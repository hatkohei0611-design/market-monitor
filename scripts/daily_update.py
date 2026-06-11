# -*- coding: utf-8 -*-
"""
market-monitor 日次更新スクリプト
1. EDGAR日次インデックスから Form 4 を取得し、Officer/Director の P/S 件数を日次集計
2. S&P 500 を Stooq から取得し、252日高値比ドローダウンを計算
3. クラスター密度(過去63営業日の比率>1.0日数)とステートを判定
4. docs/ にダッシュボード(HTML + チャートPNG)を生成

設計メモ:
- バックテスト(SEC四半期データ)と同じ定義: NONDERIV取引のP/S、Officer/Director、
  filing単位カウント、提出日ベース
- 1回の実行で処理する日数は MAX_DAYS_PER_RUN まで(初回キャッチアップは複数回実行)
- ネットワーク失敗時もダッシュボード生成は必ず行い、警告として表示する
"""

import datetime as dt
import io
import json
import os
import re
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401  (日本語フォント設定)
import matplotlib.pyplot as plt
import pandas as pd
import requests

# ===== 設定 =====
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "data", "insider_daily.csv")
SPX_CSV = os.path.join(ROOT, "data", "spx.csv")
STATE_JSON = os.path.join(ROOT, "data", "state.json")
DOCS = os.path.join(ROOT, "docs")

MAX_DAYS_PER_RUN = int(os.environ.get("MAX_DAYS", "8"))
CLUSTER_WIN = 63
THRESH = 1.0
REQ_INTERVAL = 0.13  # 約7.7req/s (SEC上限10req/sに余裕)

SEC_CONTACT = os.environ.get("SEC_CONTACT", "anonymous@example.com")
UA = {"User-Agent": f"market-monitor research {SEC_CONTACT}"}
SESSION = requests.Session()

warnings = []  # ダッシュボードに表示する警告


def log(msg):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


def fetch(url, ok404=False):
    r = SESSION.get(url, headers=UA, timeout=60)
    time.sleep(REQ_INTERVAL)
    if r.status_code == 404 and ok404:
        return None
    r.raise_for_status()
    return r


# ============================================================
# 1. EDGAR Form 4 日次集計
# ============================================================
def load_insider():
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV, encoding="utf-8-sig", parse_dates=["date"])
        return df
    warnings.append("data/insider_daily.csv が未配置です(四半期データのシードを推奨)")
    return pd.DataFrame(columns=["date", "buy_filings", "sell_filings"])


def load_state():
    if os.path.exists(STATE_JSON):
        return json.load(open(STATE_JSON))
    return {}


def save_state(st):
    json.dump(st, open(STATE_JSON, "w"), indent=1)


def parse_form4(txt):
    """Form 4 全文から (officer/directorか, NONDERIVのP有無, S有無) を返す"""
    rel = re.search(r"<isOfficer>\s*(1|true)", txt, re.I) or \
          re.search(r"<isDirector>\s*(1|true)", txt, re.I)
    if not rel:
        return False, False, False
    m = re.search(r"<nonDerivativeTable>(.*?)</nonDerivativeTable>",
                  txt, re.S | re.I)
    if not m:
        return True, False, False
    codes = set(re.findall(r"<transactionCode>\s*([A-Za-z])", m.group(1)))
    codes = {c.upper() for c in codes}
    return True, "P" in codes, "S" in codes


def process_day(day):
    """1営業日分のForm 4を集計。戻り値 (buy, sell) / 休日はNone"""
    q = (day.month - 1) // 3 + 1
    idx_url = (f"https://www.sec.gov/Archives/edgar/daily-index/"
               f"{day.year}/QTR{q}/form.{day:%m%d%y}.idx")
    r = fetch(idx_url, ok404=True)
    if r is None:
        return None  # 休日・週末
    paths = []
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ("4", "4/A"):
            paths.append(parts[-1])
    log(f"  {day:%Y-%m-%d}: Form4 {len(paths)}件を解析")
    buy = sell = err = 0
    for p in paths:
        try:
            txt = fetch("https://www.sec.gov/Archives/" + p).text
            is_od, has_p, has_s = parse_form4(txt)
            if is_od:
                buy += int(has_p)
                sell += int(has_s)
        except Exception:
            err += 1
    if err:
        log(f"  解析失敗 {err}件(続行)")
        if err > len(paths) * 0.05:
            warnings.append(f"{day:%Y-%m-%d}: Form4解析失敗が{err}件と多め")
    return buy, sell


def update_insider():
    df = load_insider()
    st = load_state()
    if st.get("last_processed"):
        last = dt.date.fromisoformat(st["last_processed"])
    elif len(df):
        last = df["date"].max().date()
    else:
        last = dt.date.today() - dt.timedelta(days=80)  # シードなし時の初期値
    # EDGARの当日インデックスは米国時間夜に完成するため、2日前まで処理
    target_end = dt.date.today() - dt.timedelta(days=2)
    day = last + dt.timedelta(days=1)
    done = 0
    new_rows = []
    while day <= target_end and done < MAX_DAYS_PER_RUN:
        if day.weekday() < 5:  # 平日のみ
            try:
                res = process_day(day)
            except Exception as e:
                warnings.append(f"{day}: EDGAR取得エラー ({e}) — 次回再試行")
                log(f"  ERROR {day}: {e}")
                break  # この日で停止し、次回ここから再開
            if res is not None:
                new_rows.append({"date": pd.Timestamp(day),
                                 "buy_filings": res[0],
                                 "sell_filings": res[1]})
            done += 1
        st["last_processed"] = day.isoformat()
        day += dt.timedelta(days=1)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df = df.drop_duplicates("date", keep="last").sort_values("date")
        df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
        log(f"insider: {len(new_rows)}日分追加 (〜{df['date'].max().date()})")
    save_state(st)
    remaining = max(0, (target_end - dt.date.fromisoformat(
        st.get("last_processed", target_end.isoformat()))).days)
    if remaining > 2:
        warnings.append(f"キャッチアップ中: 残り約{remaining}日分 "
                        f"(Actionsを再実行すると進みます)")
    return df


# ============================================================
# 2. S&P 500 (Stooq)
# ============================================================
def update_spx():
    try:
        r = fetch("https://stooq.com/q/d/l/?s=%5Espx&i=d")
        spx = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"])
        spx = spx.rename(columns={"Date": "date", "Close": "close"})
        spx = spx[["date", "close"]].dropna()
        if len(spx) > 1000:
            spx.to_csv(SPX_CSV, index=False)
            log(f"SPX: {len(spx)}日 (〜{spx['date'].max().date()})")
            return spx
        raise ValueError("Stooqの返却データが短すぎます")
    except Exception as e:
        warnings.append(f"S&P 500取得失敗 ({e}) — 前回キャッシュを使用")
        if os.path.exists(SPX_CSV):
            return pd.read_csv(SPX_CSV, parse_dates=["date"])
        return pd.DataFrame(columns=["date", "close"])


# ============================================================
# 3. 指標計算とステート判定
# ============================================================
def compute(df, spx):
    out = {}
    if len(df) == 0:
        return out
    df = df.sort_values("date").copy()
    df["ratio"] = df["buy_filings"] / df["sell_filings"].replace(0, pd.NA)
    df["sig"] = (df["ratio"] > THRESH).astype(int)
    df["density"] = df["sig"].rolling(CLUSTER_WIN, min_periods=1).sum()
    out["df"] = df
    last = df.iloc[-1]
    out["date"] = last["date"].date()
    out["ratio"] = float(last["ratio"]) if pd.notna(last["ratio"]) else None
    out["density"] = int(last["density"])
    out["buy"] = int(last["buy_filings"])
    out["sell"] = int(last["sell_filings"])
    d = out["density"]
    if d >= 21:
        out["state"] = ("③ パニック", "#cc3333",
                        "分割買い開始の候補圏。ピークアウト確認まで第1トランシェに留める")
    elif d >= 6:
        out["state"] = ("② 警戒", "#cc8833",
                        "死の谷ゾーン。新規買い禁止・現金温存(検証: 6M-5.0%/勝率56%)")
    else:
        out["state"] = ("① 平常", "#2a7d4f", "シグナルなし。通常運用")
    if len(df) < CLUSTER_WIN:
        warnings.append(f"密度のウォームアップ中 (履歴{len(df)}日 < 63日)")
    if len(spx):
        spx = spx.sort_values("date").copy()
        spx["dd"] = spx["close"] / spx["close"].rolling(252, min_periods=60).max() - 1
        out["spx"] = spx
        out["spx_close"] = float(spx["close"].iloc[-1])
        out["dd"] = float(spx["dd"].iloc[-1])
        out["spx_date"] = spx["date"].iloc[-1].date()
        gap = (pd.Timestamp(out["spx_date"]) - pd.Timestamp(out["date"])).days
        if gap > 5:
            warnings.append(f"インサイダーデータがS&P 500より{gap}日遅れています"
                            f"(EDGARキャッチアップ中は正常)")
    return out


# ============================================================
# 4. ダッシュボード生成
# ============================================================
def make_charts(res):
    df = res.get("df")
    if df is None or len(df) < 5:
        return False
    d1 = df.tail(260)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax = axes[0]
    ax.plot(d1["date"], d1["density"], color="#1a3a6b", lw=1.6)
    ax.axhspan(6, 21, color="#cc8833", alpha=0.12)
    ax.axhline(21, color="#cc3333", ls="--", lw=1)
    ax.axhline(6, color="#cc8833", ls="--", lw=1)
    ax.set_ylabel("クラスター密度 (63日)")
    ax.set_title("クラスター密度 (橙帯=警戒6-20 / 赤線=パニック21)")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(d1["date"], d1["ratio"], color="#4878CF", width=1.0)
    ax.axhline(1.0, color="#cc3333", ls="--", lw=1)
    ax.set_ylabel("件数比率 (buy/sell)")
    ax.set_title("日次 buy/sell 比率 (赤線=1.0)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS, "chart_insider.png"), dpi=110)
    plt.close()

    if "spx" in res:
        s1 = res["spx"].tail(260)
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        axes[0].plot(s1["date"], s1["close"], color="black", lw=1.4)
        axes[0].set_title("S&P 500 (直近1年)")
        axes[0].grid(alpha=0.3)
        axes[1].fill_between(s1["date"], s1["dd"] * 100, 0,
                             color="#cc3333", alpha=0.4)
        axes[1].axhline(-5, color="gray", ls=":", lw=1)
        axes[1].axhline(-15, color="gray", ls=":", lw=1)
        axes[1].set_ylabel("DD (%)")
        axes[1].set_title("252日高値比ドローダウン")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS, "chart_spx.png"), dpi=110)
        plt.close()
    return True


def build_html(res):
    now_utc = dt.datetime.utcnow()
    now_jst = now_utc + dt.timedelta(hours=9)
    if res:
        name, color, advice = res["state"]
        ratio_s = f"{res['ratio']:.3f}" if res["ratio"] is not None else "—"
        cards = f"""
  <div class="grid">
    <div class="card state" style="background:{color}">
      <div class="lbl">現在のステート</div>
      <div class="big">{name}</div>
      <div class="small">{advice}</div>
    </div>
    <div class="card"><div class="lbl">クラスター密度</div>
      <div class="big">{res['density']} <span class="unit">/63日</span></div>
      <div class="small">閾値: 警戒6 / パニック21</div></div>
    <div class="card"><div class="lbl">本日の比率 ({res['date']})</div>
      <div class="big">{ratio_s}</div>
      <div class="small">buy {res['buy']} / sell {res['sell']}</div></div>
    <div class="card"><div class="lbl">S&amp;P 500 DD ({res.get('spx_date','—')})</div>
      <div class="big">{res.get('dd', 0)*100:+.1f}%</div>
      <div class="small">終値 {res.get('spx_close', 0):,.0f} / 252日高値比</div></div>
  </div>"""
    else:
        cards = "<p>データ未取得。Actionsの実行ログを確認してください。</p>"
    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f'<div class="warn"><b>⚠ データ注意事項</b><ul>{items}</ul></div>'
    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Monitor</title>
<style>
 body{{font-family:'Hiragino Sans','Yu Gothic',sans-serif;margin:0;background:#f4f6f9;color:#222}}
 header{{background:#1a3a6b;color:#fff;padding:14px 18px}}
 header h1{{font-size:18px;margin:0}} header .sub{{font-size:11px;opacity:.8}}
 main{{max-width:860px;margin:0 auto;padding:14px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
 .card{{background:#fff;border-radius:10px;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card.state{{color:#fff}}
 .lbl{{font-size:11px;opacity:.75}} .big{{font-size:26px;font-weight:bold;margin:4px 0}}
 .unit{{font-size:13px;font-weight:normal}} .small{{font-size:11px;opacity:.8;line-height:1.5}}
 .warn{{background:#fff6e5;border:1px solid #e0b95e;border-radius:8px;padding:10px 14px;margin:12px 0;font-size:12px}}
 .warn ul{{margin:6px 0 0 18px;padding:0}}
 img{{width:100%;border-radius:10px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);margin:10px 0}}
 footer{{font-size:10.5px;color:#777;padding:14px;line-height:1.6}}
</style></head><body>
<header><h1>Market Monitor — インサイダー密度ステート</h1>
<div class="sub">最終更新: {now_jst:%Y-%m-%d %H:%M} JST ({now_utc:%H:%M} UTC) / データ: SEC EDGAR + Stooq</div></header>
<main>
{cards}
{warn_html}
<img src="chart_insider.png" alt="insider">
<img src="chart_spx.png" alt="spx">
</main>
<footer>定義: 件数比率 = Officer/DirectorのForm 4日次 buy filings / sell filings (NONDERIVのP/S・filing単位・提出日ベース)。
クラスター密度 = 過去63営業日における比率&gt;1.0の日数。ステート: ①平常0-5 / ②警戒6-20(新規買い禁止) / ③パニック21+(分割買い候補)。
本ページは検証記録に基づく私的モニターであり投資助言ではない。</footer>
</body></html>"""
    with open(os.path.join(DOCS, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main():
    os.makedirs(DOCS, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    try:
        df = update_insider()
    except Exception as e:
        traceback.print_exc()
        warnings.append(f"インサイダー更新で予期せぬエラー: {e}")
        df = load_insider()
    spx = update_spx()
    res = compute(df, spx)
    if res:
        make_charts(res)
    build_html(res)
    log("dashboard generated")
    for w in warnings:
        log(f"WARN: {w}")


if __name__ == "__main__":
    main()
