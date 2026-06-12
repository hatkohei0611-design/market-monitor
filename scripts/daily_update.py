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
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

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
TIME_BUDGET_MIN = int(os.environ.get("TIME_BUDGET_MIN", "45"))  # これを超えたら保存して終了
WORKERS = 5  # Form 4取得の並列数 (全体でSEC上限10req/sを遵守)
START_TIME = time.time()
CLUSTER_WIN = 63
THRESH = 1.0
REQ_INTERVAL = 0.13  # 約7.7req/s (SEC上限10req/sに余裕)

SEC_CONTACT = os.environ.get("SEC_CONTACT", "anonymous@example.com")
# SEC公式サンプルと同じ素朴な形式: "Sample Company Name AdminContact@example.com"
SEC_HEADERS = {
    "User-Agent": f"market-monitor {SEC_CONTACT}",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*",
}
# ブラウザ系サイト(FRED/Stooq/Yahoo)用
BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/csv,application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
SESSION = requests.Session()
_rate_lock = threading.Lock()
_last_req = [0.0]


def _rate_limit():
    """全スレッド共通で約7req/sに制限"""
    with _rate_lock:
        wait = _last_req[0] + REQ_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


warnings = []  # ダッシュボードに表示する警告


def log(msg):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


def fetch(url, ok404=False, headers=None, tries=4):
    """403/429/5xx は間隔を空けてリトライする"""
    last_err = None
    for i in range(tries):
        _rate_limit()
        try:
            r = SESSION.get(url, headers=headers or SEC_HEADERS, timeout=30)
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 * (i + 1))
            continue
        if r.status_code == 404 and ok404:
            return None
        if r.status_code == 403 and "<Code>AccessDenied</Code>" in r.text[:400]:
            # S3形式のAccessDenied = ファイル不存在 (休場日・未公開日)
            if ok404:
                return None
            r.raise_for_status()
        if r.status_code in (403, 429, 500, 502, 503) and i < tries - 1:
            log(f"  HTTP {r.status_code} -> {4*(i+1)}秒待って再試行 ({url[-40:]})")
            time.sleep(4 * (i + 1))
            continue
        if r.status_code >= 400:
            body = r.text[:200].replace("\n", " ")
            log(f"  [診断] HTTP {r.status_code} body先頭: {body}")
        r.raise_for_status()
        return r
    if last_err:
        raise last_err
    r.raise_for_status()


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
               f"{day.year}/QTR{q}/form.{day:%Y%m%d}.idx")
    r = fetch(idx_url, ok404=True)
    if r is None:
        return None  # 休日・週末
    paths = []
    for line in r.text.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ("4", "4/A"):
            paths.append(parts[-1])
    log(f"  {day:%Y-%m-%d}: Form4 {len(paths)}件を解析 ({WORKERS}並列)")
    counts = {"buy": 0, "sell": 0, "err": 0}
    lock = threading.Lock()

    def work(p):
        try:
            txt = fetch("https://www.sec.gov/Archives/" + p, tries=2).text
            is_od, has_p, has_s = parse_form4(txt)
            with lock:
                if is_od:
                    counts["buy"] += int(has_p)
                    counts["sell"] += int(has_s)
        except Exception:
            with lock:
                counts["err"] += 1

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, paths))
    if counts["err"]:
        log(f"  解析失敗 {counts['err']}件(続行)")
        if counts["err"] > len(paths) * 0.05:
            warnings.append(f"{day:%Y-%m-%d}: Form4解析失敗が{counts['err']}件と多め")
    return counts["buy"], counts["sell"]


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
    done = added = 0
    while day <= target_end and done < MAX_DAYS_PER_RUN:
        if (time.time() - START_TIME) > TIME_BUDGET_MIN * 60:
            log(f"時間予算 {TIME_BUDGET_MIN}分に到達 — ここまでを保存して終了")
            break
        if day.weekday() < 5:  # 平日のみ
            try:
                res = process_day(day)
            except Exception as e:
                warnings.append(f"{day}: EDGAR取得エラー ({e}) — 次回再試行")
                log(f"  ERROR {day}: {e}")
                break  # この日で停止し、次回ここから再開
            if res is not None:
                row = pd.DataFrame([{"date": pd.Timestamp(day),
                                     "buy_filings": res[0],
                                     "sell_filings": res[1]}])
                df = pd.concat([df, row], ignore_index=True)
                df = df.drop_duplicates("date", keep="last").sort_values("date")
                df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")  # 1日ごと保存
                added += 1
            done += 1
        st["last_processed"] = day.isoformat()
        save_state(st)
        day += dt.timedelta(days=1)
    if added:
        log(f"insider: {added}日分追加 (〜{df['date'].max().date()})")
    remaining = max(0, (target_end - dt.date.fromisoformat(
        st.get("last_processed", target_end.isoformat()))).days)
    if remaining > 2:
        warnings.append(f"キャッチアップ中: 残り約{remaining}日分 "
                        f"(Actionsを再実行すると進みます)")
    return df


# ============================================================
# 2. S&P 500 (Stooq)
# ============================================================
def _spx_from_yahoo():
    """Yahoo Finance chart API (JSON)"""
    r = fetch("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
              "?range=2y&interval=1d", headers=BROWSER_HEADERS)
    j = r.json()["chart"]["result"][0]
    ts = j["timestamp"]
    cl = j["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").normalize(),
                       "close": cl})
    return df.dropna()


def _spx_from_fred():
    """FRED公式CSV (キー不要、直近10年分)"""
    r = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
              headers=BROWSER_HEADERS)
    df = pd.read_csv(io.StringIO(r.text))
    df = df.iloc[:, :2]
    df.columns = ["date", "close"]  # 列名は DATE/observation_date 両対応
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna()


def _spx_from_stooq():
    r = fetch("https://stooq.com/q/d/l/?s=%5Espx&i=d",
              headers=BROWSER_HEADERS)
    df = pd.read_csv(io.StringIO(r.text))
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError(f"Stooq想定外の列: {list(df.columns)[:5]}")
    df = df.rename(columns={cols["date"]: "date", cols["close"]: "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df[["date", "close"]].dropna()


def update_spx():
    for name, fn, min_len in [("Yahoo", _spx_from_yahoo, 400),
                              ("FRED", _spx_from_fred, 500),
                              ("Stooq", _spx_from_stooq, 1000)]:
        try:
            spx = fn()
            if len(spx) >= min_len:
                spx.to_csv(SPX_CSV, index=False)
                log(f"SPX({name}): {len(spx)}日 (〜{spx['date'].max().date()})")
                return spx
            raise ValueError(f"{name}の返却が短すぎます ({len(spx)}行)")
        except Exception as e:
            log(f"SPX {name} 失敗: {e}")
            last = e
    warnings.append(f"S&P 500取得失敗 ({last}) — 前回キャッシュを使用")
    if os.path.exists(SPX_CSV):
        return pd.read_csv(SPX_CSV, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "close"])


# ============================================================
# 2.5 AIAE (Aggregate Investor Allocation to Equities)
#     FRED Z.1 公式系列から構築 (過去検証: 2000Q1=51.59% vs 論文51.7%)
#     株式時価 = NCBEILQ027S + FBCELLQ027S (百万ドル -> /1000で十億ドルに統一)
#     負債計   = FGSDODNS + CMDEBT + BCNSDODNS + DODFFSWCMI + SLGSDODNS (十億ドル)
#     AIAE = 株式 / (株式 + 負債)
# ============================================================
AIAE_CSV = os.path.join(ROOT, "data", "aiae.csv")
AIAE_EQ = ["NCBEILQ027S", "FBCELLQ027S"]          # millions of $
AIAE_DEBT = ["FGSDODNS", "CMDEBT", "BCNSDODNS",
             "DODFFSWCMI", "SLGSDODNS"]            # billions of $


def fred_series(sid):
    """FRED系列を取得 (キー不要のfredgraph.csv。失敗時はFRED_API_KEYでAPI)"""
    try:
        r = fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                  headers=BROWSER_HEADERS)
        df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
        df.columns = ["date", "value"]
    except Exception:
        key = os.environ.get("FRED_API_KEY", "")
        if not key:
            raise
        r = fetch("https://api.stlouisfed.org/fred/series/observations"
                  f"?series_id={sid}&api_key={key}&file_type=json",
                  headers=BROWSER_HEADERS)
        obs = r.json()["observations"]
        df = pd.DataFrame({"date": [o["date"] for o in obs],
                           "value": [o["value"] for o in obs]})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].rename(sid)


def update_aiae(spx):
    """AIAE四半期系列 + S&P500補正のリアルタイム近似を返す"""
    try:
        cols = {}
        for sid in AIAE_EQ + AIAE_DEBT:
            cols[sid] = fred_series(sid)
        z1 = pd.concat(cols.values(), axis=1).dropna()
        eq = z1[AIAE_EQ].sum(axis=1) / 1000.0       # millions -> billions
        debt = z1[AIAE_DEBT].sum(axis=1)
        aiae = (eq / (eq + debt)).rename("aiae")
        # 健全性チェック (歴史レンジは概ね 20%〜55%)
        last = float(aiae.iloc[-1])
        if not (0.15 < last < 0.70):
            warnings.append(f"AIAE計算値が異常 ({last:.1%}) — 系列の単位/定義を要確認")
            return None
        out = pd.DataFrame({"aiae": aiae, "eq": eq, "debt": debt})
        out.index.name = "date"
        out.to_csv(AIAE_CSV)
        log(f"AIAE: {len(out)}四半期 (最新 {out.index[-1].date()} = {last:.2%})")
    except Exception as e:
        warnings.append(f"AIAE取得失敗 ({e}) — 前回キャッシュを使用")
        if os.path.exists(AIAE_CSV):
            out = pd.read_csv(AIAE_CSV, parse_dates=["date"], index_col="date")
        else:
            return None
    res = {"q": out, "q_date": out.index[-1].date(),
           "q_val": float(out["aiae"].iloc[-1])}
    # リアルタイム近似: 株式部分を四半期末からのS&P500変化率でスケール
    if spx is not None and len(spx):
        spx = spx.sort_values("date")
        q_end = pd.Timestamp(out.index[-1]) + pd.offsets.QuarterEnd(0)
        base = spx[spx["date"] <= q_end]["close"]
        if len(base):
            scale = float(spx["close"].iloc[-1]) / float(base.iloc[-1])
            eq_adj = float(out["eq"].iloc[-1]) * scale
            res["rt_val"] = eq_adj / (eq_adj + float(out["debt"].iloc[-1]))
            res["rt_date"] = spx["date"].iloc[-1].date()
            # 公式値の過去分布におけるリアルタイム値のパーセンタイル
            res["rt_pct"] = float((out["aiae"] < res["rt_val"]).mean())
    return res


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
def make_charts(res, aiae=None):
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

    if aiae is not None:
        q = aiae["q"]
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(q.index, q["aiae"] * 100, color="#1a3a6b", lw=1.4,
                label="AIAE (Z.1公式・四半期)")
        if "rt_val" in aiae:
            ax.scatter([pd.Timestamp(aiae["rt_date"])], [aiae["rt_val"] * 100],
                       color="#cc3333", zorder=5, s=40,
                       label=f"リアルタイム近似 {aiae['rt_val']*100:.1f}%")
        med = q["aiae"].median() * 100
        ax.axhline(med, color="gray", ls=":", lw=1, label=f"中央値 {med:.1f}%")
        ax.set_ylabel("AIAE (%)")
        ax.set_title("AIAE — 投資家の株式配分比率 (FRED Z.1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS, "chart_aiae.png"), dpi=110)
        plt.close()
    return True


def aiae_card(aiae):
    if not aiae:
        return ""
    rt = ""
    if "rt_val" in aiae:
        rt = (f"リアルタイム近似 {aiae['rt_val']*100:.1f}% "
              f"(歴史パーセンタイル {aiae['rt_pct']*100:.0f}%)")
    return (f'<div class="card"><div class="lbl">AIAE '
            f'(公式 {aiae["q_date"]})</div>'
            f'<div class="big">{aiae["q_val"]*100:.1f}%</div>'
            f'<div class="small">{rt}</div></div>')


def build_html(res, aiae=None):
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
    {aiae_card(aiae)}
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
<div class="sub">最終更新: {now_jst:%Y-%m-%d %H:%M} JST ({now_utc:%H:%M} UTC) / データ: SEC EDGAR + Yahoo/FRED</div></header>
<main>
{cards}
{warn_html}
<img src="chart_insider.png" alt="insider">
<img src="chart_spx.png" alt="spx">
<img src="chart_aiae.png" alt="aiae" onerror="this.style.display='none'">
</main>
<footer>定義: 件数比率 = Officer/DirectorのForm 4日次 buy filings / sell filings (NONDERIVのP/S・filing単位・提出日ベース)。
クラスター密度 = 過去63営業日における比率&gt;1.0の日数。
AIAE = 株式時価(NCBEILQ027S+FBCELLQ027S) ÷ (株式時価 + 5借入主体負債計)、Z.1公開ラグ2.5〜5.5ヶ月、リアルタイム近似は株式部分のみS&amp;P500で補正。ステート: ①平常0-5 / ②警戒6-20(新規買い禁止) / ③パニック21+(分割買い候補)。
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
    aiae = update_aiae(spx)
    if res:
        make_charts(res, aiae)
    build_html(res, aiae)
    log("dashboard generated")
    for w in warnings:
        log(f"WARN: {w}")


if __name__ == "__main__":
    main()
