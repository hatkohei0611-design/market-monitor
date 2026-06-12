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


def fetch(url, ok404=False, headers=None, tries=4, timeout=30):
    """403/429/5xx は間隔を空けてリトライする"""
    last_err = None
    for i in range(tries):
        _rate_limit()
        try:
            r = SESSION.get(url, headers=headers or SEC_HEADERS, timeout=timeout)
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
    """FRED系列を取得 (FRED_API_KEYがあればAPI優先、なければfredgraph.csv)"""
    key = os.environ.get("FRED_API_KEY", "")
    if key:
        r = fetch("https://api.stlouisfed.org/fred/series/observations"
                  f"?series_id={sid}&api_key={key}&file_type=json",
                  headers=BROWSER_HEADERS, timeout=60)
        obs = r.json()["observations"]
        df = pd.DataFrame({"date": [o["date"] for o in obs],
                           "value": [o["value"] for o in obs]})
    else:
        r = fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}",
                  headers=BROWSER_HEADERS, timeout=60)
        df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
        df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")["value"].rename(sid)


def update_aiae(spx):
    """AIAE四半期系列 + S&P500補正のリアルタイム近似を返す"""
    try:
        cols, detail = {}, []
        for sid in AIAE_EQ + AIAE_DEBT:
            s = fred_series(sid)
            # 単位正規化: FRED APIは百万$、fredgraphは表示単位(十億$等)で返す
            # 最新値が1e6超なら百万$単位とみなし十億$へ換算
            if abs(s.iloc[-1]) > 1e6:
                s = s / 1000.0
            cols[sid] = s
            detail.append(f"{sid}=〜{s.index[-1].date()}:{s.iloc[-1]:,.0f}B")
        log("AIAE系列診断(十億$換算後): " + " | ".join(detail))
        z1 = pd.concat(cols.values(), axis=1).dropna()
        eq = z1[AIAE_EQ].sum(axis=1)    # 全て十億$に正規化済み
        debt = z1[AIAE_DEBT].sum(axis=1)
        aiae = (eq / (eq + debt)).rename("aiae")
        last = float(aiae.iloc[-1])
        log(f"AIAE構成(最終{z1.index[-1].date()}): eq={eq.iloc[-1]:,.0f}B "
            f"debt={debt.iloc[-1]:,.0f}B -> AIAE={last:.2%}")
        stale = [sid for sid, s in cols.items()
                 if s.index[-1] < pd.Timestamp.now() - pd.Timedelta(days=730)]
        if stale:
            warnings.append(f"AIAE系列が更新停止の疑い: {','.join(stale)} "
                            f"(後継系列への切替が必要かもしれません)")
        # 健全性チェック (歴史レンジは概ね 20%〜55%)
        if not (0.15 < last < 0.70):
            warnings.append(f"AIAE計算値が異常 ({last:.1%}, eq={eq.iloc[-1]:,.0f}B/"
                            f"debt={debt.iloc[-1]:,.0f}B) — ログのAIAE系列診断を参照")
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
# 2.6 天井側モジュール: WSJブレッドス -> 自前HO判定 / FINRA MD/M2 / V1スコア
#   HO定義(2026-06-12固定): NYSE+Nasdaq合算、NH/NL両方>2.8%、
#   SPX>50営業日前(上昇トレンド)、McClellan Osc<0、NH<=2*NL
#   過去シード(data/hb_signals.csv のseed_*行)は目視集約の凍結値で定義が異なる
# ============================================================
BREADTH_CSV = os.path.join(ROOT, "data", "breadth_daily.csv")
HB_CSV = os.path.join(ROOT, "data", "hb_signals.csv")
MDM2_CSV = os.path.join(ROOT, "data", "mdm2.csv")
HO_THRESH = 0.028


def update_breadth(spx):
    """WSJ市場データから当日のNH/NL/騰落数を取得して蓄積、HO判定"""
    cols = ["date", "adv", "dec", "nh", "nl", "issues"]
    bd = (pd.read_csv(BREADTH_CSV, parse_dates=["date"])
          if os.path.exists(BREADTH_CSV) else pd.DataFrame(columns=cols))
    url = ("https://www.wsj.com/market-data/stocks/marketsdiary"
           "?id=%7B%22application%22%3A%22WSJ%22%2C%22marketsDiaryType%22"
           "%3A%22diaries%22%7D&type=mdc_marketsdiary")
    try:
        r = fetch(url, headers={**BROWSER_HEADERS,
                                "Accept": "application/json, text/plain, */*",
                                "Referer": "https://www.wsj.com/market-data"})
        j = r.json()
        found = {}
        # 実構造パース: data.instrumentSets[].headerFields[0].label = 取引所名
        #               instruments[].id -> latestClose (直近取引日の値)
        for s in (j.get("data", {}) or {}).get("instrumentSets", []):
            hdr = s.get("headerFields") or [{}]
            label = str(hdr[0].get("label", "")).lower()
            vals = {}
            for inst in s.get("instruments", []):
                try:
                    vals[str(inst.get("id", "")).lower()] = float(
                        str(inst.get("latestClose", "")).replace(",", ""))
                except (TypeError, ValueError):
                    pass
            rec = {"adv": vals.get("advances"), "dec": vals.get("declines"),
                   "nh": vals.get("newhighs"), "nl": vals.get("newlows"),
                   "issues": vals.get("issuestraded")}
            if rec["adv"] is None or rec["nh"] is None:
                continue
            if "nyse" in label and "amex" not in label and "arca" not in label:
                found.setdefault("nyse", rec)
            elif "nasdaq" in label:
                found.setdefault("nasdaq", rec)
        # フォールバック: 旧・再帰探索

        def walk(node, label=""):
            if isinstance(node, dict):
                lab = str(node.get("name", node.get("exchange", label)))
                keys = {k.lower(): k for k in node.keys()}
                def grab(*cands):
                    for c in cands:
                        for kl, ko in keys.items():
                            if c in kl:
                                v = node[ko]
                                if isinstance(v, dict):
                                    v = v.get("value", v.get("raw"))
                                try:
                                    return float(str(v).replace(",", ""))
                                except (TypeError, ValueError):
                                    pass
                    return None
                rec = {"adv": grab("advanc"), "dec": grab("declin"),
                       "nh": grab("newhigh", "new high", "52weekhigh", "hi52"),
                       "nl": grab("newlow", "new low", "52weeklow", "lo52"),
                       "issues": grab("issuestraded", "total issues", "issues")}
                ll = lab.lower()
                if rec["adv"] is not None and rec["nh"] is not None:
                    if "nyse" in ll and "amex" not in ll and "arca" not in ll:
                        found.setdefault("nyse", rec)
                    elif "nasdaq" in ll:
                        found.setdefault("nasdaq", rec)
                for v in node.values():
                    walk(v, lab)
            elif isinstance(node, list):
                for v in node:
                    walk(v, label)
        walk(j)
        if "nyse" not in found or "nasdaq" not in found:
            snippet = json.dumps(j.get("data", j), ensure_ascii=False)[:900] \
                if isinstance(j, dict) else str(j)[:300]
            log(f"[診断] WSJ data中身: {snippet}")
            raise ValueError("WSJ応答の構造が想定外 (ログの[診断]行参照)")
        if spx is not None and len(spx):
            diary_date = pd.Timestamp(spx["date"].max())
        else:
            diary_date = pd.Timestamp(dt.date.today())
        row = {"date": diary_date}
        for k in ["adv", "dec", "nh", "nl", "issues"]:
            vals = [found[m][k] for m in ("nyse", "nasdaq")
                    if found[m][k] is not None]
            row[k] = sum(vals) if vals else None
        if row["issues"] is None:  # issues欠落時はadv+dec+α近似不可 -> NH/NL率計算不可
            raise ValueError("issues traded が取得できません")
        bd = pd.concat([bd, pd.DataFrame([row])], ignore_index=True)
        bd = bd.drop_duplicates("date", keep="last").sort_values("date")
        bd.to_csv(BREADTH_CSV, index=False)
        log(f"breadth(WSJ): {row['date'].date()} NH={row['nh']:.0f} "
            f"NL={row['nl']:.0f} issues={row['issues']:.0f}")
    except Exception as e:
        warnings.append(f"WSJブレッドス取得失敗 ({e}) — 本日のHO判定スキップ")
        return bd
    # ---- HO判定 (最新日) ----
    if len(bd) >= 1 and spx is not None and len(spx) > 60:
        b = bd.iloc[-1]
        nh_pct = b["nh"] / b["issues"]
        nl_pct = b["nl"] / b["issues"]
        c1 = nh_pct > HO_THRESH and nl_pct > HO_THRESH
        c4 = b["nh"] <= 2 * b["nl"]
        spx_s = spx.sort_values("date")["close"]
        c2 = float(spx_s.iloc[-1]) > float(spx_s.iloc[-51])
        # McClellan Oscillator (ratio-adjusted, EMA19-EMA39) — 39日蓄積後に有効
        c3, mo = None, None
        if len(bd) >= 39:
            rana = (bd["adv"] - bd["dec"]) / (bd["adv"] + bd["dec"]) * 1000
            mo = float(rana.ewm(span=19).mean().iloc[-1]
                       - rana.ewm(span=39).mean().iloc[-1])
            c3 = mo < 0
        else:
            warnings.append(f"McClellan蓄積中 ({len(bd)}/39日) — HO判定は3条件版(暫定)")
        provisional = c3 is None
        fired = c1 and c2 and c4 and (c3 if c3 is not None else True)
        log(f"HO条件: NH%={nh_pct:.2%} NL%={nl_pct:.2%} 両方>2.8%={c1} "
            f"上昇トレンド={c2} MO={'%.1f' % mo if mo is not None else 'N/A'} "
            f"NH<=2NL={c4} -> {'点灯' if fired else '消灯'}")
        if fired:
            hb = (pd.read_csv(HB_CSV, parse_dates=["date"])
                  if os.path.exists(HB_CSV)
                  else pd.DataFrame(columns=["date", "signals", "source"]))
            d = pd.Timestamp(b["date"])
            srcname = "auto_provisional" if provisional else "auto"
            if not ((hb["date"] == d) &
                    (hb["source"].str.startswith("auto"))).any():
                hb = pd.concat([hb, pd.DataFrame(
                    [{"date": d, "signals": 1, "source": srcname}])],
                    ignore_index=True).sort_values("date")
                hb.to_csv(HB_CSV, index=False)
                log("HO点灯 -> hb_signals.csv に記録")
    return bd


def hb_counts():
    if not os.path.exists(HB_CSV):
        return None
    hb = pd.read_csv(HB_CSV, parse_dates=["date"])
    now = pd.Timestamp(dt.date.today())
    out = {}
    for label, days in [("3m", 91), ("6m", 182), ("12m", 365)]:
        out[label] = int(hb[hb["date"] > now - pd.Timedelta(days=days)]
                         ["signals"].sum())
    return out


def ho_level(c):
    """4段階階層型クラスター判定 (NYSE+Nasdaq合算閾値)"""
    if c is None:
        return None
    if c["3m"] >= 20 or c["12m"] >= 53:
        return 4, "Crisis (2018年級)", "#8b0000"
    if c["3m"] >= 16 and c["12m"] >= 33:
        return 3, "Alarm (2007年級)", "#cc3333"
    if c["3m"] >= 10 and c["12m"] >= 22:
        return 2, "Warning (2000年級)", "#cc6633"
    if c["6m"] >= 8 or c["12m"] >= 15:
        return 1, "Watch (注意)", "#cc9933"
    return 0, "通常", "#2a7d4f"


def update_mdm2():
    """FINRAマージンデット(月次) + FRED M2 -> V1スケールのMD/M2"""
    try:
        r = fetch("https://www.finra.org/investors/learn-to-invest/"
                  "advanced-investing/margin-statistics",
                  headers={**BROWSER_HEADERS,
                           "Sec-Fetch-Dest": "document",
                           "Sec-Fetch-Mode": "navigate",
                           "Sec-Fetch-Site": "none",
                           "Upgrade-Insecure-Requests": "1"})
        # ページ内テーブルから 月名+数値3列 の行を抽出 (debit=第1数値, $millions)
        rows = re.findall(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s\-]+"
            r"(\d{4})[^0-9]*?([\d,]{6,})", r.text)
        if not rows:
            raise ValueError("FINRAページからテーブルを抽出できず(構造変更?)")
        mon = {m: i + 1 for i, m in enumerate(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        md = pd.DataFrame(
            [{"date": pd.Timestamp(int(y), mon[m], 1),
              "md_mil": float(v.replace(",", ""))} for m, y, v in rows])
        md = md.drop_duplicates("date").sort_values("date")
        if md["md_mil"].iloc[-1] < 5e5:  # 50万(=0.5T)未満は誤抽出疑い
            raise ValueError(f"MD抽出値が異常 ({md['md_mil'].iloc[-1]:,.0f}M)")
        m2 = fred_series("M2SL")  # billions, 月次
        mm = md.set_index("date")["md_mil"].div(1000)  # -> billions
        ratio = (mm / m2).dropna()
        scaled = ratio * 100000  # V1スケール (5382 = 5.382%相当)
        out = pd.DataFrame({"mdm2_v1": scaled})
        out.index.name = "date"
        out.to_csv(MDM2_CSV)
        log(f"MD/M2: 最新 {out.index[-1].date()} = {scaled.iloc[-1]:.0f} (V1スケール)")
        return {"date": out.index[-1].date(), "val": float(scaled.iloc[-1])}
    except Exception as e:
        warnings.append(f"FINRA MD/M2取得失敗 ({e}) — キャッシュ使用 "
                        f"(継続する場合は data/mdm2.csv に月1回手動追記)")
        if os.path.exists(MDM2_CSV):
            c = pd.read_csv(MDM2_CSV, parse_dates=["date"], index_col="date")
            return {"date": c.index[-1].date(), "val": float(c["mdm2_v1"].iloc[-1])}
        return None


def compute_v1(mdm2, aiae, hbc):
    """V1スコア = MD/M2 + HB(6m) + AIAE。35点が真の天井境界"""
    if mdm2 is None or aiae is None or hbc is None:
        return None
    aiae_pct = aiae.get("rt_val", aiae["q_val"]) * 100
    s_md = max(0.0, (mdm2["val"] - 4000) / 50)
    s_hb = float(hbc["6m"])
    # FRED公式AIAEは旧較正(TradingView系)と約10pt水準が異なるため閾値50に再較正
    # (アンカー: 2026-03のAIAEスコア2.93を再現するよう接続)
    s_ai = max(0.0, aiae_pct - 50)
    total = s_md + s_hb + s_ai
    if total >= 50:
        bucket, color = "50+ 史上級バブル (2000型)", "#8b0000"
    elif total >= 35:
        bucket, color = "35-50 真の天井域 (過去1Y勝率0%)", "#cc3333"
    elif total >= 25:
        bucket, color = "25-35 偽信号警戒 (2018型混在)", "#cc6633"
    elif total >= 15:
        bucket, color = "15-25 注意 (2022型)", "#cc9933"
    else:
        bucket, color = "0-15 通常市場", "#2a7d4f"
    return {"total": total, "md": s_md, "hb": s_hb, "ai": s_ai,
            "aiae_pct": aiae_pct, "bucket": bucket, "color": color,
            "mdm2": mdm2}


# ============================================================
# 2.7 期待値モニター: 長期SPX / AIAE 10Y回帰 / 密度バケット統計
# ============================================================
SPXLONG_CSV = os.path.join(ROOT, "data", "spx_long.csv")


def update_spx_long():
    """Yahooから日次の全履歴(1927-)を取得。キャッシュ7日有効"""
    if os.path.exists(SPXLONG_CSV):
        c = pd.read_csv(SPXLONG_CSV, parse_dates=["date"])
        if c["date"].max() > pd.Timestamp.now() - pd.Timedelta(days=7):
            return c
    try:
        r = fetch("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
                  "?range=max&interval=1d", headers=BROWSER_HEADERS, timeout=60)
        j = r.json()["chart"]["result"][0]
        df = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(),
                           "close": j["indicators"]["quote"][0]["close"]}).dropna()
        df = df.drop_duplicates("date").sort_values("date")
        if len(df) > 5000:
            df.to_csv(SPXLONG_CSV, index=False)
            log(f"SPX長期: {len(df)}日 ({df['date'].min().date()}〜)")
            return df
        raise ValueError(f"長期SPXが短すぎます ({len(df)}行)")
    except Exception as e:
        warnings.append(f"SPX長期履歴の取得失敗 ({e}) — 期待値モニター一部停止")
        if os.path.exists(SPXLONG_CSV):
            return pd.read_csv(SPXLONG_CSV, parse_dates=["date"])
        return None


def aiae_regression(aiae, spxl):
    """AIAE四半期 vs 10年先S&P500年率リターン(価格)のOLS。毎日再計算"""
    if aiae is None or spxl is None or len(spxl) < 3000:
        return None
    try:
        q = aiae["q"]["aiae"].copy()
        m = spxl.set_index("date")["close"].resample("ME").last().dropna()
        rows = []
        for t, x in q.items():
            t0 = t + pd.offsets.MonthEnd(0)
            t1 = t0 + pd.DateOffset(years=10)
            if t0 in m.index:
                f = m[m.index >= t1]
                if len(f):
                    rows.append((float(x), (float(f.iloc[0]) / float(m[t0])) ** 0.1 - 1))
        if len(rows) < 40:
            return None
        import numpy as np
        x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        slope, icpt = np.polyfit(x, y, 1)
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        cur = aiae.get("rt_val", aiae["q_val"])
        pred = slope * cur + icpt
        return {"pred": float(pred), "r2": r2, "n": len(rows),
                "cur": cur, "last_q": str(q.index[-1].date())}
    except Exception as e:
        log(f"AIAE回帰エラー: {e}")
        return None


def insider_bucket_stats(df, spxl):
    """密度バケット別の先行リターン統計(全履歴から毎日再計算)"""
    if df is None or len(df) < 300 or spxl is None:
        return None
    try:
        d = df.sort_values("date").copy()
        s = spxl.sort_values("date").reset_index(drop=True)
        d = pd.merge_asof(d, s.rename(columns={"close": "px"}), on="date")
        d["idx"] = d["date"].map(
            {dt_: i for i, dt_ in enumerate(s["date"])})
        d = d.dropna(subset=["idx", "px"])
        px = s["close"].values
        def fwd(i, h):
            i = int(i)
            return px[i + h] / px[i] - 1 if i + h < len(px) else None
        d["f6"] = d["idx"].map(lambda i: fwd(i, 126))
        d["f12"] = d["idx"].map(lambda i: fwd(i, 252))
        def bucket(v):
            return "21+" if v >= 21 else ("6-20" if v >= 6 else
                                          ("1-5" if v >= 1 else "0"))
        d["bk"] = d["density"].map(bucket)
        out = {}
        for bk, g in d.dropna(subset=["f12"]).groupby("bk"):
            out[bk] = {"n": len(g),
                       "m6": float(g["f6"].mean()), "w6": float((g["f6"] > 0).mean()),
                       "m12": float(g["f12"].mean()), "w12": float((g["f12"] > 0).mean())}
        return out
    except Exception as e:
        log(f"密度統計エラー: {e}")
        return None


# V1バケット統計 (2026-05レポート固定値: 過去のMD/M2・HB月次系列が
# リポジトリにないため再計算不可。較正の出典を明示して固定表示)
V1_BUCKETS = [
    (0, 15, "0-15 通常市場", "+11〜17%", "86-96%", 215),
    (15, 25, "15-25 注意(2022型)", "-0.3〜-4.3%", "40-50%", 37),
    (25, 35, "25-35 偽信号警戒", "-5.6%", "53%", 17),
    (35, 50, "35-50 真の天井域", "-27.1%", "0%", 17),
    (50, 999, "50+ 史上級(2000型)", "-15.9%", "0%", 9),
]


def v1_bucket_row(total):
    for lo, hi, name, cagr, win, n in V1_BUCKETS:
        if lo <= total < hi:
            return {"name": name, "cagr": cagr, "win": win, "n": n}
    return None


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


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;800&family=Noto+Sans+JP:wght@400;500;700;900&display=swap');
:root{--navy:#16243f;--blue:#2b5aa0;--ink:#1c2433;--mut:#69748a;--bg:#eef1f7;
--ok:#1e8a5a;--warn:#d07f2e;--bad:#cd3a3a;--card:#ffffff}
*{box-sizing:border-box}
body{font-family:'Noto Sans JP','Inter',sans-serif;margin:0;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(120deg,#101c33,#1b3158 55%,#27548f);color:#fff;padding:26px 22px 22px}
header h1{font-size:21px;margin:0;font-weight:900;letter-spacing:.02em}
header .sub{font-size:11.5px;opacity:.75;margin-top:6px;font-weight:500}
main{max-width:980px;margin:0 auto;padding:18px 16px 30px}
h2.sec{font-size:13.5px;margin:22px 0 10px;color:var(--navy);font-weight:900;
display:flex;align-items:center;gap:8px}
h2.sec::before{content:"";width:10px;height:10px;border-radius:3px;background:var(--blue)}
h2.sec.top::before{background:var(--bad)}
h2.sec.exp::before{background:#7a4fd0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:12px}
.card{background:var(--card);border-radius:14px;padding:15px 16px;
box-shadow:0 2px 10px rgba(22,36,63,.07);border:1px solid rgba(22,36,63,.05)}
.card.state{color:#fff;border:none}
.lbl{font-size:10.5px;color:var(--mut);font-weight:700;letter-spacing:.04em;text-transform:uppercase}
.card.state .lbl{color:rgba(255,255,255,.75)}
.big{font-size:30px;font-weight:800;font-family:'Inter','Noto Sans JP',sans-serif;
margin:5px 0 3px;line-height:1.1}
.unit{font-size:13px;font-weight:500;color:var(--mut)}
.small{font-size:11px;color:var(--mut);line-height:1.65}
.card.state .small{color:rgba(255,255,255,.85)}
.bar{height:8px;border-radius:99px;background:#e7ebf3;margin:9px 0 4px;position:relative}
.bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;
background:linear-gradient(90deg,#1e8a5a,#d0b32e 45%,#cd3a3a 70%)}
.bar>b{position:absolute;top:-3px;bottom:-3px;width:2px;background:#16243f;left:58.3%}
.note{background:#eef2fa;border:1px solid #c5d2ea;border-radius:12px;
padding:11px 15px;font-size:11.5px;line-height:1.8;margin:13px 0;color:#2c3a55}
.warnbox{background:#fff7e8;border:1px solid #e6c386;border-radius:12px;
padding:11px 15px;font-size:11.5px;margin:13px 0;color:#5d4310}
.warnbox ul{margin:5px 0 0 17px;padding:0}
.warnbox li{margin:3px 0}
img.chart{width:100%;border-radius:14px;background:#fff;
box-shadow:0 2px 10px rgba(22,36,63,.07);margin:12px 0 2px}
.fineprint{font-size:10px;color:#8a93a6;line-height:1.7;margin-top:4px}
footer{font-size:10px;color:#8a93a6;padding:18px 22px 26px;line-height:1.8;max-width:980px;margin:0 auto}
"""


def _fmt_pct(v, digits=1):
    return f"{v*100:+.{digits}f}%"


def aiae_card(aiae, reg):
    if not aiae:
        return ""
    rt = ""
    if "rt_val" in aiae:
        rt = (f"リアルタイム近似 <b>{aiae['rt_val']*100:.1f}%</b>"
              f" / 歴史Pct {aiae['rt_pct']*100:.0f}%")
    pred = ""
    if reg:
        pred = (f"<br>10Y期待 <b>{_fmt_pct(reg['pred'])}/年</b>"
                f" (R²={reg['r2']:.2f}, n={reg['n']})")
    return (f'<div class="card"><div class="lbl">AIAE 公式 {aiae["q_date"]}</div>'
            f'<div class="big">{aiae["q_val"]*100:.1f}<span class="unit">%</span></div>'
            f'<div class="small">{rt}{pred}</div></div>')


def top_panel(v1, hbc, holv):
    if v1 is None and hbc is None:
        return ""
    cards = ""
    if v1:
        pos = min(98, v1["total"] / 60 * 100)
        bk = v1_bucket_row(v1["total"])
        bkrow = (f"点灯時1Y期待 <b>{bk['cagr']}</b> / 勝率{bk['win']} (n={bk['n']},"
                 f" 2026-05レポート固定値)") if bk else ""
        cards += f"""
    <div class="card state" style="background:linear-gradient(135deg,{v1['color']},#7d1f1f)">
      <div class="lbl">V1 天井スコア</div>
      <div class="big">{v1['total']:.1f}</div>
      <div class="bar" style="background:rgba(255,255,255,.25)"><i style="width:{pos}%"></i><b></b></div>
      <div class="small">{v1['bucket']}<br>MD/M2 {v1['md']:.1f} + HB {v1['hb']:.0f} + AIAE {v1['ai']:.1f}<br>{bkrow}</div>
    </div>
    <div class="card"><div class="lbl">MD/M2 — FINRA {v1['mdm2']['date']}</div>
      <div class="big">{v1['mdm2']['val']:,.0f}</div>
      <div class="small">警戒4000 / 過去天井5000超<br>(2000:6366 / 2007:5690 / 2026-01:5703)</div></div>"""
    if hbc is not None and holv is not None:
        lv, name, color = holv
        cards += f"""
    <div class="card"><div class="lbl">HOクラスター判定</div>
      <div class="big" style="color:{color}">Lv{lv}<span class="unit"> {name.split(' ')[0]}</span></div>
      <div class="small">3M:{hbc['3m']} / 6M:{hbc['6m']} / 12M:{hbc['12m']}<br>
      Watch 6M≧8 / Warning 3M≧10 / Alarm 3M≧16</div></div>"""
    return f"""
<h2 class="sec top">天井側 — V1スコアモデル</h2>
<div class="grid">{cards}</div>
<div class="note"><b>V1モデルの限界(必読)</b>: 35点境界の較正は実効2イベント(Dotcom/GFC)。
HB過去分は目視集約シード、2026-06-12以降は自前ルール(2.8%・NYSE+Nasdaq合算・WSJ)。
AIAE項はFRED公式系列用に閾値50へ再較正(2026-03アンカー接続)。点灯時期待値は固定のバックテスト値であり、
HB/AIAEの定義不連続を含む。点推定ではなく方向性として読むこと。</div>"""


def exp_panel(reg, istats, res):
    if not reg and not istats:
        return ""
    cards = ""
    if istats and res:
        cur_bk = "21+" if res["density"] >= 21 else (
            "6-20" if res["density"] >= 6 else ("1-5" if res["density"] >= 1 else "0"))
        s21 = istats.get("21+")
        if s21:
            lit = "🔔 点灯中" if cur_bk == "21+" else "未点灯(参考値)"
            cards += f"""
    <div class="card"><div class="lbl">③パニック(密度21+)点灯時 — {lit}</div>
      <div class="big">{_fmt_pct(s21['m12'])}<span class="unit"> /12M</span></div>
      <div class="small">勝率{s21['w12']*100:.0f}% / 6M {_fmt_pct(s21['m6'])} (勝率{s21['w6']*100:.0f}%)
      / n={s21['n']}日<br>全履歴(2006-)から毎日再計算。独立イベントは実質3-4回</div></div>"""
        sc = istats.get(cur_bk)
        if sc and cur_bk != "21+":
            cards += f"""
    <div class="card"><div class="lbl">現在の密度バケット ({cur_bk}) の歴史統計</div>
      <div class="big">{_fmt_pct(sc['m12'])}<span class="unit"> /12M</span></div>
      <div class="small">勝率{sc['w12']*100:.0f}% / 6M {_fmt_pct(sc['m6'])} / n={sc['n']}日
      (毎日再計算)</div></div>"""
    if reg:
        cards += f"""
    <div class="card"><div class="lbl">AIAE 10年回帰 (現在値 {reg['cur']*100:.1f}%)</div>
      <div class="big">{_fmt_pct(reg['pred'])}<span class="unit"> /年×10Y</span></div>
      <div class="small">R²={reg['r2']:.2f} / n={reg['n']}四半期 (Z.1全史×S&amp;P500価格、毎日再計算)<br>
      配当除く価格リターン。サンプル重複あり・in-sample値</div></div>"""
    return f"""
<h2 class="sec exp">期待値モニター</h2>
<div class="grid">{cards}</div>
<div class="fineprint">※ 先行リターンは重複標本のため統計的独立性なし。AIAEのR²はin-sample。
V1点灯時期待のみバックテスト固定値(過去のMD/M2・HB月次系列が必要なため再計算非対応と明記)。</div>"""


def build_html(res, aiae=None, v1=None, hbc=None, holv=None, reg=None, istats=None):
    now_utc = dt.datetime.utcnow()
    now_jst = now_utc + dt.timedelta(hours=9)
    if res:
        name, color, advice = res["state"]
        ratio_s = f"{res['ratio']:.3f}" if res["ratio"] is not None else "—"
        grad = {"#2a7d4f": "linear-gradient(135deg,#1e8a5a,#136443)",
                "#cc8833": "linear-gradient(135deg,#d07f2e,#a05a14)",
                "#cc3333": "linear-gradient(135deg,#cd3a3a,#8b1f1f)"}.get(color, color)
        cards = f"""
  <div class="grid">
    <div class="card state" style="background:{grad}">
      <div class="lbl">現在のステート</div>
      <div class="big">{name}</div>
      <div class="small">{advice}</div>
    </div>
    <div class="card"><div class="lbl">クラスター密度</div>
      <div class="big">{res['density']}<span class="unit"> /63日</span></div>
      <div class="small">閾値: 警戒6 / パニック21</div></div>
    <div class="card"><div class="lbl">本日の比率 ({res['date']})</div>
      <div class="big">{ratio_s}</div>
      <div class="small">buy {res['buy']} / sell {res['sell']}</div></div>
    <div class="card"><div class="lbl">S&amp;P 500 DD ({res.get('spx_date','—')})</div>
      <div class="big">{res.get('dd', 0)*100:+.1f}<span class="unit">%</span></div>
      <div class="small">終値 {res.get('spx_close', 0):,.0f} / 252日高値比</div></div>
    {aiae_card(aiae, reg)}
  </div>"""
    else:
        cards = "<p>データ未取得。Actionsのログを確認してください。</p>"
    warn_html = ""
    if warnings:
        items = "".join(f"<li>{w}</li>" for w in warnings)
        warn_html = f'<div class="warnbox"><b>⚠ データ注意事項</b><ul>{items}</ul></div>'
    html = f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Monitor</title>
<style>{CSS}</style></head><body>
<header><h1>Market Monitor</h1>
<div class="sub">底値検出 × 天井警戒 の両面監視 / 最終更新 {now_jst:%Y-%m-%d %H:%M} JST
 / SEC EDGAR · WSJ · FRED · Yahoo</div></header>
<main>
<h2 class="sec">底側 — インサイダー密度ステート</h2>
{cards}
{top_panel(v1, hbc, holv)}
{exp_panel(reg, istats, res)}
{warn_html}
<img class="chart" src="chart_insider.png" alt="insider">
<img class="chart" src="chart_spx.png" alt="spx">
<img class="chart" src="chart_aiae.png" alt="aiae" onerror="this.style.display='none'">
</main>
<footer>定義: 件数比率 = Officer/DirectorのForm 4日次 buy/sell filings (NONDERIV P/S・filing単位・提出日)。
クラスター密度 = 過去63営業日の比率&gt;1.0日数。ステート: ①平常0-5 / ②警戒6-20(新規買い禁止) / ③パニック21+(分割買い候補)。
AIAE = Z.1株式時価 ÷ (株式時価+5部門負債)、公開ラグ2.5-5.5ヶ月、RT近似は株式部分のみS&amp;P500補正。
V1 = MD/M2スコア + HB6M信号数 + max(0, AIAE-50)。本ページは検証記録に基づく私的モニターであり投資助言ではない。</footer>
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
    update_breadth(spx)
    hbc = hb_counts()
    mdm2 = update_mdm2()
    v1 = compute_v1(mdm2, aiae, hbc)
    holv = ho_level(hbc)
    spxl = update_spx_long()
    reg = aiae_regression(aiae, spxl)
    istats = insider_bucket_stats(res.get("df") if res else None, spxl)
    if res:
        make_charts(res, aiae)
    build_html(res, aiae, v1, hbc, holv, reg, istats)
    log("dashboard generated")
    for w in warnings:
        log(f"WARN: {w}")


if __name__ == "__main__":
    main()
