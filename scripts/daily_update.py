# -*- coding: utf-8 -*-
"""
market-monitor 日次更新スクリプト
1. EDGAR日次インデックスから Form 4 を取得し、Officer/Director の P/S 件数を日次集計
2. S&P 500 を Stooq から取得し、252日高値比ドローダウンを計算
3. クラスター密度(過去63営業日の比率>1.0日数)とステートを判定
4. docs/ にダッシュボード(HTML + チャートPNG)を生成
5. シリコンサイクル(WSTS 3MMA YoY) と FINRA MD/M2 を自動取得
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
TIME_BUDGET_MIN = int(os.environ.get("TIME_BUDGET_MIN", "45"))
WORKERS = 5
START_TIME = time.time()
CLUSTER_WIN = 63
THRESH = 1.0
REQ_INTERVAL = 0.13

SEC_CONTACT = os.environ.get("SEC_CONTACT", "anonymous@example.com")
SEC_HEADERS = {
    "User-Agent": f"market-monitor {SEC_CONTACT}",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "*/*",
}
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
    with _rate_lock:
        wait = _last_req[0] + REQ_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_req[0] = time.time()


warnings = []


def log(msg):
    print(f"[{dt.datetime.utcnow():%H:%M:%S}] {msg}", flush=True)


def fetch(url, ok404=False, headers=None, tries=4, timeout=30):
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
    q = (day.month - 1) // 3 + 1
    idx_url = (f"https://www.sec.gov/Archives/edgar/daily-index/"
               f"{day.year}/QTR{q}/form.{day:%Y%m%d}.idx")
    r = fetch(idx_url, ok404=True)
    if r is None:
        return None
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
        last = dt.date.today() - dt.timedelta(days=80)
    target_end = dt.date.today() - dt.timedelta(days=2)
    day = last + dt.timedelta(days=1)
    done = added = 0
    while day <= target_end and done < MAX_DAYS_PER_RUN:
        if (time.time() - START_TIME) > TIME_BUDGET_MIN * 60:
            log(f"時間予算 {TIME_BUDGET_MIN}分に到達 — ここまでを保存して終了")
            break
        if day.weekday() < 5:
            try:
                res = process_day(day)
            except Exception as e:
                warnings.append(f"{day}: EDGAR取得エラー ({e}) — 次回再試行")
                log(f"  ERROR {day}: {e}")
                break
            if res is not None:
                row = pd.DataFrame([{"date": pd.Timestamp(day),
                                     "buy_filings": res[0],
                                     "sell_filings": res[1]}])
                df = pd.concat([df, row], ignore_index=True)
                df = df.drop_duplicates("date", keep="last").sort_values("date")
                df.to_csv(DATA_CSV, index=False, encoding="utf-8-sig")
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
    r = fetch("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
              "?range=2y&interval=1d", headers=BROWSER_HEADERS)
    j = r.json()["chart"]["result"][0]
    ts = j["timestamp"]
    cl = j["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"date": pd.to_datetime(ts, unit="s").normalize(),
                       "close": cl})
    return df.dropna()


def _spx_from_fred():
    r = fetch("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
              headers=BROWSER_HEADERS)
    df = pd.read_csv(io.StringIO(r.text))
    df = df.iloc[:, :2]
    df.columns = ["date", "close"]
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
# ============================================================
AIAE_CSV = os.path.join(ROOT, "data", "aiae.csv")
AIAE_EQ = ["NCBEILQ027S", "FBCELLQ027S"]
AIAE_DEBT = ["FGSDODNS", "CMDEBT", "BCNSDODNS",
             "DODFFSWCMI", "SLGSDODNS"]


def fred_series(sid):
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
    try:
        cols, detail = {}, []
        for sid in AIAE_EQ + AIAE_DEBT:
            s = fred_series(sid)
            if abs(s.iloc[-1]) > 1e6:
                s = s / 1000.0
            cols[sid] = s
            detail.append(f"{sid}=〜{s.index[-1].date()}:{s.iloc[-1]:,.0f}B")
        log("AIAE系列診断(十億$換算後): " + " | ".join(detail))
        z1 = pd.concat(cols.values(), axis=1).dropna()
        eq = z1[AIAE_EQ].sum(axis=1)
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
    if spx is not None and len(spx):
        spx = spx.sort_values("date")
        q_end = pd.Timestamp(out.index[-1]) + pd.offsets.QuarterEnd(0)
        base = spx[spx["date"] <= q_end]["close"]
        if len(base):
            scale = float(spx["close"].iloc[-1]) / float(base.iloc[-1])
            eq_adj = float(out["eq"].iloc[-1]) * scale
            res["rt_val"] = eq_adj / (eq_adj + float(out["debt"].iloc[-1]))
            res["rt_date"] = spx["date"].iloc[-1].date()
            res["rt_pct"] = float((out["aiae"] < res["rt_val"]).mean())
    return res


# ============================================================
# 2.6 天井側モジュール: WSJブレッドス -> 自前HO判定 / FINRA MD/M2 / V1スコア
# ============================================================
BREADTH_CSV = os.path.join(ROOT, "data", "breadth_daily.csv")
HB_CSV = os.path.join(ROOT, "data", "hb_signals.csv")
MDM2_CSV = os.path.join(ROOT, "data", "mdm2.csv")
HO_THRESH = 0.028


def update_breadth(spx):
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
        if row["issues"] is None:
            raise ValueError("issues traded が取得できません")
        bd = pd.concat([bd, pd.DataFrame([row])], ignore_index=True)
        bd = bd.drop_duplicates("date", keep="last").sort_values("date")
        bd.to_csv(BREADTH_CSV, index=False)
        log(f"breadth(WSJ): {row['date'].date()} NH={row['nh']:.0f} "
            f"NL={row['nl']:.0f} issues={row['issues']:.0f}")
    except Exception as e:
        warnings.append(f"WSJブレッドス取得失敗 ({e}) — 本日のHO判定スキップ")
        return bd
    if len(bd) >= 1 and spx is not None and len(spx) > 60:
        b = bd.iloc[-1]
        nh_pct = b["nh"] / b["issues"]
        nl_pct = b["nl"] / b["issues"]
        c1 = nh_pct > HO_THRESH and nl_pct > HO_THRESH
        c4 = b["nh"] <= 2 * b["nl"]
        spx_s = spx.sort_values("date")["close"]
        c2 = float(spx_s.iloc[-1]) > float(spx_s.iloc[-51])
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


MDRAW_CSV = os.path.join(ROOT, "data", "md_raw.csv")


FINRA_XLSX_FIXED = ("https://www.finra.org/sites/default/files/"
                    "2021-03/margin-statistics.xlsx")
FINRA_MARGIN_PAGE = ("https://www.finra.org/investors/learn-to-invest/"
                     "advanced-investing/margin-statistics")


def _fetch_finra_xlsx():
    """FINRA Excelを取得しDataFrame[date, md_mil]を返す。
    第1候補=固定URL / 第2候補=ページからxlsxリンク抽出"""
    import io as _io
    candidates = [FINRA_XLSX_FIXED]
    # ページからxlsxリンクも候補に追加
    try:
        from urllib.parse import urljoin
        html = fetch(FINRA_MARGIN_PAGE, headers={**BROWSER_HEADERS,
                     "Sec-Fetch-Dest": "document",
                     "Sec-Fetch-Mode": "navigate",
                     "Upgrade-Insecure-Requests": "1"}).text
        for link in re.findall(r'href=["\']([^"\']*?\.xlsx[^"\']*)["\']',
                               html, re.I):
            url = urljoin(FINRA_MARGIN_PAGE, link)
            if url not in candidates:
                candidates.append(url)
    except Exception:
        pass

    last_err = None
    for url in candidates:
        try:
            r = fetch(url, headers=BROWSER_HEADERS, timeout=40)
            content = r.content
            if len(content) < 10000:
                last_err = f"{url}: サイズ過小({len(content)}B)"
                continue
            xls = pd.ExcelFile(_io.BytesIO(content), engine="openpyxl")
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                if df.shape[1] < 2:
                    continue
                cols = [str(c) for c in df.columns]
                debit_col = next(
                    (c for c in cols[1:] if "debit" in c.lower()), None)
                if debit_col is None and (
                        "month" in cols[0].lower() or "year" in cols[0].lower()):
                    debit_col = cols[1]  # 添付構造: B列がDebit
                if debit_col is None:
                    continue
                out = df[[cols[0], debit_col]].copy()
                out.columns = ["date", "md_mil"]
                out["date"] = pd.to_datetime(out["date"], errors="coerce")
                out["md_mil"] = pd.to_numeric(
                    out["md_mil"].astype(str).str.replace(",", ""),
                    errors="coerce")
                out = out.dropna(subset=["date", "md_mil"]).sort_values("date")
                if len(out) >= 12 and out["md_mil"].iloc[-1] > 5e5:
                    log(f"FINRA Excel取得成功: {url[-40:]} "
                        f"シート'{sheet}' {len(out)}ヶ月 "
                        f"(最新{out['date'].iloc[-1]:%Y-%m}="
                        f"{out['md_mil'].iloc[-1]:,.0f}M)")
                    return out
            last_err = f"{url}: Debit列を特定できず"
        except Exception as e:
            last_err = f"{url}: {e}"
            continue
    raise RuntimeError(f"FINRA Excel全候補で失敗 ({last_err})")


def update_mdm2():
    """FINRAマージンデット(月次) + FRED M2 -> V1スケールのMD/M2
    優先順: ①FINRA公式Excel直接取得 ②md_raw.csv(手動の生Debit値)
            ③mdm2.csvキャッシュ"""
    md = None
    try:
        md = _fetch_finra_xlsx()
        # 取得成功したら md_raw.csv も更新しておく(次回フォールバック用の鮮度維持)
        try:
            bak = md.rename(columns={"md_mil": "debit_mil"})[["date", "debit_mil"]].copy()
            bak["date"] = bak["date"].dt.strftime("%Y-%m-%d")
            bak.to_csv(MDRAW_CSV, index=False)
        except Exception:
            pass
    except Exception as e:
        if os.path.exists(MDRAW_CSV):
            md = pd.read_csv(MDRAW_CSV, parse_dates=["date"]).rename(
                columns={"debit_mil": "md_mil"}).sort_values("date")
            log(f"FINRA Excel取得失敗 -> md_raw.csv使用 "
                f"(最新 {md['date'].iloc[-1].date()}: "
                f"{md['md_mil'].iloc[-1]:,.0f}M)")
            warnings.append(f"FINRA自動取得失敗({e}) — md_raw.csvキャッシュ使用")
        else:
            warnings.append(f"FINRA MD/M2取得失敗 ({e}) — キャッシュ使用 "
                            f"(data/md_raw.csv に月1回、生のDebit値を追記してください)")
    try:
        if md is None:
            raise ValueError("MDデータなし")
        m2 = fred_series("M2SL")  # 月次
        if abs(m2.iloc[-1]) > 1e6:  # API=百万$なら十億$へ正規化
            m2 = m2 / 1000.0
        m2m = m2.reindex(md.set_index("date").index, method="ffill")
        scaled = (md.set_index("date")["md_mil"] / m2m * 100).dropna()  # V1スケール
        out = pd.DataFrame({"mdm2_v1": scaled})
        out.index.name = "date"
        out.to_csv(MDM2_CSV)
        log(f"MD/M2: 最新 {out.index[-1].date()} = {scaled.iloc[-1]:.0f} (V1スケール)")
        return {"date": out.index[-1].date(), "val": float(scaled.iloc[-1])}
    except Exception as e:
        warnings.append(f"MD/M2計算失敗 ({e}) — 前回キャッシュを使用")
        if os.path.exists(MDM2_CSV):
            c = pd.read_csv(MDM2_CSV, parse_dates=["date"], index_col="date")
            return {"date": c.index[-1].date(), "val": float(c["mdm2_v1"].iloc[-1])}
        return None


def compute_v1(mdm2, aiae, hbc):
    if mdm2 is None or aiae is None or hbc is None:
        return None
    aiae_pct = aiae.get("rt_val", aiae["q_val"]) * 100
    s_md = max(0.0, (mdm2["val"] - 4000) / 50)
    s_hb = float(hbc["6m"])
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
    if os.path.exists(SPXLONG_CSV):
        c = pd.read_csv(SPXLONG_CSV, parse_dates=["date"])
        if c["date"].max() > pd.Timestamp.now() - pd.Timedelta(days=7):
            return c
    def _grab(url, min_rows):
        r = fetch(url, headers=BROWSER_HEADERS, timeout=60)
        j = r.json()["chart"]["result"][0]
        df = pd.DataFrame({"date": pd.to_datetime(j["timestamp"], unit="s").normalize(),
                           "close": j["indicators"]["quote"][0]["close"]}).dropna()
        df = df.drop_duplicates("date").sort_values("date")
        if len(df) < min_rows:
            raise ValueError(f"行数不足 ({len(df)})")
        return df
    try:
        p2 = int(dt.datetime.now().timestamp())
        try:
            df = _grab("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
                       f"?period1=-2208988800&period2={p2}&interval=1d", 5000)
            log(f"SPX長期(日次): {len(df)}行 ({df['date'].min().date()}〜)")
        except Exception as e1:
            log(f"日次フル履歴失敗 ({e1}) -> 月次にフォールバック")
            df = _grab("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"
                       "?range=max&interval=1mo", 600)
            log(f"SPX長期(月次): {len(df)}行 ({df['date'].min().date()}〜)")
        df.to_csv(SPXLONG_CSV, index=False)
        return df
    except Exception as e:
        warnings.append(f"SPX長期履歴の取得失敗 ({e}) — 期待値モニター一部停止")
        if os.path.exists(SPXLONG_CSV):
            return pd.read_csv(SPXLONG_CSV, parse_dates=["date"])
        return None


def aiae_regression(aiae, spxl):
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
    if df is None or len(df) < 300 or spxl is None:
        return None
    try:
        d = df.sort_values("date").copy()
        s = spxl.sort_values("date").reset_index(drop=True)
        d["date"] = pd.to_datetime(d["date"]).astype("datetime64[ns]")
        s["date"] = pd.to_datetime(s["date"]).astype("datetime64[ns]")
        gap = s["date"].diff().dt.days.median()
        h6, h12 = (6, 12) if gap > 15 else (126, 252)
        d = pd.merge_asof(d, s.rename(columns={"close": "px"}), on="date")
        d["idx"] = d["date"].map(
            {dt_: i for i, dt_ in enumerate(s["date"])})
        if gap > 15:
            d["idx"] = d["date"].map(
                lambda t: s["date"].searchsorted(t, side="right") - 1)
        d = d.dropna(subset=["idx", "px"])
        px = s["close"].values
        def fwd(i, h):
            i = int(i)
            return px[i + h] / px[i] - 1 if 0 <= i and i + h < len(px) else None
        d["f6"] = d["idx"].map(lambda i: fwd(i, h6))
        d["f12"] = d["idx"].map(lambda i: fwd(i, h12))
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
# 2.8 シリコンサイクル (WSTS 3MMA YoY) — 半導体底値検出
#   WSTS公式サイトからExcelを自動取得 -> 3MMA Worldwide抽出 -> YoY計算。
#   取得失敗時は data/silicon_3mma.csv キャッシュにフォールバック。
#   買いシグナル: YoY 3MMAが前月<=0->当月>0 へプラス転換 かつ
#                直前12Mに-5%以下まで沈んだ「本物の底」のみ (ノイズ除去)
#   -> 24Mホールド戦略 (1990-2023の9回, SOX検証で勝率100%/中央値+74%)
# ============================================================
SILICON_CSV = os.path.join(ROOT, "data", "silicon_3mma.csv")
WSTS_PAGE = "https://www.wsts.org/67/Historical-Billings-Report"


def _fetch_wsts_silicon():
    """WSTSサイトからExcelを自動取得し3MMA Worldwide売上を返す。
    戻り値: pd.Series (index=月初日, value=3MMA売上)。失敗時は例外。"""
    import io as _io
    from urllib.parse import urljoin
    # ページHTMLからxlsxリンクを抽出
    html = fetch(WSTS_PAGE, headers={**BROWSER_HEADERS,
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"}).text
    links = []
    for pat in [r'href=["\']([^"\']*?\.xlsx[^"\']*)["\']',
                r'(https?://[^\s"\'<>]*?\.xlsx)']:
        links += re.findall(pat, html, re.I)
    links = list(dict.fromkeys(links))
    links.sort(key=lambda u: (0 if re.search(r'histor|billing', u, re.I) else 1))
    if not links:
        raise ValueError("WSTSページに.xlsxリンクなし(JS生成/同意要の可能性)")

    last_err = None
    for link in links:
        url = urljoin(WSTS_PAGE, link)
        try:
            content = fetch(url, headers=BROWSER_HEADERS, timeout=40).content
            if len(content) < 10000:
                last_err = f"{url}: サイズ過小"; continue
            xls = pd.ExcelFile(_io.BytesIO(content), engine="openpyxl")
            mma = next((s for s in xls.sheet_names if "mma" in s.lower()), None)
            if mma is None:
                last_err = f"{url}: 3MMAシートなし"; continue
            ws = xls.parse(mma, header=None)
            recs, year = [], None
            for _, row in ws.iterrows():
                a = row.iloc[0]
                if pd.notna(a) and isinstance(a, (int, float)) and 1986 <= a <= 2035:
                    year = int(a); continue
                if pd.notna(a) and str(a).strip() == "Worldwide" and year:
                    for m in range(1, 13):
                        if m < len(row):
                            v = row.iloc[m]
                            if pd.notna(v) and isinstance(v, (int, float)) and v > 0:
                                recs.append((pd.Timestamp(f"{year}-{m:02d}-01"),
                                             float(v)))
            if len(recs) >= 100:
                s = pd.Series([v for _, v in recs],
                              index=[d for d, _ in recs]).sort_index()
                log(f"WSTS自動取得成功: {url[-45:]} {len(s)}ヶ月 "
                    f"(最新{s.index[-1]:%Y-%m})")
                return s
            last_err = f"{url}: 抽出{len(recs)}件と不足"
        except Exception as e:
            last_err = f"{url}: {e}"; continue
    raise RuntimeError(f"WSTS Excel取得失敗 ({last_err})")


def update_silicon():
    """WSTSシリコンサイクルを取得・計算。自動取得→CSVキャッシュの順"""
    s = None
    # ① WSTSサイトから自動取得
    try:
        s = _fetch_wsts_silicon()
        # 成功したらCSVキャッシュも更新(次回フォールバック用の鮮度維持)
        try:
            os.makedirs(os.path.dirname(SILICON_CSV), exist_ok=True)
            bak = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in s.index],
                                "sales_3mma": [round(v, 1) for v in s.values]})
            bak.to_csv(SILICON_CSV, index=False)
        except Exception:
            pass
    except Exception as e:
        # ② CSVキャッシュにフォールバック
        if os.path.exists(SILICON_CSV):
            df = pd.read_csv(SILICON_CSV, parse_dates=["date"]).sort_values("date")
            df = df[df["sales_3mma"] > 0]
            s = df.set_index("date")["sales_3mma"]
            log(f"WSTS自動取得失敗 -> silicon_3mma.csv使用 (最新{s.index[-1]:%Y-%m})")
            warnings.append(f"WSTS自動取得失敗({e}) — CSVキャッシュ使用")
        else:
            warnings.append(f"WSTS取得失敗({e}) かつCSVキャッシュなし — "
                            f"data/silicon_3mma.csvを配置してください")
            return None
    try:
        if s is None or len(s) < 13:
            warnings.append("シリコンサイクル: データ13ヶ月未満でYoY計算不可")
            return None
        # YoY (3MMAは既に計算済みなので前年同月比のみ)
        yoy = (s / s.shift(12) - 1) * 100
        yoy = yoy.dropna()
        if len(yoy) < 2:
            return None
        # 0%プラス転換 (買いシグナル) を検出。
        # ノイズ除去フィルタ: 直前12ヶ月以内にYoYが-5%以下まで沈んだ
        # 「本物のサイクル底」からの転換のみ採用 (2008-02/2013-05等の
        # ゼロ近傍ダマシを除外。バックテスト9回と整合)
        DEPTH_THRESHOLD = -5.0
        DEPTH_LOOKBACK = 12
        cross_dates = []
        for i in range(1, len(yoy)):
            if yoy.iloc[i-1] <= 0 and yoy.iloc[i] > 0:
                start = max(0, i - DEPTH_LOOKBACK)
                recent_min = yoy.iloc[start:i].min()
                if recent_min <= DEPTH_THRESHOLD:
                    cross_dates.append(yoy.index[i])

        cur_yoy = float(yoy.iloc[-1])
        cur_date = yoy.index[-1]
        prev_yoy = float(yoy.iloc[-2])
        # モメンタム (3ヶ月変化)
        mom3 = float(yoy.iloc[-1] - yoy.iloc[-4]) if len(yoy) >= 4 else None

        # 最後の買いシグナルからの経過月数
        last_signal = cross_dates[-1] if cross_dates else None
        months_since = None
        in_hold_window = False
        hold_exit_date = None
        if last_signal:
            months_since = ((cur_date.year - last_signal.year) * 12
                            + (cur_date.month - last_signal.month))
            # 24Mホールド戦略の保有期間内か
            in_hold_window = months_since < 24
            # 売却予定月 (シグナル+24M)
            exit_y = last_signal.year + (last_signal.month - 1 + 24) // 12
            exit_m = (last_signal.month - 1 + 24) % 12 + 1
            hold_exit_date = f"{exit_y}-{exit_m:02d}"

        # 当月が買いシグナル点灯か (直近でプラス転換したばかり)
        just_fired = (last_signal is not None
                      and last_signal == cur_date)

        # データ鮮度チェック (WSTSは約2ヶ月ラグ。3ヶ月以上古ければ警告)
        now = pd.Timestamp(dt.date.today())
        data_age_months = ((now.year - cur_date.year) * 12
                           + (now.month - cur_date.month))
        if data_age_months > 3:
            warnings.append(f"シリコンサイクル: データが{data_age_months}ヶ月前"
                            f"({cur_date:%Y-%m})で停止 — WSTS Blue Bookで更新を")

        # 状態判定
        if just_fired:
            state = ("🟢 買いシグナル点灯", "#1f9d68",
                     f"YoYが0%をプラス転換({cur_yoy:+.1f}%)。"
                     f"24Mホールド戦略のエントリー (過去9回勝率100%/中央値+74%)")
        elif in_hold_window:
            state = ("🟢 ホールド期間中", "#1f9d68",
                     f"{last_signal:%Y-%m}買いシグナルから{months_since}ヶ月経過。"
                     f"{hold_exit_date}まで保有 (24Mホールド戦略)")
        elif cur_yoy < 0:
            state = ("🔵 下降局面 (次の底待ち)", "#3a6ea5",
                     f"YoY {cur_yoy:+.1f}%。マイナス圏。"
                     f"再びプラス転換で次の買いシグナル")
        else:
            state = ("⚪ 中立", "#93a0b8",
                     f"YoY {cur_yoy:+.1f}%。直近シグナル{last_signal:%Y-%m}は"
                     f"{months_since}ヶ月前(24M超で保有終了)")

        # パーセンタイル (現在のYoYが過去全体の何%地点か)
        pctile = float((yoy < cur_yoy).sum() / len(yoy) * 100)

        # チャート生成
        make_silicon_chart(yoy, cross_dates, s)

        return {
            "date": cur_date,
            "yoy": cur_yoy,
            "prev_yoy": prev_yoy,
            "mom3": mom3,
            "pctile": pctile,
            "cross_dates": cross_dates,
            "n_signals": len(cross_dates),
            "last_signal": last_signal,
            "months_since": months_since,
            "in_hold_window": in_hold_window,
            "hold_exit_date": hold_exit_date,
            "just_fired": just_fired,
            "state": state,
            "data_age_months": data_age_months,
        }
    except Exception as e:
        log(f"シリコンサイクル計算エラー: {e}")
        warnings.append(f"シリコンサイクル計算失敗 ({e})")
        return None


def make_silicon_chart(yoy, cross_dates, sales):
    """シリコンサイクルYoY + 買いシグナル色分けチャート"""
    try:
        y = yoy[yoy.index >= pd.Timestamp("1994-01-01")]
        fig, ax = plt.subplots(figsize=(11, 4.6))
        # 0ライン基準の塗り分け
        ax.fill_between(y.index, y.values, 0, where=y.values >= 0,
                        color="#1f9d68", alpha=0.18)
        ax.fill_between(y.index, y.values, 0, where=y.values < 0,
                        color="#cc3333", alpha=0.15)
        ax.plot(y.index, y.values, color="#2b5aa0", lw=1.6)
        ax.axhline(0, color="#16243f", lw=0.9)

        # 買いシグナル(0%プラス転換) を緑の縦線+マーカー
        for cd in cross_dates:
            if cd >= y.index[0]:
                yv = float(yoy.loc[cd])
                ax.axvline(cd, color="#1f9d68", lw=1, ls="--", alpha=0.55)
                ax.scatter([cd], [yv], color="#1f9d68", s=70, zorder=5,
                           edgecolors="white", lw=1.2, marker="^")

        # 24Mホールド期間を薄緑の帯で表示
        for cd in cross_dates:
            if cd >= y.index[0]:
                exit_d = cd + pd.DateOffset(months=24)
                ax.axvspan(cd, min(exit_d, y.index[-1]),
                           color="#1f9d68", alpha=0.05)

        # 現在値ラベル
        cur = y.iloc[-1]
        ax.scatter([y.index[-1]], [cur], color="#cc3333", s=55, zorder=6)
        ax.annotate(f"{cur:+.0f}%", xy=(y.index[-1], cur),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#cc3333",
                    va="center")

        ax.set_ylabel("YoY 3MMA (%)")
        ax.set_title("シリコンサイクル (WSTS 3MMA 前年比) — ▲緑=買いシグナル(0%転換)/緑帯=24M保有")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(DOCS, "chart_silicon.png"), dpi=110)
        plt.close()
        return True
    except Exception as e:
        log(f"シリコンサイクルチャート失敗: {e}")
        return False


def silicon_panel(sil):
    """シリコンサイクルのカードパネルHTML"""
    if sil is None:
        return ""
    name, color, advice = sil["state"]
    grad = {"#1f9d68": "linear-gradient(135deg,#1f9d68,#0d5436)",
            "#3a6ea5": "linear-gradient(135deg,#3a6ea5,#16314f)"}.get(color)
    if grad:
        st_open = f'<div class="card state" style="background:{grad}">'
    else:
        st_open = '<div class="card">'

    # モメンタム表示
    mom_s = ""
    if sil["mom3"] is not None:
        arrow = "↑加速" if sil["mom3"] > 0 else "↓減速"
        mom_s = f" / Mom(3M) {sil['mom3']:+.1f} {arrow}"

    # 次のシグナル/保有状態の補足
    if sil["in_hold_window"]:
        hold_s = (f"保有中: {sil['last_signal']:%Y-%m}買い → "
                  f"{sil['hold_exit_date']}売り予定")
    elif sil["yoy"] < 0:
        hold_s = "マイナス圏。再プラス転換で次の買い場"
    else:
        hold_s = f"直近買い{sil['last_signal']:%Y-%m}は保有期間終了"

    # 最後の3つの買いシグナル
    recent_signals = sil["cross_dates"][-3:] if sil["cross_dates"] else []
    sig_list = " / ".join(f"{d:%Y-%m}" for d in recent_signals)

    return f"""
<h2 class="sec">底側 — シリコンサイクル (半導体)</h2>
<div class="grid">
    {st_open}
      <div class="lbl">現在のステート</div>
      <div class="big" style="font-size:22px">{name}</div>
      <div class="small">{advice}</div>
    </div>
    <div class="card"><div class="lbl">YoY 3MMA ({sil['date']:%Y-%m})</div>
      <div class="big">{sil['yoy']:+.1f}<span class="unit">%</span></div>
      <div class="small">歴史Pct {sil['pctile']:.0f}%{mom_s}<br>{hold_s}</div></div>
    <div class="card"><div class="lbl">買いシグナル履歴 (0%プラス転換)</div>
      <div class="big">{sil['n_signals']}<span class="unit">回</span></div>
      <div class="small">直近: {sig_list}<br>
      24Mホールドで過去勝率100%/中央値+74%</div></div>
</div>
<img class="chart" src="chart_silicon.png" alt="silicon">
<div class="note"><b>シリコンサイクル戦略</b>: WSTS世界半導体売上の3ヶ月移動平均YoYが
0%をプラス転換した月にSOX/半導体をロング、24ヶ月後に機械的に売却。
過去9回(1997-2023)で勝率100%・中央値+74%。エントリーシグナルは明確だが
天井検出は困難(別途検証)。データは月次・約2ヶ月ラグ。投資助言ではない。</div>"""


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
    df["excess"] = (df["ratio"] - 1.0).clip(lower=0).fillna(0)
    df["c_score"] = df["excess"].rolling(CLUSTER_WIN, min_periods=1).sum()
    out["df"] = df
    last = df.iloc[-1]
    out["date"] = last["date"].date()
    out["ratio"] = float(last["ratio"]) if pd.notna(last["ratio"]) else None
    out["density"] = int(last["density"])
    out["c_score"] = float(last["c_score"])
    out["c_peak_63d"] = float(df["c_score"].tail(CLUSTER_WIN).max())
    out["buy"] = int(last["buy_filings"])
    out["sell"] = int(last["sell_filings"])
    c = out["c_score"]
    if c >= 15:
        out["state"] = ("③ 歴史的水準", "#1f9d68",
                        f"歴史的買い場(C={c:.2f}≥15)。GFC/COVID級のシグナル。"
                        f"分割買い・コアポジション構築の好機")
    elif c >= 7:
        out["state"] = ("③ 点灯", "#1f9d68",
                        f"案C点灯(C={c:.2f}≥7)。分割買い候補圏。"
                        f"12M期待+30.5%/勝率100%(検証20年・n=357)")
    elif c >= 4:
        out["state"] = ("② 中立(注意)", "#cc8833",
                        f"立ち上がり監視域(C={c:.2f})。閾値7まで残り{7-c:.2f}。"
                        f"買い候補のWatchlist整理タイミング")
    else:
        out["state"] = ("① 沈黙", "#2a7d4f",
                        f"案C沈黙(C={c:.2f}<4)。通常運用。"
                        f"63日内ピーク C={out['c_peak_63d']:.2f}")
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;700;800&family=Noto+Sans+JP:wght@400;500;700;900&family=Cormorant+Garamond:wght@600&display=swap');
:root{--gold:#d4af6a;--gold2:#f0d9a8;--ink:#e9edf6;--mut:#93a0b8;--ok:#2fae74;
--warn:#e0954a;--bad:#e25656;--line:rgba(255,255,255,.09)}
*{box-sizing:border-box}
body{font-family:'Noto Sans JP','Inter',sans-serif;margin:0;color:var(--ink);
background:#0c1120;background-image:
radial-gradient(900px 480px at 85% -10%, rgba(63,94,168,.28), transparent 60%),
radial-gradient(700px 420px at -10% 25%, rgba(122,79,208,.16), transparent 55%),
radial-gradient(800px 500px at 50% 110%, rgba(212,175,106,.07), transparent 60%)}
header{padding:34px 24px 26px;border-bottom:1px solid var(--line);
background:linear-gradient(120deg, rgba(16,24,46,.9), rgba(24,38,72,.55));
backdrop-filter:blur(6px)}
header .wrap{max-width:1020px;margin:0 auto}
header h1{font-size:24px;margin:0;font-weight:900;letter-spacing:.04em;
background:linear-gradient(90deg,#fff,var(--gold2) 65%,var(--gold));
-webkit-background-clip:text;background-clip:text;color:transparent}
header .sub{font-size:11.5px;color:var(--mut);margin-top:8px;font-weight:500;letter-spacing:.03em}
header .rule{width:64px;height:2px;background:linear-gradient(90deg,var(--gold),transparent);
margin-top:14px;border-radius:2px}
main{max-width:1020px;margin:0 auto;padding:22px 18px 36px}
h2.sec{font-size:13px;margin:26px 0 12px;font-weight:900;letter-spacing:.1em;
color:var(--gold2);display:flex;align-items:center;gap:10px;text-transform:none}
h2.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:13px}
.card{background:linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
border:1px solid var(--line);border-radius:16px;padding:16px 17px;
box-shadow:0 10px 30px rgba(3,7,18,.45);backdrop-filter:blur(5px);
transition:transform .15s ease}
.card:hover{transform:translateY(-2px)}
.card.state{border:none;color:#fff}
.lbl{font-size:10px;color:var(--mut);font-weight:700;letter-spacing:.1em;text-transform:uppercase}
.card.state .lbl{color:rgba(255,255,255,.8)}
.big{font-size:32px;font-weight:800;font-family:'Inter','Noto Sans JP',sans-serif;
margin:6px 0 4px;line-height:1.05;text-shadow:0 2px 14px rgba(0,0,0,.35)}
.unit{font-size:13px;font-weight:600;color:var(--mut)}
.card.state .unit{color:rgba(255,255,255,.75)}
.small{font-size:11px;color:var(--mut);line-height:1.7}
.card.state .small{color:rgba(255,255,255,.88)}
.bar{height:8px;border-radius:99px;background:rgba(0,0,0,.35);margin:10px 0 5px;position:relative;overflow:visible}
.bar>i{position:absolute;left:0;top:0;bottom:0;border-radius:99px;
background:linear-gradient(90deg,#2fae74,#e6c34a 45%,#e25656 70%);
box-shadow:0 0 12px rgba(230,195,74,.35)}
.bar>b{position:absolute;top:-3px;bottom:-3px;width:2px;background:#fff;opacity:.85;left:58.3%}
.note{background:rgba(63,94,168,.13);border:1px solid rgba(120,150,210,.25);border-radius:14px;
padding:12px 16px;font-size:11.5px;line-height:1.85;margin:15px 0;color:#c6d2e8}
.warnbox{background:rgba(212,160,70,.1);border:1px solid rgba(212,175,106,.3);border-radius:14px;
padding:12px 16px;font-size:11.5px;margin:15px 0;color:#e8d3a8}
.warnbox ul{margin:6px 0 0 17px;padding:0}
.warnbox li{margin:4px 0}
img.chart{width:100%;border-radius:16px;background:#fff;padding:6px;
border:1px solid var(--line);box-shadow:0 10px 30px rgba(3,7,18,.45);margin:14px 0 2px}
.fineprint{font-size:10px;color:#71809c;line-height:1.8;margin-top:6px}
footer{font-size:10px;color:#71809c;padding:22px 24px 30px;line-height:1.9;
max-width:1020px;margin:0 auto;border-top:1px solid var(--line)}
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


def lookback_v1(aiae):
    if not aiae:
        return []
    now = pd.Timestamp(dt.date.today())
    targets = [("1日前", 1, "d"), ("1週間前", 7, "d"), ("1ヶ月前", 30, "d"),
               ("半年前", 182, "d"), ("1年前", 365, "d")]
    hist = None
    hist_f = os.path.join(ROOT, "data", "score_history.csv")
    if os.path.exists(hist_f):
        hist = pd.read_csv(hist_f, parse_dates=["date"]).dropna(subset=["v1"])
        hist = hist.set_index("date")["v1"].sort_index()
    monthly = None
    if os.path.exists(MDM2_CSV):
        mm = pd.read_csv(MDM2_CSV, parse_dates=["date"]).set_index("date")["mdm2_v1"]
        hb = pd.read_csv(HB_CSV, parse_dates=["date"]) if os.path.exists(HB_CSV) else None
        q = aiae["q"]["aiae"]
        m_rows = []
        for d, mv in mm.items():
            h6 = (hb[(hb["date"] > d - pd.Timedelta(days=182)) &
                     (hb["date"] <= d)]["signals"].sum() if hb is not None else 0)
            av = q[q.index <= d]
            if not len(av):
                continue
            v1 = (max(0, (mv - 4000) / 50) + float(h6)
                  + max(0, float(av.iloc[-1]) * 100 - 50))
            m_rows.append((d, v1))
        if m_rows:
            monthly = pd.DataFrame(m_rows, columns=["date", "v1"]).set_index("date")["v1"]
    out = []
    for label, n, _ in targets:
        target = now - pd.Timedelta(days=n)
        val, src_name = None, None
        if hist is not None and len(hist):
            mask = hist.index <= target
            if mask.any():
                val = float(hist[mask].iloc[-1])
                src_name = "日次実測"
        if val is None and monthly is not None and len(monthly):
            mask = monthly.index <= target
            if mask.any():
                val = float(monthly[mask].iloc[-1])
                src_name = "月次再構成"
        out.append((label, val, src_name, target.strftime("%Y-%m-%d")))
    return out


def lookback_panel(aiae, v1_now):
    items = lookback_v1(aiae)
    if not items:
        return ""
    if v1_now is None:
        return ""
    cur = v1_now["total"]
    cells = ""
    for label, val, src_name, asof in items:
        if val is None:
            cells += f"""
      <div class="lb-cell"><div class="lb-lbl">{label}</div>
        <div class="lb-val" style="color:var(--mut)">—</div>
        <div class="lb-sub">{asof} データなし</div></div>"""
            continue
        diff = cur - val
        arrow = "↗" if diff > 0.5 else ("↘" if diff < -0.5 else "→")
        dc = "#e25656" if diff > 0.5 else ("#7ce3ae" if diff < -0.5 else "var(--mut)")
        tag = "実測" if src_name == "日次実測" else "再構成"
        cells += f"""
      <div class="lb-cell"><div class="lb-lbl">{label}</div>
        <div class="lb-val">{val:.1f}</div>
        <div class="lb-sub" style="color:{dc}">{arrow} 差 {diff:+.1f}<br>
        <span style="color:var(--mut);font-size:9px">{tag}・{asof}</span></div></div>"""
    css = """<style>
.lb-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-top:12px}
.lb-cell{background:rgba(255,255,255,.04);border:1px solid var(--line);
border-radius:11px;padding:10px 8px;text-align:center}
.lb-lbl{font-size:10px;color:var(--mut);font-weight:700;letter-spacing:.06em}
.lb-val{font-size:22px;font-weight:800;font-family:'Inter',sans-serif;margin:4px 0}
.lb-sub{font-size:10px;line-height:1.5}
@media(max-width:640px){.lb-grid{grid-template-columns:repeat(2,1fr)}}
</style>"""
    return f"""
{css}
<h2 class="sec">V1スコア — 時系列スナップショット</h2>
<div class="card" style="padding:14px 17px">
  <div class="small" style="margin-bottom:6px">現在 <b style="color:var(--ink);font-size:20px">{cur:.1f}</b> との比較。
  実測=2026-06-12以降の日次蓄積 / 再構成=md_raw×M2+HBシード+AIAE公式の月末値</div>
  <div class="lb-grid">{cells}</div>
</div>"""


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
    <div class="card state" style="background:linear-gradient(135deg,{v1['color']},#5e1414);box-shadow:0 10px 35px rgba(226,86,86,.25)">
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


def exp_panel(reg, istats, res, v1):
    cards = ""
    c = res["c_score"] if res else 0
    if c >= 15:
        cards += f"""
    <div class="card state" style="background:linear-gradient(135deg,#1f9d68,#0d5436);box-shadow:0 10px 35px rgba(31,157,104,.3)">
      <div class="lbl">① 底側案C — 🚨 歴史的水準 (C={c:.2f}≥15)</div>
      <div class="big">+33.0%<span class="unit"> /1Y期待</span></div>
      <div class="small">勝率100% / 累積: 3Y+56% / 5Y+106% / 10Y+240%(年率+13%)<br>
      n=277日(GFC/欧州/COVID級, 20年検証)</div></div>"""
    elif c >= 7:
        cards += f"""
    <div class="card state" style="background:linear-gradient(135deg,#1f9d68,#0d5436)">
      <div class="lbl">① 底側案C — 🔔 点灯中 (C={c:.2f}≥7)</div>
      <div class="big">+30.5%<span class="unit"> /1Y期待</span></div>
      <div class="small">勝率100% / 累積: 3Y+57% / 5Y+102% / 10Y+242%(年率+13%)<br>
      n=357日(4独立イベント, 20年検証・偽点灯ゼロ)</div></div>"""
    elif c >= 4:
        cards += f"""
    <div class="card" style="background:linear-gradient(135deg,#e0954a,#8a4d10);color:#fff;border:none">
      <div class="lbl" style="color:rgba(255,255,255,.8)">① 底側案C — ⚠ 中立(立ち上がり監視)</div>
      <div class="big">C={c:.2f}<span class="unit"> /4≤C&lt;7</span></div>
      <div class="small" style="color:rgba(255,255,255,.88)">閾値7まで残り{7-c:.2f}<br>
      点灯したら12M期待+30.5%/勝率100%(参考)</div></div>"""
    else:
        cards += f"""
    <div class="card" style="opacity:.85">
      <div class="lbl">① 底側案C</div>
      <div class="big" style="color:var(--mut);font-size:24px">沈黙</div>
      <div class="small">現在C={c:.2f} (閾値4で中立, 7で点灯)<br>
      63日内ピーク: {res['c_peak_63d']:.2f} / 参考A密度: {res['density']}/63日<br>
      点灯時の歴史期待: 12M+30.5%/勝率100% (n=357)</div></div>"""
    lit_t = v1 is not None and v1["total"] >= 35
    if lit_t:
        bk = v1_bucket_row(v1["total"])
        cards += f"""
    <div class="card state" style="background:linear-gradient(135deg,#e25656,#5e1414)">
      <div class="lbl">② V1天井モデル — 🚨 点灯中 (スコア{v1['total']:.1f})</div>
      <div class="big">{bk['cagr']}<span class="unit"> /1Y期待</span></div>
      <div class="small">勝率{bk['win']} / n={bk['n']} — {bk['name']}<br>
      2026-05バックテスト固定値 (実効2イベント、点推定でなく方向性)</div></div>"""
    else:
        sc = f"{v1['total']:.1f}" if v1 else "—"
        cards += f"""
    <div class="card" style="opacity:.8">
      <div class="lbl">② V1天井モデル</div>
      <div class="big" style="color:var(--mut);font-size:24px">点灯なし</div>
      <div class="small">現在スコア {sc} — 点灯条件: 35+ (真の天井域)</div></div>"""
    if reg:
        cum = (1 + reg["pred"]) ** 10 - 1
        if cum >= 0.20:
            mood = "🟢 強気"
            style = 'class="card state" style="background:linear-gradient(135deg,#1f9d68,#0d5436)"'
        elif cum <= -0.10:
            mood = "🔴 弱気"
            style = 'class="card state" style="background:linear-gradient(135deg,#e25656,#5e1414)"'
        else:
            mood = "⚪ 中立"
            style = 'class="card"'
        cards += f"""
    <div {style}>
      <div class="lbl">③ AIAE 10年回帰 (現在値 {reg['cur']*100:.1f}%) — {mood}</div>
      <div class="big">{_fmt_pct(cum)}<span class="unit"> /10年累積</span></div>
      <div class="small">年率換算 {_fmt_pct(reg['pred'])}/年 / R²={reg['r2']:.2f} / n={reg['n']}四半期<br>
      Z.1全史×S&amp;P500価格・毎日再計算。配当除く・in-sample<br>
      判定: 累積+20%以上=強気 / -10%以下=弱気 / 中間=中立</div></div>"""
    if not cards:
        return ""
    return f"""
<h2 class="sec">期待値モニター</h2>
<div class="grid">{cards}</div>
<div class="fineprint">※ ①②は点灯時のみ期待値を表示(未点灯時の大きな数字は誤解を招くため)。
①②が同時点灯した場合は歴史的に2008年10月型(暴落進行中の底値買い局面)であり、①の分割買いルールを優先しつつ②の継続下落リスクを併記する。
先行リターンは重複標本のため統計的独立性なし。</div>"""


def make_v1_chart(aiae):
    try:
        if not (os.path.exists(MDM2_CSV) and os.path.exists(HB_CSV) and aiae):
            return False
        mm = pd.read_csv(MDM2_CSV, parse_dates=["date"]).set_index("date")["mdm2_v1"]
        hb = pd.read_csv(HB_CSV, parse_dates=["date"])
        q = aiae["q"]["aiae"]
        rows = []
        for d, mv in mm.items():
            hb6 = hb[(hb["date"] > d - pd.Timedelta(days=182)) &
                     (hb["date"] <= d)]["signals"].sum()
            av = q[q.index <= d]
            if not len(av):
                continue
            v1 = max(0, (mv - 4000) / 50) + hb6 + max(0, float(av.iloc[-1]) * 100 - 50)
            rows.append((d, v1, mv))
        m = pd.DataFrame(rows, columns=["date", "v1", "mdm2"]).sort_values("date")
        hist_f = os.path.join(ROOT, "data", "score_history.csv")
        h = (pd.read_csv(hist_f, parse_dates=["date"]).dropna(subset=["v1"])
             if os.path.exists(hist_f) else pd.DataFrame(columns=["date", "v1", "mdm2"]))
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})
        a1.axhspan(35, 50, color="#cd3a3a", alpha=.10)
        a1.axhspan(50, 100, color="#7d1f1f", alpha=.12)
        a1.axhline(35, color="#cd3a3a", lw=1, ls="--", alpha=.8)
        a1.plot(m["date"], m["v1"], "o-", color="#2b5aa0", lw=1.8, ms=5,
                label="月次再構成 (md_raw×M2 + HBシード + AIAE公式)")
        if len(h):
            a1.plot(h["date"], h["v1"], "-", color="#cd3a3a", lw=2,
                    label="日次実測 (2026-06-12〜)")
        a1.set_ylim(0, max(60, m["v1"].max() + 8))
        a1.set_ylabel("V1スコア")
        a1.set_title("V1天井スコアとMD/M2の推移 (2025-12〜 / 35=真の天井境界)")
        a1.legend(fontsize=8, loc="lower right")
        a1.grid(alpha=.25)
        a2.axhline(4000, color="#cc8833", lw=1, ls="--", alpha=.7)
        a2.axhline(5000, color="#cd3a3a", lw=1, ls="--", alpha=.7)
        a2.plot(m["date"], m["mdm2"], "o-", color="#1e8a5a", lw=1.8, ms=5)
        if len(h) and "mdm2" in h.columns and h["mdm2"].notna().any():
            a2.plot(h["date"], h["mdm2"], "-", color="#1e8a5a", lw=1.5, alpha=.6)
        a2.set_ylabel("MD/M2")
        a2.grid(alpha=.25)
        a2.annotate("4000=警戒 / 5000=過去天井圏", xy=(0.01, 0.05),
                    xycoords="axes fraction", fontsize=8, color="#666")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "chart_v1.png"), dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        log(f"V1チャート作成失敗: {e}")
        return False


def make_v1_long_chart(aiae):
    try:
        if not (os.path.exists(MDM2_CSV) and os.path.exists(SPXLONG_CSV) and aiae):
            return False
        mm = pd.read_csv(MDM2_CSV, parse_dates=["date"]).set_index("date")["mdm2_v1"]
        if (mm.index[-1] - mm.index[0]).days < 730:
            return False
        q = aiae["q"]["aiae"]
        core, full = [], []
        hb = pd.read_csv(HB_CSV, parse_dates=["date"]) if os.path.exists(HB_CSV) else None
        for d, mv in mm.items():
            av = q[q.index <= d]
            if not len(av):
                continue
            c = max(0, (mv - 4000) / 50) + max(0, float(av.iloc[-1]) * 100 - 50)
            core.append((d, c))
            if hb is not None and d >= pd.Timestamp("2025-12-01"):
                h6 = hb[(hb["date"] > d - pd.Timedelta(days=182)) &
                        (hb["date"] <= d)]["signals"].sum()
                full.append((d, c + h6))
        cdf = pd.DataFrame(core, columns=["date", "v1"]).set_index("date")
        fdf = pd.DataFrame(full, columns=["date", "v1"]).set_index("date") if full else None
        spxl = pd.read_csv(SPXLONG_CSV, parse_dates=["date"])
        sp = spxl[spxl["date"] >= cdf.index[0]].set_index("date")["close"]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11.5, 7.6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2]})
        a1.semilogy(sp.index, sp.values, color="#16243f", lw=1.3, label="S&P 500 (左軸, log)")
        a1.set_ylabel("S&P 500")
        a1.set_title("S&P 500 × V1コアスコア × MD/M2 (1997〜)")
        b1 = a1.twinx()
        b1.fill_between(cdf.index, cdf["v1"], color="#cd3a3a", alpha=.30,
                        label="V1コア (MD/M2+AIAE, HB除く)")
        if fdf is not None and len(fdf):
            b1.plot(fdf.index, fdf["v1"], color="#8b0000", lw=2.2,
                    label="フルV1 (HB込み, 2025-12〜)")
        b1.axhline(35, color="#cd3a3a", lw=1, ls="--", alpha=.7)
        b1.set_ylabel("V1スコア (右軸)")
        b1.set_ylim(0, max(60, cdf["v1"].max() + 8))
        ln1, lb1 = a1.get_legend_handles_labels()
        ln2, lb2 = b1.get_legend_handles_labels()
        a1.legend(ln1 + ln2, lb1 + lb2, fontsize=8, loc="upper left")
        a2.plot(mm.index, mm.values, color="#1e8a5a", lw=1.5)
        a2.axhline(4000, color="#cc8833", lw=1, ls="--", alpha=.7)
        a2.axhline(5000, color="#cd3a3a", lw=1, ls="--", alpha=.7)
        a2.set_ylabel("MD/M2")
        a2.grid(alpha=.25)
        for d, lab in [("2000-03-01", "2000"), ("2007-07-01", "2007"),
                       ("2021-10-01", "2021"), ("2026-04-01", "2026")]:
            t = pd.Timestamp(d)
            if t in mm.index:
                a2.annotate(lab, xy=(t, mm[t]), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=8, color="#8b0000", fontweight="bold")
        a2.annotate("注: HB系列は2025-12以降のみ存在。長期線はHBを除くコアスコア",
                    xy=(0.01, 0.04), xycoords="axes fraction", fontsize=8, color="#666")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(DOCS, "chart_v1_long.png"), dpi=110)
        plt.close(fig)
        return True
    except Exception as e:
        log(f"長期チャート作成失敗: {e}")
        return False


def build_html(res, aiae=None, v1=None, hbc=None, holv=None, reg=None,
               istats=None, sil=None):
    now_utc = dt.datetime.utcnow()
    now_jst = now_utc + dt.timedelta(hours=9)
    if res:
        name, color, advice = res["state"]
        ratio_s = f"{res['ratio']:.3f}" if res["ratio"] is not None else "—"
        grad = {"#cc8833": "linear-gradient(135deg,#e0954a,#8a4d10)",
                "#1f9d68": "linear-gradient(135deg,#1f9d68,#0d5436)"}.get(color)
        if grad:
            st_open = f'<div class="card state" style="background:{grad}">'
        else:
            st_open = '<div class="card">'
        cards = f"""
  <div class="grid">
    {st_open}
      <div class="lbl">現在のステート</div>
      <div class="big">{name}</div>
      <div class="small">{advice}</div>
    </div>
    <div class="card"><div class="lbl">案C スコア (Σ超過分/63日)</div>
      <div class="big">{res['c_score']:.2f}</div>
      <div class="small">沈黙&lt;4 / 中立4-7 / 🔔点灯≥7 / 🚨歴史的≥15<br>
      63日内ピーク: {res['c_peak_63d']:.2f} / 参考A密度: {res['density']}/63日</div></div>
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
<header><div class="wrap"><h1>MARKET MONITOR</h1>
<div class="sub">底値検出 × 天井警戒 の両面監視 — 最終更新 {now_jst:%Y-%m-%d %H:%M} JST
 / SEC EDGAR · WSJ · FRED · Yahoo · WSTS</div><div class="rule"></div></div></header>
<main>
<h2 class="sec">底側 — インサイダー密度ステート</h2>
{cards}
{silicon_panel(sil)}
{top_panel(v1, hbc, holv)}
{exp_panel(reg, istats, res, v1)}
{lookback_panel(aiae, v1)}
{warn_html}
<img class="chart" src="chart_insider.png" alt="insider">
<img class="chart" src="chart_spx.png" alt="spx">
<img class="chart" src="chart_aiae.png" alt="aiae" onerror="this.style.display='none'">
<img class="chart" src="chart_v1.png" alt="v1" onerror="this.style.display='none'">
<img class="chart" src="chart_v1_long.png" alt="v1long" onerror="this.style.display='none'">
</main>
<footer>定義: 件数比率 = Officer/DirectorのForm 4日次 buy/sell filings (NONDERIV P/S・filing単位・提出日)。
クラスター密度 = 過去63営業日の比率&gt;1.0日数。ステート: ①平常0-5 / ②警戒6-20(新規買い禁止) / ③パニック21+(分割買い候補)。
AIAE = Z.1株式時価 ÷ (株式時価+5部門負債)、公開ラグ2.5-5.5ヶ月、RT近似は株式部分のみS&amp;P500補正。
V1 = MD/M2スコア + HB6M信号数 + max(0, AIAE-50)。シリコンサイクル = WSTS世界半導体売上3MMA前年比。
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
    update_breadth(spx)
    hbc = hb_counts()
    mdm2 = update_mdm2()
    v1 = compute_v1(mdm2, aiae, hbc)
    holv = ho_level(hbc)
    spxl = update_spx_long()
    reg = aiae_regression(aiae, spxl)
    istats = insider_bucket_stats(res.get("df") if res else None, spxl)
    sil = update_silicon()
    try:
        hist_f = os.path.join(ROOT, "data", "score_history.csv")
        h = (pd.read_csv(hist_f, parse_dates=["date"])
             if os.path.exists(hist_f) else
             pd.DataFrame(columns=["date", "v1", "density", "aiae_rt", "hb6m", "mdm2"]))
        row = {"date": pd.Timestamp(dt.date.today()),
               "v1": round(v1["total"], 2) if v1 else None,
               "density": res["density"] if res else None,
               "aiae_rt": round(aiae.get("rt_val", float("nan")) * 100, 2) if aiae else None,
               "hb6m": hbc["6m"] if hbc else None,
               "mdm2": round(mdm2["val"]) if mdm2 else None}
        h = pd.concat([h, pd.DataFrame([row])], ignore_index=True)
        h = h.drop_duplicates("date", keep="last").sort_values("date")
        h.to_csv(hist_f, index=False)
    except Exception as e:
        log(f"スコア履歴ロギング失敗: {e}")
    if res:
        make_charts(res, aiae)
    make_v1_chart(aiae)
    make_v1_long_chart(aiae)
    build_html(res, aiae, v1, hbc, holv, reg, istats, sil)
    log("dashboard generated")
    for w in warnings:
        log(f"WARN: {w}")


if __name__ == "__main__":
    main()
