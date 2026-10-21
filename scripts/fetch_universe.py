"""
Build the US investable universe for the diversification platform.

Stages (run in order; each one reads the previous stage's output from disk):
  symbols  Download the Nasdaq Trader symbol directories (all US-listed securities)
           and drop leveraged/inverse/buffer/ETN funds and non-common shares by name.
  prices   Download full daily price history from Yahoo Finance in checkpointed
           batches. Re-running resumes from the last finished batch.
  filter   Apply the history / liquidity / price rules and write the filtered
           universe plus daily and weekly adjusted-close price files.
  info     (optional) Fetch ETF metadata (category, AUM, expense ratio, issuer)
           for the filtered ETFs, used to sanity-check the list.

Usage:
  python scripts/fetch_universe.py                 # symbols + prices + filter
  python scripts/fetch_universe.py --stage info    # one stage only
"""
import argparse
import io
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PRICE_DIR = RAW / "prices"
UNIVERSE = ROOT / "data" / "universe"
PROCESSED = ROOT / "data" / "processed"

SYMBOL_URLS = {
    "nasdaq": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "other": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
OTHER_EXCHANGES = {"A": "NYSE American", "N": "NYSE", "P": "NYSE Arca", "Z": "Cboe BZX", "V": "IEX"}

# Filter thresholds
MIN_YEARS = 7               # must cover the 2020 crash and the 2022 stock-bond sell-off
MIN_ADV_USD = 1_000_000     # median daily dollar volume over the last year
MIN_STOCK_PRICE = 5.0       # excludes penny stocks
ADV_WINDOW = 252            # trading days in one year
MAX_STALE_DAYS = 10         # last price older than this => delisted / halted
DUP_CORR = 0.995            # ETFs whose daily returns correlate above this are the same exposure
DUP_WINDOW_YEARS = 3
PRICE_START = "2000-01-01"  # earliest date kept in the processed price files (covers 2008)

# Download pacing
BATCH_SIZE = 100
PAUSE_SECONDS = 3
MAX_RETRIES = 4

# --- name-based filters ------------------------------------------------------
ETF_NAME_RULES = [
    ("leveraged/inverse", re.compile(
        r"\b(\d(\.\d+)?x|ultrashort|ultra(pro)?(?![- ]short)|inverse|leveraged|daily target|-\d+x|bear|bull"
        r"|short(?![- ]?(term|duration|maturity|treasury|dated|bond|income|municipal)))\b", re.I)),
    ("buffer/defined-outcome", re.compile(
        r"\b(buffer|defined outcome|accelerated|outcome|floor|structured)\b", re.I)),
    ("volatility product", re.compile(r"\b(vix|volatility futures|volatility etf)\b", re.I)),
    ("ETN", re.compile(r"\b(etn|exchange traded notes?|notes? due)\b", re.I)),
    ("dated target-maturity", re.compile(r"\b20[2-4]\d\b")),
]
NON_COMMON = re.compile(
    r"\b(warrants?|rights?|units?|preferred|pfd|notes?|debentures?|subordinated|series [a-z])\b|%", re.I)
SHARE_CLASS = re.compile(r"[A-Z]+\.[A-B]")      # BRK.B, BF.B are fine
CLOSED_END = re.compile(r"\bfund\b|closed end|beneficial interest|\bBDC\b", re.I)
ADR = re.compile(r"american depositary|\bADS\b|\bADR\b", re.I)
BAD_FIN_STATUS = {"D", "E", "Q", "G", "H", "J", "K"}  # Nasdaq deficient / delinquent / bankrupt flags

log = logging.getLogger("universe")


def name_filter_reason(row) -> str:
    if row.is_etf:
        for reason, rx in ETF_NAME_RULES:
            if rx.search(row["name"]):
                return reason
        return ""
    if re.search(r"[$.\-=+^]", row.symbol) and not SHARE_CLASS.fullmatch(row.symbol):
        return "symbol suffix (preferred/warrant/unit)"
    if NON_COMMON.search(row["name"]):
        return "non-common share"
    if row.fin_status in BAD_FIN_STATUS:
        return "deficient/delinquent filer"
    return ""


# --- stage 1: symbols --------------------------------------------------------
def stage_symbols():
    (RAW / "symbols").mkdir(parents=True, exist_ok=True)
    UNIVERSE.mkdir(parents=True, exist_ok=True)
    frames = []
    for src, url in SYMBOL_URLS.items():
        text = requests.get(url, timeout=30).text
        (RAW / "symbols" / f"{src}listed.txt").write_text(text, encoding="utf-8")
        df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
        df = df[~df.iloc[:, 0].str.startswith("File Creation Time")]
        if src == "nasdaq":
            frames.append(pd.DataFrame({
                "symbol": df["Symbol"], "name": df["Security Name"], "exchange": "Nasdaq",
                "is_etf": df["ETF"] == "Y", "is_test": df["Test Issue"] == "Y",
                "fin_status": df["Financial Status"]}))
        else:
            frames.append(pd.DataFrame({
                "symbol": df["ACT Symbol"], "name": df["Security Name"],
                "exchange": df["Exchange"].map(OTHER_EXCHANGES).fillna(df["Exchange"]),
                "is_etf": df["ETF"] == "Y", "is_test": df["Test Issue"] == "Y", "fin_status": "N"}))

    sym = pd.concat(frames, ignore_index=True)
    sym = sym[~sym.is_test].drop(columns="is_test").drop_duplicates("symbol")
    sym["type"] = np.where(sym.is_etf, "ETF", "Stock")
    sym["subtype"] = np.select(
        [sym.is_etf, sym["name"].str.contains(CLOSED_END), sym["name"].str.contains(ADR)],
        ["ETF", "Closed-end fund/BDC", "ADR"], default="Common stock")
    sym["yf_symbol"] = sym.symbol.str.replace(".", "-", regex=False)
    sym["name_filter_reason"] = sym.apply(name_filter_reason, axis=1)
    sym.to_csv(UNIVERSE / "symbols_all.csv", index=False)

    kept = sym[sym.name_filter_reason == ""]
    log.info("symbols: %d listed (%d ETFs); %d kept after name filter (%d ETFs, %d stocks)",
             len(sym), sym.is_etf.sum(), len(kept), kept.is_etf.sum(), (~kept.is_etf).sum())
    log.info("removed by reason:\n%s", sym[sym.name_filter_reason != ""]
             .groupby(["type", "name_filter_reason"]).size().to_string())


# --- stage 2: prices ---------------------------------------------------------
def _download(tickers: list[str], threads: bool = True) -> pd.DataFrame:
    """Daily history for tickers as long format: date, ticker, close, adj_close, volume."""
    df = yf.download(tickers, period="max", auto_adjust=False, actions=False,
                     progress=False, threads=threads, group_by="column")
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close", "volume"])
    long = df[["Close", "Adj Close", "Volume"]].stack(level=1).reset_index()
    long.columns = ["date", "ticker", "close", "adj_close", "volume"]
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.astype({"ticker": "string", "close": "float32", "adj_close": "float32", "volume": "float64"})


def stage_prices():
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    sym = pd.read_csv(UNIVERSE / "symbols_all.csv", keep_default_na=False)
    tickers = sorted(sym.loc[sym.name_filter_reason == "", "yf_symbol"])
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    log.info("prices: %d tickers in %d batches of %d", len(tickers), len(batches), BATCH_SIZE)

    missing: list[str] = []
    t0 = time.time()
    for i, batch in enumerate(batches):
        path = PRICE_DIR / f"batch_{i:04d}.parquet"
        if path.exists():
            continue
        for attempt in range(1, MAX_RETRIES + 1):
            data = _download(batch)
            got = set(data.ticker.unique())
            # Most of a batch coming back empty means Yahoo is throttling us, not bad tickers.
            if len(got) >= 0.5 * len(batch) or attempt == MAX_RETRIES:
                break
            wait = 60 * attempt
            log.warning("batch %d: only %d/%d returned, likely rate-limited; sleeping %ds",
                        i, len(got), len(batch), wait)
            time.sleep(wait)
        data.to_parquet(path, index=False)
        missing += [t for t in batch if t not in got]
        log.info("batch %d/%d done: %d/%d tickers, %.0fs elapsed",
                 i + 1, len(batches), len(got), len(batch), time.time() - t0)
        time.sleep(PAUSE_SECONDS)

    # One slower, single-threaded pass over anything that failed in the bulk pass.
    if missing:
        log.info("retrying %d missing tickers single-threaded", len(missing))
        retry = pd.concat([_download(missing[i:i + 20], threads=False)
                           for i in range(0, len(missing), 20)], ignore_index=True)
        if not retry.empty:
            retry.to_parquet(PRICE_DIR / "batch_retry.parquet", index=False)
        still = sorted(set(missing) - set(retry.ticker.unique()))
        (PRICE_DIR / "no_data.txt").write_text("\n".join(still), encoding="utf-8")
        log.info("%d tickers have no Yahoo data (listed in no_data.txt)", len(still))
    log.info("prices done in %.1f min", (time.time() - t0) / 60)


# --- stage 3: filter ---------------------------------------------------------
def _ticker_stats(g: pd.DataFrame) -> dict:
    g = g.sort_values("date")
    recent = g.tail(ADV_WINDOW)
    return {
        "first_date": g.date.iloc[0], "last_date": g.date.iloc[-1], "n_days": len(g),
        "last_close": float(g.close.iloc[-1]),
        "adv_usd": float((recent.close * recent.volume).median()),
        "zero_volume_pct": float((recent.volume == 0).mean()),
    }


def duplicate_clusters(px: pd.DataFrame, etf_adv: pd.Series) -> pd.DataFrame:
    """Group ETFs that are effectively the same exposure (e.g. SPY/VOO/IVV).

    Two ETFs are linked when their daily returns over the last DUP_WINDOW_YEARS correlate
    >= DUP_CORR; clusters are the connected components. The most liquid member of each
    cluster is its representative, so suggestions never offer three S&P 500 funds.
    """
    cols = [t for t in etf_adv.index if t in px.columns]
    window = px.loc[px.index >= px.index[-1] - pd.DateOffset(years=DUP_WINDOW_YEARS), cols]
    rets = window.pct_change(fill_method=None).iloc[1:]
    corr = rets.corr(min_periods=int(0.8 * len(rets))).to_numpy()

    parent = list(range(len(cols)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in zip(*np.where(np.triu(corr >= DUP_CORR, k=1))):
        parent[root(a)] = root(b)

    out = pd.DataFrame({"yf_symbol": cols, "root": [root(i) for i in range(len(cols))]})
    out["adv"] = etf_adv[cols].to_numpy()
    rep = out.sort_values("adv", ascending=False).groupby("root").yf_symbol.first()
    out["dup_cluster"] = out.root.map(rep)              # cluster named after its representative
    out["dup_representative"] = out.yf_symbol == out.dup_cluster
    return out[["yf_symbol", "dup_cluster", "dup_representative"]]


def stage_filter():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sym = pd.read_csv(UNIVERSE / "symbols_all.csv", keep_default_na=False)
    files = sorted(PRICE_DIR.glob("batch_*.parquet"))

    # Pass 1: per-ticker stats, one batch file at a time to keep memory low.
    stats = []
    for f in files:
        df = pd.read_parquet(f)
        stats += [{"yf_symbol": t, **_ticker_stats(g)} for t, g in df.groupby("ticker")]
    stats = pd.DataFrame(stats).drop_duplicates("yf_symbol", keep="last")

    as_of = stats.last_date.max()
    history_cutoff = as_of - pd.DateOffset(years=MIN_YEARS)
    audit = sym.merge(stats, on="yf_symbol", how="left")
    audit["years_history"] = ((audit.last_date - audit.first_date).dt.days / 365.25).round(1)

    def final_reason(r):
        if r.name_filter_reason:
            return r.name_filter_reason
        if pd.isna(r.first_date):
            return "no price data"
        if r.last_date < as_of - pd.Timedelta(days=MAX_STALE_DAYS):
            return "stale (delisted/halted)"
        if r.first_date > history_cutoff:
            return f"history < {MIN_YEARS}y"
        if r.adv_usd < MIN_ADV_USD:
            return "illiquid (ADV < $1M)"
        if r.type == "Stock" and r.last_close < MIN_STOCK_PRICE:
            return f"price < ${MIN_STOCK_PRICE:.0f}"
        return ""

    audit["exclude_reason"] = audit.apply(final_reason, axis=1)
    audit["included"] = audit.exclude_reason == ""
    audit.to_csv(UNIVERSE / "universe_audit.csv", index=False)

    keep = audit[audit.included].sort_values(["type", "adv_usd"], ascending=[True, False])

    # Pass 2: adjusted closes for the included tickers only, wide format (date x ticker).
    wanted = set(keep.yf_symbol)
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=["date", "ticker", "adj_close"])
        parts.append(df[df.ticker.isin(wanted) & (df.date >= PRICE_START)])
    px = (pd.concat(parts).drop_duplicates(["date", "ticker"], keep="last")
          .pivot(index="date", columns="ticker", values="adj_close").sort_index())
    px.to_parquet(PROCESSED / "prices_daily.parquet", compression="zstd")
    px.resample("W-FRI").last().to_parquet(PROCESSED / "prices_weekly.parquet", compression="zstd")

    clusters = duplicate_clusters(px, keep[keep.type == "ETF"].set_index("yf_symbol").adv_usd)
    keep = keep.merge(clusters, on="yf_symbol", how="left")
    cols = ["symbol", "yf_symbol", "name", "type", "subtype", "exchange", "first_date",
            "years_history", "adv_usd", "last_close", "zero_volume_pct", "dup_cluster", "dup_representative"]
    keep[cols].to_csv(UNIVERSE / "filtered_universe.csv", index=False)
    log.info("ETFs: %d tickers, %d distinct exposures after merging near-duplicates",
             (keep.type == "ETF").sum(), keep.dup_representative.fillna(False).astype(bool).sum())

    log.info("filter (as of %s): %d ETFs and %d stocks included", as_of.date(),
             (keep.type == "ETF").sum(), (keep.type == "Stock").sum())
    log.info("exclusions:\n%s", audit[~audit.included].groupby(["type", "exclude_reason"])
             .size().to_string())


# --- stage 4: ETF metadata ---------------------------------------------------
INFO_FIELDS = ["longName", "category", "fundFamily", "totalAssets", "netExpenseRatio",
               "fundInceptionDate", "legalType", "quoteType"]


def stage_info():
    uni = pd.read_csv(UNIVERSE / "filtered_universe.csv", keep_default_na=False)
    out_path = UNIVERSE / "etf_info.csv"
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["yf_symbol"])
    todo = [t for t in uni.loc[uni.type == "ETF", "yf_symbol"] if t not in set(done.yf_symbol)]
    log.info("info: %d ETFs to fetch (%d cached)", len(todo), len(done))

    def save(rows):
        nonlocal done
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(out_path, index=False)

    rows, consecutive_failures = [], 0
    for i, t in enumerate(todo, 1):
        for attempt in range(1, 3):
            try:
                info = yf.Ticker(t).info
                # When Yahoo blocks the metadata endpoint, yfinance returns a near-empty
                # dict instead of raising, so treat a missing quoteType as a failure.
                if not info.get("quoteType"):
                    raise RuntimeError("empty response (metadata endpoint blocked/rate-limited)")
                rows.append({"yf_symbol": t, **{k: info.get(k) for k in INFO_FIELDS}})
                consecutive_failures = 0
                break
            except Exception as e:
                log.warning("%s: %s (attempt %d)", t, e, attempt)
                time.sleep(20 * attempt)
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                save(rows)
                log.error("info: Yahoo is refusing metadata requests; stopped at %d/%d. "
                          "Re-run later - finished tickers are cached.", i, len(todo))
                return
        if i % 50 == 0 or i == len(todo):  # checkpoint so a crash doesn't lose progress
            save(rows)
            rows = []
            log.info("info: %d/%d", i, len(todo))
        time.sleep(0.3)


STAGES = {"symbols": stage_symbols, "prices": stage_prices, "filter": stage_filter, "info": stage_info}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all",
                        help="'all' runs symbols, prices and filter")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    for name in (["symbols", "prices", "filter"] if args.stage == "all" else [args.stage]):
        STAGES[name]()


if __name__ == "__main__":
    main()
