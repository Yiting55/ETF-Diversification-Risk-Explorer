"""Price and universe access.

Prices come from data/processed/prices_daily.parquet (built by scripts/fetch_universe.py).
Tickers outside that file - e.g. funds younger than 7 years such as SGOV or IBIT - are
downloaded from Yahoo Finance on first use and cached in data/cache/, so a live demo
never depends on the network for tickers it has already seen.
"""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_CSV = ROOT / "data" / "universe" / "filtered_universe.csv"
PRICES = ROOT / "data" / "processed" / "prices_daily.parquet"
CACHE = ROOT / "data" / "cache" / "extra_prices.parquet"


def normalize_ticker(t: str) -> str:
    """User input -> Yahoo symbol: ' brk.b ' -> 'BRK-B'."""
    return t.strip().upper().replace(".", "-")


def load_universe() -> pd.DataFrame:
    return pd.read_csv(UNIVERSE_CSV, keep_default_na=False)


class PriceStore:
    """Daily adjusted closes (date x ticker) for any US ticker."""

    def __init__(self, prices_path: Path = PRICES, cache_path: Path = CACHE):
        self.prices_path = prices_path
        self.cache_path = cache_path
        self.local = set(pq.read_schema(prices_path).names) - {"date"}

    def _cached(self) -> pd.DataFrame:
        return pd.read_parquet(self.cache_path) if self.cache_path.exists() else pd.DataFrame()

    def _fetch(self, tickers: list[str]) -> pd.DataFrame:
        import logging
        import yfinance as yf  # only needed for tickers outside the local file
        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # unknown tickers are reported by get()
        df = yf.download(tickers, period="max", auto_adjust=True, progress=False,
                         threads=False, multi_level_index=True)
        if df.empty:
            return pd.DataFrame()
        px = df["Close"].dropna(how="all", axis=1).astype("float32")
        px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
        cached = self._cached()
        merged = px if cached.empty else cached.drop(columns=px.columns, errors="ignore").join(px, how="outer")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.cache_path, compression="zstd")
        return px

    def get(self, tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
        """Return (prices, unknown_tickers). Unknown = no data anywhere, even on Yahoo."""
        local = [t for t in tickers if t in self.local]
        frames = [pd.read_parquet(self.prices_path, columns=local)] if local else []

        rest = [t for t in tickers if t not in self.local]
        if rest:
            cached = self._cached()
            hit = [t for t in rest if t in cached.columns]
            if hit:
                frames.append(cached[hit])
            miss = [t for t in rest if t not in cached.columns]
            if miss:
                fetched = self._fetch(miss)
                if not fetched.empty:
                    frames.append(fetched)

        prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
        unknown = [t for t in tickers if t not in prices.columns]
        return prices[[t for t in tickers if t in prices.columns]], unknown
