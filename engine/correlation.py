"""Correlation views with fixed holdings and explicit historical windows."""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from .metrics import estimate_covariance
from .scenarios import Scenario


def average_pairwise(corr: pd.DataFrame) -> float:
    """Unweighted mean of distinct pairs; undefined for a single holding."""
    values = corr.to_numpy()
    pairs = values[np.triu_indices(len(values), k=1)]
    return float(pairs.mean()) if len(pairs) else float("nan")


def rolling_correlation(prices: pd.DataFrame, window: int = 60) -> pd.Series:
    """Average pairwise Pearson correlation using complete trailing return windows.

    Every point uses the same columns and all `window` daily returns. Missing
    prices are never forward-filled and incomplete or constant windows stay gaps.
    Passing two columns gives the correlation for that pair.
    """
    if window < 2:
        raise ValueError("Rolling correlation requires a window of at least two days.")
    result = pd.Series(np.nan, index=prices.index, name="Correlation")
    if len(prices.columns) < 2:
        return result
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    complete = returns.notna().all(axis=1).rolling(window).sum().eq(window)
    varying = returns.rolling(window).std().gt(np.finfo(float).eps).all(axis=1)
    # Accumulate one pair at a time instead of allocating a date × N × N matrix.
    total = pd.Series(0.0, index=prices.index)
    count = 0
    for a, b in combinations(prices.columns, 2):
        total += returns[a].rolling(window).corr(returns[b])
        count += 1
    return (total / count).where(complete & varying).clip(-1, 1).rename("Correlation")


@dataclass
class CorrelationWindow:
    corr: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    shrinkage: float
    low_confidence: bool


@dataclass
class CorrelationComparison:
    baseline: CorrelationWindow | None = None
    stress: CorrelationWindow | None = None
    excluded: list[str] = field(default_factory=list)
    note: str = ""


def compare_correlations(prices: pd.DataFrame, scenario: Scenario,
                         min_obs: int = 15, min_coverage: float = 0.95) -> CorrelationComparison:
    """Compare stress with the preceding calendar year, using identical holdings.

    Require 95% price coverage in both periods and nonconstant returns. Recompute
    both estimates on the common set, rather than slicing a larger shrunk matrix.
    """
    baseline = prices.loc[scenario.start - pd.DateOffset(years=1):
                          scenario.start - pd.Timedelta(days=1)]
    stress = prices.loc[scenario.start:scenario.end]
    out = CorrelationComparison()
    if baseline.empty or stress.empty:
        out.note = "Not enough data for a pre-period baseline and stress comparison."
        return out
    kept = [t for t in prices if baseline[t].notna().mean() >= min_coverage
            and stress[t].notna().mean() >= min_coverage]
    # Avoid filling or bridging over missing prices when forming daily returns.
    periods = [p[kept].pct_change(fill_method=None).iloc[1:]
               .replace([np.inf, -np.inf], np.nan) for p in (baseline, stress)]
    kept = [t for t in kept if all(r[t].std() > np.finfo(float).eps for r in periods)]
    out.excluded = [t for t in prices if t not in kept]
    if len(kept) < 2:
        out.note = "At least two holdings with usable history in both periods are needed."
        return out
    windows = []
    for prices_in_period, returns in zip((baseline, stress), periods):
        returns = returns[kept].dropna()
        if len(returns) < min_obs:
            out.note = "Not enough overlapping daily returns in both periods for comparison."
            return out
        short = len(returns) < 3 * len(kept)
        try:
            _, corr, shrinkage = estimate_covariance(returns, shrink=short)
        except ValueError:
            out.note = "Correlation is unavailable: the overlapping returns have insufficient variation."
            return out
        # Include the starting price used by the first daily return.
        first = prices_in_period.index.get_loc(returns.index[0]) - 1
        windows.append(CorrelationWindow(
            pd.DataFrame(corr, index=kept, columns=kept), prices_in_period.index[first],
            returns.index[-1], len(returns), shrinkage, short))
    out.baseline, out.stress = windows
    return out
