"""Portfolio -> diversification results for a normal window and any stress windows.

Date handling rules (shown to the user in `notes`, never applied silently):
  * Normal window: if a holding is younger than the lookback, the window starts at the
    youngest holding's first price, so every holding is measured over the same dates.
  * Stress window: a holding with no data for that period (it did not exist yet) is
    left out of that scenario and the remaining weights are rescaled.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import PriceStore, normalize_ticker
from .metrics import diversification_metrics, dollar_view, estimate_covariance, replay
from .scenarios import DEFAULT_LOOKBACK, LOOKBACK_CHOICES, STRESS_SCENARIOS, Scenario, normal_window

MIN_COVERAGE = 0.95          # share of a stress window's trading days a holding must have
MIN_OBS = 15                 # below this a covariance estimate is meaningless
DAYS_PER_HOLDING = 3         # fewer trading days than 3 x holdings -> shrink + low confidence
SHORT_HISTORY_DAYS = 252     # normal window shorter than one year -> flag as low confidence


@dataclass
class WindowResult:
    scenario: Scenario
    tickers: list[str]
    weights: np.ndarray
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    metrics: dict
    corr: pd.DataFrame
    shrinkage: float
    dollars: dict
    replay: dict
    excluded: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    low_confidence: bool = False


@dataclass
class Analysis:
    normal: WindowResult
    stress: dict[str, WindowResult]
    unknown: list[str]
    value: float
    lookbacks: dict[int, WindowResult | None] = field(default_factory=dict)


def clean_holdings(holdings: dict[str, float]) -> pd.Series:
    # Build from pairs, not a dict comprehension: "voo" and "VOO" must add up, not overwrite.
    pairs = [(normalize_ticker(t), float(x)) for t, x in holdings.items() if t and t.strip()]
    w = pd.Series([x for _, x in pairs], index=[t for t, _ in pairs], dtype=float)
    if not np.isfinite(w.to_numpy()).all():
        raise ValueError("Weights must be finite numbers.")
    w = w.groupby(level=0).sum()
    if (w < 0).any():
        raise ValueError("Weights must be non-negative (long-only portfolios).")
    w = w[w > 0]
    if w.empty:
        raise ValueError("Add at least one holding with a positive weight.")
    return w / w.sum()


def analyze_window(prices: pd.DataFrame, weights: pd.Series, scenario: Scenario, value: float,
                   align_start: bool) -> WindowResult | None:
    px = prices.loc[scenario.start:scenario.end, weights.index]
    notes, excluded = [], {}
    if px.empty:
        return None

    if align_start:
        first = px.apply(pd.Series.first_valid_index).dropna()
        late = first[first > px.index[0] + pd.Timedelta(days=7)]
        if not late.empty:
            px = px.loc[late.max():]
            notes.append(f"Window starts {late.max():%d %b %Y} because {late.idxmax()} only has "
                         f"prices from then; all holdings are measured over the same dates.")
    else:
        coverage = px.notna().mean()
        for t in coverage[coverage < MIN_COVERAGE].index:
            first = prices[t].first_valid_index()
            excluded[t] = (f"no prices for this period (first price {first:%d %b %Y})"
                           if first is not None else "no prices")
        if excluded:
            notes.append(f"Left out of this scenario: {', '.join(excluded)}. "
                         "Remaining weights were rescaled to 100%.")

    kept = [t for t in weights.index if t not in excluded]
    if not kept:
        return None
    w = weights[kept] / weights[kept].sum()
    px = px[kept].dropna()
    rets = px.pct_change().iloc[1:]
    if len(rets) < MIN_OBS:
        return None

    too_short = len(rets) < DAYS_PER_HOLDING * len(kept)
    cov, corr, shrink = estimate_covariance(rets, shrink=too_short)
    m = diversification_metrics(w.to_numpy(), cov)
    low_conf = too_short or (align_start and len(rets) < SHORT_HISTORY_DAYS)
    if low_conf:
        notes.append(f"Only {len(rets)} trading days of data for {len(kept)} holdings, so "
                     "correlations are estimated with low confidence"
                     + (f" (shrunk {shrink:.0%} toward the average correlation)." if too_short else "."))
    return WindowResult(
        scenario=scenario, tickers=kept, weights=w.to_numpy(), start=px.index[0], end=px.index[-1],
        n_obs=len(rets), metrics=m, corr=pd.DataFrame(corr, index=kept, columns=kept),
        shrinkage=shrink, dollars=dollar_view(m, value), replay=replay(px, w.to_numpy()),
        excluded=excluded, notes=notes, low_confidence=low_conf,
    )


def analyze(holdings: dict[str, float], value: float = 10_000, lookback_years: int = DEFAULT_LOOKBACK,
            stress: list[Scenario] | None = None, store: PriceStore | None = None) -> Analysis:
    weights = clean_holdings(holdings)
    store = store or PriceStore()
    prices, unknown = store.get(list(weights.index))
    weights = weights.drop(unknown)
    if weights.empty:
        raise ValueError(f"No price data found for: {', '.join(unknown)}")
    weights = weights / weights.sum()

    as_of = prices.index.max()
    normal = analyze_window(prices, weights, normal_window(as_of, lookback_years), value, align_start=True)
    if normal is None:
        raise ValueError("Not enough overlapping price history to measure correlations.")

    scenarios = list(STRESS_SCENARIOS.values()) if stress is None else stress
    results = {}
    for s in scenarios:
        r = analyze_window(prices, weights, s, value, align_start=False)
        if r is not None:
            results[s.key] = r
    lookbacks = {lookback_years: normal}
    for years in LOOKBACK_CHOICES:
        if years != lookback_years:
            try:
                lookbacks[years] = analyze_window(
                    prices, weights, normal_window(as_of, years), value, align_start=True)
            except ValueError:
                lookbacks[years] = None
    return Analysis(normal=normal, stress=results, unknown=unknown, value=value, lookbacks=lookbacks)
