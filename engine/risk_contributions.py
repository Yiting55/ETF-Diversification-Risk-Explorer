"""Portfolio risk contributions by holding and historical window.

Weights are dollar capital allocations (or any proportional capital values). Each
window normalizes the available long-only allocations to one, estimates annualized
covariance from daily log returns, and reports Euler contributions to volatility.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, estimate_covariance
from .scenarios import STRESS_SCENARIOS, Scenario

MIN_OBS = 15
DAYS_PER_HOLDING = 3


@dataclass
class RiskContributionWindow:
    """Annualized risk statistics for one aligned date window."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int
    portfolio_volatility: float
    variance: float
    contributions: pd.DataFrame
    covariance: pd.DataFrame
    low_confidence: bool = False
    shrinkage: float = 0.0
    excluded: list[str] = field(default_factory=list)


def _window_result(prices: pd.DataFrame, capital: pd.Series, scenario: Scenario,
                   align_start: bool = False) -> RiskContributionWindow | None:
    """Compute one window with complete observations and the app shrinkage rule."""
    px = prices.loc[scenario.start:scenario.end, capital.index]
    if px.empty:
        return None

    excluded: list[str] = []
    if align_start:
        # Match the app's historical overview: align all holdings at the latest first price.
        first = px.apply(pd.Series.first_valid_index).dropna()
        if first.empty:
            return None
        start = first.max()
        px = px.loc[start:]
    else:
        coverage = px.notna().mean()
        excluded = list(coverage[coverage < 0.95].index)
        usable = [t for t in capital.index if t not in excluded]
        if not usable:
            return None
        px = px[usable]
        capital = capital[usable]

    # No forward filling: only price dates where every included holding is observed.
    px = px.dropna(how="any")
    if len(px) < MIN_OBS + 1:
        return None
    # Log returns, as requested; reject invalid/non-positive prices explicitly.
    if (px <= 0).any().any():
        raise ValueError("Prices must be positive to calculate log returns.")
    returns = np.log(px).diff().iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < MIN_OBS:
        return None

    short = len(returns) < DAYS_PER_HOLDING * len(capital)
    cov_values, _, shrinkage = estimate_covariance(returns, shrink=short)
    cov = pd.DataFrame(cov_values, index=capital.index, columns=capital.index)
    w = capital / capital.sum()
    wv = w.to_numpy(dtype=float)
    sigma_w = cov_values @ wv
    variance = float(wv @ sigma_w)
    if not np.isfinite(variance) or variance <= np.finfo(float).eps:
        raise ValueError("Portfolio volatility is zero or too small to calculate risk contributions.")
    port_vol = float(np.sqrt(variance))
    marginal = sigma_w / port_vol
    absolute = wv * marginal
    percentage = absolute / port_vol
    # Keep numerical noise from disrupting an otherwise exact Euler identity.
    if not np.isclose(percentage.sum(), 1.0, atol=1e-8):
        raise ArithmeticError("Risk contributions failed the Euler decomposition check.")

    table = pd.DataFrame({
        "capital_weight": wv,
        "asset_volatility": np.sqrt(np.diag(cov_values)),
        "marginal_contribution_to_risk": marginal,
        "absolute_risk_contribution": absolute,
        "percentage_risk_contribution": percentage,
    }, index=capital.index)
    table.index.name = "ticker"
    return RiskContributionWindow(
        label=scenario.label, start=returns.index[0], end=returns.index[-1],
        n_obs=len(returns), portfolio_volatility=port_vol, variance=variance,
        contributions=table, covariance=cov, low_confidence=short,
        shrinkage=shrinkage, excluded=excluded,
    )


def calculate_risk_contributions(weights: dict, price_returns_df: pd.DataFrame,
                                 windows: dict[str, Scenario] | None = None,
                                 full_start=None, full_end=None) -> dict[str, RiskContributionWindow]:
    """Calculate annualized risk attribution for full, COVID and 2022 windows.

    Args:
        weights: Ticker to dollar capital allocation. Zero allocations are ignored;
            negative allocations are rejected because the app supports long-only portfolios.
        price_returns_df: Date-indexed adjusted price levels (date x ticker), despite the
            legacy parameter name. Returns are calculated internally as daily log returns.
        windows: Optional mapping of scenario keys to custom ``Scenario`` objects. By
            default this uses the app's named COVID and 2022 stress windows.
        full_start/full_end: Optional bounds for the full-history window. Defaults to the
            available price history.

    Returns a mapping with ``full``, ``covid`` and ``2022`` keys where a window has enough
    data. Each window has a ``contributions`` DataFrame and annualized covariance matrix.
    """
    if not isinstance(price_returns_df, pd.DataFrame) or price_returns_df.empty:
        raise ValueError("Provide a non-empty date-indexed price DataFrame.")
    if not isinstance(price_returns_df.index, pd.DatetimeIndex):
        raise ValueError("Price data must have a DatetimeIndex.")

    pairs = [(str(t).strip().upper(), float(v)) for t, v in weights.items() if str(t).strip()]
    capital = pd.Series([v for _, v in pairs], index=[t for t, _ in pairs], dtype=float)
    if capital.empty or not np.isfinite(capital.to_numpy()).all():
        raise ValueError("Provide at least one holding with a finite capital weight.")
    capital = capital.groupby(level=0).sum()
    if (capital < 0).any():
        raise ValueError("Negative weights and short positions are not supported.")
    capital = capital[capital > 0]
    if capital.empty:
        raise ValueError("At least one holding must have a positive capital weight.")

    missing = [t for t in capital.index if t not in price_returns_df.columns]
    if missing:
        raise ValueError(f"Missing price data for: {', '.join(missing)}")
    prices = price_returns_df.loc[:, capital.index].copy()
    prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        prices = prices.groupby(level=0).last()
    if not prices.columns.is_unique:
        raise ValueError("Price data must have unique ticker columns.")

    as_of = prices.index.max() if full_end is None else pd.Timestamp(full_end)
    start = prices.index.min() if full_start is None else pd.Timestamp(full_start)
    scenarios = {
        "full": Scenario("full", "Full historical window", start, as_of,
                         "All available historical data."),
        **(STRESS_SCENARIOS if windows is None else windows),
    }
    results = {}
    for key, scenario in scenarios.items():
        result = _window_result(prices, capital, scenario, align_start=(key == "full"))
        if result is not None:
            results[key] = result
    if not results:
        raise ValueError("No requested window has enough overlapping price data.")
    return results
