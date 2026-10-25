"""Historical block-bootstrap Monte Carlo scenarios for a long-only portfolio.

This is an empirical scenario generator, not a predictive model. It resamples the
historical portfolio return series in contiguous blocks, which preserves short-run
return clustering better than independently drawing individual days.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Simulated wealth paths and summary statistics."""

    paths: np.ndarray
    terminal_values: np.ndarray
    max_drawdowns: np.ndarray
    starting_value: float
    horizon_days: int
    n_simulations: int
    block_days: int
    n_observations: int
    history_start: pd.Timestamp
    history_end: pd.Timestamp

    @property
    def summary(self) -> dict[str, float]:
        """Return useful outcome quantiles and empirical probabilities."""
        q = np.percentile(self.terminal_values, [5, 25, 50, 75, 95])
        dd = np.percentile(self.max_drawdowns, [5, 50])
        return {
            "terminal_p05": float(q[0]),
            "terminal_p25": float(q[1]),
            "terminal_median": float(q[2]),
            "terminal_p75": float(q[3]),
            "terminal_p95": float(q[4]),
            "probability_below_start": float(np.mean(self.terminal_values < self.starting_value)),
            "max_drawdown_p05": float(dd[0]),
            "max_drawdown_median": float(dd[1]),
        }


def simulate_portfolio(daily_returns: pd.DataFrame, weights: dict[str, float],
                       starting_value: float = 10_000, horizon_days: int = 252,
                       n_simulations: int = 2_000, block_days: int = 21,
                       seed: int = 42) -> MonteCarloResult:
    """Simulate daily rebalanced portfolio outcomes by block-bootstrapping history.

    ``daily_returns`` must contain simple daily returns (for example, adjusted price
    ``pct_change``), with dates as its index and tickers as columns. Asset returns are
    combined with the normalized target weights each day, so the simulated portfolio is
    rebalanced to those weights daily. Historical blocks are sampled with replacement.
    """
    if not isinstance(daily_returns, pd.DataFrame) or daily_returns.empty:
        raise ValueError("Provide a non-empty DataFrame of daily returns.")
    if not isinstance(daily_returns.index, pd.DatetimeIndex):
        raise ValueError("Daily returns must have a DatetimeIndex.")
    if daily_returns.columns.has_duplicates:
        raise ValueError("Daily returns must have unique ticker columns.")

    try:
        pairs = [(str(t).strip().upper(), float(v)) for t, v in weights.items()
                 if str(t).strip()]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Weights must be a ticker-to-capital mapping of finite numbers.") from exc
    if not pairs:
        raise ValueError("Provide at least one holding with a positive weight.")
    capital = pd.Series([v for _, v in pairs], index=[t for t, _ in pairs], dtype=float)
    if not np.isfinite(capital.to_numpy()).all():
        raise ValueError("Weights must be finite numbers.")
    capital = capital.groupby(level=0).sum()
    if (capital < 0).any():
        raise ValueError("Negative weights and short positions are not supported.")
    capital = capital[capital > 0]
    if capital.empty:
        raise ValueError("At least one holding must have a positive weight.")

    columns = {str(c).strip().upper(): c for c in daily_returns.columns}
    missing = [t for t in capital.index if t not in columns]
    if missing:
        raise ValueError(f"Missing daily returns for: {', '.join(missing)}")
    if not np.isfinite(starting_value) or starting_value <= 0:
        raise ValueError("Starting portfolio value must be a positive finite number.")
    for name, value in (("horizon_days", horizon_days), ("n_simulations", n_simulations),
                        ("block_days", block_days), ("seed", seed)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
    if horizon_days < 1 or n_simulations < 1 or block_days < 1 or seed < 0:
        raise ValueError("Horizon, simulation count, and block length must be positive; seed cannot be negative.")

    selected = [columns[t] for t in capital.index]
    asset_returns = daily_returns[selected].copy()
    asset_returns.columns = capital.index
    asset_returns = asset_returns.replace([np.inf, -np.inf], np.nan).dropna(how="any").sort_index()
    if len(asset_returns) < block_days:
        raise ValueError(f"Need at least {block_days} complete historical days for the selected block length.")
    observations = asset_returns.to_numpy(dtype=float)
    if (observations <= -1).any():
        raise ValueError("Daily simple returns must be greater than -100%.")

    w = (capital / capital.sum()).to_numpy(dtype=float)
    # Convert the cross-asset history to the target-weight, daily-rebalanced portfolio.
    portfolio_returns = observations @ w
    n_obs = len(portfolio_returns)
    n_blocks = int(np.ceil(horizon_days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_obs - block_days + 1, size=(n_simulations, n_blocks))
    offsets = np.arange(block_days)
    sample_indices = (starts[:, :, None] + offsets).reshape(n_simulations, -1)[:, :horizon_days]
    sampled = portfolio_returns[sample_indices]

    paths = np.empty((n_simulations, horizon_days + 1), dtype=float)
    paths[:, 0] = starting_value
    paths[:, 1:] = starting_value * np.cumprod(1.0 + sampled, axis=1)
    peaks = np.maximum.accumulate(paths, axis=1)
    max_drawdowns = np.min(paths / peaks - 1.0, axis=1)
    return MonteCarloResult(
        paths=paths, terminal_values=paths[:, -1].copy(), max_drawdowns=max_drawdowns,
        starting_value=float(starting_value), horizon_days=int(horizon_days),
        n_simulations=int(n_simulations), block_days=int(block_days),
        n_observations=n_obs, history_start=asset_returns.index[0],
        history_end=asset_returns.index[-1],
    )
