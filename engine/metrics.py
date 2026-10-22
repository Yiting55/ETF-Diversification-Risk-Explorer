"""Diversification metrics.

Everything here is derived from one quantity, the diversification ratio

    DR = (sum_i w_i * sigma_i) / sigma_p,     sigma_p = sqrt(w' Sigma w)

i.e. "risk if nothing offset anything" divided by "risk you actually have".

    effective_bets  = DR^2       volatility-based equivalent number of uncorrelated bets
                                  (N uncorrelated, equal-vol, equal-weight assets -> exactly N;
                                   perfectly correlated assets -> 1)
    risk_reduction  = 1 - 1/DR   share of risk removed by diversification
    weight_count    = 1/sum w^2  how many holdings you *look* like you have (ignores correlation)
    risk_share_i    = w_i (Sigma w)_i / sigma_p^2   each holding's share of total variance; sums to 1
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
Z_95 = 1.645  # one-sided 95% normal quantile, for the "bad year" dollar view


def _constant_correlation_shrinkage(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004), "Honey, I Shrunk the Sample Covariance Matrix".

    Shrinks the sample covariance toward a target where every pair shares the *average*
    correlation, with the intensity that minimises expected estimation error. Unlike the
    usual identity target, this leaves the overall correlation level intact - shrinking
    toward zero correlation would make short crisis windows look falsely diversified.
    """
    t, n = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / t
    var = np.diag(sample)
    sd = np.sqrt(var)
    if n == 1:
        return sample, 0.0
    r_bar = ((sample / np.outer(sd, sd)).sum() - n) / (n * (n - 1))
    prior = r_bar * np.outer(sd, sd)
    np.fill_diagonal(prior, var)

    y = x ** 2
    phi_mat = y.T @ y / t - sample ** 2
    phi = phi_mat.sum()
    theta = (x ** 3).T @ x / t - var[:, None] * sample      # theta_ii,ij
    np.fill_diagonal(theta, 0)
    rho = np.diag(phi_mat).sum() + r_bar * ((sd[None, :] / sd[:, None]) * theta).sum()
    gamma = ((sample - prior) ** 2).sum()
    shrinkage = float(np.clip((phi - rho) / gamma / t, 0, 1)) if gamma > 0 else 1.0
    return shrinkage * prior + (1 - shrinkage) * sample, shrinkage


def estimate_covariance(returns: pd.DataFrame, shrink: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Annualised covariance and correlation of daily returns, plus the shrinkage intensity.

    With enough days per holding the sample correlation is used as-is. Shrinkage is only
    for short windows (few days per holding, e.g. the 23-day COVID crash with many
    holdings), where the sample estimate is noisy. It is applied to standardised returns
    (i.e. to the correlation matrix) and rescaled by each asset's own volatility, so a
    T-bill fund keeps its tiny volatility instead of being pulled toward NVDA's.
    """
    x = returns.to_numpy(dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Risk estimates require at least two days of finite returns.")
    vol = x.std(axis=0, ddof=1)
    constant = returns.columns[vol <= np.finfo(float).eps]
    if len(constant):
        raise ValueError(f"Cannot estimate diversification: no measurable price variation for {', '.join(map(str, constant))}.")
    z = (x - x.mean(axis=0)) / vol
    if shrink:
        corr, shrinkage = _constant_correlation_shrinkage(z)
    else:
        corr, shrinkage = z.T @ z, 0.0
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    cov = corr * np.outer(vol, vol) * TRADING_DAYS
    return cov, corr, shrinkage


def diversification_metrics(weights: np.ndarray, cov: np.ndarray) -> dict:
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if (w.ndim != 1 or not len(w) or cov.shape != (len(w), len(w))
            or not np.isfinite(w).all() or not np.isfinite(cov).all()
            or (w < 0).any() or not np.isclose(w.sum(), 1)
            or (np.diag(cov) < 0).any()):
        raise ValueError("Diversification requires finite covariance data and non-negative weights summing to 100%.")
    vols = np.sqrt(np.diag(cov))
    undiversified_vol = float(w @ vols)            # what volatility would be with all correlations = 1
    variance = float(w @ cov @ w)
    if undiversified_vol <= 0 or variance <= undiversified_vol ** 2 * 1e-12:
        raise ValueError("Portfolio volatility is zero or near zero; the diversification score is not reliable.")
    port_vol = float(np.sqrt(variance))
    dr = undiversified_vol / port_vol
    return {
        "portfolio_vol": port_vol,
        "undiversified_vol": undiversified_vol,
        "diversification_ratio": dr,
        "effective_bets": dr ** 2,
        "weight_count": float(1 / np.sum(w ** 2)),
        "uncorrelated_bets": float(undiversified_vol ** 2 / np.sum((w * vols) ** 2)),
        "risk_reduction": 1 - 1 / dr,
        "asset_vols": vols,
        "risk_share": w * (cov @ w) / port_vol ** 2,
        # Hedges (negatively correlated holdings) can push DR^2 above the number of holdings.
        "hedged": dr ** 2 > len(w) + 1e-9,
    }


def dollar_view(metrics: dict, value: float) -> dict:
    """Parametric one-year 95% loss: 'a bad year (1-in-20) could cost about ...'."""
    return {
        "bad_year_loss": Z_95 * metrics["portfolio_vol"] * value,
        "bad_year_loss_undiversified": Z_95 * metrics["undiversified_vol"] * value,
    }


def replay(prices: pd.DataFrame, weights: np.ndarray) -> dict:
    """Buy-and-hold the portfolio through a window, starting from the given weights."""
    growth = prices / prices.iloc[0]                 # each holding's value of $1 invested
    portfolio = growth @ np.asarray(weights, dtype=float)
    drawdown = portfolio / portfolio.cummax() - 1
    return {
        "growth": growth,
        "portfolio": portfolio,
        "total_return": float(portfolio.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "holding_returns": growth.iloc[-1] - 1,
    }
