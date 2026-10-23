"""Tests for annualized Euler risk contributions."""
import numpy as np
import pandas as pd
import pytest

from engine.risk_contributions import calculate_risk_contributions
from engine.scenarios import Scenario


@pytest.fixture
def synthetic_prices():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2019-01-01", "2023-12-29")
    equity = rng.normal(0.0003, 0.012, len(idx))
    bond = -0.25 * equity + rng.normal(0.0001, 0.004, len(idx))
    returns = pd.DataFrame({"SPY": equity, "AGG": bond}, index=idx)
    return (1 + returns).cumprod() * 100


def test_risk_contributions_sum_to_one_and_volatility_euler(synthetic_prices):
    result = calculate_risk_contributions({"SPY": 60, "AGG": 40}, synthetic_prices,
                                          windows={})["full"]
    table = result.contributions
    assert table["percentage_risk_contribution"].sum() == pytest.approx(1.0)
    assert table["absolute_risk_contribution"].sum() == pytest.approx(result.portfolio_volatility)
    assert table["capital_weight"].to_dict() == pytest.approx({"SPY": 0.6, "AGG": 0.4})


def test_annualized_asset_volatility_uses_log_returns(synthetic_prices):
    out = calculate_risk_contributions({"SPY": 1}, synthetic_prices, windows={})["full"]
    log_returns = np.log(synthetic_prices["SPY"]).diff().dropna()
    assert out.contributions.loc["SPY", "asset_volatility"] == pytest.approx(
        log_returns.std(ddof=1) * np.sqrt(252))


def test_builtin_stress_windows_and_custom_window(synthetic_prices):
    custom = {"custom": Scenario("custom", "Custom", pd.Timestamp("2021-01-01"),
                                 pd.Timestamp("2021-12-31"), "")}
    result = calculate_risk_contributions({"SPY": 60, "AGG": 40}, synthetic_prices,
                                          windows=custom)
    assert set(result) == {"full", "custom"}
    assert result["custom"].start.year == 2021


def test_zero_weights_ignored_and_duplicate_tickers_aggregated(synthetic_prices):
    out = calculate_risk_contributions({"spy": 30, "SPY": 30, "AGG": 40, "ZERO": 0},
                                       synthetic_prices, windows={})["full"]
    assert out.contributions["capital_weight"].to_dict() == pytest.approx({"SPY": 0.6, "AGG": 0.4})


@pytest.mark.parametrize("weights", [{"SPY": -1}, {"SPY": 0}, {"SPY": float("nan")}])
def test_invalid_weights_raise(synthetic_prices, weights):
    with pytest.raises(ValueError):
        calculate_risk_contributions(weights, synthetic_prices, windows={})


def test_missing_ticker_price_raises(synthetic_prices):
    with pytest.raises(ValueError, match="Missing price data"):
        calculate_risk_contributions({"NOPE": 100}, synthetic_prices)


def test_incomplete_daily_prices_are_not_forward_filled(synthetic_prices):
    p = synthetic_prices.copy()
    p.loc[p.index[100], "AGG"] = np.nan
    out = calculate_risk_contributions({"SPY": 50, "AGG": 50}, p, windows={})["full"]
    assert out.n_obs < len(p) - 1


def test_streamlit_mismatch_callout_selects_largest_gap():
    from types import SimpleNamespace
    from app.streamlit_app import risk_mismatch_insight
    res = SimpleNamespace(tickers=["SPY", "AGG"], weights=np.array([0.6, 0.4]),
                          metrics={"risk_share": np.array([0.89, 0.11])})
    insight = risk_mismatch_insight(res)
    assert "SPY" in insight and "60.0%" in insight and "89.0%" in insight
