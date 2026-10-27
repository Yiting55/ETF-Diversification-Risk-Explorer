"""Tests for the isolated historical block-bootstrap simulator."""
import numpy as np
import pandas as pd
import pytest

from engine.monte_carlo import simulate_portfolio


@pytest.fixture
def return_history():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=600)
    equity = rng.normal(0.0004, 0.012, len(dates))
    bonds = -0.2 * equity + rng.normal(0.0001, 0.004, len(dates))
    return pd.DataFrame({"VOO": equity, "AGG": bonds}, index=dates)


def test_constant_returns_produce_expected_terminal_value():
    dates = pd.bdate_range("2024-01-01", periods=30)
    returns = pd.DataFrame({"A": 0.01, "B": 0.02}, index=dates)
    result = simulate_portfolio(returns, {"A": 50, "B": 50}, starting_value=1_000,
                                horizon_days=10, n_simulations=20, block_days=5, seed=1)
    expected = 1_000 * (1 + 0.015) ** 10
    assert result.terminal_values == pytest.approx(np.full(20, expected))
    assert result.paths.shape == (20, 11)


def test_simulation_is_reproducible_and_reports_summary(return_history):
    args = (return_history, {"VOO": 60, "AGG": 40})
    a = simulate_portfolio(*args, horizon_days=42, n_simulations=50, block_days=5, seed=9)
    b = simulate_portfolio(*args, horizon_days=42, n_simulations=50, block_days=5, seed=9)
    assert np.array_equal(a.paths, b.paths)
    assert set(a.summary) == {
        "terminal_p05", "terminal_p25", "terminal_median", "terminal_p75", "terminal_p95",
        "probability_below_start", "max_drawdown_p05", "max_drawdown_median",
    }
    assert np.all(a.max_drawdowns <= 0)
    assert 0 <= a.summary["probability_below_start"] <= 1


def test_capital_weights_are_normalized(return_history):
    result = simulate_portfolio(return_history, {"VOO": 30, "AGG": 20},
                                horizon_days=10, n_simulations=5, block_days=1)
    single_weight = simulate_portfolio(return_history[["VOO"]], {"VOO": 1},
                                       horizon_days=10, n_simulations=5, block_days=1)
    assert result.paths.shape == single_weight.paths.shape


@pytest.mark.parametrize("weights", [{"VOO": -1}, {"VOO": 0}, {"MISSING": 100}, {}])
def test_invalid_weights_or_missing_ticker_raise(return_history, weights):
    with pytest.raises(ValueError):
        simulate_portfolio(return_history, weights, horizon_days=10, n_simulations=5, block_days=1)


def test_incomplete_rows_are_dropped_jointly(return_history):
    history = return_history.copy()
    history.iloc[0, 0] = np.nan
    result = simulate_portfolio(history, {"VOO": 60, "AGG": 40},
                                horizon_days=10, n_simulations=5, block_days=1)
    assert result.n_observations == len(history) - 1


def test_block_length_must_fit_history(return_history):
    with pytest.raises(ValueError, match="complete historical days"):
        simulate_portfolio(return_history.iloc[:10], {"VOO": 100},
                           horizon_days=10, n_simulations=5, block_days=11)
