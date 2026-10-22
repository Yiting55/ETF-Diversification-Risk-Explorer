"""Checks against the real price file (skipped if scripts/fetch_universe.py hasn't been run)."""
import pytest

from engine.data import PRICES

pytestmark = pytest.mark.skipif(not PRICES.exists(), reason="run scripts/fetch_universe.py first")


@pytest.fixture(scope="module")
def result():
    from engine import analyze
    return analyze({"VOO": 30, "QQQ": 25, "VTI": 15, "NVDA": 10, "AGG": 10, "GLD": 10})


def test_stress_windows_match_known_spx_drawdowns():
    from engine import analyze
    r = analyze({"SPY": 1})
    assert r.stress["covid"].replay["total_return"] == pytest.approx(-0.34, abs=0.02)
    assert r.stress["2022"].replay["total_return"] == pytest.approx(-0.24, abs=0.02)


def test_single_holding_is_one_bet():
    from engine import analyze
    assert analyze({"SPY": 1}).normal.metrics["effective_bets"] == pytest.approx(1)


def test_overlapping_equity_funds_behave_like_few_bets(result):
    m = result.normal.metrics
    assert m["weight_count"] > 4            # looks diversified by weight...
    assert m["effective_bets"] < 3          # ...but behaves like far fewer bets


def test_near_identical_funds_stay_correlated_in_short_crisis_window(result):
    # VOO and VTI are ~0.99 correlated; 23 days of COVID data must not make them look different.
    assert result.stress["covid"].corr.loc["VOO", "VTI"] > 0.95


def test_duplicate_and_lowercase_tickers_are_merged():
    from engine import analyze
    r = analyze({"voo": 50, "VOO": 50, "agg": 100})
    assert r.normal.tickers == ["AGG", "VOO"] or r.normal.tickers == ["VOO", "AGG"]
    assert sorted(r.normal.weights) == pytest.approx([0.5, 0.5])


def test_lookback_comparison_reuses_selected_result(result):
    assert set(result.lookbacks) == {3, 5, 7}
    assert result.lookbacks[5] is result.normal
    assert result.lookbacks[7].start < result.lookbacks[3].start
    assert all(w.end == result.normal.end for w in result.lookbacks.values())


def test_short_history_lookbacks_report_same_actual_dates():
    import numpy as np
    import pandas as pd
    from engine import analyze

    dates = pd.bdate_range("2026-01-01", periods=100)
    prices = pd.DataFrame({"NEW": 100 * np.cumprod(1 + np.random.default_rng(4).normal(0, 0.01, 100))}, index=dates)

    class Store:
        def get(self, tickers):
            return prices[tickers], []

    r = analyze({"NEW": 1}, stress=[], store=Store())
    assert len({(w.start, w.end) for w in r.lookbacks.values()}) == 1
    assert all(w.low_confidence for w in r.lookbacks.values())


def test_nonfinite_weights_are_rejected():
    from engine.analyze import clean_holdings
    with pytest.raises(ValueError, match="finite numbers"):
        clean_holdings({"SPY": float("nan")})
