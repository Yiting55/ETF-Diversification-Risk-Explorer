import numpy as np
import pandas as pd
import pytest

from engine.metrics import diversification_metrics, estimate_covariance, replay


def cov_from(vols, corr):
    v = np.asarray(vols)
    return np.asarray(corr) * np.outer(v, v)


def test_uncorrelated_equal_assets_count_as_n_bets():
    n = 5
    m = diversification_metrics(np.full(n, 1 / n), cov_from([0.2] * n, np.eye(n)))
    assert m["effective_bets"] == pytest.approx(n)
    assert m["weight_count"] == pytest.approx(n)
    assert m["risk_reduction"] == pytest.approx(1 - 1 / np.sqrt(n))


def test_perfectly_correlated_assets_are_one_bet():
    n = 4
    m = diversification_metrics(np.full(n, 1 / n), cov_from([0.1, 0.2, 0.3, 0.4], np.ones((n, n))))
    assert m["effective_bets"] == pytest.approx(1)
    assert m["risk_reduction"] == pytest.approx(0, abs=1e-12)


def test_risk_shares_sum_to_one_and_track_concentration():
    corr = np.array([[1, 0.9, 0.1], [0.9, 1, 0.1], [0.1, 0.1, 1]])
    w = np.array([0.4, 0.4, 0.2])
    m = diversification_metrics(w, cov_from([0.25, 0.25, 0.05], corr))
    assert m["risk_share"].sum() == pytest.approx(1)
    assert m["risk_share"][2] < w[2]          # low-vol, low-corr asset adds less risk than its weight


def test_hedge_can_exceed_holding_count():
    m = diversification_metrics(np.array([0.5, 0.5]), cov_from([0.2, 0.2], [[1, -0.8], [-0.8, 1]]))
    assert m["effective_bets"] > 2 and m["hedged"]


def test_shrinkage_preserves_each_assets_volatility():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, [0.0005, 0.02], size=(500, 2)))   # T-bill-like vs stock-like
    cov, corr, _ = estimate_covariance(rets, shrink=True)
    sample_vol = rets.std().to_numpy() * np.sqrt(252)
    assert np.sqrt(np.diag(cov)) == pytest.approx(sample_vol)
    assert np.diag(corr) == pytest.approx(1)


def test_shrinkage_keeps_average_correlation_on_short_samples():
    # 23 days, 6 assets: the COVID-crash case. Shrinkage must not drag correlations toward 0.
    rng = np.random.default_rng(1)
    common = rng.normal(0, 0.03, size=(23, 1))
    rets = pd.DataFrame(common + rng.normal(0, 0.01, size=(23, 6)))
    _, corr, shrink = estimate_covariance(rets, shrink=True)
    sample = rets.corr().to_numpy()
    off = ~np.eye(6, dtype=bool)
    assert 0 <= shrink <= 1
    assert corr[off].mean() == pytest.approx(sample[off].mean(), abs=0.01)


def test_replay_buy_and_hold():
    px = pd.DataFrame({"A": [100, 50], "B": [100, 100]})
    r = replay(px, np.array([0.5, 0.5]))
    assert r["total_return"] == pytest.approx(-0.25)
    assert r["max_drawdown"] == pytest.approx(-0.25)


def test_zero_correlation_baseline_isolates_correlation_effect():
    w = np.array([0.6, 0.4])
    vols = [0.3, 0.1]
    uncorrelated = diversification_metrics(w, cov_from(vols, np.eye(2)))
    correlated = diversification_metrics(w, cov_from(vols, [[1, 0.8], [0.8, 1]]))
    assert correlated["uncorrelated_bets"] == pytest.approx(uncorrelated["effective_bets"])
    assert correlated["effective_bets"] < correlated["uncorrelated_bets"]
    assert correlated["uncorrelated_bets"] < correlated["weight_count"]


def test_negative_correlation_improves_score_over_baseline():
    m = diversification_metrics(np.array([0.5, 0.5]), cov_from([0.2, 0.2], [[1, -0.5], [-0.5, 1]]))
    assert m["effective_bets"] == pytest.approx(4)
    assert m["uncorrelated_bets"] == pytest.approx(2)


@pytest.mark.parametrize("shrink", [False, True])
def test_constant_prices_have_clear_error(shrink):
    with pytest.raises(ValueError, match="no measurable price variation for CASH"):
        estimate_covariance(pd.DataFrame({"CASH": [0.0] * 20}), shrink=shrink)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_returns_are_rejected(bad):
    with pytest.raises(ValueError, match="finite returns"):
        estimate_covariance(pd.DataFrame({"A": [0.01, bad, -0.01]}), shrink=False)


@pytest.mark.parametrize("correlation", [-1, -1 + 1e-13])
def test_zero_or_near_zero_portfolio_volatility_has_clear_error(correlation):
    with pytest.raises(ValueError, match="zero or near zero"):
        diversification_metrics(np.array([0.5, 0.5]),
                                cov_from([0.2, 0.2], [[1, correlation], [correlation, 1]]))


def test_low_but_nonzero_volatility_is_supported():
    m = diversification_metrics(np.array([1.0]), np.array([[1e-16]]))
    assert m["effective_bets"] == pytest.approx(1)


def test_nonfinite_covariance_is_rejected():
    with pytest.raises(ValueError, match="finite covariance"):
        diversification_metrics(np.array([1.0]), np.array([[np.nan]]))
