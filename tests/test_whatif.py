"""Tests for engine/whatif.py."""
import numpy as np
import pandas as pd
import pytest

from engine.data import PRICES
from engine.metrics import diversification_metrics
from engine.whatif import (capm_expected_returns, mark_pareto, new_weights, ranking_stability,
                           ranking_table, rf_from_proxy, run_whatif)


class FakeStore:
    """Synthetic daily prices, 2016-2026."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2016-01-04", "2026-10-20")
        n = len(idx)
        eq = rng.normal(0.0004, 0.010, n)
        bond = -0.3 * eq + rng.normal(0.0001, 0.004, n)
        r = {
            "VOO": eq + rng.normal(0, 0.002, n),
            "QQQ": 1.2 * eq + rng.normal(0, 0.005, n),
            "VT": 0.95 * eq + rng.normal(0, 0.002, n),
            "EEM": 0.9 * eq + rng.normal(0, 0.008, n),
            "VXUS": 0.9 * eq + rng.normal(0, 0.004, n),
            "VNQ": 0.8 * eq + rng.normal(0, 0.008, n),
            "IEF": bond,
            "TLT": 2.0 * bond + rng.normal(0, 0.003, n),
            "GLD": rng.normal(0.0002, 0.009, n),
            "DBC": 0.3 * eq + rng.normal(0, 0.010, n),
            "BIL": np.full(n, 0.04 / 252),
        }
        for i in range(1, 6):
            r[f"STK{i}"] = eq + rng.normal(0, 0.006, n)
        self.prices = (1 + pd.DataFrame(r, index=idx)).cumprod() * 100

    def get(self, tickers):
        have = [t for t in tickers if t in self.prices.columns]
        return self.prices[have], [t for t in tickers if t not in self.prices.columns]


@pytest.fixture(scope="module")
def wi():
    return run_whatif({"VOO": 60, "QQQ": 40}, store=FakeStore())


def test_new_weights_sum_to_one_and_scale_pro_rata():
    w = pd.Series({"VOO": 0.6, "QQQ": 0.4})
    nw = new_weights(w, "GLD", 0.2)
    assert nw.sum() == pytest.approx(1)
    assert nw["VOO"] == pytest.approx(0.48) and nw["GLD"] == pytest.approx(0.2)


def test_new_weights_candidate_already_held():
    nw = new_weights(pd.Series({"VOO": 0.5, "GLD": 0.5}), "GLD", 0.1)
    assert nw["GLD"] == pytest.approx(0.55) and len(nw) == 2


def test_alpha_out_of_range():
    with pytest.raises(ValueError):
        new_weights(pd.Series({"VOO": 1.0}), "GLD", 1.0)


def test_capm_market_beta_is_one_and_formula():
    p = FakeStore().prices
    er = capm_expected_returns(p, "2021-01-01", "2026-10-20", "VT", rf=0.04, mrp=0.05)
    assert er["VT"] == pytest.approx(0.09)
    assert er["BIL"] == pytest.approx(0.04, abs=1e-6)


def test_rf_from_proxy():
    assert rf_from_proxy(FakeStore().prices, "BIL") == pytest.approx((1 + 0.04 / 252) ** 252 - 1)


def test_base_matches_main_engine(wi):
    from engine.analyze import analyze
    a = analyze({"VOO": 60, "QQQ": 40}, store=FakeStore())
    assert wi.base["normal"].metrics["effective_bets"] == pytest.approx(a.normal.metrics["effective_bets"])
    assert wi.base["covid"].metrics["effective_bets"] == pytest.approx(a.stress["covid"].metrics["effective_bets"])


def test_all_windows_and_candidates_present(wi):
    assert set(wi.base) == {"normal", "covid", "2022"}
    assert set(wi.grid[wi.grid.window == "normal"].candidate) == {"(current)", *wi.candidates}


def test_hedges_beat_more_equity(wi):
    t = ranking_table(wi, "normal", 0.1)
    assert t.loc["IEF", "d_bets"] > t.loc["VXUS", "d_bets"]
    assert t.loc["GLD", "d_bets"] > t.loc["EEM", "d_bets"]
    assert list(t["rank"]) == list(range(1, len(t) + 1))


def test_stress_table_uses_drawdown(wi):
    t = ranking_table(wi, "2022", 0.1)
    assert t["d_max_dd"].notna().all()


def test_pareto():
    t = pd.DataFrame({"x": [0.3, 0.2, 0.1], "y": [-0.01, 0.0, -0.02]}, index=list("abc"))
    eff = mark_pareto(t, "x", "y")["efficient"]
    assert eff["a"] and eff["b"] and not eff["c"]


def test_stability_shares_sum_to_one(wi):
    s = ranking_stability(wi, n_boot=30)
    assert s.sum() == pytest.approx(1)


def test_dr_squared_hand_example():
    # 50/50, vols 20% and 10%, correlation 0.5: var_p = 0.0175, DR = 0.15 / sqrt(0.0175)
    cov = np.array([[0.04, 0.01], [0.01, 0.01]])
    m = diversification_metrics(np.array([0.5, 0.5]), cov)
    assert m["effective_bets"] == pytest.approx(0.0225 / 0.0175)


def test_shrinkage_choice_is_frozen_across_candidates():
    # 23 COVID returns: 7 holdings are not shrunk, 8 would be
    h = {"VOO": 1, "QQQ": 1, "STK1": 1, "STK2": 1, "STK3": 1, "STK4": 1, "STK5": 1}
    wi = run_whatif(h, store=FakeStore())
    b = wi.base["covid"]
    assert b.n_obs >= 3 * len(b.tickers) and b.shrinkage == 0
    g = wi.grid[(wi.grid.window == "covid") & (wi.grid.candidate != "(current)")]
    assert not g["low_confidence"].any()


def test_already_held_candidate_is_flagged(wi):
    wi2 = run_whatif({"VOO": 50, "GLD": 50}, store=FakeStore())
    t = ranking_table(wi2, "normal", 0.1)
    assert t.loc["GLD", "held"] == "already held"
    assert t.loc["IEF", "held"] == ""


def test_rank_by_drawdown_in_stress(wi):
    t = ranking_table(wi, "2022", 0.1, rank_by="d_max_dd")
    assert t["d_max_dd"].is_monotonic_decreasing


def test_summary_mentions_top_candidate(wi):
    from engine.whatif import summary_sentence
    t = ranking_table(wi, "normal", 0.1)
    assert t.index[0] in summary_sentence(wi, t, "normal", 0.1)
    t2 = ranking_table(wi, "covid", 0.1, rank_by="d_max_dd")
    s2 = summary_sentence(wi, t2, "covid", 0.1, rank_by="d_max_dd")
    assert "fall" in s2


real = pytest.mark.skipif(not PRICES.exists(), reason="run scripts/fetch_universe.py first")


@real
def test_real_data_base_matches_main_tab():
    from engine import analyze
    h = {"VOO": 30, "QQQ": 25, "VTI": 15, "NVDA": 10, "AGG": 10, "GLD": 10}
    w, a = run_whatif(h), analyze(h)
    assert w.base["normal"].metrics["effective_bets"] == pytest.approx(a.normal.metrics["effective_bets"])
    assert w.base["2022"].metrics["effective_bets"] == pytest.approx(a.stress["2022"].metrics["effective_bets"])
    assert 0 < w.rf < 0.1


def test_new_money_equals_pro_rata_mix():
    from engine.whatif import purchase_alpha
    w = pd.Series({"VOO": 0.6, "QQQ": 0.4})
    V, A = 10_000, 2_500
    by_money = (V * w).add(pd.Series({"GLD": A}), fill_value=0) / (V + A)
    pd.testing.assert_series_equal(by_money.sort_index(), new_weights(w, "GLD", purchase_alpha(V, A)).sort_index(),
                                   check_names=False)


def test_purchase_alpha_validation():
    from engine.whatif import purchase_alpha
    assert purchase_alpha(10_000, 2_500) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        purchase_alpha(10_000, 0)


def test_check_purchase_impact_and_dollars():
    from engine.whatif import check_purchase, purchase_sentence
    pc = check_purchase({"VOO": 60, "QQQ": 40}, "tlt", 2_500, value=10_000, store=FakeStore())
    assert pc.ticker == "TLT" and pc.alpha == pytest.approx(0.2)
    assert set(pc.impact.index) == {"normal", "covid", "2022"}
    r = pc.impact.loc["normal"]
    assert r["bets_after"] > r["bets_before"]
    assert r["loss_after"] == pytest.approx(r["dd_after"] * 12_500)
    assert "TLT" in purchase_sentence(pc)


def test_check_purchase_more_equity_vs_hedge():
    from engine.whatif import check_purchase
    eq = check_purchase({"VOO": 60, "QQQ": 40}, "VT", 2_500, store=FakeStore()).impact.loc["normal"]
    hedge = check_purchase({"VOO": 60, "QQQ": 40}, "IEF", 2_500, store=FakeStore()).impact.loc["normal"]
    assert hedge["bets_after"] - hedge["bets_before"] > eq["bets_after"] - eq["bets_before"]


def test_check_purchase_unknown_ticker():
    from engine.whatif import check_purchase
    with pytest.raises(ValueError):
        check_purchase({"VOO": 100}, "NOPE", 1_000, store=FakeStore())


def test_custom_candidates_are_compared():
    wi2 = run_whatif({"VOO": 60, "QQQ": 40}, candidates={"IEF": "bonds", "VT": "world"}, store=FakeStore())
    assert set(ranking_table(wi2, "normal", 0.1).index) == {"IEF", "VT"}
