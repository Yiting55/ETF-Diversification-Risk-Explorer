import numpy as np
import pandas as pd
import pytest

from engine.correlation import average_pairwise, compare_correlations, rolling_correlation
from engine.scenarios import custom_window


def sample_prices(n=500):
    rng = np.random.default_rng(17)
    returns = rng.normal(0, 0.01, (n, 3))
    return pd.DataFrame(100 * np.cumprod(1 + returns, axis=0),
                        index=pd.bdate_range("2020-01-01", periods=n), columns=["A", "B", "C"])


def test_rolling_average_matches_all_pairs_in_trailing_sample():
    prices = sample_prices(100)
    result = rolling_correlation(prices, 20)
    expected = prices.pct_change(fill_method=None).iloc[-20:].corr()
    assert result.iloc[-1] == pytest.approx(average_pairwise(expected))
    assert result.iloc[:20].isna().all()
    assert result.iloc[20:].notna().all()


def test_selected_pair_matches_pearson_and_does_not_use_future_data():
    prices = sample_prices(100)[["A", "B"]]
    before = rolling_correlation(prices, 20)
    assert before.iloc[50] == pytest.approx(prices.pct_change().iloc[31:51].corr().iloc[0, 1])
    prices.iloc[51:] *= 2
    pd.testing.assert_series_equal(before.iloc[:51], rolling_correlation(prices, 20).iloc[:51])


def test_missing_prices_leave_gaps_without_changing_pair_membership():
    prices = sample_prices(100)
    prices.iloc[40, 2] = np.nan
    result = rolling_correlation(prices, 20)
    assert result.iloc[39] == pytest.approx(average_pairwise(prices.pct_change(fill_method=None).iloc[20:40].corr()))
    assert result.iloc[40:61].isna().all()
    assert pd.notna(result.iloc[61])


def test_single_holding_short_and_constant_windows_are_undefined():
    prices = sample_prices(40)
    assert rolling_correlation(prices[["A"]], 20).isna().all()
    assert rolling_correlation(prices, 60).isna().all()
    prices["C"] = 100.0
    assert rolling_correlation(prices, 20).isna().all()
    assert np.isnan(average_pairwise(pd.DataFrame([[1.0]])))


def test_comparison_uses_previous_year_and_identical_holdings():
    prices = sample_prices()
    scenario = custom_window("2021-02-01", "2021-05-01")
    result = compare_correlations(prices, scenario)
    assert result.baseline.end < scenario.start
    assert result.baseline.start >= scenario.start - pd.DateOffset(years=1)
    assert result.stress.start >= scenario.start
    assert result.stress.end <= scenario.end
    assert list(result.baseline.corr) == list(result.stress.corr) == list(prices)
    baseline = prices.loc["2020-02-01":"2021-01-31"].pct_change().iloc[1:]
    assert result.baseline.n_obs == len(baseline)
    np.testing.assert_allclose(result.baseline.corr, baseline.corr(), atol=1e-12)


def test_young_and_constant_holdings_are_excluded_from_both_periods():
    prices = sample_prices()
    prices.loc[:"2021-01-31", "NEW"] = np.nan
    prices.loc["2021-02-01":, "NEW"] = prices.loc["2021-02-01":, "A"]
    prices["CASH"] = 100.0
    result = compare_correlations(prices, custom_window("2021-02-01", "2021-05-01"))
    assert result.excluded == ["NEW", "CASH"]
    assert list(result.baseline.corr) == list(result.stress.corr) == ["A", "B", "C"]


@pytest.mark.parametrize("start,end", [("1990-01-01", "1990-02-01"),
                                       ("2021-02-01", "2021-02-05")])
def test_unavailable_periods_have_explanation(start, end):
    result = compare_correlations(sample_prices(), custom_window(start, end))
    assert result.baseline is None and result.stress is None
    assert result.note


def test_short_stress_sample_reports_shrinkage():
    prices = sample_prices()
    # 12 holdings, but only 20 return observations in the stress period.
    prices = pd.concat([prices.add_suffix(str(i)) for i in range(4)], axis=1)
    result = compare_correlations(prices, custom_window("2021-02-01", "2021-03-01"))
    assert result.stress.low_confidence
    assert 0 <= result.stress.shrinkage <= 1
    assert not result.baseline.low_confidence


def test_comparison_detects_increase_in_correlation():
    prices = sample_prices()
    start = pd.Timestamp("2021-02-01")
    # Make B identical to A during stress; baseline returns remain independent.
    prices.loc[start:, "B"] = prices.loc[start:, "A"].to_numpy()
    result = compare_correlations(prices, custom_window(start, "2021-05-01"))
    assert result.stress.corr.loc["A", "B"] == pytest.approx(1)
    assert result.stress.corr.loc["A", "B"] - result.baseline.corr.loc["A", "B"] > 0.8
