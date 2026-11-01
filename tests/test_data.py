import numpy as np
import pandas as pd
import pytest

from portfolio_risk import data


def weekly(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="W-MON"), dtype=float)


def test_currency_conversion_multiplies_by_aligned_fx():
    sgd = weekly([10, 11, 12, 13])
    fx = weekly([0.70, 0.72, 0.71, 0.73])
    pd.testing.assert_series_equal(data.convert_to_usd(sgd, fx), sgd * fx)


def test_currency_conversion_forward_fills_missing_fx_weeks():
    sgd = weekly([10, 11, 12])
    fx = sgd.copy()
    fx[:] = [0.7, 0.7, 0.8]
    fx = fx.drop(fx.index[1])
    out = data.convert_to_usd(sgd, fx)
    assert out.iloc[1] == pytest.approx(11 * 0.7)
    assert out.iloc[2] == pytest.approx(12 * 0.8)


def test_holiday_gap_is_forward_filled_not_merged():
    a = weekly([100, 101, 102, 103])
    b = a.drop(a.index[2]) * 2
    prices = pd.DataFrame({"A": a, "B": b})
    r = data.build_returns(prices)
    assert len(r) == 4
    assert r["B"].iloc[2] == pytest.approx(0.0)
    assert r["B"].iloc[3] == pytest.approx(103 * 2 / (101 * 2) - 1)


def test_common_window_truncates_to_shortest_history():
    long = weekly(np.linspace(100, 120, 10))
    short = weekly(np.linspace(50, 55, 6), start="2024-01-29")
    r = data.build_returns(pd.DataFrame({"L": long, "S": short}))
    window = data.common_window(r, ["L", "S"])
    assert len(window) == 5
    assert window.index[0] == short.index[1]


def test_roll_jump_flagged_for_futures_only():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 2)), columns=["CL=F", "SPY"])
    r.index = pd.date_range("2021-01-04", periods=200, freq="W-MON")
    r.iloc[50] = 0.5
    flags = data.flag_roll_jumps(r, ["CL=F"])
    assert list(flags["week"]) == [r.index[50]]


def test_high_correlation_pairs_detected():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    df = pd.DataFrame({"A": x, "B": x + rng.normal(0, 0.001, 100), "C": rng.normal(size=100)})
    pairs = data.high_correlations(df)
    assert list(zip(pairs["asset_a"], pairs["asset_b"])) == [("A", "B")]


def test_default_currency():
    assert data.default_currency("D05.SI") == "SGD"
    assert data.default_currency("AAPL") == "USD"


def test_common_window_tolerates_duplicate_tickers():
    r = data.build_returns(pd.DataFrame({"A": weekly([1, 2, 3, 4]), "VOO": weekly([5, 6, 7, 8])}))
    assert list(data.common_window(r, ["A", "VOO", "VOO"]).columns) == ["A", "VOO"]
