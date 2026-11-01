import numpy as np
import pandas as pd
import pytest

from portfolio_risk import analytics


def test_weights_normalise_to_one():
    w = analytics.normalise_weights(pd.Series({"A": 30.0, "B": 50.0, "C": 70.0}))
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(0.2)


def test_weights_reject_zero_total():
    with pytest.raises(ValueError):
        analytics.normalise_weights(pd.Series({"A": 0.0, "B": 0.0}))


def test_voo_beta_is_one(returns):
    b = analytics.betas(returns, returns["VOO"])
    assert b["VOO"] == pytest.approx(1.0)


def test_portfolio_beta_is_weighted_sum(returns):
    b = analytics.betas(returns, returns["VOO"])
    w = pd.Series(1 / 3, index=["A", "B", "C"])
    port = analytics.portfolio_returns(returns, w)
    assert port.cov(returns["VOO"]) / returns["VOO"].var() == pytest.approx(w @ b[w.index])


def test_annual_covariance_is_weekly_times_52(returns):
    cols = ["A", "B", "C"]
    annual = analytics.annual_covariance(returns[cols], "sample")
    np.testing.assert_allclose(annual.to_numpy(), returns[cols].cov().to_numpy() * 52)


def test_ledoit_wolf_is_52_times_its_weekly_estimate(returns):
    from sklearn.covariance import LedoitWolf

    cols = ["A", "B", "C"]
    weekly = LedoitWolf().fit(returns[cols].to_numpy()).covariance_
    annual = analytics.annual_covariance(returns[cols], "ledoit_wolf")
    np.testing.assert_allclose(annual.to_numpy(), weekly * 52)


def test_portfolio_variance_matches_manual_three_asset_example():
    cov = pd.DataFrame(
        [[0.04, 0.006, 0.002], [0.006, 0.09, 0.012], [0.002, 0.012, 0.16]],
        index=list("ABC"),
        columns=list("ABC"),
    )
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    manual = (
        0.5**2 * 0.04 + 0.3**2 * 0.09 + 0.2**2 * 0.16
        + 2 * (0.5 * 0.3 * 0.006 + 0.5 * 0.2 * 0.002 + 0.3 * 0.2 * 0.012)
    )
    assert analytics.portfolio_variance(w, cov) == pytest.approx(manual)
    assert analytics.portfolio_volatility(w, cov) == pytest.approx(np.sqrt(manual))


def test_risk_contributions_sum_to_one(returns):
    cov = analytics.annual_covariance(returns[["A", "B", "C", "D"]], "sample")
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    assert analytics.risk_contributions(w, cov).sum() == pytest.approx(1.0)


def test_shrunk_returns_move_toward_cross_sectional_mean(returns):
    cols = ["A", "B", "C", "D"]
    raw = analytics.expected_returns(returns[cols], "raw")
    shrunk = analytics.expected_returns(returns[cols], "shrunk", 0.5)
    np.testing.assert_allclose(shrunk, 0.5 * raw + 0.5 * raw.mean())
    assert analytics.expected_returns(returns[cols], "shrunk", 1.0).nunique() == 1


def test_max_drawdown_and_var():
    r = pd.Series([0.10, -0.20, 0.05, -0.10])
    wealth = [1.1, 0.88, 0.924, 0.8316]
    assert analytics.max_drawdown(r) == pytest.approx(0.8316 / 1.1 - 1)
    assert analytics.historical_var(pd.Series(np.arange(-50, 50) / 100)) > 0


def test_ewma_with_a_very_long_half_life_matches_ledoit_wolf(returns):
    r = returns[["A", "B", "C", "D"]]
    ewma = analytics.annual_covariance(r, "ewma", half_life=1e9)
    lw = analytics.annual_covariance(r, "ledoit_wolf")
    np.testing.assert_allclose(ewma.to_numpy(), lw.to_numpy(), atol=1e-6)


def test_ewma_weights_recent_weeks_more_than_ledoit_wolf():
    rng = np.random.default_rng(3)
    n = 260
    scale = np.where(np.arange(n) >= n - 52, 0.04, 0.02)
    r = pd.DataFrame(rng.normal(0, 1, (n, 3)) * scale[:, None], columns=["A", "B", "C"])
    ewma = analytics.annual_covariance(r, "ewma", half_life=52)
    lw = analytics.annual_covariance(r, "ledoit_wolf")
    assert ewma.loc["A", "A"] > lw.loc["A", "A"]


def test_effective_weeks():
    assert 149 < analytics.effective_weeks(10_000, 52) < 151
    assert analytics.effective_weeks(100, 1e12) == pytest.approx(100)


def test_ewma_rejects_a_non_positive_half_life(returns):
    with pytest.raises(ValueError):
        analytics.annual_covariance(returns[["A", "B"]], "ewma", half_life=0)
