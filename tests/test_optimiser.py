import math

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from portfolio_risk import analytics, optimiser

COLS = ["A", "B", "C", "D", "E"]


@pytest.fixture
def inputs(returns):
    r = returns[COLS]
    return analytics.expected_returns(r, "raw"), analytics.annual_covariance(r, "sample")


def check_valid(w: pd.Series, cap):
    assert (w >= -1e-9).all()
    assert w.sum() == pytest.approx(1.0)
    if cap is not None:
        assert (w <= cap + 1e-5).all()


@pytest.mark.parametrize("cap", [None, 0.3, 0.2])
def test_weights_valid_and_capped(inputs, cap):
    mu, cov = inputs
    check_valid(optimiser.min_variance(cov, cap).weights, cap)
    check_valid(optimiser.max_sharpe(mu, cov, 0.01, cap).weights, cap)


def test_risk_parity_equalises_risk_contributions(inputs):
    _, cov = inputs
    res = optimiser.risk_parity(cov, None)
    check_valid(res.weights, None)
    rc = analytics.risk_contributions(res.weights, cov)
    np.testing.assert_allclose(rc, 1 / len(cov), atol=1e-4)


def test_risk_parity_warns_on_cap_breach(inputs):
    _, cov = inputs
    res = optimiser.risk_parity(cov, 0.2001)
    assert res.warnings


def test_max_sharpe_beats_single_assets_and_equal_weight(inputs):
    mu, cov = inputs
    rf = 0.01
    w = optimiser.max_sharpe(mu, cov, rf, None).weights
    sharpe = lambda x: (x @ mu - rf) / analytics.portfolio_volatility(x, cov)
    best = sharpe(w)
    for asset in COLS:
        single = pd.Series(0.0, index=COLS)
        single[asset] = 1.0
        assert best >= sharpe(single) - 1e-6
    assert best >= sharpe(pd.Series(1 / len(COLS), index=COLS)) - 1e-6


def test_min_variance_is_lowest_variance_portfolio(inputs):
    _, cov = inputs
    w = optimiser.min_variance(cov, None).weights
    for asset in COLS:
        single = pd.Series(0.0, index=COLS)
        single[asset] = 1.0
        assert analytics.portfolio_variance(w, cov) <= analytics.portfolio_variance(single, cov) + 1e-9


@pytest.mark.parametrize("cap, n", [(0.2, 4), (0.5, 1), (0.15, 6)])
def test_refuses_too_few_assets(returns, cap, n):
    assert n < math.ceil(1 / cap)
    cols = COLS[: min(n, 5)]
    mu = analytics.expected_returns(returns[cols], "raw")
    cov = analytics.annual_covariance(returns[cols], "sample")
    with pytest.raises(optimiser.OptimisationError):
        optimiser.min_variance(cov, cap)
    with pytest.raises(optimiser.OptimisationError):
        optimiser.max_sharpe(mu, cov, 0.0, cap)
    with pytest.raises(optimiser.OptimisationError):
        optimiser.efficient_frontier(mu, cov, cap)


def test_max_sharpe_errors_when_no_asset_beats_risk_free(inputs):
    mu, cov = inputs
    with pytest.raises(optimiser.OptimisationError):
        optimiser.max_sharpe(mu, cov, 5.0, None)


def test_frontier_is_monotone_and_complete(inputs):
    mu, cov = inputs
    f = optimiser.efficient_frontier(mu, cov, 0.4)
    assert len(f) == 40
    assert f["return"].is_monotonic_increasing
    assert f["volatility"].is_monotonic_increasing or (f["volatility"].diff().dropna() > -1e-6).all()


def reference_frontier_weights(mu, cov, cap, points=40):
    """The original formulation: a fresh quad_form problem and variable for every target."""
    n = len(cov)
    psd = cp.psd_wrap((cov.to_numpy() + cov.to_numpy().T) / 2)

    def solve(objective, constraints, w):
        cp.Problem(objective, constraints).solve(solver=cp.CLARABEL, **optimiser.SOLVER_OPTIONS)
        return w.value

    def box(w):
        out = [cp.sum(w) == 1, w >= 0]
        return out + ([w <= cap] if cap is not None else [])

    w = cp.Variable(n)
    low = float(mu.to_numpy() @ solve(cp.Minimize(cp.quad_form(w, psd)), box(w), w))
    w = cp.Variable(n)
    high = float(mu.to_numpy() @ solve(cp.Maximize(mu.to_numpy() @ w), box(w), w))
    rows = []
    for target in np.linspace(low, high, points):
        w = cp.Variable(n)
        value = solve(cp.Minimize(cp.quad_form(w, psd)), [*box(w), mu.to_numpy() @ w >= target - 1e-9], w)
        row = pd.Series(value, index=cov.index)
        row[row.abs() < 1e-10] = 0.0
        rows.append(row / row.sum())
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.mark.parametrize("cap", [None, 0.4, 0.25])
def test_forty_point_frontier_matches_non_parametrised_version(inputs, cap):
    mu, cov = inputs
    fast = optimiser.frontier_weights(mu, cov, cap, points=40)
    slow = reference_frontier_weights(mu, cov, cap, points=40)
    assert fast.shape == slow.shape == (40, len(COLS))
    assert (fast - slow).abs().to_numpy().max() < 1e-6


def test_frontier_problem_is_dpp_and_compiled_once(inputs):
    mu, cov = inputs
    ws = optimiser.Workspace(len(cov))
    optimiser.frontier_weights(mu, cov, 0.4, points=5, workspace=ws)
    built = dict(ws._built)
    assert all(problem.is_dpp() for problem, _ in built.values())
    optimiser.frontier_weights(mu * 1.1, cov * 0.9, 0.3, points=5, workspace=ws)
    assert ws._built.keys() == built.keys()
    assert all(ws._built[k][0] is built[k][0] for k in built)


def test_shared_workspace_gives_the_same_answers_as_a_fresh_one(inputs):
    mu, cov = inputs
    ws = optimiser.Workspace(len(cov))
    optimiser.min_variance(cov * 2, 0.5, ws)
    for shared, fresh in [
        (optimiser.min_variance(cov, 0.3, ws), optimiser.min_variance(cov, 0.3)),
        (optimiser.max_sharpe(mu, cov, 0.01, 0.3, ws), optimiser.max_sharpe(mu, cov, 0.01, 0.3)),
    ]:
        assert (shared.weights - fresh.weights).abs().max() < 1e-8
