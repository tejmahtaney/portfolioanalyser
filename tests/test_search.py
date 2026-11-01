import numpy as np
import pandas as pd
import pytest

from portfolio_risk import analytics, optimiser, search

HELD = ["H1", "H2", "H3"]


def make_returns(n=600, seed=3, extra=None, start="2014-01-06"):
    """Correlated stocks driven by one factor, plus IND, which is independent of everything."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.002, 0.02, n)
    names = HELD + [f"C{i}" for i in range(6)]
    loadings = rng.uniform(0.6, 1.4, len(names))
    frame = pd.DataFrame(
        factor[:, None] * loadings + rng.normal(0, 0.01, (n, len(names))),
        index=pd.date_range(start, periods=n, freq="W-MON"),
        columns=names,
    )
    frame["IND"] = rng.normal(0.002, 0.02, n)
    for name, values in (extra or {}).items():
        frame[name] = values
    return frame


def holdings(**kw):
    return pd.Series({"H1": 50.0, "H2": 30.0, "H3": 20.0}, **kw)


def candidates(returns):
    return [c for c in returns.columns if c not in HELD]


# --- solver --------------------------------------------------------------------------------------------------


def random_covariance(rng):
    """Annual covariance of 2 to 24 assets: volatilities 10 to 50%, one common factor, varied idiosyncratic risk."""
    n = int(rng.integers(2, 25))
    f = rng.normal(size=n) * 0.5 + 0.7
    corr = np.outer(f, f) * 0.5 + np.diag(rng.uniform(0.2, 1, n))
    d = np.sqrt(np.diag(corr))
    vol = rng.uniform(0.1, 0.5, n)
    return corr / np.outer(d, d) * np.outer(vol, vol)


def spread(w, cov):
    contrib = w * (cov @ w)
    return contrib.max() / contrib.min() - 1


def test_solver_matches_cvxpy_on_random_covariances():
    """optimiser.risk_parity stops short of the exact solution on some matrices (its risk contributions differ by up
    to ~1e-4), so the 1e-6 comparison is made where the reference itself is converged. The solver under test is
    held to the exact condition on every matrix."""
    rng = np.random.default_rng(0)
    compared = 0
    for _ in range(50):
        cov = random_covariance(rng)
        fast = search.risk_parity_weights(cov)
        assert spread(fast, cov) < 1e-9
        names = [f"A{i}" for i in range(len(cov))]
        slow = optimiser.risk_parity(pd.DataFrame(cov, index=names, columns=names), None).weights.to_numpy()
        assert np.abs(fast - slow).max() < 1e-4
        if spread(slow, cov) < 1e-6:
            compared += 1
            assert np.abs(fast - slow).max() < 1e-6
    assert compared >= 40


def test_solver_equalises_risk_contributions():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(8, 20))
    cov = a @ a.T
    w = search.risk_parity_weights(cov)
    contrib = w * (cov @ w)
    assert contrib.max() / contrib.min() == pytest.approx(1, abs=1e-8)


# --- band ----------------------------------------------------------------------------------------------------


def test_band_leaves_small_gap_untraded_and_trades_large_gap():
    table = search.trade_table(
        pd.Series({"A": 0.0405, "B": 0.06}), pd.Series({"A": 0.04, "B": 0.04})
    )
    assert np.isnan(table.loc["A", "Trade"])
    assert table.loc["B", "Trade"] == pytest.approx(0.02)
    assert table.loc["B", "Current"] == 0.04 and table.loc["B", "Target"] == 0.06


def test_band_trades_new_positions_and_full_exits():
    table = search.trade_table(pd.Series({"NEW": 0.03}), pd.Series({"OLD": 0.02}))
    assert table.loc["NEW", "Trade"] == pytest.approx(0.03)
    assert table.loc["OLD", "Trade"] == pytest.approx(-0.02)


# --- search --------------------------------------------------------------------------------------------------


def test_uncorrelated_candidate_is_in_top_set():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=1)
    assert res.finalists[0].tickers == ("IND",)
    assert len(res.finalists) == 5
    ratios = [f.diversification_ratio for f in res.finalists]
    assert ratios == sorted(ratios, reverse=True)


def test_uncorrelated_candidate_is_in_top_set_at_depth_two():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=2)
    assert "IND" in res.finalists[0].tickers


def test_held_tickers_are_not_candidates():
    r = make_returns()
    res = search.search(r, holdings(), [*HELD, "IND"], max_new=1)
    assert [f.tickers for f in res.finalists] == [("IND",)]


def test_keep_mode_preserves_internal_proportions():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=2, mode="keep")
    props = holdings() / holdings().sum()
    for f in res.finalists:
        held = f.weights[HELD]
        assert (held / held.sum() - props).abs().max() < 1e-9
        assert f.weights.sum() == pytest.approx(1)


def test_reweight_mode_changes_proportions():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=1, mode="reweight")
    held = res.finalists[0].weights[HELD]
    props = holdings() / holdings().sum()
    assert (held / held.sum() - props).abs().max() > 1e-3


def test_short_history_candidate_is_excluded_and_listed():
    r = make_returns()
    r.loc[r.index[:-100], "C0"] = np.nan  # 100 weeks only
    res = search.search(r, holdings(), candidates(r), max_new=1, unavailable={"GONE": "Delisted."})
    assert "C0" in res.excluded and "100" in res.excluded["C0"]
    assert all("C0" not in f.tickers for f in res.finalists)
    res = search.search(r, holdings(), [*candidates(r), "GONE"], max_new=1, unavailable={"GONE": "Delisted."})
    assert res.excluded["GONE"] == "Delisted."


def test_short_history_holding_is_an_error_naming_the_ticker():
    r = make_returns()
    r.loc[r.index[:-100], "H2"] = np.nan
    with pytest.raises(ValueError, match="H2"):
        search.search(r, holdings(), candidates(r), max_new=1)


def test_candidate_with_156_to_full_weeks_is_searched_but_not_backtested():
    r = make_returns()
    r.loc[r.index[:-200], "C1"] = np.nan
    res = search.search(r, holdings(), ["C1", "IND"], max_new=1, beam_width=2)
    by_ticker = {f.tickers: f for f in res.finalists}
    assert by_ticker[("C1",)].backtest.status == search.NO_BACKTEST
    assert by_ticker[("IND",)].backtest.status in (search.PASSES, search.FAILS)
    assert "C1" not in res.excluded


def greedy(returns, held, props, cands, max_new, half_life=52):
    """Plain greedy search, scored through analytics and cvxpy rather than the search module's solver."""
    window = returns.iloc[-search.WINDOW_WEEKS :]

    def ratio(names):
        cov = analytics.annual_covariance(window[names], "ewma", half_life)
        w = optimiser.risk_parity(cov, None).weights
        sigma = pd.Series(np.sqrt(np.diag(cov)), index=cov.index)
        return float(w @ sigma / analytics.portfolio_volatility(w, cov))

    chosen, best = [], None
    for _ in range(max_new):
        scored = {c: ratio([*held, *chosen, c]) for c in sorted(cands) if c not in chosen}
        pick = max(scored, key=lambda c: (scored[c], c))
        if best is not None and scored[pick] <= best:
            break
        chosen.append(pick)
        best = scored[pick]
    return tuple(sorted(chosen))


def test_beam_width_one_equals_greedy():
    r = make_returns()
    cands = candidates(r)
    res = search.search(r, holdings(), cands, max_new=3, beam_width=1, backtest=False)
    assert len(res.finalists) == 1
    assert res.finalists[0].tickers == greedy(r, HELD, None, cands, 3)


def test_wider_beam_is_never_worse_than_greedy():
    r = make_returns()
    narrow = search.search(r, holdings(), candidates(r), max_new=3, beam_width=1, backtest=False)
    wide = search.search(r, holdings(), candidates(r), max_new=3, beam_width=5, backtest=False)
    assert wide.finalists[0].diversification_ratio >= narrow.finalists[0].diversification_ratio - 1e-12


def test_progress_reports_each_depth_and_best_by_depth():
    r = make_returns()
    calls = []
    res = search.search(
        r, holdings(), candidates(r), max_new=3, backtest=False, progress=lambda d, m, b: calls.append((d, m))
    )
    assert calls[:3] == [(1, 3), (2, 3), (3, 3)][: len(calls)]
    assert set(res.best_by_depth) == set(range(1, res.depth))


def test_max_new_out_of_range():
    r = make_returns()
    for bad in (0, 6):
        with pytest.raises(ValueError):
            search.search(r, holdings(), candidates(r), max_new=bad)


# --- backtest ------------------------------------------------------------------------------------------------


def covid_returns(crash):
    """Weekly returns on dates that include Feb to Apr 2020, which sit in the out-of-sample region."""
    r = make_returns(n=600, start="2014-06-02")
    if crash:
        spring = (r.index >= "2020-02-03") & (r.index <= "2020-04-27")
        r.loc[spring, :] -= 0.12
    return r


def test_covid_figures_differ_when_crash_is_planted():
    r = covid_returns(crash=True)
    res = search.search(r, holdings(), candidates(r), max_new=1, beam_width=1)
    bt = res.finalists[0].backtest
    assert bt.status in (search.PASSES, search.FAILS)
    assert bt.ex_covid.weeks < bt.full.weeks
    assert bt.ex_covid.sharpe != pytest.approx(bt.full.sharpe)
    assert bt.ex_covid.max_drawdown > bt.full.max_drawdown  # the crash drove the full-sample drawdown


def test_covid_figures_barely_move_without_a_crash():
    r = covid_returns(crash=False)
    bt = search.search(r, holdings(), candidates(r), max_new=1, beam_width=1).finalists[0].backtest
    assert abs(bt.ex_covid.max_drawdown - bt.full.max_drawdown) < 0.2
    assert not bt.covid_sensitive or bt.status != bt.status_ex_covid


def test_fails_when_sharpe_is_more_than_one_se_below_baseline():
    base = pd.Series(np.random.default_rng(0).normal(0.004, 0.01, 300))
    bad = base - 0.02
    figures, status = search._figures(bad, base, 0.0)
    assert status == search.FAILS
    figures, status = search._figures(base, base, 0.0)
    assert status == search.PASSES
    sr = figures.baseline_sharpe
    assert figures.standard_error == pytest.approx(np.sqrt((1 + sr**2 / 2) / (300 / 52)))


def test_keep_mode_baseline_is_current_weights_held_fixed():
    r = make_returns()
    props = holdings() / holdings().sum()
    base = search._baseline(r[HELD], HELD, props, "keep", 52, 156, 13)
    assert np.allclose(base, r[HELD].iloc[156:] @ props)


def test_finalist_order_follows_diversification_ratio_not_sharpe():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=2)
    ratios = [f.diversification_ratio for f in res.finalists]
    assert ratios == sorted(ratios, reverse=True)
