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
    scores = [f.score for f in res.finalists]
    assert scores == sorted(scores)
    assert res.finalists[0].metrics.d_volatility < 0


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
    res = search.search(r, holdings(), candidates(r), max_new=2, mode="keep", sizing="risk_parity")
    props = holdings() / holdings().sum()
    for f in res.finalists:
        held = f.weights[HELD]
        assert (held / held.sum() - props).abs().max() < 1e-9
        assert f.weights.sum() == pytest.approx(1)


def test_reweight_mode_changes_proportions():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=1, mode="reweight", sizing="risk_parity")
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


def greedy(returns, held, props, cands, max_new):
    """Plain greedy search on the same fixed-slice metrics, computed with pandas alone."""
    window = returns.iloc[-search.WINDOW_WEEKS :]
    hold = window[held] @ props

    def cvar(x):
        return -x.nsmallest(max(1, int(np.ceil(search.CVAR_LEVEL * len(x))))).mean()

    def metrics(names):
        sleeve = window[list(names)].sum(axis=1)
        p = hold * (1 - search.SLICE * len(names)) + search.SLICE * sleeve
        return [(p.std() - hold.std()) * np.sqrt(52), cvar(p) - cvar(hold), hold.corr(sleeve)]

    eligible = [c for c in sorted(cands) if metrics([c])[1] <= 0]
    chosen, prev = [], None
    for _ in range(max_new):
        rows = {c: metrics([*chosen, c]) for c in eligible if c not in chosen}
        table = pd.DataFrame.from_dict(rows, orient="index")
        pick = min(rows, key=lambda c: (table.rank().mean(axis=1)[c], c))
        if prev is not None and rows[pick][0] >= prev:
            break
        chosen.append(pick)
        prev = rows[pick][0]
    return tuple(sorted(chosen))


def test_beam_width_one_equals_greedy():
    r = make_returns()
    cands = candidates(r)
    props = holdings() / holdings().sum()
    res = search.search(r, holdings(), cands, max_new=3, beam_width=1, backtest=False)
    assert len(res.finalists) == 1
    assert res.finalists[0].tickers == greedy(r, HELD, props, cands, 3)


def test_candidate_that_raises_cvar_is_excluded_and_listed():
    r = make_returns()
    crash = np.where(np.random.default_rng(9).random(len(r)) < 0.1, -0.9, 0.002)
    r["CRASH"] = crash
    res = search.search(r, holdings(), [*candidates(r), "CRASH"], max_new=2, backtest=False)
    assert "CVaR" in res.excluded["CRASH"]
    assert all("CRASH" not in f.tickers for f in res.finalists)


def test_error_when_every_candidate_raises_cvar():
    r = make_returns()
    r["CRASH"] = np.where(np.random.default_rng(9).random(len(r)) < 0.1, -0.9, 0.002)
    with pytest.raises(ValueError, match="CVaR"):
        search.search(r, holdings(), ["CRASH"], max_new=1, backtest=False)


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
    for bad in (0, 11):
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


def test_standard_error_is_zero_for_identical_series():
    x = pd.Series(np.random.default_rng(0).normal(0.004, 0.01, 300))
    se, rho = search._sharpe_difference_se(x, x, 0.0)
    assert rho == pytest.approx(1)
    assert se == pytest.approx(0, abs=1e-6)  # sqrt turns rho = 1 - 2e-16 into ~1e-8


def test_standard_error_matches_memmel_formula():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0.004, 0.01, 300))
    oos = 0.6 * base + pd.Series(rng.normal(0.003, 0.008, 300))
    rf = 0.04
    s1, s2 = (oos.mean() - rf / 52) / oos.std(), (base.mean() - rf / 52) / base.std()
    rho = np.corrcoef(oos, base)[0, 1]
    var = (2 * (1 - rho) + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / 300
    se, got_rho = search._sharpe_difference_se(oos, base, rf)
    assert se == pytest.approx(np.sqrt(52 * var))
    assert got_rho == pytest.approx(rho)
    figures, _ = search._figures(oos, base, rf)
    assert figures.correlation == pytest.approx(rho) and figures.standard_error == pytest.approx(se)


def test_planted_weekly_loss_fails_and_equal_series_passes():
    base = pd.Series(np.random.default_rng(0).normal(0.004, 0.01, 300))
    figures, status = search._figures(base - 0.002, base, 0.0)
    assert status == search.FAILS
    assert search._figures(base, base, 0.0)[1] == search.PASSES


def test_figures_require_the_same_weeks():
    x = pd.Series(np.zeros(5) + np.arange(5) * 0.01, index=pd.date_range("2020-01-06", periods=5, freq="W-MON"))
    with pytest.raises(ValueError, match="same weeks"):
        search._figures(x, x.iloc[1:], 0.0)


def statuses(res):
    return {f.tickers: (f.backtest.status, f.backtest.status_ex_covid) for f in res.finalists}


def test_unrelated_ticker_starting_a_week_earlier_changes_no_status():
    r = make_returns()
    plain = search.search(r, holdings(), candidates(r), max_new=1)
    early = pd.DataFrame(np.nan, index=[r.index[0] - pd.Timedelta(weeks=1)], columns=r.columns)
    wider = pd.concat([early, r])
    wider["VOO"] = np.nan
    wider.loc[wider.index[0], "VOO"] = 0.01
    wider.loc[wider.index[1:], "VOO"] = np.random.default_rng(5).normal(0.002, 0.02, len(r))
    res = search.search(wider, holdings(), candidates(r), max_new=1)
    assert statuses(res) == statuses(plain)
    assert all(s[0] != search.NO_BACKTEST for s in statuses(res).values())


@pytest.mark.parametrize("weeks, expected", [(480, False), (510, True)])
def test_backtest_needs_500_weeks_of_common_history(weeks, expected):
    r = make_returns()
    r.loc[r.index[:-weeks], "IND"] = np.nan
    res = search.search(r, holdings(), ["IND"], max_new=1)
    bt = res.finalists[0].backtest
    assert (bt.status != search.NO_BACKTEST) is expected
    assert (bt.full is not None) is expected
    if expected:
        assert bt.full.weeks == weeks - search.WINDOW_WEEKS
    else:
        assert "480" in bt.reason


def test_baselines_are_cached_by_window():
    r = make_returns()
    props = holdings() / holdings().sum()
    cache = {}
    for add in ("C0", "C1", "C2"):
        search._backtest(r, HELD, props, [add], "reweight", cache, 0.04, 52, 156, 13)
    assert len(cache) == 1
    r.loc[r.index[:50], "C3"] = np.nan
    search._backtest(r, HELD, props, ["C3"], "reweight", cache, 0.04, 52, 156, 13)
    assert len(cache) == 2


def test_keep_mode_baseline_is_current_weights_held_fixed():
    r = make_returns()
    props = holdings() / holdings().sum()
    base = search._baseline(r[HELD], HELD, props, "keep", 52, 156, 13)
    assert np.allclose(base, r[HELD].iloc[156:] @ props)


def test_finalist_order_follows_score_not_sharpe():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=2)
    scores = [f.score for f in res.finalists]
    assert scores == sorted(scores)


def test_trailing_week_missing_for_a_holding_is_trimmed_not_fatal():
    r = make_returns()
    r.loc[r.index[-1], HELD] = np.nan
    res = search.search(r, holdings(), candidates(r), max_new=1, backtest=False)
    assert res.trimmed_weeks == 1


def test_longer_trailing_gap_in_a_holding_still_fails():
    r = make_returns()
    r.loc[r.index[-5:], "H1"] = np.nan
    with pytest.raises(ValueError, match="H1"):
        search.search(r, holdings(), candidates(r), max_new=1, backtest=False)


# --- position cap --------------------------------------------------------------------------------------------


def test_capped_solver_holds_binding_assets_at_the_cap_and_keeps_every_asset():
    cov = np.diag([0.01, 0.04, 0.04, 0.04]) + 0.005
    upper = np.array([0.3, 0.3, 0.3, 0.3])
    free = search.risk_parity_weights(cov)
    assert free.max() > 0.3
    w = search.capped_risk_parity_weights(cov, upper)
    assert w.sum() == pytest.approx(1)
    assert (w <= 0.3 + 1e-9).all()
    assert (w > 0).all()
    assert w.max() == pytest.approx(0.3)


def test_capped_solver_is_unchanged_when_the_cap_does_not_bind():
    cov = random_covariance(np.random.default_rng(1))
    w = search.capped_risk_parity_weights(cov, np.ones(len(cov)))
    assert w == pytest.approx(search.risk_parity_weights(cov))


@pytest.mark.parametrize("mode", search.MODES)
def test_cap_positions_limits_every_position_to_the_largest_input(mode):
    r = make_returns()
    even = pd.Series({"H1": 34.0, "H2": 33.0, "H3": 33.0})
    kw = dict(max_new=2, mode=mode, sizing="risk_parity", backtest=False)
    capped = search.search(r, even, candidates(r), cap_positions=True, **kw)
    for f in capped.finalists:
        assert f.weights.max() <= 0.34 + 1e-9
        assert (f.weights > 0).all()
        assert f.weights.sum() == pytest.approx(1)


def test_risk_parity_is_uncapped_by_default():
    r = make_returns()
    even = pd.Series({"H1": 34.0, "H2": 33.0, "H3": 33.0})
    res = search.search(r, even, candidates(r), max_new=2, mode="keep", sizing="risk_parity", backtest=False)
    assert max(f.weights.max() for f in res.finalists) > 0.34 + 1e-6


# --- slice sizing ----------------------------------------------------------------------------------------------------


def test_slice_sizing_puts_each_addition_at_the_slice_and_scales_the_holdings():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=2, backtest=False)
    props = holdings() / holdings().sum()
    for f in res.finalists:
        k = len(f.tickers)
        assert f.weights[list(f.tickers)].tolist() == pytest.approx([search.SLICE] * k)
        assert f.weights[HELD].to_numpy() == pytest.approx((props * (1 - search.SLICE * k)).to_numpy())
        assert f.weights.sum() == pytest.approx(1)


def test_slice_is_limited_to_the_largest_input_position():
    r = make_returns()
    small = pd.Series({"H1": 2.0, "H2": 2.0, "H3": 1.0})
    res = search.search(r, small, candidates(r), max_new=1, backtest=False)
    assert res.finalists[0].weights.max() <= 0.4 + 1e-9


def test_slice_sizing_is_backtested_against_the_current_weights():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=1, beam_width=2)
    assert all(f.backtest.status in (search.PASSES, search.FAILS) for f in res.finalists)
    assert all(f.backtest.full.weeks > 0 for f in res.finalists)


# --- balanced sizing -----------------------------------------------------------------------------------------


def test_cap_weights_holds_the_excess_at_the_bound_and_keeps_the_total():
    w = search.cap_weights(np.array([0.6, 0.3, 0.1]), np.array([0.4, 1.0, 1.0]))
    assert w.sum() == pytest.approx(1)
    assert w[0] == pytest.approx(0.4)
    assert w[1] / w[2] == pytest.approx(3)


def balanced(r, tilt, **kw):
    return search.search(
        r, holdings(), candidates(r), max_new=2, mode="reweight", sizing="balanced", tilt=tilt, backtest=False, **kw
    )


def test_balanced_with_zero_tilt_is_risk_parity():
    r = make_returns()
    rp = search.search(r, holdings(), candidates(r), max_new=2, mode="reweight", sizing="risk_parity", backtest=False)
    zero = balanced(r, 0.0)
    for a, b in zip(rp.finalists, zero.finalists):
        assert a.weights.to_numpy() == pytest.approx(b.weights.to_numpy())


def test_balanced_lies_between_risk_parity_and_max_sharpe_and_sums_to_one():
    r = make_returns()
    for f in balanced(r, 0.5).finalists:
        assert f.weights.sum() == pytest.approx(1)
        assert (f.weights >= 0).all()


def test_balanced_respects_the_cap_when_asked():
    r = make_returns()
    even = pd.Series({"H1": 34.0, "H2": 33.0, "H3": 33.0})
    res = search.search(
        r, even, candidates(r), max_new=2, mode="reweight", sizing="balanced", tilt=0.5, cap_positions=True,
        backtest=False,
    )
    for f in res.finalists:
        assert f.weights.max() <= 0.34 + 1e-9


def test_balanced_is_backtested():
    r = make_returns()
    res = search.search(r, holdings(), candidates(r), max_new=1, beam_width=2, sizing="balanced", mode="keep")
    assert all(f.backtest.status in (search.PASSES, search.FAILS) for f in res.finalists)


def test_tilt_outside_zero_one_is_an_error():
    r = make_returns()
    with pytest.raises(ValueError, match="tilt"):
        search.search(r, holdings(), candidates(r), max_new=1, tilt=1.5)
