"""Drives the real Streamlit script with AppTest. Downloads are mocked and no network is used."""
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from portfolio_risk import analytics, data, optimiser, ui, universe, weights as wt

TIMEOUT = 60


def script():
    from portfolio_risk.ui import main

    main()


@pytest.fixture
def calls(returns, monkeypatch):
    st.cache_data.clear()
    counts = {"download": 0, "optimiser": 0, "summarise": 0, "held": 0}

    def fake_load(tickers, *args, **kwargs):
        counts["download"] += 1
        keep = [c for c in returns.columns if c in set(tickers) | {"VOO"}]
        return data.MarketData(returns=returns[keep], currencies={c: "USD" for c in keep})

    frame = pd.DataFrame(
        {"ticker": list(returns.columns), "name": [f"Name {c}" for c in returns.columns], "asset_class": "Index fund"}
    )
    monkeypatch.setattr(data, "load_market_data", fake_load)
    monkeypatch.setattr(universe, "load_universe", lambda: (frame, None))
    for name in ["min_variance", "max_sharpe", "risk_parity", "efficient_frontier"]:
        original = getattr(optimiser, name)

        def counted(*args, _original=original, **kwargs):
            counts["optimiser"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(optimiser, name, counted)
    original_summarise = analytics.summarise

    def counted_summarise(*args, **kwargs):
        counts["summarise"] += 1
        return original_summarise(*args, **kwargs)

    monkeypatch.setattr(analytics, "summarise", counted_summarise)
    original_held = ui.build_held_estimates

    def counted_held(*args, **kwargs):
        counts["held"] += 1
        return original_held(*args, **kwargs)

    monkeypatch.setattr(ui, "build_held_estimates", counted_held)
    yield counts
    st.cache_data.clear()


def launch(selected):
    at = AppTest.from_function(script, default_timeout=TIMEOUT)
    at.session_state["selected"] = selected
    at.run()
    assert not at.exception
    return at


def editor_key(at):
    return next(k for k in at.session_state.keys() if k.startswith("editor-") and k.split("-")[1].isdigit())


def edit(at, values: dict[int, float]):
    at.session_state[editor_key(at)] = {
        "edited_rows": {row: {"Weight (%)": v} for row, v in values.items()}, "added_rows": [], "deleted_rows": [],
    }
    at.run()
    assert not at.exception


def click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_changing_a_weight_does_not_download(calls):
    at = launch(["A", "B", "C"])
    assert calls["download"] == 1
    for value in [10.0, 12.75, 0.001, 55.5, 33.33]:
        edit(at, {0: value, 1: 5.0, 2: 7.0})
    assert calls["download"] == 1


def test_changing_a_weight_does_not_rerun_the_optimiser(calls):
    at = launch(["A", "B", "C", "D", "E"])
    assert calls["optimiser"] == 0
    click(at, "Run optimisation")
    after_run = calls["optimiser"]
    assert after_run > 0
    assert not at.error
    for value in [10.0, 12.75, 0.001, 55.5]:
        edit(at, {0: value, 1: 20.0, 2: 30.0})
    assert calls["optimiser"] == after_run
    assert calls["download"] == 1


def test_optimiser_runs_again_only_when_asked_after_inputs_change(calls):
    at = launch(["A", "B", "C", "D", "E"])
    click(at, "Run optimisation")
    first = calls["optimiser"]
    at.radio[0].set_value("Sample").run()
    assert calls["optimiser"] == first
    click(at, "Run optimisation")
    assert calls["optimiser"] > first


def test_weight_edit_recomputes_weight_dependent_statistics_only(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: 1.0, 1: 1.0, 2: 1.0})
    before = calls["summarise"]
    edit(at, {0: 50.0, 1: 1.0, 2: 1.0})
    assert calls["summarise"] == before + 1


def test_changing_settings_downloads_nothing_new_but_changing_tickers_does(calls):
    at = launch(["A", "B", "C"])
    at.radio[0].set_value("Sample").run()
    assert calls["download"] == 1
    at.session_state["selected"] = ["A", "B", "C", "D"]
    at.run()
    assert calls["download"] == 2


def test_ticker_added_and_removed_keeps_typed_weights(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: 12.75, 1: 0.001, 2: 40.0})
    at.session_state["selected"] = ["A", "B", "C", "D"]
    at.run()
    assert at.session_state["weights"] == {"A": 12.75, "B": 0.001, "C": 40.0, "D": 0.0}
    at.session_state["selected"] = ["A", "C", "D"]
    at.run()
    assert at.session_state["weights"] == {"A": 12.75, "C": 40.0, "D": 0.0}
    at.session_state["selected"] = ["B", "A", "C", "D"]
    at.run()
    assert at.session_state["weights"]["A"] == 12.75 and at.session_state["weights"]["B"] == 0.0
    assert not at.exception


def test_summary_shows_typed_and_normalised_totals(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: 33.33, 1: 33.33, 2: 33.34})
    assert metric(at, "Typed total") == "100.00"
    assert metric(at, "Normalised total") == "100.00%"
    assert any("is 100." in c.value for c in at.caption)
    edit(at, {0: 50.0, 1: 25.0, 2: 0.5})
    assert metric(at, "Typed total") == "75.50"
    assert metric(at, "Normalised total") == "100.00%"
    assert any("not 100" in c.value for c in at.caption)


def test_zero_total_shows_message_and_skips_calculations(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: 0.0, 1: 0.0, 2: 0.0})
    assert any("typed total is 0" in w.value for w in at.warning)
    assert calls["summarise"] == 0
    assert metric(at, "Normalised total") == "n/a"
    assert any("positive weight" in i.value for i in at.info)


def test_negative_weight_shows_inline_error_and_skips_calculations(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: -5.0, 1: 10.0, 2: 10.0})
    assert any("must not be negative: A" in e.value for e in at.error)
    assert calls["summarise"] == 0


def test_blank_weight_is_treated_as_zero(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: None, 1: 10.0, 2: 30.0})
    assert at.session_state["weights"]["A"] == 0.0
    assert metric(at, "Typed total") == "40.00"


def test_equal_and_clear_buttons_write_session_state(calls):
    at = launch(["A", "B", "C"])
    click(at, "Equal weight")
    assert at.session_state["weights"] == {t: 100 / 3 for t in "ABC"}
    edit(at, {0: 70.0})
    assert at.session_state["weights"]["A"] == 70.0
    click(at, "Clear weights")
    assert at.session_state["weights"] == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert calls["download"] == 1


def test_decimal_weights_reach_state_at_full_precision(calls):
    at = launch(["A", "B"])
    edit(at, {0: 0.001, 1: 99.999})
    assert at.session_state["weights"] == {"A": 0.001, "B": 99.999}


def test_portfolio_snapshot_skips_zero_total_without_dividing(returns):
    est = ui.build_estimates(
        data.MarketData(returns=returns), ["A", "B", "C"], dict(cov_method="sample", return_method="raw", shrinkage=0.5), set()
    )
    assert ui.portfolio_snapshot(wt.parse(pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})), est, 0.01) is None
    snap = ui.portfolio_snapshot(wt.parse(pd.Series({"A": 0.001, "B": 99.999, "C": 0.0})), est, 0.01)
    assert snap.w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.isfinite(snap.summary["Volatility"])


PORTFOLIO_METRICS = ["Expected return", "Volatility", "Sharpe ratio", "Beta to VOO", "Max drawdown", "95% weekly VaR"]
METHODS = [(cov, ret) for cov in ("Ledoit-Wolf", "Sample") for ret in ("Shrunk", "Raw historical mean")]


def portfolio_metrics(at):
    return {label: metric(at, label) for label in PORTFOLIO_METRICS}


def choose_methods(at, cov, ret):
    at.radio[0].set_value(cov)
    at.radio[1].set_value(ret)
    at.run()
    assert not at.exception


def add_ticker(at, ticker):
    at.session_state["selected"] = [*at.session_state["selected"], ticker]
    at.run()
    assert not at.exception


@pytest.mark.parametrize("cov,ret", METHODS)
def test_zero_weight_ticker_with_full_history_leaves_portfolio_metrics_unchanged(calls, returns, cov, ret):
    at = launch(["A", "B", "C"])
    choose_methods(at, cov, ret)
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0})
    before = portfolio_metrics(at)
    add_ticker(at, "D")
    assert at.session_state["weights"]["D"] == 0.0
    assert portfolio_metrics(at) == before
    settings = dict(
        cov_method="ledoit_wolf" if cov == "Ledoit-Wolf" else "sample", return_method="shrunk" if ret == "Shrunk" else "raw", shrinkage=0.5
    )
    est = ui.build_estimates(data.MarketData(returns=returns), ["A", "B", "C"], settings, set())
    snap = ui.portfolio_snapshot(wt.parse(pd.Series({"A": 50.0, "B": 30.0, "C": 20.0})), est, 0.04)
    assert before["Expected return"] == ui.pct(snap.summary["Expected return"])
    assert before["Volatility"] == ui.pct(snap.summary["Volatility"])


@pytest.mark.parametrize("cov,ret", METHODS)
def test_zero_weight_ticker_with_shorter_history_leaves_portfolio_metrics_unchanged(calls, returns, cov, ret):
    returns.loc[returns.index[:200], "D"] = np.nan
    at = launch(["A", "B", "C"])
    choose_methods(at, cov, ret)
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0})
    before = portfolio_metrics(at)
    assert not any("Frontier, Recommendations and Backtest" in c.value for c in at.caption)
    add_ticker(at, "D")
    assert portfolio_metrics(at) == before
    assert any("**300 weeks** used" in m.value for m in at.markdown)
    assert any("Frontier, Recommendations and Backtest tabs use the shared window" in c.value for c in at.caption)


def test_portfolio_tab_states_the_weeks_and_range_used_for_held_assets(calls, returns):
    returns.loc[returns.index[:200], "D"] = np.nan
    at = launch(["A", "B", "C", "D"])
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0, 3: 0.0})
    first, last = returns.index[0], returns.index[-1]
    assert any(f"**300 weeks** used for the held assets, from {first:%d %b %Y} to {last:%d %b %Y}" in m.value for m in at.markdown)
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0, 3: 10.0})
    assert any("**100 weeks** used" in m.value for m in at.markdown)
    assert not any("shared window" in c.value for c in at.caption)


def recommendations_script(frame):
    from portfolio_risk import data, ui, weights as wt

    settings = dict(cov_method="ledoit_wolf", return_method="shrunk", shrinkage=0.5)
    tickers = ["A", "B", "C", "D", "E"]
    est = ui.build_estimates(data.MarketData(returns=frame), tickers, settings, set())
    p = dict(settings, risk_free=0.04, cap=0.5)
    bundle = ui.compute_bundle(est, p, ui.optimisation_key(est, p))
    parsed = wt.parse({"A": 50.0, "B": 30.0, "C": 20.0, "D": 0.0, "E": 0.0})
    held = ui.build_held_estimates(data.MarketData(returns=frame), ["A", "B", "C"], settings)
    ui.recommendations_tab(bundle, ui.portfolio_snapshot(parsed, est, 0.04, held))


def test_recommendations_current_row_stays_on_shared_estimates_and_says_so(returns):
    at = AppTest.from_function(recommendations_script, args=(returns,), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Current row" in c.value and "shared estimates" in c.value for c in at.caption)
    shared = analytics.annual_covariance(returns[["A", "B", "C", "D", "E"]].dropna(), "ledoit_wolf")
    w = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2, "D": 0.0, "E": 0.0})
    expected = ui.pct(analytics.portfolio_volatility(w, shared))
    stats = next(df.value for df in at.dataframe if "Volatility" in df.value.columns)
    assert ui.pct(stats.loc["Current", "Volatility"]) == expected


def test_weight_edit_that_keeps_the_held_set_does_not_re_estimate(calls):
    at = launch(["A", "B", "C", "D", "E"])
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0, 3: 0.0, 4: 0.0})
    assert calls["held"] == 1
    for value in [10.0, 12.75, 0.001, 55.5]:
        edit(at, {0: value, 1: 30.0, 2: 20.0, 3: 0.0, 4: 0.0})
    assert calls["held"] == 1
    assert calls["download"] == 1
    edit(at, {0: 10.0, 1: 30.0, 2: 20.0, 3: 5.0, 4: 0.0})
    assert calls["held"] == 2
    edit(at, {0: 10.0, 1: 30.0, 2: 20.0, 3: 0.0, 4: 0.0})
    assert calls["held"] == 2
    assert calls["download"] == 1


def test_all_selected_assets_held_reuses_the_shared_estimates(calls):
    at = launch(["A", "B", "C"])
    edit(at, {0: 50.0, 1: 30.0, 2: 20.0})
    edit(at, {0: 40.0, 1: 30.0, 2: 30.0})
    assert calls["held"] == 0


@pytest.mark.parametrize("cov,ret", METHODS)
def test_single_held_asset_produces_finite_metrics(calls, returns, cov, ret):
    at = launch(["A", "B", "C"])
    choose_methods(at, cov, ret)
    edit(at, {0: 0.0, 1: 100.0, 2: 0.0})
    assert not any("at least two" in i.value for i in at.info)
    values = portfolio_metrics(at)
    for label, text in values.items():
        assert np.isfinite(float(text.rstrip("%"))), (label, text)
    window = returns.dropna(subset=["B", "VOO"])
    assert values["Volatility"] == ui.pct(analytics.annual_covariance(window[["B"]], "ledoit_wolf" if cov == "Ledoit-Wolf" else "sample").loc["B", "B"] ** 0.5)


def test_single_held_asset_has_full_risk_contribution_and_dashes_for_the_rest(returns):
    est = ui.build_estimates(
        data.MarketData(returns=returns), ["A", "B", "C"], dict(cov_method="ledoit_wolf", return_method="shrunk", shrinkage=0.5), set()
    )
    held = ui.build_held_estimates(
        data.MarketData(returns=returns), ["B"], dict(cov_method="ledoit_wolf", return_method="shrunk", shrinkage=0.5)
    )
    snap = ui.portfolio_snapshot(wt.parse(pd.Series({"A": 0.0, "B": 7.0, "C": 0.0})), est, 0.04, held)
    assert snap.table.loc["B", "Risk contribution"] == pytest.approx(1.0)
    assert snap.table.loc[["A", "C"], ["Risk contribution", "Expected return", "Volatility", "Beta to VOO"]].isna().all().all()
    assert snap.table["Weight (%)"].to_dict() == {"A": 0.0, "B": 100.0, "C": 0.0}
    assert snap.w.to_dict() == {"A": 0.0, "B": 1.0, "C": 0.0}
    html = ui.format_stats(snap.table, na_rep="–").to_html()
    assert html.count("–") == 8
    assert snap.summary["Volatility"] == pytest.approx(held.cov.loc["B", "B"] ** 0.5)


def test_sharpe_ratio_shows_na_when_it_is_nan(calls, monkeypatch):
    monkeypatch.setattr(analytics, "sharpe_ratio", lambda *args: float("nan"))
    at = launch(["A", "B", "C"])
    edit(at, {0: 10.0, 1: 20.0, 2: 30.0})
    assert metric(at, "Sharpe ratio") == "n/a"


def test_optimiser_still_stops_below_two_selected_assets(calls):
    at = launch(["A"])
    assert any("at least two" in i.value for i in at.info)


def test_excel_current_rows_use_held_estimates_and_keep_zero_weight_tickers(returns):
    from io import BytesIO

    settings = dict(cov_method="ledoit_wolf", return_method="shrunk", shrinkage=0.5)
    md = data.MarketData(returns=returns)
    est = ui.build_estimates(md, ["A", "B", "C", "D"], settings, set())
    held = ui.build_held_estimates(md, ["A", "B", "C"], settings)
    parsed = wt.parse({"A": 50.0, "B": 30.0, "C": 20.0, "D": 0.0})
    snap = ui.portfolio_snapshot(parsed, est, 0.04, held)
    p = dict(settings, risk_free=0.04, cap=0.5)
    bundle = ui.compute_bundle(est, p, ui.optimisation_key(est, p))
    names = pd.Series({t: f"Name {t}" for t in "ABCD"})
    book = ui.export_workbook_bytes(snap, names, est, p, bundle)
    portfolio = pd.read_excel(BytesIO(book), sheet_name="Portfolio").set_index("Ticker")
    assert portfolio["Weight (%)"].tolist() == pytest.approx([50, 30, 20, 0])
    assert portfolio.loc["D", ["Risk contribution (%)", "Expected return (%)", "Volatility (%)"]].isna().all()
    stats = pd.read_excel(BytesIO(book), sheet_name="Recommended statistics").set_index("Portfolio")
    assert stats.loc["Current", "Volatility (%)"] == pytest.approx(round(snap.summary["Volatility"] * 100, 4))
    assert stats.loc["Current", "Expected return (%)"] == pytest.approx(round(snap.summary["Expected return"] * 100, 4))
