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
    counts = {"download": 0, "optimiser": 0, "summarise": 0}

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
