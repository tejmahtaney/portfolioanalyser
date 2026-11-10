"""Drives the Add positions page with AppTest. Downloads are mocked and no network is used."""
import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from portfolio_risk import data, excel_io, search, ui_search, universe

TIMEOUT = 120
HELD = ["H1", "H2", "H3"]
CANDIDATES = ["C0", "C1", "C2", "IND"]
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def make_returns(n=600, seed=3):
    rng = np.random.default_rng(seed)
    factor = rng.normal(0.002, 0.02, n)
    names = HELD + [c for c in CANDIDATES if c != "IND"]
    loadings = rng.uniform(0.6, 1.4, len(names))
    frame = pd.DataFrame(
        factor[:, None] * loadings + rng.normal(0, 0.01, (n, len(names))),
        index=pd.date_range("2014-01-06", periods=n, freq="W-MON"),
        columns=names,
    )
    frame["IND"] = rng.normal(0.002, 0.02, n)
    return frame


def script():
    from portfolio_risk.ui_search import page

    page()


def two_pages_script():
    """Runs one page per script run, so the other page's widgets are not rendered, as when changing tabs."""
    import streamlit as st

    from portfolio_risk import ui, ui_search

    (ui.main if st.session_state.get("show") == "risk" else ui_search.page)()


def portfolio_script():
    from portfolio_risk.ui import main

    main()


def portfolio_file(weights: dict[str, float]) -> bytes:
    """A workbook as the Portfolio risk page exports it, weights in percent."""
    series = pd.Series(weights)
    portfolio = pd.DataFrame({"Name": [f"Name {t}" for t in series.index], "Weight": series / series.sum()})
    return excel_io.export_workbook(portfolio, {"Risk-free rate": "4.00%"})


@pytest.fixture
def env(monkeypatch):
    """Synthetic returns behind data.load_market_data, and a counter on search.search."""
    st.cache_data.clear()
    state = {"returns": make_returns(), "searches": 0}

    def fake_load(tickers, *args, **kwargs):
        keep = [c for c in state["returns"].columns if c in set(tickers) | {"VOO"}]
        return data.MarketData(returns=state["returns"][keep], currencies={c: "USD" for c in keep})

    names = list(make_returns().columns)
    frame = pd.DataFrame({"ticker": names, "name": [f"Name {c}" for c in names], "asset_class": "Index fund"})
    monkeypatch.setattr(data, "load_market_data", fake_load)
    monkeypatch.setattr(universe, "load_universe", lambda: (frame, None))
    original = search.search

    def counted(*args, **kwargs):
        state["searches"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(search, "search", counted)
    yield state
    st.cache_data.clear()


def launch(fn=script, upload=True):
    at = AppTest.from_function(fn, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    if upload:
        at.file_uploader[0].set_value(("portfolio.xlsx", portfolio_file({"H1": 50, "H2": 30, "H3": 20}), XLSX))
        at.run()
        assert not at.exception
    return at


def run_search(at, candidates=("C0", "C1", "C2", "IND"), max_new=2):
    at.multiselect[0].set_value(list(candidates))
    at.slider[0].set_value(max_new)
    at.run()
    next(b for b in at.button if b.label == "Run search").click()
    at.run()
    assert not at.exception
    return at


def trade_column(at):
    return list(at.dataframe[2].value["Trade"])


def test_page_runs_end_to_end_on_a_synthetic_portfolio(env):
    at = run_search(launch())
    assert not at.error
    assert env["searches"] == 1

    holdings = at.dataframe[0].value
    assert list(holdings.index) == HELD
    assert holdings["Weight (%)"].tolist() == [50.0, 30.0, 20.0]

    finalists = at.dataframe[1].value
    assert list(finalists.columns) == [
        "Added", "Δ Volatility", "Δ CVaR", "Correlation", "Diversification ratio", "Volatility", "Effective assets", "Backtest", "Sharpe (out of sample)",
        "Sharpe (baseline)", "Max drawdown", "Without Feb–Apr 2020",
    ]
    assert finalists["Δ Volatility"].str.startswith("-").iloc[0]
    assert set(finalists["Backtest"]) <= {search.PASSES, search.FAILS}

    trades = at.dataframe[2].value
    assert list(trades.columns) == ["Name", "Target", "Current", "Trade"]
    assert all(c.endswith("%") for c in trades["Target"])
    assert at.selectbox[0].value.startswith("1. ")
    assert any("Weights use the last 156 weeks" in c.value for c in at.caption)
    assert any("backtest covers 10 years" in c.value for c in at.caption)
    assert any(c.value.startswith('"fails" means') for c in at.caption)
    assert any("not financial advice" in c.value for c in at.caption)
    assert at.download_button[0].label == "Export target weights"


def test_run_button_is_the_only_trigger(env):
    at = launch()
    at.multiselect[0].set_value(["C0", "C1"])
    at.slider[0].set_value(2)
    at.run()
    assert env["searches"] == 0
    assert not at.dataframe[1:]  # no results yet
    next(b for b in at.button if b.label == "Run search").click()
    at.run()
    assert env["searches"] == 1
    at.radio[1].set_value("Risk parity").run()
    assert env["searches"] == 1
    assert any("Press Run search" in i.value for i in at.info)


def test_balanced_sizing_runs_and_the_share_slider_changes_the_result_key(env):
    at = launch()
    at.radio[1].set_value("Balanced (risk parity + max Sharpe)")
    at.run()
    assert not at.slider[1].disabled
    at = run_search(at)
    assert not at.error
    assert env["searches"] == 1
    at.slider[1].set_value(80).run()
    assert env["searches"] == 1
    assert any("Press Run search" in i.value for i in at.info)


def test_changing_a_band_threshold_does_not_rerun_the_search(env):
    at = run_search(launch())
    assert env["searches"] == 1
    default_trades = trade_column(at)
    assert "–" in default_trades  # the default band leaves some trades blank
    at.number_input(key="ap_band_abs").set_value(0.0)
    at.number_input(key="ap_band_rel").set_value(0.0)
    at.run()
    assert not at.exception
    assert env["searches"] == 1
    assert "–" not in trade_column(at)
    at.number_input(key="ap_band_abs").set_value(5.0)
    at.number_input(key="ap_band_rel").set_value(25.0)
    at.run()
    assert env["searches"] == 1
    assert trade_column(at) == default_trades


def test_budget_for_new_positions_defaults_to_15_and_is_part_of_the_result_key(env):
    at = launch()
    assert at.number_input(key="ap_budget").value == 15.0
    at.radio(key="ap_sizing").set_value("Risk parity")
    at = run_search(at)
    assert env["searches"] == 1
    targets = at.dataframe[2].value["Target"].str.rstrip("%").astype(float)
    assert targets.iloc[len(HELD):].sum() == pytest.approx(15.0, abs=0.05)
    at.number_input(key="ap_budget").set_value(30.0)
    at.run()
    assert any("Press Run search" in i.value for i in at.info)


def test_inputs_and_result_survive_a_visit_to_the_other_page(env):
    at = AppTest.from_function(two_pages_script, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    at.file_uploader[0].set_value(("portfolio.xlsx", portfolio_file({"H1": 50, "H2": 30, "H3": 20}), XLSX))
    at.run()
    run_search(at)
    assert env["searches"] == 1
    ratios = list(at.dataframe[1].value["Added"])

    at.session_state["show"] = "risk"
    at.run()
    assert not at.exception
    at.session_state["show"] = "search"
    at.run()
    assert not at.exception
    assert env["searches"] == 1
    assert list(at.dataframe[1].value["Added"]) == ratios
    assert at.slider[0].value == 2


def test_exported_target_weights_import_on_the_portfolio_page(env):
    at = run_search(launch())
    finalist = at.session_state[ui_search.RESULTS_KEY]
    stored = next(iter(finalist.values()))
    chosen = stored.result.finalists[0]
    names = pd.Series({t: f"Name {t}" for t in chosen.weights.index})
    book = ui_search.export_target_weights(chosen, names, "reweight")

    exported = excel_io.import_weights(pd.io.common.BytesIO(book))
    assert set(exported.index) == set(HELD) | set(chosen.tickers)

    page = AppTest.from_function(portfolio_script, default_timeout=TIMEOUT)
    page.run()
    page.sidebar.file_uploader[0].set_value(("target.xlsx", book, XLSX))
    page.run()
    assert not page.exception
    assert not page.sidebar.error
    assert sorted(page.session_state["selected"]) == sorted(chosen.weights.index)
    got = pd.Series(page.session_state["weights"])
    assert got.index.tolist() == exported.index.tolist()
    assert np.allclose(got / got.sum(), chosen.weights.reindex(got.index), atol=1e-4)


def test_every_finalist_without_backtest_names_the_short_holding(env):
    returns = env["returns"].copy()
    returns.loc[returns.index[:200], "H3"] = np.nan  # 400 weeks left
    env["returns"] = returns
    at = run_search(launch())
    assert not at.error
    assert set(at.dataframe[1].value["Backtest"]) == {search.NO_BACKTEST}
    assert any("H3 (400 weeks)" in w.value and "under 500" in w.value for w in at.warning)


def test_a_short_candidate_alone_gets_the_generic_reason(env):
    returns = env["returns"].copy()
    returns.loc[returns.index[:200], "C0"] = np.nan
    returns.loc[returns.index[:200], "C1"] = np.nan
    returns.loc[returns.index[:200], "C2"] = np.nan
    returns.loc[returns.index[:200], "IND"] = np.nan
    env["returns"] = returns
    at = run_search(launch())
    assert not any("current holdings" in w.value for w in at.warning)
    assert any("Common history is" in i.value for i in at.info)


def test_excluded_candidates_are_listed_in_a_collapsed_expander(env):
    returns = env["returns"].copy()
    returns.loc[returns.index[:500], "C2"] = np.nan  # fewer than 156 weeks
    env["returns"] = returns
    at = run_search(launch())
    expander = next(e for e in at.expander if e.label.startswith("Excluded"))
    assert not expander.proto.expanded
    assert "C2" in expander.dataframe[0].value.index


def test_trade_display_shows_dash_for_blank_trades():
    weights = pd.Series({"A": 0.5, "B": 0.5})
    trades = pd.DataFrame({"Target": weights, "Current": [0.49, 0.2], "Trade": [np.nan, 0.3]})
    finalist = search.Finalist(
        ("B",), search.Metrics(-0.01, -0.01, 0.2), 1.0, 1.0, 0.1, 2.0, weights, trades, search.Backtest(search.NO_BACKTEST)
    )
    out = ui_search.trade_display(finalist, 0.05, 0.25, pd.Series({"A": "Alpha"}))
    assert out.loc["A", "Trade"] == "–"
    assert out.loc["B", "Trade"] == "+30.00%"
    assert out.loc["A", "Name"] == "Alpha" and out.loc["B", "Name"] == "B"


def test_entrypoint_navigation_opens_on_the_portfolio_risk_page(env):
    at = AppTest.from_file("../streamlit_app.py", default_timeout=TIMEOUT)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Portfolio risk"
