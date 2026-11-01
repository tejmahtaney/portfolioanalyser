from __future__ import annotations

import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import analytics, backtest, charts, data, excel_io, optimiser, universe
from . import weights as wt
from .config import (
    BENCHMARK,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE,
    DEFAULT_SHRINKAGE,
    MIN_WEEKS_WARNING,
)

DISCLAIMER = "Results are based on historical data and are not financial advice. Past performance does not predict future returns."
WEIGHT_COLUMN = "Weight (%)"
OPTIMISER_TABS = ("Frontier", "Recommendations", "Backtest")
MAX_REMEMBERED_RUNS = 6


@dataclass
class Estimates:
    """Everything that depends only on the selected tickers and the settings, never on typed weights."""

    tickers: list[str]
    md: data.MarketData
    window: pd.DataFrame
    returns: pd.DataFrame
    cov: pd.DataFrame | None = None
    mu: pd.Series | None = None
    betas: pd.Series | None = None
    flags: pd.DataFrame = field(default_factory=pd.DataFrame)
    pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    info: str | None = None


@dataclass
class Snapshot:
    """Statistics that depend on the weights. Cheap matrix operations, recomputed on every edit.

    `w` spans every selected ticker and feeds the optimiser tabs. `summary`, `var95`, `beta` and `table` come from `est`,
    the estimates over the held tickers only. `shared` is the estimates over every selected ticker.
    """

    w: pd.Series
    summary: dict[str, float]
    var95: float
    beta: float | None
    table: pd.DataFrame
    est: Estimates
    shared: Estimates


@dataclass
class OptBundle:
    """Optimiser, frontier and backtest output. Independent of the typed weights."""

    key: tuple
    est: Estimates
    risk_free: float
    cap: float | None
    results: dict[str, optimiser.Result] | None = None
    frontier: pd.DataFrame | None = None
    frontier_fig: go.Figure | None = None
    oos: pd.DataFrame | None = None
    errors: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


@st.cache_data(ttl=3600, show_spinner=False)
def cached_universe():
    return universe.load_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...]):
    return data.load_market_data(list(tickers))


def short_history_warning(weeks: int) -> str:
    return f"Only {weeks} weeks of common history remain, fewer than {MIN_WEEKS_WARNING}. Estimates will be unreliable."


def build_estimates(
    md: data.MarketData, selected: list[str], settings: dict, futures: set[str], min_assets: int = 2
) -> Estimates:
    tickers = [t for t in selected if t in md.returns.columns]
    empty = pd.DataFrame()
    if not tickers:
        return Estimates(tickers, md, empty, empty, error="None of the selected assets could be loaded.")
    bench_ok = BENCHMARK in md.returns.columns
    window = data.common_window(md.returns, tickers + ([BENCHMARK] if bench_ok else []))
    if window.empty:
        return Estimates(tickers, md, window, empty, error="The selected assets share no common history.")
    est = Estimates(tickers, md, window, window[tickers])
    if not bench_ok:
        est.warnings.append(f"{BENCHMARK} could not be loaded, so betas are unavailable.")
    if len(window) < MIN_WEEKS_WARNING:
        est.warnings.append(short_history_warning(len(window)))
    if len(tickers) < min_assets:
        est.info = "Select at least two assets to estimate a covariance matrix."
        return est
    est.cov = analytics.annual_covariance(est.returns, settings["cov_method"])
    est.mu = analytics.expected_returns(est.returns, settings["return_method"], settings["shrinkage"])
    est.betas = analytics.betas(est.returns, window[BENCHMARK]) if bench_ok else None
    est.flags = data.flag_roll_jumps(est.returns, [t for t in tickers if t in futures])
    est.pairs = data.high_correlations(est.returns)
    if not est.pairs.empty:
        est.warnings.append(
            "Highly correlated pairs (above 0.98): " + ", ".join(f"{a}/{b}" for a, b in zip(est.pairs.asset_a, est.pairs.asset_b))
        )
    return est


@st.cache_data(ttl=3600, show_spinner=False)
def cached_estimates(
    tickers: tuple[str, ...], cov_method: str, return_method: str, shrinkage: float, _futures: frozenset = frozenset()
) -> Estimates:
    """Keyed on the sorted ticker tuple and the settings; typed weights never reach this function."""
    settings = dict(cov_method=cov_method, return_method=return_method, shrinkage=shrinkage)
    return build_estimates(cached_market_data(tickers), list(tickers), settings, set(_futures))


def held_tickers(parsed: wt.Parsed, tickers: list[str]) -> tuple[str, ...]:
    """Sorted tickers whose normalised weight is above 0."""
    if parsed.normalised is None:
        return ()
    return tuple(sorted(t for t in tickers if parsed.normalised.get(t, 0.0) > 0))


def build_held_estimates(md: data.MarketData, held: list[str], settings: dict) -> Estimates:
    """Window, covariance, expected returns and betas over the held tickers alone. One asset is enough."""
    return build_estimates(md, held, settings, set(), min_assets=1)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_held_estimates(
    held: tuple[str, ...], cov_method: str, return_method: str, shrinkage: float, _selected: tuple[str, ...]
) -> Estimates:
    """Keyed on the sorted held tuple and the settings. `_selected` only locates the cached market data of the full selection."""
    settings = dict(cov_method=cov_method, return_method=return_method, shrinkage=shrinkage)
    return build_held_estimates(cached_market_data(_selected), list(held), settings)


def held_estimates(parsed: wt.Parsed, est: Estimates, p: dict) -> Estimates | None:
    """Estimates over the held tickers. When every selected ticker is held these are the shared estimates."""
    held = held_tickers(parsed, est.tickers)
    if not held:
        return None
    if held == tuple(sorted(est.tickers)):
        return est
    return cached_held_estimates(held, p["cov_method"], p["return_method"], p["shrinkage"], tuple(sorted(p["selected"])))


def portfolio_snapshot(parsed: wt.Parsed, est: Estimates, risk_free: float, held_est: Estimates | None = None) -> Snapshot | None:
    """Returns None when there is nothing to calculate: a zero total, negative weights or no usable held estimates.

    The statistics use `held_est`, which defaults to `est`; tickers outside it get empty estimate cells.
    """
    if parsed.normalised is None:
        return None
    held_est = est if held_est is None else held_est
    if held_est.error or held_est.cov is None:
        return None
    w = parsed.normalised.reindex(est.tickers)
    wh = w.reindex(held_est.tickers)
    summary = analytics.summarise(wh, held_est.mu, held_est.cov, held_est.returns, risk_free)
    port = analytics.portfolio_returns(held_est.returns, wh)
    table = pd.DataFrame(
        {
            "Weight (%)": wt.largest_remainder(wh * 100),
            "Risk contribution": analytics.risk_contributions(wh, held_est.cov),
            "Expected return": held_est.mu[wh.index],
            "Volatility": pd.Series({t: held_est.cov.loc[t, t] ** 0.5 for t in wh.index}),
        }
    )
    beta = None
    if held_est.betas is not None:
        table["Beta to VOO"] = held_est.betas[wh.index]
        beta = float(wh @ held_est.betas[wh.index])
    table = table.reindex(est.tickers)
    table["Weight (%)"] = table["Weight (%)"].fillna(0.0)
    return Snapshot(w, summary, analytics.historical_var(port), beta, table, held_est, est)


def optimise_all(mu, cov, rf, cap, workspace=None) -> dict[str, optimiser.Result]:
    return {
        "Minimum variance": optimiser.min_variance(cov, cap, workspace),
        "Maximum Sharpe": optimiser.max_sharpe(mu, cov, rf, cap, workspace),
        "Risk parity": optimiser.risk_parity(cov, cap),
    }


def optimisation_key(est: Estimates, p: dict) -> tuple:
    return (tuple(est.tickers), p["cov_method"], p["return_method"], p["shrinkage"], p["risk_free"], p["cap"])


def compute_bundle(est: Estimates, p: dict, key: tuple) -> OptBundle:
    started = time.perf_counter()
    bundle = OptBundle(key, est, p["risk_free"], p["cap"])
    try:
        optimiser.check_feasible(len(est.tickers), p["cap"])
    except optimiser.OptimisationError as exc:
        bundle.errors["all"] = str(exc)
        return bundle
    workspace = optimiser.Workspace(len(est.tickers))
    try:
        bundle.results = optimise_all(est.mu, est.cov, p["risk_free"], p["cap"], workspace)
    except (optimiser.OptimisationError, ValueError) as exc:
        bundle.errors["optimiser"] = str(exc)
    try:
        bundle.frontier = optimiser.efficient_frontier(est.mu, est.cov, p["cap"], workspace=workspace)
        marks = {
            n: (analytics.portfolio_volatility(r.weights, est.cov), float(r.weights @ est.mu[r.weights.index]))
            for n, r in (bundle.results or {}).items()
        }
        assets = pd.DataFrame({"return": est.mu, "volatility": pd.Series({t: est.cov.loc[t, t] ** 0.5 for t in est.cov.index})})
        bundle.frontier_fig = charts.frontier_figure(bundle.frontier, assets, marks)
    except (optimiser.OptimisationError, ValueError) as exc:
        bundle.errors["frontier"] = str(exc)
    try:
        bundle.oos = backtest.run_backtest(
            est.returns, None, p["risk_free"], p["cap"], p["cov_method"], p["return_method"], p["shrinkage"]
        )
    except (optimiser.OptimisationError, ValueError) as exc:
        bundle.errors["backtest"] = str(exc)
    bundle.seconds = time.perf_counter() - started
    return bundle


def remember(state: MutableMapping, bundle: OptBundle) -> None:
    runs = state.setdefault("opt_runs", {})
    runs.pop(bundle.key, None)
    runs[bundle.key] = bundle
    while len(runs) > MAX_REMEMBERED_RUNS:
        runs.pop(next(iter(runs)))
    state["opt_last"] = bundle.key


def recall(state: MutableMapping, key: tuple) -> OptBundle | None:
    return state.get("opt_runs", {}).get(key)


def latest(state: MutableMapping) -> OptBundle | None:
    key = state.get("opt_last")
    return None if key is None else recall(state, key)


def bump_editor() -> None:
    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1


def apply_import(names: pd.Series) -> None:
    upload = st.sidebar.file_uploader("Import from Excel", type="xlsx", help="Use a workbook exported from this app.")
    if upload is None or upload.file_id == st.session_state.get("imported_id"):
        return
    st.session_state["imported_id"] = upload.file_id
    try:
        weights = excel_io.import_weights(upload)
    except Exception as exc:
        st.sidebar.error(f"Import failed: {exc}")
        return
    unknown = [t for t in weights.index if t not in names.index]
    weights = weights.drop(unknown)
    if unknown:
        st.sidebar.warning("Not in the universe, skipped: " + ", ".join(unknown))
    if weights.empty:
        st.sidebar.error("Import failed: none of the tickers are in the universe.")
        return
    st.session_state["selected"] = list(weights.index)
    st.session_state["weights"] = {t: float(v) for t, v in weights.items()}
    bump_editor()
    st.sidebar.success(f"Imported {len(weights)} assets.")


def sidebar(uni: pd.DataFrame) -> dict:
    names = uni.set_index("ticker")["name"]
    st.sidebar.header("Portfolio")
    apply_import(names)
    selected = st.sidebar.multiselect(
        "Assets", options=list(names.index), format_func=lambda t: f"{t} — {names[t]}",
        placeholder="Search by ticker or name", key="selected",
    )
    st.sidebar.header("Assumptions")
    risk_free = st.sidebar.number_input("Risk-free rate (%)", value=DEFAULT_RISK_FREE * 100, step=0.25, format="%.2f") / 100
    cov_label = st.sidebar.radio("Covariance method", ["Ledoit-Wolf", "Sample"])
    ret_label = st.sidebar.radio("Expected return method", ["Shrunk", "Raw historical mean"])
    shrinkage = DEFAULT_SHRINKAGE
    if ret_label == "Shrunk":
        shrinkage = st.sidebar.slider("Shrinkage intensity", 0.0, 1.0, DEFAULT_SHRINKAGE, 0.05)
    else:
        st.sidebar.warning(
            "Raw historical means are noisy estimates. Optimisers amplify that noise and tend to concentrate in recent winners."
        )
    use_cap = st.sidebar.checkbox("Limit maximum weight per asset", value=True)
    cap = None
    if use_cap:
        cap = st.sidebar.number_input(
            "Max weight per asset (%)", min_value=1.0, max_value=100.0, value=DEFAULT_MAX_WEIGHT * 100, step=1.0
        ) / 100
    return dict(
        selected=selected, risk_free=risk_free, cap=cap, shrinkage=shrinkage,
        cov_method="ledoit_wolf" if cov_label == "Ledoit-Wolf" else "sample",
        return_method="shrunk" if ret_label == "Shrunk" else "raw",
    )


def set_weights(values: dict[str, float]) -> None:
    st.session_state["weights"] = {**st.session_state.get("weights", {}), **values}
    bump_editor()


def weight_table(shown: list[str], names: pd.Series) -> pd.Series:
    """Editor over the weight dict. The frame is rebuilt from the dict only when the rows or the version change."""
    state = st.session_state
    stored = state["weights"]
    signature = (state.get("editor_version", 0), tuple(shown))
    if state.get("editor_signature") != signature:
        state["editor_signature"] = signature
        state["editor_counter"] = state.get("editor_counter", 0) + 1
        state["editor_frame"] = pd.DataFrame(
            {
                "Ticker": shown,
                "Name": [names[t] for t in shown],
                WEIGHT_COLUMN: pd.Series([stored.get(t, 0.0) for t in shown], dtype="float64"),
            }
        )
    edited = st.data_editor(
        state["editor_frame"], hide_index=True, width="stretch", key=f"editor-{state['editor_counter']}",
        disabled=["Ticker", "Name"],
        column_config={WEIGHT_COLUMN: st.column_config.NumberColumn(step=0.01, format="%.2f", min_value=0.0)},
    )
    raw = edited.set_index("Ticker")[WEIGHT_COLUMN]
    stored.update(wt.parse(raw).typed.to_dict())
    return raw


def show_weight_summary(slot, parsed: wt.Parsed) -> None:
    with slot:
        cols = st.columns([1, 1, 2])
        cols[0].metric("Typed total", f"{parsed.total:,.2f}")
        cols[1].metric("Normalised total", "100.00%" if parsed.normalised is not None else "n/a")
        if abs(parsed.total - 100) <= 1e-9:
            note = "The typed total is 100."
        else:
            note = f"The typed total is {parsed.total:,.6g}, not 100. Weights are divided by their sum before any calculation."
        cols[2].caption(note)
        if parsed.negatives:
            st.error("Weights must not be negative: " + ", ".join(parsed.negatives) + ". Calculations are skipped.")
        elif parsed.is_zero:
            st.warning("The typed total is 0. Enter at least one positive weight; weight-dependent calculations are skipped.")


def pct(x: float) -> str:
    return f"{x:.2%}"


def format_stats(table: pd.DataFrame, na_rep: str | None = None) -> pd.DataFrame:
    return table.style.format(
        {c: ("{:.2f}" if "Sharpe" in c or "Beta" in c or "(%)" in c else "{:.2%}") for c in table.columns}, na_rep=na_rep
    )


def portfolio_tab(snap: Snapshot | None, held_error: str | None = None):
    if held_error:
        st.error(f"The held assets cannot be estimated. {held_error}")
        return
    if snap is None:
        st.info("Enter at least one positive weight to see portfolio statistics.")
        return
    held, shared = snap.est, snap.shared
    window = held.window
    st.write(f"**{len(window)} weeks** used for the held assets, from {window.index[0]:%d %b %Y} to {window.index[-1]:%d %b %Y}.")
    differs = not window.index.equals(shared.window.index)
    if differs:
        st.caption("The Frontier, Recommendations and Backtest tabs use the shared window of all selected assets.")
        if len(window) < MIN_WEEKS_WARNING:
            st.warning(short_history_warning(len(window)))
    sharpe = snap.summary["Sharpe ratio"]
    cols = st.columns(6)
    cols[0].metric("Expected return", pct(snap.summary["Expected return"]))
    cols[1].metric("Volatility", pct(snap.summary["Volatility"]))
    cols[2].metric("Sharpe ratio", "n/a" if pd.isna(sharpe) else f"{sharpe:.2f}")
    cols[3].metric("Beta to VOO", "n/a" if snap.beta is None else f"{snap.beta:.2f}")
    cols[4].metric("Max drawdown", pct(snap.summary["Max drawdown"]))
    cols[5].metric("95% weekly VaR", pct(snap.var95), help="Loss exceeded in 5% of weeks.")
    st.dataframe(format_stats(snap.table, na_rep="–"), width="stretch")


def frontier_tab(bundle: OptBundle, snap: Snapshot | None):
    if bundle.frontier_fig is None:
        return
    fig = go.Figure(bundle.frontier_fig)
    if snap is not None:
        est = bundle.est
        charts.add_portfolio_point(
            fig, "Your portfolio", analytics.portfolio_volatility(snap.w, est.cov), float(snap.w @ est.mu[snap.w.index])
        )
    st.plotly_chart(fig, width="stretch")
    if bundle.cap is not None:
        st.caption(
            "With a weight cap, the frontier ends at the highest return reachable under that cap, "
            "which can be below the best single asset's return."
        )


def recommendations_tab(bundle: OptBundle, snap: Snapshot | None):
    results, est = bundle.results, bundle.est
    for r in results.values():
        for message in r.warnings:
            st.warning(message)
    weights = {n: r.weights for n, r in results.items()}
    if snap is not None:
        weights = {"Current": snap.w, **weights}
    table, dropped = wt.display_table(weights, exempt=["Current"])
    table.loc["Total"] = table.sum()
    st.subheader("Weights (%)")
    st.dataframe(table.astype("object").map(lambda v: "–" if pd.isna(v) else f"{v:.2f}").style.set_properties(**{"text-align": "right"}), width="stretch")
    removed = len(est.tickers) - (len(table) - 1)
    st.caption(
        f"Positions below 1e-6 are left out of the optimiser columns ("
        + ", ".join(f"{n}: {dropped[n]}" for n in results)
        + f"), and {removed} asset row{'s' if removed != 1 else ''} removed. "
        "Shown to 2 decimal places, rounded so that each column sums to 100.00."
    )
    st.subheader("Portfolio statistics")
    if snap is not None:
        st.caption(
            "The Current row and the Your portfolio point on the frontier use the shared estimates of all selected assets, "
            "so they can differ from the Portfolio tab when some selected assets have zero weight."
        )
    exact = pd.DataFrame(weights)
    summary = pd.DataFrame({n: analytics.summarise(exact[n], est.mu, est.cov, est.returns, bundle.risk_free) for n in exact}).T
    st.dataframe(format_stats(summary), width="stretch")


def backtest_tab(bundle: OptBundle, snap: Snapshot | None):
    st.caption(
        "Walk-forward: fit on 156 weeks, hold the fitted weights for the next 13, roll forward. "
        "Your weights and equal weight stay fixed. The backtest ignores transaction costs and taxes."
    )
    oos = bundle.oos
    if snap is not None:
        oos = backtest.add_user_weights(oos, bundle.est.returns, snap.w)
    st.write(f"Out-of-sample period: {oos.index[0]:%d %b %Y} to {oos.index[-1]:%d %b %Y} ({len(oos)} weeks).")
    st.dataframe(format_stats(backtest.performance_table(oos, bundle.risk_free)), width="stretch")
    st.plotly_chart(charts.cumulative_figure(oos), width="stretch")


def optimiser_tab(name: str, bundle: OptBundle | None, current: bool, snap: Snapshot | None):
    """Renders one optimiser tab from a stored bundle; a bundle for other inputs is shown without weight overlays."""
    if bundle is None:
        st.info("Press Run optimisation to compute this tab.")
        return
    if not current:
        st.warning("These results are for an earlier selection or settings. Press Run optimisation to update them.")
        snap = None
    if "all" in bundle.errors:
        st.error(f"Optimisation is disabled. {bundle.errors['all']}")
        return
    part = {"Frontier": "frontier", "Recommendations": "optimiser", "Backtest": "backtest"}[name]
    if part in bundle.errors:
        st.error(bundle.errors[part])
        return
    {"Frontier": frontier_tab, "Recommendations": recommendations_tab, "Backtest": backtest_tab}[name](bundle, snap)


def quality_tab(md, tickers, window, futures_flags, corr_pairs, uni):
    st.write(f"**{len(window)} weeks** used, from {window.index[0]:%d %b %Y} to {window.index[-1]:%d %b %Y}.")
    if md.failures:
        st.error("Download failed for: " + "; ".join(f"{t} ({e})" for t, e in md.failures.items()))
    if md.stale:
        st.warning("Using stale cached prices for: " + ", ".join(md.stale))
    currencies = pd.Series({t: md.currencies[t] for t in tickers if t in md.currencies}, name="Currency")
    non_usd = currencies[currencies != "USD"]
    if not non_usd.empty:
        st.info("Converted to USD at weekly rates: " + ", ".join(f"{t} ({c})" for t, c in non_usd.items()))
    st.subheader("Futures roll jumps")
    st.caption("Futures returns exclude roll yield and collateral interest.")
    if futures_flags.empty:
        st.write("No weekly futures return beyond 4 standard deviations.")
    else:
        st.dataframe(
            futures_flags.style.format({"return": "{:.2%}", "z_score": "{:.1f}", "week": lambda d: f"{d:%d %b %Y}"}),
            width="stretch", hide_index=True,
        )
    st.subheader("Correlation")
    if corr_pairs.empty:
        st.write("No pair of selected assets has a correlation above 0.98.")
    else:
        st.dataframe(corr_pairs.style.format({"correlation": "{:.3f}"}), width="stretch", hide_index=True)


def export_workbook_bytes(snap: Snapshot, names, est: Estimates, p: dict, bundle: OptBundle | None) -> bytes:
    """The Current rows use the held-asset estimates; the optimiser columns use the shared estimates `est`."""
    w, held = snap.w, snap.est
    wh = w.reindex(held.tickers)
    portfolio = pd.DataFrame(
        {
            "Name": [names[t] for t in w.index],
            "Weight": w,
            "Risk contribution": analytics.risk_contributions(wh, held.cov).reindex(w.index),
            "Expected return": held.mu[wh.index].reindex(w.index),
            "Volatility": pd.Series({t: held.cov.loc[t, t] ** 0.5 for t in wh.index}).reindex(w.index),
        }
    )
    if held.betas is not None:
        portfolio["Beta to VOO"] = held.betas[wh.index].reindex(w.index)
    weights = statistics = None
    if bundle is not None and bundle.results is not None:
        weights = pd.DataFrame({"Current": w, **{n: r.weights for n, r in bundle.results.items()}})
        rows = {"Current": analytics.summarise(wh, held.mu, held.cov, held.returns, p["risk_free"])}
        rows.update(
            {n: analytics.summarise(weights[n], est.mu, est.cov, est.window[w.index], p["risk_free"]) for n in weights.columns[1:]}
        )
        statistics = pd.DataFrame(rows).T
    settings = {
        "Risk-free rate": f"{p['risk_free']:.2%}",
        "Covariance method": p["cov_method"],
        "Expected return method": p["return_method"],
        "Shrinkage intensity": str(p["shrinkage"]),
        "Max weight per asset": "off" if p["cap"] is None else f"{p['cap']:.0%}",
        "Weeks used": str(len(est.window)),
        "Window": f"{est.window.index[0]:%Y-%m-%d} to {est.window.index[-1]:%Y-%m-%d}",
    }
    if not held.window.index.equals(est.window.index):
        settings["Weeks used (held assets)"] = str(len(held.window))
        settings["Window (held assets)"] = f"{held.window.index[0]:%Y-%m-%d} to {held.window.index[-1]:%Y-%m-%d}"
    return excel_io.export_workbook(portfolio, settings, weights, statistics)


@st.fragment
def workspace(est: Estimates, p: dict, uni: pd.DataFrame):
    """Weight table, summary and tabs. A weight edit reruns only this fragment, using cached estimates and optimiser output."""
    state = st.session_state
    names = uni.set_index("ticker")["name"]
    shown = [t for t in p["selected"] if t in est.tickers]
    state["weights"] = wt.sync(state.get("weights", {}), p["selected"])

    summary_slot = st.container()
    left, right, _ = st.columns([1, 1, 4])
    left.button("Equal weight", on_click=set_weights, args=(wt.equal(shown),), width="stretch")
    right.button("Clear weights", on_click=set_weights, args=({t: 0.0 for t in shown},), width="stretch")
    parsed = wt.parse(weight_table(shown, names))
    show_weight_summary(summary_slot, parsed)
    held_est = held_estimates(parsed, est, p)
    snap = portfolio_snapshot(parsed, est, p["risk_free"], held_est)

    key = optimisation_key(est, p)
    bundle = recall(state, key)
    controls = st.container()
    run = controls.button("Run optimisation", help="The optimiser, frontier and backtest do not depend on the typed weights.")
    if snap is not None:
        controls.download_button(
            "Export to Excel",
            lambda: export_workbook_bytes(snap, names, est, p, bundle),
            file_name="portfolio-risk.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    progress = st.empty()
    tabs = st.tabs(["Portfolio", *OPTIMISER_TABS, "Data quality"], key="main-tabs", on_change="rerun")

    # Compute before filling the tabs: the previous result stays on screen until the new elements replace it.
    if run or (latest(state) is None and any(tab.open is True for tab in tabs[1:4])):
        with progress, st.spinner("Optimising, building the frontier and running the backtest…"):
            bundle = compute_bundle(est, p, key)
            remember(state, bundle)
    shown_bundle = bundle or latest(state)

    with tabs[0]:
        if tabs[0].open is not False:
            st.caption(DISCLAIMER)
            portfolio_tab(snap, held_est.error if held_est is not None else None)
    for name, tab in zip(OPTIMISER_TABS, tabs[1:4]):
        with tab:
            if tab.open is not False:
                st.caption(DISCLAIMER)
                optimiser_tab(name, shown_bundle, bundle is not None, snap)
    with tabs[4]:
        if tabs[4].open is not False:
            st.caption(DISCLAIMER)
            quality_tab(est.md, est.tickers, est.window, est.flags, est.pairs, uni)


def load_estimates(p: dict, uni: pd.DataFrame) -> Estimates:
    tickers = tuple(sorted(p["selected"]))
    futures = frozenset(uni.loc[uni["asset_class"] == "Commodity future", "ticker"]) & set(tickers)
    with st.status("Loading prices…", expanded=False) as status:
        est = cached_estimates(tickers, p["cov_method"], p["return_method"], p["shrinkage"], futures)
        status.update(label=f"Prices ready for {len(est.tickers)} of {len(tickers)} assets", state="complete")
        for line in est.md.log:
            st.write(line)
    return est


def main():
    st.set_page_config(page_title="Portfolio risk", layout="wide")
    st.title("Portfolio risk")
    st.caption(DISCLAIMER)

    uni, uni_warning = cached_universe()
    if uni_warning:
        st.warning(uni_warning)
    p = sidebar(uni)
    if not p["selected"]:
        st.info("Select assets in the sidebar to begin.")
        return

    est = load_estimates(p, uni)
    for t, err in est.md.failures.items():
        if t in p["selected"]:
            st.error(f"Could not load {t}: {err}")
    if est.error:
        st.error(est.error)
        st.stop()
    for message in est.warnings:
        st.warning(message)
    if est.info:
        st.info(est.info)
        st.stop()
    workspace(est, p, uni)
