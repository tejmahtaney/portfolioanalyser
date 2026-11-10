"""Streamlit page "Add positions": runs search.py on an uploaded portfolio and shows the finalists."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import excel_io, search, ui
from .search import Finalist, SearchResult

SOURCES = ("Watchlist", "S&P 500")
MODE_LABELS = {"Reweight everything": "reweight", "Keep current holdings": "keep"}
SIZING_LABELS = {
    f"Fixed {search.SLICE:.0%} per addition": "slice", "Risk parity": "risk_parity",
    "Balanced (risk parity + max Sharpe)": "balanced",
}
MAX_REMEMBERED_SEARCHES = 6
DASH = "–"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Widget keys. Streamlit drops a widget's state while another page is showing, so these are copied aside and restored.
K_SOURCE, K_WATCH, K_MAX_NEW, K_MODE = "ap_source", "ap_watchlist", "ap_max_new", "ap_mode"
K_SIZING, K_CAP, K_TILT, K_BUDGET = "ap_sizing", "ap_cap", "ap_tilt", "ap_budget"
K_BAND_ABS, K_BAND_REL, K_FINALIST = "ap_band_abs", "ap_band_rel", "ap_finalist"
DEFAULTS = {
    K_SOURCE: SOURCES[0], K_WATCH: [], K_MAX_NEW: 3, K_MODE: "Reweight everything",
    K_SIZING: next(iter(SIZING_LABELS)), K_CAP: False, K_TILT: int(search.DEFAULT_TILT * 100),
    K_BUDGET: search.DEFAULT_NEW_BUDGET * 100,
    K_BAND_ABS: search.BAND_ABSOLUTE * 100, K_BAND_REL: search.BAND_RELATIVE * 100, K_FINALIST: None,
}
HOLDINGS_KEY, RESULTS_KEY = "ap_holdings", "ap_results"


@dataclass
class Holdings:
    name: str
    weights: pd.Series  # percent, as imported


@dataclass
class Stored:
    result: SearchResult
    held_weeks: dict[str, int]  # weeks of returns per held ticker, for the "no backtest" explanation
    mode: str


# --- Pure helpers --------------------------------------------------------------------------------------------


def _fmt(x: float | None, spec: str) -> str:
    return DASH if x is None or pd.isna(x) else format(x, spec)


def finalist_label(rank: int, finalist: Finalist) -> str:
    return f"{rank}. {', '.join(finalist.tickers)}"


def covid_flag(finalist: Finalist) -> str:
    bt = finalist.backtest
    return f"⚠ {bt.status} → {bt.status_ex_covid}" if bt.covid_sensitive else ""


def finalist_table(result: SearchResult) -> pd.DataFrame:
    rows = []
    for f in result.finalists:
        bt = f.backtest
        full = bt.full
        rows.append(
            {
                "Added": ", ".join(f.tickers),
                "Δ Volatility": f"{f.metrics.d_volatility:+.2%}",
                "Δ CVaR": f"{f.metrics.d_cvar:+.2%}",
                "Correlation": f"{f.metrics.correlation:.2f}",
                "Diversification ratio": f"{f.diversification_ratio:.3f}",
                "Volatility": f"{f.volatility:.2%}",
                "Effective assets": f"{f.effective_assets:.1f}",
                "Backtest": bt.status,
                "Sharpe (out of sample)": _fmt(full and full.sharpe, ".2f"),
                "Sharpe (baseline)": _fmt(full and full.baseline_sharpe, ".2f"),
                "Max drawdown": _fmt(full and full.max_drawdown, ".1%"),
                "Without Feb–Apr 2020": covid_flag(f),
            }
        )
    return pd.DataFrame(rows, index=range(1, len(rows) + 1))


def trade_display(finalist: Finalist, absolute: float, relative: float, names: pd.Series) -> pd.DataFrame:
    """Target, current and trade in percent, with the band applied at the given thresholds (fractions)."""
    trades = search.trade_table(finalist.weights, finalist.trades["Current"], absolute, relative)
    trades = trades.sort_values("Target", ascending=False)
    return pd.DataFrame(
        {
            "Name": [names.get(t, t) for t in trades.index],
            "Target": [f"{x:.2%}" for x in trades["Target"]],
            "Current": [f"{x:.2%}" for x in trades["Current"]],
            "Trade": [DASH if pd.isna(x) else f"{x:+.2%}" for x in trades["Trade"]],
        },
        index=trades.index.rename("Ticker"),
    )


def no_backtest_notice(stored: Stored) -> tuple[str, str] | None:
    """(kind, message) when no finalist could be backtested; names the short holdings if they are the cause."""
    finalists = stored.result.finalists
    if not finalists or any(f.backtest.status != search.NO_BACKTEST for f in finalists):
        return None
    short = {t: w for t, w in stored.held_weeks.items() if w < search.MIN_BACKTEST_WEEKS}
    if short:
        listed = ", ".join(f"{t} ({w} weeks)" for t, w in sorted(short.items(), key=lambda kv: kv[1]))
        return "warning", (
            f"No backtest: current holdings with under {search.MIN_BACKTEST_WEEKS} weeks of history: {listed}."
        )
    return "info", finalists[0].backtest.reason or "No backtest."


def export_target_weights(finalist: Finalist, names: pd.Series, mode: str) -> bytes:
    """The finalist's target weights in the Portfolio sheet format that the Portfolio risk page imports."""
    target = finalist.weights
    portfolio = pd.DataFrame({"Name": [names.get(t, t) for t in target.index], "Weight": target.to_numpy()}, index=target.index)
    settings = {
        "Source": "Add positions search",
        "Added": ", ".join(finalist.tickers),
        "Mode": mode,
        "Diversification ratio": f"{finalist.diversification_ratio:.3f}",
    }
    return excel_io.export_workbook(portfolio, settings)


# --- Session state -------------------------------------------------------------------------------------------


def restore_inputs() -> None:
    state = st.session_state
    for key, default in DEFAULTS.items():
        if key not in state:
            state[key] = state.get(f"kept_{key}", default)


def keep_inputs() -> None:
    for key in DEFAULTS:
        if key in st.session_state:
            st.session_state[f"kept_{key}"] = st.session_state[key]


def search_key(
    holdings: pd.Series, candidates: tuple, max_new: int, mode: str, sizing: str, cap: bool, tilt: int,
    budget: float = 100.0,
) -> tuple:
    held = tuple(sorted((t, round(float(w), 6)) for t, w in holdings.items() if w > 0))
    return held, candidates, max_new, mode, sizing, cap, tilt, round(float(budget), 6)


def remember(key: tuple, stored: Stored) -> None:
    runs = st.session_state.setdefault(RESULTS_KEY, {})
    runs.pop(key, None)
    runs[key] = stored
    while len(runs) > MAX_REMEMBERED_SEARCHES:
        runs.pop(next(iter(runs)))


# --- Page ----------------------------------------------------------------------------------------------------


def read_holdings() -> Holdings | None:
    upload = st.file_uploader(
        "Current portfolio", type="xlsx", help="A workbook exported from the Portfolio risk page.", key="ap_upload"
    )
    if upload is not None and upload.file_id != st.session_state.get("ap_upload_id"):
        st.session_state["ap_upload_id"] = upload.file_id
        try:
            st.session_state[HOLDINGS_KEY] = Holdings(upload.name, excel_io.import_weights(upload))
        except Exception as exc:
            st.session_state.pop(HOLDINGS_KEY, None)
            st.error(f"Import failed: {exc}")
            return None
    return st.session_state.get(HOLDINGS_KEY)


def show_holdings(holdings: Holdings) -> pd.Series:
    """Shows the holdings read and returns those with a weight above zero, in percent."""
    weights = holdings.weights
    held = weights[weights > 0]
    st.caption(f"Read {len(held)} holdings from {holdings.name}.")
    shown = held.rename("Weight (%)").rename_axis("Ticker").to_frame()
    shown["Weight (%)"] = (shown["Weight (%)"] / held.sum() * 100).round(2)
    st.dataframe(shown)
    if held.empty:
        st.error("The portfolio has no holdings with a weight above zero.")
    return held


def candidate_inputs(uni: pd.DataFrame) -> tuple:
    """The candidate spec, ("sp500",) or ("watchlist", sorted tickers). It doubles as part of the result key."""
    names = uni.set_index("ticker")["name"]
    source = st.radio("Candidates", SOURCES, horizontal=True, key=K_SOURCE)
    if source == "S&P 500":
        return ("sp500",)
    watch = st.multiselect(
        "Watchlist", options=list(names.index), format_func=lambda t: f"{t} — {names[t]}",
        placeholder="Search by ticker or name", key=K_WATCH,
    )
    return ("watchlist", tuple(sorted(watch)))


def run_search(
    held_pct: pd.Series, candidates: tuple, max_new: int, mode_label: str, sizing_label: str, cap: bool, tilt: int,
    budget: float, key: tuple,
) -> None:
    mode, sizing = MODE_LABELS[mode_label], SIZING_LABELS[sizing_label]
    bar = st.progress(0.0, text="Loading prices…")

    def on_load(message: str) -> None:
        bar.progress(0.0, text=message)

    def on_depth(depth: int, total: int, best: search.ScoredSet) -> None:
        bar.progress(depth / total, text=f"Additions {depth} of {total}: best {', '.join(best.tickers)} ({best.metrics.d_volatility:+.2%} volatility)")

    try:
        warning = None
        if candidates[0] == "sp500":
            tickers, warning = search.sp500_candidates()
        else:
            tickers = list(candidates[1])
        returns, failures = search.load_history(list(held_pct.index), tickers, progress=on_load)
        result = search.search(
            returns, held_pct, tickers, max_new=max_new, mode=mode, sizing=sizing, cap_positions=cap,
            tilt=tilt / 100, new_budget=budget / 100, unavailable=failures, progress=on_depth,
        )
    except Exception as exc:
        bar.empty()
        st.error(f"Search failed: {exc}")
        return
    bar.empty()
    if warning:
        st.warning(warning)
    held_weeks = {t: int(returns[t].notna().sum()) if t in returns else 0 for t in held_pct.index}
    if result.trimmed_weeks:
        st.warning(
            f"The latest {result.trimmed_weeks} week(s) were left out: at least one holding has no price for them yet."
        )
    remember(key, Stored(result, held_weeks, mode))


def show_results(stored: Stored, names: pd.Series, absolute: float, relative: float) -> None:
    result = stored.result
    if not result.finalists:
        st.info("No finalists.")
        return
    st.subheader("Finalists")
    st.caption(
        f"Ranked by the mean rank on change in volatility, change in CVaR and correlation with your holdings, each "
        f"addition at {search.SLICE:.0%}. "
        f"Holdings alone: diversification ratio {result.holdings_ratio:.3f}."
    )
    notice = no_backtest_notice(stored)
    if notice:
        getattr(st, notice[0])(notice[1])
    st.dataframe(finalist_table(result))

    labels = [finalist_label(i, f) for i, f in enumerate(result.finalists, 1)]
    if st.session_state.get(K_FINALIST) not in labels:
        st.session_state[K_FINALIST] = labels[0]
    choice = st.selectbox("Finalist", labels, key=K_FINALIST)
    finalist = result.finalists[labels.index(choice)]

    st.subheader(f"Trades for {', '.join(finalist.tickers)}")
    st.dataframe(trade_display(finalist, absolute, relative, names))
    st.caption(f"{DASH} means the gap is inside the no-trade band.")
    st.download_button(
        "Export target weights", export_target_weights(finalist, names, stored.mode),
        file_name=f"target_weights_{'_'.join(finalist.tickers)}.xlsx", mime=XLSX_MIME,
    )

    if result.excluded:
        with st.expander(f"Excluded candidates ({len(result.excluded)})", expanded=False):
            st.dataframe(pd.Series(result.excluded, name="Reason").rename_axis("Ticker").to_frame())


def captions() -> None:
    st.caption(f"Weights use the last {search.WINDOW_WEEKS} weeks with recent weeks weighted more.")
    st.caption("The backtest covers 10 years.")
    st.caption('"fails" means an out-of-sample Sharpe more than one standard error below the baseline.')
    st.caption("Results are historical and are not financial advice.")


def page() -> None:
    st.set_page_config(page_title="Add positions", layout="wide")
    st.title("Add positions")
    st.write(
        f"Finds up to {search.MAX_NEW_LIMIT} stocks to add to your portfolio so that, at risk parity, it is as diversified as possible."
    )
    restore_inputs()
    uni, uni_warning = ui.cached_universe()
    if uni_warning:
        st.warning(uni_warning)
    names = uni.set_index("ticker")["name"]

    holdings = read_holdings()
    held_pct = show_holdings(holdings) if holdings is not None else None
    candidates = candidate_inputs(uni)
    max_new = st.slider("Maximum new positions", 1, search.MAX_NEW_LIMIT, key=K_MAX_NEW)
    sizing_label = st.radio(
        "Sizing", list(SIZING_LABELS), horizontal=True, key=K_SIZING,
        help="Fixed: each addition at that weight, holdings scaled down in proportion. "
        "Risk parity: sized for equal risk. Balanced: risk parity blended with the max Sharpe weights.",
    )
    mode_label = st.radio(
        "Mode (risk parity and balanced)", list(MODE_LABELS), horizontal=True, key=K_MODE,
        disabled=SIZING_LABELS[sizing_label] == "slice",
    )
    cap = st.checkbox(
        "Cap positions at my largest holding (risk parity and balanced)", key=K_CAP,
        disabled=SIZING_LABELS[sizing_label] == "slice",
        help="No recommended position is larger than the largest weight in your portfolio. Positions above it "
        "are held at it and the rest re-solved; none is dropped.",
    )
    tilt = st.slider(
        "Max Sharpe share (%)", 0, 100, step=5, key=K_TILT, disabled=SIZING_LABELS[sizing_label] != "balanced",
        help="Balanced sizing: 0 is pure risk parity, 100 is pure max Sharpe. Max Sharpe uses shrunk historical "
        "returns, so it favours what did best over the window.",
    )
    budget = st.number_input(
        "New positions in total: at most (% of the portfolio)", min_value=1.0, max_value=100.0, step=1.0, key=K_BUDGET,
        help="The additions together never take more than this. Without a limit, risk parity gives each addition "
        "the same risk as your whole current portfolio (keep mode) or as any single holding (reweight mode), and "
        "low-volatility additions then take most of the weight. 100 switches the limit off.",
    )
    left, right = st.columns(2)
    absolute = left.number_input(
        "Band: absolute (points)", min_value=0.0, max_value=100.0, step=0.5, key=K_BAND_ABS,
        help="A trade is skipped when the gap is smaller than this many percentage points and the relative size below.",
    ) / 100
    relative = right.number_input(
        "Band: relative (%)", min_value=0.0, max_value=100.0, step=5.0, key=K_BAND_REL,
        help="...and smaller than this percentage of the target weight.",
    ) / 100
    keep_inputs()

    ready = held_pct is not None and not held_pct.empty and (candidates[0] == "sp500" or bool(candidates[1]))
    key = search_key(held_pct, candidates, max_new, MODE_LABELS[mode_label], SIZING_LABELS[sizing_label], cap, tilt, budget) if ready else None
    if st.button("Run search", type="primary", disabled=not ready):
        run_search(held_pct, candidates, max_new, mode_label, sizing_label, cap, tilt, budget, key)
    if not ready:
        st.info("Upload a portfolio and choose candidates to begin.")
    else:
        stored = st.session_state.get(RESULTS_KEY, {}).get(key)
        if stored is None:
            st.info("Press Run search to see results for these inputs.")
        else:
            show_results(stored, names, absolute, relative)
    captions()
