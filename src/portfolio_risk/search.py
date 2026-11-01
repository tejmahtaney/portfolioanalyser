"""Search for new stocks to add to a portfolio under risk parity.

Stage 1 is a beam search on the diversification ratio of the risk parity portfolio of holdings plus additions,
estimated on the last WINDOW_WEEKS weeks with an EWMA covariance. Stage 2 runs a walk-forward backtest over the
full history for each finalist and compares its out-of-sample Sharpe ratio with a baseline that has no additions.

Weights are fractions that sum to 1. Nothing here imports Streamlit.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import analytics, data, universe
from .config import (
    BACKTEST_HOLD_WEEKS,
    BACKTEST_TRAIN_WEEKS,
    DEFAULT_RISK_FREE,
    EWMA_HALF_LIFE_WEEKS,
    PERIODS_PER_YEAR,
)
from .optimiser import OptimisationError

HISTORY_PERIOD_SEARCH = "10y"
WINDOW_WEEKS = BACKTEST_TRAIN_WEEKS
BEAM_WIDTH = 5
MAX_NEW_LIMIT = 5
BAND_ABSOLUTE = 0.05
BAND_RELATIVE = 0.25
COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2020-04-30")
MODES = ("reweight", "keep")
BLOCK = "current holdings"

PASSES, FAILS, NO_BACKTEST = "passes", "fails", "no backtest"


# --- Fast risk parity solver ---------------------------------------------------------------------------------

_NEWTON_MAX_ITER = 200
_NEWTON_TOL = 1e-18
_NEWTON_NEAR = 1e-6  # inside this decrement full steps converge quadratically; the objective is too noisy to test


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    """Long-only risk parity weights by damped Newton on min 0.5 y'Σy - Σ log y, then w = y / Σy.

    The optimum has equal risk contributions, w_i (Σw)_i the same for every i. Same problem as
    optimiser.risk_parity, without cvxpy's per-call set-up cost.
    """
    cov = (cov + cov.T) / 2
    n = len(cov)
    y = 1.0 / np.sqrt(np.diag(cov))
    y *= np.sqrt(n / (y @ cov @ y))  # y'Σy = n at the optimum
    f = 0.5 * y @ cov @ y - np.log(y).sum()
    diag = np.diag_indices(n)
    for _ in range(_NEWTON_MAX_ITER):
        grad = cov @ y - 1.0 / y
        hess = cov.copy()
        hess[diag] += 1.0 / y**2
        step = -np.linalg.solve(hess, grad)
        decrement = -grad @ step
        t = 1.0
        while True:
            trial = y + t * step
            if (trial > 0).all():
                f_trial = 0.5 * trial @ cov @ trial - np.log(trial).sum()
                if decrement < _NEWTON_NEAR or f_trial <= f - 1e-4 * t * decrement:
                    break
            t *= 0.5
            if t < 1e-14:  # float noise: the step no longer lowers the objective, so y is as good as it gets
                return y / y.sum()
        y, f = trial, f_trial
        if decrement < _NEWTON_TOL:
            return y / y.sum()
    raise OptimisationError("Risk parity: the Newton solver did not converge.")


# --- Result types --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredSet:
    tickers: tuple[str, ...]  # sorted
    ratio: float  # diversification ratio


@dataclass
class BacktestFigures:
    sharpe: float
    max_drawdown: float
    baseline_sharpe: float
    baseline_max_drawdown: float
    standard_error: float
    weeks: int


@dataclass
class Backtest:
    status: str  # "passes", "fails" or "no backtest"
    full: BacktestFigures | None = None
    ex_covid: BacktestFigures | None = None  # leaves out 1 Feb to 30 Apr 2020
    status_ex_covid: str | None = None
    covid_sensitive: bool = False  # pass or fail differs between the two evaluations
    reason: str | None = None


@dataclass
class Finalist:
    tickers: tuple[str, ...]
    diversification_ratio: float
    volatility: float
    effective_assets: float
    weights: pd.Series  # target weights over holdings and additions
    trades: pd.DataFrame  # Target, Current, Trade (NaN inside the band)
    backtest: Backtest


@dataclass
class SearchResult:
    finalists: list[Finalist]  # ordered by diversification ratio, best first
    best_by_depth: dict[int, ScoredSet]  # best set at each depth below the final one
    depth: int  # size of the finalist sets; below max_new when the search stopped early
    holdings_ratio: float  # diversification ratio of the holdings alone, for reference
    excluded: dict[str, str] = field(default_factory=dict)


Progress = Callable[[int, int, ScoredSet], None]  # depth, max_new, best set at that depth


# --- Inputs --------------------------------------------------------------------------------------------------


def sp500_candidates() -> tuple[list[str], str | None]:
    table, warning = universe.load_sp500()
    return table["ticker"].tolist(), warning


def load_history(
    holdings: Iterable[str], candidates: Iterable[str], progress: Callable[[str], None] | None = None
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Ten years of weekly returns and the tickers that could not be loaded, with their reasons."""
    tickers = list(dict.fromkeys([*holdings, *candidates]))
    market = data.load_market_data(tickers, progress=progress, period=HISTORY_PERIOD_SEARCH)
    return market.returns, market.failures


def _clean_holdings(holdings: pd.Series) -> pd.Series:
    held = holdings[holdings > 0]
    held.index = [str(t).strip() for t in held.index]
    if held.empty:
        raise ValueError("The portfolio has no holdings with a weight above zero.")
    return analytics.normalise_weights(held)


# --- Estimation ----------------------------------------------------------------------------------------------


def _frame(window: pd.DataFrame, held: list[str], props: pd.Series, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """The returns the optimiser sees, and the labels that are always in the portfolio.

    Keep mode replaces the holdings by one column: their returns at the current proportions.
    """
    if mode == "reweight":
        return window, held
    return window.assign(**{BLOCK: window[held] @ props[held]}), [BLOCK]


@dataclass
class _Fit:
    labels: list[str]
    cov: np.ndarray
    w: np.ndarray

    @property
    def ratio(self) -> float:
        return float(self.w @ np.sqrt(np.diag(self.cov)) / self.volatility)

    @property
    def volatility(self) -> float:
        return float(np.sqrt(self.w @ self.cov @ self.w))


def _fit(frame: pd.DataFrame, base: list[str], adds: Sequence[str], half_life: float) -> _Fit:
    labels = [*base, *adds]
    cov = analytics.annual_covariance(frame[labels], "ewma", half_life).to_numpy()
    return _Fit(labels, cov, risk_parity_weights(cov))


def _expand(fit: _Fit, held: list[str], props: pd.Series, mode: str) -> pd.Series:
    """Weights over holdings and additions. In keep mode the block share is spread over the holdings."""
    w = pd.Series(fit.w, index=fit.labels)
    if mode == "reweight":
        return w
    block = w.pop(BLOCK)
    return pd.concat([block * props[held], w])


# --- Band ----------------------------------------------------------------------------------------------------


def trade_table(
    target: pd.Series, current: pd.Series, absolute: float = BAND_ABSOLUTE, relative: float = BAND_RELATIVE
) -> pd.DataFrame:
    """Target, current and trade per ticker. The trade is NaN when the gap is inside the band: smaller than
    `absolute` and smaller than `relative` times the target."""
    tickers = list(dict.fromkeys([*current.index, *target.index]))
    tgt = target.reindex(tickers).fillna(0.0)
    cur = current.reindex(tickers).fillna(0.0)
    gap = tgt - cur
    inside = (gap.abs() < absolute) & (gap.abs() < relative * tgt)
    return pd.DataFrame({"Target": tgt, "Current": cur, "Trade": gap.mask(inside)})


# --- Stage 2: walk-forward backtest --------------------------------------------------------------------------


def _sharpe(oos: pd.Series, risk_free: float) -> float:
    ret = oos.mean() * PERIODS_PER_YEAR
    return analytics.sharpe_ratio(ret, risk_free, oos.std() * np.sqrt(PERIODS_PER_YEAR))


def _figures(oos: pd.Series, base: pd.Series, risk_free: float) -> tuple[BacktestFigures, str]:
    sr, sr_base = _sharpe(oos, risk_free), _sharpe(base, risk_free)
    se = float(np.sqrt((1 + sr_base**2 / 2) / (len(oos) / PERIODS_PER_YEAR)))
    status = FAILS if sr < sr_base - se else PASSES
    figures = BacktestFigures(
        sr, analytics.max_drawdown(oos), sr_base, analytics.max_drawdown(base), se, len(oos)
    )
    return figures, status


def _walk_forward(
    panel: pd.DataFrame, held: list[str], props: pd.Series, adds: Sequence[str], mode: str, half_life: float,
    train: int, hold: int,
) -> pd.Series:
    """Out-of-sample weekly returns of risk parity over holdings plus `adds`, refitted every `hold` weeks."""
    frame, base = _frame(panel, held, props, mode)
    parts = []
    for start in range(train, len(panel), hold):
        fit = _fit(frame.iloc[start - train : start], base, adds, half_life)
        w = _expand(fit, held, props, mode)
        parts.append(panel.iloc[start : start + hold][w.index] @ w)
    return pd.concat(parts)


def _baseline(
    panel: pd.DataFrame, held: list[str], props: pd.Series, mode: str, half_life: float, train: int, hold: int
) -> pd.Series:
    """Keep mode: the current weights held fixed. Reweight mode: risk parity on the holdings alone."""
    if mode == "keep":
        return (panel[held] @ props[held]).iloc[train:]
    return _walk_forward(panel, held, props, [], mode, half_life, train, hold)


def _without_covid(oos: pd.Series) -> pd.Series:
    return oos[(oos.index < COVID_START) | (oos.index > COVID_END)]


def _backtest(
    calendar: pd.DataFrame, held: list[str], props: pd.Series, adds: Sequence[str], mode: str,
    baseline: pd.Series | None, risk_free: float, half_life: float, train: int, hold: int,
) -> Backtest:
    assets = [*held, *adds]
    short = [t for t in assets if calendar[t].isna().any()]
    if short:
        return Backtest(NO_BACKTEST, reason="Less than the full history for: " + ", ".join(short))
    if len(calendar) <= train:
        return Backtest(NO_BACKTEST, reason=f"Needs more than {train} weeks; {len(calendar)} available.")
    oos = _walk_forward(calendar[assets], held, props, adds, mode, half_life, train, hold)
    full, status = _figures(oos, baseline, risk_free)
    ex, status_ex = _figures(_without_covid(oos), _without_covid(baseline), risk_free)
    return Backtest(status, full, ex, status_ex, status != status_ex)


# --- Search --------------------------------------------------------------------------------------------------


def search(
    returns: pd.DataFrame,
    holdings: pd.Series,
    candidates: Iterable[str],
    max_new: int = 3,
    mode: str = "reweight",
    absolute_band: float = BAND_ABSOLUTE,
    relative_band: float = BAND_RELATIVE,
    beam_width: int = BEAM_WIDTH,
    risk_free: float = DEFAULT_RISK_FREE,
    half_life: float = EWMA_HALF_LIFE_WEEKS,
    unavailable: Mapping[str, str] | None = None,
    progress: Progress | None = None,
    backtest: bool = True,
) -> SearchResult:
    """Finds the sets of up to `max_new` new tickers that make the risk parity portfolio most diversified.

    returns: weekly returns, ten years for the backtest. holdings: weights as from excel_io.import_weights (any
    scale; normalised here). unavailable: tickers that could not be loaded, with reasons (load_history).
    """
    if not 1 <= max_new <= MAX_NEW_LIMIT:
        raise ValueError(f"max_new must be between 1 and {MAX_NEW_LIMIT}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    if len(returns) < WINDOW_WEEKS:
        raise ValueError(f"The search needs {WINDOW_WEEKS} weeks of returns; {len(returns)} available.")

    props = _clean_holdings(holdings)
    held = list(props.index)
    window = returns.iloc[-WINDOW_WEEKS:]
    calendar = returns.loc[returns.notna().any(axis=1)]

    for t in held:
        if t not in returns or window[t].isna().any():
            have = int(window[t].notna().sum()) if t in returns else 0
            raise ValueError(f"Holding {t} has returns for {have} of the last {WINDOW_WEEKS} weeks.")
        if not window[t].std() > 0:
            raise ValueError(f"Holding {t} has no price variation over the last {WINDOW_WEEKS} weeks.")

    excluded: dict[str, str] = {}
    eligible: list[str] = []
    for t in dict.fromkeys(str(c).strip() for c in candidates):
        if t in props.index:
            continue
        if t not in returns:
            excluded[t] = (unavailable or {}).get(t, "No price history.")
        elif window[t].isna().any():
            excluded[t] = f"Returns for only {int(window[t].notna().sum())} of the last {WINDOW_WEEKS} weeks."
        elif not window[t].std() > 0:
            excluded[t] = f"No price variation over the last {WINDOW_WEEKS} weeks."
        else:
            eligible.append(t)
    if not eligible:
        raise ValueError("No candidate has enough history to be tested.")
    eligible.sort()
    max_new = min(max_new, len(eligible))

    frame, base = _frame(window, held, props, mode)

    def score(adds: Iterable[str]) -> ScoredSet:
        adds = tuple(sorted(adds))
        return ScoredSet(adds, _fit(frame, base, adds, half_life).ratio)

    def best(sets: Iterable[ScoredSet]) -> list[ScoredSet]:
        return sorted(sets, key=lambda s: (-s.ratio, s.tickers))[:beam_width]

    beams = [best(score([c]) for c in eligible)]
    if progress:
        progress(1, max_new, beams[0][0])
    for depth in range(2, max_new + 1):
        seen: set[tuple[str, ...]] = set()
        scored = []
        for kept in beams[-1]:
            for c in eligible:
                key = tuple(sorted((*kept.tickers, c)))
                if c in kept.tickers or key in seen:
                    continue
                seen.add(key)
                scored.append(score(key))
        nxt = best(scored)
        if progress:
            progress(depth, max_new, nxt[0])
        if nxt[0].ratio <= beams[-1][0].ratio:  # no extension raised the best score
            break
        beams.append(nxt)

    base_baseline = None
    if backtest and len(calendar) > WINDOW_WEEKS and not calendar[held].isna().any().any():
        base_baseline = _baseline(
            calendar[held], held, props, mode, half_life, WINDOW_WEEKS, BACKTEST_HOLD_WEEKS
        )

    finalists = []
    for chosen in beams[-1]:
        fit = _fit(frame, base, chosen.tickers, half_life)
        weights = _expand(fit, held, props, mode)
        if base_baseline is None:
            result = Backtest(NO_BACKTEST, reason="Backtest not available for the current holdings.")
        else:
            result = _backtest(
                calendar, held, props, chosen.tickers, mode, base_baseline, risk_free, half_life,
                WINDOW_WEEKS, BACKTEST_HOLD_WEEKS,
            )
        finalists.append(
            Finalist(
                chosen.tickers, fit.ratio, fit.volatility, float(1 / (weights**2).sum()), weights,
                trade_table(weights, props, absolute_band, relative_band), result,
            )
        )
    holdings_ratio = _fit(frame, base, (), half_life).ratio
    return SearchResult(
        finalists, {d + 1: b[0] for d, b in enumerate(beams[:-1])}, len(beams), holdings_ratio, excluded
    )
