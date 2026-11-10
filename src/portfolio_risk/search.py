"""Search for new stocks to add to a portfolio.

Stage 1 is a beam search over additions held at a fixed small slice each (SLICE), with the holdings scaled down in
proportion. A candidate that raises the portfolio's CVaR is dropped. A set is ranked by the mean of its ranks on
the change in volatility, the change in CVaR and the correlation of the additions with the holdings, over the last
WINDOW_WEEKS weeks. The finalists are then sized in one of three ways: each addition at SLICE (the default, or the
largest input position when that is smaller); by risk parity with an EWMA covariance; or "balanced", risk parity
blended with the max Sharpe weights. The last two can cap every position at the largest input position. Under every
sizing the additions together take at most a set share of the portfolio (new_budget). Stage 2 runs
a walk-forward backtest over the full history for each finalist and compares its out-of-sample Sharpe ratio with a
baseline that has no additions.

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
from . import optimiser
from .optimiser import OptimisationError

HISTORY_PERIOD_SEARCH = "10y"
WINDOW_WEEKS = BACKTEST_TRAIN_WEEKS
MAX_TRIM_WEEKS = 2
BEAM_WIDTH = 5
MIN_BACKTEST_WEEKS = 500
MAX_NEW_LIMIT = 10
BAND_ABSOLUTE = 0.05
BAND_RELATIVE = 0.25
COVID_START = pd.Timestamp("2020-02-01")
COVID_END = pd.Timestamp("2020-04-30")
MODES = ("reweight", "keep")
SIZINGS = ("slice", "risk_parity", "balanced")
DEFAULT_TILT = 0.5  # share of max Sharpe in balanced sizing
DEFAULT_NEW_BUDGET = 0.15  # largest total weight of the additions offered by the page
BLOCK = "current holdings"
SLICE = 0.05  # weight of each addition while ranking, and in the recommendation under slice sizing
CVAR_LEVEL = 0.10  # share of the worst weeks averaged for CVaR

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


def capped_risk_parity_weights(cov: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Risk parity with w_i <= upper_i. An asset above its bound is held at the bound and the rest are
    re-solved on what remains, repeating until none is over. No asset is dropped: every free asset keeps a
    positive weight."""
    n = len(cov)
    if upper.sum() < 1 - 1e-12:
        raise OptimisationError("The position cap leaves the weights unable to sum to 100%.")
    w = risk_parity_weights(cov)
    free = np.ones(n, bool)
    while True:
        over = free & (w > upper + 1e-12)
        if not over.any():
            return w
        free &= ~over
        fixed = ~free
        w = np.where(fixed, upper, 0.0)
        if free.any():
            w[free] = risk_parity_weights(cov[np.ix_(free, free)]) * (1.0 - upper[fixed].sum())


def cap_weights(w: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Holds any weight above its bound at the bound and spreads the excess over the others in proportion."""
    w = w.copy()
    free = np.ones(len(w), bool)
    while True:
        over = free & (w > upper + 1e-12)
        if not over.any():
            return w
        free &= ~over
        w[over] = upper[over]
        if not free.any() or w[free].sum() <= 0:
            return w
        w[free] *= (1.0 - w[~free].sum()) / w[free].sum()


def allocate(relative: np.ndarray, total: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Weights summing to `total`, each inside [lower, upper], as close to proportional to `relative` as the bounds
    allow: w_i = clip(s * relative_i, lower_i, upper_i), with the scale s found by bisection."""
    if len(relative) == 0:
        return relative.astype(float)
    if lower.sum() > total + 1e-12 or upper.sum() < total - 1e-12:
        raise OptimisationError("The position limits cannot be met together.")
    relative = np.maximum(relative, 1e-12)
    low, high = 0.0, float((upper / relative).max())
    for _ in range(200):
        mid = (low + high) / 2
        if np.clip(mid * relative, lower, upper).sum() < total:
            low = mid
        else:
            high = mid
    w = np.clip(high * relative, lower, upper)
    free = (w > lower + 1e-12) & (w < upper - 1e-12)
    if free.any():  # removes the bisection residue
        w[free] += (total - w.sum()) * w[free] / w[free].sum()
    return w


# --- Result types --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """Effect of adding the tickers at SLICE each, against the holdings alone. Lower is better for all three."""

    d_volatility: float  # annual
    d_cvar: float  # weekly loss, positive is worse
    correlation: float  # weekly returns of the additions with those of the holdings


@dataclass(frozen=True)
class ScoredSet:
    tickers: tuple[str, ...]  # sorted
    metrics: Metrics
    score: float  # mean rank over the sets scored together, lower is better


@dataclass
class BacktestFigures:
    sharpe: float
    max_drawdown: float
    baseline_sharpe: float
    baseline_max_drawdown: float
    standard_error: float  # annual, of the Sharpe difference (Memmel-corrected Jobson-Korkie)
    correlation: float  # of the weekly out-of-sample returns of the finalist and the baseline
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
    metrics: Metrics
    score: float
    diversification_ratio: float
    volatility: float
    effective_assets: float
    weights: pd.Series  # target weights over holdings and additions
    trades: pd.DataFrame  # Target, Current, Trade (NaN inside the band)
    backtest: Backtest


@dataclass
class SearchResult:
    finalists: list[Finalist]  # ordered by score, best first
    best_by_depth: dict[int, ScoredSet]  # best set at each depth below the final one
    depth: int  # size of the finalist sets; below max_new when the search stopped early
    holdings_ratio: float  # diversification ratio of the holdings alone, for reference
    excluded: dict[str, str] = field(default_factory=dict)
    trimmed_weeks: int = 0  # trailing weeks dropped because a holding had no bar yet


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


def _trim_incomplete_tail(returns: pd.DataFrame, held: list[str]) -> tuple[pd.DataFrame, int]:
    """Drops up to MAX_TRIM_WEEKS trailing weeks in which a holding has no return yet.

    The calendar is shared with futures and FX, which can have the current week's bar before equities do.
    A longer gap is left alone so that a stalled holding still fails the completeness check.
    """
    present = [t for t in held if t in returns]
    if not present:
        return returns, 0
    complete = returns[present].notna().all(axis=1).to_numpy().nonzero()[0]
    trailing = len(returns) - 1 - int(complete[-1]) if len(complete) else 0
    if not 0 < trailing <= MAX_TRIM_WEEKS:
        return returns, 0
    return returns.iloc[:-trailing], trailing


# --- Estimation ----------------------------------------------------------------------------------------------


def _frame(window: pd.DataFrame, held: list[str], props: pd.Series, mode: str) -> tuple[pd.DataFrame, list[str]]:
    """The returns the optimiser sees, and the labels that are always in the portfolio.

    Keep mode replaces the holdings by one column: their returns at the current proportions.
    """
    if mode == "reweight":
        return window, held
    return window.assign(**{BLOCK: window[held] @ props[held]}), [BLOCK]


@dataclass(frozen=True)
class Sizing:
    """cap: largest weight any single stock may take. tilt: share of the max Sharpe weights in the blend with risk
    parity (0 is pure risk parity). risk_free: for the max Sharpe weights. budget: largest total weight of the
    additions."""

    cap: float = 1.0
    tilt: float = 0.0
    risk_free: float = DEFAULT_RISK_FREE
    budget: float = 1.0


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


def _fit(
    frame: pd.DataFrame, base: list[str], adds: Sequence[str], half_life: float, sizing: Sizing = Sizing()
) -> _Fit:
    """Risk parity weights, blended with max Sharpe weights when sizing.tilt > 0 (kept as risk parity when no
    portfolio beats the risk-free rate). The keep-mode block is a share of the holdings, which already sit at or
    under a cap set from them, so it is bounded by 1."""
    labels = [*base, *adds]
    cov_df = analytics.annual_covariance(frame[labels], "ewma", half_life)
    cov = cov_df.to_numpy()
    upper = np.array([1.0 if label == BLOCK else sizing.cap for label in labels])
    w = capped_risk_parity_weights(cov, upper)
    if sizing.tilt > 0:
        mu = analytics.expected_returns(frame[labels])
        try:
            sharpe_w = optimiser.max_sharpe(mu, cov_df, sizing.risk_free, None).weights.to_numpy()
        except OptimisationError:
            sharpe_w = None
        if sharpe_w is not None:
            w = cap_weights((1 - sizing.tilt) * w + sizing.tilt * sharpe_w, upper)
    return _Fit(labels, cov, _limit_additions(w, len(base), upper, sizing))


def _limit_additions(w: np.ndarray, n_base: int, upper: np.ndarray, sizing: Sizing) -> np.ndarray:
    """Scales the additions down to sizing.budget in total when they exceed it, and the rest up to fill the gap.
    Each side keeps its internal proportions, as far as the cap allows."""
    total = float(w[n_base:].sum())
    if n_base == len(w) or total <= sizing.budget + 1e-12:
        return w
    zero = np.zeros(len(w))
    out = np.empty(len(w))
    out[n_base:] = allocate(w[n_base:], sizing.budget, zero[n_base:], upper[n_base:])
    out[:n_base] = allocate(w[:n_base], 1 - sizing.budget, zero[:n_base], upper[:n_base])
    return out


def _expand(fit: _Fit, held: list[str], props: pd.Series, mode: str) -> pd.Series:
    """Weights over holdings and additions. In keep mode the block share is spread over the holdings."""
    w = pd.Series(fit.w, index=fit.labels)
    if mode == "reweight":
        return w
    block = w.pop(BLOCK)
    return pd.concat([block * props[held], w])


def _cvar(returns: np.ndarray) -> float:
    worst = max(1, int(np.ceil(CVAR_LEVEL * len(returns))))
    return float(-np.sort(returns)[:worst].mean())


def _slice_size(props: pd.Series, n_adds: int = 1, budget: float = 1.0) -> float:
    """Weight of each addition: SLICE, or the largest input position or an equal share of the budget for
    additions when one of those is smaller."""
    return min(SLICE, float(props.max()), budget / n_adds)


def _slice_returns(hold: pd.Series, panel: pd.DataFrame, adds: Sequence[str], size: float) -> pd.Series:
    """Weekly returns of the holdings scaled to 1 - size * len(adds), with each addition at `size`."""
    return hold * (1 - size * len(adds)) + size * panel[list(adds)].sum(axis=1)


def _slice_weights(props: pd.Series, adds: Sequence[str], size: float) -> pd.Series:
    return pd.concat([props * (1 - size * len(adds)), pd.Series(size, index=list(adds))])


def _stats(window: pd.DataFrame, weights: pd.Series, half_life: float) -> tuple[float, float]:
    """(diversification ratio, volatility) of fixed weights under the EWMA covariance."""
    cov = analytics.annual_covariance(window[list(weights.index)], "ewma", half_life).to_numpy()
    w = weights.to_numpy()
    vol = float(np.sqrt(w @ cov @ w))
    return float(w @ np.sqrt(np.diag(cov)) / vol), vol


def _set_metrics(window: pd.DataFrame, hold: pd.Series, adds: tuple[str, ...], size: float) -> Metrics:
    """Metrics of the holdings plus the additions at `size` each, against the holdings alone. `hold` is the weekly
    return of the holdings at their current proportions."""
    sleeve = window[list(adds)].sum(axis=1).to_numpy()
    h = hold.to_numpy()
    p = _slice_returns(hold, window, adds, size).to_numpy()
    d_vol = (p.std(ddof=1) - h.std(ddof=1)) * np.sqrt(PERIODS_PER_YEAR)
    return Metrics(float(d_vol), _cvar(p) - _cvar(h), float(np.corrcoef(h, sleeve)[0, 1]))


def _rank(sets: Iterable[tuple[tuple[str, ...], Metrics]]) -> list[ScoredSet]:
    """Orders sets by the mean of their ranks on the three metrics, best first."""
    sets = list(sets)
    table = pd.DataFrame([[m.d_volatility, m.d_cvar, m.correlation] for _, m in sets])
    scores = table.rank(method="average").mean(axis=1)
    scored = [ScoredSet(tickers, m, float(sc)) for (tickers, m), sc in zip(sets, scores)]
    return sorted(scored, key=lambda x: (x.score, x.tickers))


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


def _sharpe_difference_se(oos: pd.Series, base: pd.Series, risk_free: float) -> tuple[float, float]:
    """(annual standard error, correlation) of the difference of two Sharpe ratios on the same weeks.

    Jobson-Korkie with Memmel's (2003) correction, on weekly Sharpe ratios s1, s2 and correlation rho:
    Var = [2(1 - rho) + 0.5 (s1^2 + s2^2 - 2 s1 s2 rho^2)] / T. The annual standard error is sqrt(52 Var).
    """
    weekly_rf = risk_free / PERIODS_PER_YEAR
    s1 = (oos.mean() - weekly_rf) / oos.std()
    s2 = (base.mean() - weekly_rf) / base.std()
    rho = float(oos.corr(base))
    var = (2 * (1 - rho) + 0.5 * (s1**2 + s2**2 - 2 * s1 * s2 * rho**2)) / len(oos)
    return float(np.sqrt(PERIODS_PER_YEAR * max(var, 0.0))), rho


def _figures(oos: pd.Series, base: pd.Series, risk_free: float) -> tuple[BacktestFigures, str]:
    if not oos.index.equals(base.index):
        raise ValueError("The finalist and the baseline must be evaluated on the same weeks.")
    sr, sr_base = _sharpe(oos, risk_free), _sharpe(base, risk_free)
    se, rho = _sharpe_difference_se(oos, base, risk_free)
    status = FAILS if sr - sr_base < -se else PASSES
    figures = BacktestFigures(
        sr, analytics.max_drawdown(oos), sr_base, analytics.max_drawdown(base), se, rho, len(oos)
    )
    return figures, status


def _walk_forward(
    panel: pd.DataFrame, held: list[str], props: pd.Series, adds: Sequence[str], mode: str, half_life: float,
    train: int, hold: int, sizing: Sizing = Sizing(),
) -> pd.Series:
    """Out-of-sample weekly returns of risk parity over holdings plus `adds`, refitted every `hold` weeks."""
    frame, base = _frame(panel, held, props, mode)
    parts = []
    for start in range(train, len(panel), hold):
        fit = _fit(frame.iloc[start - train : start], base, adds, half_life, sizing)
        w = _expand(fit, held, props, mode)
        parts.append(panel.iloc[start : start + hold][w.index] @ w)
    return pd.concat(parts)


def _baseline(
    panel: pd.DataFrame, held: list[str], props: pd.Series, mode: str, half_life: float, train: int, hold: int,
    sizing: Sizing = Sizing(),
) -> pd.Series:
    """Keep mode: the current weights held fixed. Reweight mode: risk parity on the holdings alone."""
    if mode == "keep":
        return (panel[held] @ props[held]).iloc[train:]
    return _walk_forward(panel, held, props, [], mode, half_life, train, hold, sizing)


def _without_covid(oos: pd.Series) -> pd.Series:
    return oos[(oos.index < COVID_START) | (oos.index > COVID_END)]


def _backtest(
    returns: pd.DataFrame, held: list[str], props: pd.Series, adds: Sequence[str], mode: str,
    baselines: dict[int, pd.Series], risk_free: float, half_life: float, train: int, hold: int,
    sizing: str = "risk_parity", fit_sizing: Sizing = Sizing(),
) -> Backtest:
    """Backtests on the weeks where every asset of the finalist has returns. The baseline is computed on those
    same weeks; `baselines` caches it by window, since most finalists share one."""
    window = data.common_window(returns, [*held, *adds])
    if len(window) < MIN_BACKTEST_WEEKS:
        return Backtest(NO_BACKTEST, reason=f"Common history is {len(window)} weeks; {MIN_BACKTEST_WEEKS} needed.")
    key = hash(window.index.asi8.tobytes())
    if sizing == "slice":
        if key not in baselines:
            baselines[key] = _baseline(window[held], held, props, "keep", half_life, train, hold)
        baseline = baselines[key]
        size = _slice_size(props, len(adds), fit_sizing.budget)
        oos = _slice_returns(window[held] @ props[held], window, adds, size).iloc[train:]
    else:
        if key not in baselines:
            baselines[key] = _baseline(window[held], held, props, mode, half_life, train, hold, fit_sizing)
        baseline = baselines[key]
        oos = _walk_forward(window, held, props, adds, mode, half_life, train, hold, fit_sizing)
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
    sizing: str = "slice",
    cap_positions: bool = False,
    tilt: float = DEFAULT_TILT,
    new_budget: float = 1.0,
    absolute_band: float = BAND_ABSOLUTE,
    relative_band: float = BAND_RELATIVE,
    beam_width: int = BEAM_WIDTH,
    risk_free: float = DEFAULT_RISK_FREE,
    half_life: float = EWMA_HALF_LIFE_WEEKS,
    unavailable: Mapping[str, str] | None = None,
    progress: Progress | None = None,
    backtest: bool = True,
) -> SearchResult:
    """Finds the sets of up to `max_new` new tickers that most improve the holdings, then sizes them by risk parity.

    returns: weekly returns, ten years for the backtest. holdings: weights as from excel_io.import_weights (any
    scale; normalised here). new_budget: largest total weight of the additions, under every sizing; ranking
    still uses SLICE per addition. unavailable: tickers that could not be loaded, with reasons (load_history).
    """
    if not 1 <= max_new <= MAX_NEW_LIMIT:
        raise ValueError(f"max_new must be between 1 and {MAX_NEW_LIMIT}.")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}.")
    if sizing not in SIZINGS:
        raise ValueError(f"sizing must be one of {SIZINGS}.")
    if not 0 <= tilt <= 1:
        raise ValueError("tilt must be between 0 and 1.")
    if not 0 < new_budget <= 1:
        raise ValueError("new_budget must be above 0 and at most 1.")

    props = _clean_holdings(holdings)
    held = list(props.index)
    returns, trimmed = _trim_incomplete_tail(returns, held)
    if len(returns) < WINDOW_WEEKS:
        raise ValueError(f"The search needs {WINDOW_WEEKS} weeks of returns; {len(returns)} available.")
    window = returns.iloc[-WINDOW_WEEKS:]

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
    hold = window[held] @ props[held]
    size = _slice_size(props)
    passed = []
    for t in sorted(eligible):
        if _set_metrics(window, hold, (t,), size).d_cvar > 0:
            excluded[t] = f"Raises the portfolio's CVaR at a {size:.0%} position."
        else:
            passed.append(t)
    if not passed:
        raise ValueError(f"No candidate lowers the portfolio's CVaR at a {size:.0%} position.")
    eligible = passed
    max_new = min(max_new, len(eligible))

    frame, base = _frame(window, held, props, mode)
    fit_sizing = Sizing(  # cap and tilt: risk parity and balanced sizing only
        float(props.max()) if cap_positions else 1.0, tilt if sizing == "balanced" else 0.0, risk_free, new_budget
    )

    def best(keys: Iterable[tuple[str, ...]]) -> list[ScoredSet]:
        return _rank((k, _set_metrics(window, hold, k, size)) for k in keys)[:beam_width]

    beams = [best((c,) for c in eligible)]
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
                scored.append(key)
        nxt = best(scored)
        if progress:
            progress(depth, max_new, nxt[0])
        if nxt[0].metrics.d_volatility >= beams[-1][0].metrics.d_volatility:  # no extension lowered volatility
            break
        beams.append(nxt)

    baselines: dict[int, pd.Series] = {}
    finalists = []
    for chosen in beams[-1]:
        if sizing == "slice":
            weights = _slice_weights(props, chosen.tickers, _slice_size(props, len(chosen.tickers), new_budget))
            ratio, volatility = _stats(window, weights, half_life)
        else:
            fit = _fit(frame, base, chosen.tickers, half_life, fit_sizing)
            weights = _expand(fit, held, props, mode)
            ratio, volatility = fit.ratio, fit.volatility
        if backtest:
            result = _backtest(
                returns, held, props, chosen.tickers, mode, baselines, risk_free, half_life,
                WINDOW_WEEKS, BACKTEST_HOLD_WEEKS, sizing, fit_sizing,
            )
        else:
            result = Backtest(NO_BACKTEST, reason="Backtest not requested.")
        finalists.append(
            Finalist(
                chosen.tickers, chosen.metrics, chosen.score, ratio, volatility,
                float(1 / (weights**2).sum()), weights,
                trade_table(weights, props, absolute_band, relative_band), result,
            )
        )
    holdings_ratio = _fit(frame, base, (), half_life, Sizing(fit_sizing.cap)).ratio
    return SearchResult(
        finalists, {d + 1: b[0] for d, b in enumerate(beams[:-1])}, len(beams), holdings_ratio, excluded, trimmed
    )
