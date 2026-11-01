"""Long-only, fully invested optimisers built on cvxpy.

Minimum variance, maximum Sharpe, maximum return and the frontier are built once per Workspace with cp.Parameter for
the covariance factor, cap, expected returns and target return. They are DPP compliant, so later solves with new
values reuse the compiled problem.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from .config import FRONTIER_POINTS

SOLVER = cp.CLARABEL
# Tight tolerances pin down weights in flat directions of the objective; the defaults are the fallback.
SOLVER_OPTIONS = dict(tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)


class OptimisationError(RuntimeError):
    pass


@dataclass
class Result:
    weights: pd.Series
    warnings: list[str] = field(default_factory=list)


def min_assets(cap: float | None) -> int:
    return 1 if cap is None else math.ceil(round(1 / cap, 9))


def check_feasible(n_assets: int, cap: float | None) -> None:
    need = min_assets(cap)
    if n_assets < need:
        raise OptimisationError(
            f"A {cap:.0%} cap per asset needs at least {need} assets; {n_assets} selected."
        )


def _solve(problem: cp.Problem, label: str) -> None:
    failure = None
    for attempt, options in enumerate((SOLVER_OPTIONS, {})):
        if attempt:
            # Re-solving the same object after an inaccurate result repeats it; a new Problem over the same variables does not.
            problem = cp.Problem(problem.objective, problem.constraints)
        try:
            with warnings.catch_warnings():
                if options:
                    warnings.filterwarnings("ignore", message="Solution may be inaccurate")
                problem.solve(solver=SOLVER, **options)
        except cp.SolverError as exc:
            failure = OptimisationError(f"{label}: solver failed ({exc}).")
            continue
        if problem.status == cp.OPTIMAL:
            return
        failure = OptimisationError(f"{label}: no optimal solution (status: {problem.status}).")
    raise failure


def _series(values, index) -> pd.Series:
    w = pd.Series(np.asarray(values).ravel(), index=index)
    w[w.abs() < 1e-10] = 0.0
    return w / w.sum()


class Workspace:
    """Compiled problems for n assets. Not thread safe: use one per computation."""

    def __init__(self, n: int):
        self.n = n
        self.factor = cp.Parameter((n, n))
        self.cap = cp.Parameter(nonneg=True)
        self.mu = cp.Parameter(n)
        self.excess = cp.Parameter(n)
        self.target = cp.Parameter()
        self._built: dict[str, tuple[cp.Problem, cp.Variable]] = {}

    def set_cov(self, cov: pd.DataFrame) -> None:
        """Stores F with F'F = cov, so w'cov w = ||F w||². The eigen form also handles singular matrices."""
        sym = (cov.to_numpy() + cov.to_numpy().T) / 2
        values, vectors = np.linalg.eigh(sym)
        self.factor.value = np.sqrt(np.clip(values, 0, None))[:, None] * vectors.T

    def set_cap(self, cap: float | None) -> None:
        self.cap.value = 1.0 if cap is None else cap

    def _get(self, name: str):
        if name not in self._built:
            self._built[name] = getattr(self, f"_build_{name}")()
        return self._built[name]

    def _simplex(self, w: cp.Variable) -> list:
        return [cp.sum(w) == 1, w >= 0, w <= self.cap]

    def _build_min_variance(self):
        w = cp.Variable(self.n)
        return cp.Problem(cp.Minimize(cp.sum_squares(self.factor @ w)), self._simplex(w)), w

    def _build_frontier(self):
        w = cp.Variable(self.n)
        constraints = [*self._simplex(w), self.mu @ w >= self.target - 1e-9]
        return cp.Problem(cp.Minimize(cp.sum_squares(self.factor @ w)), constraints), w

    def _build_max_return(self):
        w = cp.Variable(self.n)
        return cp.Problem(cp.Maximize(self.mu @ w), self._simplex(w)), w

    def _build_max_sharpe(self):
        y = cp.Variable(self.n)
        kappa = cp.Variable(nonneg=True)
        constraints = [self.excess @ y == 1, y >= 0, cp.sum(y) == kappa, y <= self.cap * kappa]
        return cp.Problem(cp.Minimize(cp.sum_squares(self.factor @ y)), constraints), y


def _workspace(workspace: Workspace | None, n: int) -> Workspace:
    if workspace is None:
        return Workspace(n)
    if workspace.n != n:
        raise ValueError(f"Workspace is for {workspace.n} assets, not {n}.")
    return workspace


def min_variance(cov: pd.DataFrame, cap: float | None, workspace: Workspace | None = None) -> Result:
    check_feasible(len(cov), cap)
    ws = _workspace(workspace, len(cov))
    ws.set_cov(cov)
    ws.set_cap(cap)
    problem, w = ws._get("min_variance")
    _solve(problem, "Minimum variance")
    return Result(_series(w.value, cov.index))


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, risk_free: float, cap: float | None, workspace: Workspace | None = None
) -> Result:
    """Convex form: minimise y'Σy s.t. (μ-rf)'y = 1, y = w·kappa, then w = y / kappa."""
    check_feasible(len(cov), cap)
    ws = _workspace(workspace, len(cov))
    if max_achievable_return(mu, cap, ws) <= risk_free:
        raise OptimisationError(
            "Maximum Sharpe: no portfolio within the constraints has an expected return above the risk-free rate."
        )
    ws.set_cov(cov)
    ws.set_cap(cap)
    ws.excess.value = (mu - risk_free).to_numpy()
    problem, y = ws._get("max_sharpe")
    _solve(problem, "Maximum Sharpe")
    return Result(_series(y.value, cov.index))


def risk_parity(cov: pd.DataFrame, cap: float | None) -> Result:
    """Cap is ignored in the solve; a breach is reported as a warning.

    Built per call with a constant matrix rather than as a DPP problem: the factor form that DPP needs
    is less reliable alongside the log term (optimal_inaccurate on some backtest windows).
    """
    sym = (cov.to_numpy() + cov.to_numpy().T) / 2
    y = cp.Variable(len(cov))
    _solve(cp.Problem(cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(sym)) - cp.sum(cp.log(y)))), "Risk parity")
    weights = _series(y.value, cov.index)
    warnings = []
    if cap is not None and (weights > cap + 1e-9).any():
        over = ", ".join(weights.index[weights > cap + 1e-9])
        warnings.append(f"Risk parity weights exceed the {cap:.0%} cap for: {over}. The cap is not applied to this portfolio.")
    return Result(weights, warnings)


def max_achievable_return(mu: pd.Series, cap: float | None, workspace: Workspace | None = None) -> float:
    ws = _workspace(workspace, len(mu))
    ws.set_cap(cap)
    ws.mu.value = mu.to_numpy()
    problem, w = ws._get("max_return")
    _solve(problem, "Maximum return")
    return float(mu.to_numpy() @ w.value)


def frontier_weights(
    mu: pd.Series, cov: pd.DataFrame, cap: float | None, points: int = FRONTIER_POINTS,
    workspace: Workspace | None = None,
) -> pd.DataFrame:
    """One row of weights per frontier point. Targets run from the minimum variance return up to the cap-limited maximum."""
    check_feasible(len(cov), cap)
    ws = _workspace(workspace, len(cov))
    low = float(mu @ min_variance(cov, cap, ws).weights)
    high = max_achievable_return(mu, cap, ws)
    ws.mu.value = mu.to_numpy()
    problem, w = ws._get("frontier")
    rows = []
    for target in np.linspace(low, high, points):
        ws.target.value = target
        _solve(problem, f"Frontier point at {target:.2%} return")
        rows.append(_series(w.value, cov.index))
    return pd.DataFrame(rows).reset_index(drop=True)


def efficient_frontier(
    mu: pd.Series, cov: pd.DataFrame, cap: float | None, points: int = FRONTIER_POINTS,
    workspace: Workspace | None = None,
) -> pd.DataFrame:
    weights = frontier_weights(mu, cov, cap, points, workspace)
    return pd.DataFrame(
        {
            "return": [float(mu @ row) for _, row in weights.iterrows()],
            "volatility": [float(np.sqrt(row @ cov @ row)) for _, row in weights.iterrows()],
        }
    )
