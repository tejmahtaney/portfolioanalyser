"""Walk-forward backtest: fit on a training window, hold the weights over the next block."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analytics, optimiser
from .config import BACKTEST_HOLD_WEEKS, BACKTEST_TRAIN_WEEKS, EWMA_HALF_LIFE_WEEKS, PERIODS_PER_YEAR

STRATEGIES = ["Your weights", "Equal weight", "Minimum variance", "Maximum Sharpe", "Risk parity"]


def run_backtest(
    returns: pd.DataFrame,
    user_weights: pd.Series | None,
    risk_free: float,
    cap: float | None,
    cov_method: str,
    return_method: str,
    shrinkage: float,
    train: int = BACKTEST_TRAIN_WEEKS,
    hold: int = BACKTEST_HOLD_WEEKS,
    half_life: float = EWMA_HALF_LIFE_WEEKS,
) -> pd.DataFrame:
    """Out-of-sample weekly returns per strategy, with constant weights within each holding block.

    Pass user_weights=None to leave out "Your weights": the other strategies do not depend on them, so the result
    can be cached and the column added afterwards with add_user_weights.
    """
    if len(returns) <= train:
        raise ValueError(f"The backtest needs more than {train} weeks of data; {len(returns)} available.")
    equal = pd.Series(1 / returns.shape[1], index=returns.columns)
    workspace = optimiser.Workspace(returns.shape[1])
    parts = []
    for start in range(train, len(returns), hold):
        fit = returns.iloc[start - train : start]
        test = returns.iloc[start : start + hold]
        cov = analytics.annual_covariance(fit, cov_method, half_life)
        mu = analytics.expected_returns(fit, return_method, shrinkage)
        try:
            weights = {
                "Equal weight": equal,
                "Minimum variance": optimiser.min_variance(cov, cap, workspace).weights,
                "Maximum Sharpe": optimiser.max_sharpe(mu, cov, risk_free, cap, workspace).weights,
                "Risk parity": optimiser.risk_parity(cov, cap).weights,
            }
            if user_weights is not None:
                weights["Your weights"] = user_weights
        except optimiser.OptimisationError as exc:
            raise optimiser.OptimisationError(
                f"Backtest window ending {fit.index[-1]:%d %b %Y}: {exc}"
            ) from exc
        parts.append(pd.DataFrame({name: test @ w[test.columns] for name, w in weights.items()}))
    oos = pd.concat(parts)
    return oos[[c for c in STRATEGIES if c in oos]]


def add_user_weights(oos: pd.DataFrame, returns: pd.DataFrame, user_weights: pd.Series) -> pd.DataFrame:
    """Adds the fixed-weight strategy, which needs no refitting: its weights are the same in every block."""
    mine = (returns.loc[oos.index, user_weights.index] @ user_weights).rename("Your weights")
    combined = pd.concat([mine, oos.drop(columns="Your weights", errors="ignore")], axis=1)
    return combined[STRATEGIES]


def performance_table(oos: pd.DataFrame, risk_free: float) -> pd.DataFrame:
    rows = {}
    for name in oos:
        r = oos[name]
        ret = r.mean() * PERIODS_PER_YEAR
        vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
        rows[name] = {
            "Annualised return": ret,
            "Volatility": vol,
            "Sharpe ratio": analytics.sharpe_ratio(ret, risk_free, vol),
            "Max drawdown": analytics.max_drawdown(r),
        }
    return pd.DataFrame(rows).T
