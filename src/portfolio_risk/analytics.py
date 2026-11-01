"""Portfolio statistics. Inputs are weekly percentage-change returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import PERIODS_PER_YEAR


def normalise_weights(weights: pd.Series) -> pd.Series:
    if (weights < 0).any():
        raise ValueError("Weights must not be negative.")
    total = weights.sum()
    if not total > 0:
        raise ValueError("Weights must sum to more than zero.")
    return weights / total


def annual_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    if method == "ledoit_wolf":
        weekly = pd.DataFrame(
            LedoitWolf().fit(returns.to_numpy()).covariance_, index=returns.columns, columns=returns.columns
        )
    elif method == "sample":
        weekly = returns.cov()
    else:
        raise ValueError(f"Unknown covariance method: {method}")
    return weekly * PERIODS_PER_YEAR


def expected_returns(returns: pd.DataFrame, method: str = "shrunk", intensity: float = 0.5) -> pd.Series:
    mean = returns.mean() * PERIODS_PER_YEAR
    if method == "raw":
        return mean
    if method == "shrunk":
        return (1 - intensity) * mean + intensity * mean.mean()
    raise ValueError(f"Unknown expected return method: {method}")


def betas(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """cov(asset, benchmark) / var(benchmark), on the same weekly returns."""
    aligned = returns.loc[benchmark.index.intersection(returns.index)]
    bench = benchmark.loc[aligned.index]
    var = bench.var()
    return aligned.apply(lambda col: col.cov(bench)) / var


def portfolio_variance(w: pd.Series, cov: pd.DataFrame) -> float:
    return float(w.to_numpy() @ cov.to_numpy() @ w.to_numpy())


def portfolio_volatility(w: pd.Series, cov: pd.DataFrame) -> float:
    return float(np.sqrt(portfolio_variance(w, cov)))


def risk_contributions(w: pd.Series, cov: pd.DataFrame) -> pd.Series:
    marginal = pd.Series(cov.to_numpy() @ w.to_numpy(), index=w.index)
    return w * marginal / portfolio_variance(w, cov)


def sharpe_ratio(expected_return: float, risk_free: float, volatility: float) -> float:
    return (expected_return - risk_free) / volatility if volatility > 0 else float("nan")


def portfolio_returns(returns: pd.DataFrame, w: pd.Series) -> pd.Series:
    return returns[w.index] @ w


def max_drawdown(series: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative growth, as a negative number."""
    wealth = np.concatenate([[1.0], (1 + series).cumprod().to_numpy()])
    return float((wealth / np.maximum.accumulate(wealth) - 1).min())


def historical_var(series: pd.Series, level: float = 0.95) -> float:
    """Weekly VaR, expressed as a positive loss."""
    return float(-series.quantile(1 - level))


def summarise(
    w: pd.Series, mu: pd.Series, cov: pd.DataFrame, returns: pd.DataFrame, risk_free: float
) -> dict[str, float]:
    exp_ret = float(w @ mu)
    vol = portfolio_volatility(w, cov)
    return {
        "Expected return": exp_ret,
        "Volatility": vol,
        "Sharpe ratio": sharpe_ratio(exp_ret, risk_free, vol),
        "Max drawdown": max_drawdown(portfolio_returns(returns, w)),
    }
