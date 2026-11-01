import pandas as pd

from portfolio_risk import backtest


def test_backtest_covers_only_out_of_sample_weeks(returns):
    cols = ["A", "B", "C", "D", "E"]
    r = returns[cols]
    w = pd.Series(0.2, index=cols)
    oos = backtest.run_backtest(r, w, 0.0, 0.4, "ledoit_wolf", "shrunk", 0.5, train=156, hold=13)
    assert list(oos.columns) == backtest.STRATEGIES
    assert len(oos) == len(r) - 156
    assert oos.index[0] == r.index[156]
    table = backtest.performance_table(oos, 0.0)
    assert list(table.columns) == ["Annualised return", "Volatility", "Sharpe ratio", "Max drawdown"]


def test_backtest_passes_method_and_half_life_to_every_refit(returns, monkeypatch):
    cols = ["A", "B", "C", "D", "E"]
    r = returns[cols]
    seen = []
    original = backtest.analytics.annual_covariance

    def spy(fit, method="ledoit_wolf", half_life=52):
        seen.append((method, half_life, len(fit)))
        return original(fit, method, half_life)

    monkeypatch.setattr(backtest.analytics, "annual_covariance", spy)
    backtest.run_backtest(r, None, 0.0, 0.4, "ewma", "shrunk", 0.5, train=156, hold=13, half_life=26)
    assert len(seen) == len(range(156, len(r), 13))
    assert {(m, h) for m, h, _ in seen} == {("ewma", 26)}
