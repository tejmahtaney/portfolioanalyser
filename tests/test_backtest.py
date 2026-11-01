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
