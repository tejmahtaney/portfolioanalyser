import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n, k = 300, 6
    market = rng.normal(0.003, 0.02, n)
    betas = np.array([1.0, 0.6, 1.3, 0.2, 0.9, 0.0])
    drift = np.array([0.0020, 0.0032, 0.0025, 0.0018, 0.0038, 0.0015])
    noise = rng.normal(0, 0.01, (n, k))
    data = market[:, None] * betas + drift + noise
    idx = pd.date_range("2020-01-06", periods=n, freq="W-MON")
    return pd.DataFrame(data, index=idx, columns=["VOO", "A", "B", "C", "D", "E"])
