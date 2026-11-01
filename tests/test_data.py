import numpy as np
import pandas as pd
import pytest

from portfolio_risk import data


def weekly(values, start="2024-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="W-MON"), dtype=float)


def test_currency_conversion_multiplies_by_aligned_fx():
    sgd = weekly([10, 11, 12, 13])
    fx = weekly([0.70, 0.72, 0.71, 0.73])
    pd.testing.assert_series_equal(data.convert_to_usd(sgd, fx), sgd * fx)


def test_currency_conversion_forward_fills_missing_fx_weeks():
    sgd = weekly([10, 11, 12])
    fx = sgd.copy()
    fx[:] = [0.7, 0.7, 0.8]
    fx = fx.drop(fx.index[1])
    out = data.convert_to_usd(sgd, fx)
    assert out.iloc[1] == pytest.approx(11 * 0.7)
    assert out.iloc[2] == pytest.approx(12 * 0.8)


def test_holiday_gap_is_forward_filled_not_merged():
    a = weekly([100, 101, 102, 103])
    b = a.drop(a.index[2]) * 2
    prices = pd.DataFrame({"A": a, "B": b})
    r = data.build_returns(prices)
    assert len(r) == 4
    assert r["B"].iloc[2] == pytest.approx(0.0)
    assert r["B"].iloc[3] == pytest.approx(103 * 2 / (101 * 2) - 1)


def test_common_window_truncates_to_shortest_history():
    long = weekly(np.linspace(100, 120, 10))
    short = weekly(np.linspace(50, 55, 6), start="2024-01-29")
    r = data.build_returns(pd.DataFrame({"L": long, "S": short}))
    window = data.common_window(r, ["L", "S"])
    assert len(window) == 5
    assert window.index[0] == short.index[1]


def test_roll_jump_flagged_for_futures_only():
    rng = np.random.default_rng(1)
    r = pd.DataFrame(rng.normal(0, 0.01, (200, 2)), columns=["CL=F", "SPY"])
    r.index = pd.date_range("2021-01-04", periods=200, freq="W-MON")
    r.iloc[50] = 0.5
    flags = data.flag_roll_jumps(r, ["CL=F"])
    assert list(flags["week"]) == [r.index[50]]


def test_high_correlation_pairs_detected():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    df = pd.DataFrame({"A": x, "B": x + rng.normal(0, 0.001, 100), "C": rng.normal(size=100)})
    pairs = data.high_correlations(df)
    assert list(zip(pairs["asset_a"], pairs["asset_b"])) == [("A", "B")]


def test_default_currency():
    assert data.default_currency("D05.SI") == "SGD"
    assert data.default_currency("AAPL") == "USD"


def test_common_window_tolerates_duplicate_tickers():
    r = data.build_returns(pd.DataFrame({"A": weekly([1, 2, 3, 4]), "VOO": weekly([5, 6, 7, 8])}))
    assert list(data.common_window(r, ["A", "VOO", "VOO"]).columns) == ["A", "VOO"]


def test_build_returns_does_not_fill_past_a_series_end():
    a = weekly(np.linspace(100, 130, 30))
    b = weekly(np.linspace(50, 60, 20))
    r = data.build_returns(pd.DataFrame({"A": a, "B": b}))
    assert r["B"].iloc[:20].notna().sum() == 19
    assert r["B"].iloc[20:].isna().all()
    assert (r["B"].iloc[20:] != 0).all()
    assert r["A"].iloc[1:].notna().all()
    assert len(data.common_window(r, ["A", "B"])) == 19


def test_interior_gap_is_still_filled_when_the_series_also_ends_early():
    a = weekly(np.linspace(100, 130, 12))
    b = weekly(np.linspace(50, 60, 8)).drop(weekly(range(8)).index[3])
    prices = pd.DataFrame({"A": a, "B": b})
    filled = data.ffill_within_range(prices)
    assert filled["B"].iloc[3] == pytest.approx(prices["B"].iloc[2])
    assert filled["B"].iloc[8:].isna().all()
    assert data.build_returns(prices)["B"].iloc[3] == pytest.approx(0.0)


class FakeYahoo:
    """Stands in for yfinance: batch closes by ticker, currency metadata by ticker."""

    def __init__(self, monkeypatch, tmp_path, currencies, fail_lookups=False):
        self.currencies, self.fail_lookups = currencies, fail_lookups
        self.downloaded = []
        monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(data.yf, "download", self.download)
        monkeypatch.setattr(data.yf, "Ticker", self.ticker)

    def download(self, tickers, **kwargs):
        self.downloaded.extend(tickers)
        idx = pd.date_range("2024-01-01", periods=12, freq="W-MON")
        cols = {t: pd.DataFrame({"Close": np.linspace(10, 12, 12) * (0.7 if t.endswith("=X") else 1)}, index=idx) for t in tickers}
        return pd.concat(cols, axis=1)

    def ticker(self, symbol):
        outer = self

        class T:
            @property
            def fast_info(self):
                if outer.fail_lookups:
                    raise RuntimeError("no fast_info")
                return {"currency": outer.currencies[symbol]}

            def get_history_metadata(self):
                if outer.fail_lookups:
                    raise RuntimeError("no metadata")
                return {"currency": outer.currencies[symbol]}

        return T()


def cached_source(tmp_path, ticker):
    return pd.read_parquet(data._cache_path(ticker))[["currency", "currency_source"]].iloc[0].tolist()


def test_batch_downloaded_si_ticker_quoted_in_usd_is_stored_and_used_as_usd(monkeypatch, tmp_path):
    yahoo = FakeYahoo(monkeypatch, tmp_path, {"J36.SI": "USD", "VOO": "USD"})
    md = data.load_market_data(["J36.SI"])
    assert md.currencies["J36.SI"] == "USD"
    np.testing.assert_allclose(md.prices["J36.SI"].dropna(), np.linspace(10, 12, 12))
    assert cached_source(tmp_path, "J36.SI") == ["USD", "metadata"]


def test_old_cache_with_guessed_sgd_is_corrected_on_next_load(monkeypatch, tmp_path):
    yahoo = FakeYahoo(monkeypatch, tmp_path, {"J36.SI": "USD", "VOO": "USD"})
    idx = pd.date_range("2024-01-01", periods=12, freq="W-MON")
    path = data._cache_path("J36.SI")
    path.parent.mkdir(parents=True)
    pd.DataFrame({"close": np.linspace(10, 12, 12), "currency": "SGD"}, index=idx).to_parquet(path)
    mtime = path.stat().st_mtime_ns

    md = data.load_market_data(["J36.SI"])

    assert "J36.SI" not in yahoo.downloaded  # prices were fresh, only the currency was looked up
    assert md.currencies["J36.SI"] == "USD"
    assert cached_source(tmp_path, "J36.SI") == ["USD", "metadata"]
    assert path.stat().st_mtime_ns == mtime  # correcting the currency does not make old prices look fresh


def test_failed_currency_lookup_falls_back_to_the_suffix_guess(monkeypatch, tmp_path):
    yahoo = FakeYahoo(monkeypatch, tmp_path, {}, fail_lookups=True)
    md = data.load_market_data(["J36.SI"])
    assert md.currencies["J36.SI"] == "SGD"
    assert md.currencies["VOO"] == "USD"
    assert cached_source(tmp_path, "J36.SI") == ["SGD", "guess"]
    assert md.failures == {}

    again = data.load_market_data(["J36.SI"])  # a guess on disk is retried, and failing again does not block
    assert again.currencies["J36.SI"] == "SGD"
    assert not again.returns.empty
