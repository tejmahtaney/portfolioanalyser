"""Price download, caching, currency conversion and weekly return construction."""
from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    BENCHMARK,
    CACHE_DIR,
    CACHE_MAX_AGE_HOURS,
    HIGH_CORRELATION,
    HISTORY_PERIOD,
    ROLL_JUMP_SIGMAS,
)


@dataclass
class MarketData:
    returns: pd.DataFrame
    currencies: dict[str, str] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    log: list[str] = field(default_factory=list)


def default_currency(ticker: str) -> str:
    return "SGD" if ticker.endswith(".SI") else "USD"


def _cache_path(ticker: str):
    return CACHE_DIR / "prices" / f"{ticker.replace('=', '_').replace('/', '_')}.parquet"


def _download(ticker: str) -> tuple[pd.Series, str | None]:
    tk = yf.Ticker(ticker)
    hist = tk.history(period=HISTORY_PERIOD, interval="1wk", auto_adjust=True)
    if hist.empty or "Close" not in hist:
        raise ValueError("no price data returned")
    close = hist["Close"].dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close[~close.index.duplicated(keep="last")]
    currency = None
    try:
        currency = tk.get_history_metadata().get("currency")
    except Exception:
        pass
    if not currency:
        try:
            currency = tk.fast_info["currency"]
        except Exception:
            pass
    return close.astype(float), currency


def _read_cache(ticker: str) -> tuple[pd.Series, str, bool] | None:
    """(close prices, currency, is_fresh) from the parquet cache, or None when there is no file."""
    path = _cache_path(ticker)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    fresh = (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600
    return frame["close"], frame["currency"].iloc[0], fresh


def _write_cache(ticker: str, close: pd.Series, currency: str) -> None:
    path = _cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"close": close, "currency": currency}).to_parquet(path)


def fetch_prices(ticker: str) -> tuple[pd.Series, str, bool]:
    """Returns (close prices, currency, used_stale_cache). Raises if there is no usable data."""
    cached = _read_cache(ticker)
    if cached is not None and cached[2]:
        return cached[0], cached[1], False
    try:
        close, currency = _download(ticker)
    except Exception:
        if cached is not None:
            return cached[0], cached[1], True
        raise
    currency = currency or default_currency(ticker)
    _write_cache(ticker, close, currency)
    return close, currency, False


def _batch_close(frame: pd.DataFrame, ticker: str, single: bool) -> pd.Series | None:
    try:
        close = frame["Close"] if single and not isinstance(frame.columns, pd.MultiIndex) else frame[ticker]["Close"]
    except KeyError:
        return None
    close = close.dropna()
    if close.empty:
        return None
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    return close[~close.index.duplicated(keep="last")].astype(float)


def download_batch(tickers: list[str]) -> dict[str, pd.Series]:
    """One yf.download call for every ticker. Tickers that come back empty are left out."""
    if not tickers:
        return {}
    frame = yf.download(
        tickers, period=HISTORY_PERIOD, interval="1wk", auto_adjust=True,
        group_by="ticker", threads=True, progress=False,
    )
    if frame is None or frame.empty:
        return {}
    found = {t: _batch_close(frame, t, len(tickers) == 1) for t in tickers}
    return {t: s for t, s in found.items() if s is not None}


def fx_ticker(currency: str) -> str:
    return f"{currency}USD=X"


def convert_to_usd(prices: pd.Series, fx: pd.Series) -> pd.Series:
    """Multiplies a local-currency series by the FX rate aligned to its dates."""
    calendar = fx.index.union(prices.index)
    aligned = fx.reindex(calendar).ffill().reindex(prices.index)
    return prices * aligned


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills holiday gaps on a shared calendar, then takes percentage changes.

    Rows are deliberately not dropped first: that would turn holiday weeks into two-week returns.
    """
    filled = prices.sort_index().ffill()
    return filled.pct_change(fill_method=None)


def common_window(returns: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    unique = list(dict.fromkeys(tickers))
    return returns.dropna(subset=unique)[unique]


def _known_currency(ticker: str) -> str:
    cached = _read_cache(ticker)
    return cached[1] if cached is not None else default_currency(ticker)


def _collect(
    wanted: list[str], max_workers: int, note: Callable[[str], None]
) -> tuple[dict[str, tuple[pd.Series, str]], dict[str, str], list[str]]:
    """Fresh disk cache first, then one batched download for the rest, then per-ticker calls for the leftovers."""
    results: dict[str, tuple[pd.Series, str]] = {}
    failures: dict[str, str] = {}
    stale: list[str] = []
    todo = []
    for t in wanted:
        cached = _read_cache(t)
        if cached is not None and cached[2]:
            results[t] = (cached[0], cached[1])
        else:
            todo.append(t)
    note(f"{len(results)} of {len(wanted)} tickers read from the disk cache.")
    if not todo:
        return results, failures, stale

    note(f"Downloading {len(todo)} tickers in one batch…")
    try:
        batch = download_batch(todo)
    except Exception as exc:
        note(f"Batch download failed ({exc}); falling back to one call per ticker.")
        batch = {}
    for t, close in batch.items():
        currency = _known_currency(t)
        _write_cache(t, close, currency)
        results[t] = (close, currency)
    missing = [t for t in todo if t not in batch]
    note(f"Batch returned {len(batch)} of {len(todo)} tickers.")

    def work(t: str):
        try:
            return t, fetch_prices(t), None
        except Exception as exc:
            return t, None, str(exc) or type(exc).__name__

    if missing:
        note("Per-ticker fallback for: " + ", ".join(missing))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for t, res, err in pool.map(work, missing):
                if err is not None:
                    failures[t] = err
                    continue
                series, currency, was_stale = res
                results[t] = (series, currency)
                if was_stale:
                    stale.append(t)
    return results, failures, stale


def load_market_data(
    tickers: list[str], max_workers: int = 4, progress: Callable[[str], None] | None = None
) -> MarketData:
    wanted = list(dict.fromkeys([*tickers, BENCHMARK]))
    log: list[str] = []

    def note(message: str) -> None:
        log.append(message)
        if progress is not None:
            progress(message)

    # Likely FX pairs ride along in the same batch instead of costing a second round trip.
    fx_guess = sorted({fx_ticker(c) for c in map(_known_currency, wanted) if c != "USD"})
    results, failures, stale = _collect(list(dict.fromkeys([*wanted, *fx_guess])), max_workers, note)
    fx_fetched = {t: results.pop(t) for t in fx_guess if t in results}
    for t in fx_guess:
        if t in failures:
            failures.pop(t)

    currencies = {t: ccy for t, (_, ccy) in results.items()}
    fx: dict[str, pd.Series] = {}
    for ccy in sorted({c for c in currencies.values() if c != "USD"}):
        try:
            if fx_ticker(ccy) in fx_fetched:
                fx[ccy] = fx_fetched[fx_ticker(ccy)][0]
            else:
                series, _, was_stale = fetch_prices(fx_ticker(ccy))
                fx[ccy] = series
                if was_stale:
                    stale.append(fx_ticker(ccy))
        except Exception as exc:
            for t in [t for t, c in currencies.items() if c == ccy]:
                failures[t] = f"FX rate {fx_ticker(ccy)} unavailable ({exc})"
                results.pop(t)

    usd = {
        t: s if currencies[t] == "USD" else convert_to_usd(s, fx[currencies[t]])
        for t, (s, _) in results.items()
        if t in currencies and (currencies[t] == "USD" or currencies[t] in fx)
    }
    if not usd:
        return MarketData(pd.DataFrame(), currencies, failures, stale, log=log)
    prices = pd.DataFrame(usd).sort_index().ffill()
    return MarketData(build_returns(prices), {t: currencies[t] for t in usd}, failures, stale, prices, log)


def flag_roll_jumps(returns: pd.DataFrame, futures: list[str]) -> pd.DataFrame:
    """Weekly returns beyond ROLL_JUMP_SIGMAS standard deviations in futures series."""
    rows = []
    for t in futures:
        if t not in returns:
            continue
        r = returns[t].dropna()
        sd = r.std()
        if not sd or np.isnan(sd):
            continue
        z = (r - r.mean()) / sd
        for date in z[z.abs() > ROLL_JUMP_SIGMAS].index:
            rows.append({"ticker": t, "week": date, "return": r[date], "z_score": z[date]})
    return pd.DataFrame(rows, columns=["ticker", "week", "return", "z_score"])


def high_correlations(returns: pd.DataFrame, threshold: float = HIGH_CORRELATION) -> pd.DataFrame:
    corr = returns.corr()
    rows = [
        {"asset_a": a, "asset_b": b, "correlation": corr.loc[a, b]}
        for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1 :]
        if corr.loc[a, b] > threshold
    ]
    return pd.DataFrame(rows, columns=["asset_a", "asset_b", "correlation"])
