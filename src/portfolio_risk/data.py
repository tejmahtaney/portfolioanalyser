"""Price download, caching, currency conversion and weekly return construction."""
from __future__ import annotations

import os
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
    HISTORY_PERIODS,
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


def snap_to_monday(close: pd.Series) -> pd.Series:
    """Dates every weekly bar on the Monday of its week; if two bars share a week, the later one wins.

    Yahoo can date the current partial week on its last trading day, and exchanges disagree, so raw bar dates
    cannot be trusted to line up across tickers.
    """
    close = close.sort_index()
    index = pd.DatetimeIndex(close.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    close = close.set_axis(index - pd.to_timedelta(index.day_of_week, unit="D"))
    return close[~close.index.duplicated(keep="last")]


def check_period(period: str) -> str:
    if period not in HISTORY_PERIODS:
        raise ValueError(f"Unknown history period {period!r}; use one of {', '.join(HISTORY_PERIODS)}.")
    return period


def _years(period: str) -> int:
    return int(period.removesuffix("y"))


def _period_start(period: str) -> pd.Timestamp:
    """Monday of the week holding the date one period ago; bars before it are outside the period."""
    start = pd.Timestamp.now().normalize() - pd.DateOffset(years=_years(period))
    return start - pd.Timedelta(days=start.day_of_week)


def _download(ticker: str, period: str = HISTORY_PERIOD) -> tuple[pd.Series, str | None]:
    tk = yf.Ticker(ticker)
    hist = tk.history(period=period, interval="1wk", auto_adjust=True)
    if hist.empty or "Close" not in hist:
        raise ValueError("no price data returned")
    close = snap_to_monday(hist["Close"].dropna())
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


def _read_cache(
    ticker: str, period: str = HISTORY_PERIOD, ignore_period: bool = False
) -> tuple[pd.Series, str, bool, str] | None:
    """(close prices, currency, is_fresh, currency_source) from the parquet cache, or None when there is no file.

    currency_source is "metadata" or "guess". Files written before the column existed count as "guess".

    One file per ticker, tagged with the period it was downloaded for. Files written before the tag existed hold
    HISTORY_PERIOD. A file covering less than the requested period counts as no file; one covering more is trimmed
    to the request. ignore_period skips both and returns the file as stored.
    """
    path = _cache_path(ticker)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    stored = frame["period"].iloc[0] if "period" in frame else HISTORY_PERIOD
    if not ignore_period and stored != period:
        if stored not in HISTORY_PERIODS or _years(stored) < _years(period):
            return None
        frame = frame[frame.index >= _period_start(period)]
        if frame.empty:
            return None
    fresh = (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600
    source = frame["currency_source"].iloc[0] if "currency_source" in frame else "guess"
    return snap_to_monday(frame["close"]), frame["currency"].iloc[0], fresh, "metadata" if source == "metadata" else "guess"


def _write_cache(ticker: str, close: pd.Series, currency: str, source: str, period: str = HISTORY_PERIOD) -> None:
    path = _cache_path(ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"close": close, "currency": currency, "currency_source": source, "period": period}).to_parquet(path)


def _update_currency(ticker: str, currency: str) -> None:
    """Rewrites only the currency of a cache file, keeping its prices and its age so old prices do not look fresh."""
    path = _cache_path(ticker)
    stamp = path.stat()
    frame = pd.read_parquet(path)
    frame["currency"], frame["currency_source"] = currency, "metadata"
    frame.to_parquet(path)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))


def lookup_currency(ticker: str) -> str | None:
    """Quote currency from Yahoo metadata, or None when neither source answers."""
    tk = yf.Ticker(ticker)
    for read in (lambda: tk.fast_info["currency"], lambda: tk.get_history_metadata().get("currency")):
        try:
            currency = read()
        except Exception:
            continue
        if isinstance(currency, str) and currency:
            return currency
    return None


def fetch_prices(ticker: str, period: str = HISTORY_PERIOD) -> tuple[pd.Series, str, bool]:
    """Returns (close prices, currency, used_stale_cache). Raises if there is no usable data."""
    cached = _read_cache(ticker, period)
    if cached is not None and cached[2]:
        return cached[0], cached[1], False
    try:
        close, currency = _download(ticker, period)
    except Exception:
        if cached is not None:
            return cached[0], cached[1], True
        raise
    source = "metadata" if currency else "guess"
    currency = currency or default_currency(ticker)
    _write_cache(ticker, close, currency, source, period)
    return close, currency, False


def _batch_close(frame: pd.DataFrame, ticker: str, single: bool) -> pd.Series | None:
    try:
        close = frame["Close"] if single and not isinstance(frame.columns, pd.MultiIndex) else frame[ticker]["Close"]
    except KeyError:
        return None
    close = close.dropna()
    if close.empty:
        return None
    return snap_to_monday(close).astype(float)


def download_batch(tickers: list[str], period: str = HISTORY_PERIOD) -> dict[str, pd.Series]:
    """One yf.download call for every ticker. Tickers that come back empty are left out."""
    if not tickers:
        return {}
    frame = yf.download(
        tickers, period=period, interval="1wk", auto_adjust=True,
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


def ffill_within_range(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills each column up to its own last valid date. A trailing gap stays NaN."""
    ended = prices.notna().iloc[::-1].cummax().iloc[::-1]
    return prices.ffill().where(ended)


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Forward-fills holiday gaps on a shared calendar, then takes percentage changes.

    Rows are deliberately not dropped first: that would turn holiday weeks into two-week returns.
    A series is filled only up to its own last valid date, so one that ended early has NaN returns after it
    (common_window then ends the shared window there) instead of flat prices and zero returns.
    """
    filled = ffill_within_range(prices.sort_index())
    return filled.pct_change(fill_method=None)


def common_window(returns: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    unique = list(dict.fromkeys(tickers))
    return returns.dropna(subset=unique)[unique]


def _known_currency(ticker: str) -> str:
    cached = _read_cache(ticker, ignore_period=True)
    return cached[1] if cached is not None else default_currency(ticker)


def _collect(
    wanted: list[str], max_workers: int, note: Callable[[str], None], period: str = HISTORY_PERIOD
) -> tuple[dict[str, tuple[pd.Series, str]], dict[str, str], list[str]]:
    """Fresh disk cache first, then one batched download for the rest, then per-ticker calls for the leftovers.

    Currencies not yet confirmed by Yahoo metadata (batch downloads, cache files marked "guess") are looked up
    in the same thread pool as the per-ticker calls.
    """
    results: dict[str, tuple[pd.Series, str]] = {}
    failures: dict[str, str] = {}
    stale: list[str] = []
    todo = []
    unconfirmed: dict[str, pd.Series] = {}  # fresh cache files whose currency is only a guess
    for t in wanted:
        cached = _read_cache(t, period)
        if cached is not None and cached[2]:
            results[t] = (cached[0], cached[1])
            if cached[3] != "metadata" and not t.endswith("=X"):
                unconfirmed[t] = cached[0]
        else:
            todo.append(t)
    note(f"{len(results)} of {len(wanted)} tickers read from the disk cache.")
    if not todo and not unconfirmed:
        return results, failures, stale

    batch: dict[str, pd.Series] = {}
    if todo:
        note(f"Downloading {len(todo)} tickers in one batch…")
        try:
            batch = download_batch(todo, period)
        except Exception as exc:
            note(f"Batch download failed ({exc}); falling back to one call per ticker.")
        note(f"Batch returned {len(batch)} of {len(todo)} tickers.")
    missing = [t for t in todo if t not in batch]

    known: dict[str, str] = {}  # batch tickers with a metadata currency already on disk
    for t in batch:
        cached = _read_cache(t, ignore_period=True)
        if cached is not None and cached[3] == "metadata":
            known[t] = cached[1]
    to_look_up = [*unconfirmed, *(t for t in batch if t not in known and not t.endswith("=X"))]

    def work(t: str):
        try:
            return t, fetch_prices(t, period), None
        except Exception as exc:
            return t, None, str(exc) or type(exc).__name__

    def look_up(t: str):
        try:
            return lookup_currency(t)
        except Exception:
            return None

    if missing:
        note("Per-ticker fallback for: " + ", ".join(missing))
    if to_look_up:
        note("Looking up the currency of: " + ", ".join(to_look_up))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending_lookups = pool.map(look_up, to_look_up)
        pending_fetches = pool.map(work, missing)
        lookups, fetched = list(pending_lookups), list(pending_fetches)
    looked_up = dict(zip(to_look_up, lookups))

    for t, close in batch.items():
        if t in known:
            currency, source = known[t], "metadata"
        elif looked_up.get(t):
            currency, source = looked_up[t], "metadata"
        else:
            currency, source = default_currency(t), "guess"
        _write_cache(t, close, currency, source, period)
        results[t] = (close, currency)
    for t, close in unconfirmed.items():
        if looked_up.get(t):
            _update_currency(t, looked_up[t])
            results[t] = (close, looked_up[t])
    for t, res, err in fetched:
        if err is not None:
            failures[t] = err
            continue
        series, currency, was_stale = res
        results[t] = (series, currency)
        if was_stale:
            stale.append(t)
    return results, failures, stale


def load_market_data(
    tickers: list[str],
    max_workers: int = 4,
    progress: Callable[[str], None] | None = None,
    period: str = HISTORY_PERIOD,
) -> MarketData:
    check_period(period)
    wanted = list(dict.fromkeys([*tickers, BENCHMARK]))
    log: list[str] = []

    def note(message: str) -> None:
        log.append(message)
        if progress is not None:
            progress(message)

    # Likely FX pairs ride along in the same batch instead of costing a second round trip.
    fx_guess = sorted({fx_ticker(c) for c in map(_known_currency, wanted) if c != "USD"})
    results, failures, stale = _collect(list(dict.fromkeys([*wanted, *fx_guess])), max_workers, note, period)
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
                series, _, was_stale = fetch_prices(fx_ticker(ccy), period)
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
    prices = ffill_within_range(pd.DataFrame(usd).sort_index())
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
