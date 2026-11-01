"""Investable universe. Add entries to the static lists below; S&P 500 is scraped."""
from __future__ import annotations

import time
from io import StringIO

import pandas as pd
import requests

from .config import CACHE_DIR, CACHE_MAX_AGE_HOURS

STI = [
    ("D05.SI", "DBS Group"),
    ("O39.SI", "OCBC"),
    ("U11.SI", "United Overseas Bank"),
    ("Z74.SI", "Singtel"),
    ("C6L.SI", "Singapore Airlines"),
    ("9CI.SI", "CapitaLand Investment"),
    ("C38U.SI", "CapitaLand Integrated Commercial Trust"),
    ("A17U.SI", "CapitaLand Ascendas REIT"),
    ("BN4.SI", "Keppel"),
    ("S63.SI", "ST Engineering"),
    ("C09.SI", "City Developments"),
    ("U96.SI", "Sembcorp Industries"),
    ("F34.SI", "Wilmar International"),
    ("Y92.SI", "Thai Beverage"),
    ("G13.SI", "Genting Singapore"),
    ("S68.SI", "Singapore Exchange"),
    ("BS6.SI", "Yangzijiang Shipbuilding"),
    ("5E2.SI", "Seatrium"),
    ("J36.SI", "Jardine Matheson"),
    ("H78.SI", "Hongkong Land"),
    ("C07.SI", "Jardine Cycle & Carriage"),
    ("M44U.SI", "Mapletree Logistics Trust"),
    ("ME8U.SI", "Mapletree Industrial Trust"),
    ("N2IU.SI", "Mapletree Pan Asia Commercial Trust"),
    ("S58.SI", "SATS"),
    ("V03.SI", "Venture Corporation"),
    ("U14.SI", "UOL Group"),
    ("AJBU.SI", "Keppel DC REIT"),
    ("BUOU.SI", "Frasers Logistics & Commercial Trust"),
    ("J69U.SI", "Frasers Centrepoint Trust"),
]

INDEX_FUNDS = [
    ("SPY", "SPDR S&P 500 ETF"),
    ("VOO", "Vanguard S&P 500 ETF"),
    ("IVV", "iShares Core S&P 500 ETF"),
    ("VTI", "Vanguard Total Stock Market ETF"),
    ("QQQ", "Invesco QQQ Trust"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF"),
    ("IWM", "iShares Russell 2000 ETF"),
    ("VEA", "Vanguard FTSE Developed Markets ETF"),
    ("VWO", "Vanguard FTSE Emerging Markets ETF"),
    ("VXUS", "Vanguard Total International Stock ETF"),
    ("BND", "Vanguard Total Bond Market ETF"),
    ("ES3.SI", "SPDR Straits Times Index ETF"),
    ("G3B.SI", "Nikko AM Singapore STI ETF"),
]

FUTURES = [
    ("GC=F", "Gold futures"),
    ("SI=F", "Silver futures"),
    ("CL=F", "WTI crude oil futures"),
    ("BZ=F", "Brent crude oil futures"),
    ("NG=F", "Natural gas futures"),
    ("HG=F", "Copper futures"),
    ("PL=F", "Platinum futures"),
]

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE = CACHE_DIR / "sp500.parquet"

COLUMNS = ["ticker", "name", "asset_class"]


def yahoo_ticker(symbol: str) -> str:
    return symbol.replace(".", "-")


def _static() -> pd.DataFrame:
    rows = (
        [(t, n, "STI constituent") for t, n in STI]
        + [(t, n, "Index fund") for t, n in INDEX_FUNDS]
        + [(t, n, "Commodity future") for t, n in FUTURES]
    )
    return pd.DataFrame(rows, columns=COLUMNS)


def _cache_fresh(path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < CACHE_MAX_AGE_HOURS * 3600


def load_sp500() -> tuple[pd.DataFrame, str | None]:
    """Returns (constituents, warning). Falls back to a stale cache if the scrape fails."""
    if _cache_fresh(SP500_CACHE):
        return pd.read_parquet(SP500_CACHE), None
    try:
        resp = requests.get(SP500_URL, headers={"User-Agent": "portfolio-risk/0.1"}, timeout=15)
        resp.raise_for_status()
        table = pd.read_html(StringIO(resp.text), match="Symbol")[0]
        df = pd.DataFrame(
            {
                "ticker": table["Symbol"].astype(str).map(yahoo_ticker),
                "name": table["Security"].astype(str),
                "asset_class": "S&P 500 constituent",
            }
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SP500_CACHE)
        return df, None
    except Exception as exc:
        if SP500_CACHE.exists():
            return pd.read_parquet(SP500_CACHE), f"S&P 500 list could not be refreshed ({exc}); using the cached copy."
        return pd.DataFrame(columns=COLUMNS), f"S&P 500 list could not be loaded ({exc}); it is omitted from the universe."


def load_universe() -> tuple[pd.DataFrame, str | None]:
    sp500, warning = load_sp500()
    df = pd.concat([sp500, _static()], ignore_index=True).drop_duplicates("ticker", keep="last")
    return df.reset_index(drop=True), warning
