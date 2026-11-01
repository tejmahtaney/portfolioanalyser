"""Typed-weight handling: parsing, normalisation, selection bookkeeping and display rounding. No Streamlit here."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import analytics

DROP_BELOW = 1e-6
DISPLAY_DECIMALS = 2


@dataclass(frozen=True)
class Parsed:
    typed: pd.Series
    total: float
    negatives: list[str]
    normalised: pd.Series | None

    @property
    def is_zero(self) -> bool:
        return not self.negatives and not self.total > 0


def parse(raw: pd.Series | Mapping[str, float]) -> Parsed:
    """Blanks count as 0 and values stay float64. Negatives or a zero total leave `normalised` as None."""
    typed = pd.to_numeric(pd.Series(raw), errors="coerce").astype("float64")
    typed = typed.where(np.isfinite(typed), 0.0)
    negatives = [str(t) for t in typed.index[typed < 0]]
    total = float(typed.sum())
    normalised = analytics.normalise_weights(typed) if not negatives and total > 0 else None
    return Parsed(typed, total, negatives, normalised)


def sync(stored: Mapping[str, float], tickers: Iterable[str]) -> dict[str, float]:
    """Keeps the weight of every ticker still selected and gives new ones 0.0."""
    return {t: float(stored.get(t, 0.0)) for t in tickers}


def equal(tickers: Iterable[str]) -> dict[str, float]:
    tickers = list(tickers)
    return {t: 100.0 / len(tickers) for t in tickers}


def largest_remainder(values: pd.Series, decimals: int = DISPLAY_DECIMALS) -> pd.Series:
    """Percentages rounded to `decimals` places that add up to exactly 100. For display only.

    Values are rescaled to sum to 100 first, each is rounded down, and the units still missing go to the largest
    fractional remainders.
    """
    total_units = 100 * 10**decimals
    if len(values) == 0 or not values.sum() > 0:
        return pd.Series(0.0, index=values.index)
    scaled = values.to_numpy(dtype="float64") / values.sum() * total_units
    units = np.floor(scaled).astype("int64")
    short = int(total_units - units.sum())
    if short > 0:
        units[np.argsort(-(scaled - units), kind="stable")[:short]] += 1
    return pd.Series(units / 10**decimals, index=values.index)


def display_table(
    columns: Mapping[str, pd.Series], drop_below: float = DROP_BELOW, exempt: Iterable[str] = ()
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Percent table, each column summing to 100.00, plus how many positions each column dropped.

    Columns are fractions that sum to 1. In a column that is not exempt, a position under `drop_below` is blanked
    and the rest are rounded to sum to 100.00. A row is removed when it is under `drop_below` in every column.
    """
    exempt = set(exempt)
    shown, dropped = {}, {}
    for name, w in columns.items():
        keep = w >= drop_below
        if name in exempt:
            shown[name], dropped[name] = largest_remainder(w), 0
            continue
        dropped[name] = int((~keep).sum())
        shown[name] = largest_remainder(w[keep]).reindex(w.index) if keep.any() else pd.Series(np.nan, index=w.index)
    alive = pd.DataFrame({name: w >= drop_below for name, w in columns.items()}).any(axis=1)
    return pd.DataFrame(shown)[alive], dropped
