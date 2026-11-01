import numpy as np
import pandas as pd
import pytest

from portfolio_risk import weights as wt


def test_33_33_33_34_normalise_to_exactly_one():
    parsed = wt.parse(pd.Series({"A": 33.33, "B": 33.33, "C": 33.34}))
    assert parsed.normalised.sum() == pytest.approx(1.0, abs=1e-12)
    assert parsed.total == pytest.approx(100.0, abs=1e-12)


def test_tiny_and_large_weights_are_kept_unrounded():
    parsed = wt.parse(pd.Series({"A": 0.001, "B": 99.999}))
    assert parsed.typed["A"] == 0.001
    assert parsed.typed["B"] == 99.999
    assert parsed.typed.dtype == np.float64
    assert parsed.normalised["A"] == pytest.approx(1e-5, rel=1e-12)
    assert parsed.normalised["B"] == pytest.approx(0.99999, rel=1e-12)


@pytest.mark.parametrize("typed", [0.5, 12.75, 0.001, 250.0])
def test_decimal_and_over_100_values_accepted(typed):
    parsed = wt.parse(pd.Series({"A": typed, "B": 1.0}))
    assert parsed.typed["A"] == typed
    assert parsed.normalised is not None
    assert parsed.normalised.sum() == pytest.approx(1.0, abs=1e-12)


def test_blanks_count_as_zero():
    parsed = wt.parse(pd.Series({"A": None, "B": 5.0, "C": np.nan}))
    assert list(parsed.typed) == [0.0, 5.0, 0.0]
    assert parsed.normalised["B"] == 1.0


def test_negatives_are_rejected_and_named():
    parsed = wt.parse(pd.Series({"A": -1.0, "B": 5.0}))
    assert parsed.negatives == ["A"]
    assert parsed.normalised is None


@pytest.mark.parametrize("raw", [{"A": 0.0, "B": 0.0}, {"A": None, "B": np.nan}])
def test_zero_total_has_nothing_to_calculate(raw):
    parsed = wt.parse(pd.Series(raw))
    assert parsed.is_zero
    assert parsed.normalised is None


def test_largest_remainder_known_case():
    out = wt.largest_remainder(pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}) * 100)
    assert sorted(out) == [33.33, 33.33, 33.34]


@pytest.mark.parametrize("seed", range(200))
def test_largest_remainder_always_sums_to_100(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 40))
    raw = rng.dirichlet(np.full(n, rng.choice([0.05, 0.5, 5.0]))) * 100
    out = wt.largest_remainder(pd.Series(raw))
    assert round(out.sum() * 100) == 10_000
    assert out.sum() == pytest.approx(100.0, abs=1e-9)
    assert (out - raw).abs().max() <= 0.01 + 1e-9


def test_largest_remainder_handles_exact_values_and_unnormalised_input():
    assert list(wt.largest_remainder(pd.Series([25.0, 25.0, 50.0]))) == [25.0, 25.0, 50.0]
    assert wt.largest_remainder(pd.Series([2.0, 6.0])).sum() == pytest.approx(100.0)


def test_adding_a_ticker_keeps_every_existing_weight():
    stored = {"A": 12.75, "B": 0.001}
    out = wt.sync(stored, ["A", "B", "C"])
    assert out == {"A": 12.75, "B": 0.001, "C": 0.0}


def test_removing_a_ticker_keeps_the_rest():
    stored = {"A": 12.75, "B": 0.001, "C": 30.0}
    assert wt.sync(stored, ["A", "C"]) == {"A": 12.75, "C": 30.0}


def test_equal_weights_are_not_rounded():
    out = wt.equal(["A", "B", "C"])
    assert out["A"] == 100 / 3
    assert sum(out.values()) == pytest.approx(100.0, abs=1e-12)


def test_display_table_drops_dust_and_still_sums_to_100():
    w = pd.Series({"A": 0.6, "B": 0.4 - 5e-7, "C": 5e-7})
    other = pd.Series({"A": 0.2, "B": 0.3, "C": 0.5})
    table, dropped = wt.display_table({"Opt": w, "Other": other})
    assert dropped == {"Opt": 1, "Other": 0}
    assert np.isnan(table.loc["C", "Opt"])
    assert round(table["Opt"].sum() * 100) == 10_000
    assert round(table["Other"].sum() * 100) == 10_000


def test_display_table_removes_rows_that_are_dust_everywhere_and_exempts_current():
    cur = pd.Series({"A": 0.5, "B": 0.5, "C": 0.0})
    opt = pd.Series({"A": 1.0, "B": 0.0, "C": 0.0})
    table, dropped = wt.display_table({"Current": cur, "Opt": opt}, exempt=["Current"])
    assert list(table.index) == ["A", "B"]
    assert dropped == {"Current": 0, "Opt": 2}
    assert round(table["Current"].sum() * 100) == 10_000
