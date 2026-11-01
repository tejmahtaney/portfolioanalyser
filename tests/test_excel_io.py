from io import BytesIO

import pandas as pd
import pytest

from portfolio_risk import excel_io


def sample():
    idx = ["A", "B", "C"]
    portfolio = pd.DataFrame(
        {
            "Name": ["Alpha", "Beta", "Gamma"],
            "Weight": [0.5, 0.3, 0.2],
            "Risk contribution": [0.6, 0.3, 0.1],
            "Expected return": [0.1, 0.08, 0.05],
            "Volatility": [0.2, 0.15, 0.1],
            "Beta to VOO": [1.1, 0.9, 0.4],
        },
        index=idx,
    )
    weights = pd.DataFrame({"Current": [0.5, 0.3, 0.2], "Minimum variance": [0.1, 0.3, 0.6]}, index=idx)
    stats = pd.DataFrame(
        {"Expected return": [0.09, 0.06], "Volatility": [0.12, 0.08], "Sharpe ratio": [0.4, 0.3], "Max drawdown": [-0.2, -0.1]},
        index=["Current", "Minimum variance"],
    )
    return portfolio, weights, stats


def test_round_trip_restores_weights_in_percent():
    portfolio, weights, stats = sample()
    blob = excel_io.export_workbook(portfolio, {"Risk-free rate": "4.00%"}, weights, stats)
    restored = excel_io.import_weights(BytesIO(blob))
    assert list(restored.index) == ["A", "B", "C"]
    assert restored.tolist() == pytest.approx([50, 30, 20])


def test_export_contains_all_recommendation_sheets():
    portfolio, weights, stats = sample()
    book = pd.read_excel(BytesIO(excel_io.export_workbook(portfolio, {}, weights, stats)), sheet_name=None)
    assert {"Portfolio", "Recommended weights", "Recommended statistics", "Settings"} <= set(book)
    assert book["Recommended weights"]["Minimum variance"].tolist() == pytest.approx([10, 30, 60])


def test_export_treats_any_beta_column_as_unscaled():
    portfolio, _, _ = sample()
    portfolio = portfolio.rename(columns={"Beta to VOO": "Beta to SPY"})
    portfolio["Beta to SPY"] = [1.123456, 0.9, 0.4]
    book = pd.read_excel(BytesIO(excel_io.export_workbook(portfolio, {})), sheet_name="Portfolio")
    assert "Beta to SPY" in book.columns
    assert "Beta to SPY (%)" not in book.columns
    assert book["Beta to SPY"].tolist() == pytest.approx([1.1235, 0.9, 0.4])


def test_export_without_recommendations_still_imports():
    portfolio, _, _ = sample()
    blob = excel_io.export_workbook(portfolio, {})
    assert excel_io.import_weights(BytesIO(blob)).sum() == pytest.approx(100)


def test_import_rejects_bad_files():
    buf = BytesIO()
    pd.DataFrame({"Ticker": ["A", "A"], "Weight (%)": [1, 2]}).to_excel(buf, sheet_name="Portfolio", index=False)
    with pytest.raises(ValueError, match="more than once"):
        excel_io.import_weights(BytesIO(buf.getvalue()))
    buf = BytesIO()
    pd.DataFrame({"x": [1]}).to_excel(buf, sheet_name="Other", index=False)
    with pytest.raises(ValueError, match="no 'Portfolio' sheet"):
        excel_io.import_weights(BytesIO(buf.getvalue()))
