"""Excel export of the portfolio and recommendations, and import of weights from that export."""
from __future__ import annotations

from io import BytesIO

import pandas as pd

PORTFOLIO_SHEET = "Portfolio"
WEIGHT_COLUMN = "Weight (%)"


def _percent(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame * 100).round(4)


def export_workbook(
    portfolio: pd.DataFrame,
    settings: dict[str, str],
    weights: pd.DataFrame | None = None,
    statistics: pd.DataFrame | None = None,
) -> bytes:
    """`portfolio` is indexed by ticker, with a Name column and fractional columns; weights/statistics are fractions."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        table = portfolio.copy()
        names = table.pop("Name")
        out = _percent(table)
        out.columns = [f"{c} (%)" if c != "Beta to VOO" else c for c in out.columns]
        if "Beta to VOO" in out:
            out["Beta to VOO"] = table["Beta to VOO"].round(4)
        out.insert(0, "Name", names)
        out.rename_axis("Ticker").reset_index().to_excel(writer, sheet_name=PORTFOLIO_SHEET, index=False)
        if weights is not None:
            _percent(weights).rename_axis("Ticker").reset_index().to_excel(writer, sheet_name="Recommended weights", index=False)
        if statistics is not None:
            stats = statistics.copy()
            stats["Sharpe ratio"] = stats["Sharpe ratio"].round(4)
            for c in ("Expected return", "Volatility", "Max drawdown"):
                stats[c] = (stats[c] * 100).round(4)
            stats = stats.rename(
                columns={"Expected return": "Expected return (%)", "Volatility": "Volatility (%)", "Max drawdown": "Max drawdown (%)"}
            )
            stats.rename_axis("Portfolio").reset_index().to_excel(writer, sheet_name="Recommended statistics", index=False)
        pd.DataFrame({"Setting": list(settings), "Value": list(settings.values())}).to_excel(
            writer, sheet_name="Settings", index=False
        )
        for sheet in writer.book.worksheets:
            for col in sheet.columns:
                sheet.column_dimensions[col[0].column_letter].width = max(len(str(c.value or "")) for c in col) + 3
    return buffer.getvalue()


def import_weights(file) -> pd.Series:
    """Reads ticker weights (in percent) from the Portfolio sheet of a file this app exported."""
    try:
        frame = pd.read_excel(file, sheet_name=PORTFOLIO_SHEET)
    except ValueError:
        raise ValueError(f"The workbook has no '{PORTFOLIO_SHEET}' sheet.") from None
    if not {"Ticker", WEIGHT_COLUMN} <= set(frame.columns):
        raise ValueError(f"The '{PORTFOLIO_SHEET}' sheet needs 'Ticker' and '{WEIGHT_COLUMN}' columns.")
    frame = frame.dropna(subset=["Ticker"])
    frame["Ticker"] = frame["Ticker"].astype(str).str.strip()
    if frame["Ticker"].duplicated().any():
        raise ValueError("The workbook lists a ticker more than once.")
    weights = pd.to_numeric(frame[WEIGHT_COLUMN], errors="coerce")
    if weights.isna().any() or (weights < 0).any():
        raise ValueError("Weights must be non-negative numbers.")
    if frame.empty:
        raise ValueError("The workbook contains no assets.")
    return pd.Series(weights.to_numpy(), index=frame["Ticker"].to_numpy(), name=WEIGHT_COLUMN)
