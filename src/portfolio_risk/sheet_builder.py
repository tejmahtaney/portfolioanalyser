"""Builds a self-contained workbook whose analytics are live formulas (Excel, LibreOffice, Google Sheets).

Prices are written as values (USD, weekly, forward-filled, common window); everything downstream is a formula.
Usage: python -m portfolio_risk.sheet_builder -o portfolio.xlsx VOO:10 NVDA:5 ...
"""
from __future__ import annotations

import argparse

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter as col

from . import data, universe
from .config import (
    BENCHMARK,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_RISK_FREE,
    DEFAULT_SHRINKAGE,
    HIGH_CORRELATION,
    MIN_WEEKS_WARNING,
    ROLL_JUMP_SIGMAS,
)

INPUT = Font(color="0000FF")
BOLD = Font(bold=True)
FLAG = PatternFill("solid", start_color="F8CBAD")
PCT, NUM = "0.00%", "0.0000"


def _header(ws, row, labels, start=1):
    for i, label in enumerate(labels):
        ws.cell(row, start + i, label).font = BOLD


def _price_window(prices: pd.DataFrame, returns: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Prices from the week before the common return window to the end."""
    window = data.common_window(returns, tickers)
    start = prices.index.get_loc(window.index[0]) - 1
    return prices.iloc[start:][tickers]


def build_workbook(
    holdings: dict[str, float], prices: pd.DataFrame, names: dict[str, str]
) -> Workbook:
    assets = list(holdings)
    columns = [BENCHMARK, *assets]
    px = _price_window(prices, data.build_returns(prices), list(dict.fromkeys(columns)))
    px = px[columns] if BENCHMARK in px else px
    n, weeks = len(assets), len(px) - 1
    last = weeks + 1
    first_asset, last_asset = col(3), col(2 + n)

    wb = Workbook()
    inputs = wb.active
    inputs.title = "Inputs"
    prices_ws = wb.create_sheet("Prices")
    rets = wb.create_sheet("Returns")
    cov_ws = wb.create_sheet("Covariance")
    corr_ws = wb.create_sheet("Correlation")
    stats = wb.create_sheet("Assets")
    port = wb.create_sheet("Portfolio")
    quality = wb.create_sheet("Data quality")
    solver = wb.create_sheet("Optimiser")

    # Inputs
    inputs["A1"] = "Portfolio risk: inputs (blue cells are editable)"
    inputs["A1"].font = BOLD
    for r, (label, value, fmt) in enumerate(
        [
            ("Risk-free rate", DEFAULT_RISK_FREE, PCT),
            ("Shrinkage intensity (0 = raw historical mean)", DEFAULT_SHRINKAGE, "0.00"),
            ("Periods per year", 52, "0"),
            ("Max weight per asset (checked in Optimiser)", DEFAULT_MAX_WEIGHT, PCT),
            ("VaR confidence level", 0.95, "0%"),
        ],
        start=3,
    ):
        inputs.cell(r, 1, label)
        c = inputs.cell(r, 2, value)
        c.font, c.number_format = INPUT, fmt
    RF, SHRINK, PPY, CAP, VARL = (f"Inputs!$B${r}" for r in range(3, 8))
    _header(inputs, 9, ["Ticker", "Name", "Weight (%)", "Normalised weight"])
    w0, w1 = 10, 9 + n
    for i, t in enumerate(assets):
        r = w0 + i
        inputs.cell(r, 1, t)
        inputs.cell(r, 2, names.get(t, ""))
        cw = inputs.cell(r, 3, holdings[t])
        cw.font = INPUT
        inputs.cell(r, 4, f"=C{r}/SUM($C${w0}:$C${w1})").number_format = PCT
    inputs.cell(w1 + 1, 2, "Typed total (%)")
    inputs.cell(w1 + 1, 3, f"=SUM(C{w0}:C{w1})")
    inputs.cell(w1 + 3, 1, "Change tickers or prices by rebuilding the workbook; weights, rates and the shrinkage can be edited here.")
    inputs.column_dimensions["A"].width = 46
    inputs.column_dimensions["B"].width = 34

    # Prices (values) and Returns (formulas)
    _header(prices_ws, 1, ["Date", f"{BENCHMARK} (benchmark)", *assets])
    _header(rets, 1, ["Date", f"{BENCHMARK} (benchmark)", *assets, "Portfolio return", "Cumulative growth", "Drawdown"])
    for r, (date, row) in enumerate(px.iterrows(), start=2):
        prices_ws.cell(r, 1, date.to_pydatetime()).number_format = "yyyy-mm-dd"
        for j, t in enumerate(columns):
            prices_ws.cell(r, 2 + j, float(row.iloc[j]))
    port_c, cum_c, dd_c = col(3 + n), col(4 + n), col(5 + n)
    for r in range(2, last + 1):
        rets.cell(r, 1, f"=Prices!A{r + 1}").number_format = "yyyy-mm-dd"
        for j in range(n + 1):
            c = col(2 + j)
            rets.cell(r, 2 + j, f"=Prices!{c}{r + 1}/Prices!{c}{r}-1").number_format = PCT
        rets.cell(r, 3 + n, f"=SUMPRODUCT({first_asset}{r}:{last_asset}{r},Covariance!$B${n + 3}:${col(n + 1)}${n + 3})").number_format = PCT
        prev = "1" if r == 2 else f"{cum_c}{r - 1}"
        rets.cell(r, 4 + n, f"={prev}*(1+{port_c}{r})").number_format = NUM
        rets.cell(r, 5 + n, f"={cum_c}{r}/MAX(1,MAX({cum_c}$2:{cum_c}{r}))-1").number_format = PCT
    rng = lambda c: f"Returns!${c}$2:${c}${last}"

    # Covariance and correlation
    for ws in (cov_ws, corr_ws):
        for j, t in enumerate(assets):
            ws.cell(1, 2 + j, t).font = BOLD
            ws.cell(2 + j, 1, t).font = BOLD
    for i in range(n):
        for j in range(n):
            a, b = col(3 + i), col(3 + j)
            cov_ws.cell(2 + i, 2 + j, f"=_xlfn.COVARIANCE.S({rng(a)},{rng(b)})*{PPY}").number_format = "0.000000"
            corr_ws.cell(2 + i, 2 + j, f"=CORREL({rng(a)},{rng(b)})").number_format = "0.000"
    cov_ws.cell(n + 3, 1, "Normalised weights").font = BOLD
    for j in range(n):
        cov_ws.cell(n + 3, 2 + j, f"=Inputs!$D${w0 + j}").number_format = PCT
    cov_ws.cell(n + 5, 1, "Annualised sample covariance (weekly covariance × periods per year). Ledoit-Wolf is not available in the sheet.")
    corr_rng = f"B2:{col(n + 1)}{n + 1}"
    corr_ws.conditional_formatting.add(
        corr_rng, FormulaRule(formula=[f"AND(B2>{HIGH_CORRELATION},ROW()-1<>COLUMN()-1)"], fill=FLAG)
    )
    corr_ws.cell(n + 3, 1, f"Pairs above {HIGH_CORRELATION}")
    corr_ws.cell(n + 3, 2, f'=(COUNTIF({corr_rng},">{HIGH_CORRELATION}")-{n})/2')

    # Per-asset statistics
    _header(stats, 1, ["Ticker", "Weight", "Annualised mean return", "Expected return (shrunk)", "Volatility", "Beta to VOO", "(Σw)ᵢ", "Risk contribution"])
    for i, t in enumerate(assets):
        r, c = 2 + i, col(3 + i)
        stats.cell(r, 1, t)
        stats.cell(r, 2, f"=Inputs!D{w0 + i}").number_format = PCT
        stats.cell(r, 3, f"=AVERAGE({rng(c)})*{PPY}").number_format = PCT
        stats.cell(r, 4, f"=(1-{SHRINK})*C{r}+{SHRINK}*AVERAGE($C$2:$C${n + 1})").number_format = PCT
        stats.cell(r, 5, f"=SQRT(Covariance!{col(2 + i)}{r})").number_format = PCT
        stats.cell(r, 6, f"=SLOPE({rng(c)},{rng('B')})").number_format = "0.000"
        stats.cell(r, 7, f"=SUMPRODUCT(Covariance!$B{r}:${col(n + 1)}{r},Covariance!$B${n + 3}:${col(n + 1)}${n + 3})").number_format = "0.000000"
        stats.cell(r, 8, f"=B{r}*G{r}/Portfolio!$B$6").number_format = PCT
    stats.cell(n + 2, 1, "Total").font = BOLD
    stats.cell(n + 2, 2, f"=SUM(B2:B{n + 1})").number_format = PCT
    stats.cell(n + 2, 8, f"=SUM(H2:H{n + 1})").number_format = PCT
    W, MU, BETA = (f"Assets!${c}$2:${c}${n + 1}" for c in "BDF")

    # Portfolio summary
    rows = [
        ("Weeks used", f"=COUNT({rng('B')})", "0"),
        ("Date range", f'=TEXT(MIN({rng("A")}),"yyyy-mm-dd")&" to "&TEXT(MAX({rng("A")}),"yyyy-mm-dd")', "@"),
        ("Expected return", f"=SUMPRODUCT({W},{MU})", PCT),
        ("Volatility", "=SQRT(B6)", PCT),
        ("Variance (w'Σw)", f"=SUMPRODUCT({W},Assets!$G$2:$G${n + 1})", "0.000000"),
        ("Sharpe ratio", f"=(B4-{RF})/B5", "0.000"),
        ("Beta to VOO", f"=SUMPRODUCT({W},{BETA})", "0.000"),
        ("Max drawdown", f"=MIN({rng(dd_c)})", PCT),
        (f"Historical VaR (weekly, loss)", f"=-PERCENTILE({rng(port_c)},1-{VARL})", PCT),
    ]
    port["A1"] = "Portfolio (calculated from the normalised weights)"
    port["A1"].font = BOLD
    for r, (label, formula, fmt) in enumerate(rows, start=2):
        port.cell(r, 1, label)
        port.cell(r, 2, formula).number_format = fmt
    port["A12"] = "Warnings"
    port["B12"] = f'=IF(B2<{MIN_WEEKS_WARNING},"Fewer than {MIN_WEEKS_WARNING} weeks of data: estimates are unreliable","")'
    port["A14"] = "Results are historical and are not financial advice."
    port.column_dimensions["A"].width = 30
    port.column_dimensions["B"].width = 26

    # Data quality
    _header(quality, 1, ["Ticker", "Futures?", "Largest |z-score|", f"Weeks beyond {ROLL_JUMP_SIGMAS:g}σ"])
    for i, t in enumerate(assets):
        r, c = 2 + i, col(3 + i)
        quality.cell(r, 1, t)
        quality.cell(r, 2, f'=RIGHT(A{r},2)="=F"')
        quality.cell(r, 3, f"=MAX(MAX({rng(c)})-AVERAGE({rng(c)}),AVERAGE({rng(c)})-MIN({rng(c)}))/STDEV({rng(c)})").number_format = "0.0"
        quality.cell(r, 4, f"=SUMPRODUCT(--(ABS({rng(c)}-AVERAGE({rng(c)}))/STDEV({rng(c)})>{ROLL_JUMP_SIGMAS}))")
        if t.endswith("=F"):
            letter = col(3 + i)
            rets.conditional_formatting.add(
                f"{letter}2:{letter}{last}",
                FormulaRule(formula=[f"ABS({letter}2-AVERAGE({letter}$2:{letter}${last}))/STDEV({letter}$2:{letter}${last})>{ROLL_JUMP_SIGMAS}"], fill=FLAG),
            )
    r = n + 3
    quality.cell(r, 1, "Weeks used")
    quality.cell(r, 2, "=Portfolio!B2")
    quality.cell(r + 1, 1, "Date range")
    quality.cell(r + 1, 2, "=Portfolio!B3")
    quality.cell(r + 2, 1, f"Pairs with correlation above {HIGH_CORRELATION}")
    quality.cell(r + 2, 2, f"=Correlation!B{n + 3}")
    quality.cell(r + 4, 1, "Prices are weekly closes converted to USD, forward-filled over holiday gaps and cut to the common window when the workbook was built.")
    quality.cell(r + 5, 1, "Futures returns exclude roll yield and collateral interest. Weeks flagged by the 4σ rule are shaded on the Returns sheet.")
    quality.column_dimensions["A"].width = 44

    # Optimiser (Solver-ready)
    solver["A1"] = "Trial portfolio for Solver (Excel Solver, or the Solver add-on in Google Sheets)"
    solver["A1"].font = BOLD
    _header(solver, 4, ["Ticker", "Trial weight", "(Σw)ᵢ", "Expected return"])
    s0, s1 = 5, 4 + n
    for i, t in enumerate(assets):
        r = s0 + i
        solver.cell(r, 1, t)
        solver.cell(r, 2, 1 / n).font = INPUT
        solver.cell(r, 2).number_format = PCT
        solver.cell(r, 4, f"=Assets!D{2 + i}").number_format = PCT
    # a helper row of trial weights avoids array formulas
    solver.cell(3, 1, "Trial weights (row)")
    for j in range(n):
        solver.cell(3, 2 + j, f"=$B${s0 + j}").number_format = PCT
    for i in range(n):
        r = s0 + i
        solver.cell(r, 3, f"=SUMPRODUCT(Covariance!$B{2 + i}:${col(n + 1)}{2 + i},$B$3:${col(n + 1)}$3)").number_format = "0.000000"
    out = s1 + 2
    for k, (label, formula, fmt) in enumerate(
        [
            ("Sum of weights (must be 100%)", f"=SUM(B{s0}:B{s1})", PCT),
            ("Largest weight", f"=MAX(B{s0}:B{s1})", PCT),
            ("Expected return", f"=SUMPRODUCT(B{s0}:B{s1},D{s0}:D{s1})", PCT),
            ("Variance", f"=SUMPRODUCT(B{s0}:B{s1},C{s0}:C{s1})", "0.000000"),
            ("Volatility", f"=SQRT(B{out + 3})", PCT),
            ("Sharpe ratio", f"=(B{out + 2}-{RF})/B{out + 4}", "0.000"),
            ("Cap check", f'=IF(B{out + 1}>{CAP}+0.000001,"Breaches cap","Within cap")', "@"),
        ]
    ):
        solver.cell(out + k, 1, label)
        solver.cell(out + k, 2, formula).number_format = fmt
    notes = [
        "Minimum variance: minimise the Variance cell by changing the trial weights, with weights ≥ 0, sum = 100% and largest weight ≤ cap.",
        "Maximum Sharpe: maximise the Sharpe ratio cell with the same constraints (use a nonlinear solver; start from equal weights).",
        "Risk parity and the efficient frontier are not set up in the sheet.",
        "Solver results are local optima from a general-purpose solver, not the exact convex solutions the Python app computes.",
    ]
    for k, text in enumerate(notes):
        solver.cell(out + 9 + k, 1, text)
    solver.column_dimensions["A"].width = 34

    for ws in wb.worksheets:
        ws.freeze_panes = "B2" if ws.title in ("Prices", "Returns") else None
        if ws.title in ("Prices", "Returns"):
            ws.column_dimensions["A"].width = 12
    return wb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("holdings", nargs="+", help="TICKER:WEIGHT pairs, weights in percent")
    parser.add_argument("-o", "--output", default="portfolio.xlsx")
    args = parser.parse_args()
    holdings = {}
    for item in args.holdings:
        ticker, _, weight = item.partition(":")
        holdings[ticker] = float(weight or 1)
    md = data.load_market_data(list(holdings))
    if md.failures:
        raise SystemExit("Download failed for: " + "; ".join(f"{t} ({e})" for t, e in md.failures.items()))
    uni, _ = universe.load_universe()
    names = dict(zip(uni["ticker"], uni["name"]))
    build_workbook(holdings, md.prices, names).save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
