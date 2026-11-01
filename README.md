# portfolio-risk

Streamlit app that estimates risk and return for a user-built portfolio and recommends optimised weights
(minimum variance, maximum Sharpe, risk parity), with an efficient frontier and a walk-forward backtest.

## Install

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
streamlit run streamlit_app.py
```

## Weights and what recomputes

Weights are typed as decimals (for example 0.5, 12.75 or 0.001), stored as float64, and divided by their sum before any calculation. Rounding to 2 decimal places happens for display only, with the largest remainder method so each displayed column sums to 100.00. A typed total of 0 shows a message and skips the weight-dependent calculations.

| Changes | Recomputes |
| --- | --- |
| A typed weight that keeps the same tickers held | Volatility, beta, risk contributions, VaR, drawdown, and the user's point on the frontier. Nothing is downloaded and nothing is estimated. |
| A typed weight that moves a ticker between zero and positive | The same, plus the common window, covariance, expected returns and betas of the held tickers. Cached per sorted held set and settings, and built from the prices already loaded, so nothing is downloaded. |
| Tickers or estimation settings | Prices (only tickers missing from the disk cache), covariance, expected returns for all selected tickers, and for the held tickers |
| Press **Run optimisation**, or open Frontier, Recommendations or Backtest for the first time | Optimiser, frontier and backtest, kept in the session per tickers and settings |

A ticker is held when its normalised weight is above 0. The Portfolio tab (metrics and table) and the Current sheets of the Excel export use estimates built from the held tickers alone: their own common window, covariance and expected returns, with the same sidebar settings. Adding a ticker at weight 0 therefore changes none of those numbers, and one held asset is enough. Zero-weight rows stay in the Portfolio table with "–" in the estimate columns. The Portfolio tab states the weeks and date range used.

The optimiser, frontier, recommendations and backtest use the shared estimates over all selected tickers, whatever the weights. In the Recommendations tab the Current statistics row, and the Your portfolio point on the frontier, are computed on those shared estimates, so they can differ from the Portfolio tab when a selected ticker has zero weight or a shorter history.

## Excel import and export

**Export to Excel** writes the portfolio, the recommended weights and statistics for every optimised portfolio, and the settings used. **Import from Excel** (sidebar) reads the `Portfolio` sheet of such a file and restores the selected assets and their weights; the sidebar assumptions are not restored.

## Spreadsheet version

Builds a workbook whose analytics are live formulas, so it can be used in Excel, LibreOffice or Google Sheets (upload the file to Drive) without the app:

```bash
python -m portfolio_risk.sheet_builder -o portfolio.xlsx VOO:10 NVDA:5 D05.SI:20 GC=F:15 BND:50
```

Arguments are `TICKER:WEIGHT` pairs, weights in percent. Prices are written as values (USD, weekly, forward-filled, common window); weights, risk-free rate and shrinkage stay editable in the `Inputs` sheet. The sheet covers returns, covariance (sample only), correlation, beta, risk contributions, drawdown, VaR and data quality. Optimisation is a Solver-ready trial portfolio on the `Optimiser` sheet. Ledoit-Wolf, risk parity, the efficient frontier and the backtest are not in the sheet.

## Test

```bash
pytest
```

Tests use synthetic data and need no network access.

## Layout

| Module | Concern |
| --- | --- |
| `universe.py` | Asset universe. Add entries to `STI`, `INDEX_FUNDS` or `FUTURES`; S&P 500 is scraped from Wikipedia |
| `data.py` | One batched yfinance download for tickers missing from the parquet cache (per-ticker fallback), USD conversion, weekly returns, data-quality checks |
| `analytics.py` | Covariance, expected returns, beta, risk contribution, drawdown, VaR |
| `optimiser.py` | cvxpy optimisers and efficient frontier. DPP problems built once per `Workspace` and reused across the 40 frontier points and backtest windows |
| `weights.py` | Typed-weight parsing, normalisation, selection bookkeeping and largest-remainder display rounding |
| `backtest.py` | Walk-forward backtest |
| `excel_io.py` | Excel export and import |
| `sheet_builder.py` | Formula-driven workbook generator |
| `charts.py` | Plotly figures |
| `ui.py` | Streamlit interface. Estimates are cached on tickers and settings; the weight table and tabs run in one fragment |

Prices are cached in `.cache/` (override with `PORTFOLIO_RISK_CACHE`) and refreshed after 24 hours.

Results are historical and are not financial advice.
