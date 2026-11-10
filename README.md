# portfolio-risk

Streamlit app that estimates risk and return for a user-built portfolio and recommends optimised weights
(minimum variance, maximum Sharpe, risk parity), with an efficient frontier and a walk-forward backtest.

## Install

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
streamlit run streamlit_app.py
```

## Weights

Type weights as decimals, for example 0.5 or 12.75. The app divides them by their sum before any calculation. Displayed weights are rounded to 2 decimal places so each column sums to 100.00. A typed total of 0 skips the weight-dependent statistics.

The Portfolio tab and the Current rows of the Excel export use only assets with a weight above 0, over their own common history. An asset at weight 0 does not change these numbers. The Frontier, Recommendations and Backtest tabs use all selected assets over their shared history, so their Current figures can differ from the Portfolio tab.

The optimiser, frontier and backtest run when you press **Run optimisation** or first open one of those tabs. Editing weights does not rerun them.

## Excel import and export

**Export to Excel** writes the portfolio, the recommended weights and statistics for every optimised portfolio, and the settings used. **Import from Excel** (sidebar) reads the `Portfolio` sheet of such a file and restores the selected assets and their weights; the sidebar assumptions are not restored.

## Add positions

The second page of the app finds stocks to add to a portfolio. Upload a workbook exported from the Portfolio risk page, pick candidates (a watchlist from the universe, or the S&P 500), the maximum number of new positions (1 to 10) and a mode: **Reweight everything** recomputes all weights at risk parity; **Keep current holdings** holds their proportions fixed and sizes them as one block. Press **Run search**; changing an input does not rerun it, and the result is kept in the session for those inputs.

**New positions in total** limits the combined weight of the additions (default 15%). When the sizing gives them more, they are scaled down to the limit and the holdings scaled up, each side keeping its internal proportions. Without the limit, risk parity gives every addition the same risk as the whole current portfolio (keep mode) or as any one holding (reweight mode); the search favours low-volatility stocks, which need more weight for the same risk, so three additions can take 60% or more. Set it to 100 to switch the limit off.

Each finalist shows its diversification ratio, volatility, effective number of assets and backtest status, with the out-of-sample Sharpe ratio against a baseline without additions. A finalist is flagged when its status changes if February to April 2020 is left out. If a current holding has under 500 weeks of history no finalist can be backtested, and the page names the holding and its weeks. Selecting a finalist shows its trades; a trade inside the no-trade band (default 5 points and 25% of the target, both adjustable without rerunning) shows as "–". **Export target weights** writes an xlsx that the Portfolio risk page can import.

## Spreadsheet version

Builds a workbook whose analytics are live formulas, so it can be used in Excel, LibreOffice or Google Sheets (upload the file to Drive) without the app:

```bash
python -m portfolio_risk.sheet_builder -o portfolio.xlsx VOO:10 NVDA:5 D05.SI:20 GC=F:15 BND:50
```

Arguments are `TICKER:WEIGHT` pairs, weights in percent. Prices are written as values (USD, weekly with every bar dated to the Monday of its week, forward-filled, common window); weights, risk-free rate and shrinkage stay editable in the `Inputs` sheet. The sheet covers returns, covariance (sample only), correlation, beta, risk contributions, drawdown, VaR and data quality. Optimisation is a Solver-ready trial portfolio on the `Optimiser` sheet. Ledoit-Wolf, risk parity, the efficient frontier and the backtest are not in the sheet.

## Test

```bash
pytest
```

Tests use synthetic data and need no network access.

## Layout

| Module | Concern |
| --- | --- |
| `universe.py` | Asset universe. Add entries to `STI`, `INDEX_FUNDS` or `FUTURES`; S&P 500 is scraped from Wikipedia |
| `data.py` | History period (`5y` default, `10y` option; the cache file is tagged with its period). One batched yfinance download for tickers missing from the parquet cache (per-ticker fallback), USD conversion, weekly returns (bar dates snapped to Monday), data-quality checks |
| `analytics.py` | Covariance (Ledoit-Wolf, sample, and an optional exponentially weighted `ewma`), expected returns, beta, risk contribution, drawdown, VaR |
| `optimiser.py` | cvxpy optimisers and efficient frontier. DPP problems built once per `Workspace` and reused across the 40 frontier points and backtest windows |
| `weights.py` | Typed-weight parsing, normalisation, selection bookkeeping and largest-remainder display rounding |
| `backtest.py` | Walk-forward backtest |
| `search.py` | Finds stocks to add to a portfolio under risk parity: beam search on the diversification ratio (last 156 weeks, EWMA covariance), walk-forward backtest filter on 10 years, no-trade band. Has a Newton risk parity solver for the search loop. No Streamlit code |
| `ui_search.py` | Streamlit page for `search.py` ("Add positions") |
| `excel_io.py` | Excel export and import |
| `sheet_builder.py` | Formula-driven workbook generator |
| `charts.py` | Plotly figures |
| `ui.py` | Streamlit interface (Portfolio risk page); `streamlit_app.py` joins both pages with `st.navigation` |

Prices are converted to USD and cached in `.cache/` for 24 hours. Set `PORTFOLIO_RISK_CACHE` to use another folder.

Results are historical and are not financial advice.
