# Paused: portfolio-risk refit (2026-10-27)

## Done
- Batched `yf.download` with per-ticker fallback (`data.py`), verified against the live service.
- DPP `Workspace` in `optimiser.py`, Clarabel with tight tolerances and a fallback. Risk parity stays on the original `quad_form` form, because the factor form came back `optimal_inaccurate` on 2 of 12 backtest windows.
- `weights.py`: parsing, normalisation, `sync`, largest-remainder rounding, `display_table`.
- `ui.py`: cached estimates, one fragment (editor plus lazy tabs), Run optimisation, Equal/Clear buttons.
- Backtest takes `user_weights=None`; `add_user_weights` adds that column afterwards.
- 271 tests pass. The call-count tests were mutation-checked.
- Real server and Chromium run showed no exceptions.
- README updated.

## Timings (15 tickers, cached data, AppTest, medians)
- Weight edit: 0.72 s to 0.09 s.
- Ticker change: 0.74 s to 0.18 s.
- Full optimisation run (function level, warm process): about 0.58 s to 0.33 s.
- In-app Run click measured 0.7-0.9 s, i.e. slower than function level. I was bisecting this: the same code ran 0.73 s in a process that imported `portfolio_risk.ui` and 0.33 s without it. Cause unknown; the last check (background task bszymtbmf) imports modules one by one.

## Still to do
- Find the cause of the in-app slowdown, or report it as unexplained.
- Final test run, then write the report: before/after table and the unverified list.

## Unverified or caveats
- Typing decimals (0.001) in the real data_editor was not verified. Synthetic canvas input did not register. `step=0.01` may limit precision to 2 dp; the editor already shows 6.6667 as 6.66.
- Switching Clarabel for the old OSQP changes weights slightly. On the real data the differences are below 1e-6 apart from the last frontier point.
- Currency for batch-downloaded tickers comes from the cache or the `.SI` suffix, not from Yahoo metadata.
- st.status lists the load steps after loading finishes rather than live.
- "–" cells in the recommendations table are left-aligned.
- PID 7270 is the user's own streamlit server. I did not touch it.
- Scratch files, including a copy of the old code for timings, are in the session scratchpad.
