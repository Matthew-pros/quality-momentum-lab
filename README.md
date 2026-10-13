# Quality Momentum Lab — research app v2

A Streamlit research environment for screening equities, reviewing price and fundamental data, testing monthly momentum rules, sizing positions with ATR, and reconciling FIFO tax lots.

## What the app implements

- **Price ingestion:** one non-threaded Yahoo batch with sequential retry for partial failures, or import a user-supplied/licensed long-format OHLCV CSV; per-ticker quality flags for stale data, invalid prices and extreme daily moves.
- **Technical factors:** momentum 12–1M and 6–1M, MA50/MA200 trend, MA200 slope, ADX(14), ATR(20), 20-/55-session breakout and 50-session low trigger.
- **Official filing fundamentals:** SEC EDGAR Company Facts for supported filers, with every fact filtered by its filing date for as-of snapshots. Enter a contact email in the sidebar or set the Streamlit secret `SEC_CONTACT_EMAIL`.
- **Explicit data coverage:** missing metrics stay missing; the app does not manufacture neutral 50-point fundamental scores.
- **Optional revisions and sentiment:** timestamped 0–100 CSV inputs, aligned point-in-time to decision dates; no fake default sentiment.
- **Monthly backtest:** score-driven top-N portfolio, trend/ADX/breakout gates, score hysteresis, limit to replacements per month, equal-weight or inverse-ATR weights, transaction costs on gross traded notional, equity curve, drawdown/volatility/Sharpe, a rebalance audit and transaction-cost sensitivity.
- **Stop variants:** no stop/month-end rebalance, month-end 50-session low, daily-bar ATR trailing stop, daily-bar 50-session low, or both. Daily stops use information from the previous close and approximate gap-aware fills: open when the market gaps through the stop, otherwise at the stop level. High-water marks persist across monthly rebalance dates, and stop exits incur modeled transaction costs. This is not an intraday execution engine.
- **Chronological performance split:** 60% train / 20% validation / 20% holdout summary for net return, CAGR, volatility, Sharpe, drawdown, turnover, stops and benchmark. This is a diagnostic split, not a walk-forward optimizer; the holdout is only genuinely untouched if parameters are not adjusted after viewing it.
- **Standalone DCF lab:** an FCFE-style discounted-cash-flow scenario from annual FCF proxy and editable assumptions, with a value-per-share sensitivity table. It is not part of the composite score or trading backtest; currency, ADR share count and cash-flow-definition checks remain a manual prerequisite.
- **Risk sizing:** long-only quantity planner using ATR stop distance, risk budget, max position and total exposure caps.
- **FIFO lot ledger:** buys/sells matched chronologically, purchase/sale fees allocated, realized matches and remaining inventory exports. For CZK-only transactions it flags potentially relevant Czech securities exemptions (annual gross proceeds test / three-year holding test) based on 2026 official guidance; it does not calculate statutory tax due or handle non-CZK FX automatically.
- **Automated regression checks:** syntax compilation plus synthetic tests for indicator units, future-data separation, hysteresis, signal timestamps, SEC as-of snapshots, CSV price ingestion, Streamlit UI smoke rendering and FIFO accounting.

## Important limits (not a claim of production-grade investment performance)

1. **Yahoo Finance:** `yfinance` is an unofficial client and Yahoo may return HTTP 401/invalid-crumb errors or block usage. It is kept as an experimental source only, and public use/redistribution requires checking Yahoo's terms and appropriate rights. For a public app, use the **Import OHLCV CSV** provider with data for which you have suitable rights. The app deliberately no longer calls Yahoo's unstable `.info` endpoint for fundamentals.
2. **SEC coverage:** SEC Company Facts only covers mapped SEC filers and available XBRL tags. Accounting definitions and fiscal periods differ; reported values need data-quality and issuer-by-issuer review.
3. **Valuation estimates:** P/E and FCF yield based on current price × the latest reported shares outstanding can be materially wrong for ADRs, dual-class companies or stale share counts. These estimates are excluded from scoring by default.
4. **Backtest universe:** the entered symbol list is a current user-selected universe, not a point-in-time, survivorship-bias-free database. The engine stops instead of silently dropping held securities with missing future returns.
5. **Execution realism:** the monthly prototype applies turnover costs to close-to-close returns; it does not model exact opening fills, bid/ask spreads, market impact, every corporate action or FX conversion.
6. **Tax:** the FIFO tool is a lot ledger with year-specific preliminary flags. For CZK-only entries, it can mark potential value-test/time-test cases and warn about historic-year 40m limits or a possible notification duty for exempt income above CZK 5m; it is not a Czech tax calculation and does not apply all exemptions, tax rates, loss offsets, inheritance/corporate-action rules, or FX conversion.
7. **HMM and external sentiment:** HMM remains deferred until simpler features pass out-of-sample tests. External sentiment/revision signals are accepted only with timestamps and source metadata.

Do not interpret scores as forecasts or backtest performance as expected future returns.

## Run locally

Python 3.12 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud deployment

Streamlit Community Cloud does not use `runtime.txt` to select the Python version. In the app's advanced settings, select Python 3.12 (or another explicitly tested version), then save/redeploy as needed. Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Set secrets in the Streamlit app's settings under **Advanced settings → Secrets**, for example:

```toml
SEC_CONTACT_EMAIL = "your-contact-email@example.com"
```

Do not commit real contact details or API keys to a public repository.

## SEC contact email

The contact email is sent in the SEC request's User-Agent header. To set it on Streamlit Community Cloud, add a secret named `SEC_CONTACT_EMAIL`, or enter the address in the running app. Use an address you are comfortable sharing with the SEC request endpoint. SEC EDGAR public data APIs are described here: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

## OHLCV price CSV schema

When Yahoo Finance is unavailable or unsuitable for a public deployment, choose **Import OHLCV CSV** in the app. Upload one long-format CSV with one row per date and ticker:

```csv
Date,Ticker,Open,High,Low,Close,Adj Close,Volume
2025-01-02,MSFT,420.00,423.00,419.00,422.50,422.50,18000000
2025-01-02,AAPL,248.00,251.00,247.00,250.00,250.00,20000000
```

Required columns are `Date,Ticker,Open,High,Low,Close`; `Adj Close` and `Volume` are optional. If `Adj Close` is omitted, momentum falls back to `Close` and dividend adjustment is not included. Duplicate ticker/date rows are rejected. Edit the investment universe to match the tickers contained in the file, and include the `SPY` rows if you want the SPY benchmark. Use only data you have permission to use and redistribute.

## Optional-signal CSV schemas

Sentiment CSV: `Date,Ticker,Sentiment score,Source`

Revision CSV: `Date,Ticker,Revision score,Source`

Scores must be numeric from 0 to 100. Historical decisions use only the latest row on or before the decision date. No sentiment or analyst-revision source is bundled, because live/current-only values would introduce look-ahead in backtests.

## Tax lot CSV schema

Required: `Date,Ticker,Type,Quantity,Price`

Optional: `Fees,Currency`

Type is `BUY` or `SELL`; quantities are positive. Sales that exceed available FIFO inventory are rejected. Transactions are processed in their stated currency; the tool does not aggregate currencies into a common base currency.

## Test suite

```bash
pip install pytest
pytest -q
```

GitHub Actions runs these checks on pushes and pull requests. See [ROADMAP.md](ROADMAP.md) for production readiness gates.
