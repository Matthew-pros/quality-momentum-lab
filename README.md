# Quality Momentum Lab — research app v2.2.0

A Streamlit research environment for screening equities, reviewing price and fundamental data, testing monthly momentum rules, sizing positions with ATR, and reconciling FIFO tax lots.

## What the app implements

- **Public-safe price mode:** the default public app uses deterministic synthetic DEMO prices, never calls a market-data API at startup, labels every synthetic series, and skips real-company SEC fundamentals. For actual securities, switch to **Import OHLCV CSV** and upload a data set you have permission to use and publicly display. The Yahoo adapter is optional/non-public and is not part of the default public runtime.
- **Price quality/provenance:** per-ticker flags for stale series, invalid prices and extreme daily moves; invalid OHLC rows are excluded from technical calculations while large moves remain flagged for manual review. CSV exports retain source mode, uploaded filename and whether adjusted close was provided or dividends may be omitted.
- **Technical factors:** momentum 12–1M and 6–1M, MA50/MA200 trend, MA200 slope, ADX(14), ATR(20), 20-/55-session breakout and 50-session low trigger. Indicators and daily stop logic use adjusted OHLC where available to avoid split-driven false jumps; latest ATR is converted to the quoted-price scale for share sizing.
- **Official filing fundamentals:** SEC EDGAR Company Facts for supported filers, with every fact filtered by its filing date for as-of snapshots; separate SEC submissions metadata supplies SIC codes/descriptions and a coarse SIC major-industry group. It is not a GICS/ICB sector feed. Enter a contact email in the sidebar or set the Streamlit secret `SEC_CONTACT_EMAIL`.
- **Explicit data coverage:** missing metrics stay missing; the app does not manufacture neutral 50-point fundamental scores.
- **Optional revisions and sentiment:** timestamped 0–100 CSV inputs, aligned point-in-time to decision dates; no fake default sentiment.
- **Monthly backtest:** score-driven top-N portfolio, trend/ADX/breakout gates, score hysteresis, replacement limits, optional cap of holdings per coarse SEC SIC group, equal-weight or inverse-ATR weights, entry/exit counts and holding-duration stats, equity curve, drawdown/volatility/Sharpe, rebalance audit and transaction-cost sensitivity.
- **Execution-cost assumptions:** separate commission, half-spread and slippage in bps applied to gross traded notional; stop exits add modelled sell turnover. This is still a cost proxy, not a broker fill simulation, and does not include fixed fees, market impact or FX.
- **Stop variants:** no stop/month-end rebalance, month-end 50-session low, daily-bar ATR trailing stop, daily-bar 50-session low, or both. Daily stops use information from the previous close and approximate gap-aware fills: open when the market gaps through the stop, otherwise at the stop level. High-water marks persist across monthly rebalance dates, and stop exits incur modeled transaction costs. This is not an intraday execution engine.
- **Chronological performance split:** 60% train / 20% validation / 20% holdout summary for net return, CAGR, volatility, Sharpe, drawdown, turnover, stops and benchmark. This is a diagnostic split, not a walk-forward optimizer; the holdout is only genuinely untouched if parameters are not adjusted after viewing it.
- **Train-selected holdout search:** compares up to a 3×3 local grid of minimum score and hysteresis using the first 60% of the return history and selects on training Sharpe (training CAGR/turnover as tie-breakers). Only the selected configuration's later 40% is shown as holdout; candidate holdout results are intentionally hidden. This is single train/test selection, not rolling walk-forward, and can still overfit if the grid is repeatedly changed after viewing holdout results.
- **Bootstrap uncertainty bands:** seeded moving-block bootstrap of the net monthly return path, reporting 5th/median/95th percentiles for CAGR, Sharpe, drawdown and terminal equity. This is a descriptive sequence-sensitivity diagnostic, not a guaranteed confidence interval and not a cure for data bias.
- **Standalone DCF lab:** an FCFE-style discounted-cash-flow scenario from annual FCF proxy and editable assumptions, with a value-per-share sensitivity table. It is not part of the composite score or trading backtest; currency, ADR share count and cash-flow-definition checks remain a manual prerequisite.
- **Risk sizing:** long-only quantity planner using ATR stop distance, risk budget per position, max position weight, total exposure and portfolio-level aggregate initial-risk cap. Whole-share rounding keeps the reported risk at or below the requested cap (subject to the stop-fill assumptions).
- **FIFO lot ledger:** buys/sells matched chronologically, purchase/sale fees allocated, realized matches and remaining inventory exports. For CZK-only transactions it flags potentially relevant Czech securities exemptions (annual gross proceeds test / three-year holding test) based on 2026 official guidance; it does not calculate statutory tax due or handle non-CZK FX automatically.
- **Automated regression checks:** syntax compilation plus synthetic tests for indicator units, future-data separation, hysteresis, signal timestamps, SEC as-of snapshots, CSV price ingestion, Streamlit UI smoke rendering and FIFO accounting.

## Important limits (not a claim of production-grade investment performance)

1. **Public-data rights:** the deployed/public UI does not fetch or redistribute Yahoo prices. Its default is clearly labeled synthetic data; real prices are user-supplied through CSV. The optional Yahoo adapter is intended only for explicitly configured, non-public experimentation and requires checking the provider's terms and data rights. SEC facts are disabled in synthetic-demo mode.
2. **SEC coverage:** SEC Company Facts only covers mapped SEC filers and available XBRL tags. Accounting definitions and fiscal periods differ; reported values need data-quality and issuer-by-issuer review.
3. **Valuation estimates:** P/E and FCF yield based on current price × the latest reported shares outstanding can be materially wrong for ADRs, dual-class companies or stale share counts. These estimates are excluded from scoring by default.
4. **Backtest universe:** the entered symbol list is a current user-selected universe, not a point-in-time, survivorship-bias-free database. The engine stops instead of silently dropping held securities with missing future returns.
5. **Execution realism:** the monthly prototype applies turnover costs to close-to-close returns; it does not model exact opening fills, bid/ask spreads, market impact, every corporate action or FX conversion.
6. **Tax:** the FIFO tool is a lot ledger with year-specific preliminary flags. It marks the CZK 100k annual value-test case, the CZK 40m time-test cap applicable to tax year 2025 only, and that cap's removal for securities from 2026. It never applies the 2025 cap retroactively to 2024 or earlier. From 14 February 2025, exempt cryptoasset sales could also count toward the 2025 cap; cryptoassets are not included in this app's lot ledger. The potential CZK 5m §38v notification check is evaluated against each source sale row, even if FIFO divides it among acquisition lots; that row-to-receipt match is a review approximation, not a legal conclusion. It is not a Czech tax calculation and does not apply all exemptions, tax rates, loss offsets, inheritance/corporate-action rules, all other exempt receipts, or FX conversion.
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
