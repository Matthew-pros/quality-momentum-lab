# Quality Momentum Lab — research app v2

A Streamlit research environment for screening equities, reviewing price and fundamental data, testing monthly momentum rules, sizing positions with ATR, and reconciling FIFO tax lots.

## What the app implements

- **Price ingestion:** one non-threaded batch download with sequential retry for missing ticker histories; per-ticker row coverage and first/last price dates.
- **Technical factors:** momentum 12–1M and 6–1M, MA50/MA200 trend, MA200 slope, ADX(14), ATR(20), 20-/55-session breakout and 50-session low trigger.
- **Official filing fundamentals:** SEC EDGAR Company Facts for supported filers, with every fact filtered by its filing date for as-of snapshots. Enter a contact email in the sidebar or set the Streamlit secret `SEC_CONTACT_EMAIL`.
- **Explicit data coverage:** missing metrics stay missing; the app does not manufacture neutral 50-point fundamental scores.
- **Optional revisions and sentiment:** timestamped 0–100 CSV inputs, aligned point-in-time to decision dates; no fake default sentiment.
- **Monthly backtest:** score-driven top-N portfolio, trend/ADX/breakout gates, score hysteresis, limit to replacements per month, transaction costs on gross traded notional, equity curve, drawdown/volatility/Sharpe and rebalance audit.
- **Risk sizing:** long-only quantity planner using ATR stop distance, risk budget, max position and total exposure caps.
- **FIFO lot ledger:** buys/sells matched chronologically, purchase/sale fees allocated, realized matches and remaining inventory exports.
- **Automated regression checks:** syntax compilation plus synthetic tests for indicator units, future-data separation, hysteresis, signal timestamps and FIFO accounting.

## Important limits (not a claim of production-grade investment performance)

1. **Yahoo Finance:** `yfinance` is an unofficial client and its maintainers say Yahoo Finance data is intended for personal use. Public redistribution or broader use requires checking Yahoo's terms and appropriate rights. The app uses Yahoo for exploratory price history only; it deliberately no longer calls Yahoo's unstable `.info` endpoint for fundamental fields.
2. **SEC coverage:** SEC Company Facts only covers mapped SEC filers and available XBRL tags. Accounting definitions and fiscal periods differ; reported values need data-quality and issuer-by-issuer review.
3. **Valuation estimates:** P/E and FCF yield based on current price × the latest reported shares outstanding can be materially wrong for ADRs, dual-class companies or stale share counts. These estimates are excluded from scoring by default.
4. **Backtest universe:** the entered symbol list is a current user-selected universe, not a point-in-time, survivorship-bias-free database. The engine stops instead of silently dropping held securities with missing future returns.
5. **Execution realism:** the monthly prototype applies turnover costs to close-to-close returns; it does not model exact opening fills, bid/ask spreads, market impact, every corporate action or FX conversion.
6. **Tax:** the FIFO tool is a lot ledger, not a Czech tax calculation. It does not apply exemptions, tax rates, holding-period tests, loss offsets, or CZK conversion.
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

## SEC contact email

The contact email is sent in the SEC request's User-Agent header. To set it on Streamlit Community Cloud, add a secret named `SEC_CONTACT_EMAIL`, or enter the address in the running app. Use an address you are comfortable sharing with the SEC request endpoint. SEC EDGAR public data APIs are described here: https://www.sec.gov/search-filings/edgar-application-programming-interfaces

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
