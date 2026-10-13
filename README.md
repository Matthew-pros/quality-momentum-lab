# Quality × Value × Momentum Lab (MVP)

Streamlit research app for equity screening and a basic monthly momentum backtest.

## Features
- Historical adjusted prices via `yfinance`
- Momentum 12–1M and 6–1M
- MA50/MA200 trend filter and 55-day breakout
- ADX(14) and ATR-derived volatility
- Indicative current fundamentals from Yahoo Finance
- Composite candidate score
- Basic monthly long-only momentum backtest with configurable turnover cost
- CSV exports

## Important limitations
This is a research prototype, not a validated investment system:
- Yahoo Finance fundamental fields are current snapshots, not point-in-time history.
- The backtest uses a user-supplied current ticker universe and therefore has survivorship bias.
- It does not model tax-lot accounting or Czech tax law.
- It does not yet include social sentiment, analyst estimate revisions, a full DCF, or HMM.
- Data availability, completeness, terms and rate limits are not guaranteed.

Do not use performance from this prototype as evidence of future returns.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Free deployment
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Open Streamlit Community Cloud and connect GitHub.
3. Create app and choose `app.py`.
4. For private use, review the platform's privacy settings before deploying.
5. Never commit API keys or personal portfolio data to a public repository.

## Data providers
- `yfinance` is an unofficial open-source interface to Yahoo Finance data. Review Yahoo's terms and applicable usage rights.
- Future fundamental layer should prefer official SEC EDGAR company filings for US issuers, plus a licensed/appropriate source for non-US filings.
- Alpha Vantage and Financial Modeling Prep offer free tiers with limits; check current terms and quotas before use.

## Roadmap to a robust version
1. Build point-in-time fundamentals and filings availability dates.
2. Add historical constituent universe including delisted stocks.
3. Add dividend, FX, spread, slippage, commissions and tax-lot-aware simulation.
4. Add analyst revisions and news sentiment only where historical timestamps and usage rights are available.
5. Add walk-forward and sensitivity tests, then compare to a benchmark.
6. Consider HMM only if it improves out-of-sample risk-adjusted results net of costs.
