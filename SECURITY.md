# Security policy

## Supported version

Security fixes are applied to the current `main` branch. This is a public research application, not a managed brokerage or trading service.

## Reporting a vulnerability

Please do not publish exploitable issues, access tokens, personal data or sensitive logs in a public GitHub issue. Use GitHub's private vulnerability reporting / Security Advisories feature for this repository when available. If private reporting is not enabled, open a minimal public issue asking for a private contact channel without including technical exploit details.

## API tokens and uploaded files

- Never commit a Tiingo token, SEC contact information intended to be private, Streamlit secrets, broker credentials, transaction exports or personal data.
- The app uses a user's Tiingo token only for requests to Tiingo EOD; the app does not intentionally write raw tokens to source control or application logs. Session-scoped data caching stores a one-way token fingerprint rather than the token as a cache key.
- Any token entered into a hosted app is processed by that hosted application's runtime. Use a token with the minimum permissions available, review the host and provider terms, and rotate it immediately if it may have been exposed.
- The default public mode uses synthetic demo data. Real price data requires a user-supplied CSV or the user's own provider token; do not use data in ways that exceed the applicable licence.
- Uploaded CSVs are processed in the running session. The app does not promise durable private storage, encrypted portfolio vault functionality, broker-grade data retention, or automatic deletion guarantees beyond the hosting platform's session lifecycle.
- Do not upload passwords, broker statements with account identifiers, or unrelated sensitive documents.

## Trading and investment controls

The application does not submit orders. Scores and backtests are research outputs, not verified forecasts. Before using real money, verify data sources, units, currencies, transaction costs and tax treatment independently. See `README.md` for the known data and simulation limitations.
