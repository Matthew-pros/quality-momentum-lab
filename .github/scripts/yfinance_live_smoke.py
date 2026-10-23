"""Network smoke test for the real, free Yahoo Finance data adapter.

This is separate from the deterministic unit tests by design: it verifies the external
provider is reachable and returns recent real OHLCV for liquid US-listed symbols.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# GitHub Actions runs this file as a script; ensure repository root is importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qlab.data import download_market_data  # noqa: E402


def main() -> int:
    symbols = ["MSFT", "SPY"]
    prices, report = download_market_data(symbols, period="1mo")
    failed = []

    for ticker in symbols:
        frame = prices.get(ticker)
        row = report.loc[report["Ticker"].astype(str).eq(ticker)]
        if frame is None or frame.empty:
            message = row["Message"].iloc[0] if not row.empty else "No provider report returned"
            failed.append(f"{ticker}: no usable daily price history ({message})")
            continue

        if "Close" not in frame:
            failed.append(f"{ticker}: Close field is missing")
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if len(close) < 10:
            failed.append(f"{ticker}: only {len(close)} usable daily bars; expected at least 10")
            continue
        if not (close.gt(0).all() and close.map(pd.notna).all()):
            failed.append(f"{ticker}: price values are not all finite and positive")
            continue

        latest = pd.Timestamp(close.index[-1])
        if latest.tzinfo is not None:
            latest = latest.tz_convert("UTC").tz_localize(None)
        age_days = (pd.Timestamp.now().normalize() - latest.normalize()).days
        if age_days > 10:
            failed.append(f"{ticker}: latest daily bar is {age_days} days old (limit 10)")
            continue

        currency = str(frame.attrs.get("currency", "UNKNOWN")).upper()
        if currency != "USD":
            failed.append(f"{ticker}: Yahoo quote currency is {currency}, expected USD for the smoke universe")
            continue

        print(
            f"PASS {ticker}: {len(close)} daily bars, latest={latest.date()}, "
            f"close={float(close.iloc[-1]):.2f} USD, source=Yahoo Finance via yfinance"
        )

    if failed:
        print("Yahoo Finance live-data smoke test FAILED:", file=sys.stderr)
        for issue in failed:
            print(f" - {issue}", file=sys.stderr)
        if not report.empty:
            print("\nProvider report:", file=sys.stderr)
            print(report[["Ticker", "Status", "Message", "Currency"]].to_string(index=False), file=sys.stderr)
        return 1

    print(f"PASS: Yahoo Finance returned fresh daily USD OHLCV for all {len(symbols)} smoke-test symbols.")
    print(f"Checked at {datetime.now(timezone.utc).isoformat()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
