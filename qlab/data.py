"""Data adapters for Quality Momentum Lab.

Yahoo Finance is experimental OHLCV only through the unofficial yfinance interface.
SEC Company Facts supplies standardized reported facts for mapped SEC filers.
"""
from __future__ import annotations

from datetime import date
import hashlib
import time
from typing import Any

import numpy as np
import pandas as pd
import requests


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def normalize_tickers(text_or_tickers: str | list[str]) -> list[str]:
    raw = text_or_tickers.split(",") if isinstance(text_or_tickers, str) else text_or_tickers
    return list(dict.fromkeys(str(x).strip().upper() for x in raw if str(x).strip()))


def validate_usd_price_basis(
    tickers: list[str] | tuple[str, ...],
    currencies: dict[str, str],
    confirm_unknown_usd: bool = False,
    missing_tickers: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    """Validate that portfolio-valued calculations have a common USD price basis.

    This function never converts currencies. Missing metadata can only be treated
    as USD after an explicit user attestation; known non-USD codes cannot be
    overridden by that attestation.
    """
    symbols = normalize_tickers(list(tickers))
    missing = sorted(set(normalize_tickers(list(missing_tickers))) & set(symbols))
    unknown_codes = {"", "UNKNOWN", "NAN", "NONE", "<NA>"}
    unknown = sorted(
        ticker for ticker in symbols
        if ticker not in missing and str(currencies.get(ticker, "UNKNOWN")).strip().upper() in unknown_codes
    )
    non_usd_pairs = {
        (ticker, str(currencies[ticker]).strip().upper())
        for ticker in symbols
        if ticker in currencies
        and str(currencies[ticker]).strip().upper() not in unknown_codes | {"USD"}
    }
    non_usd_currencies = sorted({currency for _, currency in non_usd_pairs})
    confirmed_unknown = bool(confirm_unknown_usd)
    ready = not missing and not non_usd_currencies and (not unknown or confirmed_unknown)
    return {
        "ready": ready,
        "missing_tickers": missing,
        "unknown_tickers": unknown,
        "non_usd_currencies": non_usd_currencies,
        "confirm_unknown_usd": confirmed_unknown,
        "basis": "USD" if ready else "Unverified / incompatible",
    }


def _extract_ticker_frame(raw: pd.DataFrame, ticker: str, all_tickers: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        if len(all_tickers) != 1:
            return pd.DataFrame()
        out = raw.copy()
    else:
        out = None
        for level in range(raw.columns.nlevels):
            try:
                if ticker in raw.columns.get_level_values(level):
                    out = raw.xs(ticker, axis=1, level=level, drop_level=True).copy()
                    break
            except (KeyError, ValueError, TypeError):
                continue
        if out is None:
            return pd.DataFrame()
        if isinstance(out.columns, pd.MultiIndex):
            flattened = []
            for col in out.columns:
                values = [str(x) for x in col]
                flattened.append(next((x for x in values if x in PRICE_FIELDS), values[-1]))
            out.columns = flattened
    out.columns = [str(c) for c in out.columns]
    out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]
    keep = [c for c in PRICE_FIELDS if c in out.columns]
    if "Close" not in keep and "Adj Close" not in keep:
        return pd.DataFrame()
    out = out[keep].copy()
    for column in out.columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "Close" not in out and "Adj Close" in out:
        out["Close"] = out["Adj Close"]
    if "Adj Close" not in out and "Close" in out:
        out["Adj Close"] = out["Close"]
    return out.dropna(subset=["Close"], how="all")


def _add_adjusted_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    close = pd.to_numeric(frame.get("Close", pd.Series(index=frame.index, dtype=float)), errors="coerce")
    adj_close = pd.to_numeric(frame.get("Adj Close", close), errors="coerce")
    factor = (adj_close / close).replace([np.inf, -np.inf], np.nan)
    for raw_col, adjusted_col in [
        ("Open", "Adj Open"), ("High", "Adj High"), ("Low", "Adj Low")
    ]:
        if raw_col in frame:
            frame[adjusted_col] = pd.to_numeric(frame[raw_col], errors="coerce") * factor
    return frame


def assess_price_quality(
    prices: dict[str, pd.DataFrame],
    report: pd.DataFrame | None = None,
    now=None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Flag stale/extreme price series and exclude invalid OHLC rows from calculations."""
    if report is None:
        report = pd.DataFrame({"Ticker": list(prices)})
    else:
        report = report.copy()
    if "Ticker" not in report:
        report["Ticker"] = list(prices)
    report = report.drop(
        columns=["Calendar days stale", "Invalid price rows", "Max abs daily return %", "Quality"],
        errors="ignore",
    )
    now_ts = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    if now_ts.tzinfo:
        now_ts = now_ts.tz_localize(None)
    quality_rows = []
    sanitized: dict[str, pd.DataFrame] = {}
    for ticker in report["Ticker"].astype(str):
        frame = prices.get(ticker)
        quality_row: dict[str, Any] = {
            "Ticker": ticker,
            "Calendar days stale": np.nan,
            "Invalid price rows": np.nan,
            "Max abs daily return %": np.nan,
            "Quality": "MISSING",
            "Quality message": "No valid price series returned",
        }
        if frame is None or frame.empty or "Close" not in frame:
            quality_rows.append(quality_row)
            continue
        clean = frame.copy().sort_index()
        close = pd.to_numeric(clean["Close"], errors="coerce")
        invalid = (~np.isfinite(close) | (close <= 0))
        numeric_prices: dict[str, pd.Series] = {"Close": close}
        for column in ["Open", "High", "Low"]:
            if column in clean:
                numeric_prices[column] = pd.to_numeric(clean[column], errors="coerce")
                invalid |= (
                    ~np.isfinite(numeric_prices[column])
                    | (numeric_prices[column] <= 0)
                )
        if "Adj Close" in clean:
            adjusted_close = pd.to_numeric(clean["Adj Close"], errors="coerce")
            invalid |= (~np.isfinite(adjusted_close) | (adjusted_close <= 0))
        if {"Open", "High", "Low", "Close"}.issubset(numeric_prices):
            open_price = numeric_prices["Open"]
            high = numeric_prices["High"]
            low = numeric_prices["Low"]
            invalid |= high < low
            invalid |= high < pd.concat([open_price, close], axis=1).max(axis=1)
            invalid |= low > pd.concat([open_price, close], axis=1).min(axis=1)
        if "Volume" in clean:
            volume = pd.to_numeric(clean["Volume"], errors="coerce")
            # Missing volume is allowed; a negative volume is not.
            invalid |= volume.notna() & (volume < 0)
        invalid_count = int(invalid.sum())
        # Keep the original invalid-row count in report but prevent invalid values from
        # contaminating technicals, price levels, returns, or position sizing.
        price_columns = [
            column for column in [
                "Open", "High", "Low", "Close", "Adj Close",
                "Adj Open", "Adj High", "Adj Low",
            ] if column in clean
        ]
        clean.loc[invalid, price_columns] = np.nan
        sanitized[ticker] = clean

        ret_col = "Adj Close" if "Adj Close" in clean else "Close"
        ret = pd.to_numeric(clean[ret_col], errors="coerce").pct_change(fill_method=None)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        max_move = float(ret.abs().max() * 100) if not ret.empty else np.nan
        last_valid = clean["Close"].last_valid_index()
        if last_valid is None:
            stale = int((now_ts - pd.Timestamp(clean.index.max()).normalize()).days)
        else:
            last_date = pd.Timestamp(last_valid)
            if last_date.tzinfo:
                last_date = last_date.tz_localize(None)
            stale = max(0, int((now_ts - last_date.normalize()).days))
        flags = []
        if stale > 7:
            flags.append(f"stale {stale} calendar days")
        if invalid_count:
            flags.append(f"{invalid_count} invalid price rows excluded")
        if pd.notna(max_move) and max_move >= 30:
            flags.append(f"large daily return {max_move:.1f}%")
        quality_row.update({
            "Calendar days stale": stale,
            "Invalid price rows": invalid_count,
            "Max abs daily return %": max_move,
            "Quality": "REVIEW" if flags else "OK",
            "Quality message": "; ".join(flags),
        })
        quality_rows.append(quality_row)
    qdf = pd.DataFrame(quality_rows)
    report = report.merge(qdf, on="Ticker", how="left", suffixes=("", "_quality"))
    base_message = report["Message"].fillna("").astype(str) if "Message" in report else pd.Series("", index=report.index)
    quality_message = report["Quality message"].fillna("").astype(str)
    report["Message"] = [
        "; ".join(part for part in [base, quality] if part)
        for base, quality in zip(base_message, quality_message)
    ]
    report = report.drop(columns=["Quality message"])
    for column in ["Rows", "Calendar days stale", "Invalid price rows", "Max abs daily return %", "Missing %"]:
        if column in report:
            report[column] = pd.to_numeric(report[column], errors="coerce")
    # Refresh reported valid row count and missingness after invalid data were excluded.
    for ticker, clean in sanitized.items():
        mask = report["Ticker"].astype(str).eq(ticker)
        if mask.any():
            valid_close = pd.to_numeric(clean["Close"], errors="coerce")
            report.loc[mask, "Rows"] = int(valid_close.notna().sum())
            report.loc[mask, "Missing %"] = round(float(valid_close.isna().mean() * 100), 2)
    # Preserve price series that were absent from the quality report (e.g. extra CSV tickers).
    report_tickers = set(report["Ticker"].astype(str))
    for ticker, frame in prices.items():
        if ticker not in sanitized and ticker not in report_tickers:
            sanitized[ticker] = frame.copy()
    return sanitized, report

def generate_demo_market_data(
    tickers: str | list[str],
    period: str = "10y",
    now=None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Generate deterministic synthetic OHLCV data for public UI demonstrations.

    This function never reaches an external service and the outputs are not real
    security prices or signals about real companies. A ticker string only seeds its
    synthetic path; it does not map that path to a real security.
    """
    symbols = normalize_tickers(tickers)
    report_columns = [
        "Ticker", "Status", "Rows", "First date", "Last date", "Missing %",
        "Message", "Source",
    ]
    if not symbols:
        return {}, pd.DataFrame(columns=report_columns)
    days_by_period = {"5y": 1260, "10y": 2520, "max": 3780}
    observations = days_by_period.get(str(period).lower(), 2520)
    end = pd.Timestamp(now if now is not None else pd.Timestamp.now()).normalize()
    dates = pd.bdate_range(end=end, periods=observations)
    prices: dict[str, pd.DataFrame] = {}
    report_rows = []
    for position, ticker in enumerate(symbols):
        seed_bytes = hashlib.sha256(ticker.encode("utf-8")).digest()[:8]
        seed = int.from_bytes(seed_bytes, "little", signed=False)
        rng = np.random.default_rng(seed)
        annual_drift = 0.00010 + (seed % 31) / 100_000
        daily_vol = 0.008 + ((seed >> 8) % 8) / 1000
        log_returns = rng.normal(annual_drift, daily_vol, size=len(dates))
        base = 25.0 + (position % 12) * 11.0
        close = base * np.exp(np.cumsum(log_returns))
        open_price = close * np.exp(rng.normal(0.0, daily_vol * 0.35, size=len(dates)))
        high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.012, size=len(dates)))
        low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.012, size=len(dates))
        )
        volume = rng.integers(250_000, 15_000_000, size=len(dates), endpoint=False)
        frame = pd.DataFrame({
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }, index=dates)
        prices[ticker] = frame
        report_rows.append({
            "Ticker": ticker, "Status": "OK", "Rows": len(frame),
            "First date": dates.min().date().isoformat(),
            "Last date": dates.max().date().isoformat(),
            "Missing %": 0.0,
            "Message": "SYNTHETIC DEMO ONLY — not real market data",
            "Source": "Synthetic demo",
        })
    report = pd.DataFrame(report_rows, columns=report_columns)
    prices, report = assess_price_quality(prices, report, now=end)
    report["Source"] = "Synthetic demo"
    report["Message"] = report["Message"].fillna("").astype(str).apply(
        lambda message: "; ".join(part for part in [message, "SYNTHETIC DEMO ONLY — not real market data"] if part)
    )
    return prices, report


def download_market_data(
    tickers: str | list[str],
    period: str = "10y",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download one non-threaded batch and retry only partial failures, sequentially."""
    symbols = normalize_tickers(tickers)
    report_columns = [
        "Ticker", "Status", "Rows", "First date", "Last date", "Missing %", "Message",
    ]
    if not symbols:
        return {}, pd.DataFrame(columns=report_columns)
    try:
        import yfinance as yf  # Optional developer-only adapter; not used by the public UI.
    except ImportError:
        return {}, pd.DataFrame([
            {
                "Ticker": ticker, "Status": "MISSING", "Rows": 0,
                "First date": "", "Last date": "", "Missing %": 100.0,
                "Message": "Optional yfinance package is not installed; use licensed OHLCV CSV instead.",
            }
            for ticker in symbols
        ], columns=report_columns)
    raw = pd.DataFrame()
    batch_error = ""
    try:
        raw = yf.download(
            tickers=symbols if len(symbols) > 1 else symbols[0],
            period=period,
            auto_adjust=False,
            actions=False,
            group_by="ticker",
            threads=False,
            progress=False,
            timeout=30,
            repair=False,
        )
    except Exception as exc:
        batch_error = f"{type(exc).__name__}: {exc}"
    prices: dict[str, pd.DataFrame] = {}
    initial_report = []
    for ticker in symbols:
        frame = _extract_ticker_frame(raw, ticker, symbols)
        if not frame.empty:
            prices[ticker] = _add_adjusted_ohlc(frame)
        initial_report.append({
            "Ticker": ticker,
            "Status": "OK" if ticker in prices else "MISSING",
            "Rows": len(prices[ticker]) if ticker in prices else 0,
            "First date": prices[ticker].index.min().date().isoformat() if ticker in prices else "",
            "Last date": prices[ticker].index.max().date().isoformat() if ticker in prices else "",
            "Missing %": round(float(prices[ticker]["Close"].isna().mean() * 100), 2) if ticker in prices else 100.0,
            "Message": "" if ticker in prices else (batch_error or "Provider returned no rows for this ticker"),
        })
    # A total outage/401 should not cause a burst of per-symbol retries.
    if prices and len(prices) < len(symbols):
        for ticker in symbols:
            if ticker in prices:
                continue
            try:
                one = yf.download(
                    tickers=ticker, period=period, auto_adjust=False, actions=False,
                    group_by="column", threads=False, progress=False, timeout=30, repair=False,
                )
                frame = _extract_ticker_frame(one, ticker, [ticker])
                if not frame.empty:
                    prices[ticker] = _add_adjusted_ohlc(frame)
            except Exception:
                continue
    report = pd.DataFrame(initial_report)
    for ticker, frame in prices.items():
        mask = report["Ticker"].eq(ticker)
        report.loc[mask, ["Status", "Rows", "First date", "Last date", "Missing %", "Message"]] = [
            "OK", len(frame), frame.index.min().date().isoformat(), frame.index.max().date().isoformat(),
            round(float(frame["Close"].isna().mean() * 100), 2), "",
        ]
    return assess_price_quality(prices, report)


def load_ohlcv_csv(
    source,
    expected_tickers: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load a user-supplied long-format OHLCV CSV without reaching an external price API.

    Required columns: Date, Ticker, Open, High, Low, Close. Optional: Adj Close, Volume, Currency.
    Currency is validated per ticker and exposed as frame metadata/report data. It does not convert prices.
    If Adj Close is absent, Close is used as a clearly documented non-dividend-adjusted proxy.
    Duplicate ticker/date rows are rejected instead of being silently discarded.
    """
    try:
        raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV: {type(exc).__name__}: {exc}") from exc
    raw.columns = [str(col).strip() for col in raw.columns]
    required = {"Date", "Ticker", "Open", "High", "Low", "Close"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            "OHLCV CSV requires columns: Date, Ticker, Open, High, Low, Close. "
            f"Missing: {', '.join(sorted(missing))}"
        )
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Ticker"] = raw["Ticker"].astype(str).str.upper().str.strip()
    if raw["Date"].isna().any():
        raise ValueError("CSV contains one or more unparseable Date values.")
    if raw["Ticker"].eq("").any():
        raise ValueError("CSV contains an empty Ticker.")
    for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if column in raw:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
    if "Currency" in raw:
        raw["Currency"] = raw["Currency"].astype("string").str.upper().str.strip().replace("", pd.NA)
    if "Adj Close" not in raw:
        raw["Adj Close"] = raw["Close"]
        adj_note = "Adj Close absent: momentum uses Close and does not include dividend adjustments."
    else:
        adj_note = ""
    duplicate = raw.duplicated(["Ticker", "Date"], keep=False)
    if duplicate.any():
        samples = raw.loc[duplicate, ["Ticker", "Date"]].head(5)
        items = ", ".join(f"{r.Ticker}@{pd.Timestamp(r.Date).date()}" for r in samples.itertuples(index=False))
        raise ValueError(f"Duplicate ticker/date observations detected: {items}. Resolve duplicates in the CSV.")
    prices: dict[str, pd.DataFrame] = {}
    currency_by_ticker: dict[str, str] = {}
    for ticker, part in raw.groupby("Ticker", sort=True):
        currency = "UNKNOWN"
        if "Currency" in part:
            if part["Currency"].isna().any():
                raise ValueError(
                    f"{ticker}: Currency is blank on one or more rows. Supply the ticker's currency consistently for all rows, "
                    "or omit the Currency column and confirm the common USD basis in the application."
                )
            currencies = sorted(set(part["Currency"].astype(str).str.upper()))
            if len(currencies) != 1:
                raise ValueError(f"{ticker}: Currency changes within the imported series: {', '.join(currencies)}.")
            currency = currencies[0]
            if not currency.isalpha() or not (3 <= len(currency) <= 5):
                raise ValueError(f"{ticker}: invalid Currency code {currency!r}. Use a code such as USD, EUR, GBP or CZK.")
        currency_by_ticker[ticker] = currency
        part = part.sort_values("Date").set_index("Date")
        fields = [column for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if column in part.columns]
        frame = part[fields].copy()
        frame = _add_adjusted_ohlc(frame)
        frame.attrs["currency"] = currency
        prices[ticker] = frame

    report_tickers = normalize_tickers(expected_tickers or list(prices))
    report_rows = []
    now = pd.Timestamp.now().normalize()
    for ticker in report_tickers:
        frame = prices.get(ticker)
        if frame is None or frame.empty:
            report_rows.append({
                "Ticker": ticker, "Currency": currency_by_ticker.get(ticker, "UNKNOWN"),
                "Status": "MISSING", "Rows": 0, "First date": "", "Last date": "",
                "Missing %": 100.0, "Message": "Ticker not present in uploaded CSV" + (f"; {adj_note}" if adj_note else ""),
            })
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce")
        report_rows.append({
            "Ticker": ticker, "Currency": currency_by_ticker.get(ticker, "UNKNOWN"),
            "Status": "OK", "Rows": int(close.notna().sum()),
            "First date": frame.index.min().date().isoformat(), "Last date": frame.index.max().date().isoformat(),
            "Missing %": round(float(close.isna().mean() * 100), 2),
            "Message": adj_note,
        })
    report = pd.DataFrame(report_rows)
    return assess_price_quality(prices, report, now=now)


def sic_group(sic: int | str | None) -> str:
    """Coarse SIC major group, not a GICS/ICB sector classification."""
    if sic in (None, ""):
        return "Unclassified"
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return "Unclassified"
    if 100 <= code <= 999:
        return "Agriculture / Forestry / Fishing"
    if 1000 <= code <= 1499:
        return "Mining"
    if 1500 <= code <= 1799:
        return "Construction"
    if 2000 <= code <= 3999:
        return "Manufacturing"
    if 4000 <= code <= 4999:
        return "Transportation / Communications / Utilities"
    if 5000 <= code <= 5999:
        return "Wholesale / Retail Trade"
    if 6000 <= code <= 6799:
        return "Finance / Insurance / Real Estate"
    if 7000 <= code <= 8999:
        return "Services"
    if 9000 <= code <= 9999:
        return "Public Administration / Other"
    return "Unclassified"


def sec_user_agent(email: str) -> str:
    return f"QualityMomentumLab/1.0 (https://github.com/Matthew-pros/quality-momentum-lab; contact: {email.strip()})"


def _sec_get_json(url: str, email: str) -> dict[str, Any]:
    if not email or "@" not in email:
        raise ValueError("Enter a valid contact email to identify requests to SEC EDGAR.")
    headers = {
        "User-Agent": sec_user_agent(email),
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }
    last_error = None
    for attempt in range(4):
        try:
            # Stay below SEC's published fair-access request rate even during cache misses.
            time.sleep(0.12)
            response = requests.get(url, headers=headers, timeout=25)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    wait = min(8.0, max(0.5, float(retry_after)))
                except (TypeError, ValueError):
                    wait = min(8.0, 0.5 * (2 ** attempt))
                if attempt < 3:
                    time.sleep(wait)
                    continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= 3:
                break
            time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    if last_error:
        raise last_error
    raise RuntimeError(f"SEC request failed without a response: {url}")


def load_sec_ticker_map(email: str) -> dict[str, dict[str, Any]]:
    payload = _sec_get_json(SEC_TICKER_MAP_URL, email)
    result: dict[str, dict[str, Any]] = {}
    for item in payload.values() if isinstance(payload, dict) else []:
        ticker = str(item.get("ticker", "")).upper().strip()
        if ticker:
            result[ticker] = {"cik": int(item["cik_str"]), "title": item.get("title", ticker)}
    return result


def load_sec_company_facts(
    ticker: str,
    email: str,
    ticker_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    item = ticker_map.get(ticker.upper())
    if not item:
        raise LookupError(f"{ticker}: ticker is not in the SEC company ticker map")
    cik = int(item["cik"])
    payload = _sec_get_json(SEC_COMPANY_FACTS_URL.format(cik=f"{cik:010d}"), email)
    sic = None
    sic_description = ""
    try:
        submissions = _sec_get_json(SEC_SUBMISSIONS_URL.format(cik=f"{cik:010d}"), email)
        sic_value = submissions.get("sic")
        sic = int(sic_value) if sic_value not in (None, "") else None
        sic_description = str(submissions.get("sicDescription", "") or "")
    except Exception as exc:
        # The company facts endpoint remains usable if optional SIC metadata is unavailable.
        sic_description = f"SEC submissions metadata unavailable: {type(exc).__name__}"
    return {
        "ticker": ticker.upper(), "cik": cik, "title": item["title"],
        "payload": payload, "sic": sic, "sic_description": sic_description,
        "sic_group": sic_group(sic),
    }


def _concept_records(
    company_facts: dict[str, Any],
    tags: list[str],
    cutoff: str | date,
) -> list[dict[str, Any]]:
    """Return observations disclosed strictly before cutoff, for close-based decisions."""
    cutoff_text = pd.Timestamp(cutoff).date().isoformat()
    facts = company_facts.get("facts", {})
    taxonomies = [facts.get("us-gaap", {}), facts.get("dei", {}), facts.get("ifrs-full", {})]
    for tag in tags:
        for taxonomy in taxonomies:
            node = taxonomy.get(tag)
            if not node:
                continue
            records = []
            for unit, observations in node.get("units", {}).items():
                for obs in observations:
                    filed, end = str(obs.get("filed", "")), str(obs.get("end", ""))
                    # The feed has a date, not a dependable intraday filing timestamp.
                    if not filed or not end or filed >= cutoff_text or end > cutoff_text:
                        continue
                    row = dict(obs)
                    row["unit"], row["tag"] = unit, tag
                    records.append(row)
            if records:
                return records
    return []


def _preferred_xbrl_unit(records: list[dict[str, Any]]) -> str:
    """Choose one consistent XBRL unit, preferring the unit with most observations.

    USD is preferred only when observation counts tie. This does not convert other
    currencies into USD; it merely prevents mixing parallel units inside one metric.
    """
    units = sorted({str(row.get("unit", "")) for row in records if row.get("unit")})
    if not units:
        return ""
    counts = {
        unit: sum(1 for row in records if str(row.get("unit", "")) == unit)
        for unit in units
    }
    return sorted(
        units,
        key=lambda unit: (
            -counts[unit],
            0 if unit.upper() == "USD" else 1,
            1 if "/" in unit else 0,
            unit,
        ),
    )[0]


def _latest_instant_unit(
    company_facts: dict[str, Any], tags: list[str], cutoff: str | date
) -> str:
    records = _concept_records(company_facts, tags, cutoff)
    candidates = [row for row in records if not row.get("start")]
    return _preferred_xbrl_unit(candidates)


def _annual_series(company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> pd.Series:
    records = _concept_records(company_facts, tags, cutoff)
    chosen_unit = _preferred_xbrl_unit(records)
    if chosen_unit:
        records = [row for row in records if str(row.get("unit", "")) == chosen_unit]
    chosen: dict[str, dict[str, Any]] = {}
    for row in records:
        if not str(row.get("form", "")).startswith(("10-K", "20-F", "40-F")):
            continue
        start, end = row.get("start"), row.get("end")
        if not start:
            continue
        try:
            days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        except Exception:
            continue
        if not 320 <= days <= 410:
            continue
        key = str(end)
        if key not in chosen or str(row.get("filed", "")) > str(chosen[key].get("filed", "")):
            chosen[key] = row
    if not chosen:
        return pd.Series(dtype=float)
    pairs = {}
    for end, row in chosen.items():
        try:
            pairs[pd.Timestamp(end)] = float(row["val"])
        except (TypeError, ValueError, KeyError):
            continue
    result = pd.Series(pairs, dtype=float).sort_index() if pairs else pd.Series(dtype=float)
    result.attrs["unit"] = chosen_unit
    return result


def _latest_instant(
    company_facts: dict[str, Any], tags: list[str], cutoff: str | date
) -> tuple[float, str, str]:
    records = _concept_records(company_facts, tags, cutoff)
    candidates = [row for row in records if not row.get("start")]
    chosen_unit = _preferred_xbrl_unit(candidates)
    if chosen_unit:
        candidates = [row for row in candidates if str(row.get("unit", "")) == chosen_unit]
    if not candidates:
        return np.nan, "", ""
    candidates.sort(key=lambda row: (str(row.get("end", "")), str(row.get("filed", ""))))
    row = candidates[-1]
    try:
        value = float(row["val"])
    except (TypeError, ValueError, KeyError):
        value = np.nan
    return value, str(row.get("end", "")), str(row.get("filed", ""))


def _latest_annual_with_filing(
    series: pd.Series,
    company_facts: dict[str, Any],
    tags: list[str],
    cutoff: str | date,
) -> tuple[float, str, str]:
    records = _concept_records(company_facts, tags, cutoff)
    by_end: dict[str, dict[str, Any]] = {}
    for row in records:
        if not str(row.get("form", "")).startswith(("10-K", "20-F", "40-F")) or not row.get("start"):
            continue
        try:
            days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
        except Exception:
            continue
        if not 320 <= days <= 410:
            continue
        end = str(row.get("end", ""))
        if end not in by_end or str(row.get("filed", "")) > str(by_end[end].get("filed", "")):
            by_end[end] = row
    if series.empty:
        return np.nan, "", ""
    end_ts = series.index[-1].date().isoformat()
    row = by_end.get(end_ts, {})
    return float(series.iloc[-1]), end_ts, str(row.get("filed", ""))


def sec_fundamental_snapshot(
    fact_bundle: dict[str, Any],
    as_of: str | date,
    latest_price: float = np.nan,
    price_currency: str | None = None,
) -> dict[str, Any]:
    """Compute reported fundamental ratios using only records filed before as_of."""
    company_facts = fact_bundle["payload"]
    cutoff = pd.Timestamp(as_of).date()
    revenue_tags = [
        "RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    ]
    income_tags = ["NetIncomeLoss", "ProfitLoss"]
    cfo_tags = ["NetCashProvidedByUsedInOperatingActivities"]
    capex_tags = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
    revenue = _annual_series(company_facts, revenue_tags, cutoff)
    income = _annual_series(company_facts, income_tags, cutoff)
    cfo = _annual_series(company_facts, cfo_tags, cutoff)
    capex = _annual_series(company_facts, capex_tags, cutoff)

    revenue_unit = str(revenue.attrs.get("unit", "") or "")
    income_unit = str(income.attrs.get("unit", "") or "")
    cfo_unit = str(cfo.attrs.get("unit", "") or "")
    capex_unit = str(capex.attrs.get("unit", "") or "")
    revenue_last = float(revenue.iloc[-1]) if len(revenue) else np.nan
    revenue_prev = float(revenue.iloc[-2]) if len(revenue) > 1 else np.nan
    income_last = float(income.iloc[-1]) if len(income) else np.nan
    income_prev = float(income.iloc[-2]) if len(income) > 1 else np.nan
    cfo_last = float(cfo.iloc[-1]) if len(cfo) else np.nan
    same_end_capex = capex.get(cfo.index[-1], np.nan) if len(cfo) else np.nan
    fcf_unit = cfo_unit if cfo_unit and cfo_unit == capex_unit else ""
    fcf = (
        cfo_last - abs(float(same_end_capex))
        if pd.notna(cfo_last) and pd.notna(same_end_capex) and fcf_unit
        else np.nan
    )

    equity_tags = [
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    ]
    debt_noncurrent_tags = [
        "LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebt"
    ]
    debt_current_tags = [
        "LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent", "ShortTermBorrowings"
    ]
    equity, equity_end, equity_filed = _latest_instant(company_facts, equity_tags, cutoff)
    debt_noncurrent, debt_end, debt_filed = _latest_instant(company_facts, debt_noncurrent_tags, cutoff)
    debt_current, _, _ = _latest_instant(company_facts, debt_current_tags, cutoff)
    equity_unit = _latest_instant_unit(company_facts, equity_tags, cutoff)
    debt_noncurrent_unit = _latest_instant_unit(company_facts, debt_noncurrent_tags, cutoff)
    debt_current_unit = _latest_instant_unit(company_facts, debt_current_tags, cutoff)
    debt_unit = debt_noncurrent_unit if pd.notna(debt_noncurrent) else debt_current_unit
    if pd.notna(debt_noncurrent) and pd.notna(debt_current) and debt_noncurrent_unit != debt_current_unit:
        debt = np.nan
        debt_unit = ""
    else:
        debt = (
            max(0.0, debt_noncurrent) + max(0.0, debt_current)
            if pd.notna(debt_noncurrent) or pd.notna(debt_current) else np.nan
        )
    shares, shares_end, shares_filed = _latest_instant(company_facts, ["EntityCommonStockSharesOutstanding"], cutoff)
    price_currency_code = str(price_currency or "").strip().upper()
    price_financial_unit_matches = bool(
        price_currency_code and income_unit and price_currency_code == income_unit
    )
    market_cap = (
        float(latest_price) * shares
        if price_financial_unit_matches and pd.notna(latest_price) and pd.notna(shares)
        and latest_price > 0 and shares > 0
        else np.nan
    )

    def growth(current, previous):
        if pd.notna(current) and pd.notna(previous) and previous > 0:
            return (current / previous - 1) * 100
        return np.nan

    revenue_growth = growth(revenue_last, revenue_prev)
    earnings_growth = growth(income_last, income_prev)
    roe = (
        income_last / equity * 100
        if income_unit and income_unit == equity_unit and pd.notna(income_last) and pd.notna(equity) and equity > 0
        else np.nan
    )
    profit_margin = (
        income_last / revenue_last * 100
        if income_unit and income_unit == revenue_unit and pd.notna(income_last) and pd.notna(revenue_last) and revenue_last > 0
        else np.nan
    )
    fcf_margin = (
        fcf / revenue_last * 100
        if fcf_unit and fcf_unit == revenue_unit and pd.notna(fcf) and pd.notna(revenue_last) and revenue_last > 0
        else np.nan
    )
    debt_equity = (
        debt / equity * 100
        if debt_unit and debt_unit == equity_unit and pd.notna(debt) and pd.notna(equity) and equity > 0
        else np.nan
    )
    pe = (
        market_cap / income_last
        if price_financial_unit_matches and pd.notna(market_cap) and pd.notna(income_last) and income_last > 0
        else np.nan
    )
    fcf_yield = (
        fcf / market_cap * 100
        if price_currency_code and price_currency_code == fcf_unit and pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0
        else np.nan
    )
    latest_period = revenue.index[-1].date().isoformat() if len(revenue) else ""
    revenue_filed = _latest_annual_with_filing(revenue, company_facts, revenue_tags, cutoff)[2]
    income_filed = _latest_annual_with_filing(income, company_facts, income_tags, cutoff)[2]
    cfo_filed = _latest_annual_with_filing(cfo, company_facts, cfo_tags, cutoff)[2]
    filing_dates = [d for d in [revenue_filed, income_filed, cfo_filed, equity_filed, debt_filed, shares_filed] if d]
    latest_filed = max(filing_dates) if filing_dates else ""
    fields = [revenue_last, income_last, revenue_growth, earnings_growth, roe, profit_margin, fcf, fcf_margin]
    coverage_pct = sum(bool(pd.notna(v)) for v in fields) / len(fields) * 100
    status = "OK" if coverage_pct >= 50 and len(filing_dates) >= 3 else "Partial SEC coverage"

    sic_code = fact_bundle.get("sic")
    sic_description = str(fact_bundle.get("sic_description", "") or "")
    sic_category = fact_bundle.get("sic_group") or sic_group(sic_code)
    return {
        "Ticker": fact_bundle["ticker"], "Company": fact_bundle["title"],
        "SEC SIC": sic_code, "SEC SIC Description": sic_description, "SIC group": sic_category,
        "Source": "SEC EDGAR Company Facts", "As of": cutoff.isoformat(),
        "Financial period": latest_period, "Latest filing date": latest_filed,
        "Revenue": revenue_last, "Revenue unit": revenue_unit, "Revenue growth %": revenue_growth,
        "Net income": income_last, "Net income unit": income_unit, "Earnings growth %": earnings_growth,
        "ROE %": roe, "Profit margin %": profit_margin,
        "CFO": cfo_last, "CFO unit": cfo_unit, "Capex unit": capex_unit,
        "FCF": fcf, "FCF unit": fcf_unit, "FCF margin %": fcf_margin,
        "Debt": debt, "Debt unit": debt_unit, "Equity": equity, "Equity unit": equity_unit, "Debt/Equity %": debt_equity,
        "Shares outstanding (reported)": shares, "Estimated market cap": market_cap,
        "P/E (annual EPS proxy)": pe, "FCF yield %": fcf_yield,
        "Price currency": price_currency_code or "UNKNOWN",
        "Price / net-income unit match": price_financial_unit_matches,
        "Fundamental coverage %": coverage_pct, "Fundamental status": status,
        "Data note": (
            "Annual values use one selected XBRL unit per metric. Ratios are suppressed where financial units differ. "
            "Market cap/P-E/FCF yield require price-currency confirmation; this app does not convert currencies. "
            "Current price × last reported shares remains an estimate; facts are filtered by filing date."
        ),
    }


def _all_filing_dates(company_facts: dict[str, Any]) -> list[date]:
    """Collect filed dates once so historical snapshots can be recalculated only on filing events."""
    found: set[date] = set()
    for taxonomy in company_facts.get("facts", {}).values():
        for node in taxonomy.values():
            for observations in node.get("units", {}).values():
                for row in observations:
                    filed = row.get("filed")
                    if not filed:
                        continue
                    try:
                        found.add(pd.Timestamp(filed).date())
                    except (TypeError, ValueError):
                        continue
    return sorted(found)


def _reprice_snapshot(snapshot: dict[str, Any], as_of: date, latest_price: float) -> dict[str, Any]:
    """Update only price-dependent valuation fields while financial disclosures are unchanged."""
    row = dict(snapshot)
    row["As of"] = as_of.isoformat()
    shares = pd.to_numeric(pd.Series([row.get("Shares outstanding (reported)", np.nan)]), errors="coerce").iloc[0]
    income = pd.to_numeric(pd.Series([row.get("Net income", np.nan)]), errors="coerce").iloc[0]
    fcf = pd.to_numeric(pd.Series([row.get("FCF", np.nan)]), errors="coerce").iloc[0]
    market_cap = (
        float(latest_price) * float(shares)
        if pd.notna(latest_price) and latest_price > 0 and pd.notna(shares) and shares > 0
        else np.nan
    )
    row["Estimated market cap"] = market_cap
    row["P/E (annual EPS proxy)"] = market_cap / income if pd.notna(market_cap) and pd.notna(income) and income > 0 else np.nan
    row["FCF yield %"] = fcf / market_cap * 100 if pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0 else np.nan
    return row


def sec_snapshot_to_history(
    fact_bundle: dict[str, Any],
    dates: list[pd.Timestamp],
    prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Build filing-date-safe monthly snapshots, recalculating fundamentals on new filing events only.

    This avoids repeatedly scanning a full SEC XBRL payload for every month in a multi-year test.
    A filing dated on the decision day is withheld until a later decision date to avoid close-time leakage.
    """
    requested = pd.DatetimeIndex(pd.to_datetime(dates), name="Decision date")
    if requested.empty:
        return pd.DataFrame(index=requested)
    ordered = requested.sort_values().unique()
    filing_dates = _all_filing_dates(fact_bundle["payload"])
    price_series = None
    if prices is not None:
        price_series = pd.to_numeric(prices, errors="coerce").copy()
        price_series.index = pd.to_datetime(price_series.index)
        if price_series.index.tz is not None:
            price_series.index = price_series.index.tz_localize(None)
        price_series = price_series.sort_index()

    rows_by_date: dict[pd.Timestamp, dict[str, Any]] = {}
    prior_calculated_date = None
    last_snapshot: dict[str, Any] | None = None
    for ts in ordered:
        cutoff = pd.Timestamp(ts).date()
        latest_price = np.nan
        if price_series is not None:
            known = price_series.loc[price_series.index <= ts].dropna()
            if not known.empty:
                latest_price = float(known.iloc[-1])
        has_new_filing = prior_calculated_date is None or any(
            prior_calculated_date < filed < cutoff for filed in filing_dates
        )
        if last_snapshot is None or has_new_filing:
            last_snapshot = sec_fundamental_snapshot(fact_bundle, cutoff, latest_price)
        else:
            last_snapshot = _reprice_snapshot(last_snapshot, cutoff, latest_price)
        rows_by_date[pd.Timestamp(ts)] = dict(last_snapshot)
        prior_calculated_date = cutoff

    out_rows = [rows_by_date[pd.Timestamp(ts)] for ts in requested]
    return pd.DataFrame(out_rows, index=requested)
