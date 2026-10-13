"""Data adapters for Quality Momentum Lab.

Yahoo Finance is experimental OHLCV only through the unofficial yfinance interface.
SEC Company Facts supplies standardized reported facts for mapped SEC filers.
"""
from __future__ import annotations

from datetime import date
import time
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
PRICE_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def normalize_tickers(text_or_tickers: str | list[str]) -> list[str]:
    raw = text_or_tickers.split(",") if isinstance(text_or_tickers, str) else text_or_tickers
    return list(dict.fromkeys(str(x).strip().upper() for x in raw if str(x).strip()))


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
    """Add deterministic freshness, invalid-price, and extreme-return flags per ticker."""
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
        close = pd.to_numeric(frame["Close"], errors="coerce")
        invalid = (~np.isfinite(close) | (close <= 0))
        if "High" in frame and "Low" in frame:
            high = pd.to_numeric(frame["High"], errors="coerce")
            low = pd.to_numeric(frame["Low"], errors="coerce")
            invalid |= high.notna() & low.notna() & (high < low)
        ret_col = "Adj Close" if "Adj Close" in frame else "Close"
        ret = pd.to_numeric(frame[ret_col], errors="coerce").pct_change(fill_method=None)
        ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
        max_move = float(ret.abs().max() * 100) if not ret.empty else np.nan
        last_date = pd.Timestamp(frame.index.max())
        if last_date.tzinfo:
            last_date = last_date.tz_localize(None)
        stale = max(0, int((now_ts - last_date.normalize()).days))
        invalid_count = int(invalid.sum())
        flags = []
        if stale > 7:
            flags.append(f"stale {stale} calendar days")
        if invalid_count:
            flags.append(f"{invalid_count} invalid price rows")
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
    return {"ticker": ticker.upper(), "cik": cik, "title": item["title"], "payload": payload}


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


def _annual_series(company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> pd.Series:
    records = _concept_records(company_facts, tags, cutoff)
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
    return pd.Series(pairs, dtype=float).sort_index() if pairs else pd.Series(dtype=float)


def _latest_instant(
    company_facts: dict[str, Any], tags: list[str], cutoff: str | date
) -> tuple[float, str, str]:
    records = _concept_records(company_facts, tags, cutoff)
    candidates = [row for row in records if not row.get("start")]
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

    revenue_last = float(revenue.iloc[-1]) if len(revenue) else np.nan
    revenue_prev = float(revenue.iloc[-2]) if len(revenue) > 1 else np.nan
    income_last = float(income.iloc[-1]) if len(income) else np.nan
    income_prev = float(income.iloc[-2]) if len(income) > 1 else np.nan
    cfo_last = float(cfo.iloc[-1]) if len(cfo) else np.nan
    same_end_capex = capex.get(cfo.index[-1], np.nan) if len(cfo) else np.nan
    fcf = cfo_last - abs(float(same_end_capex)) if pd.notna(cfo_last) and pd.notna(same_end_capex) else np.nan

    equity, equity_end, equity_filed = _latest_instant(company_facts, [
        "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
    ], cutoff)
    debt_noncurrent, debt_end, debt_filed = _latest_instant(company_facts, [
        "LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent", "LongTermDebt"
    ], cutoff)
    debt_current, _, _ = _latest_instant(company_facts, [
        "LongTermDebtCurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent", "ShortTermBorrowings"
    ], cutoff)
    debt = (
        max(0.0, debt_noncurrent) + max(0.0, debt_current)
        if pd.notna(debt_noncurrent) or pd.notna(debt_current) else np.nan
    )
    shares, shares_end, shares_filed = _latest_instant(company_facts, ["EntityCommonStockSharesOutstanding"], cutoff)
    market_cap = (
        float(latest_price) * shares
        if pd.notna(latest_price) and pd.notna(shares) and latest_price > 0 and shares > 0
        else np.nan
    )

    def growth(current, previous):
        if pd.notna(current) and pd.notna(previous) and previous > 0:
            return (current / previous - 1) * 100
        return np.nan

    revenue_growth = growth(revenue_last, revenue_prev)
    earnings_growth = growth(income_last, income_prev)
    roe = income_last / equity * 100 if pd.notna(income_last) and pd.notna(equity) and equity > 0 else np.nan
    profit_margin = income_last / revenue_last * 100 if pd.notna(income_last) and pd.notna(revenue_last) and revenue_last > 0 else np.nan
    fcf_margin = fcf / revenue_last * 100 if pd.notna(fcf) and pd.notna(revenue_last) and revenue_last > 0 else np.nan
    debt_equity = debt / equity * 100 if pd.notna(debt) and pd.notna(equity) and equity > 0 else np.nan
    pe = market_cap / income_last if pd.notna(market_cap) and pd.notna(income_last) and income_last > 0 else np.nan
    fcf_yield = fcf / market_cap * 100 if pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0 else np.nan
    latest_period = revenue.index[-1].date().isoformat() if len(revenue) else ""
    revenue_filed = _latest_annual_with_filing(revenue, company_facts, revenue_tags, cutoff)[2]
    income_filed = _latest_annual_with_filing(income, company_facts, income_tags, cutoff)[2]
    cfo_filed = _latest_annual_with_filing(cfo, company_facts, cfo_tags, cutoff)[2]
    filing_dates = [d for d in [revenue_filed, income_filed, cfo_filed, equity_filed, debt_filed, shares_filed] if d]
    latest_filed = max(filing_dates) if filing_dates else ""
    fields = [revenue_last, income_last, revenue_growth, earnings_growth, roe, profit_margin, fcf, fcf_margin]
    coverage_pct = sum(bool(pd.notna(v)) for v in fields) / len(fields) * 100
    status = "OK" if coverage_pct >= 50 and len(filing_dates) >= 3 else "Partial SEC coverage"

    return {
        "Ticker": fact_bundle["ticker"], "Company": fact_bundle["title"],
        "Source": "SEC EDGAR Company Facts", "As of": cutoff.isoformat(),
        "Financial period": latest_period, "Latest filing date": latest_filed,
        "Revenue": revenue_last, "Revenue growth %": revenue_growth,
        "Net income": income_last, "Earnings growth %": earnings_growth,
        "ROE %": roe, "Profit margin %": profit_margin,
        "CFO": cfo_last, "FCF": fcf, "FCF margin %": fcf_margin,
        "Debt": debt, "Equity": equity, "Debt/Equity %": debt_equity,
        "Shares outstanding (reported)": shares, "Estimated market cap": market_cap,
        "P/E (annual EPS proxy)": pe, "FCF yield %": fcf_yield,
        "Fundamental coverage %": coverage_pct, "Fundamental status": status,
        "Data note": "Current price × last reported shares is only an estimate; annual ratios use annual filing facts available before the as-of date.",
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
