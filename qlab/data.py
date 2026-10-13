"""Data adapters for Quality Momentum Lab.

Yahoo Finance is used only for exploratory daily OHLCV in this adapter. Its
unofficial interface can be rate-limited and must not be assumed to grant
public redistribution rights. Fundamental fields come from official SEC EDGAR
Company Facts when the operator supplies a contact email for the SEC User-Agent.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests
import time
import yfinance as yf


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
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
            out.columns = [next((str(x) for x in col if str(x) in PRICE_FIELDS), str(col[-1])) for col in out.columns]
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


def download_market_data(tickers: str | list[str], period: str = "10y") -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download one non-threaded batch to avoid concurrent yfinance SQLite locks."""
    symbols = normalize_tickers(tickers)
    if not symbols:
        return {}, pd.DataFrame(columns=["Ticker", "Status", "Rows", "First date", "Last date", "Missing %", "Message"])
    raw = pd.DataFrame()
    first_error = ""
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
        first_error = f"{type(exc).__name__}: {exc}"

    prices: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, Any]] = []
    for ticker in symbols:
        frame = _extract_ticker_frame(raw, ticker, symbols)
        message = ""
        if frame.empty and first_error:
            message = first_error
        if not frame.empty:
            # Adj Close captures split/dividend adjustment. Adjust High/Low with the
            # same factor for indicator consistency while keeping raw Close visible.
            factor = (frame["Adj Close"] / frame["Close"]).replace([np.inf, -np.inf], np.nan)
            frame["Adj Open"] = frame["Open"] * factor if "Open" in frame else np.nan
            frame["Adj High"] = frame["High"] * factor if "High" in frame else np.nan
            frame["Adj Low"] = frame["Low"] * factor if "Low" in frame else np.nan
            prices[ticker] = frame
        reports.append({
            "Ticker": ticker,
            "Status": "OK" if not frame.empty else "MISSING",
            "Rows": int(frame["Close"].notna().sum()) if not frame.empty else 0,
            "First date": frame.index.min().date().isoformat() if not frame.empty else "",
            "Last date": frame.index.max().date().isoformat() if not frame.empty else "",
            "Missing %": round(float(frame["Close"].isna().mean() * 100), 2) if not frame.empty else 100.0,
            "Message": message if message else ("" if not frame.empty else "Provider returned no rows for this ticker"),
        })
    # Retry a partial batch only for missing symbols, sequentially. If every
    # symbol failed, it is probably a provider/rate-limit outage; do not hammer it.
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
                    factor = (frame["Adj Close"] / frame["Close"]).replace([np.inf, -np.inf], np.nan)
                    frame["Adj Open"] = frame["Open"] * factor if "Open" in frame else np.nan
                    frame["Adj High"] = frame["High"] * factor if "High" in frame else np.nan
                    frame["Adj Low"] = frame["Low"] * factor if "Low" in frame else np.nan
                    prices[ticker] = frame
            except Exception:
                continue
    report = pd.DataFrame(reports)
    for ticker in prices:
        if not report.empty and (report["Ticker"] == ticker).any():
            report.loc[report["Ticker"] == ticker, ["Status", "Rows", "First date", "Last date", "Missing %", "Message"]] = [
                "OK", len(prices[ticker]), prices[ticker].index.min().date().isoformat(),
                prices[ticker].index.max().date().isoformat(), round(prices[ticker]["Close"].isna().mean() * 100, 2), ""
            ]
    return prices, report


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
    values = payload.values() if isinstance(payload, dict) else []
    for item in values:
        ticker = str(item.get("ticker", "")).upper().strip()
        if ticker:
            result[ticker] = {
                "cik": int(item["cik_str"]),
                "title": item.get("title", ticker),
            }
    return result


def load_sec_company_facts(ticker: str, email: str, ticker_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item = ticker_map.get(ticker.upper())
    if not item:
        raise LookupError(f"{ticker}: ticker is not in the SEC company ticker map")
    cik = int(item["cik"])
    payload = _sec_get_json(SEC_COMPANY_FACTS_URL.format(cik=f"{cik:010d}"), email)
    return {"ticker": ticker.upper(), "cik": cik, "title": item["title"], "payload": payload}


def _concept_records(company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> list[dict[str, Any]]:
    """Return records for the first available US-GAAP/DEI tag known by cutoff."""
    cutoff_text = pd.Timestamp(cutoff).date().isoformat()
    facts = company_facts.get("facts", {})
    taxonomies = [facts.get("us-gaap", {}), facts.get("dei", {}), facts.get("ifrs-full", {})]
    for tag in tags:
        for taxonomy in taxonomies:
            node = taxonomy.get(tag)
            if not node:
                continue
            records: list[dict[str, Any]] = []
            units = node.get("units", {})
            for unit, observations in units.items():
                for obs in observations:
                    filed = str(obs.get("filed", ""))
                    end = str(obs.get("end", ""))
                    # Conservative close-based backtest convention: a filing dated
                    # on the decision day is assumed unavailable until the next session,
                    # because SEC Company Facts gives a filing date but not a reliable
                    # intraday timestamp for this purpose.
                    if not filed or not end or filed >= cutoff_text or end > cutoff_text:
                        continue
                    row = dict(obs)
                    row["unit"] = unit
                    row["tag"] = tag
                    records.append(row)
            if records:
                return records
    return []


def _annual_series(company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> pd.Series:
    records = _concept_records(company_facts, tags, cutoff)
    chosen: dict[str, dict[str, Any]] = {}
    for row in records:
        form = str(row.get("form", ""))
        if not (form.startswith("10-K") or form.startswith("20-F") or form.startswith("40-F")):
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
    result = pd.Series({pd.Timestamp(k): float(v["val"]) for k, v in chosen.items()})
    result.index = pd.DatetimeIndex(result.index)
    return result.sort_index()


def _latest_instant(company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> tuple[float, str, str]:
    records = _concept_records(company_facts, tags, cutoff)
    candidates = [r for r in records if not r.get("start")]
    if not candidates:
        return np.nan, "", ""
    candidates.sort(key=lambda r: (str(r.get("end", "")), str(r.get("filed", ""))))
    row = candidates[-1]
    return float(row["val"]), str(row.get("end", "")), str(row.get("filed", ""))


def _latest_annual_with_filing(series: pd.Series, company_facts: dict[str, Any], tags: list[str], cutoff: str | date) -> tuple[float, str, str]:
    records = _concept_records(company_facts, tags, cutoff)
    by_end: dict[str, dict[str, Any]] = {}
    for row in records:
        if not (str(row.get("form", "")).startswith(("10-K", "20-F", "40-F"))):
            continue
        if not row.get("start"):
            continue
        try:
            days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
        except Exception:
            continue
        if not 320 <= days <= 410 or pd.Timestamp(row["end"]) not in series.index:
            continue
        end = str(row["end"])
        if end not in by_end or str(row.get("filed", "")) > str(by_end[end].get("filed", "")):
            by_end[end] = row
    if series.empty:
        return np.nan, "", ""
    end_ts = series.index[-1]
    row = by_end.get(end_ts.date().isoformat(), {})
    return float(series.iloc[-1]), end_ts.date().isoformat(), str(row.get("filed", ""))


def sec_fundamental_snapshot(fact_bundle: dict[str, Any], as_of: str | date, latest_price: float = np.nan) -> dict[str, Any]:
    """Compute current or historical point-in-time ratios, filtering every fact by filed date."""
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
    shares, shares_end, shares_filed = _latest_instant(company_facts, [
        "EntityCommonStockSharesOutstanding"
    ], cutoff)
    market_cap = float(latest_price) * shares if pd.notna(latest_price) and pd.notna(shares) and latest_price > 0 and shares > 0 else np.nan

    def growth(curr, prev):
        if pd.notna(curr) and pd.notna(prev) and prev > 0:
            return (curr / prev - 1) * 100
        return np.nan

    latest_period = revenue.index[-1].date().isoformat() if len(revenue) else ""
    revenue_filed = _latest_annual_with_filing(revenue, company_facts, revenue_tags, cutoff)[2]
    income_filed = _latest_annual_with_filing(income, company_facts, income_tags, cutoff)[2]
    cfo_filed = _latest_annual_with_filing(cfo, company_facts, cfo_tags, cutoff)[2]
    filing_dates = [d for d in [revenue_filed, income_filed, cfo_filed, equity_filed, debt_filed, shares_filed] if d]
    latest_filed = max(filing_dates) if filing_dates else ""
    coverage_fields = [
        revenue_last, income_last, growth(revenue_last, revenue_prev),
        growth(income_last, income_prev),
        income_last / equity * 100 if pd.notna(income_last) and pd.notna(equity) and equity > 0 else np.nan,
        income_last / revenue_last * 100 if pd.notna(income_last) and pd.notna(revenue_last) and revenue_last > 0 else np.nan,
        fcf, fcf / revenue_last * 100 if pd.notna(fcf) and pd.notna(revenue_last) and revenue_last > 0 else np.nan,
    ]
    coverage_pct = sum(bool(pd.notna(value)) for value in coverage_fields) / len(coverage_fields) * 100
    status = "OK" if coverage_pct >= 50 and len(filing_dates) >= 3 else "Partial SEC coverage"

    return {
        "Ticker": fact_bundle["ticker"],
        "Company": fact_bundle["title"],
        "Source": "SEC EDGAR Company Facts",
        "As of": cutoff.isoformat(),
        "Financial period": latest_period,
        "Latest filing date": latest_filed,
        "Revenue": revenue_last,
        "Revenue growth %": growth(revenue_last, revenue_prev),
        "Net income": income_last,
        "Earnings growth %": growth(income_last, income_prev),
        "ROE %": income_last / equity * 100 if pd.notna(income_last) and pd.notna(equity) and equity > 0 else np.nan,
        "Profit margin %": income_last / revenue_last * 100 if pd.notna(income_last) and pd.notna(revenue_last) and revenue_last > 0 else np.nan,
        "CFO": cfo_last,
        "FCF": fcf,
        "FCF margin %": fcf / revenue_last * 100 if pd.notna(fcf) and pd.notna(revenue_last) and revenue_last > 0 else np.nan,
        "Debt": debt,
        "Equity": equity,
        "Debt/Equity %": debt / equity * 100 if pd.notna(debt) and pd.notna(equity) and equity > 0 else np.nan,
        "Shares outstanding (reported)": shares,
        "Estimated market cap": market_cap,
        "P/E (annual EPS proxy)": market_cap / income_last if pd.notna(market_cap) and pd.notna(income_last) and income_last > 0 else np.nan,
        "FCF yield %": fcf / market_cap * 100 if pd.notna(fcf) and pd.notna(market_cap) and market_cap > 0 else np.nan,
        "Fundamental coverage %": coverage_pct,
        "Fundamental status": status,
        "Data note": "Current market price × latest reported shares is an estimate; annual ratios use the latest annual filing public by the as-of date.",
    }


def sec_snapshot_to_history(
    fact_bundle: dict[str, Any],
    dates: list[pd.Timestamp],
    prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Build filing-date-safe annual fundamental snapshots at requested decision dates."""
    rows = []
    for ts in dates:
        price = np.nan
        if prices is not None:
            known = prices.loc[prices.index <= ts].dropna()
            if not known.empty:
                price = float(known.iloc[-1])
        rows.append(sec_fundamental_snapshot(fact_bundle, ts.date(), price))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates, name="Decision date"))
