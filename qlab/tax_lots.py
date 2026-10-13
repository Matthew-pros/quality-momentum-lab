"""Tax-lot ledger utilities. Computes realized proceeds and cost basis, not statutory tax."""
from __future__ import annotations

from collections import defaultdict, deque
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = {"Date", "Ticker", "Type", "Quantity", "Price"}
OPTIONAL_COLUMNS = {"Fees", "Currency"}


def normalize_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Transaction file is missing columns: " + ", ".join(sorted(missing)))
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out["Type"] = out["Type"].astype(str).str.upper().str.strip()
    out["Quantity"] = pd.to_numeric(out["Quantity"], errors="coerce")
    out["Price"] = pd.to_numeric(out["Price"], errors="coerce")
    if "Fees" in out:
        out["Fees"] = pd.to_numeric(out["Fees"], errors="coerce").fillna(0.0)
    else:
        out["Fees"] = 0.0
    if "Currency" not in out:
        out["Currency"] = "USD"
    else:
        out["Currency"] = out["Currency"].fillna("USD").astype(str).str.upper()
    if out["Date"].isna().any():
        raise ValueError("One or more transaction dates could not be parsed.")
    if not out["Type"].isin(["BUY", "SELL"]).all():
        raise ValueError("Transaction Type must be BUY or SELL.")
    if (out["Quantity"] <= 0).any() or (out["Price"] < 0).any() or (out["Fees"] < 0).any():
        raise ValueError("Quantity must be > 0; price and fees cannot be negative.")
    if out["Ticker"].eq("").any():
        raise ValueError("Ticker cannot be empty.")
    return out.sort_values(["Date"]).reset_index(drop=True)


def fifo_ledger(frame: pd.DataFrame, as_of=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return realized trade matches, open lots, and annual realized P/L summaries in source currency."""
    tx = normalize_transactions(frame)
    if as_of is not None:
        tx = tx.loc[tx["Date"] <= pd.Timestamp(as_of)]
    lots = defaultdict(deque)
    realized = []
    for txrow in tx.to_dict("records"):
        ticker = txrow["Ticker"]
        qty = float(txrow["Quantity"])
        price = float(txrow["Price"])
        fees = float(txrow["Fees"])
        fee_per_share = fees / qty
        currency = str(txrow.get("Currency", "USD"))
        if txrow["Type"] == "BUY":
            lots[ticker].append({
                "Ticker": ticker, "Acquisition date": txrow["Date"],
                "Remaining quantity": qty, "Unit cost incl. fee": price + fee_per_share,
                "Currency": currency,
            })
            continue
        available = sum(lot["Remaining quantity"] for lot in lots[ticker])
        if qty > available + 1e-9:
            raise ValueError(
                f"SELL {ticker} on {txrow['Date'].date()} exceeds FIFO inventory "
                f"({qty:g} sold, {available:g} available). Short sales are not supported."
            )
        remaining = qty
        while remaining > 1e-9:
            lot = lots[ticker][0]
            matched = min(remaining, lot["Remaining quantity"])
            basis = matched * lot["Unit cost incl. fee"]
            proceeds = matched * price - matched * fee_per_share
            gain = proceeds - basis
            days = (txrow["Date"] - lot["Acquisition date"]).days
            holding_period_end = lot["Acquisition date"] + pd.DateOffset(years=3)
            passes_3y_test = pd.Timestamp(txrow["Date"]).normalize() > pd.Timestamp(holding_period_end).normalize()
            realized.append({
                "Ticker": ticker, "Sale date": txrow["Date"],
                "Acquisition date": lot["Acquisition date"],
                "Holding days": days,
                "Holding period > 3 years": bool(passes_3y_test),
                "Quantity matched": matched,
                "Unit sale price": price,
                "Gross sale proceeds": matched * price,
                "Allocated sale fees": matched * fee_per_share,
                "Cost basis incl. buy fees": basis, "Net proceeds": proceeds,
                "Realized gain/loss": gain, "Currency": currency,
            })
            lot["Remaining quantity"] -= matched
            remaining -= matched
            if lot["Remaining quantity"] <= 1e-9:
                lots[ticker].popleft()
    open_rows = []
    for ticker, queue in lots.items():
        for lot in queue:
            if lot["Remaining quantity"] > 1e-9:
                row = dict(lot)
                row["Holding days"] = (pd.Timestamp(as_of or tx["Date"].max()) - row["Acquisition date"]).days
                row["Cost basis remaining"] = row["Remaining quantity"] * row["Unit cost incl. fee"]
                open_rows.append(row)
    realized_df = pd.DataFrame(realized)
    open_df = pd.DataFrame(open_rows)
    if realized_df.empty:
        annual = pd.DataFrame(columns=[
            "Year", "Ticker", "Currency", "Realized gain/loss", "Matched quantity",
            "Gross sale proceeds", "Tax review status",
        ])
    else:
        realized_df["Year"] = realized_df["Sale date"].dt.year
        realized_df["Tax review status"] = "Review tax treatment and FX conversion"
        for year, group in realized_df.groupby("Year"):
            year_mask = realized_df["Year"].eq(year)
            currencies = set(group["Currency"].astype(str).str.upper())
            long_enough = year_mask & realized_df["Holding period > 3 years"].astype(bool)
            short_holds = year_mask & ~realized_df["Holding period > 3 years"].astype(bool)
            if currencies == {"CZK"}:
                annual_gross = float(group["Gross sale proceeds"].sum())
                if annual_gross <= 100_000:
                    realized_df.loc[year_mask, "Tax review status"] = (
                        "Potential value-test exemption: annual gross proceeds <= CZK 100,000; verify full securities portfolio"
                    )
                    continue

                if int(year) <= 2025:
                    long_gross = float(group.loc[group["Holding period > 3 years"].astype(bool), "Gross sale proceeds"].sum())
                    if long_gross > 40_000_000:
                        realized_df.loc[long_enough, "Tax review status"] = (
                            "Historic-year 40m CZK exemption cap may apply; proportional calculation and other exempt sales need review"
                        )
                    else:
                        realized_df.loc[long_enough, "Tax review status"] = (
                            "Potential 3-year time-test exemption; verify historic 40m CZK aggregate cap and eligibility"
                        )
                else:
                    realized_df.loc[long_enough, "Tax review status"] = (
                        "Potential 3-year time-test exemption (40m cap removed from 2026); verify full eligibility"
                    )
                    long_gross = float(group.loc[group["Holding period > 3 years"].astype(bool), "Gross sale proceeds"].sum())
                    if long_gross > 5_000_000:
                        realized_df.loc[long_enough, "Tax review status"] = (
                            "Potential time-test exemption; review notification duty for exempt income above CZK 5m"
                        )
                realized_df.loc[short_holds, "Tax review status"] = (
                    "Time test not met and value test exceeded: review taxable income, deductible costs, and applicable-year rules"
                )
            else:
                # The value test and historic limit are annual CZK tests. Without dated
                # FX conversion across the full tax household portfolio, do not decide them.
                if int(year) <= 2025:
                    historic_note = "; historic 40m CZK cap may apply"
                else:
                    historic_note = "; 40m CZK securities cap removed from 2026"
                realized_df.loc[long_enough, "Tax review status"] = (
                    "Potential 3-year time-test exemption; convert proceeds to CZK and review full portfolio"
                    + historic_note
                )
                realized_df.loc[short_holds, "Tax review status"] = (
                    "Holding period <= 3 years; annual value test needs full CZK proceeds/FX review"
                    + historic_note
                )
        annual = realized_df.groupby(["Year", "Ticker", "Currency"], as_index=False).agg(
            **{
                "Realized gain/loss": ("Realized gain/loss", "sum"),
                "Matched quantity": ("Quantity matched", "sum"),
                "Gross sale proceeds": ("Gross sale proceeds", "sum"),
                "Tax review status": ("Tax review status", lambda x: " | ".join(sorted(set(x)))),
            }
        )
    return realized_df, open_df, annual
