"""Portfolio-held position review queue with thesis/technical exit flags."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_bool(value) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            return None
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true","1","yes","y","t"}:
        return True
    if text in {"false","0","no","n","f"}:
        return False
    return None


def build_holdings_review(
    screener: pd.DataFrame,
    held_tickers: list[str],
    exit_score_floor: float = 35.0,
    min_fundamental_coverage: float = 50.0,
    check_fundamentals: bool = True,
) -> pd.DataFrame:
    """Review only the positions the user already holds; do not make automatic orders.

    Price-data or membership gaps are surfaced as a data review, never translated into
    an automatic exit. A 50-session-low break or explicit trend break takes priority.
    """
    for name, value in [
        ("exit_score_floor", exit_score_floor),
        ("min_fundamental_coverage", min_fundamental_coverage),
    ]:
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if not 0 <= float(exit_score_floor) <= 100:
        raise ValueError("exit_score_floor must be between 0 and 100.")
    if not 0 <= float(min_fundamental_coverage) <= 100:
        raise ValueError("min_fundamental_coverage must be between 0 and 100.")

    held = [str(t).strip().upper() for t in held_tickers if str(t).strip()]
    if len(held) != len(set(held)):
        raise ValueError("Held ticker list contains duplicates; enter each held position once.")
    if not held:
        return pd.DataFrame(columns=[
            "Ticker", "Review action", "Reasons", "Current score", "Fundamental coverage %",
            "Trend OK", "Below 50d low", "Signal",
        ])

    if screener is None or screener.empty or "Ticker" not in screener:
        lookup = pd.DataFrame(index=pd.Index([], name="Ticker"))
    else:
        frame = screener.copy()
        frame["Ticker"] = frame["Ticker"].astype(str).str.strip().str.upper()
        if frame["Ticker"].duplicated().any():
            duplicates = frame.loc[frame["Ticker"].duplicated(keep=False), "Ticker"].unique().tolist()
            raise ValueError("Screener contains duplicate tickers: " + ", ".join(duplicates[:10]))
        lookup = frame.set_index("Ticker", drop=True)

    rows = []
    priority = {
        "EXIT REVIEW — technical rule broken":0,
        "DATA REVIEW — holding missing":1,
        "FUNDAMENTALS REVIEW — coverage incomplete":2,
        "THESIS REVIEW — score below floor":3,
        "HOLD / no configured exit trigger":4,
    }
    for ticker in held:
        if ticker not in lookup.index:
            rows.append({
                "Ticker":ticker,
                "Review action":"DATA REVIEW — holding missing",
                "Reasons":"Ticker is not in the current price universe/screener. Verify symbol, membership, provider coverage and delisting status before considering any trade.",
                "Current score":np.nan,
                "Fundamental coverage %":np.nan,
                "Trend OK":None,
                "Below 50d low":None,
                "Signal":"",
            })
            continue

        row = lookup.loc[ticker]
        score_value = row.get("Fundamental/composite score", row.get("Composite score", np.nan))
        if pd.isna(score_value) and "Technical score" in row:
            score_value = row.get("Technical score", np.nan)
        try:
            score = float(score_value)
            if not np.isfinite(score):
                score = np.nan
        except (TypeError, ValueError):
            score = np.nan
        try:
            coverage = float(row.get("Fundamental coverage %", np.nan))
            if not np.isfinite(coverage):
                coverage = np.nan
        except (TypeError, ValueError):
            coverage = np.nan

        trend = _as_bool(row.get("Trend OK")) if "Trend OK" in row else None
        below_low = _as_bool(row.get("Below 50d low")) if "Below 50d low" in row else None
        signal = str(row.get("Signal", "") or "")
        reasons = []

        if below_low is True:
            reasons.append("Close has breached the prior 50-session low")
        if trend is False:
            reasons.append("Trend filter is no longer OK")
        if score <= float(exit_score_floor):
            reasons.append(f"Composite score is at/below the review floor ({exit_score_floor:.1f})")
        if trend is None:
            reasons.append("Trend status is missing; verify price history/indicators before deciding")
        if below_low is None:
            reasons.append("50-session-low status is missing; verify price history/indicators before deciding")
        if check_fundamentals and (pd.isna(coverage) or coverage < float(min_fundamental_coverage)):
            reasons.append(
                f"Fundamental coverage is incomplete ({'missing' if pd.isna(coverage) else f'{coverage:.0f}'}%; "
                f"review threshold {min_fundamental_coverage:.0f}%)"
            )

        if below_low is True or trend is False:
            action = "EXIT REVIEW — technical rule broken"
        elif trend is None or below_low is None:
            action = "DATA REVIEW — holding missing"
        elif check_fundamentals and (pd.isna(coverage) or coverage < float(min_fundamental_coverage)):
            action = "FUNDAMENTALS REVIEW — coverage incomplete"
        elif pd.isna(score) or score <= float(exit_score_floor):
            action = "THESIS REVIEW — score below floor"
        else:
            action = "HOLD / no configured exit trigger"
        rows.append({
            "Ticker":ticker,
            "Review action":action,
            "Reasons":"; ".join(reasons) if reasons else "No configured technical/score exit trigger detected",
            "Current score":score,
            "Fundamental coverage %":coverage,
            "Trend OK":trend,
            "Below 50d low":below_low,
            "Signal":signal,
        })

    out = pd.DataFrame(rows)
    out["_priority"] = out["Review action"].map(priority).fillna(99)
    return out.sort_values(["_priority", "Ticker"]).drop(columns="_priority").reset_index(drop=True)
