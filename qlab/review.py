"""Snapshot-to-snapshot screening review for disciplined, low-turnover workflows."""
from __future__ import annotations

import numpy as np
import pandas as pd


SCORE_COLUMNS = ("Fundamental/composite score", "Composite score")
MOMENTUM_COLUMNS = ("Momentum 12-1M", "12-1M")
SIGNAL_COLUMNS = ("Signal", "Action", "Recommendation")
DATE_COLUMNS = ("Latest filing date", "Fundamental filing date", "Filing date")


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def _bool_series(series: pd.Series) -> pd.Series:
    def parse(value) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value) if np.isfinite(float(value)) else False
        return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}
    return series.map(parse).astype(bool)


def _normalize_snapshot(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Ticker"])
    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    if "Ticker" not in out:
        raise ValueError(f"{label} snapshot requires a Ticker column.")
    out["Ticker"] = out["Ticker"].astype(str).str.strip().str.upper()
    if out["Ticker"].eq("").any() or out["Ticker"].str.lower().isin({"nan", "none"}).any():
        raise ValueError(f"{label} snapshot contains a blank ticker.")
    duplicates = out.loc[out["Ticker"].duplicated(keep=False), "Ticker"].unique().tolist()
    if duplicates:
        raise ValueError(f"{label} snapshot contains duplicate ticker rows: {', '.join(duplicates[:10])}.")
    return out.set_index("Ticker", drop=True)


def compare_screener_snapshots(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    score_change_points: float = 7.0,
    momentum_change: float = 0.10,
) -> pd.DataFrame:
    """Compare two exported screener snapshots and return an explainable review queue.

    A flag is a prompt for human review, not an automatic buy/sell instruction.
    momentum_change is an absolute return change (0.10 = 10 percentage points).
    """
    for name, value in [("score_change_points", score_change_points), ("momentum_change", momentum_change)]:
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative.")

    cur = _normalize_snapshot(current, "Current")
    prev = _normalize_snapshot(previous, "Previous")
    tickers = sorted(set(cur.index) | set(prev.index))
    if not tickers:
        return pd.DataFrame(columns=[
            "Ticker", "Review action", "Material change", "Reasons",
            "Current score", "Previous score", "Score change",
            "Current 12-1M", "Previous 12-1M", "12-1M change",
        ])

    cur_score_col = _first_existing(current, SCORE_COLUMNS)
    prev_score_col = _first_existing(previous, SCORE_COLUMNS)
    cur_mom_col = _first_existing(current, MOMENTUM_COLUMNS)
    prev_mom_col = _first_existing(previous, MOMENTUM_COLUMNS)
    cur_signal_col = _first_existing(current, SIGNAL_COLUMNS)
    prev_signal_col = _first_existing(previous, SIGNAL_COLUMNS)
    cur_date_col = _first_existing(current, DATE_COLUMNS)
    prev_date_col = _first_existing(previous, DATE_COLUMNS)

    rows = []
    for ticker in tickers:
        current_row = cur.loc[ticker] if ticker in cur.index else None
        previous_row = prev.loc[ticker] if ticker in prev.index else None
        is_new = previous_row is None
        is_removed = current_row is None

        def value(row, col, default=np.nan):
            if row is None or col is None or col not in row.index:
                return default
            result = row[col]
            return default if pd.isna(result) else result

        current_score = pd.to_numeric(pd.Series([value(current_row, cur_score_col)]), errors="coerce").iloc[0]
        previous_score = pd.to_numeric(pd.Series([value(previous_row, prev_score_col)]), errors="coerce").iloc[0]
        score_delta = current_score - previous_score if pd.notna(current_score) and pd.notna(previous_score) else np.nan

        current_momentum = pd.to_numeric(pd.Series([value(current_row, cur_mom_col)]), errors="coerce").iloc[0]
        previous_momentum = pd.to_numeric(pd.Series([value(previous_row, prev_mom_col)]), errors="coerce").iloc[0]
        momentum_delta = current_momentum - previous_momentum if pd.notna(current_momentum) and pd.notna(previous_momentum) else np.nan

        current_trend = _bool_series(pd.Series([value(current_row, "Trend OK", False)])).iloc[0]
        previous_trend = _bool_series(pd.Series([value(previous_row, "Trend OK", False)])).iloc[0]
        low_breach = _bool_series(pd.Series([value(current_row, "Below 50d low", False)])).iloc[0]

        current_signal = str(value(current_row, cur_signal_col, "")).strip() if current_row is not None else ""
        previous_signal = str(value(previous_row, prev_signal_col, "")).strip() if previous_row is not None else ""
        signal_changed = bool(current_signal and previous_signal and current_signal != previous_signal)

        current_filing = str(value(current_row, cur_date_col, "")).strip() if current_row is not None else ""
        previous_filing = str(value(previous_row, prev_date_col, "")).strip() if previous_row is not None else ""
        filing_updated = False
        if current_filing and previous_filing:
            current_dt = pd.to_datetime(current_filing, errors="coerce")
            previous_dt = pd.to_datetime(previous_filing, errors="coerce")
            filing_updated = bool(pd.notna(current_dt) and pd.notna(previous_dt) and current_dt > previous_dt)
        elif current_filing and not previous_filing:
            filing_updated = True

        score_deteriorated = bool(pd.notna(score_delta) and score_delta <= -float(score_change_points))
        momentum_deteriorated = bool(pd.notna(momentum_delta) and momentum_delta <= -float(momentum_change))
        trend_weakened = bool(previous_trend and not current_trend)
        reasons = []
        if is_new:
            reasons.append("Ticker is new in current snapshot")
        elif is_removed:
            reasons.append("Ticker is absent from current snapshot; verify universe/data coverage")
        else:
            if low_breach:
                reasons.append("Close is below the prior 50-session low")
            if trend_weakened:
                reasons.append("Trend filter changed from OK to not OK")
            if score_deteriorated:
                reasons.append(f"Composite score fell by {abs(float(score_delta)):.1f} points")
            elif pd.notna(score_delta) and score_delta >= float(score_change_points):
                reasons.append(f"Composite score rose by {float(score_delta):.1f} points")
            if momentum_deteriorated:
                reasons.append(f"12-1M momentum deteriorated by {abs(float(momentum_delta))*100:.1f} percentage points")
            elif pd.notna(momentum_delta) and momentum_delta >= float(momentum_change):
                reasons.append(f"12-1M momentum improved by {float(momentum_delta)*100:.1f} percentage points")
            if signal_changed:
                reasons.append(f"Signal changed: {previous_signal} → {current_signal}")
            if filing_updated:
                reasons.append("A newer fundamental filing is present")

        material = bool(
            is_new or is_removed or low_breach or trend_weakened
            or score_deteriorated or momentum_deteriorated or signal_changed or filing_updated
            or (pd.notna(score_delta) and score_delta >= float(score_change_points))
            or (pd.notna(momentum_delta) and momentum_delta >= float(momentum_change))
        )
        if is_new:
            action = "NEW — review"
        elif is_removed:
            action = "NOT PRESENT — check data/universe"
        elif low_breach or trend_weakened:
            action = "EXIT REVIEW — thesis/technical break"
        elif score_deteriorated or momentum_deteriorated:
            action = "REVIEW — material deterioration"
        elif filing_updated:
            action = "FUNDAMENTALS REVIEW"
        elif signal_changed or (pd.notna(score_delta) and abs(score_delta) >= float(score_change_points)) or (pd.notna(momentum_delta) and abs(momentum_delta) >= float(momentum_change)):
            action = "REVIEW — material change"
        else:
            action = "NO MATERIAL CHANGE"

        rows.append({
            "Ticker": ticker,
            "Review action": action,
            "Material change": material,
            "Reasons": "; ".join(reasons) if reasons else "No configured material change detected",
            "Current score": float(current_score) if pd.notna(current_score) else np.nan,
            "Previous score": float(previous_score) if pd.notna(previous_score) else np.nan,
            "Score change": float(score_delta) if pd.notna(score_delta) else np.nan,
            "Current 12-1M": float(current_momentum) if pd.notna(current_momentum) else np.nan,
            "Previous 12-1M": float(previous_momentum) if pd.notna(previous_momentum) else np.nan,
            "12-1M change": float(momentum_delta) if pd.notna(momentum_delta) else np.nan,
            "Trend OK now": bool(current_trend) if current_row is not None else False,
            "Below 50d low now": bool(low_breach) if current_row is not None else False,
            "Signal changed": bool(signal_changed),
            "New filing": bool(filing_updated),
        })

    priority = {
        "EXIT REVIEW — thesis/technical break": 0,
        "REVIEW — material deterioration": 1,
        "NOT PRESENT — check data/universe": 2,
        "NEW — review": 3,
        "FUNDAMENTALS REVIEW": 4,
        "REVIEW — material change": 5,
        "NO MATERIAL CHANGE": 6,
    }
    out = pd.DataFrame(rows)
    out["_priority"] = out["Review action"].map(priority).fillna(99)
    return out.sort_values(["_priority", "Material change", "Ticker"], ascending=[True, False, True]).drop(columns="_priority").reset_index(drop=True)
