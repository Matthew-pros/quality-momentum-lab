"""Timestamped optional signals imported by CSV; no fabricated sentiment or revisions."""
from __future__ import annotations

import pandas as pd
import numpy as np


def parse_signal_csv(file, score_column: str) -> pd.DataFrame:
    frame = pd.read_csv(file)
    required = {"Date", "Ticker", score_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Signal CSV must include: {', '.join(sorted(required))}. Missing: {', '.join(sorted(missing))}")
    out = frame.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Ticker"] = out["Ticker"].astype(str).str.upper().str.strip()
    out[score_column] = pd.to_numeric(out[score_column], errors="coerce")
    if out["Date"].isna().any() or out["Ticker"].eq("").any():
        raise ValueError("Signal file contains invalid Date or Ticker values.")
    if out[score_column].isna().any() or not out[score_column].between(0, 100).all():
        raise ValueError(f"{score_column} must be numeric on a 0–100 scale.")
    if "Source" not in out:
        out["Source"] = "User uploaded"
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def latest_signals(signals: pd.DataFrame, score_column: str, as_of=None) -> dict[str, dict]:
    if signals is None or signals.empty:
        return {}
    end = pd.Timestamp(as_of or pd.Timestamp.now(tz=None))
    s = signals.loc[signals["Date"] <= end]
    if s.empty:
        return {}
    s = s.sort_values("Date").groupby("Ticker", as_index=False).tail(1)
    return {
        row["Ticker"]: {score_column: float(row[score_column]), f"{score_column} date": row["Date"].date().isoformat(), f"{score_column} source": row["Source"]}
        for row in s.to_dict("records")
    }


def signal_history(signals: pd.DataFrame | None, score_column: str, decision_dates) -> dict[str, pd.DataFrame]:
    """Forward-fill last known score at each decision date without ever using future uploads."""
    if signals is None or signals.empty:
        return {}
    dates = pd.DatetimeIndex(decision_dates)
    result = {}
    for ticker, group in signals.groupby("Ticker"):
        series = group.set_index("Date")[score_column].sort_index()
        series = series[~series.index.duplicated(keep="last")]
        idx = series.index.union(dates).sort_values()
        expanded = series.reindex(idx).ffill().reindex(dates)
        result[str(ticker)] = pd.DataFrame({score_column: expanded.to_numpy()}, index=dates)
    return result
