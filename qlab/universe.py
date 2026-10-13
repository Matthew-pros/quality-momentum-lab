"""Point-in-time equity-universe membership interval helpers."""
from __future__ import annotations

import pandas as pd


def parse_universe_membership(source) -> pd.DataFrame:
    """Parse membership intervals with inclusive start/end dates.

    Required columns are Ticker and Start date; End date is optional/blank for an
    open-ended membership. A ticker may have several non-overlapping membership
    intervals. Overlap is rejected because it usually indicates an ambiguous
    historical universe definition.
    """
    try:
        raw = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read universe membership CSV: {type(exc).__name__}: {exc}") from exc
    raw.columns = [str(column).strip() for column in raw.columns]
    aliases = {str(column).strip().lower().replace("_", " "): column for column in raw.columns}
    ticker_col = aliases.get("ticker") or aliases.get("symbol")
    start_col = aliases.get("start date") or aliases.get("start")
    end_col = aliases.get("end date") or aliases.get("end")
    if ticker_col is None or start_col is None:
        raise ValueError(
            "Universe membership CSV requires Ticker and Start date columns; End date is optional."
        )
    ticker_values = raw[ticker_col].astype("string").str.upper().str.strip()
    if ticker_values.isna().any() or ticker_values.isin(["", "NAN", "<NA>", "NONE"]).any():
        raise ValueError("Universe membership contains an empty Ticker.")

    start_text = raw[start_col].astype("string").str.strip()
    start_blank = raw[start_col].isna() | start_text.eq("")
    start_values = pd.to_datetime(start_text.mask(start_blank), errors="coerce", utc=True)
    if start_blank.any() or start_values.isna().any():
        raise ValueError("Every membership interval needs a valid Start date.")
    start_values = start_values.dt.tz_localize(None).dt.normalize()

    if end_col:
        end_text = raw[end_col].astype("string").str.strip()
        end_blank = raw[end_col].isna() | end_text.eq("")
        end_values = pd.to_datetime(end_text.mask(end_blank), errors="coerce", utc=True)
        malformed_end = (~end_blank) & end_values.isna()
        if malformed_end.any():
            raise ValueError("A non-empty End date must be a valid date; leave it blank only for open-ended membership.")
        end_values = end_values.dt.tz_localize(None).dt.normalize()
        end_values = end_values.mask(end_blank)
    else:
        end_values = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")

    frame = pd.DataFrame({
        "Ticker": ticker_values.astype(str),
        "Start date": start_values,
        "End date": end_values,
    })
    invalid = frame["End date"].notna() & frame["End date"].lt(frame["Start date"])
    if invalid.any():
        raise ValueError("End date cannot be earlier than Start date.")
    frame = frame.sort_values(["Ticker", "Start date"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["Ticker", "Start date"]).any():
        raise ValueError("Duplicate Ticker / Start date interval detected.")
    for ticker, group in frame.groupby("Ticker", sort=False):
        prior_end = None
        prior_open = False
        for row in group.itertuples(index=False):
            start = row[1]
            end = row[2]
            if prior_open:
                raise ValueError(f"{ticker}: an open-ended membership interval cannot be followed by another interval.")
            if prior_end is not None and start <= prior_end:
                raise ValueError(f"{ticker}: membership intervals overlap on or before {start.date()}.")
            prior_end = end if pd.notna(end) else None
            prior_open = pd.isna(end)
    return frame[["Ticker", "Start date", "End date"]]


def universe_members_asof(history: pd.DataFrame | None, as_of) -> set[str] | None:
    """Return active members on the inclusive as-of date, or None for static-universe mode."""
    if history is None:
        return None
    if history.empty:
        return set()
    date = pd.Timestamp(as_of)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    date = date.normalize()
    start = pd.to_datetime(history["Start date"], errors="coerce")
    end = pd.to_datetime(history["End date"], errors="coerce")
    if getattr(start.dt, "tz", None) is not None:
        start = start.dt.tz_localize(None)
    if getattr(end.dt, "tz", None) is not None:
        end = end.dt.tz_localize(None)
    active = (start.dt.normalize() <= date) & (end.isna() | (end.dt.normalize() >= date))
    return set(history.loc[active, "Ticker"].astype(str).str.upper())


def tickers_in_membership_file(history: pd.DataFrame | None) -> list[str]:
    """List distinct tickers included in a point-in-time membership file."""
    if history is None or history.empty:
        return []
    return sorted(set(history["Ticker"].astype(str).str.upper().str.strip()))
