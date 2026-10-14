"""Technical indicators, cross-sectional scoring, portfolio rules, and research backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

from qlab.universe import universe_members_asof


def _num(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators on a split/dividend-adjusted OHLC basis where possible."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    raw_close = _num(frame.get("Close", frame.get("Adj Close", pd.Series(dtype=float))))
    if raw_close.empty:
        return pd.DataFrame()
    adjusted_close = _num(frame.get("Adj Close", raw_close)).reindex(raw_close.index)
    factor = (adjusted_close / raw_close).replace([np.inf, -np.inf], np.nan)
    close = adjusted_close

    def adjusted_ohlc(raw_name: str, adjusted_name: str) -> pd.Series:
        if adjusted_name in frame:
            values = _num(frame[adjusted_name]).reindex(close.index)
        elif raw_name in frame:
            values = _num(frame[raw_name]).reindex(close.index) * factor
        else:
            values = close.copy()
        return values.fillna(close)

    high = adjusted_ohlc("High", "Adj High")
    low = adjusted_ohlc("Low", "Adj Low")
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = true_range.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx14 = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean().clip(lower=0, upper=100)
    atr20 = true_range.ewm(alpha=1/20, min_periods=20, adjust=False).mean()

    out = pd.DataFrame(index=close.index)
    out["Close"] = close
    out["Adj Close"] = adjusted_close
    out["Close raw"] = raw_close
    out["High"] = high
    out["Low"] = low
    out["MA50"] = close.rolling(50, min_periods=50).mean()
    out["MA200"] = close.rolling(200, min_periods=200).mean()
    out["MA200_prev20"] = close.rolling(200, min_periods=200).mean().shift(20)
    out["Momentum 12-1M"] = close.shift(21) / close.shift(252) - 1
    out["Momentum 6-1M"] = close.shift(21) / close.shift(126) - 1
    out["ADX14"] = adx14
    out["ATR14"] = atr14
    # ATR is stored on the adjusted-price basis; prepare_screener converts the latest
    # value back to the raw quote scale for position sizing.
    out["ATR20"] = atr20
    out["ATR20%"] = atr20 / close * 100
    out["High20 prior"] = high.shift(1).rolling(20, min_periods=20).max()
    out["High55 prior"] = high.shift(1).rolling(55, min_periods=55).max()
    out["Low50 prior"] = low.shift(1).rolling(50, min_periods=50).min()
    out["Breakout20"] = close > out["High20 prior"]
    out["Breakout55"] = close > out["High55 prior"]
    out["Trend OK"] = (close > out["MA200"]) & (out["MA50"] > out["MA200"]) & (out["MA200"] > out["MA200_prev20"])
    out["Below 50d low"] = close < out["Low50 prior"]
    out["Daily return"] = close.pct_change(fill_method=None)
    out["Volatility20%"] = out["Daily return"].rolling(20, min_periods=20).std() * np.sqrt(252) * 100
    return out.replace([np.inf, -np.inf], np.nan)

def _precompute_indicator_cache(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute daily indicators once for a fixed price dataset shared by parameter runs."""
    cache: dict[str, pd.DataFrame] = {}
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        source_col = "Adj Close" if "Adj Close" in frame else "Close"
        if source_col not in frame or len(_num(frame[source_col]).dropna()) < 300:
            continue
        cache[ticker] = calculate_indicators(frame)
    return cache


def percentile_score(series: pd.Series, higher_better: bool = True) -> pd.Series:
    """Rank valid observations only; missing inputs remain missing, never fake-neutral."""
    values = _num(series)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.notna()
    if valid.sum() == 1:
        result.loc[valid] = 50.0
    elif valid.sum() >= 2:
        result.loc[valid] = values.loc[valid].rank(pct=True, ascending=higher_better) * 100
    return result


def weighted_average(frame: pd.DataFrame, weights: dict[str, float], default=np.nan) -> pd.Series:
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        values = _num(frame[column])
        valid = values.notna()
        numerator.loc[valid] += values.loc[valid] * float(weight)
        denominator.loc[valid] += float(weight)
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(default)


def score_cross_section(rows: pd.DataFrame, sentiment_enabled: bool = False, revisions_enabled: bool = False) -> pd.DataFrame:
    """Create factor scores. Each factor ranks only observed values in the same decision universe."""
    df = rows.copy()
    sentiment_enabled = sentiment_enabled or ("Sentiment score" in df and df["Sentiment score"].notna().any())
    revisions_enabled = revisions_enabled or ("Revision score" in df and df["Revision score"].notna().any())
    for col in [
        "ROE %", "Profit margin %", "FCF margin %", "Revenue growth %",
        "Earnings growth %", "FCF yield %", "P/E (annual EPS proxy)",
        "Momentum 12-1M", "Momentum 6-1M", "ADX14", "ATR20%",
    ]:
        if col not in df:
            df[col] = np.nan

    df["Quality score"] = weighted_average(pd.DataFrame({
        "ROE": percentile_score(df["ROE %"]),
        "Margin": percentile_score(df["Profit margin %"]),
        "FCF margin": percentile_score(df["FCF margin %"]),
    }, index=df.index), {"ROE": 0.35, "Margin": 0.30, "FCF margin": 0.35})

    positive_pe = _num(df["P/E (annual EPS proxy)"]).where(_num(df["P/E (annual EPS proxy)"]) > 0)
    df["Value score"] = weighted_average(pd.DataFrame({
        "FCF yield": percentile_score(df["FCF yield %"]),
        "PE": percentile_score(positive_pe, higher_better=False),
    }, index=df.index), {"FCF yield": 0.65, "PE": 0.35})

    df["Growth score"] = weighted_average(pd.DataFrame({
        "Revenue growth": percentile_score(df["Revenue growth %"]),
        "Earnings growth": percentile_score(df["Earnings growth %"]),
    }, index=df.index), {"Revenue growth": 0.5, "Earnings growth": 0.5})

    df["Momentum score"] = weighted_average(pd.DataFrame({
        "12-1M": percentile_score(df["Momentum 12-1M"]),
        "6-1M": percentile_score(df["Momentum 6-1M"]),
    }, index=df.index), {"12-1M": 0.5, "6-1M": 0.5})

    trend = df.get("Trend OK", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    df["Trend score"] = np.where(
        trend, 50 + _num(df["ADX14"]).clip(0, 50).fillna(0), 0.0
    )
    price_parts = pd.DataFrame({
        "Momentum": df["Momentum score"],
        "Trend": df["Trend score"],
        "Breakout20": df.get("Breakout20", pd.Series(False, index=df.index)).astype(float) * 100,
        "Breakout55": df.get("Breakout55", pd.Series(False, index=df.index)).astype(float) * 100,
    }, index=df.index)
    df["Technical score"] = weighted_average(price_parts, {
        "Momentum": 0.55, "Trend": 0.25, "Breakout20": 0.10, "Breakout55": 0.10
    })

    composite_parts = pd.DataFrame({
        "Quality": df["Quality score"],
        "Value": df["Value score"],
        "Growth": df["Growth score"],
        "Momentum": df["Momentum score"],
        "Trend": df["Trend score"],
    }, index=df.index)
    df["Fundamental/composite score"] = weighted_average(composite_parts, {
        "Quality": 0.27, "Value": 0.20, "Growth": 0.18, "Momentum": 0.22, "Trend": 0.13
    })
    fundamental_cols = ["ROE %", "Profit margin %", "FCF margin %", "Revenue growth %", "Earnings growth %", "FCF yield %"]
    df["Fundamental coverage %"] = df[fundamental_cols].notna().mean(axis=1) * 100
    no_fundamentals = df["Fundamental coverage %"].fillna(0) <= 0
    df.loc[no_fundamentals, "Fundamental/composite score"] = df.loc[no_fundamentals, "Technical score"]
    # Optional timestamped sources adjust the final base score but are never filled with neutral fakes.
    if sentiment_enabled and "Sentiment score" in df:
        df["Fundamental/composite score"] = weighted_average(pd.DataFrame({
            "Base": df["Fundamental/composite score"],
            "Sentiment": _num(df["Sentiment score"]),
        }, index=df.index), {"Base": 0.92, "Sentiment": 0.08})
    if revisions_enabled and "Revision score" in df:
        df["Fundamental/composite score"] = weighted_average(pd.DataFrame({
            "Base": df["Fundamental/composite score"],
            "Revisions": _num(df["Revision score"]),
        }, index=df.index), {"Base": 0.85, "Revisions": 0.15})
    df["Score status"] = np.where(
        df["Fundamental coverage %"] >= 60, "Fundamentals + price",
        np.where(df["Fundamental coverage %"] > 0, "Partial fundamentals / price", "Price-only score")
    )
    below_low = df.get("Below 50d low", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    breakout = df.get("Breakout55", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    score = _num(df["Fundamental/composite score"]).fillna(_num(df["Technical score"]))
    df["Signal"] = np.select(
        [below_low, trend & breakout & (score >= 65), trend, df["Momentum score"].notna()],
        ["Review exit: below 50-day low", "Breakout candidate", "Trend watch", "Momentum watch"],
        default="Insufficient price history",
    )
    if df["Fundamental coverage %"].max(skipna=True) == 0:
        df["Signal"] = df["Signal"].astype(str) + " · no fundamentals"
    return df


def prepare_screener(
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, dict],
    optional_signals: dict[str, dict] | None = None,
    allow_estimated_valuation: bool = False,
) -> pd.DataFrame:
    rows = []
    optional_signals = optional_signals or {}
    for ticker, frame in prices.items():
        indicators = calculate_indicators(frame)
        if indicators.empty:
            continue
        available = indicators.dropna(subset=["Close"])
        if available.empty:
            continue
        latest = available.iloc[-1]
        f = fundamentals.get(ticker, {})
        row = {
            "Ticker": ticker,
            "Company": f.get("Company", ticker),
            "SIC group": f.get("SIC group", "Unclassified"),
            "SEC SIC": f.get("SEC SIC", np.nan),
            "SEC SIC Description": f.get("SEC SIC Description", ""),
            "Source": f.get("Source", "Prices only"),
            "Financial period": f.get("Financial period", ""),
            "Latest filing date": f.get("Latest filing date", ""),
            "Price date": available.index[-1].date().isoformat(),
            "Price": float(frame.loc[available.index[-1], "Close"]) if "Close" in frame else float(latest["Close"]),
            "Market cap estimate": f.get("Estimated market cap", np.nan),
            "Revenue": f.get("Revenue", np.nan),
            "Net income": f.get("Net income", np.nan),
            "P/E (annual EPS proxy)": f.get("P/E (annual EPS proxy)", np.nan),
            "ROE %": f.get("ROE %", np.nan),
            "Profit margin %": f.get("Profit margin %", np.nan),
            "FCF": f.get("FCF", np.nan),
            "FCF margin %": f.get("FCF margin %", np.nan),
            "FCF yield %": f.get("FCF yield %", np.nan),
            "Revenue growth %": f.get("Revenue growth %", np.nan),
            "Earnings growth %": f.get("Earnings growth %", np.nan),
            "Debt/Equity %": f.get("Debt/Equity %", np.nan),
            "Fundamental status": f.get("Fundamental status", "Not loaded"),
            "Momentum 12-1M": latest.get("Momentum 12-1M", np.nan),
            "Momentum 6-1M": latest.get("Momentum 6-1M", np.nan),
            "MA50": latest.get("MA50", np.nan),
            "MA200": latest.get("MA200", np.nan),
            "MA200 rising": bool(latest.get("MA200", np.nan) > latest.get("MA200_prev20", np.nan)) if pd.notna(latest.get("MA200", np.nan)) and pd.notna(latest.get("MA200_prev20", np.nan)) else False,
            "ADX14": latest.get("ADX14", np.nan),
            "ATR20": (
                float(latest.get("ATR20", np.nan)) /
                float(latest.get("Adj Close", np.nan) / frame.loc[available.index[-1], "Close"])
                if pd.notna(latest.get("ATR20", np.nan))
                and pd.notna(latest.get("Adj Close", np.nan))
                and pd.notna(frame.loc[available.index[-1], "Close"])
                and frame.loc[available.index[-1], "Close"] > 0
                and latest.get("Adj Close", 0) > 0
                else latest.get("ATR20", np.nan)
            ),
            "ATR20%": latest.get("ATR20%", np.nan),
            "Breakout20": bool(latest.get("Breakout20", False)),
            "Breakout55": bool(latest.get("Breakout55", False)),
            "Below 50d low": bool(latest.get("Below 50d low", False)),
            "Trend OK": bool(latest.get("Trend OK", False)),
        }
        row.update(optional_signals.get(ticker, {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if not allow_estimated_valuation:
        scored = score_cross_section(df.assign(**{
            "FCF yield %": np.nan,
            "P/E (annual EPS proxy)": np.nan,
        }), sentiment_enabled="Sentiment score" in df and df["Sentiment score"].notna().any(),
           revisions_enabled="Revision score" in df and df["Revision score"].notna().any())
        for column in ["Quality score", "Value score", "Growth score", "Momentum score", "Trend score", "Technical score", "Fundamental/composite score", "Fundamental coverage %", "Score status", "Signal"]:
            df[column] = scored[column]
    else:
        df = score_cross_section(df, sentiment_enabled="Sentiment score" in df and df["Sentiment score"].notna().any(),
                                 revisions_enabled="Revision score" in df and df["Revision score"].notna().any())
    return df


def select_with_hysteresis(
    scores: pd.Series,
    eligible: pd.Series,
    previous: set[str],
    top_n: int,
    min_improvement: float = 7.0,
    exit_score_floor: float = 30.0,
    max_replacements: int = 2,
    sector_map: dict[str, str] | None = None,
    max_sector_positions: int | None = None,
) -> set[str]:
    valid = scores.dropna()
    valid = valid[eligible.reindex(valid.index).fillna(False).astype(bool)]
    ranked = valid.sort_values(ascending=False, kind="mergesort")
    ranked_names = [ticker for ticker in sorted(ranked.index, key=lambda t: (-float(ranked[t]), str(t)))]
    sectors = sector_map or {}
    sector_limit = int(max_sector_positions) if max_sector_positions is not None else None

    def sector_of(ticker):
        value = str(sectors.get(str(ticker), "Unclassified") or "Unclassified").strip()
        return value if value and value.lower() not in {"unclassified", "unknown", "none", "nan"} else None

    def may_add(ticker, selected):
        sector = sector_of(ticker)
        if sector_limit is None or sector_limit <= 0 or sector is None:
            return True
        same_sector = sum(1 for held in selected if sector_of(held) == sector)
        return same_sector < sector_limit

    def initial_fill():
        chosen = []
        for ticker in ranked_names:
            if may_add(ticker, chosen):
                chosen.append(ticker)
            if len(chosen) >= top_n:
                break
        return set(chosen)

    if not previous:
        # Initial construction fills the target count where possible; no replacement cap yet.
        return initial_fill()

    incumbents = [
        ticker for ticker in previous
        if ticker in ranked.index and float(ranked[ticker]) >= exit_score_floor
    ]
    # Stable score/ticker ordering also gives a deterministic choice when a sector cap
    # forces an existing concentration to be reduced.
    current = []
    for ticker in sorted(incumbents, key=lambda name: (-float(ranked[name]), str(name))):
        if may_add(ticker, current):
            current.append(ticker)
    replacements = 0

    # Mandatory exits can leave vacancies. Limit replacement entries to the configured count.
    for ticker in ranked_names:
        if len(current) >= top_n or replacements >= max_replacements:
            break
        if ticker not in current and may_add(ticker, current):
            current.append(ticker)
            replacements += 1

    while len(current) >= top_n and top_n > 0 and replacements < max_replacements:
        incumbents = [t for t in current if t in ranked.index]
        if not incumbents:
            break
        weakest = min(incumbents, key=lambda t: (float(ranked[t]), str(t)))
        challenger = None
        for candidate in ranked_names:
            if candidate in current:
                continue
            trial = [t for t in current if t != weakest]
            if may_add(candidate, trial):
                challenger = candidate
                break
        if challenger is None or float(ranked[challenger]) < float(ranked[weakest]) + min_improvement:
            break
        current.remove(weakest)
        current.append(challenger)
        replacements += 1
    return set(current[:top_n])


def _build_monthly_cross_section(
    prices: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    fundamentals_history: dict[str, pd.DataFrame] | None,
    optional_history: dict[str, pd.DataFrame] | None,
    indicator_cache: dict[str, pd.DataFrame] | None = None,
    universe_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    active_tickers = universe_members_asof(universe_history, signal_date)
    for ticker, frame in prices.items():
        if active_tickers is not None and ticker not in active_tickers:
            continue
        daily = (indicator_cache or {}).get(ticker)
        if daily is None:
            daily = calculate_indicators(frame)
        history = daily.loc[daily.index <= signal_date]
        if history.empty:
            continue
        r = history.iloc[-1]
        if pd.isna(r.get("Momentum 12-1M")) or pd.isna(r.get("Momentum 6-1M")):
            continue
        base = {
            "Ticker": ticker,
            "Momentum 12-1M": r.get("Momentum 12-1M", np.nan),
            "Momentum 6-1M": r.get("Momentum 6-1M", np.nan),
            "ADX14": r.get("ADX14", np.nan),
            "ATR20%": r.get("ATR20%", np.nan),
            "Trend OK": bool(r.get("Trend OK", False)),
            "Breakout20": bool(r.get("Breakout20", False)),
            "Breakout55": bool(r.get("Breakout55", False)),
            "Below 50d low": bool(r.get("Below 50d low", False)),
        }
        fh = (fundamentals_history or {}).get(ticker)
        if fh is not None and not fh.empty:
            known = fh.loc[fh.index <= signal_date]
            if not known.empty:
                snap = known.iloc[-1].to_dict()
                for key in [
                    "ROE %", "Profit margin %", "FCF margin %", "FCF yield %",
                    "P/E (annual EPS proxy)", "Revenue growth %", "Earnings growth %",
                ]:
                    base[key] = snap.get(key, np.nan)
        oh = (optional_history or {}).get(ticker)
        if oh is not None and not oh.empty:
            known = oh.loc[oh.index <= signal_date]
            if not known.empty:
                for key, value in known.iloc[-1].to_dict().items():
                    base[key] = value
        rows.append(base)
    return score_cross_section(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _precompute_monthly_cross_sections(
    prices: dict[str, pd.DataFrame],
    fundamentals_history: dict[str, pd.DataFrame] | None = None,
    optional_history: dict[str, pd.DataFrame] | None = None,
    indicator_cache: dict[str, pd.DataFrame] | None = None,
    universe_history: pd.DataFrame | None = None,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Score each point-in-time month once when evaluating multiple parameter candidates.

    Factor percentile ranks are cross-sectional and depend on source history/date, but
    not on min_score or hysteresis. Sharing this cache avoids recalculating and re-ranking
    the same universe for every fold/candidate combination.
    """
    adjusted: dict[str, pd.Series] = {}
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        source_col = "Adj Close" if "Adj Close" in frame else "Close"
        if source_col not in frame:
            continue
        series = _num(frame[source_col]).dropna()
        if len(series) >= 300:
            adjusted[ticker] = series
    if len(adjusted) < 2:
        return {}
    monthly = pd.DataFrame(adjusted).sort_index().resample("ME").last().dropna(how="all")
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    cache = indicator_cache or _precompute_indicator_cache(prices)
    for i in range(12, len(monthly) - 1):
        signal_date = monthly.index[i]
        out[pd.Timestamp(signal_date)] = _build_monthly_cross_section(
            prices, signal_date, fundamentals_history, optional_history,
            indicator_cache=cache, universe_history=universe_history
        )
    return out


def _prepare_shared_backtest_kwargs(
    prices: dict[str, pd.DataFrame],
    kwargs: dict,
) -> dict:
    """Precompute indicators and point-in-time scored cross-sections once per dataset."""
    prepared = dict(kwargs)
    membership = prepared.get("universe_history")
    if membership is not None:
        if not {"Ticker", "Start date", "End date"}.issubset(membership.columns):
            raise ValueError("universe_history must be parsed and contain Ticker, Start date, End date.")
        missing = sorted(set(membership["Ticker"].astype(str).str.upper()) - set(prices))
        if missing:
            raise ValueError(
                "Historical membership references ticker(s) without any OHLCV series: "
                + ", ".join(missing[:20])
                + (f" … and {len(missing) - 20} more" if len(missing) > 20 else "")
                + ". Supply their historical prices, including delisted securities."
            )
    indicators = prepared.get("precomputed_indicators")
    if indicators is None:
        indicators = _precompute_indicator_cache(prices)
        prepared["precomputed_indicators"] = indicators
    if prepared.get("precomputed_cross_sections") is None:
        use_fundamentals = bool(prepared.get("use_fundamentals", True))
        fundamentals = prepared.get("fundamentals_history") if use_fundamentals else None
        optional = prepared.get("optional_history")
        prepared["precomputed_cross_sections"] = _precompute_monthly_cross_sections(
            prices,
            fundamentals_history=fundamentals,
            optional_history=optional,
            indicator_cache=indicators,
            universe_history=prepared.get("universe_history"),
        )
    return prepared


def _target_weights(
    selected: list[str],
    cross_section: pd.DataFrame,
    method: str = "Equal weight",
    max_position_weight: float = 0.35,
) -> dict[str, float]:
    """Build target weights; inverse ATR weights are capped with any remainder left in cash."""
    if not selected:
        return {}
    if method != "Inverse ATR":
        return {ticker: 1.0 / len(selected) for ticker in selected}

    table = cross_section.set_index("Ticker")
    volatility = pd.to_numeric(table.reindex(selected).get("ATR20%", pd.Series(index=selected, dtype=float)), errors="coerce")
    volatility = volatility.where(volatility > 0)
    fallback = float(volatility.median()) if volatility.notna().any() else 1.0
    volatility = volatility.fillna(fallback)
    scores = (1.0 / volatility).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scores.sum() <= 0:
        return {ticker: 1.0 / len(selected) for ticker in selected}

    cap = float(np.clip(max_position_weight, 0.05, 1.0))
    remaining = list(selected)
    target: dict[str, float] = {ticker: 0.0 for ticker in selected}
    budget = 1.0
    while remaining and budget > 1e-12:
        active_scores = scores.reindex(remaining)
        raw = active_scores / active_scores.sum() if active_scores.sum() > 0 else pd.Series(1.0 / len(remaining), index=remaining)
        capped = [ticker for ticker in remaining if budget * float(raw[ticker]) > cap + 1e-12]
        if not capped:
            for ticker in remaining:
                target[ticker] = budget * float(raw[ticker])
            budget = 0.0
            break
        for ticker in capped:
            target[ticker] = cap
            budget -= cap
            remaining.remove(ticker)
        if budget < 0:
            budget = 0.0
    # Floating-point roundoff tolerance; unused budget represents cash when all caps bind.
    return target


def _simulate_position_month(
    frame: pd.DataFrame,
    indicators: pd.DataFrame,
    signal_date: pd.Timestamp,
    return_period_end: pd.Timestamp,
    stop_mode: str = "Month-end 50-day low",
    stop_atr_multiple: float = 3.0,
    prior_high_water: float | None = None,
) -> tuple[float, str, str, float]:
    """Simulate one holding segment on the same adjusted-price scale as indicators.

    Stop levels use information available through the previous close. In a gap below
    the stop, the model fills at the adjusted open; otherwise it fills at the stop
    level when the adjusted day's low crosses it. Daily OHLC cannot reveal the exact
    intraday ordering of high and low, so today's high never tightens today's stop.
    Returns segment return, exit reason, exit date and adjusted high-water mark.
    """
    if frame is None or frame.empty:
        return np.nan, "MISSING", "", np.nan
    frame = frame.sort_index()
    close_raw = _num(frame.get("Close", frame.get("Adj Close", pd.Series(dtype=float))))
    if close_raw.empty:
        return np.nan, "MISSING", "", np.nan
    adjusted_close = _num(frame.get("Adj Close", close_raw)).reindex(frame.index)
    factor = (adjusted_close / close_raw).replace([np.inf, -np.inf], np.nan)

    def adjusted_series(raw_name: str, adjusted_name: str) -> pd.Series:
        if adjusted_name in frame:
            values = _num(frame[adjusted_name]).reindex(frame.index)
        elif raw_name in frame:
            values = _num(frame[raw_name]).reindex(frame.index) * factor
        else:
            values = adjusted_close.copy()
        return values

    adjusted_open = adjusted_series("Open", "Adj Open")
    adjusted_high = adjusted_series("High", "Adj High")
    adjusted_low = adjusted_series("Low", "Adj Low")
    known = frame.index[frame.index <= signal_date]
    if len(known) == 0:
        return np.nan, "MISSING", "", np.nan
    entry_date = known[-1]
    entry_raw = close_raw.loc[entry_date]
    entry_adjusted = adjusted_close.loc[entry_date]
    if pd.isna(entry_raw) or not np.isfinite(float(entry_raw)) or float(entry_raw) <= 0:
        return np.nan, "MISSING", "", np.nan
    if pd.isna(entry_adjusted) or not np.isfinite(float(entry_adjusted)) or float(entry_adjusted) <= 0:
        entry_adjusted = entry_raw
    entry_adjusted = float(entry_adjusted)

    path = frame.loc[(frame.index > entry_date) & (frame.index <= return_period_end)]
    if path.empty:
        return np.nan, "MISSING", "", np.nan

    # High-water is explicitly kept on the adjusted price basis. A new position
    # starts at its entry close; an incumbent carries the saved high since entry.
    high_water = (
        float(prior_high_water)
        if prior_high_water is not None and np.isfinite(prior_high_water)
        else entry_adjusted
    )
    high_water = max(high_water, entry_adjusted)
    prior_atr = np.nan
    if indicators is not None and entry_date in indicators.index and "ATR20" in indicators:
        prior_atr = pd.to_numeric(pd.Series([indicators.loc[entry_date, "ATR20"]]), errors="coerce").iloc[0]

    daily_indicators = indicators if indicators is not None else pd.DataFrame()
    for dt in path.index:
        adjusted_bar_close = pd.to_numeric(pd.Series([adjusted_close.loc[dt]]), errors="coerce").iloc[0]
        adjusted_bar_open = pd.to_numeric(pd.Series([adjusted_open.loc[dt]]), errors="coerce").iloc[0]
        adjusted_bar_low = pd.to_numeric(pd.Series([adjusted_low.loc[dt]]), errors="coerce").iloc[0]
        adjusted_bar_high = pd.to_numeric(pd.Series([adjusted_high.loc[dt]]), errors="coerce").iloc[0]
        if (
            not np.isfinite(adjusted_bar_close) or adjusted_bar_close <= 0
            or not np.isfinite(adjusted_bar_open) or adjusted_bar_open <= 0
            or not np.isfinite(adjusted_bar_low) or adjusted_bar_low <= 0
        ):
            continue

        levels: list[tuple[str, float]] = []
        if stop_mode in {"ATR trailing stop", "ATR + daily 50-day low"} and np.isfinite(prior_atr) and prior_atr > 0:
            atr_level = high_water - float(stop_atr_multiple) * float(prior_atr)
            if np.isfinite(atr_level) and atr_level > 0:
                levels.append(("ATR trailing stop", atr_level))
        if stop_mode in {"Daily 50-day low", "ATR + daily 50-day low"} and dt in daily_indicators.index:
            low50 = pd.to_numeric(
                pd.Series([daily_indicators.loc[dt].get("Low50 prior", np.nan)]),
                errors="coerce",
            ).iloc[0]
            if np.isfinite(low50) and low50 > 0:
                levels.append(("Daily 50-day low", float(low50)))

        stop_level = max((level for _, level in levels), default=np.nan)
        if np.isfinite(stop_level) and (adjusted_bar_open <= stop_level or adjusted_bar_low <= stop_level):
            fill_adjusted = float(adjusted_bar_open if adjusted_bar_open <= stop_level else stop_level)
            reason_names = [
                name for name, level in levels
                if np.isclose(level, stop_level, rtol=1e-8, atol=1e-8)
            ]
            reason = " + ".join(reason_names) if reason_names else "Daily stop"
            return fill_adjusted / entry_adjusted - 1, reason, pd.Timestamp(dt).date().isoformat(), np.nan

        if np.isfinite(adjusted_bar_high) and adjusted_bar_high > 0:
            high_water = max(high_water, float(adjusted_bar_high))
        if dt in daily_indicators.index and "ATR20" in daily_indicators:
            current_atr = pd.to_numeric(
                pd.Series([daily_indicators.loc[dt, "ATR20"]]), errors="coerce"
            ).iloc[0]
            if np.isfinite(current_atr) and current_atr > 0:
                prior_atr = float(current_atr)

    last_date = path.index[-1]
    last_adjusted = adjusted_close.loc[last_date] if last_date in adjusted_close.index else np.nan
    if pd.isna(last_adjusted) or not np.isfinite(float(last_adjusted)) or float(last_adjusted) <= 0:
        return np.nan, "MISSING", "", np.nan
    return float(last_adjusted / entry_adjusted - 1), "HOLD", "", high_water


def run_monthly_backtest(
    prices: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    top_n: int = 5,
    cost_bps: float = 15,
    initial: float = 10_000,
    min_adx: float = 10,
    min_score: float = 35,
    hysteresis: float = 7,
    max_replacements: int = 2,
    use_fundamentals: bool = True,
    fundamentals_history: dict[str, pd.DataFrame] | None = None,
    optional_history: dict[str, pd.DataFrame] | None = None,
    require_breakout: str = "None",
    weighting_method: str = "Equal weight",
    max_position_weight: float = 0.35,
    stop_mode: str = "Month-end 50-day low",
    stop_atr_multiple: float = 3.0,
    sector_map: dict[str, str] | None = None,
    max_sector_positions: int | None = 2,
    commission_bps: float | None = None,
    half_spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    evaluation_start_date: pd.Timestamp | str | None = None,
    evaluation_end_date: pd.Timestamp | str | None = None,
    precomputed_indicators: dict[str, pd.DataFrame] | None = None,
    precomputed_cross_sections: dict[pd.Timestamp, pd.DataFrame] | None = None,
    universe_history: pd.DataFrame | None = None,
    initial_holdings: set[str] | list[str] | None = None,
    initial_weights: dict[str, float] | None = None,
    initial_stop_high_water: dict[str, float] | None = None,
    initial_holding_entry_dates: dict[str, pd.Timestamp] | None = None,
) -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame | None]:
    """Month-end signals; selected positions earn the next month return.

    Prices are adjusted for corporate actions if the provider has supplied Adj Close.
    Orders are modeled at the signal month's close, represented by close-to-close
    returns for the next month, with costs on gross traded notional. This is not an
    intraday fill simulation; the current symbol list can still create survivorship bias.
    """
    valid_stop_modes = {
        "No stop / month-end rebalancing",
        "Month-end 50-day low",
        "ATR trailing stop",
        "Daily 50-day low",
        "ATR + daily 50-day low",
    }
    if stop_mode not in valid_stop_modes:
        raise ValueError(f"Unsupported stop_mode: {stop_mode}")
    if stop_atr_multiple <= 0:
        raise ValueError("stop_atr_multiple must be positive.")
    components = [half_spread_bps, slippage_bps] + ([] if commission_bps is None else [commission_bps])
    if any(not np.isfinite(float(value)) or float(value) < 0 for value in components):
        raise ValueError("Commission, spread and slippage cost assumptions must be finite and non-negative.")
    effective_cost_bps = float(cost_bps) if commission_bps is None else float(commission_bps) + float(half_spread_bps) + float(slippage_bps)
    if effective_cost_bps < 0 or not np.isfinite(effective_cost_bps):
        raise ValueError("Total cost bps must be finite and non-negative.")
    if not prices:
        return None, None, None
    if universe_history is not None:
        required_membership_columns = {"Ticker", "Start date", "End date"}
        if not required_membership_columns.issubset(universe_history.columns):
            raise ValueError("universe_history must be parsed and contain Ticker, Start date, End date.")
        missing_membership_prices = sorted(set(universe_history["Ticker"].astype(str).str.upper()) - set(prices))
        if missing_membership_prices:
            raise ValueError(
                "Historical membership references ticker(s) without any OHLCV series: "
                + ", ".join(missing_membership_prices[:20])
                + (f" … and {len(missing_membership_prices) - 20} more" if len(missing_membership_prices) > 20 else "")
                + ". Supply their historical prices, including delisted securities."
            )
    adjusted = {}
    indicator_cache: dict[str, pd.DataFrame] = {}
    shared_indicators = precomputed_indicators or {}
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        col = "Adj Close" if "Adj Close" in frame else "Close"
        series = _num(frame[col]).dropna()
        if len(series) >= 300:
            adjusted[ticker] = series
            cached_indicators = shared_indicators.get(ticker)
            if cached_indicators is not None and not cached_indicators.empty:
                indicator_cache[ticker] = cached_indicators
            else:
                indicator_cache[ticker] = calculate_indicators(frame)
    if len(adjusted) < 2:
        return None, None, None
    daily_prices = pd.DataFrame(adjusted).sort_index()
    monthly = daily_prices.resample("ME").last()
    monthly = monthly.dropna(how="all")
    if len(monthly) < 15:
        return None, None, None
    monthly_returns = monthly.pct_change(fill_method=None)
    start_cutoff = pd.Timestamp(evaluation_start_date) if evaluation_start_date is not None else None
    end_cutoff = pd.Timestamp(evaluation_end_date) if evaluation_end_date is not None else None
    if monthly.index.tz is None:
        if start_cutoff is not None and start_cutoff.tzinfo is not None:
            start_cutoff = start_cutoff.tz_localize(None)
        if end_cutoff is not None and end_cutoff.tzinfo is not None:
            end_cutoff = end_cutoff.tz_localize(None)
    else:
        if start_cutoff is not None:
            start_cutoff = start_cutoff.tz_localize(monthly.index.tz) if start_cutoff.tzinfo is None else start_cutoff.tz_convert(monthly.index.tz)
        if end_cutoff is not None:
            end_cutoff = end_cutoff.tz_localize(monthly.index.tz) if end_cutoff.tzinfo is None else end_cutoff.tz_convert(monthly.index.tz)
    if start_cutoff is not None and end_cutoff is not None and start_cutoff > end_cutoff:
        raise ValueError("evaluation_start_date must be on or before evaluation_end_date.")

    if not np.isfinite(float(initial)) or float(initial) <= 0:
        raise ValueError("Initial equity must be a positive finite value.")
    previous_holdings = {str(t).strip().upper() for t in (initial_holdings or set()) if str(t).strip()}
    if len(previous_holdings) != len(initial_holdings or set()):
        raise ValueError("Initial holdings contain duplicate/blank tickers.")
    unknown_initial = sorted(previous_holdings - set(adjusted))
    if unknown_initial:
        raise ValueError("Initial holdings are missing price histories: " + ", ".join(unknown_initial[:10]))
    previous_weights = {str(t).strip().upper(): float(w) for t, w in (initial_weights or {}).items()}
    if set(previous_weights) != previous_holdings:
        if previous_holdings and not previous_weights:
            previous_weights = {ticker: 1.0 / len(previous_holdings) for ticker in previous_holdings}
        elif not previous_holdings and previous_weights:
            raise ValueError("Initial weights were supplied without initial holdings.")
        elif set(previous_weights) != previous_holdings:
            raise ValueError("Initial holdings and initial weight tickers must match exactly.")
    if any(not np.isfinite(w) or w < 0 for w in previous_weights.values()):
        raise ValueError("Initial portfolio weights must be finite and non-negative.")
    if sum(previous_weights.values()) > 1.0 + 1e-8:
        raise ValueError("Initial portfolio weights cannot exceed 100% of NAV; residual cash is allowed.")
    initial_high_water = {str(t).strip().upper(): float(v) for t, v in (initial_stop_high_water or {}).items()}
    if set(initial_high_water) - previous_holdings:
        raise ValueError("Initial trailing-stop high-water marks must belong to current holdings.")
    if any(not np.isfinite(v) or v <= 0 for v in initial_high_water.values()):
        raise ValueError("Initial trailing-stop high-water marks must be positive finite prices.")
    initial_entries = {}
    for ticker, value in (initial_holding_entry_dates or {}).items():
        name = str(ticker).strip().upper()
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        initial_entries[name] = stamp
    if set(initial_entries) - previous_holdings:
        raise ValueError("Initial holding-entry dates must belong to current holdings.")
    if previous_holdings and not initial_entries:
        initial_entries = {ticker: pd.Timestamp(monthly.index[start_idx]) for ticker in previous_holdings}
    if set(initial_entries) != previous_holdings:
        raise ValueError("Initial holding-entry dates must match initial holdings.")
    rows, audit = [], []
    equity = float(initial)
    # Persist the post-close high-water mark across rebalance months for existing positions.
    stop_high_water: dict[str, float] = initial_high_water.copy()
    holding_entry_dates: dict[str, pd.Timestamp] = initial_entries.copy()
    closed_holding_days: list[int] = []
    total_entry_events = 0
    total_exit_events = 0
    # First usable decision requires at least 252 trading days of history in the daily indicators.
    start_idx = 12
    for i in range(start_idx, len(monthly) - 1):
        decision_date = monthly.index[i]
        if start_cutoff is not None and decision_date < start_cutoff:
            continue
        if end_cutoff is not None and decision_date > end_cutoff:
            continue
        cached_cross = (precomputed_cross_sections or {}).get(pd.Timestamp(decision_date))
        cross = cached_cross.copy(deep=True) if cached_cross is not None else _build_monthly_cross_section(
            prices, decision_date,
            fundamentals_history if use_fundamentals else None,
            optional_history, indicator_cache=indicator_cache,
            universe_history=universe_history
        )
        if cross.empty:
            continue
        score_col = "Fundamental/composite score"
        cross[score_col] = cross[score_col].where(cross[score_col].notna(), cross["Technical score"])
        eligible = cross["Trend OK"].fillna(False).astype(bool) & (cross["ADX14"].fillna(0) >= min_adx)
        # Month-end and daily 50-day-low modes both avoid carrying a name into a
        # new month if its close has already breached the prior 50-session low.
        if stop_mode in {"Month-end 50-day low", "Daily 50-day low", "ATR + daily 50-day low"}:
            eligible &= ~cross["Below 50d low"].fillna(False).astype(bool)
        eligible &= cross[score_col].fillna(-np.inf) >= min_score
        if use_fundamentals:
            # Do not pretend a company with no disclosed data was factor-ranked.
            eligible &= cross["Fundamental coverage %"].fillna(0) >= 50
        if require_breakout == "20-day":
            eligible &= cross["Breakout20"].fillna(False).astype(bool)
        elif require_breakout == "55-day":
            eligible &= cross["Breakout55"].fillna(False).astype(bool)
        scores = cross.set_index("Ticker")[score_col]
        eligible.index = cross["Ticker"]
        picks = select_with_hysteresis(
            scores, eligible, previous_holdings, top_n,
            min_improvement=hysteresis, exit_score_floor=min_score,
            max_replacements=max_replacements,
            sector_map=sector_map, max_sector_positions=max_sector_positions,
        )
        row_return_date = monthly.index[i + 1]
        selected = sorted(picks)
        selected_set = set(selected)
        signal_exits = sorted(previous_holdings - selected_set)
        new_entries = sorted(selected_set - previous_holdings)
        low_at_signal = set(
            cross.loc[cross["Below 50d low"].fillna(False).astype(bool), "Ticker"].astype(str)
        )
        signal_exit_details = []
        for ticker in signal_exits:
            entry_date = holding_entry_dates.pop(ticker, decision_date)
            held_days = max(0, int((pd.Timestamp(decision_date) - pd.Timestamp(entry_date)).days))
            closed_holding_days.append(held_days)
            total_exit_events += 1
            if ticker in low_at_signal and stop_mode == "Month-end 50-day low":
                reason = "Month-end 50-day low"
            elif ticker in low_at_signal and stop_mode in {"Daily 50-day low", "ATR + daily 50-day low"}:
                reason = "Close below prior 50-day low at rebalance"
            else:
                reason = "Rebalance / eligibility"
            signal_exit_details.append(f"{ticker}:{reason}:{held_days}d")
        for ticker in new_entries:
            holding_entry_dates[ticker] = pd.Timestamp(decision_date)
            total_entry_events += 1
        next_returns = monthly_returns.loc[row_return_date, selected] if selected else pd.Series(dtype=float)
        new_weights = _target_weights(
            selected, cross, method=weighting_method, max_position_weight=max_position_weight
        )
        names = set(previous_weights) | set(new_weights)
        rebalance_turnover = sum(abs(new_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in names)

        # Run the chosen daily-bar stop variant. Month-end-only models continue to
        # use adjusted close-to-close returns; daily stops persist high-water marks.
        ticker_returns: dict[str, float] = {}
        ticker_exit_reason: dict[str, str] = {}
        ticker_exit_date: dict[str, str] = {}
        updated_high_water: dict[str, float] = {}
        daily_stop_modes = {"ATR trailing stop", "Daily 50-day low", "ATR + daily 50-day low"}
        for ticker in selected:
            if stop_mode in daily_stop_modes:
                prior_high = stop_high_water.get(ticker) if ticker in previous_holdings else None
                ret, reason, exit_date, high_water = _simulate_position_month(
                    prices[ticker], indicator_cache.get(ticker), decision_date, row_return_date,
                    stop_mode=stop_mode, stop_atr_multiple=stop_atr_multiple,
                    prior_high_water=prior_high,
                )
                if reason == "MISSING" or not np.isfinite(ret):
                    raise ValueError(
                        f"Missing daily price path for {ticker} from {decision_date.date()} "
                        f"through {row_return_date.date()}; daily-stop result halted."
                    )
                if reason == "HOLD" and pd.isna(next_returns.get(ticker, np.nan)):
                    raise ValueError(
                        f"No verified month-end return for {ticker} during {row_return_date.date()} "
                        "and no stop exit was recorded; result halted instead of forward-filling."
                    )
                ticker_returns[ticker] = float(ret)
                ticker_exit_reason[ticker] = reason
                ticker_exit_date[ticker] = exit_date
                if reason == "HOLD" and np.isfinite(high_water):
                    updated_high_water[ticker] = float(high_water)
            else:
                ret = next_returns.get(ticker, np.nan)
                if pd.isna(ret):
                    raise ValueError(
                        f"Missing monthly return for held ticker {ticker} during {row_return_date.date()}; "
                        "the result was stopped rather than silently ignoring a missing/delisted security."
                    )
                ticker_returns[ticker] = float(ret)
                ticker_exit_reason[ticker] = "HOLD"
                ticker_exit_date[ticker] = ""

        gross_return = float(sum(new_weights.get(ticker, 0.0) * ticker_returns[ticker] for ticker in selected)) if selected else 0.0
        stopped = {ticker for ticker in selected if ticker_exit_reason[ticker] != "HOLD"}
        stop_exit_details = []
        for ticker in sorted(stopped):
            entry_date = holding_entry_dates.pop(ticker, decision_date)
            exit_date = pd.Timestamp(ticker_exit_date[ticker]) if ticker_exit_date[ticker] else row_return_date
            held_days = max(0, int((exit_date - pd.Timestamp(entry_date)).days))
            closed_holding_days.append(held_days)
            total_exit_events += 1
            stop_exit_details.append(f"{ticker}:{ticker_exit_reason[ticker]}:{held_days}d")
        # A stop is a new sell trade during the month; charge costs on the estimated
        # market value actually liquidated, in addition to any rebalance trades.
        stop_turnover = sum(
            max(0.0, new_weights.get(ticker, 0.0) * (1 + ticker_returns[ticker]))
            for ticker in stopped
        )
        turnover = rebalance_turnover + stop_turnover
        transaction_cost = turnover * effective_cost_bps / 10_000
        stop_transaction_cost = stop_turnover * effective_cost_bps / 10_000
        net_return = gross_return - transaction_cost
        equity *= (1 + net_return)
        survivors = [ticker for ticker in selected if ticker not in stopped]
        drifted_weights: dict[str, float] = {}
        if 1 + net_return > 0:
            for ticker in survivors:
                drifted_weights[ticker] = (
                    new_weights.get(ticker, 0.0) * (1 + ticker_returns[ticker]) / (1 + net_return)
                )
        effective_exposure = sum(drifted_weights.values()) * 100
        sic_group_weights: dict[str, float] = {}
        for ticker, weight in drifted_weights.items():
            group = str((sector_map or {}).get(ticker, "Unclassified") or "Unclassified")
            sic_group_weights[group] = sic_group_weights.get(group, 0.0) + float(weight)
        max_sic_group_weight_pct = max(sic_group_weights.values(), default=0.0) * 100
        rows.append({
            "Date": row_return_date,
            "Signal date": decision_date,
            "Gross return": gross_return,
            "Transaction cost": transaction_cost,
            "Net return": net_return,
            "Equity": equity,
            "Holdings": ", ".join(selected),
            "Number of holdings": len(selected),
            "Entry events": len(new_entries),
            "Exit events": len(signal_exits) + len(stopped),
            "New entries": ", ".join(new_entries),
            "Signal exits": " | ".join(signal_exit_details),
            "Stops triggered": len(stopped),
            "Stop exits": " | ".join(stop_exit_details),
            "Stop dates": ", ".join(f"{ticker}:{ticker_exit_date[ticker]}" for ticker in sorted(stopped) if ticker_exit_date[ticker]),
            "Gross exposure %": effective_exposure,
            "Target weights": ", ".join(f"{ticker}:{new_weights[ticker]:.3f}" for ticker in selected),
            "Surviving weights": ", ".join(f"{ticker}:{drifted_weights[ticker]:.3f}" for ticker in sorted(drifted_weights)),
            "SIC group weights": " | ".join(f"{group}:{weight:.1%}" for group, weight in sorted(sic_group_weights.items())),
            "Max SIC group weight %": max_sic_group_weight_pct,
            "Distinct SIC groups": len(sic_group_weights),
            "Rebalance turnover": rebalance_turnover,
            "Stop turnover": stop_turnover,
            "Gross turnover": turnover,
            "Stop transaction cost": stop_transaction_cost,
            "Estimated cost (bps of portfolio)": transaction_cost * 10_000,
            "Selected score mean": float(scores.reindex(selected).mean()) if selected else np.nan,
        })
        audit.append({
            "Signal date": decision_date,
            "Return period": row_return_date,
            "Previous holdings": ", ".join(sorted(previous_holdings)),
            "Selected holdings": ", ".join(selected),
            "Target weights": ", ".join(f"{ticker}:{new_weights[ticker]:.3f}" for ticker in selected),
            "Ticker returns": ", ".join(f"{ticker}:{ticker_returns[ticker]:.4f}" for ticker in selected),
            "Exit rules": ", ".join(f"{ticker}:{ticker_exit_reason[ticker]}" for ticker in selected),
            "Stop dates": ", ".join(f"{ticker}:{ticker_exit_date[ticker]}" for ticker in selected if ticker_exit_date[ticker]),
            "Entry events": len(new_entries),
            "New entries": ", ".join(new_entries),
            "Signal exits": " | ".join(signal_exit_details),
            "Stops triggered": len(stopped),
            "Stop exits": " | ".join(stop_exit_details),
            "Stop mode": stop_mode,
            "Weighting method": weighting_method,
            "Max positions per SIC group": max_sector_positions if max_sector_positions else "Off",
            "Selected SIC groups": " | ".join(f"{ticker}:{(sector_map or {}).get(ticker, 'Unclassified')}" for ticker in selected),
            "Eligible names": ", ".join(sorted(eligible[eligible].index)),
            "Min ADX": min_adx,
            "Min score": min_score,
            "Hysteresis points": hysteresis,
            "Rebalance turnover": rebalance_turnover,
            "Stop turnover": stop_turnover,
            "Gross turnover": turnover,
            "Stop transaction cost": stop_transaction_cost,
            "Cost bps": effective_cost_bps,
            "Commission bps": commission_bps if commission_bps is not None else np.nan,
            "Half-spread bps": half_spread_bps,
            "Slippage bps": slippage_bps,
        })
        previous_holdings = set(survivors)
        previous_weights = drifted_weights
        stop_high_water = updated_high_water
    if not rows:
        return None, None, None
    curve = pd.DataFrame(rows).set_index("Date")
    number_of_months = len(curve)
    years = number_of_months / 12.0
    cagr = (curve["Equity"].iloc[-1] / float(initial)) ** (1 / years) - 1
    volatility = curve["Net return"].std(ddof=1) * np.sqrt(12)
    sharpe = (curve["Net return"].mean() * 12) / volatility if pd.notna(volatility) and volatility > 0 else np.nan
    wealth = np.r_[float(initial), curve["Equity"].to_numpy()]
    peaks = np.maximum.accumulate(wealth)[1:]
    drawdown = curve["Equity"].to_numpy() / peaks - 1
    metrics = {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe (rf=0)": sharpe,
        "Max drawdown": float(np.min(drawdown)),
        "Final equity": float(curve["Equity"].iloc[-1]),
        "Months": number_of_months,
        "Total transaction costs %": float(curve["Transaction cost"].sum() * 100),
        "Average monthly turnover %": float(curve["Gross turnover"].mean() * 100),
        "Average gross exposure %": float(curve["Gross exposure %"].mean()),
        "Entry events": int(total_entry_events),
        "Exit events": int(total_exit_events),
        "Average holding days (closed positions)": float(np.mean(closed_holding_days)) if closed_holding_days else np.nan,
        "Median holding days (closed positions)": float(np.median(closed_holding_days)) if closed_holding_days else np.nan,
        "Average max SIC group weight %": float(curve["Max SIC group weight %"].mean()),
    }
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.dropna().resample("ME").last().pct_change(fill_method=None)
        aligned = curve["Net return"].to_frame("Strategy").join(bench.rename("Benchmark"), how="inner").dropna()
        if not aligned.empty:
            metrics["Benchmark CAGR"] = float((1 + aligned["Benchmark"]).prod() ** (12 / len(aligned)) - 1)
            metrics["Benchmark period months"] = len(aligned)
            metrics["Strategy CAGR aligned"] = float((1 + aligned["Strategy"]).prod() ** (12 / len(aligned)) - 1)
            curve["Benchmark return"] = bench.reindex(curve.index)
            curve["Benchmark equity"] = float(initial) * (1 + bench.reindex(curve.index).fillna(0)).cumprod()
    curve.attrs["end_state"] = {
        "holdings": sorted(previous_holdings),
        "weights": dict(previous_weights),
        "stop_high_water": dict(stop_high_water),
        "holding_entry_dates": dict(holding_entry_dates),
        "equity": float(equity),
    }
    return curve, metrics, pd.DataFrame(audit)


def compare_stop_variants(
    prices: dict[str, pd.DataFrame],
    stop_modes: tuple[str, ...],
    **backtest_kwargs,
) -> pd.DataFrame:
    """Compare exit-rule variants with identical universe, factor and cost settings.

    Each variant is independently simulated. This is intended for model diagnostics,
    not for choosing a winner using the final holdout or as evidence of predictive edge.
    """
    rows = []
    common_kwargs = _prepare_shared_backtest_kwargs(prices, dict(backtest_kwargs))
    for mode in stop_modes:
        row: dict[str, object] = {"Stop mode": mode, "Status": "OK", "Error": ""}
        try:
            _, metrics, _ = run_monthly_backtest(prices=prices, stop_mode=mode, **common_kwargs)
            if metrics is None:
                row["Status"] = "Insufficient data"
            else:
                row.update(metrics)
        except Exception as exc:
            row["Status"] = "Failed"
            row["Error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return pd.DataFrame(rows)


def cost_sensitivity(
    curve: pd.DataFrame,
    initial: float = 10_000,
    cost_scenarios_bps: tuple[float, ...] = (0, 15, 30, 50),
) -> pd.DataFrame:
    """Reprice the same audited signal path under alternative transaction-cost assumptions."""
    if curve is None or curve.empty:
        return pd.DataFrame()
    required = {"Gross return", "Gross turnover"}
    if not required.issubset(curve.columns):
        raise ValueError(f"Backtest curve must include: {', '.join(sorted(required))}")
    gross = pd.to_numeric(curve["Gross return"], errors="coerce").fillna(0.0)
    turnover = pd.to_numeric(curve["Gross turnover"], errors="coerce").fillna(0.0).clip(lower=0)
    rows = []
    months = len(curve)
    years = months / 12.0
    for bps in cost_scenarios_bps:
        if float(bps) < 0:
            raise ValueError("Transaction costs cannot be negative.")
        estimated_cost = turnover * float(bps) / 10_000
        net = gross - estimated_cost
        equity = float(initial) * (1 + net).cumprod()
        final_equity = float(equity.iloc[-1])
        cagr = (final_equity / float(initial)) ** (1 / years) - 1 if years > 0 and final_equity > 0 else np.nan
        volatility = float(net.std(ddof=1) * np.sqrt(12)) if len(net) > 1 else np.nan
        sharpe = float(net.mean() * 12 / volatility) if pd.notna(volatility) and volatility > 0 else np.nan
        wealth = np.r_[float(initial), equity.to_numpy()]
        high_water = np.maximum.accumulate(wealth)[1:]
        drawdown = equity.to_numpy() / high_water - 1
        rows.append({
            "Cost (bps)": float(bps),
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe (rf=0)": sharpe,
            "Max drawdown": float(np.min(drawdown)),
            "Final equity": final_equity,
            "Total modelled cost (% of starting equity sum)": float(estimated_cost.sum() * 100),
            "Average monthly turnover %": float(turnover.mean() * 100),
        })
    return pd.DataFrame(rows)


def chronological_period_summary(
    curve: pd.DataFrame,
    initial: float = 10_000,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> pd.DataFrame:
    """Summarize a fixed strategy across chronological train/validation/holdout periods.

    This is a diagnostic split, not an automated walk-forward parameter optimizer.
    The holdout is genuinely untouched only if the operator does not adjust strategy
    settings after examining it.
    """
    if curve is None or curve.empty:
        return pd.DataFrame()
    if "Net return" not in curve:
        raise ValueError("Curve is missing Net return.")
    if not (0 < train_fraction < 1) or not (0 <= validation_fraction < 1):
        raise ValueError("Train and validation fractions must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fractions must leave a non-empty holdout.")
    returns = pd.to_numeric(curve["Net return"], errors="coerce")
    if returns.isna().any():
        raise ValueError("Cannot split a curve with missing net returns.")
    n = len(returns)
    train_end = int(np.floor(n * train_fraction))
    validation_end = int(np.floor(n * (train_fraction + validation_fraction)))
    bounds = [
        ("Train", 0, train_end),
        ("Validation", train_end, validation_end),
        ("Holdout", validation_end, n),
    ]
    rows = []
    for label, start, end in bounds:
        part = curve.iloc[start:end]
        part_returns = returns.iloc[start:end]
        if part.empty:
            continue
        wealth = float(initial) * (1 + part_returns).cumprod()
        total_return = float((1 + part_returns).prod() - 1)
        years = len(part_returns) / 12.0
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else np.nan
        volatility = float(part_returns.std(ddof=1) * np.sqrt(12)) if len(part_returns) > 1 else np.nan
        sharpe = float(part_returns.mean() * 12 / volatility) if pd.notna(volatility) and volatility > 0 else np.nan
        wealth_with_initial = np.r_[float(initial), wealth.to_numpy()]
        peaks = np.maximum.accumulate(wealth_with_initial)[1:]
        drawdown = wealth.to_numpy() / peaks - 1
        row = {
            "Period": label,
            "Start date": pd.Timestamp(part.index[0]).date().isoformat(),
            "End date": pd.Timestamp(part.index[-1]).date().isoformat(),
            "Months": int(len(part)),
            "Total return": total_return,
            "CAGR": cagr,
            "Volatility": volatility,
            "Sharpe (rf=0)": sharpe,
            "Max drawdown": float(np.min(drawdown)),
            "Average turnover %": float(pd.to_numeric(part.get("Gross turnover", pd.Series(0, index=part.index)), errors="coerce").mean() * 100),
            "Average exposure %": float(pd.to_numeric(part.get("Gross exposure %", pd.Series(0, index=part.index)), errors="coerce").mean()),
            "Stops triggered": int(pd.to_numeric(part.get("Stops triggered", pd.Series(0, index=part.index)), errors="coerce").fillna(0).sum()),
        }
        if "Benchmark return" in part:
            benchmark_returns = pd.to_numeric(part["Benchmark return"], errors="coerce").dropna()
            row["Benchmark CAGR"] = (
                float((1 + benchmark_returns).prod() ** (12 / len(benchmark_returns)) - 1)
                if len(benchmark_returns) and (1 + benchmark_returns).prod() > 0 else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def industry_exposure_summary(
    audit: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize historical target-weight exposure by the available SIC grouping.

    Uses the target weights saved in the rebalance audit. Unclassified names remain visible
    rather than being silently assigned to a guessed sector. Since the map is the current
    SEC SIC mapping, it is a diagnostic, not a historical point-in-time industry series.
    """
    columns = [
        "SIC group", "Average weight %", "Maximum weight %", "Rebalance months invested",
        "Share of rebalance months %", "Average number of positions",
    ]
    if audit is None or audit.empty or "Target weights" not in audit:
        return pd.DataFrame(columns=columns)
    sector_map = sector_map or {}
    rows = []
    all_groups = set(str(value or "Unclassified") for value in sector_map.values())
    parsed_periods: list[dict[str, float]] = []
    for _, record in audit.iterrows():
        text_weights = str(record.get("Target weights", "") or "").strip()
        ticker_weights: dict[str, float] = {}
        if text_weights:
            for token in text_weights.split(","):
                if ":" not in token:
                    continue
                ticker, raw_weight = token.rsplit(":", 1)
                try:
                    weight = float(raw_weight)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(weight) and weight >= 0:
                    ticker_weights[ticker.strip().upper()] = weight
        period_weights: dict[str, float] = {}
        for ticker, weight in ticker_weights.items():
            group = str(sector_map.get(ticker, "Unclassified") or "Unclassified")
            period_weights[group] = period_weights.get(group, 0.0) + weight
            all_groups.add(group)
        parsed_periods.append(period_weights)
    periods = max(1, len(parsed_periods))
    for group in sorted(all_groups):
        values = np.array([period.get(group, 0.0) for period in parsed_periods], dtype=float)
        rows.append({
            "SIC group": group,
            "Average weight %": float(values.mean() * 100) if len(values) else 0.0,
            "Maximum weight %": float(values.max() * 100) if len(values) else 0.0,
            "Rebalance months invested": int((values > 0).sum()),
            "Share of rebalance months %": float((values > 0).sum() / periods * 100),
            "Average number of positions": float(np.mean([
                sum(
                    1 for ticker, weight in (
                        (
                            token.rsplit(":", 1)[0].strip().upper(),
                            float(token.rsplit(":", 1)[1]),
                        )
                        for token in str(record.get("Target weights", "") or "").split(",")
                        if ":" in token
                    )
                    if weight > 0 and str(sector_map.get(ticker, "Unclassified") or "Unclassified") == group
                )
                for _, record in audit.iterrows()
            ])) if len(parsed_periods) else 0.0,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        "Average weight %", ascending=False, kind="mergesort"
    ).reset_index(drop=True)


def train_select_holdout_search(
    prices: dict[str, pd.DataFrame],
    min_score_candidates: tuple[float, ...] = (25.0, 35.0, 45.0),
    hysteresis_candidates: tuple[float, ...] = (3.0, 7.0, 12.0),
    train_fraction: float = 0.60,
    min_train_months: int = 12,
    min_holdout_months: int = 6,
    **backtest_kwargs,
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame, dict[str, float]]:
    """Select entry score/hysteresis using train-period Sharpe, then report only the chosen holdout.

    The same fixed parameter set is run through historical time, using only past data
    for each monthly signal. Candidate selection uses returns from the train segment
    only. The chosen candidate's later holdout is reported separately; candidate holdout
    metrics are intentionally not exposed to avoid tuning to the test period.

    This is a single train/holdout parameter-selection test, not a full rolling
    walk-forward optimizer. It does not correct survivorship bias or data-source errors.
    """
    if not prices:
        raise ValueError("Price history is empty.")
    if not (0.25 <= train_fraction <= 0.85):
        raise ValueError("Train fraction must be between 25% and 85%.")
    if min_train_months < 6 or min_holdout_months < 3:
        raise ValueError("Require at least 6 training months and 3 holdout months.")
    score_values = sorted(set(float(x) for x in min_score_candidates))
    hysteresis_values = sorted(set(float(x) for x in hysteresis_candidates))
    if not score_values or any(not np.isfinite(x) or x < 0 or x > 100 for x in score_values):
        raise ValueError("Training score candidates must be finite values from 0 to 100.")
    if not hysteresis_values or any(not np.isfinite(x) or x < 0 or x > 100 for x in hysteresis_values):
        raise ValueError("Hysteresis candidates must be finite, non-negative values.")

    backtest_kwargs = _prepare_shared_backtest_kwargs(prices, dict(backtest_kwargs))
    candidate_curves: dict[tuple[float, float], pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    split_idx: int | None = None
    for score in score_values:
        for hysteresis in hysteresis_values:
            config = {"min_score": score, "hysteresis": hysteresis}
            row: dict[str, object] = {
                "Min score": score,
                "Hysteresis": hysteresis,
                "Status": "OK",
                "Train months": 0,
                "Train CAGR": np.nan,
                "Train Sharpe": np.nan,
                "Train max drawdown": np.nan,
                "Train turnover %": np.nan,
            }
            try:
                curve, _, _ = run_monthly_backtest(
                    prices=prices, **{**backtest_kwargs, **config}
                )
                if curve is None or curve.empty:
                    row["Status"] = "Insufficient data"
                    result_rows.append(row)
                    continue
                this_split = int(np.floor(len(curve) * train_fraction))
                if this_split < min_train_months or len(curve) - this_split < min_holdout_months:
                    row["Status"] = "Insufficient train/holdout history"
                    result_rows.append(row)
                    continue
                # Require all candidates to use the same chronological split; different
                # valid-curve lengths would otherwise make train scores incomparable.
                if split_idx is None:
                    split_idx = this_split
                elif this_split != split_idx:
                    row["Status"] = "Incomparable date coverage"
                    result_rows.append(row)
                    continue
                train = curve.iloc[:this_split]
                r = pd.to_numeric(train["Net return"], errors="coerce")
                if r.isna().any():
                    row["Status"] = "Missing training returns"
                    result_rows.append(row)
                    continue
                total_factor = float(np.prod(1 + r.to_numpy()))
                train_cagr = total_factor ** (12 / len(r)) - 1 if total_factor > 0 else np.nan
                train_volatility = float(r.std(ddof=1) * np.sqrt(12)) if len(r) > 1 else np.nan
                train_sharpe = float(r.mean() * 12 / train_volatility) if np.isfinite(train_volatility) and train_volatility > 0 else np.nan
                equity = np.r_[1.0, np.cumprod(1 + r.to_numpy())]
                drawdown = equity[1:] / np.maximum.accumulate(equity)[1:] - 1
                row.update({
                    "Train months": int(len(train)),
                    "Train CAGR": float(train_cagr),
                    "Train Sharpe": float(train_sharpe),
                    "Train max drawdown": float(np.min(drawdown)),
                    "Train turnover %": float(pd.to_numeric(train.get("Gross turnover", pd.Series(0, index=train.index)), errors="coerce").mean() * 100),
                })
                candidate_curves[(score, hysteresis)] = curve
            except Exception as exc:
                row["Status"] = "Failed"
                row["Error"] = f"{type(exc).__name__}: {exc}"
            result_rows.append(row)

    table = pd.DataFrame(result_rows)
    valid = table.loc[(table["Status"] == "OK") & table["Train Sharpe"].apply(lambda x: pd.notna(x) and np.isfinite(x))]
    if valid.empty:
        # Sharpe may be undefined for a constant return path; then use train CAGR only.
        valid = table.loc[(table["Status"] == "OK") & table["Train CAGR"].apply(lambda x: pd.notna(x) and np.isfinite(x))]
        selection_metric = "Train CAGR"
    else:
        selection_metric = "Train Sharpe"
    if valid.empty or split_idx is None:
        raise ValueError("No candidate had sufficient comparable data and a valid train-period objective.")
    ranked = valid.sort_values(
        [selection_metric, "Train CAGR", "Train turnover %", "Min score", "Hysteresis"],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    best_row = ranked.iloc[0]
    best_config = {
        "min_score": float(best_row["Min score"]),
        "hysteresis": float(best_row["Hysteresis"]),
        "training_objective": selection_metric,
        "train_months": float(best_row["Train months"]),
    }
    table["Selected by train"] = (
        table["Status"].eq("OK")
        & table["Min score"].eq(best_config["min_score"])
        & table["Hysteresis"].eq(best_config["hysteresis"])
    )
    selected_curve = candidate_curves[(best_config["min_score"], best_config["hysteresis"])]
    holdout = selected_curve.iloc[split_idx:].copy()
    initial = float(backtest_kwargs.get("initial", 10_000))
    holdout_returns = pd.to_numeric(holdout["Net return"], errors="coerce")
    holdout["Equity"] = initial * (1 + holdout_returns).cumprod()
    benchmark_returns = pd.to_numeric(holdout.get("Benchmark return", pd.Series(index=holdout.index, dtype=float)), errors="coerce")
    if benchmark_returns.notna().any():
        holdout["Benchmark equity"] = initial * (1 + benchmark_returns.fillna(0)).cumprod()
    total_factor = float(np.prod(1 + holdout_returns.to_numpy()))
    holdout_cagr = total_factor ** (12 / len(holdout_returns)) - 1 if total_factor > 0 else np.nan
    holdout_vol = float(holdout_returns.std(ddof=1) * np.sqrt(12)) if len(holdout_returns) > 1 else np.nan
    holdout_sharpe = float(holdout_returns.mean() * 12 / holdout_vol) if np.isfinite(holdout_vol) and holdout_vol > 0 else np.nan
    holdout_wealth = np.r_[1.0, np.cumprod(1 + holdout_returns.to_numpy())]
    holdout_dd = holdout_wealth[1:] / np.maximum.accumulate(holdout_wealth)[1:] - 1
    holdout_metrics: dict[str, float] = {
        "Holdout months": float(len(holdout)),
        "Holdout CAGR": float(holdout_cagr),
        "Holdout volatility": float(holdout_vol),
        "Holdout Sharpe (rf=0)": float(holdout_sharpe),
        "Holdout max drawdown": float(np.min(holdout_dd)),
        "Holdout final equity": float(holdout["Equity"].iloc[-1]),
        "Holdout average turnover %": float(pd.to_numeric(holdout.get("Gross turnover", pd.Series(0, index=holdout.index)), errors="coerce").mean() * 100),
    }
    benchmark_good = benchmark_returns.dropna()
    if len(benchmark_good) and float(np.prod(1 + benchmark_good.to_numpy())) > 0:
        holdout_metrics["Holdout benchmark CAGR"] = float(np.prod(1 + benchmark_good.to_numpy()) ** (12 / len(benchmark_good)) - 1)
        holdout_metrics["Holdout strategy CAGR aligned"] = float(
            (1 + holdout_returns.loc[benchmark_good.index]).prod() ** (12 / len(benchmark_good)) - 1
        )
    best_config["train_score"] = float(best_row[selection_metric])
    return table.sort_values(["Selected by train", selection_metric, "Train CAGR"], ascending=[False, False, False], na_position="last"), best_config, holdout, holdout_metrics


def _apply_terminal_liquidation_cost(
    curve: pd.DataFrame,
    initial: float,
    cost_bps: float,
) -> pd.DataFrame:
    """Charge the cost of liquidating any residual end-of-test exposure."""
    if curve is None or curve.empty:
        return curve
    out = curve.copy()
    if "Gross exposure %" not in out or "Equity" not in out:
        return out
    exposure = max(0.0, float(pd.to_numeric(pd.Series([out.iloc[-1]["Gross exposure %"]]), errors="coerce").fillna(0).iloc[0]) / 100)
    if exposure <= 0 or cost_bps <= 0:
        out.loc[out.index[-1], "Terminal liquidation turnover"] = 0.0
        out.loc[out.index[-1], "Terminal liquidation cost"] = 0.0
        return out
    end_equity = float(out.iloc[-1]["Equity"])
    before_equity = float(initial) if len(out) == 1 else float(out.iloc[-2]["Equity"])
    if end_equity <= 0 or before_equity <= 0:
        out.loc[out.index[-1], "Terminal liquidation turnover"] = exposure
        out.loc[out.index[-1], "Terminal liquidation cost"] = 0.0
        return out
    liquidation_turnover = exposure * end_equity / before_equity
    cash_cost = end_equity * exposure * float(cost_bps) / 10_000
    final_equity = max(0.0, end_equity - cash_cost)
    extra_return_cost = cash_cost / before_equity
    last = out.index[-1]
    out.loc[last, "Equity"] = final_equity
    out.loc[last, "Net return"] = final_equity / before_equity - 1
    if "Transaction cost" in out:
        out.loc[last, "Transaction cost"] = float(out.loc[last, "Transaction cost"]) + extra_return_cost
    if "Gross turnover" in out:
        out.loc[last, "Gross turnover"] = float(out.loc[last, "Gross turnover"]) + liquidation_turnover
    if "Estimated cost (bps of portfolio)" in out:
        out.loc[last, "Estimated cost (bps of portfolio)"] = (
            float(out.loc[last, "Estimated cost (bps of portfolio)"]) + extra_return_cost * 10_000
        )
    out.loc[last, "Terminal liquidation turnover"] = liquidation_turnover
    out.loc[last, "Terminal liquidation cost"] = extra_return_cost
    return out


def expanding_walk_forward(
    prices: dict[str, pd.DataFrame],
    min_score_candidates: tuple[float, ...] = (25.0, 35.0, 45.0),
    hysteresis_candidates: tuple[float, ...] = (3.0, 7.0, 12.0),
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    min_train_months: int = 18,
    **backtest_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run an expanding-window walk-forward search with isolated, flat-start OOS episodes.

    Each fold chooses min-score and hysteresis solely from past return history. The
    selected configuration is then run on the following test segment with the portfolio
    reset to cash; entry/rebalance/stop costs are applied, and any remaining end exposure
    is liquidated with modelled costs. Fold test returns do not overlap. This yields a
    conservative sequence of independent OOS episodes, not a broker-execution simulation.
    """
    if not prices:
        raise ValueError("Price history is empty.")
    for name, value in [("train_months", train_months), ("test_months", test_months), ("step_months", step_months), ("min_train_months", min_train_months)]:
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if train_months < min_train_months:
        raise ValueError("train_months must be at least min_train_months.")
    if test_months < 3:
        raise ValueError("test_months must be at least 3.")
    if step_months < test_months:
        raise ValueError("step_months must be >= test_months so out-of-sample returns do not overlap.")
    score_values = sorted(set(float(value) for value in min_score_candidates))
    hysteresis_values = sorted(set(float(value) for value in hysteresis_candidates))
    if not score_values or any(not np.isfinite(value) or value < 0 or value > 100 for value in score_values):
        raise ValueError("min_score_candidates must be finite values from 0 to 100.")
    if not hysteresis_values or any(not np.isfinite(value) or value < 0 or value > 100 for value in hysteresis_values):
        raise ValueError("hysteresis_candidates must be finite, non-negative values.")
    cost_bps = float(backtest_kwargs.get("cost_bps", 15.0))
    if not np.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("cost_bps must be finite and non-negative.")

    adjusted: dict[str, pd.Series] = {}
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        source_col = "Adj Close" if "Adj Close" in frame else "Close"
        series = _num(frame[source_col]).dropna()
        if len(series) >= 300:
            adjusted[ticker] = series
    if len(adjusted) < 2:
        raise ValueError("Need at least two usable price histories with 300+ observations.")
    monthly = pd.DataFrame(adjusted).sort_index().resample("ME").last().dropna(how="all")
    if len(monthly) < 13 + train_months + test_months:
        raise ValueError(
            f"Insufficient monthly history for expanding walk-forward: need approximately "
            f"{13 + train_months + test_months} month-end observations, received {len(monthly)}."
        )

    backtest_kwargs = _prepare_shared_backtest_kwargs(prices, dict(backtest_kwargs))
    base_initial = float(backtest_kwargs.get("initial", 10_000))
    start_idx = 12  # match the 12-month burn-in used by run_monthly_backtest
    fold_candidates: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    oos_segments: list[pd.DataFrame] = []
    fold_number = 0
    for test_start_idx in range(start_idx + train_months, len(monthly)):
        if (test_start_idx - (start_idx + train_months)) % step_months != 0:
            continue
        if test_start_idx + test_months > len(monthly) - 1:
            break
        fold_number += 1
        train_end_decision = monthly.index[test_start_idx - 1]
        test_start_decision = monthly.index[test_start_idx]
        test_end_decision = monthly.index[test_start_idx + test_months - 1]
        fold_table_rows: list[dict[str, object]] = []
        viable: list[dict[str, object]] = []
        for score in score_values:
            for hyst in hysteresis_values:
                record: dict[str, object] = {
                    "Fold": fold_number,
                    "Train end": train_end_decision,
                    "Min score": score,
                    "Hysteresis": hyst,
                    "Status": "OK",
                    "Train months": 0,
                    "Train CAGR": np.nan,
                    "Train Sharpe": np.nan,
                    "Train max drawdown": np.nan,
                    "Train turnover %": np.nan,
                    "Selected by train": False,
                }
                kwargs = dict(backtest_kwargs)
                kwargs.update({
                    "min_score": score,
                    "hysteresis": hyst,
                    "evaluation_start_date": None,
                    "evaluation_end_date": train_end_decision,
                })
                try:
                    train_curve, _, _ = run_monthly_backtest(prices=prices, **kwargs)
                    if train_curve is None or train_curve.empty or len(train_curve) != test_start_idx - start_idx:
                        record["Status"] = "Incomparable training coverage"
                        fold_table_rows.append(record)
                        continue
                    train_curve = _apply_terminal_liquidation_cost(
                        train_curve, initial=base_initial, cost_bps=cost_bps
                    )
                    train_returns = pd.to_numeric(train_curve["Net return"], errors="coerce")
                    if train_returns.isna().any() or (1 + train_returns).le(0).any():
                        record["Status"] = "Invalid train returns"
                        fold_table_rows.append(record)
                        continue
                    factor = float(np.prod(1 + train_returns.to_numpy()))
                    train_cagr = factor ** (12 / len(train_returns)) - 1
                    train_vol = float(train_returns.std(ddof=1) * np.sqrt(12)) if len(train_returns) > 1 else np.nan
                    train_sharpe = float(train_returns.mean() * 12 / train_vol) if np.isfinite(train_vol) and train_vol > 0 else np.nan
                    wealth = np.r_[1.0, np.cumprod(1 + train_returns.to_numpy())]
                    drawdown = wealth[1:] / np.maximum.accumulate(wealth)[1:] - 1
                    record.update({
                        "Train months": int(len(train_returns)),
                        "Train CAGR": float(train_cagr),
                        "Train Sharpe": float(train_sharpe),
                        "Train max drawdown": float(np.min(drawdown)),
                        "Train turnover %": float(pd.to_numeric(train_curve.get("Gross turnover", pd.Series(0, index=train_curve.index)), errors="coerce").mean() * 100),
                    })
                    viable.append(record)
                except Exception as exc:
                    record["Status"] = "Failed"
                    record["Error"] = f"{type(exc).__name__}: {exc}"
                fold_table_rows.append(record)

        objective = "Train Sharpe"
        eligible = [
            row for row in viable
            if pd.notna(row["Train Sharpe"]) and np.isfinite(float(row["Train Sharpe"]))
        ]
        if not eligible:
            objective = "Train CAGR"
            eligible = [
                row for row in viable
                if pd.notna(row["Train CAGR"]) and np.isfinite(float(row["Train CAGR"]))
            ]
        if not eligible:
            raise ValueError(
                f"Fold {fold_number} has no comparable parameter candidate with valid training metrics."
            )
        winner = sorted(
            eligible,
            key=lambda row: (
                -float(row[objective]),
                -float(row["Train CAGR"]),
                float(row["Train turnover %"]),
                float(row["Min score"]),
                float(row["Hysteresis"]),
            ),
        )[0]
        for row in fold_table_rows:
            if row["Min score"] == winner["Min score"] and row["Hysteresis"] == winner["Hysteresis"] and row["Status"] == "OK":
                row["Selected by train"] = True
        fold_candidates.extend(fold_table_rows)

        test_kwargs = dict(backtest_kwargs)
        test_kwargs.update({
            "min_score": float(winner["Min score"]),
            "hysteresis": float(winner["Hysteresis"]),
            "evaluation_start_date": test_start_decision,
            "evaluation_end_date": test_end_decision,
        })
        test_curve, _, _ = run_monthly_backtest(prices=prices, **test_kwargs)
        expected_test_dates = monthly.index[test_start_idx + 1:test_start_idx + test_months + 1]
        if test_curve is None or len(test_curve) != test_months or not test_curve.index.equals(expected_test_dates):
            actual = 0 if test_curve is None else len(test_curve)
            raise ValueError(
                f"Fold {fold_number} expected {test_months} non-overlapping OOS return months, "
                f"but backtest returned {actual}."
            )
        test_curve = _apply_terminal_liquidation_cost(test_curve, base_initial, cost_bps)
        test_curve["Fold"] = fold_number
        test_curve["Selected min score"] = float(winner["Min score"])
        test_curve["Selected hysteresis"] = float(winner["Hysteresis"])
        test_returns = pd.to_numeric(test_curve["Net return"], errors="coerce")
        if test_returns.isna().any() or (1 + test_returns).le(0).any():
            raise ValueError(f"Fold {fold_number} has invalid OOS returns.")
        factor = float(np.prod(1 + test_returns.to_numpy()))
        test_cagr = factor ** (12 / len(test_returns)) - 1
        test_vol = float(test_returns.std(ddof=1) * np.sqrt(12)) if len(test_returns) > 1 else np.nan
        test_sharpe = float(test_returns.mean() * 12 / test_vol) if np.isfinite(test_vol) and test_vol > 0 else np.nan
        wealth = np.r_[1.0, np.cumprod(1 + test_returns.to_numpy())]
        drawdown = wealth[1:] / np.maximum.accumulate(wealth)[1:] - 1
        benchmark_returns = pd.to_numeric(
            test_curve.get("Benchmark return", pd.Series(index=test_curve.index, dtype=float)),
            errors="coerce",
        )
        benchmark_good = benchmark_returns.dropna()
        fold_rows.append({
            "Fold": fold_number,
            "Training end decision": train_end_decision,
            "Test start decision": test_start_decision,
            "OOS return start": test_curve.index[0],
            "OOS return end": test_curve.index[-1],
            "Train months": int(winner["Train months"]),
            "Selected min score": float(winner["Min score"]),
            "Selected hysteresis": float(winner["Hysteresis"]),
            "Training objective": objective,
            "Training score": float(winner[objective]),
            "Train CAGR": float(winner["Train CAGR"]),
            "Train Sharpe": float(winner["Train Sharpe"]) if pd.notna(winner["Train Sharpe"]) else np.nan,
            "OOS months": int(len(test_curve)),
            "OOS CAGR": float(test_cagr),
            "OOS Sharpe": float(test_sharpe) if np.isfinite(test_sharpe) else np.nan,
            "OOS max drawdown": float(np.min(drawdown)),
            "OOS turnover %": float(pd.to_numeric(test_curve["Gross turnover"], errors="coerce").mean() * 100),
            "Terminal liquidation turnover %": float(test_curve["Terminal liquidation turnover"].iloc[-1] * 100),
            "OOS benchmark CAGR": (
                float(np.prod(1 + benchmark_good.to_numpy()) ** (12 / len(benchmark_good)) - 1)
                if len(benchmark_good) and np.prod(1 + benchmark_good.to_numpy()) > 0 else np.nan
            ),
        })
        oos_segments.append(test_curve)

    if not oos_segments:
        raise ValueError(
            f"Not enough history for an expanding walk-forward split with {train_months} training "
            f"months and {test_months} test months. Use a longer history or smaller windows."
        )
    oos = pd.concat(oos_segments).sort_index()
    if oos.index.duplicated().any():
        raise ValueError("Walk-forward generated overlapping OOS return dates; choose step_months >= test_months.")
    oos_returns = pd.to_numeric(oos["Net return"], errors="coerce")
    oos["Equity"] = base_initial * (1 + oos_returns).cumprod()
    benchmark_series = pd.to_numeric(oos.get("Benchmark return", pd.Series(index=oos.index, dtype=float)), errors="coerce")
    if benchmark_series.notna().any():
        oos["Benchmark equity"] = base_initial * (1 + benchmark_series.fillna(0)).cumprod()

    oos_years = len(oos_returns) / 12
    oos_factor = float(np.prod(1 + oos_returns.to_numpy()))
    oos_vol = float(oos_returns.std(ddof=1) * np.sqrt(12)) if len(oos_returns) > 1 else np.nan
    oos_sharpe = float(oos_returns.mean() * 12 / oos_vol) if np.isfinite(oos_vol) and oos_vol > 0 else np.nan
    oos_wealth = np.r_[base_initial, oos["Equity"].to_numpy()]
    oos_drawdown = oos_wealth[1:] / np.maximum.accumulate(oos_wealth)[1:] - 1
    metrics: dict[str, float] = {
        "OOS folds": float(len(fold_rows)),
        "OOS months": float(len(oos)),
        "OOS CAGR": float(oos_factor ** (1 / oos_years) - 1) if oos_years > 0 and oos_factor > 0 else np.nan,
        "OOS volatility": float(oos_vol) if np.isfinite(oos_vol) else np.nan,
        "OOS Sharpe (rf=0)": float(oos_sharpe) if np.isfinite(oos_sharpe) else np.nan,
        "OOS max drawdown": float(np.min(oos_drawdown)),
        "OOS final equity": float(oos["Equity"].iloc[-1]),
        "OOS average turnover %": float(pd.to_numeric(oos["Gross turnover"], errors="coerce").mean() * 100),
        "Terminal liquidation cost % of starting equity sum": float(pd.to_numeric(oos["Terminal liquidation cost"], errors="coerce").fillna(0).sum() * 100),
    }
    bench_good = benchmark_series.dropna()
    if len(bench_good) and float(np.prod(1 + bench_good.to_numpy())) > 0:
        metrics["OOS benchmark CAGR"] = float(np.prod(1 + bench_good.to_numpy()) ** (12 / len(bench_good)) - 1)
        aligned = pd.concat([oos_returns.rename("strategy"), benchmark_series.rename("benchmark")], axis=1).dropna()
        if len(aligned) and float(np.prod(1 + aligned["benchmark"].to_numpy())) > 0:
            metrics["OOS strategy CAGR aligned"] = float(np.prod(1 + aligned["strategy"].to_numpy()) ** (12 / len(aligned)) - 1)
    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(fold_candidates).sort_values(
            ["Fold", "Selected by train", "Train Sharpe", "Train CAGR"],
            ascending=[True, False, False, False], na_position="last", kind="mergesort",
        ),
        oos,
        metrics,
    )


def moving_block_bootstrap_uncertainty(
    curve: pd.DataFrame,
    initial: float = 10_000,
    bootstrap_samples: int = 500,
    block_length: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Estimate descriptive uncertainty bands by resampling contiguous monthly-return blocks.

    This is a stationary-regime sensitivity diagnostic, not a guarantee of frequentist
    coverage and not a fix for look-ahead/survivorship bias. It resamples the already
    generated *net* return path; it does not refit/re-optimize the strategy.
    """
    if curve is None or curve.empty or "Net return" not in curve:
        return pd.DataFrame()
    returns = pd.to_numeric(curve["Net return"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("Net returns must be finite for bootstrap uncertainty.")
    if (returns <= -1).any():
        raise ValueError("Monthly returns must be greater than -100% for compounding.")
    n = len(returns)
    if bootstrap_samples < 100 or bootstrap_samples > 10_000:
        raise ValueError("Bootstrap samples must be between 100 and 10,000.")
    if block_length < 1 or block_length > max(1, n // 2):
        raise ValueError("Block length must be positive and no more than half of the sample length.")
    if not np.isfinite(float(initial)) or initial <= 0:
        raise ValueError("Initial capital must be positive and finite.")

    def metrics(sample: np.ndarray) -> tuple[float, float, float, float]:
        factors = 1.0 + sample
        equity = float(initial) * np.cumprod(factors)
        total_factor = float(np.prod(factors))
        cagr = total_factor ** (12.0 / len(sample)) - 1 if total_factor > 0 else np.nan
        volatility = float(np.std(sample, ddof=1) * np.sqrt(12)) if len(sample) > 1 else np.nan
        sharpe = float(np.mean(sample) * 12 / volatility) if np.isfinite(volatility) and volatility > 0 else np.nan
        wealth = np.r_[float(initial), equity]
        peaks = np.maximum.accumulate(wealth)[1:]
        drawdown = equity / peaks - 1
        return cagr, sharpe, float(np.min(drawdown)), float(equity[-1])

    observed = metrics(returns)
    rng = np.random.default_rng(seed)
    per_replication = np.empty((bootstrap_samples, 4), dtype=float)
    block_offsets = np.arange(block_length, dtype=int)
    blocks_needed = int(np.ceil(n / block_length))
    for b in range(bootstrap_samples):
        starts = rng.integers(0, n, size=blocks_needed)
        indices = ((starts[:, None] + block_offsets[None, :]) % n).reshape(-1)[:n]
        per_replication[b] = metrics(returns[indices])

    labels = ["CAGR", "Sharpe (rf=0)", "Max drawdown", "Final equity"]
    rows = []
    for j, label in enumerate(labels):
        samples = per_replication[:, j]
        finite = samples[np.isfinite(samples)]
        rows.append({
            "Metric": label,
            "Observed": observed[j],
            "5th percentile": float(np.quantile(finite, 0.05)) if len(finite) else np.nan,
            "Median bootstrap": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
            "95th percentile": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
            "Bootstrap samples": int(bootstrap_samples),
            "Block length (months)": int(block_length),
            "Seed": int(seed),
        })
    return pd.DataFrame(rows)


def calculate_position_sizes(
    screener: pd.DataFrame,
    selected_tickers: list[str],
    capital: float,
    risk_budget_pct: float = 0.5,
    stop_atr_multiple: float = 3.0,
    max_position_pct: float = 20.0,
    max_total_exposure_pct: float = 95.0,
    max_total_risk_pct: float = 3.0,
) -> pd.DataFrame:
    """ATR-based long-only sizing planner, not an order-execution system."""
    if screener.empty or not selected_tickers:
        return pd.DataFrame()
    required_columns = {"Ticker", "Price", "ATR20"}
    missing_columns = required_columns - set(screener.columns)
    if missing_columns:
        raise ValueError("Screener is missing required sizing columns: " + ", ".join(sorted(missing_columns)))
    if not np.isfinite(float(capital)) or capital <= 0:
        raise ValueError("Capital must be a positive finite value.")
    for name, value in [
        ("risk_budget_pct", risk_budget_pct),
        ("stop_atr_multiple", stop_atr_multiple),
        ("max_position_pct", max_position_pct),
        ("max_total_exposure_pct", max_total_exposure_pct),
        ("max_total_risk_pct", max_total_risk_pct),
    ]:
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    for name, value in [
        ("risk_budget_pct", risk_budget_pct),
        ("max_position_pct", max_position_pct),
        ("max_total_exposure_pct", max_total_exposure_pct),
        ("max_total_risk_pct", max_total_risk_pct),
    ]:
        if float(value) > 100:
            raise ValueError(f"{name} cannot exceed 100%.")
    if stop_atr_multiple <= 0:
        raise ValueError("stop_atr_multiple must be positive.")
    if screener["Ticker"].astype(str).str.upper().duplicated().any():
        duplicates = screener.loc[screener["Ticker"].astype(str).str.upper().duplicated(keep=False), "Ticker"].astype(str).unique().tolist()
        raise ValueError("Screener has duplicate ticker rows: " + ", ".join(duplicates[:10]))
    normalized_selected = [str(ticker).strip().upper() for ticker in selected_tickers]
    if len(normalized_selected) != len(set(normalized_selected)):
        raise ValueError("Selected tickers contain duplicates; select each ticker once.")
    indexed = screener.copy()
    indexed["Ticker"] = indexed["Ticker"].astype(str).str.upper().str.strip()
    indexed = indexed.set_index("Ticker")
    missing_selected = sorted(set(normalized_selected) - set(indexed.index))
    if missing_selected:
        raise ValueError("Selected ticker(s) are missing from the current screener: " + ", ".join(missing_selected))
    rows = []
    for ticker in normalized_selected:
        r = indexed.loc[ticker]
        try:
            price = float(r.get("Price", np.nan))
            atr = float(r.get("ATR20", np.nan))
        except (TypeError, ValueError):
            price, atr = np.nan, np.nan
        if not np.isfinite(price) or price <= 0 or not np.isfinite(atr) or atr <= 0:
            rows.append({
                "Ticker": ticker,
                "Price": price if np.isfinite(price) and price > 0 else np.nan,
                "ATR20": atr if np.isfinite(atr) and atr > 0 else np.nan,
                "Stop distance": 0.0,
                "Provisional stop": np.nan,
                "Shares (risk-sized, capped)": 0,
                "Position value": 0.0,
                "Weight %": 0.0,
                "Initial risk budget": 0.0,
                "Risk if stop fills at stop price": 0.0,
                "Sizing status": "INSUFFICIENT DATA — no valid positive price or ATR(20)",
            })
            continue
        risk_money = float(capital) * float(risk_budget_pct) / 100
        stop_distance = float(stop_atr_multiple) * atr
        raw_shares = risk_money / stop_distance
        cap_value = float(capital) * float(max_position_pct) / 100
        shares = max(0, int(min(raw_shares, cap_value / price)))
        stop_price = max(0.0, price - stop_distance)
        rows.append({
            "Ticker": ticker,
            "Price": price,
            "ATR20": atr,
            "Stop distance": stop_distance,
            "Provisional stop": stop_price,
            "Shares (risk-sized, capped)": shares,
            "Position value": shares * price,
            "Weight %": (shares * price / capital * 100) if capital else np.nan,
            "Initial risk budget": shares * stop_distance,
            "Risk if stop fills at stop price": shares * stop_distance,
            "Sizing status": "OK" if shares > 0 else "ZERO SHARES — caps/risk budget too small for one share",
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    gross_value = result["Position value"].sum()
    cap = float(capital) * float(max_total_exposure_pct) / 100
    if gross_value > cap and gross_value > 0:
        scale = cap / gross_value
        result["Shares (risk-sized, capped)"] = np.floor(result["Shares (risk-sized, capped)"] * scale).astype(int)
        result["Position value"] = result["Shares (risk-sized, capped)"] * result["Price"]
        result["Initial risk budget"] = result["Shares (risk-sized, capped)"] * result["Stop distance"]
        result["Risk if stop fills at stop price"] = result["Initial risk budget"]
        result["Weight %"] = result["Position value"] / float(capital) * 100

    # Apply a portfolio-level open-position risk cap after per-position and gross
    # exposure limits. Integer share rounding keeps realized model risk at or below cap.
    total_risk_cap = float(capital) * float(max_total_risk_pct) / 100
    total_risk = float(result["Initial risk budget"].sum())
    if total_risk > total_risk_cap and total_risk > 0:
        scale = total_risk_cap / total_risk
        result["Shares (risk-sized, capped)"] = np.floor(result["Shares (risk-sized, capped)"] * scale).astype(int)
        result["Position value"] = result["Shares (risk-sized, capped)"] * result["Price"]
        result["Initial risk budget"] = result["Shares (risk-sized, capped)"] * result["Stop distance"]
        result["Risk if stop fills at stop price"] = result["Initial risk budget"]
        result["Weight %"] = result["Position value"] / float(capital) * 100

    result["Portfolio risk cap %"] = float(max_total_risk_pct)
    result["Portfolio risk used %"] = result["Initial risk budget"] / float(capital) * 100
    result["Total portfolio risk %"] = float(result["Initial risk budget"].sum() / float(capital) * 100)
    return result
