"""Technical indicators, cross-sectional scoring, portfolio rules, and research backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _num(series):
    return pd.to_numeric(series, errors="coerce").astype(float)


def calculate_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily indicators using adjusted OHLC, excluding the current bar from breakouts."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    close = _num(frame.get("Close", frame.get("Adj Close", pd.Series(dtype=float))))
    if close.empty:
        return pd.DataFrame()
    total_return_close = _num(frame.get("Adj Close", close)).reindex(close.index)
    high = _num(frame.get("High", close)).reindex(close.index).fillna(close)
    low = _num(frame.get("Low", close)).reindex(close.index).fillna(close)
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
    out["Adj Close"] = total_return_close
    out["High"] = high
    out["Low"] = low
    out["MA50"] = close.rolling(50, min_periods=50).mean()
    out["MA200"] = close.rolling(200, min_periods=200).mean()
    out["MA200_prev20"] = close.rolling(200, min_periods=200).mean().shift(20)
    out["Momentum 12-1M"] = total_return_close.shift(21) / total_return_close.shift(252) - 1
    out["Momentum 6-1M"] = total_return_close.shift(21) / total_return_close.shift(126) - 1
    out["ADX14"] = adx14
    out["ATR14"] = atr14
    out["ATR20"] = atr20
    out["ATR20%"] = atr20 / close * 100
    out["High20 prior"] = high.shift(1).rolling(20, min_periods=20).max()
    out["High55 prior"] = high.shift(1).rolling(55, min_periods=55).max()
    out["Low50 prior"] = low.shift(1).rolling(50, min_periods=50).min()
    out["Breakout20"] = close > out["High20 prior"]
    out["Breakout55"] = close > out["High55 prior"]
    out["Trend OK"] = (close > out["MA200"]) & (out["MA50"] > out["MA200"]) & (out["MA200"] > out["MA200_prev20"])
    out["Below 50d low"] = close < out["Low50 prior"]
    out["Daily return"] = total_return_close.pct_change(fill_method=None)
    out["Volatility20%"] = out["Daily return"].rolling(20, min_periods=20).std() * np.sqrt(252) * 100
    out["Close raw"] = close
    return out.replace([np.inf, -np.inf], np.nan)


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
            "Source": f.get("Source", "Prices only"),
            "Financial period": f.get("Financial period", ""),
            "Latest filing date": f.get("Latest filing date", ""),
            "Price date": available.index[-1].date().isoformat(),
            "Price": float(frame["Close"].dropna().iloc[-1]) if "Close" in frame else float(latest["Close"]),
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
            "ATR20": latest.get("ATR20", np.nan),
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
) -> set[str]:
    valid = scores.dropna()
    valid = valid[eligible.reindex(valid.index).fillna(False).astype(bool)]
    ranked = valid.sort_values(ascending=False)
    ranked_names = list(ranked.index)
    if not previous:
        # Fill initial portfolio once; the replacement cap governs subsequent rebalance dates.
        return set(ranked_names[:top_n])
    current = [
        ticker for ticker in previous
        if ticker in ranked.index and float(ranked[ticker]) >= exit_score_floor
    ]
    replacements = 0
    # Mandatory exits can leave vacancies. Limit replacement entries to the configured
    # count so that a rule violation does not trigger an uncontrolled full rebalance.
    for ticker in ranked_names:
        if len(current) >= top_n or replacements >= max_replacements:
            break
        if ticker not in current:
            current.append(ticker)
            replacements += 1
    while len(current) >= top_n and top_n > 0 and replacements < max_replacements:
        incumbents = [t for t in current if t in ranked.index]
        challengers = [t for t in ranked_names if t not in current]
        if not incumbents or not challengers:
            break
        weakest = min(incumbents, key=lambda t: float(ranked[t]))
        challenger = challengers[0]
        if float(ranked[challenger]) < float(ranked[weakest]) + min_improvement:
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
) -> pd.DataFrame:
    rows = []
    for ticker, frame in prices.items():
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
) -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame | None]:
    """Month-end signals; selected positions earn the next month return.

    Prices are adjusted for corporate actions if the provider has supplied Adj Close.
    Orders are modeled at the signal month's close, represented by close-to-close
    returns for the next month, with costs on gross traded notional. This is not an
    intraday fill simulation; the current symbol list can still create survivorship bias.
    """
    if not prices:
        return None, None, None
    adjusted = {}
    indicator_cache = {}
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        col = "Adj Close" if "Adj Close" in frame else "Close"
        series = _num(frame[col]).dropna()
        if len(series) >= 300:
            adjusted[ticker] = series
            indicator_cache[ticker] = calculate_indicators(frame)
    if len(adjusted) < 2:
        return None, None, None
    daily_prices = pd.DataFrame(adjusted).sort_index()
    monthly = daily_prices.resample("ME").last()
    monthly = monthly.dropna(how="all")
    if len(monthly) < 15:
        return None, None, None
    monthly_returns = monthly.pct_change(fill_method=None)
    rows, audit = [], []
    equity = float(initial)
    previous_holdings: set[str] = set()
    previous_weights: dict[str, float] = {}
    # First usable decision requires at least 252 trading days of history in the daily indicators.
    start_idx = 12
    for i in range(start_idx, len(monthly) - 1):
        decision_date = monthly.index[i]
        cross = _build_monthly_cross_section(
            prices, decision_date,
            fundamentals_history if use_fundamentals else None,
            optional_history, indicator_cache=indicator_cache
        )
        if cross.empty:
            continue
        score_col = "Fundamental/composite score"
        cross[score_col] = cross[score_col].where(cross[score_col].notna(), cross["Technical score"])
        eligible = cross["Trend OK"].fillna(False).astype(bool) & (cross["ADX14"].fillna(0) >= min_adx)
        # 50-session low is an explicit month-end exit trigger. This does not
        # simulate intraday stop fills; those require higher-frequency execution data.
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
        )
        row_return_date = monthly.index[i + 1]
        selected = sorted(picks)
        # A missing return for a portfolio constituent invalidates the backtest result;
        # never silently drop that name and inflate results.
        next_returns = monthly_returns.loc[row_return_date, selected] if selected else pd.Series(dtype=float)
        if len(selected) and next_returns.isna().any():
            missing = list(next_returns.index[next_returns.isna()])
            raise ValueError(
                f"Missing monthly return for held ticker(s) {missing} during {row_return_date.date()}; "
                "the result was stopped rather than silently ignoring a missing/delisted security."
            )
        new_weights = _target_weights(
            selected, cross, method=weighting_method, max_position_weight=max_position_weight
        )
        names = set(previous_weights) | set(new_weights)
        turnover = sum(abs(new_weights.get(t, 0.0) - previous_weights.get(t, 0.0)) for t in names)
        transaction_cost = turnover * float(cost_bps) / 10_000
        gross_return = float(sum(new_weights.get(ticker, 0.0) * float(next_returns[ticker]) for ticker in selected)) if selected else 0.0
        net_return = gross_return - transaction_cost
        equity *= (1 + net_return)
        rows.append({
            "Date": row_return_date,
            "Signal date": decision_date,
            "Gross return": gross_return,
            "Transaction cost": transaction_cost,
            "Net return": net_return,
            "Equity": equity,
            "Holdings": ", ".join(selected),
            "Number of holdings": len(selected),
            "Gross exposure %": sum(new_weights.values()) * 100,
            "Target weights": ", ".join(f"{ticker}:{new_weights[ticker]:.3f}" for ticker in selected),
            "Gross turnover": turnover,
            "Estimated cost (bps of portfolio)": transaction_cost * 10_000,
            "Selected score mean": float(scores.reindex(selected).mean()) if selected else np.nan,
        })
        audit.append({
            "Signal date": decision_date,
            "Return period": row_return_date,
            "Previous holdings": ", ".join(sorted(previous_holdings)),
            "Selected holdings": ", ".join(selected),
            "Target weights": ", ".join(f"{ticker}:{new_weights[ticker]:.3f}" for ticker in selected),
            "Weighting method": weighting_method,
            "Eligible names": ", ".join(sorted(eligible[eligible].index)),
            "Min ADX": min_adx,
            "Min score": min_score,
            "Hysteresis points": hysteresis,
            "Gross turnover": turnover,
            "Cost bps": cost_bps,
        })
        previous_holdings = picks
        previous_weights = new_weights
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
    }
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.dropna().resample("ME").last().pct_change(fill_method=None)
        aligned = curve["Net return"].to_frame("Strategy").join(bench.rename("Benchmark"), how="inner").dropna()
        if not aligned.empty:
            metrics["Benchmark CAGR"] = float((1 + aligned["Benchmark"]).prod() ** (12 / len(aligned)) - 1)
            metrics["Benchmark period months"] = len(aligned)
            metrics["Strategy CAGR aligned"] = float((1 + aligned["Strategy"]).prod() ** (12 / len(aligned)) - 1)
            curve["Benchmark equity"] = float(initial) * (1 + bench.reindex(curve.index).fillna(0)).cumprod()
    return curve, metrics, pd.DataFrame(audit)


def calculate_position_sizes(
    screener: pd.DataFrame,
    selected_tickers: list[str],
    capital: float,
    risk_budget_pct: float = 0.5,
    stop_atr_multiple: float = 3.0,
    max_position_pct: float = 20.0,
    max_total_exposure_pct: float = 95.0,
) -> pd.DataFrame:
    """ATR-based long-only sizing planner, not an order-execution system."""
    if screener.empty or not selected_tickers:
        return pd.DataFrame()
    indexed = screener.set_index("Ticker")
    rows = []
    for ticker in selected_tickers:
        if ticker not in indexed.index:
            continue
        r = indexed.loc[ticker]
        price = float(r.get("Price", np.nan))
        atr = float(r.get("ATR20", np.nan))
        if not np.isfinite(price) or price <= 0 or not np.isfinite(atr) or atr <= 0:
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
    return result
