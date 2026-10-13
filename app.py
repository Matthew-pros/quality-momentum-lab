import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import date

st.set_page_config(page_title="Quality × Value × Momentum Lab", page_icon="📈", layout="wide")

DEFAULT_TICKERS = "MSFT,AAPL,GOOGL,AMZN,META,NVDA,AVGO,ORCL,ADBE,CRM,CSCO,INTU,ASML,TSM,LLY,JNJ,AZN,NVO,RELX,OTIS,SPY,ACWI"
BENCHMARKS = ["SPY", "ACWI", "QQQ"]

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(tickers_text: str, period: str):
    tickers = list(dict.fromkeys([x.strip().upper() for x in tickers_text.split(",") if x.strip()]))
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by="column", threads=True)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
            volume = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame(index=close.index)
        else:
            close = raw.xs("Close", axis=1, level=1)
            volume = raw.xs("Volume", axis=1, level=1) if "Volume" in raw.columns.get_level_values(1) else pd.DataFrame(index=close.index)
    else:
        ticker = tickers[0]
        close = raw[["Close"]].rename(columns={"Close": ticker})
        volume = raw[["Volume"]].rename(columns={"Volume": ticker}) if "Volume" in raw else pd.DataFrame(index=close.index)
    if isinstance(close, pd.Series):
        close = close.to_frame()
    if isinstance(volume, pd.Series):
        volume = volume.to_frame()
    return close.dropna(how="all"), volume

@st.cache_data(ttl=86400, show_spinner=False)
def load_info(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}

def safe_num(value):
    try:
        v = float(value)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan

def calc_adx(high, low, close, n=14):
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = pd.concat([(high-low), (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/n, min_periods=n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, min_periods=n, adjust=False).mean(), atr

@st.cache_data(ttl=3600, show_spinner=False)
def load_ohlc(ticker: str, period: str):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

def technical_row(ticker, close, period):
    ohlc = load_ohlc(ticker, period)
    if ohlc.empty or "Close" not in ohlc:
        return None
    c = ohlc["Close"].dropna()
    if len(c) < 220:
        return None
    h = ohlc["High"].reindex(c.index)
    l = ohlc["Low"].reindex(c.index)
    adx, atr = calc_adx(h, l, c)
    r_12_1 = c.iloc[-22] / c.iloc[-252] - 1 if len(c) >= 252 else np.nan
    r_6_1 = c.iloc[-22] / c.iloc[-126] - 1 if len(c) >= 126 else np.nan
    ma50 = c.rolling(50).mean().iloc[-1]
    ma200 = c.rolling(200).mean().iloc[-1]
    ma200_prev = c.rolling(200).mean().iloc[-21] if len(c) > 220 else np.nan
    true_range = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr20 = true_range.ewm(alpha=1/20, min_periods=20, adjust=False).mean()
    high55 = h.shift(1).rolling(55).max().iloc[-1]
    low50 = l.shift(1).rolling(50).min().iloc[-1]
    price = c.iloc[-1]
    return {
        "Ticker": ticker, "Price": price, "12-1M": r_12_1, "6-1M": r_6_1,
        "MA50": ma50, "MA200": ma200, "MA200 rising": bool(ma200 > ma200_prev) if pd.notna(ma200_prev) else False,
        "ADX14": adx.iloc[-1], "ATR20%": (atr20.iloc[-1] / price * 100),
        "Breakout55": bool(price > high55) if pd.notna(high55) else False,
        "Below 50d low": bool(price < low50) if pd.notna(low50) else False,
        "Trend OK": bool(price > ma200 and ma50 > ma200 and ma200 > ma200_prev) if pd.notna(ma200_prev) else False
    }

def percentile_score(series, higher_better=True):
    """Cross-sectional percentile; missing observations remain missing."""
    s = pd.to_numeric(series, errors="coerce")
    observed = s.notna()
    result = pd.Series(np.nan, index=s.index, dtype=float)
    if observed.sum() == 0:
        return result
    if observed.sum() == 1:
        result.loc[observed] = 50.0
        return result
    result.loc[observed] = s.loc[observed].rank(pct=True, ascending=higher_better) * 100
    return result


def weighted_average(frame, weights, default=50.0):
    """Row-wise weighted mean that does not treat missing data as a real score."""
    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = values.notna()
        numerator.loc[valid] += values.loc[valid] * weight
        denominator.loc[valid] += weight
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(default)


def combine_percentiles(frame, features):
    """Features are (column, weight, higher_is_better) tuples."""
    ranked = pd.DataFrame(index=frame.index)
    weights = {}
    for column, weight, higher_better in features:
        ranked[column] = percentile_score(frame[column], higher_better=higher_better)
        weights[column] = weight
    return weighted_average(ranked, weights)

def get_fundamentals(ticker):
    info = load_info(ticker)
    return {
        "Ticker": ticker,
        "Company": info.get("shortName", ticker),
        "Sector": info.get("sector", "Unknown"),
        "Market cap": safe_num(info.get("marketCap")),
        "P/E": safe_num(info.get("trailingPE")),
        "Forward P/E": safe_num(info.get("forwardPE")),
        "Price/Sales": safe_num(info.get("priceToSalesTrailing12Months")),
        "ROE %": safe_num(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") is not None else np.nan,
        "Debt/Equity": safe_num(info.get("debtToEquity")),
        "FCF": safe_num(info.get("freeCashflow")),
        "FCF margin %": (
            safe_num(info.get("freeCashflow")) / safe_num(info.get("totalRevenue")) * 100
            if pd.notna(safe_num(info.get("freeCashflow")))
            and pd.notna(safe_num(info.get("totalRevenue")))
            and safe_num(info.get("totalRevenue")) > 0 else np.nan
        ),
        "Profit margin %": safe_num(info.get("profitMargins")) * 100 if info.get("profitMargins") is not None else np.nan,
        "Revenue growth %": safe_num(info.get("revenueGrowth")) * 100 if info.get("revenueGrowth") is not None else np.nan,
        "Earnings growth %": safe_num(info.get("earningsGrowth")) * 100 if info.get("earningsGrowth") is not None else np.nan,
        "Analyst target upside %": (safe_num(info.get("targetMeanPrice")) / safe_num(info.get("currentPrice")) - 1) * 100
            if safe_num(info.get("targetMeanPrice")) and safe_num(info.get("currentPrice")) else np.nan,
    }

def run_monthly_backtest(close, top_n=5, cost_bps=10, initial=10000):
    # Educational prototype: month-end ranking, next-month holding, long-only equal weight.
    # Excludes historical fundamentals, delisted stocks, tax-lot treatment, and point-in-time universe.
    px = close.dropna(axis=1, how="all").copy()
    if len(px) < 300 or px.shape[1] < 2:
        return None, None
    monthly = px.resample("ME").last().dropna(how="all")
    if len(monthly) < 14:
        return None, None
    rets = monthly.pct_change(fill_method=None)
    rows = []
    equity = initial
    prev = set()
    for i in range(13, len(monthly)):
        # 12-1M momentum using month-end data, skipping latest month
        mom12 = monthly.iloc[i-1] / monthly.iloc[i-12] - 1
        mom6 = monthly.iloc[i-1] / monthly.iloc[i-6] - 1
        trend = monthly.iloc[i-1] > monthly.iloc[max(0, i-12):i].mean()
        score = (mom12.rank(pct=True) + mom6.rank(pct=True)) / 2
        eligible = score[trend].dropna().sort_values(ascending=False)
        picks = set(eligible.head(top_n).index)
        if not picks:
            picks = set()
        old_weights = {ticker: 1.0 / len(prev) for ticker in prev} if prev else {}
        new_weights = {ticker: 1.0 / len(picks) for ticker in picks} if picks else {}
        traded_tickers = set(old_weights) | set(new_weights)
        # Gross traded notional: buys + sells as a fraction of portfolio value.
        turnover = sum(abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in traded_tickers)
        cost = turnover * cost_bps / 10000
        selected_returns = rets.iloc[i].reindex(list(picks)) if picks else pd.Series(dtype=float)
        # Explicitly keep equal weights if a selected name has a missing monthly return.
        next_ret = selected_returns.fillna(0.0).mean() if picks else 0.0
        strat_ret = (next_ret if pd.notna(next_ret) else 0.0) - cost
        equity *= (1 + strat_ret)
        rows.append({"Date": monthly.index[i], "Return": strat_ret, "Equity": equity, "Holdings": ", ".join(sorted(picks)), "Gross turnover": turnover})
        prev = picks
    if not rows:
        return None, None
    curve = pd.DataFrame(rows).set_index("Date")
    years = len(curve) / 12.0
    cagr = (curve["Equity"].iloc[-1] / initial) ** (1/years) - 1
    vol = curve["Return"].std(ddof=1) * np.sqrt(12)
    sharpe = (curve["Return"].mean() * 12) / vol if vol > 0 else np.nan
    high_water = np.maximum.accumulate(np.r_[initial, curve["Equity"].to_numpy()])[1:]
    dd = curve["Equity"] / high_water - 1
    metrics = {
        "CAGR": cagr, "Volatility": vol, "Sharpe (rf=0)": sharpe,
        "Max drawdown": dd.min(), "Final equity": curve["Equity"].iloc[-1],
        "Months": len(curve)
    }
    return curve, metrics

st.title("Quality × Value × Momentum Lab")
st.caption("Výzkumný prototyp pro screening akcií a základní backtest. Není to investiční doporučení.")
with st.sidebar:
    st.header("Nastavení")
    tickers_text = st.text_area("Ticker universe (oddělený čárkami)", DEFAULT_TICKERS, height=120)
    period = st.selectbox("Historie cen", ["5y", "10y", "max"], index=1)
    st.caption("Pro 12–1M momentum a MA200 potřebujeme alespoň přibližně 1 rok dat; delší historie je vhodnější.")
    st.divider()
    st.subheader("Backtest")
    top_n = st.slider("Počet pozic", 3, 15, 5)
    cost_bps = st.slider("Náklady na změnu portfolia (bps)", 0, 100, 15, 5)
    initial = st.number_input("Počáteční kapitál (USD)", min_value=1000, value=10000, step=1000)
    run_button = st.button("Načíst data a přepočítat", type="primary")

if "run" not in st.session_state:
    st.session_state.run = True
if run_button:
    st.cache_data.clear()
    st.session_state.run = True

if st.session_state.run:
    tickers = list(dict.fromkeys([x.strip().upper() for x in tickers_text.split(",") if x.strip()]))
    if len(tickers) < 2:
        st.error("Zadej alespoň dva tickery.")
        st.stop()
    with st.spinner("Načítám cenová data… bezplatné zdroje mohou občas odpovídat pomalu."):
        close, volume = load_prices(tickers_text, period)
    if close.empty:
        st.error("Data se nepodařilo načíst. Zkontroluj tickery nebo to zkus později.")
        st.stop()

    latest_data_date = close.dropna(how="all").index.max()
    price_qc_rows = []
    for ticker in tickers:
        values = close[ticker].dropna() if ticker in close.columns else pd.Series(dtype=float)
        price_qc_rows.append({
            "Ticker": ticker,
            "Rows with price": int(values.notna().sum()),
            "First date": values.index.min().date().isoformat() if not values.empty else None,
            "Last date": values.index.max().date().isoformat() if not values.empty else None,
            "Missing price series": values.empty
        })
    price_qc = pd.DataFrame(price_qc_rows)
    st.caption(
        f"Kontrola cenových dat: poslední dostupné datum {latest_data_date.date().isoformat()} · "
        f"řady s cenami {int((~price_qc['Missing price series']).sum())}/{len(tickers)} · "
        "zdroj Yahoo Finance přes yfinance"
    )
    with st.expander("Kontrola úplnosti cenových dat"):
        st.dataframe(price_qc, use_container_width=True, hide_index=True)
        missing_tickers = price_qc.loc[price_qc["Missing price series"], "Ticker"].tolist()
        if missing_tickers:
            st.warning("Chybí cenová řada pro: " + ", ".join(missing_tickers))
        st.download_button(
            "Stáhnout kontrolu cenových dat CSV",
            price_qc.to_csv(index=False).encode("utf-8"),
            "price_data_quality.csv",
            "text/csv"
        )

    tab1, tab2, tab3 = st.tabs(["🔎 Screener", "📉 Backtest (experimentální)", "📚 Data & metodika"])
    with tab1:
        st.subheader("Technické a fundamentální skóre")
        st.info("Fundamentální údaje z Yahoo Finance jsou orientační aktuální snapshot, nikoli point-in-time historická data. Proto se v tomto MVP nepoužívají k simulaci historických nákupů.")
        prog = st.progress(0)
        rows = []
        for i, ticker in enumerate(tickers):
            try:
                tech = technical_row(ticker, close, period)
                fund = get_fundamentals(ticker)
                if tech:
                    rows.append({**fund, **tech})
            except Exception:
                pass
            prog.progress((i+1)/len(tickers))
        if not rows:
            st.warning("Nedostatek dat pro výpočet indikátorů. Zkus delší historii nebo jiné tickery.")
        else:
            df = pd.DataFrame(rows)
            df["_Forward P/E positive"] = df["Forward P/E"].where(df["Forward P/E"] > 0)
            df["_Price/Sales positive"] = df["Price/Sales"].where(df["Price/Sales"] > 0)
            df["Momentum score"] = combine_percentiles(df, [
                ("12-1M", 0.5, True), ("6-1M", 0.5, True)
            ])
            df["Quality proxy"] = combine_percentiles(df, [
                ("ROE %", 0.25, True),
                ("Profit margin %", 0.25, True),
                ("FCF margin %", 0.25, True),
                ("Revenue growth %", 0.25, True)
            ])
            # Positive valuation multiples only; negative P/E is not treated as "cheap".
            df["Value proxy"] = combine_percentiles(df, [
                ("_Forward P/E positive", 0.6, False),
                ("_Price/Sales positive", 0.4, False)
            ])
            df["Growth proxy"] = combine_percentiles(df, [
                ("Revenue growth %", 0.5, True),
                ("Earnings growth %", 0.5, True)
            ])
            df["Trend score"] = np.where(
                df["Trend OK"],
                50 + df["ADX14"].clip(0, 50).fillna(0),
                0
            )
            score_parts = pd.DataFrame({
                "Quality": df["Quality proxy"],
                "Value": df["Value proxy"],
                "Growth": df["Growth proxy"],
                "Momentum": df["Momentum score"],
                "Trend": df["Trend score"]
            }, index=df.index)
            df["Composite score"] = weighted_average(score_parts, {
                "Quality": 0.28, "Value": 0.22, "Growth": 0.18,
                "Momentum": 0.22, "Trend": 0.10
            })
            fundamental_columns = [
                "ROE %", "Profit margin %", "FCF margin %",
                "Revenue growth %", "Earnings growth %", "Forward P/E", "Price/Sales"
            ]
            df["Fundamental coverage %"] = df[fundamental_columns].notna().mean(axis=1) * 100
            df["Fundamental status"] = np.where(
                df["Fundamental coverage %"] >= 70, "Adequate snapshot",
                np.where(df["Fundamental coverage %"] >= 40, "Partial data", "Low coverage")
            )
            df["Signal"] = np.select(
                [df["Below 50d low"], df["Trend OK"] & df["Breakout55"] & (df["Composite score"] >= 65), df["Trend OK"]],
                ["Review / exit trigger", "Breakout candidate", "Trend: watch"],
                default="No entry"
            )
            display_cols = ["Ticker","Company","Sector","Price","Composite score","Quality proxy","Value proxy","Growth proxy","Momentum score","Fundamental coverage %","Fundamental status","12-1M","6-1M","ADX14","ATR20%","Trend OK","Breakout55","P/E","Forward P/E","ROE %","FCF margin %","Signal"]
            display_cols = [c for c in display_cols if c in df.columns]
            shown = df.sort_values("Composite score", ascending=False)[display_cols].copy()
            st.dataframe(shown, use_container_width=True, hide_index=True, column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Composite score": st.column_config.NumberColumn(format="%.1f"),
                "Quality proxy": st.column_config.NumberColumn(format="%.1f"),
                "Value proxy": st.column_config.NumberColumn(format="%.1f"),
                "Momentum score": st.column_config.NumberColumn(format="%.1f"),
                "12-1M": st.column_config.NumberColumn(format="%.1%"),
                "6-1M": st.column_config.NumberColumn(format="%.1%"),
                "ADX14": st.column_config.NumberColumn(format="%.1f"),
                "ATR20%": st.column_config.NumberColumn(format="%.1f%%"),
            })
            st.download_button("Stáhnout screener CSV", df.to_csv(index=False).encode("utf-8"), "screener.csv", "text/csv")
            chosen = st.selectbox("Detail akcie", df["Ticker"].tolist())
            ohlc = load_ohlc(chosen, period)
            if not ohlc.empty:
                chart = px.line(ohlc, y="Close", title=f"{chosen} — adjusted close")
                st.plotly_chart(chart, use_container_width=True)
                st.write(df[df["Ticker"] == chosen].T.rename(columns={df[df["Ticker"] == chosen].index[0]: "Value"}))

    with tab2:
        st.subheader("Základní měsíční momentum backtest")
        st.warning("Tento test je pouze prototyp: používá současný seznam tickerů, postrádá historické delistované firmy a point-in-time fundamenty, a nezohledňuje daně. Náklady v bps se aplikují na hrubě obchodovaný objem (nákupy + prodeje). Výsledky nejsou důkazem obchodovatelnosti.")
        curve, metrics = run_monthly_backtest(close, top_n=top_n, cost_bps=cost_bps, initial=initial)
        if curve is None:
            st.warning("Nedostatek dat pro backtest. Zvol delší historii a širší seznam tickerů.")
        else:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CAGR", f"{metrics['CAGR']:.1%}")
            c2.metric("Sharpe (rf=0)", f"{metrics['Sharpe (rf=0)']:.2f}")
            c3.metric("Max drawdown", f"{metrics['Max drawdown']:.1%}")
            c4.metric("Konečná hodnota", f"${metrics['Final equity']:,.0f}")
            st.plotly_chart(px.line(curve, y="Equity", title="Equity curve"), use_container_width=True)
            st.dataframe(curve, use_container_width=True)
            st.download_button("Stáhnout equity curve CSV", curve.to_csv().encode("utf-8"), "backtest_equity.csv", "text/csv")
    with tab3:
        st.subheader("Datové zdroje a limity")
        st.markdown("""
        **Použité zdroje**
        - **Yahoo Finance přes `yfinance`** — historické ceny a orientační aktuální základní metriky. `yfinance` je neoficiální open-source klient a jeho dokumentace uvádí, že Yahoo data jsou určena pro osobní použití; ověř podmínky užití pro konkrétní nasazení.
        - **Bez API klíče** — technické indikátory se počítají lokálně z OHLC dat.
        - **Další fáze: SEC EDGAR Company Facts** pro americké firmy a oficiální výkazy; vyžaduje správné mapování tickerů/CIK a respektování limitů.
        - **Volitelně FMP nebo Alpha Vantage** — bezplatné limity se mohou měnit a nejsou vhodné pro neomezené skenování velkého globálního univerza.

        **Důležité metodické limity**
        1. Aktuální fundamenty nesmí být použity v historickém backtestu jako by byly známy tehdy.
        2. Současný seznam tickerů vytváří survivorship bias.
        3. Sociální sentiment a analytické revize nejsou v této první verzi zahrnuty.
        4. Daň není zahrnuta do výsledků backtestu. Pro českého daňového rezidenta je potřeba lot-level evidence a přesná implementace aktuálních daňových pravidel.
        5. Cenová data a jejich dostupnost se mohou měnit; ověř práva k použití a případnou možnost redistribuce.

        **Doporučené použití:** screening a výzkumný prototyp, ne automatické zadávání pokynů ani tvrzení o očekávaném výnosu.
        """)
