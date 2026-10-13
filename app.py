
from __future__ import annotations

import io
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from qlab.data import download_market_data, load_sec_ticker_map, load_sec_company_facts, sec_fundamental_snapshot, sec_snapshot_to_history
from qlab.strategy import prepare_screener, calculate_indicators, run_monthly_backtest, calculate_position_sizes
from qlab.optional_signals import parse_signal_csv, latest_signals, signal_history
from qlab.tax_lots import fifo_ledger

APP_VERSION = "2.0.0"
DEFAULT_UNIVERSE = "MSFT,AAPL,GOOGL,AMZN,META,NVDA,AVGO,ORCL,ADBE,CRM,CSCO,INTU,LLY,JNJ,AZN,NVO,RELX,OTIS,ASML,TSM"
BENCHMARKS = ("SPY",)
EXCLUDED_FUNDS = {"SPY", "ACWI", "QQQ", "VTI"}

st.set_page_config(page_title="Quality Momentum Lab", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background: radial-gradient(1100px 520px at 10% -5%, rgba(54,111,138,.24), transparent 65%), #0b1017; }
    [data-testid="stSidebar"] { background: #0e1621; border-right: 1px solid rgba(142,166,191,.16); }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px; }
    .qlab-eyebrow { color: #72e0c1; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
    .qlab-title { font-size: clamp(2rem, 4vw, 3.3rem); letter-spacing: -.045em; font-weight: 800; line-height: 1.05; margin: .3rem 0 .6rem 0; }
    .qlab-subtitle { color: #b4c2d0; font-size: 1rem; max-width: 900px; line-height: 1.55; }
    .qlab-panel { padding: 1rem 1.1rem; background: linear-gradient(135deg, rgba(25,39,54,.95), rgba(16,25,36,.92)); border: 1px solid rgba(151,180,204,.18); border-radius: 16px; }
    .qlab-small { color: #92a4b7; font-size: .83rem; }
    div[data-testid="stMetric"] { background: rgba(24,37,50,.72); border: 1px solid rgba(142,166,191,.15); border-radius: 14px; padding: .8rem 1rem; }
    div[data-testid="stMetricLabel"] { color: #92a4b7; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; padding: .55rem 1rem; }
    div.stButton > button[kind="primary"] { background: #72e0c1; color: #09221d; border: 0; font-weight: 800; }
    div.stButton > button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_market_data(symbols: tuple[str, ...], period: str):
    return download_market_data(list(symbols), period)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_sec_map(email: str):
    return load_sec_ticker_map(email)

@st.cache_data(ttl=86400, show_spinner=False)
def cached_sec_bundle(ticker: str, email: str, cik: int, title: str):
    return load_sec_company_facts(ticker, email, {ticker: {"cik": int(cik), "title": title}})

def _secret_email():
    try:
        value = st.secrets.get("SEC_CONTACT_EMAIL", "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get("SEC_CONTACT_EMAIL", "")

def _signals(uploaded, score_column):
    if uploaded is None:
        return None
    try:
        return parse_signal_csv(io.BytesIO(uploaded.getvalue()), score_column)
    except Exception as exc:
        st.sidebar.error(f"Soubor {uploaded.name}: {exc}")
        return None

def _format_detail_dict(row):
    out = {}
    for key, value in row.items():
        try:
            if pd.isna(value):
                out[str(key)] = ""
                continue
        except (TypeError, ValueError):
            pass
        out[str(key)] = str(value)
    return out

def _equity_chart(close, ma50=None, ma200=None, title="Cenový vývoj"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=close.index, y=close.values, name="Close", line=dict(width=2.2, color="#72e0c1")))
    if ma50 is not None:
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="MA50", line=dict(width=1.3, color="#f6bd60")))
    if ma200 is not None:
        fig.add_trace(go.Scatter(x=ma200.index, y=ma200.values, name="MA200", line=dict(width=1.5, color="#e07a9a")))
    fig.update_layout(title=title, template="plotly_dark", height=380, margin=dict(l=12,r=12,t=48,b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title=None,
        yaxis_title="Cena v měně kotace", hovermode="x unified")
    return fig

def _daily_returns(frame):
    col = "Adj Close" if "Adj Close" in frame else "Close"
    return pd.to_numeric(frame[col], errors="coerce").dropna()

def _make_fundamental_history(prices, bundles, use_estimated_value):
    close_frame = pd.DataFrame({
        ticker: frame["Adj Close"] if "Adj Close" in frame else frame["Close"]
        for ticker, frame in prices.items() if not frame.empty
    })
    monthly_dates = list(close_frame.resample("ME").last().dropna(how="all").index)
    histories = {}
    for ticker, bundle in bundles.items():
        if ticker not in prices or not monthly_dates:
            continue
        history = sec_snapshot_to_history(bundle, monthly_dates, prices[ticker]["Close"])
        if not use_estimated_value:
            for column in ["FCF yield %", "P/E (annual EPS proxy)"]:
                if column in history:
                    history[column] = np.nan
        histories[ticker] = history
    return histories

def _download_frame(prices):
    series = []
    for ticker, frame in prices.items():
        if not frame.empty:
            col = "Adj Close" if "Adj Close" in frame else "Close"
            series.append(frame[col].rename(ticker))
    return pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()

# -------------------------------- Controls --------------------------------
with st.sidebar:
    st.markdown("### QUALITY MOMENTUM LAB")
    st.caption(f"Research console · build {APP_VERSION}")
    ticker_text = st.text_area("Investiční univerzum", DEFAULT_UNIVERSE, height=106)
    period = st.selectbox("Historie cen", ["5y", "10y", "max"], index=1)
    st.caption("Ceny: Yahoo Finance přes neoficiální yfinance klient. Ověř podmínky poskytovatele pro veřejné použití.")
    st.divider()
    st.subheader("SEC fundamenty")
    sec_email = st.text_input("Kontaktní e-mail pro SEC EDGAR", value=_secret_email(),
        help="Odesílá se v identifikační hlavičce User-Agent při požadavcích na SEC. Lze nastavit přes Streamlit secret SEC_CONTACT_EMAIL.")
    use_estimated_value = st.checkbox("Použít orientační ocenění (P/E a FCF yield)", value=False,
        help="Odhady používají poslední nahlášené akcie v oběhu × cenu; u ADR, dual-class akcií nebo zastaralého počtu akcií mohou být nepřesné.")
    st.divider()
    st.subheader("Pravidla portfolia")
    top_n = st.slider("Počet pozic", 3, 15, 5)
    min_adx = st.slider("Minimální ADX", 0, 40, 10)
    min_score = st.slider("Minimální vstupní skóre", 0, 80, 35)
    hysteresis = st.slider("Hystereze skóre", 0, 25, 7, help="Nová akcie musí být lepší o tento počet bodů, aby vytlačila stávající pozici.")
    max_replacements = st.slider("Max. výměn za rebalancování", 0, 15, 2)
    breakout_filter = st.selectbox("Vstupní průraz", ["None", "20-day", "55-day"], index=0)
    cost_bps = st.slider("Náklady na zobchodovaný objem (bps)", 0, 100, 15, 5)
    st.divider()
    capital = st.number_input("Modelový kapitál (USD)", min_value=1000.0, value=25000.0, step=1000.0)
    risk_budget_pct = st.slider("Riziko na pozici (% kapitálu)", 0.1, 2.0, 0.5, 0.1)
    stop_atr = st.slider("Stop distance (× ATR20)", 1.0, 5.0, 3.0, 0.5)
    max_position_pct = st.slider("Max. váha pozice (%)", 5, 40, 20, 1)
    max_exposure_pct = st.slider("Max. celková expozice (%)", 25, 100, 95, 5)
    st.divider()
    st.subheader("Časované doplňkové signály")
    sentiment_upload = st.file_uploader("Sentiment CSV (0–100)", type=["csv"], key="sentiment_csv")
    revision_upload = st.file_uploader("Revize odhadů CSV (0–100)", type=["csv"], key="revision_csv")
    st.caption("Požadované sloupce: Date, Ticker, Sentiment score / Revision score, Source je volitelný.")
    if st.button("Obnovit všechna data", type="primary", width="stretch"):
        st.cache_data.clear()
        st.session_state.pop("backtest_result", None)
        st.rerun()

universe = [x for x in dict.fromkeys(s.strip().upper() for s in ticker_text.split(",") if s.strip()) if x not in EXCLUDED_FUNDS]
if len(universe) < 2:
    st.error("Zadej alespoň dvě akcie. Benchmarková ETF SPY/ACWI/QQQ/VTI se automaticky vyřazují z investičního univerza.")
    st.stop()
symbols = tuple(dict.fromkeys(universe + list(BENCHMARKS)))
sentiment_df = _signals(sentiment_upload, "Sentiment score")
revision_df = _signals(revision_upload, "Revision score")
sentiment_latest = latest_signals(sentiment_df, "Sentiment score") if sentiment_df is not None else {}
revision_latest = latest_signals(revision_df, "Revision score") if revision_df is not None else {}
optional_latest = {}
for ticker in set(sentiment_latest) | set(revision_latest):
    merged = {}
    merged.update(sentiment_latest.get(ticker, {}))
    merged.update(revision_latest.get(ticker, {}))
    optional_latest[ticker] = merged

with st.spinner("Načítám cenová data v dávce, bez paralelních požadavků…"):
    try:
        price_data_all, price_report = cached_market_data(symbols, period)
    except Exception as exc:
        st.error(f"Stahování cen selhalo: {type(exc).__name__}: {exc}")
        st.stop()
prices = {ticker: price_data_all[ticker] for ticker in universe if ticker in price_data_all and not price_data_all[ticker].empty}
if not prices:
    st.error("Zdroj nevrátil použitelnou cenu pro žádný zadaný ticker. Zkontroluj omezení zdroje nebo to zkus později.")
    st.stop()

# ------------------------------- SEC facts --------------------------------
fundamentals, sec_bundles, sec_errors = {}, {}, []
valid_email = "@" in sec_email and "." in sec_email.split("@")[-1]
if valid_email:
    try:
        sec_map = cached_sec_map(sec_email.strip())
        missing_map = [t for t in universe if t not in sec_map and t not in EXCLUDED_FUNDS]
        valid_tickers = [t for t in universe if t in sec_map and t not in EXCLUDED_FUNDS]
        progress = st.progress(0, text="Načítám SEC EDGAR fundamentals…")
        for idx, ticker in enumerate(valid_tickers):
            try:
                issuer = sec_map[ticker]
                bundle = cached_sec_bundle(ticker, sec_email.strip(), int(issuer["cik"]), str(issuer["title"]))
                sec_bundles[ticker] = bundle
                raw_close = prices[ticker]["Close"].dropna() if ticker in prices else pd.Series(dtype=float)
                last_price = float(raw_close.iloc[-1]) if not raw_close.empty else np.nan
                fundamentals[ticker] = sec_fundamental_snapshot(bundle, datetime.now(timezone.utc).date(), last_price)
            except Exception as exc:
                sec_errors.append({"Ticker": ticker, "Error": f"{type(exc).__name__}: {exc}"})
            progress.progress((idx + 1) / max(1, len(valid_tickers)), text=f"SEC data: {idx + 1}/{len(valid_tickers)}")
        progress.empty()
        for ticker in missing_map:
            fundamentals[ticker] = {"Ticker": ticker, "Company": ticker, "Fundamental status": "No SEC mapping", "Data note": "Ticker not found in SEC company ticker map."}
    except Exception as exc:
        st.warning(f"SEC EDGAR se nepodařilo načíst: {type(exc).__name__}: {exc}. Screener poběží v režimu cenových dat.")
else:
    st.sidebar.info("Fundamentální vrstva je vypnutá. Zadej kontaktní e-mail pro načtení oficiálních SEC EDGAR výkazů. Žádné fundamentální údaje se nedoplňují skóre 50.")

screener = prepare_screener(prices, fundamentals, optional_signals=optional_latest, allow_estimated_valuation=use_estimated_value)
if screener.empty:
    st.error("Z cenových dat se nepodařilo vytvořit screener.")
    st.stop()

# ------------------------------ Main header ------------------------------
st.markdown('<div class="qlab-eyebrow">Systematic equity research · v2</div>', unsafe_allow_html=True)
st.markdown('<div class="qlab-title">Quality × Value × Momentum</div>', unsafe_allow_html=True)
st.markdown('<div class="qlab-subtitle">Výzkumné prostředí pro výběr akcií, dlouhodobé momentum, fundamentální kvalitu a řízení obratu portfolia. Důraz je na dohledatelných zdrojích, časové správnosti a výsledcích po nákladech — nikoli na falešné přesnosti.</div>', unsafe_allow_html=True)
latest_price_date = max(frame.index.max() for frame in prices.values())
stamp = pd.Timestamp(latest_price_date)
age_days = (pd.Timestamp.now().normalize() - stamp.tz_localize(None).normalize()).days if stamp.tzinfo else (pd.Timestamp.now().normalize() - stamp.normalize()).days
valid_current_fundamentals = sum(1 for t in fundamentals if fundamentals[t].get("Source") == "SEC EDGAR Company Facts" and fundamentals[t].get("Fundamental status") == "OK")
trend_count = int(screener["Trend OK"].fillna(False).sum())
k1,k2,k3,k4 = st.columns(4)
k1.metric("Akcie v univerzu", str(len(universe)), f"{len(prices)} cenových řad")
k2.metric("Cenová data", stamp.date().isoformat(), "zdrojové datum")
k3.metric("SEC fundamenty", f"{valid_current_fundamentals}/{len(universe)}", "úplnější snapshoty")
k4.metric("V trendu", f"{trend_count}/{len(screener)}", "MA50 / MA200 / sklon MA200")
if age_days > 7:
    st.warning(f"Pozor: poslední dostupná cena je stará {age_days} dní. Výsledek nemusí odrážet aktuální trh.")
if sec_errors:
    with st.expander(f"Chyby SEC načítání ({len(sec_errors)})"):
        st.dataframe(pd.DataFrame(sec_errors), width="stretch", hide_index=True)
if not use_estimated_value:
    st.info("Valuační skóre je ve výchozím nastavení vypnuté: kapitalizace odvozená z posledního nahlášeného počtu akcií nemusí odpovídat obchodované třídě/ADR. Zapnutí používá P/E a FCF yield jako orientační odhady.")
if sentiment_df is None and revision_df is None:
    st.caption("Sentiment a revize analytických odhadů jsou vypnuté, dokud není přiložen historicky časovaný zdroj.")

tab_overview, tab_screener, tab_strategy, tab_portfolio, tab_data, tab_method = st.tabs([
    "Přehled", "Screener", "Strategie & backtest", "Portfolio & loty", "Kvalita dat", "Metodika"
])

with tab_overview:
    left,right = st.columns([1.3,1])
    with left:
        st.subheader("Kandidáti podle skóre")
        best = screener.sort_values("Fundamental/composite score", ascending=False, na_position="last").head(10)
        fig = px.bar(best.sort_values("Fundamental/composite score", ascending=True), x="Fundamental/composite score", y="Ticker",
            orientation="h", title="Composite score (0–100)", template="plotly_dark",
            color="Fundamental/composite score", color_continuous_scale=["#243347","#438b91","#72e0c1"])
        fig.update_layout(height=430, margin=dict(l=10,r=10,t=45,b=10), coloraxis_showscale=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Signály")
        st.dataframe(screener["Signal"].value_counts().rename_axis("Signal").reset_index(name="Počet"), width="stretch", hide_index=True)
        p1,p2 = st.columns(2)
        p1.metric("Pokrytí fundamenty", f"{screener['Fundamental coverage %'].mean():.0f}%")
        p2.metric("Breakout 55 dní", str(int(screener["Breakout55"].fillna(False).sum())))
        st.markdown('<div class="qlab-panel"><b>Interpretace skóre</b><br><span class="qlab-small">Skóre je relativní pořadí uvnitř právě načteného univerza, nikoli očekávaný výnos. Chybějící fundamenty se nemění na neutrální hodnoty.</span></div>', unsafe_allow_html=True)

with tab_screener:
    st.subheader("Screener a podkladové metriky")
    search_col,signal_col,trend_col = st.columns([1.2,1,1])
    search = search_col.text_input("Filtrovat ticker nebo název", "")
    signal_options = ["Vše"] + sorted(screener["Signal"].dropna().astype(str).unique().tolist())
    signal_filter = signal_col.selectbox("Signál", signal_options, index=0)
    trend_filter = trend_col.selectbox("Trend", ["Vše","Pouze Trend OK","Pouze mimo trend"], index=0)
    shown = screener.copy()
    if search:
        shown = shown[shown["Ticker"].str.contains(search.upper(), na=False) | shown["Company"].astype(str).str.contains(search, case=False, na=False)]
    if signal_filter != "Vše":
        shown = shown[shown["Signal"] == signal_filter]
    if trend_filter == "Pouze Trend OK":
        shown = shown[shown["Trend OK"]]
    elif trend_filter == "Pouze mimo trend":
        shown = shown[~shown["Trend OK"]]
    display_cols = [
        "Ticker","Company","Price","Fundamental/composite score","Technical score","Quality score","Value score",
        "Growth score","Momentum score","Fundamental coverage %","Fundamental status","ROE %","Profit margin %",
        "FCF margin %","Revenue growth %","Earnings growth %","FCF yield %","Momentum 12-1M","Momentum 6-1M",
        "ADX14","ATR20%","Breakout20","Breakout55","Trend OK","Below 50d low","Signal","Price date","Latest filing date"
    ]
    display_cols = [c for c in display_cols if c in shown.columns]
    st.dataframe(shown.sort_values("Fundamental/composite score", ascending=False, na_position="last")[display_cols],
        width="stretch", hide_index=True, height=520, column_config={
        "Fundamental/composite score": st.column_config.ProgressColumn("Composite score", min_value=0, max_value=100, format="%.1f"),
        "Technical score": st.column_config.ProgressColumn("Technical score", min_value=0, max_value=100, format="%.1f"),
        "Quality score": st.column_config.NumberColumn(format="%.1f"), "Value score": st.column_config.NumberColumn(format="%.1f"),
        "Growth score": st.column_config.NumberColumn(format="%.1f"), "Momentum score": st.column_config.NumberColumn(format="%.1f"),
        "Fundamental coverage %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Price": st.column_config.NumberColumn(format="%.2f"), "Momentum 12-1M": st.column_config.NumberColumn(format="%.1%"),
        "Momentum 6-1M": st.column_config.NumberColumn(format="%.1%"), "ATR20%": st.column_config.NumberColumn(format="%.2f%"),
        "ADX14": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("Stáhnout kompletní screener CSV", shown.to_csv(index=False).encode("utf-8"), "quality_momentum_screener.csv", "text/csv")
    st.divider()
    selected_ticker = st.selectbox("Detail akcie", screener["Ticker"].tolist())
    if selected_ticker in prices:
        frame = prices[selected_ticker]
        indicators = calculate_indicators(frame)
        st.plotly_chart(_equity_chart(frame["Close"].dropna(), indicators["MA50"], indicators["MA200"],
            f"{selected_ticker} · Close / MA50 / MA200"), width="stretch")
        row = screener.loc[screener["Ticker"] == selected_ticker].iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("Composite", "—" if pd.isna(row["Fundamental/composite score"]) else f"{row['Fundamental/composite score']:.1f}/100")
        b.metric("12–1M momentum", "—" if pd.isna(row["Momentum 12-1M"]) else f"{row['Momentum 12-1M']:.1%}")
        c.metric("ADX(14)", "—" if pd.isna(row["ADX14"]) else f"{row['ADX14']:.1f}")
        d.metric("ATR(20) %", "—" if pd.isna(row["ATR20%"]) else f"{row['ATR20%']:.2f}%")
        with st.expander("Všechny metriky a metadata"):
            st.json(_format_detail_dict(row.to_dict()))

with tab_strategy:
    st.subheader("Měsíční rebalancování · nízká obrátka")
    st.markdown("Signál se vytváří na konci měsíce a výnos portfolia se počítá za následující měsíc. Náklady se aplikují na hrubý zobchodovaný objem. Hystereze omezuje výměny při malém rozdílu skóre.")
    sec_coverage = sum(1 for t in sec_bundles if t in prices) / max(1, len(universe))
    default_pit = sec_coverage >= 0.70
    use_pit = st.checkbox(f"Zapnout point-in-time SEC fundamenty v backtestu (coverage nyní {sec_coverage:.0%})",
        value=default_pit, disabled=sec_coverage < 0.40,
        help="Historický signál používá jen fakta zveřejněná do dne rozhodnutí. Zůstává survivorship bias, protože univerzum je současný seznam.")
    st.caption("Signálové parametry jsou vlevo. Stop-loss 50denní low / 3×ATR je v kalkulátoru pozic; tento měsíční backtest nepředstírá intradenní exekuci stopů.")
    signature = (tuple(universe), period, top_n, min_adx, min_score, hysteresis, max_replacements,
        cost_bps, breakout_filter, use_pit, use_estimated_value, bool(sentiment_df is not None),
        bool(revision_df is not None), capital)
    run_backtest = st.button("Spustit / přepočítat backtest", type="primary", key="run_backtest")
    saved = st.session_state.get("backtest_result")
    if run_backtest:
        benchmark_series = _daily_returns(price_data_all["SPY"]) if "SPY" in price_data_all else None
        fund_histories, optional_histories = {}, {}
        try:
            with st.spinner("Počítám historické signály, point-in-time fundamenty a rebalancování…"):
                if use_pit and sec_bundles:
                    fund_histories = _make_fundamental_history(prices, sec_bundles, use_estimated_value)
                all_closes = pd.DataFrame({t: f["Adj Close"] if "Adj Close" in f else f["Close"] for t,f in prices.items() if not f.empty})
                monthly_dates = list(all_closes.resample("ME").last().dropna(how="all").index)
                if sentiment_df is not None:
                    optional_histories.update(signal_history(sentiment_df, "Sentiment score", monthly_dates))
                if revision_df is not None:
                    optional_histories.update(signal_history(revision_df, "Revision score", monthly_dates))
                curve, metrics, audit = run_monthly_backtest(
                    prices=prices, benchmark=benchmark_series, top_n=top_n, cost_bps=cost_bps,
                    initial=capital, min_adx=min_adx, min_score=min_score, hysteresis=hysteresis,
                    max_replacements=max_replacements, use_fundamentals=use_pit,
                    fundamentals_history=fund_histories, optional_history=optional_histories,
                    require_breakout=breakout_filter)
                st.session_state["backtest_result"] = {"curve":curve,"metrics":metrics,"audit":audit,"signature":signature,"error":None}
        except Exception as exc:
            st.session_state["backtest_result"] = {"curve":None,"metrics":None,"audit":None,"signature":signature,"error":f"{type(exc).__name__}: {exc}"}
        saved = st.session_state.get("backtest_result")
    if saved is None:
        st.info("Backtest ještě nebyl spuštěn. Spusť výpočet tlačítkem výše.")
    elif saved.get("signature") != signature:
        st.warning("Výsledky odpovídají předchozí konfiguraci. Spusť přepočet.")
    elif saved.get("error"):
        st.error("Výsledek není vykázán, protože data/return series nejsou úplná: " + saved["error"])
    elif saved.get("curve") is None:
        st.warning("Nedostatek historie nebo platných titulů pro backtest.")
    else:
        curve, metrics, audit = saved["curve"], saved["metrics"], saved["audit"]
        a,b,c,d = st.columns(4)
        a.metric("CAGR strategie", f"{metrics['CAGR']:.1%}")
        b.metric("Volatilita p.a.", f"{metrics['Volatility']:.1%}")
        c.metric("Sharpe (rf=0)", "—" if pd.isna(metrics["Sharpe (rf=0)"]) else f"{metrics['Sharpe (rf=0)']:.2f}")
        d.metric("Max drawdown", f"{metrics['Max drawdown']:.1%}")
        e,f,g,h = st.columns(4)
        e.metric("Konečná hodnota", f"${metrics['Final equity']:,.0f}")
        f.metric("Testované měsíce", str(metrics["Months"]))
        g.metric("Obrat / měsíc", f"{metrics['Average monthly turnover %']:.1f}%")
        h.metric("Součet modelových nákladů", f"{metrics['Total transaction costs %']:.2f}%")
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=curve.index,y=curve["Equity"],name="Strategie",line=dict(width=2.5,color="#72e0c1")))
        if "Benchmark equity" in curve:
            equity_fig.add_trace(go.Scatter(x=curve.index,y=curve["Benchmark equity"],name="Benchmark SPY",line=dict(width=1.7,color="#f6bd60")))
        equity_fig.update_layout(template="plotly_dark",title="Růst modelového kapitálu",height=410,
            margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",hovermode="x unified")
        st.plotly_chart(equity_fig, width="stretch")
        ret_fig = px.bar(curve.reset_index(), x="Date", y="Net return", title="Čistý měsíční výnos", template="plotly_dark")
        ret_fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(ret_fig, width="stretch")
        with st.expander("Audit každého rebalancování"):
            st.dataframe(audit, width="stretch", hide_index=True)
        st.dataframe(curve, width="stretch")
        st.download_button("Stáhnout equity curve CSV",curve.to_csv().encode("utf-8"),"backtest_curve.csv","text/csv")
        st.download_button("Stáhnout audit obchodních rozhodnutí CSV",audit.to_csv(index=False).encode("utf-8"),"backtest_audit.csv","text/csv")
        st.warning("Tento test stále není survivorship-bias-free: vychází ze současného uživatelského seznamu tickerů. Close-to-close model nákladů nenahrazuje plnou exekuci na open/spread/slippage a daňový model.")

with tab_portfolio:
    st.subheader("ATR position sizing a evidence lotů")
    st.markdown("Kalkulátor převádí maximální riziko na počet kusů při zadaném ATR stopu a limitech expozice. Nezadává pokyny brokerovi.")
    candidates = screener.sort_values("Fundamental/composite score",ascending=False,na_position="last")
    suggested = candidates.loc[candidates["Trend OK"],"Ticker"].head(min(top_n,5)).tolist()
    picked = st.multiselect("Tituly ke kalkulaci",screener["Ticker"].tolist(),default=suggested)
    sizes = calculate_position_sizes(screener,picked,capital,risk_budget_pct=risk_budget_pct,
        stop_atr_multiple=stop_atr,max_position_pct=max_position_pct,max_total_exposure_pct=max_exposure_pct)
    if sizes.empty:
        st.info("Vyber akcie s dostupnou cenou a ATR(20).")
    else:
        a,b,c = st.columns(3)
        a.metric("Modelová alokace",f"${sizes['Position value'].sum():,.0f}")
        b.metric("Expozice kapitálu",f"{sizes['Weight %'].sum():.1f}%")
        c.metric("Odhadované počáteční riziko",f"${sizes['Initial risk budget'].sum():,.0f}")
        st.dataframe(sizes,width="stretch",hide_index=True,column_config={
            "Weight %":st.column_config.NumberColumn(format="%.2f%%"),"Price":st.column_config.NumberColumn(format="%.2f"),
            "ATR20":st.column_config.NumberColumn(format="%.2f"),"Provisional stop":st.column_config.NumberColumn(format="%.2f"),
            "Position value":st.column_config.NumberColumn(format="$%.2f"),"Initial risk budget":st.column_config.NumberColumn(format="$%.2f")})
        st.download_button("Stáhnout plán pozic CSV",sizes.to_csv(index=False).encode("utf-8"),"position_sizing.csv","text/csv")
    st.divider()
    st.markdown("#### FIFO tax-lot ledger")
    st.write("Nahraj historii transakcí. Výpočet páruje prodeje s nejstaršími nákupními loty, započítá dodané poplatky a ukáže realizovaný zisk/ztrátu i zbylé loty. Neaplikuje české daňové osvobození, sazby, časové testy ani přepočet měn.")
    template = pd.DataFrame([
        {"Date":"2024-01-15","Ticker":"MSFT","Type":"BUY","Quantity":10,"Price":400,"Fees":1,"Currency":"USD"},
        {"Date":"2025-03-01","Ticker":"MSFT","Type":"SELL","Quantity":4,"Price":420,"Fees":1,"Currency":"USD"},
    ])
    st.download_button("Stáhnout ukázkový CSV formát lotů",template.to_csv(index=False).encode("utf-8"),"tax_lot_template.csv","text/csv")
    ledger_upload = st.file_uploader("Transakční historie CSV",type=["csv"],key="tax_lot_file")
    if ledger_upload is not None:
        try:
            transactions = pd.read_csv(io.BytesIO(ledger_upload.getvalue()))
            realized,open_lots,annual = fifo_ledger(transactions)
            a,b,c = st.columns(3)
            a.metric("Realizovaný zisk / ztráta",f"{realized['Realized gain/loss'].sum():,.2f}" if not realized.empty else "0.00")
            b.metric("Realizované páry",str(len(realized)))
            c.metric("Otevřené loty",str(len(open_lots)))
            if not realized.empty:
                st.markdown("**Realizované páry (FIFO)**")
                st.dataframe(realized,width="stretch",hide_index=True)
                st.download_button("Stáhnout realizované páry",realized.to_csv(index=False).encode("utf-8"),"fifo_realized.csv","text/csv")
            if not open_lots.empty:
                st.markdown("**Otevřené loty**")
                st.dataframe(open_lots,width="stretch",hide_index=True)
                st.download_button("Stáhnout otevřené loty",open_lots.to_csv(index=False).encode("utf-8"),"fifo_open_lots.csv","text/csv")
            st.markdown("**Roční souhrn podle měny a tickeru**")
            st.dataframe(annual,width="stretch",hide_index=True)
        except Exception as exc:
            st.error(f"FIFO import selhal: {type(exc).__name__}: {exc}")

with tab_data:
    st.subheader("Kontrola kvality dat")
    st.markdown("Chybná data vedou k sebejistým, ale chybným výsledkům. Proto je zde samostatný panel zdrojů a úplnosti.")
    data_cols = [c for c in ["Ticker","Status","Rows","First date","Last date","Missing %","Message"] if c in price_report.columns]
    st.dataframe(price_report[data_cols],width="stretch",hide_index=True,height=430)
    st.download_button("Stáhnout quality report CSV",price_report.to_csv(index=False).encode("utf-8"),"data_quality_report.csv","text/csv")
    combined = _download_frame(price_data_all)
    if not combined.empty:
        st.download_button("Stáhnout adjustované cenové řady CSV",combined.to_csv(index_label="Date").encode("utf-8"),"adjusted_price_history.csv","text/csv")
    if valid_email:
        sec_report = []
        for ticker in universe:
            record = fundamentals.get(ticker,{})
            sec_report.append({"Ticker":ticker,"Source":record.get("Source","No SEC data"),
                "Financial period":record.get("Financial period",""),"Latest filing date":record.get("Latest filing date",""),
                "Fundamental status":record.get("Fundamental status","Not loaded"),"Data note":record.get("Data note","")})
        st.markdown("#### Fundamentální zdroj a stáří výkazů")
        sec_df = pd.DataFrame(sec_report)
        st.dataframe(sec_df,width="stretch",hide_index=True)
        st.download_button("Stáhnout SEC provenance CSV",sec_df.to_csv(index=False).encode("utf-8"),"sec_provenance.csv","text/csv")
    if price_report["Status"].eq("MISSING").any():
        st.error("Některé cenové řady chybí. Backtest se nesmí používat bez vyřešení dat.")
    else:
        st.success("Cenové řady byly vráceny pro všechny tickery označené jako dostupné při tomto stažení.")
    st.caption(f"Načteno v tomto běhu · {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    st.caption("Dostupná historická řada neznamená ověřenou burzovní exekuci; poskytovatel může data opravit nebo omezit.")

with tab_method:
    st.subheader("Pravidla modelu")
    st.markdown(
        """
        ### Faktory a relativní skóre
        - **Quality:** ROE, zisková marže, FCF marže.
        - **Value:** FCF yield a kladné P/E pouze při zapnutí orientačního ocenění. Chybějící pole zůstávají chybějící.
        - **Growth:** růst ročních tržeb a zisku z veřejných výkazů; nejde o konsenzuální revize analytiků.
        - **Momentum:** 12–1M a 6–1M s vynecháním posledních přibližně 21 obchodních dnů.
        - **Trend:** Close nad MA200, MA50 nad MA200 a rostoucí MA200; ADX(14) měří sílu trendu.
        - **Breakout:** Close nad předchozími high za 20 nebo 55 seancí, bez aktuální svíčky v referenčním maximu.
        - **Risk:** ATR(20), limit pozice a modelový stop ve vzdálenosti násobku ATR.

        ### Časová správnost
        Fundamentální údaje z SEC se filtrují podle filed date a historické rozhodnutí nepoužije výkaz zveřejněný později. Tím se ale neřeší survivorship bias současného tickerového seznamu.

        ### Omezení
        - Yahoo Finance přes yfinance je neoficiální rozhraní; ověř podmínky poskytovatele. Data nejsou licencovaným produkčním feedem.
        - SEC Company Facts pokrývá reportující emitenty a standardizované XBRL tagy; metriky se mohou lišit účetními standardy a dostupností tagu.
        - P/E a FCF yield používají hrubý odhad tržní kapitalizace z ceny a posledního nahlášeného počtu akcií. U ADR, duálních tříd a změn kapitálu mohou být chybné, proto jsou výchozím stavem mimo skóre.
        - Backtest používá současný seznam a není survivorship-bias-free. Měsíční close-to-close simulace není přesný broker fill model.
        - Sentiment/revize jsou volitelné časované CSV; historický model vybírá jen poslední známý záznam k rozhodovacímu dni.
        - FIFO eviduje pořizovací náklady a poplatky; neurčuje českou daňovou povinnost a nepřepočítává měny.
        - HMM je záměrně odložen, dokud základní strategie neprojde out-of-sample testem. Jinak by zvyšoval riziko přeoptimalizace.

        ### Cesta k důvěryhodným výsledkům
        Historické univerzum včetně delistovaných titulů, jasná práva k datům, point-in-time fundamenty, dividendy/FX/corporate actions, spread a skluz, walk-forward validace a předem odložené testovací období.
        """
    )
    st.markdown("**Roadmap:** [ROADMAP.md](https://github.com/Matthew-pros/quality-momentum-lab/blob/main/ROADMAP.md) · [Automatické testy](https://github.com/Matthew-pros/quality-momentum-lab/actions)")
    st.caption("Výzkumný nástroj; nejde o investiční, daňové ani právní doporučení.")
