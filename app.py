
from __future__ import annotations

import hashlib
import io
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from qlab.data import generate_demo_market_data, load_ohlcv_csv, load_sec_ticker_map, load_sec_company_facts, sec_fundamental_snapshot, sec_snapshot_to_history
from qlab.strategy import prepare_screener, calculate_indicators, run_monthly_backtest, calculate_position_sizes, cost_sensitivity, chronological_period_summary, moving_block_bootstrap_uncertainty, train_select_holdout_search, expanding_walk_forward, compare_stop_variants, industry_exposure_summary
from qlab.optional_signals import parse_signal_csv, latest_signals, signal_history
from qlab.tax_lots import fifo_ledger
from qlab.universe import parse_universe_membership, tickers_in_membership_file, universe_members_asof
from qlab.valuation import calculate_fcfe_dcf, dcf_sensitivity

APP_VERSION = "2.4.0"
DEFAULT_UNIVERSE = "MSFT,AAPL,GOOGL,AMZN,META,NVDA,AVGO,ORCL,ADBE,CRM,CSCO,INTU,LLY,JNJ,AZN,NVO,RELX,OTIS,ASML,TSM"
DEFAULT_DEMO_UNIVERSE = "SIM01,SIM02,SIM03,SIM04,SIM05,SIM06,SIM07,SIM08,SIM09,SIM10,SIM11,SIM12"
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

def safe_float(value, default=0.0):
    try:
        num = float(value)
        return num if np.isfinite(num) else default
    except (TypeError, ValueError):
        return default

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

def _download_ohlcv_long(prices):
    frames = []
    columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for ticker, frame in prices.items():
        if frame is None or frame.empty:
            continue
        part = frame.copy()
        part.index = pd.to_datetime(part.index)
        part["Ticker"] = str(ticker).upper()
        part["Date"] = part.index.date.astype(str)
        for column in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if column not in part:
                part[column] = np.nan
        frames.append(part[columns[1:]].assign(Date=part["Date"])[columns])
    return pd.concat(frames, ignore_index=True).sort_values(["Ticker", "Date"]) if frames else pd.DataFrame(columns=columns)

# -------------------------------- Controls --------------------------------
with st.sidebar:
    st.markdown("### QUALITY MOMENTUM LAB")
    st.caption(f"Research console · build {APP_VERSION}")
    price_provider = st.selectbox(
        "Zdroj cenových dat",
        [
            "Demo — syntetická data (bez externího API)",
            "Import OHLCV CSV (vlastní/licencovaná data)",
        ],
        index=0,
        help="Veřejný režim se spouští nad syntetickými daty, aby nepublikoval tržní data bez ověřené licence. Pro skutečné ceny nahraj CSV ze zdroje, který dovoluje veřejné zobrazení."
    )
    is_demo_mode = price_provider.startswith("Demo")
    ticker_text = st.text_area(
        "Investiční univerzum",
        DEFAULT_DEMO_UNIVERSE if is_demo_mode else DEFAULT_UNIVERSE,
        height=106,
        key="demo_universe" if is_demo_mode else "market_universe",
        disabled=is_demo_mode,
        help="Demo režim používá SIM01–SIM12 a generuje pouze syntetické řady. Při nahrání CSV historického členství se tickery z tohoto CSV stanou pro backtest autoritativním univerzem."
    )
    price_csv_upload = st.file_uploader(
        "Historie OHLCV CSV",
        type=["csv"],
        key="ohlcv_csv",
        disabled=price_provider != "Import OHLCV CSV (vlastní/licencovaná data)",
        help="Long format: Date,Ticker,Open,High,Low,Close; volitelně Adj Close,Volume. Jedna řádka na ticker a datum."
    )
    universe_history_upload = st.file_uploader(
        "Historické složení investičního univerza (volitelné CSV)",
        type=["csv"],
        key="universe_membership_csv",
        disabled=is_demo_mode or price_provider != "Import OHLCV CSV (vlastní/licencovaná data)",
        help="Intervaly členství: Ticker,Start date,End date. End date může být prázdné pro otevřený interval. Ticker list v souboru bude autoritativní; zahrň i později delistované tituly."
    )
    membership_template = pd.DataFrame([
        {"Ticker":"AAA","Start date":"2015-01-01","End date":"2020-12-31"},
        {"Ticker":"AAA","Start date":"2021-01-01","End date":""},
        {"Ticker":"BBB","Start date":"2018-06-01","End date":""},
    ])
    st.download_button(
        "Stáhnout šablonu členství CSV",
        membership_template.to_csv(index=False).encode("utf-8"),
        "universe_membership_template.csv", "text/csv",
        help="Jedna řádka na interval členství; Start/End jsou včetně uvedeného dne, prázdný End date znamená otevřený interval."
    )
    st.caption("Point-in-time univerzum snižuje survivorship bias pouze tehdy, když CSV obsahuje úplné historické členství i OHLCV řady všech tehdejších titulů včetně delistovaných.")
    period = st.selectbox("Historie cen", ["5y", "10y", "max"], index=1)
    if is_demo_mode:
        st.warning("DEMO: všechny ceny, výnosy a signály jsou syntetické, nejsou to kotace skutečných společností. SEC fundamenty se v demo režimu nenačítají.")
    else:
        st.caption("CSV režim nevolá žádný externí cenový API endpoint. Použij jen data, pro která máš práva k analýze a veřejnému zobrazení.")
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
    max_sector_positions = st.slider(
        "Max. pozic na SIC skupinu", 0, 8, 2,
        help="0 = vypnuto. Jde o hrubé SEC SIC industry group, nikoli GICS sektor. Chybějící klasifikace se nekontroluje; historické změny klasifikace nejsou k dispozici."
    )
    breakout_filter = st.selectbox("Vstupní průraz", ["None", "20-day", "55-day"], index=0)
    weighting_method = st.selectbox("Váhy pozic v backtestu", ["Equal weight", "Inverse ATR"], index=0,
        help="Inverse ATR váží nižší volatilitu vyšší vahou; limit pozice může ponechat část portfolia v hotovosti.")
    backtest_max_weight = st.slider("Max. váha pro Inverse ATR (%)", 10, 50, 35, 5, disabled=weighting_method != "Inverse ATR") / 100
    stop_mode = st.selectbox(
        "Výstupní pravidlo v backtestu",
        ["Month-end 50-day low", "No stop / month-end rebalancing", "ATR trailing stop", "Daily 50-day low", "ATR + daily 50-day low"],
        index=0,
        help="Denní stop varianty používají denní OHLC, stop založený na datech předchozího close a gap-aware fill na open. Není to intradenní simulace."
    )
    st.caption("ATR stop využívá hodnotu z pole „Stop distance (× ATR20)“ níže.")
    commission_bps = st.slider("Komise na zobchodovaný objem (bps)", 0, 20, 1, 1)
    half_spread_bps = st.slider("Odhad poloviny bid/ask spreadu (bps)", 0, 50, 5, 1)
    slippage_bps = st.slider("Skluz při exekuci (bps)", 0, 50, 5, 1)
    cost_bps = commission_bps + half_spread_bps + slippage_bps
    st.caption(f"Celkové modelované náklady: {cost_bps} bps na každou jednotku zobchodovaného objemu; nejde o odhad konkrétního brokera.")
    st.divider()
    capital = st.number_input("Modelový kapitál (USD)", min_value=1000.0, value=25000.0, step=1000.0)
    risk_budget_pct = st.slider("Riziko na pozici (% kapitálu)", 0.1, 2.0, 0.5, 0.1)
    max_total_risk_pct = st.slider("Max. souhrnné počáteční riziko portfolia (%)", 0.5, 10.0, 3.0, 0.5,
        help="Celkový modelový risk do ATR stopů napříč vybranými pozicemi. Po zaokrouhlení počtu akcií se rozpočet nemá překročit.")
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

universe = (
    [x for x in DEFAULT_DEMO_UNIVERSE.split(",") if x]
    if is_demo_mode
    else [x for x in dict.fromkeys(s.strip().upper() for s in ticker_text.split(",") if s.strip()) if x not in EXCLUDED_FUNDS]
)
benchmark_ticker = "SIM_BENCHMARK" if is_demo_mode else "SPY"
universe_history = None
universe_history_signature = "static-universe"
if universe_history_upload is not None:
    try:
        universe_history = parse_universe_membership(io.BytesIO(universe_history_upload.getvalue()))
        benchmark_rows = universe_history["Ticker"].isin(EXCLUDED_FUNDS)
        if benchmark_rows.any():
            st.warning("Benchmarková ETF byla z historického investičního univerza odstraněna: " + ", ".join(sorted(set(universe_history.loc[benchmark_rows, "Ticker"]))))
            universe_history = universe_history.loc[~benchmark_rows].reset_index(drop=True)
        historical_tickers = tickers_in_membership_file(universe_history)
        universe = historical_tickers
        universe_history_signature = hashlib.sha256(universe_history_upload.getvalue()).hexdigest()
    except Exception as exc:
        st.error(f"Historické složení univerza není validní: {type(exc).__name__}: {exc}")
        st.stop()
if len(universe) < 2:
    st.error("Zadej alespoň dvě akcie. Benchmarková ETF SPY/ACWI/QQQ/VTI se automaticky vyřazují z investičního univerza.")
    st.stop()
symbols = tuple(dict.fromkeys(universe + [benchmark_ticker]))
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

with st.spinner("Načítám cenová data a kontroluji kvalitu…"):
    try:
        if is_demo_mode:
            price_data_all, price_report = generate_demo_market_data(list(symbols), period)
        elif price_provider == "Import OHLCV CSV (vlastní/licencovaná data)":
            if price_csv_upload is None:
                st.info("Vybraný je CSV režim. Nahraj historii OHLCV v levém panelu a případně uprav investiční univerzum tak, aby odpovídalo tickerům v souboru.")
                st.stop()
            price_data_all, price_report = load_ohlcv_csv(
                io.BytesIO(price_csv_upload.getvalue()),
                expected_tickers=list(symbols),
            )
        else:
            raise ValueError("Nepodporovaný zdroj cenových dat. Zvol demo nebo import OHLCV CSV.")
    except Exception as exc:
        st.error(f"Načítání cen selhalo: {type(exc).__name__}: {exc}")
        st.stop()
prices = {ticker: price_data_all[ticker] for ticker in universe if ticker in price_data_all and not price_data_all[ticker].empty}
if not prices:
    st.error("Zdroj nevrátil použitelnou cenu pro žádný zadaný ticker. Zkontroluj omezení zdroje nebo to zkus později.")
    st.stop()
if universe_history is not None:
    membership_tickers = set(tickers_in_membership_file(universe_history))
    missing_price_history = sorted(membership_tickers - set(prices))
    if missing_price_history:
        st.error(
            "Historické složení univerza obsahuje tickery bez cenové historie v CSV: "
            + ", ".join(missing_price_history[:30])
            + (f" … a dalších {len(missing_price_history) - 30}" if len(missing_price_history) > 30 else "")
            + ". Pro point-in-time test je nutné dodat OHLCV i pro tehdejší tituly včetně těch později delistovaných."
        )
        st.stop()

price_data_signature = hashlib.sha256()
if not is_demo_mode and price_csv_upload is not None:
    price_data_signature.update(price_csv_upload.getvalue())
else:
    for ticker in sorted(prices):
        frame = prices[ticker]
        price_data_signature.update(ticker.encode("utf-8"))
        price_data_signature.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes())
price_data_signature = price_data_signature.hexdigest()

# Preserve source provenance in every downloadable data-quality report.
price_source_label = "Deterministic synthetic DEMO" if is_demo_mode else "User-supplied OHLCV CSV"
source_filename = "" if is_demo_mode or price_csv_upload is None else price_csv_upload.name
if "Price source" in price_report:
    price_report = price_report.drop(columns=["Price source"])
if "Source file" in price_report:
    price_report = price_report.drop(columns=["Source file"])
price_report.insert(0, "Price source", price_source_label)
price_report.insert(1, "Source file", source_filename)
if not is_demo_mode and price_csv_upload is not None:
    no_adj_close = price_report.get("Message", pd.Series("", index=price_report.index)).fillna("").astype(str).str.contains("Adj Close absent")
    price_report["Adjustment status"] = np.where(
        no_adj_close,
        "Close-only; dividend adjustment not included",
        "Adj Close supplied; corporate-action basis not independently verified",
    )
else:
    price_report["Adjustment status"] = "Synthetic adjusted series" if is_demo_mode else "Unknown"

# ------------------------------- SEC facts --------------------------------
fundamentals, sec_bundles, sec_errors = {}, {}, []
valid_email = (not is_demo_mode) and "@" in sec_email and "." in sec_email.split("@")[-1]
if valid_email:
    try:
        sec_map = cached_sec_map(sec_email.strip())
        missing_map = [t for t in universe if t not in sec_map and t not in EXCLUDED_FUNDS]
        valid_tickers = [t for t in universe if t in sec_map and t not in EXCLUDED_FUNDS]
        progress = st.progress(0, text="Načítám SEC EDGAR fundamentals…")
        for idx, ticker in enumerate(valid_tickers):
            # Keep request rate below the SEC fair-access ceiling when cache misses occur.
            if idx:
                time.sleep(0.12)
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
    if is_demo_mode:
        st.sidebar.info("SEC fundamenty jsou v demo režimu vypnuté, protože ceny jsou syntetické. Pro skutečný research přepni na vlastní/licencované cenové CSV a zadej SEC kontaktní e-mail.")
    else:
        st.sidebar.info("Fundamentální vrstva je vypnutá. Zadej kontaktní e-mail pro načtení oficiálních SEC EDGAR výkazů. Žádné fundamentální údaje se nedoplňují skóre 50.")

all_screener = prepare_screener(prices, fundamentals, optional_signals=optional_latest, allow_estimated_valuation=use_estimated_value)
if all_screener.empty:
    st.error("Z cenových dat se nepodařilo vytvořit screener.")
    st.stop()
sector_map_all = all_screener.set_index("Ticker")["SIC group"].fillna("Unclassified").astype(str).to_dict() if "SIC group" in all_screener else {}
screener = all_screener
membership_screener_is_current = universe_history is None
if universe_history is not None:
    last_observation_date = max(pd.Timestamp(frame.index.max()) for frame in prices.values() if not frame.empty)
    if last_observation_date.tzinfo is not None:
        last_observation_date = last_observation_date.tz_localize(None)
    current_members = universe_members_asof(universe_history, last_observation_date)
    current_screener = all_screener.loc[all_screener["Ticker"].isin(current_members or set())].copy()
    if not current_screener.empty:
        screener = current_screener
        st.info(
            f"Screener zobrazuje {len(screener)} titulů, které byly v univerzu k poslednímu datu cen {last_observation_date.date()}. "
            "Historický backtest používá členství podle data každého signálu. Historické SIC klasifikace zůstávají nedostupné."
        )
    else:
        membership_screener_is_current = False
        st.warning(
            f"Historický membership CSV nemá žádné aktivní členy k poslednímu datu cen ({last_observation_date.date()}). "
            "Screener zůstává jako historický přehled všech nahraných symbolů; nepoužívej ho jako seznam aktuálně obchodovatelných titulů."
        )

# ------------------------------ Main header ------------------------------
st.markdown(f'<div class="qlab-eyebrow">Systematic equity research · build {APP_VERSION}</div>', unsafe_allow_html=True)
st.markdown('<div class="qlab-title">Quality × Value × Momentum</div>', unsafe_allow_html=True)
st.markdown('<div class="qlab-subtitle">Výzkumné prostředí pro výběr akcií, dlouhodobé momentum, fundamentální kvalitu a řízení obratu portfolia. Důraz je na dohledatelných zdrojích, časové správnosti a výsledcích po nákladech — nikoli na falešné přesnosti.</div>', unsafe_allow_html=True)
latest_price_date = max(frame.index.max() for frame in prices.values())
stamp = pd.Timestamp(latest_price_date)
age_days = (pd.Timestamp.now().normalize() - stamp.tz_localize(None).normalize()).days if stamp.tzinfo else (pd.Timestamp.now().normalize() - stamp.normalize()).days
screener_tickers = set(screener["Ticker"].astype(str))
valid_current_fundamentals = sum(
    1 for t in screener_tickers
    if t in fundamentals and fundamentals[t].get("Source") == "SEC EDGAR Company Facts"
    and fundamentals[t].get("Fundamental status") == "OK"
)
trend_count = int(screener["Trend OK"].fillna(False).sum())
k1,k2,k3,k4 = st.columns(4)
universe_metric_label = "Členové screeneru" if universe_history is not None and membership_screener_is_current else "Tickery v datové sadě"
universe_metric_delta = f"{len(universe)} historických tickerů" if universe_history is not None else f"{len(prices)} cenových řad"
k1.metric(universe_metric_label, str(len(screener)), universe_metric_delta)
k2.metric("Cenová data", stamp.date().isoformat(), "zdrojové datum")
k3.metric("SEC fundamenty", f"{valid_current_fundamentals}/{len(screener)}", "úplnější snapshoty aktuálního screeneru")
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

tab_overview, tab_screener, tab_strategy, tab_valuation, tab_portfolio, tab_data, tab_method = st.tabs([
    "Přehled", "Screener", "Strategie & backtest", "DCF valuace", "Portfolio & loty", "Kvalita dat", "Metodika"
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
        st.subheader("Odvětvové skupiny podle SEC SIC")
        group_counts = screener["SIC group"].fillna("Unclassified").value_counts().rename_axis("SIC group").reset_index(name="Počet")
        st.dataframe(group_counts, width="stretch", hide_index=True, height=240)
        st.caption("Jde o hrubou skupinu podle SIC kódu, nikoli standardizovanou GICS/ICB sektorovou klasifikaci.")
        st.markdown('<div class="qlab-panel"><b>Interpretace skóre</b><br><span class="qlab-small">Skóre je relativní pořadí uvnitř právě načteného univerza, nikoli očekávaný výnos. Chybějící fundamenty se nemění na neutrální hodnoty.</span></div>', unsafe_allow_html=True)

with tab_screener:
    st.subheader("Screener a podkladové metriky")
    search_col,industry_col,signal_col,trend_col = st.columns([1.25,1.2,1,1])
    search = search_col.text_input("Filtrovat ticker nebo název", "")
    industry_options = ["Vše"] + sorted(screener["SIC group"].fillna("Unclassified").astype(str).unique().tolist())
    industry_filter = industry_col.selectbox("Odvětví (SEC SIC)", industry_options, index=0)
    signal_options = ["Vše"] + sorted(screener["Signal"].dropna().astype(str).unique().tolist())
    signal_filter = signal_col.selectbox("Signál", signal_options, index=0)
    trend_filter = trend_col.selectbox("Trend", ["Vše","Pouze Trend OK","Pouze mimo trend"], index=0)
    shown = screener.copy()
    if search:
        shown = shown[shown["Ticker"].str.contains(search.upper(), na=False) | shown["Company"].astype(str).str.contains(search, case=False, na=False)]
    if industry_filter != "Vše":
        shown = shown[shown["SIC group"].fillna("Unclassified").astype(str) == industry_filter]
    if signal_filter != "Vše":
        shown = shown[shown["Signal"] == signal_filter]
    if trend_filter == "Pouze Trend OK":
        shown = shown[shown["Trend OK"]]
    elif trend_filter == "Pouze mimo trend":
        shown = shown[~shown["Trend OK"]]
    display_cols = [
        "Ticker","Company","SIC group","SEC SIC","SEC SIC Description","Price","Fundamental/composite score","Technical score","Quality score","Value score",
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
        "Price": st.column_config.NumberColumn(format="%.2f"), "Momentum 12-1M": st.column_config.NumberColumn(format="percent"),
        "Momentum 6-1M": st.column_config.NumberColumn(format="percent"), "ATR20%": st.column_config.NumberColumn(format="%.2f%"),
        "ADX14": st.column_config.NumberColumn(format="%.1f"),
    })
    st.download_button("Stáhnout kompletní screener CSV", shown.to_csv(index=False).encode("utf-8"), "quality_momentum_screener.csv", "text/csv")
    st.divider()
    selected_ticker = st.selectbox("Detail akcie", screener["Ticker"].tolist())
    if selected_ticker in prices:
        frame = prices[selected_ticker]
        indicators = calculate_indicators(frame)
        chart_close = frame["Adj Close"].dropna() if "Adj Close" in frame else frame["Close"].dropna()
        st.plotly_chart(_equity_chart(chart_close, indicators["MA50"], indicators["MA200"],
            f"{selected_ticker} · Adjusted close / MA50 / MA200"), width="stretch")
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
    st.markdown("Signál vzniká na konci měsíce a vážený výnos se sleduje v následujícím měsíci. Výstupy lze porovnat jako žádný stop, měsíční close pod 50denním minimem, denní 50denní low, ATR trailing stop nebo kombinaci. Denní stop používá gap-aware plnění z OHLC, nikoli intradenní tickovou historii.")
    sec_coverage = sum(
        1 for ticker in sec_bundles
        if ticker in prices and safe_float(fundamentals.get(ticker, {}).get("Fundamental coverage %", 0)) >= 50
    ) / max(1, len(universe))
    default_pit = sec_coverage >= 0.70
    use_pit = st.checkbox(f"Zapnout point-in-time SEC fundamenty v backtestu (coverage nyní {sec_coverage:.0%})",
        value=default_pit, disabled=sec_coverage < 0.40,
        help="Historický signál používá jen fakta zveřejněná do dne rozhodnutí. Zůstává survivorship bias, protože univerzum je současný seznam.")
    st.caption("Signálové parametry jsou vlevo. Stop-loss 50denní low / 3×ATR je v kalkulátoru pozic; tento měsíční backtest nepředstírá intradenní exekuci stopů.")
    # Current SIC metadata is shown for all uploaded securities; it is not point-in-time sector history.
    sector_map = sector_map_all
    classified_sector_count = sum(1 for ticker in universe if sector_map.get(ticker, "Unclassified") not in {"Unclassified", "Unknown", ""})
    if max_sector_positions > 0 and classified_sector_count < len(universe):
        st.info(f"Limit SIC skupin pokrývá {classified_sector_count}/{len(universe)} titulů. Neklasifikované tituly jsou omezením vynechány; současná klasifikace nemusí odpovídat historickému SIC za celou dobu testu.")
    signature = (tuple(universe), period, price_data_signature, universe_history_signature,
        top_n, min_adx, min_score, hysteresis, max_replacements,
        max_sector_positions, commission_bps, half_spread_bps, slippage_bps, cost_bps, breakout_filter,
        weighting_method, backtest_max_weight, stop_mode, stop_atr,
        use_pit, use_estimated_value, bool(sentiment_df is not None), bool(revision_df is not None), capital)
    run_backtest = st.button("Spustit / přepočítat backtest", type="primary", key="run_backtest")
    compare_stop_button = st.button("Porovnat všechna výstupní pravidla", key="compare_stop_variants")
    train_holdout_button = st.button(
        "Vybrat parametry na tréninku a ověřit holdout",
        key="train_select_holdout_button",
        help="Porovná 9 variant min-score × hystereze pouze na prvních 60 % výnosové historie; na posledních 40 % ukáže pouze výsledky vítěze zvoleného tréninkovým Sharpe. Nejde o rolling walk-forward."
    )
    walk_forward_button = st.button(
        "Spustit expanding walk-forward",
        key="expanding_walk_forward_button",
        help="V každém foldu vybere min-score a hysterezi jen z minulých dat, potom otestuje vítěze na neprolínajícím se šestíměsíčním OOS okně. Portfolio se na začátku každého OOS okna resetuje na hotovost a na konci se modelově zlikviduje."
    )
    saved = st.session_state.get("backtest_result")
    if run_backtest:
        benchmark_series = _daily_returns(price_data_all[benchmark_ticker]) if benchmark_ticker in price_data_all else None
        fund_histories, optional_histories = {}, {}
        try:
            with st.spinner("Počítám historické signály, point-in-time fundamenty a rebalancování…"):
                if use_pit and sec_bundles:
                    fund_histories = _make_fundamental_history(prices, sec_bundles, use_estimated_value)
                all_closes = pd.DataFrame({t: f["Adj Close"] if "Adj Close" in f else f["Close"] for t,f in prices.items() if not f.empty})
                monthly_dates = list(all_closes.resample("ME").last().dropna(how="all").index)
                sentiment_history = signal_history(sentiment_df, "Sentiment score", monthly_dates) if sentiment_df is not None else {}
                revision_history = signal_history(revision_df, "Revision score", monthly_dates) if revision_df is not None else {}
                for ticker in set(sentiment_history) | set(revision_history):
                    pieces = []
                    if ticker in sentiment_history:
                        pieces.append(sentiment_history[ticker])
                    if ticker in revision_history:
                        pieces.append(revision_history[ticker])
                    optional_histories[ticker] = pd.concat(pieces, axis=1)
                curve, metrics, audit = run_monthly_backtest(
                    prices=prices, benchmark=benchmark_series, top_n=top_n, cost_bps=cost_bps,
                    initial=capital, min_adx=min_adx, min_score=min_score, hysteresis=hysteresis,
                    max_replacements=max_replacements, use_fundamentals=use_pit,
                    fundamentals_history=fund_histories, optional_history=optional_histories,
                    require_breakout=breakout_filter, weighting_method=weighting_method,
                    max_position_weight=backtest_max_weight, stop_mode=stop_mode,
                    stop_atr_multiple=stop_atr, sector_map=sector_map,
                    max_sector_positions=max_sector_positions if max_sector_positions > 0 else None,
                    commission_bps=commission_bps, half_spread_bps=half_spread_bps,
                    slippage_bps=slippage_bps, universe_history=universe_history)
                st.session_state["backtest_result"] = {
                    "curve":curve,"metrics":metrics,"audit":audit,"signature":signature,"error":None,
                    "fundamentals_history":fund_histories,"optional_history":optional_histories
                }
        except Exception as exc:
            st.session_state["backtest_result"] = {
                "curve":None,"metrics":None,"audit":None,"signature":signature,
                "error":f"{type(exc).__name__}: {exc}",
                "fundamentals_history":fund_histories,"optional_history":optional_histories
            }
        saved = st.session_state.get("backtest_result")
    if compare_stop_button:
        if saved is None or saved.get("signature") != signature:
            st.warning("Nejprve spusť backtest pro aktuální konfiguraci; srovnání musí používat stejnou konfiguraci.")
        elif saved.get("error") or saved.get("curve") is None:
            st.warning("Stop varianty nelze porovnat, dokud základní backtest nemá platný výsledek.")
        else:
            benchmark_series = _daily_returns(price_data_all[benchmark_ticker]) if benchmark_ticker in price_data_all else None
            with st.spinner("Porovnávám stop pravidla se stejnými vstupními parametry…"):
                variants = compare_stop_variants(
                    prices=prices,
                    stop_modes=(
                        "No stop / month-end rebalancing",
                        "Month-end 50-day low",
                        "ATR trailing stop",
                        "Daily 50-day low",
                        "ATR + daily 50-day low",
                    ),
                    benchmark=benchmark_series, top_n=top_n, cost_bps=cost_bps,
                    initial=capital, min_adx=min_adx, min_score=min_score, hysteresis=hysteresis,
                    max_replacements=max_replacements, use_fundamentals=use_pit,
                    fundamentals_history=saved.get("fundamentals_history", {}),
                    optional_history=saved.get("optional_history", {}),
                    require_breakout=breakout_filter, weighting_method=weighting_method,
                    max_position_weight=backtest_max_weight, stop_atr_multiple=stop_atr,
                    sector_map=sector_map,
                    max_sector_positions=max_sector_positions if max_sector_positions > 0 else None,
                    commission_bps=commission_bps, half_spread_bps=half_spread_bps,
                    slippage_bps=slippage_bps, universe_history=universe_history,
                )
                st.session_state["stop_variant_comparison"] = {"signature":signature,"table":variants}
        saved = st.session_state.get("backtest_result")
    if train_holdout_button:
        if saved is None or saved.get("signature") != signature or saved.get("error") or saved.get("curve") is None:
            st.warning("Nejprve spusť platný základní backtest pro aktuální data a parametry. Historie SEC/sentimentu se převezme z tohoto výpočtu.")
        else:
            try:
                score_candidates = tuple(sorted(set([
                    max(0.0, min_score - 10.0), float(min_score), min(80.0, min_score + 10.0)
                ])))
                hysteresis_candidates = tuple(sorted(set([
                    max(0.0, hysteresis - 4.0), float(hysteresis), min(25.0, hysteresis + 4.0)
                ])))
                benchmark_series = _daily_returns(price_data_all[benchmark_ticker]) if benchmark_ticker in price_data_all else None
                with st.spinner("Vybírám konfiguraci pouze na tréninkové části a měřím ji na odděleném holdoutu…"):
                    candidates, chosen, holdout_curve, holdout_metrics = train_select_holdout_search(
                        prices=prices,
                        min_score_candidates=score_candidates,
                        hysteresis_candidates=hysteresis_candidates,
                        train_fraction=0.60,
                        benchmark=benchmark_series, top_n=top_n, cost_bps=cost_bps,
                        initial=capital, min_adx=min_adx, min_score=min_score, hysteresis=hysteresis,
                        max_replacements=max_replacements, use_fundamentals=use_pit,
                        fundamentals_history=saved.get("fundamentals_history", {}),
                        optional_history=saved.get("optional_history", {}),
                        require_breakout=breakout_filter, weighting_method=weighting_method,
                        max_position_weight=backtest_max_weight, stop_mode=stop_mode,
                        stop_atr_multiple=stop_atr, sector_map=sector_map,
                        max_sector_positions=max_sector_positions if max_sector_positions > 0 else None,
                        commission_bps=commission_bps, half_spread_bps=half_spread_bps,
                        slippage_bps=slippage_bps, universe_history=universe_history,
                    )
                st.session_state["train_select_holdout"] = {
                    "signature": signature, "candidates": candidates, "chosen": chosen,
                    "curve": holdout_curve, "metrics": holdout_metrics, "error": None,
                }
            except Exception as exc:
                st.session_state["train_select_holdout"] = {
                    "signature": signature, "error": f"{type(exc).__name__}: {exc}",
                }

    if walk_forward_button:
        if saved is None or saved.get("signature") != signature or saved.get("error") or saved.get("curve") is None:
            st.session_state["expanding_walk_forward"] = {
                "signature": signature,
                "error": "Nejprve spusť platný základní backtest pro aktuální data a parametry.",
            }
        else:
            try:
                score_candidates = tuple(sorted(set([
                    max(0.0, min_score - 10.0), float(min_score), min(80.0, min_score + 10.0)
                ])))
                hysteresis_candidates = tuple(sorted(set([
                    max(0.0, hysteresis - 4.0), float(hysteresis), min(25.0, hysteresis + 4.0)
                ])))
                benchmark_series = _daily_returns(price_data_all[benchmark_ticker]) if benchmark_ticker in price_data_all else None
                with st.spinner("Běží expanding walk-forward: trénink pouze z minulosti, oddělené OOS foldy a likvidace na hranicích…"):
                    folds, candidate_table, oos_curve, wf_metrics = expanding_walk_forward(
                        prices=prices,
                        min_score_candidates=score_candidates,
                        hysteresis_candidates=hysteresis_candidates,
                        train_months=24, test_months=6, step_months=6, min_train_months=18,
                        benchmark=benchmark_series, top_n=top_n, cost_bps=cost_bps,
                        initial=capital, min_adx=min_adx, min_score=min_score, hysteresis=hysteresis,
                        max_replacements=max_replacements, use_fundamentals=use_pit,
                        fundamentals_history=saved.get("fundamentals_history", {}),
                        optional_history=saved.get("optional_history", {}),
                        require_breakout=breakout_filter, weighting_method=weighting_method,
                        max_position_weight=backtest_max_weight, stop_mode=stop_mode,
                        stop_atr_multiple=stop_atr, sector_map=sector_map,
                        max_sector_positions=max_sector_positions if max_sector_positions > 0 else None,
                        commission_bps=commission_bps, half_spread_bps=half_spread_bps,
                        slippage_bps=slippage_bps, universe_history=universe_history,
                    )
                st.session_state["expanding_walk_forward"] = {
                    "signature": signature, "folds": folds, "candidates": candidate_table,
                    "curve": oos_curve, "metrics": wf_metrics, "error": None,
                }
            except Exception as exc:
                st.session_state["expanding_walk_forward"] = {
                    "signature": signature, "error": f"{type(exc).__name__}: {exc}",
                }

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
        h.metric("Průměrná expozice", f"{metrics['Average gross exposure %']:.1f}%")
        i,j,k,l = st.columns(4)
        i.metric("Vstupy do pozic", str(metrics["Entry events"]))
        j.metric("Výstupy z pozic", str(metrics["Exit events"]))
        avg_days = metrics["Average holding days (closed positions)"]
        k.metric("Průměrná doba držení", "—" if pd.isna(avg_days) else f"{avg_days:.0f} dní")
        l.metric("Denní stop výstupy", str(int(curve["Stops triggered"].sum())))
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(x=curve.index,y=curve["Equity"],name="Strategie",line=dict(width=2.5,color="#72e0c1")))
        if "Benchmark equity" in curve:
            equity_fig.add_trace(go.Scatter(x=curve.index,y=curve["Benchmark equity"],name=f"Benchmark {benchmark_ticker}",line=dict(width=1.7,color="#f6bd60")))
        equity_fig.update_layout(template="plotly_dark",title="Růst modelového kapitálu",height=410,
            margin=dict(l=10,r=10,t=45,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",hovermode="x unified")
        st.plotly_chart(equity_fig, width="stretch")
        ret_fig = px.bar(curve.reset_index(), x="Date", y="Net return", title="Čistý měsíční výnos", template="plotly_dark")
        ret_fig.update_layout(height=280,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(ret_fig, width="stretch")
        st.markdown("#### Historická expozice podle SIC skupiny")
        exposure_summary = industry_exposure_summary(audit, sector_map=sector_map)
        if exposure_summary.empty:
            st.caption("Pro vybraný backtest není k dispozici audit vah.")
        else:
            st.dataframe(exposure_summary, width="stretch", hide_index=True, column_config={
                "Average weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Maximum weight %": st.column_config.NumberColumn(format="%.2f%%"),
                "Share of rebalance months %": st.column_config.NumberColumn(format="%.1f%%"),
                "Average number of positions": st.column_config.NumberColumn(format="%.2f"),
            })
            st.caption("Souhrn přiřazuje historicky držené tickery k současné SEC SIC skupině; nejde o point-in-time sektorovou historii ani o GICS/ICB klasifikaci.")
            st.download_button("Stáhnout SIC expozice CSV", exposure_summary.to_csv(index=False).encode("utf-8"), "industry_exposure_summary.csv", "text/csv")
        with st.expander("Audit každého rebalancování"):
            st.dataframe(audit, width="stretch", hide_index=True)
        st.dataframe(curve, width="stretch")
        if "SIC group weights" in curve:
            st.markdown("#### Expozice podle současné SEC SIC skupiny")
            st.caption("Skupiny jsou součtem koncových váh pozic podle aktuálního SEC SIC metadata. Nejde o historickou point-in-time klasifikaci ani o standardizovanou GICS sektorovou klasifikaci.")
            st.dataframe(curve[["SIC group weights", "Max SIC group weight %", "Distinct SIC groups"]], width="stretch")
        st.download_button("Stáhnout equity curve CSV",curve.to_csv().encode("utf-8"),"backtest_curve.csv","text/csv")
        st.download_button("Stáhnout audit obchodních rozhodnutí CSV",audit.to_csv(index=False).encode("utf-8"),"backtest_audit.csv","text/csv")
        st.markdown("#### Citlivost na transakční náklady")
        sensitivity_bps = tuple(sorted(set([0.0, float(cost_bps), 15.0, 30.0, 50.0])))
        sensitivity = cost_sensitivity(curve, initial=capital, cost_scenarios_bps=sensitivity_bps)
        st.caption("Stejná historická rozhodnutí a váhy, pouze přepočet cost bps. Nejde o nové přepočítání rozhodovací logiky; nezahrnuje změnu obchodů způsobenou vyššími náklady.")
        st.dataframe(sensitivity, width="stretch", hide_index=True, column_config={
            "Cost (bps)": st.column_config.NumberColumn(format="%.0f"),
            "CAGR": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent"),
            "Sharpe (rf=0)": st.column_config.NumberColumn(format="%.2f"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Final equity": st.column_config.NumberColumn(format="$%.2f"),
        })
        sensitivity_fig = px.line(sensitivity, x="Cost (bps)", y="CAGR", markers=True,
            title="CAGR při různých nákladech", template="plotly_dark")
        sensitivity_fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(sensitivity_fig, width="stretch")
        st.markdown("#### Chronologická kontrola období")
        period_summary = chronological_period_summary(curve, initial=capital)
        st.dataframe(period_summary, width="stretch", hide_index=True, column_config={
            "Total return": st.column_config.NumberColumn(format="percent"),
            "CAGR": st.column_config.NumberColumn(format="percent"),
            "Volatility": st.column_config.NumberColumn(format="percent"),
            "Max drawdown": st.column_config.NumberColumn(format="percent"),
            "Benchmark CAGR": st.column_config.NumberColumn(format="percent"),
        })
        st.download_button("Stáhnout rozdělení období CSV", period_summary.to_csv(index=False).encode("utf-8"), "chronological_period_summary.csv", "text/csv")
        st.caption("Rozdělení je chronologické 60/20/20 (train/validation/holdout), se stejnými parametry a bez optimalizace uvnitř jednotlivých částí. Holdout je opravdu nedotčený pouze tehdy, pokud parametry neupravuješ podle jeho výsledku.")
        st.markdown("#### Nejistota výsledků · moving-block bootstrap")
        st.caption("Resampling zachovává krátké sekvence měsíčních výnosů, aby výsledek nebyl založen jen na pořadí jednoho konkrétního průchodu. Je to diagnostický interval, nikoli záruka statistického pokrytí nebo náhrada point-in-time univerza.")
        try:
            block_length = min(6, max(1, len(curve) // 2))
            uncertainty = moving_block_bootstrap_uncertainty(
                curve, initial=capital, bootstrap_samples=500, block_length=block_length, seed=42
            )
            uncertainty_display = uncertainty.copy()
            for idx, row in uncertainty_display.iterrows():
                metric = row["Metric"]
                columns = ["Observed", "5th percentile", "Median bootstrap", "95th percentile"]
                if metric in {"CAGR", "Max drawdown"}:
                    uncertainty_display.loc[idx, columns] = [
                        f"{float(row[col]):.1%}" if np.isfinite(float(row[col])) else "—" for col in columns
                    ]
                elif metric == "Final equity":
                    uncertainty_display.loc[idx, columns] = [
                        f"${float(row[col]):,.0f}" if np.isfinite(float(row[col])) else "—" for col in columns
                    ]
                else:
                    uncertainty_display.loc[idx, columns] = [
                        f"{float(row[col]):.2f}" if np.isfinite(float(row[col])) else "—" for col in columns
                    ]
            st.dataframe(uncertainty_display, width="stretch", hide_index=True)
            st.download_button("Stáhnout bootstrap intervaly CSV", uncertainty.to_csv(index=False).encode("utf-8"), "bootstrap_uncertainty.csv", "text/csv")
        except Exception as exc:
            st.warning(f"Bootstrap nejistoty nebyl vypočten: {type(exc).__name__}: {exc}")
        trained = st.session_state.get("train_select_holdout")
        if trained and trained.get("signature") == signature:
            st.markdown("#### Parametry zvolené na tréninkové části · holdout vítězné konfigurace")
            if trained.get("error"):
                st.error("Train/holdout test nebyl dokončen: " + trained["error"])
            else:
                chosen = trained["chosen"]
                hm = trained["metrics"]
                st.caption(
                    f"Výběr pouze na train: min-score {chosen['min_score']:.0f}, "
                    f"hystereze {chosen['hysteresis']:.0f}, cíl {chosen['training_objective']} = "
                    f"{chosen['train_score']:.2f}; trénink {chosen['train_months']:.0f} měsíců. "
                    "Holdout výsledky ostatních kandidátů záměrně nejsou v tabulce."
                )
                oa, ob, oc, od = st.columns(4)
                oa.metric("Holdout CAGR", f"{hm['Holdout CAGR']:.1%}")
                ob.metric("Holdout Sharpe", "—" if not np.isfinite(hm["Holdout Sharpe (rf=0)"]) else f"{hm['Holdout Sharpe (rf=0)']:.2f}")
                oc.metric("Holdout max. drawdown", f"{hm['Holdout max drawdown']:.1%}")
                od.metric("Holdout měsíce", str(int(hm["Holdout months"])))
                oe, of, og = st.columns(3)
                oe.metric("Holdout konečná hodnota", "$" + format(hm["Holdout final equity"], ",.0f"))
                of.metric("Holdout obrat / měsíc", f"{hm['Holdout average turnover %']:.1f}%")
                if "Holdout benchmark CAGR" in hm:
                    og.metric("Holdout benchmark CAGR", f"{hm['Holdout benchmark CAGR']:.1%}")
                holdout_fig = go.Figure()
                holdout_fig.add_trace(go.Scatter(
                    x=trained["curve"].index, y=trained["curve"]["Equity"],
                    name="Vybraná strategie — holdout", line=dict(width=2.3, color="#72e0c1")
                ))
                if "Benchmark equity" in trained["curve"]:
                    holdout_fig.add_trace(go.Scatter(
                        x=trained["curve"].index, y=trained["curve"]["Benchmark equity"],
                        name="Benchmark — holdout", line=dict(width=1.7, color="#f6bd60")
                    ))
                holdout_fig.update_layout(
                    template="plotly_dark", title="Oddělený holdout · parametry vybrány bez znalosti jeho výsledku",
                    height=330, margin=dict(l=10,r=10,t=45,b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(holdout_fig, width="stretch")
                st.markdown("##### Tréninkové výsledky kandidátů")
                st.caption("V této tabulce jsou pouze metriky z prvních 60 % historie. Po prohlédnutí holdoutu už parametry neupravuj, jinak se testovací část stane součástí vývoje.")
                candidate_table = trained["candidates"]
                st.dataframe(candidate_table, width="stretch", hide_index=True, column_config={
                    "Train CAGR": st.column_config.NumberColumn(format="percent"),
                    "Train Sharpe": st.column_config.NumberColumn(format="%.2f"),
                    "Train max drawdown": st.column_config.NumberColumn(format="percent"),
                    "Train turnover %": st.column_config.NumberColumn(format="%.1f%%"),
                })
                st.download_button(
                    "Stáhnout tabulku tréninkových kandidátů CSV",
                    candidate_table.to_csv(index=False).encode("utf-8"),
                    "train_parameter_candidates.csv", "text/csv",
                )
                st.download_button(
                    "Stáhnout vítězný holdout equity CSV",
                    trained["curve"].to_csv().encode("utf-8"),
                    "train_selected_holdout_curve.csv", "text/csv",
                )
                st.warning("Jde o jednorázový výběr parametrů na train a test na pozdější části, nikoli rolling walk-forward. Opakované úpravy gridu podle holdoutu ruší jeho nezávislost; současný ticker universe navíc stále vytváří survivorship bias.")
        variant_saved = st.session_state.get("stop_variant_comparison")
        if variant_saved and variant_saved.get("signature") == signature:
            st.markdown("#### Srovnání výstupních pravidel")
            st.caption("Stejné univerzum, faktory, vstupní pravidla, váhy a náklady; liší se pouze zvolené výstupní pravidlo. Tabulku používej jako citlivostní analýzu, ne jako výběr vítěze podle holdout.")
            variant_table = variant_saved["table"]
            st.dataframe(variant_table, width="stretch", hide_index=True, column_config={
                "CAGR": st.column_config.NumberColumn(format="percent"),
                "Volatility": st.column_config.NumberColumn(format="percent"),
                "Max drawdown": st.column_config.NumberColumn(format="percent"),
                "Average monthly turnover %": st.column_config.NumberColumn(format="percent"),
                "Average max SIC group weight %": st.column_config.NumberColumn(format="percent"),
            })
            st.download_button("Stáhnout srovnání stop variant CSV", variant_table.to_csv(index=False).encode("utf-8"), "stop_variant_comparison.csv", "text/csv")
        st.warning("Test stále není survivorship-bias-free: vychází ze současného uživatelského seznamu tickerů. Denní stop je aproximace z OHLC (gap na open nebo fill na stop úrovni), nikoli intradenní exekuce. Cost model rozlišuje komisi, polovinu spreadu a skluz, ale neobsahuje fixní poplatky, market impact, FX ani daňový model.")

    wf_saved = st.session_state.get("expanding_walk_forward")
    if wf_saved and wf_saved.get("signature") == signature:
        st.divider()
        st.subheader("Expanding walk-forward · opakované out-of-sample foldy")
        st.caption("V každém foldu se min-score × hystereze vyberou pouze podle předchozí historie (Sharpe, případně CAGR). Následujících šest měsíců tvoří test. Každý fold začíná s hotovostí, platí náklady na vstup/rebalancování a na konci se zbylé pozice modelově zlikvidují. Foldy se nepřekrývají.")
        if wf_saved.get("error"):
            st.warning("Walk-forward nebyl dokončen: " + wf_saved["error"])
        else:
            wf_metrics = wf_saved["metrics"]
            wa, wb, wc, wd = st.columns(4)
            wa.metric("Agregovaný OOS CAGR", f"{wf_metrics['OOS CAGR']:.1%}")
            wb.metric("Agregovaný OOS Sharpe", "—" if not np.isfinite(wf_metrics["OOS Sharpe (rf=0)"]) else f"{wf_metrics['OOS Sharpe (rf=0)']:.2f}")
            wc.metric("OOS max. drawdown", f"{wf_metrics['OOS max drawdown']:.1%}")
            wd.metric("Nezávislé foldy / měsíce", f"{int(wf_metrics['OOS folds'])} / {int(wf_metrics['OOS months'])}")
            we, wf, wg = st.columns(3)
            we.metric("OOS konečná hodnota", "$" + format(wf_metrics["OOS final equity"], ",.0f"))
            wf.metric("OOS obrat / měsíc", f"{wf_metrics['OOS average turnover %']:.1f}%")
            if "OOS benchmark CAGR" in wf_metrics:
                wg.metric(f"OOS benchmark {benchmark_ticker} CAGR", f"{wf_metrics['OOS benchmark CAGR']:.1%}")
            wf_curve = wf_saved["curve"]
            fig_wf = go.Figure()
            fig_wf.add_trace(go.Scatter(x=wf_curve.index, y=wf_curve["Equity"], name="Expanding walk-forward OOS", line=dict(width=2.4, color="#72e0c1")))
            if "Benchmark equity" in wf_curve:
                fig_wf.add_trace(go.Scatter(x=wf_curve.index, y=wf_curve["Benchmark equity"], name=f"Benchmark {benchmark_ticker}", line=dict(width=1.7, color="#f6bd60")))
            fig_wf.update_layout(template="plotly_dark", title="Agregovaná OOS křivka po poplatcích a závěrečné likvidaci", height=360,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
            st.plotly_chart(fig_wf, width="stretch")
            st.markdown("#### Výsledky jednotlivých walk-forward foldů")
            st.dataframe(wf_saved["folds"], width="stretch", hide_index=True, column_config={
                "OOS CAGR": st.column_config.NumberColumn(format="percent"),
                "OOS max drawdown": st.column_config.NumberColumn(format="percent"),
                "OOS turnover %": st.column_config.NumberColumn(format="percent"),
                "OOS benchmark CAGR": st.column_config.NumberColumn(format="percent"),
                "Train CAGR": st.column_config.NumberColumn(format="percent"),
            })
            st.download_button("Stáhnout walk-forward foldy CSV", wf_saved["folds"].to_csv(index=False).encode("utf-8"), "expanding_walk_forward_folds.csv", "text/csv")
            st.markdown("#### Kandidáti a výběr pouze podle tréninku")
            st.caption("Tabulka nezobrazuje OOS metriky nevybraných konfigurací, aby neumožňovala jejich dodatečné ladění podle testovacího období.")
            st.dataframe(wf_saved["candidates"], width="stretch", hide_index=True, column_config={
                "Train CAGR": st.column_config.NumberColumn(format="percent"),
                "Train Sharpe": st.column_config.NumberColumn(format="%.2f"),
                "Train max drawdown": st.column_config.NumberColumn(format="percent"),
                "Train turnover %": st.column_config.NumberColumn(format="%.1f%%"),
            })
            st.download_button("Stáhnout walk-forward kandidáty CSV", wf_saved["candidates"].to_csv(index=False).encode("utf-8"), "expanding_walk_forward_candidates.csv", "text/csv")
            st.download_button("Stáhnout OOS walk-forward equity CSV", wf_curve.to_csv().encode("utf-8"), "expanding_walk_forward_oos_curve.csv", "text/csv")
            st.warning("Ani expanding walk-forward neodstraňuje survivorship bias současného tickerového seznamu, chyby poskytovatele nebo nepřesnost OHLC exekuce. Používej jej jako kontrolu stability, ne jako důkaz budoucího výnosu.")

with tab_valuation:
    st.subheader("DCF / scénářová hodnota akcie")
    if is_demo_mode:
        st.warning("Jsi v demo režimu: zobrazovaná cena je syntetická, nikoli aktuální cena vybraného reálného tickeru. DCF scénář počítej až po vložení ověřeného FCF, počtu akcií a ceny ze stejné měny/třídy.")
    st.markdown(
        "Tato valuace je samostatný FCFE-style scénář; **nemění Composite score ani výběr pozic v backtestu**. Používá roční FCF proxy = provozní cash flow − capex, konstantní růst po zadaný počet let a terminální růst. Výsledek je velmi citlivý na předpoklady."
    )
    dcf_ticker = st.selectbox("Akcie k valuaci", screener["Ticker"].tolist(), key="dcf_ticker")
    dcf_snapshot = fundamentals.get(dcf_ticker, {})
    dcf_row = screener.loc[screener["Ticker"] == dcf_ticker].iloc[0]
    sec_fcf = safe_float(dcf_snapshot.get("FCF"), 0.0)
    sec_shares = safe_float(dcf_snapshot.get("Shares outstanding (reported)"), 0.0)
    current_price = safe_float(dcf_row.get("Price"), 0.0)
    growth_raw = dcf_snapshot.get("Revenue growth %")
    has_revenue_growth = pd.notna(growth_raw) if growth_raw is not None else False
    revenue_growth = safe_float(growth_raw, 0.0) if has_revenue_growth else 0.0
    default_growth = float(np.clip(revenue_growth, -20.0, 30.0))
    st.caption(
        f"Zdroj: {dcf_snapshot.get('Source', 'nenačteno')} · období: {dcf_snapshot.get('Financial period', '—') or '—'} · "
        f"poslední podání: {dcf_snapshot.get('Latest filing date', '—') or '—'}"
    )
    if not has_revenue_growth:
        st.info("SEC růst tržeb není dostupný. Výchozích 0 % pro růst FCF je pouze explicitní výchozí scénář, nikoli odhadovaný údaj.")
    if sec_shares <= 0:
        st.warning("SEC neposkytl použitelný počet akcií v oběhu. Pole níže začíná na 1 pouze kvůli validaci vstupu; nahraď jej ověřeným počtem akcií odpovídajícím stejné obchodované třídě a měně.")
    if current_price <= 0:
        st.warning("Pro zvolený titul není dostupná kladná cena. DCF lze počítat, ale nebude možné smysluplně porovnat hodnotu s trhem.")
    if sec_fcf <= 0:
        st.warning("V SEC snapshotu není k dispozici kladné roční FCF. Níže můžeš zadat vlastní normalizovanou hodnotu; negativní/kolísavé FCF není vhodné bez dalšího vysvětlení vložit do jednoduchého terminálního DCF.")
    input_left, input_right = st.columns(2)
    with input_left:
        base_fcf = st.number_input(
            "Normalizované roční FCF (v reportovací měně)",
            min_value=0.0, value=max(0.0, sec_fcf), step=1_000_000.0, format="%.2f",
            help="Předvyplněno z posledního dostupného ročního výkazu. Provozní cash flow minus capex není univerzální FCFE definice; zkontroluj výkaz."
        )
        shares_input = st.number_input(
            "Počet akcií v oběhu (kusy)",
            min_value=1.0, value=max(1.0, sec_shares), step=1_000_000.0, format="%.0f",
            help="SEC snapshot může obsahovat zastaralý či jinak definovaný počet akcií. U ADR a více tříd je nutné ověřit poměr a odpovídající obchodovanou třídu."
        )
        quote_price = st.number_input(
            "Aktuální cena podle screeneru",
            min_value=0.0, value=max(0.0, current_price), step=1.0, format="%.4f",
            help="Pro porovnání musí být cena a FCF/share inputs ve srovnatelných měnových a jednotkových konvencích."
        )
    with input_right:
        growth_pct = st.slider("Růst FCF ročně (%)", -20.0, 30.0, default_growth, 1.0)
        cost_equity_pct = st.slider("Náklady vlastního kapitálu (%)", 5.0, 25.0, 10.0, 0.5)
        terminal_growth_pct = st.slider("Terminální růst (%)", 0.0, 4.0, 2.5, 0.25)
        forecast_years = st.slider("Délka prognózy (roky)", 3, 10, 5)
    confirmed_units = st.checkbox(
        "Ověřil/a jsem, že FCF, počet akcií a cena jsou ve srovnatelných měnových a třídních jednotkách.",
        value=False,
        help="Aplikace nezná u každého XBRL faktu spolehlivou jednotku/ADR poměr a automaticky neprovádí FX konverzi."
    )
    if not confirmed_units:
        st.info("Potvrď jednotky až po jejich kontrole. U zahraničního emitenta, ADR nebo dual-class akcie mohou FCF a cena být v různých měnách nebo odpovídat různým počtům akcií.")
    elif base_fcf <= 0:
        st.error("Zadej kladné normalizované FCF. Model s nulovým či záporným FCF a perpetuitním terminálním růstem by v této jednoduché podobě nebyl smysluplný.")
    else:
        try:
            dcf_result = calculate_fcfe_dcf(
                base_fcf=base_fcf,
                annual_growth_rate=growth_pct / 100,
                cost_of_equity=cost_equity_pct / 100,
                terminal_growth_rate=terminal_growth_pct / 100,
                years=forecast_years,
                shares_outstanding=shares_input,
            )
            a,b,c1,d = st.columns(4)
            a.metric("Modelová hodnota / akcii", f"{dcf_result['Implied value per share']:,.2f}")
            b.metric("Aktuální cena", f"{quote_price:,.2f}")
            upside = (dcf_result["Implied value per share"] / quote_price - 1) * 100 if quote_price > 0 else np.nan
            c1.metric("Odchylka vůči ceně", "—" if not np.isfinite(upside) else f"{upside:+.1f}%")
            d.metric("Terminální hodnota / PV", f"{dcf_result['Terminal value share %']:.1f}%")
            details = pd.DataFrame([
                {"Položka": "PV explicitního FCF", "Hodnota": dcf_result["PV forecast FCF"]},
                {"Položka": "PV terminální hodnoty", "Hodnota": dcf_result["PV terminal value"]},
                {"Položka": "Celková implikovaná hodnota equity", "Hodnota": dcf_result["Implied equity value"]},
                {"Položka": "Použité FCF", "Hodnota": dcf_result["Base FCF"]},
            ])
            st.dataframe(details, width="stretch", hide_index=True, column_config={"Hodnota": st.column_config.NumberColumn(format="%.2f")})
            growth_scenarios = sorted(set([
                max(-0.20, (growth_pct - 5.0) / 100),
                growth_pct / 100,
                min(0.30, (growth_pct + 5.0) / 100),
            ]))
            discount_scenarios = sorted(set([
                max(0.05, (cost_equity_pct - 2.0) / 100),
                cost_equity_pct / 100,
                min(0.25, (cost_equity_pct + 2.0) / 100),
            ]))
            sensitivity = dcf_sensitivity(
                base_fcf=base_fcf, shares_outstanding=shares_input,
                growth_rates=growth_scenarios, discount_rates=discount_scenarios,
                terminal_growth_rate=terminal_growth_pct / 100, years=forecast_years,
            )
            st.markdown("#### Citlivost hodnoty na růst FCF a diskontní sazbu")
            st.dataframe(sensitivity, width="stretch", hide_index=True, column_config={
                "FCF growth %": st.column_config.NumberColumn(format="%.1f%%"),
                **{column: st.column_config.NumberColumn(format="%.2f") for column in sensitivity.columns if column != "FCF growth %"},
            })
            matrix = sensitivity.set_index("FCF growth %")
            heat = px.imshow(
                matrix, aspect="auto", text_auto=".2f",
                labels={"x": "Náklady vlastního kapitálu", "y": "Růst FCF (%)", "color": "Hodnota / akcii"},
                title="DCF sensitivity · hodnota na akcii", template="plotly_dark",
            )
            heat.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(heat, width="stretch")
            st.download_button("Stáhnout DCF scénáře CSV", sensitivity.to_csv(index=False).encode("utf-8"), "dcf_sensitivity.csv", "text/csv")
        except Exception as exc:
            st.error(f"DCF scénář nešlo spočítat: {type(exc).__name__}: {exc}")
    st.warning(
        "DCF model je citlivý na předpoklady, nezohledňuje všechny účetní/finanční položky, dluhy v enterprise-value modelu, "
        "měnové konverze ani rozdíly v ADR. Výstup není cenový cíl ani doporučení a není součástí obchodního skóre."
    )

with tab_portfolio:
    st.subheader("ATR position sizing a evidence lotů")
    st.markdown("Kalkulátor převádí maximální riziko na počet kusů při zadaném ATR stopu a limitech expozice. Nezadává pokyny brokerovi.")
    candidates = screener.sort_values("Fundamental/composite score",ascending=False,na_position="last")
    suggested = candidates.loc[candidates["Trend OK"],"Ticker"].head(min(top_n,5)).tolist()
    picked = st.multiselect("Tituly ke kalkulaci",screener["Ticker"].tolist(),default=suggested)
    sizes = calculate_position_sizes(screener,picked,capital,risk_budget_pct=risk_budget_pct,
        stop_atr_multiple=stop_atr,max_position_pct=max_position_pct,max_total_exposure_pct=max_exposure_pct,
        max_total_risk_pct=max_total_risk_pct)
    if sizes.empty:
        st.info("Vyber akcie s dostupnou cenou a ATR(20).")
    else:
        a,b,c = st.columns(3)
        a.metric("Modelová alokace",f"${sizes['Position value'].sum():,.0f}")
        b.metric("Expozice kapitálu",f"{sizes['Weight %'].sum():.1f}%")
        risk_used = float(sizes["Initial risk budget"].sum())
        c.metric("Počáteční riziko portfolia",f"${risk_used:,.0f}",f"{risk_used / capital * 100:.2f}% z kapitálu (limit {max_total_risk_pct:.1f}%)")
        st.dataframe(sizes,width="stretch",hide_index=True,column_config={
            "Weight %":st.column_config.NumberColumn(format="%.2f%%"),"Price":st.column_config.NumberColumn(format="%.2f"),
            "ATR20":st.column_config.NumberColumn(format="%.2f"),"Provisional stop":st.column_config.NumberColumn(format="%.2f"),
            "Position value":st.column_config.NumberColumn(format="$%.2f"),"Initial risk budget":st.column_config.NumberColumn(format="$%.2f")})
        st.download_button("Stáhnout plán pozic CSV",sizes.to_csv(index=False).encode("utf-8"),"position_sizing.csv","text/csv")
    st.divider()
    st.markdown("#### FIFO tax-lot ledger")
    st.write("Nahraj historii transakcí. FIFO páruje prodeje s nejstaršími nákupními loty, započítá dodané poplatky a ukáže realizovaný zisk/ztrátu, hrubé prodeje i zbylé loty.")
    st.info("Daňové štítky jsou pouze orientační. Hodnotový test do 100 000 Kč se posuzuje za celý rok; od roku 2026 je pro cenné papíry při splnění časového testu zrušen 40milionový limit, který se vztahoval na rok 2025. Nepřenášej tento limit automaticky na roky 2024 a starší. Pro rok 2025 se do limitu může započítávat i osvobozený prodej kryptoaktiv od 14. 2. 2025 — tento modul kryptoaktiva neeviduje. Oznamovací hranice 5 mil. Kč se vztahuje k jednotlivému osvobozenému příjmu, ne k ročnímu součtu. Modul používá původní prodejní řádek jako aproximaci jednotlivého příjmu i tehdy, když FIFO rozdělí prodej mezi loty; skutečné právní posouzení může záviset na konkrétních okolnostech. U jiných měn aplikace nepočítá automatický převod do CZK. Ověř úplnost všech osvobozených příjmů i celého portfolia a příslušné výjimky; nejde o výpočet daně.")
    st.markdown("[Finanční správa ČR — daňové novinky pro rok 2026](https://financnisprava.gov.cz/cs/financni-sprava/media-a-verejnost/tiskove-zpravy-gfr/tiskove-zpravy-2026/danove-novinky-pro-rok-2026)")
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
            realized_currencies = sorted(realized["Currency"].fillna("").astype(str).str.upper().unique()) if not realized.empty else []
            if len(realized_currencies) == 1:
                realized_gain_label = f"Realizovaný zisk / ztráta ({realized_currencies[0]})"
                realized_gain_value = f"{realized['Realized gain/loss'].sum():,.2f}"
            elif len(realized_currencies) > 1:
                realized_gain_label = "Realizovaný zisk / ztráta"
                realized_gain_value = "Více měn — viz tabulka"
            else:
                realized_gain_label = "Realizovaný zisk / ztráta"
                realized_gain_value = "0.00"
            a,b,c = st.columns(3)
            a.metric(realized_gain_label, realized_gain_value)
            b.metric("Realizované páry",str(len(realized)))
            c.metric("Otevřené loty",str(len(open_lots)))
            if len(realized_currencies) > 1:
                st.caption("P/L se nesčítá napříč různými měnami bez konverze do společné měny.")
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
    data_cols = [c for c in [
        "Price source", "Source file", "Ticker", "Status", "Quality", "Rows", "First date", "Last date",
        "Calendar days stale", "Missing %", "Invalid price rows", "Max abs daily return %", "Adjustment status", "Message"
    ] if c in price_report.columns]
    st.dataframe(price_report[data_cols],width="stretch",hide_index=True,height=430, column_config={
        "Calendar days stale": st.column_config.NumberColumn(format="%.0f"),
        "Missing %": st.column_config.NumberColumn(format="%.2f%%"),
        "Invalid price rows": st.column_config.NumberColumn(format="%.0f"),
        "Max abs daily return %": st.column_config.NumberColumn(format="%.2f%%"),
    })
    st.download_button("Stáhnout quality report CSV",price_report.to_csv(index=False).encode("utf-8"),"data_quality_report.csv","text/csv")
    combined = _download_frame(price_data_all)
    if not combined.empty:
        st.download_button("Stáhnout adjustované cenové řady CSV (wide)",combined.to_csv(index_label="Date").encode("utf-8"),"adjusted_price_history_wide.csv","text/csv")
    long_ohlcv = _download_ohlcv_long(price_data_all)
    if not long_ohlcv.empty:
        st.download_button(
            "Stáhnout OHLCV CSV pro opětovný import (long format)",
            long_ohlcv.to_csv(index=False).encode("utf-8"),
            "ohlcv_history_long.csv", "text/csv",
            help="Formát kompatibilní s CSV importem této aplikace. V demo režimu obsahuje výhradně syntetické ceny.",
        )
    if universe_history is not None:
        membership_asof = max(pd.Timestamp(frame.index.max()) for frame in prices.values() if not frame.empty)
        if membership_asof.tzinfo is not None:
            membership_asof = membership_asof.tz_localize(None)
        active_members = universe_members_asof(universe_history, membership_asof)
        membership_report = universe_history.copy()
        membership_report["Included on last price date"] = membership_report["Ticker"].isin(active_members or set())
        st.markdown("#### Point-in-time složení investičního univerza")
        st.caption(
            f"Použité intervaly jsou včetně Start/End date; poslední cenové datum je {membership_asof.date()}. "
            "Backtest nejprve omezí univerzum podle tohoto historického členství a teprve potom spočítá relativní faktorové ranky."
        )
        st.dataframe(membership_report, width="stretch", hide_index=True)
        st.download_button(
            "Stáhnout použitou historii členství CSV",
            membership_report.to_csv(index=False).encode("utf-8"),
            "point_in_time_universe_membership.csv", "text/csv",
        )
        st.warning(
            "Tato funkce snižuje survivorship bias pouze tehdy, když členství pokrývá celý testovaný interval "
            "a OHLCV soubor obsahuje všechny tehdejší tituly včetně delistovaných. Neověřuje úplnost zdroje třetí stranou."
        )

    if valid_email:
        sec_report = []
        for ticker in universe:
            record = fundamentals.get(ticker,{})
            sec_report.append({"Ticker":ticker,"Source":record.get("Source","No SEC data"),
                "SEC SIC":record.get("SEC SIC", ""),
                "SEC SIC Description":record.get("SEC SIC Description", ""),
                "SIC group":record.get("SIC group", "Unclassified"),
                "Financial period":record.get("Financial period",""),"Latest filing date":record.get("Latest filing date",""),
                "Fundamental status":record.get("Fundamental status","Not loaded"),"Data note":record.get("Data note","")})
        st.markdown("#### Fundamentální zdroj a stáří výkazů")
        sec_df = pd.DataFrame(sec_report)
        st.dataframe(sec_df,width="stretch",hide_index=True)
        st.download_button("Stáhnout SEC provenance CSV",sec_df.to_csv(index=False).encode("utf-8"),"sec_provenance.csv","text/csv")
    if price_report["Status"].eq("MISSING").any():
        st.error("Některé cenové řady chybí. Backtest se nesmí používat bez vyřešení dat.")
    elif "Quality" in price_report and price_report["Quality"].eq("REVIEW").any():
        flagged = price_report.loc[price_report["Quality"].eq("REVIEW"), "Ticker"].tolist()
        st.warning("Některé řady vyžadují kontrolu (stáří dat, podezřelý pohyb nebo neplatné hodnoty): " + ", ".join(flagged) + ". Neplatné cenové řádky jsou z výpočtů vyřazeny; velké pohyby mohou být reálné a nejsou automaticky odstraněny.")
    else:
        st.success("V základní automatické kontrole nejsou zjištěné chybějící ani podezřelé cenové řady.")
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
        Fundamentální údaje z SEC se filtrují podle filed date a historické rozhodnutí nepoužije výkaz zveřejněný později. Volitelný intervalový CSV soubor (Ticker, Start date, End date) omezuje cross-section pouze na tituly, které podle dodaných dat patřily do univerza k rozhodovacímu datu. Členství se filtruje **před** výpočtem relativních percentilových skóre.

        ### Omezení
        - Veřejná aplikace nepoužívá tržní API automaticky. Výchozí data jsou syntetická; pro reálné ceny nahraj pouze CSV, jehož licence dovoluje veřejné zobrazení.
        - SEC Company Facts pokrývá reportující emitenty a standardizované XBRL tagy; metriky se mohou lišit účetními standardy a dostupností tagu.
        - P/E a FCF yield používají hrubý odhad tržní kapitalizace z ceny a posledního nahlášeného počtu akcií. U ADR, duálních tříd a změn kapitálu mohou být chybné, proto jsou výchozím stavem mimo skóre.
        - Bez CSV historického členství backtest používá pevný seznam tickerů a není survivorship-bias-free. S intervalovým CSV a úplnými cenovými řadami delistovaných titulů lze toto zkreslení omezit, nikoli automaticky vyloučit. Měsíční a denní OHLC simulace nejsou přesný broker fill model.
        - Sentiment/revize jsou volitelné časované CSV; historický model vybírá jen poslední známý záznam k rozhodovacímu dni.
        - FIFO eviduje pořizovací náklady a poplatky; neurčuje českou daňovou povinnost a nepřepočítává měny.
        - Expanding walk-forward volí parametry z minulých dat a testuje je v neprolínajících se OOS foldech; každý fold začíná na hotovosti a na konci modelově likviduje pozice. Nejde o kontinuitní broker účet a neřeší survivorship bias.
        - DCF je samostatný FCFE-style scénář, nikoli součást Composite score. Před použitím je třeba ověřit měnu, jednotky FCF, počet akcií, ADR poměr a příslušnou obchodovanou třídu.
        - HMM je záměrně odložen, dokud základní strategie neprojde robustní out-of-sample testy. Jinak by zvyšoval riziko přeoptimalizace.

        ### Cesta k důvěryhodným výsledkům
        Historické univerzum včetně delistovaných titulů, jasná práva k datům, point-in-time fundamenty, dividendy/FX/corporate actions, spread a skluz, walk-forward validace a předem odložené testovací období.
        """
    )
    st.markdown("**Roadmap:** [ROADMAP.md](https://github.com/Matthew-pros/quality-momentum-lab/blob/main/ROADMAP.md) · [Automatické testy](https://github.com/Matthew-pros/quality-momentum-lab/actions)")
    st.caption("Výzkumný nástroj; nejde o investiční, daňové ani právní doporučení.")
