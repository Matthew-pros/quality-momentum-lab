"""Offline Streamlit smoke test: runs the UI against deterministic synthetic OHLCV data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
from streamlit.testing.v1 import AppTest

import qlab.data


def _fake_market_data(tickers, period="10y"):
    index = pd.bdate_range(end=pd.Timestamp("2026-10-12"), periods=700)
    t = np.arange(len(index), dtype=float)
    result = {}
    report = []
    for position, ticker in enumerate(tickers):
        close = (35.0 + 4.0 * position) * np.exp((0.0002 + position * 0.00001) * t)
        close = close * (1 + 0.025 * np.sin(t / (35 + position)))
        frame = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.012,
                "Low": close * 0.988,
                "Close": close,
                "Adj Close": close * (1 + t * 0.000005),
                "Volume": 500_000 + (position * 10_000),
            },
            index=index,
        )
        result[ticker] = frame
        report.append({
            "Ticker": ticker,
            "Status": "OK",
            "Rows": len(frame),
            "First date": index.min().date().isoformat(),
            "Last date": index.max().date().isoformat(),
            "Missing %": 0.0,
            "Message": "",
        })
    return result, pd.DataFrame(report)


def test_streamlit_home_and_tabs_render_without_live_network_calls(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(qlab.data, "generate_demo_market_data", _fake_market_data)
    st.cache_data.clear()

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=40)
    app.run()

    assert not app.exception, "\n".join(str(e.value) for e in app.exception)
    rendered_markdown = "\\n".join(str(element.value) for element in app.markdown)
    assert "QUALITY MOMENTUM LAB" in rendered_markdown
    rendered_subheaders = {str(element.value) for element in app.subheader}
    assert "Screener a podkladové metriky" in rendered_subheaders
    assert "Kontrola kvality dat" in rendered_subheaders


def test_streamlit_backtest_and_train_holdout_buttons_run_offline(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(qlab.data, "generate_demo_market_data", _fake_market_data)
    st.cache_data.clear()

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception, "\\n".join(str(e.value) for e in app.exception)

    run_button = next(button for button in app.button if button.label == "Spustit / přepočítat backtest")
    run_button.click().run()
    assert not app.exception, "\\n".join(str(e.value) for e in app.exception)
    assert "backtest_result" in app.session_state
    assert app.session_state["backtest_result"]["error"] is None

    holdout_button = next(button for button in app.button if button.label == "Vybrat parametry na tréninku a ověřit holdout")
    holdout_button.click().run()
    assert not app.exception, "\\n".join(str(e.value) for e in app.exception)
    assert "train_select_holdout" in app.session_state
    result = app.session_state["train_select_holdout"]
    assert result.get("error") is None
    assert result["metrics"]["Holdout months"] >= 6
    assert result["candidates"]["Selected by train"].sum() == 1


def test_streamlit_dcf_form_calculates_and_renders_sensitivity_offline(monkeypatch):
    monkeypatch.delenv("SEC_CONTACT_EMAIL", raising=False)
    monkeypatch.setattr(qlab.data, "generate_demo_market_data", _fake_market_data)
    st.cache_data.clear()

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
    app.run()
    assert not app.exception, "\\n".join(str(e.value) for e in app.exception)

    fcf = next(item for item in app.number_input if item.label == "Normalizované roční FCF (v reportovací měně)")
    shares = next(item for item in app.number_input if item.label == "Počet akcií v oběhu (kusy)")
    confirmation = next(item for item in app.checkbox if item.label.startswith("Ověřil/a jsem"))
    fcf.set_value(3_000_000_000)
    shares.set_value(1_000_000_000)
    confirmation.set_value(True)
    app.run()

    assert not app.exception, "\\n".join(str(e.value) for e in app.exception)
    rendered_labels = {str(item.label) for item in app.metric}
    assert "Modelová hodnota / akcii" in rendered_labels
    assert "Terminální hodnota / PV" in rendered_labels
    assert any("Citlivost hodnoty na růst FCF a diskontní sazbu" in str(item.value) for item in app.markdown)
