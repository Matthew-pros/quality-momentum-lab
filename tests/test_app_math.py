import io
import numpy as np
import pandas as pd
import pytest

from qlab.optional_signals import parse_signal_csv, latest_signals, signal_history
from qlab.strategy import (
    calculate_indicators, percentile_score, weighted_average,
    score_cross_section, select_with_hysteresis, run_monthly_backtest,
)
from qlab.tax_lots import fifo_ledger


def synthetic_prices(n=1600):
    index = pd.bdate_range("2018-01-01", periods=n)
    t = np.arange(n, dtype=float)
    values = {}
    params = [
        (40, 0.00075, 0.025, 42.0),
        (80, 0.00055, 0.040, 53.0),
        (120, 0.00035, 0.030, 61.0),
    ]
    for ticker, (base, drift, amplitude, freq) in zip(["AAA", "BBB", "CCC"], params):
        close = base * np.exp(drift * t) * (1 + amplitude * np.sin(t / freq + base / 17))
        # Small positive dividend adjustment over time for total-return tests.
        adj = close * (1 + 0.000015 * t)
        frame = pd.DataFrame({
            "Open": close * 0.998,
            "High": close * 1.012,
            "Low": close * 0.988,
            "Close": close,
            "Adj Close": adj,
            "Volume": 1_000_000,
        }, index=index)
        values[ticker] = frame
    return values


def test_percentile_score_respects_direction_and_keeps_missing_values():
    values = pd.Series([20.0, 10.0, np.nan], index=["high", "low", "missing"])
    higher = percentile_score(values, higher_better=True)
    lower = percentile_score(values, higher_better=False)
    assert higher["high"] > higher["low"]
    assert lower["low"] > lower["high"]
    assert np.isnan(higher["missing"])
    assert np.isnan(lower["missing"])


def test_weighted_average_reweights_available_inputs_and_does_not_fabricate_neutral_score():
    frame = pd.DataFrame({"quality": [80.0, np.nan, np.nan], "momentum": [40.0, 70.0, np.nan]},
                         index=["both", "momentum_only", "neither"])
    result = weighted_average(frame, {"quality": 0.6, "momentum": 0.4})
    assert np.isclose(result["both"], 64.0)
    assert np.isclose(result["momentum_only"], 70.0)
    assert np.isnan(result["neither"])


def test_technical_indicators_have_expected_units_and_exclude_current_bar_from_breakout():
    prices = synthetic_prices()
    frame = prices["AAA"].copy()
    indicators = calculate_indicators(frame)
    last = indicators.index[-1]
    factor = frame["Adj Close"] / frame["Close"]
    adjusted_high = frame["High"] * factor
    previous_55_high = adjusted_high.iloc[-56:-1].max()
    assert indicators["ATR20"].dropna().gt(0).all()
    assert indicators["ATR20%"].dropna().gt(0).all()
    assert indicators["ADX14"].dropna().between(0, 100).all()
    assert np.isclose(indicators.loc[last, "High55 prior"], previous_55_high)
    assert indicators.loc[last, "Close"] == frame["Adj Close"].iloc[-1]
    assert indicators.loc[last, "Close raw"] == frame["Close"].iloc[-1]
    assert indicators.loc[last, "Momentum 12-1M"] == pytest.approx(
        frame["Adj Close"].iloc[-22] / frame["Adj Close"].iloc[-253] - 1
    )


def test_hysteresis_does_not_replace_a_holding_for_a_small_score_change():
    scores = pd.Series({"HELD": 70.0, "NEW": 74.0, "BEST": 90.0})
    eligible = pd.Series({"HELD": True, "NEW": True, "BEST": True})
    selected = select_with_hysteresis(scores, eligible, {"HELD"}, top_n=1,
                                      min_improvement=7, exit_score_floor=30,
                                      max_replacements=1)
    # Initial refill is only for vacancies; the incumbent is replaced only by a challenger
    # with a score at least 7 points higher.
    assert selected == {"BEST"}


def test_backtest_signal_precedes_return_period_and_audit_is_exportable():
    prices = synthetic_prices()
    curve, metrics, audit = run_monthly_backtest(
        prices, top_n=2, cost_bps=10, initial=10_000, min_adx=0, min_score=0,
        hysteresis=7, max_replacements=1, use_fundamentals=False,
    )
    assert curve is not None and metrics is not None and audit is not None
    assert len(curve) > 12
    assert (pd.to_datetime(audit["Signal date"]) < pd.to_datetime(audit["Return period"])).all()
    assert curve["Transaction cost"].ge(0).all()
    assert (curve["Number of holdings"] <= 2).all()
    assert metrics["Months"] == len(curve)
    assert np.isfinite(metrics["CAGR"])
    assert metrics["Max drawdown"] <= 0


def test_backtest_first_selection_does_not_use_prices_after_signal_date():
    prices = synthetic_prices()
    original, _, audit_original = run_monthly_backtest(
        prices, top_n=2, cost_bps=0, initial=10_000, min_adx=0, min_score=0,
        use_fundamentals=False,
    )
    assert original is not None and audit_original is not None
    decision = pd.Timestamp(audit_original.iloc[0]["Signal date"])
    future_prices = {ticker: frame.copy() for ticker, frame in prices.items()}
    for frame in future_prices.values():
        future = frame.index > decision
        frame.loc[future, ["Open", "High", "Low", "Close", "Adj Close"]] *= 1.3
    changed, _, audit_changed = run_monthly_backtest(
        future_prices, top_n=2, cost_bps=0, initial=10_000, min_adx=0, min_score=0,
        use_fundamentals=False,
    )
    assert changed is not None and audit_changed is not None
    assert audit_changed.iloc[0]["Selected holdings"] == audit_original.iloc[0]["Selected holdings"]
    assert changed.iloc[0]["Net return"] != pytest.approx(original.iloc[0]["Net return"])


def test_signal_csv_is_asof_and_does_not_use_future_rows():
    csv = b"Date,Ticker,Sentiment score,Source\n2024-01-01,AAA,55,feed1\n2024-03-01,AAA,80,feed1\n"
    signals = parse_signal_csv(io.BytesIO(csv), "Sentiment score")
    now = latest_signals(signals, "Sentiment score", as_of="2024-02-01")
    assert now["AAA"]["Sentiment score"] == 55.0
    history = signal_history(signals, "Sentiment score", pd.to_datetime(["2024-02-01", "2024-04-01"]))
    assert history["AAA"].iloc[0, 0] == 55.0
    assert history["AAA"].iloc[1, 0] == 80.0


def test_fifo_ledger_matches_lots_and_allocates_fees():
    tx = pd.DataFrame([
        {"Date":"2024-01-01","Ticker":"AAA","Type":"BUY","Quantity":10,"Price":100,"Fees":10,"Currency":"USD"},
        {"Date":"2024-02-01","Ticker":"AAA","Type":"BUY","Quantity":5,"Price":120,"Fees":5,"Currency":"USD"},
        {"Date":"2024-03-01","Ticker":"AAA","Type":"SELL","Quantity":12,"Price":130,"Fees":12,"Currency":"USD"},
    ])
    realized, open_lots, annual = fifo_ledger(tx)
    assert np.isclose(realized["Quantity matched"].sum(), 12)
    assert len(realized) == 2
    assert open_lots["Remaining quantity"].sum() == pytest.approx(3)
    # Buy fee/unit 1, sell fee/unit 1: first lot P/L = 10*(129-101).
    assert realized.iloc[0]["Realized gain/loss"] == pytest.approx(280)
    assert annual["Realized gain/loss"].sum() == pytest.approx(realized["Realized gain/loss"].sum())


def test_fifo_rejects_short_sales():
    tx = pd.DataFrame([{"Date":"2024-01-01","Ticker":"AAA","Type":"SELL","Quantity":1,"Price":10}])
    with pytest.raises(ValueError, match="exceeds FIFO inventory"):
        fifo_ledger(tx)


def test_yfinance_multiindex_columns_split_into_per_ticker_frames():
    from qlab.data import _extract_ticker_frame, normalize_tickers

    index = pd.bdate_range("2025-01-01", periods=3)
    columns = pd.MultiIndex.from_product(
        [["AAA", "BBB"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    )
    values = np.arange(36, dtype=float).reshape(3, 12) + 10
    raw = pd.DataFrame(values, index=index, columns=columns)
    aa = _extract_ticker_frame(raw, "AAA", ["AAA", "BBB"])
    bb = _extract_ticker_frame(raw, "BBB", ["AAA", "BBB"])
    assert list(aa.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert list(bb.columns) == list(aa.columns)
    assert aa.shape == (3, 6) and bb.shape == (3, 6)
    assert normalize_tickers("aaa, BBB,aaa,,") == ["AAA", "BBB"]


def test_sec_fundamentals_respect_filing_date_as_of_cutoff():
    from qlab.data import sec_fundamental_snapshot

    def annual_rows(values, filed_dates):
        rows = []
        for year, value, filed in zip([2021, 2022, 2023], values, filed_dates):
            rows.append({
                "start": f"{year}-01-01", "end": f"{year}-12-31",
                "val": value, "filed": filed, "form": "10-K", "fy": year, "fp": "FY",
            })
        return {"USD": rows}

    facts = {
        "us-gaap": {
            "Revenues": {"units": annual_rows([100, 150, 180], ["2022-02-15", "2023-02-15", "2024-02-15"])},
            "NetIncomeLoss": {"units": annual_rows([10, 15, 20], ["2022-02-15", "2023-02-15", "2024-02-15"])},
            "NetCashProvidedByUsedInOperatingActivities": {"units": annual_rows([18, 25, 30], ["2022-02-15", "2023-02-15", "2024-02-15"])},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": annual_rows([4, 5, 8], ["2022-02-15", "2023-02-15", "2024-02-15"])},
            "StockholdersEquity": {"units": {"USD": [
                {"end": "2022-12-31", "val": 40, "filed": "2023-02-15", "form": "10-K"},
                {"end": "2023-12-31", "val": 60, "filed": "2024-02-15", "form": "10-K"},
            ]}},
            "LongTermDebtNoncurrent": {"units": {"USD": [
                {"end": "2022-12-31", "val": 12, "filed": "2023-02-15", "form": "10-K"},
            ]}},
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                {"end": "2022-12-31", "val": 100, "filed": "2023-02-15", "form": "10-K"},
            ]}},
        },
    }
    snapshot = sec_fundamental_snapshot(
        {"ticker": "AAA", "title": "AAA Corporation", "payload": {"facts": facts}},
        as_of="2023-06-30", latest_price=2.0,
    )
    assert snapshot["Revenue"] == 150
    assert snapshot["Revenue growth %"] == pytest.approx(50)
    assert snapshot["Net income"] == 15
    assert snapshot["Earnings growth %"] == pytest.approx(50)
    assert snapshot["FCF"] == pytest.approx(20)
    assert snapshot["Profit margin %"] == pytest.approx(10)
    assert snapshot["ROE %"] == pytest.approx(37.5)
    assert snapshot["Financial period"] == "2022-12-31"
    assert snapshot["Latest filing date"] <= "2023-06-30"


def test_empty_or_invalid_price_frame_is_not_accepted_as_data():
    from qlab.data import _extract_ticker_frame

    empty = _extract_ticker_frame(pd.DataFrame(), "AAA", ["AAA"])
    assert empty.empty
    raw = pd.DataFrame({"Volume": [100, 200]}, index=pd.date_range("2025-01-01", periods=2))
    assert _extract_ticker_frame(raw, "AAA", ["AAA"]).empty


def test_sec_fact_filed_on_decision_date_is_deferred_to_avoid_close_time_leakage():
    from qlab.data import _annual_series

    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [
                    {"start":"2021-01-01","end":"2021-12-31","val":100,"filed":"2022-02-15","form":"10-K"},
                    {"start":"2022-01-01","end":"2022-12-31","val":150,"filed":"2023-02-15","form":"10-K"},
                ]}}
            }
        }
    }
    result = _annual_series(facts, ["Revenues"], "2023-02-15")
    assert len(result) == 1
    assert result.iloc[0] == 100


def test_inverse_atr_weights_reduce_high_volatility_position_and_respect_cap():
    from qlab.strategy import _target_weights

    cross = pd.DataFrame({
        "Ticker": ["LOWVOL", "HIGHVOL", "MIDVOL"],
        "ATR20%": [1.0, 5.0, 2.0],
    })
    weights = _target_weights(["LOWVOL", "HIGHVOL", "MIDVOL"], cross, "Inverse ATR", 0.40)
    assert weights["LOWVOL"] > weights["HIGHVOL"]
    assert all(weight <= 0.40 + 1e-12 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_strict_position_cap_can_leave_residual_cash():
    from qlab.strategy import _target_weights

    cross = pd.DataFrame({"Ticker": ["A", "B", "C"], "ATR20%": [1.0, 1.0, 1.0]})
    weights = _target_weights(["A", "B", "C"], cross, "Inverse ATR", 0.20)
    assert all(weight <= 0.20 + 1e-12 for weight in weights.values())
    assert sum(weights.values()) == pytest.approx(0.60)


def test_czech_tax_review_flags_value_test_and_three_year_test_without_claiming_tax_due():
    from qlab.tax_lots import fifo_ledger

    short_hold_czk = pd.DataFrame([
        {"Date":"2024-01-10","Ticker":"AAA","Type":"BUY","Quantity":10,"Price":50,"Fees":0,"Currency":"CZK"},
        {"Date":"2025-11-10","Ticker":"AAA","Type":"SELL","Quantity":10,"Price":60,"Fees":0,"Currency":"CZK"},
    ])
    realized_small, _, _ = fifo_ledger(short_hold_czk)
    assert not realized_small.iloc[0]["Holding period > 3 years"]
    assert "value-test exemption" in realized_small.iloc[0]["Tax review status"]

    long_hold_czk = pd.DataFrame([
        {"Date":"2020-01-10","Ticker":"BBB","Type":"BUY","Quantity":1000,"Price":100,"Fees":0,"Currency":"CZK"},
        {"Date":"2024-02-10","Ticker":"BBB","Type":"SELL","Quantity":1000,"Price":500,"Fees":0,"Currency":"CZK"},
    ])
    realized_large, _, annual_large = fifo_ledger(long_hold_czk)
    assert realized_large.iloc[0]["Holding period > 3 years"]
    assert "3-year time-test exemption" in realized_large.iloc[0]["Tax review status"]
    assert annual_large.iloc[0]["Gross sale proceeds"] == pytest.approx(500_000)


def test_monthly_replacement_cap_limits_refills_after_exit():
    from qlab.strategy import select_with_hysteresis

    scores = pd.Series({"A": 80.0, "B": 78.0, "C": 75.0, "D": 72.0, "E": 70.0})
    eligible = pd.Series({"A": True, "B": False, "C": True, "D": True, "E": True})
    selected = select_with_hysteresis(
        scores, eligible, previous={"A", "B", "C"}, top_n=3,
        min_improvement=7.0, exit_score_floor=30.0, max_replacements=1,
    )
    assert selected == {"A", "C", "D"}


def test_initial_portfolio_can_fill_target_count_before_monthly_replacement_limit():
    from qlab.strategy import select_with_hysteresis

    scores = pd.Series({"A": 90.0, "B": 80.0, "C": 70.0, "D": 60.0})
    eligible = pd.Series(True, index=scores.index)
    selected = select_with_hysteresis(
        scores, eligible, previous=set(), top_n=3,
        min_improvement=7.0, exit_score_floor=30.0, max_replacements=1,
    )
    assert selected == {"A", "B", "C"}


def test_csv_price_adapter_loads_long_format_and_reports_universe_gaps():
    import io
    from qlab.data import load_ohlcv_csv

    rows = []
    dates = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=4)
    for ticker, base in [("AAA", 20.0), ("BBB", 50.0)]:
        for i, date in enumerate(dates):
            close = base + i * 0.1
            rows.append({
                "Date": date.date().isoformat(), "Ticker": ticker,
                "Open": close - 0.05, "High": close + 0.20, "Low": close - 0.20,
                "Close": close, "Adj Close": close, "Volume": 1000 + i,
            })
    payload = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    prices, report = load_ohlcv_csv(io.BytesIO(payload), expected_tickers=["AAA", "BBB", "SPY"])
    assert set(prices) == {"AAA", "BBB"}
    assert len(prices["AAA"]) == 4
    assert report.loc[report["Ticker"] == "SPY", "Status"].iloc[0] == "MISSING"
    assert report.loc[report["Ticker"] == "AAA", "Quality"].iloc[0] == "OK"
    assert "Adj Close" in prices["AAA"]


def test_csv_price_adapter_rejects_duplicate_ticker_dates_and_missing_columns():
    import io
    from qlab.data import load_ohlcv_csv

    duplicate_csv = pd.DataFrame([
        {"Date":"2026-01-02","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10},
        {"Date":"2026-01-02","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10},
    ]).to_csv(index=False).encode("utf-8")
    with pytest.raises(ValueError, match="Duplicate ticker/date"):
        load_ohlcv_csv(io.BytesIO(duplicate_csv), expected_tickers=["AAA"])

    incomplete = pd.DataFrame([{"Date":"2026-01-02","Ticker":"AAA","Close":10}]).to_csv(index=False).encode("utf-8")
    with pytest.raises(ValueError, match="requires columns"):
        load_ohlcv_csv(io.BytesIO(incomplete), expected_tickers=["AAA"])


def test_market_quality_report_flags_stale_invalid_and_extreme_prices():
    from qlab.data import assess_price_quality

    index = pd.to_datetime(["2026-09-20", "2026-09-21", "2026-09-22"])
    frame = pd.DataFrame({
        "Open": [10.0, 10.0, 0.0], "High": [11.0, 20.0, 0.0],
        "Low": [9.0, 9.0, 0.0], "Close": [10.0, 19.0, 0.0],
        "Adj Close": [10.0, 19.0, 0.0],
    }, index=index)
    cleaned, report = assess_price_quality({"AAA": frame}, now="2026-10-13")
    row = report.iloc[0]
    assert row["Quality"] == "REVIEW"
    assert row["Calendar days stale"] > 7
    assert row["Invalid price rows"] == 1
    assert row["Max abs daily return %"] >= 30
    assert np.isnan(cleaned["AAA"].loc[index[-1], "Close"])
    assert row["Rows"] == 2


def test_sec_history_reprices_between_filings_without_changing_reported_fundamentals():
    from qlab.data import sec_snapshot_to_history

    def annual(values, filed):
        return {"USD": [
            {"start":"2021-01-01","end":"2021-12-31","val":values[0],"filed":filed[0],"form":"10-K"},
            {"start":"2022-01-01","end":"2022-12-31","val":values[1],"filed":filed[1],"form":"10-K"},
        ]}
    facts = {"facts":{"us-gaap":{
        "Revenues":{"units":annual([100,150],["2022-02-15","2023-02-15"])},
        "NetIncomeLoss":{"units":annual([10,15],["2022-02-15","2023-02-15"])},
        "NetCashProvidedByUsedInOperatingActivities":{"units":annual([18,25],["2022-02-15","2023-02-15"])},
        "PaymentsToAcquirePropertyPlantAndEquipment":{"units":annual([4,5],["2022-02-15","2023-02-15"])},
        "StockholdersEquity":{"units":{"USD":[
            {"end":"2022-12-31","val":60,"filed":"2023-02-15","form":"10-K"},
        ]}},
        "LongTermDebtNoncurrent":{"units":{"USD":[
            {"end":"2022-12-31","val":12,"filed":"2023-02-15","form":"10-K"},
        ]}},
        "EntityCommonStockSharesOutstanding":{"units":{"shares":[
            {"end":"2022-12-31","val":100,"filed":"2023-02-15","form":"10-K"},
        ]}},
    }}}
    bundle = {"ticker":"AAA","title":"AAA Inc.","payload":facts}
    dates = pd.to_datetime(["2023-03-31","2023-04-30"])
    prices = pd.Series([10.0,20.0], index=pd.to_datetime(["2023-03-31","2023-04-28"]))
    history = sec_snapshot_to_history(bundle, list(dates), prices, price_currency="USD")
    assert history.iloc[0]["Revenue"] == history.iloc[1]["Revenue"] == 150
    assert history.iloc[0]["As of"] == "2023-03-31"
    assert history.iloc[1]["As of"] == "2023-04-30"
    assert history.iloc[1]["Estimated market cap"] == pytest.approx(2 * history.iloc[0]["Estimated market cap"])
    assert history.iloc[1]["P/E (annual EPS proxy)"] == pytest.approx(2 * history.iloc[0]["P/E (annual EPS proxy)"])


def test_cost_sensitivity_keeps_signals_fixed_and_higher_cost_lowers_terminal_equity():
    from qlab.strategy import cost_sensitivity

    curve = pd.DataFrame({
        "Gross return": [0.03, -0.02, 0.04, 0.01],
        "Gross turnover": [1.0, 0.2, 0.5, 0.4],
    }, index=pd.date_range("2024-01-31", periods=4, freq="ME"))
    table = cost_sensitivity(curve, initial=10_000, cost_scenarios_bps=(0, 15, 50))
    assert table.loc[table["Cost (bps)"] == 0, "Final equity"].iloc[0] > table.loc[table["Cost (bps)"] == 15, "Final equity"].iloc[0]
    assert table.loc[table["Cost (bps)"] == 15, "Final equity"].iloc[0] > table.loc[table["Cost (bps)"] == 50, "Final equity"].iloc[0]
    assert table["Average monthly turnover %"].nunique() == 1


def test_daily_atr_trailing_stop_uses_gap_open_and_returns_exit_reason():
    from qlab.strategy import calculate_indicators, _simulate_position_month

    index = pd.bdate_range("2024-01-01", periods=360)
    close = pd.Series(50.0 + np.arange(len(index)) * 0.20, index=index)
    frame = pd.DataFrame({
        "Open": close - 0.10,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Adj Close": close,
        "Volume": 500_000,
    }, index=index)
    signal_date = index[320]
    stop_date = index[321]
    # Simulate an overnight gap below the trailing stop on the first session after signal.
    frame.loc[stop_date, ["Open", "High", "Low", "Close", "Adj Close"]] = [90.0, 92.0, 88.0, 91.0, 91.0]
    indicators = calculate_indicators(frame)

    ret, reason, exit_date, high_water = _simulate_position_month(
        frame, indicators, signal_date, index[340],
        stop_mode="ATR trailing stop", stop_atr_multiple=3.0,
    )
    assert ret < 0
    assert reason == "ATR trailing stop"
    assert exit_date == stop_date.date().isoformat()
    # Gap-aware implementation must use the open (90), not a more optimistic stop level.
    entry_adjusted = frame.loc[signal_date, "Adj Close"]
    assert ret == pytest.approx(90.0 / entry_adjusted - 1)


def test_monthly_backtest_supports_daily_stop_mode_and_audit_columns():
    from qlab.strategy import run_monthly_backtest

    curve, metrics, audit = run_monthly_backtest(
        synthetic_prices(1600), top_n=2, cost_bps=10, initial=10_000,
        min_adx=0, min_score=0, hysteresis=7, max_replacements=1,
        use_fundamentals=False, stop_mode="ATR trailing stop", stop_atr_multiple=3.0,
    )
    assert curve is not None and metrics is not None and audit is not None
    assert "Stops triggered" in curve.columns
    assert "Stop mode" in audit.columns
    assert "Ticker returns" in audit.columns
    assert "Stop turnover" in curve.columns
    assert "Entry events" in curve.columns
    assert "Exit events" in curve.columns
    assert "Average holding days (closed positions)" in metrics
    assert (curve["Gross turnover"] >= curve["Rebalance turnover"] - 1e-12).all()
    assert np.allclose(curve["Transaction cost"], curve["Gross turnover"] * 10 / 10_000)


def test_atr_trailing_stop_carries_high_water_from_prior_month_not_all_time_peak():
    from qlab.strategy import calculate_indicators, _simulate_position_month

    index = pd.bdate_range("2024-01-01", periods=500)
    close = pd.Series(100.0, index=index)
    frame = pd.DataFrame({
        "Open": close.copy(), "High": close + 1.0, "Low": close - 1.0,
        "Close": close.copy(), "Adj Close": close.copy(), "Volume": 100_000,
    }, index=index)
    first_signal = index[300]
    spike_date = index[301]
    frame.loc[spike_date, ["Open","High","Low","Close","Adj Close"]] = [119.0,121.0,118.0,120.0,120.0]
    frame.loc[index[302:321], ["Open","High","Low","Close","Adj Close"]] = [120.0,121.0,119.0,120.0,120.0]
    indicators = calculate_indicators(frame)
    _, reason1, _, high_water = _simulate_position_month(
        frame, indicators, first_signal, index[320], stop_mode="ATR trailing stop", stop_atr_multiple=3.0
    )
    assert reason1 == "HOLD"
    assert high_water == pytest.approx(121.0)

    # The incumbent's stored high-water makes a stop active even though the
    # current entry close alone would produce a lower stop level.
    second_signal = index[320]
    stop_date = index[321]
    frame.loc[stop_date, ["Open","High","Low","Close","Adj Close"]] = [114.0,116.0,113.0,115.0,115.0]
    indicators = calculate_indicators(frame)
    ret, reason2, exit_date, _ = _simulate_position_month(
        frame, indicators, second_signal, index[340],
        stop_mode="ATR trailing stop", stop_atr_multiple=3.0, prior_high_water=high_water
    )
    assert reason2 == "ATR trailing stop"
    assert exit_date == stop_date.date().isoformat()
    assert ret < 0


def test_chronological_period_summary_splits_in_time_order_and_reports_holdout():
    from qlab.strategy import chronological_period_summary

    index = pd.date_range("2020-01-31", periods=20, freq="ME")
    returns = np.array([0.01] * 12 + [-0.03] * 4 + [0.02] * 4)
    curve = pd.DataFrame({
        "Net return": returns,
        "Gross turnover": np.full(20, 0.2),
        "Gross exposure %": np.full(20, 90.0),
        "Stops triggered": np.zeros(20, dtype=int),
        "Benchmark return": np.full(20, 0.005),
    }, index=index)
    summary = chronological_period_summary(curve, initial=10_000)
    assert summary["Period"].tolist() == ["Train", "Validation", "Holdout"]
    assert summary["Months"].tolist() == [12, 4, 4]
    assert summary.iloc[0]["End date"] < summary.iloc[1]["Start date"]
    assert summary.iloc[1]["End date"] < summary.iloc[2]["Start date"]
    assert summary.iloc[2]["CAGR"] > summary.iloc[1]["CAGR"]
    assert "Benchmark CAGR" in summary.columns


def test_hysteresis_tie_breaking_is_deterministic_by_ticker():
    from qlab.strategy import select_with_hysteresis

    scores = pd.Series({"B": 75.0, "A": 75.0, "C": 80.0, "D": 70.0})
    eligible = pd.Series(True, index=scores.index)
    selected = select_with_hysteresis(
        scores, eligible, previous={"B", "A"}, top_n=2,
        min_improvement=0.0, exit_score_floor=0.0, max_replacements=1,
    )
    assert selected == {"B", "C"}


def test_fcfe_dcf_reconciles_terminal_value_and_per_share_equity_value():
    from qlab.valuation import calculate_fcfe_dcf

    result = calculate_fcfe_dcf(
        base_fcf=100.0, annual_growth_rate=0.0, cost_of_equity=0.10,
        terminal_growth_rate=0.02, years=5, shares_outstanding=10.0,
    )
    assert result["PV forecast FCF"] > 0
    assert result["PV terminal value"] > 0
    assert result["Implied equity value"] == pytest.approx(
        result["PV forecast FCF"] + result["PV terminal value"]
    )
    assert result["Implied value per share"] == pytest.approx(result["Implied equity value"] / 10.0)
    assert result["Terminal value share %"] > 50


def test_dcf_rejects_nonpositive_fcf_and_invalid_terminal_discount_relationship():
    from qlab.valuation import calculate_fcfe_dcf

    with pytest.raises(ValueError, match="Base free cash flow must be positive"):
        calculate_fcfe_dcf(0.0, 0.05, 0.10, 0.02)
    with pytest.raises(ValueError, match="must be greater than terminal growth"):
        calculate_fcfe_dcf(100.0, 0.05, 0.02, 0.02)


def test_dcf_sensitivity_value_falls_when_discount_rate_rises():
    from qlab.valuation import dcf_sensitivity

    table = dcf_sensitivity(
        base_fcf=100.0, shares_outstanding=10.0,
        growth_rates=[0.05], discount_rates=[0.08, 0.10, 0.12],
        terminal_growth_rate=0.02, years=5,
    )
    values = table.iloc[0, 1:].to_numpy(dtype=float)
    assert values[0] > values[1] > values[2]


def test_czech_notification_threshold_is_per_sale_not_annual_total_and_respects_fifo_lot_splits():
    from qlab.tax_lots import fifo_ledger

    split_sale = pd.DataFrame([
        {"Date":"2020-01-01","Ticker":"AAA","Type":"BUY","Quantity":20000,"Price":50,"Fees":0,"Currency":"CZK"},
        {"Date":"2020-06-01","Ticker":"AAA","Type":"BUY","Quantity":30000,"Price":70,"Fees":0,"Currency":"CZK"},
        {"Date":"2024-07-01","Ticker":"AAA","Type":"SELL","Quantity":50000,"Price":200,"Fees":0,"Currency":"CZK"},
    ])
    realized, _, _ = fifo_ledger(split_sale)
    assert len(realized) == 2  # the sale is split across two FIFO acquisitions
    assert realized["Sale transaction ID"].nunique() == 1
    assert (realized["Gross sale proceeds"].sum()) == pytest.approx(10_000_000)
    assert realized["Tax review status"].str.contains("§38v notice").all()

    multiple_sales = pd.DataFrame([
        {"Date":"2020-01-01","Ticker":"BBB","Type":"BUY","Quantity":40000,"Price":100,"Fees":0,"Currency":"CZK"},
        {"Date":"2024-07-01","Ticker":"BBB","Type":"SELL","Quantity":20000,"Price":200,"Fees":0,"Currency":"CZK"},
        {"Date":"2024-08-01","Ticker":"BBB","Type":"SELL","Quantity":20000,"Price":200,"Fees":0,"Currency":"CZK"},
    ])
    realized_multi, _, _ = fifo_ledger(multiple_sales)
    assert realized_multi["Gross sale proceeds"].sum() == pytest.approx(8_000_000)
    assert realized_multi["Sale transaction ID"].nunique() == 2
    assert not realized_multi["Tax review status"].str.contains("§38v notice").any()


def test_sic_groups_are_explicit_coarse_groups_not_claimed_gics_sectors():
    from qlab.data import sic_group

    assert sic_group(3571) == "Manufacturing"
    assert sic_group(6021) == "Finance / Insurance / Real Estate"
    assert sic_group(7372) == "Services"
    assert sic_group(None) == "Unclassified"
    assert sic_group("not-a-code") == "Unclassified"


def test_sector_cap_limits_initial_portfolio_and_leaves_unknown_sector_unconstrained():
    from qlab.strategy import select_with_hysteresis

    scores = pd.Series({"A": 95.0, "B": 90.0, "C": 85.0, "D": 80.0, "E": 75.0})
    eligible = pd.Series(True, index=scores.index)
    groups = {"A":"Manufacturing", "B":"Manufacturing", "C":"Services", "D":"Finance", "E":"Unclassified"}
    selected = select_with_hysteresis(
        scores, eligible, previous=set(), top_n=4,
        sector_map=groups, max_sector_positions=1,
    )
    assert len(selected) == 4
    assert sum(groups[ticker] == "Manufacturing" for ticker in selected) == 1
    assert "E" in selected  # unclassified names are not falsely all treated as one sector


def test_sector_cap_reduces_existing_concentration_deterministically_and_respects_replacement_limit():
    from qlab.strategy import select_with_hysteresis

    scores = pd.Series({"A": 88.0, "B": 82.0, "C": 80.0, "D": 79.0, "E": 65.0})
    eligible = pd.Series(True, index=scores.index)
    groups = {"A":"Manufacturing", "B":"Manufacturing", "C":"Services", "D":"Finance", "E":"Utilities"}
    selected = select_with_hysteresis(
        scores, eligible, previous={"A", "B", "C"}, top_n=3,
        min_improvement=7, exit_score_floor=30, max_replacements=1,
        sector_map=groups, max_sector_positions=1,
    )
    assert "A" in selected and "B" not in selected
    assert len(selected) == 3
    assert len({groups[t] for t in selected}) == 3


def test_component_costs_sum_on_rebalance_and_stop_turnover():
    curve, _, audit = run_monthly_backtest(
        synthetic_prices(1600), top_n=2, cost_bps=0, initial=10_000,
        min_adx=0, min_score=0, hysteresis=7, max_replacements=1,
        use_fundamentals=False, commission_bps=1.0, half_spread_bps=4.0, slippage_bps=6.0,
        stop_mode="ATR trailing stop",
    )
    assert curve is not None and audit is not None
    assert np.allclose(curve["Transaction cost"], curve["Gross turnover"] * 11.0 / 10_000)
    assert set(audit["Commission bps"]) == {1.0}
    assert set(audit["Half-spread bps"]) == {4.0}
    assert set(audit["Slippage bps"]) == {6.0}
    assert set(audit["Cost bps"]) == {11.0}


def test_adjusted_ohlc_prevents_false_split_jump_in_trend_and_stop_indicators():
    from qlab.strategy import calculate_indicators

    index = pd.bdate_range("2020-01-01", periods=320)
    adjusted = pd.Series(100.0 + np.arange(len(index)) * 0.03, index=index)
    raw = adjusted.copy()
    # A 2-for-1 split cuts the raw quoted close in half; adjusted close remains continuous.
    split_index = index[250]
    raw.loc[index >= split_index] = raw.loc[index >= split_index] / 2.0
    frame = pd.DataFrame({
        "Open": raw * 0.999,
        "High": raw * 1.01,
        "Low": raw * 0.99,
        "Close": raw,
        "Adj Close": adjusted,
        "Volume": 1_000_000,
    }, index=index)
    indicators = calculate_indicators(frame)
    assert indicators.loc[split_index, "Close"] == pytest.approx(adjusted.loc[split_index])
    assert indicators.loc[split_index, "Close raw"] == pytest.approx(raw.loc[split_index])
    assert not bool(indicators.loc[split_index, "Below 50d low"])
    assert 1.5 < indicators.loc[split_index, "ATR20%"] < 2.5


def test_atr_position_sizing_enforces_aggregate_portfolio_risk_cap_after_integer_rounding():
    from qlab.strategy import calculate_position_sizes

    screener = pd.DataFrame([
        {"Ticker":"AAA","Price":100.0,"ATR20":10.0},
        {"Ticker":"BBB","Price":100.0,"ATR20":10.0},
        {"Ticker":"CCC","Price":100.0,"ATR20":10.0},
        {"Ticker":"DDD","Price":100.0,"ATR20":10.0},
        {"Ticker":"EEE","Price":100.0,"ATR20":10.0},
    ])
    sized = calculate_position_sizes(
        screener, ["AAA","BBB","CCC","DDD","EEE"], capital=10_000,
        risk_budget_pct=1.0, stop_atr_multiple=3.0,
        max_position_pct=100, max_total_exposure_pct=100,
        max_total_risk_pct=2.0,
    )
    assert not sized.empty
    assert sized["Initial risk budget"].sum() <= 200.0 + 1e-9
    assert sized["Total portfolio risk %"].iloc[0] <= 2.0 + 1e-9
    assert sized["Weight %"].sum() <= 100.0 + 1e-9


def test_atr_position_sizing_rejects_invalid_risk_inputs():
    from qlab.strategy import calculate_position_sizes

    screener = pd.DataFrame([{"Ticker":"AAA","Price":100.0,"ATR20":5.0}])
    with pytest.raises(ValueError, match="max_total_risk_pct"):
        calculate_position_sizes(screener, ["AAA"], 10_000, max_total_risk_pct=-1)


def test_czech_40m_securities_cap_is_only_flagged_for_tax_year_2025():
    from qlab.tax_lots import fifo_ledger

    rows = []
    for year, ticker in [(2024, "Y24"), (2025, "Y25"), (2026, "Y26")]:
        rows.append({"Date":"2020-01-01","Ticker":ticker,"Type":"BUY","Quantity":100_000,"Price":100,"Fees":0,"Currency":"CZK"})
        rows.append({"Date":f"{year}-09-01","Ticker":ticker,"Type":"SELL","Quantity":100_000,"Price":450,"Fees":0,"Currency":"CZK"})
    realized, _, _ = fifo_ledger(pd.DataFrame(rows))
    status_by_year = realized.groupby(realized["Sale date"].dt.year)["Tax review status"].agg(lambda x: " | ".join(x.astype(str)))
    assert "40m" not in status_by_year.loc[2024].lower()
    assert "2025 czk 40m time-test cap" in status_by_year.loc[2025].lower()
    assert "40m" not in status_by_year.loc[2026].lower()
    assert "§38v notice" in status_by_year.loc[2024]
    assert "§38v notice" in status_by_year.loc[2025]
    assert "§38v notice" in status_by_year.loc[2026]


def test_stop_variant_comparison_uses_same_common_settings_and_reports_failures_explicitly():
    from qlab.strategy import compare_stop_variants

    table = compare_stop_variants(
        prices=synthetic_prices(1600),
        stop_modes=("No stop / month-end rebalancing", "ATR trailing stop"),
        top_n=2, cost_bps=0, initial=10_000, min_adx=0, min_score=0,
        hysteresis=7, max_replacements=1, use_fundamentals=False,
        weighting_method="Equal weight", stop_atr_multiple=3.0,
    )
    assert table["Stop mode"].tolist() == ["No stop / month-end rebalancing", "ATR trailing stop"]
    assert table["Status"].eq("OK").all()
    assert table["CAGR"].notna().all()
    assert "Average holding days (closed positions)" in table.columns


def test_fifo_rejects_currency_mismatch_instead_of_subtracting_incomparable_amounts():
    from qlab.tax_lots import fifo_ledger

    transactions = pd.DataFrame([
        {"Date":"2024-01-01","Ticker":"AAA","Type":"BUY","Quantity":10,"Price":100,"Fees":0,"Currency":"USD"},
        {"Date":"2024-04-01","Ticker":"AAA","Type":"SELL","Quantity":10,"Price":3000,"Fees":0,"Currency":"CZK"},
    ])
    with pytest.raises(ValueError, match="Currency mismatch"):
        fifo_ledger(transactions)


def test_synthetic_demo_prices_are_deterministic_and_explicitly_non_real():
    from qlab.data import generate_demo_market_data

    first, first_report = generate_demo_market_data(
        ["SIM01", "SIM_BENCHMARK"], period="5y", now="2026-10-13"
    )
    second, second_report = generate_demo_market_data(
        ["SIM01", "SIM_BENCHMARK"], period="5y", now="2026-10-13"
    )
    pd.testing.assert_frame_equal(first["SIM01"], second["SIM01"])
    assert len(first["SIM01"]) >= 1200
    assert first["SIM01"]["Close"].gt(0).all()
    assert (first["SIM01"]["High"] >= first["SIM01"][["Open","Close"]].max(axis=1)).all()
    assert (first["SIM01"]["Low"] <= first["SIM01"][["Open","Close"]].min(axis=1)).all()
    assert first_report["Source"].eq("Synthetic demo").all()
    assert first_report["Message"].str.contains("SYNTHETIC DEMO ONLY").all()
    assert first_report["Quality"].eq("OK").all()


def test_market_quality_rejects_inconsistent_ohlc_ranges_and_negative_volume():
    from qlab.data import assess_price_quality

    index = pd.to_datetime(["2026-10-01", "2026-10-02", "2026-10-05"])
    frame = pd.DataFrame({
        "Open": [10.0, 12.0, 10.0],
        "High": [11.0, 11.0, 11.0],
        "Low": [9.0, 9.0, 9.0],
        "Close": [10.5, 10.0, 10.2],
        "Adj Close": [10.5, 10.0, 10.2],
        "Volume": [100.0, 100.0, -1.0],
    }, index=index)
    prices, report = assess_price_quality({"AAA": frame}, now="2026-10-13")
    row = report.iloc[0]
    assert row["Quality"] == "REVIEW"
    assert row["Invalid price rows"] == 2
    assert pd.isna(prices["AAA"].loc[index[1], "Close"])
    assert pd.isna(prices["AAA"].loc[index[2], "Close"])
    assert prices["AAA"].loc[index[0], "Close"] == pytest.approx(10.5)


def test_industry_exposure_summary_reports_weights_not_just_position_counts():
    from qlab.strategy import industry_exposure_summary

    audit = pd.DataFrame({
        "Signal date": pd.to_datetime(["2024-01-31","2024-02-29","2024-03-31"]),
        "Target weights": ["AAA:0.600, BBB:0.400", "BBB:0.500, CCC:0.500", "AAA:1.000"],
    })
    summary = industry_exposure_summary(
        audit,
        sector_map={"AAA":"Technology", "BBB":"Finance / Insurance / Real Estate"},
    ).set_index("SIC group")
    assert summary.loc["Technology", "Average weight %"] == pytest.approx((60 + 0 + 100) / 3)
    assert summary.loc["Finance / Insurance / Real Estate", "Average weight %"] == pytest.approx((40 + 50 + 0) / 3)
    assert summary.loc["Unclassified", "Average weight %"] == pytest.approx((0 + 50 + 0) / 3)
    assert summary.loc["Technology", "Rebalance months invested"] == 2
    assert summary.loc["Finance / Insurance / Real Estate", "Maximum weight %"] == pytest.approx(50.0)
    assert summary["Average weight %"].sum() == pytest.approx(100.0)


def test_moving_block_bootstrap_is_reproducible_and_reports_observed_return_metrics():
    from qlab.strategy import moving_block_bootstrap_uncertainty

    curve = pd.DataFrame({
        "Net return": [0.02, 0.01, -0.04, 0.03, 0.01, -0.01, 0.025, -0.005,
                       0.015, 0.01, -0.02, 0.04, 0.02, -0.01, 0.015, 0.005,
                       -0.03, 0.025, 0.01, 0.02, -0.015, 0.03, 0.012, -0.01],
    }, index=pd.date_range("2024-01-31", periods=24, freq="ME"))
    a = moving_block_bootstrap_uncertainty(curve, initial=10_000, bootstrap_samples=100, block_length=3, seed=7)
    b = moving_block_bootstrap_uncertainty(curve, initial=10_000, bootstrap_samples=100, block_length=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert a["Metric"].tolist() == ["CAGR", "Sharpe (rf=0)", "Max drawdown", "Final equity"]
    observed_final = 10_000 * np.prod(1 + curve["Net return"].to_numpy())
    assert a.loc[a["Metric"] == "Final equity", "Observed"].iloc[0] == pytest.approx(observed_final)
    assert (a["5th percentile"] <= a["Median bootstrap"]).all()
    assert (a["Median bootstrap"] <= a["95th percentile"]).all()


def test_moving_block_bootstrap_rejects_invalid_returns_and_too_few_samples():
    from qlab.strategy import moving_block_bootstrap_uncertainty

    curve = pd.DataFrame({"Net return":[0.01, -1.0, 0.02]})
    with pytest.raises(ValueError, match="greater than -100%"):
        moving_block_bootstrap_uncertainty(curve, bootstrap_samples=100, block_length=1)
    good = pd.DataFrame({"Net return":[0.01] * 12})
    with pytest.raises(ValueError, match="between 100 and 10,000"):
        moving_block_bootstrap_uncertainty(good, bootstrap_samples=20, block_length=2)


def test_dcf_rejects_growth_below_minus_100_percent_and_noninteger_horizon():
    from qlab.valuation import calculate_fcfe_dcf

    with pytest.raises(ValueError, match="greater than -100%"):
        calculate_fcfe_dcf(100, -1.1, 0.10, 0.02, years=5)
    with pytest.raises(ValueError, match="integer from 1 to 20"):
        calculate_fcfe_dcf(100, 0.03, 0.10, 0.02, years=2.5)


def test_train_only_parameter_search_reports_holdout_for_only_the_training_winner():
    from qlab.strategy import train_select_holdout_search

    candidates, chosen, holdout, metrics = train_select_holdout_search(
        prices=synthetic_prices(1600),
        min_score_candidates=(0.0, 20.0),
        hysteresis_candidates=(0.0, 7.0),
        train_fraction=0.60,
        min_train_months=12,
        min_holdout_months=6,
        top_n=2,
        cost_bps=10,
        initial=10_000,
        min_adx=0,
        min_score=0,
        hysteresis=7,
        max_replacements=1,
        use_fundamentals=False,
        stop_mode="No stop / month-end rebalancing",
    )
    assert len(candidates) == 4
    assert candidates["Selected by train"].sum() == 1
    assert "Holdout CAGR" not in candidates.columns
    assert chosen["min_score"] in {0.0, 20.0}
    assert chosen["hysteresis"] in {0.0, 7.0}
    assert len(holdout) >= 6
    assert metrics["Holdout months"] == len(holdout)
    assert np.isfinite(metrics["Holdout CAGR"])
    assert holdout.index.is_monotonic_increasing


def test_train_only_parameter_search_rejects_insufficient_history():
    from qlab.strategy import train_select_holdout_search

    tiny = synthetic_prices(400)
    with pytest.raises(ValueError, match="No candidate had sufficient"):
        train_select_holdout_search(
            prices=tiny,
            min_score_candidates=(20.0,),
            hysteresis_candidates=(7.0,),
            train_fraction=0.60,
            min_train_months=30,
            min_holdout_months=24,
            top_n=2,
            cost_bps=0,
            initial=10_000,
            min_adx=0,
            min_score=20,
            use_fundamentals=False,
        )


def test_dcf_matches_closed_form_and_value_per_share_units():
    from qlab.valuation import calculate_fcfe_dcf

    fcf, growth, rate, terminal, years, shares = 100.0, 0.0, 0.10, 0.02, 5, 10.0
    result = calculate_fcfe_dcf(fcf, growth, rate, terminal, years=years, shares_outstanding=shares)
    pv_explicit = sum(fcf / ((1 + rate) ** year) for year in range(1, years + 1))
    terminal_value = fcf * (1 + terminal) / (rate - terminal)
    pv_terminal = terminal_value / ((1 + rate) ** years)
    expected_equity = pv_explicit + pv_terminal

    assert result["PV forecast FCF"] == pytest.approx(pv_explicit)
    assert result["Terminal value at forecast horizon"] == pytest.approx(terminal_value)
    assert result["PV terminal value"] == pytest.approx(pv_terminal)
    assert result["Implied equity value"] == pytest.approx(expected_equity)
    assert result["Implied value per share"] == pytest.approx(expected_equity / shares)
    assert result["Terminal value share %"] == pytest.approx(pv_terminal / expected_equity * 100)


def test_dcf_sensitivity_is_monotonic_for_positive_fcf_and_valid_rates():
    from qlab.valuation import dcf_sensitivity

    matrix = dcf_sensitivity(
        base_fcf=100.0,
        shares_outstanding=10.0,
        growth_rates=[0.00, 0.05, 0.10],
        discount_rates=[0.08, 0.10, 0.12],
        terminal_growth_rate=0.02,
        years=5,
    ).set_index("FCF growth %")
    discount_columns = [f"Cost of equity {rate:.1f}%" for rate in (8.0, 10.0, 12.0)]
    # A faster FCF growth assumption raises value; a higher discount rate lowers value.
    assert matrix.loc[5.0, discount_columns[0]] > matrix.loc[0.0, discount_columns[0]]
    assert matrix.loc[10.0, discount_columns[1]] > matrix.loc[5.0, discount_columns[1]]
    assert matrix.loc[5.0, discount_columns[0]] > matrix.loc[5.0, discount_columns[1]]
    assert matrix.loc[5.0, discount_columns[1]] > matrix.loc[5.0, discount_columns[2]]


def test_dcf_sensitivity_returns_nan_when_discount_rate_does_not_exceed_terminal_growth():
    from qlab.valuation import dcf_sensitivity

    matrix = dcf_sensitivity(
        base_fcf=100.0,
        shares_outstanding=10.0,
        growth_rates=[0.03],
        discount_rates=[0.02, 0.05],
        terminal_growth_rate=0.02,
        years=5,
    )
    assert pd.isna(matrix.loc[0, "Cost of equity 2.0%"])
    assert np.isfinite(matrix.loc[0, "Cost of equity 5.0%"])


def test_dcf_rejects_nonpositive_shares_invalid_rates_and_nonpositive_fcf():
    from qlab.valuation import calculate_fcfe_dcf, dcf_sensitivity

    with pytest.raises(ValueError, match="positive for this simplified DCF"):
        calculate_fcfe_dcf(0, 0.03, 0.10, 0.02)
    with pytest.raises(ValueError, match="Shares outstanding must be positive"):
        calculate_fcfe_dcf(100, 0.03, 0.10, 0.02, shares_outstanding=0)
    with pytest.raises(ValueError, match="Shares outstanding must be positive"):
        dcf_sensitivity(100, -5, [0.03], [0.10], 0.02)


def test_backtest_evaluation_window_resets_portfolio_and_uses_only_requested_decisions():
    from qlab.strategy import run_monthly_backtest

    prices = synthetic_prices(1300)
    month_ends = prices["AAA"]["Adj Close"].resample("ME").last().dropna().index
    start, end = month_ends[25], month_ends[30]
    curve, _, audit = run_monthly_backtest(
        prices=prices, top_n=2, cost_bps=10, initial=10_000,
        min_adx=0, min_score=0, use_fundamentals=False,
        stop_mode="No stop / month-end rebalancing",
        evaluation_start_date=start, evaluation_end_date=end,
    )
    assert curve is not None and audit is not None
    assert len(curve) == 6
    assert pd.Timestamp(audit.iloc[0]["Signal date"]) == start
    assert pd.Timestamp(audit.iloc[-1]["Signal date"]) == end
    assert curve.index[0] == month_ends[26]
    assert curve.index[-1] == month_ends[31]
    assert audit.iloc[0]["Previous holdings"] == ""


def test_expanding_walk_forward_selects_on_past_only_and_returns_nonoverlapping_oos_folds():
    from qlab.strategy import expanding_walk_forward

    prices = synthetic_prices(1300)
    folds, candidates, oos, metrics = expanding_walk_forward(
        prices=prices,
        min_score_candidates=(0.0, 20.0),
        hysteresis_candidates=(0.0, 7.0),
        train_months=12,
        test_months=3,
        step_months=6,
        min_train_months=12,
        top_n=2,
        cost_bps=10,
        initial=10_000,
        min_adx=0,
        min_score=0,
        hysteresis=7,
        max_replacements=1,
        use_fundamentals=False,
        stop_mode="No stop / month-end rebalancing",
    )
    assert len(folds) >= 4
    assert len(candidates) == len(folds) * 4
    assert candidates.groupby("Fold")["Selected by train"].sum().eq(1).all()
    assert not any("OOS" in col for col in candidates.columns)
    assert len(oos) == int(folds["OOS months"].sum())
    assert oos.index.is_monotonic_increasing and not oos.index.duplicated().any()
    assert (folds["Training end"] if "Training end" in folds else folds["Training end decision"] < folds["OOS return start"]).all()
    assert "Terminal liquidation turnover" in oos.columns
    assert np.allclose(oos["Equity"], 10_000 * (1 + oos["Net return"]).cumprod())
    assert metrics["OOS months"] == len(oos)
    assert np.isfinite(metrics["OOS CAGR"])


def test_expanding_walk_forward_rejects_overlapping_folds_and_insufficient_history():
    from qlab.strategy import expanding_walk_forward

    prices = synthetic_prices(1300)
    with pytest.raises(ValueError, match="step_months must be >= test_months"):
        expanding_walk_forward(
            prices=prices, min_score_candidates=(20.0,), hysteresis_candidates=(5.0,),
            train_months=12, test_months=6, step_months=3, min_train_months=12,
            top_n=2, use_fundamentals=False, stop_mode="No stop / month-end rebalancing",
        )
    with pytest.raises(ValueError, match="Insufficient monthly history"):
        expanding_walk_forward(
            prices=synthetic_prices(400), min_score_candidates=(20.0,), hysteresis_candidates=(5.0,),
            train_months=36, test_months=12, step_months=12, min_train_months=24,
            top_n=2, use_fundamentals=False, stop_mode="No stop / month-end rebalancing",
        )


def test_terminal_liquidation_cost_uses_end_nav_and_trades_remaining_exposure():
    from qlab.strategy import _apply_terminal_liquidation_cost

    curve = pd.DataFrame({
        "Equity": [1_100.0],
        "Net return": [0.10],
        "Transaction cost": [0.0],
        "Gross turnover": [0.20],
        "Gross exposure %": [80.0],
        "Estimated cost (bps of portfolio)": [0.0],
    }, index=pd.to_datetime(["2025-01-31"]))
    out = _apply_terminal_liquidation_cost(curve, initial=1_000.0, cost_bps=10.0)
    expected_cash_cost = 1_100.0 * 0.80 * 10 / 10_000
    assert out.iloc[0]["Equity"] == pytest.approx(1_100.0 - expected_cash_cost)
    assert out.iloc[0]["Net return"] == pytest.approx((1_100.0 - expected_cash_cost) / 1_000.0 - 1)
    assert out.iloc[0]["Terminal liquidation cost"] == pytest.approx(expected_cash_cost / 1_000.0)
    assert out.iloc[0]["Terminal liquidation turnover"] == pytest.approx(0.80 * 1_100.0 / 1_000.0)
    assert out.iloc[0]["Gross turnover"] == pytest.approx(0.20 + 0.80 * 1_100.0 / 1_000.0)


def test_precomputed_monthly_cross_sections_match_direct_backtest_results():
    from qlab.strategy import (
        _precompute_indicator_cache, _precompute_monthly_cross_sections, run_monthly_backtest,
    )

    prices = synthetic_prices(900)
    indicator_cache = _precompute_indicator_cache(prices)
    monthly_cross = _precompute_monthly_cross_sections(prices, indicator_cache=indicator_cache)
    assert monthly_cross
    direct_curve, direct_metrics, direct_audit = run_monthly_backtest(
        prices=prices, top_n=2, cost_bps=10, initial=10_000,
        min_adx=0, min_score=0, hysteresis=7, max_replacements=1,
        use_fundamentals=False, stop_mode="No stop / month-end rebalancing",
    )
    cached_curve, cached_metrics, cached_audit = run_monthly_backtest(
        prices=prices, top_n=2, cost_bps=10, initial=10_000,
        min_adx=0, min_score=0, hysteresis=7, max_replacements=1,
        use_fundamentals=False, stop_mode="No stop / month-end rebalancing",
        precomputed_indicators=indicator_cache,
        precomputed_cross_sections=monthly_cross,
    )
    pd.testing.assert_frame_equal(direct_curve, cached_curve)
    pd.testing.assert_frame_equal(direct_audit, cached_audit)
    assert direct_metrics.keys() == cached_metrics.keys()
    for key in direct_metrics:
        if pd.isna(direct_metrics[key]):
            assert pd.isna(cached_metrics[key])
        else:
            assert cached_metrics[key] == pytest.approx(direct_metrics[key])


def test_daily_atr_stop_does_not_trigger_on_raw_price_split_when_adjusted_series_is_flat():
    from qlab.strategy import _simulate_position_month

    index = pd.bdate_range("2024-01-01", periods=360)
    split_start = 321
    raw_close = np.full(len(index), 100.0)
    raw_open = np.full(len(index), 100.0)
    raw_high = np.full(len(index), 101.0)
    raw_low = np.full(len(index), 99.0)
    adjustment = np.ones(len(index))
    # A two-for-one split halves all raw OHLC prices; Adj Close stays on a
    # continuous total-return basis. No adjusted OHLC columns are supplied,
    # forcing the stop simulator to derive adjustment factors itself.
    raw_close[split_start:] = 50.0
    raw_open[split_start:] = 50.0
    raw_high[split_start:] = 50.5
    raw_low[split_start:] = 49.5
    adjustment[split_start:] = 2.0
    adjusted_close = raw_close * adjustment
    frame = pd.DataFrame({
        "Open": raw_open, "High": raw_high, "Low": raw_low,
        "Close": raw_close, "Adj Close": adjusted_close,
        "Volume": 100_000,
    }, index=index)
    # Deterministic adjusted ATR/indicator path. The stop at 97 is far below the
    # correctly-adjusted post-split low (99); comparing raw low would falsely exit.
    indicators = pd.DataFrame({
        "ATR20": np.full(len(index), 1.0),
        "Low50 prior": np.full(len(index), 95.0),
    }, index=index)
    signal_date = index[320]
    ret, reason, exit_date, _ = _simulate_position_month(
        frame, indicators, signal_date, index[340],
        stop_mode="ATR trailing stop", stop_atr_multiple=3.0,
    )
    assert reason == "HOLD"
    assert exit_date == ""
    assert ret == pytest.approx(0.0)


def test_point_in_time_universe_parser_and_membership_boundaries():
    import io
    from qlab.universe import parse_universe_membership, universe_members_asof, tickers_in_membership_file

    csv = b"Ticker,Start date,End date\nAAA,2010-01-01,2020-12-31\nAAA,2021-01-01,\nBBB,2015-06-01,2024-12-31\nCCC,2022-01-01,\n"
    history = parse_universe_membership(io.BytesIO(csv))
    assert tickers_in_membership_file(history) == ["AAA", "BBB", "CCC"]
    assert universe_members_asof(history, "2018-06-30") == {"AAA", "BBB"}
    assert universe_members_asof(history, "2021-01-01") == {"AAA", "BBB"}
    assert universe_members_asof(history, "2023-01-31") == {"AAA", "BBB", "CCC"}
    assert universe_members_asof(history, "2025-06-30") == {"AAA", "CCC"}
    assert universe_members_asof(None, "2025-06-30") is None


def test_point_in_time_universe_parser_rejects_overlapping_and_open_ended_intervals():
    import io
    from qlab.universe import parse_universe_membership

    overlap = b"Ticker,Start date,End date\nAAA,2020-01-01,2020-12-31\nAAA,2020-12-31,\n"
    with pytest.raises(ValueError, match="intervals overlap"):
        parse_universe_membership(io.BytesIO(overlap))

    after_open_end = b"Ticker,Start date,End date\nAAA,2020-01-01,\nAAA,2022-01-01,2022-12-31\n"
    with pytest.raises(ValueError, match="open-ended membership"):
        parse_universe_membership(io.BytesIO(after_open_end))

    bad_start = b"Ticker,Start date,End date\nAAA,not-a-date,\n"
    with pytest.raises(ValueError, match="valid Start date"):
        parse_universe_membership(io.BytesIO(bad_start))

    bad_end = b"Ticker,Start date,End date\nAAA,2020-01-01,not-a-date\n"
    with pytest.raises(ValueError, match="non-empty End date must be a valid date"):
        parse_universe_membership(io.BytesIO(bad_end))


def test_point_in_time_membership_filters_before_cross_sectional_factor_ranking():
    from qlab.strategy import _build_monthly_cross_section, _precompute_indicator_cache
    from qlab.universe import parse_universe_membership

    prices = synthetic_prices(1000)
    decision_date = pd.Timestamp("2020-06-30")
    history = parse_universe_membership(pd.DataFrame([
        {"Ticker":"AAA","Start date":"2018-01-01","End date":""},
        {"Ticker":"BBB","Start date":"2018-01-01","End date":""},
        {"Ticker":"CCC","Start date":"2021-01-01","End date":""},
    ]))
    cache = _precompute_indicator_cache(prices)
    point_in_time = _build_monthly_cross_section(
        prices, decision_date, None, None, indicator_cache=cache, universe_history=history
    )
    contemporaneous_universe = _build_monthly_cross_section(
        {ticker: prices[ticker] for ticker in ["AAA","BBB"]},
        decision_date, None, None,
        indicator_cache={ticker: cache[ticker] for ticker in ["AAA","BBB"]},
    )
    assert set(point_in_time["Ticker"]) == {"AAA", "BBB"}
    assert set(contemporaneous_universe["Ticker"]) == {"AAA", "BBB"}
    left = point_in_time.set_index("Ticker").sort_index()
    right = contemporaneous_universe.set_index("Ticker").sort_index()
    pd.testing.assert_series_equal(left["Momentum score"], right["Momentum score"])
    pd.testing.assert_series_equal(left["Fundamental/composite score"], right["Fundamental/composite score"])


def test_backtest_never_selects_a_ticker_before_its_membership_start_date():
    from qlab.strategy import run_monthly_backtest
    from qlab.universe import parse_universe_membership

    prices = synthetic_prices(1300)
    history = parse_universe_membership(pd.DataFrame([
        {"Ticker":"AAA","Start date":"2018-01-01","End date":""},
        {"Ticker":"BBB","Start date":"2018-01-01","End date":""},
        {"Ticker":"CCC","Start date":"2021-01-01","End date":""},
    ]))
    curve, metrics, audit = run_monthly_backtest(
        prices=prices, top_n=2, cost_bps=10, initial=10_000,
        min_adx=0, min_score=0, hysteresis=0, max_replacements=2,
        use_fundamentals=False, stop_mode="No stop / month-end rebalancing",
        universe_history=history,
    )
    assert curve is not None and metrics is not None and audit is not None
    decisions = pd.to_datetime(audit["Signal date"])
    before_ccc = audit.loc[decisions < pd.Timestamp("2021-01-01"), "Selected holdings"]
    assert before_ccc.map(lambda names: "CCC" not in names.split(", ")).all()
    from qlab.strategy import _build_monthly_cross_section, _precompute_indicator_cache
    late_cross = _build_monthly_cross_section(
        prices, pd.Timestamp("2021-06-30"), None, None,
        indicator_cache=_precompute_indicator_cache(prices), universe_history=history,
    )
    assert "CCC" in set(late_cross["Ticker"])


def test_backtest_fails_fast_when_point_in_time_universe_lists_unavailable_ohlcv_symbol():
    from qlab.strategy import run_monthly_backtest
    from qlab.universe import parse_universe_membership

    history = parse_universe_membership(pd.DataFrame([
        {"Ticker":"AAA","Start date":"2018-01-01","End date":""},
        {"Ticker":"DELISTED","Start date":"2018-01-01","End date":"2021-05-31"},
    ]))
    with pytest.raises(ValueError, match="without any OHLCV series.*DELISTED"):
        run_monthly_backtest(
            prices=synthetic_prices(600),
            use_fundamentals=False,
            universe_history=history,
        )


def test_csv_price_adapter_preserves_currency_metadata_and_rejects_mixed_currency_within_ticker():
    import io
    from qlab.data import load_ohlcv_csv

    csv = pd.DataFrame([
        {"Date":"2025-01-02","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10,"Currency":"USD"},
        {"Date":"2025-01-03","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10.5,"Currency":"USD"},
        {"Date":"2025-01-02","Ticker":"BBB","Open":20,"High":21,"Low":19,"Close":20,"Currency":"EUR"},
        {"Date":"2025-01-03","Ticker":"BBB","Open":20,"High":21,"Low":19,"Close":20.5,"Currency":"EUR"},
    ]).to_csv(index=False).encode("utf-8")
    prices, report = load_ohlcv_csv(io.BytesIO(csv), expected_tickers=["AAA","BBB"])
    assert prices["AAA"].attrs["currency"] == "USD"
    assert prices["BBB"].attrs["currency"] == "EUR"
    currency = report.set_index("Ticker")["Currency"].to_dict()
    assert currency == {"AAA":"USD","BBB":"EUR"}

    mixed = pd.DataFrame([
        {"Date":"2025-01-02","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10,"Currency":"USD"},
        {"Date":"2025-01-03","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10.5,"Currency":"EUR"},
    ]).to_csv(index=False).encode("utf-8")
    with pytest.raises(ValueError, match="Currency changes within the imported series"):
        load_ohlcv_csv(io.BytesIO(mixed), expected_tickers=["AAA"])


def test_csv_price_adapter_requires_currency_for_all_rows_when_currency_column_is_present():
    import io
    from qlab.data import load_ohlcv_csv

    payload = pd.DataFrame([
        {"Date":"2025-01-02","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10,"Currency":"USD"},
        {"Date":"2025-01-03","Ticker":"AAA","Open":10,"High":11,"Low":9,"Close":10.5,"Currency":""},
    ]).to_csv(index=False).encode("utf-8")
    with pytest.raises(ValueError, match="Currency is blank on one or more rows"):
        load_ohlcv_csv(io.BytesIO(payload), expected_tickers=["AAA"])


def test_usd_price_basis_gate_requires_usd_metadata_or_explicit_attestation():
    from qlab.data import validate_usd_price_basis

    assert validate_usd_price_basis(["AAA", "BBB"], {"AAA":"USD", "BBB":"USD"})["ready"]
    unknown = validate_usd_price_basis(["AAA", "BBB"], {"AAA":"USD", "BBB":"UNKNOWN"})
    assert not unknown["ready"]
    assert unknown["unknown_tickers"] == ["BBB"]
    attested = validate_usd_price_basis(
        ["AAA", "BBB"], {"AAA":"USD", "BBB":"UNKNOWN"}, confirm_unknown_usd=True
    )
    assert attested["ready"]
    mixed = validate_usd_price_basis(
        ["AAA", "BBB"], {"AAA":"USD", "BBB":"EUR"}, confirm_unknown_usd=True
    )
    assert not mixed["ready"]
    assert mixed["non_usd_currencies"] == ["EUR"]
    missing = validate_usd_price_basis(
        ["AAA", "DELISTED"], {"AAA":"USD"}, missing_tickers=["DELISTED"]
    )
    assert not missing["ready"]
    assert missing["missing_tickers"] == ["DELISTED"]


def test_sec_financial_ratios_and_valuation_are_suppressed_when_xbrl_units_disagree():
    from qlab.data import sec_fundamental_snapshot

    def observations(values, filed, unit="USD"):
        return {unit: [
            {"start":f"{year}-01-01","end":f"{year}-12-31","val":value,"filed":filing,"form":"10-K"}
            for year, value, filing in zip([2021, 2022], values, filed)
        ]}
    facts = {"facts":{"us-gaap":{
        "Revenues":{"units":observations([100,150],["2022-02-15","2023-02-15"],"USD")},
        "NetIncomeLoss":{"units":observations([10,15],["2022-02-15","2023-02-15"],"EUR")},
        "NetCashProvidedByUsedInOperatingActivities":{"units":observations([18,25],["2022-02-15","2023-02-15"],"USD")},
        "PaymentsToAcquirePropertyPlantAndEquipment":{"units":observations([4,5],["2022-02-15","2023-02-15"],"EUR")},
        "StockholdersEquity":{"units":{"USD":[
            {"end":"2022-12-31","val":60,"filed":"2023-02-15","form":"10-K"},
        ]}},
        "EntityCommonStockSharesOutstanding":{"units":{"shares":[
            {"end":"2022-12-31","val":100,"filed":"2023-02-15","form":"10-K"},
        ]}},
    }}}
    result = sec_fundamental_snapshot(
        {"ticker":"MIX","title":"Mixed Units Ltd","payload":facts},
        as_of="2024-06-30", latest_price=20, price_currency="USD",
    )
    assert result["Revenue unit"] == "USD"
    assert result["Net income unit"] == "EUR"
    assert result["FCF unit"] == ""
    assert pd.isna(result["FCF"])
    assert pd.isna(result["Profit margin %"])
    assert pd.isna(result["ROE %"])
    assert pd.isna(result["FCF margin %"])
    assert pd.isna(result["Estimated market cap"])
    assert pd.isna(result["P/E (annual EPS proxy)"])
    assert pd.isna(result["FCF yield %"])
    assert result["Revenue growth %"] == pytest.approx(50.0)
    assert result["Earnings growth %"] == pytest.approx(50.0)


def test_sec_estimated_multiples_require_explicit_matching_quote_currency():
    from qlab.data import sec_fundamental_snapshot

    def annual(values, filings):
        return {"USD": [
            {"start":f"{year}-01-01","end":f"{year}-12-31","val":value,"filed":filed,"form":"10-K"}
            for year, value, filed in zip([2021, 2022], values, filings)
        ]}
    facts = {"facts":{"us-gaap":{
        "Revenues":{"units":annual([100,150],["2022-02-15","2023-02-15"])},
        "NetIncomeLoss":{"units":annual([10,15],["2022-02-15","2023-02-15"])},
        "NetCashProvidedByUsedInOperatingActivities":{"units":annual([18,25],["2022-02-15","2023-02-15"])},
        "PaymentsToAcquirePropertyPlantAndEquipment":{"units":annual([4,5],["2022-02-15","2023-02-15"])},
        "StockholdersEquity":{"units":{"USD":[{"end":"2022-12-31","val":60,"filed":"2023-02-15","form":"10-K"}]}},
        "EntityCommonStockSharesOutstanding":{"units":{"shares":[{"end":"2022-12-31","val":100,"filed":"2023-02-15","form":"10-K"}]}},
    }}}
    bundle={"ticker":"AAA","title":"AAA Inc.","payload":facts}
    unknown_quote = sec_fundamental_snapshot(bundle,"2024-06-30",latest_price=20)
    usd_quote = sec_fundamental_snapshot(bundle,"2024-06-30",latest_price=20,price_currency="USD")
    eur_quote = sec_fundamental_snapshot(bundle,"2024-06-30",latest_price=20,price_currency="EUR")
    assert pd.isna(unknown_quote["Estimated market cap"])
    assert pd.isna(unknown_quote["P/E (annual EPS proxy)"])
    assert usd_quote["Estimated market cap"] == pytest.approx(2_000)
    assert usd_quote["P/E (annual EPS proxy)"] == pytest.approx(2_000 / 15)
    assert pd.isna(eur_quote["Estimated market cap"])
    assert pd.isna(eur_quote["P/E (annual EPS proxy)"])


def test_annual_series_selects_one_unit_instead_of_mixing_parallel_currency_observations():
    from qlab.data import _annual_series

    facts = {"facts":{"us-gaap":{
        "Revenues":{"units":{
            "USD":[
                {"start":"2021-01-01","end":"2021-12-31","val":100,"filed":"2022-02-15","form":"10-K"},
                {"start":"2022-01-01","end":"2022-12-31","val":150,"filed":"2023-02-15","form":"10-K"},
            ],
            "EUR":[
                {"start":"2021-01-01","end":"2021-12-31","val":900,"filed":"2022-02-15","form":"10-K"},
                {"start":"2022-01-01","end":"2022-12-31","val":1200,"filed":"2023-02-15","form":"10-K"},
            ],
        }}
    }}}
    result = _annual_series(facts, ["Revenues"], "2024-06-30")
    assert result.attrs["unit"] == "USD"
    assert result.to_list() == [100.0, 150.0]
    assert result.iloc[-1] != 1200.0


def test_price_quality_report_uses_last_valid_bar_and_marks_fully_invalid_ticker_missing():
    from qlab.data import assess_price_quality

    index = pd.to_datetime(["2026-10-08","2026-10-09","2026-10-12","2026-10-13"])
    partially_invalid = pd.DataFrame({
        "Open":[10.0,10.0,10.0,0.0],
        "High":[11.0,11.0,11.0,0.0],
        "Low":[9.0,9.0,9.0,0.0],
        "Close":[10.0,10.5,10.8,0.0],
        "Adj Close":[10.0,10.5,10.8,0.0],
    },index=index)
    fully_invalid = pd.DataFrame({
        "Open":[0.0,0.0],"High":[0.0,0.0],"Low":[0.0,0.0],
        "Close":[0.0,np.nan],"Adj Close":[0.0,np.nan],
    },index=pd.to_datetime(["2026-10-12","2026-10-13"]))
    report = pd.DataFrame([
        {"Ticker":"AAA","Status":"OK","Rows":4,"First date":"2026-10-08","Last date":"2026-10-13","Missing %":0.0,"Message":""},
        {"Ticker":"BAD","Status":"OK","Rows":2,"First date":"2026-10-12","Last date":"2026-10-13","Missing %":0.0,"Message":""},
    ])
    sanitized, out = assess_price_quality({"AAA":partially_invalid,"BAD":fully_invalid},report,now="2026-10-13")
    aa=out.set_index("Ticker").loc["AAA"]
    bad=out.set_index("Ticker").loc["BAD"]
    assert aa["Last date"] == "2026-10-12"
    assert aa["Rows"] == 3
    assert pd.isna(sanitized["AAA"].loc[pd.Timestamp("2026-10-13"),"Close"])
    assert bad["Status"] == "MISSING"
    assert bad["Rows"] == 0
    assert bad["Missing %"] == 100.0
    assert not pd.to_numeric(sanitized["BAD"]["Close"],errors="coerce").notna().any()


def test_snapshot_review_flags_material_deterioration_and_preserves_hold_positions():
    from qlab.review import compare_screener_snapshots

    previous = pd.DataFrame([
        {"Ticker":"AAA","Composite score":82.0,"Momentum 12-1M":0.35,"Trend OK":True,
         "Below 50d low":False,"Signal":"Breakout candidate","Latest filing date":"2025-10-01"},
        {"Ticker":"BBB","Composite score":61.0,"Momentum 12-1M":0.12,"Trend OK":True,
         "Below 50d low":False,"Signal":"Trend watch","Latest filing date":"2025-10-01"},
        {"Ticker":"CCC","Composite score":50.0,"Momentum 12-1M":0.10,"Trend OK":True,
         "Below 50d low":False,"Signal":"Trend watch","Latest filing date":"2025-10-01"},
    ])
    current = pd.DataFrame([
        {"Ticker":"AAA","Fundamental/composite score":81.0,"Momentum 12-1M":0.34,"Trend OK":True,
         "Below 50d low":False,"Signal":"Breakout candidate","Latest filing date":"2025-10-01"},
        {"Ticker":"BBB","Fundamental/composite score":48.0,"Momentum 12-1M":-0.02,"Trend OK":False,
         "Below 50d low":True,"Signal":"Review / exit trigger","Latest filing date":"2025-12-01"},
        {"Ticker":"DDD","Fundamental/composite score":55.0,"Momentum 12-1M":0.15,"Trend OK":True,
         "Below 50d low":False,"Signal":"Trend watch","Latest filing date":"2025-12-01"},
    ])
    review = compare_screener_snapshots(current, previous, score_change_points=7, momentum_change=0.10)
    by_ticker = review.set_index("Ticker")
    assert by_ticker.loc["AAA","Review action"] == "NO MATERIAL CHANGE"
    assert by_ticker.loc["BBB","Review action"] == "EXIT REVIEW — thesis/technical break"
    assert by_ticker.loc["BBB","Material change"]
    assert "fell by 13.0 points" in by_ticker.loc["BBB","Reasons"]
    assert by_ticker.loc["DDD","Review action"] == "NEW — review"
    assert by_ticker.loc["CCC","Review action"] == "NOT PRESENT — check data/universe"


def test_snapshot_review_rejects_duplicate_tickers_and_bad_thresholds():
    from qlab.review import compare_screener_snapshots

    duplicated = pd.DataFrame([{"Ticker":"AAA","Composite score":50},{"Ticker":"aaa","Composite score":55}])
    with pytest.raises(ValueError, match="duplicate ticker"):
        compare_screener_snapshots(duplicated, duplicated)
    good = pd.DataFrame([{"Ticker":"AAA","Composite score":50}])
    with pytest.raises(ValueError, match="score_change_points"):
        compare_screener_snapshots(good, good, score_change_points=-1)


def test_fifo_rejects_missing_or_infinite_numeric_inputs():
    from qlab.tax_lots import fifo_ledger

    for field, value in [("Quantity", np.nan), ("Price", np.inf), ("Fees", -np.inf)]:
        tx = pd.DataFrame([{
            "Date":"2024-01-01","Ticker":"AAA","Type":"BUY",
            "Quantity":10.0,"Price":100.0,"Fees":0.0,"Currency":"USD",
        }])
        tx.loc[0, field] = value
        with pytest.raises(ValueError, match="finite numbers"):
            fifo_ledger(tx)


def test_fifo_never_assumes_usd_when_currency_is_missing():
    from qlab.tax_lots import fifo_ledger

    tx = pd.DataFrame([
        {"Date":"2020-01-01","Ticker":"AAA","Type":"BUY","Quantity":10,"Price":100,"Fees":0},
        {"Date":"2024-02-01","Ticker":"AAA","Type":"SELL","Quantity":10,"Price":200,"Fees":0},
    ])
    realized, _, _ = fifo_ledger(tx)
    assert realized.iloc[0]["Currency"] == "UNKNOWN"
    assert "convert proceeds to CZK" in realized.iloc[0]["Tax review status"]
    assert "Potential value-test exemption" not in realized.iloc[0]["Tax review status"]
