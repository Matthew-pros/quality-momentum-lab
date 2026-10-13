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
    previous_55_high = frame["High"].iloc[-56:-1].max()
    assert indicators["ATR20"].dropna().gt(0).all()
    assert indicators["ATR20%"].dropna().gt(0).all()
    assert indicators["ADX14"].dropna().between(0, 100).all()
    assert np.isclose(indicators.loc[last, "High55 prior"], previous_55_high)
    assert indicators.loc[last, "Close"] == frame["Close"].iloc[-1]
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
