import ast
from pathlib import Path

import numpy as np
import pandas as pd


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
SOURCE = APP_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(APP_PATH))

# app.py contains the Streamlit UI at module level, so load only pure math
# functions for unit tests without executing the UI or making network calls.
PURE_FUNCTIONS = {
    "calc_adx",
    "percentile_score",
    "weighted_average",
    "combine_percentiles",
    "run_monthly_backtest",
}
function_nodes = [
    node for node in TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in PURE_FUNCTIONS
]
found = {node.name for node in function_nodes}
assert found == PURE_FUNCTIONS, f"Missing pure functions: {PURE_FUNCTIONS - found}"
for node in function_nodes:
    node.decorator_list = []

module = ast.Module(body=function_nodes, type_ignores=[])
namespace = {"pd": pd, "np": np}
exec(compile(module, str(APP_PATH), "exec"), namespace)


def test_percentile_score_respects_direction_and_keeps_missing_values():
    values = pd.Series([20.0, 10.0, np.nan], index=["high", "low", "missing"])
    higher = namespace["percentile_score"](values, higher_better=True)
    lower = namespace["percentile_score"](values, higher_better=False)

    assert higher["high"] > higher["low"]
    assert lower["low"] > lower["high"]
    assert np.isnan(higher["missing"])
    assert np.isnan(lower["missing"])


def test_weighted_average_reweights_available_inputs_only():
    frame = pd.DataFrame({
        "quality": [80.0, np.nan, np.nan],
        "momentum": [40.0, 70.0, np.nan],
    }, index=["both", "momentum_only", "neither"])

    result = namespace["weighted_average"](frame, {"quality": 0.6, "momentum": 0.4})

    assert np.isclose(result["both"], 64.0)
    assert np.isclose(result["momentum_only"], 70.0)
    assert np.isclose(result["neither"], 50.0)


def test_adx_and_atr_are_finite_nonnegative_for_synthetic_trend():
    index = pd.bdate_range("2020-01-01", periods=150)
    close = pd.Series(np.linspace(50, 100, len(index)), index=index)
    high = close + 1.5
    low = close - 1.5

    adx, atr = namespace["calc_adx"](high, low, close, n=14)

    assert adx.dropna().gt(0).all()
    assert atr.dropna().gt(0).all()
    assert adx.dropna().le(100).all()


def test_backtest_uses_return_for_month_following_signal_month():
    index = pd.bdate_range("2010-01-01", periods=5000)
    t = np.arange(len(index), dtype=float)
    prices = pd.DataFrame({
        "AAA": 50 * np.exp(0.00045 * t) * (1 + 0.025 * np.sin(t / 70)),
        "BBB": 70 * np.exp(0.00025 * t) * (1 + 0.035 * np.sin(t / 53 + 1)),
        "CCC": 90 * np.exp(0.00010 * t) * (1 + 0.020 * np.sin(t / 45 + 2)),
    }, index=index)

    curve, metrics = namespace["run_monthly_backtest"](
        prices, top_n=2, cost_bps=10, initial=10_000
    )
    monthly = prices.resample("ME").last().dropna(how="all")
    returns = monthly.pct_change(fill_method=None)
    i = 13

    momentum_12_1 = monthly.iloc[i - 1] / monthly.iloc[i - 12] - 1
    momentum_6_1 = monthly.iloc[i - 1] / monthly.iloc[i - 6] - 1
    trend = monthly.iloc[i - 1] > monthly.iloc[max(0, i - 12):i].mean()
    score = (momentum_12_1.rank(pct=True) + momentum_6_1.rank(pct=True)) / 2
    picks = set(score[trend].dropna().sort_values(ascending=False).head(2).index)
    expected_gross_return = returns.iloc[i].reindex(list(picks)).fillna(0.0).mean()
    expected_cost = (1.0 if picks else 0.0) * 10 / 10_000

    assert curve is not None
    assert metrics is not None
    assert curve.index[0] == monthly.index[i]
    assert np.isclose(curve["Return"].iloc[0], expected_gross_return - expected_cost)
    assert len(curve) == len(monthly) - 13


def test_backtest_drawdown_includes_initial_capital_as_high_water_mark():
    index = pd.bdate_range("2010-01-01", periods=5000)
    t = np.arange(len(index), dtype=float)
    prices = pd.DataFrame({
        "AAA": 100 * np.exp(0.0002 * t) * (1 + 0.08 * np.sin(t / 32)),
        "BBB": 100 * np.exp(0.0001 * t) * (1 + 0.07 * np.sin(t / 29 + 1)),
        "CCC": 100 * np.exp(0.0003 * t) * (1 + 0.06 * np.sin(t / 38 + 2)),
    }, index=index)

    curve, metrics = namespace["run_monthly_backtest"](
        prices, top_n=2, cost_bps=0, initial=10_000
    )
    assert curve is not None and metrics is not None
    wealth_with_initial = np.r_[10_000, curve["Equity"].to_numpy()]
    high_water = np.maximum.accumulate(wealth_with_initial)[1:]
    expected_max_drawdown = (curve["Equity"].to_numpy() / high_water - 1).min()
    assert np.isclose(metrics["Max drawdown"], expected_max_drawdown)
