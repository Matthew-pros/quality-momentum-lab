"""Scenario-based discounted cash-flow valuation helper.

Uses reported operating cash flow less capital expenditure as a simplified FCF/FCFE proxy.
The model is educational and deliberately separate from ranking/backtest signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_fcfe_dcf(
    base_fcf: float,
    annual_growth_rate: float,
    cost_of_equity: float,
    terminal_growth_rate: float,
    years: int = 5,
    shares_outstanding: float | None = None,
) -> dict[str, float]:
    """Calculate an equity-style DCF scenario from free cash flow and cost of equity."""
    values = [base_fcf, annual_growth_rate, cost_of_equity, terminal_growth_rate]
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("DCF inputs must be finite numbers.")
    if base_fcf <= 0:
        raise ValueError("Base free cash flow must be positive for this simplified DCF scenario.")
    if years < 1 or years > 20:
        raise ValueError("Forecast horizon must be from 1 to 20 years.")
    if cost_of_equity <= terminal_growth_rate:
        raise ValueError("Cost of equity must be greater than terminal growth.")
    if cost_of_equity <= -1 or terminal_growth_rate <= -1:
        raise ValueError("Discount and terminal-growth rates are outside valid ranges.")
    if shares_outstanding is not None and (not np.isfinite(float(shares_outstanding)) or shares_outstanding <= 0):
        raise ValueError("Shares outstanding must be positive when supplied.")

    annual_growth = float(annual_growth_rate)
    discount = float(cost_of_equity)
    terminal_growth = float(terminal_growth_rate)
    base = float(base_fcf)

    forecast_values = np.array([base * (1 + annual_growth) ** year for year in range(1, years + 1)], dtype=float)
    discounted_forecast = forecast_values / np.array([(1 + discount) ** year for year in range(1, years + 1)], dtype=float)
    terminal_cash_flow = float(forecast_values[-1] * (1 + terminal_growth))
    terminal_value = terminal_cash_flow / (discount - terminal_growth)
    discounted_terminal_value = terminal_value / ((1 + discount) ** years)
    pv_forecast = float(discounted_forecast.sum())
    equity_value = pv_forecast + float(discounted_terminal_value)

    result = {
        "Base FCF": base,
        "PV forecast FCF": pv_forecast,
        "Terminal FCF (year after forecast)": terminal_cash_flow,
        "Terminal value at forecast horizon": float(terminal_value),
        "PV terminal value": float(discounted_terminal_value),
        "Implied equity value": float(equity_value),
        "Terminal value share %": float(discounted_terminal_value / equity_value * 100) if equity_value else np.nan,
        "Forecast years": float(years),
        "FCF growth %": annual_growth * 100,
        "Cost of equity %": discount * 100,
        "Terminal growth %": terminal_growth * 100,
    }
    if shares_outstanding is not None:
        result["Shares outstanding"] = float(shares_outstanding)
        result["Implied value per share"] = float(equity_value / float(shares_outstanding))
    return result


def dcf_sensitivity(
    base_fcf: float,
    shares_outstanding: float,
    growth_rates: list[float],
    discount_rates: list[float],
    terminal_growth_rate: float,
    years: int = 5,
) -> pd.DataFrame:
    """Build a value-per-share sensitivity matrix across growth and discount assumptions."""
    if not np.isfinite(float(shares_outstanding)) or shares_outstanding <= 0:
        raise ValueError("Shares outstanding must be positive.")
    rows = []
    for growth in growth_rates:
        row = {"FCF growth %": float(growth) * 100}
        for discount in discount_rates:
            label = f"Cost of equity {float(discount)*100:.1f}%"
            if discount <= terminal_growth_rate or base_fcf <= 0:
                row[label] = np.nan
                continue
            values = calculate_fcfe_dcf(
                base_fcf=float(base_fcf),
                annual_growth_rate=float(growth),
                cost_of_equity=float(discount),
                terminal_growth_rate=float(terminal_growth_rate),
                years=years,
                shares_outstanding=float(shares_outstanding),
            )
            row[label] = values["Implied value per share"]
        rows.append(row)
    return pd.DataFrame(rows)
