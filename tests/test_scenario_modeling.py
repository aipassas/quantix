"""Tests for scenario_modeling.py — DCF/risk scenario shocks wired into
the existing engines, the dividend-cut real-number calculation, and
scenario persistence.

The most important test in this file is the risk-shock no-op invariant:
a scenario with volatility_multiplier=1.0 and mean_return_shift=0.0 must
reproduce EVERY base risk metric to floating-point precision, since it's
mathematically a pass-through. This directly encodes a real bug caught
while building this — the shocked-series reconstruction was silently
dropping the single oldest return observation in the window, which moved
Sharpe by several points even under a supposed no-op.
"""
import datetime
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

import scenario_modeling as sm
from scenario_modeling import (
    SCENARIO_TYPES,
    ScenarioDefinition,
    apply_dcf_scenario,
    apply_risk_scenario,
    default_scenario,
    delete_scenario,
    dividend_cut_impact,
    load_scenarios,
    run_scenario,
    save_scenario,
)


class _FakeEngine:
    """Mirrors FundamentalAnalysisEngine.intrinsic_price()'s signature
    with a simple, hand-checkable formula: value falls as discount rate
    rises, rises as growth rate rises — enough to test the SHOCK WIRING
    (does apply_dcf_scenario pass the right shocked numbers through)
    without needing real statement data."""
    def intrinsic_price(self, growth_rate: float, discount_rate: float) -> float:
        return 100.0 * (1 + growth_rate) / discount_rate


# --- default_scenario --------------------------------------------------------

def test_default_scenario_covers_all_three_required_types():
    for stype in ("dividend_cut", "recession", "sector_shift"):
        scenario = default_scenario(stype)
        assert scenario.scenario_type == stype
        assert scenario.created_at  # timestamped, not blank

    assert set(SCENARIO_TYPES) == {"dividend_cut", "recession", "sector_shift"}


def test_default_scenario_rejects_unknown_type():
    with pytest.raises(ValueError):
        default_scenario("hyperinflation")


def test_default_dividend_cut_has_no_growth_or_volatility_shock():
    """Central design point: this app's DCF is dividend-policy-invariant,
    so the DEFAULT dividend-cut scenario must not silently move the DCF
    or risk numbers unless the user opts into a discount-rate add-on."""
    scenario = default_scenario("dividend_cut")
    assert scenario.growth_rate_delta == 0.0
    assert scenario.discount_rate_delta == 0.0
    assert scenario.volatility_multiplier == 1.0
    assert scenario.mean_return_shift == 0.0
    assert scenario.dividend_cut_pct > 0


def test_default_recession_shocks_growth_discount_and_volatility():
    scenario = default_scenario("recession")
    assert scenario.growth_rate_delta < 0
    assert scenario.discount_rate_delta > 0
    assert scenario.volatility_multiplier > 1.0
    assert scenario.mean_return_shift < 0


def test_default_sector_shift_shocks_discount_rate():
    scenario = default_scenario("sector_shift")
    assert scenario.discount_rate_delta > 0
    assert scenario.growth_rate_delta == 0.0  # sector re-rating, not a growth call


# --- apply_dcf_scenario -----------------------------------------------------

def test_dcf_scenario_computes_correct_shocked_values():
    engine = _FakeEngine()
    scenario = ScenarioDefinition(name="t", scenario_type="recession", growth_rate_delta=-0.05, discount_rate_delta=0.02)
    result = apply_dcf_scenario(engine, base_growth_rate=0.08, base_discount_rate=0.09, scenario=scenario)

    assert result.ok
    assert result.shocked_growth_rate == pytest.approx(0.03)
    assert result.shocked_discount_rate == pytest.approx(0.11)
    assert result.base_intrinsic_price == pytest.approx(engine.intrinsic_price(0.08, 0.09))
    assert result.shocked_intrinsic_price == pytest.approx(engine.intrinsic_price(0.03, 0.11))
    expected_pct = (result.shocked_intrinsic_price - result.base_intrinsic_price) / result.base_intrinsic_price * 100
    assert result.pct_change == pytest.approx(expected_pct)


def test_dcf_scenario_no_op_reproduces_base_exactly():
    engine = _FakeEngine()
    scenario = ScenarioDefinition(name="t", scenario_type="recession")  # all deltas 0.0
    result = apply_dcf_scenario(engine, base_growth_rate=0.06, base_discount_rate=0.09, scenario=scenario)
    assert result.pct_change == pytest.approx(0.0, abs=1e-9)
    assert result.base_intrinsic_price == pytest.approx(result.shocked_intrinsic_price)


def test_dcf_scenario_refuses_when_discount_rate_collapses_to_terminal_growth():
    from config import DCF
    engine = _FakeEngine()
    # Push the shocked discount rate at/below DCF.terminal_growth_rate.
    huge_shock = ScenarioDefinition(name="t", scenario_type="sector_shift", discount_rate_delta=-0.20)
    result = apply_dcf_scenario(engine, base_growth_rate=0.05, base_discount_rate=DCF.terminal_growth_rate + 0.01, scenario=huge_shock)
    assert result.ok is False
    assert "terminal growth rate" in result.reason


def test_dcf_scenario_handles_engine_exception_gracefully():
    class _BrokenEngine:
        def intrinsic_price(self, g, d):
            raise ZeroDivisionError("boom")
    result = apply_dcf_scenario(_BrokenEngine(), 0.05, 0.09, default_scenario("recession"))
    assert result.ok is False
    assert "boom" in result.reason


# --- apply_risk_scenario: the no-op invariant is the load-bearing test ------

def test_risk_scenario_no_op_reproduces_every_metric_exactly(clean_ohlcv):
    """The core correctness guarantee: multiplier=1.0, shift=0.0 must be
    mathematically a pass-through. This is the exact invariant whose
    violation (dropping the oldest return observation) was caught live
    while building this feature."""
    no_op = ScenarioDefinition(name="t", scenario_type="dividend_cut")  # defaults: multiplier=1.0, shift=0.0
    result = apply_risk_scenario(clean_ohlcv, no_op, confidence_level=0.95)

    assert result.ok
    assert result.shocked_var_pct == pytest.approx(result.base_var_pct, abs=1e-9)
    assert result.shocked_cvar_pct == pytest.approx(result.base_cvar_pct, abs=1e-9)
    assert result.shocked_sharpe == pytest.approx(result.base_sharpe, abs=1e-6)
    assert result.shocked_max_drawdown_pct == pytest.approx(result.base_max_drawdown_pct, abs=1e-9)


def test_risk_scenario_higher_volatility_widens_var_and_cvar(clean_ohlcv):
    scenario = ScenarioDefinition(name="t", scenario_type="recession", volatility_multiplier=2.0)
    result = apply_risk_scenario(clean_ohlcv, scenario, confidence_level=0.95)
    assert result.ok
    # A wider return distribution means a MORE NEGATIVE (larger-magnitude) VaR/CVaR.
    assert result.shocked_var_pct < result.base_var_pct
    assert result.shocked_cvar_pct < result.base_cvar_pct


def test_risk_scenario_negative_mean_shift_worsens_drawdown(clean_ohlcv):
    scenario = ScenarioDefinition(name="t", scenario_type="recession", mean_return_shift=-0.01)
    result = apply_risk_scenario(clean_ohlcv, scenario, confidence_level=0.95)
    assert result.ok
    assert result.shocked_max_drawdown_pct < result.base_max_drawdown_pct


def test_risk_scenario_empty_df_is_refused_not_crashed():
    result = apply_risk_scenario(pd.DataFrame(), default_scenario("recession"), confidence_level=0.95)
    assert result.ok is False
    assert result.reason


# --- dividend_cut_impact: the one real, non-fabricated number ---------------

def test_dividend_cut_impact_computes_yield_from_price_not_yahoo_field():
    """Yield is DERIVED (dividendRate / price), never trusted directly
    from Yahoo's own dividendYield field — that field's scale has been
    unreliable elsewhere in this app's own experience."""
    info = {"dividendRate": 2.0, "dividendYield": 999.0}  # deliberately wrong/irrelevant Yahoo field
    result = dividend_cut_impact(info, current_price=100.0, cut_pct=50.0)
    assert result.applicable
    assert result.current_yield_pct == pytest.approx(2.0)  # 2.00 / 100.00 * 100, NOT 999.0
    assert result.shocked_annual_dividend == pytest.approx(1.0)
    assert result.lost_annual_income_per_share == pytest.approx(1.0)
    assert result.shocked_yield_pct == pytest.approx(1.0)


def test_dividend_cut_impact_not_applicable_when_no_dividend():
    result = dividend_cut_impact({"dividendRate": None}, current_price=100.0, cut_pct=50.0)
    assert result.applicable is False
    assert "does not report" in result.detail


def test_dividend_cut_impact_not_applicable_without_price():
    result = dividend_cut_impact({"dividendRate": 2.0}, current_price=None, cut_pct=50.0)
    assert result.applicable is False


def test_dividend_cut_impact_full_cut_zeroes_dividend():
    result = dividend_cut_impact({"dividendRate": 3.0}, current_price=50.0, cut_pct=100.0)
    assert result.shocked_annual_dividend == pytest.approx(0.0)
    assert result.shocked_yield_pct == pytest.approx(0.0)


# --- run_scenario orchestration ----------------------------------------------

def test_run_scenario_only_computes_dividend_for_dividend_cut_type(clean_ohlcv):
    engine = _FakeEngine()
    info = {"dividendRate": 2.0, "currentPrice": 100.0}
    recession = default_scenario("recession")
    result = run_scenario(engine, clean_ohlcv, info, 0.06, 0.09, recession, confidence_level=0.95)
    assert result.dividend.applicable is False


def test_run_scenario_computes_dividend_for_dividend_cut_type(clean_ohlcv):
    engine = _FakeEngine()
    info = {"dividendRate": 2.0, "currentPrice": 100.0}
    cut = default_scenario("dividend_cut")
    result = run_scenario(engine, clean_ohlcv, info, 0.06, 0.09, cut, confidence_level=0.95)
    assert result.dividend.applicable


def test_run_scenario_implied_portfolio_value_change_scales_with_investment(clean_ohlcv):
    engine = _FakeEngine()
    info = {"currentPrice": 100.0}
    recession = default_scenario("recession")
    result = run_scenario(engine, clean_ohlcv, info, 0.06, 0.09, recession, confidence_level=0.95, investment_amount=5000)
    expected = 5000 * (result.dcf.pct_change / 100)
    assert result.implied_portfolio_value_change == pytest.approx(expected)


# --- persistence -----------------------------------------------------------------

def test_save_and_load_scenario_round_trip(tmp_path):
    path = tmp_path / "store.json"
    scenario = default_scenario("recession")
    save_scenario(scenario, path)
    loaded = load_scenarios(path)
    assert len(loaded) == 1
    assert loaded[0].name == "Recession"
    assert loaded[0].growth_rate_delta == scenario.growth_rate_delta


def test_save_scenario_updates_existing_by_name(tmp_path):
    path = tmp_path / "store.json"
    first = default_scenario("recession")
    save_scenario(first, path)

    edited = ScenarioDefinition(name="Recession", scenario_type="recession", growth_rate_delta=-0.99, created_at="now")
    save_scenario(edited, path)

    loaded = load_scenarios(path)
    assert len(loaded) == 1  # updated in place, not duplicated
    assert loaded[0].growth_rate_delta == pytest.approx(-0.99)


def test_delete_scenario_removes_only_the_named_one(tmp_path):
    path = tmp_path / "store.json"
    save_scenario(default_scenario("recession"), path)
    save_scenario(default_scenario("sector_shift"), path)
    remaining = delete_scenario("Recession", path)
    assert [s.name for s in remaining] == ["Sector Multiple Compression"]


def test_load_scenarios_missing_file_returns_empty(tmp_path):
    assert load_scenarios(tmp_path / "nope.json") == []


def test_load_scenarios_corrupt_file_degrades_to_empty_not_raise(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{not valid json")
    assert load_scenarios(path) == []


def test_save_scenarios_trims_to_max_saved(tmp_path):
    from config import SCENARIO_MODELING
    path = tmp_path / "store.json"
    scenarios = [
        ScenarioDefinition(name=f"S{i}", scenario_type="recession", created_at="now")
        for i in range(SCENARIO_MODELING.max_saved_scenarios + 10)
    ]
    sm.save_scenarios(scenarios, path)
    loaded = load_scenarios(path)
    assert len(loaded) == SCENARIO_MODELING.max_saved_scenarios
    assert loaded[-1].name == f"S{SCENARIO_MODELING.max_saved_scenarios + 9}"  # kept the MOST RECENT, not the oldest
