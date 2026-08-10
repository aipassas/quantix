"""Tests for the CAPM beta fallback chain in fundamental_analysis.py —
beta_estimate(), wacc(), and run_dcf()'s beta/beta_source/beta_r_squared
fields. The regression itself (portfolio_analytics.compute_capm_beta) is
tested in tests/test_portfolio_analytics.py; this covers how
FundamentalAnalysisEngine consumes an already-regressed beta.
"""
import datetime

import pytest

from config import RISK, DCF
from financial_standardization import StandardizedFinancials
from fundamental_analysis import FundamentalAnalysisEngine


def _std(ticker="TEST", **overrides) -> StandardizedFinancials:
    fields = dict(
        ticker=ticker, long_name=None, business_summary=None, website=None, sector=None,
        pe_ratio=None, peg_ratio=None, price_to_book=None, net_margin=0.15, return_on_equity=None,
        debt_to_equity=1.0, current_ratio=1.5, beta=None, earnings_growth=None,
        held_pct_insiders=None, held_pct_institutions=None,
        market_cap=1_000_000_000.0, shares_outstanding=100_000_000.0, current_price=50.0,
        total_assets=None, current_assets=None, current_liabilities=None,
        stockholders_equity=None, total_liabilities=None, total_debt=0.0, total_debt_from_statement=None,
        retained_earnings=0.0, inventory=0.0, cash_and_equivalents=0.0,
        total_revenue=1_000_000_000.0, ebit=200_000_000.0, interest_expense=0.0, net_income=100_000_000.0,
        gross_profit=None, operating_income=None, pretax_income=None, tax_provision=None,
        free_cash_flow=150_000_000.0, depreciation_and_amortization=0.0,
        most_recent_quarter=None, validation=None, data_fallbacks=(),
        revenue_history=(), ebit_history=(), depreciation_history=(), capex_history=(),
        change_in_working_capital_history=(),
    )
    fields.update(overrides)
    return StandardizedFinancials(**fields)


def _dated_history(start_year, values):
    return tuple((datetime.date(start_year + i, 12, 31), v) for i, v in enumerate(values))


# --- beta_estimate() fallback chain -----------------------------------------

def test_regressed_beta_takes_priority_when_supplied():
    engine = FundamentalAnalysisEngine(_std(beta=1.2))
    beta, source = engine.beta_estimate(regressed_beta=1.8)
    assert (beta, source) == (1.8, "regressed")


def test_falls_back_to_yahoo_beta_when_no_regression():
    engine = FundamentalAnalysisEngine(_std(beta=1.2))
    beta, source = engine.beta_estimate(regressed_beta=None)
    assert (beta, source) == (1.2, "yahoo_reported")


def test_falls_back_to_market_assumption_when_nothing_available():
    engine = FundamentalAnalysisEngine(_std(beta=None))
    beta, source = engine.beta_estimate(regressed_beta=None)
    assert (beta, source) == (1.0, "market_assumption")


# --- wacc() actually uses the resolved beta ---------------------------------

def test_wacc_changes_with_regressed_beta():
    engine = FundamentalAnalysisEngine(_std(beta=1.0, total_debt=0.0))
    wacc_default = engine.wacc(regressed_beta=None)
    wacc_high_beta = engine.wacc(regressed_beta=2.0)
    assert wacc_high_beta > wacc_default  # higher beta -> higher cost of equity -> higher WACC


def test_wacc_matches_hand_computed_capm_with_regressed_beta():
    engine = FundamentalAnalysisEngine(_std(beta=1.0, total_debt=0.0, interest_expense=0.0))
    wacc = engine.wacc(regressed_beta=1.5)
    expected_cost_of_equity = RISK.risk_free_rate + 1.5 * (DCF.market_return - RISK.risk_free_rate)
    # No debt -> WACC collapses to pure cost of equity.
    assert wacc == pytest.approx(expected_cost_of_equity)


# --- run_dcf() discloses which beta source was actually used ---------------

def _runnable_std(**overrides):
    return _std(
        total_revenue=1000.0, ebit=200.0, shares_outstanding=100.0, current_price=50.0,
        revenue_history=_dated_history(2021, [900.0, 950.0, 1000.0]),
        ebit_history=_dated_history(2021, [180.0, 190.0, 200.0]),
        **overrides,
    )


def test_run_dcf_reports_regressed_beta_and_r_squared():
    engine = FundamentalAnalysisEngine(_runnable_std(beta=1.0))
    result = engine.run_dcf(0.10, regressed_beta=1.7, beta_r_squared=0.62)
    assert result.ok is True
    assert result.beta == 1.7
    assert result.beta_source == "regressed"
    assert result.beta_r_squared == 0.62


def test_run_dcf_reports_yahoo_beta_when_no_regression_supplied():
    engine = FundamentalAnalysisEngine(_runnable_std(beta=1.3))
    result = engine.run_dcf(0.10)
    assert result.beta == 1.3
    assert result.beta_source == "yahoo_reported"
    assert result.beta_r_squared is None


def test_run_dcf_reports_market_assumption_when_nothing_available():
    engine = FundamentalAnalysisEngine(_runnable_std(beta=None))
    result = engine.run_dcf(0.10)
    assert result.beta == 1.0
    assert result.beta_source == "market_assumption"
    assert result.beta_r_squared is None


def test_run_dcf_never_reports_r_squared_for_a_non_regressed_beta():
    """Even if a caller passes a stale beta_r_squared alongside
    regressed_beta=None (shouldn't happen, but never trust it blindly),
    the result must not attribute an R² to a beta that wasn't regressed."""
    engine = FundamentalAnalysisEngine(_runnable_std(beta=1.1))
    result = engine.run_dcf(0.10, regressed_beta=None, beta_r_squared=0.9)
    assert result.beta_source == "yahoo_reported"
    assert result.beta_r_squared is None
