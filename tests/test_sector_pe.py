"""Tests for sector-adjusted P/E Scorecard thresholds — SCORECARD.pe_range_for()
(config.py) and its use in FundamentalAnalysisEngine._build_checks().

PEG is deliberately unaffected by sector (see pe_range_for()'s docstring),
covered here too so that stays true if someone touches it later.
"""
import pytest

from config import SCORECARD
from financial_standardization import StandardizedFinancials
from fundamental_analysis import FundamentalAnalysisEngine


def _std(ticker="TEST", **overrides) -> StandardizedFinancials:
    fields = dict(
        ticker=ticker, long_name=None, business_summary=None, website=None, sector=None,
        pe_ratio=None, peg_ratio=None, price_to_book=None, net_margin=0.15, return_on_equity=None,
        debt_to_equity=1.0, current_ratio=1.5, beta=1.0, earnings_growth=None,
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


def _pe_check(sector, pe):
    engine = FundamentalAnalysisEngine(_std(sector=sector, pe_ratio=pe))
    metrics = engine.analyze()
    return next(c for c in metrics.checks if c.key == "pe_ratio")


# --- SCORECARD.pe_range_for() ---------------------------------------------

def test_pe_range_for_configured_sector_returns_override():
    assert SCORECARD.pe_range_for("Technology") == (15.0, 65.0)
    assert SCORECARD.pe_range_for("Utilities") == (12.0, 22.0)


def test_pe_range_for_unconfigured_sector_falls_back_to_global():
    assert SCORECARD.pe_range_for("Industrials") == SCORECARD.pe_range
    assert SCORECARD.pe_range_for(None) == SCORECARD.pe_range
    assert SCORECARD.pe_range_for("Some Sector Yahoo Invents Tomorrow") == SCORECARD.pe_range


def test_pe_range_for_both_financials_spellings_match():
    assert SCORECARD.pe_range_for("Financial Services") == SCORECARD.pe_range_for("Financials")


# --- _build_checks() integration -------------------------------------------

def test_high_growth_tech_pe_passes_under_sector_band_fails_under_global_band():
    """The acceptance-criteria spot check: PE 55 is a normal, healthy
    multiple for a high-growth tech name but would fail the old flat
    (10, 45) band."""
    check = _pe_check("Technology", 55.0)
    assert check.passed is True
    assert 55.0 > SCORECARD.pe_range[1]  # confirms the global band alone would have failed it


def test_expensive_utility_fails_under_sector_band_passes_under_global_band():
    """The mirror-image case: PE 42 comfortably clears the old flat (10, 45)
    band, but is expensive relative to Utilities' typical bond-proxy
    multiple — the sector band catches what the flat band missed."""
    check = _pe_check("Utilities", 42.0)
    assert check.passed is False
    assert SCORECARD.pe_range[0] <= 42.0 <= SCORECARD.pe_range[1]  # confirms the global band alone would have passed it


def test_unconfigured_sector_uses_global_band_unchanged():
    check = _pe_check("Industrials", 50.0)
    assert check.passed is False  # 50 > global upper bound of 45
    assert "sector-adjusted" not in check.benchmark


def test_configured_sector_benchmark_display_flags_sector_adjustment():
    check = _pe_check("Technology", 55.0)
    assert "sector-adjusted" in check.benchmark
    assert "15" in check.benchmark and "65" in check.benchmark


def test_missing_pe_check_is_none_not_a_crash():
    check = _pe_check("Technology", None)
    assert check.passed is None


# --- PEG stays sector-blind --------------------------------------------------

def test_peg_check_identical_across_sectors():
    tech = FundamentalAnalysisEngine(_std(sector="Technology", peg_ratio=1.8)).analyze()
    util = FundamentalAnalysisEngine(_std(sector="Utilities", peg_ratio=1.8)).analyze()
    tech_peg = next(c for c in tech.checks if c.key == "peg_ratio")
    util_peg = next(c for c in util.checks if c.key == "peg_ratio")
    assert tech_peg.passed == util_peg.passed == True
    assert tech_peg.benchmark == util_peg.benchmark


# --- Debt-to-Equity's existing sector adjustment is untouched --------------

def test_debt_to_equity_sector_adjustment_still_works():
    check = FundamentalAnalysisEngine(
        _std(sector="Financial Services", debt_to_equity=3.0)
    ).analyze()
    de_check = next(c for c in check.checks if c.key == "debt_to_equity")
    assert de_check.passed is True  # 3.0 < financials_max_debt_to_equity (4.0), would fail the generic 2.5 threshold
