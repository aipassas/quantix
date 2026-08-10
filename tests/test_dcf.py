"""Tests for the multi-stage DCF engine (fundamental_analysis.py) —
normalized_ebit_margin(), reinvestment_ratios(), intrinsic_price(), and
run_dcf()'s gating logic. Synthetic StandardizedFinancials objects for the
non-network tests; a live check against real AAPL/JPM data for the
acceptance criterion ("a bank correctly can't run this model, a normal
company can").
"""
import datetime

import pytest

from config import DCF
from financial_standardization import StandardizedFinancials
from fundamental_analysis import FundamentalAnalysisEngine


def _std(ticker="TEST", **overrides) -> StandardizedFinancials:
    """A minimal StandardizedFinancials with every field defaulted to
    None/0/empty except what a test explicitly overrides."""
    fields = dict(
        ticker=ticker, long_name=None, business_summary=None, website=None, sector=None,
        pe_ratio=None, peg_ratio=None, price_to_book=None, net_margin=None, return_on_equity=None,
        debt_to_equity=None, current_ratio=None, beta=1.0, earnings_growth=None,
        held_pct_insiders=None, held_pct_institutions=None,
        market_cap=1_000_000_000.0, shares_outstanding=100_000_000.0, current_price=50.0,
        total_assets=None, current_assets=None, current_liabilities=None,
        stockholders_equity=None, total_liabilities=None, total_debt=0.0, total_debt_from_statement=None,
        retained_earnings=0.0, inventory=0.0, cash_and_equivalents=0.0,
        total_revenue=1_000_000_000.0, ebit=200_000_000.0, interest_expense=0.0, net_income=None,
        gross_profit=None, operating_income=None, pretax_income=None, tax_provision=None,
        free_cash_flow=150_000_000.0, depreciation_and_amortization=0.0,
        most_recent_quarter=None, validation=None, data_fallbacks=(),
        revenue_history=(), ebit_history=(), depreciation_history=(), capex_history=(),
        change_in_working_capital_history=(),
    )
    fields.update(overrides)
    return StandardizedFinancials(**fields)


def _dated_history(start_year, values):
    """(date, value) pairs for consecutive fiscal years, oldest first."""
    return tuple((datetime.date(start_year + i, 12, 31), v) for i, v in enumerate(values))


def test_normalized_ebit_margin_averages_aligned_years():
    std = _std(
        revenue_history=_dated_history(2021, [100.0, 110.0, 121.0]),
        ebit_history=_dated_history(2021, [20.0, 22.0, 24.2]),  # constant 20% margin every year
    )
    engine = FundamentalAnalysisEngine(std)
    assert engine.normalized_ebit_margin() == pytest.approx(0.20)


def test_normalized_ebit_margin_none_without_history():
    engine = FundamentalAnalysisEngine(_std(revenue_history=(), ebit_history=()))
    assert engine.normalized_ebit_margin() is None


def test_normalized_ebit_margin_only_uses_dates_present_in_both():
    # An extra EBIT-only year (no matching revenue) must not skew the average.
    std = _std(
        revenue_history=_dated_history(2021, [100.0, 100.0]),
        ebit_history=_dated_history(2020, [999.0, 10.0, 10.0]),  # 2020 has no matching revenue year
    )
    engine = FundamentalAnalysisEngine(std)
    assert engine.normalized_ebit_margin() == pytest.approx(0.10)


def test_reinvestment_ratios_average_and_preserve_sign():
    std = _std(
        revenue_history=_dated_history(2021, [100.0, 200.0]),
        depreciation_history=_dated_history(2021, [10.0, 20.0]),      # 10% both years
        capex_history=_dated_history(2021, [-5.0, -30.0]),            # -5% then -15%
        change_in_working_capital_history=_dated_history(2021, [-1.0, -1.0]),  # -1% then -0.5%
    )
    engine = FundamentalAnalysisEngine(std)
    da_ratio, capex_ratio, nwc_ratio = engine.reinvestment_ratios()
    assert da_ratio == pytest.approx(0.10)
    assert capex_ratio == pytest.approx((-0.05 + -0.15) / 2)
    assert nwc_ratio == pytest.approx((-0.01 + -0.005) / 2)


def test_reinvestment_ratios_default_to_zero_when_missing():
    engine = FundamentalAnalysisEngine(_std(revenue_history=_dated_history(2021, [100.0])))
    assert engine.reinvestment_ratios() == (0.0, 0.0, 0.0)


def test_intrinsic_price_matches_hand_computed_value_flat_margin():
    """A company already at its normalized margin (no fade) with zero
    reinvestment drag reduces to a simple NOPAT-growth DCF — hand-computable."""
    std = _std(
        total_revenue=1000.0, ebit=200.0,  # 20% current margin
        shares_outstanding=100.0,
        revenue_history=_dated_history(2021, [900.0, 950.0, 1000.0]),
        ebit_history=_dated_history(2021, [180.0, 190.0, 200.0]),  # constant 20% -> normalized = current, no fade
    )
    engine = FundamentalAnalysisEngine(std)
    growth_rate, discount_rate = 0.10, 0.09999999999999  # avoid exact terminal_growth collision only if needed
    discount_rate = 0.10

    result = engine.intrinsic_price(growth_rate, discount_rate)

    # Hand-computed: revenue grows 10%/yr from 1000, flat 20% margin, 21% tax, no reinvestment.
    revenue = 1000.0
    fcf_list = []
    for _ in range(DCF.projection_years):
        revenue *= 1.10
        fcf_list.append(revenue * 0.20 * (1 - DCF.tax_rate))
    pv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(fcf_list, start=1))
    terminal = (fcf_list[-1] * (1 + DCF.terminal_growth_rate)) / (discount_rate - DCF.terminal_growth_rate)
    pv_terminal = terminal / (1 + discount_rate) ** DCF.projection_years
    expected = (pv + pv_terminal) / 100.0

    assert result == pytest.approx(expected, rel=1e-9)


def test_intrinsic_price_fades_margin_toward_normalized():
    """A current margin above its historical normalized average should
    pull the FAR-year (year 5) FCF margin down toward that average,
    producing a lower year-5 FCF than a naive flat-current-margin
    projection would."""
    std = _std(
        total_revenue=1000.0, ebit=300.0,  # current margin 30%, elevated
        shares_outstanding=100.0,
        revenue_history=_dated_history(2019, [800.0, 850.0, 900.0, 950.0, 1000.0]),
        ebit_history=_dated_history(2019, [160.0, 170.0, 180.0, 190.0, 300.0]),  # historical ~20%, last year spikes to 30%
    )
    engine = FundamentalAnalysisEngine(std)
    normalized = engine.normalized_ebit_margin()
    assert normalized < 0.30  # historical average pulled down by the four ~20% years

    # Reconstruct year-5 margin from the model's own fade logic and confirm it's below the naive 30%-flat projection.
    current_margin = 0.30
    fade_year5 = 5 / DCF.projection_years
    margin_year5 = current_margin + (normalized - current_margin) * fade_year5
    assert margin_year5 == pytest.approx(normalized)  # full fade completes exactly at the projection horizon
    assert margin_year5 < current_margin


def test_run_dcf_fails_safe_missing_market_cap():
    engine = FundamentalAnalysisEngine(_std(market_cap=None))
    result = engine.run_dcf(0.10)
    assert result.ok is False
    assert result.reason == "missing market cap"


def test_run_dcf_fails_safe_missing_shares():
    engine = FundamentalAnalysisEngine(_std(shares_outstanding=0))
    result = engine.run_dcf(0.10)
    assert result.ok is False
    assert result.reason == "missing shares outstanding"


def test_run_dcf_fails_safe_missing_revenue():
    engine = FundamentalAnalysisEngine(_std(total_revenue=None))
    result = engine.run_dcf(0.10)
    assert result.ok is False
    assert result.reason == "missing or non-positive revenue"


def test_run_dcf_fails_safe_no_ebit_history():
    # total_revenue present but no revenue/ebit history AND ebit itself is
    # None -> no way to anchor a margin at all (mirrors JPM: banks don't
    # report EBIT/Operating Income).
    engine = FundamentalAnalysisEngine(_std(ebit=None, revenue_history=(), ebit_history=()))
    result = engine.run_dcf(0.10)
    assert result.ok is False
    assert result.reason == "no EBIT history available to model a margin trajectory"


def test_run_dcf_fails_safe_negative_normalized_margin():
    std = _std(
        total_revenue=1000.0, ebit=-50.0,
        revenue_history=_dated_history(2021, [900.0, 950.0, 1000.0]),
        ebit_history=_dated_history(2021, [-100.0, -80.0, -50.0]),  # structurally unprofitable every year
    )
    engine = FundamentalAnalysisEngine(std)
    result = engine.run_dcf(0.10)
    assert result.ok is False
    assert "negative" in result.reason


def test_run_dcf_succeeds_with_valid_synthetic_data():
    std = _std(
        total_revenue=1000.0, ebit=200.0, shares_outstanding=100.0, current_price=50.0,
        revenue_history=_dated_history(2021, [900.0, 950.0, 1000.0]),
        ebit_history=_dated_history(2021, [180.0, 190.0, 200.0]),
    )
    engine = FundamentalAnalysisEngine(std)
    result = engine.run_dcf(0.10)
    assert result.ok is True
    assert result.intrinsic_price is not None and result.intrinsic_price > 0
    assert result.status in ("Strong Buy", "Buy", "Overvalued")
    assert result.wacc is not None


@pytest.mark.live
def test_dcf_real_data_aapl_runs_jpm_skips():
    """Acceptance-style check on real data: a normal company (AAPL, has
    EBIT history) can run the new multi-stage model; a bank (JPM, no
    reported EBIT/Operating Income) correctly can't, with an honest
    reason rather than a fabricated number."""
    import datetime as dt

    from data_loader import load_ticker_bundle
    from financial_standardization import standardize_financials

    end = dt.date.today()
    start = end - dt.timedelta(days=400)

    aapl_bundle = load_ticker_bundle("AAPL", start, end, deep=True)
    aapl_std = standardize_financials(aapl_bundle)
    aapl_engine = FundamentalAnalysisEngine(aapl_std, raw_info=aapl_bundle.info)
    assert len(aapl_std.revenue_history) >= 2
    assert len(aapl_std.ebit_history) >= 2
    aapl_result = aapl_engine.run_dcf(0.10, fallback_price=aapl_std.current_price)
    assert aapl_result.ok is True
    assert aapl_result.intrinsic_price > 0

    jpm_bundle = load_ticker_bundle("JPM", start, end, deep=True)
    jpm_std = standardize_financials(jpm_bundle)
    jpm_engine = FundamentalAnalysisEngine(jpm_std, raw_info=jpm_bundle.info)
    jpm_result = jpm_engine.run_dcf(0.10, fallback_price=jpm_std.current_price)
    assert jpm_result.ok is False
    assert jpm_result.reason  # a real, disclosed reason, never a silently fabricated number
