"""Tests for portfolio_analytics.py — correlation, covariance, and
portfolio diversification across a multi-ticker basket."""
import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import (
    build_aligned_returns, compute_correlation_matrix, compute_covariance_matrix,
    compute_portfolio_diversification, compute_capm_beta, compute_performance_attribution,
)
from config import RISK


@pytest.fixture
def four_ticker_baskets(ohlcv_factory):
    """A: base series. B: identical returns to A (perfectly correlated).
    C: exact negative of A's returns (perfectly anti-correlated).
    D: independent random returns (near-zero correlation)."""
    base = ohlcv_factory(n=400, seed=42, start_price=100.0)
    base_returns = np.log(base["Close"] / base["Close"].shift(1)).dropna().values

    dates = base.index
    b_close = 50 * np.exp(np.cumsum(np.concatenate([[0], base_returns])))
    c_close = 200 * np.exp(np.cumsum(np.concatenate([[0], -base_returns])))
    d = ohlcv_factory(n=400, seed=99, start_price=80.0)
    d = d.set_index(dates)  # align index for the alignment step

    return {
        "A": base,
        "B": pd.DataFrame({"Close": b_close}, index=dates),
        "C": pd.DataFrame({"Close": c_close}, index=dates),
        "D": d,
    }


def test_alignment_includes_all_valid_tickers(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    assert alignment.included_tickers == ["A", "B", "C", "D"]
    assert alignment.excluded_tickers == []
    assert alignment.sufficient_data


def test_correlation_known_relationships(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    corr = compute_correlation_matrix(alignment.returns)
    assert corr.loc["A", "B"] == pytest.approx(1.0, abs=1e-9)
    assert corr.loc["A", "C"] == pytest.approx(-1.0, abs=1e-9)
    assert abs(corr.loc["A", "D"]) < 0.2


def test_covariance_matches_manual(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    engine = compute_covariance_matrix(alignment.returns)
    manual = alignment.returns.cov() * 252
    pd.testing.assert_frame_equal(engine, manual)


def test_diversification_matches_textbook_two_asset_formula(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    two_asset = alignment.returns[["A", "D"]]
    weights = {"A": 0.6, "D": 0.4}

    result = compute_portfolio_diversification(two_asset, weights=weights)

    cov = compute_covariance_matrix(two_asset).values
    w = np.array([0.6, 0.4])
    manual_variance = w[0]**2 * cov[0, 0] + w[1]**2 * cov[1, 1] + 2 * w[0] * w[1] * cov[0, 1]
    manual_vol = np.sqrt(manual_variance)

    assert result.portfolio_volatility == pytest.approx(manual_vol, abs=1e-9)
    assert result.diversification_benefit > 0


def test_diversification_zero_benefit_for_perfectly_correlated_pair(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    perfect_pair = alignment.returns[["A", "B"]]
    result = compute_portfolio_diversification(perfect_pair, weights={"A": 0.5, "B": 0.5})
    assert result.diversification_benefit == pytest.approx(0.0, abs=1e-6)


def test_diversification_weight_normalization_invariant(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    two_asset = alignment.returns[["A", "D"]]
    normalized = compute_portfolio_diversification(two_asset, weights={"A": 0.6, "D": 0.4})
    unnormalized = compute_portfolio_diversification(two_asset, weights={"A": 6.0, "D": 4.0})
    assert normalized.portfolio_volatility == pytest.approx(unnormalized.portfolio_volatility, abs=1e-9)


def test_diversification_none_on_empty_or_single_ticker():
    assert compute_portfolio_diversification(pd.DataFrame()) is None


def test_diversification_none_on_zero_weights(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    two_asset = alignment.returns[["A", "D"]]
    assert compute_portfolio_diversification(two_asset, weights={"A": 0.0, "D": 0.0}) is None


def test_excluded_ticker_disclosed_with_reason(four_ticker_baskets):
    baskets = dict(four_ticker_baskets)
    baskets["EMPTY"] = pd.DataFrame()
    alignment = build_aligned_returns(baskets)
    assert "EMPTY" in alignment.excluded_tickers
    assert "EMPTY" in alignment.exclusion_reasons


# --- compute_capm_beta() ----------------------------------------------------

def test_beta_is_one_for_identical_returns(four_ticker_baskets):
    """B has IDENTICAL log returns to A (just a different starting price) —
    textbook beta of 1.0 and a perfect R²."""
    result = compute_capm_beta(four_ticker_baskets["B"], four_ticker_baskets["A"], min_observations=20)
    assert result is not None
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_beta_is_negative_one_for_inverted_returns(four_ticker_baskets):
    """C is the exact negative of A's log returns — beta of -1.0, still a
    perfect R² (a perfect NEGATIVE linear relationship is still perfect)."""
    result = compute_capm_beta(four_ticker_baskets["C"], four_ticker_baskets["A"], min_observations=20)
    assert result is not None
    assert result.beta == pytest.approx(-1.0, abs=1e-9)
    assert result.r_squared == pytest.approx(1.0, abs=1e-9)


def test_beta_matches_manual_ols_slope(four_ticker_baskets):
    """D is independent of A — cross-check against numpy's own OLS fit
    rather than another hand-derivation of the same formula."""
    result = compute_capm_beta(four_ticker_baskets["D"], four_ticker_baskets["A"], min_observations=20)
    assert result is not None

    ticker_returns = np.log(four_ticker_baskets["D"]["Close"] / four_ticker_baskets["D"]["Close"].shift(1)).dropna()
    bench_returns = np.log(four_ticker_baskets["A"]["Close"] / four_ticker_baskets["A"]["Close"].shift(1)).dropna()
    aligned = pd.DataFrame({"t": ticker_returns, "b": bench_returns}).dropna()
    manual_beta = np.polyfit(aligned["b"], aligned["t"], 1)[0]

    assert result.beta == pytest.approx(manual_beta, rel=1e-9)


def test_beta_none_when_insufficient_observations(four_ticker_baskets):
    result = compute_capm_beta(
        four_ticker_baskets["A"].iloc[:10], four_ticker_baskets["B"].iloc[:10], min_observations=60,
    )
    assert result is None


def test_beta_none_when_benchmark_has_zero_variance(four_ticker_baskets):
    flat = four_ticker_baskets["A"].copy()
    flat["Close"] = 100.0  # constant price -> zero-variance returns
    result = compute_capm_beta(four_ticker_baskets["A"], flat, min_observations=20)
    assert result is None


def test_beta_observation_count_reflects_actual_overlap(four_ticker_baskets):
    result = compute_capm_beta(four_ticker_baskets["A"], four_ticker_baskets["B"], min_observations=20)
    assert result.observation_count == len(four_ticker_baskets["A"]) - 1  # one lost to the first day's pct_change/log-return NaN


def test_beta_handles_tz_naive_vs_tz_aware_index_mismatch(four_ticker_baskets):
    """finance.py's ticker df is tz-stripped by price_processing.py; a
    benchmark df fetched straight from data_loader.py's macro bundle isn't
    guaranteed to be — pandas raises constructing a DataFrame from two
    Series with mismatched tz-awareness, so this must not crash."""
    ticker_df = four_ticker_baskets["A"]
    benchmark_df = four_ticker_baskets["B"].copy()
    benchmark_df.index = benchmark_df.index.tz_localize("UTC")

    result = compute_capm_beta(ticker_df, benchmark_df, min_observations=20)

    assert result is not None
    naive_result = compute_capm_beta(four_ticker_baskets["A"], four_ticker_baskets["B"], min_observations=20)
    assert result.beta == pytest.approx(naive_result.beta)


# --- compute_performance_attribution() --------------------------------------

def test_attribution_systematic_plus_selection_reconstructs_total_excess_return():
    """The acceptance criterion itself: Systematic + Selection must sum
    back to Total Excess Return, for arbitrary inputs — guaranteed by
    construction (Selection is defined as the residual), not approximated."""
    for ticker_ret, bench_ret, beta, days in [
        (25.0, 15.0, 1.3, 365), (-10.0, 5.0, 0.7, 90), (0.0, 0.0, 1.0, 30),
        (50.0, -20.0, -0.5, 730), (8.0, 8.0, 1.0, 180),
    ]:
        result = compute_performance_attribution(ticker_ret, bench_ret, beta, days)
        assert result.systematic_pct + result.selection_pct == pytest.approx(result.total_excess_return_pct, abs=1e-9)


def test_attribution_matches_hand_computed_values():
    result = compute_performance_attribution(
        ticker_return_pct=30.0, benchmark_return_pct=20.0, beta=1.2,
        period_days=365, annual_risk_free_rate=0.04,
    )
    risk_free_pct = ((1.04) ** 1.0 - 1) * 100  # period_days=365 -> period_years=365/365.25, close to 1.0
    period_years = 365 / 365.25
    risk_free_pct = ((1.04) ** period_years - 1) * 100

    expected_total_excess = 30.0 - risk_free_pct
    expected_systematic = 1.2 * (20.0 - risk_free_pct)
    expected_selection = expected_total_excess - expected_systematic

    assert result.risk_free_period_pct == pytest.approx(risk_free_pct)
    assert result.total_excess_return_pct == pytest.approx(expected_total_excess)
    assert result.systematic_pct == pytest.approx(expected_systematic)
    assert result.selection_pct == pytest.approx(expected_selection)


def test_attribution_zero_beta_means_all_selection():
    """No market exposure -> the entire excess return is attributed to
    selection, none to systematic."""
    result = compute_performance_attribution(20.0, 15.0, beta=0.0, period_days=365)
    assert result.systematic_pct == pytest.approx(0.0)
    assert result.selection_pct == pytest.approx(result.total_excess_return_pct)


def test_attribution_pure_market_replication_has_zero_selection():
    """beta=1.0 and the ticker's return exactly equals the benchmark's —
    textbook 'just tracked the index,' no stock-specific skill -> selection
    should be exactly zero."""
    result = compute_performance_attribution(12.0, 12.0, beta=1.0, period_days=180)
    assert result.selection_pct == pytest.approx(0.0, abs=1e-9)


def test_attribution_risk_free_rate_compounds_not_linear():
    """A 2-year period at a 4% annual rate: compounding (1.04^2 - 1 =
    8.16%) must differ from naive linear scaling (2 * 4% = 8.00%) — proves
    the compounding choice is actually being used, not silently reduced to
    a linear approximation."""
    result = compute_performance_attribution(0.0, 0.0, beta=1.0, period_days=730, annual_risk_free_rate=0.04)
    linear_approx_pct = 0.04 * (730 / 365.25) * 100
    assert result.risk_free_period_pct == pytest.approx(((1.04 ** (730 / 365.25)) - 1) * 100)
    assert result.risk_free_period_pct != pytest.approx(linear_approx_pct, abs=1e-6)


def test_attribution_defaults_to_config_risk_free_rate():
    result_explicit = compute_performance_attribution(10.0, 5.0, beta=1.0, period_days=365, annual_risk_free_rate=RISK.risk_free_rate)
    result_default = compute_performance_attribution(10.0, 5.0, beta=1.0, period_days=365)
    assert result_explicit.risk_free_period_pct == pytest.approx(result_default.risk_free_period_pct)


def test_attribution_zero_period_days_gives_zero_risk_free_rate():
    result = compute_performance_attribution(5.0, 3.0, beta=1.0, period_days=0)
    assert result.risk_free_period_pct == pytest.approx(0.0)
    assert result.total_excess_return_pct == pytest.approx(5.0)
