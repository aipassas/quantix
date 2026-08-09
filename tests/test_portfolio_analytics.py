"""Tests for portfolio_analytics.py — correlation, covariance, and
portfolio diversification across a multi-ticker basket."""
import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import (
    build_aligned_returns, compute_correlation_matrix, compute_covariance_matrix,
    compute_portfolio_diversification,
)


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
