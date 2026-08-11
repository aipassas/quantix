"""Tests for portfolio_analytics.py — correlation, covariance, and
portfolio diversification across a multi-ticker basket."""
import numpy as np
import pandas as pd
import pytest

from portfolio_analytics import (
    build_aligned_returns, compute_correlation_matrix, compute_covariance_matrix,
    compute_portfolio_diversification, compute_capm_beta, compute_performance_attribution,
    compute_min_variance_portfolio, compute_max_sharpe_portfolio, compute_efficient_frontier,
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


# --- Markowitz optimization (compute_min_variance_portfolio / compute_max_sharpe_portfolio / compute_efficient_frontier) ---

def test_min_variance_matches_two_asset_closed_form_solution(four_ticker_baskets):
    """A and C are exact opposites (C = -A's returns) with equal variance —
    the textbook 2-asset min-variance weight is exactly 0.5/0.5, and the
    resulting PORTFOLIO variance is exactly zero (perfectly offsetting).
    Cross-checked against the closed-form formula
    w1* = (var2 - cov12) / (var1 + var2 - 2*cov12), not just plausibility."""
    baskets = {"A": four_ticker_baskets["A"], "C": four_ticker_baskets["C"]}
    alignment = build_aligned_returns(baskets)
    cov = compute_covariance_matrix(alignment.returns)
    var_a, var_c, cov_ac = cov.loc["A", "A"], cov.loc["C", "C"], cov.loc["A", "C"]
    expected_w_a = (var_c - cov_ac) / (var_a + var_c - 2 * cov_ac)

    result = compute_min_variance_portfolio(alignment.returns)

    assert result is not None
    assert result.weights["A"] == pytest.approx(expected_w_a, abs=1e-4)
    assert result.weights["A"] == pytest.approx(0.5, abs=1e-3)
    assert result.volatility == pytest.approx(0.0, abs=1e-6)
    assert result.sharpe_ratio is None  # undefined at ~zero volatility, never fabricated


def test_min_variance_weights_are_long_only_and_sum_to_one(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    result = compute_min_variance_portfolio(alignment.returns)
    assert result is not None
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert all(w >= -1e-9 for w in result.weights.values())


def test_max_sharpe_weights_are_long_only_and_sum_to_one(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    result = compute_max_sharpe_portfolio(alignment.returns)
    assert result is not None
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert all(w >= -1e-9 for w in result.weights.values())


def test_max_sharpe_portfolio_sharpe_at_least_as_good_as_equal_weighted(four_ticker_baskets):
    """The max-Sharpe portfolio is optimal OVER the same feasible set
    (long-only, weights-sum-to-1) the equal-weighted portfolio belongs to
    — it can never do worse."""
    alignment = build_aligned_returns(four_ticker_baskets)
    max_sharpe = compute_max_sharpe_portfolio(alignment.returns)
    n = alignment.returns.shape[1]
    equal_weights = {t: 1.0 / n for t in alignment.returns.columns}

    cov = compute_covariance_matrix(alignment.returns).to_numpy()
    mean_returns = alignment.returns.mean().to_numpy() * RISK.trading_days_per_year
    w_eq = np.full(n, 1.0 / n)
    eq_return = float(w_eq @ mean_returns)
    eq_vol = float(np.sqrt(w_eq @ cov @ w_eq))
    eq_sharpe = (eq_return - RISK.risk_free_rate) / eq_vol

    assert max_sharpe.sharpe_ratio >= eq_sharpe - 1e-9


def test_min_variance_volatility_no_higher_than_equal_weighted(four_ticker_baskets):
    """The min-variance portfolio is optimal for volatility over the SAME
    feasible set the equal-weighted portfolio belongs to."""
    alignment = build_aligned_returns(four_ticker_baskets)
    min_var = compute_min_variance_portfolio(alignment.returns)
    diversification = compute_portfolio_diversification(alignment.returns)  # the equal-weighted portfolio
    assert min_var.volatility <= diversification.portfolio_volatility + 1e-9


def test_none_on_single_ticker():
    single = pd.DataFrame({"A": np.random.default_rng(1).normal(0, 0.01, 100)})
    assert compute_min_variance_portfolio(single) is None
    assert compute_max_sharpe_portfolio(single) is None
    assert compute_efficient_frontier(single) is None


def test_allow_short_permits_weights_outside_long_only_bounds():
    """A well-conditioned (not exactly singular) 2-asset case, deliberately
    constructed via the closed-form 2-asset min-variance formula so the true
    UNCONSTRAINED optimum requires w_X > 1 / w_Y < 0 (a genuine short in Y):
    var_X=1, var_Y=4, cov_XY=1.9 (corr=0.95) -> w_X* = (4-1.9)/(1+4-2*1.9) ≈ 1.79.
    long-only must clamp to the boundary (w_X=1, w_Y=0); allow_short must
    reach the true optimum with a real negative weight and strictly lower
    volatility, since it optimizes over a strictly larger feasible set."""
    rng = np.random.default_rng(21)
    cov = np.array([[1.0, 1.9], [1.9, 4.0]])
    daily_returns = rng.multivariate_normal([0.0, 0.0], cov / 252, size=500)
    returns_df = pd.DataFrame(daily_returns, columns=["X", "Y"])

    long_only = compute_min_variance_portfolio(returns_df, allow_short=False)
    shorting = compute_min_variance_portfolio(returns_df, allow_short=True)

    assert long_only is not None and shorting is not None
    assert all(-1e-6 <= w <= 1 + 1e-6 for w in long_only.weights.values())
    assert shorting.weights["Y"] < -0.1, "the unconstrained optimum should genuinely short Y"
    assert shorting.volatility < long_only.volatility


def test_efficient_frontier_min_variance_endpoint_matches_min_variance_portfolio(four_ticker_baskets):
    alignment = build_aligned_returns(four_ticker_baskets)
    frontier = compute_efficient_frontier(alignment.returns, num_points=15)
    min_var = compute_min_variance_portfolio(alignment.returns)

    assert frontier is not None
    assert frontier.frontier_returns[0] == pytest.approx(min_var.expected_return, abs=1e-6)
    assert frontier.frontier_volatilities[0] == pytest.approx(min_var.volatility, abs=1e-3)


def test_efficient_frontier_volatility_non_decreasing_along_returns(four_ticker_baskets):
    """Starting from the global minimum-variance point and requiring
    progressively higher target returns, the minimum achievable volatility
    for each target must be monotonically non-decreasing — the defining
    shape of the upper (efficient) half of the frontier."""
    alignment = build_aligned_returns(four_ticker_baskets)
    frontier = compute_efficient_frontier(alignment.returns, num_points=20)
    assert frontier is not None
    vols = frontier.frontier_volatilities
    assert all(vols[i] <= vols[i + 1] + 1e-6 for i in range(len(vols) - 1))


def test_efficient_frontier_equal_weighted_matches_diversification_section(four_ticker_baskets):
    """The frontier's equal_weighted reference point must be the exact
    same portfolio the existing Portfolio Diversification section already
    evaluates — same aligned returns, same annualized covariance matrix."""
    alignment = build_aligned_returns(four_ticker_baskets)
    frontier = compute_efficient_frontier(alignment.returns)
    diversification = compute_portfolio_diversification(alignment.returns)
    assert frontier.equal_weighted.volatility == pytest.approx(diversification.portfolio_volatility, abs=1e-9)
