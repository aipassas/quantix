"""Tests for risk_analytics.py — historical volatility, VaR/CVaR, Maximum
Drawdown, Sharpe/Sortino/Calmar, and the composite Risk Score.

Cross-validated against manual numpy/scipy computations and hand-worked
cases with known closed-form answers, the same approach used ad-hoc
throughout this module's development (see RISK_ANALYTICS.md).
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from risk_analytics import (
    compute_log_returns, compute_rolling_volatility, compute_annualized_volatility,
    compute_annualized_return, compute_sharpe_ratio, interpret_sharpe_ratio,
    compute_downside_deviation, compute_sortino_ratio,
    compute_historical_var, compute_parametric_var, compute_expected_shortfall,
    interpret_tail_risk,
    compute_drawdown_series, compute_max_drawdown,
    compute_calmar_ratio, interpret_calmar_ratio,
    compute_risk_score,
    compute_hurst_exponent,
)


# --- Log returns / volatility ---

def test_log_returns_first_bar_is_nan(clean_ohlcv):
    log_returns = compute_log_returns(clean_ohlcv)
    assert pd.isna(log_returns.iloc[0])
    assert log_returns.iloc[1:].notna().all()


def test_annualized_volatility_matches_manual(clean_ohlcv):
    engine = compute_annualized_volatility(clean_ohlcv)
    manual_returns = compute_log_returns(clean_ohlcv).dropna()
    manual = manual_returns.std(ddof=1) * np.sqrt(252)
    assert engine == pytest.approx(manual, abs=1e-10)


def test_rolling_volatility_warmup_is_nan(clean_ohlcv):
    rolling = compute_rolling_volatility(clean_ohlcv, window=21)
    assert rolling.iloc[:21].isna().all()
    assert rolling.iloc[21:].notna().all()


def test_annualized_volatility_none_on_insufficient_data():
    tiny = pd.DataFrame({"Close": [100.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    assert compute_annualized_volatility(tiny) is None


# --- Annualized return / Sharpe ---

def test_annualized_return_matches_log_telescoping_identity(clean_ohlcv):
    """Sum of daily log returns telescopes exactly to ln(P_end/P_start) —
    this identity is why mean(log returns) * trading_days is a clean CAGR
    approximation rather than an ad-hoc one."""
    engine = compute_annualized_return(clean_ohlcv)
    log_returns = compute_log_returns(clean_ohlcv).dropna()
    total_log_return = np.log(clean_ohlcv["Close"].iloc[-1] / clean_ohlcv["Close"].iloc[0])
    expected = total_log_return / len(log_returns) * 252
    assert engine == pytest.approx(expected, abs=1e-9)


def test_sharpe_ratio_matches_manual(clean_ohlcv):
    for rf in (0.0, 0.02, 0.04):
        engine = compute_sharpe_ratio(clean_ohlcv, risk_free_rate=rf)
        log_returns = compute_log_returns(clean_ohlcv).dropna()
        manual = (log_returns.mean() * 252 - rf) / (log_returns.std(ddof=1) * np.sqrt(252))
        assert engine == pytest.approx(manual, abs=1e-10)


def test_sharpe_ratio_none_on_zero_volatility():
    flat = pd.DataFrame({"Close": [100.0] * 30}, index=pd.bdate_range("2024-01-01", periods=30))
    assert compute_sharpe_ratio(flat, risk_free_rate=0.04) is None


def test_interpret_sharpe_ratio_bands():
    assert interpret_sharpe_ratio(-0.5).label.endswith("Poor")
    assert interpret_sharpe_ratio(0.5).label.endswith("Sub-optimal")
    assert "Very" not in interpret_sharpe_ratio(1.5).label
    assert interpret_sharpe_ratio(2.5).label.endswith("Very Good")
    assert "Exceptional" in interpret_sharpe_ratio(4.0).label
    assert interpret_sharpe_ratio(None) is None


# --- Sortino ---

def test_downside_deviation_matches_textbook_5point_case():
    """Known case: returns [+2%, +2%, -1%, -3%, +1%].
    Downside deviation = sqrt(mean(min(r,0)^2)) = sqrt((0+0+0.0001+0.0009+0)/5)."""
    returns = [0.02, 0.02, -0.01, -0.03, 0.01]
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * np.exp(r))
    df = pd.DataFrame({"Close": prices}, index=pd.bdate_range("2024-01-01", periods=len(prices)))

    engine = compute_downside_deviation(df, trading_days_per_year=252)
    expected_daily = np.sqrt((0 + 0 + 0.01**2 + 0.03**2 + 0) / 5)
    expected = expected_daily * np.sqrt(252)
    assert engine == pytest.approx(expected, abs=1e-9)


def test_sortino_exceeds_sharpe_on_positively_skewed_series():
    """The whole point of Sortino: rare large upside jumps shouldn't be
    penalized as 'risk' the way Sharpe's total-volatility denominator does."""
    rng = np.random.default_rng(9)
    n = 400
    log_rets = rng.normal(0.0004, 0.010, n) + (rng.random(n) < 0.03) * 0.04
    prices = 100 * np.exp(np.cumsum(log_rets))
    df = pd.DataFrame({"Close": prices}, index=pd.bdate_range("2023-01-02", periods=n))

    sharpe = compute_sharpe_ratio(df, risk_free_rate=0.04)
    sortino = compute_sortino_ratio(df, risk_free_rate=0.04)
    assert sortino > sharpe


def test_sortino_none_on_all_positive_returns():
    prices = pd.Series([100, 101, 102, 103, 104, 105], index=pd.bdate_range("2024-01-01", periods=6))
    df = pd.DataFrame({"Close": prices})
    assert compute_downside_deviation(df) == 0.0
    assert compute_sortino_ratio(df, risk_free_rate=0.04) is None


# --- VaR / CVaR ---

def test_historical_var_matches_manual_percentile(clean_ohlcv):
    for confidence in (0.90, 0.95, 0.99):
        engine = compute_historical_var(clean_ohlcv, confidence, lookback=252)
        log_returns = compute_log_returns(clean_ohlcv).dropna().tail(252)
        manual = np.percentile(log_returns, (1 - confidence) * 100, method="linear")
        assert engine == pytest.approx(manual, abs=1e-10)


def test_parametric_var_matches_manual_normal_fit(clean_ohlcv):
    for confidence in (0.90, 0.95, 0.99):
        engine = compute_parametric_var(clean_ohlcv, confidence, lookback=252)
        log_returns = compute_log_returns(clean_ohlcv).dropna().tail(252)
        mu, sigma = log_returns.mean(), log_returns.std()
        manual = mu + norm.ppf(1 - confidence) * sigma
        assert engine == pytest.approx(manual, abs=1e-10)


def test_var_none_below_min_observations(clean_ohlcv):
    tiny = clean_ohlcv.iloc[:10]
    assert compute_historical_var(tiny, 0.95) is None
    assert compute_parametric_var(tiny, 0.95) is None


def test_expected_shortfall_matches_manual_tail_mean(clean_ohlcv):
    for confidence in (0.90, 0.95, 0.99):
        engine = compute_expected_shortfall(clean_ohlcv, confidence, lookback=252)
        log_returns = compute_log_returns(clean_ohlcv).dropna().tail(252)
        var_threshold = log_returns.quantile(1 - confidence)
        manual = log_returns[log_returns <= var_threshold].mean()
        assert engine == pytest.approx(manual, abs=1e-10)


def test_expected_shortfall_never_less_extreme_than_var(clean_ohlcv):
    """CVaR must always be further into the loss tail than VaR by definition."""
    for confidence in (0.90, 0.95, 0.99):
        var = compute_historical_var(clean_ohlcv, confidence, lookback=252)
        cvar = compute_expected_shortfall(clean_ohlcv, confidence, lookback=252)
        assert cvar <= var


def test_interpret_tail_risk_none_on_missing_input():
    assert interpret_tail_risk(None, -0.03, 0.95) is None
    text = interpret_tail_risk(-0.02, -0.035, 0.95)
    assert "1.50" in text  # (3.5% - 2.0%) = 1.5pp


# --- Maximum Drawdown ---

def test_max_drawdown_hand_worked_case():
    """Prices: 100, 110, 120, 90, 80, 100, 130, 125.
    Peak=120, Trough=80 -> drawdown = (80-120)/120 = -1/3. Recovers at 130 (2 trading days later)."""
    dates = pd.bdate_range("2024-01-01", periods=8)
    prices = pd.Series([100, 110, 120, 90, 80, 100, 130, 125], index=dates)

    result = compute_max_drawdown(prices)
    assert result.max_drawdown == pytest.approx(-1 / 3, abs=1e-9)
    assert result.peak_date == dates[2] and result.peak_price == 120
    assert result.trough_date == dates[4] and result.trough_price == 80
    assert result.recovered is True
    assert result.recovery_date == dates[6]
    assert result.recovery_days == 2


def test_max_drawdown_monotonic_increase_is_zero():
    prices = pd.Series([100, 105, 110, 120, 130], index=pd.bdate_range("2024-02-01", periods=5))
    result = compute_max_drawdown(prices)
    assert result.max_drawdown == 0.0
    assert result.recovered is True
    assert result.recovery_days == 0


def test_max_drawdown_never_recovered():
    prices = pd.Series([100, 120, 80, 90, 95], index=pd.bdate_range("2024-03-01", periods=5))
    result = compute_max_drawdown(prices)
    assert result.recovered is False
    assert result.recovery_date is None
    assert result.recovery_days is None


def test_max_drawdown_scale_invariant(clean_ohlcv):
    prices = clean_ohlcv["Close"]
    returns = prices.pct_change()
    cum_growth = (1 + returns).cumprod()
    cum_growth.iloc[0] = 1.0

    price_result = compute_max_drawdown(prices)
    growth_result = compute_max_drawdown(cum_growth)
    assert price_result.max_drawdown == pytest.approx(growth_result.max_drawdown, abs=1e-9)


# --- Calmar ---

def test_calmar_ratio_matches_manual_composition(clean_ohlcv):
    engine = compute_calmar_ratio(clean_ohlcv)
    annual_return = compute_annualized_return(clean_ohlcv)
    dd = compute_max_drawdown(clean_ohlcv["Close"])
    manual = annual_return / abs(dd.max_drawdown)
    assert engine == pytest.approx(manual, abs=1e-9)


def test_calmar_ratio_none_on_zero_drawdown():
    prices = pd.DataFrame({"Close": [100, 105, 110, 120, 130]}, index=pd.bdate_range("2024-02-01", periods=5))
    assert compute_calmar_ratio(prices) is None


def test_interpret_calmar_ratio_bands():
    assert interpret_calmar_ratio(-0.5).label.endswith("Poor")
    assert interpret_calmar_ratio(6.0) is not None and "Excellent" in interpret_calmar_ratio(6.0).label


# --- Composite Risk Score ---

def test_risk_score_renormalizes_when_a_factor_is_excluded():
    """Excluding a factor must drop its weight from BOTH the numerator and
    denominator — not zero it into the numerator while keeping the weight."""
    full = compute_risk_score(
        rolling_volatility=0.15, historical_var=-0.08, expected_shortfall=-0.06,
        max_drawdown=-0.30, sharpe_ratio=1.25, sortino_ratio=2.5, calmar_ratio=0.0,
        altman_z=2.99,  # scores 100
    )
    excluded = compute_risk_score(
        rolling_volatility=0.15, historical_var=-0.08, expected_shortfall=-0.06,
        max_drawdown=-0.30, sharpe_ratio=1.25, sortino_ratio=2.5, calmar_ratio=0.0,
        altman_z=None,
    )
    assert excluded.excluded_factors == ["Altman Z-Score"]
    # Manual recomputation over the remaining 7 factors:
    remaining = [f for f in full.factors if f.label != "Altman Z-Score"]
    manual_score = sum(f.sub_score * f.weight for f in remaining) / sum(f.weight for f in remaining)
    assert excluded.score == pytest.approx(round(manual_score, 1), abs=1e-6)


def test_risk_score_clamps_beyond_anchors():
    result = compute_risk_score(
        rolling_volatility=0.01,   # far better than the best anchor (0.15) -> still 100
        historical_var=-0.50,      # far worse than the worst anchor (-0.08) -> still 0
        expected_shortfall=None, max_drawdown=None, sharpe_ratio=None,
        sortino_ratio=None, calmar_ratio=None, altman_z=None,
    )
    vol_factor = next(f for f in result.factors if f.label == "Annualized Volatility")
    var_factor = next(f for f in result.factors if f.label == "1-Day Historical VaR")
    assert vol_factor.sub_score == 100.0
    assert var_factor.sub_score == 0.0


def test_risk_score_all_missing_is_zero_no_crash():
    result = compute_risk_score(None, None, None, None, None, None, None, None)
    assert result.score == 0.0
    assert len(result.excluded_factors) == 8


# --- Hurst Exponent (classical R/S analysis) --------------------------------

def _price_df_from_returns(log_returns):
    """A minimal Close-only DataFrame from a log-return series — all
    compute_hurst_exponent() reads is df['Close'], via compute_log_returns()."""
    close = 100 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame({"Close": close})


def test_hurst_random_walk_reads_near_half():
    """A pure random walk's returns are i.i.d. — no memory to detect.
    Averaged over several seeds since any one sample is noisy; the
    classical R/S statistic has a known small positive bias at finite
    sample sizes, so the tolerance band is centered slightly above 0.5,
    not exactly on it (documented in compute_hurst_exponent()'s docstring)."""
    values = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        returns = rng.normal(0, 1, 250)
        h = compute_hurst_exponent(_price_df_from_returns(returns))
        assert h is not None
        values.append(h)
    assert 0.45 < np.mean(values) < 0.65


def test_hurst_trending_series_reads_notably_above_random_walk():
    """AR(1) with a strong positive coefficient (phi=0.6) injects genuine
    positive autocorrelation — persistent/trending behavior — into the
    return series. Compared directly against a random walk on the SAME
    seeds so the comparison isn't confounded by sample noise."""
    trending, random_walk = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, 250)
        ar_returns = np.zeros(250)
        for t in range(1, 250):
            ar_returns[t] = 0.6 * ar_returns[t - 1] + noise[t]
        trending.append(compute_hurst_exponent(_price_df_from_returns(ar_returns)))
        random_walk.append(compute_hurst_exponent(_price_df_from_returns(noise)))
    assert np.mean(trending) > np.mean(random_walk) + 0.1
    assert np.mean(trending) > 0.6


def test_hurst_mean_reverting_series_reads_notably_below_random_walk():
    """AR(1) with a strong negative coefficient (phi=-0.5) injects genuine
    anti-persistence — mean-reverting behavior."""
    mean_reverting, random_walk = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, 250)
        ar_returns = np.zeros(250)
        for t in range(1, 250):
            ar_returns[t] = -0.5 * ar_returns[t - 1] + noise[t]
        mean_reverting.append(compute_hurst_exponent(_price_df_from_returns(ar_returns)))
        random_walk.append(compute_hurst_exponent(_price_df_from_returns(noise)))
    assert np.mean(mean_reverting) < np.mean(random_walk) - 0.05
    assert np.mean(mean_reverting) < 0.5


def test_hurst_operates_on_returns_not_raw_price_levels():
    """A random-walk PRICE series is itself a cumulative sum (non-stationary)
    — running R/S directly on price levels would read H near 1.0, not the
    expected ~0.5, for a genuinely memoryless process. This is the specific
    pitfall the implementation must avoid."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0, 1, 500)
    df = _price_df_from_returns(returns)
    h = compute_hurst_exponent(df)
    assert h < 0.75, "H should reflect the memoryless RETURN process (~0.5), not the price level's own random-walk integration (~1.0)"


def test_hurst_none_on_insufficient_history():
    df = _price_df_from_returns(np.random.default_rng(1).normal(0, 1, 15))
    assert compute_hurst_exponent(df, min_window=10) is None


def test_hurst_invariant_to_constant_price_rescaling():
    """Log returns (and therefore H) are exactly invariant under a constant
    multiplicative rescaling of price: log(k*P_t / k*P_{t-1}) == log(P_t /
    P_{t-1}) — the k cancels. A precise, deterministic cross-check that
    doesn't depend on statistical sampling."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0, 1, 200)
    df = _price_df_from_returns(returns)
    rescaled_df = df.copy()
    rescaled_df["Close"] = df["Close"] * 37.5

    h_original = compute_hurst_exponent(df)
    h_rescaled = compute_hurst_exponent(rescaled_df)
    assert h_original == pytest.approx(h_rescaled, abs=1e-9)


def test_hurst_deterministic_for_identical_input():
    df = _price_df_from_returns(np.random.default_rng(5).normal(0, 1, 150))
    assert compute_hurst_exponent(df) == compute_hurst_exponent(df.copy())
