"""Risk Analytics Engine for Quantix.

Every quantitative risk metric derived from price history — historical
volatility, Historical/Parametric VaR, Expected Shortfall (CVaR), Maximum
Drawdown, Sharpe Ratio, Sortino Ratio, Calmar Ratio, plus a composite Risk
Score synthesizing all of them (and Altman Z, from fundamental_analysis.py)
into one 0-100 figure — is calculated here, in one place, instead of
inline in finance.py. This mirrors technical_indicators.py's role for
chart indicators: finance.py consumes the results and renders them; it
performs no risk arithmetic of its own.

Every function here expects an already-cleaned OHLCV DataFrame — the
output of price_processing.py's process_price_data() — so metrics are
never computed on duplicate timestamps or structurally invalid bars.

Returns convention: this module uses logarithmic returns
(ln(P_t / P_t-1)), not simple returns, matching Bloomberg/TradingView
historical-volatility methodology and giving time-additive returns for
multi-day aggregation.

VaR/CVaR sign convention: a Value at Risk figure is returned as a signed
log return — negative for a loss, e.g. -0.02 for a "5th-percentile day
loses 2%". This matches the sign of the underlying returns rather than
flipping to a positive "loss magnitude", so it can be plotted or compared
against a return series directly without a sign correction.

See RISK_ANALYTICS.md for the full reference: every formula, normalization
anchor the composite Risk Score uses, and how each metric was validated.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import RISK
from user_thresholds import effective_risk


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Day-over-day log returns of Close. First observation is NaN (no prior bar)."""
    return np.log(df["Close"] / df["Close"].shift(1))


def compute_rolling_volatility(
    df: pd.DataFrame,
    window: int,
    trading_days_per_year: Optional[int] = None,
) -> pd.Series:
    """Rolling annualized historical volatility (stdev of log returns × sqrt(trading days)).

    Uses `min_periods=window` so the first `window` bars are genuinely NaN
    rather than a volatility estimate built on a partial sample.
    """
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    log_returns = compute_log_returns(df)
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(trading_days_per_year)


def compute_annualized_volatility(
    df: pd.DataFrame,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Full-sample annualized historical volatility over the entire input range.

    Returns None (never a fabricated number) when there are fewer than two
    valid log returns to estimate a standard deviation from.
    """
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    log_returns = compute_log_returns(df).dropna()
    if len(log_returns) < 2:
        return None
    return log_returns.std() * np.sqrt(trading_days_per_year)


def compute_hurst_exponent(df: pd.DataFrame, min_window: int = 10) -> Optional[float]:
    """Hurst exponent via classical Rescaled Range (R/S) analysis on daily
    log returns — Hurst's original 1951 method, not a single-scale
    approximation. For a log-spaced range of window sizes n, splits the
    return series into non-overlapping windows, computes each window's
    rescaled range R/S (R = the range of the window's mean-centered
    cumulative sum, S = the window's own standard deviation), and averages
    R/S across every window of that size. H is the slope of log(mean R/S)
    vs log(n) across window sizes, since E[R/S] scales as c * n^H.

    Applied to log RETURNS, not raw price levels: a random-walk PRICE
    series is itself already a cumulative sum (non-stationary), so running
    R/S directly on price levels reads H near 1.0 rather than the expected
    0.5 for a genuinely memoryless process — verified empirically before
    writing this (H~1.0 on raw price vs H~0.55 on returns for the same
    synthetic random walk). Log returns are the stationary series this
    test is actually designed for.

    H ~ 0.5: random walk (no memory). H > 0.5: trending/persistent
    (positive autocorrelation). H < 0.5: mean-reverting/anti-persistent.
    Cross-checked against synthetic series before shipping: a pure random
    walk reads ~0.55-0.58 on a 1-year (~250 observation) daily sample (a
    well-documented small-sample upward bias in the classical R/S
    statistic itself — Anis-Lloyd/Peters bias — not a defect in this
    implementation), a strongly trending AR(1) series (phi=0.6) reads
    ~0.71, and a strongly mean-reverting one (phi=-0.5) reads ~0.46 on the
    same sample size — correctly ordered and clearly separated.

    Returns None (never a fabricated number) when there's too little
    history to form at least 2 distinct window sizes.
    """
    log_returns = compute_log_returns(df).dropna().to_numpy()
    n = len(log_returns)
    max_window = n // 2
    if max_window < min_window:
        return None

    window_sizes = sorted(set(
        int(size) for size in np.logspace(np.log10(min_window), np.log10(max_window), 20)
        if int(size) >= min_window
    ))

    rs_values = []
    valid_sizes = []
    for size in window_sizes:
        num_windows = n // size
        if num_windows < 1:
            continue
        rs_for_size = []
        for i in range(num_windows):
            window = log_returns[i * size:(i + 1) * size]
            deviations = window - window.mean()
            cumulative = np.cumsum(deviations)
            r = cumulative.max() - cumulative.min()
            s = window.std(ddof=0)
            if s > 0:
                rs_for_size.append(r / s)
        if rs_for_size:
            rs_values.append(np.mean(rs_for_size))
            valid_sizes.append(size)

    if len(valid_sizes) < 2:
        return None

    slope, _ = np.polyfit(np.log(valid_sizes), np.log(rs_values), 1)
    return float(slope)


def compute_annualized_return(
    df: pd.DataFrame,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Full-sample annualized return: mean(log returns) × trading days per year.

    This is the log-return analogue of CAGR — summing log returns over the
    period equals ln(P_end / P_start), so the mean scaled by trading days is
    a standard, time-additive way to annualize. Kept on the same log-return
    convention as compute_annualized_volatility() so a Sharpe-style ratio
    built from both isn't mixing return conventions in numerator and
    denominator.

    Returns None (never a fabricated number) under the same insufficient-
    data condition as compute_annualized_volatility().
    """
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    log_returns = compute_log_returns(df).dropna()
    if len(log_returns) < 2:
        return None
    return log_returns.mean() * trading_days_per_year


def compute_sharpe_ratio(
    df: pd.DataFrame,
    risk_free_rate: Optional[float] = None,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Annualized Sharpe Ratio: excess return per unit of total volatility.

    (annualized return − risk-free rate) / annualized volatility, both legs
    computed from the same log-return series so they're on a consistent
    basis. Returns None when there isn't enough data to estimate either leg,
    or when volatility is exactly zero (a flat, zero-variance price series —
    the ratio is undefined, not infinite or fabricated).
    """
    risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK.risk_free_rate
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    annual_return = compute_annualized_return(df, trading_days_per_year)
    annual_vol = compute_annualized_volatility(df, trading_days_per_year)
    if annual_return is None or not annual_vol:
        return None
    return (annual_return - risk_free_rate) / annual_vol


@dataclass
class SharpeInterpretation:
    label: str
    explanation: str
    limitation: str


def interpret_sharpe_ratio(value: Optional[float]) -> Optional[SharpeInterpretation]:
    """Plain-language quality band for an annualized Sharpe Ratio, ready for direct UI display.

    Bands follow the common industry convention (Investopedia/CFA-curriculum
    style): <0 poor, 0-1 sub-optimal, 1-2 good, 2-3 very good, >3 exceptional
    (and unusual enough to be worth double-checking rather than taken at
    face value — often a short sample or a low-volatility fluke).

    Returns None if `value` is missing, so callers can render an "N/A" state
    instead of interpreting a fabricated number.
    """
    if value is None:
        return None
    limitation = (
        "Sharpe penalizes upside and downside volatility equally and assumes "
        "roughly normal returns — a strategy with rare, large gains (positive "
        "skew) or fat tails can look worse or better than it really is. "
        "Compare against Sortino (downside-only) for a fuller picture."
    )
    if value < 0:
        return SharpeInterpretation("Poor", "Negative risk-adjusted return — the asset underperformed the risk-free rate over this period once volatility is accounted for.", limitation)
    if value < 1.0:
        return SharpeInterpretation("Sub-optimal", "Positive but modest risk-adjusted return — returns are not compensating investors much for the volatility taken on.", limitation)
    if value < 2.0:
        return SharpeInterpretation("Good", "A solid risk-adjusted return — widely considered an acceptable-to-good Sharpe by institutional standards.", limitation)
    if value < 3.0:
        return SharpeInterpretation("Very Good", "A strong risk-adjusted return, notably above typical market benchmarks.", limitation)
    return SharpeInterpretation("Exceptional (verify)", "Unusually high for a sustained strategy — worth double-checking the sample period isn't too short or unusually low-volatility, rather than assuming it's simply an outstanding result.", limitation)


def compute_downside_deviation(
    df: pd.DataFrame,
    target_return: float = 0.0,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Annualized downside deviation — the Sortino/Fouse semi-deviation below `target_return`.

    sqrt(mean(min(R_i − target, 0)²)) × sqrt(trading days), averaged over
    EVERY observation (not just the negative-return subset) — a day that
    beats the target correctly contributes zero rather than being excluded
    from the sample size. This is the textbook Sortino formula, deliberately
    different from a plain std() of only the negative days (which changes
    both the denominator's sample size and what's being measured).

    `target_return` is a per-period (daily) rate, 0.0 by default — the
    conventional Minimum Acceptable Return most Sortino implementations use.

    Returns None under the same insufficient-data condition as the other
    annualized metrics in this module.
    """
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    log_returns = compute_log_returns(df).dropna()
    if len(log_returns) < 2:
        return None
    downside = np.minimum(log_returns - target_return, 0.0)
    return np.sqrt((downside ** 2).mean()) * np.sqrt(trading_days_per_year)


def compute_sortino_ratio(
    df: pd.DataFrame,
    risk_free_rate: Optional[float] = None,
    target_return: float = 0.0,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Annualized Sortino Ratio: excess return per unit of DOWNSIDE-only volatility.

    (annualized return − risk-free rate) / downside deviation — the same
    numerator as compute_sharpe_ratio(), but a denominator that only
    penalizes returns below `target_return`, making it more appropriate than
    Sharpe for asymmetric/positively-skewed return distributions.

    Returns None when there isn't enough data, or when downside deviation is
    exactly zero (no observations fell below the target at all — the ratio
    is undefined, not infinite).
    """
    risk_free_rate = risk_free_rate if risk_free_rate is not None else RISK.risk_free_rate
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    annual_return = compute_annualized_return(df, trading_days_per_year)
    downside_dev = compute_downside_deviation(df, target_return, trading_days_per_year)
    if annual_return is None or not downside_dev:
        return None
    return (annual_return - risk_free_rate) / downside_dev


def _lookback_log_returns(df: pd.DataFrame, lookback: Optional[int]) -> pd.Series:
    """Log returns restricted to the most recent `lookback` observations (or all of them)."""
    log_returns = compute_log_returns(df).dropna()
    if lookback is not None:
        log_returns = log_returns.tail(lookback)
    return log_returns


def compute_historical_var(
    df: pd.DataFrame,
    confidence_level: float,
    lookback: Optional[int] = None,
    min_observations: Optional[int] = None,
) -> Optional[float]:
    """1-day Historical VaR: the empirical percentile of the return distribution.

    Non-parametric — makes no assumption about the shape of returns, just reads
    the (1 - confidence_level) percentile straight off the actual historical
    sample (e.g. the 5th percentile for a 95% confidence level). Expressed as a
    signed log return (negative = a loss), over `lookback` most-recent trading
    days (or the whole available range if `lookback` is None).

    Returns None — never a fabricated figure — when there are fewer than
    `min_observations` returns to estimate a percentile from; a percentile off
    a handful of observations is not a meaningful tail estimate.
    """
    min_observations = min_observations or RISK.var_min_observations
    log_returns = _lookback_log_returns(df, lookback)
    if len(log_returns) < min_observations:
        return None
    return log_returns.quantile(1.0 - confidence_level)


def compute_parametric_var(
    df: pd.DataFrame,
    confidence_level: float,
    lookback: Optional[int] = None,
    min_observations: Optional[int] = None,
) -> Optional[float]:
    """1-day Parametric (variance-covariance) VaR, assuming normally distributed returns.

    Fits a normal distribution to the sample mean/stdev of log returns over
    `lookback` most-recent trading days, then reads the same (1 - confidence
    level) quantile off that fitted normal curve instead of the raw empirical
    sample — the classic RiskMetrics-style approach. Comparing this against
    compute_historical_var() on the same window is exactly how fat tails /
    skew in the real return distribution show up (they diverge).

    Returns None under the same insufficient-data condition as the historical
    version.
    """
    min_observations = min_observations or RISK.var_min_observations
    log_returns = _lookback_log_returns(df, lookback)
    if len(log_returns) < min_observations:
        return None
    mu = log_returns.mean()
    sigma = log_returns.std()
    z = norm.ppf(1.0 - confidence_level)
    return mu + z * sigma


def compute_expected_shortfall(
    df: pd.DataFrame,
    confidence_level: float,
    lookback: Optional[int] = None,
    min_observations: Optional[int] = None,
) -> Optional[float]:
    """1-day Expected Shortfall (Conditional VaR): the average loss in the tail beyond Historical VaR.

    Where Historical VaR answers "what's the loss at the cutoff", CVaR answers
    "given that the loss exceeds the cutoff, what's the average loss" — a
    strictly more informative tail-risk figure (and the reason Basel III moved
    bank capital requirements from VaR to Expected Shortfall). Computed as the
    mean of every log return at or below compute_historical_var()'s threshold,
    over the same `lookback` window and `confidence_level`.

    Returns None under the same insufficient-data condition as the VaR
    functions above.
    """
    min_observations = min_observations or RISK.var_min_observations
    log_returns = _lookback_log_returns(df, lookback)
    if len(log_returns) < min_observations:
        return None
    var_threshold = log_returns.quantile(1.0 - confidence_level)
    tail = log_returns[log_returns <= var_threshold]
    if tail.empty:
        return None
    return tail.mean()


def interpret_tail_risk(
    var_value: Optional[float],
    cvar_value: Optional[float],
    confidence_level: float,
) -> Optional[str]:
    """Plain-language interpretation of a VaR/CVaR pair, ready for direct UI display.

    Returns None if either input is missing, so callers can render an "N/A"
    state instead of a sentence built on a fabricated number.
    """
    if var_value is None or cvar_value is None:
        return None
    tail_gap = cvar_value - var_value
    return (
        f"On the worst {1.0 - confidence_level:.0%} of trading days (beyond the "
        f"{confidence_level:.0%} VaR cutoff of {var_value * 100:.2f}%), the average "
        f"loss is {cvar_value * 100:.2f}% — {abs(tail_gap) * 100:.2f} percentage points "
        f"worse than the VaR cutoff alone. That gap is the tail risk VaR alone doesn't capture."
    )


@dataclass
class MaxDrawdownResult:
    """The single worst peak-to-trough decline in a price/equity series."""
    max_drawdown: float          # negative fraction, e.g. -0.32 for a 32% decline
    peak_date: pd.Timestamp
    peak_price: float
    trough_date: pd.Timestamp
    trough_price: float
    recovered: bool
    recovery_date: Optional[pd.Timestamp]
    recovery_days: Optional[int]  # trading days from trough to recovery; None if not yet recovered


def compute_drawdown_series(prices: pd.Series) -> pd.Series:
    """Running drawdown from the running peak, as a fraction (0 = at a new high, negative = below it).

    Works on any price or cumulative-growth series — the formula is scale
    invariant, so the same function serves a raw Close price series and a
    strategy's cumulative-return index equally.
    """
    running_peak = prices.cummax()
    return (prices - running_peak) / running_peak


def compute_max_drawdown(prices: pd.Series) -> Optional[MaxDrawdownResult]:
    """The largest peak-to-trough decline in `prices`, plus its recovery period.

    Recovery is defined as the price closing back at or above the prior
    peak; `recovery_days` counts trading days from the trough to that point.
    If the series ends before that happens, `recovered` is False and
    `recovery_date`/`recovery_days` are None — never a fabricated recovery.

    Returns None only when there's no price data at all.
    """
    prices = prices.dropna()
    if len(prices) < 1:
        return None

    drawdown = compute_drawdown_series(prices)
    trough_date = drawdown.idxmin()
    trough_price = prices.loc[trough_date]
    max_dd = drawdown.loc[trough_date]

    prices_to_trough = prices.loc[:trough_date]
    peak_price = prices_to_trough.cummax().iloc[-1]
    peak_date = prices_to_trough[prices_to_trough == peak_price].index[-1]

    post_trough = prices.loc[trough_date:]
    recovery_mask = post_trough >= peak_price
    if recovery_mask.any():
        recovery_date = post_trough[recovery_mask].index[0]
        recovery_days = len(prices.loc[trough_date:recovery_date]) - 1
        recovered = True
    else:
        recovery_date = None
        recovery_days = None
        recovered = False

    return MaxDrawdownResult(
        max_drawdown=max_dd,
        peak_date=peak_date,
        peak_price=peak_price,
        trough_date=trough_date,
        trough_price=trough_price,
        recovered=recovered,
        recovery_date=recovery_date,
        recovery_days=recovery_days,
    )


def compute_calmar_ratio(
    df: pd.DataFrame,
    trading_days_per_year: Optional[int] = None,
) -> Optional[float]:
    """Calmar Ratio: annualized return divided by the magnitude of Maximum Drawdown.

    Unlike Sharpe/Sortino (which divide by a volatility estimate), Calmar
    compares return directly against the single worst realized loss — a
    tail-risk-focused measure popular for evaluating trend-following/managed-
    futures strategies, where a smooth equity curve with one bad drawdown is
    exactly the failure mode volatility-based ratios can miss.

    Built from this module's own compute_annualized_return() and
    compute_max_drawdown() — the same buy-and-hold drawdown figure already
    shown elsewhere in the app, not a separately computed one.

    Returns None when there isn't enough data, or when Maximum Drawdown is
    exactly zero (the price never fell below a prior high — the ratio is
    undefined, not infinite).
    """
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    annual_return = compute_annualized_return(df, trading_days_per_year)
    dd_result = compute_max_drawdown(df["Close"])
    if annual_return is None or dd_result is None or dd_result.max_drawdown == 0:
        return None
    return annual_return / abs(dd_result.max_drawdown)


@dataclass
class CalmarInterpretation:
    label: str
    explanation: str


def interpret_calmar_ratio(value: Optional[float]) -> Optional[CalmarInterpretation]:
    """Plain-language quality band for an annualized Calmar Ratio, ready for direct UI display.

    Bands follow the common practitioner convention for this ratio (looser
    and less standardized than Sharpe's CFA-curriculum bands, since Calmar
    is a newer, less academically formalized measure): <0 poor, 0-1 below
    average, 1-3 good, 3-5 very good, >5 excellent (and, like Sharpe's >3
    band, worth double-checking rather than taken purely at face value — a
    very high Calmar often just means the selected date range happened to
    contain a shallow drawdown, not that the strategy is robust to a bad one).

    Returns None if `value` is missing, so callers can render an "N/A" state
    instead of interpreting a fabricated number.
    """
    if value is None:
        return None
    if value < 0:
        return CalmarInterpretation("Poor", "Negative — the annualized return didn't even cover the worst drawdown experienced, let alone compensate for it.")
    if value < 1.0:
        return CalmarInterpretation("Below Average", "Positive but modest — annualized return is smaller than the worst peak-to-trough loss over this period.")
    if value < 3.0:
        return CalmarInterpretation("Good", "A solid return relative to the worst drawdown experienced — a commonly cited acceptable range for trend-following strategies.")
    if value < 5.0:
        return CalmarInterpretation("Very Good", "A strong return-to-drawdown profile, notably better than typical benchmarks.")
    return CalmarInterpretation("Excellent (verify)", "Unusually high — often a sign the selected date range happened to avoid a deep drawdown rather than proof the strategy is resilient to one. Check Max Drawdown's own recovery period before relying on this.")


def _anchored_score(value: Optional[float], anchors: Tuple[float, float]) -> Optional[float]:
    """0-100 sub-score: `anchors[0]` (best) maps to 100, `anchors[1]` (worst) maps to 0, clamped, linear between."""
    if value is None:
        return None
    best, worst = anchors
    if best == worst:
        return 100.0
    fraction = (value - worst) / (best - worst)
    return float(np.clip(fraction, 0.0, 1.0) * 100.0)


def _factor_status(sub_score: Optional[float]) -> str:
    """A word rather than a coloured dot.

    Text survives being copied out of a table, is unambiguous to a
    colour-blind reader, and does not depend on the terminal rendering an
    emoji font — the dots were showing as tofu boxes on some machines.
    """
    if sub_score is None:
        return "Not rated"
    if sub_score >= 70:
        return "Strong"
    if sub_score >= 40:
        return "Adequate"
    return "Weak"


@dataclass
class RiskFactor:
    label: str
    value_display: str
    sub_score: Optional[float]  # 0-100, None if this factor couldn't be computed
    weight: float
    status: str                 # "Strong" / "Adequate" / "Weak" / "Not rated" 


@dataclass
class RiskScoreResult:
    score: float
    grade: str
    grade_color: str          # hex, for the gauge; was an emoji the caller had to map
    factors: List[RiskFactor] = field(default_factory=list)
    excluded_factors: List[str] = field(default_factory=list)


def _risk_grade(score: float) -> Tuple[str, str]:
    """(grade label, hex colour)."""
    if score >= 75:
        return "Low Risk", "#22c55e"
    if score >= 50:
        return "Moderate Risk", "#eab308"
    if score >= 30:
        return "Elevated Risk", "#f97316"
    return "High Risk", "#ef4444"


def compute_risk_score(
    rolling_volatility: Optional[float],
    historical_var: Optional[float],
    expected_shortfall: Optional[float],
    max_drawdown: Optional[float],
    sharpe_ratio: Optional[float],
    sortino_ratio: Optional[float],
    calmar_ratio: Optional[float],
    altman_z: Optional[float],
) -> RiskScoreResult:
    """Combine every risk metric in this section into one 0-100 Composite Risk Score.

    Takes already-computed metric values rather than a raw DataFrame — this
    function is pure synthesis on top of the other functions in this module
    (and Altman Z from fundamental_analysis.py), not a new source of risk
    arithmetic. Each factor is normalized to a 0-100 sub-score against the
    anchors in RISK.risk_score_*_anchors (higher sub-score = lower risk /
    better performance, regardless of the factor's own natural direction),
    then combined via a weighted average.

    A factor that couldn't be computed (e.g. Altman Z for a bank with no
    classified balance sheet) is excluded from the average entirely and the
    remaining weights renormalized — the same "don't penalize what can't be
    checked" principle data_quality.py uses for field completeness, rather
    than silently treating "unknown" as "bad."
    """
    candidates = [
        ("Annualized Volatility", rolling_volatility, RISK.risk_score_vol_anchors, RISK.risk_score_weight_volatility, lambda v: f"{v * 100:.2f}%"),
        ("1-Day Historical VaR", historical_var, RISK.risk_score_var_anchors, RISK.risk_score_weight_var, lambda v: f"{v * 100:.2f}%"),
        ("Expected Shortfall (CVaR)", expected_shortfall, RISK.risk_score_cvar_anchors, RISK.risk_score_weight_cvar, lambda v: f"{v * 100:.2f}%"),
        ("Maximum Drawdown", max_drawdown, RISK.risk_score_drawdown_anchors, RISK.risk_score_weight_max_drawdown, lambda v: f"{v * 100:.2f}%"),
        ("Sharpe Ratio", sharpe_ratio, RISK.risk_score_sharpe_anchors, RISK.risk_score_weight_sharpe, lambda v: f"{v:.2f}"),
        ("Sortino Ratio", sortino_ratio, RISK.risk_score_sortino_anchors, RISK.risk_score_weight_sortino, lambda v: f"{v:.2f}"),
        ("Calmar Ratio", calmar_ratio, RISK.risk_score_calmar_anchors, RISK.risk_score_weight_calmar, lambda v: f"{v:.2f}"),
        # Anchored on the EFFECTIVE safe zone: the per-factor weights stay
        # fixed (deliberately not user-editable), but the point at which
        # Altman is considered fully safe has to agree with the verdict badge
        # the user retuned, or the score and the badge would contradict.
        ("Altman Z-Score", altman_z, (effective_risk().altman_safe_zone, 0.0), RISK.risk_score_weight_altman_z, lambda v: f"{v:.2f}"),
    ]

    factors: List[RiskFactor] = []
    excluded: List[str] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for label, value, anchors, weight, formatter in candidates:
        sub_score = _anchored_score(value, anchors)
        factors.append(RiskFactor(
            label=label,
            value_display=formatter(value) if value is not None else "N/A",
            sub_score=sub_score,
            weight=weight,
            status=_factor_status(sub_score),
        ))
        if sub_score is None:
            excluded.append(label)
        else:
            weighted_sum += sub_score * weight
            weight_total += weight

    score = (weighted_sum / weight_total) if weight_total > 0 else 0.0
    grade, grade_color = _risk_grade(score)

    return RiskScoreResult(
        score=round(score, 1),
        grade=grade,
        grade_color=grade_color,
        factors=factors,
        excluded_factors=excluded,
    )
