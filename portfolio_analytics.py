"""Portfolio Analytics Engine for Quantix.

Multi-ticker correlation, covariance, and diversification metrics — the
portfolio-level counterpart to risk_analytics.py's single-ticker risk
metrics. finance.py consumes the results and renders them; this module
performs no data fetching of its own — callers pass in already-fetched
price histories (via data_loader.load_price_history_only()), the same
"one fetch per dataset" principle every other module in this app follows.

Returns convention: reuses risk_analytics.compute_log_returns() for each
ticker's log returns, so correlation/covariance here are computed on
exactly the same return definition as every single-ticker risk metric
elsewhere in the app — not a second, differently-defined "returns."

Never-fabricate convention: a ticker with no usable price data is
excluded and disclosed via `excluded_tickers`/`exclusion_reasons`, never
silently dropped or backfilled. Correlation/covariance/diversification
results below the observation-count floor still compute (the numbers
aren't wrong, just noisy) — `sufficient_data` flags this so the caller can
decide whether to warn rather than this module silently deciding for it.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import RISK
from risk_analytics import compute_log_returns


@dataclass
class ReturnsAlignmentResult:
    returns: pd.DataFrame
    included_tickers: List[str]
    excluded_tickers: List[str]
    exclusion_reasons: Dict[str, str] = field(default_factory=dict)
    observation_count: int = 0
    sufficient_data: bool = False


def build_aligned_returns(
    price_histories: Dict[str, pd.DataFrame],
    lookback: Optional[int] = None,
    min_observations: Optional[int] = None,
) -> ReturnsAlignmentResult:
    """Align log returns for multiple tickers onto a shared trading-day index.

    Each ticker's log returns are computed independently, then inner-joined
    on their date index (pandas auto-aligns when building a DataFrame from
    named Series, and `.dropna(how="any")` keeps only dates every included
    ticker actually traded on) — so a ticker with a shorter history doesn't
    silently corrupt the others' returns with NaN-filled dates.

    A ticker with no price data at all is excluded upfront (`excluded_tickers`
    / `exclusion_reasons`). A ticker that has data but produces zero overlap
    with the others after alignment would simply contribute an all-NaN
    column and get dropped by the same `dropna` — recorded the same way.
    """
    min_observations = min_observations or RISK.correlation_min_observations

    per_ticker_returns: Dict[str, pd.Series] = {}
    excluded: List[str] = []
    reasons: Dict[str, str] = {}

    for ticker, df in price_histories.items():
        if df is None or df.empty or "Close" not in df.columns:
            excluded.append(ticker)
            reasons[ticker] = "no price data available"
            continue
        log_returns = compute_log_returns(df).dropna()
        if lookback is not None:
            log_returns = log_returns.tail(lookback)
        if log_returns.empty:
            excluded.append(ticker)
            reasons[ticker] = "no valid return observations in the selected window"
            continue
        per_ticker_returns[ticker] = log_returns

    if len(per_ticker_returns) < 2:
        return ReturnsAlignmentResult(
            returns=pd.DataFrame(),
            included_tickers=list(per_ticker_returns.keys()),
            excluded_tickers=excluded,
            exclusion_reasons=reasons,
            observation_count=0,
            sufficient_data=False,
        )

    returns_df = pd.DataFrame(per_ticker_returns).dropna(how="any")
    observation_count = len(returns_df)

    for ticker in per_ticker_returns:
        if ticker not in reasons and returns_df[ticker].isna().all():
            excluded.append(ticker)
            reasons[ticker] = "no overlapping trading days with the rest of the basket"

    included = [t for t in per_ticker_returns if t not in reasons]
    returns_df = returns_df[included]

    return ReturnsAlignmentResult(
        returns=returns_df,
        included_tickers=included,
        excluded_tickers=excluded,
        exclusion_reasons=reasons,
        observation_count=observation_count,
        sufficient_data=observation_count >= min_observations and len(included) >= 2,
    )


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of aligned log returns."""
    return returns.corr()


@dataclass
class BetaRegressionResult:
    beta: float
    r_squared: float
    observation_count: int


def compute_capm_beta(
    ticker_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    min_observations: Optional[int] = None,
) -> Optional[BetaRegressionResult]:
    """OLS beta of the ticker's log returns regressed against the
    benchmark's log returns: beta = Cov(ticker, benchmark) / Var(benchmark),
    the closed-form slope of a single-variable linear regression (same
    coefficient np.polyfit(benchmark, ticker, 1)[0] would give). R² (the
    squared Pearson correlation) is returned alongside so a low-confidence
    regression is visible to the caller rather than presented with false
    precision — this is the fundamental_analysis.py CAPM/WACC input the
    "CAPM Beta Regression" feature replaces the static Yahoo-reported beta
    with.

    Returns None (never a fabricated or degenerate beta) when there isn't
    enough overlapping history to regress reliably, or when the benchmark
    itself has zero variance over the window — the caller falls back to
    Yahoo's reported beta, then to a declared 1.0 market-beta assumption.
    """
    min_observations = min_observations or RISK.beta_regression_min_observations
    ticker_returns = compute_log_returns(ticker_df).dropna()
    benchmark_returns = compute_log_returns(benchmark_df).dropna()
    # The ticker and benchmark price histories don't necessarily go through
    # the same tz-normalization pipeline before reaching here (finance.py's
    # own ticker df is stripped of tz via price_processing.py; a benchmark
    # fetched straight from data_loader.py's macro bundle isn't) — a
    # tz-naive/tz-aware mismatch would otherwise make pandas raise
    # constructing the aligned frame below, so both are normalized to
    # tz-naive right here rather than trusting the caller already did it.
    if ticker_returns.index.tz is not None:
        ticker_returns.index = ticker_returns.index.tz_localize(None)
    if benchmark_returns.index.tz is not None:
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
    aligned = pd.DataFrame({"ticker": ticker_returns, "benchmark": benchmark_returns}).dropna()
    if len(aligned) < min_observations:
        return None

    benchmark_variance = aligned["benchmark"].var()
    if not benchmark_variance:
        return None

    beta = aligned["ticker"].cov(aligned["benchmark"]) / benchmark_variance
    r_squared = aligned["ticker"].corr(aligned["benchmark"]) ** 2
    return BetaRegressionResult(beta=float(beta), r_squared=float(r_squared), observation_count=len(aligned))


@dataclass
class PerformanceAttributionResult:
    total_excess_return_pct: float
    systematic_pct: float
    selection_pct: float
    beta_used: float
    risk_free_period_pct: float


def compute_performance_attribution(
    ticker_return_pct: float,
    benchmark_return_pct: float,
    beta: float,
    period_days: int,
    annual_risk_free_rate: Optional[float] = None,
) -> PerformanceAttributionResult:
    """Decompose a ticker's period return vs. its benchmark into a
    systematic (market-beta-driven) component and a selection
    (stock-specific residual) component — the standard Jensen's-alpha-style
    attribution, one level deeper than the simple `ticker return - benchmark
    return` alpha the Relative Strength & Alpha Generation section already
    shows:

        Total Excess Return = Ticker Return - Risk-Free Rate (prorated to the period)
        Systematic          = beta * (Benchmark Return - Risk-Free Rate)
        Selection            = Total Excess Return - Systematic

    Selection is DEFINED as the residual, so Systematic + Selection always
    reconstructs Total Excess Return exactly, by construction — not an
    approximation that happens to be close.

    `beta` is a plain float rather than this module's own
    BetaRegressionResult so the caller can pass whichever step of the
    regressed -> Yahoo-reported -> 1.0 assumption fallback chain
    (fundamental_analysis.FundamentalAnalysisEngine.beta_estimate()) it
    actually resolved to, without this function needing to know about that
    chain itself.

    The annual risk-free rate is compounded to the period length — (1 +
    r)^(period_days/365.25) - 1 — rather than scaled linearly, matching how
    this app's DCF/CAPM treat the risk-free rate as an annualized figure
    everywhere else. Defaults to config.RISK.risk_free_rate when not given.
    """
    annual_risk_free_rate = annual_risk_free_rate if annual_risk_free_rate is not None else RISK.risk_free_rate
    period_years = period_days / 365.25 if period_days > 0 else 0.0
    risk_free_period_pct = ((1 + annual_risk_free_rate) ** period_years - 1) * 100

    total_excess_return_pct = ticker_return_pct - risk_free_period_pct
    benchmark_excess_pct = benchmark_return_pct - risk_free_period_pct
    systematic_pct = beta * benchmark_excess_pct
    selection_pct = total_excess_return_pct - systematic_pct

    return PerformanceAttributionResult(
        total_excess_return_pct=total_excess_return_pct,
        systematic_pct=systematic_pct,
        selection_pct=selection_pct,
        beta_used=beta,
        risk_free_period_pct=risk_free_period_pct,
    )


def compute_covariance_matrix(returns: pd.DataFrame, trading_days_per_year: Optional[int] = None) -> pd.DataFrame:
    """Annualized covariance matrix of aligned log returns."""
    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    return returns.cov() * trading_days_per_year


@dataclass
class DiversificationResult:
    portfolio_volatility: float
    weighted_average_volatility: float
    diversification_benefit: float        # weighted average minus portfolio; positive = real benefit
    diversification_ratio: Optional[float]  # weighted average / portfolio; None if portfolio vol is ~0


def compute_portfolio_diversification(
    returns: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    trading_days_per_year: Optional[int] = None,
) -> Optional[DiversificationResult]:
    """Portfolio volatility (accounting for correlation) vs. the weighted average
    of each holding's own volatility (ignoring correlation) — the gap between
    the two IS the diversification benefit.

    Portfolio variance uses the standard Markowitz formula, w^T Σ w, where Σ
    is the annualized covariance matrix — for two assets this expands to
    w1²σ1² + w2²σ2² + 2w1w2·cov12, the textbook two-asset case.

    `weights` defaults to equal-weight across every column in `returns` if
    omitted; if provided, weights are normalized to sum to 1 (and any ticker
    not in `returns` is ignored) so callers don't have to pre-normalize.

    Returns None when there are fewer than 2 tickers, or when the supplied
    weights sum to zero (undefined) — never a fabricated result.
    """
    if returns.empty or returns.shape[1] < 2:
        return None

    trading_days_per_year = trading_days_per_year or RISK.trading_days_per_year
    tickers = list(returns.columns)

    if weights is None:
        w = np.full(len(tickers), 1.0 / len(tickers))
    else:
        raw = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
        total = raw.sum()
        if total <= 0:
            return None
        w = raw / total

    cov_matrix = compute_covariance_matrix(returns, trading_days_per_year).values
    portfolio_variance = float(w @ cov_matrix @ w)
    portfolio_vol = float(np.sqrt(max(portfolio_variance, 0.0)))

    individual_vols = np.sqrt(np.clip(np.diag(cov_matrix), 0.0, None))
    weighted_avg_vol = float(np.dot(w, individual_vols))

    diversification_ratio = (weighted_avg_vol / portfolio_vol) if portfolio_vol > 1e-9 else None

    return DiversificationResult(
        portfolio_volatility=portfolio_vol,
        weighted_average_volatility=weighted_avg_vol,
        diversification_benefit=weighted_avg_vol - portfolio_vol,
        diversification_ratio=diversification_ratio,
    )
