"""Centralized configuration for Quantix.

Every tunable constant in the app — scorecard thresholds, DCF assumptions,
watchlist baskets, chart defaults, risk parameters — lives here instead of
being scattered as magic numbers throughout finance.py. Grouped by concern
so the setting you want is easy to find and change in one place.

Does NOT include the cache TTLs in data_loader.py or the quality-scoring
constants in data_quality.py — those already live as clearly-named
module-level constants right next to the logic they configure, and moving
them here would separate them from their context for no real benefit.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class WatchlistConfig:
    """Fast pre-screen thresholds used to scan the tech/diversified baskets.

    Deliberately stricter than SCORECARD below: this runs across a whole
    basket of tickers to surface only the strongest candidates, whereas the
    Scorecard runs a deeper analysis on a single ticker the user already
    chose to research.
    """
    tech_basket: Tuple[str, ...] = ("AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "META")
    diversified_basket: Tuple[str, ...] = ("JPM", "V", "MA", "LLY", "UNH", "COST", "WMT", "AVGO", "ASML", "CAT")
    min_net_margin: float = 0.10
    max_debt_to_equity: float = 1.0
    min_current_ratio: float = 1.2
    pe_range: Tuple[float, float] = (15, 45)
    cards_shown: int = 4


@dataclass(frozen=True)
class ScorecardConfig:
    """Thresholds and weights for the Strategic Investment Scorecard.

    Sector-adjusted: Financial Services companies use
    `financials_max_debt_to_equity` in place of `max_debt_to_equity` — banks
    are structurally more leveraged as a business model (deposits and
    borrowings ARE the business), so the corporate-leverage threshold that
    flags a non-financial company as risky isn't a meaningful signal for one.
    Every other threshold is shared across sectors: a full per-sector
    benchmark table wasn't built out for the other ~10 GICS sectors since
    there's no live external industry-benchmark source in this environment
    to validate invented numbers against, and Debt-to-Equity was the one
    threshold with a concrete, demonstrated cross-sector problem.

    A metric with no computable value (common for Financials — see
    fundamental_analysis.py) is excluded from both the numerator and
    denominator of the score rather than counted as a failure, so missing
    data doesn't silently penalize companies structurally unable to report
    it. `weights` then determines how much each *evaluable* metric counts
    toward the score — applied uniformly across sectors (only the D/E
    threshold itself varies by sector, not the weights).
    """
    min_net_margin: float = 0.10
    max_debt_to_equity: float = 2.5
    financials_max_debt_to_equity: float = 4.0
    # Yahoo's `sector` field has used both spellings across versions.
    financials_sector_names: Tuple[str, ...] = ("Financial Services", "Financials")
    min_current_ratio: float = 1.0
    max_beta: float = 1.5
    pe_range: Tuple[float, float] = (10, 45)
    peg_range: Tuple[float, float] = (0, 2.5)
    min_interest_coverage: float = 3.0
    min_roic_pct: float = 10.0
    min_fcf_yield_pct: float = 4.0  # shown in the Master Matrix; not one of the scoreboard flags
    high_alignment_pct: float = 75
    moderate_alignment_pct: float = 40
    # Core financial-health signals (profitability, leverage, capital
    # efficiency) count for more than secondary considerations (valuation
    # multiples, volatility) when computing the weighted Blueprint Alignment
    # score. Any key not listed defaults to a weight of 1.0.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "net_margin": 1.5,
        "debt_to_equity": 1.5,
        "roic": 1.5,
        "interest_coverage": 1.0,
        "current_ratio": 1.0,
        "pe_ratio": 0.75,
        "peg_ratio": 0.75,
        "beta": 0.5,
    })

    def max_debt_to_equity_for(self, sector: Optional[str]) -> float:
        if sector in self.financials_sector_names:
            return self.financials_max_debt_to_equity
        return self.max_debt_to_equity

    def weight_for(self, key: str) -> float:
        return self.weights.get(key, 1.0)


@dataclass(frozen=True)
class DCFAssumptions:
    market_return: float = 0.10
    tax_rate: float = 0.21
    terminal_growth_rate: float = 0.02
    projection_years: int = 5
    strong_buy_margin_of_safety: float = 20.0
    sensitivity_wacc_delta: float = 0.02
    sensitivity_growth_delta: float = 0.05
    sensitivity_steps: int = 3


@dataclass(frozen=True)
class RiskConfig:
    risk_free_rate: float = 0.04  # shared by Sharpe/Sortino and the DCF's CAPM cost of equity
    trading_days_per_year: int = 252  # annualization factor shared by every risk metric
    vix_high_risk_threshold: float = 25.0
    altman_safe_zone: float = 2.99
    altman_grey_zone: float = 1.81
    hurst_mean_reverting_below: float = 0.45
    hurst_trending_above: float = 0.55
    backtest_buy_z_score: float = -2.0
    backtest_sell_z_score: float = 0.0
    kelly_half_factor: float = 2.0             # half-Kelly for risk management
    kelly_macro_risk_extra_factor: float = 2.0  # additional halving when the VIX risk flag is active
    var_confidence_levels: Tuple[float, ...] = (0.90, 0.95, 0.99)
    var_confidence_default: float = 0.95
    var_min_observations: int = 20  # below this, a percentile/normal-fit estimate is too unstable to show
    correlation_min_observations: int = 20  # below this, a correlation/covariance estimate is too unstable to show

    # Composite Risk Score: normalization anchors (value at 0-score <-> 100-score)
    # and weights for the Risk Dashboard's single 0-100 summary figure. Weights
    # sum to 1.0 over whichever factors are actually computable for the ticker;
    # unavailable factors (e.g. Altman Z for banks) are excluded and the rest
    # renormalized, same "don't penalize what can't be checked" principle as
    # data_quality.py's field-completeness scoring.
    # Each anchor tuple is (best, worst): the value that scores 100, then the
    # value that scores 0, linearly interpolated (and clamped) between.
    risk_score_vol_anchors: Tuple[float, float] = (0.15, 0.60)        # 15% ann. vol -> 100, 60% -> 0
    risk_score_var_anchors: Tuple[float, float] = (0.0, -0.08)        # 0% 1-day VaR -> 100, -8% -> 0
    risk_score_cvar_anchors: Tuple[float, float] = (0.0, -0.12)       # 0% CVaR -> 100, -12% -> 0
    risk_score_drawdown_anchors: Tuple[float, float] = (0.0, -0.60)   # 0% drawdown -> 100, -60% -> 0
    risk_score_sharpe_anchors: Tuple[float, float] = (2.5, 0.0)       # Sharpe 2.5 -> 100, 0 -> 0
    risk_score_sortino_anchors: Tuple[float, float] = (2.5, 0.0)      # same scale as Sharpe
    risk_score_calmar_anchors: Tuple[float, float] = (5.0, 0.0)       # Calmar 5 -> 100, 0 -> 0
    # Altman Z uses its own safe-zone threshold as the 100-score anchor (see below)

    risk_score_weight_volatility: float = 0.15
    risk_score_weight_var: float = 0.15
    risk_score_weight_cvar: float = 0.15
    risk_score_weight_max_drawdown: float = 0.20
    risk_score_weight_sharpe: float = 0.10
    risk_score_weight_sortino: float = 0.10
    risk_score_weight_calmar: float = 0.10
    risk_score_weight_altman_z: float = 0.05


@dataclass(frozen=True)
class MonteCarloConfig:
    num_simulations: int = 1000
    forecast_days: int = 60
    plotted_paths: int = 150
    upside_bias_threshold_pct: float = 65.0
    downside_bias_threshold_pct: float = 40.0


@dataclass(frozen=True)
class ChartDefaults:
    default_ticker: str = "AAPL"
    default_benchmark: str = "SPY"
    default_lookback_days: int = 365
    sma_default: int = 20
    sma_range: Tuple[int, int] = (5, 200)
    rsi_default: int = 14
    rsi_range: Tuple[int, int] = (5, 50)
    atr_default: int = 14
    atr_range: Tuple[int, int] = (5, 50)
    stochastic_k_range: Tuple[int, int] = (5, 50)
    adx_range: Tuple[int, int] = (5, 50)
    vol_window_default: int = 21
    vol_window_range: Tuple[int, int] = (5, 120)
    var_lookback_default: int = 252
    var_lookback_range: Tuple[int, int] = (30, 500)
    risk_free_rate_range_pct: Tuple[float, float] = (0.0, 10.0)
    portfolio_lookback_default: int = 252
    portfolio_lookback_range: Tuple[int, int] = (60, 500)
    portfolio_default_basket: str = "AAPL, MSFT, GOOGL, JPM"
    dcf_growth_default_pct: int = 15
    dcf_growth_range_pct: Tuple[int, int] = (1, 35)
    dcf_wacc_default_pct: int = 9
    dcf_wacc_range_pct: Tuple[int, int] = (5, 15)


@dataclass(frozen=True)
class TechnicalConfig:
    """Settings for technical_indicators.py — the technical analysis
    calculation layer (SMA, RSI, MACD, Bollinger Bands, ATR, Stochastic,
    ADX, Ichimoku Cloud, and OBV)."""
    # The three most widely-used SMA periods (short/medium/long-term trend),
    # shown as an optional overlay alongside the user's custom-length line
    # (CHART_DEFAULTS.sma_default/sma_range) — standard across virtually
    # every charting platform, not a Quantix-specific choice.
    sma_trio_periods: Tuple[int, ...] = (20, 50, 200)
    sma_trio_colors: Tuple[str, ...] = ("#38bdf8", "#facc15", "#f87171")  # short/medium/long, distinct from the custom line's orange

    # Classic Wilder RSI thresholds — the universal default shown by
    # TradingView and virtually every other charting platform out of the
    # box. Fixed, not user-configurable: RSI's period is already the one
    # adjustable control (sidebar "RSI Length").
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # Standard MACD periods (fast/slow/signal EMA spans) — the universal
    # 12/26/9 default virtually every platform ships with. Fixed, not
    # user-configurable: unlike SMA/RSI, MACD's periods are rarely tuned in
    # practice, and three new sliders would clutter the sidebar for a
    # setting almost nobody changes.
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9

    # Standard Bollinger Band width — the universal 2.0-standard-deviation
    # default. Fixed, not user-configurable: the band's period reuses the
    # existing SMA Length slider (CHART_DEFAULTS.sma_default/sma_range), so
    # width is the only remaining parameter, and 2.0σ is what virtually
    # every platform ships with.
    bollinger_num_std: float = 2.0

    # ATR-based stop-loss: Suggested Stop = Current Price − multiplier × ATR.
    # 2.0 is the most common moderate default across trading literature —
    # tight enough to matter, wide enough to avoid ordinary daily noise
    # triggering it. Fixed, not user-configurable; ATR's own period already
    # has a dedicated sidebar slider (CHART_DEFAULTS.atr_default/atr_range).
    # Long-only (Current Price − N×ATR): this app's framing is buy-side
    # throughout (Kelly Criterion, DCF, Quality Score all assume going
    # long, never shorting), so only a downside stop is shown.
    atr_stop_multiplier: float = 2.0

    # Standard Stochastic Oscillator periods (the "Slow Stochastic" —
    # pre-smoothed %K — is what TradingView and pandas_ta both actually
    # ship as their default "Stochastic", confirmed by reading pandas_ta's
    # source rather than the simpler textbook Fast Stochastic formula).
    # Overbought/oversold at 80/20 — Stochastic's own universal convention,
    # deliberately different from RSI's 70/30.
    stochastic_k_period: int = 14
    stochastic_d_period: int = 3
    stochastic_smooth_k: int = 3
    stochastic_overbought: float = 80.0
    stochastic_oversold: float = 20.0

    # ADX trend-strength threshold — the standard Wilder convention: ADX
    # above this is considered "trending" (favors trend-following),
    # below is considered "non-trending/choppy" (favors mean-reversion) —
    # directly relevant to this app's own mean-reversion Backtest section.
    adx_period: int = 14
    adx_trend_threshold: float = 25.0

    # Standard Ichimoku Cloud periods — virtually universal across every
    # platform that implements it; rarely tuned in practice, unlike SMA/RSI.
    ichimoku_tenkan_period: int = 9
    ichimoku_kijun_period: int = 26
    ichimoku_senkou_b_period: int = 52


@dataclass(frozen=True)
class PeerDefaults:
    """Smart default peer tickers per target, to make testing/demoing feel seamless."""
    fallback: str = "MA, AXP, PYPL"
    by_ticker: Dict[str, str] = field(default_factory=lambda: {
        "AAPL": "MSFT, GOOGL, META",
        "MSFT": "AAPL, GOOGL, ORCL",
        "NVDA": "AMD, INTC, TSM",
        "CROX": "SKX, DECK, NKE",
        "GOOGL": "META, AMZN, MSFT",
    })

    def for_ticker(self, ticker: str) -> str:
        return self.by_ticker.get(ticker, self.fallback)


@dataclass(frozen=True)
class TearSheetConfig:
    """Thresholds for the CIO verdict narrative (STRONG BUY / HOLD / AVOID) and briefing."""
    strong_buy_min_score_pct: float = 75
    strong_buy_min_margin_of_safety: float = -30
    hold_watchlist_min_score_pct: float = 50
    hold_watchlist_min_margin_of_safety: float = -25
    high_insider_ownership_pct: float = 5


@dataclass(frozen=True)
class OutlierBoundsConfig:
    """Sanity bounds for the Financial Metrics Validation Report's outlier
    detection — a metric whose magnitude exceeds its bound is flagged as
    "check this," separately from whether it agrees with Yahoo's own figure
    (a value can be internally consistent and cross-check cleanly while
    still being an outlier, or vice versa).

    Every bound here is a judgment call, not derived from a live external
    industry-benchmark source (same disclosed limitation as the sector
    thresholds in ScorecardConfig) — set comfortably above every legitimate
    value already observed in this app (e.g. AAPL's real ROE of ~152% from
    heavy buybacks stays well under the 500% bound below), so this fires on
    values that are very likely a data or calculation error, not merely an
    unusual-but-real company.
    """
    max_abs_net_margin_pct: float = 200.0
    max_gross_margin_pct: float = 100.0       # cost of revenue can't be negative, so >100% is essentially always a data issue
    max_operating_margin_pct: float = 100.0
    max_abs_roa_pct: float = 100.0
    max_abs_roe_pct: float = 500.0            # heavy-buyback companies (AAPL) legitimately exceed 100%; bound sits well above that
    max_abs_roic_pct: float = 200.0
    max_current_ratio: float = 20.0
    max_quick_ratio: float = 20.0
    max_debt_to_equity: float = 10.0          # comfortably above even the sector-relaxed Financials scorecard threshold of 4.0
    max_pe_ratio: float = 300.0               # extreme-growth stocks can legitimately sit in the low hundreds
    max_price_to_book: float = 100.0
    max_abs_peg_ratio: float = 10.0
    max_abs_ev_ebitda: float = 100.0
    max_abs_fcf_yield_pct: float = 50.0
    max_beta: float = 5.0


@dataclass(frozen=True)
class QualityConfig:
    """Bands and weights for the multi-factor Company Quality Classification
    — a complementary, differently-framed view from the Strategic Investment
    Scorecard (which stays a pass/fail checklist). This blends five factors
    into one 0-100 quality score and category.

    Every band/weight here is a disclosed judgment call (same limitation as
    OutlierBoundsConfig/ScorecardConfig's sector thresholds — no live
    external quality-rating source to calibrate against). Most factors use
    linear bands (`(worst, best)`: `worst` scores 0, `best` scores 100,
    clamped, inverted for metrics where lower is better). Valuation
    deliberately does NOT reward cheapness — standard quality-investing
    methodology (e.g. MSCI Quality Index) excludes valuation entirely,
    since excellent businesses often justly trade at premium multiples; here
    each valuation metric instead scores 100 at an "ideal"/fairly-priced
    center point and falls off proportionally to relative distance from it
    in either direction, so both extreme cheapness and extreme
    expensiveness get flagged.

    A metric that isn't computable for a company (common for banks — see
    fundamental_analysis.py) is excluded from its factor's average rather
    than scored as 0; a factor with zero computable metrics is excluded
    from the overall weighted score entirely — same "don't penalize missing
    data" principle as the Scorecard redesign.
    """
    # --- Profitability (linear bands, % values) ---
    net_margin_band_pct: Tuple[float, float] = (0.0, 25.0)
    gross_margin_band_pct: Tuple[float, float] = (0.0, 60.0)
    operating_margin_band_pct: Tuple[float, float] = (0.0, 30.0)
    roa_band_pct: Tuple[float, float] = (0.0, 15.0)

    # --- Financial Stability ---
    # Debt-to-Equity is inverted (lower is better) and sector-aware: the
    # 100-point anchor is 0, the 0-point anchor is 2x the sector's Scorecard
    # threshold (SCORECARD.max_debt_to_equity_for), so Financials and other
    # sectors each get a proportionally fair band without a separate table.
    debt_to_equity_zero_point_multiplier: float = 2.0
    current_ratio_band: Tuple[float, float] = (0.0, 3.0)
    interest_coverage_band: Tuple[float, float] = (0.0, 10.0)
    # Reuses the same Altman zone boundaries already shown elsewhere in the
    # app (RISK.altman_grey_zone / altman_safe_zone) rather than inventing a
    # third set of numbers for the same underlying concept.
    altman_z_band: Tuple[float, float] = (0.0, 3.5)

    # --- Growth (single Yahoo-reported figure, decimal e.g. 0.15 = 15%) ---
    earnings_growth_band_pct: Tuple[float, float] = (-10.0, 25.0)

    # --- Valuation ("ideal"/fairly-priced center points, not "cheapest wins") ---
    pe_ideal: float = 20.0
    peg_ideal: float = 1.0    # the textbook Peter Lynch "fairly priced" PEG
    price_to_book_ideal: float = 3.0
    ev_ebitda_ideal: float = 12.0

    # --- Capital Efficiency ---
    roic_band_pct: Tuple[float, float] = (0.0, 20.0)
    roe_band_pct: Tuple[float, float] = (0.0, 30.0)
    # Naturally sector-dependent (asset-light software vs. asset-heavy
    # banks/utilities) — a single global band is a coarser approximation
    # here than the sector-adjusted Debt-to-Equity treatment above.
    asset_turnover_band: Tuple[float, float] = (0.0, 1.5)

    # --- Factor weights (sum to 1.0) ---
    weight_profitability: float = 0.25
    weight_financial_stability: float = 0.25
    weight_capital_efficiency: float = 0.20
    weight_growth: float = 0.15
    weight_valuation: float = 0.15

    # --- Category thresholds ---
    elite_min_score: float = 85.0
    high_min_score: float = 70.0
    average_min_score: float = 50.0
    below_average_min_score: float = 30.0


WATCHLIST = WatchlistConfig()
SCORECARD = ScorecardConfig()
DCF = DCFAssumptions()
RISK = RiskConfig()
MONTE_CARLO = MonteCarloConfig()
CHART_DEFAULTS = ChartDefaults()
TECHNICAL = TechnicalConfig()
PEER_DEFAULTS = PeerDefaults()
TEAR_SHEET = TearSheetConfig()
OUTLIER_BOUNDS = OutlierBoundsConfig()
QUALITY = QualityConfig()
