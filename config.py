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
    # Sector-adjusted P/E override table. Deliberately NOT a full ~11-sector
    # GICS benchmark table — same disclosed limitation as Debt-to-Equity
    # above: no live external industry-benchmark source in this environment
    # to validate invented numbers against. Configured only for sectors with
    # a well-established, widely-cited structural reason the flat (10, 45)
    # band misprices them: asset-light growth sectors sustainably trade
    # above it; leverage-driven, regulated, or commodity-cyclical sectors
    # sustainably trade below it. Every other sector (Healthcare, Consumer
    # Cyclical, Industrials, Communication Services, Basic Materials, Real
    # Estate) falls back to the global band via pe_range_for() rather than a
    # guessed number — Healthcare and Communication Services in particular
    # mix structurally different sub-industries under one Yahoo sector label
    # (biotech vs. staple pharma; legacy telecom vs. mega-cap growth ad-tech)
    # so no single band is defensible for either without finer-grained data
    # this app doesn't have.
    sector_pe_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Technology": (15.0, 65.0),         # asset-light, scalable margins — the market has sustained a growth premium here for decades
        "Financial Services": (6.0, 18.0),  # leverage-driven earnings and regulatory capital rules cap what the market affords banks
        "Financials": (6.0, 18.0),          # Yahoo's alternate spelling for the same sector, mirrors financials_sector_names above
        "Utilities": (12.0, 22.0),          # regulated, low-single-digit earnings growth — priced as a bond proxy, not a growth equity
        "Energy": (6.0, 22.0),              # commodity-cyclical earnings swing with oil/gas prices — historically discounted to the broad market
        "Consumer Defensive": (14.0, 28.0), # stable, low-growth staples — priced above deep-value cyclicals but below growth sectors
    })
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

    def pe_range_for(self, sector: Optional[str]) -> Tuple[float, float]:
        """The sector-adjusted P/E band, or the global fallback band if the
        sector has no configured override (see sector_pe_ranges above).

        PEG is deliberately NOT given the same per-sector treatment: PEG
        already divides P/E by the growth rate, which is specifically what
        makes a flat P/E band misleading across growth-premium vs. low-growth
        sectors in the first place. A PEG of ~1.0 reads as "fairly priced
        relative to its own growth" whether the underlying stock is a
        high-growth/high-P/E name or a low-growth/low-P/E one — the sector
        distortion P/E has is already normalized out.
        """
        return self.sector_pe_ranges.get(sector, self.pe_range)

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
    # A regressed beta needs more history than a simple correlation to be
    # stable — 60 trading days (~3 months) is a floor, not the standard
    # academic 2-year/weekly window, since the app already loads whatever
    # date range the user selected rather than a second, separate fetch.
    beta_regression_min_observations: int = 60

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
    # Block length for the bootstrap-resampling simulation method. Trading
    # days are resampled in contiguous blocks (not individually) so that
    # whatever short-run autocorrelation/volatility clustering actually
    # occurred historically survives into the simulated paths.
    bootstrap_block_days: int = 5
    min_history_days_for_bootstrap: int = 30


@dataclass(frozen=True)
class WatchlistPanelConfig:
    """The sidebar's quick-switch watchlist — a user-maintained list, not
    the fixed Institutional Watchlist baskets above (which are a scoring
    universe, not a personal tracking list). `default_tickers` only seeds
    the FIRST-EVER watchlist (named `default_watchlist_name`) the first
    time this app runs with no saved store file yet; after that, every
    list's contents and which one is active come from the persisted
    store, and the user's own edits win.

    max_tickers is a UI bound, not a data one: every row costs a (cached)
    quote lookup and two lines of sidebar height, so the cap keeps the
    panel scannable and the fetch cheap rather than reflecting any Yahoo
    limit.
    """
    default_tickers: Tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "GOOGL")
    default_watchlist_name: str = "My Watchlist"
    max_tickers: int = 10
    max_watchlists: int = 10  # UI bound on the multi-list switcher, not a storage limit
    store_filename: str = "watchlist_store.json"

    # Recently-viewed strip (chips under the symbol header). Separate cap
    # from max_tickers: this list is accumulated automatically by
    # navigating, not curated, so it's kept shorter to stay a single
    # scannable row rather than a wall of chips.
    max_recent_tickers: int = 8


@dataclass(frozen=True)
class RealtimeAlertsConfig:
    """Real-Time Alert Engine — in-tab polling, not a background worker;
    in-app notification only, not email/push; rules and trigger history
    persisted to a local file, per signed-in user when auth is configured.
    Each of those three is a scope decision made explicitly with the user
    before this was built, not a silent simplification — see
    realtime_alerts.py's module docstring for the full reasoning."""
    poll_interval_seconds: int = 60
    store_filename: str = "alert_rules_store.json"
    max_rules: int = 20        # UI bound: keeps the rule list scannable and every poll's fetch count bounded
    max_history: int = 100     # trimmed on every save so the store file can't grow unbounded


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """Defaults for the Portfolio Backtester — the multi-ticker, weighted,
    rebalanced counterpart to the single-ticker Algorithmic Backtesting
    Simulator. Runs the SAME strategy already configured there across a
    basket, rather than exposing a second, separate strategy definition."""
    # UI/fetch bound, not a Yahoo limit: each basket ticker costs its own
    # deep fetch + indicator computation, so this keeps a run tractable and
    # the per-ticker contribution table scannable.
    max_tickers: int = 8
    default_rebalance_frequency: str = "monthly"   # one of REBALANCE_FREQUENCIES in portfolio_backtester.py
    default_rebalance_threshold_pct: float = 5.0    # only consulted when threshold-based rebalancing is enabled


@dataclass(frozen=True)
class MLPipelineConfig:
    """The Momentum Continuation classifier — a BASELINE model by design
    (the originating task's own wording), not an ensemble or deep model:
    a single, interpretable Logistic Regression trained on features this
    app already computes elsewhere (RSI, MACD, SMA structure, trailing
    returns, rolling volatility, volume) rather than new indicator math.
    See ml_pipeline.py's module docstring for the full design reasoning,
    including why predicting stock direction is a genuinely hard problem
    this app is careful not to overclaim skill at."""
    label_horizon_days: int = 10        # forward window the label looks ahead over — distinct from the trailing feature windows
    train_lookback_days: int = 1500     # ~6 years of daily bars per training ticker
    test_fraction: float = 0.2          # most-recent slice of the COMBINED timeline held out — never a random shuffle, which would leak future information into training
    min_training_rows: int = 200        # below this, training is refused rather than fit on too little data to mean anything
    model_filename: str = "ml_momentum_model.joblib"
    history_filename: str = "ml_training_history.json"
    max_history: int = 50               # trimmed on every save so the history file can't grow unbounded


@dataclass(frozen=True)
class ScenarioModelingConfig:
    """Default shocks for the three required scenario types (Dividend Cut,
    Recession, Sector Multiple Compression) — starting points a user edits
    before running, not forced assumptions. Every shock is expressed
    ENTIRELY through parameters the existing DCF/risk engines already
    accept (growth_rate, discount_rate, and a shock to the historical
    return series for VaR/CVaR/Sharpe/Max Drawdown) — see
    scenario_modeling.py's module docstring for why Dividend Cut in
    particular needed real thought: this app's DCF is unlevered-FCF-based,
    which is dividend-policy-invariant by the Modigliani-Miller theorem,
    so a dividend cut has no direct, non-fabricated DCF lever the way a
    growth or discount-rate shock does."""
    default_growth_rate_delta_recession: float = -0.06     # -6pp off the DCF's revenue growth assumption
    default_discount_rate_delta_recession: float = 0.02    # +2pp WACC, a widened equity risk premium
    default_volatility_multiplier_recession: float = 1.6
    default_mean_return_shift_recession: float = -0.0015   # per-day log-return shift

    default_discount_rate_delta_sector_shift: float = 0.025  # +2.5pp WACC — the direct DCF equivalent of a lower market-clearing multiple
    default_volatility_multiplier_sector_shift: float = 1.2

    default_dividend_cut_pct: float = 50.0
    # Optional, off by default: an ADDITIONAL WACC add-on representing
    # hypothesized market repricing of perceived risk after a cut
    # announcement — a disclosed assumption the user opts into, never
    # applied automatically, since it isn't derived from the cut itself.
    default_discount_rate_delta_dividend_cut: float = 0.0

    store_filename: str = "scenario_store.json"
    max_saved_scenarios: int = 30
    default_investment_amount: float = 10_000.0  # illustrative-only dollar base for the "portfolio value impact" figure


@dataclass(frozen=True)
class WalkForwardConfig:
    """Defaults for the Backtest section's optional walk-forward mode.
    Trading days, not calendar months, to match how every other lookback
    in this app (Monte Carlo's forecast_days, VaR's lookback, etc.) is
    already expressed — ~6 months train / ~2 months test is a common
    walk-forward convention, not a fitted or validated split (this engine
    has no parameters to fit; see strategy_builder.run_walk_forward_backtest
    for what the train segment actually does here)."""
    default_train_days: int = 126
    default_test_days: int = 42
    min_train_days: int = 20
    min_test_days: int = 10


@dataclass(frozen=True)
class BacktestCostConfig:
    """Default transaction-cost assumption for the Backtesting Simulator —
    a flat basis-point charge on every entry/exit, not a full market-impact
    model (no size/liquidity/volatility dependency). On by default at a
    sensible value rather than zero: 10 bps (0.10%) per leg is a common
    round-number stand-in for commission + bid/ask spread + slippage on a
    liquid large-cap name, deliberately not fine-tuned per ticker since
    there's no live execution-cost data source in this environment to
    validate a more precise number against — the same disclosed-assumption
    convention as every other judgment-call constant in this file."""
    default_cost_bps: float = 10.0
    max_cost_bps: float = 100.0


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
class CompetitiveBenchmarkingConfig:
    """Outperform/laggard flagging for the Peer Competitor Matrix's
    per-metric comparison. A peer sitting close to the group average on a
    metric isn't meaningfully outperforming OR lagging — it's just near
    the pack — so a flag only fires once a value is at least this many
    percent away from the group average, in the favorable or unfavorable
    direction for that metric. Not a statistical significance test (peer
    groups here are far too small, typically 2-6 names, for one to be
    meaningful) — a disclosed, simple distance threshold, not a
    fabricated p-value."""
    outperform_threshold_pct: float = 10.0
    max_peers: int = 6


@dataclass(frozen=True)
class OnboardingConfig:
    """The first-run guided walkthrough. "First-time user" here means
    "onboarding hasn't been completed/skipped on THIS local instance" —
    a single local flag file, not genuine per-browser/per-visitor
    detection. When auth is configured this flag is per signed-in user, so
    each person gets the walkthrough once. Signed out — or on an instance
    with no provider set up — it stays a single instance-wide flag, so
    once anyone completes or skips it, it won't auto-trigger again for
    anyone else sharing that signed-out profile —
    "Replay Tutorial" in the sidebar's System tab is how it's seen again
    on purpose. See onboarding.py's module docstring for why this is a
    native step-by-step panel rather than a spotlight-style overlay tour:
    a real, already-proven constraint in this codebase, not a stylistic
    choice."""
    state_filename: str = "onboarding_state.json"


@dataclass(frozen=True)
class ThemeConfig:
    """Dark/light mode for the app's own chrome (the OLED CSS injection
    and Plotly chart templates) — NOT the CIO Tear Sheet, which is a
    deliberately-white printed-report facsimile independent of the app
    theme (see theme.py). Persisted the same way every other cross-restart
    preference in this app is: a single local JSON file, scoped per
    signed-in user when auth is configured (see auth.py)."""
    state_filename: str = "theme_state.json"
    default_theme: str = "dark"


@dataclass(frozen=True)
class CollaborationConfig:
    """Per-ticker notes with @-mentions.

    Identity may be either SELF-DECLARED or AUTHENTICATED, and the
    difference is carried all the way into the notification email rather
    than being flattened away. Before auth.py existed every author name
    was a claim; now a note written while signed in via OIDC carries a
    verified name. A reader deciding how much weight to put on "Ana says
    sell" needs to know which of those they're looking at, so the two
    cases get two different closing lines instead of one hedged one.

    Mentions resolve only against the roster the user curates, which is
    what bounds who the app can ever email."""
    store_filename: str = "collaboration_store.json"
    max_members: int = 25
    max_note_chars: int = 2000
    mention_subject_template: str = "{author} mentioned you in a Quantix note on {ticker}"
    mention_body_template: str = (
        "Hi {name},\n\n"
        "{author} mentioned you in a note on {ticker} in Quantix:\n\n"
        "  {body}\n\n"
        "Open Quantix to reply. {identity_note}"
    )
    # Notes written while signed out, on an instance with auth off, or by
    # anyone who simply typed a name into the box.
    identity_note_self_declared: str = (
        "Note that Quantix has no user accounts — the author name above is "
        "self-declared rather than authenticated."
    )
    # Notes written while signed in. The issuer is named because "verified"
    # is only meaningful if you know who did the verifying.
    identity_note_authenticated: str = (
        "The author was signed in when they wrote this, so the name above is a "
        "verified identity from {issuer} rather than a self-declared label."
    )


@dataclass(frozen=True)
class PortfolioConfig:
    """Actual holdings and the performance dashboard built on them.

    A HOLDING CARRIES ITS PURCHASE DATE, and that is the load-bearing
    decision. With only ticker+shares there is no way to know what was
    held last year, so a "return over time" chart would have to assume
    today's basket was held for the whole window — back-projecting
    today's winners onto the past and flattering the result. Recording
    when each position started lets the series begin each holding on its
    own date instead.

    TIME-WEIGHTED RETURN HEADLINES THE BENCHMARK COMPARISON. Positions
    opened at different times are cash flows, and money-weighted return
    is affected by their timing — so comparing it against an index's
    return is apples-to-oranges. TWR strips that out, which is what
    makes "did my picks beat the S&P 500" a fair question. Money-weighted
    is shown alongside, because it is the truer answer to "what did my
    money actually do".

    SHAPED FOR MULTIPLE PORTFOLIOS, holding one. The store mirrors
    WatchlistStore's active/lists layout so the separate Multi-Portfolio
    Management task becomes a UI change rather than a migration.
    """
    store_filename: str = "portfolio_store.json"
    default_portfolio_name: str = "My Portfolio"
    default_benchmark: str = "SPY"
    max_holdings: int = 50
    max_portfolios: int = 10
    max_name_chars: int = 40
    # Below this many aligned trading days, an annualised figure is
    # extrapolation rather than measurement. Matches the floor
    # historical_comparison.py and the risk endpoint already use.
    min_observations: int = 30


@dataclass(frozen=True)
class DigestConfig:
    """The weekly email digest.

    IT RUNS AS A SEPARATE PROCESS ON AN OS SCHEDULER, NOT INSIDE THE APP.
    The originating task wants a digest that reaches an INACTIVE user
    "without requiring them to check manually" — so by definition it has
    to run while Streamlit is shut. Streamlit executes nothing when no
    browser tab is open (realtime_alerts.py documents the same limit for
    in-tab alert polling), so digest.py is a standalone script driven by
    cron or launchd. The app generates the schedule line; installing it
    stays the user's deliberate act, because a job that mails out on its
    own should never be arranged silently.

    THE SETTINGS STORE IS SHARED, KEYED BY OWNER — the same reasoning as
    api_keys.py. A cron-run script has no Streamlit session, so
    auth.current_user() is None inside it; settings filed under a user's
    namespace would be invisible to the process that has to read them.
    Each record carries owner_key instead, which also lets ONE scheduled
    run send every configured user's digest.

    NO PORTFOLIO SECTION. The task also asks for "portfolio changes", but
    nothing in Quantix stores holdings — there is no positions store, and
    designing one belongs to the Multi-Portfolio Management task. The
    digest says so in plain words rather than dressing the watchlist up as
    a portfolio, which would misstate what the numbers mean.
    """
    store_filename: str = "digest_store.json"
    default_period_days: int = 7
    min_period_days: int = 1
    max_period_days: int = 90
    # Watchlist rows shown before the digest truncates. A digest long
    # enough to scroll defeats the point of a digest.
    max_movers_shown: int = 25
    max_alerts_shown: int = 20
    subject_template: str = "Quantix digest — {period} — {headline}"


@dataclass(frozen=True)
class SupportConfig:
    """In-app help and support.

    NO LIVE CHAT, DELIBERATELY. The originating task asked for a "chat/help
    widget" to "reduce support response time". A chat box implies someone
    is staffing it; on a locally-run instance nobody is, and an unanswered
    chat is worse than no chat because it makes a promise the app cannot
    keep. So this is self-serve search first, with an explicit outbound
    report for what search can't answer.

    The help corpus is NOT duplicated here — it is assembled from the 57
    metric definitions and 13 chart explanations already in metric_help.py
    plus the FAQ in support.py. One source per fact, so help text cannot
    drift from the tooltips describing the same metric.
    """
    # Where a support report is emailed. Empty means "no destination
    # configured", which is the honest default: this app ships with no
    # support organisation behind it, and inventing an address would send
    # someone's bug report into a void.
    support_address: str = ""
    max_subject_chars: int = 120
    max_body_chars: int = 4000
    # Log lines attached to a report when the user opts in. Enough to show
    # what led to a failure without shipping the whole file.
    diagnostics_log_lines: int = 50
    search_results_shown: int = 6
    categories: Tuple[str, ...] = ("Question", "Bug report", "Feature request", "Data looks wrong")


@dataclass(frozen=True)
class ApiKeysConfig:
    """Scoped API keys for programmatic (non-human) access to Quantix.

    THE STORE IS SHARED, NOT PER-USER, AND THAT IS DELIBERATE. Every other
    piece of personal state is namespaced per signed-in user (see auth.py).
    Keys cannot be: the API server is a separate process with no Streamlit
    session, so auth.current_user() is None inside it. A key filed under a
    user's namespace would be invisible to the exact process whose job is
    to verify it. So the store is shared and each record carries the
    owner's namespace key instead — which is also what lets an
    owner-scoped endpoint resolve the right user's watchlists.

    ONLY THE HASH IS STORED. The secret is shown once, at creation, and
    never again — recoverable keys are the single most common way key
    systems leak, and a store that cannot reveal a key cannot leak one.

    READ-ONLY BY DESIGN. The originating task mentions "headless trading
    bots"; Quantix has no brokerage integration and this API deliberately
    exposes no write or trade path at all. Every scope below is a read.
    """
    store_filename: str = "api_keys_store.json"
    key_prefix: str = "qtx"
    # 32 bytes of urlsafe randomness. Long enough that online guessing is
    # hopeless and offline guessing is irrelevant against a hashed store.
    secret_bytes: int = 32
    id_length: int = 8          # public, shown in the UI to identify a key
    max_keys_per_owner: int = 20
    default_expiry_days: int = 90
    max_expiry_days: int = 365
    max_name_chars: int = 60
    # Bound to loopback unless deliberately changed. This app is run
    # locally; a default of 0.0.0.0 would silently publish someone's
    # financial analysis to their whole network the first time they
    # started the server.
    default_host: str = "127.0.0.1"
    default_port: int = 8787   # Streamlit owns 8501; keep well clear of it


@dataclass(frozen=True)
class ThresholdsConfig:
    """Where the user's overrides of the shipped valuation/risk thresholds
    live. The DEFAULTS stay in ScorecardConfig/RiskConfig above, with the
    reasoning for each; this only names the store file. See
    user_thresholds.py for which subset is editable and why the rest
    deliberately isn't."""
    store_filename: str = "threshold_overrides.json"


@dataclass(frozen=True)
class FavoritesConfig:
    """Favorites (starred tickers) + recently-viewed, the "quick access"
    strip under the symbol header.

    Deliberately SEPARATE from the sidebar's named watchlists rather than
    reusing them: watchlists are multiple, named, curated baskets
    ("Dividend Payers", "Tech"), so a pin action against whichever list
    happens to be active would be context-dependent and would dirty a
    deliberately-curated basket. Favorites are one flat set that means
    "always show me these, whichever watchlist I'm in." That split was
    settled with the user before this was built, not assumed.

    max_favorites / max_chips are UI bounds, not storage ones: everything
    renders as ONE row of chips, so these caps are really "how many
    equal-width buttons still fit legibly across that row". Tuned
    in-browser rather than guessed, twice: at max_chips=10 the per-chip
    column came out ~87px and labels like "★ AAPL" wrapped mid-word, and
    after the buttons gained a 1px themed border the same thing happened
    again at 8. Hence 7, which leaves ~86px of usable label width — a
    starred five-letter ticker ("★ GOOGL", the longest realistic label)
    fits on one line. The chips are also nowrap in CSS, so any future
    squeeze degrades to an ellipsis rather than back to a mid-word break.

    max_favorites is held below max_chips on purpose, so a fully-starred
    row still leaves room for a couple of recents rather than crowding
    them out entirely."""
    store_filename: str = "favorites_store.json"
    max_favorites: int = 5
    max_chips: int = 7  # total favorites + recents chips rendered at once


@dataclass(frozen=True)
class EmailReportConfig:
    """Non-secret defaults for emailing the CIO Tear Sheet PDF. The SMTP
    server/credentials themselves are deliberately NOT here — they're
    read at send time from Streamlit secrets (.streamlit/secrets.toml)
    or environment variables, never hardcoded or persisted by this app
    (see email_report.py)."""
    default_subject_template: str = "Quantix Tear Sheet — {ticker} ({date})"
    default_body_template: str = (
        "Attached is the Quantix Institutional Tear Sheet for {ticker}, generated {date}.\n\n"
        "Algorithmic execution carries inherent risk. Verify all execution parameters via broker."
    )


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
WALK_FORWARD = WalkForwardConfig()
WATCHLIST_PANEL = WatchlistPanelConfig()
BACKTEST_COST = BacktestCostConfig()
REALTIME_ALERTS = RealtimeAlertsConfig()
PORTFOLIO_BACKTEST = PortfolioBacktestConfig()
ML_PIPELINE = MLPipelineConfig()
SCENARIO_MODELING = ScenarioModelingConfig()
COMPETITIVE_BENCHMARKING = CompetitiveBenchmarkingConfig()
ONBOARDING = OnboardingConfig()
THEME = ThemeConfig()
THRESHOLDS = ThresholdsConfig()
COLLABORATION = CollaborationConfig()
FAVORITES = FavoritesConfig()
API_KEYS = ApiKeysConfig()
SUPPORT = SupportConfig()
DIGEST = DigestConfig()
PORTFOLIO = PortfolioConfig()
EMAIL_REPORT = EmailReportConfig()
CHART_DEFAULTS = ChartDefaults()
TECHNICAL = TechnicalConfig()
PEER_DEFAULTS = PeerDefaults()
TEAR_SHEET = TearSheetConfig()
OUTLIER_BOUNDS = OutlierBoundsConfig()
QUALITY = QualityConfig()
