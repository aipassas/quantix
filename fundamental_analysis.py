"""Fundamental Analysis Engine for Quantix.

Every financial ratio derived from a company's financial statements is
calculated here, in one place, instead of being computed inline throughout
finance.py. The dashboard consumes the results and renders them; it performs
no ratio arithmetic of its own.

See FUNDAMENTALS.md for the full reference: every ratio's formula, null
handling, assumptions, accounting/valuation standard citations where
applicable (Altman Z-Score, CAPM), and a guide for extending this module.
The summary below is a quick lookup, not a replacement for it.

Scope is deliberately statement-derived fundamentals — profitability,
leverage, liquidity, cash-flow quality, valuation, distress (Altman Z) and
discounted cash flow. Metrics computed from *price returns* rather than
financial statements (Sharpe, Sortino, VaR, CVaR, drawdown) belong to the
risk analytics layer and are intentionally not here.

Adding a new ratio means adding one MetricCheck in _build_checks(): it then
appears in the Master Matrix and, if flagged, counts toward the Scorecard,
without touching the dashboard.

The Strategic Investment Scorecard's Blueprint Alignment score is weighted
(see FundamentalMetrics.score_pct and config.SCORECARD.weights) — core
financial-health signals (profitability, leverage, capital efficiency) count
for more than secondary ones (valuation multiples, volatility) — and
sector-adjusted for Debt-to-Equity specifically (see
SCORECARD.max_debt_to_equity_for and _build_checks()). A metric with no
computable value for a given company is excluded from the score entirely
rather than counted as a failure (FundamentalMetrics.evaluable_checks), so
sectors with different reporting norms (e.g. banks lacking a classified
balance sheet) aren't penalized just for missing fields that were never
going to be reported.

Formula reference
-----------------
Profitability (validated against Yahoo's own reported ratios — see
validate_profitability()):
    Net Margin        = Net Income / Total Revenue
    Gross Margin      = Gross Profit / Total Revenue           (absent for banks/financials — no cost of revenue)
    Operating Margin  = Operating Income / Total Revenue        (absent for banks/financials)
    ROA               = Net Income / Total Assets
    ROE               = Net Income / Stockholders Equity        (can be extreme/negative for heavy-buyback
                                                                  companies with small or negative book equity —
                                                                  mathematically correct, not a bug)
ROIC              = NOPAT / (Total Debt + Stockholders Equity)
                    where NOPAT = EBIT × (1 − effective tax rate)
                    effective tax rate = Tax Provision / Pretax Income when Pretax Income > 0,
                    else the assumed statutory rate (config.DCF.tax_rate). Tax-adjusting EBIT
                    is the standard institutional ROIC definition — using raw EBIT overstates
                    return by counting tax as available to capital providers.
Liquidity (validated against Yahoo's own reported ratios — see
validate_liquidity(); informational only, does not feed the Scorecard/Matrix):
    Current Ratio (computed) = Current Assets / Current Liabilities
    Quick Ratio              = (Current Assets − Inventory) / Current Liabilities
Leverage (validated — see validate_leverage()):
    Debt-to-Equity    = Total Debt / Stockholders Equity (statement-computed;
                        this IS the canonical value used in the Scorecard/Matrix
                        — Yahoo's debtToEquity is only a fallback when
                        Stockholders Equity itself isn't reported)
Interest Coverage = EBIT / |Interest Expense|
FCF Yield         = Free Cash Flow / Market Cap
Valuation (validated against Yahoo's own reported ratios — see
validate_valuation(); P/E and Price-to-Book stay Yahoo-sourced as canonical,
this only cross-checks them — EV/EBITDA is a new metric with no prior value):
    P/E (computed)    = Market Cap / Net Income              (None for non-positive earnings — not a
                                                                meaningful multiple, same for Price-to-Book below)
    Price-to-Book     = Market Cap / Stockholders Equity      (None for non-positive book value)
    EV/EBITDA         = Enterprise Value / EBITDA
                        EV     = Market Cap + Total Debt − Cash & Equivalents
                        EBITDA = EBIT + Depreciation & Amortization
                        (None when EBITDA is non-positive, same reasoning as P/E)
Altman Z (public manufacturing, 1968):
    Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 0.999·X5
    X1 = (Current Assets − Current Liabilities) / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets                  (raw EBIT per the original Altman definition — NOT NOPAT)
    X4 = Market Cap / Total Liabilities
    X5 = Total Revenue / Total Assets
WACC (CAPM)       = E/(E+D)·[rf + β(Rm − rf)] + D/(E+D)·[(IntExp/Debt)(1 − tax)]
Intrinsic value   = 2-stage DCF: PV of N years of FCF growth, plus a Gordon
                    Growth terminal value, divided by shares outstanding.
Margin of safety  = (Intrinsic − Market Price) / Intrinsic

P/E, PEG, price/book, current ratio and beta are reported by the data
provider and arrive already normalized from financial_standardization.py;
debt/equity is statement-computed there instead (see above). All are
surfaced here so that every metric the dashboard shows has a single origin.
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from config import DCF, OUTLIER_BOUNDS, QUALITY, RISK, SCORECARD, WATCHLIST
from financial_standardization import StandardizedFinancials, normalize_debt_to_equity
from logging_setup import get_logger, log_event

logger = get_logger("fundamentals")


def _fmt(value: Optional[float], suffix: str = "", decimals: int = 2) -> str:
    """Display string for a metric, or 'N/A' when the input was unavailable."""
    return "N/A" if value is None else f"{value:.{decimals}f}{suffix}"


@dataclass
class MetricCheck:
    """One evaluated metric: its value, how to show it, and whether it passes.

    `in_scorecard` and `in_matrix` are separate flags on purpose — a metric
    can be scored without being shown as a Matrix row, or vice versa (FCF
    Yield is a Matrix row but not one of the 8 scoreboard flags). Keeping
    both flags explicit preserves that distinction instead of burying it in
    the dashboard.
    """
    key: str
    category: str
    label: str
    value: Optional[float]
    display: str
    benchmark: str
    passed: Optional[bool]      # None = not evaluable (missing data)
    in_scorecard: bool = True
    in_matrix: bool = True
    weight: float = 1.0         # relative contribution to the weighted Blueprint Alignment score; see config.SCORECARD.weights

    @property
    def status_icon(self) -> str:
        if self.passed is None:
            return "⚪"
        return "🟢" if self.passed else "🔴"


@dataclass
class ProfitabilityCheck:
    """One statement-derived ratio, independently computed from raw
    statement data and cross-checked against Yahoo's own separately-reported
    ratio for the same concept (grossMargins, operatingMargins,
    returnOnAssets, returnOnEquity, profitMargins, currentRatio, quickRatio)
    — the practical substitute here for reconciling against a real annual
    report. Despite the name, also used for liquidity ratios (see
    validate_liquidity()); `suffix` controls whether values render as a
    percentage ("%") or a plain ratio ("").

    `reference_pct` is None when Yahoo doesn't report an equivalent (ROIC has
    no such field) or the metric is structurally absent for this company
    (Gross/Operating Margin for banks) — in which case `agrees` is also None,
    meaning "not evaluable," not "failed."
    """
    key: str
    label: str
    formula: str
    computed_pct: Optional[float]
    reference_pct: Optional[float]
    agrees: Optional[bool]
    suffix: str = "%"

    @property
    def computed_display(self) -> str:
        return _fmt(self.computed_pct, self.suffix)

    @property
    def reference_display(self) -> str:
        return _fmt(self.reference_pct, self.suffix)

    @property
    def status_icon(self) -> str:
        if self.agrees is None:
            return "⚪"
        return "🟢" if self.agrees else "🟡"


@dataclass
class OutlierFlag:
    """One metric whose computed value exceeds its configured sanity bound
    (config.OUTLIER_BOUNDS) — a distinct signal from ProfitabilityCheck's
    `agrees`: a value can cross-check cleanly against Yahoo and still be an
    outlier (both sources could share the same underlying data error), or
    disagree with Yahoo while being perfectly plausible on its own."""
    key: str
    category: str
    label: str
    display: str
    note: str


@dataclass
class MetricsValidationSummary:
    """Consolidated view across every metric this engine validates —
    Profitability, Liquidity, Leverage, and Valuation — plus outlier and
    incomplete-calculation flags that don't belong to any single category.
    Built by validate_all_metrics(); powers the Financial Metrics Validation
    Report (an overview) alongside the four detailed per-category reports
    (which this does not replace).
    """
    checks: List[Tuple[str, ProfitabilityCheck]]  # (category, check)
    outliers: List[OutlierFlag] = field(default_factory=list)
    fallback_notes: List[str] = field(default_factory=list)

    @property
    def evaluated_checks(self) -> List[Tuple[str, ProfitabilityCheck]]:
        return [(cat, c) for cat, c in self.checks if c.computed_pct is not None]

    @property
    def disagreements(self) -> List[Tuple[str, ProfitabilityCheck]]:
        return [(cat, c) for cat, c in self.checks if c.agrees is False]

    @property
    def disagreement_count(self) -> int:
        return len(self.disagreements)

    @property
    def outlier_count(self) -> int:
        return len(self.outliers)

    @property
    def fallback_count(self) -> int:
        return len(self.fallback_notes)

    @property
    def total_issues(self) -> int:
        return self.disagreement_count + self.outlier_count + self.fallback_count

    @property
    def is_clean(self) -> bool:
        return self.total_issues == 0


# A computed ratio and Yahoo's own reported ratio can legitimately differ —
# Yahoo's figure is often trailing-twelve-month while ours is the most recent
# annual period — so agreement uses a relative tolerance rather than an exact
# match. 15% keeps this a genuine correctness check without flagging normal
# TTM-vs-annual drift as a formula error. Shared by both the profitability and
# liquidity validation reports.
_AGREEMENT_TOLERANCE = 0.15


def _values_agree(computed: Optional[float], reference: Optional[float]) -> Optional[bool]:
    if computed is None or reference is None:
        return None
    denominator = max(abs(reference), 1e-9)
    # bool(...) so this is always a native Python bool, never numpy.bool_ —
    # computed/reference are often numpy.float64 (pandas Series extraction),
    # and an `is False`/`is True` check downstream would silently never match
    # a numpy.bool_ result.
    return bool(abs(computed - reference) / denominator <= _AGREEMENT_TOLERANCE)


# Per-metric sanity bound from config.OUTLIER_BOUNDS, and whether the bound
# applies to the magnitude (symmetric — the metric can be legitimately very
# negative, e.g. FCF Yield) or only to the raw value (e.g. Current Ratio,
# which is never meaningfully negative). Metrics not listed here (Total Debt,
# Interest Coverage — see validate_leverage()) have no defined bound: a very
# high Interest Coverage is always good news, not a red flag, and Total Debt
# is a raw dollar amount rather than a bounded ratio.
_OUTLIER_BOUNDS_TABLE = {
    "net_margin": (OUTLIER_BOUNDS.max_abs_net_margin_pct, True),
    "gross_margin": (OUTLIER_BOUNDS.max_gross_margin_pct, False),
    "operating_margin": (OUTLIER_BOUNDS.max_operating_margin_pct, False),
    "roa": (OUTLIER_BOUNDS.max_abs_roa_pct, True),
    "roe": (OUTLIER_BOUNDS.max_abs_roe_pct, True),
    "roic": (OUTLIER_BOUNDS.max_abs_roic_pct, True),
    "current_ratio": (OUTLIER_BOUNDS.max_current_ratio, False),
    "quick_ratio": (OUTLIER_BOUNDS.max_quick_ratio, False),
    "debt_to_equity": (OUTLIER_BOUNDS.max_debt_to_equity, False),
    "pe_ratio": (OUTLIER_BOUNDS.max_pe_ratio, False),
    "price_to_book": (OUTLIER_BOUNDS.max_price_to_book, False),
    "peg_ratio": (OUTLIER_BOUNDS.max_abs_peg_ratio, True),
    "ev_ebitda": (OUTLIER_BOUNDS.max_abs_ev_ebitda, True),
    "fcf_yield": (OUTLIER_BOUNDS.max_abs_fcf_yield_pct, True),
}


def _outlier_note(key: str, computed: Optional[float]) -> Optional[str]:
    """None if the value is missing, has no defined bound, or is within it;
    otherwise a short note naming the bound that was exceeded."""
    if computed is None:
        return None
    bound = _OUTLIER_BOUNDS_TABLE.get(key)
    if bound is None:
        return None
    limit, symmetric = bound
    magnitude = abs(computed) if symmetric else computed
    if magnitude <= limit:
        return None
    sign = "±" if symmetric else ""
    return f"magnitude exceeds the sanity bound of {sign}{limit:g} — verify this figure before relying on it"


def _linear_score(value: Optional[float], band: Tuple[float, float]) -> Optional[float]:
    """0-100, clamped, linear between band[0] (-> 0) and band[1] (-> 100).
    Pass band[0] > band[1] for a metric where lower is better (e.g.
    Debt-to-Equity) — the interpolation direction inverts automatically."""
    if value is None:
        return None
    lo, hi = band
    if hi == lo:
        return None
    pct = (value - lo) / (hi - lo) * 100
    return max(0.0, min(100.0, pct))


def _ideal_score(value: Optional[float], ideal: float) -> Optional[float]:
    """100 at `ideal`, falling off proportionally to relative distance in
    either direction — used for Valuation metrics, where the goal is
    "reasonably priced" rather than "as cheap as possible" (see
    config.QualityConfig's docstring for the rationale)."""
    if value is None or ideal == 0:
        return None
    score = 100 * (1 - abs(value - ideal) / abs(ideal))
    return max(0.0, min(100.0, score))


@dataclass
class QualityFactorMetric:
    """One input metric behind a QualityFactor's score."""
    label: str
    value: Optional[float]
    display: str
    sub_score: Optional[float]  # 0-100, or None if this metric wasn't computable


@dataclass
class QualityFactor:
    """One of the five dimensions (Profitability, Financial Stability,
    Growth, Valuation, Capital Efficiency) behind the overall Company
    Quality score — see CompanyQuality and classify_company_quality()."""
    name: str
    weight: float
    metrics: List[QualityFactorMetric] = field(default_factory=list)

    @property
    def evaluable_metrics(self) -> List[QualityFactorMetric]:
        return [m for m in self.metrics if m.sub_score is not None]

    @property
    def score(self) -> Optional[float]:
        """Average of this factor's evaluable metric sub-scores, or None if
        none of them were computable — excluded metrics don't drag the
        average down, same principle as the Scorecard's missing-data handling."""
        evaluable = self.evaluable_metrics
        if not evaluable:
            return None
        return sum(m.sub_score for m in evaluable) / len(evaluable)


@dataclass
class CompanyQuality:
    """Multi-factor company quality classification — a complementary,
    differently-framed view from the Strategic Investment Scorecard (which
    stays a pass/fail checklist against configured thresholds). This blends
    five weighted factors into one 0-100 score and category.
    """
    factors: List[QualityFactor] = field(default_factory=list)

    @property
    def evaluable_factors(self) -> List[QualityFactor]:
        return [f for f in self.factors if f.score is not None]

    @property
    def overall_score(self) -> Optional[float]:
        """Weighted average over evaluable factors only — a factor with zero
        computable metrics (e.g. Growth with no Yahoo growth figure) is
        excluded entirely rather than scored as 0, and the remaining
        factors' weights are implicitly renormalized by the division below."""
        evaluable = self.evaluable_factors
        total_weight = sum(f.weight for f in evaluable)
        if not total_weight:
            return None
        return sum((f.score or 0) * f.weight for f in evaluable) / total_weight

    @property
    def category(self) -> str:
        score = self.overall_score
        if score is None:
            return "Not Ratable"
        if score >= QUALITY.elite_min_score:
            return "Elite Quality"
        if score >= QUALITY.high_min_score:
            return "High Quality"
        if score >= QUALITY.average_min_score:
            return "Average Quality"
        if score >= QUALITY.below_average_min_score:
            return "Below Average"
        return "Weak Quality"

    @property
    def category_icon(self) -> str:
        return {
            "Elite Quality": "🟢", "High Quality": "🟢", "Average Quality": "🟡",
            "Below Average": "🟠", "Weak Quality": "🔴", "Not Ratable": "⚪",
        }[self.category]


@dataclass
class WatchlistScore:
    """Result of the fast 4-point basket pre-screen (see screen_watchlist)."""
    ticker: str
    score: float
    status: str
    pe_ratio: float
    net_margin_pct: float


@dataclass
class DCFResult:
    """Outcome of the multi-stage DCF. `ok` False means it could not be run."""
    ok: bool
    reason: Optional[str] = None
    wacc: Optional[float] = None
    current_price: Optional[float] = None
    intrinsic_price: Optional[float] = None
    margin_of_safety_pct: Optional[float] = None
    status: Optional[str] = None          # Strong Buy / Buy / Overvalued
    status_color: Optional[str] = None    # streamlit delta_color
    beta: Optional[float] = None
    beta_source: Optional[str] = None     # "regressed" / "yahoo_reported" / "market_assumption"
    beta_r_squared: Optional[float] = None  # only set when beta_source == "regressed"


@dataclass
class FundamentalMetrics:
    """Every statement-derived metric for one ticker."""
    ticker: str
    sector: Optional[str]  # drives the sector-adjusted Debt-to-Equity threshold in _build_checks()

    # Provider-reported, normalized upstream
    net_margin: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    pe_ratio: Optional[float]
    peg_ratio: Optional[float]
    beta: Optional[float]

    # Calculated here
    roic_pct: Optional[float]
    interest_coverage: Optional[float]
    fcf_yield_pct: Optional[float]

    # Profitability — independently computed from statements (see
    # validate_profitability() for the full cross-checked report)
    gross_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    roa_pct: Optional[float] = None
    profitability_checks: List[ProfitabilityCheck] = field(default_factory=list)

    # Liquidity — independently computed from statements (see
    # validate_liquidity() for the full cross-checked report). Current Ratio
    # itself is still the provider-reported value above; these are purely
    # for the validation report.
    liquidity_checks: List[ProfitabilityCheck] = field(default_factory=list)

    # Leverage — Debt-to-Equity here IS the value in `debt_to_equity` above
    # (statement-computed, see financial_standardization.py); this list adds
    # Total Debt source verification and Interest Coverage documentation
    # alongside it (see validate_leverage()).
    leverage_checks: List[ProfitabilityCheck] = field(default_factory=list)

    # Valuation — P/E and Price-to-Book here stay Yahoo-sourced canonically
    # (standardized.pe_ratio / standardized.price_to_book); this list adds
    # independent cross-checks plus EV/EBITDA (new metric) and PEG/FCF Yield
    # documentation (see validate_valuation()).
    valuation_checks: List[ProfitabilityCheck] = field(default_factory=list)

    # Consolidated overview across all four categories above, plus outlier
    # detection and incomplete-calculation flags — see validate_all_metrics().
    metrics_validation: Optional[MetricsValidationSummary] = None

    # Multi-factor quality classification (Profitability, Financial
    # Stability, Growth, Valuation, Capital Efficiency) — a complementary,
    # differently-framed view from the Scorecard/Matrix above, not a
    # replacement. See classify_company_quality().
    company_quality: Optional[CompanyQuality] = None

    # Distress
    altman_z: Optional[float] = None
    altman_verdict: str = "N/A"
    altman_missing_inputs: List[str] = field(default_factory=list)

    # Evaluated checks driving both the Scorecard and the Master Matrix
    checks: List[MetricCheck] = field(default_factory=list)

    @property
    def scorecard_checks(self) -> List[MetricCheck]:
        return [c for c in self.checks if c.in_scorecard]

    @property
    def matrix_checks(self) -> List[MetricCheck]:
        return [c for c in self.checks if c.in_matrix]

    @property
    def evaluable_checks(self) -> List[MetricCheck]:
        """Scorecard checks with a computable value. A metric that's
        structurally unavailable for this company (e.g. ROIC for a bank with
        no reported EBIT) is excluded here rather than counted as a failure
        — otherwise companies in sectors with different reporting norms
        would be silently penalized for data gaps, not genuine weaknesses."""
        return [c for c in self.scorecard_checks if c.passed is not None]

    @property
    def green_flags(self) -> int:
        return sum(1 for c in self.evaluable_checks if c.passed)

    @property
    def total_checks(self) -> int:
        return len(self.evaluable_checks)

    @property
    def score_pct(self) -> float:
        """Weighted Blueprint Alignment score (0-100) over evaluable checks
        only — see config.SCORECARD.weights. Not a simple green_flags/
        total_checks average: core financial-health metrics count for more
        than secondary ones (valuation, volatility)."""
        total_weight = sum(c.weight for c in self.evaluable_checks)
        if not total_weight:
            return 0.0
        passed_weight = sum(c.weight for c in self.evaluable_checks if c.passed)
        return (passed_weight / total_weight) * 100

    @property
    def alignment_verdict(self) -> str:
        if self.score_pct >= SCORECARD.high_alignment_pct:
            return "high"
        if self.score_pct >= SCORECARD.moderate_alignment_pct:
            return "moderate"
        return "low"


class FundamentalAnalysisEngine:
    """Calculates every statement-derived metric for one standardized ticker.

    Construct with a StandardizedFinancials, then call analyze() for the
    ratios/scorecard and run_dcf() for the valuation model (kept separate
    because the DCF is parameterized by the user's growth-rate input).

    `raw_info` is optional and used ONLY by validate_profitability() to
    cross-check our computed ratios against Yahoo's own separately-reported
    ones (grossMargins, operatingMargins, returnOnAssets, returnOnEquity,
    profitMargins). It's intentionally not folded into StandardizedFinancials
    — that module holds one canonical value per concept, whereas this is a
    second, independent reference for comparison, not a source of truth.
    """

    def __init__(self, standardized: StandardizedFinancials, raw_info: Optional[dict] = None):
        self.std = standardized
        self._info = raw_info or {}

    # ----- individual ratios -------------------------------------------------

    def _computed_effective_tax_rate(self) -> Optional[float]:
        """Tax Provision / Pretax Income, or None when the inputs are
        missing or produce an implausible rate (one-time items can push
        this outside [0, 50%]) — the None case is what effective_tax_rate()
        falls back on, and effective_tax_rate_used_fallback() reports."""
        pretax, tax = self.std.pretax_income, self.std.tax_provision
        if pretax is None or tax is None or pretax <= 0:
            return None
        rate = tax / pretax
        return rate if 0 <= rate <= 0.5 else None

    def effective_tax_rate(self) -> float:
        """Tax Provision / Pretax Income when available and sane, else the
        assumed statutory rate (config.DCF.tax_rate). A company's own
        reported effective rate is more accurate than a flat assumption —
        multinational effective rates routinely sit below the US statutory
        21% — but is only trustworthy when Pretax Income is positive and the
        resulting rate falls in a plausible range; one-time items can
        otherwise produce a nonsensical ratio.
        """
        rate = self._computed_effective_tax_rate()
        return rate if rate is not None else DCF.tax_rate

    def effective_tax_rate_used_fallback(self) -> bool:
        """True when effective_tax_rate() had to fall back to the assumed
        statutory rate — surfaced by the Financial Metrics Validation Report
        as an incomplete-calculation flag, since ROIC is then built on an
        assumption rather than this company's own reported tax rate."""
        return self._computed_effective_tax_rate() is None

    def roic_pct(self) -> Optional[float]:
        """NOPAT / (Total Debt + Stockholders Equity).

        NOPAT (Net Operating Profit After Tax) = EBIT × (1 − effective tax
        rate). This is the standard institutional ROIC definition — using
        raw, pre-tax EBIT (as an earlier version of this engine did) counts
        tax as if it were available to capital providers, overstating the
        return investors actually capture.
        """
        ebit, equity = self.std.ebit, self.std.stockholders_equity
        debt = self.std.total_debt  # optional field: already 0 when unreported
        if ebit is None or equity is None or (debt + equity) == 0:
            return None
        nopat = ebit * (1 - self.effective_tax_rate())
        return (nopat / (debt + equity)) * 100

    def interest_coverage(self) -> Optional[float]:
        ebit, interest = self.std.ebit, self.std.interest_expense
        if ebit is None or interest is None or abs(interest) == 0:
            return None
        return ebit / abs(interest)

    def fcf_yield_pct(self) -> Optional[float]:
        fcf, mcap = self.std.free_cash_flow, self.std.market_cap
        if fcf is None or not mcap:
            return None
        return (fcf / mcap) * 100

    # ----- distress ----------------------------------------------------------

    def altman_z_score(self):
        """Altman Z-Score plus its verdict and any missing inputs.

        Returns (z, verdict, missing_inputs). z is None when the score can't
        be computed — several legitimate company types (notably banks) don't
        report a classified balance sheet, so this is an expected outcome
        rather than an error.
        """
        s = self.std
        inputs = {
            "Total Assets": s.total_assets,
            "Current Assets": s.current_assets,
            "Current Liabilities": s.current_liabilities,
            "EBIT / Operating Income": s.ebit,
            "Total Revenue": s.total_revenue,
            "Market Cap": s.market_cap,
            "Total Liabilities (Net Minority Interest)": s.total_liabilities,
        }
        missing = [name for name, value in inputs.items() if value is None]

        # Total Assets/Liabilities are denominators in every X-ratio — zero or
        # negative values are accounting nonsense (never legitimately absent
        # like a bank's missing classified balance sheet), so they're surfaced
        # as their own descriptive reason rather than silently producing an
        # empty `missing` list the UI would have nothing to explain with.
        if s.total_assets is not None and s.total_assets <= 0:
            missing.append(f"Total Assets (reported as {s.total_assets:,.0f} — must be positive)")
        if s.total_liabilities is not None and s.total_liabilities <= 0:
            missing.append(f"Total Liabilities (reported as {s.total_liabilities:,.0f} — must be positive)")

        if missing:
            log_event(
                logger, logging.WARNING, "calc.skipped", section="altman_z_score",
                ticker=s.ticker, missing=", ".join(missing),
            )
            return None, "Insufficient Financial Data", missing

        working_capital = s.current_assets - s.current_liabilities
        x1 = working_capital / s.total_assets
        x2 = s.retained_earnings / s.total_assets
        x3 = s.ebit / s.total_assets
        x4 = s.market_cap / s.total_liabilities
        x5 = s.total_revenue / s.total_assets

        z = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (0.999 * x5)

        if z > RISK.altman_safe_zone:
            verdict = "🟢 Safe Zone"
        elif z >= RISK.altman_grey_zone:
            verdict = "🟡 Grey Zone"
        else:
            verdict = "🔴 Distress Zone (High Risk)"
        return z, verdict, []

    # ----- valuation ---------------------------------------------------------

    def beta_estimate(self, regressed_beta: Optional[float] = None) -> Tuple[float, str]:
        """Beta for CAPM, in explicit fallback-chain priority: a regressed
        beta (OLS slope of this ticker's returns against the user's selected
        benchmark — see portfolio_analytics.compute_capm_beta, computed in
        finance.py where both return series are already loaded) takes
        priority when the caller supplies one; otherwise Yahoo's reported
        beta; otherwise a declared 1.0 market-beta assumption. Every step
        down the chain is an explicit, disclosed choice, never a silently
        fabricated number.
        """
        s = self.std
        if regressed_beta is not None:
            return regressed_beta, "regressed"
        if s.beta is not None:
            return s.beta, "yahoo_reported"
        return 1.0, "market_assumption"

    def wacc(self, regressed_beta: Optional[float] = None) -> Optional[float]:
        """Weighted average cost of capital via CAPM plus after-tax cost of debt."""
        s = self.std
        mcap = s.market_cap
        if not mcap:
            return None
        beta, _ = self.beta_estimate(regressed_beta)
        cost_of_equity = RISK.risk_free_rate + beta * (DCF.market_return - RISK.risk_free_rate)

        interest = abs(s.interest_expense) if s.interest_expense else 0
        debt = s.total_debt if s.total_debt else 1  # avoid div-by-zero when debt is $0/unreported
        cost_of_debt = (interest / debt) * (1 - DCF.tax_rate)

        return (mcap / (mcap + debt)) * cost_of_equity + (debt / (mcap + debt)) * cost_of_debt

    def normalized_ebit_margin(self) -> Optional[float]:
        """Historical average EBIT margin across every fiscal year where both
        revenue and EBIT are reported (typically ~4-5 years from yfinance) —
        the level the projected margin fades TOWARD in intrinsic_price(), so
        a single unusually strong or weak current year doesn't get
        extrapolated forever. None when there's no revenue/EBIT history to
        average — never a fabricated industry benchmark standing in for it.
        """
        s = self.std
        revenue_by_date = dict(s.revenue_history)
        ebit_by_date = dict(s.ebit_history)
        common_dates = [d for d in revenue_by_date if d in ebit_by_date and revenue_by_date[d]]
        if not common_dates:
            return None
        margins = [ebit_by_date[d] / revenue_by_date[d] for d in common_dates]
        return sum(margins) / len(margins)

    def reinvestment_ratios(self) -> Tuple[float, float, float]:
        """Average (D&A, Capex, ΔWorking Capital) as a fraction of revenue,
        across every fiscal year where both that line and revenue are
        reported. These scale the projection's reinvestment need with
        projected revenue rather than attempting a full driver-based
        build (no per-asset capex schedule, no explicit NWC turnover
        model) — a standard simplification when only historical financial
        statements are available, not a hidden shortcut.

        Capex and ΔWorking Capital are used exactly as yfinance reports
        them (already negative when they're a cash use — confirmed by
        reconstructing reported Free Cash Flow = Operating Cash Flow +
        Capital Expenditure from real statement data), so all three ratios
        are simply ADDED in intrinsic_price(), never subtracted. Each
        ratio independently defaults to 0.0 (no effect) when that specific
        statement line isn't available for this ticker, rather than
        blocking the whole DCF over one missing line.
        """
        s = self.std
        revenue_by_date = dict(s.revenue_history)

        def _avg_ratio(history: Tuple[Tuple[object, float], ...]) -> float:
            by_date = dict(history)
            common = [d for d in by_date if d in revenue_by_date and revenue_by_date[d]]
            if not common:
                return 0.0
            return sum(by_date[d] / revenue_by_date[d] for d in common) / len(common)

        return (
            _avg_ratio(s.depreciation_history),
            _avg_ratio(s.capex_history),
            _avg_ratio(s.change_in_working_capital_history),
        )

    def intrinsic_price(self, growth_rate: float, discount_rate: float) -> float:
        """2-stage DCF intrinsic value per share at the given growth/discount rates.

        Revenue is projected forward at `growth_rate` (the user's slider —
        this model doesn't override that with a historical growth rate).
        EBIT margin is projected as a linear fade from this year's margin
        to normalized_ebit_margin() over DCF.projection_years — a company
        already at its historical average margin gets a flat projection;
        one currently above or below it gets pulled toward that average
        rather than extrapolating an unusually good or bad year forever.
        Linear fade is a deliberate simplifying choice (the standard
        approach in practitioner multi-stage DCF templates absent a
        driver-based margin forecast), not a fabricated growth path.

        FCF per projected year = NOPAT + D&A + Capex + ΔWorking Capital,
        with the latter three scaled by revenue via reinvestment_ratios()'s
        historical average ratios (Capex/ΔNWC already negative-signed, see
        that method's docstring) — replacing the old single-FCF-observation
        compounding this function used before this task.

        Exposed publicly so the sensitivity analysis can re-run it across a
        grid without duplicating the model.
        """
        s = self.std
        shares = s.shares_outstanding
        revenue = s.total_revenue
        current_margin = (s.ebit / revenue) if (s.ebit is not None and revenue) else None
        normalized_margin = self.normalized_ebit_margin()
        if current_margin is None:
            current_margin = normalized_margin
        if normalized_margin is None:
            normalized_margin = current_margin
        da_ratio, capex_ratio, nwc_ratio = self.reinvestment_ratios()

        fcf_projections = []
        for year in range(1, DCF.projection_years + 1):
            revenue = revenue * (1 + growth_rate)
            fade = year / DCF.projection_years
            margin = current_margin + (normalized_margin - current_margin) * fade
            ebit = revenue * margin
            nopat = ebit * (1 - DCF.tax_rate)
            fcf = nopat + revenue * da_ratio + revenue * capex_ratio + revenue * nwc_ratio
            fcf_projections.append(fcf)

        pv_projections = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(fcf_projections, start=1))
        terminal = (fcf_projections[-1] * (1 + DCF.terminal_growth_rate)) / (discount_rate - DCF.terminal_growth_rate)
        pv_terminal = terminal / (1 + discount_rate) ** DCF.projection_years
        return (pv_projections + pv_terminal) / shares

    def run_dcf(
        self,
        growth_rate: float,
        fallback_price: Optional[float] = None,
        regressed_beta: Optional[float] = None,
        beta_r_squared: Optional[float] = None,
    ) -> DCFResult:
        """Full DCF valuation. Never raises — an un-runnable model returns
        ok=False with the reason, so the dashboard can explain itself."""
        s = self.std
        shares = s.shares_outstanding or 0
        price = s.current_price if s.current_price is not None else fallback_price
        mcap = s.market_cap
        normalized_margin = self.normalized_ebit_margin()
        current_margin = (s.ebit / s.total_revenue) if (s.ebit is not None and s.total_revenue) else None
        anchor_margin = normalized_margin if normalized_margin is not None else current_margin

        if not mcap:
            reason = "missing market cap"
        elif shares <= 0:
            reason = "missing shares outstanding"
        elif s.total_revenue is None or s.total_revenue <= 0:
            reason = "missing or non-positive revenue"
        elif anchor_margin is None:
            reason = "no EBIT history available to model a margin trajectory"
        elif anchor_margin <= 0:
            reason = "structurally negative average EBIT margin — not a meaningful DCF candidate"
        else:
            reason = None

        if reason is not None:
            log_event(
                logger, logging.WARNING, "calc.skipped", section="dcf_engine",
                ticker=s.ticker, reason=reason, market_cap=mcap,
                revenue=s.total_revenue, ebit=s.ebit, shares=shares,
            )
            return DCFResult(ok=False, reason=reason, current_price=price)

        discount_rate = self.wacc(regressed_beta)
        beta_used, beta_source = self.beta_estimate(regressed_beta)
        intrinsic = self.intrinsic_price(growth_rate, discount_rate)
        margin_of_safety = ((intrinsic - price) / intrinsic) * 100

        if margin_of_safety >= DCF.strong_buy_margin_of_safety:
            status, color = "Strong Buy", "normal"
        elif margin_of_safety > 0:
            status, color = "Buy", "normal"
        else:
            status, color = "Overvalued", "inverse"

        return DCFResult(
            ok=True, wacc=discount_rate, current_price=price,
            intrinsic_price=intrinsic, margin_of_safety_pct=margin_of_safety,
            status=status, status_color=color,
            beta=beta_used, beta_source=beta_source,
            beta_r_squared=beta_r_squared if beta_source == "regressed" else None,
        )

    # ----- basket pre-screen -------------------------------------------------

    def screen_watchlist(self) -> Optional[WatchlistScore]:
        """Fast 4-point screen for scanning a whole basket of tickers.

        Deliberately uses the stricter WATCHLIST thresholds rather than
        SCORECARD's: this runs across many candidates to surface only the
        strongest, whereas the Scorecard deep-dives a ticker the user already
        chose. Needs only quote-level fields, so it works on the lightweight
        (deep=False) bundles the basket scan loads.

        Returns None when the ticker lacks the P/E or margin needed to screen.
        """
        s = self.std
        if not s.pe_ratio or not s.net_margin:
            return None

        flags = 0
        if s.net_margin >= WATCHLIST.min_net_margin:
            flags += 1
        if s.debt_to_equity is not None and 0 < s.debt_to_equity < WATCHLIST.max_debt_to_equity:
            flags += 1
        if s.current_ratio is not None and s.current_ratio > WATCHLIST.min_current_ratio:
            flags += 1
        if WATCHLIST.pe_range[0] <= s.pe_ratio <= WATCHLIST.pe_range[1]:
            flags += 1

        score = (flags / 4) * 100
        status = "🟢 High" if score >= 75 else ("🟡 Moderate" if score >= 50 else "🔴 Low")
        return WatchlistScore(
            ticker=s.ticker, score=score, status=status,
            pe_ratio=s.pe_ratio, net_margin_pct=s.net_margin * 100,
        )

    # ----- profitability -------------------------------------------------------

    def gross_margin_pct(self) -> Optional[float]:
        gp, rev = self.std.gross_profit, self.std.total_revenue
        if gp is None or not rev:
            return None
        return (gp / rev) * 100

    def operating_margin_pct(self) -> Optional[float]:
        oi, rev = self.std.operating_income, self.std.total_revenue
        if oi is None or not rev:
            return None
        return (oi / rev) * 100

    def roa_pct(self) -> Optional[float]:
        ni, assets = self.std.net_income, self.std.total_assets
        if ni is None or not assets:
            return None
        return (ni / assets) * 100

    def net_margin_pct_computed(self) -> Optional[float]:
        """Net Margin independently computed from statements (Net Income /
        Total Revenue) — kept separate from `standardized.net_margin`, which
        is Yahoo's own pre-computed figure and remains the value the
        Scorecard/Master Matrix use. This one exists purely to cross-check
        that Yahoo's number is right."""
        ni, rev = self.std.net_income, self.std.total_revenue
        if ni is None or not rev:
            return None
        return (ni / rev) * 100

    def roe_pct_computed(self) -> Optional[float]:
        """ROE independently computed (Net Income / Stockholders Equity),
        cross-checked against Yahoo's own returnOnEquity. Can be extreme or
        negative for companies with small/negative book equity from heavy
        buybacks (AAPL is the textbook case) — a large number here is not
        necessarily a bug in either source."""
        ni, equity = self.std.net_income, self.std.stockholders_equity
        if ni is None or not equity:
            return None
        return (ni / equity) * 100

    def validate_profitability(self) -> List[ProfitabilityCheck]:
        """Every profitability ratio, independently computed and cross-checked
        against Yahoo's own separately-reported ratio for the same concept.
        This is the closest practical substitute available here for
        reconciling against a real annual report (see module docstring).
        """
        def yahoo_pct(info_key: str) -> Optional[float]:
            value = self._info.get(info_key)
            return None if value is None else value * 100

        def check(key, label, formula, computed_fn, yahoo_key):
            computed = computed_fn()
            reference = yahoo_pct(yahoo_key) if yahoo_key else None
            return ProfitabilityCheck(
                key=key, label=label, formula=formula,
                computed_pct=computed, reference_pct=reference,
                agrees=_values_agree(computed, reference),
            )

        return [
            check("net_margin", "Net Margin", "Net Income / Total Revenue",
                  self.net_margin_pct_computed, "profitMargins"),
            check("gross_margin", "Gross Margin", "Gross Profit / Total Revenue",
                  self.gross_margin_pct, "grossMargins"),
            check("operating_margin", "Operating Margin", "Operating Income / Total Revenue",
                  self.operating_margin_pct, "operatingMargins"),
            check("roa", "Return on Assets", "Net Income / Total Assets",
                  self.roa_pct, "returnOnAssets"),
            check("roe", "Return on Equity", "Net Income / Stockholders Equity",
                  self.roe_pct_computed, "returnOnEquity"),
            # Yahoo does not report ROIC; no independent reference exists, so
            # this row documents the formula without a cross-check.
            check("roic", "ROIC", "NOPAT / (Total Debt + Stockholders Equity)",
                  self.roic_pct, None),
        ]

    # ----- liquidity -------------------------------------------------------

    def current_ratio_computed(self) -> Optional[float]:
        """Current Ratio independently computed from statements (Current
        Assets / Current Liabilities) — kept separate from
        `standardized.current_ratio`, which is Yahoo's own pre-computed
        figure and remains the value the Scorecard/Master Matrix use. This
        one exists purely to cross-check that Yahoo's number is right.
        Returns None when Current Liabilities is missing or zero, rather
        than raising or dividing by zero."""
        ca, cl = self.std.current_assets, self.std.current_liabilities
        if ca is None or not cl:
            return None
        return ca / cl

    def quick_ratio_computed(self) -> Optional[float]:
        """Quick (Acid-Test) Ratio = (Current Assets − Inventory) / Current
        Liabilities. Inventory defaults to 0 on StandardizedFinancials for
        companies that structurally carry none (banks, software/services),
        so it never blocks the calculation the way a missing Current
        Liabilities does."""
        ca, cl = self.std.current_assets, self.std.current_liabilities
        if ca is None or not cl:
            return None
        return (ca - self.std.inventory) / cl

    def validate_liquidity(self) -> List[ProfitabilityCheck]:
        """Current Ratio and Quick Ratio, independently computed and
        cross-checked against Yahoo's own separately-reported ratio for the
        same concept — informational only; neither feeds the Scorecard or
        Master Matrix (Current Ratio there stays Yahoo-sourced, and Quick
        Ratio isn't a scoreboard flag at all).
        """
        def yahoo_ratio(info_key: str) -> Optional[float]:
            value = self._info.get(info_key)
            return None if value is None else float(value)

        def check(key, label, formula, computed_fn, yahoo_key):
            computed = computed_fn()
            reference = yahoo_ratio(yahoo_key)
            return ProfitabilityCheck(
                key=key, label=label, formula=formula,
                computed_pct=computed, reference_pct=reference,
                agrees=_values_agree(computed, reference),
                suffix="",
            )

        return [
            check("current_ratio", "Current Ratio", "Current Assets / Current Liabilities",
                  self.current_ratio_computed, "currentRatio"),
            check("quick_ratio", "Quick Ratio", "(Current Assets − Inventory) / Current Liabilities",
                  self.quick_ratio_computed, "quickRatio"),
        ]

    # ----- leverage ----------------------------------------------------------

    def validate_leverage(self) -> List[ProfitabilityCheck]:
        """Leverage metrics, independently verified:

        - Debt-to-Equity IS the canonical `standardized.debt_to_equity` value
          (statement-computed — see financial_standardization.py); shown here
          cross-checked against Yahoo's own debtToEquity for visibility.
        - Total Debt: the balance-sheet-only figure vs. Yahoo's info-dict
          figure, independent of the resolved (fallback-merged) value used
          everywhere else — surfaces a silent disagreement between the two
          raw sources instead of hiding it behind whichever one happened to
          be picked. Shown in $ billions.
        - Interest Coverage has no Yahoo-reported equivalent (same situation
          as ROIC in validate_profitability()) — documented here for
          completeness with no cross-check (⚪ "not evaluable").
        """
        yahoo_de = normalize_debt_to_equity(self._info.get('debtToEquity'))
        de_check = ProfitabilityCheck(
            key="debt_to_equity", label="Debt-to-Equity",
            formula="Total Debt / Stockholders Equity",
            computed_pct=self.std.debt_to_equity, reference_pct=yahoo_de,
            agrees=_values_agree(self.std.debt_to_equity, yahoo_de),
            suffix="",
        )

        statement_debt = self.std.total_debt_from_statement
        info_debt = self._info.get('totalDebt')
        statement_debt_b = None if statement_debt is None else statement_debt / 1e9
        info_debt_b = None if info_debt is None else info_debt / 1e9
        debt_check = ProfitabilityCheck(
            key="total_debt", label="Total Debt (Balance Sheet vs. Info)",
            formula="Balance Sheet 'Total Debt' vs. Yahoo info['totalDebt']",
            computed_pct=statement_debt_b, reference_pct=info_debt_b,
            agrees=_values_agree(statement_debt_b, info_debt_b),
            suffix=" B",
        )

        coverage_check = ProfitabilityCheck(
            key="interest_coverage", label="Interest Coverage",
            formula="EBIT / |Interest Expense|",
            computed_pct=self.interest_coverage(), reference_pct=None,
            agrees=None,
            suffix="x",
        )

        return [de_check, debt_check, coverage_check]

    # ----- valuation -----------------------------------------------------------

    def pe_ratio_computed(self) -> Optional[float]:
        """P/E independently computed (Market Cap / Net Income) — kept
        separate from `standardized.pe_ratio`, which stays the Yahoo-sourced
        canonical value used by the Scorecard/Master Matrix. None for
        negative/zero earnings, same convention as the canonical field."""
        mc, ni = self.std.market_cap, self.std.net_income
        if not mc or ni is None or ni <= 0:
            return None
        return mc / ni

    def price_to_book_computed(self) -> Optional[float]:
        """Price-to-Book independently computed (Market Cap / Stockholders
        Equity). None for negative/zero book value — same reasoning as P/E:
        not a meaningful multiple, not just missing data."""
        mc, equity = self.std.market_cap, self.std.stockholders_equity
        if not mc or equity is None or equity <= 0:
            return None
        return mc / equity

    def enterprise_value(self) -> Optional[float]:
        """EV = Market Cap + Total Debt − Cash & Equivalents."""
        mc = self.std.market_cap
        if not mc:
            return None
        return mc + self.std.total_debt - self.std.cash_and_equivalents

    def ebitda(self) -> Optional[float]:
        """EBITDA = EBIT + Depreciation & Amortization. Can be negative for
        a genuinely distressed company — that's a real, informative value,
        left as-is here; only the EV/EBITDA *ratio* treats a non-positive
        EBITDA as not meaningful (see ev_to_ebitda_computed)."""
        ebit = self.std.ebit
        if ebit is None:
            return None
        return ebit + self.std.depreciation_and_amortization

    def ev_to_ebitda_computed(self) -> Optional[float]:
        """EV / EBITDA. None when EBITDA is non-positive — like a negative
        P/E, an EV/EBITDA multiple from non-positive EBITDA isn't a
        meaningful valuation signal."""
        ev = self.enterprise_value()
        ebitda = self.ebitda()
        if ev is None or ebitda is None or ebitda <= 0:
            return None
        return ev / ebitda

    def validate_valuation(self) -> List[ProfitabilityCheck]:
        """Every valuation ratio, independently computed where possible and
        cross-checked against Yahoo's own separately-reported ratio for the
        same concept. P/E and Price-to-Book stay Yahoo-sourced as the
        canonical Scorecard/Master Matrix values (see standardized.pe_ratio /
        standardized.price_to_book) — this report exists purely to verify
        them, same as Current Ratio in validate_liquidity(). EV/EBITDA is a
        new metric with no prior canonical value; PEG and FCF Yield are
        documented here even though Yahoo rarely/never reports an equivalent
        (pegRatio is commonly None; FCF Yield has no Yahoo field at all).
        """
        def yahoo_ratio(info_key: str) -> Optional[float]:
            value = self._info.get(info_key)
            return None if value is None else float(value)

        pe_computed = self.pe_ratio_computed()
        pb_computed = self.price_to_book_computed()
        ev_ebitda_computed = self.ev_to_ebitda_computed()
        yahoo_ev_ebitda = yahoo_ratio('enterpriseToEbitda')
        yahoo_peg = yahoo_ratio('pegRatio')

        return [
            ProfitabilityCheck(
                key="pe_ratio", label="P/E Ratio",
                formula="Market Cap / Net Income",
                computed_pct=pe_computed, reference_pct=self.std.pe_ratio,
                agrees=_values_agree(pe_computed, self.std.pe_ratio),
                suffix="",
            ),
            ProfitabilityCheck(
                key="price_to_book", label="Price-to-Book",
                formula="Market Cap / Stockholders Equity",
                computed_pct=pb_computed, reference_pct=self.std.price_to_book,
                agrees=_values_agree(pb_computed, self.std.price_to_book),
                suffix="",
            ),
            ProfitabilityCheck(
                key="peg_ratio", label="PEG Ratio",
                formula="P/E / (Earnings Growth %)",
                computed_pct=self.std.peg_ratio, reference_pct=yahoo_peg,
                agrees=_values_agree(self.std.peg_ratio, yahoo_peg),
                suffix="",
            ),
            ProfitabilityCheck(
                key="ev_ebitda", label="EV / EBITDA",
                formula="(Market Cap + Total Debt − Cash) / (EBIT + D&A)",
                computed_pct=ev_ebitda_computed, reference_pct=yahoo_ev_ebitda,
                agrees=_values_agree(ev_ebitda_computed, yahoo_ev_ebitda),
                suffix="x",
            ),
            # No Yahoo equivalent field exists for FCF Yield — documented
            # here for completeness with no cross-check (⚪ "not evaluable"),
            # same treatment as ROIC and Interest Coverage.
            ProfitabilityCheck(
                key="fcf_yield", label="FCF Yield",
                formula="Free Cash Flow / Market Cap",
                computed_pct=self.fcf_yield_pct(), reference_pct=None,
                agrees=None,
                suffix="%",
            ),
        ]

    # ----- consolidated validation ---------------------------------------------

    def validate_all_metrics(
        self,
        profitability_checks: List[ProfitabilityCheck],
        liquidity_checks: List[ProfitabilityCheck],
        leverage_checks: List[ProfitabilityCheck],
        valuation_checks: List[ProfitabilityCheck],
    ) -> MetricsValidationSummary:
        """Consolidate the four category reports into one overview: every
        check in one place, plus outlier detection (config.OUTLIER_BOUNDS)
        and incomplete-calculation flags that don't belong to any single
        category. Takes the already-computed check lists rather than
        recomputing them, since analyze() builds all four anyway.
        """
        checks: List[Tuple[str, ProfitabilityCheck]] = (
            [("Profitability", c) for c in profitability_checks] +
            [("Liquidity", c) for c in liquidity_checks] +
            [("Leverage", c) for c in leverage_checks] +
            [("Valuation", c) for c in valuation_checks]
        )

        outliers = []
        for category, c in checks:
            note = _outlier_note(c.key, c.computed_pct)
            if note:
                outliers.append(OutlierFlag(
                    key=c.key, category=category, label=c.label,
                    display=c.computed_display, note=note,
                ))

        fallback_notes = list(self.std.data_fallbacks)
        if self.effective_tax_rate_used_fallback():
            fallback_notes.append(
                f"ROIC: effective tax rate estimated at the statutory {DCF.tax_rate * 100:.0f}% "
                "because Pretax Income/Tax Provision weren't usable"
            )

        return MetricsValidationSummary(checks=checks, outliers=outliers, fallback_notes=fallback_notes)

    # ----- company quality classification ---------------------------------------

    def asset_turnover(self) -> Optional[float]:
        """Total Revenue / Total Assets — how efficiently the company
        converts its asset base into revenue. Naturally sector-dependent
        (asset-light software vs. asset-heavy banks/utilities); the Capital
        Efficiency factor's band is a global approximation, not sector-tuned."""
        rev, assets = self.std.total_revenue, self.std.total_assets
        if rev is None or not assets:
            return None
        return rev / assets

    def classify_company_quality(self, roic: Optional[float], interest_cov: Optional[float], altman_z: Optional[float]) -> CompanyQuality:
        """Blend Profitability, Financial Stability, Growth, Valuation, and
        Capital Efficiency into one weighted quality score and category —
        see config.QualityConfig for every band/weight and the rationale
        behind each. Takes roic/interest_cov/altman_z already computed by
        analyze() rather than recomputing (avoids duplicate log_event calls
        from altman_z_score()).
        """
        s = self.std
        de_threshold = SCORECARD.max_debt_to_equity_for(s.sector)
        net_margin_pct = None if s.net_margin is None else s.net_margin * 100
        growth_pct = None if s.earnings_growth is None else s.earnings_growth * 100
        ev_ebitda = self.ev_to_ebitda_computed()
        roe = self.roe_pct_computed()
        asset_turnover = self.asset_turnover()

        def m(label: str, value: Optional[float], suffix: str, sub_score: Optional[float]) -> QualityFactorMetric:
            return QualityFactorMetric(label=label, value=value, display=_fmt(value, suffix), sub_score=sub_score)

        profitability = QualityFactor(
            name="Profitability", weight=QUALITY.weight_profitability,
            metrics=[
                m("Net Margin", net_margin_pct, "%", _linear_score(net_margin_pct, QUALITY.net_margin_band_pct)),
                m("Gross Margin", self.gross_margin_pct(), "%", _linear_score(self.gross_margin_pct(), QUALITY.gross_margin_band_pct)),
                m("Operating Margin", self.operating_margin_pct(), "%", _linear_score(self.operating_margin_pct(), QUALITY.operating_margin_band_pct)),
                m("Return on Assets", self.roa_pct(), "%", _linear_score(self.roa_pct(), QUALITY.roa_band_pct)),
            ],
        )

        financial_stability = QualityFactor(
            name="Financial Stability", weight=QUALITY.weight_financial_stability,
            metrics=[
                m("Debt-to-Equity", s.debt_to_equity, "",
                  _linear_score(s.debt_to_equity, (de_threshold * QUALITY.debt_to_equity_zero_point_multiplier, 0.0))),
                m("Current Ratio", s.current_ratio, "", _linear_score(s.current_ratio, QUALITY.current_ratio_band)),
                m("Interest Coverage", interest_cov, "x", _linear_score(interest_cov, QUALITY.interest_coverage_band)),
                m("Altman Z-Score", altman_z, "", _linear_score(altman_z, QUALITY.altman_z_band)),
            ],
        )

        growth = QualityFactor(
            name="Growth", weight=QUALITY.weight_growth,
            metrics=[
                m("Earnings/Revenue Growth", growth_pct, "%", _linear_score(growth_pct, QUALITY.earnings_growth_band_pct)),
            ],
        )

        valuation = QualityFactor(
            name="Valuation", weight=QUALITY.weight_valuation,
            metrics=[
                m("P/E Ratio", s.pe_ratio, "", _ideal_score(s.pe_ratio, QUALITY.pe_ideal)),
                m("PEG Ratio", s.peg_ratio, "", _ideal_score(s.peg_ratio, QUALITY.peg_ideal)),
                m("Price-to-Book", s.price_to_book, "", _ideal_score(s.price_to_book, QUALITY.price_to_book_ideal)),
                m("EV/EBITDA", ev_ebitda, "x", _ideal_score(ev_ebitda, QUALITY.ev_ebitda_ideal)),
            ],
        )

        capital_efficiency = QualityFactor(
            name="Capital Efficiency", weight=QUALITY.weight_capital_efficiency,
            metrics=[
                m("ROIC", roic, "%", _linear_score(roic, QUALITY.roic_band_pct)),
                m("Return on Equity", roe, "%", _linear_score(roe, QUALITY.roe_band_pct)),
                m("Asset Turnover", asset_turnover, "x", _linear_score(asset_turnover, QUALITY.asset_turnover_band)),
            ],
        )

        return CompanyQuality(factors=[profitability, financial_stability, growth, valuation, capital_efficiency])

    # ----- assembled result --------------------------------------------------

    def _build_checks(self, roic, interest_cov, fcf_yield) -> List[MetricCheck]:
        """Evaluate every metric against its configured benchmark.

        This list is the single source for both the Strategic Investment
        Scorecard and the Master Matrix — add a metric here and it shows up
        in both, no dashboard changes required. Debt-to-Equity's threshold is
        sector-adjusted (see SCORECARD.max_debt_to_equity_for); every check's
        `weight` comes from SCORECARD.weights and determines how much it
        contributes to the weighted Blueprint Alignment score.
        """
        s = self.std
        nm, de, cr = s.net_margin, s.debt_to_equity, s.current_ratio
        pe, peg, beta = s.pe_ratio, s.peg_ratio, s.beta
        de_threshold = SCORECARD.max_debt_to_equity_for(s.sector)
        pe_low, pe_high = SCORECARD.pe_range_for(s.sector)
        pe_is_sector_adjusted = (pe_low, pe_high) != SCORECARD.pe_range

        return [
            MetricCheck(
                key="net_margin", category="Profitability", label="Net Margin",
                value=nm, display=_fmt(None if nm is None else nm * 100, "%"),
                benchmark=f"> {SCORECARD.min_net_margin * 100:.0f}%",
                passed=None if nm is None else nm >= SCORECARD.min_net_margin,
                weight=SCORECARD.weight_for("net_margin"),
            ),
            MetricCheck(
                key="debt_to_equity", category="Leverage", label="Debt-to-Equity",
                value=de, display=_fmt(de),
                benchmark=f"< {de_threshold} (sector-adjusted)" if de_threshold != SCORECARD.max_debt_to_equity else f"< {de_threshold}",
                passed=None if de is None else de < de_threshold,
                weight=SCORECARD.weight_for("debt_to_equity"),
            ),
            MetricCheck(
                key="roic", category="Capital Efficiency", label="ROIC",
                value=roic, display=_fmt(roic, "%"),
                benchmark=f"> {SCORECARD.min_roic_pct:.0f}%",
                passed=None if roic is None else roic > SCORECARD.min_roic_pct,
                weight=SCORECARD.weight_for("roic"),
            ),
            # Reported in the Master Matrix but not one of the scoreboard flags.
            MetricCheck(
                key="fcf_yield", category="Cash Flow Quality", label="FCF Yield",
                value=fcf_yield, display=_fmt(fcf_yield, "%"),
                benchmark=f"> {SCORECARD.min_fcf_yield_pct:.0f}%",
                passed=None if fcf_yield is None else fcf_yield > SCORECARD.min_fcf_yield_pct,
                in_scorecard=False,
            ),
            MetricCheck(
                key="interest_coverage", category="Debt Safety", label="Interest Coverage",
                value=interest_cov, display=_fmt(interest_cov, "x", decimals=1),
                benchmark=f"> {SCORECARD.min_interest_coverage:.1f}x",
                passed=None if interest_cov is None else interest_cov > SCORECARD.min_interest_coverage,
                weight=SCORECARD.weight_for("interest_coverage"),
            ),
            MetricCheck(
                key="pe_ratio", category="Valuation (P/E)", label="P/E Ratio TTM",
                value=pe, display="N/A" if pe is None else f"{pe}",
                benchmark=(
                    f"{pe_low:.0f} - {pe_high:.0f} (sector-adjusted)" if pe_is_sector_adjusted
                    else f"{pe_low:.0f} - {pe_high:.0f}"
                ),
                passed=None if pe is None else pe_low <= pe <= pe_high,
                weight=SCORECARD.weight_for("pe_ratio"),
            ),
            # Not sector-adjusted like P/E above, deliberately: PEG already
            # divides P/E by the growth rate, which is exactly what makes a
            # flat P/E band misleading across sectors in the first place.
            # See SCORECARD.pe_range_for()'s docstring.
            MetricCheck(
                key="peg_ratio", category="Valuation (PEG)", label="PEG Ratio (Proxy)",
                value=peg, display="N/A" if peg is None else f"{peg}",
                benchmark=f"< {SCORECARD.peg_range[1]}",
                passed=None if peg is None else SCORECARD.peg_range[0] < peg <= SCORECARD.peg_range[1],
                weight=SCORECARD.weight_for("peg_ratio"),
            ),
            MetricCheck(
                key="beta", category="Volatility", label="Beta",
                value=beta, display="N/A" if beta is None else f"{beta}",
                benchmark=f"< {SCORECARD.max_beta}",
                passed=None if beta is None else beta < SCORECARD.max_beta,
                weight=SCORECARD.weight_for("beta"),
            ),
            MetricCheck(
                key="current_ratio", category="Liquidity", label="Current Ratio",
                value=cr, display=_fmt(cr),
                benchmark=f"> {SCORECARD.min_current_ratio}",
                passed=None if cr is None else cr > SCORECARD.min_current_ratio,
                weight=SCORECARD.weight_for("current_ratio"),
            ),
        ]

    def analyze(self) -> FundamentalMetrics:
        """Compute every statement-derived ratio and the evaluated checks."""
        s = self.std
        roic = self.roic_pct()
        interest_cov = self.interest_coverage()
        fcf_yield = self.fcf_yield_pct()
        altman_z, altman_verdict, altman_missing = self.altman_z_score()
        profitability_checks = self.validate_profitability()
        liquidity_checks = self.validate_liquidity()
        leverage_checks = self.validate_leverage()
        valuation_checks = self.validate_valuation()
        metrics_validation = self.validate_all_metrics(
            profitability_checks, liquidity_checks, leverage_checks, valuation_checks,
        )
        company_quality = self.classify_company_quality(roic, interest_cov, altman_z)

        metrics = FundamentalMetrics(
            ticker=s.ticker,
            sector=s.sector,
            net_margin=s.net_margin,
            debt_to_equity=s.debt_to_equity,
            current_ratio=s.current_ratio,
            pe_ratio=s.pe_ratio,
            peg_ratio=s.peg_ratio,
            beta=s.beta,
            roic_pct=roic,
            interest_coverage=interest_cov,
            fcf_yield_pct=fcf_yield,
            gross_margin_pct=self.gross_margin_pct(),
            operating_margin_pct=self.operating_margin_pct(),
            roa_pct=self.roa_pct(),
            profitability_checks=profitability_checks,
            liquidity_checks=liquidity_checks,
            leverage_checks=leverage_checks,
            valuation_checks=valuation_checks,
            metrics_validation=metrics_validation,
            company_quality=company_quality,
            altman_z=altman_z,
            altman_verdict=altman_verdict,
            altman_missing_inputs=altman_missing,
            checks=self._build_checks(roic, interest_cov, fcf_yield),
        )

        # `c.agrees` can be numpy.bool_ (computed_pct is a numpy.float64
        # extracted from a pandas Series), and `numpy.bool_(False) is False`
        # is False — an `is` identity check would silently never match.
        disagreements = [c.key for c in profitability_checks if c.agrees is not None and not c.agrees]
        if disagreements:
            log_event(
                logger, logging.WARNING, "profitability.discrepancy",
                ticker=s.ticker, metrics=", ".join(disagreements),
            )
        liquidity_disagreements = [c.key for c in liquidity_checks if c.agrees is not None and not c.agrees]
        if liquidity_disagreements:
            log_event(
                logger, logging.WARNING, "liquidity.discrepancy",
                ticker=s.ticker, metrics=", ".join(liquidity_disagreements),
            )
        leverage_disagreements = [c.key for c in leverage_checks if c.agrees is not None and not c.agrees]
        if leverage_disagreements:
            log_event(
                logger, logging.WARNING, "leverage.discrepancy",
                ticker=s.ticker, metrics=", ".join(leverage_disagreements),
            )
        valuation_disagreements = [c.key for c in valuation_checks if c.agrees is not None and not c.agrees]
        if valuation_disagreements:
            log_event(
                logger, logging.WARNING, "valuation.discrepancy",
                ticker=s.ticker, metrics=", ".join(valuation_disagreements),
            )
        if metrics_validation.outliers:
            log_event(
                logger, logging.WARNING, "metrics.outlier",
                ticker=s.ticker, metrics=", ".join(o.key for o in metrics_validation.outliers),
            )
        log_event(
            logger, logging.INFO, "metrics_validation.summary", ticker=s.ticker,
            evaluated=len(metrics_validation.evaluated_checks), of=len(metrics_validation.checks),
            disagreements=metrics_validation.disagreement_count,
            outliers=metrics_validation.outlier_count,
            fallbacks=metrics_validation.fallback_count,
        )
        log_event(
            logger, logging.INFO, "quality.classified", ticker=s.ticker,
            category=company_quality.category,
            score=None if company_quality.overall_score is None else round(company_quality.overall_score, 1),
            factors_evaluable=len(company_quality.evaluable_factors), of=len(company_quality.factors),
        )
        log_event(
            logger, logging.INFO, "fundamentals.analyzed", ticker=s.ticker,
            sector=s.sector or "unknown",
            green_flags=metrics.green_flags, of=metrics.total_checks,
            score=round(metrics.score_pct, 1), roic=roic,
            altman=altman_z, alignment=metrics.alignment_verdict,
        )
        return metrics


def analyze_fundamentals(standardized: StandardizedFinancials, raw_info: Optional[dict] = None) -> FundamentalMetrics:
    """Convenience wrapper matching the module-level style used elsewhere."""
    return FundamentalAnalysisEngine(standardized, raw_info=raw_info).analyze()
