"""ETF decomposition: what a fund holds, what it costs, and what it is.

TWO UNIT TRAPS, BOTH MEASURED. Neither is guessable from the field names.

1. Yahoo's `equity_holdings` frame labels its rows "Price/Earnings",
   "Price/Book", "Price/Sales" — and reports the RECIPROCAL of each. SPY
   comes back as 0.03976, which is not a price-to-earnings ratio of four
   hundredths; it is an earnings yield, and 1/0.03976 = 25.15, which is
   SPY's actual multiple. Verified across four funds whose ordering
   confirms it: QQQ 29.4, SPY 25.2, VTV 20.7, IWM 18.1 — growth above
   blend above value above small-cap, exactly as it should be. Taken at
   face value the app would report every ETF as trading at 0.04x
   earnings.

2. The expense ratio exists in two places in two units: `info`'s
   netExpenseRatio is a PERCENT (0.0945 meaning 0.0945%) while
   fund_operations' "Annual Report Expense Ratio" is a FRACTION
   (0.000945). Exactly 100x apart. This module takes the fraction and
   converts once, so no caller has to remember which source it came from.

WHY THE TOP TEN ARE NOT THE VALUATION BASIS. The task specifies
`weighted_pe = sum(h.weight * h.pe for h in holdings)` over the top
holdings. Yahoo returns ten of them, and for SPY those are 37.6% of the
fund — so that sum produces a P/E of 9.46 against a true 25.15, a 62%
understatement that reads like a cheap fund rather than like an error.
The fund-level figures above cover the WHOLE portfolio, so they are used
for valuation and the top ten are reported as what they actually measure:
concentration.

WHAT IS NOT AVAILABLE, so the scorecard says so rather than scoring it:
holding-level ROE and ROA, the total number of holdings, and geographic
allocation. Median market cap is present in the schema but comes back
empty for the funds probed, so style is classified on valuation alone and
the size half is reported as unavailable.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("etf_analysis")

TOP_HOLDINGS_SHOWN = 10

# Horizons for the cost-drag illustration.
DRAG_YEARS: Tuple[int, ...] = (10, 20, 30)
# The growth rate the drag is illustrated against. Declared, not derived:
# this is an illustration of what a fee costs, not a return forecast, and
# the UI says so.
DRAG_ASSUMED_GROSS_RETURN_PCT = 7.0

# Above this gap over the category average, the fee is called out. The
# task says "peer average + 0.5%".
EXPENSE_FLAG_GAP_PCT = 0.5

# Style cutoffs on the fund's price/earnings. Morningstar-ish boundaries;
# declared here so the label and the number a reader sees cannot disagree.
VALUE_PE_MAX = 18.0
GROWTH_PE_MIN = 26.0


@dataclass(frozen=True)
class Holding:
    symbol: str
    name: str
    weight_pct: float          # already a percentage, e.g. 7.55


@dataclass(frozen=True)
class EtfProfile:
    symbol: str
    category: str = ""
    family: str = ""
    legal_type: str = ""

    price_earnings: Optional[float] = None      # inverted, whole fund
    price_book: Optional[float] = None
    price_sales: Optional[float] = None
    category_price_earnings: Optional[float] = None

    expense_ratio_pct: Optional[float] = None
    category_expense_ratio_pct: Optional[float] = None
    turnover_pct: Optional[float] = None
    category_turnover_pct: Optional[float] = None

    net_assets: Optional[float] = None
    asset_mix: Dict[str, float] = field(default_factory=dict)
    sector_weights: Dict[str, float] = field(default_factory=dict)
    top_holdings: Tuple[Holding, ...] = ()

    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _invert(value) -> Optional[float]:
    """Yahoo's mislabelled yield -> the ratio it claims to be.

    Returns None for zero or nonsense rather than raising or producing an
    absurd multiple: a fund with no earnings has no P/E, and 1/0 is not
    an answer.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number <= 0:       # NaN or non-positive
        return None
    ratio = 1.0 / number
    # A P/E of ten thousand is a data artefact, not a valuation.
    return ratio if ratio < 1000 else None


def _cell(frame, row: str, column: str):
    try:
        return frame.loc[row, column]
    except Exception:
        return None


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


@st.cache_data(ttl=86400, show_spinner=False)
def load_profile(symbol: str) -> EtfProfile:
    """Everything this build can say about a fund. Never raises."""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return EtfProfile(symbol="", error="No symbol given.")
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        funds = ticker.funds_data
        info = ticker.info or {}

        overview = funds.fund_overview or {}
        equity = funds.equity_holdings
        operations = funds.fund_operations

        holdings = []
        try:
            for holding_symbol, row in funds.top_holdings.iterrows():
                weight = _number(row.get("Holding Percent"))
                holdings.append(Holding(
                    symbol=str(holding_symbol).strip().upper(),
                    name=str(row.get("Name") or "").strip(),
                    # Yahoo gives the weight as a fraction; one conversion,
                    # here, so nothing downstream has to wonder.
                    weight_pct=(weight * 100.0) if weight is not None else 0.0,
                ))
        except Exception:
            log_exception(logger, "etf.holdings_unreadable", section="etf_analysis")

        expense = _number(_cell(operations, "Annual Report Expense Ratio", symbol))
        category_expense = _number(_cell(operations, "Annual Report Expense Ratio",
                                         "Category Average"))
        turnover = _number(_cell(operations, "Annual Holdings Turnover", symbol))
        category_turnover = _number(_cell(operations, "Annual Holdings Turnover",
                                          "Category Average"))

        profile = EtfProfile(
            symbol=symbol,
            category=str(overview.get("categoryName") or ""),
            family=str(overview.get("family") or ""),
            legal_type=str(overview.get("legalType") or ""),
            price_earnings=_invert(_cell(equity, "Price/Earnings", symbol)),
            price_book=_invert(_cell(equity, "Price/Book", symbol)),
            price_sales=_invert(_cell(equity, "Price/Sales", symbol)),
            category_price_earnings=_invert(
                _cell(equity, "Price/Earnings", "Category Average")),
            # Fractions in this frame; percent everywhere in this module.
            expense_ratio_pct=expense * 100.0 if expense is not None else None,
            category_expense_ratio_pct=(category_expense * 100.0
                                        if category_expense is not None else None),
            turnover_pct=turnover * 100.0 if turnover is not None else None,
            category_turnover_pct=(category_turnover * 100.0
                                   if category_turnover is not None else None),
            net_assets=_number(info.get("netAssets") or info.get("totalAssets")),
            asset_mix={k: v for k, v in (funds.asset_classes or {}).items()
                       if _number(v)},
            sector_weights={k: v for k, v in (funds.sector_weightings or {}).items()
                            if _number(v)},
            top_holdings=tuple(holdings),
        )
        log_event(logger, logging.INFO, "etf.profile_loaded", ticker=symbol,
                  holdings=len(holdings), has_pe=profile.price_earnings is not None)
        return profile
    except Exception as e:
        log_exception(logger, "etf.profile_failed", section="etf_analysis")
        return EtfProfile(symbol=symbol,
                          error=f"Fund data is unavailable for {symbol} "
                                f"({type(e).__name__}).")


# --- derived measures ---------------------------------------------------------

def concentration_pct(holdings: Sequence[Holding]) -> Optional[float]:
    """How much of the fund the listed holdings account for.

    This is NOT a valuation input — see the module docstring. It is the
    honest reading of a top-ten list: a concentration measure.
    """
    if not holdings:
        return None
    return sum(h.weight_pct for h in holdings)


def expense_gap_pct(profile: EtfProfile) -> Optional[float]:
    if profile.expense_ratio_pct is None or profile.category_expense_ratio_pct is None:
        return None
    return profile.expense_ratio_pct - profile.category_expense_ratio_pct


def expense_is_high(profile: EtfProfile) -> bool:
    gap = expense_gap_pct(profile)
    return gap is not None and gap > EXPENSE_FLAG_GAP_PCT


def expense_drag(expense_ratio_pct: Optional[float], years: int,
                 gross_return_pct: float = DRAG_ASSUMED_GROSS_RETURN_PCT) -> Optional[float]:
    """Percent of the gross outcome given up to fees over `years`.

    Compounded, not multiplied: a 0.5% fee over 30 years costs far more
    than 15% because it compounds against the whole balance every year.
    Returns None rather than 0.0 when the ratio is unknown, so "no fee
    reported" cannot be mistaken for "no fee".
    """
    if expense_ratio_pct is None or years <= 0:
        return None
    gross = (1.0 + gross_return_pct / 100.0) ** years
    net = (1.0 + (gross_return_pct - expense_ratio_pct) / 100.0) ** years
    if gross <= 0:
        return None
    return (1.0 - net / gross) * 100.0


def valuation_gap_pct(profile: EtfProfile) -> Optional[float]:
    """(fund P/E - category P/E) / category P/E, as a percentage."""
    if profile.price_earnings is None or not profile.category_price_earnings:
        return None
    return ((profile.price_earnings - profile.category_price_earnings)
            / profile.category_price_earnings) * 100.0


# The style words Yahoo's own category strings use ("Large Value",
# "Mid-Cap Growth", "Large Blend"). Its categories are Morningstar-derived
# and carry SIZE as well as style, which is the half this module cannot
# compute — median market cap is in the schema but comes back empty.
_STYLE_WORDS = ("value", "growth", "blend")


def style_label(profile: EtfProfile) -> str:
    """The fund's style, preferring the source's own classification.

    Yahoo's category is used first, and my P/E cutoffs only as a fallback,
    because they disagreed on a real fund: VTV is categorised "Large
    Value" and prices at 20.7x, which a cutoff of 18 calls "Blend". The
    cutoffs are a guess about where value ends; the category is the
    provider's actual classification and carries the size band too. Where
    both exist the category wins and the multiple is shown beside it, so
    a reader can see the basis rather than taking a label on trust.
    """
    category = (profile.category or "").strip()
    if category and any(word in category.lower() for word in _STYLE_WORDS):
        return category

    pe = profile.price_earnings
    if pe is None:
        return "Not classified — no category or fund-level P/E reported"
    if pe <= VALUE_PE_MAX:
        return f"Value (inferred from {pe:.1f}x earnings)"
    if pe >= GROWTH_PE_MIN:
        return f"Growth (inferred from {pe:.1f}x earnings)"
    return f"Blend (inferred from {pe:.1f}x earnings)"


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    score: Optional[float]      # 0-10, or None when the input is missing
    detail: str


def quality_scorecard(profile: EtfProfile) -> Tuple[List[ScoreComponent], Optional[float], str]:
    """(components, overall 1-10 or None, how it was computed).

    Scores only what the data supports and says how many components
    contributed, following this app's "N of M evaluable" convention.
    Holding-level ROE/ROA and the total holding count are not available
    from this source, so they are listed as unscored rather than being
    silently dropped — a five-part scorecard secretly computed from three
    parts is a worse lie than an honest three.
    """
    components: List[ScoreComponent] = []

    gap = expense_gap_pct(profile)
    if gap is None:
        components.append(ScoreComponent(
            "Cost vs category", None, "No expense ratio reported."))
    else:
        # Cheaper than peers is better; 10 at 0.5pp below, 0 at 0.5pp above.
        score = max(0.0, min(10.0, 5.0 - gap * 10.0))
        components.append(ScoreComponent(
            "Cost vs category", score,
            f"{profile.expense_ratio_pct:.2f}% vs {profile.category_expense_ratio_pct:.2f}% "
            f"category average."))

    covered = concentration_pct(profile.top_holdings)
    if covered is None:
        components.append(ScoreComponent(
            "Concentration", None, "No holdings reported."))
    else:
        # 10 when the top ten are a small slice, 0 when they are most of it.
        score = max(0.0, min(10.0, (100.0 - covered) / 7.0))
        components.append(ScoreComponent(
            "Concentration", score,
            f"Top {len(profile.top_holdings)} holdings are {covered:.1f}% of the fund."))

    if profile.turnover_pct is None:
        components.append(ScoreComponent(
            "Turnover", None, "No turnover reported."))
    else:
        # Lower turnover means lower internal trading cost and tax drag.
        score = max(0.0, min(10.0, 10.0 - profile.turnover_pct / 20.0))
        detail = f"{profile.turnover_pct:.1f}% a year"
        if profile.category_turnover_pct is not None:
            detail += f" vs {profile.category_turnover_pct:.1f}% category average"
        components.append(ScoreComponent("Turnover", score, detail + "."))

    components.append(ScoreComponent(
        "Holding quality (ROE/ROA)", None,
        "Not available: this source reports fund-level valuation but no "
        "holding-level profitability."))
    components.append(ScoreComponent(
        "Diversification (holding count)", None,
        "Not available: this source lists only the top "
        f"{TOP_HOLDINGS_SHOWN}, not the full roster."))

    scored = [c.score for c in components if c.score is not None]
    if not scored:
        return components, None, "Nothing could be scored from the available data."
    overall = sum(scored) / len(scored)
    return (components, overall,
            f"{len(scored)} of {len(components)} components could be scored.")
