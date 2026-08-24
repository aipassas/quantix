"""Automated data quality assessment for Quantix.

Before the investment engine runs its calculations, this module answers one
question: can this data be trusted? It combines three independent signals
that already exist elsewhere in the pipeline into one score instead of
requiring the user to piece it together from several separate panels:

  - Field completeness (financial_validation.py): are the required/optional
    statement fields actually present?
  - Freshness: how old is the most recent reported quarter?
  - Fetch reliability (data_loader.py): did retries have to kick in, or did
    any optional dataset come back empty?

Required-field completeness dominates the score, since that's what actually
degrades calculations to "N/A" elsewhere in the app — staleness and fetch
warnings are real but secondary quality signals.

QUALITY MEANS SOMETHING DIFFERENT PER ASSET CLASS, and not noticing that
produced a real misreport: every ETF scored 17.5/100 "Poor". SPY and TLT
both did. That was not a low score, it was a category error — the module
graded a fund against corporate filings that funds do not and will never
make, and then counted the same non-fact THREE times: required
completeness 0%, optional completeness 0%, and the three "statement data
unavailable" warnings dragging fetch reliability to 70. A fund has no
income statement; "the income statement is missing" is not evidence that
its data is untrustworthy. Meanwhile the things a fund DOES report —
expense ratio, net assets, category, family, a current price series —
were measured by nothing.

So the dimensions are declared per class, the same way asset_class.py
declares capabilities: a class either has a dimension or does not, and
the weights of the ones it has sum to 1. A fund is scored on fund data, a
crypto on price data, an equity on filings exactly as before. Verified:
AAPL still scores 98.5, so the equity path is untouched.

WHAT IS NOT SCORED, DELIBERATELY. Top holdings and sector weightings are
absent BY NATURE for bond and commodity funds — measured: SPY and QQQ
disclose ten holdings and ten or eleven sectors, TLT and GLD disclose
none. Scoring them would repeat the original mistake one level down,
penalising a bond fund for not being an equity fund. They are not
reported here either: they are not in the info dict (checked — no
holdings/sector key exists on it), they come from a separate funds_data
call, and this function runs on every page render. The Fund
Decomposition panel already shows them where they exist.
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import asset_class
from data_loader import STATEMENT_LABELS, TickerBundle, MacroBundle
from financial_standardization import StandardizedFinancials
from logging_setup import get_logger, log_event

logger = get_logger("data_quality")

STALENESS_THRESHOLD_DAYS = 120   # SEC quarterly filings are due within ~45 days of quarter-end
STALENESS_FLOOR_DAYS = 365       # score reaches 0 once data is this old

WEIGHT_REQUIRED_COMPLETENESS = 0.60
WEIGHT_OPTIONAL_COMPLETENESS = 0.15
WEIGHT_FRESHNESS = 0.15
WEIGHT_FETCH_RELIABILITY = 0.10

# A price series more than this many days old is not current. Four covers
# a long weekend plus a holiday on either side; equities and ETFs skip
# weekends while crypto does not, so this is deliberately tolerant rather
# than counting expected trading days per class.
PRICE_CURRENT_DAYS = 4
PRICE_STALE_FLOOR_DAYS = 30      # price score reaches 0 here

# The fields every fund reports. Measured across SPY, QQQ, TLT, VTV, ARKK
# and GLD — all six report all of these, including the bond fund and the
# commodity trust, so a gap here is a real gap rather than a class quirk.
FUND_PROFILE_FIELDS: Tuple[str, ...] = (
    "netExpenseRatio", "totalAssets", "category", "fundFamily",
    "legalType", "navPrice",
)


@dataclass(frozen=True)
class Dimension:
    """One thing quality is measured on, for the classes that have it."""
    key: str            # attribute on DataQualityReport holding the 0-100 score
    label: str
    weight: float
    help: str


REQUIRED_FIELDS_DIM = Dimension(
    "required_completeness_pct", "Required fields", 0.60,
    "Share of required balance sheet / income statement / cash flow "
    "fields present.")
OPTIONAL_FIELDS_DIM = Dimension(
    "optional_completeness_pct", "Optional fields", 0.15,
    "Share of optional statement fields present.")
FRESHNESS_DIM = Dimension(
    "freshness_score", "Filing freshness", 0.15,
    "Age of the most recently reported quarter.")
FUND_PROFILE_DIM = Dimension(
    "fund_profile_score", "Fund profile", 0.50,
    "Expense ratio, net assets, category, family, legal type and NAV — "
    "what a fund reports in place of filings.")
PRICE_HISTORY_DIM = Dimension(
    "price_history_score", "Price history", 0.20,
    "Whether the price series is present and current.")
PRICE_HISTORY_ONLY_DIM = Dimension(
    "price_history_score", "Price history", 0.60,
    "Whether the price series is present and current. For an instrument "
    "with no issuer this is the data the whole page is built from.")
FETCH_DIM = Dimension(
    "fetch_reliability_score", "Fetch reliability", 0.10,
    "Penalised for retried or failed downloads.")
FETCH_FUND_DIM = Dimension(
    "fetch_reliability_score", "Fetch reliability", 0.30,
    "Penalised for retried or failed downloads. A statement this "
    "instrument never files is not counted.")
FETCH_PRICE_DIM = Dimension(
    "fetch_reliability_score", "Fetch reliability", 0.40,
    "Penalised for retried or failed downloads. A statement this "
    "instrument never files is not counted.")

# Which dimensions apply to which class. Declared, not inferred — the same
# discipline asset_class.py uses for capabilities, and for the same
# reason: "this field is None" and "this field cannot exist here" are
# different facts and only one of them is a quality problem.
_EQUITY_DIMS = (REQUIRED_FIELDS_DIM, OPTIONAL_FIELDS_DIM, FRESHNESS_DIM, FETCH_DIM)
_FUND_DIMS = (FUND_PROFILE_DIM, PRICE_HISTORY_DIM, FETCH_FUND_DIM)
_PRICE_ONLY_DIMS = (PRICE_HISTORY_ONLY_DIM, FETCH_PRICE_DIM)

DIMENSIONS: Dict[str, Tuple[Dimension, ...]] = {
    asset_class.EQUITY: _EQUITY_DIMS,
    asset_class.ETF: _FUND_DIMS,
    asset_class.CRYPTO: _PRICE_ONLY_DIMS,
    asset_class.FOREX: _PRICE_ONLY_DIMS,
    asset_class.FUTURE: _PRICE_ONLY_DIMS,
    asset_class.INDEX: _PRICE_ONLY_DIMS,
    asset_class.UNKNOWN: _PRICE_ONLY_DIMS,
}


def dimensions_for(klass: str) -> Tuple[Dimension, ...]:
    """The dimensions quality is measured on for this class.

    An unrecognised class gets the price-only set rather than the equity
    set: assuming filings exist is what produced the original bug.
    """
    return DIMENSIONS.get(klass, _PRICE_ONLY_DIMS)


def _grade(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 55: return "Fair"
    return "Poor"


def _grade_icon(score: float) -> str:
    if score >= 75: return ""
    if score >= 55: return ""
    return ""


# Badge colours, one per grade. The task named three — green Excellent,
# yellow Good, red Poor — but this module has always graded on FOUR
# levels, and Fair (55-74) is the interesting one: data good enough to
# read but not to lean on. Collapsing it into either neighbour would
# misreport, so it gets amber between yellow and red rather than being
# rounded away to fit the sentence.
#
# These are the app's existing semantic colours, not new ones: the green
# and red are the same pair used for gain and loss everywhere else, which
# is correct here — this badge is a judgement about trustworthiness and
# reads in the same direction.
GRADE_COLOURS = {
    "Excellent": "#00ea77",
    "Good": "#eab308",
    "Fair": "#f97316",
    "Poor": "#ef4444",
}


def grade_colour(grade: str) -> str:
    """The badge colour for a grade. Unknown grades read as Poor rather
    than as a default green — an unrecognised grade is not evidence of
    good data."""
    return GRADE_COLOURS.get(grade, GRADE_COLOURS["Poor"])


_FILING_MEANINGS = {
    "Excellent": "Every required field is present and current. "
                 "Figures on this page can be read at face value.",
    "Good": "Minor gaps or slightly stale filings. The headline "
            "figures hold; check the detail before quoting a ratio.",
    "Fair": "Enough is missing or out of date that some ratios are "
            "estimates. Treat derived figures as indicative.",
    "Poor": "Key inputs are absent or badly stale. Valuations and "
            "ratios on this page may be unreliable.",
}

# The same four grades, said without reference to filings. An instrument
# with no statements was being told "every required field is present and
# current" — a sentence about evidence that had not been looked at, under
# a score that had not counted it.
_NON_FILING_MEANINGS = {
    "Excellent": "Everything this instrument reports is present, and its "
                 "price series is current. Figures on this page can be "
                 "read at face value.",
    "Good": "A small gap in what this instrument reports, or a price "
            "series a few days behind. The headline figures hold.",
    "Fair": "Enough of what this instrument reports is missing, or its "
            "price series is stale enough, that derived figures are "
            "indicative rather than exact.",
    "Poor": "Key inputs are absent or the price series is badly out of "
            "date. Figures on this page may be unreliable.",
}

_UNKNOWN_GRADE = "This grade is not recognised; treat the data as unverified."


def grade_meaning(grade: str, klass: str = asset_class.EQUITY) -> str:
    """One line on what the grade means for the numbers on screen.

    Worded for the evidence that was actually weighed: a fund's grade is
    not a statement about filings, so it must not be explained as one.
    """
    meanings = (_FILING_MEANINGS
                if asset_class.supports(klass, asset_class.FUNDAMENTALS)
                else _NON_FILING_MEANINGS)
    return meanings.get(grade, _UNKNOWN_GRADE)


@dataclass
class DataQualityReport:
    score: float
    grade: str
    grade_icon: str

    required_completeness_pct: float
    optional_completeness_pct: float
    freshness_score: float
    fetch_reliability_score: float

    most_recent_quarter: Optional[datetime.date]
    staleness_days: Optional[int]
    is_stale: bool

    missing_required_fields: List[str] = field(default_factory=list)
    missing_optional_fields: List[str] = field(default_factory=list)
    fetch_warnings: List[str] = field(default_factory=list)
    fetch_errors: List[str] = field(default_factory=list)

    # Which kind of instrument this was scored AS, and on what. The
    # statement-shaped fields above are still computed for every class so
    # the detail panel can list what is absent, but only the dimensions
    # below carry weight in `score`.
    asset_class: str = asset_class.EQUITY
    dimensions: Tuple[Dimension, ...] = _EQUITY_DIMS
    fund_profile_score: float = 0.0
    price_history_score: float = 0.0
    missing_fund_fields: List[str] = field(default_factory=list)
    price_age_days: Optional[int] = None

    @property
    def issue_count(self) -> int:
        """How many real, countable gaps there are.

        Counts only what was actually MEASURED for this class. Counting
        the fund-profile fields for a cryptocurrency reported "6
        field-level issue(s)" against an instrument that has no expense
        ratio or fund family to report — the original category error
        creeping back in one level down, so the count is derived from the
        dimensions rather than from whichever lists happen to be
        populated.
        """
        keys = {d.key for d in self.dimensions}
        count = len(self.fetch_errors)
        if "required_completeness_pct" in keys:
            count += (len(self.missing_required_fields)
                      + len(self.missing_optional_fields)
                      + len(self.fetch_warnings))
        if "fund_profile_score" in keys:
            count += len(self.missing_fund_fields)
        return count

    @property
    def scored_on(self) -> str:
        """One line naming what the score actually measured, for a reader
        who would otherwise assume it means the same thing everywhere."""
        return ", ".join(d.label.lower() for d in self.dimensions)


def _completeness_pct(present: int, total: int) -> float:
    return 100.0 if total == 0 else (present / total) * 100.0


def _is_statement_warning(warning: str) -> bool:
    """Does this warning say a corporate filing was absent?

    Matched against data_loader's own STATEMENT_LABELS rather than a
    literal copied here, so rewording the message there cannot silently
    turn "this fund has no income statement" back into a fetch failure.
    """
    lowered = warning.lower()
    return any(label in lowered for label in STATEMENT_LABELS)


def _fund_profile(info: Optional[dict]) -> Tuple[float, List[str]]:
    """How much of what a fund reports in place of filings is present."""
    info = info if isinstance(info, dict) else {}
    missing = [f for f in FUND_PROFILE_FIELDS if info.get(f) is None]
    present = len(FUND_PROFILE_FIELDS) - len(missing)
    return _completeness_pct(present, len(FUND_PROFILE_FIELDS)), missing


def _price_history(bundle: TickerBundle) -> Tuple[float, Optional[int]]:
    """Is the price series present and current? (score, age in days).

    Scored on CURRENCY, not on length. The user picks the date range, so
    a short series is a choice rather than a defect — but a series whose
    last bar is weeks old means the technicals, risk and simulation
    panels are all drawing stale conclusions, and for an instrument with
    no filings that is the whole basis of the page.
    """
    history = getattr(bundle, "price_history", None)
    if history is None or len(history) == 0:
        return 0.0, None
    try:
        last = history.index[-1]
        last_date = last.date() if hasattr(last, "date") else last
        age = (datetime.date.today() - last_date).days
    except (AttributeError, TypeError, IndexError):
        # An index that is not dates at all: present, but its currency
        # cannot be verified. Same posture as an unverifiable filing date.
        return 70.0, None
    if age <= PRICE_CURRENT_DAYS:
        return 100.0, age
    span = PRICE_STALE_FLOOR_DAYS - PRICE_CURRENT_DAYS
    return max(0.0, 100.0 - ((age - PRICE_CURRENT_DAYS) / span) * 100.0), age


def assess_data_quality(standardized: StandardizedFinancials, ticker_bundle: TickerBundle,
                        macro_bundle: MacroBundle,
                        klass: Optional[str] = None) -> DataQualityReport:
    """One data quality score (0-100) for the ticker currently under analysis,
    measured on the dimensions that exist for its ASSET CLASS.

    `klass` defaults to classifying the bundle's own info, so callers that
    do not care get the right answer; passing it explicitly lets a caller
    that has already classified avoid doing it twice.
    """
    if klass is None:
        klass = asset_class.classify(ticker_bundle.info, ticker_bundle.ticker)
    dimensions = dimensions_for(klass)

    validation = standardized.validation

    required_total = sum(1 for stmt in validation.statements for c in stmt.checks if c.required)
    required_present = sum(1 for stmt in validation.statements for c in stmt.checks if c.required and c.present)
    optional_total = sum(1 for stmt in validation.statements for c in stmt.checks if not c.required)
    optional_present = sum(1 for stmt in validation.statements for c in stmt.checks if not c.required and c.present)

    required_pct = _completeness_pct(required_present, required_total)
    optional_pct = _completeness_pct(optional_present, optional_total)

    most_recent_quarter = standardized.most_recent_quarter
    staleness_days: Optional[int] = None
    is_stale = False
    if most_recent_quarter is None:
        # Can't verify freshness — don't penalize for something we can't check,
        # but don't reward it as if it were confirmed fresh either.
        freshness_score = 70.0
    else:
        staleness_days = (datetime.date.today() - most_recent_quarter).days
        if staleness_days <= STALENESS_THRESHOLD_DAYS:
            freshness_score = 100.0
        else:
            is_stale = True
            span = STALENESS_FLOOR_DAYS - STALENESS_THRESHOLD_DAYS
            freshness_score = max(0.0, 100.0 - ((staleness_days - STALENESS_THRESHOLD_DAYS) / span) * 100.0)

    fetch_warnings = list(ticker_bundle.warnings) + list(macro_bundle.warnings)
    fetch_errors = list(ticker_bundle.errors)

    # A class with no filings must not be charged for not having filed.
    # These three warnings ("income statement data unavailable" and its
    # two siblings) fire on every single fund, and counting them was the
    # third place the same non-fact was being deducted for.
    counted_warnings = fetch_warnings
    if not asset_class.supports(klass, asset_class.FUNDAMENTALS):
        counted_warnings = [w for w in fetch_warnings if not _is_statement_warning(w)]
    fetch_reliability_score = max(
        0.0, 100.0 - len(fetch_errors) * 30.0 - len(counted_warnings) * 10.0)

    fund_profile_score, missing_fund_fields = _fund_profile(ticker_bundle.info)
    price_history_score, price_age_days = _price_history(ticker_bundle)

    # Only the dimensions this class actually has contribute. Their
    # weights sum to 1 by construction, asserted in the tests, so no
    # class is quietly graded out of less than 100.
    available = {
        "required_completeness_pct": required_pct,
        "optional_completeness_pct": optional_pct,
        "freshness_score": freshness_score,
        "fetch_reliability_score": fetch_reliability_score,
        "fund_profile_score": fund_profile_score,
        "price_history_score": price_history_score,
    }
    score = sum(d.weight * available[d.key] for d in dimensions)


    log_event(
        logger,
        logging.WARNING if _grade(score) in ("Poor", "Fair") else logging.INFO,
        "quality.assessed", ticker=standardized.ticker, score=round(score, 1),
        grade=_grade(score), asset_class=klass, required_pct=round(required_pct, 1),
        optional_pct=round(optional_pct, 1), freshness=round(freshness_score, 1),
        fetch=round(fetch_reliability_score, 1), stale=is_stale,
        staleness_days=staleness_days, fund_profile=round(fund_profile_score, 1),
        price_history=round(price_history_score, 1),
    )

    return DataQualityReport(
        score=round(score, 1),
        grade=_grade(score),
        grade_icon=_grade_icon(score),
        required_completeness_pct=round(required_pct, 1),
        optional_completeness_pct=round(optional_pct, 1),
        freshness_score=round(freshness_score, 1),
        fetch_reliability_score=round(fetch_reliability_score, 1),
        most_recent_quarter=most_recent_quarter,
        staleness_days=staleness_days,
        is_stale=is_stale,
        missing_required_fields=[f"{stmt.statement_name}: {name}" for stmt in validation.statements for name in stmt.missing_required],
        missing_optional_fields=[f"{stmt.statement_name}: {name}" for stmt in validation.statements for name in stmt.missing_optional],
        fetch_warnings=fetch_warnings,
        fetch_errors=fetch_errors,
        asset_class=klass,
        dimensions=dimensions,
        fund_profile_score=round(fund_profile_score, 1),
        price_history_score=round(price_history_score, 1),
        missing_fund_fields=missing_fund_fields,
        price_age_days=price_age_days,
    )
