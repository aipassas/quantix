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
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from data_loader import TickerBundle, MacroBundle
from financial_standardization import StandardizedFinancials
from logging_setup import get_logger, log_event

logger = get_logger("data_quality")

STALENESS_THRESHOLD_DAYS = 120   # SEC quarterly filings are due within ~45 days of quarter-end
STALENESS_FLOOR_DAYS = 365       # score reaches 0 once data is this old

WEIGHT_REQUIRED_COMPLETENESS = 0.60
WEIGHT_OPTIONAL_COMPLETENESS = 0.15
WEIGHT_FRESHNESS = 0.15
WEIGHT_FETCH_RELIABILITY = 0.10


def _grade(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 75: return "Good"
    if score >= 55: return "Fair"
    return "Poor"


def _grade_icon(score: float) -> str:
    if score >= 75: return ""
    if score >= 55: return ""
    return ""


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


def _completeness_pct(present: int, total: int) -> float:
    return 100.0 if total == 0 else (present / total) * 100.0


def assess_data_quality(standardized: StandardizedFinancials, ticker_bundle: TickerBundle, macro_bundle: MacroBundle) -> DataQualityReport:
    """Combine field-level validation, freshness, and fetch reliability into
    one data quality score (0-100) for the ticker currently under analysis.
    """
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
    fetch_reliability_score = max(0.0, 100.0 - len(fetch_errors) * 30.0 - len(fetch_warnings) * 10.0)

    score = (
        WEIGHT_REQUIRED_COMPLETENESS * required_pct
        + WEIGHT_OPTIONAL_COMPLETENESS * optional_pct
        + WEIGHT_FRESHNESS * freshness_score
        + WEIGHT_FETCH_RELIABILITY * fetch_reliability_score
    )

    log_event(
        logger,
        logging.WARNING if _grade(score) in ("Poor", "Fair") else logging.INFO,
        "quality.assessed", ticker=standardized.ticker, score=round(score, 1),
        grade=_grade(score), required_pct=round(required_pct, 1),
        optional_pct=round(optional_pct, 1), freshness=round(freshness_score, 1),
        fetch=round(fetch_reliability_score, 1), stale=is_stale,
        staleness_days=staleness_days,
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
    )
