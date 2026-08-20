"""Ranked stock suggestions from stated preferences.

WHAT THIS DELIBERATELY IS NOT. The originating task asked for a machine
learning model that suggests stocks to buy. ml_pipeline.py's momentum
classifier was measured before anything was designed:

    test accuracy      0.4913
    majority baseline  0.5324
    edge               -0.0410
    ROC AUC            0.4792     (0.50 = coin flip, 3,120 held-out rows)

No detectable edge — marginally below chance, consistent with noise.
ml_pipeline's own docstring predicted this: liquid-market prices already
reflect public information, so a baseline model on public technical
features should not be expected to show one.

Ranking suggestions by those probabilities would present noise as
intelligence on a screen someone uses to decide where money goes. That is
the most consequential version of the fabricated number this codebase
refuses everywhere else, so this module makes no prediction at all.

WHAT IT DOES INSTEAD. It ranks how well each candidate MATCHES THE
CRITERIA THE USER STATED — sector, risk tolerance, valuation leaning —
translated into thresholds over metrics the app already computes and
already discloses. The honest claim is "this fits what you asked for",
never "this will go up". There are no buy/sell/hold labels, no target
prices and no implied timing, because nothing here supports one.

IT REUSES THE SCREENER RATHER THAN DUPLICATING IT. screener.py already
evaluates a universe against ScreenCriterion filters using the same
metric functions as the single-ticker analysis. This module adds the
three things it lacks for this job: translating plain preferences into
those criteria, RANKING rather than pass/fail, and explaining why each
candidate placed where it did. No metric arithmetic lives here.

THRESHOLDS ARE MULTIPLES OF THE APP'S OWN CONFIGURED VALUES. A
conservative profile tightens SCORECARD.max_beta and max_debt_to_equity;
an aggressive one loosens them. Inventing a second set of numbers
alongside the first is how two sources of truth start disagreeing.

UNEVALUABLE CRITERIA ARE EXCLUDED FROM BOTH SIDES of the match score,
matching the Blueprint scorecard's convention exactly: a company is not
penalised for a metric it structurally cannot report. That also means a
candidate with very few evaluable criteria is not comparable to one with
many, so it is held back rather than flattered by a small denominator.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import RECOMMENDATIONS, SCORECARD
from logging_setup import get_logger, log_event, log_exception
from screener import ScreenCriterion
from user_thresholds import effective_scorecard

logger = get_logger("recommendations")


@dataclass(frozen=True)
class Preferences:
    """What the user said they want. Every field maps to criteria that
    are shown on screen — nothing here is applied invisibly."""
    sectors: Tuple[str, ...] = ()          # empty means any sector
    risk_profile: str = RECOMMENDATIONS.default_risk_profile
    valuation: str = RECOMMENDATIONS.default_valuation
    require_profitable: bool = True


@dataclass(frozen=True)
class CriterionOutcome:
    label: str
    metric: str
    operator: str
    threshold: float
    value: Optional[float] = None

    @property
    def evaluable(self) -> bool:
        return self.value is not None

    @property
    def passed(self) -> Optional[bool]:
        if self.value is None:
            return None
        if self.operator == "<=":
            return self.value <= self.threshold
        if self.operator == ">=":
            return self.value >= self.threshold
        return None


@dataclass(frozen=True)
class Suggestion:
    ticker: str
    sector: str = ""
    outcomes: Tuple[CriterionOutcome, ...] = ()
    status: str = "ok"
    detail: str = ""

    @property
    def evaluable(self) -> Tuple[CriterionOutcome, ...]:
        return tuple(o for o in self.outcomes if o.evaluable)

    @property
    def matched(self) -> Tuple[CriterionOutcome, ...]:
        return tuple(o for o in self.evaluable if o.passed)

    @property
    def missed(self) -> Tuple[CriterionOutcome, ...]:
        return tuple(o for o in self.evaluable if not o.passed)

    @property
    def unavailable(self) -> Tuple[CriterionOutcome, ...]:
        return tuple(o for o in self.outcomes if not o.evaluable)

    @property
    def match_pct(self) -> Optional[float]:
        """Share of EVALUABLE criteria met.

        Unevaluable criteria are excluded from both numerator and
        denominator, exactly as the Blueprint scorecard does — a company
        is not marked down for a figure it structurally cannot report.
        """
        if not self.evaluable:
            return None
        return len(self.matched) / len(self.evaluable) * 100

    @property
    def confidence_adjusted_pct(self) -> Optional[float]:
        """Match rate adjusted downward for how little it rests on.

        RANKING BY RAW PERCENTAGE REWARDS HAVING FEWER CRITERIA APPLY.
        Observed live: under a conservative profile JPM ranked FIRST at
        75%, judged on four criteria, because it is a bank and Altman Z
        and current ratio are structurally unavailable for it — while
        AVGO, judged on all six, ranked below at 67%. The bank looked
        best partly because two of the hardest tests could not be
        applied to it.

        This is the Wilson score lower bound, the standard answer to
        ranking proportions with unequal denominators. Three of four
        stays plausibly high; three of four when the other two simply
        could not be measured no longer outranks four of six. The RAW
        percentage is still what gets displayed — this only orders the
        list, and the evaluable count is shown next to every row so the
        basis is visible either way.
        """
        n = len(self.evaluable)
        if not n:
            return None
        import math

        z = 1.0                      # ~84%; enough to separate 4 from 8
        p = len(self.matched) / n
        denominator = 1 + z * z / n
        centre = p + z * z / (2 * n)
        margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return max(0.0, (centre - margin) / denominator) * 100

    @property
    def comparable(self) -> bool:
        """Whether enough criteria could be evaluated for the score to
        mean anything. "Matched 2 of 2" describes missing data, not a
        good fit, and ranking it above a company judged on eight would
        be actively misleading."""
        return len(self.evaluable) >= RECOMMENDATIONS.min_evaluable_criteria


# --- translating preferences into criteria ------------------------------------

def _scaled(value: float, multiplier: float, invert: bool = False) -> float:
    """Scale a shipped threshold by the risk profile.

    Ceilings scale directly — an aggressive profile tolerates more beta.
    Floors scale inversely, so a conservative profile demands a HIGHER
    minimum margin rather than a lower one. Without the inversion,
    "conservative" would quietly relax half the criteria.
    """
    return value / multiplier if invert else value * multiplier


def criteria_for(prefs: Preferences) -> Tuple[CriterionOutcome, ...]:
    """The criteria a preference set expands into, with no values yet.

    Returned as data rather than applied internally so the UI can show
    the user exactly what "Conservative" means in numbers before they
    act on any result. A preference whose effect is invisible is
    indistinguishable from one that does nothing.
    """
    scorecard = effective_scorecard()
    multiplier = RECOMMENDATIONS.risk_profiles.get(
        prefs.risk_profile, RECOMMENDATIONS.risk_profiles[RECOMMENDATIONS.default_risk_profile])

    out: List[CriterionOutcome] = []

    def add(label, metric, operator, threshold):
        out.append(CriterionOutcome(label=label, metric=metric,
                                    operator=operator, threshold=round(threshold, 2)))

    # Risk appetite — ceilings, so they scale directly with the profile.
    add("Beta at or below", "beta", "<=", _scaled(scorecard.max_beta, multiplier))
    add("Leverage (D/E) at or below", "debt_to_equity", "<=",
        _scaled(scorecard.max_debt_to_equity, multiplier))
    add("Annualised volatility at or below", "annual_volatility_pct", "<=",
        _scaled(30.0, multiplier))
    # Altman Z is a floor: a conservative profile wants a HIGHER score.
    add("Altman Z at or above", "altman_z", ">=", _scaled(1.81, multiplier, invert=True))

    # Quality floors — inverted for the same reason.
    if prefs.require_profitable:
        add("Net margin at or above", "net_margin_pct", ">=",
            _scaled(scorecard.min_net_margin * 100, multiplier, invert=True))
    add("Current ratio at or above", "current_ratio", ">=",
        _scaled(scorecard.min_current_ratio, multiplier, invert=True))

    # Valuation leaning.
    if prefs.valuation == "Value-leaning":
        add("P/E at or below", "pe_ratio", "<=", scorecard.pe_range[1] * 0.6)
        add("Price/book at or below", "price_to_book", "<=", 4.0)
    elif prefs.valuation == "Growth-leaning":
        # PEG rather than P/E: a growth company with a high multiple and
        # the earnings growth to match is exactly what this preference
        # means, and a raw P/E ceiling would exclude it.
        add("PEG at or below", "peg_ratio", "<=", scorecard.peg_range[1])

    return tuple(out)


def as_screen_criteria(outcomes: Sequence[CriterionOutcome]) -> Tuple[ScreenCriterion, ...]:
    """Criteria in the form screener.run_screen consumes.

    Passed to the screener purely so it computes the metric VALUES; the
    pass/fail it returns is deliberately ignored here, because this
    module needs a ranking rather than a filter.
    """
    return tuple(ScreenCriterion(metric=o.metric, operator=o.operator, threshold=o.threshold)
                 for o in outcomes)


# --- ranking ------------------------------------------------------------------

def rank(prefs: Preferences, universe: Sequence[str],
         screener: Optional[Callable] = None,
         sector_lookup: Optional[Callable[[str], str]] = None,
         limit: Optional[int] = None) -> Tuple[Tuple[Suggestion, ...], Tuple[str, ...]]:
    """Rank `universe` by how well each matches `prefs`.

    Returns (ranked, notes). `screener` and `sector_lookup` are injected
    so this is testable without the network.

    Sorting is by match percentage, then by how many criteria could be
    evaluated — a company judged on eight criteria ranks above one judged
    on four at the same percentage, because the first figure is better
    evidenced. Ties then break alphabetically so the order is stable
    between runs rather than shuffling on every rerun.
    """
    limit = limit if limit is not None else RECOMMENDATIONS.max_suggestions
    outcomes = criteria_for(prefs)
    notes: List[str] = []

    if not universe:
        return (), ("No tickers to consider — add some to a watchlist first.",)

    if screener is None:
        from screener import run_screen
        screener = run_screen
    if sector_lookup is None:
        sector_lookup = _default_sector_lookup

    try:
        results = screener(tuple(universe), as_screen_criteria(outcomes))
    except Exception as e:
        log_exception(logger, "recommendations.screen_failed", section="recommendations")
        return (), (f"Couldn't evaluate the universe ({type(e).__name__}).",)

    suggestions: List[Suggestion] = []
    filtered_by_sector = 0

    for result in results:
        sector = sector_lookup(result.ticker) or ""
        if prefs.sectors and sector not in prefs.sectors:
            filtered_by_sector += 1
            continue

        values = getattr(result, "values", {}) or {}
        suggestions.append(Suggestion(
            ticker=result.ticker,
            sector=sector,
            outcomes=tuple(
                CriterionOutcome(o.label, o.metric, o.operator, o.threshold,
                                 value=values.get(o.metric))
                for o in outcomes
            ),
            status=getattr(result, "status", "ok"),
            detail=getattr(result, "detail", ""),
        ))

    if filtered_by_sector:
        notes.append(
            f"{filtered_by_sector} candidate(s) excluded because their sector wasn't one "
            f"you selected."
        )

    comparable = [s for s in suggestions if s.comparable]
    thin = [s for s in suggestions if not s.comparable]
    if thin:
        notes.append(
            f"{len(thin)} candidate(s) had fewer than {RECOMMENDATIONS.min_evaluable_criteria} "
            f"evaluable criteria and were left out of the ranking — a high match on two "
            f"criteria says more about missing data than about the company."
        )

    # Ordered by the confidence-adjusted figure, not the raw one — see
    # Suggestion.confidence_adjusted_pct for the live case that made the
    # difference. Ties still break alphabetically so the list is stable
    # between reruns rather than reshuffling.
    comparable.sort(key=lambda s: (-(s.confidence_adjusted_pct or 0), -len(s.evaluable), s.ticker))
    log_event(logger, logging.INFO, "recommendations.ranked",
              considered=len(results), ranked=len(comparable))
    return tuple(comparable[:limit]), tuple(notes)


def _default_sector_lookup(ticker: str) -> str:
    """Sector from the shallow (info-only) bundle the app already caches,
    so this costs nothing beyond what the watchlist scan already fetched.
    Never raises — an unknown sector just can't match a sector filter."""
    try:
        from data_loader import load_ticker_bundle
        return str((load_ticker_bundle(ticker, deep=False).info or {}).get("sector") or "")
    except Exception:
        return ""


def available_sectors(universe: Sequence[str],
                      sector_lookup: Optional[Callable[[str], str]] = None) -> Tuple[str, ...]:
    """Sectors actually present in the universe, sorted.

    Offering sectors that nothing in the universe belongs to would let
    someone build a preference set that can only ever return nothing.
    """
    sector_lookup = sector_lookup or _default_sector_lookup
    found = {sector_lookup(t) for t in universe}
    return tuple(sorted(s for s in found if s))
