"""Achievement badges, for what you did and never for what the market did.

THE TICKET'S TWO NAMED BADGES ARE DECLINED, AND THE NUMBERS ARE WHY.
"Consistent Investor — N consecutive profitable trades" was measured
before anything was designed, over 73,457 real observations across thirty
large caps and ten years:

    a random 3-month position is profitable    65.2%   (not 50%)
    a random 6-month position                  70.0%
    a random 1-year position                   75.0%

At a 65.2% base rate, five profitable positions in a row happens to 11.8%
of people BY CHANCE — about one in eight, with no skill whatever. And it
measures the calendar rather than the person: five-in-a-row-by-chance ran
at 25.9% in 2017 and 2.3% in 2022, so the SAME badge is eleven times
easier to earn in one year than another. A badge that a rising market
hands out is not an achievement, it is a thermometer with a ribbon on it.

"First 5-Bagger" is the same failure with a bigger number. The contest
measurement already established that the largest gains land on a
top-quartile-volatility name 58% of the time against 25% by chance, and
that winning does not repeat. Awarding a 5x congratulates somebody for
having held the most volatile thing available, and teaches every reader
that this is the target — in an app that refuses to make a prediction
anywhere else.

SO EVERY BADGE HERE IS PROCESS. Recording a thesis before the outcome was
known; going back to review it once it was; keeping the habit running;
entering the contest before the month began; building a research setup
worth returning to. None of these can be handed to someone by a bull
market, and none is easier in one year than another. investment_journal
already freezes conviction the moment an outcome is seen, so the record
these are counted from is one hindsight cannot edit.

NOTHING IS PERSISTED AND NOTHING IS COLLECTED. `evaluate()` takes records
the caller already loaded and returns state; this module owns no store,
adds no switch and asks for no consent. One source of truth, nothing to
fall out of step, and no new file to declare. Cumulative counts never go
backwards, so a badge does not flicker between renders. Deleting the
entries that earned one does remove it, which is correct — the evidence
is the thing being recognised, not a trophy detached from it.

PRIVATE. Nothing here reaches the feed, the profile or the leaderboard.
There is already a leaderboard for what is genuinely comparable between
people, and it is opt-in and floored for good reasons.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config import BADGES
from logging_setup import get_logger

logger = get_logger("badges")

JOURNAL = "journal"
STREAK = "streak"
CONTEST = "contest"
RESEARCH = "research"

SOURCES: Tuple[str, ...] = (JOURNAL, STREAK, CONTEST, RESEARCH)

WHY_NO_OUTCOME_BADGES = (
    "No badge here is awarded for a profit, and that is a measurement rather than a "
    "preference. Across 73,457 observations, a random three-month position was "
    "profitable 65.2% of the time — so five profitable positions in a row happens to "
    "about one person in eight by chance alone, and it ran at 25.9% in 2017 against "
    "2.3% in 2022. A badge a rising market hands out measures the year, not you. "
    "Everything below is something you did."
)

NOTHING_YET = (
    "No badges yet. They are earned by recording a decision before you know how it "
    "turns out, coming back to review it afterwards, and keeping that up — never by "
    "a position going well."
)


@dataclass(frozen=True)
class BadgeSpec:
    """One badge. `target` is a count of a DELIBERATE ACT, never a return."""
    id: str
    name: str
    blurb: str
    source: str
    target: int
    unit: str


# The set. Ordered within each source from easiest to hardest, so a
# reader meets the one they are closest to first.
#
# A test asserts the EXACT count. Adding a badge should break it and be
# extended deliberately — relaxing it to >= would let one silently
# disappear, the same invariant quick_stats and support already carry.
BADGE_SPECS: Tuple[BadgeSpec, ...] = (
    # --- the journal: a record hindsight cannot edit ---
    BadgeSpec("first_thesis", "Wrote it down",
              "Recorded one investment decision with your reasoning, before you "
              "knew how it would turn out.", JOURNAL, 1, "decisions"),
    BadgeSpec("ten_theses", "Ten on the record",
              "Ten decisions recorded in advance. Enough that the pattern starts "
              "to be about you rather than about one call.", JOURNAL, 10, "decisions"),
    BadgeSpec("fifty_theses", "Fifty on the record",
              "Fifty decisions recorded in advance. A record long enough to read "
              "your own judgement back from.", JOURNAL, 50, "decisions"),
    BadgeSpec("first_review", "Went back and looked",
              "Reviewed one decision after its outcome was visible. The hardest "
              "habit in investing and the one nobody does.", JOURNAL, 1, "reviews"),
    BadgeSpec("ten_reviews", "Ten reviewed",
              "Went back to ten decisions once the results were in.",
              JOURNAL, 10, "reviews"),
    BadgeSpec("twelve_months_journal", "A year of decisions",
              "Recorded a decision in twelve different months. Measured in "
              "distinct months, so it cannot be earned in one afternoon.",
              JOURNAL, 12, "months"),

    # --- streaks: the habit itself ---
    BadgeSpec("week_streak", "A week running",
              "Seven consecutive days with a deliberate action. No grace days — "
              "a missed day ends the run.", STREAK, 7, "days"),
    BadgeSpec("month_streak", "A month running",
              "Thirty consecutive days with a deliberate action, unbroken.",
              STREAK, 30, "days"),
    BadgeSpec("hundred_days", "A hundred days in",
              "A hundred separate days on which you did something deliberate here, "
              "consecutive or not.", STREAK, 100, "active days"),

    # --- the contest: a call made before the window opened ---
    BadgeSpec("first_contest", "Called it in advance",
              "Entered the monthly contest once. Entries close before the month "
              "begins, so the pick was made blind.", CONTEST, 1, "months entered"),
    BadgeSpec("six_contests", "Six months in",
              "Entered six months. Participation and follow-through — never the "
              "result, which the contest deliberately does not rank.",
              CONTEST, 6, "months entered"),
    BadgeSpec("twelve_contests", "A full year entered",
              "Entered twelve months, each pick made before its month began.",
              CONTEST, 12, "months entered"),

    # --- the research setup you built ---
    BadgeSpec("first_screen", "Built a screen",
              "Saved one screener of your own, rather than running the same "
              "filters again by hand.", RESEARCH, 1, "saved screens"),
    BadgeSpec("five_screens", "Five screens saved",
              "Five screeners of your own, which is a research process rather "
              "than a one-off question.", RESEARCH, 5, "saved screens"),
    BadgeSpec("three_watchlists", "Organised",
              "Three named watchlists kept in parallel, which is a structure "
              "rather than a pile.", RESEARCH, 3, "watchlists"),
)


@dataclass(frozen=True)
class Evidence:
    """The counts every badge is derived from.

    Plain integers on purpose. Passing the stores themselves would let a
    badge reach into a record it was never meant to read, and would make
    this module depend on five other modules' shapes.
    """
    decisions: int = 0
    reviews: int = 0
    journal_months: int = 0
    longest_streak: int = 0
    active_days: int = 0
    contest_months: int = 0
    saved_screens: int = 0
    watchlists: int = 0

    def count_for(self, spec: BadgeSpec) -> int:
        return {
            "first_thesis": self.decisions, "ten_theses": self.decisions,
            "fifty_theses": self.decisions,
            "first_review": self.reviews, "ten_reviews": self.reviews,
            "twelve_months_journal": self.journal_months,
            "week_streak": self.longest_streak, "month_streak": self.longest_streak,
            "hundred_days": self.active_days,
            "first_contest": self.contest_months,
            "six_contests": self.contest_months,
            "twelve_contests": self.contest_months,
            "first_screen": self.saved_screens, "five_screens": self.saved_screens,
            "three_watchlists": self.watchlists,
        }.get(spec.id, 0)


@dataclass(frozen=True)
class Badge:
    spec: BadgeSpec
    progress: int = 0

    @property
    def earned(self) -> bool:
        return self.progress >= self.spec.target

    @property
    def remaining(self) -> int:
        return max(self.spec.target - self.progress, 0)

    @property
    def fraction(self) -> float:
        """0.0-1.0, capped. Used for a progress bar, which cannot take
        more than 1.0 without raising."""
        if self.spec.target <= 0:
            return 0.0
        return min(self.progress / self.spec.target, 1.0)

    def sentence(self) -> str:
        if self.earned:
            return f"{self.spec.name} — earned."
        return (f"{self.spec.name} — {self.progress} of {self.spec.target} "
                f"{self.spec.unit}.")


def evidence_from(journal_entries: Sequence = (), streak=None,
                  contest_entries: Sequence = (), saved_screens: int = 0,
                  watchlists: int = 0) -> Evidence:
    """Build the counts from records the caller already loaded.

    Every field is read defensively: this is a decorative panel and it
    must not take down the page it sits in over one malformed record.

    A REVIEW MEANS ONE WRITTEN AFTER THE OUTCOME WAS VISIBLE, which is
    what `reviewed_at` records. Counting entries that merely HAVE a
    review field, or counting the reasoning written at decision time,
    would award the hardest habit in investing for doing nothing.
    """
    entries = list(journal_entries or ())
    decisions = len(entries)
    reviews = sum(1 for e in entries
                  if str(getattr(e, "reviewed_at", "") or "").strip())
    months = {str(getattr(e, "decided_on", "") or "")[:7] for e in entries}
    months.discard("")

    return Evidence(
        decisions=decisions,
        reviews=reviews,
        journal_months=len(months),
        longest_streak=int(getattr(streak, "longest", 0) or 0),
        active_days=int(getattr(streak, "total_days", 0) or 0),
        contest_months=len({str(getattr(e, "period", "") or "")
                            for e in (contest_entries or ())} - {""}),
        saved_screens=max(int(saved_screens or 0), 0),
        watchlists=max(int(watchlists or 0), 0),
    )


def evaluate(evidence: Evidence) -> Tuple[Badge, ...]:
    """Every badge with its progress, earned ones first.

    Unearned badges are ordered by how close they are, so the panel can
    show what is within reach rather than a wall of things not done.
    """
    badges = [Badge(spec, evidence.count_for(spec)) for spec in BADGE_SPECS]
    badges.sort(key=lambda b: (not b.earned, -b.fraction, b.spec.target,
                               b.spec.id))
    return tuple(badges)


def earned(badges: Sequence[Badge]) -> Tuple[Badge, ...]:
    return tuple(b for b in badges if b.earned)


def next_up(badges: Sequence[Badge], limit: int = 3) -> Tuple[Badge, ...]:
    """The closest unearned badges that have actually been STARTED.

    A badge at zero is not "nearly there", and showing it as the next
    step would be inventing momentum. Falls back to nothing rather than
    padding the list.
    """
    started = [b for b in badges if not b.earned and b.progress > 0]
    return tuple(started[:limit])


def summary(badges: Sequence[Badge]) -> str:
    got = len(earned(badges))
    if got == 0:
        return NOTHING_YET
    return (f"{got} of {len(badges)} earned — each one for something you did, "
            f"never for a position going well.")


def by_source(badges: Sequence[Badge]) -> Dict[str, Tuple[Badge, ...]]:
    out: Dict[str, List[Badge]] = {source: [] for source in SOURCES}
    for badge in badges:
        out.setdefault(badge.spec.source, []).append(badge)
    return {source: tuple(items) for source, items in out.items()}
