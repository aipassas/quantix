"""Empty states that tell you what to do, and a button that does it.

WHY A SHARED RENDERER. The app already had helpful empty-state TEXT in
about a dozen places — the watchlist's "No tickers yet", the real-time
engine's "No active rules", the saved screeners' "Build a screen below".
What none of them had was a way to act on the advice. Left to each call
site, "and an action button" becomes a dozen slightly different layouts,
so the shape lives here once and each caller supplies only the words and
what the button does.

WHAT COUNTS AS AN ACTION. It has to change state, not gesture at a
control. Streamlit cannot focus an input or scroll to an element, so
"click here to jump to the add box" is not available and pretending
otherwise produces a button that appears to do nothing. Every action
wired to this module completes the job in one click — adds the ticker,
creates the rule, drops the filter — and each is reversible by the
control it points at.

THE SCREENER'S EMPTY STATE IS NOT EMPTY. "No stocks match. Try adjusting
filters" describes a blank result, but this screener always returns a row
per ticker with its own pass/fail; zero PASSING is not zero results, and
the table of failures stays on screen because it is the evidence. So the
guidance here does not say "try adjusting filters" and leave the reader
to guess which — blocking_criteria() counts how many tickers each
criterion actually rejected and names the worst one. That turns a
platitude into a fact the user can act on, and it is computable because
ScreenResult carries one pass/fail per criterion.

A criterion that could not be EVALUATED for a ticker (criteria_passes[i]
is None, meaning the metric was unavailable) is counted separately from
one the ticker genuinely failed. Folding the two together would report a
filter as "rejecting" companies it never got to judge.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import streamlit as st

from logging_setup import get_logger, log_event

logger = get_logger("empty_states")


def render(headline: str, guidance: str, *,
           action_label: Optional[str] = None,
           key: Optional[str] = None,
           help_text: Optional[str] = None,
           container=None) -> bool:
    """Draw an empty state. Returns True when its action was clicked.

    `container` lets a caller render into st.sidebar without this module
    knowing anything about where it is being used.
    """
    target = container if container is not None else st
    target.caption(f"**{headline}**")
    target.caption(guidance)
    if not action_label or not key:
        return False
    return bool(target.button(action_label, key=key, width="stretch",
                              help=help_text))


# --- why the screener returned nothing ----------------------------------------

@dataclass(frozen=True)
class Blocker:
    """One criterion, and how many tickers it turned away."""
    index: int
    text: str            # "P/E Ratio < 15"
    failed: int          # genuinely evaluated and failed
    unavailable: int     # metric could not be computed for that ticker
    considered: int      # tickers this criterion was evaluated against

    @property
    def share(self) -> float:
        return self.failed / self.considered if self.considered else 0.0

    def sentence(self) -> str:
        part = f"{self.text} — {self.failed} of {self.considered} failed it"
        if self.unavailable:
            part += f", and {self.unavailable} could not be measured"
        return part


def blocking_criteria(results: Sequence, criteria: Sequence) -> Tuple[Blocker, ...]:
    """Criteria ordered by how many tickers they rejected, worst first.

    `results` are ScreenResults and `criteria` the ScreenCriterions they
    were run against, positionally aligned — result.criteria_passes[i]
    belongs to criteria[i]. A result whose list is short (a fetch error,
    say) simply contributes nothing to the criteria past its end rather
    than being counted as a failure.
    """
    from screener import criterion_text

    blockers = []
    for index, criterion in enumerate(criteria):
        failed = unavailable = considered = 0
        for result in results:
            passes = getattr(result, "criteria_passes", None) or []
            if index >= len(passes):
                continue
            considered += 1
            outcome = passes[index]
            if outcome is None:
                unavailable += 1
            elif not outcome:
                failed += 1
        blockers.append(Blocker(
            index=index,
            text=criterion_text(getattr(criterion, "metric", ""),
                                getattr(criterion, "operator", "?"),
                                getattr(criterion, "threshold", None)),
            failed=failed, unavailable=unavailable, considered=considered,
        ))
    # Most-rejected first; ties broken by original order so the readout is
    # stable between runs rather than reshuffling on equal counts.
    return tuple(sorted(blockers, key=lambda b: (-b.failed, b.index)))


def screener_guidance(results: Sequence, criteria: Sequence) -> Tuple[str, Optional[Blocker]]:
    """(sentence, the criterion worth dropping first).

    Returns a blocker only when one actually rejected something — with a
    single criterion, or when every ticker failed on missing data, there
    is nothing useful to offer and the caller shows the sentence alone.
    """
    if not criteria:
        return "There are no filters to adjust.", None
    if not results:
        return ("Nothing was screened, so there is nothing to loosen — "
                "check the ticker universe."), None

    blockers = blocking_criteria(results, criteria)
    worst = blockers[0] if blockers else None
    if worst is None or worst.failed == 0:
        unavailable = sum(b.unavailable for b in blockers)
        if unavailable:
            return ("No ticker failed a filter outright — the metrics behind "
                    "them could not be computed for these companies."), None
        return "No filter rejected anything, so the universe itself is empty.", None

    sentence = f"Most restrictive: {worst.sentence()}."
    # Offering to drop the only filter would leave a screen that screens
    # nothing, which is not a useful next step.
    return sentence, (worst if len(criteria) > 1 else None)


def log_action(name: str, **fields) -> None:
    log_event(logger, logging.INFO, f"empty_state.{name}", **fields)
