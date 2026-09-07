"""Feature walkthroughs: step-by-step how-tos for each major feature.

THE TASK ASKED FOR VIDEOS AND THIS SHIPS WRITING. That is a deliberate
substitution of medium, not a silent downgrade, and the reasoning is the
same one support.py used to refuse a live-chat widget: a promise nobody
can keep is worse than an honest absence.

I cannot record video. Shipping a library of embeds pointing at URLs
that do not exist would put dead players in the app; shipping "video
coming soon" cards would be a promise with no date behind it. What the
ticket actually wants is the OUTCOME — "self-serve education that
reduces repetitive 'how do I use X' questions" — and written steps
deliver that outcome today, for every feature, in a form that is
searchable, copy-pasteable, screen-readable, and diffable when the UI
changes.

THE VIDEO PATH IS WIRED, NOT REMOVED. Every walkthrough carries an
optional `video_url`. Set one and the panel renders a player above the
written steps; leave it empty and the steps stand alone with no dead
link and no apology. So whoever can record them adds a URL per entry
and the library becomes what the ticket described, without a rewrite.

WHY THIS IS NOT ALREADY COVERED, measured rather than assumed. The help
corpus is 90 articles — but 77 of those are metric and chart
DEFINITIONS pulled from metric_help ("what does P/E mean"), and of the
13 task-oriented FAQ entries most answer "why isn't this working"
rather than "how do I do this". Counting genuine how-tos: three, against
roughly fifteen major features. The gap the ticket names is real.

THESE JOIN THE EXISTING SEARCH RATHER THAN STARTING A SECOND ONE.
support.build_index() folds these in, so a user searching "how do I
export" finds the walkthrough in the same box that already answers
"what is WACC". Building a separate search over a separate corpus would
mean the answer depends on which box you happened to type into. Note the
dependency runs support -> walkthroughs and never back, so this module
imports nothing from support and the two cannot form a cycle.

STEPS NAME THEIR CONTROLS, AND A TEST ENFORCES IT. Each step may record
the exact on-screen label of the control it tells the reader to use.
tests/test_walkthroughs.py asserts every one of those labels still
appears in finance.py, so renaming a button fails the suite instead of
silently leaving instructions that no longer match the app. A how-to
that rots is worse than no how-to: it costs the reader their trust in
the rest of the documentation.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Shown once at the top of the library, so the absence is stated rather
# than discovered by a reader hunting for a play button.
VIDEOS_NOT_RECORDED = (
    "These walkthroughs are written rather than filmed. No screen recordings "
    "ship with this build, and rather than show empty players or a "
    "\"coming soon\" card, each guide gives the steps in full. Every entry "
    "here has a slot for a video: add a URL and a player appears above the "
    "steps for that guide."
)


@dataclass(frozen=True)
class Step:
    """One instruction.

    `control` is the literal on-screen label of the button, tab, expander
    or field this step tells the reader to use. It is not decoration: a
    test asserts every non-empty value still exists in finance.py, which
    is what stops a renamed control from leaving stale instructions
    behind.
    """
    text: str
    control: str = ""


@dataclass(frozen=True)
class Walkthrough:
    id: str
    title: str
    summary: str
    where: str                      # where in the app this lives
    steps: Tuple[Step, ...]
    keywords: Tuple[str, ...] = ()
    # Empty until someone records one. See the module docstring: an entry
    # with no video renders its steps alone rather than an empty player.
    video_url: str = ""

    @property
    def has_video(self) -> bool:
        return bool((self.video_url or "").strip())

    @property
    def body(self) -> str:
        """The walkthrough as flowing text, for the search index and for
        anywhere that wants one string rather than a structure."""
        lines = [self.summary, "", f"Where: {self.where}", ""]
        for number, step in enumerate(self.steps, start=1):
            suffix = f" ({step.control})" if step.control else ""
            lines.append(f"{number}. {step.text}{suffix}")
        return "\n".join(lines)

    @property
    def controls(self) -> Tuple[str, ...]:
        return tuple(s.control for s in self.steps if s.control)


WALKTHROUGHS: Tuple[Walkthrough, ...] = (
    Walkthrough(
        id="wt_analyse_ticker",
        title="How do I analyse a stock?",
        summary=(
            "Everything in Quantix hangs off one symbol and one date range. "
            "Set those and every tab recalculates against them."
        ),
        where="Sidebar, then the tab strip across the top",
        keywords=("start", "begin", "first", "ticker", "symbol", "analyse", "analyze"),
        steps=(
            Step("Type the symbol into the sidebar. Non-US listings need their "
                 "exchange suffix — VWCE alone will not resolve, VWCE.DE will.",
                 control="Stock Ticker"),
            Step("If you only know the company name, search for it instead and "
                 "pick the listing you meant.", control="Find a ticker"),
            Step("Choose how much history to load. This is not cosmetic: a "
                 "one-year range cannot produce a 200-day moving average or a "
                 "one-year volatility figure, and those panels will say so.",
                 control="Analysis range"),
            Step("Work across the tabs. They are ordered from summary to "
                 "detail, and ⌘1–⌘8 jump straight to the first eight."),
        ),
    ),
    Walkthrough(
        id="wt_screener",
        title="How do I screen for stocks?",
        summary=(
            "Build a filter set, run it against the universe, and save the "
            "ones you want to reuse."
        ),
        where="Sidebar → the screener panel",
        keywords=("screen", "screener", "filter", "criteria", "find stocks"),
        steps=(
            Step("Open the screener.", control="Find stocks matching your criteria"),
            Step("Add a criterion, then choose its metric, operator and "
                 "threshold. Sector is categorical, so its operators are "
                 "is / is not rather than < / >.", control="+ Add Filter"),
            Step("Run it. Results that could not be judged — because the "
                 "company does not report that field — are counted separately "
                 "from results that failed the test.", control="Find matches"),
            Step("Keep a screen you will run again.", control="Save this screen"),
            Step("Check what the screener cannot do before trusting an empty "
                 "result.", control="What this screener cannot filter on"),
        ),
    ),
    Walkthrough(
        id="wt_screener_words",
        title="How do I build a screen by typing it in plain English?",
        summary=(
            "The screener accepts a sentence and turns it into criteria, so "
            "you do not have to know which metric is called what."
        ),
        where="Sidebar → the screener panel",
        keywords=("voice", "plain english", "natural language", "speak", "dictate"),
        steps=(
            Step("Open the plain-English box.", control="Ask in words"),
            Step("Describe what you want — for example \"profitable tech "
                 "companies with a P/E under 20 and a dividend above 2%\"."),
            Step("Read what it understood. Anything it could not parse is "
                 "listed back to you rather than dropped silently, so you can "
                 "add that filter by hand."),
            Step("Apply the criteria, then adjust them like any others."),
        ),
    ),
    Walkthrough(
        id="wt_dcf",
        title="How do I read the DCF valuation?",
        summary=(
            "A two-stage discounted cash flow with a CAPM-derived discount "
            "rate, plus a grid showing how sensitive the answer is to the two "
            "assumptions that move it most."
        ),
        where="Fundamentals & Valuation tab",
        keywords=("dcf", "valuation", "intrinsic", "wacc", "fair value"),
        steps=(
            Step("Open the model.", control="Professional Multi-Stage DCF Valuation"),
            Step("Check the beta line first. It says whether beta was "
                 "regressed against your benchmark, taken from Yahoo, or "
                 "fallen back to a declared 1.0 — the three are not equally "
                 "trustworthy."),
            Step("Read the sensitivity grid, not just the headline number. A "
                 "DCF whose value swings wildly across a one-point change in "
                 "WACC is telling you the estimate is fragile.",
                 control="Sensitivity Analysis: WACC vs Terminal Growth"),
            Step("If it reports that it could not compute, that is a real "
                 "answer: companies with structurally negative EBIT have no "
                 "meaningful DCF, and Quantix will not print $0.00 instead."),
        ),
    ),
    Walkthrough(
        id="wt_alerts",
        title="How do I set an alert?",
        summary=(
            "Rules are evaluated against the loaded data and raise a "
            "notification when they are met."
        ),
        where="Sidebar → alerts, and the bell in the header",
        keywords=("alert", "alerts", "notify", "notification", "rule", "trigger"),
        steps=(
            Step("Add a rule and set its condition.", control="+ Add Alert Rule"),
            Step("Evaluate the rules now rather than waiting.",
                 control="Check Alerts"),
            Step("Read how far the alerts actually reach before relying on "
                 "them.", control="How these alerts work"),
            Step("Alerts are evaluated while the page is open. They are not a "
                 "background service and will not reach you with the tab "
                 "closed — for that, set up the email digest or Slack."),
        ),
    ),
    Walkthrough(
        id="wt_watchlist",
        title="How do I keep a watchlist?",
        summary="Named lists of symbols, saved to your account.",
        where="Sidebar → watchlists",
        keywords=("watchlist", "watch", "list", "follow", "track"),
        steps=(
            Step("Create or rename a list.", control="Manage Watchlists"),
            Step("Add the symbol you are looking at to the active list.",
                 control="Add"),
            Step("Click any symbol on the list to switch the whole app to it."),
        ),
    ),
    Walkthrough(
        id="wt_portfolio",
        title="How do I track a portfolio?",
        summary=(
            "Enter what you actually hold — shares, cost and date — and the "
            "portfolio tab values it and measures its risk as one book."
        ),
        where="Portfolio tab",
        keywords=("portfolio", "holdings", "positions", "shares", "cost basis"),
        steps=(
            Step("Create a portfolio.", control="Manage portfolios"),
            Step("Add each holding: ticker, share count, price paid and the "
                 "date you bought.", control="Add position"),
            Step("Read the correlation and frontier panels once you hold more "
                 "than one thing — the point of them is the interaction "
                 "between positions, which no single position shows.",
                 control="Efficient Frontier"),
            Step("Nothing here is sent anywhere. Holdings are stored on this "
                 "machine, under your account."),
        ),
    ),
    Walkthrough(
        id="wt_export",
        title="How do I export a report?",
        summary=(
            "Three formats, all built from the analysis currently on screen."
        ),
        where="Sidebar → export",
        keywords=("export", "pdf", "powerpoint", "excel", "report", "download", "share"),
        steps=(
            Step("Open the export panel.", control="Export this analysis"),
            Step("A print-ready tear sheet.", control="Generate PDF Report"),
            Step("A slide deck, one section per panel.",
                 control="Generate PowerPoint"),
            Step("The figures as numbers rather than pictures of numbers.",
                 control="Generate Excel"),
            Step("Exports are always dark-themed regardless of your light/dark "
                 "setting. That is deliberate: a document that leaves the "
                 "building should not change appearance based on a personal "
                 "viewing preference."),
        ),
    ),
    Walkthrough(
        id="wt_journal",
        title="How do I keep a decision journal?",
        summary=(
            "Record why you decided something, then come back to it. It "
            "deliberately does not score you right or wrong."
        ),
        where="Overview tab → Investment Journal",
        keywords=("journal", "decision", "diary", "reasoning", "conviction", "review"),
        steps=(
            Step("Write down the action, your conviction, and the reasoning — "
                 "specifically the thesis, the trigger, and what would change "
                 "your mind.", control="Save entry"),
            Step("Come back later. Each entry shows the move since, the "
                 "holding period, and the benchmark over the same window."),
            Step("Add a review note rather than editing the original "
                 "reasoning. The original is the only record of what you "
                 "actually believed at the time."),
            Step("Conviction locks once an outcome has been shown to you. "
                 "Conviction is only evidence if it was recorded before the "
                 "result was known."),
            Step("Pattern reads need ten matured decisions in a band before "
                 "they say anything — a hit rate over two decisions is noise.",
                 control="What your record shows"),
        ),
    ),
    Walkthrough(
        id="wt_earnings",
        title="How do I search what management said?",
        summary=(
            "Full text of the earnings documents a company filed with the "
            "SEC, searchable across every quarter at once."
        ),
        where="Fundamentals & Valuation tab → Earnings Materials",
        keywords=("earnings", "transcript", "filings", "sec", "edgar",
                  "management", "commentary", "call"),
        steps=(
            Step("Choose how many quarters to read, then fetch them.",
                 control="Load filings from SEC"),
            Step("Search a term across all of them at once. Matching is "
                 "whole-word, so \"AI\" will not match \"said\"."),
            Step("Read the coverage line. It says whether any of what you "
                 "loaded is management commentary or whether the company "
                 "files results only."),
            Step("These are filed documents, not call transcripts. The "
                 "analyst Q&A from an earnings call is not filed with the SEC "
                 "and is not available here — use the company's "
                 "investor-relations site for the call itself."),
        ),
    ),
    Walkthrough(
        id="wt_historical",
        title="How do I compare a stock to how it looked before?",
        summary=(
            "Replays the whole analysis as of an earlier date and shows it "
            "beside today's."
        ),
        where="Overview tab",
        keywords=("historical", "before", "past", "compare", "then", "earlier"),
        steps=(
            Step("Open the comparison.", control="Historical Comparison"),
            Step("Pick the date to replay as of."),
            Step("Read the two side by side. The point is what changed in the "
                 "story, not only in the price."),
        ),
    ),
    Walkthrough(
        id="wt_thresholds",
        title="How do I change what counts as good?",
        summary=(
            "The scorecard judges companies against thresholds you can move."
        ),
        where="Sidebar → thresholds",
        keywords=("threshold", "thresholds", "scorecard", "criteria", "customise",
                  "customize", "blueprint"),
        steps=(
            Step("Open the threshold editor.", control="Custom Thresholds"),
            Step("Adjust any threshold. The Blueprint Alignment score "
                 "recalculates against your values, not the defaults."),
            Step("A metric a company cannot report is excluded from the score "
                 "rather than counted as a failure, so a bank is not punished "
                 "for having no meaningful current ratio."),
        ),
    ),
    Walkthrough(
        id="wt_team_notes",
        title="How do I share notes with a colleague?",
        summary=(
            "Per-ticker notes with @-mentions that email the person "
            "mentioned."
        ),
        where="Overview tab → team notes",
        keywords=("notes", "team", "collaborate", "mention", "share", "colleague"),
        steps=(
            Step("Add your colleagues to the roster first. A mention only "
                 "resolves against people already on it, which is what stops "
                 "a typo from mailing a stranger."),
            Step("Write the note and @-mention whoever should see it.",
                 control="Post note"),
            Step("Each note records whether its author was signed in or "
                 "simply typed a name, because \"Ana says sell\" means "
                 "different things in those two cases."),
            Step("Notes are shared across everyone using this instance, "
                 "deliberately — watchlists and settings are per-user, but a "
                 "thread nobody else can read is not collaboration."),
        ),
    ),
    Walkthrough(
        id="wt_shortcuts",
        title="What keyboard shortcuts are there?",
        summary="The ones that actually reach the page, measured rather than assumed.",
        where="Anywhere in the app",
        keywords=("keyboard", "shortcut", "shortcuts", "hotkey", "keys"),
        steps=(
            Step("Press ? to see the full list at any time."),
            Step("⌘1–⌘8 switch tabs; ⌘K focuses the ticker box."),
            Step("Alt+1 through Alt+9 are bound to the physical number keys, "
                 "so they work on macOS where Option+1 types a different "
                 "character."),
            Step("⌘N, ⌘T and ⌘W are reserved by the browser and never reach "
                 "the page, so nothing is bound to them."),
        ),
    ),
)


_BY_ID: Dict[str, Walkthrough] = {w.id: w for w in WALKTHROUGHS}


def by_id(walkthrough_id: str) -> Optional[Walkthrough]:
    return _BY_ID.get((walkthrough_id or "").strip())


def all_walkthroughs() -> Tuple[Walkthrough, ...]:
    return WALKTHROUGHS


def with_video() -> Tuple[Walkthrough, ...]:
    """Those that have a recording attached. Empty in this build — see
    VIDEOS_NOT_RECORDED — and the panel uses this to decide whether to
    mention video at all rather than hard-coding the absence."""
    return tuple(w for w in WALKTHROUGHS if w.has_video)


def search(query: str) -> Tuple[Walkthrough, ...]:
    """Walkthroughs matching a query, best first.

    Kept deliberately simple because the real search is support.search()
    over the assembled index; this exists for callers that want only
    walkthroughs, such as a "related how-to" link beside a panel.
    """
    terms = [t for t in (query or "").lower().split() if t]
    if not terms:
        return ()
    scored = []
    for walkthrough in WALKTHROUGHS:
        haystack = (walkthrough.title + " " + walkthrough.body + " "
                    + " ".join(walkthrough.keywords)).lower()
        score = sum(1 for t in terms if t in haystack)
        if score:
            scored.append((score, walkthrough))
    scored.sort(key=lambda pair: (-pair[0], pair[1].title))
    return tuple(w for _, w in scored)
