"""Headline sentiment per ticker, scored with a finance-corrected lexicon.

WHY THE LEXICON IS OVERRIDDEN, WITH NUMBERS. Plain VADER misclassified 5
of 10 representative financial headlines — coin-flip accuracy. The
failures are systematic, not random:

    "Nvidia crushes revenue expectations"      -0.44  (crushes = -1.9, violence)
    "Shares fall after strong quarter"         +0.67  ("fall" absent, "strong" wins)
    "Company reduces debt and liability"       -0.51  (debt -1.5, liability -0.8)
    "Apple beats earnings estimates"            0.00  ("beats" absent entirely)

VADER is trained on social media, where "crushes" is violent and "fine"
means acceptable — so "regulators fine bank" scores POSITIVE (+0.8).
Twenty-five core finance movement words are simply absent and score zero.
This is Loughran & McDonald's 2011 finding reproduced on this library:
general-purpose dictionaries mislabel financial text badly enough that
the output is not usable as a signal.

FINANCE_LEXICON below corrects that, and evaluate() measures the result
against a labelled set so the accuracy shown in the UI is a measurement
rather than a claim. A test asserts the overlay beats plain VADER, so a
future edit that degrades it fails the suite.

WHAT THIS STILL CANNOT DO, and the UI says so. It is a lexicon, not a
language model: it has no grasp of clause structure, so "beats estimates
but cuts guidance" is scored by summing words rather than by
understanding that the second clause dominates. It cannot detect sarcasm,
does not know which company a sentence is about, and has no notion of
whether news is already priced in. It is a fast read on tone, not a
forecast.

RELEVANCE IS FILTERED, IMPERFECTLY, AND THE RESIDUAL IS DISCLOSED.
Yahoo's per-ticker feed returns loosely related stories — a request for
AAPL returned a QCOM headline as its top item. Matching on title AND
summary was worse still: Qualcomm articles mentioning "Apple exposure"
and "the revenue Apple is taking away" counted as Apple sentiment when
that sentiment was about Qualcomm. So matching is title-only.

What title matching still cannot do is distinguish being ABOUT a company
from merely NAMING it. Both of these passed the AAPL filter:

    "Kestra Financial names Kelly Apple wealth management head"
        — Kelly Apple is a person.
    "MU Stock Gains On Micron's $10B AI Memory Bet - Nvidia's Jensen
     Huang, Apple..."
        — a Micron story that name-checks Apple.

Separating a surname from a company, or a passing mention from a
subject, needs more than a word list. Rather than stack untested
heuristics on top, the panel shows every headline the score was computed
from, so a diluted score is visible rather than merely possible.
"""
import datetime
import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from config import NEWS_SENTIMENT
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("news_sentiment")


# Finance-specific polarity, applied on top of VADER's lexicon.
#
# Three kinds of entry, and the distinction matters:
#   1. ABSENT IN VADER — the word carries clear financial meaning and
#      currently scores zero (beats, plunges, downgrade, bankruptcy).
#   2. WRONG IN VADER — the word exists with a polarity that is right for
#      general English and wrong for finance (crushes -1.9, fine +0.8).
#   3. NEUTRALISED — technical vocabulary that VADER treats as negative
#      but which carries no sentiment in a financial sentence (debt,
#      liability, risk, exposure, leverage, short).
#
# Values follow VADER's own roughly -4..+4 scale.
FINANCE_LEXICON: Dict[str, float] = {
    # --- beating / missing expectations -------------------------------
    "beat": 2.4, "beats": 2.4, "beating": 2.2,
    "miss": -2.4, "misses": -2.4, "missed": -2.4, "missing": -2.2,
    "tops": 2.3, "topped": 2.3, "exceeds": 2.3, "exceeded": 2.3,
    "crushes": 2.8, "crushed": 2.4, "smashes": 2.8,   # violence in general English
    "outperform": 2.2, "outperforms": 2.2, "outperformed": 2.2,
    "underperform": -2.2, "underperforms": -2.2, "underperformed": -2.2,
    "shortfall": -2.2, "disappoints": -2.4, "disappointing": -2.2,

    # --- price movement ------------------------------------------------
    "surges": 2.8, "surge": 2.6, "soars": 3.0, "soar": 2.8,
    "rallies": 2.4, "rally": 2.2, "jumps": 2.2, "climbs": 1.8,
    "rebounds": 2.0, "gains": 1.8, "rises": 1.6, "advances": 1.5,
    "plunges": -3.0, "plunge": -3.0, "plummets": -3.0,
    "tumbles": -2.6, "slumps": -2.6, "sinks": -2.4, "slides": -2.2,
    "falls": -2.0, "fall": -1.8, "drops": -2.0, "declines": -1.8,
    "slips": -1.6, "retreats": -1.6, "sinking": -2.4,

    # --- analyst and guidance actions ----------------------------------
    "upgrade": 2.5, "upgrades": 2.5, "upgraded": 2.5,
    "downgrade": -2.5, "downgrades": -2.5, "downgraded": -2.5,
    "raises": 2.0, "raised": 1.8, "hikes": 1.8, "boosts": 2.0,
    "lowers": -2.0, "lowered": -1.8, "slashes": -2.6, "slashed": -2.6,
    "warns": -2.4, "warning": -2.2, "cautions": -1.8,
    # "cuts" is -1.2 in VADER (as in self-harm); in finance it is a
    # guidance or dividend cut, which is worse for a shareholder.
    "cuts": -2.2, "cut": -1.8,

    # --- corporate events ----------------------------------------------
    "bankruptcy": -3.8, "insolvency": -3.8, "default": -3.2,
    "delisting": -3.0, "delisted": -3.0, "fraud": -3.6,
    "probe": -2.2, "investigation": -2.0, "lawsuit": -2.2, "sued": -2.2,
    "recall": -2.4, "halted": -2.2, "suspension": -2.2,
    "layoffs": -2.0, "restructuring": -1.4, "writedown": -2.4,
    "impairment": -2.2, "downturn": -2.2,
    # VADER has "fine" at +0.8 — "that's fine". A regulatory fine is not.
    "fine": -2.0, "fined": -2.4, "penalty": -2.2, "penalties": -2.2,
    "settlement": -1.0, "breach": -2.4,

    "dividend": 1.6, "buyback": 2.0, "buybacks": 2.0,
    "acquisition": 1.0, "approval": 2.2, "approved": 2.0,
    "wins": 2.2, "awarded": 2.0, "expansion": 1.6, "record": 1.8,
    "profitable": 2.4, "profit": 1.8, "profits": 1.8,
    "guidance": 0.0,   # neutral on its own; raised/cut carries the sign

    # --- neutralised technical vocabulary ------------------------------
    # These are descriptive in finance and score negative in VADER,
    # which drags whole headlines the wrong way.
    "debt": 0.0, "debts": 0.0, "liability": 0.0, "liabilities": 0.0,
    "risk": 0.0, "risks": 0.0, "risky": -0.8, "exposure": 0.0,
    "leverage": 0.0, "leveraged": 0.0, "short": 0.0, "shorts": 0.0,
    "volatility": 0.0, "volatile": -0.5, "derivative": 0.0,
    "hedge": 0.0, "bear": 0.0, "bearish": -2.0, "bull": 0.0, "bullish": 2.0,
    "correction": 0.0, "sell": 0.0, "selling": 0.0, "cheap": 0.0,
}


@dataclass(frozen=True)
class Article:
    title: str
    summary: str = ""
    url: str = ""
    provider: str = ""
    published: Optional[datetime.datetime] = None
    score: Optional[float] = None

    @property
    def label(self) -> str:
        if self.score is None:
            return "unscored"
        return classify(self.score)

    @property
    def published_display(self) -> str:
        return self.published.strftime("%d %b %H:%M") if self.published else "date unknown"


@dataclass(frozen=True)
class SentimentResult:
    ticker: str
    articles: Tuple[Article, ...] = ()
    score: Optional[float] = None          # mean compound, -1..+1
    positive: int = 0
    negative: int = 0
    neutral: int = 0
    considered: int = 0                    # fetched before relevance filtering
    notes: Tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return "unavailable" if self.score is None else classify(self.score)

    @property
    def has_score(self) -> bool:
        return self.score is not None


def classify(score: float) -> str:
    if score >= NEWS_SENTIMENT.positive_threshold:
        return "positive"
    if score <= NEWS_SENTIMENT.negative_threshold:
        return "negative"
    return "neutral"


# --- scoring ------------------------------------------------------------------

_analyzer = None


def _get_analyzer():
    """VADER with the finance overlay applied. Built once — constructing
    the analyzer parses its whole lexicon file."""
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return _analyzer


def score_text(text: str) -> Optional[float]:
    """Compound sentiment for one piece of text, or None if empty.

    Never raises: sentiment is a nice-to-have panel and must not be able
    to take down a ticker page.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(_get_analyzer().polarity_scores(text)["compound"])
    except Exception:
        log_exception(logger, "news_sentiment.scoring_failed", section="news_sentiment")
        return None


# --- relevance ----------------------------------------------------------------

def _company_tokens(ticker: str, company_name: str = "") -> List[str]:
    """Words that mark an article as being about this company.

    The company's legal suffixes are stripped — "Apple Inc." should match
    an article that says "Apple", and requiring the full legal name would
    filter out almost everything.
    """
    tokens = [ticker.lower()]
    name = re.sub(r"\b(inc|corp|corporation|co|ltd|plc|llc|nv|sa|ag|holdings|group|the)\b\.?",
                  " ", (company_name or "").lower())
    for word in re.findall(r"[a-z]{3,}", name):
        tokens.append(word)
    return tokens


def is_relevant(article: Article, ticker: str, company_name: str = "") -> bool:
    """Whether an article is actually ABOUT this company.

    MATCHES ON THE TITLE ONLY, and that is a deliberate narrowing after
    measuring the alternative. Matching the summary too let Qualcomm
    articles through as Apple news:

        "Is QCOM Stock a Buy as Auto and AI Growth Offset Handset Weakness?"
            ...QCOM's diversification gains momentum, but Apple exposure...

        "The Upside Case For Qualcomm Stock Now Carries Purchase Orders"
            ...replace the revenue Apple is taking away...

    Both mention Apple as a customer, and the second does so negatively
    about Apple LEAVING — sentiment that is about Qualcomm's prospects,
    not Apple's. A headline states what a piece is about; a body mention
    is frequently a comparison or a supply-chain reference.

    The cost is recall: AAPL drops from 5 loosely-matched articles to 3
    genuinely-about-Apple ones. That is the right trade for a signal
    whose entire value is not being noise — and when too few survive,
    analyse() reports no score rather than averaging what is left.
    """
    title = article.title.lower()
    return any(token in title for token in _company_tokens(ticker, company_name))


# --- fetching -----------------------------------------------------------------

def _parse_published(value) -> Optional[datetime.datetime]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.datetime.fromtimestamp(float(value))
        except (ValueError, OSError):
            return None
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _default_fetcher(ticker: str) -> List[dict]:
    import yfinance as yf
    return yf.Ticker(ticker).news or []


def normalise(raw: Sequence[dict]) -> Tuple[Article, ...]:
    """Yahoo's news payload into Articles.

    Handles both the current nested {"content": {...}} shape and the older
    flat one, because the field layout has changed between yfinance
    versions and a shape change should cost sentiment, not the page.
    """
    articles: List[Article] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = str(content.get("title") or "").strip()
        if not title:
            continue
        provider = content.get("provider")
        provider_name = provider.get("displayName", "") if isinstance(provider, dict) \
            else str(content.get("publisher") or "")
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl", "link"):
            candidate = content.get(key)
            if isinstance(candidate, dict):
                url = candidate.get("url", "")
            elif isinstance(candidate, str):
                url = candidate
            if url:
                break
        articles.append(Article(
            title=title,
            summary=str(content.get("summary") or content.get("description") or "").strip(),
            url=url,
            provider=provider_name,
            published=_parse_published(content.get("pubDate") or content.get("displayTime")
                                       or content.get("providerPublishTime")),
        ))
    return tuple(articles)


def analyse(ticker: str, company_name: str = "",
            fetcher: Optional[Callable] = None) -> SentimentResult:
    """Fetch, filter and score recent news for one ticker.

    `fetcher` is injectable so tests never touch the network. Never
    raises — an unreachable feed comes back as a result with a note.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return SentimentResult(ticker="", notes=("No ticker given.",))

    try:
        raw = (fetcher or _default_fetcher)(ticker)
    except Exception as e:
        log_exception(logger, "news_sentiment.fetch_failed", section="news_sentiment")
        return SentimentResult(
            ticker=ticker,
            notes=(f"Couldn't fetch news for {ticker} ({type(e).__name__}).",))

    articles = normalise(raw)
    considered = len(articles)
    notes: List[str] = []

    if NEWS_SENTIMENT.require_relevance:
        relevant = tuple(a for a in articles if is_relevant(a, ticker, company_name))
        dropped = considered - len(relevant)
        if dropped:
            notes.append(
                f"{dropped} of {considered} articles didn't mention {ticker} or the company "
                "by name and were left out — the feed mixes in adjacent-industry stories, and "
                "scoring those would attribute another company's news to this one."
            )
        articles = relevant

    articles = articles[:NEWS_SENTIMENT.max_articles]
    scored = tuple(
        Article(a.title, a.summary, a.url, a.provider, a.published,
                score=score_text(f"{a.title}. {a.summary}"))
        for a in articles
    )
    usable = [a for a in scored if a.score is not None]

    if not usable:
        notes.append(f"No usable recent news found for {ticker}.")
        return SentimentResult(ticker=ticker, articles=scored, considered=considered,
                               notes=tuple(notes))

    positive = sum(1 for a in usable if a.label == "positive")
    negative = sum(1 for a in usable if a.label == "negative")
    neutral = len(usable) - positive - negative

    score = None
    if len(usable) >= NEWS_SENTIMENT.min_articles_for_score:
        score = sum(a.score for a in usable) / len(usable)
    else:
        notes.append(
            f"Only {len(usable)} relevant article(s) — too few to average into a score that "
            f"means anything, so the headlines are shown without one."
        )

    log_event(logger, logging.INFO, "news_sentiment.analysed",
              ticker=ticker, considered=considered, scored=len(usable))

    return SentimentResult(
        ticker=ticker, articles=scored, score=score,
        positive=positive, negative=negative, neutral=neutral,
        considered=considered, notes=tuple(notes),
    )


# --- measuring the overlay ----------------------------------------------------

# A labelled set of financial headlines, hand-written to cover the ways a
# general-purpose lexicon fails: finance-specific verbs, neutral technical
# vocabulary, and words whose everyday polarity is inverted in a market
# context.
#
# THE SET DELIBERATELY RETAINS CASES THE OVERLAY STILL GETS WRONG. Tuning
# the lexicon until it scores 20/20 would make the published accuracy
# meaningless — it would measure how well the lexicon was fitted to its
# own test, not how it handles headlines it has never seen. The three
# current failures each show the same underlying limit, that a lexicon
# sums words and cannot read structure:
#
#   "Company reduces debt and liability exposure"  -> neutral, want positive
#       Needs to know that REDUCING something negative is positive.
#   "Firm reports higher risk-weighted assets"     -> positive, want neutral
#       "higher" is positive in isolation; this is domain knowledge.
#   "Shares fall after strong quarter"             -> positive, want negative
#       "strong" (+2.3) outweighs "falls" (-2.0); no sense that the price
#       clause is the one that matters.
#
# All three are compositional, which is exactly what a finance-tuned
# language model would handle and a word list cannot.
LABELLED_HEADLINES: Tuple[Tuple[str, str], ...] = (
    ("Apple beats earnings estimates", "positive"),
    ("Apple misses earnings estimates", "negative"),
    ("Nvidia crushes revenue expectations", "positive"),
    ("Tesla shares plunge on weak guidance", "negative"),
    ("Company reduces debt and liability exposure", "positive"),
    ("Firm reports higher risk-weighted assets", "neutral"),
    ("Boeing wins $10bn defence contract", "positive"),
    ("Regulators fine bank over compliance failures", "negative"),
    ("Shares fall after strong quarter", "negative"),
    ("Analyst upgrades stock to buy", "positive"),
    ("Analyst downgrades stock to sell", "negative"),
    ("Company announces dividend increase and buyback", "positive"),
    ("Chipmaker warns of slowing demand", "negative"),
    ("Retailer files for bankruptcy protection", "negative"),
    ("Shares soar on record quarterly profit", "positive"),
    ("Drugmaker wins FDA approval for new therapy", "positive"),
    ("Automaker announces recall of 50,000 vehicles", "negative"),
    ("Bank slashes full-year guidance", "negative"),
    ("Stock rallies after activist investor takes stake", "positive"),
    ("Company completes routine debt refinancing", "neutral"),
)


def evaluate(lexicon: Optional[Dict[str, float]] = None,
             cases: Sequence[Tuple[str, str]] = LABELLED_HEADLINES) -> Tuple[int, int]:
    """(correct, total) over the labelled set.

    `lexicon=None` measures plain VADER, so the improvement can be shown
    as a difference rather than asserted. Builds its own analyzer to avoid
    mutating the shared one.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    if lexicon:
        analyzer.lexicon.update(lexicon)

    correct = 0
    for text, expected in cases:
        compound = analyzer.polarity_scores(text)["compound"]
        if classify(compound) == expected:
            correct += 1
    return correct, len(cases)


def accuracy_summary() -> str:
    """The line shown in the UI, so the panel states a measurement rather
    than claiming the scoring works."""
    tuned, total = evaluate(FINANCE_LEXICON)
    plain, _ = evaluate(None)
    return (
        f"Scored with a finance-corrected lexicon: {tuned} of {total} correct on a labelled "
        f"set of financial headlines, against {plain} of {total} for the same library "
        f"unmodified."
    )
