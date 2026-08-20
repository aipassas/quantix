"""Tests for news_sentiment.py.

The load-bearing tests here are about not shipping a confident number
that means nothing:

  - THE OVERLAY MUST BEAT PLAIN VADER, measured on a labelled set. A
    future lexicon edit that degrades accuracy fails the suite.
  - THE PUBLISHED ACCURACY MUST BE THE MEASURED ONE. The UI states a
    figure; a test recomputes it rather than trusting the string.
  - IRRELEVANT ARTICLES MUST BE FILTERED. Yahoo's per-ticker feed mixes
    in other companies, and scoring those attributes someone else's news
    to this ticker.
  - TOO FEW ARTICLES MUST YIELD NO SCORE, not an average of two things.

No network: the fetcher is injected.
"""
import datetime

import pytest

import news_sentiment as ns
from config import NEWS_SENTIMENT
from news_sentiment import (
    FINANCE_LEXICON,
    LABELLED_HEADLINES,
    Article,
    accuracy_summary,
    analyse,
    classify,
    evaluate,
    is_relevant,
    normalise,
    score_text,
)


def yahoo_item(title, summary="", provider="Reuters", pub="2026-08-20T10:00:00Z", url="http://x"):
    return {"id": title, "content": {
        "title": title, "summary": summary,
        "provider": {"displayName": provider},
        "pubDate": pub, "canonicalUrl": {"url": url},
    }}


def fetcher(items):
    return lambda ticker: items


# --- the reason this module overrides the lexicon -----------------------------

def test_the_finance_overlay_beats_plain_vader():
    """THE CENTRAL CLAIM, MEASURED. Plain VADER is trained on social media
    and misreads financial text; if the overlay ever stopped helping,
    shipping it would be pointless."""
    tuned, total = evaluate(FINANCE_LEXICON)
    plain, _ = evaluate(None)
    assert tuned > plain, f"overlay {tuned}/{total} is no better than plain VADER {plain}/{total}"


def test_the_overlay_is_substantially_better_not_marginally():
    """Pins the improvement so a future edit can't quietly erode it to
    within noise while still technically 'beating' plain VADER."""
    tuned, total = evaluate(FINANCE_LEXICON)
    plain, _ = evaluate(None)
    assert tuned - plain >= 6, f"only +{tuned - plain} correct; the overlay has degraded"
    assert tuned / total >= 0.80


def test_plain_vader_really_is_bad_at_this():
    """Documents the premise. If a future VADER release fixed financial
    vocabulary, this fails and the overlay should be reconsidered rather
    than carried forever."""
    plain, total = evaluate(None)
    assert plain / total < 0.6, (
        f"plain VADER now scores {plain}/{total} — it may no longer need correcting"
    )


def test_the_published_accuracy_is_the_measured_one():
    """The UI states a figure. It has to be computed, not written down —
    a hardcoded claim would drift the moment the lexicon changed."""
    tuned, total = evaluate(FINANCE_LEXICON)
    plain, _ = evaluate(None)
    summary = accuracy_summary()
    assert f"{tuned} of {total}" in summary
    assert f"{plain} of {total}" in summary


def test_the_labelled_set_still_contains_failures():
    """A set the lexicon scores perfectly would measure how well it was
    fitted to its own test, not how it handles unseen headlines."""
    tuned, total = evaluate(FINANCE_LEXICON)
    assert tuned < total, "the labelled set has been tuned to a perfect score; it now proves nothing"


def test_every_labelled_case_has_a_valid_expectation():
    for text, expected in LABELLED_HEADLINES:
        assert text.strip()
        assert expected in ("positive", "negative", "neutral")


# --- the specific corrections -------------------------------------------------

@pytest.mark.parametrize("headline,expected", [
    ("Apple beats earnings estimates", "positive"),
    ("Apple misses earnings estimates", "negative"),
    ("Nvidia crushes revenue expectations", "positive"),
    ("Analyst downgrades stock to sell", "negative"),
    ("Analyst upgrades stock to buy", "positive"),
    ("Retailer files for bankruptcy protection", "negative"),
    ("Regulators fine bank over compliance failures", "negative"),
])
def test_finance_verbs_are_scored_correctly(headline, expected):
    assert classify(score_text(headline)) == expected


@pytest.mark.parametrize("term", ["debt", "liability", "risk", "exposure", "leverage"])
def test_neutral_technical_vocabulary_carries_no_polarity(term):
    """These are descriptive in finance. VADER scores them negative,
    which drags whole headlines the wrong way."""
    assert FINANCE_LEXICON[term] == 0.0


def test_a_regulatory_fine_is_not_fine():
    """VADER has "fine" at +0.8 — the everyday sense. In a market
    headline it is a penalty."""
    assert FINANCE_LEXICON["fine"] < 0


def test_crushing_expectations_is_good_news():
    assert FINANCE_LEXICON["crushes"] > 0


def test_scoring_empty_text_returns_none():
    assert score_text("") is None
    assert score_text("   ") is None


def test_classification_thresholds_follow_config():
    assert classify(NEWS_SENTIMENT.positive_threshold) == "positive"
    assert classify(NEWS_SENTIMENT.negative_threshold) == "negative"
    assert classify(0.0) == "neutral"


# --- relevance ----------------------------------------------------------------

def test_an_article_about_another_company_is_not_relevant():
    """The observed failure: a request for AAPL returned a QCOM headline
    as its top item. Scoring it would attribute Qualcomm's news to
    Apple."""
    article = Article(title="QCOM Falls 30.6% in 3 Months as Handset Risks Challenge Auto Growth")
    assert is_relevant(article, "AAPL", "Apple Inc.") is False


def test_an_article_naming_the_ticker_is_relevant():
    assert is_relevant(Article(title="AAPL rises on strong iPhone demand"), "AAPL", "Apple Inc.")


def test_an_article_naming_the_company_is_relevant():
    assert is_relevant(Article(title="Apple unveils new product line"), "AAPL", "Apple Inc.")


def test_a_body_mention_alone_is_not_enough():
    """MEASURED NARROWING, NOT A GUESS. Matching the summary let real
    Qualcomm articles through as Apple news — both mentioned Apple as a
    customer, and one did so negatively about Apple taking revenue away.
    That sentiment is about Qualcomm. A headline states what a piece is
    about; a body mention is often a comparison or supply-chain
    reference.

    The cost is recall, accepted deliberately: too few relevant articles
    produces no score rather than an average of someone else's news.
    """
    qualcomm = Article(
        title="Is QCOM Stock a Buy as Auto and AI Growth Offset Handset Weakness?",
        summary="QCOM's diversification gains momentum, but Apple exposure and handset "
                "weakness keep the buy case in question.")
    assert is_relevant(qualcomm, "AAPL", "Apple Inc.") is False
    assert is_relevant(qualcomm, "QCOM", "QUALCOMM Incorporated") is True


def test_legal_suffixes_do_not_prevent_a_match():
    """Requiring the full legal name would filter out nearly everything —
    headlines say "Apple", not "Apple Inc.". """
    assert is_relevant(Article(title="Apple gains"), "AAPL", "Apple Inc.")
    assert is_relevant(Article(title="Microsoft gains"), "MSFT", "Microsoft Corporation")


def test_suffix_words_alone_do_not_make_an_article_relevant():
    """Otherwise every article containing "group" or "holdings" would
    match any company with that in its name."""
    assert is_relevant(Article(title="Group holdings rise across the market"),
                       "AAPL", "Apple Inc.") is False


# --- fetching and normalising -------------------------------------------------

def test_the_current_nested_yahoo_shape_is_parsed():
    articles = normalise([yahoo_item("Apple beats", summary="Strong quarter")])
    assert len(articles) == 1
    assert articles[0].title == "Apple beats"
    assert articles[0].provider == "Reuters"
    assert articles[0].published == datetime.datetime(2026, 8, 20, 10, 0)


def test_the_older_flat_shape_is_also_parsed():
    """yfinance has changed this payload between versions; a shape change
    should cost sentiment, not the page."""
    articles = normalise([{
        "title": "Apple beats", "publisher": "Reuters",
        "providerPublishTime": 1787000000, "link": "http://x",
    }])
    assert len(articles) == 1 and articles[0].provider == "Reuters"


def test_items_without_a_title_are_skipped():
    assert normalise([{"content": {"summary": "no title here"}}, yahoo_item("Real")]) \
        == normalise([yahoo_item("Real")])


def test_malformed_items_do_not_break_parsing():
    articles = normalise([None, "garbage", 42, yahoo_item("Real headline")])
    assert len(articles) == 1


def test_an_unparseable_date_does_not_drop_the_article():
    articles = normalise([yahoo_item("Apple beats", pub="not-a-date")])
    assert len(articles) == 1 and articles[0].published is None
    assert articles[0].published_display == "date unknown"


# --- the assembled analysis ---------------------------------------------------

def test_a_positive_run_of_news_scores_positive():
    items = [yahoo_item(t) for t in (
        "Apple beats earnings estimates",
        "Apple soars on record quarterly profit",
        "Apple wins FDA approval for new health feature",
        "Apple upgraded to buy by analyst",
    )]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    assert result.has_score and result.label == "positive"
    assert result.positive >= 3


def test_a_negative_run_of_news_scores_negative():
    items = [yahoo_item(t) for t in (
        "Apple misses earnings estimates",
        "Apple plunges on weak guidance",
        "Apple faces regulatory probe over app store",
        "Apple downgraded to sell",
    )]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    assert result.has_score and result.label == "negative"
    assert result.negative >= 3


def test_too_few_relevant_articles_yields_no_score():
    """An average of two headlines is not a signal. Reporting one would
    be the module's whole failure mode."""
    result = analyse("AAPL", "Apple Inc.", fetcher([yahoo_item("Apple beats estimates")]))
    assert result.has_score is False
    assert result.score is None
    assert any("too few" in n.lower() for n in result.notes)
    assert result.articles           # but the headlines are still shown


def test_irrelevant_articles_are_filtered_and_the_drop_is_explained():
    items = [yahoo_item("QCOM falls on handset risks"),
             yahoo_item("Intel cuts guidance"),
             yahoo_item("Apple beats estimates"),
             yahoo_item("Apple soars on profit"),
             yahoo_item("Apple wins approval")]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    assert result.considered == 5
    assert len(result.articles) == 3
    assert any("didn't mention" in n for n in result.notes)


def test_counts_add_up_to_the_scored_articles():
    items = [yahoo_item(t) for t in (
        "Apple beats estimates", "Apple plunges on guidance", "Apple holds annual meeting")]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    scored = [a for a in result.articles if a.score is not None]
    assert result.positive + result.negative + result.neutral == len(scored)


def test_articles_are_capped():
    items = [yahoo_item(f"Apple news item {i}") for i in range(NEWS_SENTIMENT.max_articles + 15)]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    assert len(result.articles) <= NEWS_SENTIMENT.max_articles


def test_a_failing_feed_degrades_to_a_note():
    """Sentiment is a supporting panel. It must not be able to take down
    a ticker page."""
    def boom(ticker):
        raise RuntimeError("Yahoo is down")
    result = analyse("AAPL", "Apple Inc.", boom)
    assert result.has_score is False
    assert any("couldn't fetch" in n.lower() for n in result.notes)


def test_an_empty_feed_says_so():
    result = analyse("AAPL", "Apple Inc.", fetcher([]))
    assert result.has_score is False
    assert any("no usable" in n.lower() for n in result.notes)


def test_a_missing_ticker_is_refused():
    assert analyse("", fetcher=fetcher([])).has_score is False


def test_the_ticker_is_normalised():
    result = analyse("aapl", "Apple Inc.", fetcher([yahoo_item("Apple beats")]))
    assert result.ticker == "AAPL"


def test_each_article_carries_its_own_score_and_label():
    items = [yahoo_item(t) for t in (
        "Apple beats estimates", "Apple plunges on weak guidance", "Apple wins approval")]
    result = analyse("AAPL", "Apple Inc.", fetcher(items))
    by_title = {a.title: a for a in result.articles}
    assert by_title["Apple beats estimates"].label == "positive"
    assert by_title["Apple plunges on weak guidance"].label == "negative"
