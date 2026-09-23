"""What other accounts looked at: counts, never names, never yourself.

THE TEST THAT MATTERS MOST is test_a_row_cannot_carry_an_identity. The
feed names a publisher to the people who FOLLOW them; this widget is
visible to every account on the instance, including ones that follow
nobody. A wider audience must carry strictly less.

The second is test_this_is_not_the_market_trending_panel. ticker_discovery
already reports most-active, gainers and losers from live Yahoo screens
and is already in the sidebar. Rendering that here under a social label
would be the same data wearing the wrong claim — the exact mistake
ticker_discovery's own docstring was written to avoid.
"""
import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import peer_trending as pt
from config import PEER_TRENDING, FOLLOWING

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"
NOW = datetime.datetime(2026, 9, 23, 12, 0, 0)


class _Profile:
    def __init__(self, user_key, streams=(pt.STREAM,), name="Someone"):
        self.user_key = user_key
        self.name = name
        self.streams = tuple(streams)

    def shares(self, stream):
        return stream in self.streams


class _Profiles:
    def __init__(self, *profiles):
        self.profiles = tuple(profiles)

    def get(self, key):
        return next((p for p in self.profiles if p.user_key == key), None)


class _Event:
    def __init__(self, actor_key, ticker, hours_ago=1, stream=pt.STREAM):
        self.actor_key = actor_key
        self.ticker = ticker
        self.stream = stream
        self.at = (NOW - datetime.timedelta(hours=hours_ago)).isoformat(
            timespec="seconds")


class _Feed:
    def __init__(self, *events):
        self.events = tuple(events)


def _sharers(*keys):
    return _Profiles(*[_Profile(k) for k in keys])


# --- it is not the market panel -----------------------------------------------

def test_this_is_not_the_market_trending_panel():
    """ticker_discovery already does the market. This must not duplicate
    it or quietly relabel it."""
    src = Path(pt.__file__).read_text()
    assert "ticker_discovery" not in src.replace(
        "ticker_discovery's", "").replace("ticker_discovery already", "") \
        or "import ticker_discovery" not in src
    imports = [ln for ln in src.splitlines()
               if ln.startswith(("import ", "from ")) and "ticker_discovery" in ln]
    assert imports == [], "this module must not reach into the market screens"
    assert "yfinance" not in src and "most_actives" not in src


def test_the_empty_state_points_at_the_market_panel_rather_than_faking_one():
    assert "Find a ticker" in pt.MARKET_TRENDING_INSTEAD
    assert "market" in pt.MARKET_TRENDING_INSTEAD.lower()
    assert "market data wearing a social label" in pt.NOBODY_SHARING


# --- nothing new is collected -------------------------------------------------

def test_the_module_owns_no_store():
    """The viewed stream is the collection and the consent. A second
    store would be a second copy to fall out of step, and a second
    opt-in nobody was asked for."""
    src = Path(pt.__file__).read_text()
    # The IMPORT, not just the call. Banning "shared_path(" with its
    # paren passes a build that merely imports the name — proven by a
    # poison. This module has no business touching local_store at all.
    # Stripped before matching: an import smuggled in INSIDE a function
    # is indented, and a filter anchored to column zero misses it — also
    # proven by a poison.
    persistence = [ln.strip() for ln in src.splitlines()
                   if ln.strip().startswith(("import ", "from "))
                   and ("local_store" in ln or ln.strip() == "import json")]
    assert persistence == [], f"this module must persist nothing: {persistence}"
    for banned in ("shared_path(", "store_path(", "atomic_write_text",
                   "save_store", "json.dumps", "write_text"):
        assert banned not in src, f"{banned} — this module must persist nothing"
    assert not any(name for name in dir(pt)
                   if name in ("load_store", "save_store", "_store_path"))


def test_it_reads_only_the_viewed_stream():
    """Folding in watchlist or journal events would use a consent given
    for a different purpose."""
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"),
                 _Event("a", "MSFT", stream="watchlist"),
                 _Event("b", "MSFT", stream="watchlist"))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["NVDA"]


def test_the_stream_name_matches_followings():
    import following
    assert pt.STREAM == following.STREAM_VIEWED


# --- consent ------------------------------------------------------------------

def test_an_account_with_the_stream_off_does_not_count():
    """Re-checked on every read, so switching off takes effect before any
    purge runs — the discipline following.feed_for already uses."""
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    profiles = _Profiles(_Profile("a"), _Profile("b", streams=()))
    result = pt.trending(feed, profiles, "me", now=NOW)
    assert result.rows == ()
    assert result.sharing_accounts == 1


def test_an_account_with_no_profile_does_not_count():
    feed = _Feed(_Event("a", "NVDA"), _Event("ghost", "NVDA"))
    result = pt.trending(feed, _sharers("a"), "me", now=NOW)
    assert result.rows == ()


def test_nobody_sharing_is_distinguished_from_nobody_looking():
    """Different answers. One means the feature is unused; the other
    means it is quiet."""
    quiet = pt.trending(_Feed(), _sharers("a", "b"), "me", now=NOW)
    assert quiet.reason != pt.NOBODY_SHARING
    assert "at least 2 different accounts" in quiet.reason

    unused = pt.trending(_Feed(_Event("a", "NVDA")), _Profiles(), "me", now=NOW)
    assert unused.reason == pt.NOBODY_SHARING


def test_an_account_sharing_but_silent_still_counts_as_participating():
    result = pt.trending(_Feed(), _sharers("a", "b", "c"), "me", now=NOW)
    assert result.sharing_accounts == 3


# --- your own views -----------------------------------------------------------

def test_your_own_views_are_never_counted():
    """Reflecting your own browsing back as a social signal is the most
    misleading version of social proof there is."""
    feed = _Feed(_Event("me", "TSLA"), _Event("me", "TSLA", hours_ago=2),
                 _Event("a", "TSLA"), _Event("b", "TSLA"))
    result = pt.trending(feed, _sharers("me", "a", "b"), "me", now=NOW)
    (row,) = result.rows
    assert row.ticker == "TSLA" and row.viewers == 2, "the reader was counted"
    assert result.sharing_accounts == 2, "the reader is not one of the others"


def test_a_ticker_only_you_looked_at_does_not_appear():
    feed = _Feed(_Event("me", "TSLA"), _Event("me", "TSLA", hours_ago=3))
    result = pt.trending(feed, _sharers("me", "a", "b"), "me", now=NOW)
    assert result.rows == ()


def test_a_signed_out_reader_excludes_nobody():
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    result = pt.trending(feed, _sharers("a", "b"), "", now=NOW)
    assert result.rows[0].viewers == 2


# --- the floor ----------------------------------------------------------------

def test_one_viewer_is_a_person_not_a_trend():
    feed = _Feed(_Event("a", "NVDA"))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert result.rows == ()
    assert "not a trend" in result.reason


def test_the_floor_is_inclusive_at_its_own_boundary():
    feed = _Feed(*[_Event(f"k{i}", "NVDA") for i in range(PEER_TRENDING.min_viewers)])
    result = pt.trending(feed, _sharers(*[f"k{i}" for i in range(PEER_TRENDING.min_viewers)]),
                         "me", now=NOW)
    assert result.rows and result.rows[0].viewers == PEER_TRENDING.min_viewers


def test_the_floor_is_deliberately_lower_than_peer_comparisons():
    """Different disclosures. A return percentile states other people's
    figures; 'two accounts opened NVDA' names nobody."""
    from config import PEER_COMPARISON
    assert PEER_TRENDING.min_viewers < PEER_COMPARISON.min_cohort
    assert PEER_TRENDING.min_viewers >= 2, "one viewer must never be listed"


# --- distinct accounts, not events --------------------------------------------

def test_the_same_account_twice_is_one_viewer():
    """The feed dedupes within 24h, but relying on that would make this
    correct only by coincidence — the two windows are configured
    separately."""
    feed = _Feed(_Event("a", "NVDA", hours_ago=1),
                 _Event("a", "NVDA", hours_ago=2),
                 _Event("a", "NVDA", hours_ago=3),
                 _Event("b", "NVDA", hours_ago=1))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert result.rows[0].viewers == 2, "events were counted instead of accounts"


def test_the_two_windows_are_independent_settings():
    """If this ever relied on following's dedupe, changing one would
    silently break the other."""
    feed = _Feed(_Event("a", "NVDA", hours_ago=1),
                 _Event("a", "NVDA", hours_ago=FOLLOWING.viewed_dedupe_hours + 1),
                 _Event("b", "NVDA", hours_ago=1))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW,
                         window_hours=FOLLOWING.viewed_dedupe_hours + 10)
    assert result.rows[0].viewers == 2


# --- the window ---------------------------------------------------------------

def test_an_event_outside_the_window_is_not_counted():
    feed = _Feed(_Event("a", "NVDA", hours_ago=1),
                 _Event("b", "NVDA", hours_ago=PEER_TRENDING.window_hours + 1))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert result.rows == (), "a stale view kept a ticker on the list"


def test_an_event_inside_the_window_is_counted():
    feed = _Feed(_Event("a", "NVDA", hours_ago=1),
                 _Event("b", "NVDA", hours_ago=PEER_TRENDING.window_hours - 1))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert result.rows[0].viewers == 2


def test_an_event_with_no_timestamp_is_dropped_rather_than_counted():
    stale = _Event("b", "NVDA")
    stale.at = ""
    result = pt.trending(_Feed(_Event("a", "NVDA"), stale),
                         _sharers("a", "b"), "me", now=NOW)
    assert result.rows == ()


def test_the_window_is_reported_so_now_is_not_implied():
    """'Right now' is not available — events carry a timestamp, not a
    session."""
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert result.window_hours == PEER_TRENDING.window_hours
    assert f"{PEER_TRENDING.window_hours} hours" in pt.caption(result)


# --- ordering -----------------------------------------------------------------

def test_the_most_viewed_ticker_ranks_first():
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"), _Event("c", "NVDA"),
                 _Event("a", "MSFT"), _Event("b", "MSFT"))
    result = pt.trending(feed, _sharers("a", "b", "c"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["NVDA", "MSFT"]
    assert [t.viewers for t in result.rows] == [3, 2]


def test_an_equal_count_breaks_on_recency_then_alphabetically():
    feed = _Feed(_Event("a", "AAA", hours_ago=5), _Event("b", "AAA", hours_ago=5),
                 _Event("a", "ZZZ", hours_ago=1), _Event("b", "ZZZ", hours_ago=1))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["ZZZ", "AAA"], "newer first"


def test_a_full_tie_breaks_alphabetically_so_two_readers_agree():
    feed = _Feed(_Event("a", "ZZZ"), _Event("b", "ZZZ"),
                 _Event("a", "AAA"), _Event("b", "AAA"),
                 _Event("a", "MMM"), _Event("b", "MMM"))
    first = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    second = pt.trending(_Feed(*reversed(feed.events)), _sharers("a", "b"),
                         "me", now=NOW)
    assert [t.ticker for t in first.rows] == [t.ticker for t in second.rows]
    assert [t.ticker for t in first.rows] == ["AAA", "MMM", "ZZZ"]


def test_the_list_is_capped():
    events = []
    for i in range(20):
        events += [_Event("a", f"T{i:02d}"), _Event("b", f"T{i:02d}")]
    result = pt.trending(_Feed(*events), _sharers("a", "b"), "me", now=NOW,
                         max_rows=4)
    assert len(result.rows) == 4


# --- no identities ------------------------------------------------------------

def test_a_row_cannot_carry_an_identity():
    """A wider audience than the feed must carry strictly less than it."""
    import dataclasses
    fields = [f.name for f in dataclasses.fields(pt.Trend)]
    assert fields == ["ticker", "viewers", "latest_at"]
    for banned in ("name", "user_key", "actor", "profile"):
        assert banned not in fields


def test_no_name_reaches_the_rendered_row():
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    profiles = _Profiles(_Profile("a", name="Ana Ruiz"),
                         _Profile("b", name="Ben Osei"))
    result = pt.trending(feed, profiles, "me", now=NOW)
    rendered = " ".join(t.sentence() for t in result.rows) + pt.caption(result)
    assert "Ana" not in rendered and "Ben" not in rendered


def test_the_sentence_agrees_in_number():
    assert pt.Trend("X", 1, "").sentence().endswith("1 account")
    assert pt.Trend("X", 2, "").sentence().endswith("2 accounts")


def test_the_caption_says_your_own_views_are_excluded():
    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    text = pt.caption(pt.trending(feed, _sharers("a", "b"), "me", now=NOW))
    assert "Your own views are not counted" in text
    assert "no row names anyone" in text


def test_there_is_no_caption_when_there_is_nothing_to_caption():
    assert pt.caption(pt.Trends(reason="x")) == ""


# --- robustness ---------------------------------------------------------------

def test_a_malformed_event_is_skipped_rather_than_taking_the_panel_down():
    """This renders inside the portfolio tab. following.load_feed already
    drops malformed records, so nothing should arrive broken — but one
    bad row must not take the tab down with it."""
    class _Broken:
        stream = pt.STREAM          # gets past the cheapest filter

    feed = _Feed(_Event("a", "NVDA"), _Event("b", "NVDA"))
    feed.events = feed.events + (_Broken(),)
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["NVDA"]
    assert result.rows[0].viewers == 2, "the broken row was counted as a viewer"


def test_a_blank_ticker_is_dropped():
    feed = _Feed(_Event("a", "  "), _Event("b", "  "),
                 _Event("a", "NVDA"), _Event("b", "NVDA"))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["NVDA"]


def test_a_lowercase_ticker_is_normalised_and_merged():
    feed = _Feed(_Event("a", "nvda"), _Event("b", "NVDA"))
    result = pt.trending(feed, _sharers("a", "b"), "me", now=NOW)
    assert [t.ticker for t in result.rows] == ["NVDA"]
    assert result.rows[0].viewers == 2, "case split one ticker into two"


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- What others are looking at ---")
    end = src.index("# --- Streak ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_counts_through_the_module():
    panel = _panel()
    assert "pt_trending(" in panel
    assert "sorted(" not in panel, "the ranking belongs in the module"


def test_the_panel_excludes_the_reader():
    """A caller passing no viewer key would count the reader's own
    browsing into the social signal."""
    import re
    call = re.search(r"pt_trending\((.*?)\)", _panel(), re.S)
    assert call, "no pt_trending call found"
    assert "_peer_key" in call.group(1)


def test_the_panel_shows_the_reason_when_there_is_nothing():
    assert "_pt_result.reason" in _panel()


def test_the_panel_points_at_the_market_panel_when_empty():
    assert "PT_MARKET_TRENDING_INSTEAD" in _panel()


def test_the_panel_can_open_a_trending_ticker():
    panel = _panel()
    assert "_pending_ticker" in panel
    assert "pt_open_" in panel


def test_the_panel_carries_the_caption():
    assert "pt_caption(" in _panel()
