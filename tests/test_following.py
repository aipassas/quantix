"""Profiles, following, and the colleagues feed.

The privacy model is one rule and the tests are arranged around it: a
follower's session never opens another account's files. The sharing
account publishes from its own session, only for streams it switched
on; the reader filters what was published. So the tests check that

  - nothing is published without a profile, or with a stream off,
  - switching a stream off is retroactive and deleting a profile purges,
  - the reader re-checks the publisher's CURRENT switches,
  - and the module never reads a per-user store on the reading side.

There are no experts. A profile has a self-described title.
"""
import datetime
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import following as f

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def _ana(streams=(f.STREAM_JOURNAL,)):
    store, err = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana",
                                  title="Energy analyst", streams=streams)
    assert err == ""
    return store


# --- nothing shared by default ------------------------------------------------

def test_no_profile_means_nothing_is_published():
    feed = f.publish(f.FeedStore(), f.ProfilesStore(), "k-ana",
                     f.STREAM_JOURNAL, "AAPL", "Buy")
    assert feed.events == ()


def test_a_new_profile_shares_nothing():
    store, _ = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana")
    assert store.get("k-ana").streams == ()
    feed = f.publish(f.FeedStore(), store, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    assert feed.events == ()


def test_a_stream_that_is_off_publishes_nothing_even_when_others_are_on():
    profiles = _ana(streams=(f.STREAM_JOURNAL,))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_WATCHLIST, "MSFT", "added")
    assert feed.events == ()
    feed = f.publish(feed, profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    assert [e.stream for e in feed.events] == [f.STREAM_JOURNAL]


def test_publish_checks_the_switch_itself_not_the_caller():
    """A hook that forgets to check must still share nothing."""
    profiles, _ = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana", streams=())
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked")
    assert feed.events == ()


def test_an_unknown_stream_is_never_enabled():
    store, _ = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana",
                                streams=("journal", "portfolio_value", "everything"))
    assert store.get("k-ana").streams == ("journal",)


# --- the privacy boundary -----------------------------------------------------

def test_publish_takes_a_summary_never_a_store():
    """The signature IS the boundary."""
    import inspect
    params = set(inspect.signature(f.publish).parameters)
    assert params == {"feed", "profiles", "actor_key", "stream", "ticker", "summary", "now"}
    for forbidden in ("holdings", "entry", "rule", "watchlist", "portfolio", "store"):
        assert forbidden not in params


def test_the_reading_side_never_touches_a_per_user_store():
    """feed_for takes the caller's OWN follow list and the shared stores.
    There is no code path from the reader to another account's files."""
    import inspect
    params = set(inspect.signature(f.feed_for).parameters)
    assert params == {"follows", "feed", "profiles", "limit"}
    source = Path(f.__file__).read_text()
    body = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    # store_path (per-user) is used for exactly one thing: whom I follow.
    assert body.count("store_path(") == 1
    assert "store_path(FOLLOWING.follows_filename)" in body


def test_a_feed_event_carries_only_the_published_line(tmp_path):
    profiles = _ana()
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy · High")
    path = tmp_path / "feed.json"
    f.save_feed(feed, path)
    raw = json.loads(path.read_text())
    assert set(raw["events"][0]) == {"id", "actor_key", "stream", "ticker", "summary", "at"}


def test_follows_are_per_user_and_profiles_are_shared():
    import auth
    assert f.FOLLOWING.follows_filename in auth.PER_USER_STORES
    assert f.FOLLOWING.profiles_filename in auth.SHARED_STORES
    assert f.FOLLOWING.feed_filename in auth.SHARED_STORES


# --- opting out is retroactive ------------------------------------------------

def test_switching_a_stream_off_purges_its_history():
    profiles = _ana(streams=(f.STREAM_JOURNAL, f.STREAM_VIEWED))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    feed = f.publish(feed, profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked")
    before = profiles.get("k-ana")
    profiles, _ = f.upsert_profile(profiles, "k-ana", "Ana", "Energy analyst",
                                   streams=(f.STREAM_VIEWED,))
    feed = f.apply_stream_change(feed, before, profiles.get("k-ana"))
    assert [e.stream for e in feed.events] == [f.STREAM_VIEWED]


def test_the_reader_hides_a_switched_off_stream_before_any_purge():
    """The publisher's choice takes effect on the next READ, not the next
    save — even if the cleanup never runs."""
    profiles = _ana(streams=(f.STREAM_JOURNAL,))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    profiles, _ = f.upsert_profile(profiles, "k-ana", "Ana", streams=())   # off, no purge
    me = f.follow(f.FollowStore(), "k-ana")
    assert f.feed_for(me, feed, profiles) == ()


def test_deleting_a_profile_purges_everything_and_ends_following():
    profiles = _ana(streams=(f.STREAM_JOURNAL, f.STREAM_VIEWED))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    feed = f.publish(feed, profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked")
    profiles = f.delete_profile(profiles, "k-ana")
    feed = f.purge(feed, "k-ana")
    assert feed.events == ()
    me = f.follow(f.FollowStore(), "k-ana")
    assert f.feed_for(me, feed, profiles) == ()
    assert f.followable(profiles) == ()


def test_purge_leaves_other_accounts_alone():
    profiles = _ana()
    profiles, _ = f.upsert_profile(profiles, "k-ben", "Ben", streams=(f.STREAM_JOURNAL,))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "a")
    feed = f.publish(feed, profiles, "k-ben", f.STREAM_JOURNAL, "AAPL", "b")
    assert [e.actor_key for e in f.purge(feed, "k-ana").events] == ["k-ben"]


# --- following ----------------------------------------------------------------

def test_only_followed_accounts_appear_in_the_feed():
    profiles = _ana()
    profiles, _ = f.upsert_profile(profiles, "k-ben", "Ben", streams=(f.STREAM_JOURNAL,))
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "a")
    feed = f.publish(feed, profiles, "k-ben", f.STREAM_JOURNAL, "AAPL", "b")
    me = f.follow(f.FollowStore(), "k-ana")
    assert [i.actor.name for i in f.feed_for(me, feed, profiles)] == ["Ana"]


def test_following_nobody_shows_nothing():
    profiles = _ana()
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "a")
    assert f.feed_for(f.FollowStore(), feed, profiles) == ()


def test_the_feed_is_newest_first_and_bounded():
    profiles = _ana(streams=(f.STREAM_JOURNAL,))
    feed = f.FeedStore()
    for i in range(5):
        feed = f.publish(feed, profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", f"n{i}",
                         now=datetime.datetime(2026, 9, 11, 10, i))
    me = f.follow(f.FollowStore(), "k-ana")
    items = f.feed_for(me, feed, profiles, limit=3)
    assert [i.event.summary for i in items] == ["n4", "n3", "n2"]


def test_follow_and_unfollow_are_idempotent():
    s = f.follow(f.FollowStore(), "k-ana")
    s = f.follow(s, "k-ana")
    assert s.following == ("k-ana",)
    s = f.unfollow(s, "k-ana")
    s = f.unfollow(s, "k-ana")
    assert s.following == ()


def test_you_are_not_in_your_own_followable_list():
    profiles = _ana()
    assert f.followable(profiles, viewer_key="k-ana") == ()
    assert [p.name for p in f.followable(profiles, viewer_key="k-ben")] == ["Ana"]


def test_an_account_without_a_profile_is_not_followable():
    """Becoming visible is an act, not a default."""
    assert f.followable(f.ProfilesStore(), "k-anyone") == ()


# --- no experts ---------------------------------------------------------------

def test_a_title_is_self_described_and_optional():
    store, _ = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana")
    assert store.get("k-ana").title == ""
    store, _ = f.upsert_profile(store, "k-ana", "Ana", title="  Energy analyst  ")
    assert store.get("k-ana").title == "Energy analyst"


def test_a_title_has_a_length_cap():
    _, err = f.upsert_profile(f.ProfilesStore(), "k-ana", "Ana", title="x" * 500)
    assert "under" in err


def test_the_module_never_calls_anyone_an_expert():
    """The word implies a grantor that does not exist here."""
    source = Path(f.__file__).read_text()
    body = re.sub(r'""".*?"""', "", source, flags=re.DOTALL)
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    assert "expert" not in body.lower()


def test_the_feed_line_shows_the_title_as_part_of_the_name():
    profiles = _ana()
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "Buy")
    item = f.feed_for(f.follow(f.FollowStore(), "k-ana"), feed, profiles)[0]
    assert "Ana (Energy analyst)" in item.line
    assert "journal entry" in item.line and "AAPL" in item.line


# --- dedupe and bounds --------------------------------------------------------

def test_repeat_views_inside_the_window_collapse_to_one():
    profiles = _ana(streams=(f.STREAM_VIEWED,))
    t = datetime.datetime(2026, 9, 11, 10, 0)
    feed = f.FeedStore()
    for hours in (0, 3, 20):
        feed = f.publish(feed, profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked",
                         now=t + datetime.timedelta(hours=hours))
    assert len(feed.events) == 1


def test_a_view_after_the_window_is_a_new_event():
    profiles = _ana(streams=(f.STREAM_VIEWED,))
    t = datetime.datetime(2026, 9, 11, 10, 0)
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked", now=t)
    feed = f.publish(feed, profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked",
                     now=t + datetime.timedelta(hours=30))
    assert len(feed.events) == 2


def test_different_tickers_do_not_dedupe_against_each_other():
    profiles = _ana(streams=(f.STREAM_VIEWED,))
    t = datetime.datetime(2026, 9, 11, 10, 0)
    feed = f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_VIEWED, "NVDA", "looked", now=t)
    feed = f.publish(feed, profiles, "k-ana", f.STREAM_VIEWED, "AAPL", "looked", now=t)
    assert len(feed.events) == 2


def test_the_feed_store_is_bounded(monkeypatch):
    import dataclasses
    monkeypatch.setattr(f, "FOLLOWING", dataclasses.replace(f.FOLLOWING, max_events=3))
    profiles = _ana(streams=(f.STREAM_JOURNAL,))
    feed = f.FeedStore()
    for i in range(10):
        feed = f.publish(feed, profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", f"n{i}")
    assert len(feed.events) == 3
    assert feed.events[-1].summary == "n9"


def test_an_empty_summary_publishes_nothing():
    profiles = _ana()
    assert f.publish(f.FeedStore(), profiles, "k-ana", f.STREAM_JOURNAL, "AAPL", "  ").events == ()


# --- persistence --------------------------------------------------------------

def test_profiles_round_trip(tmp_path):
    store = _ana(streams=(f.STREAM_JOURNAL, f.STREAM_ALERT))
    path = tmp_path / "p.json"
    f.save_profiles(store, path)
    loaded = f.load_profiles(path).get("k-ana")
    assert loaded.name == "Ana" and loaded.title == "Energy analyst"
    assert set(loaded.streams) == {f.STREAM_JOURNAL, f.STREAM_ALERT}


def test_a_corrupt_shared_file_is_never_overwritten(tmp_path):
    for loader, saver, empty in ((f.load_profiles, f.save_profiles, f.ProfilesStore()),
                                 (f.load_feed, f.save_feed, f.FeedStore())):
        path = tmp_path / "x.json"
        path.write_text("{ nope")
        loaded = loader(path)
        assert loaded.corrupt is True
        assert saver(loaded, path) is False
        assert path.read_text() == "{ nope"


def test_follows_round_trip(tmp_path):
    path = tmp_path / "f.json"
    f.save_follows(f.follow(f.follow(f.FollowStore(), "a"), "b"), path)
    assert f.load_follows(path).following == ("a", "b")


def test_a_malformed_event_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "feed.json"
    path.write_text(json.dumps({"events": [
        {"actor_key": "a", "stream": "journal", "ticker": "AAPL", "summary": "ok"},
        {"actor_key": "", "stream": "journal"},
        {"actor_key": "b", "stream": "not-a-stream"},
        "junk",
    ]}))
    assert len(f.load_feed(path).events) == 1


# --- UI wiring ----------------------------------------------------------------

SRC = FINANCE.read_text()


def test_every_stream_has_a_publish_hook():
    """A stream nobody can publish to is a switch that does nothing."""
    for stream in ("STREAM_WATCHLIST", "STREAM_ALERT", "STREAM_JOURNAL", "STREAM_VIEWED"):
        assert f"_fl_publish(following.{stream}" in SRC, stream


def test_the_publish_helper_cannot_raise_into_its_caller():
    start = SRC.index("def _fl_publish(")
    end = SRC.index("\n\n", start + 40)
    body = SRC[start:end]
    assert "try:" in body and "except Exception:" in body


def test_the_helper_checks_the_stream_switch_before_loading_the_feed():
    """The common case is 'not shared'; it must cost a profile read, not
    a feed read and write."""
    start = SRC.index("def _fl_publish(")
    end = SRC.index("\n\n", start + 40)
    body = SRC[start:end]
    assert body.index(".shares(stream)") < body.index("following.load_feed()")


def test_only_new_watchlist_tickers_are_published():
    assert "_fl_added not in _wl_store_before_tickers" in SRC


def test_the_profile_panel_seeds_widgets_once_then_uses_keys_only():
    """value= alongside key= would revert the user's edit (CLAUDE.md)."""
    start = SRC.index('with st.sidebar.expander("Profile & following"')
    end = SRC.index("# --- Sidebar", start + 10)
    panel = SRC[start:end]
    assert '"fl_seeded_for"' in panel
    for line in panel.splitlines():
        if 'key="fl_title"' in line or 'key=f"fl_stream_' in line:
            assert "value=" not in line, line


def test_switching_off_in_the_panel_purges_and_deleting_purges_everything():
    start = SRC.index('with st.sidebar.expander("Profile & following"')
    end = SRC.index("# --- Sidebar", start + 10)
    panel = SRC[start:end]
    assert "following.apply_stream_change(" in panel
    assert "following.purge(_fl_feed, _auth_user.key)" in panel


def test_the_feed_is_rendered_and_read_only():
    start = SRC.index("COLLEAGUES FEED (following)")
    end = SRC.index('with st.expander(f"Team Notes', start)
    block = SRC[start:end]
    assert "following.feed_for(" in block
    assert "following.publish(" not in block
    assert "save_" not in block


def test_the_ui_never_calls_anyone_an_expert():
    for name in ('with st.sidebar.expander("Profile & following"', "COLLEAGUES FEED (following)"):
        start = SRC.index(name)
        block = SRC[start:start + 6000]
        visible = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))
        # the one permitted mention is the help text saying there is NO such badge
        assert visible.lower().count("expert") <= 1
