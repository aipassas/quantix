"""Opt-in peer comparison across accounts on one instance.

Three things here are load-bearing, and each is a way the feature could
be quietly wrong rather than visibly broken:

  1. ONLY A PERCENTAGE IS SHARED. Not holdings, not tickers, not market
     value. A rate of return is a very different disclosure from what
     someone owns and how much of it.
  2. THE WINDOW IS IDENTICAL FOR EVERYONE. The portfolio dashboard
     measures from each portfolio's own earliest purchase, so ranking
     those figures would compare a three-year return with a three-month
     one and call the difference skill.
  3. A SMALL COHORT GETS NO PERCENTILE. With one other participant "you
     beat 100%" states their return exactly.
"""
import datetime
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import peer_comparison as pc
from portfolio_holdings import Holding, time_weighted_return

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


@pytest.fixture
def cfg(monkeypatch):
    import dataclasses

    def apply(**overrides):
        monkeypatch.setattr(pc, "PEER_COMPARISON",
                            dataclasses.replace(pc.PEER_COMPARISON, **overrides))
    return apply


def _cohort(size, mine=5.0, base=0.0):
    """A store with `size` others plus me."""
    store, _ = pc.publish(pc.PeerStore(), "me", mine)
    for i in range(size):
        store, _ = pc.publish(store, f"other{i}", base + i)
    return store


# --- the privacy boundary -----------------------------------------------------

def test_only_a_return_is_persisted(tmp_path):
    """THE LOAD-BEARING ASSERTION. Whatever else changes, the file must
    never gain a field describing what someone owns."""
    store, _ = pc.publish(pc.PeerStore(), "abc", 12.5)
    path = tmp_path / "peer.json"
    assert pc.save_store(store, path) is True

    raw = json.loads(path.read_text())
    assert set(raw) == {"entries"}
    for entry in raw["entries"]:
        assert set(entry) == {"user_key", "period", "twr_pct", "updated_at"}


def test_publish_is_never_handed_a_portfolio():
    """The signature IS the boundary: there is no parameter through which
    holdings or a market value could arrive."""
    import inspect
    params = set(inspect.signature(pc.publish).parameters)
    assert params == {"store", "user_key", "twr_pct", "period"}
    for forbidden in ("holdings", "value", "market_value", "cost", "tickers"):
        assert forbidden not in params


def test_the_entry_dataclass_carries_nothing_extra():
    import dataclasses
    fields = {f.name for f in dataclasses.fields(pc.PeerEntry)}
    assert fields == {"user_key", "period", "twr_pct", "updated_at"}


def test_the_store_is_shared_because_a_comparison_cannot_be_per_user():
    source = Path(pc.__file__).read_text()
    assert "shared_path(" in source


def test_the_shared_store_is_declared_in_auth():
    """test_auth fails otherwise, and the declaration is where a reader
    finds out this file crosses an account boundary."""
    import auth
    assert pc.PEER_COMPARISON.store_filename in auth.SHARED_STORES


# --- opting in and out --------------------------------------------------------

def test_nothing_is_shared_until_you_publish():
    store = pc.PeerStore()
    assert pc.is_participating(store, "me") is False
    assert pc.compare(store, "me").has_result is False


def test_publishing_then_comparing_needs_no_second_step():
    store, error = pc.publish(pc.PeerStore(), "me", 4.2)
    assert error == ""
    assert pc.is_participating(store, "me") is True


def test_publishing_twice_replaces_rather_than_duplicates():
    store, _ = pc.publish(pc.PeerStore(), "me", 1.0)
    store, _ = pc.publish(store, "me", 9.0)
    mine = [e for e in store.entries if e.user_key == "me"]
    assert len(mine) == 1 and mine[0].twr_pct == 9.0


def test_withdrawing_removes_every_period_not_just_this_one():
    """Opting out has to mean gone; leaving last month's figure behind
    would make the choice cosmetic."""
    store, _ = pc.publish(pc.PeerStore(), "me", 1.0, period="2026-08")
    store, _ = pc.publish(store, "me", 2.0, period="2026-09")
    store, _ = pc.publish(store, "other", 3.0, period="2026-09")
    after = pc.withdraw(store, "me")
    assert [e.user_key for e in after.entries] == ["other"]


def test_publishing_without_a_return_is_refused_not_stored_as_zero():
    """A portfolio with no monthly return must not enter the ranking at
    0%, which would place it below everyone who had one."""
    store, error = pc.publish(pc.PeerStore(), "me", None)
    assert store.entries == ()
    assert "no return to share" in error


def test_publishing_signed_out_is_refused():
    store, error = pc.publish(pc.PeerStore(), "", 5.0)
    assert store.entries == () and "Sign in" in error


# --- the cohort floor ---------------------------------------------------------

@pytest.mark.parametrize("others", [0, 1, 2, 3, 4])
def test_a_small_cohort_gets_no_percentile(others):
    """With one other participant, "you beat 100%" states their return
    exactly; with two it narrows it to a half."""
    result = pc.compare(_cohort(others), "me")
    assert result.percentile is None
    assert result.has_result is False
    assert str(others) in result.reason


def test_at_the_floor_a_percentile_appears():
    result = pc.compare(_cohort(pc.PEER_COMPARISON.min_cohort), "me")
    assert result.has_result is True
    assert 0.0 <= result.percentile <= 100.0


def test_the_percentile_counts_only_those_you_beat():
    # me = 5.0; others 0,1,2,3,4,5,6,7,8 -> five are strictly below.
    result = pc.compare(_cohort(9, mine=5.0), "me")
    assert result.cohort_size == 9
    assert result.beat_count == 5
    assert result.percentile == pytest.approx(100 * 5 / 9)


def test_you_are_not_counted_in_your_own_cohort():
    result = pc.compare(_cohort(6), "me")
    assert result.cohort_size == 6, "the cohort is the OTHER participants"


def test_an_equal_return_does_not_count_as_beaten():
    store, _ = pc.publish(pc.PeerStore(), "me", 5.0)
    for i in range(6):
        store, _ = pc.publish(store, f"other{i}", 5.0)
    assert pc.compare(store, "me").beat_count == 0


def test_the_floor_is_configurable_for_a_larger_deployment(cfg):
    cfg(min_cohort=2)
    assert pc.compare(_cohort(2), "me").has_result is True


def test_not_sharing_gives_a_reason_not_a_number():
    store = _cohort(9)
    store = pc.withdraw(store, "me")
    result = pc.compare(store, "me")
    assert result.has_result is False
    assert "not sharing" in result.reason


def test_only_the_same_period_is_compared():
    """Entries from another month must not pad the cohort."""
    store, _ = pc.publish(pc.PeerStore(), "me", 5.0, period="2026-09")
    for i in range(9):
        store, _ = pc.publish(store, f"other{i}", float(i), period="2026-08")
    result = pc.compare(store, "me", "2026-09")
    assert result.cohort_size == 0 and result.has_result is False


# --- the identical window -----------------------------------------------------

def test_the_window_return_is_not_the_inception_to_date_return():
    """THE REASON window_return EXISTS. The same portfolio gives very
    different figures over its whole life and over one month; ranking
    the first across people measures who started sooner."""
    index = pd.date_range("2026-07-01", "2026-09-30", freq="D")
    value = pd.Series(range(100, 100 + len(index)), index=index, dtype=float)
    flows = pd.Series(0.0, index=index)

    lifetime = time_weighted_return(value, flows)
    monthly = pc.window_return(value, flows, "2026-09")
    assert lifetime is not None and monthly is not None
    assert monthly < lifetime / 2, (
        f"a month ({monthly:.1f}%) must not be reported as the lifetime "
        f"figure ({lifetime:.1f}%)")


def test_a_month_with_fewer_than_two_valuations_has_no_return():
    """A portfolio opened yesterday has no monthly return, and 0% would
    enter it in the ranking as though it did."""
    index = pd.to_datetime(["2026-09-30"])
    value = pd.Series([100.0], index=index)
    assert pc.window_return(value, pd.Series(0.0, index=index), "2026-09") is None


def test_an_empty_series_has_no_return():
    assert pc.window_return(pd.Series(dtype=float), None, "2026-09") is None


def test_a_malformed_period_yields_no_return_rather_than_raising():
    index = pd.date_range("2026-09-01", "2026-09-10", freq="D")
    value = pd.Series(1.0, index=index)
    assert pc.window_return(value, None, "not-a-period") is None


def test_period_bounds_cover_the_whole_month():
    assert pc.period_bounds("2026-09") == (datetime.date(2026, 9, 1),
                                           datetime.date(2026, 9, 30))
    assert pc.period_bounds("2026-02") == (datetime.date(2026, 2, 1),
                                           datetime.date(2026, 2, 28))
    assert pc.period_bounds("2026-12") == (datetime.date(2026, 12, 1),
                                           datetime.date(2026, 12, 31))


def test_a_leap_february_is_handled():
    assert pc.period_bounds("2024-02")[1] == datetime.date(2024, 2, 29)


def test_the_period_label_is_the_same_for_everyone_on_a_given_day():
    day = datetime.date(2026, 9, 17)
    assert pc.period_for(day) == "2026-09"


# --- flows --------------------------------------------------------------------

def test_a_deposit_inside_the_month_is_not_counted_as_a_gain():
    """Without the flow series a purchase reads as spectacular
    performance, and whoever deposited most would top the ranking.
    Measured: 109% against a true 6.55%."""
    index = pd.date_range("2026-09-01", "2026-09-10", freq="D")
    value = pd.Series([1000, 1010, 1020, 1030, 2040, 2050, 2060, 2070, 2080, 2090],
                      index=index, dtype=float)
    holdings = (Holding("AAPL", 10, 100.0, datetime.date(2026, 9, 5)),)

    flows = pc.flows_from_holdings(holdings, index)
    honest = pc.window_return(value, flows, "2026-09")
    inflated = pc.window_return(value, pd.Series(0.0, index=index), "2026-09")

    assert flows.loc[pd.Timestamp("2026-09-05")] == pytest.approx(1000.0)
    assert honest < 20.0
    assert inflated > 100.0


def test_flows_land_on_the_purchase_date_only():
    index = pd.date_range("2026-09-01", "2026-09-05", freq="D")
    holdings = (Holding("AAPL", 2, 50.0, datetime.date(2026, 9, 3)),)
    flows = pc.flows_from_holdings(holdings, index)
    assert flows.sum() == pytest.approx(100.0)
    assert int((flows != 0).sum()) == 1


def test_a_purchase_outside_the_index_is_dropped_not_misplaced():
    index = pd.date_range("2026-09-01", "2026-09-05", freq="D")
    holdings = (Holding("AAPL", 2, 50.0, datetime.date(2020, 1, 1)),)
    assert pc.flows_from_holdings(holdings, index).sum() == 0.0


def test_no_holdings_gives_a_zero_flow_series():
    index = pd.date_range("2026-09-01", "2026-09-03", freq="D")
    assert pc.flows_from_holdings((), index).sum() == 0.0


# --- persistence --------------------------------------------------------------

def test_a_round_trip_preserves_entries(tmp_path):
    store = _cohort(3)
    path = tmp_path / "peer.json"
    pc.save_store(store, path)
    loaded = pc.load_store(path)
    assert len(loaded.entries) == len(store.entries)
    assert loaded.corrupt is False


def test_a_missing_store_is_empty_not_corrupt(tmp_path):
    loaded = pc.load_store(tmp_path / "nope.json")
    assert loaded.entries == () and loaded.corrupt is False


def test_an_unreadable_store_is_never_overwritten(tmp_path):
    """It holds other accounts' entries, not just yours."""
    path = tmp_path / "peer.json"
    path.write_text("{ not json")
    loaded = pc.load_store(path)
    assert loaded.corrupt is True
    assert pc.save_store(loaded, path) is False
    assert path.read_text() == "{ not json"


def test_a_malformed_entry_is_dropped_not_fatal(tmp_path):
    path = tmp_path / "peer.json"
    path.write_text(json.dumps({"entries": [
        {"user_key": "a", "period": "2026-09", "twr_pct": 1.0},
        {"user_key": "", "period": "2026-09", "twr_pct": 1.0},
        {"user_key": "c", "period": "2026-09", "twr_pct": "not a number"},
        "not a dict",
    ]}))
    assert len(pc.load_store(path).entries) == 1


# --- UI wiring ----------------------------------------------------------------

def test_the_panel_is_wired_into_the_app():
    source = FINANCE.read_text()
    assert "import peer_comparison" in source
    assert "peer_comparison.compare(" in source


def test_the_app_recomputes_the_month_rather_than_reusing_the_headline():
    """Publishing _pf_perf.twr_pct would share an inception-to-date
    figure and silently reintroduce the mismatched-window bug."""
    source = FINANCE.read_text()
    assert "peer_comparison.window_return(" in source
    start = source.index("PEER COMPARISON (opt-in)")
    end = source.index("if _pf_perf.mwr_pct is not None:")
    # Comments stripped first: the block's own explanation NAMES the
    # figure it is avoiding, so a plain substring check matches the
    # warning rather than the code. This project has been bitten by
    # exactly that in a CSS test.
    block = "\n".join(line for line in source[start:end].splitlines()
                      if not line.strip().startswith("#"))
    assert "publish(" in block
    assert "_pf_perf.twr_pct" not in block, (
        "the published figure must be the month-scoped one")


def test_the_app_passes_flows_so_a_deposit_is_not_a_gain():
    source = FINANCE.read_text()
    assert "peer_comparison.flows_from_holdings(" in source


def test_sharing_is_opt_in_behind_a_button():
    source = FINANCE.read_text()
    start = source.index("PEER COMPARISON (opt-in)")
    end = source.index("if _pf_perf.mwr_pct is not None:")
    block = source[start:end]
    publish_at = block.index("peer_comparison.publish(")
    button_at = block.index('key="peer_publish"')
    assert button_at < publish_at, "nothing may be published without the click"


def test_withdrawing_is_offered_wherever_sharing_is():
    source = FINANCE.read_text()
    assert 'key="peer_withdraw"' in source
