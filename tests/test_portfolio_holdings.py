"""Tests for portfolio_holdings.py.

This module reports how someone's money performed, so the tests that
matter most are the ones where a plausible-looking implementation gives a
confidently wrong number:

  - A DEPOSIT MUST NOT LOOK LIKE A GAIN. Naive "value at end vs value at
    start" reports a portfolio that merely received money as having
    returned enormously. That is the single most misleading thing this
    module could do, and it is why time-weighted return exists here.
  - A HOLDING MUST NOT PREDATE ITS PURCHASE. Without per-holding start
    dates the chart back-projects today's basket onto a past that didn't
    contain it, flattering the result with hindsight.
  - AN UNPRICEABLE TICKER MUST BE NAMED, not silently dropped — a
    missing row reads as "it didn't move".

No network: prices are injected.
"""
import datetime
import json

import pandas as pd
import pytest

from config import PORTFOLIO
from portfolio_holdings import (
    Holding,
    PortfolioStore,
    add_holding,
    build_performance,
    build_value_series,
    load_store,
    money_weighted_return,
    rebase_benchmark,
    remove_holding,
    save_store,
    time_weighted_return,
)

IDX = pd.bdate_range("2026-01-01", periods=10)
D0 = IDX[0].date()
D5 = IDX[5].date()


def flat(value=1.0, n=10):
    return pd.Series([value] * n, index=IDX)


def loader(table):
    def load(ticker, start, end):
        if ticker not in table:
            raise KeyError(ticker)
        return table[ticker]
    return load


# --- time-weighted return -----------------------------------------------------

def test_a_single_holding_returns_its_price_change():
    prices = {"A": pd.Series([1.0] * 5 + [1.1] * 5, index=IDX)}
    value, flows = build_value_series((Holding("A", 100, 1.0, D0),), prices)
    assert time_weighted_return(value, flows) == pytest.approx(10.0)


def test_a_deposit_is_not_a_gain():
    """THE PROPERTY THIS WHOLE MODULE TURNS ON.

    A is held from day 0; B is bought on day 5. Prices never move. The
    portfolio's value doubles from 100 to 200 purely because money
    arrived. A naive end-vs-start calculation reports +100%. The correct
    answer is 0% — nothing appreciated.
    """
    prices = {"A": flat(), "B": flat()}
    holdings = (Holding("A", 100, 1.0, D0), Holding("B", 100, 1.0, D5))
    value, flows = build_value_series(holdings, prices)

    assert value.iloc[0] == pytest.approx(100.0)
    assert value.iloc[-1] == pytest.approx(200.0)          # value doubled...
    assert time_weighted_return(value, flows) == pytest.approx(0.0)  # ...return did not


def test_repeated_deposits_still_yield_zero_when_nothing_moves():
    prices = {t: flat() for t in "ABCD"}
    holdings = tuple(Holding(t, 100, 1.0, IDX[i * 2].date()) for i, t in enumerate("ABCD"))
    value, flows = build_value_series(holdings, prices)
    assert time_weighted_return(value, flows) == pytest.approx(0.0)


def test_appreciation_after_a_staggered_purchase_is_measured_correctly():
    prices = {"A": flat(), "B": pd.Series([1.0] * 6 + [2.0] * 4, index=IDX)}
    holdings = (Holding("A", 100, 1.0, D0), Holding("B", 100, 1.0, D5))
    value, flows = build_value_series(holdings, prices)
    # 200 -> 300 once the purchase has settled: +50%.
    assert time_weighted_return(value, flows) == pytest.approx(50.0)


def test_position_size_changes_the_return():
    """Sanity check that weighting is real: a larger stake in the winner
    must move the number."""
    prices = {"A": flat(), "B": pd.Series([1.0] * 6 + [2.0] * 4, index=IDX)}
    small = build_value_series((Holding("A", 100, 1.0, D0), Holding("B", 100, 1.0, D5)), prices)
    large = build_value_series((Holding("A", 100, 1.0, D0), Holding("B", 500, 1.0, D5)), prices)
    assert time_weighted_return(*large) > time_weighted_return(*small)


def test_a_fall_is_negative():
    prices = {"A": pd.Series([1.0] * 5 + [0.8] * 5, index=IDX)}
    value, flows = build_value_series((Holding("A", 100, 1.0, D0),), prices)
    assert time_weighted_return(value, flows) == pytest.approx(-20.0)


def test_no_return_from_an_empty_or_single_point_series():
    assert time_weighted_return(pd.Series(dtype=float), pd.Series(dtype=float)) is None
    single = pd.Series([1.0], index=IDX[:1])
    assert time_weighted_return(single, single) is None


def test_days_before_anything_was_owned_do_not_produce_infinite_returns():
    """V(t-1) is zero before the first purchase. Dividing by it would be
    inf or NaN; those days must simply contribute nothing."""
    prices = {"A": flat()}
    value, flows = build_value_series((Holding("A", 100, 1.0, D5),), prices)
    assert value.iloc[0] == 0.0
    result = time_weighted_return(value, flows)
    assert result is not None and result == pytest.approx(0.0)


# --- the value series ---------------------------------------------------------

def test_a_holding_contributes_nothing_before_its_purchase_date():
    """Without this, the chart back-projects today's basket onto a past
    that didn't contain it — the hindsight distortion that makes a
    snapshot-only model dishonest."""
    prices = {"A": flat()}
    value, _ = build_value_series((Holding("A", 100, 1.0, D5),), prices)
    assert (value.iloc[:5] == 0).all()
    assert value.iloc[5] == pytest.approx(100.0)


def test_a_purchase_date_on_a_non_trading_day_still_counts():
    """A date landing on a weekend must attach to the next trading day,
    not vanish."""
    saturday = datetime.date(2026, 1, 3)
    assert saturday.weekday() == 5
    prices = {"A": flat()}
    value, flows = build_value_series((Holding("A", 100, 1.0, saturday),), prices)
    assert flows.sum() == pytest.approx(100.0)
    assert value.iloc[-1] == pytest.approx(100.0)


def test_the_cash_flow_records_cost_not_market_value():
    """Cost basis is what was actually paid. Recording market value
    instead would silently erase any gain made before the position was
    entered into the app."""
    prices = {"A": pd.Series([5.0] * 10, index=IDX)}
    _, flows = build_value_series((Holding("A", 10, 2.0, D0),), prices)
    assert flows.sum() == pytest.approx(20.0)   # 10 shares at cost 2.00


def test_no_holdings_gives_empty_series():
    value, flows = build_value_series((), {})
    assert value.empty and flows.empty


# --- money-weighted return ----------------------------------------------------

def test_irr_of_a_simple_one_year_double_digit_gain():
    holdings = (Holding("A", 100, 1.0, datetime.date(2025, 1, 1)),)
    result = money_weighted_return(holdings, 110.0, datetime.date(2026, 1, 1))
    assert result == pytest.approx(10.0, abs=0.05)


def test_irr_annualises_over_multiple_years():
    """100 -> 121 over two years is 10% a year, not 21%."""
    holdings = (Holding("A", 100, 1.0, datetime.date(2024, 1, 1)),)
    result = money_weighted_return(holdings, 121.0, datetime.date(2026, 1, 1))
    assert result == pytest.approx(10.0, abs=0.05)


def test_irr_is_negative_on_a_loss():
    holdings = (Holding("A", 100, 1.0, datetime.date(2025, 1, 1)),)
    assert money_weighted_return(holdings, 90.0, datetime.date(2026, 1, 1)) < 0


def test_irr_is_undefined_without_a_final_value():
    holdings = (Holding("A", 100, 1.0, datetime.date(2025, 1, 1)),)
    assert money_weighted_return(holdings, 0.0, datetime.date(2026, 1, 1)) is None


def test_irr_is_undefined_when_everything_happens_on_one_day():
    """No elapsed time means no rate to solve for. Returning a number
    here would be inventing one."""
    same_day = datetime.date(2026, 1, 1)
    holdings = (Holding("A", 100, 1.0, same_day),)
    assert money_weighted_return(holdings, 110.0, same_day) is None


def test_irr_and_twr_diverge_when_timing_matters():
    """The reason both are reported. Buying more just before a rise
    flatters IRR while leaving TWR unchanged — reporting only IRR against
    an index would credit timing as stock-picking skill."""
    prices = {"A": pd.Series([1.0] * 6 + [2.0] * 4, index=IDX)}
    early = (Holding("A", 100, 1.0, D0),)
    topped_up = (Holding("A", 100, 1.0, D0), Holding("A", 900, 1.0, D5))

    twr_early = time_weighted_return(*build_value_series(early, prices))
    twr_topped = time_weighted_return(*build_value_series(topped_up, prices))
    assert twr_early == pytest.approx(twr_topped, abs=0.01)  # picks unchanged


# --- benchmark ----------------------------------------------------------------

def test_benchmark_is_rebased_to_the_portfolio_opening_value():
    """Raw index levels and a pound value on one axis say nothing."""
    benchmark = pd.Series([5000.0, 5500.0], index=IDX[:2])
    rebased = rebase_benchmark(benchmark, to_value=1000.0, index=IDX[:2])
    assert rebased.iloc[0] == pytest.approx(1000.0)
    assert rebased.iloc[1] == pytest.approx(1100.0)   # +10% preserved


def test_rebasing_an_empty_benchmark_returns_none():
    assert rebase_benchmark(pd.Series(dtype=float), 1000.0, IDX) is None


# --- assembling the dashboard -------------------------------------------------

def test_performance_reports_value_cost_and_gain():
    prices = {"A": pd.Series([1.0] * 5 + [1.5] * 5, index=IDX), "SPY": flat(100.0)}
    result = build_performance((Holding("A", 100, 1.0, D0),), loader(prices), end=IDX[-1].date())
    assert result.market_value == pytest.approx(150.0)
    assert result.cost_total == pytest.approx(100.0)
    assert result.total_gain == pytest.approx(50.0)
    assert result.holdings[0].gain_pct == pytest.approx(50.0)


def test_excess_return_is_measured_against_the_benchmark():
    prices = {"A": pd.Series([1.0] * 5 + [1.2] * 5, index=IDX),
              "SPY": pd.Series([100.0] * 5 + [110.0] * 5, index=IDX)}
    result = build_performance((Holding("A", 100, 1.0, D0),), loader(prices), end=IDX[-1].date())
    assert result.benchmark_return_pct == pytest.approx(10.0)
    assert result.excess_vs_benchmark_pct == pytest.approx(result.twr_pct - 10.0)


def test_excess_uses_time_weighted_not_money_weighted():
    """Differencing a money-weighted return against an index is
    apples-to-oranges; the property is asserted rather than assumed."""
    prices = {"A": pd.Series([1.0] * 5 + [1.2] * 5, index=IDX),
              "SPY": pd.Series([100.0] * 5 + [110.0] * 5, index=IDX)}
    result = build_performance((Holding("A", 100, 1.0, D0),), loader(prices), end=IDX[-1].date())
    assert result.excess_vs_benchmark_pct == pytest.approx(
        result.twr_pct - result.benchmark_return_pct)


def test_an_unpriceable_holding_is_named_not_dropped():
    """A row missing from the dashboard reads as "it didn't move"."""
    prices = {"A": flat(), "SPY": flat(100.0)}
    holdings = (Holding("A", 100, 1.0, D0), Holding("GHOST", 50, 2.0, D0))
    result = build_performance(holdings, loader(prices), end=IDX[-1].date())

    assert "GHOST" in result.excluded
    ghost = next(h for h in result.holdings if h.ticker == "GHOST")
    assert ghost.ok is False and ghost.unavailable
    assert any("GHOST" in n for n in result.notes)


def test_an_excluded_holding_is_left_out_of_the_totals():
    """Counting a holding whose price is unknown would require inventing
    one. It stays visible but uncounted, and the notes say so."""
    prices = {"A": flat(), "SPY": flat(100.0)}
    holdings = (Holding("A", 100, 1.0, D0), Holding("GHOST", 50, 2.0, D0))
    result = build_performance(holdings, loader(prices), end=IDX[-1].date())
    assert result.cost_total == pytest.approx(100.0)      # not 100 + 100
    assert any("aren't counted" in n for n in result.notes)


def test_a_missing_benchmark_degrades_without_a_comparison():
    prices = {"A": flat()}          # no SPY
    result = build_performance((Holding("A", 100, 1.0, D0),), loader(prices), end=IDX[-1].date())
    assert result.benchmark_return_pct is None
    assert result.excess_vs_benchmark_pct is None
    assert any("benchmark" in n.lower() for n in result.notes)


def test_an_empty_portfolio_says_so_rather_than_showing_zeros():
    result = build_performance((), loader({}), end=IDX[-1].date())
    assert result.twr_pct is None
    assert any("no holdings" in n.lower() for n in result.notes)


def test_a_short_history_is_flagged():
    short = pd.bdate_range("2026-01-01", periods=5)
    prices = {"A": pd.Series([1.0] * 5, index=short), "SPY": pd.Series([100.0] * 5, index=short)}
    result = build_performance((Holding("A", 100, 1.0, short[0].date()),),
                               loader(prices), end=short[-1].date())
    assert any("too short" in n.lower() for n in result.notes)


def test_the_period_starts_at_the_earliest_purchase():
    prices = {"A": flat(), "B": flat(), "SPY": flat(100.0)}
    holdings = (Holding("A", 1, 1.0, D5), Holding("B", 1, 1.0, D0))
    result = build_performance(holdings, loader(prices), end=IDX[-1].date())
    assert result.period_start == D0


# --- editing and persistence --------------------------------------------------

def test_add_and_remove_a_holding():
    store, err = add_holding(PortfolioStore(), "AAPL", 10, 150.0, D0)
    assert err is None and len(store.holdings()) == 1
    assert remove_holding(store, 0).holdings() == ()


def test_the_same_ticker_can_be_held_as_two_lots():
    """Two purchases of one ticker at different prices and dates are
    distinct lots. Removing by ticker would delete the wrong one, which
    is why removal is by position."""
    store, _ = add_holding(PortfolioStore(), "AAPL", 10, 100.0, D0)
    store, _ = add_holding(store, "AAPL", 5, 200.0, D5)
    assert len(store.holdings()) == 2

    store = remove_holding(store, 0)
    remaining = store.holdings()
    assert len(remaining) == 1 and remaining[0].cost_basis == 200.0


def test_removing_an_out_of_range_index_is_a_no_op():
    store, _ = add_holding(PortfolioStore(), "AAPL", 10, 150.0, D0)
    assert len(remove_holding(store, 99).holdings()) == 1


@pytest.mark.parametrize("shares,cost", [(0, 10.0), (-5, 10.0), (10, -1.0)])
def test_impossible_quantities_are_refused(shares, cost):
    _, err = add_holding(PortfolioStore(), "AAPL", shares, cost, D0)
    assert err is not None


def test_a_future_purchase_date_is_refused():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    _, err = add_holding(PortfolioStore(), "AAPL", 10, 150.0, tomorrow)
    assert err is not None


def test_a_ticker_is_required():
    _, err = add_holding(PortfolioStore(), "   ", 10, 150.0, D0)
    assert err is not None


def test_holdings_are_capped():
    store = PortfolioStore()
    for i in range(PORTFOLIO.max_holdings):
        store, err = add_holding(store, f"T{i}", 1, 1.0, D0)
        assert err is None
    _, err = add_holding(store, "ONEMORE", 1, 1.0, D0)
    assert err is not None and "full" in err.lower()


def test_round_trip(tmp_path):
    path = tmp_path / "p.json"
    store, _ = add_holding(PortfolioStore(), "AAPL", 10.5, 150.25, D0)
    save_store(store, path)
    restored = load_store(path).holdings()[0]
    assert restored.ticker == "AAPL"
    assert restored.shares == pytest.approx(10.5)
    assert restored.cost_basis == pytest.approx(150.25)
    assert restored.purchase_date == D0


def test_missing_store_is_empty(tmp_path):
    assert load_store(tmp_path / "nope.json").holdings() == ()


def test_corrupt_store_degrades_to_empty(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json")
    assert load_store(path).holdings() == ()


def test_one_malformed_holding_does_not_discard_the_portfolio(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"active": "My Portfolio", "portfolios": {"My Portfolio": [
        {"ticker": "AAPL", "shares": 10, "cost_basis": 100.0, "purchase_date": "2026-01-01"},
        {"ticker": "BAD", "shares": "not-a-number", "cost_basis": 1, "purchase_date": "2026-01-01"},
        {"ticker": "NODATE", "shares": 1, "cost_basis": 1},
    ]}}))
    holdings = load_store(path).holdings()
    assert [h.ticker for h in holdings] == ["AAPL"]


def test_an_active_name_that_no_longer_exists_falls_back(tmp_path):
    """A keyed selectbox whose stored value falls outside its options
    raises — this app has hit that before."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"active": "Deleted", "portfolios": {"Kept": []}}))
    assert load_store(path).active == "Kept"


def test_save_leaves_no_leftover_temp_file(tmp_path):
    path = tmp_path / "p.json"
    store, _ = add_holding(PortfolioStore(), "AAPL", 1, 1.0, D0)
    save_store(store, path)
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]
