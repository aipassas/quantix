"""Tests for portfolio_backtester.py — weight normalization, rebalance
period-boundary detection, and the weighted/rebalanced portfolio
combination engine.

strategy_builder.run_backtest() is monkeypatched to return hand-crafted,
exact per-ticker daily returns rather than running real indicator math, so
the weight-drift/rebalance/contribution arithmetic can be verified against
numbers worked out by hand — not just "looks plausible" on real data.
"""
import pandas as pd
import pytest

import portfolio_backtester as pb
from portfolio_backtester import (
    REBALANCE_FREQUENCIES,
    _period_key,
    normalize_weights,
    run_portfolio_backtest,
)
from strategy_builder import BacktestResult


def _fake_backtest_result(returns: pd.Series) -> BacktestResult:
    """A BacktestResult shaped just enough for portfolio_backtester to
    consume — only df['Net_Strategy_Returns'] is actually read by it."""
    return BacktestResult(
        df=pd.DataFrame({"Net_Strategy_Returns": returns}, index=returns.index),
        entry_series=pd.Series(dtype=bool), exit_series=pd.Series(dtype=bool),
        total_strategy_return_pct=0.0, total_buy_hold_return_pct=0.0, max_drawdown_pct=0.0,
        win_rate_pct=None, trade_count=0, cost_bps=0.0,
        total_net_strategy_return_pct=0.0, net_max_drawdown_pct=0.0, total_cost_pct=0.0,
    )


def _dummy_input_df(dates, ticker: str) -> pd.DataFrame:
    """A stand-in for an indicator-prepared ticker DataFrame — content is
    irrelevant since run_backtest itself is monkeypatched; only needs to be
    non-empty (an all-index, no-column frame is `.empty` in pandas) and
    carries the ticker name so the monkeypatched run_backtest can look up
    its canned result."""
    df = pd.DataFrame({"Close": [1.0] * len(dates)}, index=dates)
    df.attrs["ticker"] = ticker
    return df


def _patch_canned_results(monkeypatch, canned: dict):
    monkeypatch.setattr(
        pb, "run_backtest",
        lambda df, rule, sma, rsi, cost_bps=0.0: canned[df.attrs["ticker"]],
    )


# --- normalize_weights -------------------------------------------------------

def test_normalize_weights_rescales_to_sum_one():
    result, err = normalize_weights({"A": 2.0, "B": 2.0})
    assert err is None
    assert result == {"A": 0.5, "B": 0.5}


def test_normalize_weights_rejects_negative():
    result, err = normalize_weights({"A": -0.5, "B": 1.5})
    assert result is None
    assert "long-only" in err.lower() or "negative" in err.lower()


def test_normalize_weights_rejects_zero_sum():
    result, err = normalize_weights({"A": 0.0, "B": 0.0})
    assert result is None
    assert "positive" in err.lower()


def test_normalize_weights_single_ticker():
    result, err = normalize_weights({"A": 7.0})
    assert err is None
    assert result == {"A": 1.0}


# --- _period_key (rebalance boundary detection) -------------------------------

def test_period_key_monthly_changes_across_month_boundary():
    jan31 = pd.Timestamp("2024-01-31")
    feb1 = pd.Timestamp("2024-02-01")
    assert _period_key(jan31, "monthly") != _period_key(feb1, "monthly")


def test_period_key_monthly_same_within_month():
    jan5 = pd.Timestamp("2024-01-05")
    jan31 = pd.Timestamp("2024-01-31")
    assert _period_key(jan5, "monthly") == _period_key(jan31, "monthly")


def test_period_key_quarterly_groups_three_months():
    jan = pd.Timestamp("2024-01-15")
    mar = pd.Timestamp("2024-03-15")
    apr = pd.Timestamp("2024-04-15")
    assert _period_key(jan, "quarterly") == _period_key(mar, "quarterly")
    assert _period_key(mar, "quarterly") != _period_key(apr, "quarterly")


def test_period_key_annually_groups_whole_year():
    assert _period_key(pd.Timestamp("2024-01-01"), "annually") == _period_key(pd.Timestamp("2024-12-31"), "annually")
    assert _period_key(pd.Timestamp("2024-12-31"), "annually") != _period_key(pd.Timestamp("2025-01-01"), "annually")


def test_period_key_rejects_unknown_frequency():
    with pytest.raises(ValueError):
        _period_key(pd.Timestamp("2024-01-01"), "weekly")


def test_all_declared_frequencies_except_none_are_handled():
    for freq in REBALANCE_FREQUENCIES:
        if freq == "none":
            continue
        _period_key(pd.Timestamp("2024-01-01"), freq)  # must not raise


# --- run_portfolio_backtest: weight drift, no rebalancing --------------------

def test_no_rebalance_weight_drifts_and_compounds_correctly(monkeypatch):
    """A jumps 10% on day 1 then sits flat; B is flat throughout. Hand-
    derived: day-1 contribution = 0.5*0.10 = 5%, total compounded return =
    5% (nothing moves after day 1), and A's post-day-1 weight should drift
    to 0.55/1.05 = 0.52381 (up from its 0.5 target) since it's now worth
    more of the portfolio."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    a_returns = pd.Series([0.10, 0.0, 0.0, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.5, "B": 0.5}, sma_length=20, rsi_length=14,
        rebalance_frequency="none",
    )

    assert err is None
    assert result.rebalance_dates == ()
    assert result.total_return_pct == pytest.approx(5.0, abs=1e-9)
    assert result.contribution_pct["A"] == pytest.approx(5.0, abs=1e-9)
    assert result.contribution_pct["B"] == pytest.approx(0.0, abs=1e-9)
    assert result.df.iloc[0]["Weight_A"] == pytest.approx(0.5)  # day 1 held at target — nothing to drift from yet
    assert result.df.iloc[1]["Weight_A"] == pytest.approx(0.55 / 1.05)


def test_contribution_always_sums_to_daily_return_total(monkeypatch):
    """By construction (Portfolio_Return[t] = sum_i weight_i[t] * return_i[t]),
    contributions must sum EXACTLY to the sum of daily portfolio returns —
    the module docstring's claim, checked directly rather than assumed."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    a_returns = pd.Series([0.03, -0.02, 0.01, 0.00, 0.02], index=dates)
    b_returns = pd.Series([-0.01, 0.04, 0.00, -0.03, 0.01], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.6, "B": 0.4}, sma_length=20, rsi_length=14,
        rebalance_frequency="monthly",
    )

    assert err is None
    assert sum(result.contribution_pct.values()) == pytest.approx(result.df["Portfolio_Return"].sum() * 100)


# --- run_portfolio_backtest: periodic rebalancing -----------------------------

def test_monthly_rebalance_triggers_on_first_trading_day_of_new_month(monkeypatch):
    """A jumps 10% on day 1 (Jan 30), both flat on Jan 31, B jumps 20% on
    Feb 1 — the first day of a new month, so the reset to 50/50 must apply
    BEFORE that day's return is earned (the standard convention). Hand-
    derived: day-1 contribution 5% (A), day-4 contribution 10% (B, at the
    freshly reset 50% weight) -> total 15.5% compounded, one rebalance on
    Feb 1 exactly."""
    dates = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"])
    a_returns = pd.Series([0.10, 0.0, 0.0, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0, 0.20, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.5, "B": 0.5}, sma_length=20, rsi_length=14,
        rebalance_frequency="monthly",
    )

    assert err is None
    assert result.rebalance_dates == (pd.Timestamp("2024-02-01"),)
    assert result.total_return_pct == pytest.approx(15.5, abs=1e-9)
    assert result.contribution_pct["A"] == pytest.approx(5.0, abs=1e-9)
    assert result.contribution_pct["B"] == pytest.approx(10.0, abs=1e-9)


def test_no_frequency_means_no_periodic_rebalance_even_across_months(monkeypatch):
    dates = pd.to_datetime(["2024-01-30", "2024-02-01"])
    a_returns = pd.Series([0.10, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.5, "B": 0.5}, sma_length=20, rsi_length=14,
        rebalance_frequency="none",
    )
    assert err is None
    assert result.rebalance_dates == ()


# --- run_portfolio_backtest: threshold-based rebalancing ----------------------

def test_threshold_rebalance_triggers_once_drift_exceeds_threshold(monkeypatch):
    """A jumps 30% on day 1 alone, pushing its drifted weight to
    0.65/1.15 = 56.5% — a 6.5-point drift past the 50% target, past the
    5-point threshold. The check runs at the START of day 2 (using day 1's
    close-of-day drift), so day 2 opens already reset to 50/50."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    a_returns = pd.Series([0.30, 0.0, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.5, "B": 0.5}, sma_length=20, rsi_length=14,
        rebalance_frequency="none", rebalance_threshold_pct=5.0,
    )

    assert err is None
    assert result.rebalance_dates == (pd.Timestamp("2024-01-03"),)
    assert result.df.iloc[1]["Weight_A"] == pytest.approx(0.5)  # reset, not left at the drifted 56.5%


def test_threshold_rebalance_does_not_trigger_below_threshold(monkeypatch):
    """Same shape as the trigger test but a smaller jump (2%) that drifts
    weight by under a point — must NOT rebalance."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    a_returns = pd.Series([0.02, 0.0, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B")},
        rule=None, target_weights={"A": 0.5, "B": 0.5}, sma_length=20, rsi_length=14,
        rebalance_frequency="none", rebalance_threshold_pct=5.0,
    )
    assert err is None
    assert result.rebalance_dates == ()


# --- exclusion handling and weight re-spreading -------------------------------

def test_ticker_with_no_price_history_is_excluded_and_weight_respread(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    a_returns = pd.Series([0.01, 0.0], index=dates)
    b_returns = pd.Series([0.0, 0.0], index=dates)
    _patch_canned_results(monkeypatch, {"A": _fake_backtest_result(a_returns), "B": _fake_backtest_result(b_returns)})

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "B": _dummy_input_df(dates, "B"), "C": pd.DataFrame()},
        rule=None, target_weights={"A": 0.4, "B": 0.4, "C": 0.2}, sma_length=20, rsi_length=14,
    )
    assert err is None
    assert result.excluded_tickers == ("C",)
    assert "C" in result.exclusion_reasons
    # 0.4/0.4 re-spread across the survivors must stay a 50/50 SPLIT, not
    # silently leave the basket at 80% total exposure.
    assert result.target_weights == pytest.approx({"A": 0.5, "B": 0.5})


def test_all_tickers_failing_returns_aggregate_error_not_a_crash():
    result, err = run_portfolio_backtest(
        {"C": pd.DataFrame()}, rule=None, target_weights={"C": 1.0}, sma_length=20, rsi_length=14,
    )
    assert result is None
    assert "C" in err


def test_run_backtest_exception_excludes_that_ticker_without_crashing(monkeypatch):
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])

    def _boom(df, rule, sma, rsi, cost_bps=0.0):
        if df.attrs["ticker"] == "BAD":
            raise RuntimeError("indicator column missing")
        return _fake_backtest_result(pd.Series([0.01, 0.0], index=dates))

    monkeypatch.setattr(pb, "run_backtest", _boom)

    result, err = run_portfolio_backtest(
        {"A": _dummy_input_df(dates, "A"), "BAD": _dummy_input_df(dates, "BAD")},
        rule=None, target_weights={"A": 0.5, "BAD": 0.5}, sma_length=20, rsi_length=14,
    )
    assert err is None
    assert result.included_tickers == ("A",)
    assert "BAD" in result.exclusion_reasons
    assert result.target_weights == {"A": 1.0}
