"""Tests for strategy_builder.py — the no-code strategy rule schema,
condition evaluator, and generalized backtest engine.

All synthetic (no network calls) — see conftest.py for the shared OHLCV
fixtures.
"""
import numpy as np
import pandas as pd
import pytest

from risk_analytics import compute_max_drawdown
from strategy_builder import (
    StrategyCondition,
    StrategyRule,
    classic_mean_reversion,
    condition_library,
    evaluate_condition,
    evaluate_condition_set,
    run_backtest,
    run_walk_forward_backtest,
)
from technical_indicators import (
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    compute_sma_lines,
    detect_bollinger_breakouts,
    detect_macd_crossovers,
    detect_sma_crossovers,
)

SMA_LENGTH = 20
RSI_LENGTH = 14


@pytest.fixture
def prepared_df(clean_ohlcv):
    """clean_ohlcv with every column a strategy condition might read —
    mirrors exactly what finance.py's indicator-computation block already
    guarantees is present on `df` before the Strategy Builder section runs."""
    df = clean_ohlcv.copy()
    df = compute_sma_lines(df, [SMA_LENGTH])
    df[f"RSI_{RSI_LENGTH}"] = compute_rsi(df, RSI_LENGTH)
    df = compute_macd(df)
    df = compute_bollinger_bands(df, SMA_LENGTH)
    df["Returns"] = df["Close"].pct_change()
    df["Mean"] = df["Close"].rolling(window=SMA_LENGTH).mean()
    df["Std"] = df["Close"].rolling(window=SMA_LENGTH).std()
    df["Z_Score"] = (df["Close"] - df["Mean"]) / df["Std"]
    return df


def test_condition_library_covers_required_indicators():
    lib = condition_library(SMA_LENGTH, RSI_LENGTH)
    assert {"rsi", "zscore", "sma_cross", "macd_cross", "bollinger_breakout"} <= set(lib.keys())


def test_rsi_level_condition_fires_on_exactly_expected_bars(prepared_df):
    condition = StrategyCondition(indicator="rsi", operator="<", threshold=30.0)
    fired = evaluate_condition(prepared_df, condition, SMA_LENGTH, RSI_LENGTH)
    expected = prepared_df[f"RSI_{RSI_LENGTH}"] < 30.0
    pd.testing.assert_series_equal(fired, expected, check_names=False)


def test_zscore_level_condition_fires_on_exactly_expected_bars(prepared_df):
    condition = StrategyCondition(indicator="zscore", operator=">", threshold=1.5)
    fired = evaluate_condition(prepared_df, condition, SMA_LENGTH, RSI_LENGTH)
    expected = prepared_df["Z_Score"] > 1.5
    pd.testing.assert_series_equal(fired, expected, check_names=False)


def test_sma_crossover_event_condition_matches_detector_exactly(prepared_df):
    signals = detect_sma_crossovers(prepared_df, SMA_LENGTH)
    bullish_dates = {s.date for s in signals if s.kind == "bullish"}
    assert bullish_dates, "fixture should produce at least one bullish SMA crossover to make this test meaningful"

    condition = StrategyCondition(indicator="sma_cross", operator="bullish")
    fired = evaluate_condition(prepared_df, condition, SMA_LENGTH, RSI_LENGTH)
    assert set(prepared_df.index[fired]) == bullish_dates


def test_macd_crossover_event_condition_matches_detector_exactly(prepared_df):
    signals = detect_macd_crossovers(prepared_df)
    bearish_dates = {s.date for s in signals if s.kind == "bearish"}
    assert bearish_dates, "fixture should produce at least one bearish MACD crossover to make this test meaningful"

    condition = StrategyCondition(indicator="macd_cross", operator="bearish")
    fired = evaluate_condition(prepared_df, condition, SMA_LENGTH, RSI_LENGTH)
    assert set(prepared_df.index[fired]) == bearish_dates


def test_bollinger_breakout_event_condition_matches_detector_exactly(prepared_df):
    breakouts = detect_bollinger_breakouts(prepared_df)
    upper_dates = {b.date for b in breakouts if b.kind == "upper"}
    assert upper_dates, "fixture should produce at least one upper-band breakout to make this test meaningful"

    condition = StrategyCondition(indicator="bollinger_breakout", operator="upper")
    fired = evaluate_condition(prepared_df, condition, SMA_LENGTH, RSI_LENGTH)
    assert set(prepared_df.index[fired]) == upper_dates


def test_condition_set_and_logic_is_true_only_when_both_true(prepared_df):
    always_true = StrategyCondition(indicator="rsi", operator="<", threshold=100.0)   # RSI is always < 100
    sometimes_true = StrategyCondition(indicator="rsi", operator="<", threshold=30.0)
    combined = evaluate_condition_set(prepared_df, (always_true, sometimes_true), "AND", SMA_LENGTH, RSI_LENGTH)
    expected = prepared_df[f"RSI_{RSI_LENGTH}"] < 30.0
    pd.testing.assert_series_equal(combined, expected, check_names=False)


def test_condition_set_or_logic_is_true_when_either_true(prepared_df):
    never_true = StrategyCondition(indicator="rsi", operator="<", threshold=-1.0)   # RSI is never negative
    sometimes_true = StrategyCondition(indicator="rsi", operator="<", threshold=30.0)
    combined = evaluate_condition_set(prepared_df, (never_true, sometimes_true), "OR", SMA_LENGTH, RSI_LENGTH)
    expected = prepared_df[f"RSI_{RSI_LENGTH}"] < 30.0
    pd.testing.assert_series_equal(combined, expected, check_names=False)


def test_empty_condition_set_never_fires(prepared_df):
    combined = evaluate_condition_set(prepared_df, (), "AND", SMA_LENGTH, RSI_LENGTH)
    assert not combined.any()


def test_backtest_regression_matches_original_hardcoded_zscore_strategy(prepared_df):
    """Exact-match regression check (Asana acceptance criterion): the
    generalized engine running the Classic Mean-Reversion preset must
    produce IDENTICAL Position/Cum_Strategy/max-drawdown to finance.py's
    original hardcoded implementation, reproduced here line-for-line as it
    existed before the refactor."""
    buy_z, sell_z = -2.0, 0.0

    # --- original hardcoded logic (finance.py, pre-refactor) ---
    original = prepared_df.copy()
    original["Signal"] = 0
    original.loc[original["Z_Score"] < buy_z, "Signal"] = 1
    original.loc[original["Z_Score"] > sell_z, "Signal"] = -1
    original["Position"] = original["Signal"].replace(0, np.nan).ffill().fillna(0)
    original["Position"] = original["Position"].clip(lower=0)
    original["Strategy_Returns"] = original["Position"].shift(1) * original["Returns"]
    original["Cum_Strategy"] = (1 + original["Strategy_Returns"]).cumprod()
    original_total_return = (original["Cum_Strategy"].iloc[-1] - 1) * 100
    original_dd = compute_max_drawdown(original["Cum_Strategy"])

    # --- new generalized engine, same preset ---
    rule = classic_mean_reversion(buy_z_score=buy_z, sell_z_score=sell_z)
    result = run_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH)

    pd.testing.assert_series_equal(result.df["Position"], original["Position"], check_names=False)
    pd.testing.assert_series_equal(result.df["Cum_Strategy"], original["Cum_Strategy"], check_names=False)
    assert result.total_strategy_return_pct == pytest.approx(original_total_return)
    assert result.max_drawdown_pct == pytest.approx(original_dd.max_drawdown * 100)


def test_backtest_win_rate_and_trade_count_are_sane(prepared_df):
    rule = classic_mean_reversion(buy_z_score=-2.0, sell_z_score=0.0)
    result = run_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH)
    assert result.trade_count >= 0
    if result.win_rate_pct is not None:
        assert 0.0 <= result.win_rate_pct <= 100.0


def test_backtest_with_no_entry_conditions_never_enters_position(prepared_df):
    rule = StrategyRule(name="Custom", entry_conditions=(), entry_logic="AND",
                         exit_conditions=(StrategyCondition("rsi", ">", 70.0),), exit_logic="AND")
    result = run_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH)
    assert (result.df["Position"] == 0).all()
    assert result.trade_count == 0


# --- run_walk_forward_backtest() --------------------------------------------

def test_walk_forward_insufficient_history_returns_explicit_reason(prepared_df):
    short_df = prepared_df.iloc[:30]
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(short_df, rule, SMA_LENGTH, RSI_LENGTH, train_days=126, test_days=42)
    assert result.ok is False
    assert result.reason is not None and "30" in result.reason
    assert result.windows == ()
    assert result.stitched_equity_curve is None


def test_walk_forward_window_boundaries_are_sequential_non_overlapping_and_counted_correctly(prepared_df):
    train_days, test_days = 50, 25
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days, test_days)

    assert result.ok is True
    for i in range(1, len(result.windows)):
        assert result.windows[i].test_start > result.windows[i - 1].test_end

    expected_window_count = (len(prepared_df) - train_days) // test_days
    assert result.window_count == expected_window_count
    assert len(result.stitched_equity_curve) == expected_window_count * test_days


def test_walk_forward_drops_partial_final_window_rather_than_fabricating_it(prepared_df):
    train_days, test_days = 50, 30  # 300-bar fixture: (300-50)=250, 250 // 30 = 8 remainder 10 -> a genuine partial leftover
    total_days = len(prepared_df)
    remainder = (total_days - train_days) % test_days
    assert remainder != 0  # sanity: this really exercises a non-exact division

    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days, test_days)
    assert result.window_count == (total_days - train_days) // test_days


def test_walk_forward_position_continuity_across_train_test_boundary(prepared_df):
    """A position opened inside window 0's TRAIN segment, that never exits,
    must still be open at the test segment's very first bar — not wrongly
    reset to flat just because the reported window starts there. Proven
    precisely: if continuity held, the stitched curve's first-bar return
    exactly equals that day's real price return (Position=1); if it didn't,
    it would be exactly 0 (Position wrongly flat)."""
    df = prepared_df.copy()
    train_days, test_days = 40, 20
    df["Z_Score"] = -1.0  # between buy(-2) and sell(0): holds an open position, never re-fires either signal
    df.iloc[5, df.columns.get_loc("Z_Score")] = -3.0  # one bar deep in train: fires the entry

    rule = classic_mean_reversion(buy_z_score=-2.0, sell_z_score=0.0)
    result = run_walk_forward_backtest(df, rule, SMA_LENGTH, RSI_LENGTH, train_days, test_days)

    assert result.ok is True
    first_stitched_return = result.stitched_equity_curve.iloc[0] - 1
    actual_price_return_at_test_start = df["Returns"].iloc[train_days]
    assert first_stitched_return == pytest.approx(actual_price_return_at_test_start)


def test_walk_forward_pooled_trade_count_and_win_rate_are_sane(prepared_df):
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days=50, test_days=25)
    assert result.trade_count >= 0
    assert result.trade_count == sum(w.trade_count for w in result.windows)
    if result.win_rate_pct is not None:
        assert 0.0 <= result.win_rate_pct <= 100.0


def test_walk_forward_max_drawdown_matches_manual_computation_on_stitched_curve(prepared_df):
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days=50, test_days=25)
    manual_dd = compute_max_drawdown(result.stitched_equity_curve)
    assert result.max_drawdown_pct == pytest.approx(manual_dd.max_drawdown * 100)


def test_walk_forward_total_return_matches_stitched_curve_final_value(prepared_df):
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days=50, test_days=25)
    assert result.total_oos_return_pct == pytest.approx((result.stitched_equity_curve.iloc[-1] - 1) * 100)


def test_walk_forward_per_window_return_matches_its_slice_of_the_stitched_curve(prepared_df):
    train_days, test_days = 50, 25
    rule = classic_mean_reversion(-2.0, 0.0)
    result = run_walk_forward_backtest(prepared_df, rule, SMA_LENGTH, RSI_LENGTH, train_days, test_days)

    for i, window in enumerate(result.windows):
        segment = result.stitched_equity_curve.iloc[i * test_days:(i + 1) * test_days]
        start_value = result.stitched_equity_curve.iloc[i * test_days - 1] if i > 0 else 1.0
        manual_return_pct = (segment.iloc[-1] / start_value - 1) * 100
        assert window.strategy_return_pct == pytest.approx(manual_return_pct)
