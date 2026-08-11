"""Tests for realtime_alerts.py — the Real-Time Alert Engine's rule
evaluation, edge-triggering, and local persistence.

Network-dependent pieces (_load_indicator_frame, load_quote_snapshots,
compute_watchlist_snapshots) are exercised via stubs/synthetic frames, not
the network, so trigger logic is verified deterministically.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from realtime_alerts import (
    AlertRule,
    EvaluationResult,
    TriggerEvent,
    _evaluate_fundamental_rule,
    _evaluate_indicator_rule,
    _evaluate_price_rule,
    detect_new_triggers,
    load_store,
    new_rule_id,
    save_store,
)
from risk_alerts import TickerRiskSnapshot
from watchlist_panel import QuoteSnapshot


def _rule(**kwargs) -> AlertRule:
    defaults = dict(id=new_rule_id(), ticker="AAPL", created_at="2026-01-01T00:00:00")
    defaults.update(kwargs)
    return AlertRule(**defaults)


# --- AlertRule.label ---------------------------------------------------------

def test_label_price_rule():
    r = _rule(trigger_type="price_above", threshold=250.0)
    assert r.label == "AAPL: Price rises to or above $250.00"


def test_label_fundamental_rule():
    r = _rule(trigger_type="fundamental", metric="altman_z", operator="<", threshold=1.81)
    assert "AAPL:" in r.label and "Altman Z-Score" in r.label and "< 1.81" in r.label


def test_label_indicator_rule_has_no_placeholder_braces():
    r = _rule(trigger_type="sma_cross_bullish")
    assert "{" not in r.label and "}" not in r.label


# --- _evaluate_price_rule ----------------------------------------------------

def test_price_above_met_when_price_at_or_over_threshold():
    r = _rule(trigger_type="price_above", threshold=100.0)
    q = QuoteSnapshot(ticker="AAPL", price=100.0, previous_close=99.0, change_pct=1.0, status="ok")
    result = _evaluate_price_rule(r, q)
    assert result.is_met is True and result.status == "ok"


def test_price_below_not_met_when_price_above_threshold():
    r = _rule(trigger_type="price_below", threshold=100.0)
    q = QuoteSnapshot(ticker="AAPL", price=101.0, previous_close=99.0, change_pct=1.0, status="ok")
    assert _evaluate_price_rule(r, q).is_met is False


def test_price_rule_unavailable_quote_is_not_met_not_error():
    r = _rule(trigger_type="price_above", threshold=100.0)
    q = QuoteSnapshot(ticker="AAPL", status="unavailable", detail="no data")
    result = _evaluate_price_rule(r, q)
    assert result.is_met is False
    assert result.status == "fetch_error"
    assert result.detail == "no data"


# --- _evaluate_fundamental_rule -----------------------------------------------

def test_fundamental_rule_met():
    r = _rule(trigger_type="fundamental", metric="risk_score", operator="<", threshold=50.0)
    snap = TickerRiskSnapshot(ticker="AAPL", values={"risk_score": 40.0}, status="ok")
    result = _evaluate_fundamental_rule(r, snap)
    assert result.is_met is True
    assert "40.0" in result.detail


def test_fundamental_rule_missing_metric_is_insufficient_not_met():
    r = _rule(trigger_type="fundamental", metric="altman_z", operator="<", threshold=1.81)
    snap = TickerRiskSnapshot(ticker="AAPL", values={}, status="ok")
    result = _evaluate_fundamental_rule(r, snap)
    assert result.is_met is False
    assert result.status == "insufficient_data"


def test_fundamental_rule_snapshot_fetch_error_propagates():
    r = _rule(trigger_type="fundamental", metric="altman_z", operator="<", threshold=1.81)
    snap = TickerRiskSnapshot(ticker="AAPL", values={}, status="fetch_error", detail="could not load")
    result = _evaluate_fundamental_rule(r, snap)
    assert result.is_met is False
    assert result.status == "fetch_error"
    assert result.detail == "could not load"


# --- _evaluate_indicator_rule --------------------------------------------------

def test_indicator_rule_no_data_is_insufficient():
    r = _rule(trigger_type="sma_cross_bullish")
    result = _evaluate_indicator_rule(r, None, "no price history returned")
    assert result.is_met is False
    assert result.status == "insufficient_data"
    assert result.detail == "no price history returned"


def test_sma_bullish_crossover_met_on_the_latest_bar(clean_ohlcv):
    from technical_indicators import compute_sma_lines
    from config import CHART_DEFAULTS

    df = compute_sma_lines(clean_ohlcv, [CHART_DEFAULTS.sma_default])
    # Force a bullish crossover exactly on the final bar: previous bar
    # below its SMA, final bar above.
    sma_col = f"SMA_{CHART_DEFAULTS.sma_default}"
    df.loc[df.index[-2], "Close"] = df.loc[df.index[-2], sma_col] - 1.0
    df.loc[df.index[-1], "Close"] = df.loc[df.index[-1], sma_col] + 1.0
    df = compute_sma_lines(df.drop(columns=[sma_col]), [CHART_DEFAULTS.sma_default])

    r = _rule(trigger_type="sma_cross_bullish")
    result = _evaluate_indicator_rule(r, df, "")
    assert result.is_met is True


def test_sma_bearish_rule_not_met_when_latest_crossover_is_bullish(clean_ohlcv):
    from technical_indicators import compute_sma_lines
    from config import CHART_DEFAULTS

    sma_col = f"SMA_{CHART_DEFAULTS.sma_default}"
    df = compute_sma_lines(clean_ohlcv, [CHART_DEFAULTS.sma_default])
    df.loc[df.index[-2], "Close"] = df.loc[df.index[-2], sma_col] - 1.0
    df.loc[df.index[-1], "Close"] = df.loc[df.index[-1], sma_col] + 1.0
    df = compute_sma_lines(df.drop(columns=[sma_col]), [CHART_DEFAULTS.sma_default])

    r = _rule(trigger_type="sma_cross_bearish")
    result = _evaluate_indicator_rule(r, df, "")
    assert result.is_met is False


def test_rsi_overbought_met_when_series_pinned_high():
    # A monotonically rising series drives Wilder RSI toward 100 —
    # deterministically overbought without depending on synthetic-fixture
    # randomness.
    idx = pd.date_range("2025-01-01", periods=60, freq="D")
    close = pd.Series(range(100, 160), index=idx, dtype=float)
    df = pd.DataFrame({"Close": close})

    from technical_indicators import compute_rsi
    from config import CHART_DEFAULTS
    df["RSI"] = compute_rsi(df, CHART_DEFAULTS.rsi_default)

    r = _rule(trigger_type="rsi_overbought")
    result = _evaluate_indicator_rule(r, df, "")
    assert result.is_met is True


def test_rsi_warmup_period_is_insufficient_not_false_negative():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 99.0]}, index=idx)
    df["RSI"] = float("nan")

    r = _rule(trigger_type="rsi_oversold")
    result = _evaluate_indicator_rule(r, df, "")
    assert result.is_met is False
    assert result.status == "insufficient_data"


# --- detect_new_triggers (edge-triggering) ------------------------------------

def test_first_poll_with_no_prior_state_treats_current_true_as_new():
    results = {"r1": EvaluationResult(rule_id="r1", is_met=True, detail="")}
    assert detect_new_triggers(results, {}) == ["r1"]


def test_still_active_condition_does_not_retrigger():
    results = {"r1": EvaluationResult(rule_id="r1", is_met=True, detail="")}
    assert detect_new_triggers(results, {"r1": True}) == []


def test_condition_clearing_then_retriggering_fires_again():
    results = {"r1": EvaluationResult(rule_id="r1", is_met=True, detail="")}
    # Was False last poll (already cleared once) -> True now: a genuine new trigger.
    assert detect_new_triggers(results, {"r1": False}) == ["r1"]


def test_not_met_rule_never_appears_in_new_triggers():
    results = {"r1": EvaluationResult(rule_id="r1", is_met=False, detail="")}
    assert detect_new_triggers(results, {}) == []


# --- persistence ---------------------------------------------------------------

def test_load_store_missing_file_returns_empty(tmp_path):
    rules, history = load_store(tmp_path / "does_not_exist.json")
    assert rules == [] and history == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "store.json"
    rule = _rule(trigger_type="price_above", threshold=250.0)
    event = TriggerEvent(rule_id=rule.id, ticker="AAPL", trigger_type="price_above", detail="hit", triggered_at="2026-01-01T00:00:01")

    save_store([rule], [event], path)
    rules, history = load_store(path)

    assert rules == [rule]
    assert history == [event]


def test_load_store_corrupt_file_degrades_to_empty_not_raise(tmp_path):
    path = tmp_path / "store.json"
    path.write_text("{not valid json at all")
    rules, history = load_store(path)
    assert rules == [] and history == []


def test_save_store_trims_history_to_configured_max(tmp_path):
    from config import REALTIME_ALERTS

    path = tmp_path / "store.json"
    events = [
        TriggerEvent(rule_id="r1", ticker="AAPL", trigger_type="price_above", detail=f"e{i}", triggered_at=f"t{i}")
        for i in range(REALTIME_ALERTS.max_history + 20)
    ]
    save_store([], events, path)
    _, history = load_store(path)
    assert len(history) == REALTIME_ALERTS.max_history
    # The trim keeps the MOST RECENT events, not the oldest.
    assert history[-1].detail == f"e{REALTIME_ALERTS.max_history + 19}"


def test_save_store_writes_atomically_no_tmp_file_left_behind(tmp_path):
    path = tmp_path / "store.json"
    save_store([], [], path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()
