"""Tests for screener.py — the multi-ticker screening engine.

Pure-logic tests (operators, ScreenResult.passed_all, metric registry
sanity) run by default. Full end-to-end screens against real tickers are
network-dependent and live behind @pytest.mark.live, same convention as
the rest of this suite (see tests/test_live_sanity.py).
"""
import datetime

import pytest

from screener import (
    METRICS,
    METRICS_BY_KEY,
    OPERATORS,
    ScreenCriterion,
    ScreenResult,
    run_screen,
)


def test_metric_registry_has_unique_keys():
    keys = [m.key for m in METRICS]
    assert len(keys) == len(set(keys))
    assert set(METRICS_BY_KEY.keys()) == set(keys)


def test_metric_registry_tiers_are_valid():
    assert {m.tier for m in METRICS} <= {"shallow", "deep", "price"}


@pytest.mark.parametrize("op,v,t,expected", [
    (">", 10, 5, True), (">", 5, 10, False),
    ("<", 5, 10, True), ("<", 10, 5, False),
    (">=", 5, 5, True), (">=", 4, 5, False),
    ("<=", 5, 5, True), ("<=", 6, 5, False),
])
def test_operators_evaluate_correctly(op, v, t, expected):
    assert OPERATORS[op](v, t) is expected


def test_passed_all_none_when_any_criterion_unevaluated():
    # A metric that couldn't be computed for this ticker (None) must not be
    # treated as an automatic failure — passed_all should report "unknown"
    # (None), not False.
    result = ScreenResult(ticker="X", criteria_passes=[True, None])
    assert result.passed_all is None


def test_passed_all_true_only_when_every_criterion_passes():
    assert ScreenResult(ticker="X", criteria_passes=[True, True]).passed_all is True
    assert ScreenResult(ticker="X", criteria_passes=[True, False]).passed_all is False


def test_passed_all_none_for_empty_criteria():
    assert ScreenResult(ticker="X", criteria_passes=[]).passed_all is None


@pytest.mark.live
def test_run_screen_end_to_end_on_real_tickers():
    """Mixed shallow (P/E), price (RSI), and deep (Altman Z) criteria across
    a small real universe including one invalid ticker — checks that valid
    tickers are evaluated consistently against their own reported values and
    that a bad ticker is reported, never silently dropped."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=400)
    criteria = (
        ScreenCriterion(metric="pe_ratio", operator="<", threshold=1000),  # trivially true if P/E exists
        ScreenCriterion(metric="rsi", operator=">", threshold=-1),          # trivially true if RSI exists
        ScreenCriterion(metric="altman_z", operator=">", threshold=-1000),  # trivially true if Altman Z exists
    )
    tickers = ("AAPL", "MSFT", "NOTATICKERXYZ123")
    results = run_screen(tickers, criteria, start, end, risk_free_rate=0.04)

    by_ticker = {r.ticker: r for r in results}
    assert set(by_ticker) == set(tickers)

    for t in ("AAPL", "MSFT"):
        r = by_ticker[t]
        assert r.status == "ok", f"{t} should have every criterion's metric computable, got status={r.status} detail={r.detail}"
        assert r.passed_all is True
        assert r.values["pe_ratio"] is not None
        assert r.values["rsi"] is not None
        assert r.values["altman_z"] is not None
        # Re-derive pass/fail independently from the reported values to
        # confirm the engine's own evaluation matches its own reported data.
        for criterion, passed in zip(criteria, r.criteria_passes):
            assert passed == OPERATORS[criterion.operator](r.values[criterion.metric], criterion.threshold)

    bad = by_ticker["NOTATICKERXYZ123"]
    assert bad.status == "fetch_error"
    assert bad.detail  # a human-readable reason, never silently empty
    assert all(p is None for p in bad.criteria_passes)
    assert bad.passed_all is None
