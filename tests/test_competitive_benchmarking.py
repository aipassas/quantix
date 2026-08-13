"""Tests for competitive_benchmarking.py — peer-group metric building,
group-average computation, outperform/laggard flagging, and the overall
per-peer verdict roll-up.
"""
import pytest

from competitive_benchmarking import (
    METRICS,
    BenchmarkRow,
    MetricFlag,
    PeerMetrics,
    build_benchmark_rows,
    build_peer_metrics,
    flag_metric,
    group_average,
)


def _row(ticker, is_target=False, **values):
    return PeerMetrics(ticker=ticker, is_target=is_target, values=values)


# --- build_peer_metrics -------------------------------------------------------

class _FakeStd:
    pe_ratio = 20.0
    peg_ratio = 1.5
    price_to_book = 5.0
    debt_to_equity = 0.4
    return_on_equity = 0.25   # decimal -> should become 25.0
    net_margin = 0.30         # decimal -> should become 30.0
    earnings_growth = 0.10    # decimal -> should become 10.0


def test_build_peer_metrics_converts_decimals_to_percent():
    row = build_peer_metrics("AAPL", _FakeStd(), is_target=True, momentum_pct=12.5)
    assert row.values["return_on_equity"] == pytest.approx(25.0)
    assert row.values["net_margin"] == pytest.approx(30.0)
    assert row.values["earnings_growth"] == pytest.approx(10.0)
    assert row.values["momentum_pct"] == pytest.approx(12.5)
    assert row.values["pe_ratio"] == 20.0
    assert row.is_target is True


def test_build_peer_metrics_handles_none_fields_without_crashing():
    class _EmptyStd:
        pe_ratio = None
        peg_ratio = None
        price_to_book = None
        debt_to_equity = None
        return_on_equity = None
        net_margin = None
        earnings_growth = None
    row = build_peer_metrics("X", _EmptyStd(), is_target=False, momentum_pct=None)
    assert all(v is None for v in row.values.values())


# --- group_average -------------------------------------------------------------

def test_group_average_ignores_missing_values():
    rows = [_row("A", pe_ratio=10.0), _row("B", pe_ratio=None), _row("C", pe_ratio=20.0)]
    assert group_average(rows, "pe_ratio") == pytest.approx(15.0)


def test_group_average_none_when_nobody_has_it():
    rows = [_row("A", pe_ratio=None), _row("B", pe_ratio=None)]
    assert group_average(rows, "pe_ratio") is None


def test_group_average_single_value_is_itself():
    rows = [_row("A", pe_ratio=42.0)]
    assert group_average(rows, "pe_ratio") == pytest.approx(42.0)


# --- flag_metric -----------------------------------------------------------------

def test_flag_metric_unavailable_when_value_missing():
    flag = flag_metric(None, average=10.0, higher_is_better=True)
    assert flag.verdict == "unavailable"


def test_flag_metric_unavailable_when_average_missing():
    flag = flag_metric(10.0, average=None, higher_is_better=True)
    assert flag.verdict == "unavailable"


def test_flag_metric_unavailable_when_average_is_zero():
    """Avoids a division by zero in the percent-distance calc."""
    flag = flag_metric(5.0, average=0.0, higher_is_better=True)
    assert flag.verdict == "unavailable"


def test_flag_metric_outperform_higher_is_better():
    # 20 vs average 10 = +100% distance, favorable when higher is better.
    flag = flag_metric(20.0, average=10.0, higher_is_better=True)
    assert flag.verdict == "outperform"


def test_flag_metric_laggard_higher_is_better():
    flag = flag_metric(5.0, average=10.0, higher_is_better=True)
    assert flag.verdict == "laggard"


def test_flag_metric_outperform_lower_is_better():
    # A cheaper P/E than average is favorable when lower is better.
    flag = flag_metric(5.0, average=10.0, higher_is_better=False)
    assert flag.verdict == "outperform"


def test_flag_metric_laggard_lower_is_better():
    flag = flag_metric(20.0, average=10.0, higher_is_better=False)
    assert flag.verdict == "laggard"


def test_flag_metric_in_line_within_threshold():
    from config import COMPETITIVE_BENCHMARKING
    # A value just under the threshold distance from average.
    tiny_gap = 10.0 * (1 + (COMPETITIVE_BENCHMARKING.outperform_threshold_pct - 1) / 100)
    flag = flag_metric(tiny_gap, average=10.0, higher_is_better=True)
    assert flag.verdict == "in_line"


def test_flag_metric_icon_matches_verdict():
    assert flag_metric(20.0, 10.0, True).icon == "🟢"
    assert flag_metric(5.0, 10.0, True).icon == "🔴"
    assert flag_metric(10.0, 10.0, True).icon == "⚪"
    assert flag_metric(None, 10.0, True).icon == "⚪"


# --- build_benchmark_rows (overall verdict roll-up) ---------------------------

def test_build_benchmark_rows_target_participates_in_the_average():
    """The target's own value pulls the group average exactly like any
    peer's — this is a comparison among equals, not target-vs-peer-only."""
    rows = [
        _row("TARGET", is_target=True, pe_ratio=100.0),
        _row("PEER", pe_ratio=10.0),
    ]
    result = build_benchmark_rows(rows)
    target_row = next(r for r in result if r.ticker == "TARGET")
    # Average of (100, 10) = 55, so target's own metric feeds its own average.
    assert target_row.flags["pe_ratio"].group_average == pytest.approx(55.0)


def test_build_benchmark_rows_clear_outperformer():
    rows = [
        _row("A", pe_ratio=5.0, return_on_equity=50.0, net_margin=50.0, earnings_growth=50.0, momentum_pct=50.0, price_to_book=1.0, debt_to_equity=0.1, peg_ratio=0.1),
        _row("B", pe_ratio=50.0, return_on_equity=5.0, net_margin=5.0, earnings_growth=5.0, momentum_pct=5.0, price_to_book=10.0, debt_to_equity=2.0, peg_ratio=3.0),
    ]
    result = build_benchmark_rows(rows)
    a = next(r for r in result if r.ticker == "A")
    assert a.overall_verdict == "Outperformer"
    assert a.overall_icon == "🟢"


def test_build_benchmark_rows_clear_laggard():
    rows = [
        _row("A", pe_ratio=5.0, return_on_equity=50.0, net_margin=50.0),
        _row("B", pe_ratio=50.0, return_on_equity=5.0, net_margin=5.0),
    ]
    result = build_benchmark_rows(rows)
    b = next(r for r in result if r.ticker == "B")
    assert b.overall_verdict == "Laggard"


def test_build_benchmark_rows_no_evaluable_metrics_is_not_enough_data():
    rows = [_row("A"), _row("B")]  # no values at all
    result = build_benchmark_rows(rows)
    assert all(r.overall_verdict == "Not Enough Data" for r in result)
    assert all(r.overall_icon == "⚪" for r in result)


def test_build_benchmark_rows_preserves_row_count_and_tickers():
    rows = [_row("A", pe_ratio=10.0), _row("B", pe_ratio=20.0), _row("C", pe_ratio=30.0)]
    result = build_benchmark_rows(rows)
    assert [r.ticker for r in result] == ["A", "B", "C"]


def test_all_declared_metrics_appear_in_every_row_flags():
    rows = [_row("A", pe_ratio=10.0), _row("B", pe_ratio=20.0)]
    result = build_benchmark_rows(rows)
    for row in result:
        assert set(row.flags.keys()) == {m.key for m in METRICS}
