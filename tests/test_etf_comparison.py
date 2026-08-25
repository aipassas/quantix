"""Side-by-side fund comparison.

The arithmetic here is easy to get wrong in ways nothing crashes on: two
funds rebased from their own first dates silently compare different
windows, a "3-year return" computed over one year of data is not one, and
a tie marked as a winner invents a difference the data does not show.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import etf_comparison as ec


ROOT = Path(__file__).resolve().parent.parent
FINANCE = (ROOT / "finance.py").read_text(encoding="utf-8")


class _Holding:
    def __init__(self, symbol, weight_pct=1.0):
        self.symbol, self.name, self.weight_pct = symbol, symbol, weight_pct


class _Profile:
    ok = True

    def __init__(self, symbol, **over):
        self.symbol = symbol
        self.expense_ratio_pct = over.get("expense_ratio_pct", 0.10)
        self.net_assets = over.get("net_assets", 1e10)
        self.price_earnings = over.get("price_earnings", 20.0)
        self.top_holdings = tuple(
            _Holding(s, w) for s, w in over.get("holdings", ()))


def _closes(**series):
    index = pd.date_range("2025-01-01", periods=len(next(iter(series.values()))),
                          freq="B")
    return pd.DataFrame({k: pd.Series(v, index=index, dtype="float64")
                         for k, v in series.items()})


# --- rebasing -----------------------------------------------------------------

def test_both_series_start_at_100_on_the_first_date_they_both_have():
    """Rebasing each column at its own first valid date would start two
    funds on different days and compare different windows."""
    frame = _closes(A=[np.nan, 10.0, 11.0, 12.0], B=[5.0, 5.0, 6.0, 7.0])
    out = ec.rebased(frame)
    assert list(out.iloc[0]) == [100.0, 100.0]
    assert len(out) == 3, "the row where A is missing is dropped for BOTH"
    assert out["A"].iloc[-1] == pytest.approx(120.0)
    assert out["B"].iloc[-1] == pytest.approx(140.0)


def test_rebasing_nothing_returns_nothing():
    assert ec.rebased(None) is None
    assert ec.rebased(pd.DataFrame()) is None
    assert ec.rebased(_closes(A=[1.0])) is None, "one bar is not a series"


# --- returns ------------------------------------------------------------------

def test_a_three_year_return_over_one_year_of_data_is_not_reported():
    """Silently shortening the window would make every young fund look
    like it had a full history."""
    frame = _closes(A=list(np.linspace(100, 120, 251)))
    assert ec.total_return_pct(frame, "A") == pytest.approx(20.0)
    assert ec.total_return_pct(frame, "A", bars=756) is None


def test_a_return_over_the_whole_window_uses_both_ends():
    frame = _closes(A=[100.0, 50.0, 150.0])
    assert ec.total_return_pct(frame, "A") == pytest.approx(50.0)


def test_returns_of_an_absent_symbol_are_absent_not_zero():
    frame = _closes(A=[100.0, 110.0])
    assert ec.total_return_pct(frame, "MISSING") is None
    assert ec.total_return_pct(None, "A") is None
    assert ec.volatility_pct(None, "A") is None
    assert ec.sharpe(None, "A") is None


def test_a_flat_series_has_no_sharpe_rather_than_a_division_by_zero():
    frame = _closes(A=[100.0] * 50)
    assert ec.volatility_pct(frame, "A") == pytest.approx(0.0)
    assert ec.sharpe(frame, "A") is None


def test_volatility_is_annualised():
    rng = np.random.default_rng(0)
    daily = rng.normal(0, 0.01, 500)
    prices = 100 * np.cumprod(1 + daily)
    vol = ec.volatility_pct(_closes(A=list(prices)), "A")
    # 1% daily vol annualises to roughly 16%.
    assert 12.0 < vol < 20.0, vol


# --- the grid -----------------------------------------------------------------

def test_a_fund_that_reports_nothing_holds_none_not_zero():
    """A zero is absorbed into whatever average the reader forms; a blank
    is skipped."""
    rows = ec.build_rows({"A": None}, None, {})
    for row in rows:
        assert row.values["A"] is None, row.key
        assert ec.format_value(row, "A") == "Not reported"


def test_the_grid_covers_every_declared_metric():
    profiles = {"A": _Profile("A"), "B": _Profile("B")}
    frame = _closes(A=list(np.linspace(100, 120, 300)),
                    B=list(np.linspace(100, 110, 300)))
    rows = ec.build_rows(profiles, frame, {"A": 1.5, "B": 2.5})
    assert [r.key for r in rows] == [m[0] for m in ec.METRICS]
    assert len(rows) == 8


def test_the_expense_row_ranks_lower_as_better():
    profiles = {"A": _Profile("A", expense_ratio_pct=0.03),
                "B": _Profile("B", expense_ratio_pct=0.75)}
    rows = {r.key: r for r in ec.build_rows(profiles, None, {})}
    assert ec.best_symbol(rows["expense_ratio_pct"]) == "A"


def test_size_is_not_ranked():
    """A fund is not superior for holding more assets."""
    profiles = {"A": _Profile("A", net_assets=1e11),
                "B": _Profile("B", net_assets=1e9)}
    rows = {r.key: r for r in ec.build_rows(profiles, None, {})}
    assert rows["net_assets"].lower_is_better is None
    assert ec.best_symbol(rows["net_assets"]) is None


def test_a_tie_has_no_winner():
    """Marking one arbitrarily invents a difference the data does not
    show."""
    profiles = {"A": _Profile("A", expense_ratio_pct=0.10),
                "B": _Profile("B", expense_ratio_pct=0.10)}
    rows = {r.key: r for r in ec.build_rows(profiles, None, {})}
    assert ec.best_symbol(rows["expense_ratio_pct"]) is None


def test_one_fund_reporting_a_metric_is_not_a_comparison():
    profiles = {"A": _Profile("A", expense_ratio_pct=0.10),
                "B": _Profile("B", expense_ratio_pct=None)}
    rows = {r.key: r for r in ec.build_rows(profiles, None, {})}
    assert ec.best_symbol(rows["expense_ratio_pct"]) is None


def test_money_is_formatted_compactly_and_percents_carry_their_sign():
    profiles = {"A": _Profile("A", net_assets=7.95e11, expense_ratio_pct=0.0945)}
    rows = {r.key: r for r in ec.build_rows(profiles, None, {})}
    assert ec.format_value(rows["net_assets"], "A") == "$795.0B"
    assert ec.format_value(rows["expense_ratio_pct"], "A") == "0.09%"


# --- holdings overlap ---------------------------------------------------------

def test_overlap_splits_shared_from_each_sides_own():
    a = _Profile("A", holdings=(("AAPL", 7.0), ("MSFT", 6.0), ("JPM", 3.0)))
    b = _Profile("B", holdings=(("AAPL", 9.0), ("MSFT", 8.0), ("AMD", 2.0)))
    overlap = ec.holdings_overlap(a, b)
    assert overlap.shared == ("AAPL", "MSFT")
    assert overlap.only_a == ("JPM",)
    assert overlap.only_b == ("AMD",)
    assert overlap.shared_weight_pct == pytest.approx(13.0)
    assert overlap.ok


def test_a_fund_with_no_disclosed_holdings_yields_an_empty_overlap():
    """Normal for a bond or commodity fund — not a failure."""
    empty = ec.holdings_overlap(_Profile("A", holdings=()),
                                _Profile("B", holdings=()))
    assert not empty.ok
    assert ec.holdings_overlap(None, None).shared == ()


def test_the_overlap_says_it_is_only_the_top_ten():
    """Yahoo returns ten holdings, about 37-46% of a fund. A reader who
    thinks this is the full overlap would badly misjudge how correlated
    two funds are."""
    assert "TOP TEN" in ec.HOLDINGS_CAVEAT
    assert "37-46%" in ec.HOLDINGS_CAVEAT
    assert ec.HOLDINGS_CAVEAT in FINANCE or "HOLDINGS_CAVEAT" in FINANCE


# --- what is not built --------------------------------------------------------

def test_tracking_error_is_declared_unavailable_rather_than_approximated():
    """It needs each fund's stated benchmark as a return series, and this
    source returns neither the mapping nor the index history. "Tracking
    error vs. something we picked" is a different number under the same
    name."""
    assert "benchmark" in ec.TRACKING_ERROR_UNAVAILABLE
    assert "tracking_error" not in {m[0] for m in ec.METRICS}
    assert "etf_comparison.TRACKING_ERROR_UNAVAILABLE" in FINANCE


# --- how it is wired ----------------------------------------------------------

def test_the_comparison_tab_only_exists_for_funds():
    assert "if tab_comparison is not None:" in FINANCE
    assert "with tab_comparison:" in FINANCE


def test_the_comparison_is_capped_and_says_so():
    assert ec.MAX_FUNDS == 3
    assert "etf_comparison.MAX_FUNDS" in FINANCE


def test_the_overlay_is_themed_like_every_other_chart():
    """Plotly's default is a white background, which is invisible against
    the app's dark theme."""
    block = FINANCE[FINANCE.index("_cmp_fig = go.Figure()"):]
    block = block[:block.index("st.plotly_chart(_cmp_fig")]
    assert "template=_plotly_template" in block
    assert "paper_bgcolor='rgba(0,0,0,0)'" in block
