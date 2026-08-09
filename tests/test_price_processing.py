"""Tests for price_processing.py — see price_processing.py's own module
docstring for the design principles being verified here (never raise,
never fabricate, minimize memory overhead)."""
import numpy as np
import pandas as pd

from price_processing import process_price_data, GAP_MULTIPLE_OF_TYPICAL_SPACING


def test_clean_data_is_reported_clean(clean_ohlcv):
    result = process_price_data(clean_ohlcv, ticker="TEST")
    assert result.is_clean
    assert result.issue_count == 0
    assert len(result.df) == len(clean_ohlcv)


def test_empty_input_returns_empty_result_with_warning():
    result = process_price_data(pd.DataFrame())
    assert result.df.empty
    assert result.warnings == ["No price data to process"]


def test_none_input_returns_empty_result():
    result = process_price_data(None)
    assert result.df.empty


def test_removes_duplicate_timestamps(clean_ohlcv):
    duplicated = pd.concat([clean_ohlcv, clean_ohlcv.iloc[[5]]])
    result = process_price_data(duplicated)
    assert result.duplicate_rows_removed == 1
    assert len(result.df) == len(clean_ohlcv)
    assert not result.df.index.duplicated().any()


def test_drops_structurally_invalid_rows(clean_ohlcv):
    corrupted = clean_ohlcv.copy()
    # High below Low: structurally impossible for a real bar.
    corrupted.iloc[10, corrupted.columns.get_loc("High")] = corrupted.iloc[10]["Low"] - 1.0
    # Negative Close: also impossible.
    corrupted.iloc[20, corrupted.columns.get_loc("Close")] = -5.0

    result = process_price_data(corrupted)
    assert result.invalid_rows_removed == 2
    assert len(result.df) == len(clean_ohlcv) - 2
    assert (result.df["High"] >= result.df["Low"]).all()
    assert (result.df["Close"] > 0).all()


def test_normalizes_tz_aware_index_to_naive(clean_ohlcv):
    tz_aware = clean_ohlcv.copy()
    tz_aware.index = tz_aware.index.tz_localize("America/New_York")
    result = process_price_data(tz_aware)
    assert result.df.index.tz is None


def test_consolidates_adj_close_column(clean_ohlcv):
    with_adj = clean_ohlcv.copy()
    # A real split adjustment scales every OHLC column proportionally, not
    # just Close in isolation — scaling Close alone would push it outside
    # [Low, High] and get the row dropped as structurally invalid, which
    # would make this test accidentally pass for the wrong reason.
    with_adj["Adj Close"] = with_adj["Close"] * 0.98
    for col in ("Open", "High", "Low", "Close"):
        with_adj[col] = with_adj[col] * 0.98

    result = process_price_data(with_adj)
    assert "Adj Close" not in result.df.columns
    np.testing.assert_array_equal(result.df["Close"].values, with_adj["Adj Close"].values)


def test_sorts_unsorted_index(clean_ohlcv):
    shuffled = clean_ohlcv.sample(frac=1.0, random_state=1)
    result = process_price_data(shuffled)
    assert result.df.index.is_monotonic_increasing


def test_ordinary_holiday_weekend_is_not_flagged_as_a_gap(ohlcv_factory):
    """Regression test for the exact false-positive this module's docstring
    documents: a 4-calendar-day gap from a 3-day US holiday weekend, against
    a 1-day typical spacing, should NOT be flagged at the current threshold
    (GAP_MULTIPLE_OF_TYPICAL_SPACING=5) — it was flagged at the old value (3).
    """
    df = ohlcv_factory(n=50, start_date="2024-01-02")  # all business days, 1-day typical spacing
    # Simulate a single 3-day-weekend-style gap: drop a Monday, leaving Friday -> Tuesday (4 calendar days).
    friday_idx = df.index[10]
    df_with_gap = df.drop(df.index[11])  # remove the very next business day
    result = process_price_data(df_with_gap)
    # A single ordinary weekend gap must not appear in possible_gaps.
    assert result.possible_gaps == [], f"False positive on an ordinary gap: {result.possible_gaps}"


def test_genuine_multi_day_gap_is_flagged(ohlcv_factory):
    df = ohlcv_factory(n=50, start_date="2024-01-02")
    # Remove 6 consecutive business days in the middle — a real gap, not a holiday weekend.
    df_with_gap = pd.concat([df.iloc[:20], df.iloc[26:]])
    result = process_price_data(df_with_gap)
    assert len(result.possible_gaps) == 1


def test_never_raises_on_missing_ohlcv_columns():
    malformed = pd.DataFrame({"Close": [100.0, 101.0, 99.0]}, index=pd.bdate_range("2024-01-01", periods=3))
    result = process_price_data(malformed)  # must not raise
    assert result.warnings  # missing-column warning recorded
    assert len(result.df) == 3  # OHLC validation skipped, but rows not dropped
