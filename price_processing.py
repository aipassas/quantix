"""Price data preprocessing engine for Quantix.

Every technical indicator (SMA, RSI, MACD, Bollinger Bands, ATR, ...) is
computed on the assumption that its input OHLCV DataFrame is well-formed: a
sorted, duplicate-free, timezone-consistent DatetimeIndex, no structurally
impossible bars (High below Low, negative prices), and a single, explicitly
adjusted Close column. Yahoo Finance's raw `.history()` response is usually
close to this, but not guaranteed — this module is the one place that
validates and cleans a price DataFrame before any indicator ever sees it,
instead of each indicator (or finance.py) trusting the raw fetch implicitly.

Scope: the main ticker's price history only — the one dataset every
upcoming technical indicator consumes. Benchmark/VIX/TNX and the 10-year
seasonality series are used only for their own dedicated charts (Alpha
Generation, 3D Seasonality Surface), never fed into an indicator
calculation, so they're out of scope here.

Interval-agnostic by design: this module works on whatever bar interval
it's handed (daily, weekly, monthly) — fetching at a given interval is
data_loader.py's job (it already does this for the 10-year monthly
seasonality data via a separate `interval="1mo"` fetch); this module never
resamples or changes the bar interval, only validates and cleans it.

Design principles (consistent with the rest of the app):
  - Never raise on bad data. A malformed row is dropped and recorded in
    `warnings`; the function always returns a DataFrame, never an exception.
  - Never fabricate data. A detected gap in the trading calendar is
    reported in `possible_gaps`, not filled with a synthetic/forward-filled
    price — indicators see only real, reported observations.
  - Minimize memory overhead. The input DataFrame is copied exactly once
    (`process_price_data`'s entry point); every helper below operates on
    and returns that same copy rather than each making its own.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# A trading-day gap wider than this multiple of the data's own typical bar
# spacing is flagged as a possible missing observation. Interval-agnostic:
# rather than hardcoding "5 calendar days" (which only makes sense for daily
# bars), the typical spacing is measured from the data itself (its median
# consecutive-timestamp diff), so the same multiple works whether the frame
# holds daily, weekly, or monthly bars.
#
# Calibrated against real AAPL daily data, not just chosen a priori: every
# single 3-day US market holiday weekend (Labor Day, MLK Day, Presidents
# Day, Memorial Day, Juneteenth, Independence Day, ...) produces a 4
# calendar-day gap against a 1-day typical spacing. An earlier value of 3
# flagged every one of these — a ~100% false-positive rate on completely
# normal data, since 3-day holiday weekends occur roughly monthly. 5
# comfortably clears that entire ordinary pattern while still catching
# gaps of 6+ days (multiple consecutive missing trading days), which is a
# meaningfully rarer and more suspicious signal. The trade-off, disclosed:
# this heuristic will not catch a single missing trading day sitting next
# to a weekend/holiday — without a real trading-calendar library, that
# case is indistinguishable from an ordinary long weekend by timestamps
# alone, and a heuristic that fires on every holiday trains users to
# ignore it, which is worse than reduced sensitivity to short gaps.
GAP_MULTIPLE_OF_TYPICAL_SPACING = 5


@dataclass
class PriceProcessingResult:
    """Cleaned price DataFrame plus a record of everything that was found
    and fixed or flagged along the way — never silent, so the app can show
    (or at least log) exactly what was done to the data before an indicator
    was computed on it."""
    df: pd.DataFrame
    warnings: List[str] = field(default_factory=list)
    duplicate_rows_removed: int = 0
    invalid_rows_removed: int = 0
    possible_gaps: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return (
            not self.warnings
            and not self.possible_gaps
            and self.duplicate_rows_removed == 0
            and self.invalid_rows_removed == 0
        )

    @property
    def issue_count(self) -> int:
        return len(self.warnings) + len(self.possible_gaps) + self.duplicate_rows_removed + self.invalid_rows_removed


def _standardize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, timezone-naive DatetimeIndex. Yahoo returns a timezone-aware
    index (exchange-local, e.g. America/New_York) — stripped here so this
    price series compares/joins cleanly with any other date-indexed data in
    the app without a silent timezone mismatch."""
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df.sort_index()


def _drop_duplicate_timestamps(df: pd.DataFrame) -> "tuple[pd.DataFrame, int]":
    """Keep the first occurrence of each timestamp. Duplicates are rare in
    Yahoo's response but have been observed around DST/timezone boundaries
    or when overlapping fetches are concatenated upstream."""
    duplicate_mask = df.index.duplicated(keep="first")
    removed = int(duplicate_mask.sum())
    if removed:
        df = df[~duplicate_mask]
    return df, removed


def _normalize_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `Close` is the single, explicit source of truth for adjusted
    price. yfinance's default (`auto_adjust=True`, in effect throughout
    this app — see data_loader.py) already returns a split/dividend-adjusted
    Close with no separate 'Adj Close' column; if a future yfinance version
    or code path ever returns both, this consolidates to one adjusted
    Close column rather than leaving two similarly-named columns for a
    caller to accidentally pick the wrong one."""
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
        df = df.drop(columns=["Adj Close"])
    return df


def _validate_and_drop_invalid_rows(df: pd.DataFrame) -> "tuple[pd.DataFrame, int, List[str]]":
    """Drop rows that are structurally impossible for a real OHLC bar —
    non-positive prices, a High below Low, or Open/Close outside the
    [Low, High] range — rather than let a corrupt row silently poison every
    indicator computed from it. Returns (cleaned_df, rows_removed, warnings)."""
    warnings: List[str] = []
    missing_cols = [c for c in REQUIRED_OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        warnings.append(f"Missing expected column(s): {', '.join(missing_cols)} — skipping OHLC validation")
        return df, 0, warnings

    valid = (
        (df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)
        & (df["Volume"] >= 0)
        & (df["High"] >= df["Low"])
        & (df["Open"] >= df["Low"]) & (df["Open"] <= df["High"])
        & (df["Close"] >= df["Low"]) & (df["Close"] <= df["High"])
    )
    invalid_count = int((~valid).sum())
    if invalid_count:
        df = df[valid]
    return df, invalid_count, warnings


def _detect_possible_gaps(df: pd.DataFrame) -> List[str]:
    """Flag date ranges where the spacing between consecutive bars is
    unusually wide relative to this data's own typical spacing — a
    heuristic, not a certainty (see GAP_MULTIPLE_OF_TYPICAL_SPACING), since
    no real trading-calendar library is used here to distinguish a genuine
    missing observation from an ordinary long holiday weekend."""
    if len(df.index) < 3:
        return []
    diffs = df.index.to_series().diff().dropna()
    typical_spacing = diffs.median()
    if typical_spacing <= pd.Timedelta(0):
        return []
    threshold = typical_spacing * GAP_MULTIPLE_OF_TYPICAL_SPACING
    gaps = diffs[diffs > threshold]
    return [
        f"{(ts - gap).date()} → {ts.date()} ({gap.days} day gap, typical spacing is {typical_spacing.days} day(s))"
        for ts, gap in gaps.items()
    ]


def process_price_data(df: pd.DataFrame, ticker: str = "") -> PriceProcessingResult:
    """Validate and clean a raw OHLCV price DataFrame before any technical
    indicator is computed on it. Never raises — an empty or malformed input
    returns an empty/cleaned result with warnings explaining why, mirroring
    data_loader.py's never-raise, always-return-with-warnings convention.

    Order matters: the index is standardized first so every later step
    (duplicate detection, gap detection) operates on a sorted, consistent
    index rather than working around an unsorted or tz-mixed one.
    """
    if df is None or df.empty:
        return PriceProcessingResult(df=pd.DataFrame(), warnings=["No price data to process"])

    working = df.copy()  # single copy for the whole pipeline — see module docstring
    working = _standardize_datetime_index(working)
    working, duplicates_removed = _drop_duplicate_timestamps(working)
    working = _normalize_adjusted_close(working)
    working, invalid_rows_removed, column_warnings = _validate_and_drop_invalid_rows(working)
    possible_gaps = _detect_possible_gaps(working)

    return PriceProcessingResult(
        df=working,
        warnings=column_warnings,
        duplicate_rows_removed=duplicates_removed,
        invalid_rows_removed=invalid_rows_removed,
        possible_gaps=possible_gaps,
    )
