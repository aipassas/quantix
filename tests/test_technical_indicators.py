"""Tests for technical_indicators.py.

Cross-validates against pandas_ta wherever a smoothing/seeding convention
matters (see TECHNICAL_ANALYSIS.md for why RSI/MACD/ATR each needed a
different, non-obvious seeding decision — these tests are the regression
guard for exactly those decisions).
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytest

from technical_indicators import (
    compute_sma_lines, detect_sma_crossovers,
    compute_rsi, interpret_rsi,
    compute_macd, detect_macd_crossovers,
    compute_bollinger_bands, detect_bollinger_breakouts,
    compute_atr, suggested_stop_loss,
)


# --- SMA ---

def test_sma_matches_manual_rolling_mean(clean_ohlcv):
    result = compute_sma_lines(clean_ohlcv, periods=[20])
    manual = clean_ohlcv["Close"].rolling(window=20, min_periods=20).mean()
    pd.testing.assert_series_equal(result["SMA_20"], manual, check_names=False)


def test_sma_warmup_period_is_nan_not_backfilled(clean_ohlcv):
    result = compute_sma_lines(clean_ohlcv, periods=[20])
    assert result["SMA_20"].iloc[:19].isna().all()
    assert result["SMA_20"].iloc[19:].notna().all()


def test_sma_crossovers_fire_once_per_crossing(clean_ohlcv):
    df = compute_sma_lines(clean_ohlcv, periods=[20])
    signals = detect_sma_crossovers(df, period=20)
    # Reconstruct the crossing count independently: count sign changes of (Close - SMA).
    above = (df["Close"] > df[f"SMA_20"]).astype(int)
    expected_crossings = int(above.diff().abs().sum())
    assert len(signals) == expected_crossings


# --- RSI ---

def test_rsi_matches_pandas_ta(clean_ohlcv):
    engine_rsi = compute_rsi(clean_ohlcv, period=14)
    reference_rsi = ta.rsi(clean_ohlcv["Close"], length=14)
    diff = (engine_rsi - reference_rsi).abs().dropna()
    assert diff.max() < 1e-8, f"RSI diverges from pandas_ta by {diff.max():.2e}"


def test_rsi_flat_market_is_50():
    flat = pd.DataFrame({"Close": [100.0] * 30}, index=pd.bdate_range("2024-01-01", periods=30))
    rsi = compute_rsi(flat, period=14)
    assert rsi.iloc[-1] == 50.0


def test_rsi_all_gains_is_100():
    rising = pd.DataFrame({"Close": np.linspace(100, 130, 30)}, index=pd.bdate_range("2024-01-01", periods=30))
    rsi = compute_rsi(rising, period=14)
    assert rsi.iloc[-1] == 100.0


def test_interpret_rsi_bands():
    assert interpret_rsi(80).label if hasattr(interpret_rsi(80), "label") else None
    assert interpret_rsi(None) is None


# --- MACD ---

def test_macd_matches_pandas_ta_exactly(clean_ohlcv):
    """Regression test for the exact seeding bug documented in
    TECHNICAL_ANALYSIS.md — an unseeded EMA diverged from pandas_ta by
    ~0.49 during warm-up; the SMA-seeded fix matches exactly."""
    result = compute_macd(clean_ohlcv)
    reference = ta.macd(clean_ohlcv["Close"], fast=12, slow=26, signal=9)

    line_diff = (result["MACD_Line"] - reference["MACD_12_26_9"]).abs().dropna()
    signal_diff = (result["MACD_Signal"] - reference["MACDs_12_26_9"]).abs().dropna()
    hist_diff = (result["MACD_Histogram"] - reference["MACDh_12_26_9"]).abs().dropna()

    assert line_diff.max() < 1e-8
    assert signal_diff.max() < 1e-8
    assert hist_diff.max() < 1e-8


def test_macd_crossovers_fire_once_per_crossing(clean_ohlcv):
    df = compute_macd(clean_ohlcv)
    signals = detect_macd_crossovers(df)
    above = (df["MACD_Line"] > df["MACD_Signal"]).astype(int)
    expected_crossings = int(above.diff().abs().dropna().sum())
    assert len(signals) == expected_crossings


# --- Bollinger Bands ---

def test_bollinger_matches_pandas_ta(clean_ohlcv):
    result = compute_bollinger_bands(clean_ohlcv, period=20, num_std=2.0)
    reference = ta.bbands(clean_ohlcv["Close"], length=20, std=2.0)
    # pandas_ta names bbands columns "BBU_{length}_{std}_{std}" — verified
    # against the installed version's actual output rather than assumed.
    upper_col = next(c for c in reference.columns if c.startswith("BBU_"))
    lower_col = next(c for c in reference.columns if c.startswith("BBL_"))

    upper_diff = (result["BB_Upper"] - reference[upper_col]).abs().dropna()
    lower_diff = (result["BB_Lower"] - reference[lower_col]).abs().dropna()
    assert upper_diff.max() < 1e-6
    assert lower_diff.max() < 1e-6


def test_bollinger_breakouts_are_close_based_not_wick_based(clean_ohlcv):
    df = compute_bollinger_bands(clean_ohlcv, period=20, num_std=2.0)
    breakouts = detect_bollinger_breakouts(df)
    # Every recorded breakout's Close must actually be outside the band on that day.
    for breakout in breakouts:
        row = df.loc[breakout.date]
        if breakout.kind in ("upper_breakout",):
            assert row["Close"] > row["BB_Upper"]
        elif breakout.kind in ("lower_breakout",):
            assert row["Close"] < row["BB_Lower"]


# --- ATR ---

def test_atr_matches_pandas_ta_exactly(clean_ohlcv):
    """Regression test for the ATR seeding convention — SMA-seeded True
    Range average, unlike RSI's unseeded smoothing despite both nominally
    using Wilder's alpha (see TECHNICAL_ANALYSIS.md §6)."""
    engine_atr = compute_atr(clean_ohlcv, period=14)
    reference_atr = ta.atr(clean_ohlcv["High"], clean_ohlcv["Low"], clean_ohlcv["Close"], length=14)
    diff = (engine_atr - reference_atr).abs().dropna()
    assert diff.max() < 1e-8


def test_suggested_stop_loss_formula():
    stop = suggested_stop_loss(current_price=100.0, current_atr=5.0, multiplier=2.0)
    assert stop == pytest.approx(90.0)


def test_suggested_stop_loss_none_on_missing_input():
    assert suggested_stop_loss(None, 5.0) is None
    assert suggested_stop_loss(100.0, None) is None
