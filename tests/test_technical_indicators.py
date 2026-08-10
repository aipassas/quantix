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
    compute_stochastic, detect_stochastic_crossovers,
    compute_anchored_vwap,
    compute_adx,
    compute_ichimoku,
    compute_obv,
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


# --- Stochastic Oscillator ---

def test_stochastic_matches_pandas_ta(clean_ohlcv):
    """The 'Slow Stochastic' (pre-smoothed %K) — TradingView's/pandas_ta's
    actual default, confirmed by reading pandas_ta's stoch() source rather
    than assuming the simpler textbook Fast Stochastic formula."""
    result = compute_stochastic(clean_ohlcv, k_period=14, d_period=3, smooth_k=3)
    reference = ta.stoch(clean_ohlcv["High"], clean_ohlcv["Low"], clean_ohlcv["Close"], k=14, d=3, smooth_k=3)
    k_col = next(c for c in reference.columns if c.startswith("STOCHk"))
    d_col = next(c for c in reference.columns if c.startswith("STOCHd"))
    assert (result["Stoch_K"] - reference[k_col]).abs().dropna().max() < 1e-8
    assert (result["Stoch_D"] - reference[d_col]).abs().dropna().max() < 1e-8


def test_stochastic_crossovers_fire_once_per_crossing(clean_ohlcv):
    result = compute_stochastic(clean_ohlcv)
    signals = detect_stochastic_crossovers(result)
    valid = result[["Stoch_K", "Stoch_D"]].dropna()
    above = (valid["Stoch_K"] > valid["Stoch_D"]).astype(int)
    expected = int(above.diff().abs().dropna().sum())
    assert len(signals) == expected


# --- Anchored VWAP ---

def test_anchored_vwap_matches_manual_calculation():
    df = pd.DataFrame({
        "High": [102, 104, 103], "Low": [98, 100, 99], "Close": [100, 103, 101], "Volume": [1000, 2000, 1500],
    }, index=pd.bdate_range("2024-01-01", periods=3))
    vwap = compute_anchored_vwap(df)
    tp = [(102 + 98 + 100) / 3, (104 + 100 + 103) / 3, (103 + 99 + 101) / 3]
    cum_pv = np.cumsum([tp[0] * 1000, tp[1] * 2000, tp[2] * 1500])
    cum_vol = np.cumsum([1000, 2000, 1500])
    expected = cum_pv / cum_vol
    assert vwap.values == pytest.approx(expected)


def test_anchored_vwap_nan_before_anchor():
    df = pd.DataFrame({
        "High": [102, 104, 103], "Low": [98, 100, 99], "Close": [100, 103, 101], "Volume": [1000, 2000, 1500],
    }, index=pd.bdate_range("2024-01-01", periods=3))
    vwap = compute_anchored_vwap(df, anchor_date=df.index[1])
    assert pd.isna(vwap.iloc[0])
    assert vwap.iloc[1:].notna().all()


# --- ADX ---

def test_adx_matches_pandas_ta(clean_ohlcv):
    """Regression test for a real seeding bug this task caught: ADX's
    internal ATR uses prenan=True (unlike compute_atr()'s own prenan=False
    default) AND a positional (not value-count) SMA seed window — getting
    either wrong shifts the seed by one bar and produces a decaying
    divergence in ADX itself, even though Plus_DI/Minus_DI still match
    (see _atr_for_adx()'s docstring)."""
    result = compute_adx(clean_ohlcv, period=14)
    reference = ta.adx(clean_ohlcv["High"], clean_ohlcv["Low"], clean_ohlcv["Close"], length=14)
    adx_col = next(c for c in reference.columns if c.startswith("ADX_"))
    dmp_col = next(c for c in reference.columns if c.startswith("DMP_"))
    dmn_col = next(c for c in reference.columns if c.startswith("DMN_"))
    assert (result["ADX"] - reference[adx_col]).abs().dropna().max() < 1e-6
    assert (result["Plus_DI"] - reference[dmp_col]).abs().dropna().max() < 1e-6
    assert (result["Minus_DI"] - reference[dmn_col]).abs().dropna().max() < 1e-6


# --- Ichimoku Cloud ---

def test_ichimoku_matches_pandas_ta(clean_ohlcv):
    """Cross-validates both VALUES and the forward/backward shift
    direction — a values-only check can pass while the cloud is plotted at
    the wrong x-position, an easy off-by-one this test guards against."""
    result = compute_ichimoku(clean_ohlcv, tenkan_period=9, kijun_period=26, senkou_b_period=52)
    hist_ref, fwd_ref = ta.ichimoku(clean_ohlcv["High"], clean_ohlcv["Low"], clean_ohlcv["Close"], tenkan=9, kijun=26, senkou=52)

    tenkan_col = next(c for c in hist_ref.columns if c.startswith("ITS_"))
    kijun_col = next(c for c in hist_ref.columns if c.startswith("IKS_"))
    senkou_a_col = next(c for c in hist_ref.columns if c.startswith("ISA_"))
    senkou_b_col = next(c for c in hist_ref.columns if c.startswith("ISB_"))
    chikou_col = next(c for c in hist_ref.columns if c.startswith("ICS_"))

    assert (result.historical["Tenkan"] - hist_ref[tenkan_col]).abs().dropna().max() < 1e-8
    assert (result.historical["Kijun"] - hist_ref[kijun_col]).abs().dropna().max() < 1e-8
    assert (result.historical["SenkouA"] - hist_ref[senkou_a_col]).abs().dropna().max() < 1e-8
    assert (result.historical["SenkouB"] - hist_ref[senkou_b_col]).abs().dropna().max() < 1e-8
    assert (result.historical["Chikou"] - hist_ref[chikou_col]).abs().dropna().max() < 1e-8

    # Forward-looking cloud: index alignment (the shift-direction check) and values.
    fwd_a_col = next(c for c in fwd_ref.columns if c.startswith("ISA_"))
    fwd_b_col = next(c for c in fwd_ref.columns if c.startswith("ISB_"))
    assert (result.forward.index == fwd_ref.index).all()
    assert (result.forward["SenkouA"] - fwd_ref[fwd_a_col]).abs().dropna().max() < 1e-8
    assert (result.forward["SenkouB"] - fwd_ref[fwd_b_col]).abs().dropna().max() < 1e-8


# --- On-Balance Volume ---

def test_obv_matches_pandas_ta(clean_ohlcv):
    engine_obv = compute_obv(clean_ohlcv)
    reference_obv = ta.obv(clean_ohlcv["Close"], clean_ohlcv["Volume"])
    diff = (engine_obv - reference_obv).abs().dropna()
    assert diff.max() < 1e-6


def test_obv_first_bar_is_nan(clean_ohlcv):
    """The first bar has no prior Close to compare against, so its
    direction is genuinely undefined — matches pandas_ta's own behavior
    (confirmed empirically, not an explicit special case in either
    implementation)."""
    engine_obv = compute_obv(clean_ohlcv)
    assert pd.isna(engine_obv.iloc[0])
