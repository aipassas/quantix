"""Opt-in real-market-data sanity checks — never run in CI or by default
(see pytest.ini's `addopts = -m "not live"`). Run explicitly with:

    pytest -m live

These catch the kind of real-world-only issue synthetic fixtures can miss
(a genuine NaN/outlier pattern in actual market data), at the cost of a
live network call to Yahoo Finance — the reason they're opt-in rather than
part of the default, fast, deterministic suite.
"""
import numpy as np
import pandas_ta as ta
import pytest

from price_processing import process_price_data
from technical_indicators import compute_rsi, compute_macd, compute_atr
from risk_analytics import compute_annualized_volatility, compute_sharpe_ratio
from fundamental_analysis import FundamentalAnalysisEngine
from data_loader import load_ticker_bundle
from financial_standardization import standardize_financials


def _real_aapl_prices():
    import datetime
    end = datetime.date.today()
    start = end - datetime.timedelta(days=400)
    bundle = load_ticker_bundle("AAPL", start, end, deep=True)
    return process_price_data(bundle.price_history, ticker="AAPL").df


@pytest.mark.live
def test_rsi_matches_pandas_ta_on_real_aapl_data():
    df = _real_aapl_prices()
    engine_rsi = compute_rsi(df, period=14)
    reference_rsi = ta.rsi(df["Close"], length=14)
    diff = (engine_rsi - reference_rsi).abs().dropna()
    assert diff.max() < 1e-6


@pytest.mark.live
def test_macd_matches_pandas_ta_on_real_aapl_data():
    df = _real_aapl_prices()
    result = compute_macd(df)
    reference = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    diff = (result["MACD_Line"] - reference["MACD_12_26_9"]).abs().dropna()
    assert diff.max() < 1e-6


@pytest.mark.live
def test_atr_matches_pandas_ta_on_real_aapl_data():
    df = _real_aapl_prices()
    engine_atr = compute_atr(df, period=14)
    reference_atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    diff = (engine_atr - reference_atr).abs().dropna()
    assert diff.max() < 1e-6


@pytest.mark.live
def test_sharpe_ratio_is_sane_on_real_aapl_data():
    df = _real_aapl_prices()
    vol = compute_annualized_volatility(df)
    sharpe = compute_sharpe_ratio(df, risk_free_rate=0.04)
    assert vol is not None and 0.05 < vol < 1.5  # a real stock's annualized vol is never ~0% or >150%
    assert sharpe is not None and -5 < sharpe < 5  # sanity bounds, not a specific expected value


@pytest.mark.live
def test_altman_z_is_safe_zone_for_real_aapl_and_msft():
    import datetime
    end = datetime.date.today()
    start = end - datetime.timedelta(days=400)
    for ticker in ("AAPL", "MSFT"):
        bundle = load_ticker_bundle(ticker, start, end, deep=True)
        standardized = standardize_financials(bundle)
        engine = FundamentalAnalysisEngine(standardized, raw_info=bundle.info or {})
        z, verdict, missing = engine.altman_z_score()
        assert z is not None, f"{ticker} should have a computable Altman Z-Score"
        assert z > 2.99, f"{ticker} (large-cap, low-leverage tech) expected in the Safe Zone, got Z={z}"
