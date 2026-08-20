"""Shared fixtures for the Quantix test suite.

Synthetic OHLCV data only — no network calls here. Tests that need real
market data live behind the `live` marker (see pytest.ini) and build their
own yfinance fetch inline, so it's obvious at a glance which tests are
network-dependent without hunting through a shared fixture for it.

Also isolates the suite from the developer's real .streamlit/secrets.toml
— see isolate_secrets below.
"""
import os

import numpy as np
import pandas as pd
import pytest
import streamlit as st


@pytest.fixture(autouse=True)
def isolate_secrets(monkeypatch):
    """Hide the real secrets.toml from every test.

    REGRESSION GUARD FOR A CREDENTIAL LEAK. Modules that read
    configuration check st.secrets BEFORE falling back to environment
    variables (see email_report._read_secret). On a machine with no
    secrets.toml that ordering is invisible, so the SMTP tests — which
    set QUANTIX_SMTP_* env vars and assert on the result — passed
    everywhere. The moment a real [smtp] section existed, those tests
    read the developer's LIVE credentials instead of their fixtures,
    failed on the mismatch, and pytest printed the real App Password
    into the failure output. That is a credential in terminal
    scrollback and in any CI log.

    Autouse and suite-wide rather than local to the SMTP tests: the same
    precedence applies to [auth] and to anything added later, so the
    isolation belongs at the boundary, not at each call site.
    """
    class _EmptySecrets(dict):
        def get(self, key, default=None):
            return default

        def __getitem__(self, key):
            raise KeyError(key)

    monkeypatch.setattr(st, "secrets", _EmptySecrets(), raising=False)
    # Real QUANTIX_* vars in the developer's shell would leak the same way.
    for name in list(os.environ):
        if name.startswith("QUANTIX_"):
            monkeypatch.delenv(name, raising=False)
    yield


def _make_ohlcv(n=300, seed=7, start_price=100.0, daily_vol=0.015, drift=0.0004, start_date="2023-01-02"):
    """A structurally valid synthetic OHLCV DataFrame: High >= Low, Open/Close
    within [Low, High], positive prices — passes price_processing.py's own
    validity checks, since indicator tests need clean input to isolate what
    they're actually testing rather than incidentally exercising cleaning too.
    """
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(drift, daily_vol, n)
    close = start_price * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range(start_date, periods=n)

    open_ = np.concatenate([[start_price], close[:-1]])
    intraday_range = np.abs(rng.normal(0, daily_vol * 0.5, n)) * close
    high = np.maximum(open_, close) + intraday_range
    low = np.clip(np.minimum(open_, close) - intraday_range, 0.01, None)
    volume = rng.integers(1_000_000, 5_000_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv_factory():
    """Factory fixture — call with kwargs (n, seed, start_price, daily_vol, drift, start_date)
    for a fresh synthetic OHLCV DataFrame with those parameters."""
    return _make_ohlcv


@pytest.fixture
def clean_ohlcv(ohlcv_factory):
    """A single default 300-bar synthetic OHLCV DataFrame, seed fixed for reproducibility."""
    return ohlcv_factory()


@pytest.fixture
def price_series(clean_ohlcv):
    """Just the Close price Series from `clean_ohlcv`, for functions that take
    a price Series directly (e.g. risk_analytics.compute_max_drawdown)."""
    return clean_ohlcv["Close"]
