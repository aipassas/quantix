"""Tests for monte_carlo.py — the GBM and block-bootstrap path simulators.

Synthetic OHLCV data for the deterministic mechanics/regression tests; a
live check against real MSFT data (a ticker with genuine historical fat
tails) for the acceptance criterion that bootstrap can produce measurably
higher kurtosis than GBM on the same historical sample.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kurtosis

from monte_carlo import simulate_bootstrap_paths, simulate_gbm_paths, terminal_stats


def test_gbm_paths_start_at_current_price():
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 200))
    paths = simulate_gbm_paths(returns, current_price=150.0, num_simulations=100, forecast_days=60, seed=42)
    assert paths.shape == (61, 100)
    assert np.all(paths[0] == 150.0)
    assert np.all(paths > 0)


def test_gbm_paths_deterministic_for_same_seed():
    returns = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 200))
    p1 = simulate_gbm_paths(returns, 150.0, 100, 60, seed=42)
    p2 = simulate_gbm_paths(returns, 150.0, 100, 60, seed=42)
    assert np.array_equal(p1, p2)


def test_gbm_paths_matches_hand_computed_first_step():
    """One simulation, one step: Price_1 = Price_0 * exp(mu + sigma*Z),
    hand-reconstructed from the same seeded normal draw."""
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, 0.005] * 10)
    current_price = 100.0
    paths = simulate_gbm_paths(returns, current_price, num_simulations=1, forecast_days=1, seed=7)

    np.random.seed(7)
    z = np.random.normal(0, 1, (1, 1))[0, 0]
    mu = returns.mean() - 0.5 * returns.std() ** 2
    expected_price_1 = current_price * np.exp(mu + returns.std() * z)

    assert paths[1, 0] == pytest.approx(expected_price_1)


def test_bootstrap_paths_start_at_current_price():
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0005, 0.015, 200))))
    paths = simulate_bootstrap_paths(close, current_price=close.iloc[-1], num_simulations=100,
                                      forecast_days=60, block_days=5, seed=42)
    assert paths.shape == (61, 100)
    assert np.all(paths[0] == close.iloc[-1])
    assert np.all(paths > 0)


def test_bootstrap_paths_deterministic_for_same_seed():
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0005, 0.015, 200))))
    p1 = simulate_bootstrap_paths(close, close.iloc[-1], 100, 60, 5, seed=42)
    p2 = simulate_bootstrap_paths(close, close.iloc[-1], 100, 60, 5, seed=42)
    assert np.array_equal(p1, p2)


def test_bootstrap_every_sampled_return_is_a_real_historical_value():
    """Never fabricates a return — every day-over-day step in every
    simulated path must equal some actual historical log return."""
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0, 0.02, 50))))
    historical_log_returns = np.log(close / close.shift(1)).dropna().to_numpy()

    paths = simulate_bootstrap_paths(close, close.iloc[-1], num_simulations=20, forecast_days=30,
                                      block_days=5, seed=9)
    daily_log_steps = np.log(paths[1:] / paths[:-1])  # (30, 20)

    for value in daily_log_steps.flatten():
        assert np.any(np.isclose(historical_log_returns, value, atol=1e-9))


def test_bootstrap_uses_contiguous_blocks_not_iid_days():
    """A block boundary aligns sampled steps with a real historical RUN of
    `block_days` consecutive returns, not independently shuffled days —
    check the first block_days steps of one path equal SOME real contiguous
    historical window, in order."""
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0, 0.02, 50))))
    historical_log_returns = np.log(close / close.shift(1)).dropna().to_numpy()

    paths = simulate_bootstrap_paths(close, close.iloc[-1], num_simulations=1, forecast_days=5,
                                      block_days=5, seed=11)
    sampled_block = np.log(paths[1:, 0] / paths[:-1, 0])

    n = len(historical_log_returns)
    found = any(
        np.allclose(historical_log_returns[start:start + 5], sampled_block, atol=1e-9)
        for start in range(n - 5 + 1)
    )
    assert found


def test_bootstrap_raises_when_history_shorter_than_one_block():
    close = pd.Series([100.0, 101.0, 99.0])  # 2 log returns, block_days=5
    with pytest.raises(ValueError):
        simulate_bootstrap_paths(close, 99.0, num_simulations=10, forecast_days=10, block_days=5, seed=1)


def test_terminal_stats_matches_hand_computed_percentiles():
    paths = np.array([
        [100.0, 100.0, 100.0],
        [90.0, 100.0, 130.0],
    ])
    pct_above, p10, p50, p90 = terminal_stats(paths, current_price=100.0)
    assert pct_above == pytest.approx(1 / 3 * 100)
    assert p50 == pytest.approx(np.percentile([90.0, 100.0, 130.0], 50))


def test_bootstrap_can_produce_measurably_fatter_tails_than_gbm_on_engineered_fat_tail_history():
    """Deterministic proof of the mechanism: a historical sample with one
    embedded extreme-crash block gives the block bootstrap a real chance to
    replay that crash in some simulated paths, producing fatter terminal-price
    tails than GBM's normal-shock model — which structurally averages the
    crash into its fitted sigma and can never replay it as a discrete event.
    """
    rng = np.random.default_rng(5)
    calm_returns = rng.normal(0.0003, 0.008, 250)
    crash_block = np.array([-0.10, -0.08, 0.03, -0.05, 0.02])  # one brutal 5-day historical crash
    log_returns_history = np.concatenate([calm_returns, crash_block])
    close = pd.Series(100 * np.exp(np.cumsum(log_returns_history)))
    simple_returns = close.pct_change().dropna()

    current_price = close.iloc[-1]
    num_simulations, forecast_days, block_days, seed = 5000, 60, 5, 123

    gbm_paths = simulate_gbm_paths(simple_returns, current_price, num_simulations, forecast_days, seed)
    bootstrap_paths = simulate_bootstrap_paths(close, current_price, num_simulations, forecast_days, block_days, seed)

    gbm_kurtosis = kurtosis(gbm_paths[-1], fisher=True)
    bootstrap_kurtosis = kurtosis(bootstrap_paths[-1], fisher=True)

    assert bootstrap_kurtosis > gbm_kurtosis


@pytest.mark.live
def test_bootstrap_reflects_real_historical_fat_tails_msft():
    """Acceptance-style live check: MSFT's trailing-year daily log returns
    have real, substantial excess kurtosis (an actual historical fat-tail
    year), so the bootstrap's 60-day terminal distribution should come out
    measurably fatter-tailed than GBM's on the identical historical sample —
    GBM is structurally incapable of reflecting that no matter the input.
    """
    import datetime as dt

    from data_loader import load_ticker_bundle
    from price_processing import process_price_data

    end = dt.date.today()
    start = end - dt.timedelta(days=400)
    bundle = load_ticker_bundle("MSFT", start, end, deep=True)
    df = process_price_data(bundle.price_history, ticker="MSFT").df
    simple_returns = df["Close"].pct_change().dropna()

    current_price = df["Close"].iloc[-1]
    seed = 12345

    gbm_paths = simulate_gbm_paths(simple_returns, current_price, 2000, 60, seed)
    bootstrap_paths = simulate_bootstrap_paths(df["Close"], current_price, 2000, 60, 5, seed)

    gbm_kurtosis = kurtosis(gbm_paths[-1], fisher=True)
    bootstrap_kurtosis = kurtosis(bootstrap_paths[-1], fisher=True)

    assert bootstrap_kurtosis > gbm_kurtosis
