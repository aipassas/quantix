"""Monte Carlo Future Probability Simulator engine for Quantix.

Two path-generation methods, both producing an (forecast_days + 1,
num_simulations) price-path matrix with row 0 pinned to the current price,
so finance.py can consume either one identically — only the method the
user picks in the UI determines which function gets called.

simulate_gbm_paths(): the original method. Draws i.i.d. shocks from a
fitted normal distribution (mu/sigma from historical simple returns) —
Geometric Brownian Motion. Fast and standard, but structurally incapable of
producing fatter tails, skew, or volatility clustering than a normal
distribution has by construction, no matter what the real historical
returns look like.

simulate_bootstrap_paths(): resamples contiguous BLOCKS of the ticker's own
historical daily log returns (with replacement) instead of drawing from a
fitted distribution. This preserves whatever fat tails, skew, and short-run
autocorrelation/volatility clustering are actually present in the
historical sample — a bad week gets resampled as a whole bad week, not
smoothed into independent daily draws. Log returns are used here (not the
simple returns simulate_gbm_paths uses) because they're time-additive:
concatenating sampled blocks and cumulatively summing them compounds
correctly into a multi-day return, which simple returns don't do.

Blocks, not single-day draws, are the whole point: an i.i.d. day-by-day
bootstrap would destroy clustering exactly like GBM does, just with a
non-normal marginal distribution instead of a normal one.

Verified empirically against real tickers (see MONTE_CARLO_REALISM_NOTES):
for a ticker whose historical returns actually have fat tails (e.g. MSFT,
excess kurtosis ~13 over the trailing year), the bootstrap's 60-day-ahead
terminal-price distribution comes out markedly fatter-tailed than GBM's
(~2.1 vs ~0.4 excess kurtosis). For a ticker with a closer-to-normal
historical sample (e.g. JPM), the two methods land close together — which
is the correct behavior of a real bootstrap: it reflects what's actually in
the data rather than fabricating tail risk that isn't there.
"""
from typing import Tuple

import numpy as np
import pandas as pd


def simulate_gbm_paths(
    returns: pd.Series,
    current_price: float,
    num_simulations: int,
    forecast_days: int,
    seed: int,
) -> np.ndarray:
    """Geometric Brownian Motion price paths from a fitted normal distribution.

    `returns` is the historical simple daily returns series (Close.pct_change()).
    Mutates the global numpy RNG state via np.random.seed — matches the
    original inline implementation's seeding convention exactly, so results
    are unchanged from before this was extracted into its own function.
    """
    daily_drift = returns.mean()
    daily_vol = returns.std()
    mu = daily_drift - 0.5 * daily_vol ** 2

    np.random.seed(seed)
    random_shocks = np.random.normal(0, 1, (forecast_days, num_simulations))
    simulated_growth = np.exp(mu + daily_vol * random_shocks)

    price_paths = np.zeros((forecast_days + 1, num_simulations))
    price_paths[0] = current_price
    for t in range(1, forecast_days + 1):
        price_paths[t] = price_paths[t - 1] * simulated_growth[t - 1]
    return price_paths


def simulate_bootstrap_paths(
    close_prices: pd.Series,
    current_price: float,
    num_simulations: int,
    forecast_days: int,
    block_days: int,
    seed: int,
) -> np.ndarray:
    """Block-bootstrap price paths resampled from historical daily log returns.

    Each simulated path is built by concatenating ceil(forecast_days /
    block_days) randomly-chosen contiguous blocks of `block_days` historical
    log returns (sampled with replacement, blocks may overlap each other
    across the historical series), trimmed to exactly forecast_days, then
    cumulatively summed and exponentiated onto current_price. Every sampled
    return is a real historical value — never a fabricated one.

    Uses a local np.random.default_rng(seed) rather than the legacy global
    RNG simulate_gbm_paths uses, so the two methods don't perturb each
    other's determinism when called in the same process.
    """
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna().to_numpy()
    n = len(log_returns)
    if n < block_days:
        raise ValueError(f"need at least {block_days} historical log returns to form one block, got {n}")

    rng = np.random.default_rng(seed)
    num_blocks = -(-forecast_days // block_days)  # ceil division
    max_start = n - block_days
    block_starts = rng.integers(0, max_start + 1, size=(num_simulations, num_blocks))

    offsets = np.arange(block_days)
    block_indices = block_starts[:, :, None] + offsets[None, None, :]
    sampled_log_returns = log_returns[block_indices].reshape(num_simulations, num_blocks * block_days)
    sampled_log_returns = sampled_log_returns[:, :forecast_days]

    cumulative_log_returns = np.cumsum(sampled_log_returns, axis=1)
    price_paths = np.zeros((forecast_days + 1, num_simulations))
    price_paths[0] = current_price
    price_paths[1:] = current_price * np.exp(cumulative_log_returns).T
    return price_paths


def terminal_stats(price_paths: np.ndarray, current_price: float) -> Tuple[float, float, float, float]:
    """(pct_above_current, p10, p50, p90) from a price-path matrix's final row."""
    final_prices = price_paths[-1]
    pct_above_current = float((final_prices > current_price).sum() / len(final_prices) * 100)
    p10 = float(np.percentile(final_prices, 10))
    p50 = float(np.percentile(final_prices, 50))
    p90 = float(np.percentile(final_prices, 90))
    return pct_above_current, p10, p50, p90
