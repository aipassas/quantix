"""Market structure for crypto: dominance, trend, and on-chain activity.

BITCOIN DOMINANCE COMES FROM A REPORTED TOTAL, not from summing a page.
CoinGecko's /global gives each coin's share of the whole market —
Bitcoin 59.14%, Ethereum 11.07% when probed. Reconstructing that from
the 250-coin universe this build loads would divide by a total that
excludes eighteen thousand other coins, inflating every share. The
reported figure is used and the universe is used only for what it is,
a ranked sample.

THE GOLDEN CROSS IS A REAL SIGNAL AND A WEAK ONE, and the panel says
both. SMA(50) crossing SMA(200) is the task's named indicator; it is
also a lagging one by construction, since a 200-day mean cannot respond
to anything until a hundred days have gone by. What matters more is
whether it can be EVALUATED at all: 200 daily bars is 200 calendar days
for crypto, so a coin listed six months ago has no 200-day average and
the cross is UNCHECKED rather than absent. Rendering an unevaluable
signal as "no cross" is an all-clear nobody performed — the same error
the bond module's SMA(200) rule was written to avoid.

SUPPORT AND RESISTANCE ARE DESCRIPTIVE. They are computed as the
extremes of the window and the round levels bracketing the current
price, and they are labelled as what they are: where the price has
been, not where it will stop. This build does not claim predictive
levels it cannot back.

ON-CHAIN TECHNICALS ARE BITCOIN-ONLY, for the reason crypto_data
records: blockchain.info indexes one chain. Where they are unavailable
the panel names the coin's chain rather than showing a Bitcoin figure
under another coin's name.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import crypto_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

FAST_SMA = 50
SLOW_SMA = 200

# A trailing average that includes the bar under test puts that bar into
# its own baseline. Volume windows here exclude the current bar for the
# same reason the ETF module's relative volume does.
VOLUME_WINDOW_DAYS = 30

UNCHECKED = "Unchecked"
FIRED = "Fired"
NOT_FIRED = "Not fired"


@dataclass(frozen=True)
class Dominance:
    symbol: str
    share_pct: Optional[float] = None
    rank: Optional[int] = None
    total_market_cap: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.share_pct is not None and not self.error


def dominance(symbol: str, market: "crypto_data.GlobalMarket") -> Dominance:
    """One coin's share of total crypto market cap."""
    if market is None or not market.ok:
        return Dominance(symbol, error=(
            getattr(market, "error", "") or "Market totals are unavailable."))
    share = market.dominance_of(symbol)
    if share is None:
        return Dominance(symbol, total_market_cap=market.total_market_cap,
                         error=("This coin is not among those the market "
                                "totals break out individually."))
    ranked = sorted(market.dominance.items(), key=lambda kv: -kv[1])
    key = crypto_data.normalise_symbol(symbol)
    rank = next((i + 1 for i, (k, _) in enumerate(ranked) if k == key), None)
    return Dominance(symbol, share, rank, market.total_market_cap)


def describe_dominance(reading: Dominance) -> str:
    if not reading.ok:
        return reading.error
    text = (f"{reading.share_pct:.2f}% of all crypto market "
            f"capitalisation")
    if reading.rank == 1:
        text += ", the largest share"
    elif reading.rank:
        text += f", the {_ordinal(reading.rank)} largest share"
    if reading.total_market_cap:
        text += f" of a ${reading.total_market_cap/1e12:,.2f}T market"
    return text + "."


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


# --- trend --------------------------------------------------------------------

@dataclass(frozen=True)
class MovingAverageCross:
    fast: Optional[float] = None
    slow: Optional[float] = None
    state: str = UNCHECKED       # Fired | Not fired | Unchecked
    golden: Optional[bool] = None
    bars_available: int = 0
    detail: str = ""

    @property
    def evaluable(self) -> bool:
        return self.state != UNCHECKED


def moving_average_cross(closes: Optional["pd.Series"],
                         fast: int = FAST_SMA,
                         slow: int = SLOW_SMA) -> MovingAverageCross:
    """The golden/death cross, or an explicit statement that it could not
    be checked.

    Indicator columns elsewhere in the app are conditional on the Chart
    Workspace's display checkboxes, so this recomputes its own averages
    rather than reading columns that exist only when a box is ticked.
    """
    series = pd.Series(closes).dropna() if closes is not None else pd.Series(dtype="float64")
    available = int(len(series))
    if available < slow:
        return MovingAverageCross(
            bars_available=available, state=UNCHECKED,
            detail=(f"A {slow}-day average needs {slow} daily bars and "
                    f"this history has {available}. The cross is "
                    f"unchecked, which is not the same as no cross."))
    fast_ma = float(series.rolling(fast).mean().iloc[-1])
    slow_ma = float(series.rolling(slow).mean().iloc[-1])
    golden = fast_ma > slow_ma
    return MovingAverageCross(
        fast=fast_ma, slow=slow_ma, state=FIRED if golden else NOT_FIRED,
        golden=golden, bars_available=available,
        detail=(
            f"The {fast}-day average ({fast_ma:,.2f}) is "
            f"{'above' if golden else 'below'} the {slow}-day "
            f"({slow_ma:,.2f}) — a {'golden' if golden else 'death'} "
            f"cross configuration. Both averages lag by construction."))


@dataclass(frozen=True)
class RangePosition:
    low: Optional[float] = None
    high: Optional[float] = None
    price: Optional[float] = None
    position_pct: Optional[float] = None      # 0 at the low, 100 at the high
    window_days: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.position_pct is not None and not self.error


def range_position(closes: Optional["pd.Series"],
                   window_days: int = 365) -> RangePosition:
    """Where the price sits inside its own recent range.

    Descriptive: these are the extremes the price actually reached, not
    levels claimed to hold. A window longer than the history available
    uses what there is and reports the shorter span.
    """
    if closes is None:
        return RangePosition(error="No price history.")
    series = pd.Series(closes).dropna()
    if len(series) < 2:
        return RangePosition(error="Price history has too few points.")
    window = series.iloc[-window_days:] if window_days else series
    low, high = float(window.min()), float(window.max())
    price = float(window.iloc[-1])
    if high <= low:
        return RangePosition(low, high, price, window_days=len(window),
                             error="The price did not move over this window.")
    return RangePosition(low, high, price,
                         100.0 * (price - low) / (high - low), len(window))


PIVOT_WINDOW_BARS = 10


def pivots(series: "pd.Series", window: int = PIVOT_WINDOW_BARS
           ) -> Tuple[List[float], List[float]]:
    """Local highs and lows: bars that are the extreme of a centred window.

    A pivot needs bars on BOTH sides, so the last `window` bars can never
    be pivots — by construction, not by omission. That is the honest
    behaviour: a high that has not yet been tested from the right is not
    a level anything has bounced off.
    """
    values = series.to_numpy(dtype="float64")
    highs: List[float] = []
    lows: List[float] = []
    for i in range(window, len(values) - window):
        neighbourhood = values[i - window: i + window + 1]
        if values[i] == neighbourhood.max():
            highs.append(float(values[i]))
        if values[i] == neighbourhood.min():
            lows.append(float(values[i]))
    return highs, lows


@dataclass(frozen=True)
class Levels:
    support: Optional[float] = None
    resistance: Optional[float] = None
    window_low: Optional[float] = None
    window_high: Optional[float] = None
    pivot_count: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return (self.support is not None or self.resistance is not None) \
            and not self.error


def support_resistance(closes: Optional["pd.Series"],
                       window_days: int = 365,
                       pivot_window: int = PIVOT_WINDOW_BARS) -> Levels:
    """The nearest pivot low below the price and pivot high above it.

    Descriptive, and labelled as such on screen: these are levels the
    price has actually turned at, not levels claimed to hold. Round
    numbers are deliberately not used — a round number is a property of
    the currency a coin is quoted in, not of the coin.

    Either side can be absent, and that is information: no pivot high
    above the price means the coin is at the top of its own range.
    """
    if closes is None:
        return Levels(error="No price history.")
    series = pd.Series(closes).dropna()
    if len(series) < 2 * pivot_window + 3:
        return Levels(error=(
            f"A pivot needs {pivot_window} bars either side; this "
            f"history has {len(series)}."))
    window = series.iloc[-window_days:] if window_days else series
    price = float(window.iloc[-1])
    highs, lows = pivots(window, pivot_window)
    below = [level for level in lows if level < price]
    above = [level for level in highs if level > price]
    return Levels(
        support=max(below) if below else None,
        resistance=min(above) if above else None,
        window_low=float(window.min()), window_high=float(window.max()),
        pivot_count=len(highs) + len(lows))


# --- volume -------------------------------------------------------------------

@dataclass(frozen=True)
class RelativeVolume:
    latest: Optional[float] = None
    average: Optional[float] = None
    ratio: Optional[float] = None
    window_days: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.ratio is not None and not self.error


def relative_volume(volumes: Optional["pd.Series"],
                    window_days: int = VOLUME_WINDOW_DAYS) -> RelativeVolume:
    """Today's volume against its trailing average, the bar EXCLUDED.

    Including the bar under test in its own baseline turns a true
    doubling of volume into roughly 190% of average rather than 200%.
    """
    if volumes is None:
        return RelativeVolume(error="No volume history.")
    series = pd.Series(volumes).dropna()
    if len(series) < window_days + 1:
        return RelativeVolume(
            window_days=window_days,
            error=(f"A {window_days}-day average needs {window_days + 1} "
                   f"bars; this history has {len(series)}."))
    latest = float(series.iloc[-1])
    average = float(series.iloc[-(window_days + 1):-1].mean())
    if average <= 0:
        return RelativeVolume(latest, average, window_days=window_days,
                              error="Average volume is zero.")
    return RelativeVolume(latest, average, latest / average, window_days)


# --- on-chain activity --------------------------------------------------------

@dataclass(frozen=True)
class ActivityReading:
    key: str
    label: str
    latest: Optional[float] = None
    change_30d_pct: Optional[float] = None
    unit: str = ""
    note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.latest is not None and not self.error


def activity(metric_key: str, series: Optional["pd.Series"],
             error: Optional[str] = None) -> ActivityReading:
    """One on-chain series reduced to a level and a 30-day change."""
    metric = crypto_data.ONCHAIN_BY_KEY.get(metric_key)
    label = metric.label if metric else metric_key
    unit = metric.unit if metric else ""
    note = metric.note if metric else ""
    if error:
        return ActivityReading(metric_key, label, unit=unit, note=note,
                               error=error)
    if series is None or len(series) == 0:
        return ActivityReading(metric_key, label, unit=unit, note=note,
                               error="No data returned.")
    clean = pd.Series(series).dropna()
    if clean.empty:
        return ActivityReading(metric_key, label, unit=unit, note=note,
                               error="No usable data.")
    latest = float(clean.iloc[-1])
    change = None
    if len(clean) > 30:
        earlier = float(clean.iloc[-31])
        if earlier:
            change = 100.0 * (latest - earlier) / earlier
    return ActivityReading(metric_key, label, latest, change, unit, note)


def activity_summary(readings: Sequence[ActivityReading]) -> str:
    """One sentence over the chain readings that resolved.

    Counts what was measured rather than scoring out of a fixed
    denominator, so a partial fetch reads as partial rather than as bad
    news about the chain.
    """
    ok = [r for r in readings if r.ok]
    if not ok:
        return ("No on-chain activity could be read. " +
                crypto_data.ONCHAIN_BITCOIN_ONLY)
    rising = [r for r in ok if r.change_30d_pct is not None
              and r.change_30d_pct > 0]
    return (f"{len(ok)} of {len(readings)} chain metrics resolved; "
            f"{len(rising)} of those are higher than 30 days ago.")
