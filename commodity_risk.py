"""Volatility, season, hedging value, and the shocks worth running.

THE ANNUALISATION FACTOR IS 252 HERE, AND THAT WAS CHECKED. After the
crypto module the tempting inference is that every non-equity class
needs its own factor. Measured over the three years to 2026-08-26,
Yahoo returns:

    GC=F 755 bars / 1096 days -> 251.6 a year
    CL=F 755 bars / 1096 days -> 251.6
    ZC=F 753 bars / 1093 days -> 251.6
    SPY  754 bars / 1095 days -> 251.5

Futures pits keep the exchange calendar, so `config.trading_days_per_year`
is right for commodities and no local constant is introduced. The
measurement is recorded because "we changed it for crypto so change it
here" is exactly the reasoning that would have been wrong.

SEASONALITY IS THE POINT, NOT A CURIOSITY. Natural gas is dearer in
January than October and corn is cheaper after harvest, and these are
structural, not anomalies to be smoothed away. But a month-of-year
average over five years rests on FIVE observations per month, which is
a small number to draw a line through. The reading therefore carries its
own sample size and declines to rank a month it has fewer than
`MIN_SEASONAL_OBSERVATIONS` sightings of. A seasonal chart that does not
say how many years it averaged invites the reader to trust a shape drawn
through three points.

THE HEDGING THESIS IS TESTABLE, so it is tested rather than asserted.
Commodities are held as an inflation and dollar hedge; whether a given
one behaves that way is a correlation against the dollar index and an
inflation-linked bond fund, computed and shown with its sample size. A
commodity that moves WITH the dollar is not doing the job it is being
bought for, and that should be visible.

BASIS RISK IS NAMED BUT NOT MEASURED. It is the risk that a futures
position diverges from the physical exposure it hedges — and measuring
it needs the physical price, which is the spot this build does not have
(see commodity_data.SPOT_UNAVAILABLE). The calendar spread's own
volatility is a related and computable quantity, and is offered as that
rather than relabelled as basis risk.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from logging_setup import get_logger, log_event

logger = get_logger(__name__)

# Measured, not inherited: futures keep the exchange calendar. See the
# module docstring for the bar counts this rests on.
BARS_PER_YEAR = 252

ANNUALISATION_NOTE = (
    "Annualised over 252 trading days. Futures keep the exchange "
    "calendar — measured at 251.6 bars a year against SPY's 251.5 — so "
    "unlike crypto this needs no separate factor."
)

VOL_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("30-day", 30), ("90-day", 90), ("1-year", 252),
)

# Five years of monthly observations is five points per month. Below
# three the "average" is not one.
MIN_SEASONAL_OBSERVATIONS = 3
SEASONAL_YEARS = 10

MIN_CORRELATION_DAYS = 60

CONTINUOUS_SERIES_NOTE = (
    "Measured on Yahoo's continuous front-month series, which splices "
    "one contract onto the next at each expiry. Its month-to-month "
    "change therefore mixes the price move with the gap between the "
    "expiring and incoming contracts — the roll. That is the series a "
    "reader watching \"the price of gas\" sees, and it is the right one "
    "for a seasonal read, but it is not a single contract's return."
)

MONTH_NAMES = ("January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November",
               "December")

BASIS_RISK_NOTE = (
    "Basis risk — the gap between a futures hedge and the physical "
    "position it covers — is not measured, because it needs a physical "
    "price and there is no spot series here. What is shown instead is "
    "the volatility of the nearest calendar spread, which moves for "
    "the same reasons and is computable from quotes that exist."
)

SHOCK_NOTE = (
    "A price shock applied to the position, not a forecast. The figure "
    "answers \"what would this cost me\", which is the question a stop "
    "or a position size is set against."
)


def _returns(closes: Optional["pd.Series"]) -> Optional["pd.Series"]:
    if closes is None:
        return None
    series = pd.Series(closes).dropna()
    if len(series) < 3:
        return None
    return series.pct_change().dropna()


# --- volatility ---------------------------------------------------------------

@dataclass(frozen=True)
class VolatilityWindow:
    label: str
    days: int
    annualised_pct: Optional[float] = None
    observations: int = 0

    @property
    def ok(self) -> bool:
        return self.annualised_pct is not None

    @property
    def status(self) -> str:
        return "Available" if self.ok else "Unavailable"


def volatility_windows(closes: Optional["pd.Series"],
                       windows: Sequence[Tuple[str, int]] = VOL_WINDOWS
                       ) -> Tuple[VolatilityWindow, ...]:
    """Annualised realised volatility, computed only on a full window.

    A window with too little history is Unavailable rather than zero: a
    contract listed two months ago has no annual volatility, and
    reporting 0% would be an all-clear nobody performed.
    """
    returns = _returns(closes)
    out: List[VolatilityWindow] = []
    for label, days in windows:
        if returns is None or len(returns) < days:
            out.append(VolatilityWindow(
                label, days, None, 0 if returns is None else len(returns)))
            continue
        window = returns.iloc[-days:]
        out.append(VolatilityWindow(
            label, days,
            float(window.std() * (BARS_PER_YEAR ** 0.5) * 100.0),
            int(len(window))))
    return tuple(out)


# --- seasonality --------------------------------------------------------------

@dataclass(frozen=True)
class SeasonalMonth:
    month: int
    average_pct: Optional[float] = None
    observations: int = 0
    positive_share: Optional[float] = None

    @property
    def name(self) -> str:
        return MONTH_NAMES[self.month - 1]

    @property
    def scored(self) -> bool:
        return (self.average_pct is not None
                and self.observations >= MIN_SEASONAL_OBSERVATIONS)


@dataclass(frozen=True)
class Seasonality:
    months: Tuple[SeasonalMonth, ...] = ()
    years_covered: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return any(m.scored for m in self.months) and not self.error

    @property
    def strongest(self) -> Optional[SeasonalMonth]:
        scored = [m for m in self.months if m.scored]
        return max(scored, key=lambda m: m.average_pct) if scored else None

    @property
    def weakest(self) -> Optional[SeasonalMonth]:
        scored = [m for m in self.months if m.scored]
        return min(scored, key=lambda m: m.average_pct) if scored else None


def seasonality(closes: Optional["pd.Series"],
                years: int = SEASONAL_YEARS) -> Seasonality:
    """Average return by calendar month, with the sample size kept.

    Every month is returned whether or not it could be scored, so a
    partial history reads as partial rather than as a set of months that
    happen not to matter.
    """
    if closes is None:
        return Seasonality(error="No price history.")
    series = pd.Series(closes).dropna()
    if len(series) < 60:
        return Seasonality(error="Too little history for a seasonal read.")
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    if years:
        cutoff = series.index[-1] - pd.DateOffset(years=int(years))
        series = series[series.index >= cutoff]
    monthly = series.resample("ME").last().dropna()
    changes = monthly.pct_change().dropna() * 100.0
    if changes.empty:
        return Seasonality(error="Not enough monthly observations.")

    months: List[SeasonalMonth] = []
    for month in range(1, 13):
        sample = changes[changes.index.month == month]
        if sample.empty:
            months.append(SeasonalMonth(month, None, 0, None))
            continue
        months.append(SeasonalMonth(
            month, float(sample.mean()), int(sample.count()),
            float((sample > 0).mean())))
    span_years = (changes.index[-1] - changes.index[0]).days / 365.25
    return Seasonality(tuple(months), span_years)


def describe_seasonality(reading: Seasonality) -> str:
    if not reading.ok:
        return reading.error or "No seasonal pattern could be measured."
    best, worst = reading.strongest, reading.weakest
    return (
        f"Over {reading.years_covered:.1f} years, {best.name} averages "
        f"{best.average_pct:+.2f}% and {worst.name} "
        f"{worst.average_pct:+.2f}%. Each month rests on about "
        f"{best.observations} observations, so read the shape rather "
        f"than the decimals.")


# --- correlation, and the hedging thesis --------------------------------------

@dataclass(frozen=True)
class Correlation:
    label: str
    symbol: str
    coefficient: Optional[float] = None
    observations: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return (self.coefficient is not None
                and self.observations >= MIN_CORRELATION_DAYS)

    @property
    def strength(self) -> str:
        if not self.ok:
            return "Unavailable"
        magnitude = abs(self.coefficient)
        if magnitude >= 0.7:
            return "Strong"
        if magnitude >= 0.4:
            return "Moderate"
        if magnitude >= 0.2:
            return "Weak"
        return "Negligible"


# The comparisons that test what a commodity is usually bought for.
HEDGE_BENCHMARKS: Tuple[Tuple[str, str, str], ...] = (
    ("US equities", "SPY",
     "A diversifier should not simply track the stock market."),
    ("US dollar", "DX-Y.NYB",
     "Commodities are priced in dollars, so a strong negative reading "
     "is the normal state — a positive one means the dollar hedge is "
     "not working."),
    ("Inflation-linked bonds", "TIP",
     "The closest tradable proxy for realised inflation expectations "
     "available here."),
)


def correlate(closes: Optional["pd.Series"],
              other_closes: Optional["pd.Series"],
              label: str, symbol: str) -> Correlation:
    """Correlation of daily returns over the days both series traded."""
    left, right = _returns(closes), _returns(other_closes)
    if left is None or right is None:
        return Correlation(label, symbol, error="Not enough price history.")
    left.index = pd.DatetimeIndex(left.index).tz_localize(None).normalize()
    right.index = pd.DatetimeIndex(right.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"a": left, "b": right}).dropna()
    if len(frame) < MIN_CORRELATION_DAYS:
        return Correlation(label, symbol, observations=len(frame),
                           error=(f"Only {len(frame)} shared trading days; "
                                  f"at least {MIN_CORRELATION_DAYS} are "
                                  f"needed."))
    coefficient = float(frame["a"].corr(frame["b"]))
    if coefficient != coefficient:
        return Correlation(label, symbol, observations=len(frame),
                           error="Correlation is undefined — a series is flat.")
    return Correlation(label, symbol, coefficient, int(len(frame)))


def hedge_verdict(dollar: Correlation) -> Tuple[str, str]:
    """Whether the dollar hedge is actually behaving.

    Stated as an observation about co-movement, never as advice. A
    commodity that rises with the dollar is not broken — it is telling
    you its own supply story is dominating the currency effect.
    """
    if not dollar.ok:
        return "Unavailable", dollar.error or "No dollar correlation."
    coefficient = dollar.coefficient
    if coefficient <= -0.3:
        return "Behaving as a dollar hedge", (
            f"Moves against the dollar at {coefficient:+.2f} over "
            f"{dollar.observations} shared days — the usual relationship "
            f"for a dollar-priced commodity.")
    if coefficient >= 0.2:
        return "Not moving against the dollar", (
            f"Moves WITH the dollar at {coefficient:+.2f} over "
            f"{dollar.observations} shared days. Its own supply and "
            f"demand are dominating the currency effect.")
    return "Largely independent of the dollar", (
        f"{coefficient:+.2f} over {dollar.observations} shared days — "
        f"close to no relationship either way.")


# --- shocks -------------------------------------------------------------------

# The task names a 20% fall. The others bracket it so the reader sees a
# curve of outcomes rather than one number chosen for them.
SHOCK_PCTS: Tuple[float, ...] = (-30.0, -20.0, -10.0, 10.0, 20.0)


@dataclass(frozen=True)
class ShockRow:
    shock_pct: float
    position_value: Optional[float] = None
    resulting_value: Optional[float] = None
    profit_loss: Optional[float] = None
    price_after: Optional[float] = None


def stress_test(price: Optional[float], position_value: Optional[float],
                shocks: Sequence[float] = SHOCK_PCTS) -> Tuple[ShockRow, ...]:
    """What a move of each size does to a stated position.

    Linear and unleveraged on purpose: this is a futures PRICE shock
    applied to an exposure the reader names. Adding a margin model would
    make the number depend on parameters this build cannot know, and a
    reader sizing a position wants the exposure arithmetic, not a
    simulated margin call.
    """
    rows: List[ShockRow] = []
    for shock in shocks:
        if position_value is None:
            rows.append(ShockRow(shock))
            continue
        resulting = float(position_value) * (1.0 + shock / 100.0)
        rows.append(ShockRow(
            shock_pct=float(shock), position_value=float(position_value),
            resulting_value=resulting,
            profit_loss=resulting - float(position_value),
            price_after=(float(price) * (1.0 + shock / 100.0)
                         if price is not None else None)))
    return tuple(rows)


def shock_in_sigmas(shock_pct: float,
                    windows: Sequence[VolatilityWindow]) -> str:
    """A shock expressed in the contract's own daily volatility.

    A 20% fall means nothing in isolation. Against a commodity whose
    annualised volatility is 30%, it is roughly a ten-sigma day and
    about a one-year move — and saying which makes the number usable.
    """
    window = next((w for w in windows if w.days == 252 and w.ok), None)
    if window is None:
        window = next((w for w in windows if w.ok), None)
    if window is None or not window.annualised_pct:
        return ""
    annual = window.annualised_pct
    daily = annual / (BARS_PER_YEAR ** 0.5)
    return (f"A {abs(shock_pct):.0f}% move is about "
            f"{abs(shock_pct) / daily:.0f} standard deviations of a single "
            f"day, or {abs(shock_pct) / annual:.2f} of a typical year, on "
            f"this contract's {window.label} volatility of {annual:.0f}%.")


def spread_volatility(curve_history: Optional["pd.DataFrame"],
                      near: str, far: str) -> Optional[float]:
    """Annualised volatility of a calendar spread, where both legs have
    history. Offered in place of basis risk, and labelled as what it is.
    """
    if curve_history is None or near not in curve_history or far not in curve_history:
        return None
    spread = (pd.Series(curve_history[far]).dropna()
              - pd.Series(curve_history[near]).dropna())
    changes = spread.diff().dropna()
    if len(changes) < 30:
        return None
    return float(changes.std() * (BARS_PER_YEAR ** 0.5))
