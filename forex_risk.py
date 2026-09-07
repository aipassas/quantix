"""Volatility, correlation, tail risk, and what unwinds a carry trade.

FX ANNUALISES OVER 260 DAYS, MEASURED. The three classes before this one
each needed checking rather than inheriting, and so does this:

    EURUSD=X 778 bars / 1096 days -> 259.3 a year
    USDJPY=X 778 bars             -> 259.3
    AUDJPY=X 778 bars             -> 259.3
    SPY      754 bars             -> 251.5
    GC=F     755 bars             -> 251.6

Currency trades around the clock on weekdays and keeps no exchange
holidays, so it gets every weekday. The difference from 252 is small —
1.6% on a volatility figure, against crypto's 20% — but it costs nothing
to be right and the measurement is what says which.

THE TASK'S VOLATILITY BAND WOULD FAIL ON HEALTHY DATA. It states
"typical 8-12% for major pairs". Measured over the year to 2026-09-07,
only FOUR of twelve majors sit inside it: EUR/USD is 5.71%, USD/CAD
4.33% and EUR/GBP 3.47%. A validation rule built on that band would flag
two thirds of the board as anomalous and teach its reader to ignore the
suite. Volatility is reported, never banded.

IMPLIED VOLATILITY IS NOT AVAILABLE, so the volatility smile the task
asks for is not drawn. A smile is a curve of implied vols across
strikes, and it needs FX option quotes that no free source publishes.
What is here is REALISED volatility from the price series, which is a
different quantity and is labelled as one.

CARRY UNWIND IS MEASURABLE, and this is the finding worth the module.
The claim that carry pairs "go up by the stairs and down by the lift" is
testable as return skew, and it holds: over five years AUD/JPY skews
-0.47 with a worst day of -5.84% against a best of +4.09%, NZD/JPY
-0.14 and GBP/JPY -0.27, while EUR/USD — nobody's carry trade — skews
+0.24. So a funded position in a high-yielder really does lose more in a
day than it ever gains in one.

One caveat that shaped the output: skew is fragile against a single bad
print. EUR/GBP returns +2.15 on the same window because the series
carries a -10.5% and a +11.6% day, moves that pair has never truly made.
So the skew is always shown WITH its extremes, letting a reader see when
one observation is doing the work.
"""
import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import forex_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

# Measured, not inherited. See the module docstring.
BARS_PER_YEAR = 260

ANNUALISATION_NOTE = (
    "Annualised over 260 days. Currency trades every weekday with no "
    "exchange holidays — measured at 259.3 bars a year against SPY's "
    "251.5 — so it gets a slightly longer year than equities."
)

NO_TYPICAL_BAND = (
    "No \"typical\" volatility band is applied. Only four of twelve "
    "majors currently sit inside the 8-12% often quoted for the asset "
    "class: EUR/USD is 5.7%, USD/CAD 4.3% and EUR/GBP 3.5%. A rule "
    "built on that band would flag most of a healthy board as "
    "anomalous."
)

VOL_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("30-day", 30), ("90-day", 90), ("1-year", 260),
)

MIN_CORRELATION_DAYS = 60
MIN_SKEW_OBSERVATIONS = 250

# 1.645 sigma is the 95th percentile of a normal distribution. Kept as a
# named constant for the tests to pin the default against; the figure
# itself is derived per confidence level, so asking for 99% gets a 99%
# number rather than a mislabelled 95% one.
Z_95 = 1.6448536269514722


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
    """Realised volatility over each window, computed only when full."""
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


# --- value at risk ------------------------------------------------------------

@dataclass(frozen=True)
class ValueAtRisk:
    horizon_days: int = 1
    confidence: float = 0.95
    parametric_pct: Optional[float] = None
    historical_pct: Optional[float] = None
    observations: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.parametric_pct is not None and not self.error

    @property
    def label(self) -> str:
        return f"{self.horizon_days}-day VaR ({self.confidence:.0%})"


def value_at_risk(closes: Optional["pd.Series"], horizon_days: int = 1,
                  confidence: float = 0.95,
                  lookback: int = 260) -> ValueAtRisk:
    """Both the parametric and the historical figure, side by side.

    They are reported together on purpose. The parametric one assumes
    normal returns and currency returns are not normal — the skew
    measured below says so — so where the historical figure is the
    larger, the normal assumption is understating the tail and the
    reader can see by how much.
    """
    returns = _returns(closes)
    if returns is None or len(returns) < 30:
        return ValueAtRisk(horizon_days, confidence,
                           error="Not enough history for a tail estimate.")
    window = returns.iloc[-lookback:]
    scale = float(horizon_days) ** 0.5
    # The z for any confidence, from the stdlib rather than a hard-coded
    # 1.645 — a table with one entry silently reports a 95% figure under
    # a 99% label the moment the caller asks for one.
    z = statistics.NormalDist().inv_cdf(confidence)
    parametric = float(window.std() * z * scale * 100.0)
    historical = float(-np.percentile(window, (1 - confidence) * 100)
                       * scale * 100.0)
    return ValueAtRisk(horizon_days, confidence, parametric, historical,
                       int(len(window)))


def describe_var(reading: ValueAtRisk) -> str:
    if not reading.ok:
        return reading.error or "Value at risk is unavailable."
    text = (f"A {reading.horizon_days}-day loss should exceed "
            f"{reading.parametric_pct:.2f}% on about one day in twenty, "
            f"assuming normal returns.")
    if reading.historical_pct is not None:
        if reading.historical_pct > reading.parametric_pct * 1.05:
            text += (f" The actual {reading.confidence:.0%} loss over the "
                     f"last {reading.observations} days was "
                     f"{reading.historical_pct:.2f}%, so the normal "
                     f"assumption understates this pair's tail.")
        else:
            text += (f" The historical figure over the last "
                     f"{reading.observations} days is "
                     f"{reading.historical_pct:.2f}%.")
    return text


# --- carry unwind -------------------------------------------------------------

@dataclass(frozen=True)
class TailProfile:
    skew: Optional[float] = None
    worst_day_pct: Optional[float] = None
    best_day_pct: Optional[float] = None
    observations: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.skew is not None and not self.error

    @property
    def scored(self) -> bool:
        return self.ok and self.observations >= MIN_SKEW_OBSERVATIONS

    @property
    def standard_error(self) -> Optional[float]:
        """The sampling error of the skew estimate, sqrt(6/n).

        Skew is a noisy statistic: 800 draws from a symmetric
        distribution give a standard error near 0.087, so a fixed
        threshold of -0.1 fires on chance about one time in eight. This
        is what makes the test below sample-size aware instead.
        """
        if self.observations < 8:
            return None
        return (6.0 / self.observations) ** 0.5

    @property
    def asymmetric_down(self) -> Optional[bool]:
        """Whether the downside skew is larger than its own noise.

        Two standard errors, so a symmetric series is not called
        asymmetric by accident — caught by a test whose "symmetric"
        fixture happened to draw a skew of -0.108.
        """
        if self.skew is None or self.standard_error is None:
            return None
        return self.skew < -2.0 * self.standard_error


def tail_profile(closes: Optional["pd.Series"]) -> TailProfile:
    """Return skew, with the extremes that produced it.

    The extremes travel with the number because skew is dominated by
    single observations: EUR/GBP reads +2.15 purely on one -10.5% and
    one +11.6% print, moves the pair has never actually made.
    """
    returns = _returns(closes)
    if returns is None or len(returns) < 30:
        return TailProfile(error="Not enough history for a tail read.")
    return TailProfile(
        skew=float(returns.skew()),
        worst_day_pct=float(returns.min() * 100.0),
        best_day_pct=float(returns.max() * 100.0),
        observations=int(len(returns)))


def unwind_risk(pair: "forex_data.Pair", carry_pct: Optional[float],
                tail: TailProfile) -> Tuple[str, str]:
    """Whether this looks like a trade that unwinds badly.

    Two conditions have to hold together: the position must be PAID to
    exist (a positive carry), and its losses must be larger than its
    gains (negative skew). Either alone is unremarkable; together they
    are the shape that empties out when risk appetite turns.
    """
    if carry_pct is None or not tail.ok:
        return "Unavailable", (
            "Needs both a carry differential and enough price history.")
    funded_by_haven = pair.quote in forex_data.SAFE_HAVENS
    if not tail.scored:
        return "Unscored", (
            f"Only {tail.observations} observations — too few to read a "
            f"tail. Skew is dominated by single days at this sample size.")
    if carry_pct > 0 and tail.asymmetric_down:
        detail = (
            f"This position is paid {carry_pct:.2f}%/yr to exist and its "
            f"returns skew {tail.skew:+.2f}: the worst day in "
            f"{tail.observations} was {tail.worst_day_pct:.2f}% against a "
            f"best of {tail.best_day_pct:+.2f}%. That asymmetry is the "
            f"carry trade's signature — the interest arrives daily and "
            f"the reversal arrives at once.")
        if funded_by_haven:
            detail += (
                f" It is funded in {pair.quote}, which is bought when "
                f"risk appetite falls, so the funding leg strengthens "
                f"in exactly the conditions that close the trade.")
        return "Carry-unwind shape", detail
    if carry_pct > 0:
        return "Paid, without the tail", (
            f"Paid {carry_pct:.2f}%/yr, and returns skew "
            f"{tail.skew:+.2f} — no marked downside asymmetry over "
            f"{tail.observations} observations.")
    return "Not a carry position", (
        f"Holding this pair costs {abs(carry_pct):.2f}%/yr in interest, "
        f"so it is not the funded trade this risk describes.")


# --- correlation --------------------------------------------------------------

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


def correlate(closes: Optional["pd.Series"],
              other_closes: Optional["pd.Series"],
              label: str, symbol: str) -> Correlation:
    left, right = _returns(closes), _returns(other_closes)
    if left is None or right is None:
        return Correlation(label, symbol, error="Not enough price history.")
    left.index = pd.DatetimeIndex(left.index).tz_localize(None).normalize()
    right.index = pd.DatetimeIndex(right.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"a": left, "b": right}).dropna()
    if len(frame) < MIN_CORRELATION_DAYS:
        return Correlation(label, symbol, observations=len(frame),
                           error=(f"Only {len(frame)} shared days; at "
                                  f"least {MIN_CORRELATION_DAYS} needed."))
    coefficient = float(frame["a"].corr(frame["b"]))
    if coefficient != coefficient:
        return Correlation(label, symbol, observations=len(frame),
                           error="Correlation is undefined — a series is flat.")
    return Correlation(label, symbol, coefficient, int(len(frame)))


def peers_for(pair: "forex_data.Pair") -> Tuple[Tuple[str, str], ...]:
    """Pairs worth comparing this one against.

    Everything sharing a leg, because a pair that moves with its
    neighbour is not a second position — it is the same bet twice.
    Verified on the case the task names: EUR/USD against GBP/USD reads
    0.81 over 515 shared days.
    """
    out: List[Tuple[str, str]] = []
    for other in forex_data.PAIRS:
        if other.code == pair.code:
            continue
        if other.base in (pair.base, pair.quote) or \
                other.quote in (pair.base, pair.quote):
            out.append((other.label, other.symbol))
    return tuple(out[:5])


def concentration_note(readings: Sequence[Correlation]) -> str:
    """One sentence on whether the neighbours are all the same trade."""
    strong = [r for r in readings if r.ok and abs(r.coefficient) >= 0.7]
    if not strong:
        return ("No neighbouring pair moves closely enough with this one "
                "to be the same position in disguise.")
    names = ", ".join(r.label for r in strong)
    return (f"{names} move with this pair at 0.7 or above. Holding them "
            f"alongside it is one bet sized larger, not a spread of "
            f"bets.")


# --- what is not measured -----------------------------------------------------

IMPLIED_VOL_UNAVAILABLE = forex_data.IMPLIED_VOL_UNAVAILABLE

INTERVENTION_UNAVAILABLE = (
    "Central bank intervention risk is not scored. Interventions are "
    "announced or inferred after the fact, and there is no forward "
    "schedule of them to read — what IS shown is the calendar of rate "
    "decisions, which is the policy risk that can be seen coming."
)

POLITICAL_RISK_UNAVAILABLE = (
    "Political and geopolitical risk is not scored. No free feed "
    "publishes it per currency, and a number assembled by hand would "
    "carry an authority it has not earned."
)
