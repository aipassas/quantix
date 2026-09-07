"""Curve shape, cost of carry, and what the term structure is saying.

THE TASK'S FORWARD-PRICE FORMULA IS DIMENSIONALLY WRONG. It gives

    P(T) = Spot + (r + s - y) x T

which adds a RATE to a PRICE. With gold near 4,683 and a net carry of
4.7%, that returns 4,683 + 0.047 = 4,683.05 for a one-year forward —
i.e. it says the forward equals spot, always. The relation is
multiplicative: F = S x (1 + cT), or F = S x e^(cT) continuously. The
real one-year gold contract trades at 4,909, which is 4.8% above the
front, not 0.001% above it.

SO THE MODULE INVERTS THE RELATION INSTEAD OF APPLYING IT. Computing a
forward price needs r, s and y — the risk-free rate, storage cost and
convenience yield. Only the first is observable here, and the other two
are exactly what a reader wants to learn. Taking them as inputs would
mean inventing two numbers to derive a third that the market already
quotes. Running it backwards uses only prices that exist:

    implied carry = ln(F_far / F_near) / (T_far - T_near)

and the risk-free rate subtracted from that leaves storage minus
convenience yield, which is the informative part. Measured on the real
gold strip: Dec-26 4,683.2 to Dec-27 4,909.0 implies 4.71%/yr
continuously compounded against a 13-week bill at 3.71% — a 1.00pp
spread that is storage and insurance net of convenience yield, which is
the textbook figure for gold. The arithmetic reproduces a known answer
on live data, which is the only reason to trust it on an unknown one.

CONTANGO AND BACKWARDATION ARE NOT A BINARY. Corn's measured curve runs
506.5 (Sep-26), 528.75, 543.75, 550.25, 551.75 (Jul-27), then 525.0 and
524.5 into the new crop. It rises for five contracts and falls for two:
front-to-back it looks like mild contango, and that reading misses the
whole point, which is that the market is pricing a harvest. Every
agricultural curve has this shape at some point in the year. A
classifier with two outcomes will therefore be wrong about agriculture
roughly half the time, so this one has four — contango, backwardation,
flat, and HUMPED, with the turning point named.

SHAPE IS READ OFF LIQUID CONTRACTS ONLY. See commodity_data: gold quotes
a May-27 contract with nineteen lots of open interest. Its price is real
in the sense that someone posted it, and meaningless in the sense that
nobody trades it. commodity_data.Curve.liquid_points does the filtering;
this module never second-guesses it.
"""
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

import commodity_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

CONTANGO = "Contango"
BACKWARDATION = "Backwardation"
FLAT = "Flat"
HUMPED = "Humped"
UNAVAILABLE = "Unavailable"

# Front-to-back moves smaller than this are called flat rather than given
# a direction. A tenth of a percent across a year of contracts is noise
# in the settlement prices, not a term structure.
FLAT_THRESHOLD_PCT = 0.10

# A step is counted as a direction change only if it moves more than this
# fraction of the curve's own front price. Without it, one settlement
# rounding turns a clean contango into a "humped" curve.
STEP_NOISE_PCT = 0.05

SPEC_FORMULA_NOTE = (
    "A forward price is not computed from assumed inputs. It needs "
    "storage cost and convenience yield, which are unobservable here "
    "and are the very things the curve is being read to learn — and the "
    "additive form \"spot + (r + s - y) x T\" adds a rate to a price, "
    "returning a one-year gold forward 0.001% above spot where the real "
    "contract trades 4.8% above it. The carry is inverted out of the "
    "quoted contracts instead."
)


# --- shape --------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    label: str = UNAVAILABLE
    front_to_back_pct: Optional[float] = None
    turning_point: str = ""          # contract label where direction changed
    contracts_used: int = 0
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.label != UNAVAILABLE


def shape(curve: "commodity_data.Curve") -> Shape:
    """Contango, backwardation, flat, or humped — with the turn named.

    The four-way answer exists because agriculture demands it: a curve
    that rises into old crop and falls into new crop is not "mild
    contango", it is a harvest, and saying so is the whole content of
    the reading.
    """
    if curve is None or not curve.ok:
        return Shape(detail=(getattr(curve, "error", "")
                             or "No curve to read."))
    points = [p for p in curve.liquid_points if p.price]
    if len(points) < 2:
        return Shape(detail="Fewer than two liquid contracts.")

    front, back = points[0], points[-1]
    change_pct = 100.0 * (back.price - front.price) / front.price

    noise = abs(front.price) * STEP_NOISE_PCT / 100.0
    directions: List[int] = []
    for earlier, later in zip(points, points[1:]):
        delta = later.price - earlier.price
        directions.append(0 if abs(delta) <= noise else (1 if delta > 0 else -1))
    moves = [d for d in directions if d != 0]

    turning = ""
    for index in range(1, len(moves)):
        if moves[index] != moves[index - 1]:
            # The turn is at the contract where the sign flips.
            flat_before = sum(1 for d in directions[:index] if d == 0)
            turning = points[index + flat_before].label
            break

    if abs(change_pct) < FLAT_THRESHOLD_PCT and not turning:
        label = FLAT
        detail = (f"{front.label} to {back.label} moves "
                  f"{change_pct:+.2f}% — no meaningful term structure.")
    elif turning:
        label = HUMPED
        detail = (
            f"The curve changes direction at {turning}: it does not "
            f"simply rise or fall, so a single contango/backwardation "
            f"label would hide what the market is pricing. Front to back "
            f"is {change_pct:+.2f}%.")
    elif all(d >= 0 for d in directions):
        label = CONTANGO
        detail = (
            f"{back.label} trades {change_pct:+.2f}% above {front.label}. "
            f"Deferred contracts priced above nearby ones is the normal "
            f"state for a storable commodity — it covers financing and "
            f"storage.")
    else:
        label = BACKWARDATION
        detail = (
            f"{back.label} trades {change_pct:+.2f}% below {front.label}. "
            f"Paying more for delivery now than later is what a market "
            f"short of immediate supply looks like.")

    return Shape(label=label, front_to_back_pct=change_pct,
                 turning_point=turning, contracts_used=len(points),
                 detail=detail)


# --- cost of carry ------------------------------------------------------------

@dataclass(frozen=True)
class Carry:
    annualised_pct: Optional[float] = None      # continuously compounded
    near_label: str = ""
    far_label: str = ""
    years: Optional[float] = None
    risk_free_pct: Optional[float] = None
    net_of_rate_pct: Optional[float] = None     # storage minus convenience
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.annualised_pct is not None and not self.error


def implied_carry(curve: "commodity_data.Curve",
                  risk_free_pct: Optional[float] = None) -> Carry:
    """The carry the market is quoting, from the two ends of the curve.

    Continuously compounded, because that is the convention the relation
    F = S e^(cT) is written in and mixing it with a simple rate is a
    silent few-tenths error over a year.
    """
    if curve is None or not curve.ok:
        return Carry(error=getattr(curve, "error", "") or "No curve.")
    points = [p for p in curve.liquid_points if p.price and p.price > 0]
    if len(points) < 2:
        return Carry(error="Fewer than two liquid contracts.")
    near, far = points[0], points[-1]
    reference = curve.as_of or pd.Timestamp.today().normalize()
    years = far.years_out(reference) - near.years_out(reference)
    if years <= 0:
        return Carry(error="The two contracts do not span any time.")
    annualised = 100.0 * math.log(far.price / near.price) / years
    net = None if risk_free_pct is None else annualised - float(risk_free_pct)
    log_event(logger, logging.INFO, "commodity_curve.carry",
              commodity=getattr(curve.commodity, "key", ""),
              carry=round(annualised, 3))
    return Carry(annualised_pct=annualised, near_label=near.label,
                 far_label=far.label, years=years,
                 risk_free_pct=risk_free_pct, net_of_rate_pct=net)


def describe_carry(carry: Carry) -> str:
    if not carry.ok:
        return carry.error or "Cost of carry is unavailable."
    text = (f"{carry.far_label} over {carry.near_label} implies "
            f"{carry.annualised_pct:+.2f}%/yr, continuously compounded "
            f"over {carry.years:.2f} years.")
    if carry.net_of_rate_pct is not None:
        text += (
            f" Against a risk-free rate of {carry.risk_free_pct:.2f}%, "
            f"that leaves {carry.net_of_rate_pct:+.2f}% — storage and "
            f"insurance net of convenience yield. A negative figure "
            f"means holders are paid to own the physical commodity, "
            f"which is what scarcity looks like.")
    return text


# --- calendar spread (the honest stand-in for basis) --------------------------

@dataclass(frozen=True)
class CalendarSpread:
    near_label: str = ""
    far_label: str = ""
    absolute: Optional[float] = None
    percent: Optional[float] = None
    annualised_pct: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.absolute is not None and not self.error


def calendar_spread(curve: "commodity_data.Curve") -> CalendarSpread:
    """The two nearest liquid contracts, differenced.

    This is what replaces basis. There is no spot price in this build
    (see commodity_data.SPOT_UNAVAILABLE), and the front contract is not
    spot — it is a dated claim with its own carry. The nearest pair
    measures immediate tightness without needing a price that does not
    exist.
    """
    if curve is None or not curve.ok:
        return CalendarSpread(error=getattr(curve, "error", "") or "No curve.")
    points = [p for p in curve.liquid_points if p.price and p.price > 0]
    if len(points) < 2:
        return CalendarSpread(error="Fewer than two liquid contracts.")
    near, far = points[0], points[1]
    absolute = far.price - near.price
    percent = 100.0 * absolute / near.price
    reference = curve.as_of or pd.Timestamp.today().normalize()
    years = far.years_out(reference) - near.years_out(reference)
    annualised = (100.0 * math.log(far.price / near.price) / years
                  if years > 0 else None)
    return CalendarSpread(near.label, far.label, absolute, percent,
                          annualised)


# --- roll yield ---------------------------------------------------------------

@dataclass(frozen=True)
class RollYield:
    annualised_pct: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.annualised_pct is not None and not self.error

    @property
    def favourable(self) -> Optional[bool]:
        return None if self.annualised_pct is None else self.annualised_pct > 0


def roll_yield(curve: "commodity_data.Curve") -> RollYield:
    """What holding a rolled long position earns or costs, before price.

    The sign convention matters and is easy to get backwards: in
    contango the next contract costs MORE, so rolling into it loses
    money and the roll yield is NEGATIVE. It is minus the calendar
    spread, annualised.
    """
    spread = calendar_spread(curve)
    if not spread.ok or spread.annualised_pct is None:
        return RollYield(error=spread.error or "No spread to annualise.")
    return RollYield(annualised_pct=-spread.annualised_pct)


ROLL_ANNUALISATION_CAVEAT = (
    "Annualised from the nearest pair of contracts, so it states what a "
    "year would cost IF the curve held its present shape. It rarely "
    "does — a one-month spread of a few percent annualises to a large "
    "number, and WTI's measured +38.8%/yr comes from a 3.2% spread over "
    "a single month."
)


def describe_roll(roll: RollYield, shape_label: str = "") -> str:
    if not roll.ok:
        return roll.error or "Roll yield is unavailable."
    if roll.annualised_pct > 0:
        return (f"Rolling a long position earns about "
                f"{roll.annualised_pct:+.2f}%/yr before any price move — "
                f"each contract is replaced by a cheaper one. This is "
                f"the tailwind a backwardated curve gives a holder.")
    return (f"Rolling a long position costs about "
            f"{roll.annualised_pct:+.2f}%/yr before any price move — "
            f"each contract is replaced by a dearer one. It is why a "
            f"commodity fund can lose money in a market that went "
            f"nowhere.")


# --- inventory against its own season -----------------------------------------

# Five years is the convention the EIA's own reporting uses, and it is
# long enough to average out one unusual year without reaching back to a
# different market regime.
SEASONAL_YEARS = 5


@dataclass(frozen=True)
class InventoryReading:
    latest: Optional[float] = None
    as_of: Optional["pd.Timestamp"] = None
    seasonal_average: Optional[float] = None
    deviation_pct: Optional[float] = None
    years_used: int = 0
    change_4w_pct: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.latest is not None and not self.error

    @property
    def scored(self) -> bool:
        return self.deviation_pct is not None and self.years_used >= 3


def read_inventory(series: Optional["pd.Series"],
                   years: int = SEASONAL_YEARS) -> InventoryReading:
    """The latest stock level against the same week in prior years.

    Compared week-of-year rather than to a flat mean, because crude
    stocks have a large and entirely normal seasonal swing — a level
    that is high for October can be low for April. A flat average would
    report the season as news.
    """
    if series is None or len(series) == 0:
        return InventoryReading(error="No inventory series.")
    clean = pd.Series(series).dropna()
    if clean.empty:
        return InventoryReading(error="Inventory series is empty.")
    latest_date = clean.index[-1]
    latest = float(clean.iloc[-1])

    week = latest_date.isocalendar().week
    prior: List[float] = []
    for offset in range(1, int(years) + 1):
        target = latest_date.year - offset
        same_week = clean[(clean.index.year == target)
                          & (clean.index.isocalendar().week == week)]
        if not same_week.empty:
            prior.append(float(same_week.iloc[0]))

    average = sum(prior) / len(prior) if prior else None
    deviation = (100.0 * (latest - average) / average
                 if average else None)

    change_4w = None
    if len(clean) > 4:
        earlier = float(clean.iloc[-5])
        if earlier:
            change_4w = 100.0 * (latest - earlier) / earlier

    return InventoryReading(latest=latest, as_of=latest_date,
                            seasonal_average=average,
                            deviation_pct=deviation, years_used=len(prior),
                            change_4w_pct=change_4w)


def inventory_verdict(reading: InventoryReading,
                      shape_label: str = "") -> Tuple[str, str]:
    """A (label, explanation) pair, read WITH the curve where possible.

    Inventory alone is a level; inventory alongside the curve is a
    story. Low stocks and backwardation agree with each other, and when
    they disagree that is worth seeing rather than smoothing over.
    """
    if not reading.ok:
        return UNAVAILABLE, reading.error or "No inventory reading."
    if not reading.scored:
        return "Unscored", (
            f"Only {reading.years_used} prior year(s) match this week of "
            f"the calendar — too few to call the level high or low.")
    deviation = reading.deviation_pct
    if deviation >= 5:
        label, base = "Ample", (
            f"Stocks are {deviation:+.1f}% against the {reading.years_used}"
            f"-year average for this week of the year.")
    elif deviation <= -5:
        label, base = "Tight", (
            f"Stocks are {deviation:+.1f}% against the {reading.years_used}"
            f"-year average for this week of the year.")
    else:
        label, base = "Normal", (
            f"Stocks are {deviation:+.1f}% against the {reading.years_used}"
            f"-year average for this week — within the usual band.")

    if shape_label == BACKWARDATION and label == "Tight":
        base += (" The curve agrees: backwardation and low stocks are the "
                 "same signal seen two ways.")
    elif shape_label == CONTANGO and label == "Ample":
        base += (" The curve agrees: contango and full storage are the "
                 "same signal seen two ways.")
    elif shape_label in (CONTANGO, BACKWARDATION) and label in ("Tight",
                                                                "Ample"):
        base += (f" Note the curve says {shape_label.lower()}, which does "
                 f"not line up with stocks reading {label.lower()} — "
                 f"worth a look rather than an average of the two.")
    return label, base
