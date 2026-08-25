"""Bond valuation: yield to maturity, duration, convexity, and scenarios.

THE TASK SHIPS A REFERENCE IMPLEMENTATION AND IT IS WRONG IN TWO WAYS
THAT MATTER. Both were measured against an independent solver before a
line of this module was written, because "the spec included code" is not
evidence the code is right:

  1. FRACTIONAL MATURITIES ARE OFF BY 50 BASIS POINTS. The reference
     counts coupons with `range(1, int(years)+1)` — truncating — while
     discounting par over the fractional exponent `(1+y)**years`. The two
     halves therefore describe different bonds. A 10.5-year 5% bond at
     par solves to 4.8137% under that code; priced with cash flows at
     0.5, 1.5 … 10.5 it is 5.3127%. Nothing raises; the number is just
     wrong, and every bond between coupon dates is a fractional bond.

  2. IT CANNOT FIND A NEGATIVE YIELD. The bisection bracket starts at
     0.0001, so a 5-year 1% bond priced at 130 returns 0.0161% instead of
     -4.2561%. Deep-premium bonds do this routinely, and negative
     sovereign yields were ordinary for years — bond_data.validate_yield
     already allows them for exactly that reason.

  Two things the reference does that are RIGHT, checked rather than
  assumed: the bisection direction is correct (price falls as yield
  rises, so a too-high price means the yield must go up), and its answers
  for ordinary integer-maturity bonds are within a basis point — 5.3617%
  against 5.3639% on a 10-year 8% bond at 120. The defects are structural,
  not arithmetic.

CONVENTION. Everything here works in PERIODS, with `periods_per_year`
explicit and defaulting to 2, because semiannual is the US market
convention and a yield quoted on the wrong frequency is a different
number wearing the same name. Measured, the gap is about 1bp on a
ten-year bond — small, but there is no reason to carry it.

WHAT IS NOT MODELLED, and is not faked:
  - Embedded options. Effective duration for a callable or putable bond
    needs a rate model and an option-adjusted spread; `effective_duration`
    here reprices under parallel shifts, which is the right calculation
    for an OPTION-FREE bond and is labelled as such. A callable bond's
    true effective duration is lower than this returns, and saying so is
    better than returning a confident wrong number.
  - Implied default probability via Merton. That needs the ISSUER's
    equity value and equity volatility alongside the debt face value, and
    this build has no mapping from a bond to its issuer's listed equity —
    the corporate bond data that would carry it needs the licence
    bond_data.BLOOMBERG_UNAVAILABLE describes.
  - Weighted-average YTM across a bond FUND's holdings. Measured in
    PHASE 2.1: bond funds disclose no holdings at all through this
    source, and the per-fund duration and maturity fields are unusable.
    `fund_expected_return` uses the fund's own yield and the duration
    MEASURED from its price behaviour instead.
"""
import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from logging_setup import get_logger, log_event

logger = get_logger("bond_valuation")

DEFAULT_PERIODS_PER_YEAR = 2
PAR = 100.0

# The solver's bracket. It reaches well below zero because negative
# yields are real, and well above any plausible coupon because a
# distressed bond's yield can be enormous.
YIELD_MIN = -0.95
YIELD_MAX = 5.00
YIELD_TOLERANCE = 1e-10          # far finer than the 1bp the task settles for

# The task's own scenario ladder, in basis points.
SCENARIO_SHIFTS_BPS: Tuple[int, ...] = (-200, -100, -50, 0, 50, 100, 200)

BASIS_POINT = 1e-4

EFFECTIVE_DURATION_SHIFT_BPS = 25   # small enough to be local, large
                                    # enough not to be lost to rounding

EMBEDDED_OPTION_NOTE = (
    "Computed for an option-free bond. A callable bond's true effective "
    "duration is lower than this — the issuer redeems when rates fall, "
    "which truncates the upside — and an option-adjusted figure needs a "
    "rate model this build does not have."
)

MERTON_UNAVAILABLE = (
    "Implied default probability is not shown. The Merton model needs the "
    "issuer's equity value and equity volatility alongside the face value "
    "of its debt, and this build has no mapping from a bond to its "
    "issuer's listed equity — that link lives in the corporate bond data "
    "which needs a licence."
)


@dataclass(frozen=True)
class CashFlow:
    """One payment, at a time measured in PERIODS from settlement."""
    period: float
    amount: float


@dataclass(frozen=True)
class Bond:
    """An option-free fixed-coupon bond.

    `years_to_maturity` may be fractional — that is the normal state of a
    bond between coupon dates, and the case the task's reference code
    gets wrong by 50bp.
    """
    coupon_rate_pct: float
    years_to_maturity: float
    par: float = PAR
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR

    @property
    def coupon_per_period(self) -> float:
        return self.par * (self.coupon_rate_pct / 100.0) / self.periods_per_year

    @property
    def total_periods(self) -> float:
        return self.years_to_maturity * self.periods_per_year


def cash_flows(bond: Bond) -> List[CashFlow]:
    """Every payment, in periods from settlement.

    Generated BACKWARD from maturity, which is how a real schedule works
    and what makes a short or long first period fall out naturally: a
    10.5-year semiannual bond pays at 1, 3, 5 … 21 half-years, not at
    1 … 20 with the redemption dangling at 21.
    """
    total = bond.total_periods
    if total <= 0:
        return []
    coupon = bond.coupon_per_period
    times: List[float] = []
    remaining = total
    # Walk back a whole period at a time until the stub is left.
    while remaining > 1e-9:
        times.append(remaining)
        remaining -= 1.0
    times.reverse()
    flows = [CashFlow(t, coupon) for t in times]
    if flows:
        last = flows[-1]
        flows[-1] = CashFlow(last.period, last.amount + bond.par)
    return flows


def price_from_yield(bond: Bond, ytm: float) -> Optional[float]:
    """Clean price per `par` at an annual yield `ytm` (as a decimal).

    Discounts every flow — coupons AND redemption — on the same period
    clock. The reference implementation uses two different clocks, which
    is where its 50bp comes from.
    """
    rate = ytm / bond.periods_per_year
    if rate <= -1.0:
        return None
    total = 0.0
    for flow in cash_flows(bond):
        total += flow.amount / ((1.0 + rate) ** flow.period)
    return total


def yield_to_maturity(bond: Bond, price: float) -> Optional[float]:
    """The annual yield that reprices `bond` to `price`. None if none exists.

    Brent over a wide bracket rather than bisection from 0.0001, so a
    premium bond with a negative yield is found rather than pinned to the
    bottom of the bracket.
    """
    if price is None or price <= 0 or not cash_flows(bond):
        return None

    def excess(ytm: float) -> float:
        value = price_from_yield(bond, ytm)
        return float("inf") if value is None else value - price

    low, high = YIELD_MIN, YIELD_MAX
    f_low, f_high = excess(low), excess(high)
    if not math.isfinite(f_low) or not math.isfinite(f_high):
        return None
    if f_low * f_high > 0:
        # No sign change: the price is outside anything this bond can be
        # worth. Better to say so than to return an endpoint.
        return None
    try:
        from scipy.optimize import brentq

        root = brentq(excess, low, high, xtol=YIELD_TOLERANCE, maxiter=200)
    except Exception:                              # noqa: BLE001
        root = _bisect(excess, low, high)
        if root is None:
            return None
    return float(root)


def _bisect(fn, low: float, high: float) -> Optional[float]:
    """Fallback when scipy is unavailable. Same bracket, same tolerance."""
    f_low = fn(low)
    for _ in range(500):
        mid = (low + high) / 2.0
        f_mid = fn(mid)
        if abs(f_mid) < 1e-12 or (high - low) < YIELD_TOLERANCE:
            return mid
        if f_low * f_mid <= 0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2.0


def current_yield_pct(bond: Bond, price: float) -> Optional[float]:
    """Annual coupon over price. Not a yield to maturity — it ignores the
    pull to par entirely, which is the whole point of a YTM."""
    if not price:
        return None
    annual_coupon = bond.par * (bond.coupon_rate_pct / 100.0)
    return (annual_coupon / price) * 100.0


# --- duration and convexity ---------------------------------------------------

def macaulay_duration(bond: Bond, ytm: float) -> Optional[float]:
    """Weighted average time to the cash flows, in YEARS.

    Returned in years, not periods, because that is what a reader means
    by duration — and the two differ by a factor of `periods_per_year`,
    which is a silent 2x if it is confused.
    """
    rate = ytm / bond.periods_per_year
    if rate <= -1.0:
        return None
    flows = cash_flows(bond)
    if not flows:
        return None
    weighted = 0.0
    total = 0.0
    for flow in flows:
        present = flow.amount / ((1.0 + rate) ** flow.period)
        weighted += present * flow.period
        total += present
    if total <= 0:
        return None
    return (weighted / total) / bond.periods_per_year


def modified_duration(bond: Bond, ytm: float) -> Optional[float]:
    """Macaulay / (1 + y/m). The percentage price change per unit yield."""
    macaulay = macaulay_duration(bond, ytm)
    if macaulay is None:
        return None
    rate = ytm / bond.periods_per_year
    if rate <= -1.0:
        return None
    return macaulay / (1.0 + rate)


def convexity(bond: Bond, ytm: float) -> Optional[float]:
    """Curvature of the price-yield line, in YEARS SQUARED.

    Positive for every option-free bond, which is why a fall in yields
    helps more than the same rise hurts.
    """
    rate = ytm / bond.periods_per_year
    if rate <= -1.0:
        return None
    flows = cash_flows(bond)
    if not flows:
        return None
    weighted = 0.0
    total = 0.0
    for flow in flows:
        present = flow.amount / ((1.0 + rate) ** flow.period)
        weighted += present * flow.period * (flow.period + 1.0)
        total += present
    if total <= 0:
        return None
    periods_squared = weighted / (total * (1.0 + rate) ** 2)
    return periods_squared / (bond.periods_per_year ** 2)


def effective_duration(bond: Bond, ytm: float,
                       shift_bps: int = EFFECTIVE_DURATION_SHIFT_BPS
                       ) -> Optional[float]:
    """Duration by repricing under parallel shifts up and down.

    For an option-free bond this agrees with modified duration to several
    decimals — which is the test that the analytic version is right. It
    is NOT an option-adjusted figure; see EMBEDDED_OPTION_NOTE.
    """
    shift = shift_bps * BASIS_POINT
    base = price_from_yield(bond, ytm)
    up = price_from_yield(bond, ytm + shift)
    down = price_from_yield(bond, ytm - shift)
    if not base or up is None or down is None or base <= 0:
        return None
    return (down - up) / (2.0 * base * shift)


# --- price impact -------------------------------------------------------------

@dataclass(frozen=True)
class PriceScenario:
    shift_bps: int
    new_yield_pct: float
    exact_price: Optional[float]
    approx_price: Optional[float]      # duration + convexity estimate
    exact_change_pct: Optional[float]
    duration_only_change_pct: Optional[float]
    approx_change_pct: Optional[float]


def estimate_price_change_pct(mod_duration: Optional[float],
                              convex: Optional[float],
                              shift_bps: float) -> Optional[float]:
    """The task's own formula:
        dP = (-ModDur x dY) + (0.5 x Convexity x dY^2)

    The convexity term is what stops the estimate understating a rally
    and overstating a sell-off — a duration-only line is symmetric, and
    a bond is not.
    """
    if mod_duration is None:
        return None
    shift = shift_bps * BASIS_POINT
    change = -mod_duration * shift
    if convex is not None:
        change += 0.5 * convex * (shift ** 2)
    return change * 100.0


def scenario_table(bond: Bond, ytm: float,
                   shifts_bps: Sequence[int] = SCENARIO_SHIFTS_BPS
                   ) -> List[PriceScenario]:
    """Exact reprice AND the duration/convexity estimate, side by side.

    Both, deliberately: the estimate is what the textbook formula gives
    and the exact reprice is what actually happens, and showing the two
    together is how a reader learns where the approximation starts to
    drift — which is at the large shifts, exactly where it matters.
    """
    base = price_from_yield(bond, ytm)
    mod = modified_duration(bond, ytm)
    convex = convexity(bond, ytm)
    rows: List[PriceScenario] = []
    for shift_bps in shifts_bps:
        new_yield = ytm + shift_bps * BASIS_POINT
        exact = price_from_yield(bond, new_yield)
        exact_change = (None if not base or exact is None
                        else (exact / base - 1.0) * 100.0)
        approx_change = estimate_price_change_pct(mod, convex, shift_bps)
        duration_only = estimate_price_change_pct(mod, None, shift_bps)
        approx_price = (None if base is None or approx_change is None
                        else base * (1.0 + approx_change / 100.0))
        rows.append(PriceScenario(
            shift_bps=shift_bps,
            new_yield_pct=new_yield * 100.0,
            exact_price=exact,
            approx_price=approx_price,
            exact_change_pct=exact_change,
            duration_only_change_pct=duration_only,
            approx_change_pct=approx_change))
    return rows


# --- valuation against par and against the curve ------------------------------

PREMIUM = "Premium"
DISCOUNT = "Discount"
AT_PAR = "At par"

PAR_TOLERANCE = 0.05     # within five cents of par is par


@dataclass(frozen=True)
class ParPosition:
    label: str
    premium_pct: float           # price over par, in percent
    detail: str


def par_position(bond: Bond, price: float, ytm: Optional[float]) -> ParPosition:
    """Premium, discount or par — and what that implies about the coupon.

    A bond trades above par exactly when its coupon exceeds its yield.
    Saying which way round it is stops "premium" being read as "good".
    """
    premium_pct = (price / bond.par - 1.0) * 100.0
    if abs(price - bond.par) <= PAR_TOLERANCE:
        return ParPosition(AT_PAR, premium_pct,
                           "Coupon and yield are the same, so there is "
                           "nothing to pull toward par.")
    if price > bond.par:
        detail = ("Priced above par because the coupon "
                  f"({bond.coupon_rate_pct:.2f}%) is above the yield")
        if ytm is not None:
            detail += f" ({ytm * 100:.2f}%)"
        detail += (". The extra price is repaid as coupon and pulls back to "
                   "par by maturity — the gain is in the income, not the "
                   "price.")
        return ParPosition(PREMIUM, premium_pct, detail)
    detail = ("Priced below par because the coupon "
              f"({bond.coupon_rate_pct:.2f}%) is below the yield")
    if ytm is not None:
        detail += f" ({ytm * 100:.2f}%)"
    detail += (". The discount is recovered as the price pulls up to par by "
               "maturity, which is return that does not depend on the "
               "coupon.")
    return ParPosition(DISCOUNT, premium_pct, detail)


RICH = "Rich vs curve"
CHEAP = "Cheap vs curve"
FAIR = "In line with curve"

CURVE_TOLERANCE_BPS = 10.0


@dataclass(frozen=True)
class CurvePosition:
    label: str
    bond_yield_pct: Optional[float] = None
    curve_yield_pct: Optional[float] = None
    spread_bps: Optional[float] = None
    method: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.spread_bps is not None


def curve_position(ytm: Optional[float], curve, years_to_maturity: float
                   ) -> CurvePosition:
    """Where a bond sits against the treasury curve at its own maturity.

    The spread IS the credit spread the task asks for: a corporate yield
    minus the treasury yield at the same maturity. It is quoted in basis
    points because that is the unit the market uses, and a spread quoted
    in percent gets misread by a factor of a hundred.

    Takes the curve object rather than a number so the maturity match is
    interpolated rather than eyeballed against the nearest point.
    """
    import bond_data

    if ytm is None:
        return CurvePosition(FAIR, detail="No yield to compare.")
    months = years_to_maturity * 12.0
    curve_yield, method = bond_data.interpolate_yield(curve, months)
    if curve_yield is None:
        return CurvePosition(
            FAIR, bond_yield_pct=ytm * 100.0, method=method,
            detail=("The curve loaded does not span this maturity, so there "
                    "is nothing to compare against."))
    spread_bps = (ytm * 100.0 - curve_yield) * 100.0
    reference = (f"the {years_to_maturity:.1f}-year point on the treasury "
                 f"curve ({curve_yield:.2f}%), taken by {method} "
                 "interpolation")
    if spread_bps > CURVE_TOLERANCE_BPS:
        label = CHEAP
        detail = f"Yields {abs(spread_bps):.0f}bp MORE than {reference}."
    elif spread_bps < -CURVE_TOLERANCE_BPS:
        label = RICH
        detail = f"Yields {abs(spread_bps):.0f}bp LESS than {reference}."
    else:
        # No "more/less than" here: within the tolerance the bond is ON
        # the curve, and phrasing it as a comparison read as
        # "4bp the same as than", which is how this was caught.
        label = FAIR
        detail = (f"Sits on {reference}, within "
                  f"{CURVE_TOLERANCE_BPS:.0f}bp.")
    if label == CHEAP:
        detail += (" That extra yield is compensation for something — "
                   "credit, liquidity or an embedded option.")
    return CurvePosition(label, ytm * 100.0, curve_yield, spread_bps,
                         method, detail)


# --- bond funds ---------------------------------------------------------------

@dataclass(frozen=True)
class FundOutlook:
    yield_pct: Optional[float]
    duration: Optional[float]
    rate_change_bps: float
    price_change_pct: Optional[float]
    total_return_pct: Optional[float]
    detail: str = ""


def fund_expected_return(yield_pct: Optional[float],
                         duration: Optional[float],
                         rate_change_bps: float) -> FundOutlook:
    """The task's "current yield - duration x expected rate change".

    `duration` must be the MEASURED one from bond_data.empirical_duration,
    not the provider's field — that field ranks a 20-year treasury fund
    below a 1-3 year one, so an expected return built on it would have
    the wrong sign of sensitivity.

    A one-year horizon is implied: the yield is an annual figure and the
    price move is treated as immediate.
    """
    if duration is None or yield_pct is None:
        return FundOutlook(yield_pct, duration, rate_change_bps, None, None,
                           "Needs both a yield and a measured duration.")
    price_change = -duration * (rate_change_bps * BASIS_POINT) * 100.0
    total = yield_pct + price_change
    direction = "rise" if rate_change_bps > 0 else "fall"
    if rate_change_bps == 0:
        detail = (f"With rates unchanged the return is the {yield_pct:.2f}% "
                  "yield alone.")
    else:
        detail = (f"A {abs(rate_change_bps):.0f}bp {direction} moves the price "
                  f"{price_change:+.2f}%, against a {yield_pct:.2f}% yield — "
                  f"{total:+.2f}% over a year, before any credit move.")
    return FundOutlook(yield_pct, duration, rate_change_bps, price_change,
                       total, detail)


def breakeven_rate_move_bps(yield_pct: Optional[float],
                            duration: Optional[float]) -> Optional[float]:
    """How far rates can rise before the yield stops covering the loss.

    The single most useful number for a bond fund holder, and it is not
    in the task: a 4% yield with duration 13 is wiped out by a 31bp move,
    which is one ordinary week.
    """
    if not duration or yield_pct is None or duration <= 0:
        return None
    return (yield_pct / duration) * 100.0
