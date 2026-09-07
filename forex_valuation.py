"""Interest-rate parity, purchasing power, and what a carry trade earns.

THE QUOTE CONVENTION IS THE WHOLE GAME, so it is stated once here and
obeyed everywhere. A pair written BASE/QUOTE — EUR/USD, AUD/JPY — is
priced in units of the QUOTE currency per one unit of the BASE. Buying
the pair means holding the base and funding it in the quote.

Getting that backwards does not raise; it silently inverts every result.
The task's own formula, "forward ≈ spot × (1 + r_domestic) / (1 +
r_foreign)", is the right shape and does not say which currency is
domestic. Under this convention the QUOTE currency is domestic:

    F = S x (1 + r_quote x T) / (1 + r_base x T)

The sanity check that pins the direction: AUD pays 4.35% and JPY pays
1.00%, so AUD/JPY must trade at a forward DISCOUNT — about 3.2% below
spot over a year. A currency you are paid to hold is expected to fall by
roughly what you are paid, which is the whole content of uncovered
parity and the reason the carry trade is not free money.

FORWARDS HERE ARE DERIVED, NEVER QUOTED. No free source publishes an FX
forward curve, so these come out of covered interest parity, which is an
arbitrage identity rather than a forecast. A real forward can sit away
from it when cross-currency basis is wide — that gap is a funding
premium, not an error in the arithmetic — and the panel says so.

PPP IS AN ANCHOR, NOT A SIGNAL. The World Bank's conversion factor is a
whole-economy measure that includes goods no one trades across borders,
so market rates depart from it by tens of percent for decades at a time.
The measurements bear that out and also confirm the method: the yen
comes out roughly 37% below its PPP rate and the franc roughly 15%
above, which are the two best-known standing facts in the currency
market. A model that reproduces those is measuring what it claims to.
Deviations are reported as deviations — never as "undervalued, therefore
buy".
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import forex_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

# Horizons the forward is quoted over, in months. The task names 1M, 3M,
# 6M and 1Y.
FORWARD_TENORS: Tuple[Tuple[str, int], ...] = (
    ("1M", 1), ("3M", 3), ("6M", 6), ("1Y", 12),
)

# A carry is only interesting against the volatility you take to earn it.
# Below this the trade is being paid less than a tenth of its own annual
# swing, which is the conventional line for "not worth the risk".
THIN_CARRY_RATIO = 0.10

PARITY_IS_NOT_A_FORECAST = (
    "Covered interest parity is an arbitrage identity, not a view. It "
    "says what a forward MUST be for borrowing in one currency and "
    "lending in the other to be a wash — not where the spot rate will "
    "go. Uncovered parity, which does make that claim, is one of the "
    "most consistently rejected propositions in economics: high-yield "
    "currencies have historically fallen by LESS than the differential, "
    "which is exactly why the carry trade has a return at all."
)


# --- forwards -----------------------------------------------------------------

@dataclass(frozen=True)
class Forward:
    tenor: str
    months: int
    rate: Optional[float] = None
    premium_pct: Optional[float] = None   # forward over spot, annualised
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.rate is not None and not self.error

    @property
    def at_discount(self) -> Optional[bool]:
        return None if self.premium_pct is None else self.premium_pct < 0


def forward_rate(spot: Optional[float], base_rate_pct: Optional[float],
                 quote_rate_pct: Optional[float], months: int,
                 tenor: str = "") -> Forward:
    """Covered interest parity, on the BASE/QUOTE convention above.

    Simple rather than compounded interest, because policy rates are
    quoted as simple annual rates and the tenors here are a year or
    less; compounding them would be a spurious refinement on inputs
    already a month stale.
    """
    label = tenor or f"{months}M"
    if spot is None or spot <= 0:
        return Forward(label, months, error="No spot rate.")
    if base_rate_pct is None or quote_rate_pct is None:
        return Forward(label, months,
                       error=("A policy rate is missing for one leg, so "
                              "parity cannot be applied."))
    years = months / 12.0
    denominator = 1.0 + (base_rate_pct / 100.0) * years
    if denominator <= 0:
        return Forward(label, months, error="Implausible base rate.")
    rate = spot * (1.0 + (quote_rate_pct / 100.0) * years) / denominator
    premium = ((rate / spot - 1.0) / years * 100.0) if years else None
    return Forward(label, months, rate, premium)


def forward_curve(spot: Optional[float], base_rate_pct: Optional[float],
                  quote_rate_pct: Optional[float],
                  tenors: Sequence[Tuple[str, int]] = FORWARD_TENORS
                  ) -> Tuple[Forward, ...]:
    return tuple(forward_rate(spot, base_rate_pct, quote_rate_pct, months,
                              tenor) for tenor, months in tenors)


def describe_forward(forward: Forward, pair: "forex_data.Pair") -> str:
    if not forward.ok:
        return forward.error or "The forward is unavailable."
    direction = "below" if forward.premium_pct < 0 else "above"
    return (
        f"The {forward.tenor} forward is {forward.rate:,.4f}, "
        f"{abs(forward.premium_pct):.2f}%/yr {direction} spot. "
        f"{'Holding' if forward.premium_pct < 0 else 'Funding'} "
        f"{pair.base} against {pair.quote} "
        f"{'earns' if forward.premium_pct < 0 else 'costs'} that much in "
        f"interest, and parity prices the forward to give it straight "
        f"back.")


# --- carry --------------------------------------------------------------------

@dataclass(frozen=True)
class Carry:
    pair_label: str = ""
    differential_pct: Optional[float] = None
    base_rate_pct: Optional[float] = None
    quote_rate_pct: Optional[float] = None
    base_as_of: str = ""
    quote_as_of: str = ""
    volatility_pct: Optional[float] = None
    ratio: Optional[float] = None          # carry per unit of volatility
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.differential_pct is not None and not self.error

    @property
    def positive(self) -> Optional[bool]:
        return None if self.differential_pct is None else self.differential_pct > 0

    @property
    def thin(self) -> Optional[bool]:
        """Whether the carry is small against the risk taken to earn it."""
        if self.ratio is None:
            return None
        return abs(self.ratio) < THIN_CARRY_RATIO


def carry(pair: "forex_data.Pair", rates: "forex_data.PolicyRates",
          volatility_pct: Optional[float] = None) -> Carry:
    """What holding the pair earns in interest, and what it risks.

    The differential alone is half a sentence. A 3.35% carry on a pair
    that swings 9% a year is a different proposition from the same carry
    on one that swings 25%, and the ratio is the standard way of saying
    which.
    """
    if rates is None or not rates.ok:
        return Carry(pair.label,
                     error=(getattr(rates, "error", "")
                            or "Policy rates are unavailable."))
    base, quote = rates.get(pair.base), rates.get(pair.quote)
    if base is None or quote is None or not (base.ok and quote.ok):
        missing = [c for c, r in ((pair.base, base), (pair.quote, quote))
                   if r is None or not r.ok]
        return Carry(pair.label,
                     error=(f"No policy rate for {', '.join(missing)}."))
    # Through PolicyRates.differential rather than subtracting again
    # here: two implementations of one rule can disagree, and a poison
    # run proved it — inverting the shared one left this path happily
    # computing the old answer.
    differential = rates.differential(pair.base, pair.quote)
    if differential is None:
        return Carry(pair.label,
                     error="The rate differential could not be computed.")
    ratio = (differential / volatility_pct
             if volatility_pct and volatility_pct > 0 else None)
    log_event(logger, logging.INFO, "forex_valuation.carry",
              pair=pair.code, differential=round(differential, 3))
    return Carry(pair.label, differential, base.rate_pct, quote.rate_pct,
                 base.as_of, quote.as_of, volatility_pct, ratio)


def describe_carry(reading: Carry) -> str:
    if not reading.ok:
        return reading.error or "Carry is unavailable."
    direction = "earns" if reading.differential_pct > 0 else "costs"
    text = (
        f"Holding {reading.pair_label} {direction} "
        f"{abs(reading.differential_pct):.2f}%/yr in interest — "
        f"{reading.base_rate_pct:.2f}% against {reading.quote_rate_pct:.2f}%"
        f" (policy rates as of {reading.base_as_of} and "
        f"{reading.quote_as_of}).")
    if reading.ratio is not None:
        text += (f" That is {abs(reading.ratio):.2f} of a year's "
                 f"volatility ({reading.volatility_pct:.1f}%)")
        text += (", which is thin against the swing you take to earn it."
                 if reading.thin else ".")
    return text


# --- purchasing power ---------------------------------------------------------

@dataclass(frozen=True)
class PppValuation:
    pair_label: str = ""
    spot: Optional[float] = None
    ppp_rate: Optional[float] = None
    deviation_pct: Optional[float] = None    # spot over PPP
    year: str = ""
    base_country: str = ""
    quote_country: str = ""
    proxy_note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.deviation_pct is not None and not self.error

    @property
    def verdict(self) -> str:
        """Whether the base currency is dear or cheap against the quote.

        Bands are wide on purpose: PPP deviations of twenty percent are
        ordinary and can persist for a decade, so a five-percent gap is
        not a finding.
        """
        if not self.ok:
            return "Unavailable"
        if self.deviation_pct >= 20:
            return "Well above PPP"
        if self.deviation_pct >= 10:
            return "Above PPP"
        if self.deviation_pct <= -20:
            return "Well below PPP"
        if self.deviation_pct <= -10:
            return "Below PPP"
        return "Near PPP"


def ppp_valuation(pair: "forex_data.Pair", spot: Optional[float],
                  factors: "forex_data.PppFactors") -> PppValuation:
    """The pair's PPP rate, and how far spot sits from it.

    A conversion factor is local currency per international dollar, so
    the QUOTE-per-BASE rate is factor(quote) / factor(base). Inverting
    that is the easy mistake and it turns every undervaluation into an
    overvaluation.
    """
    if factors is None or not factors.ok:
        return PppValuation(pair.label,
                            error=(getattr(factors, "error", "")
                                   or "PPP factors are unavailable."))
    base, quote = factors.get(pair.base), factors.get(pair.quote)
    if base is None or quote is None or not (base.ok and quote.ok):
        missing = [c for c, f in ((pair.base, base), (pair.quote, quote))
                   if f is None or not f.ok]
        return PppValuation(pair.label,
                            error=f"No PPP factor for {', '.join(missing)}.")
    if spot is None or spot <= 0:
        return PppValuation(pair.label, error="No spot rate.")
    ppp_rate = quote.factor / base.factor
    deviation = 100.0 * (spot / ppp_rate - 1.0)
    proxy = " ".join(n for n in (base.note, quote.note) if n).strip()
    return PppValuation(pair.label, spot, ppp_rate, deviation,
                        base.year or quote.year, base.country,
                        quote.country, proxy)


def describe_ppp(valuation: PppValuation) -> str:
    if not valuation.ok:
        return valuation.error or "No PPP comparison is available."
    base_label = valuation.pair_label.split("/")[0]
    direction = "above" if valuation.deviation_pct > 0 else "below"
    text = (
        f"Purchasing power puts {valuation.pair_label} at "
        f"{valuation.ppp_rate:,.4f} against a spot of "
        f"{valuation.spot:,.4f} — {base_label} trades "
        f"{abs(valuation.deviation_pct):.1f}% {direction} its PPP rate "
        f"({valuation.year} factors, {valuation.base_country} and "
        f"{valuation.quote_country}). Deviations of twenty percent are "
        f"ordinary and can last a decade: this is an anchor, not a "
        f"trade.")
    if valuation.proxy_note:
        text += " " + valuation.proxy_note
    return text


# --- the scorecard ------------------------------------------------------------

@dataclass(frozen=True)
class ScoreLine:
    key: str
    label: str
    value: str
    verdict: str
    detail: str = ""

    @property
    def scored(self) -> bool:
        return self.verdict != "Unavailable"


@dataclass(frozen=True)
class Scorecard:
    lines: Tuple[ScoreLine, ...] = ()
    dimensions_scored: int = 0
    dimensions_possible: int = 0
    summary: str = ""

    @property
    def ok(self) -> bool:
        return self.dimensions_scored > 0


def scorecard(reading: Carry, valuation: PppValuation,
              forwards: Sequence[Forward] = ()) -> Scorecard:
    """What could be measured, counted rather than graded.

    Reports how many dimensions resolved out of how many were attempted
    — the error the data-quality badge made was scoring the absence of
    evidence as bad news.
    """
    lines: List[ScoreLine] = []

    if reading.ok:
        lines.append(ScoreLine(
            "carry", "Interest carry",
            f"{reading.differential_pct:+.2f}%/yr",
            "Positive" if reading.positive else "Negative",
            describe_carry(reading)))
    else:
        lines.append(ScoreLine("carry", "Interest carry", "Unavailable",
                               "Unavailable", reading.error))

    if valuation.ok:
        lines.append(ScoreLine(
            "ppp", "Against purchasing power",
            f"{valuation.deviation_pct:+.1f}%", valuation.verdict,
            describe_ppp(valuation)))
    else:
        lines.append(ScoreLine("ppp", "Against purchasing power",
                               "Unavailable", "Unavailable",
                               valuation.error))

    year = next((f for f in forwards if f.months == 12 and f.ok), None)
    if year is not None:
        lines.append(ScoreLine(
            "forward", "1-year forward", f"{year.rate:,.4f}",
            "Discount" if year.at_discount else "Premium",
            PARITY_IS_NOT_A_FORECAST))
    else:
        lines.append(ScoreLine("forward", "1-year forward", "Unavailable",
                               "Unavailable",
                               "Needs a spot rate and both policy rates."))

    scored = sum(1 for line in lines if line.scored)
    return Scorecard(
        lines=tuple(lines), dimensions_scored=scored,
        dimensions_possible=len(lines),
        summary=(f"{scored} of {len(lines)} dimensions could be "
                 f"measured." + ("" if scored == len(lines)
                                 else " " + forex_data.BROKER_FEEDS_UNCONFIGURED)))
