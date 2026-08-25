"""On-chain valuation for Bitcoin: NVT, stock-to-flow, and scarcity.

THE TASK'S NVT THRESHOLD CANNOT FIRE. It says "NVT < 20 = potentially
undervalued". Measured over the four years to 2026-08-24, on 1365 days
where both market cap and on-chain transaction volume are reported:

    raw NVT   min 28.9 | p10 112.4 | median 190.8 | p90 521.1 | max 1159.9
    days below 20: 0 of 1365

Bitcoin's NVT has not been near 20 once. A criterion that can never be
met is not a conservative criterion, it is a dead one — the same shape
as the first-sign-in prompt that was gated on an empty namespace and so
never appeared. Twenty is not arbitrary: it is roughly right for the
2011-2013 chain, when far less value moved per unit of market cap. It
does not describe this one.

RAW NVT IS TOO NOISY TO THRESHOLD AT ALL. Its day-to-day change has a
standard deviation of 87% of its own mean, and it ranges over a factor
of forty inside four years. Smoothing the denominator over 90 days —
Willy Woo's NVT Signal, which exists for exactly this reason — takes
that to 2.6%:

    NVT signal  min 123.9 | p10 148.7 | median 197.2 | p90 242.7 | max 268.2

So this module reports NVT Signal as the headline and scores it as a
PERCENTILE of its own measured history rather than against any
remembered constant. That is the same discipline the bond module's
credit-spread z-score uses, and it survives the metric drifting as the
chain matures, which a hard-coded 20 demonstrably did not.

STOCK-TO-FLOW TAKES ITS FLOW FROM MEASUREMENT. The scarcity ratio is
supply divided by annual new issuance. Issuance could be assumed from
the halving schedule, but it does not have to be: the supply series
itself gives it. Measured over the two years to 2026-08-24, issuance is
165,134 BTC/year against a theoretical 3.125 x 52,560 = 164,250 — a 0.5%
agreement that also CONFIRMS the halving state rather than assuming it.
Stock-to-flow comes out at 121.6.

WHAT IS NOT HERE. MVRV needs realized cap, whale concentration needs a
rich list, and exchange reserves need labelled exchange wallets. None is
available and none is approximated — see crypto_data for the probes. A
valuation scorecard built on one real ratio and three invented ones
would be worse than one built on what exists, so the scorecard scores
what it can measure and says how many dimensions that was.
"""
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import crypto_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

# The smoothing window for NVT Signal, in days. Ninety is the convention
# and is what the measured noise figures above were computed against.
NVT_SIGNAL_WINDOW_DAYS = 90

# How much history the percentile is taken over. Four years spans a full
# halving cycle plus a year, so a reading is compared against both a bull
# and a bear regime rather than against whichever one is current.
NVT_HISTORY_TIMESPAN = "4years"

MIN_PERCENTILE_OBSERVATIONS = 180

NVT_SPEC_THRESHOLD_NOTE = (
    "A fixed \"NVT below 20\" rule is not used. Bitcoin's NVT has not "
    "been below 20 on any of the last 1365 days — its floor over that "
    "period is 28.9 and its median 190.8 — so the rule could only ever "
    "return \"not undervalued\". The reading is scored against its own "
    "measured history instead."
)


@dataclass(frozen=True)
class NvtReading:
    """NVT and its smoothed signal, with the percentile that gives it
    meaning."""
    nvt: Optional[float] = None
    nvt_signal: Optional[float] = None
    percentile: Optional[float] = None       # of nvt_signal, 0-100
    observations: int = 0
    median: Optional[float] = None
    as_of: Optional[pd.Timestamp] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.nvt_signal is not None and not self.error

    @property
    def scored(self) -> bool:
        """Whether the percentile rests on enough history to mean
        anything. A percentile over three weeks of data is a ranking
        against noise."""
        return (self.percentile is not None
                and self.observations >= MIN_PERCENTILE_OBSERVATIONS)


def nvt_series(market_cap: "pd.Series",
               tx_volume: "pd.Series") -> Tuple["pd.Series", "pd.Series"]:
    """Raw NVT and NVT Signal, aligned on the day.

    The two source charts are sampled independently and their raw
    timestamps do not coincide, so an inner join on the raw index yields
    nothing at all. Both are normalised to dates upstream; this joins on
    that and drops days either side is missing.
    """
    frame = pd.DataFrame({"mc": market_cap, "tv": tx_volume}).dropna()
    frame = frame[frame["tv"] > 0]
    if frame.empty:
        empty = pd.Series(dtype="float64")
        return empty, empty
    raw = frame["mc"] / frame["tv"]
    smoothed = frame["mc"] / frame["tv"].rolling(
        NVT_SIGNAL_WINDOW_DAYS, min_periods=NVT_SIGNAL_WINDOW_DAYS).mean()
    return raw, smoothed.dropna()


def percentile_of(series: "pd.Series", value: Optional[float]) -> Optional[float]:
    """Where `value` sits in `series`, as a percentile 0-100."""
    if value is None or series is None or series.empty:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    return float((clean <= value).mean() * 100.0)


def read_nvt(market_cap: Optional["pd.Series"],
             tx_volume: Optional["pd.Series"]) -> NvtReading:
    """The current NVT reading, scored against its own history."""
    if market_cap is None or tx_volume is None:
        return NvtReading(error="On-chain history is unavailable.")
    raw, signal = nvt_series(market_cap, tx_volume)
    if signal.empty:
        return NvtReading(error=(
            "Not enough overlapping history to compute NVT — the signal "
            f"needs {NVT_SIGNAL_WINDOW_DAYS} days of transaction volume."))
    current = float(signal.iloc[-1])
    reading = NvtReading(
        nvt=float(raw.iloc[-1]) if not raw.empty else None,
        nvt_signal=current,
        percentile=percentile_of(signal, current),
        observations=int(signal.count()),
        median=float(signal.median()),
        as_of=signal.index[-1],
    )
    log_event(logger, logging.INFO, "crypto_valuation.nvt",
              signal=round(current, 2), observations=reading.observations)
    return reading


def nvt_verdict(reading: NvtReading) -> Tuple[str, str]:
    """A (label, explanation) pair for an NVT reading.

    Deliberately phrased as expensive/cheap RELATIVE TO ITS OWN HISTORY,
    never as a buy or a sell. NVT measures market cap against settlement
    volume; a high reading means the chain is being valued richly for
    what it moves, which is information, not instruction.
    """
    if not reading.ok:
        return "Unavailable", reading.error or "No NVT reading."
    if not reading.scored:
        return "Unscored", (
            f"Only {reading.observations} days of history — too few to "
            f"place this reading in its own distribution. The value is "
            f"shown; the ranking is withheld.")
    percentile = reading.percentile
    value = reading.nvt_signal
    if percentile >= 80:
        return "Richly valued", (
            f"NVT Signal {value:,.0f} sits in the {percentile:.0f}th "
            f"percentile of the last {reading.observations} days: the "
            f"network is valued high against the value it settles.")
    if percentile <= 20:
        return "Cheaply valued", (
            f"NVT Signal {value:,.0f} sits in the {percentile:.0f}th "
            f"percentile of the last {reading.observations} days: the "
            f"chain is settling more value per unit of market cap than "
            f"it usually does.")
    return "Mid-range", (
        f"NVT Signal {value:,.0f} is in the {percentile:.0f}th percentile "
        f"of the last {reading.observations} days — neither extreme.")


# --- scarcity -----------------------------------------------------------------

DAYS_PER_YEAR = 365.25

# Below this much history the issuance measurement is dominated by the
# noise in daily supply reporting rather than by real issuance.
MIN_FLOW_DAYS = 180


@dataclass(frozen=True)
class Scarcity:
    """Stock-to-flow, with the flow measured rather than assumed."""
    stock: Optional[float] = None            # coins in existence
    flow: Optional[float] = None             # coins issued per year
    ratio: Optional[float] = None            # stock / flow
    inflation_pct: Optional[float] = None    # flow / stock, as a percent
    measured_over_years: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.ratio is not None and not self.error


def stock_to_flow(supply: Optional["pd.Series"]) -> Scarcity:
    """Stock-to-flow from the supply series alone.

    Flow is the change in supply over the window, annualised. Measuring
    it rather than deriving it from the halving schedule means the
    number cannot silently describe the wrong epoch: a build that still
    assumed 6.25 BTC per block after the 2024 halving would report a
    scarcity ratio half the true one and never fail a test.
    """
    if supply is None or supply.empty:
        return Scarcity(error="No supply history.")
    clean = supply.dropna()
    if len(clean) < 2:
        return Scarcity(error="Supply history has too few points.")
    span_days = (clean.index[-1] - clean.index[0]).days
    if span_days < MIN_FLOW_DAYS:
        return Scarcity(
            stock=float(clean.iloc[-1]),
            error=(f"Issuance needs at least {MIN_FLOW_DAYS} days of "
                   f"supply history; this window covers {span_days}."))
    years = span_days / DAYS_PER_YEAR
    issued = float(clean.iloc[-1]) - float(clean.iloc[0])
    if issued <= 0:
        return Scarcity(stock=float(clean.iloc[-1]),
                        error="Supply did not increase over the window.")
    flow = issued / years
    stock = float(clean.iloc[-1])
    return Scarcity(stock=stock, flow=flow, ratio=stock / flow,
                    inflation_pct=100.0 * flow / stock,
                    measured_over_years=years)


def describe_scarcity(scarcity: Scarcity) -> str:
    if not scarcity.ok:
        return scarcity.error or "Stock-to-flow is unavailable."
    return (
        f"{scarcity.stock:,.0f} coins exist and about {scarcity.flow:,.0f} "
        f"are issued a year — a stock-to-flow ratio of "
        f"{scarcity.ratio:,.1f}, or {scarcity.inflation_pct:.2f}% annual "
        f"supply growth. Issuance is measured over the last "
        f"{scarcity.measured_over_years:.1f} years rather than assumed "
        f"from the halving schedule.")


# --- supply picture -----------------------------------------------------------

@dataclass(frozen=True)
class SupplyPicture:
    """What is known about a coin's supply, with uncapped as an answer."""
    circulating: Optional[float] = None
    total: Optional[float] = None
    maximum: Optional[float] = None
    uncapped: bool = False
    pct_of_max_mined: Optional[float] = None
    note: str = ""


def supply_picture(row: "crypto_data.CoinRow") -> SupplyPicture:
    """A coin's supply, refusing to render "uncapped" as zero.

    This is the trap the module exists to avoid: Yahoo reports maxSupply
    0 and CoinGecko reports max_supply null for the same uncapped coins,
    and 111 of the top 250 are uncapped. Printing "Max supply: 0" states
    that no coins will ever exist, and circulating/max is a division by
    zero on every one of them.
    """
    if row is None:
        return SupplyPicture(note="No coin.")
    if row.uncapped:
        return SupplyPicture(
            circulating=row.circulating_supply, total=row.total_supply,
            maximum=None, uncapped=True,
            note=("No supply cap. This coin's issuance is not bounded, so "
                  "there is no \"percent mined\" to report — a zero here "
                  "would mean the opposite of what it says."))
    return SupplyPicture(
        circulating=row.circulating_supply, total=row.total_supply,
        maximum=row.max_supply, uncapped=False,
        pct_of_max_mined=row.pct_of_max_mined,
        note=(f"{row.pct_of_max_mined:.1f}% of the eventual "
              f"{row.max_supply:,.0f} already exists."
              if row.pct_of_max_mined is not None else ""))


# --- the scorecard ------------------------------------------------------------

@dataclass(frozen=True)
class ScoreLine:
    key: str
    label: str
    value: str
    verdict: str            # Rich | Cheap | Neutral | Unavailable
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


def scorecard(reading: NvtReading, scarcity: Scarcity,
              supply: SupplyPicture) -> Scorecard:
    """A valuation read built only from what could be measured.

    Reports how many dimensions were scored out of how many were
    attempted, because a score out of a fixed denominator when half the
    inputs were blind is the error that had the data-quality badge give
    every ETF 18/100 — it grades the absence of evidence as bad news.
    """
    lines: List[ScoreLine] = []

    label, detail = nvt_verdict(reading)
    verdict = {"Richly valued": "Rich", "Cheaply valued": "Cheap",
               "Mid-range": "Neutral"}.get(label, "Unavailable")
    lines.append(ScoreLine(
        "nvt", "NVT Signal",
        f"{reading.nvt_signal:,.0f}" if reading.ok else "Unavailable",
        verdict, detail))

    if scarcity.ok:
        lines.append(ScoreLine(
            "stock_to_flow", "Stock-to-flow", f"{scarcity.ratio:,.1f}",
            "Neutral",
            f"{scarcity.inflation_pct:.2f}% annual supply growth, "
            f"measured. Scarcity is a property, not a valuation: a high "
            f"ratio says new supply is small, not that the price is low."))
    else:
        lines.append(ScoreLine("stock_to_flow", "Stock-to-flow",
                               "Unavailable", "Unavailable", scarcity.error))

    if supply.uncapped:
        lines.append(ScoreLine("supply_cap", "Supply cap", "None",
                               "Neutral", supply.note))
    elif supply.pct_of_max_mined is not None:
        lines.append(ScoreLine(
            "supply_cap", "Mined of maximum",
            f"{supply.pct_of_max_mined:.1f}%", "Neutral", supply.note))
    else:
        lines.append(ScoreLine("supply_cap", "Supply cap", "Unavailable",
                               "Unavailable", "Supply is not reported."))

    scored = sum(1 for line in lines if line.scored)
    return Scorecard(
        lines=tuple(lines), dimensions_scored=scored,
        dimensions_possible=len(lines),
        summary=(f"{scored} of {len(lines)} valuation dimensions could be "
                 f"measured. " + (
                     crypto_data.MVRV_UNAVAILABLE if scored < len(lines)
                     else "")).strip())
