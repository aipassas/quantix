"""Risk for an asset that never closes.

THE ANNUALISATION FACTOR IS WRONG FOR CRYPTO, AND IT IS SHARED. config's
`trading_days_per_year = 252` is documented as "the annualization factor
shared by every risk metric", which is right for equities and wrong here
— crypto markets do not close. Measured over the three years to
2026-08-25, Yahoo returns:

    BTC-USD  1097 bars over 1096 days  -> 365.6 bars/year
    ETH-USD  1097 bars over 1096 days  -> 365.6 bars/year
    SPY       751 bars over 1093 days  -> 251.0 bars/year
    GLD       751 bars over 1093 days  -> 251.0 bars/year

Annualising Bitcoin's daily volatility with 252 gives 36.7%; with 365 it
gives 44.1%. That is not a rounding difference, it is a 20% understatement
of the single headline risk number, and it understates it in the direction
that flatters the asset. `BARS_PER_YEAR` here is 365 and every window in
this module uses it.

CORRELATION AGAINST A MARKET THAT DOES CLOSE. Bitcoin has 365 bars a year
and SPY has 251, so any correlation between them is computed on the ~251
days both trade. That is the only honest set — but it means every weekend
move in crypto is excluded from the comparison, not averaged into it, and
the panel says so. A reader who thinks a 0.4 correlation covers all of
Bitcoin's week is reading more into it than the data supports. The
overlap count is reported alongside every coefficient for the same reason.

WHAT CANNOT BE SOURCED. The task asks for a regulatory risk score by
jurisdiction, smart-contract audit status, and exchange hack history.
None of the three is available: there is no free, structured feed of
regulatory status per token per country, audit reports are PDFs from
dozens of private firms with no index, and hack history is journalism.
Each would be a judgement rendered as a number, which is worse than an
absence — so each is declared unavailable and named. Developer activity
IS sourceable, and stands in as the one maintenance signal this build
can actually measure.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from logging_setup import get_logger, log_event

logger = get_logger(__name__)

# Measured, not assumed — see the module docstring. Crypto trades every
# day, so the annualisation factor is the calendar year.
BARS_PER_YEAR = 365

# The equity factor, kept here only so the comparison note can name it.
EQUITY_BARS_PER_YEAR = 252

ANNUALISATION_NOTE = (
    "Annualised over 365 days, not the 252 used for equities: crypto "
    "trades every day. Using the equity factor would report Bitcoin's "
    "volatility about 20% lower than it is."
)

WEEKEND_OVERLAP_NOTE = (
    "Computed on the days both markets traded. Crypto trades weekends "
    "and equities do not, so weekend moves are excluded from this "
    "comparison rather than averaged into it."
)

REGULATORY_UNAVAILABLE = (
    "Regulatory status by jurisdiction is not scored. There is no free, "
    "structured feed of which tokens are permitted where, and a score "
    "assembled by hand would be an opinion presented as data — on a "
    "question where being wrong has legal consequences for the reader."
)

AUDIT_UNAVAILABLE = (
    "Smart-contract audit status is not reported. Audits are individual "
    "PDFs from dozens of private firms with no common index or machine- "
    "readable result, and an unaudited contract and an unindexed audit "
    "look identical from here."
)

HACK_HISTORY_UNAVAILABLE = (
    "Exchange hack history is not reported. It exists as journalism "
    "rather than as a dataset, and attributing an exchange's breach to "
    "a coin it happened to list would be the wrong unit of analysis."
)

# The windows the task asks for, in days.
VOL_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("7-day", 7), ("30-day", 30), ("90-day", 90), ("1-year", 365),
)


def _returns(closes: Optional["pd.Series"]) -> Optional["pd.Series"]:
    if closes is None:
        return None
    series = pd.Series(closes).dropna()
    if len(series) < 3:
        return None
    return series.pct_change().dropna()


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
        """A window with too little history is UNAVAILABLE, never zero.

        A 365-day window over a coin listed three months ago cannot be
        computed, and reporting it as 0% volatility would be an
        all-clear nobody performed.
        """
        return "Available" if self.ok else "Unavailable"


def volatility_windows(closes: Optional["pd.Series"],
                       windows: Sequence[Tuple[str, int]] = VOL_WINDOWS
                       ) -> Tuple[VolatilityWindow, ...]:
    """Annualised realised volatility over each window.

    A window is computed only when the full window's worth of returns
    exists. Scoring a 365-day volatility off 60 bars would report a
    number whose label is a lie about how much history stands behind it.
    """
    returns = _returns(closes)
    out: List[VolatilityWindow] = []
    for label, days in windows:
        if returns is None or len(returns) < days:
            out.append(VolatilityWindow(label, days, None,
                                        0 if returns is None else len(returns)))
            continue
        window = returns.iloc[-days:]
        out.append(VolatilityWindow(
            label, days,
            float(window.std() * (BARS_PER_YEAR ** 0.5) * 100.0),
            int(len(window))))
    return tuple(out)


# --- drawdown -----------------------------------------------------------------

@dataclass(frozen=True)
class Drawdown:
    max_drawdown_pct: Optional[float] = None      # negative
    peak_date: Optional[pd.Timestamp] = None
    trough_date: Optional[pd.Timestamp] = None
    current_drawdown_pct: Optional[float] = None
    days_since_peak: Optional[int] = None
    recovered: Optional[bool] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.max_drawdown_pct is not None and not self.error


def drawdown_profile(closes: Optional["pd.Series"]) -> Drawdown:
    """Worst peak-to-trough loss, and where the price sits against its
    own high now.

    The current drawdown is measured against the running peak of the
    series supplied, so it answers "how far below its high in THIS
    window" — a 1-year series and an all-time series give different and
    both-correct answers, which is why the caller's period is named on
    screen beside it.
    """
    if closes is None:
        return Drawdown(error="No price history.")
    series = pd.Series(closes).dropna()
    if len(series) < 3:
        return Drawdown(error="Price history has too few points.")
    running_peak = series.cummax()
    drawdowns = (series / running_peak - 1.0) * 100.0
    trough_date = drawdowns.idxmin()
    peak_slice = series.loc[:trough_date]
    peak_date = peak_slice.idxmax()
    current = float(drawdowns.iloc[-1])
    return Drawdown(
        max_drawdown_pct=float(drawdowns.min()),
        peak_date=peak_date, trough_date=trough_date,
        current_drawdown_pct=current,
        days_since_peak=int((series.index[-1] - peak_date).days),
        recovered=bool(current > -0.01),
    )


# --- correlation --------------------------------------------------------------

# Below this many shared days a correlation is noise dressed as a number.
MIN_CORRELATION_DAYS = 60


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
    """Correlation of daily returns over the days BOTH series traded.

    The join is on the intersection, which for crypto against an equity
    benchmark is the equity calendar. Fewer than MIN_CORRELATION_DAYS
    shared days returns unavailable rather than a coefficient: two weeks
    of overlap will produce a number, and it will mean nothing.
    """
    left, right = _returns(closes), _returns(other_closes)
    if left is None or right is None:
        return Correlation(label, symbol, error="Not enough price history.")
    # Normalising to dates first: Yahoo stamps crypto and equity bars
    # with different exchange timezones, so a raw index join silently
    # produces almost no overlap.
    left.index = pd.DatetimeIndex(left.index).tz_localize(None).normalize()
    right.index = pd.DatetimeIndex(right.index).tz_localize(None).normalize()
    frame = pd.DataFrame({"a": left, "b": right}).dropna()
    if len(frame) < MIN_CORRELATION_DAYS:
        return Correlation(label, symbol, observations=len(frame),
                           error=(f"Only {len(frame)} shared trading days; "
                                  f"at least {MIN_CORRELATION_DAYS} are "
                                  f"needed for a meaningful coefficient."))
    coefficient = float(frame["a"].corr(frame["b"]))
    if coefficient != coefficient:
        return Correlation(label, symbol, observations=len(frame),
                           error="Correlation is undefined — a series is flat.")
    return Correlation(label, symbol, coefficient, int(len(frame)))


# The comparisons the task names: stocks, gold, and other crypto. Held
# here so the panel and its test read the same list.
CORRELATION_BENCHMARKS: Tuple[Tuple[str, str], ...] = (
    ("US equities", "SPY"),
    ("Gold", "GLD"),
    ("Bitcoin", "BTC-USD"),
    ("Ethereum", "ETH-USD"),
)


def benchmarks_for(symbol: str) -> Tuple[Tuple[str, str], ...]:
    """The benchmark list with the coin itself removed.

    Correlating Bitcoin with Bitcoin returns 1.00, which is not a
    finding and takes a row that a real comparison could use.
    """
    wanted = str(symbol or "").strip().upper()
    return tuple((label, sym) for label, sym in CORRELATION_BENCHMARKS
                 if sym.upper() != wanted)


# --- leverage -----------------------------------------------------------------

@dataclass(frozen=True)
class LiquidationRead:
    leverage: float
    maintenance_margin_pct: float
    move_to_liquidation_pct: Optional[float] = None
    liquidation_price: Optional[float] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.move_to_liquidation_pct is not None and not self.error


# A typical venue holds back half a percent to one percent on a major
# pair. It is a per-venue, per-tier parameter and this build has no feed
# for it, so it is an input with a stated default rather than a fact.
DEFAULT_MAINTENANCE_MARGIN_PCT = 0.5

LIQUIDATION_NOTE = (
    "Arithmetic, not a venue quote. The distance to liquidation follows "
    "from the leverage and the maintenance margin alone; the margin "
    "varies by exchange and by position size, so treat the default as "
    "illustrative and enter your venue's own."
)


def liquidation_distance(
        price: Optional[float], leverage: float,
        maintenance_margin_pct: float = DEFAULT_MAINTENANCE_MARGIN_PCT,
        long: bool = True) -> LiquidationRead:
    """How far the price can move against a leveraged position.

    A position at L times leverage is wiped out by an adverse move of
    1/L of its notional, less the maintenance margin the venue holds
    back. At 1x there is no liquidation at all, which is reported as
    such rather than as a 100% move.
    """
    if leverage is None or leverage <= 0:
        return LiquidationRead(leverage or 0, maintenance_margin_pct,
                               error="Leverage must be positive.")
    if leverage <= 1:
        return LiquidationRead(
            leverage, maintenance_margin_pct,
            error=("An unleveraged position cannot be liquidated — there "
                   "is no borrowed margin to call."))
    move = (100.0 / leverage) - float(maintenance_margin_pct)
    if move <= 0:
        return LiquidationRead(
            leverage, maintenance_margin_pct,
            error=(f"At {leverage:g}x the maintenance margin of "
                   f"{maintenance_margin_pct:g}% already exceeds the "
                   f"position's own margin."))
    liquidation_price = None
    if price is not None and price > 0:
        factor = (1 - move / 100.0) if long else (1 + move / 100.0)
        liquidation_price = float(price) * factor
    return LiquidationRead(leverage, maintenance_margin_pct, move,
                           liquidation_price)


def liquidation_context(move_pct: Optional[float],
                        windows: Sequence[VolatilityWindow]) -> str:
    """The liquidation distance expressed in the coin's own volatility.

    A 5% move means nothing in isolation. Against a coin whose 30-day
    volatility annualises to 60%, a 5% daily move is about a 1.6 sigma
    day — which is the sentence a reader can act on.
    """
    if move_pct is None:
        return ""
    window = next((w for w in windows if w.days == 30 and w.ok), None)
    if window is None:
        window = next((w for w in windows if w.ok), None)
    if window is None:
        return ""
    daily_sigma = window.annualised_pct / (BARS_PER_YEAR ** 0.5)
    if daily_sigma <= 0:
        return ""
    sigmas = move_pct / daily_sigma
    return (f"About {sigmas:.1f} standard deviations of a typical day, "
            f"measured on this coin's {window.label} volatility "
            f"({window.annualised_pct:.0f}% annualised).")


# --- developer health ---------------------------------------------------------

@dataclass(frozen=True)
class DeveloperHealth:
    commits_4w: Optional[int] = None
    contributors: Optional[int] = None
    stars: Optional[int] = None
    verdict: str = "Unavailable"
    detail: str = ""


# Measured across the four coins probed: 108 (BTC), 41 (ETH), 171 (SOL),
# 0 (DOGE) commits in four weeks. The bands below are set to separate
# those, and a zero is reported as a fact rather than as a failure —
# a mature, feature-complete protocol legitimately goes quiet.
ACTIVE_COMMITS_4W = 20
QUIET_COMMITS_4W = 1


def developer_health(profile) -> DeveloperHealth:
    """Repository activity as the one maintenance signal available.

    This measures the reference implementation's repository, which is
    not the same as the protocol's health: a chain can be secure and
    widely used while its main client is stable and rarely touched.
    Phrased accordingly.
    """
    if profile is None or not getattr(profile, "has_developer_data", False):
        return DeveloperHealth(detail=(
            "No repository data is published for this coin."))
    commits = profile.commits_4w
    contributors = profile.contributors
    if commits is None:
        return DeveloperHealth(
            commits_4w=None, contributors=contributors,
            stars=getattr(profile, "stars", None),
            detail="Commit activity is not reported for this coin.")
    if commits >= ACTIVE_COMMITS_4W:
        verdict, detail = "Actively developed", (
            f"{commits} commits in the last four weeks"
            + (f" across {contributors} historical contributors."
               if contributors else "."))
    elif commits >= QUIET_COMMITS_4W:
        verdict, detail = "Low activity", (
            f"{commits} commits in the last four weeks. Sparse, though a "
            f"stable protocol can legitimately be quiet.")
    else:
        verdict, detail = "Dormant repository", (
            "No commits in the last four weeks. That describes the "
            "reference client's repository, not the network — the chain "
            "can run untouched.")
    return DeveloperHealth(commits, contributors,
                           getattr(profile, "stars", None), verdict, detail)
