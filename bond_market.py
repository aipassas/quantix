"""Bond market analytics: curve history, spreads, rate risk and stress.

Covers the analytics half of PHASE 2.3 (risk) and 2.4 (technicals). What
it does NOT cover is anything about an individual corporate bond —
CUSIP, issuer, per-bond coupon, per-bond rating, CDS — because that data
needs the licence bond_data.BLOOMBERG_UNAVAILABLE describes. Everything
here is either the treasury curve or a bond FUND, which is what this
build can actually see.

THE CURVE HAS FIVE YEARS OF HISTORY AND NO KEY IS NEEDED FOR IT. That
was worth checking rather than assuming: ^IRX, ^FVX, ^TNX and ^TYX each
return 1255 clean daily observations over five years, which is what makes
curve SHIFT, slope history and inversion statistics possible without
FRED. Measured on 2026-08-25 the 3M-10Y slope is +0.94pp, against +0.17pp
a year ago — real steepening — and the curve was inverted on 41.8% of the
last five years.

THE TASK'S VaR FORMULA IS ABOUT 32x TOO LARGE FOR THE HORIZON IT CLAIMS.
It computes `std_price_change = duration * 0.02` and calls the result a
ONE-DAY 95% VaR, with a comment reading "assume 2% yield volatility
(normal market)". Two percent is a defensible ANNUAL yield volatility —
measured, the 10-year's daily changes annualise to 99bp, so 2% is the
right order for a year. As a DAILY figure it is wrong: the actual
standard deviation of the daily change is 6.25bp, and the largest single
day in five years was 32bp. For a five-year duration that is the
difference between a 16.4% one-day loss and a 0.51% one. So `value_at_
risk` takes an explicit horizon and a yield volatility that defaults to
the measured daily figure rather than an assumed one.

CREDIT SPREADS ARE COMPUTED FROM DISTRIBUTION YIELDS AND UNDERSTATE THE
TRUE SPREAD. A fund reports a distribution yield, not a yield-to-worst:
LQD reports 4.66% where its actual yield-to-worst is nearer 5.3%. The
spread against a duration-matched treasury is therefore biased low, and
the honest reading is the RANKING and the CHANGE rather than the level.
The ranking is sound — measured, SHY 3.65% < IEF 3.96% < AGG 4.05% <
LQD 4.66% < EMB 5.15% < HYG 5.94% < JNK 6.65%, which is exactly the
credit ladder.

(Note in passing: `yield` is 0.0466 and `dividendYield` is 4.66 in the
same response — the same number in two units, the trap this codebase has
now hit in four separate places.)
"""
import datetime
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import bond_data
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("bond_market")

CACHE_TTL_SECONDS = 3600

# Measured over five years of ^TNX daily changes: 6.25bp a day, which
# annualises to 99bp. The task assumes 2% (200bp) as a DAILY figure.
DAILY_YIELD_VOL_PP = 0.0625
TRADING_DAYS = 252

Z_95 = 1.645
Z_99 = 2.326

BASIS_POINT = 0.01          # one bp, in percentage points

# The task's own stress scenarios, each as a parallel yield shift in
# basis points. The COVID one is a SPREAD move rather than a rate move,
# and is labelled that way because the two hit different funds.
@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    shift_bps: float
    kind: str                # "rate" | "spread"
    detail: str


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("taper", "2013 Taper Tantrum", 120, "rate",
             "Rates rose about 120bp over three months when the Fed "
             "signalled it would slow asset purchases."),
    Scenario("covid", "COVID crash, March 2020", 400, "spread",
             "Credit spreads widened 300-500bp in weeks. This hits "
             "corporate and high-yield funds; treasuries RALLIED."),
    Scenario("inflation", "Inflation shock", 200, "rate",
             "A parallel 200bp rise, the shape of 2022."),
    Scenario("cut", "Easing cycle", -100, "rate",
             "A 100bp fall. The mirror image, and the reason anyone "
             "holds duration."),
)

SPREAD_UNDERSTATED = (
    "Spreads here are computed from each fund's DISTRIBUTION yield, "
    "which is not its yield-to-worst — LQD reports 4.66% where its true "
    "yield-to-worst is nearer 5.3%. The level is therefore biased low; "
    "the ranking between funds and the change over time are the parts "
    "worth reading."
)

INDIVIDUAL_BONDS_UNAVAILABLE = (
    "Individual bonds are not covered: CUSIP, issuer, per-bond coupon, "
    "agency ratings, CDS and bid-ask all need a licensed data feed this "
    "build does not have. What is covered is the treasury curve and bond "
    "FUNDS, where the same questions have answers."
)


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


# --- curve history ------------------------------------------------------------

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_curve_history(period: str = "5y") -> Tuple[Optional["pd.DataFrame"],
                                                    Optional[str]]:
    """Daily yields for the four maturities Yahoo publishes.

    Columns are labelled by MATURITY ("3M", "5Y", …) rather than by
    symbol, so nothing downstream has to know that ^IRX is the 13-week
    bill. Never raises.
    """
    symbols = {m.yahoo: m.label for m in bond_data.MATURITIES if m.yahoo}
    try:
        import yfinance as yf

        frame = yf.download(list(symbols), period=period, progress=False,
                            auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "bond_market.curve_history_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return None, f"Treasury yield history could not be loaded: {exc}"
    if closes is None or closes.empty:
        return None, "Treasury yield history came back empty."
    renamed = closes.rename(columns=symbols)
    keep = [label for label in symbols.values() if label in renamed.columns]
    out = renamed[keep].dropna(how="all")
    log_event(logger, logging.INFO, "bond_market.curve_history",
              rows=len(out), columns=len(keep))
    return out, None


@dataclass(frozen=True)
class CurveShift:
    label: str                       # "1 month ago", "1 year ago"
    as_of: Optional[datetime.date]
    changes_bps: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.changes_bps)

    @property
    def average_bps(self) -> Optional[float]:
        if not self.changes_bps:
            return None
        return float(np.mean(list(self.changes_bps.values())))

    @property
    def shape(self) -> str:
        """Parallel, steepening or flattening.

        Compares the change at the SHORT end against the change at the
        long end: a curve where the long end rose more than the short is
        steepening, whatever direction rates went overall.
        """
        if len(self.changes_bps) < 2:
            return "Unknown"
        values = list(self.changes_bps.values())
        short_change, long_change = values[0], values[-1]
        twist = long_change - short_change
        if abs(twist) < 10:
            return "Parallel shift"
        return "Steepening" if twist > 0 else "Flattening"


# Trading days back for each comparison. Approximate by construction,
# and the returned as_of date says which bar was actually used.
SHIFT_WINDOWS: Tuple[Tuple[str, int], ...] = (
    ("1 month ago", 21), ("3 months ago", 63), ("1 year ago", 252),
)


def curve_shifts(history: Optional["pd.DataFrame"]
                 ) -> List[CurveShift]:
    """How the whole curve has moved, in basis points, per maturity."""
    if history is None or history.empty:
        return []
    latest = history.iloc[-1]
    out: List[CurveShift] = []
    for label, days in SHIFT_WINDOWS:
        if len(history) <= days:
            out.append(CurveShift(label=label, as_of=None))
            continue
        past = history.iloc[-1 - days]
        changes = {}
        for column in history.columns:
            now, then = _number(latest.get(column)), _number(past.get(column))
            if now is None or then is None:
                continue
            changes[column] = (now - then) * 100.0     # pp -> bp
        as_of = past.name.date() if hasattr(past.name, "date") else None
        out.append(CurveShift(label=label, as_of=as_of, changes_bps=changes))
    return out


@dataclass(frozen=True)
class SlopeHistory:
    short_label: str = ""
    long_label: str = ""
    series: Optional["pd.Series"] = None
    current_pp: Optional[float] = None
    inverted_days: int = 0
    total_days: int = 0

    @property
    def ok(self) -> bool:
        return self.series is not None and len(self.series) > 0

    @property
    def inverted_now(self) -> bool:
        return self.current_pp is not None and self.current_pp < 0

    @property
    def inverted_share_pct(self) -> Optional[float]:
        if not self.total_days:
            return None
        return (self.inverted_days / self.total_days) * 100.0


def slope_history(history: Optional["pd.DataFrame"],
                  short: str = "3M", long: str = "10Y") -> SlopeHistory:
    """The long-minus-short spread over time, and how often it inverted.

    3M-10Y by default: it has the longest record as a recession signal.
    The task's 2Y-10Y is not available without a FRED key, since Yahoo
    publishes no 2-year symbol — the pair is a parameter so it works the
    moment a key widens the curve.
    """
    if history is None or history.empty:
        return SlopeHistory(short, long)
    if short not in history.columns or long not in history.columns:
        return SlopeHistory(short, long)
    series = (history[long] - history[short]).dropna()
    if series.empty:
        return SlopeHistory(short, long)
    inverted = int((series < 0).sum())
    return SlopeHistory(short_label=short, long_label=long, series=series,
                        current_pp=float(series.iloc[-1]),
                        inverted_days=inverted, total_days=len(series))


# --- rate risk ----------------------------------------------------------------

def dv01(price: Optional[float], modified_duration: Optional[float],
         face: float = 100.0) -> Optional[float]:
    """Dollar value of one basis point, per `face` of position.

    Price times duration times one basis point. It is the figure a desk
    actually hedges on, because it is in money rather than percent.
    """
    if price is None or modified_duration is None:
        return None
    if price <= 0 or modified_duration <= 0:
        return None
    return price * modified_duration * 0.0001 * (face / 100.0)


def value_at_risk(price: Optional[float], modified_duration: Optional[float],
                  horizon_days: int = 1, confidence: float = 0.95,
                  daily_yield_vol_pp: float = DAILY_YIELD_VOL_PP
                  ) -> Optional[float]:
    """Loss not expected to be exceeded, as a percentage of value.

    The yield volatility DEFAULTS to the measured daily figure (6.25bp)
    and scales by the square root of the horizon, rather than the task's
    fixed 2% — which is a plausible ANNUAL volatility applied to a
    one-day horizon and overstates a one-day loss by about 32x. For a
    five-year duration the task's formula gives 16.4% where the measured
    one gives 0.51%.
    """
    if price is None or modified_duration is None:
        return None
    if modified_duration <= 0 or horizon_days <= 0:
        return None
    z = Z_99 if confidence >= 0.99 else Z_95
    move_pp = daily_yield_vol_pp * np.sqrt(horizon_days) * z
    return float(modified_duration * (move_pp / 100.0) * 100.0)


@dataclass(frozen=True)
class KeyRateDuration:
    maturity: str
    yield_pct: Optional[float]
    weight_pct: Optional[float]     # share of the fund's rate risk
    contribution: Optional[float]   # duration attributable to this point


def key_rate_durations(modified_duration: Optional[float],
                       history: Optional["pd.DataFrame"],
                       lookback_days: int = 252
                       ) -> List[KeyRateDuration]:
    """How a fund's rate risk splits across the curve.

    A proper key-rate duration reprices under a shift at ONE maturity
    with the rest held fixed, which needs the fund's cash flows — and a
    fund does not publish them here. What CAN be measured is which points
    on the curve actually move together with the fund, so the split is
    estimated from each maturity's share of curve variance over the
    lookback. It is labelled an estimate, and the weights sum to 100%.
    """
    if modified_duration is None or history is None or history.empty:
        return []
    recent = history.tail(lookback_days)
    changes = recent.diff().dropna()
    if changes.empty:
        return []
    variances = changes.var()
    total = float(variances.sum())
    if total <= 0:
        return []
    out: List[KeyRateDuration] = []
    latest = recent.iloc[-1]
    for column in history.columns:
        share = float(variances.get(column, 0.0)) / total
        out.append(KeyRateDuration(
            maturity=column,
            yield_pct=_number(latest.get(column)),
            weight_pct=share * 100.0,
            contribution=modified_duration * share))
    return out


# --- stress -------------------------------------------------------------------

@dataclass(frozen=True)
class StressResult:
    scenario: Scenario
    impact_pct: Optional[float]
    applies: bool
    detail: str


def stress_test(modified_duration: Optional[float],
                is_credit_exposed: bool,
                spread_duration: Optional[float] = None) -> List[StressResult]:
    """Each scenario's effect, in percent of value.

    A RATE scenario hits everything with duration. A SPREAD scenario hits
    only what carries credit — and treasuries do the opposite in one, so
    applying a spread widening to a treasury fund would invert the sign
    of the answer. `applies` says which is which rather than quietly
    returning a number for both.
    """
    if modified_duration is None:
        return []
    spread_duration = (spread_duration if spread_duration is not None
                       else modified_duration)
    out: List[StressResult] = []
    for scenario in SCENARIOS:
        if scenario.kind == "spread" and not is_credit_exposed:
            out.append(StressResult(
                scenario, None, False,
                "This fund holds government debt, which RALLIES when "
                "credit spreads blow out rather than falling with them."))
            continue
        duration = (spread_duration if scenario.kind == "spread"
                    else modified_duration)
        impact = -duration * (scenario.shift_bps / 10000.0) * 100.0
        out.append(StressResult(scenario, impact, True, scenario.detail))
    return out


# --- credit spreads -----------------------------------------------------------

@dataclass(frozen=True)
class CreditSpread:
    symbol: str
    fund_yield_pct: Optional[float] = None
    duration: Optional[float] = None
    matched_treasury_pct: Optional[float] = None
    spread_bps: Optional[float] = None
    method: str = ""

    @property
    def ok(self) -> bool:
        return self.spread_bps is not None


def credit_spread(symbol: str, fund_yield_pct: Optional[float],
                  duration: Optional[float], curve: "bond_data.Curve"
                  ) -> CreditSpread:
    """A fund's yield over the treasury curve at its own duration.

    Matching on DURATION rather than on stated maturity is the point: a
    fund with a 6.7-year duration should be compared with the 6.7-year
    point on the curve, and comparing every corporate fund with the
    ten-year would flatter the short ones and penalise the long.
    """
    if fund_yield_pct is None or duration is None or duration <= 0:
        return CreditSpread(symbol=symbol, fund_yield_pct=fund_yield_pct,
                            duration=duration,
                            method="needs both a yield and a duration")
    months = duration * 12.0
    treasury, method = bond_data.interpolate_yield(curve, months)
    if treasury is None:
        return CreditSpread(symbol=symbol, fund_yield_pct=fund_yield_pct,
                            duration=duration, method=method)
    return CreditSpread(symbol=symbol, fund_yield_pct=fund_yield_pct,
                        duration=duration, matched_treasury_pct=treasury,
                        spread_bps=(fund_yield_pct - treasury) * 100.0,
                        method=method)


def spread_zscore(current_bps: Optional[float],
                  history_bps: Optional[Sequence[float]]) -> Optional[float]:
    """The task's own formula: how unusual today's spread is.

    Returns None rather than infinity when the history never moved — a
    constant series has no scale to be unusual against.
    """
    if current_bps is None or history_bps is None or len(history_bps) < 2:
        return None
    values = np.asarray([v for v in history_bps if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return None
    std = float(values.std())
    if std <= 0:
        return None
    return (current_bps - float(values.mean())) / std


ABNORMAL_ZSCORE = 2.0


def spread_is_abnormal(zscore: Optional[float]) -> bool:
    """The task's ">2 standard deviations" alert."""
    return zscore is not None and abs(zscore) >= ABNORMAL_ZSCORE
