"""A screener over bond FUNDS, because individual bonds are not reachable.

WHAT THE TASK ASKS FOR AND WHY THIS IS DIFFERENT. PHASE 2.5 describes
screening individual bonds — CUSIP, issuer, coupon, maturity date,
agency rating, bid-ask — and none of that is in this build: it needs the
licensed feed bond_data.BLOOMBERG_UNAVAILABLE describes. Rather than
ship a screener over invented data, this screens the bond FUNDS that
Yahoo's ETF universe does return, on the fields that are real: yield,
duration, cost, size, credit spread and fund type.

That covers more of the task's own preset list than it might look.
"Treasuries for Safety", "Income Focus", "Short Duration" and "High
Yield" are all expressible over funds; what is lost is the per-bond
ladder builder, which needs individual maturities.

DURATION IS MEASURED, NOT READ. The provider's duration field fails the
iShares ladder test — see bond_data, where it ranks SHY above TLH — so
every duration here is regressed from the fund's own price behaviour
against the 10-year yield. That costs one batched download for the whole
bond universe: 27 funds in 1.16s, measured, which is why the universe is
built in one pass and cached rather than computed per row.

FUND TYPE IS MATCHED ON THE NAME, and says so. "Treasury" appears in
"iShares 7-10 Year Treasury Bond ETF" reliably enough to be useful, and
it is an honest text match rather than a classification field the source
does not provide.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import bond_data
import bond_market
import etf_screener
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("bond_screener")

CACHE_TTL_SECONDS = 86400
DURATION_LOOKBACK = "2y"
MAX_RESULTS_SHOWN = 50

# Fund type, matched on the name. Ordered most specific first: a fund
# named "High Yield Corporate" is high yield, not investment grade, so
# the corporate test must not win it.
TYPE_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("High Yield", ("high yield", "high-yield", "junk", "fallen angel")),
    ("Inflation-Protected", ("tips", "inflation-protected", "inflation protected")),
    ("Municipal", ("municipal", "muni")),
    ("Emerging Market", ("emerging market", "em bond", "emerging markets")),
    ("Convertible", ("convertible",)),
    ("Mortgage", ("mortgage", "mbs")),
    ("Treasury", ("treasury", "govt", "government")),
    ("Corporate (IG)", ("corporate", "credit", "investment grade")),
    ("Aggregate", ("aggregate", "total bond", "core bond", "universal")),
)

BOND_NAME_HINTS: Tuple[str, ...] = (
    "bond", "treasury", "aggregate", "corporate", "municipal", "muni",
    "high yield", "high-yield", "tips", "inflation-protected", "credit",
    "government", "mbs", "mortgage", "duration", "convertible",
)

# THE ETF UNIVERSE ALONE IS NOT ENOUGH, and that was measured rather than
# assumed. Yahoo's `top_etfs_us` screen returned 250 funds of which only
# 40 were bond funds — and 19 of the 24 largest bond ETFs were absent,
# including AGG, BND, LQD, HYG and TLT. A bond screener that cannot see
# the most widely held bond funds in the world is missing its subject, so
# these are seeded in and merged with whatever the screen returns. The
# list is declared here rather than inferred, and duplicates are dropped.
CORE_BOND_FUNDS: Tuple[str, ...] = (
    # Broad market
    "AGG", "BND", "BNDX", "SPAB",
    # Treasury, short to long
    "SGOV", "BIL", "SHV", "SHY", "VGSH", "IEI", "VGIT", "IEF", "TLH",
    "TLT", "VGLT", "GOVT",
    # Inflation-protected
    "TIP", "VTIP", "STIP", "SCHP",
    # Corporate, investment grade
    "LQD", "VCSH", "VCIT", "VCLT", "IGSB", "SPSB",
    # High yield and below
    "HYG", "JNK", "SHYG", "USHY", "ANGL",
    # Municipal, mortgage, emerging
    "MUB", "VTEB", "MBB", "VMBS", "EMB",
)

INDIVIDUAL_BONDS_NOT_SCREENED = (
    "This screens bond FUNDS, not individual bonds. Screening by CUSIP, "
    "issuer, coupon or agency rating needs a licensed bond feed this "
    "build does not have — and a screener over invented data would be "
    "worse than one that says what it covers. Yield, duration, cost, "
    "size, credit spread and fund type are all real here."
)


@dataclass(frozen=True)
class BondFundRow:
    symbol: str
    name: str = ""
    fund_type: str = "Other"
    yield_pct: Optional[float] = None
    duration: Optional[float] = None
    duration_r_squared: Optional[float] = None
    spread_bps: Optional[float] = None
    expense_ratio_pct: Optional[float] = None
    assets: Optional[float] = None
    return_1y_pct: Optional[float] = None

    @property
    def rate_loss_100bp_pct(self) -> Optional[float]:
        """What a 100bp parallel rise costs — the task's own filter."""
        if self.duration is None:
            return None
        return -self.duration


def classify(name: str) -> str:
    """The fund's type, from its name. "Other" when nothing matches."""
    low = (name or "").lower()
    for label, needles in TYPE_RULES:
        if any(needle in low for needle in needles):
            return label
    return "Other"


def looks_like_bond_fund(name: str) -> bool:
    low = (name or "").lower()
    return any(hint in low for hint in BOND_NAME_HINTS)


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _load_seeded(symbols: Tuple[str, ...]) -> List[BondFundRow]:
    """Name, yield, cost and size for the seeded funds.

    One info fetch each — they are not in the screen's own table, so
    there is nothing to read them from. Anything that fails is skipped
    rather than taking the universe down with it.
    """
    out: List[BondFundRow] = []
    if not symbols:
        return out
    try:
        import yfinance as yf
    except Exception:                              # noqa: BLE001
        return out
    for symbol in symbols:
        try:
            info = yf.Ticker(symbol).info or {}
        except Exception:                          # noqa: BLE001
            log_exception(logger, "bond_screener.seed_failed", ticker=symbol)
            continue
        name = str(info.get("longName") or info.get("shortName") or "").strip()
        if not name:
            continue
        # `yield` arrives as a FRACTION here (0.0466) while the screen's
        # own rows are percent-valued — the same 100x trap this codebase
        # keeps meeting. Normalise once, here.
        raw_yield = _number(info.get("yield"))
        yield_pct = (raw_yield * 100.0
                     if raw_yield is not None and abs(raw_yield) < 1.0
                     else raw_yield)
        expense = _number(info.get("netExpenseRatio"))
        out.append(BondFundRow(
            symbol=symbol, name=name, fund_type=classify(name),
            yield_pct=yield_pct,
            expense_ratio_pct=expense if expense else None,
            assets=_number(info.get("totalAssets") or info.get("netAssets"))))
    return out


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_bond_universe() -> Tuple[Tuple[BondFundRow, ...], Optional[str]]:
    """Every bond fund in the ETF universe, with a MEASURED duration.

    One batched price download for the whole set rather than one per
    fund — measured at 1.16s for 27 funds against 27 separate requests.
    Never raises.
    """
    rows, error = etf_screener.load_universe()
    screened = [r for r in (rows or []) if looks_like_bond_fund(r.name)]
    candidates: List[BondFundRow] = [
        BondFundRow(symbol=r.symbol, name=r.name, fund_type=classify(r.name),
                    yield_pct=r.dividend_yield_pct,
                    expense_ratio_pct=r.expense_ratio_pct, assets=r.assets,
                    return_1y_pct=r.return_1y_pct)
        for r in screened]
    seen = {c.symbol for c in candidates}
    candidates.extend(_load_seeded(tuple(s for s in CORE_BOND_FUNDS
                                         if s not in seen)))
    if not candidates:
        return (), error or "No bond funds could be loaded."

    symbols = [r.symbol for r in candidates]
    closes = None
    try:
        import yfinance as yf

        frame = yf.download(symbols + ["^TNX"], period=DURATION_LOOKBACK,
                            progress=False, auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else None
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "bond_screener.prices_failed",
                      error=f"{type(exc).__name__}: {exc}")

    curve = bond_data.load_curve()
    benchmark = (closes["^TNX"]
                 if closes is not None and "^TNX" in closes else None)

    out: List[BondFundRow] = []
    for row in candidates:
        duration = r_squared = None
        if closes is not None and row.symbol in closes and benchmark is not None:
            duration, r_squared, _ = bond_data.empirical_duration(
                closes[row.symbol], benchmark)
        spread = None
        if duration is not None and row.yield_pct is not None:
            spread = bond_market.credit_spread(
                row.symbol, row.yield_pct, duration, curve).spread_bps
        out.append(BondFundRow(
            symbol=row.symbol, name=row.name, fund_type=row.fund_type,
            yield_pct=row.yield_pct, duration=duration,
            duration_r_squared=r_squared, spread_bps=spread,
            expense_ratio_pct=row.expense_ratio_pct, assets=row.assets,
            return_1y_pct=row.return_1y_pct))
    log_event(logger, logging.INFO, "bond_screener.universe",
              funds=len(out),
              with_duration=sum(1 for r in out if r.duration is not None))
    return tuple(out), None


# --- the filters --------------------------------------------------------------

@dataclass(frozen=True)
class BondMetric:
    key: str
    label: str
    unit: str = ""
    decimals: int = 2
    kind: str = "number"        # number | text
    note: str = ""


METRICS: Tuple[BondMetric, ...] = (
    BondMetric("yield_pct", "Distribution yield", "%", 2,
               note="What the fund paid out, not its yield-to-worst."),
    BondMetric("duration", "Duration (years)", "", 2,
               note="Measured from price behaviour, not the provider's "
                    "own field — see bond_data."),
    BondMetric("spread_bps", "Credit spread", "bp", 0,
               note="Yield over the treasury curve at the fund's own "
                    "duration. Biased low; the ranking is the signal."),
    BondMetric("expense_ratio_pct", "Expense ratio", "%", 2),
    BondMetric("assets", "Assets under management", "$", 0),
    BondMetric("return_1y_pct", "1-year return", "%", 1),
    BondMetric("fund_type", "Fund type", "", 0, kind="text",
               note="Matched on the fund's name, which is a different "
                    "thing from a classification field."),
)

METRICS_BY_KEY: Dict[str, BondMetric] = {m.key: m for m in METRICS}


def operators_for(metric_key: str) -> Tuple[str, ...]:
    metric = METRICS_BY_KEY.get(metric_key)
    if metric is not None and metric.kind == "text":
        return tuple(etf_screener.CATEGORICAL_OPERATORS)
    return tuple(etf_screener.OPERATORS)


def fund_types() -> Tuple[str, ...]:
    return tuple(label for label, _ in TYPE_RULES) + ("Other",)


@dataclass(frozen=True)
class BondCriterion:
    metric: str
    operator: str
    threshold: object


@dataclass(frozen=True)
class BondMatch:
    row: BondFundRow
    unmeasured: Tuple[str, ...] = ()


def _passes(row: BondFundRow, criterion: BondCriterion) -> Optional[bool]:
    """True, False, or None when the fund does not report the metric.

    None is NOT False, for the same reason etf_screener documents: a
    fund whose duration could not be measured has not failed a duration
    filter.
    """
    metric = METRICS_BY_KEY.get(criterion.metric)
    if metric is None:
        return None
    value = getattr(row, criterion.metric, None)
    if value is None:
        return None
    if metric.kind == "text":
        test = etf_screener.CATEGORICAL_OPERATORS.get(criterion.operator)
        if test is None:
            return None
        return bool(test(str(value), str(criterion.threshold)))
    number = _number(value)
    threshold = _number(criterion.threshold)
    if number is None or threshold is None:
        return None
    test = etf_screener.OPERATORS.get(criterion.operator)
    if test is None:
        return None
    return bool(test(number, threshold))


def run(rows: Sequence[BondFundRow],
        criteria: Sequence[BondCriterion]
        ) -> Tuple[List[BondMatch], List[BondMatch]]:
    """(passed, set aside for missing data). Same contract as the ETF
    screener, so a reader who has used one has used both."""
    passed: List[BondMatch] = []
    unjudged: List[BondMatch] = []
    for row in rows:
        outcomes = [(c, _passes(row, c)) for c in criteria]
        missing = tuple(
            METRICS_BY_KEY[c.metric].label if c.metric in METRICS_BY_KEY
            else f"unknown filter “{c.metric}”"
            for c, ok in outcomes if ok is None)
        if any(ok is False for _, ok in outcomes):
            continue
        match = BondMatch(row=row, unmeasured=missing)
        (unjudged if missing else passed).append(match)
    return passed, unjudged


# --- presets ------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    criteria: Tuple[BondCriterion, ...]
    detail: str


PRESETS: Tuple[Preset, ...] = (
    Preset("Treasuries for Safety",
           (BondCriterion("fund_type", "is", "Treasury"),
            BondCriterion("duration", "<", 3.0)),
           "Government debt with little rate sensitivity — the closest "
           "thing here to cash that still pays."),
    Preset("Income Focus",
           (BondCriterion("yield_pct", ">", 4.0),
            BondCriterion("assets", ">", 1e9)),
           "Yield above 4% in a fund large enough to trade."),
    Preset("Short Duration (rate hedge)",
           (BondCriterion("duration", "<", 2.0),),
           "Under two years of duration: a 100bp rate rise costs under "
           "2%."),
    Preset("High Yield (risk / reward)",
           (BondCriterion("fund_type", "is", "High Yield"),),
           "Sub-investment-grade credit. The yield is compensation for "
           "default risk, not a free lunch."),
    Preset("Long Duration (rate bet)",
           (BondCriterion("duration", ">", 8.0),),
           "For a view that rates FALL. The same duration that pays on "
           "the way down costs on the way up."),
)

# The task's "Ladder" preset is deliberately absent: a ladder is a set of
# individual bonds with staggered MATURITY dates, and this build has no
# individual bonds. A set of funds at different durations is a different
# instrument with different reinvestment behaviour, and calling it a
# ladder would be the wrong word for it.
LADDER_UNAVAILABLE = (
    "A bond ladder is a set of individual bonds maturing in successive "
    "years, each returning its principal on a known date. Funds do not "
    "mature, so a mix of funds at different durations is not a ladder — "
    "it has no maturity dates and no principal return. Building a real "
    "one needs individual bonds."
)


def format_value(metric_key: str, value) -> str:
    """A cell, or an explicit blank. Never a fabricated zero."""
    metric = METRICS_BY_KEY.get(metric_key)
    if value is None or metric is None:
        return "Not reported"
    if metric.kind == "text":
        return str(value)
    number = _number(value)
    if number is None:
        return "Not reported"
    if metric.unit == "$":
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(number) >= cutoff:
                return f"${number / cutoff:,.1f}{suffix}"
        return f"${number:,.0f}"
    return f"{number:,.{metric.decimals}f}{metric.unit}"


def describe(criterion: BondCriterion) -> str:
    metric = METRICS_BY_KEY.get(criterion.metric)
    label = metric.label if metric else criterion.metric
    if metric is not None and metric.kind == "text":
        return f"{label} {criterion.operator} {criterion.threshold}"
    threshold = _number(criterion.threshold)
    shown = (criterion.threshold if threshold is None
             else format_value(criterion.metric, threshold))
    return f"{label} {criterion.operator} {shown}"
