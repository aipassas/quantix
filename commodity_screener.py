"""Screen the commodity board on things that were actually measured.

THE UNIVERSE IS SIXTEEN CONTRACTS, NOT A THOUSAND. The task asks to
ingest "100+ commodities"; there are not a hundred liquid futures
markets worth screening, and the sixteen here — five metals, three
energy, six agricultural, two livestock — cover the board a reader
means by "commodities". Padding the list with thinly traded contracts
would lengthen the table and shorten the time each row deserves.

EVERY FILTER READS SOMETHING MEASURED. Curve shape, implied carry, roll
yield and open interest all come from the dated contracts; volatility
and returns from the price history. What is NOT offered is a filter the
data cannot support:

  - "inventory levels" and "supply/demand balance" exist for US crude
    only, so they are shown on the crude page rather than offered as a
    universe-wide filter that fifteen of sixteen rows would be unjudged
    against.
  - "production region / OPEC vs non-OPEC" has no feed. Tagging
    contracts by hand and calling the result a filter would present an
    assumption as data — see commodity_data.GEOPOLITICAL_UNAVAILABLE.

BUILDING THE UNIVERSE IS SIXTEEN CURVES, WHICH IS SLOW. Each curve
probes up to eighteen dated contracts, so a full refresh is a few
hundred quote lookups. The screener therefore loads the CONTINUOUS
symbols for price, return and volatility — one batch download — and
attaches curve-derived columns only for the commodities whose curves are
already cached. A row with no curve yet is UNJUDGED on curve filters
rather than dropped, for the same reason the crypto screener reports its
unjudged count: silently omitting them would report a screen over
sixteen that examined four.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

import commodity_curve
import commodity_data
from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

UNIVERSE_TTL_SECONDS = 900
MAX_RESULTS_SHOWN = 30


@dataclass(frozen=True)
class CommodityRow:
    key: str
    name: str
    sector: str
    unit: str
    symbol: str
    price: Optional[float] = None
    change_1d_pct: Optional[float] = None
    return_1y_pct: Optional[float] = None
    volatility_pct: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    range_position_pct: Optional[float] = None
    open_interest: Optional[int] = None
    # Curve-derived, present only where a curve resolved.
    curve_shape: Optional[str] = None
    carry_pct: Optional[float] = None
    roll_yield_pct: Optional[float] = None

    @property
    def has_curve(self) -> bool:
        return self.curve_shape is not None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    kind: str = "number"       # number | money | percent | text
    help_text: str = ""


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("price", "Price", "number",
               "In the contract's own unit — dollars a barrel, cents a "
               "bushel — so it is not comparable across rows."),
    MetricSpec("change_1d_pct", "1-day change", "percent"),
    MetricSpec("return_1y_pct", "1-year return", "percent"),
    MetricSpec("volatility_pct", "Volatility", "percent",
               "Annualised over 252 days from the last year of prices."),
    MetricSpec("range_position_pct", "Position in 52w range", "percent",
               "0 at the 52-week low, 100 at the high."),
    MetricSpec("open_interest", "Open interest", "number",
               "Contracts held on the front month — the liquidity read."),
    MetricSpec("carry_pct", "Implied carry", "percent",
               "Annualised from the curve. Unavailable until the "
               "commodity's curve has been loaded."),
    MetricSpec("roll_yield_pct", "Roll yield", "percent",
               "Positive in backwardation, negative in contango."),
    MetricSpec("curve_shape", "Curve shape", "text",
               "Contango, Backwardation, Flat or Humped."),
    MetricSpec("sector", "Sector", "text"),
)
METRICS_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}

NUMERIC_OPERATORS: Tuple[str, ...] = (">", "<", ">=", "<=")
TEXT_OPERATORS: Tuple[str, ...] = ("is", "is not")


def operators_for(metric: str) -> Tuple[str, ...]:
    """The operators valid for one metric.

    Text metrics take is/is not and numeric ones take comparisons, which
    is why the screener's operator widget carries no Streamlit key: a
    stored ">" would raise the moment the metric changed to Sector.
    """
    spec = METRICS_BY_KEY.get(metric)
    return TEXT_OPERATORS if spec and spec.kind == "text" else NUMERIC_OPERATORS


def shapes() -> Tuple[str, ...]:
    return (commodity_curve.CONTANGO, commodity_curve.BACKWARDATION,
            commodity_curve.FLAT, commodity_curve.HUMPED)


@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    threshold: object          # float, or str for a text metric


def _value(row: CommodityRow, metric: str):
    value = getattr(row, metric, None)
    spec = METRICS_BY_KEY.get(metric)
    if spec is not None and spec.kind == "text":
        return value
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _passes(value, criterion: Criterion) -> Optional[bool]:
    """True, False, or None for "this row could not be judged"."""
    if value is None:
        return None
    spec = METRICS_BY_KEY.get(criterion.metric)
    if spec is not None and spec.kind == "text":
        same = str(value).strip().lower() == str(criterion.threshold).strip().lower()
        if criterion.operator == "is":
            return same
        if criterion.operator == "is not":
            return not same
        return None
    try:
        threshold = float(criterion.threshold)
    except (TypeError, ValueError):
        return None
    if criterion.operator == ">":
        return value > threshold
    if criterion.operator == "<":
        return value < threshold
    if criterion.operator == ">=":
        return value >= threshold
    if criterion.operator == "<=":
        return value <= threshold
    return None


@dataclass(frozen=True)
class Match:
    row: CommodityRow

    @property
    def name(self) -> str:
        return self.row.name


def run(rows: Sequence[CommodityRow],
        criteria: Sequence[Criterion]) -> Tuple[List[Match], int]:
    """Commodities meeting every criterion, and how many were unjudged."""
    matches: List[Match] = []
    unjudged = 0
    for row in rows:
        verdicts = [_passes(_value(row, c.metric), c) for c in criteria]
        if any(v is None for v in verdicts):
            unjudged += 1
            continue
        if all(verdicts):
            matches.append(Match(row))
    matches.sort(key=lambda m: (m.row.sector, m.row.name))
    log_event(logger, logging.INFO, "commodity_screener.run",
              universe=len(rows), criteria=len(criteria),
              matches=len(matches), unjudged=unjudged)
    return matches, unjudged


# --- presets ------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    criteria: Tuple[Criterion, ...]
    description: str


# The task names "Inflation hedges", "Agricultural plays" and "Energy
# transition". The first two map onto sectors that exist here. The third
# does not: lithium, cobalt and rare earths have no liquid futures
# contract on this board, so an "energy transition" preset would be
# copper alone wearing a thematic label. It is expressed as the metals
# sector instead, which is what the data can honestly support.
PRESETS: Tuple[Preset, ...] = (
    Preset("Inflation hedges",
           (Criterion("sector", "is", commodity_data.METALS),),
           "Precious and industrial metals — the classic store-of-value "
           "trade. Whether each is actually hedging is a question the "
           "dollar correlation on its own page answers."),
    Preset("Agricultural",
           (Criterion("sector", "is", commodity_data.AGRICULTURE),),
           "Grains and softs. Their curves are the ones most likely to "
           "be humped rather than simply sloped, because a harvest "
           "resets supply every year."),
    Preset("Energy",
           (Criterion("sector", "is", commodity_data.ENERGY),),
           "Crude, Brent and natural gas — the only sector here with an "
           "inventory series behind it."),
    Preset("Backwardated",
           (Criterion("curve_shape", "is", commodity_curve.BACKWARDATION),),
           "Curves where nearby contracts trade above deferred ones, "
           "which is what a market short of immediate supply looks "
           "like. Rolling a long position earns rather than costs."),
    Preset("High volatility",
           (Criterion("volatility_pct", ">", 30.0),),
           "Annualised volatility above 30%. A sizing input, not a "
           "recommendation in either direction."),
    Preset("Near 52-week highs",
           (Criterion("range_position_pct", ">", 80.0),),
           "Trading in the top fifth of the last year's range."),
)
PRESETS_BY_NAME: Dict[str, Preset] = {p.name: p for p in PRESETS}


def describe(criterion: Criterion) -> str:
    spec = METRICS_BY_KEY.get(criterion.metric)
    label = spec.label if spec else criterion.metric
    kind = spec.kind if spec else "number"
    if kind == "text":
        return f"{label} {criterion.operator} {criterion.threshold}"
    try:
        threshold = float(criterion.threshold)
    except (TypeError, ValueError):
        return f"{label} {criterion.operator} {criterion.threshold}"
    suffix = "%" if kind == "percent" else ""
    return f"{label} {criterion.operator} {threshold:g}{suffix}"


# --- the universe -------------------------------------------------------------

@st.cache_data(ttl=UNIVERSE_TTL_SECONDS, show_spinner=False)
def load_universe(curves: Optional[Dict[str, str]] = None
                  ) -> Tuple[Tuple[CommodityRow, ...], Optional[str]]:
    """One batch download of the continuous symbols.

    `curves` optionally carries already-computed curve columns keyed by
    commodity, so the screener can show them without re-probing several
    hundred dated contracts on every rerun. Rows without them stay
    UNJUDGED on curve filters rather than being dropped.
    """
    import numpy as np
    import yfinance as yf

    symbols = [c.continuous for c in commodity_data.COMMODITIES]
    try:
        data = yf.download(symbols, period="1y", progress=False,
                           auto_adjust=True, group_by="column")
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "commodity_screener.download_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return (), ("The commodity board could not be loaded — the price "
                    "download did not answer.")

    closes = data["Close"] if isinstance(data, pd.DataFrame) and "Close" in data else None
    if closes is None or closes.empty:
        return (), "No prices were returned for any commodity."

    rows: List[CommodityRow] = []
    for commodity in commodity_data.COMMODITIES:
        symbol = commodity.continuous
        series = (pd.Series(closes[symbol]).dropna()
                  if symbol in closes else pd.Series(dtype="float64"))
        if series.empty:
            rows.append(CommodityRow(commodity.key, commodity.name,
                                     commodity.sector, commodity.unit,
                                     symbol))
            continue
        price = float(series.iloc[-1])
        change_1d = (100.0 * (price / float(series.iloc[-2]) - 1.0)
                     if len(series) > 1 and series.iloc[-2] else None)
        first = float(series.iloc[0])
        return_1y = 100.0 * (price / first - 1.0) if first else None
        returns = series.pct_change().dropna()
        volatility = (float(returns.std() * (252 ** 0.5) * 100.0)
                      if len(returns) > 30 else None)
        high, low = float(series.max()), float(series.min())
        position = (100.0 * (price - low) / (high - low)
                    if high > low else None)
        extra = (curves or {}).get(commodity.key) or {}
        rows.append(CommodityRow(
            key=commodity.key, name=commodity.name, sector=commodity.sector,
            unit=commodity.unit, symbol=symbol, price=price,
            change_1d_pct=change_1d, return_1y_pct=return_1y,
            volatility_pct=volatility, high_52w=high, low_52w=low,
            range_position_pct=position,
            open_interest=extra.get("open_interest"),
            curve_shape=extra.get("curve_shape"),
            carry_pct=extra.get("carry_pct"),
            roll_yield_pct=extra.get("roll_yield_pct")))

    log_event(logger, logging.INFO, "commodity_screener.universe_loaded",
              commodities=len(rows))
    return tuple(rows), None


TABLE_COLUMNS: Tuple[str, ...] = (
    "Commodity", "Sector", "Symbol", "Price", "Unit", "1d %", "1y %",
    "Volatility %", "52w range %", "Curve", "Carry %", "Roll %",
)

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "Price", "1d %", "1y %", "Volatility %", "52w range %", "Carry %",
    "Roll %",
)


def results_frame(matches: Sequence[Match]):
    """The results table, every numeric column coerced to a number.

    A column no row reports comes back object dtype and sorts as text —
    which is exactly how the ETF screener's P/E broke. Curve columns are
    the ones at risk here, because they are absent until a curve loads.
    """
    frame = pd.DataFrame([{
        "Commodity": m.row.name,
        "Sector": m.row.sector,
        "Symbol": m.row.symbol,
        "Price": m.row.price,
        "Unit": m.row.unit,
        "1d %": m.row.change_1d_pct,
        "1y %": m.row.return_1y_pct,
        "Volatility %": m.row.volatility_pct,
        "52w range %": m.row.range_position_pct,
        "Curve": m.row.curve_shape or "Not loaded",
        "Carry %": m.row.carry_pct,
        "Roll %": m.row.roll_yield_pct,
    } for m in matches], columns=list(TABLE_COLUMNS))
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def column_config():
    """Streamlit formats. Percent columns use a printf format, NOT
    format="percent", which multiplies the stored value by 100."""
    import streamlit as st

    return {
        "Price": st.column_config.NumberColumn(
            "Price", format="%.4g",
            help="In the contract's own unit — see the Unit column. "
                 "Prices are not comparable between rows."),
        "1d %": st.column_config.NumberColumn("1d %", format="%.2f%%"),
        "1y %": st.column_config.NumberColumn("1y %", format="%.1f%%"),
        "Volatility %": st.column_config.NumberColumn(
            "Volatility %", format="%.1f%%",
            help="Annualised over 252 trading days."),
        "52w range %": st.column_config.NumberColumn(
            "52w range %", format="%.0f%%",
            help="0 at the 52-week low, 100 at the high."),
        "Curve": st.column_config.TextColumn(
            "Curve",
            help="Shape of the forward curve. \"Not loaded\" means the "
                 "dated contracts have not been probed for this "
                 "commodity yet — not that it has no curve."),
        "Carry %": st.column_config.NumberColumn(
            "Carry %", format="%.2f%%",
            help="Implied annualised cost of carry from the curve."),
        "Roll %": st.column_config.NumberColumn(
            "Roll %", format="%.2f%%",
            help="What rolling a long position earns (positive) or "
                 "costs (negative) a year, before any price move."),
    }
