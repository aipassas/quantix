"""Screen the currency board on rates, valuation and risk.

FOURTEEN PAIRS, NOT A HUNDRED. The task asks to fetch "100+ currency
pairs in under 3 seconds". There are eight currencies here with a policy
rate, a PPP factor and a liquid quote, which makes 28 possible pairs of
which about half are ever quoted the way a trader states them. The
fourteen carried are the seven majors, the five yen crosses that carry
trades actually use, and two euro crosses. Adding exotics would add rows
whose spreads and data quality this build cannot vouch for.

EVERY FILTER READS SOMETHING MEASURED. The rate differential comes from
BIS policy rates, the PPP gap from World Bank factors, volatility and
skew from the price series. What is NOT offered is a filter with nothing
behind it:

  - "countries with rising rates" needs a rate PATH, and BIS publishes a
    level. The economic calendar shows scheduled decisions instead,
    which is the forward-looking half that can be sourced.
  - "correlation to USD safe-haven flows" would need a risk-appetite
    index. The pair-to-pair correlations on each pair's own page are the
    honest version.

A PAIR THAT CANNOT BE JUDGED IS NOT ONE THAT FAILED, and the count of
unjudged rows is returned alongside the matches — the same discipline as
the crypto and commodity screeners.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

import forex_data
import forex_risk
import forex_valuation
from logging_setup import get_logger, log_event, log_exception

logger = get_logger(__name__)

UNIVERSE_TTL_SECONDS = 900
MAX_RESULTS_SHOWN = 30


@dataclass(frozen=True)
class PairRow:
    code: str
    label: str
    symbol: str
    base: str
    quote: str
    spot: Optional[float] = None
    change_1d_pct: Optional[float] = None
    return_1y_pct: Optional[float] = None
    volatility_pct: Optional[float] = None
    # Named to match commodity_screener.CommodityRow: the
    # 52-week-range quick stat is shared between the classes and
    # resolves through a single alias map.
    range_position_pct: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    rate_differential_pct: Optional[float] = None
    carry_ratio: Optional[float] = None
    ppp_deviation_pct: Optional[float] = None
    skew: Optional[float] = None
    funded_in_haven: bool = False

    @property
    def carry_positive(self) -> Optional[bool]:
        if self.rate_differential_pct is None:
            return None
        return self.rate_differential_pct > 0


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    kind: str = "number"       # number | percent | text
    help_text: str = ""


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("rate_differential_pct", "Rate differential", "percent",
               "Base policy rate minus quote, in percentage points. "
               "Positive means holding the pair earns interest."),
    MetricSpec("carry_ratio", "Carry per unit of vol", "number",
               "The differential divided by annual volatility. A large "
               "carry on a wild pair is not the same trade as the same "
               "carry on a calm one."),
    MetricSpec("ppp_deviation_pct", "Deviation from PPP", "percent",
               "How far spot sits above or below the purchasing-power "
               "rate. Twenty percent is ordinary and can persist."),
    MetricSpec("volatility_pct", "Volatility", "percent",
               "Annualised over 260 days from the last year of prices."),
    MetricSpec("skew", "Return skew", "number",
               "Over the last year of returns. Negative means losses "
               "arrive larger than gains — the carry trade's "
               "signature. Sensitive to the window and to single "
               "prints."),
    MetricSpec("return_1y_pct", "1-year return", "percent"),
    MetricSpec("change_1d_pct", "1-day change", "percent"),
    MetricSpec("range_position_pct", "Position in 52w range", "percent",
               "0 at the 52-week low, 100 at the high."),
    MetricSpec("base", "Base currency", "text"),
    MetricSpec("quote", "Quote currency", "text"),
)
METRICS_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}

NUMERIC_OPERATORS: Tuple[str, ...] = (">", "<", ">=", "<=")
TEXT_OPERATORS: Tuple[str, ...] = ("is", "is not")


def operators_for(metric: str) -> Tuple[str, ...]:
    """Text metrics take is/is not; numeric ones take comparisons.

    This is why the operator widget carries no Streamlit key: a stored
    ">" raises the moment the metric changes to a currency.
    """
    spec = METRICS_BY_KEY.get(metric)
    return TEXT_OPERATORS if spec and spec.kind == "text" else NUMERIC_OPERATORS


def currencies() -> Tuple[str, ...]:
    return tuple(c.code for c in forex_data.CURRENCIES)


@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    threshold: object


def _value(row: PairRow, metric: str):
    value = getattr(row, metric, None)
    spec = METRICS_BY_KEY.get(metric)
    if spec is not None and spec.kind == "text":
        return value
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _passes(value, criterion: Criterion) -> Optional[bool]:
    if value is None:
        return None
    spec = METRICS_BY_KEY.get(criterion.metric)
    if spec is not None and spec.kind == "text":
        same = str(value).strip().upper() == str(criterion.threshold).strip().upper()
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
    row: PairRow

    @property
    def label(self) -> str:
        return self.row.label


def run(rows: Sequence[PairRow],
        criteria: Sequence[Criterion]) -> Tuple[List[Match], int]:
    matches: List[Match] = []
    unjudged = 0
    for row in rows:
        verdicts = [_passes(_value(row, c.metric), c) for c in criteria]
        if any(v is None for v in verdicts):
            unjudged += 1
            continue
        if all(verdicts):
            matches.append(Match(row))
    matches.sort(key=lambda m: -(m.row.rate_differential_pct
                                 if m.row.rate_differential_pct is not None
                                 else -99))
    log_event(logger, logging.INFO, "forex_screener.run",
              universe=len(rows), criteria=len(criteria),
              matches=len(matches), unjudged=unjudged)
    return matches, unjudged


# --- presets ------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    criteria: Tuple[Criterion, ...]
    description: str


# The task names "Carry Trades", "Risk-Off Pairs" and "Growth"
# (commodity-linked). The first maps onto a measured differential. The
# second and third are themes rather than measurements, so they are
# expressed as the currency legs that define them — a filter on a
# currency is a fact, a filter on a theme is an opinion.
PRESETS: Tuple[Preset, ...] = (
    Preset("Positive carry",
           (Criterion("rate_differential_pct", ">", 0.0),),
           "Pairs where holding the base currency is paid interest "
           "rather than charged it. Being paid is not the same as being "
           "right: check the skew before sizing."),
    Preset("Carry worth the risk",
           (Criterion("rate_differential_pct", ">", 0.0),
            Criterion("carry_ratio", ">", 0.25)),
           "Positive carry worth at least a quarter of the pair's own "
           "annual volatility, so the interest is not dwarfed by the "
           "swing taken to earn it."),
    Preset("Yen-funded",
           (Criterion("quote", "is", "JPY"),),
           "Pairs funded in the lowest-yielding major. These are the "
           "positions that close hardest when risk appetite turns, "
           "because the funding leg is itself a haven."),
    Preset("Below purchasing power",
           (Criterion("ppp_deviation_pct", "<", -10.0),),
           "The base currency trades more than 10% under its PPP rate. "
           "A slow anchor, not a signal — gaps this size last years."),
    Preset("Calm pairs",
           (Criterion("volatility_pct", "<", 7.0),),
           "Annualised volatility under 7%. A sizing input rather than a "
           "recommendation."),
    Preset("Negative skew",
           (Criterion("skew", "<", -0.1),),
           "Pairs whose losses have arrived larger than their gains. "
           "Shown with the worst and best day on each pair's own page, "
           "because skew is fragile against a single bad print."),
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
def load_universe() -> Tuple[Tuple[PairRow, ...], Optional[str]]:
    """One batch download plus the rate and PPP tables.

    Three fetches for the whole board rather than three per pair — the
    rates and PPP calls are cached and cover every currency at once.
    """
    import yfinance as yf

    symbols = [p.symbol for p in forex_data.PAIRS]
    try:
        data = yf.download(symbols, period="1y", progress=False,
                           auto_adjust=True, group_by="column")
    except Exception as exc:                       # noqa: BLE001
        log_exception(logger, "forex_screener.download_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return (), ("The currency board could not be loaded — the price "
                    "download did not answer.")

    closes = (data["Close"]
              if isinstance(data, pd.DataFrame) and "Close" in data else None)
    if closes is None or closes.empty:
        return (), "No prices were returned for any pair."

    rates = forex_data.load_policy_rates()
    factors = forex_data.load_ppp()

    rows: List[PairRow] = []
    for pair in forex_data.PAIRS:
        series = (pd.Series(closes[pair.symbol]).dropna()
                  if pair.symbol in closes else pd.Series(dtype="float64"))
        spot = change_1d = return_1y = volatility = position = None
        high = low = skew = None
        if not series.empty:
            spot = float(series.iloc[-1])
            if len(series) > 1 and series.iloc[-2]:
                change_1d = 100.0 * (spot / float(series.iloc[-2]) - 1.0)
            first = float(series.iloc[0])
            if first:
                return_1y = 100.0 * (spot / first - 1.0)
            returns = series.pct_change().dropna()
            if len(returns) > 30:
                volatility = float(
                    returns.std() * (forex_risk.BARS_PER_YEAR ** 0.5) * 100.0)
                skew = float(returns.skew())
            high, low = float(series.max()), float(series.min())
            if high > low:
                # Clamped: when the latest price IS the window high the
                # arithmetic returns 100.00000000000001, and "position
                # in the range" is by definition inside it.
                position = min(100.0, max(
                    0.0, 100.0 * (spot - low) / (high - low)))

        differential = rates.differential(pair.base, pair.quote) if rates.ok else None
        ratio = (differential / volatility
                 if differential is not None and volatility else None)
        valuation = forex_valuation.ppp_valuation(pair, spot, factors)
        rows.append(PairRow(
            code=pair.code, label=pair.label, symbol=pair.symbol,
            base=pair.base, quote=pair.quote, spot=spot,
            change_1d_pct=change_1d, return_1y_pct=return_1y,
            volatility_pct=volatility, range_position_pct=position,
            high_52w=high, low_52w=low,
            rate_differential_pct=differential, carry_ratio=ratio,
            ppp_deviation_pct=(valuation.deviation_pct
                               if valuation.ok else None),
            skew=skew,
            funded_in_haven=pair.quote in forex_data.SAFE_HAVENS))

    log_event(logger, logging.INFO, "forex_screener.universe_loaded",
              pairs=len(rows))
    return tuple(rows), None


TABLE_COLUMNS: Tuple[str, ...] = (
    "Pair", "Spot", "1d %", "1y %", "Carry %", "Carry/vol", "PPP gap %",
    "Volatility %", "Skew", "52w range %",
)

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "Spot", "1d %", "1y %", "Carry %", "Carry/vol", "PPP gap %",
    "Volatility %", "Skew", "52w range %",
)


def results_frame(matches: Sequence[Match]):
    """Every numeric column coerced, because a column no row reports
    comes back object dtype and sorts as text."""
    frame = pd.DataFrame([{
        "Pair": m.row.label,
        "Spot": m.row.spot,
        "1d %": m.row.change_1d_pct,
        "1y %": m.row.return_1y_pct,
        "Carry %": m.row.rate_differential_pct,
        "Carry/vol": m.row.carry_ratio,
        "PPP gap %": m.row.ppp_deviation_pct,
        "Volatility %": m.row.volatility_pct,
        "Skew": m.row.skew,
        "52w range %": m.row.range_position_pct,
    } for m in matches], columns=list(TABLE_COLUMNS))
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def column_config():
    """Percent columns use a printf format, NOT format="percent", which
    multiplies the stored value by 100."""
    import streamlit as st

    return {
        "Spot": st.column_config.NumberColumn(
            "Spot", format="%.4f",
            help="Units of the quote currency per one of the base. Not "
                 "comparable across rows."),
        "1d %": st.column_config.NumberColumn("1d %", format="%.2f%%"),
        "1y %": st.column_config.NumberColumn("1y %", format="%.1f%%"),
        "Carry %": st.column_config.NumberColumn(
            "Carry %", format="%+.2f%%",
            help="Base policy rate minus quote. Positive means holding "
                 "the pair earns interest."),
        "Carry/vol": st.column_config.NumberColumn(
            "Carry/vol", format="%.2f",
            help="Carry divided by annual volatility."),
        "PPP gap %": st.column_config.NumberColumn(
            "PPP gap %", format="%+.1f%%",
            help="Spot against the purchasing-power rate. Twenty "
                 "percent is ordinary and can persist for years."),
        "Volatility %": st.column_config.NumberColumn(
            "Volatility %", format="%.1f%%",
            help="Annualised over 260 days."),
        "Skew": st.column_config.NumberColumn(
            "Skew", format="%+.2f",
            help="Over the last YEAR of returns. Negative means losses "
                 "have arrived larger than gains. Skew is fragile "
                 "against a single bad print and shifts with the "
                 "window — each pair's own page measures it over five "
                 "years and shows the extremes behind it, so the two "
                 "figures can differ."),
        "52w range %": st.column_config.NumberColumn(
            "52w range %", format="%.0f%%"),
    }
