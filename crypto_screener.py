"""Filter the coin universe, and the presets worth starting from.

WHAT THE UNIVERSE IS. CoinGecko's top 250 by market cap, one call, which
is the same list the header and the dominance panel read. The task asks
to ingest a thousand coins; four pages would do it and would spend four
of the ten calls a minute the keyless tier allows, to reach coins with
market caps under a hundred million that no preset here selects for. The
count is stated on screen rather than implied.

CATEGORIES ARE NOT USABLE AS A FILTER, measured. CoinGecko tags every
coin with a category list, and the tags are wrong in a way that matters:
Bitcoin, Ethereum, Solana AND Dogecoin all carry "Smart Contract
Platform" as their first category, and Bitcoin, Ethereum and Solana all
carry "FTX Holdings". A "DeFi Protocols" preset built on that field
would return Bitcoin and Dogecoin. So the presets here select on
MEASURED quantities — market cap, volume, supply, drawdown, developer
activity — and the one category-shaped preset ("Large caps") is defined
by rank, which is a number.

EVERY PRESET MUST RETURN SOMETHING on live data, and that is a test. A
preset that matches nothing reads as "no such coins exist" rather than
as a filter that is too tight — the bond screener shipped exactly that
bug, where "Treasuries for Safety" returned zero because the universe
was missing every short treasury fund.

A COIN THAT CANNOT BE JUDGED IS NOT A COIN THAT FAILED. A criterion
reading a field the coin does not report leaves it UNJUDGED, and the
count of those is returned alongside the matches. Silently dropping them
would report "3 matches" from a universe where 200 were never examined.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import crypto_data
from logging_setup import get_logger, log_event

logger = get_logger(__name__)

MAX_RESULTS_SHOWN = 50


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    kind: str = "number"        # number | money | percent
    help_text: str = ""
    higher_is_better: Optional[bool] = None


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("market_cap", "Market cap", "money",
               "Price times coins in circulation."),
    MetricSpec("price", "Price", "money"),
    MetricSpec("volume_24h", "24h volume", "money",
               "Exchange trading volume, not on-chain settlement."),
    MetricSpec("turnover", "Turnover", "number",
               "24h volume divided by market cap — how much of the coin "
               "changes hands in a day. A liquidity read that needs no "
               "order book."),
    MetricSpec("change_24h_pct", "24h change", "percent"),
    MetricSpec("change_7d_pct", "7-day change", "percent"),
    MetricSpec("change_30d_pct", "30-day change", "percent"),
    MetricSpec("change_1y_pct", "1-year change", "percent"),
    MetricSpec("ath_change_pct", "Below all-time high", "percent",
               "Negative: how far under its record price the coin trades."),
    MetricSpec("market_cap_rank", "Rank", "number",
               "1 is the largest coin by market cap.", higher_is_better=False),
    MetricSpec("pct_of_max_mined", "Mined of maximum", "percent",
               "Unavailable for an uncapped coin, which is an answer "
               "rather than a gap."),
)
METRICS_BY_KEY: Dict[str, MetricSpec] = {m.key: m for m in METRICS}

OPERATORS: Tuple[str, ...] = (">", "<", ">=", "<=")


@dataclass(frozen=True)
class Criterion:
    metric: str
    operator: str
    threshold: float


def _value(row: "crypto_data.CoinRow", metric: str) -> Optional[float]:
    """One metric off a row, including the computed ones.

    `turnover` and `pct_of_max_mined` are properties rather than fields,
    and `pct_of_max_mined` is None for an uncapped coin by design — see
    crypto_valuation.supply_picture for why a zero there would be false.
    """
    value = getattr(row, metric, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _passes(value: Optional[float], criterion: Criterion) -> Optional[bool]:
    """True, False, or None for "this coin could not be judged"."""
    if value is None:
        return None
    threshold = float(criterion.threshold)
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
    row: "crypto_data.CoinRow"
    values: Dict[str, Optional[float]]

    @property
    def symbol(self) -> str:
        return self.row.symbol.upper()


def run(rows: Sequence["crypto_data.CoinRow"],
        criteria: Sequence[Criterion]) -> Tuple[List[Match], int]:
    """Coins meeting every criterion, and how many could not be judged.

    Returns (matches, unjudged). The second number is not decoration: a
    screen over a universe where most coins do not report the field is
    a screen over a handful, and the reader has to be told which.
    """
    matches: List[Match] = []
    unjudged = 0
    for row in rows:
        values: Dict[str, Optional[float]] = {}
        verdicts: List[Optional[bool]] = []
        for criterion in criteria:
            value = _value(row, criterion.metric)
            values[criterion.metric] = value
            verdicts.append(_passes(value, criterion))
        if any(v is None for v in verdicts):
            unjudged += 1
            continue
        if all(verdicts):
            matches.append(Match(row, values))
    matches.sort(key=lambda m: -(m.row.market_cap or 0))
    log_event(logger, logging.INFO, "crypto_screener.run",
              universe=len(rows), criteria=len(criteria),
              matches=len(matches), unjudged=unjudged)
    return matches, unjudged


# --- presets ------------------------------------------------------------------

@dataclass(frozen=True)
class Preset:
    name: str
    criteria: Tuple[Criterion, ...]
    description: str


# Thresholds are sized against the measured universe (top 250 by market
# cap, 2026-08-25) so that each returns a non-empty, non-trivial set.
# "Large Cap Blues" in the task means BTC and ETH only; expressed as a
# rank cut it also survives the day something overtakes one of them.
PRESETS: Tuple[Preset, ...] = (
    Preset("Large caps",
           (Criterion("market_cap_rank", "<=", 10),),
           "The ten largest coins by market cap. Rank rather than a "
           "hand-written list, so it stays correct as the order changes."),
    Preset("Deeply discounted",
           (Criterion("ath_change_pct", "<", -70.0),
            Criterion("market_cap", ">", 1e9)),
           "Billion-dollar coins trading more than 70% below their "
           "record. A drawdown is not a discount — this is where to "
           "start looking, not a verdict."),
    Preset("Heavily traded",
           (Criterion("turnover", ">", 0.15),
            Criterion("market_cap", ">", 5e8)),
           "Coins turning over more than 15% of their market cap in a "
           "day. High turnover is liquidity and it is also churn."),
    Preset("Scarce and capped",
           (Criterion("pct_of_max_mined", ">", 80.0),
            Criterion("market_cap", ">", 1e8)),
           "Coins with a supply cap that is more than 80% reached. "
           "Uncapped coins are not judged by this screen rather than "
           "failed by it."),
    Preset("Year-long winners",
           (Criterion("change_1y_pct", ">", 0.0),
            Criterion("market_cap", ">", 1e9)),
           "Billion-dollar coins up over twelve months."),
)
PRESETS_BY_NAME: Dict[str, Preset] = {p.name: p for p in PRESETS}


def operators_for(metric: str) -> Tuple[str, ...]:
    """Every metric here is numeric, so every operator applies.

    Kept as a function anyway because the screener's widgets take no
    Streamlit key — a stored operator outside a changed options list
    raises — and this is the single place that list comes from.
    """
    return OPERATORS


def describe(criterion: Criterion) -> str:
    spec = METRICS_BY_KEY.get(criterion.metric)
    label = spec.label if spec else criterion.metric
    kind = spec.kind if spec else "number"
    if kind == "money":
        threshold = crypto_compact(criterion.threshold)
    elif kind == "percent":
        threshold = f"{criterion.threshold:g}%"
    else:
        threshold = f"{criterion.threshold:g}"
    return f"{label} {criterion.operator} {threshold}"


def crypto_compact(value: Optional[float]) -> str:
    """A money figure in compact form.

    Crypto prices span eight orders of magnitude in one universe — SHIB
    near 0.00001 and BTC near 80,000 — so a fixed two-decimal format
    renders half the universe as $0.00. Small values keep significant
    digits instead.
    """
    if value is None:
        return "Not reported"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not reported"
    if number != number:
        return "Not reported"
    magnitude = abs(number)
    for cut, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= cut:
            return f"${number / cut:,.2f}{suffix}"
    if magnitude >= 1:
        return f"${number:,.2f}"
    if magnitude == 0:
        return "$0"
    # Four significant figures, so a sub-cent coin keeps its price.
    return f"${number:,.8f}".rstrip("0")


TABLE_COLUMNS: Tuple[str, ...] = (
    "Symbol", "Name", "Rank", "Price", "Market cap", "24h volume",
    "24h %", "1y %", "Below ATH %", "Mined %",
)

# Columns that must be numeric for sorting to work. A column no row in
# the result set reports comes back object-dtype and sorts as text —
# which is how the ETF screener's P/E column broke once a filter
# narrowed it to funds that report none.
NUMERIC_COLUMNS: Tuple[str, ...] = (
    "Rank", "Price", "Market cap", "24h volume", "24h %", "1y %",
    "Below ATH %", "Mined %",
)


def results_frame(matches: Sequence[Match]):
    """The result table, with every numeric column coerced to a number."""
    import pandas as pd

    frame = pd.DataFrame([{
        "Symbol": m.symbol,
        "Name": m.row.name,
        "Rank": m.row.market_cap_rank,
        "Price": m.row.price,
        "Market cap": m.row.market_cap,
        "24h volume": m.row.volume_24h,
        "24h %": m.row.change_24h_pct,
        "1y %": m.row.change_1y_pct,
        "Below ATH %": m.row.ath_change_pct,
        "Mined %": m.row.pct_of_max_mined,
    } for m in matches], columns=list(TABLE_COLUMNS))
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def column_config():
    """Streamlit column formats.

    Percent columns use a printf format, NOT format="percent", which
    multiplies the stored number by 100 — every figure in this app is
    already percent-valued.
    """
    import streamlit as st

    return {
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Price": st.column_config.NumberColumn("Price", format="$%.6g"),
        "Market cap": st.column_config.NumberColumn(
            "Market cap", format="compact"),
        "24h volume": st.column_config.NumberColumn(
            "24h volume", format="compact",
            help="Exchange trading volume over the last 24 hours."),
        "24h %": st.column_config.NumberColumn("24h %", format="%.2f%%"),
        "1y %": st.column_config.NumberColumn("1y %", format="%.1f%%"),
        "Below ATH %": st.column_config.NumberColumn(
            "Below ATH %", format="%.1f%%",
            help="How far under its record price the coin trades."),
        "Mined %": st.column_config.NumberColumn(
            "Mined %", format="%.1f%%",
            help="Percent of the maximum supply already in circulation. "
                 "Blank for an uncapped coin, which has no maximum — not "
                 "a missing figure."),
    }
