"""The one-line key-metrics strip under the symbol header.

WHAT "AUTO-UPDATE" HONESTLY MEANS HERE. The strip re-renders on a timer
via st.fragment, the same mechanism the real-time alert engine already
uses, so it refreshes without a full page rerun. But the live quote
underneath it is cached for 300 seconds (watchlist_panel._load_quote), so
polling faster than that would re-render an identical number and buy
nothing but Yahoo requests. The refresh interval is therefore pinned to
that TTL, and the UI says how fresh the figure actually is rather than
implying a tick-by-tick feed this app does not have.

THE STRIP IS BUTTONS, NOT A LINE OF TEXT, because the brief asks for
click-to-expand. Custom HTML in st.markdown cannot call back into
Streamlit, so a metric rendered as styled text can never be clickable.
st.popover gives a control that reads as one compact chip and opens an
explanation in place — which satisfies "one-line stats" and "click to
expand each metric" at once instead of trading one against the other.

EVERY VALUE CAN BE ABSENT, AND SAYS SO. A metric this ticker does not
report renders as "Not reported", never as 0.00 or a blank. This strip
sits at the top of every screen, so a fabricated zero here would be the
most-seen wrong number in the app.

UNITS ARE CONVERTED IN ONE PLACE. StandardizedFinancials mixes fractions
(net_margin, return_on_equity) with already-percent values
(dividend_yield_pct). Each spec declares which it is, so the display path
never has to remember — the class of bug that has bitten this codebase
repeatedly.
"""
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_exception

logger = get_logger("quick_stats")

STORE_FILENAME = "quick_stats.json"

# Matches watchlist_panel._load_quote's cache TTL. Refreshing faster only
# re-renders the same cached number; slower would let the strip visibly lag
# the watchlist row for the same symbol.
REFRESH_SECONDS = 300

NOT_REPORTED = "Not reported"

MAX_SELECTED = 8   # beyond this the strip wraps to a second sticky row


@dataclass(frozen=True)
class StatSpec:
    key: str
    label: str
    source: str          # "quote" or "fundamental"
    kind: str = "number"  # number | money | percent | fraction_percent | text
    help_key: str = ""    # into metric_help, when one exists
    decimals: int = 2
    note: str = ""        # shown in the popover, under the value


# Ordered as offered in the picker. Everything here comes from data the
# page has already loaded — the quote for the live half, the standardized
# fundamentals for the rest — so the strip adds no fetches of its own.
STATS: Tuple[StatSpec, ...] = (
    StatSpec("price", "Price", "quote", "money", "market_price",
             note="The live quote, not the last bar of the price history the "
                  "charts and DCF are built on. Intraday those differ slightly, "
                  "which is correct: they measure different things."),
    StatSpec("change_pct", "Day", "quote", "percent",
             note="Against the previous close from the same quote."),
    StatSpec("pe_ratio", "P/E", "quote", "number",
             note="Trailing twelve months, as reported."),
    StatSpec("market_cap", "Market Cap", "fundamental", "money"),
    StatSpec("dividend_yield_pct", "Dividend", "fundamental", "percent", "dividend_yield",
             note="Trailing yield. Cross-checked against dividend rate over price, "
                  "because the reported field's units vary by data-source version."),
    StatSpec("beta", "Beta", "fundamental", "number",
             note="Yahoo's reported beta. The Risk panel regresses its own from "
                  "actual returns, which can differ."),
    StatSpec("price_to_book", "P/B", "fundamental", "number"),
    StatSpec("net_margin", "Net Margin", "fundamental", "fraction_percent"),
    StatSpec("return_on_equity", "ROE", "fundamental", "fraction_percent"),
    StatSpec("debt_to_equity", "Debt/Equity", "fundamental", "number"),
    StatSpec("current_ratio", "Current Ratio", "fundamental", "number"),
    StatSpec("sector", "Sector", "fundamental", "text"),
)
STATS_BY_KEY: Dict[str, StatSpec] = {s.key: s for s in STATS}

# Price and day change are deliberately NOT here, though the brief's
# example line led with price: the symbol header renders both directly
# above this strip in much larger type, so defaulting to them spends two
# of the row's slots repeating what the eye has already read. Both remain
# in STATS and are one click away in the picker for anyone who wants the
# strip to stand alone.
DEFAULT_KEYS: Tuple[str, ...] = ("pe_ratio", "market_cap", "dividend_yield_pct")


# --- formatting ---------------------------------------------------------------

def compact_money(value: Optional[float]) -> str:
    """$3.24T / $412.5B / $980.2M / $12,345.

    Thresholds are applied on the absolute value so a negative figure
    (equity can be) scales the same way rather than falling through to the
    plain-number branch.
    """
    if value is None:
        return NOT_REPORTED
    try:
        value = float(value)
    except (TypeError, ValueError):
        return NOT_REPORTED
    magnitude = abs(value)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= cutoff:
            return f"${value / cutoff:,.2f}{suffix}"
    return f"${value:,.2f}"


def format_value(spec: StatSpec, raw: Any) -> str:
    """`raw` rendered for the strip. Never invents a number."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return NOT_REPORTED
    if spec.kind == "text":
        return str(raw)
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return NOT_REPORTED
    if number != number:          # NaN
        return NOT_REPORTED

    if spec.kind == "money":
        return compact_money(number)
    if spec.kind == "percent":
        return f"{number:,.{spec.decimals}f}%"
    if spec.kind == "fraction_percent":
        # Stored as a fraction; shown as a percentage.
        return f"{number * 100:,.{spec.decimals}f}%"
    return f"{number:,.{spec.decimals}f}"


def raw_value(spec: StatSpec, quote, standardized) -> Any:
    """Pull one stat off whichever object owns it. Never raises."""
    try:
        if spec.source == "quote":
            if spec.key == "price":
                value = getattr(quote, "price", None)
                # Fall back to the standardized close only when the live
                # quote has nothing — better a slightly older real price
                # than "Not reported" on a ticker that clearly has one.
                return value if value is not None else getattr(standardized, "current_price", None)
            return getattr(quote, spec.key, None)
        return getattr(standardized, spec.key, None)
    except Exception:
        return None


def display(spec: StatSpec, quote, standardized) -> str:
    return format_value(spec, raw_value(spec, quote, standardized))


# --- the user's selection -----------------------------------------------------

def _path():
    return store_path(STORE_FILENAME)


def _read() -> Tuple[Dict[str, Any], bool]:
    """(data, corrupt) — the same distinction screener_templates makes.
    An unreadable file must not be quietly overwritten with defaults."""
    path = _path()
    if not path.exists():
        return {}, False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log_exception(logger, "quick_stats.unreadable", section="quick_stats")
        return {}, True
    if not isinstance(data, dict):
        return {}, True
    return data, False


def selected() -> Tuple[str, ...]:
    """The user's chosen stats, in their chosen order.

    Unknown keys are dropped rather than shown as blanks — unlike a saved
    screener criterion, a stat this version does not know cannot be
    partially honoured, and silently rendering nothing for it is what the
    picker would show anyway.
    """
    data, corrupt = _read()
    if corrupt:
        return DEFAULT_KEYS
    chosen = data.get("selected")
    if not isinstance(chosen, list):
        return DEFAULT_KEYS
    cleaned = tuple(k for k in chosen if k in STATS_BY_KEY)[:MAX_SELECTED]
    # An explicitly empty selection is a real choice — hide the strip —
    # and must not silently spring back to the defaults.
    return cleaned if cleaned or chosen == [] else DEFAULT_KEYS


def set_selected(keys: Sequence[str]) -> Tuple[bool, Optional[str]]:
    data, corrupt = _read()
    if corrupt:
        return False, ("The quick-stats preferences file on this instance can't be "
                       f"read, so saving would overwrite it. Move or delete "
                       f"{STORE_FILENAME} and try again.")
    cleaned = []
    for key in keys or []:
        if key in STATS_BY_KEY and key not in cleaned:
            cleaned.append(key)
    if len(cleaned) > MAX_SELECTED:
        return False, f"Pick at most {MAX_SELECTED} stats — more than that wraps onto a second row."
    data["selected"] = cleaned
    atomic_write_text(_path(), json.dumps(data, indent=2))
    return True, None


def reset() -> None:
    set_selected(DEFAULT_KEYS)
