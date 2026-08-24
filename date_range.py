"""The analysis date range: presets, and the arithmetic behind them.

WHAT WAS ACTUALLY WRONG. The task asks to replace "text date inputs" with
a calendar picker, but both controls were already st.date_input — they
open a calendar on click and always have. What was missing was everything
around them: no way to say "last three months" without doing the
subtraction yourself, no statement of how long the selected window
actually is, and two stacked controls where one would do.

ONE CONTROL, TWO DATES. st.date_input takes a (start, end) tuple and
renders a single calendar where you click the start and then the end.
That is the native form of "drag to select a range", and it halves the
controls on a phone. The catch is that BETWEEN those two clicks it
returns a ONE-element tuple, so any caller that unpacks two values
crashes on the intermediate state. coerce() is the whole reason this
module has a function for something as simple as a tuple.

"MAX" IS A REQUEST, NOT A PROMISE. Nothing here knows how far back a
given symbol's history goes without fetching it, so Max asks for a very
long window and the loader returns whatever exists. The label and the
summary say "as far back as the source has" rather than implying this app
knows the listing date — a chart that silently starts in 1998 because
that is all Yahoo returned should not look like a deliberate choice.

EVERY PRESET IS COMPUTED FROM A PASSED-IN `today`, never from
date.today() inside the resolver, so the tests are not a different
program on 1 January.
"""
import datetime
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

# How far back "Max" reaches. Longer than any equity history this app is
# likely to be pointed at, so the source is the binding constraint rather
# than this number — which is the honest way round.
MAX_LOOKBACK_YEARS = 30

# The lower bound offered by the calendar. Not unbounded: a typo of
# "1025" should be rejected by the control rather than sent to the loader
# as a thirty-thousand-day request.
EARLIEST_SELECTABLE = datetime.date(1970, 1, 1)


def _months_back(day: datetime.date, months: int) -> datetime.date:
    """`day` shifted back whole months, clamped to a real date.

    Calendar months, not 30-day blocks: "3M" from 31 May means 28 or 29
    February, not 2 March. Doing it by timedelta would drift the anchor
    every time and make two consecutive 1M clicks land on different days
    of the month.
    """
    month_index = day.month - 1 - months
    year = day.year + month_index // 12
    month = month_index % 12 + 1
    # Step back a day at a time from the 28th-safe day rather than
    # importing calendar just to find the month's length.
    day_of_month = day.day
    while day_of_month > 28:
        try:
            return datetime.date(year, month, day_of_month)
        except ValueError:
            day_of_month -= 1
    return datetime.date(year, month, day_of_month)


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    start_of: Callable[[datetime.date], datetime.date]
    note: str = ""


PRESETS: Tuple[Preset, ...] = (
    Preset("1M", "1M", lambda d: _months_back(d, 1)),
    Preset("3M", "3M", lambda d: _months_back(d, 3)),
    Preset("6M", "6M", lambda d: _months_back(d, 6)),
    Preset("YTD", "YTD", lambda d: datetime.date(d.year, 1, 1),
           "From 1 January of the current year."),
    Preset("1Y", "1Y", lambda d: _months_back(d, 12)),
    Preset("5Y", "5Y", lambda d: _months_back(d, 60)),
    Preset("Max", "Max", lambda d: _months_back(d, 12 * MAX_LOOKBACK_YEARS),
           "As far back as the data source has for this symbol — which is "
           "usually less than the window requested."),
)
PRESETS_BY_KEY = {p.key: p for p in PRESETS}
PRESET_KEYS: Tuple[str, ...] = tuple(p.key for p in PRESETS)


def resolve(key: str, today: datetime.date) -> Optional[Tuple[datetime.date, datetime.date]]:
    """(start, end) for a preset, or None if the key is unknown.

    None rather than a fallback range: silently substituting a different
    window for an unrecognised preset would change what the whole page is
    analysing without saying so.
    """
    preset = PRESETS_BY_KEY.get(key)
    if preset is None:
        return None
    return preset.start_of(today), today


def coerce(value, fallback: Tuple[datetime.date, datetime.date]):
    """Whatever st.date_input returned, as a usable (start, end).

    Between the two clicks of a range selection the widget returns a
    ONE-element tuple. Unpacking that raises, and this control drives
    every fetch on the page — so the half-made selection holds the
    previous range instead of taking the app down. Also orders the pair,
    since a calendar can hand back end-before-start.
    """
    if isinstance(value, datetime.date):
        value = (value,)
    try:
        dates = [v for v in value if isinstance(v, datetime.date)]
    except TypeError:
        return fallback
    if len(dates) < 2:
        return fallback
    start, end = dates[0], dates[1]
    return (start, end) if start <= end else (end, start)


def matching_preset(start: datetime.date, end: datetime.date,
                    today: datetime.date) -> Optional[str]:
    """Which preset this range corresponds to, if any.

    Lets the pill row show the current window as selected instead of
    going blank the moment the page reloads with a preset-shaped range.
    """
    for preset in PRESETS:
        if (preset.start_of(today), today) == (start, end):
            return preset.key
    return None


def describe(start: datetime.date, end: datetime.date,
             today: Optional[datetime.date] = None) -> str:
    """"24 Aug 2025 → 24 Aug 2026 · 365 days" — the span, stated.

    The task asks to "show selected dates clearly", and the count of days
    is the part that is actually hard to see: two dates a year apart look
    much like two dates three years apart at a glance.
    """
    days = (end - start).days
    span = f"{days:,} days"
    if days >= 365:
        span += f" (~{days / 365.25:.1f} years)"
    text = (f"{start.strftime('%d %b %Y')} → {end.strftime('%d %b %Y')}  ·  {span}")
    if today is not None and end > today:
        text += "  ·  ends in the future; no prices exist past today"
    return text


def problems(start: datetime.date, end: datetime.date,
             today: datetime.date) -> Sequence[str]:
    """Anything wrong with this range, in words. Empty when it is fine."""
    found = []
    if start > end:
        found.append("The start date is after the end date.")
    if (end - start).days < 2:
        found.append("That window is too short to compute indicators from.")
    if start < EARLIEST_SELECTABLE:
        found.append(f"Dates before {EARLIEST_SELECTABLE:%Y} are not supported.")
    return found
