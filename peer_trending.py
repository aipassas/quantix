"""What other accounts on this instance have been looking at.

THIS IS NOT ticker_discovery's TRENDING, AND THE DIFFERENCE IS THE WHOLE
POINT. That module reports the MARKET — Yahoo's most-active, day-gainers
and day-losers screens — and it is already wired into the sidebar's "Find
a ticker" panel. It is real, it is instance-independent, and it is not
what this ticket asks for. This one reports the PEOPLE here: which
tickers the other accounts opened. On a laptop with one account that is
honestly nothing, and the empty state says so and points at the market
panel rather than quietly rendering the same data under a social label.

NOTHING NEW IS COLLECTED. following.STREAM_VIEWED already publishes a
"looked at" event per ticker per account, opt-in, deduplicated inside a
24-hour window. So this module owns NO store, adds NO switch, and asks
for NO further consent: it takes the feed and the profiles as parameters
and counts. Switching the stream off purges its history and `trending()`
re-checks every publisher's CURRENT switch besides, so opting out drops
someone from the counts on the next read — the same discipline
following.feed_for already applies, reused rather than rebuilt.

COUNTS, NEVER NAMES. The feed names a publisher to the accounts that
FOLLOW them. This widget is visible to every account on the instance,
including ones that follow nobody, so it reaches a strictly wider
audience and therefore carries strictly less: a ticker and a number. A
`Trend` has no field through which an identity could travel, which is the
same signature-as-boundary argument peer_comparison.publish() makes.

THE FLOOR IS TWO, AND NOT peer_comparison's FIVE. That floor protects a
return, where percentile arithmetic states other people's figures almost
exactly. Here the disclosure is "at least two accounts opened NVDA" — it
names nobody and is weaker than what the stream already authorises. Two
is also simply what the word means: one person looking at something is
not a trend, it is a person, and a count of one on a small team is often
attributable to whoever is known to have the stream on. A floor of five
would mean a six-person team needs five of them on one ticker before
anything ever appears, which is a widget that never populates.

YOUR OWN VIEWS DO NOT COUNT. The ticket asks what OTHERS are looking at.
Counting the reader would reflect their own browsing back at them as a
social signal, which is the most misleading version of social proof
there is.

"RIGHT NOW" IS NOT AVAILABLE. Events carry a timestamp, not a live
session — Streamlit executes nothing when no tab is open, as
realtime_alerts already documents. A trailing window is the honest
reading and the panel names it rather than implying presence.
"""
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config import PEER_TRENDING
from logging_setup import get_logger

logger = get_logger("peer_trending")

# The stream this reads. Named here rather than imported at module scope
# so the dependency is one-directional and visible: following owns the
# collection, this owns the arithmetic.
STREAM = "viewed"

NOBODY_SHARING = (
    "No other account on this instance is sharing what it looks at, so there is "
    "nothing to show. This panel counts accounts that switched on the "
    '"Recently viewed" stream in their profile — it is deliberately empty rather '
    "than filled with market data wearing a social label."
)

BELOW_FLOOR = (
    "Nothing has been opened by at least {floor} different accounts in the last "
    "{hours} hours. One account looking at something is a person, not a trend, and "
    "on a small team a count of one is usually attributable — so single views are "
    "not listed."
)

MARKET_TRENDING_INSTEAD = (
    'For what the market is doing rather than what this team is doing, the '
    '"Find a ticker" panel in the sidebar shows the most active names, the '
    "day's gainers and the day's losers from live screens."
)


@dataclass(frozen=True)
class Trend:
    """One ticker and how many DISTINCT accounts opened it.

    Deliberately three fields, none of them an identity. There is no
    parameter here through which a name, a key or a profile could
    travel — the same reasoning peer_comparison.publish()'s signature
    carries.
    """
    ticker: str
    viewers: int
    latest_at: str

    def sentence(self) -> str:
        who = "accounts" if self.viewers != 1 else "account"
        return f"{self.ticker} — {self.viewers} {who}"


@dataclass(frozen=True)
class Trends:
    """The rows, or the reason there are none."""
    rows: Tuple[Trend, ...] = ()
    window_hours: int = 0
    sharing_accounts: int = 0      # OTHER accounts with the stream on
    reason: str = ""

    @property
    def has_result(self) -> bool:
        return bool(self.rows)


def _now() -> datetime.datetime:
    return datetime.datetime.now()


def sharing_accounts(profiles, exclude_key: str = "") -> Tuple[str, ...]:
    """Account keys with the viewed stream currently ON, excluding the
    reader. Read from the profiles rather than from the feed, so an
    account that switched the stream on but has opened nothing yet still
    counts as participating — the difference between "nobody is sharing"
    and "nobody has looked at anything", which are different answers."""
    exclude_key = (exclude_key or "").strip()
    return tuple(
        p.user_key for p in getattr(profiles, "profiles", ())
        if p.user_key != exclude_key and p.shares(STREAM)
    )


def trending(feed, profiles, viewer_key: str = "",
             now: Optional[datetime.datetime] = None,
             window_hours: Optional[int] = None,
             max_rows: Optional[int] = None) -> Trends:
    """Tickers other accounts opened inside the window, most viewers first.

    Every publisher's CURRENT switch is re-checked here rather than
    trusted from the event: a stream turned off must stop counting on the
    next read, not on the next purge.

    Distinct accounts, never event counts. The feed already collapses
    repeat views inside 24 hours, but relying on that would make this
    function's correctness depend on another module's window — and the
    two windows are configured separately.
    """
    window_hours = (PEER_TRENDING.window_hours if window_hours is None
                    else window_hours)
    max_rows = PEER_TRENDING.max_rows if max_rows is None else max_rows
    viewer_key = (viewer_key or "").strip()
    now = now or _now()

    participating = sharing_accounts(profiles, viewer_key)
    if not participating:
        return Trends(window_hours=window_hours, reason=NOBODY_SHARING)

    cutoff = (now - datetime.timedelta(hours=window_hours)).isoformat(
        timespec="seconds")
    allowed = set(participating)

    seen: Dict[str, set] = {}
    latest: Dict[str, str] = {}
    for event in getattr(feed, "events", ()):
        # Read defensively throughout. following.load_feed already drops
        # malformed records, so nothing here should be missing a field —
        # but this is a DISPLAY widget inside the portfolio tab, and one
        # bad row must not take the tab down with it.
        if getattr(event, "stream", "") != STREAM:
            continue
        actor = getattr(event, "actor_key", "")
        if actor not in allowed:
            continue                    # the reader, or a switched-off stream
        at = getattr(event, "at", "") or ""
        if not at or at < cutoff:
            continue
        ticker = (getattr(event, "ticker", "") or "").strip().upper()
        if not ticker:
            continue
        seen.setdefault(ticker, set()).add(actor)
        if at > latest.get(ticker, ""):
            latest[ticker] = at

    rows = [Trend(ticker, len(keys), latest.get(ticker, ""))
            for ticker, keys in seen.items()
            if len(keys) >= PEER_TRENDING.min_viewers]

    if not rows:
        return Trends(window_hours=window_hours,
                      sharing_accounts=len(participating),
                      reason=BELOW_FLOOR.format(floor=PEER_TRENDING.min_viewers,
                                                hours=window_hours))

    # Most viewers first, then most recently seen, then alphabetically so
    # two readers loading the same window compute the same list.
    rows.sort(key=lambda t: (-t.viewers, _negated(t.latest_at), t.ticker))
    return Trends(tuple(rows[:max_rows]), window_hours, len(participating))


class _Negated:
    """Sorts a string descending inside an otherwise ascending key."""
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __lt__(self, other) -> bool:
        return self.value > other.value

    def __eq__(self, other) -> bool:
        return isinstance(other, _Negated) and self.value == other.value


def _negated(value: str) -> _Negated:
    return _Negated(value or "")


def caption(result: Trends) -> str:
    """One line describing what the panel is counting."""
    if not result.has_result:
        return ""
    who = "account" if result.sharing_accounts == 1 else "accounts"
    return (f"Distinct accounts that opened each ticker in the last "
            f"{result.window_hours} hours, out of {result.sharing_accounts} other "
            f"{who} sharing what they look at. Your own views are not counted, and "
            f"no row names anyone.")
