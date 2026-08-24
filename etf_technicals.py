"""Technical analysis for funds: sector momentum, flow, and range position.

WHAT WAS ALREADY BUILT, checked before writing anything. The task lists
SMA(20/50/200), RSI(14), MACD(12,26,9), Bollinger(20,2) and VWAP as
"existing, adapted for ETF" — and they are existing, in
technical_indicators.py, operating on a price frame with nothing
company-specific in them. The Chart Workspace already renders the SMA
trio, an RSI subplot, a MACD subplot and volume bars coloured by up/down
day, and it is not gated on asset class, so it already worked for funds
before this module existed. asset_class.py already declares that an ETF
supports TECHNICALS. So none of that is rebuilt here. What is missing is
everything fund-SPECIFIC, and that is what this module adds.

PREMIUM / DISCOUNT TO NAV IS NOT IMPLEMENTED, and not because it was
hard. Two measurements:

  1. There is no NAV time series. `funds_data` exposes no NAV history and
     `info` carries a single scalar `navPrice`. A "plot NAV vs market
     price over time" needs a series that does not exist.

  2. The single scalar is STALE, which is worse than absent. Measured on
     2026-08-24 against the last bar of the same date: SPY -0.28%,
     QQQ -0.97%, GLD +1.53%, ARKK -2.70%. Those are impossible figures
     for the most heavily arbitraged funds in existence, where the real
     premium sits under 0.05%. Matching each navPrice against recent
     closes found why: ARKK's "NAV" of 86.24 is its 2026-08-21 close of
     86.21, QQQ's 713.26 is that day's 713.44 — a THREE-DAY-OLD close
     compared against a live price.

The task asks to "alert if premium/discount > 0.5% (unusual, possible
arbitrage)". That threshold would have fired on QQQ, GLD, TLT and ARKK
every single day, on nothing but the staleness. A permanent false
arbitrage signal on the four most liquid funds on the board is worse
than no signal, so the panel says what it cannot compute and why.
NAV_PREMIUM_UNAVAILABLE carries that sentence to the UI.

SECTOR MOMENTUM IS MEASURED THROUGH SECTOR ETFs, NOT THROUGH WEIGHT
HISTORY. The task asks to "track daily tech%, finance%, energy% vs. the
20-day avg", which needs a history of the fund's sector weights.
`funds_data.sector_weightings` is a single undated dict — there is no
time dimension in it at all, so a 20-day average of it cannot be formed
from this source, and inventing one would be fabricating the very series
the analysis rests on.

What CAN be answered, with real data, is the question underneath: which
sectors inside this fund are pulling it, and by how much. That needs the
fund's sector weights (real, current) and each sector's own price
momentum (real, from the SPDR select sector ETFs, all eleven of which
map exactly onto Yahoo's eleven sector keys and all of which return
history — verified). Weight times sector return is a contribution in
percentage points, and the sum across sectors reconstructs an estimate
of the fund's own move. It is labelled an estimate because it is one:
the proxy is the sector, not this fund's holdings within that sector.

NOT EVERY FUND HAS SECTORS. TLT and GLD report zero sector weightings —
a bond fund and a commodity trust have no equity sectors to weight. That
is an absent capability, not missing data, and it renders as such rather
than as an empty chart.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import streamlit as st

from logging_setup import get_logger, log_event, log_exception

logger = get_logger("etf_technicals")

# Yahoo's eleven sector keys -> the SPDR select sector ETF that tracks
# each. The mapping is exact: Yahoo reports exactly these eleven and
# State Street lists exactly these eleven funds. All were verified to
# return price history.
SECTOR_PROXIES: Dict[str, str] = {
    "technology": "XLK",
    "financial_services": "XLF",
    "healthcare": "XLV",
    "consumer_cyclical": "XLY",
    "consumer_defensive": "XLP",
    "energy": "XLE",
    "industrials": "XLI",
    "basic_materials": "XLB",
    "utilities": "XLU",
    "realestate": "XLRE",
    "communication_services": "XLC",
}

SECTOR_LABELS: Dict[str, str] = {
    "technology": "Technology",
    "financial_services": "Financials",
    "healthcare": "Healthcare",
    "consumer_cyclical": "Consumer Cyclical",
    "consumer_defensive": "Consumer Defensive",
    "energy": "Energy",
    "industrials": "Industrials",
    "basic_materials": "Basic Materials",
    "utilities": "Utilities",
    "realestate": "Real Estate",
    "communication_services": "Communication Services",
}

# The task's own window and threshold.
MOMENTUM_LOOKBACK_DAYS = 20
DIVERGENCE_FLAG_PCT = 2.0

# "Volume today 200% above 20-day avg = strong momentum" — read as the
# ratio reaching 200% OF the average, i.e. double. Stated explicitly
# because "200% above" and "200% of" differ by a factor of 1.5 and the
# phrase is ambiguous.
VOLUME_WINDOW = 20
VOLUME_SPIKE_RATIO_PCT = 200.0

# A 52-week window needs a 52-week window. 200 trading days is the floor
# for calling something a 52-week high; below it the reading is withheld
# rather than computed against whatever happens to be loaded, which would
# call a three-month high a 52-week one.
TRADING_DAYS_52W = 200
SMA_LONG_PERIOD = 200

CACHE_TTL_SECONDS = 3600

NAV_PREMIUM_UNAVAILABLE = (
    "Premium/discount to NAV is not shown. Yahoo exposes no NAV history, "
    "and its single `navPrice` is a stale close — measured on "
    "2026-08-24, ARKK's reported NAV was its 21 August close, giving a "
    "false -2.70% discount on a fund that arbitrages to within 0.05%. "
    "A 0.5% alert threshold would fire permanently on the most liquid "
    "funds on the board, so the figure is withheld rather than reported "
    "wrongly."
)


@dataclass(frozen=True)
class SectorMomentum:
    key: str
    label: str
    proxy: str
    weight_pct: float            # the fund's weight in this sector
    return_pct: Optional[float]  # the proxy's return over the lookback
    contribution_pct: Optional[float]   # weight x return, in points
    divergence_pct: Optional[float]     # sector return minus fund return
    flagged: bool = False        # |divergence| >= DIVERGENCE_FLAG_PCT


@dataclass(frozen=True)
class VolumeReading:
    latest: Optional[float] = None
    average: Optional[float] = None
    ratio_pct: Optional[float] = None     # latest as a % OF the average
    is_spike: bool = False
    on_up_day: Optional[bool] = None
    days_used: int = 0

    @property
    def ok(self) -> bool:
        return self.ratio_pct is not None


@dataclass(frozen=True)
class RangePosition:
    price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    position_pct: Optional[float] = None   # 0 at the low, 100 at the high
    at_new_high: bool = False
    at_new_low: bool = False
    days_used: int = 0
    sufficient: bool = False               # enough history to say "52-week"


# A signal is one of three things, and "unavailable" is not "not fired".
FIRED = "fired"
NOT_FIRED = "not fired"
UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Signal:
    name: str
    state: str
    detail: str


@dataclass(frozen=True)
class MomentumVerdict:
    label: str            # Bullish | Neutral | Bearish | Unavailable
    score: int            # net of the contributing readings
    considered: int       # how many readings could be measured
    reasons: Tuple[str, ...] = ()


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _pct_change(series: pd.Series, lookback: int) -> Optional[float]:
    """Percent change over `lookback` bars, or None if there are not that
    many bars. Never falls back to a shorter window silently — a 20-day
    momentum figure computed over four days is not a 20-day figure."""
    clean = series.dropna()
    if len(clean) <= lookback:
        return None
    first, last = _number(clean.iloc[-1 - lookback]), _number(clean.iloc[-1])
    if first is None or last is None or first == 0:
        return None
    return (last / first - 1.0) * 100.0


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_sector_returns(lookback: int = MOMENTUM_LOOKBACK_DAYS
                        ) -> Tuple[Dict[str, float], Optional[str]]:
    """(sector key -> percent return over the lookback, error).

    One batched download for all eleven proxies rather than eleven
    requests. Never raises: a failure returns an empty mapping and a
    message the panel can show, because a sector panel that cannot load
    must say so rather than render as though every sector were flat.
    """
    symbols = list(SECTOR_PROXIES.values())
    try:
        import yfinance as yf

        # Enough bars to cover the lookback with room for holidays.
        frame = yf.download(symbols, period="6mo", progress=False,
                            auto_adjust=True)
        closes = frame["Close"] if "Close" in frame else pd.DataFrame()
    except Exception as exc:                      # noqa: BLE001 - never raise
        log_exception(logger, "etf_technicals.sectors_failed",
                      error=f"{type(exc).__name__}: {exc}")
        return {}, f"Sector price data could not be loaded: {exc}"

    if closes is None or closes.empty:
        return {}, "Sector price data came back empty."

    returns: Dict[str, float] = {}
    for key, proxy in SECTOR_PROXIES.items():
        if proxy not in closes:
            continue
        value = _pct_change(closes[proxy], lookback)
        if value is not None:
            returns[key] = value

    log_event(logger, logging.INFO, "etf_technicals.sectors_loaded",
              sectors=len(returns), lookback=lookback)
    missing = [k for k in SECTOR_PROXIES if k not in returns]
    error = None
    if not returns:
        error = "No sector proxy returned enough history for this window."
    elif missing:
        error = ("No recent data for: "
                 + ", ".join(SECTOR_LABELS[k] for k in missing))
    return returns, error


def sector_momentum(weights: Optional[Dict[str, float]],
                    sector_returns: Dict[str, float],
                    fund_return_pct: Optional[float] = None
                    ) -> List[SectorMomentum]:
    """Each sector's weight, its proxy's momentum, and its contribution.

    `weights` is Yahoo's sector_weightings — FRACTIONS (0.374 for 37.4%),
    converted here so every figure this module hands out is
    percent-valued like the rest of the app.

    Sorted by contribution, largest first: the question is which sectors
    are moving the fund, so the biggest mover leads rather than the
    alphabet.
    """
    if not weights:
        return []
    rows: List[SectorMomentum] = []
    for key, raw_weight in weights.items():
        weight = _number(raw_weight)
        if weight is None:
            continue
        weight_pct = weight * 100.0
        ret = sector_returns.get(key)
        contribution = None if ret is None else weight * ret
        divergence = (None if ret is None or fund_return_pct is None
                      else ret - fund_return_pct)
        rows.append(SectorMomentum(
            key=key,
            label=SECTOR_LABELS.get(key, key.replace("_", " ").title()),
            proxy=SECTOR_PROXIES.get(key, ""),
            weight_pct=weight_pct,
            return_pct=ret,
            contribution_pct=contribution,
            divergence_pct=divergence,
            flagged=(divergence is not None
                     and abs(divergence) >= DIVERGENCE_FLAG_PCT),
        ))
    # An unmeasured sector sorts last rather than as zero — it has not
    # contributed nothing, it is simply not known.
    rows.sort(key=lambda r: (r.contribution_pct is None,
                             -(r.contribution_pct or 0.0)))
    return rows


def estimated_fund_move(rows: Sequence[SectorMomentum]) -> Optional[float]:
    """Sum of the sector contributions, or None if none could be measured.

    An ESTIMATE, and named one: the proxy is the sector as a whole, not
    this fund's particular holdings within it.
    """
    measured = [r.contribution_pct for r in rows if r.contribution_pct is not None]
    return sum(measured) if measured else None


def leaders_and_laggards(rows: Sequence[SectorMomentum]
                         ) -> Tuple[Tuple[SectorMomentum, ...],
                                    Tuple[SectorMomentum, ...]]:
    """The flagged divergences, split by direction."""
    flagged = [r for r in rows if r.flagged and r.divergence_pct is not None]
    ahead = tuple(r for r in flagged if r.divergence_pct > 0)
    behind = tuple(r for r in flagged if r.divergence_pct < 0)
    return ahead, behind


def relative_volume(df: Optional[pd.DataFrame],
                    window: int = VOLUME_WINDOW) -> VolumeReading:
    """Latest volume against its trailing average.

    The average EXCLUDES the latest bar. Including it puts the value
    being tested into the baseline it is tested against, which drags the
    ratio toward 100% exactly when the spike is largest — a doubling
    measured against a 20-bar window that contains it reads as 190%, not
    200%.
    """
    if df is None or "Volume" not in getattr(df, "columns", []) or len(df) < 2:
        return VolumeReading()
    volumes = pd.to_numeric(df["Volume"], errors="coerce").dropna()
    if len(volumes) < 2:
        return VolumeReading()
    latest = _number(volumes.iloc[-1])
    prior = volumes.iloc[-1 - window:-1] if len(volumes) > window else volumes.iloc[:-1]
    average = _number(prior.mean())
    if latest is None or average is None or average <= 0:
        return VolumeReading(latest=latest, average=average,
                             days_used=len(prior))
    on_up = None
    if {"Open", "Close"}.issubset(set(df.columns)):
        open_, close = _number(df["Open"].iloc[-1]), _number(df["Close"].iloc[-1])
        if open_ is not None and close is not None:
            on_up = close >= open_
    ratio = (latest / average) * 100.0
    return VolumeReading(latest=latest, average=average, ratio_pct=ratio,
                         is_spike=ratio >= VOLUME_SPIKE_RATIO_PCT,
                         on_up_day=on_up, days_used=len(prior))


def range_position(df: Optional[pd.DataFrame]) -> RangePosition:
    """Where the price sits in its 52-week range.

    `sufficient` is False when fewer than TRADING_DAYS_52W bars are
    loaded. The high and low are still returned — they are the loaded
    range and are useful — but the caller must not call them 52-week
    figures, because at a 3-month range they are three-month figures.
    """
    if df is None or "Close" not in getattr(df, "columns", []) or df.empty:
        return RangePosition()
    closes = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if closes.empty:
        return RangePosition()
    highs = (pd.to_numeric(df["High"], errors="coerce").dropna()
             if "High" in df.columns else closes)
    lows = (pd.to_numeric(df["Low"], errors="coerce").dropna()
            if "Low" in df.columns else closes)
    price = _number(closes.iloc[-1])
    high, low = _number(highs.max()), _number(lows.min())
    if price is None or high is None or low is None:
        return RangePosition(days_used=len(closes))
    span = high - low
    position = None if span <= 0 else ((price - low) / span) * 100.0
    return RangePosition(
        price=price, high=high, low=low, position_pct=position,
        # Compared against the bar's own high/low, so a close that only
        # equals a prior CLOSE is not called a breakout.
        at_new_high=price >= high, at_new_low=price <= low,
        days_used=len(closes), sufficient=len(closes) >= TRADING_DAYS_52W)


def _crossed_above(series: pd.Series, reference: pd.Series) -> Optional[bool]:
    pair = pd.concat([series, reference], axis=1).dropna()
    if len(pair) < 2:
        return None
    prev, now = pair.iloc[-2], pair.iloc[-1]
    return bool(prev.iloc[0] <= prev.iloc[1] and now.iloc[0] > now.iloc[1])


def _crossed_below(series: pd.Series, reference: pd.Series) -> Optional[bool]:
    pair = pd.concat([series, reference], axis=1).dropna()
    if len(pair) < 2:
        return None
    prev, now = pair.iloc[-2], pair.iloc[-1]
    return bool(prev.iloc[0] >= prev.iloc[1] and now.iloc[0] < now.iloc[1])


def signals(df: Optional[pd.DataFrame],
            sma_lines: Optional[pd.DataFrame],
            rsi: Optional[pd.Series],
            volume: Optional[VolumeReading] = None) -> List[Signal]:
    """The task's four entry/exit rules, each fired, not fired, or —
    where the history loaded cannot answer it — explicitly unavailable.

    SMA(200) needs 200 trading days. A one-year range gives 251 and a
    three-month range gives 63 (measured), so on a short range the
    sell rule genuinely cannot be evaluated. Reporting that as "not
    fired" would read as an all-clear that was never checked.
    """
    out: List[Signal] = []
    close = (pd.to_numeric(df["Close"], errors="coerce")
             if df is not None and "Close" in getattr(df, "columns", []) else None)
    sma_lines = sma_lines if sma_lines is not None else pd.DataFrame()

    def line(period: int) -> Optional[pd.Series]:
        name = f"SMA_{period}"
        if name not in sma_lines.columns:
            return None
        series = pd.to_numeric(sma_lines[name], errors="coerce").dropna()
        return series if len(series) >= 2 else None

    rsi_latest = None
    if rsi is not None:
        clean = pd.to_numeric(rsi, errors="coerce").dropna()
        rsi_latest = _number(clean.iloc[-1]) if len(clean) else None

    # Buy: price crosses above SMA(50) AND RSI < 50.
    sma50 = line(50)
    if close is None or sma50 is None:
        out.append(Signal("Buy — early momentum", UNAVAILABLE,
                          "Needs 50 days of history; the loaded range is shorter."))
    elif rsi_latest is None:
        out.append(Signal("Buy — early momentum", UNAVAILABLE,
                          "RSI could not be computed over this range."))
    else:
        crossed = _crossed_above(close, sma50)
        fired = bool(crossed) and rsi_latest < 50
        out.append(Signal(
            "Buy — early momentum", FIRED if fired else NOT_FIRED,
            f"Price {'crossed above' if crossed else 'did not cross above'} "
            f"SMA(50) on the latest bar; RSI {rsi_latest:.1f} "
            f"({'below' if rsi_latest < 50 else 'not below'} 50)."))

    # Sell: price crosses below SMA(200).
    sma200 = line(SMA_LONG_PERIOD)
    if close is None or sma200 is None:
        out.append(Signal("Sell — trend reversal", UNAVAILABLE,
                          f"Needs {SMA_LONG_PERIOD} days of history; the loaded "
                          "range is shorter, so this was not evaluated."))
    else:
        crossed = _crossed_below(close, sma200)
        out.append(Signal(
            "Sell — trend reversal", FIRED if crossed else NOT_FIRED,
            f"Price {'crossed below' if crossed else 'held above or at'} "
            f"SMA({SMA_LONG_PERIOD}) on the latest bar."))

    # Momentum: RSI > 60.
    if rsi_latest is None:
        out.append(Signal("Momentum — strong uptrend", UNAVAILABLE,
                          "RSI could not be computed over this range."))
    else:
        out.append(Signal(
            "Momentum — strong uptrend",
            FIRED if rsi_latest > 60 else NOT_FIRED,
            f"RSI {rsi_latest:.1f} ({'above' if rsi_latest > 60 else 'not above'} 60)."))

    # Support: volume spike on a down day with price holding.
    if volume is None or not volume.ok or volume.on_up_day is None:
        out.append(Signal("Support — accumulation", UNAVAILABLE,
                          "Volume data was not available for the latest bar."))
    else:
        fired = volume.is_spike and not volume.on_up_day
        out.append(Signal(
            "Support — accumulation", FIRED if fired else NOT_FIRED,
            f"Volume at {volume.ratio_pct:.0f}% of its {volume.days_used}-day "
            f"average on {'an up' if volume.on_up_day else 'a down'} day."))
    return out


def momentum_verdict(rsi_latest: Optional[float],
                     sma_lines: Optional[pd.DataFrame],
                     df: Optional[pd.DataFrame],
                     range_pos: Optional[RangePosition] = None
                     ) -> MomentumVerdict:
    """Bullish / Neutral / Bearish, from the readings that could be taken.

    Scored out of what was MEASURABLE, not out of a fixed denominator. A
    three-month range cannot speak to the 200-day trend, and averaging a
    missing reading in as zero would drag every short range toward
    Neutral and make the gauge look decisive when it was half-blind.
    """
    score = 0
    considered = 0
    reasons: List[str] = []

    if rsi_latest is not None:
        considered += 1
        if rsi_latest > 60:
            score += 1
            reasons.append(f"RSI {rsi_latest:.1f} is above 60")
        elif rsi_latest < 40:
            score -= 1
            reasons.append(f"RSI {rsi_latest:.1f} is below 40")
        else:
            reasons.append(f"RSI {rsi_latest:.1f} is mid-range")

    close = (pd.to_numeric(df["Close"], errors="coerce").dropna()
             if df is not None and "Close" in getattr(df, "columns", []) else None)
    sma_lines = sma_lines if sma_lines is not None else pd.DataFrame()
    for period in (50, SMA_LONG_PERIOD):
        name = f"SMA_{period}"
        if close is None or name not in sma_lines.columns:
            continue
        series = pd.to_numeric(sma_lines[name], errors="coerce").dropna()
        if series.empty:
            continue
        considered += 1
        latest_close, latest_sma = _number(close.iloc[-1]), _number(series.iloc[-1])
        if latest_close is None or latest_sma is None:
            considered -= 1
            continue
        if latest_close > latest_sma:
            score += 1
            reasons.append(f"price is above its {period}-day average")
        else:
            score -= 1
            reasons.append(f"price is below its {period}-day average")

    if range_pos is not None and range_pos.position_pct is not None:
        considered += 1
        if range_pos.position_pct >= 80:
            score += 1
            reasons.append(f"price sits at {range_pos.position_pct:.0f}% of its range")
        elif range_pos.position_pct <= 20:
            score -= 1
            reasons.append(f"price sits at {range_pos.position_pct:.0f}% of its range")
        else:
            reasons.append(f"price sits mid-range at {range_pos.position_pct:.0f}%")

    if considered == 0:
        return MomentumVerdict("Unavailable", 0, 0,
                               ("Not enough price history to read momentum.",))
    label = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"
    return MomentumVerdict(label, score, considered, tuple(reasons))


def describe_volume(reading: VolumeReading) -> str:
    """The task's own phrasing, with the ambiguity resolved on screen."""
    if not reading.ok:
        return "Volume for the latest bar was not reported."
    direction = ("an up day" if reading.on_up_day
                 else "a down day" if reading.on_up_day is False else "the latest bar")
    return (f"Volume is {reading.ratio_pct:.0f}% of its {reading.days_used}-day "
            f"average on {direction}"
            + (" — a spike." if reading.is_spike else "."))
