"""Technical Analysis Engine for Quantix.

Every indicator computed from price history — Simple Moving Average, RSI,
MACD, Bollinger Bands, and ATR — is calculated here, in one place, instead
of inline in finance.py. This mirrors fundamental_analysis.py's role for
statement-derived ratios: finance.py consumes the results and renders them;
it performs no indicator arithmetic of its own.

Every function here expects an already-cleaned OHLCV DataFrame — the output
of price_processing.py's process_price_data(), never a raw Yahoo fetch
directly — so indicators are never computed on duplicate timestamps,
structurally invalid bars, or a tz-inconsistent index.

See TECHNICAL_ANALYSIS.md for the full reference: every formula, the exact
smoothing/seeding convention each indicator uses (they are NOT all the
same, even within one "family" — see the _sma_seed()/_rma_sma_seeded()
docstrings below for two indicators this caught), and how each was
validated against pandas_ta. The summary below is a quick lookup, not a
replacement for it.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from config import TECHNICAL


@dataclass
class SMACrossoverSignal:
    """One crossover event: the day Close moved from one side of an SMA
    line to the other. A single row per crossing, not a continuous per-day
    state — see detect_sma_crossovers()."""
    date: pd.Timestamp
    kind: str  # "bullish" (Close crossed above) or "bearish" (Close crossed below)
    price: float
    sma_value: float
    sma_period: int

    @property
    def icon(self) -> str:
        return "🟢▲" if self.kind == "bullish" else "🔴▼"

    @property
    def label(self) -> str:
        return "Bullish Crossover" if self.kind == "bullish" else "Bearish Crossover"


def compute_sma_lines(df: pd.DataFrame, periods: Sequence[int]) -> pd.DataFrame:
    """Add one `SMA_{period}` column per period to a copy of `df` — the
    rolling mean of Close over that many bars.

    Vectorized (pandas' `.rolling().mean()` is implemented in C, no Python
    loop over rows). `min_periods=period` means the first `period - 1` rows
    of each column are genuinely NaN rather than an average of fewer
    observations than the period calls for — insufficient observations are
    left as missing, never backfilled or approximated, consistent with this
    app's no-fabricated-data principle. Duplicate periods are computed once.
    """
    result = df.copy()
    for period in dict.fromkeys(periods):  # de-duplicate, preserve order
        result[f"SMA_{period}"] = result["Close"].rolling(window=period, min_periods=period).mean()
    return result


def detect_sma_crossovers(df: pd.DataFrame, period: int) -> List[SMACrossoverSignal]:
    """Every point where Close crosses a single SMA_{period} line —
    "bullish" when price moves from at-or-below to above, "bearish" for the
    reverse. `compute_sma_lines(df, [period])` must have been called first
    (or `period` already present as an SMA_{period} column).

    One signal per crossing event, not one per day the price happens to sit
    on a given side — computed via a diff on the boolean "is price above
    the SMA" series, which by construction only changes value on the exact
    day of a crossing, so no separate deduplication step is needed.
    """
    col = f"SMA_{period}"
    if col not in df.columns:
        return []
    valid = df[["Close", col]].dropna()
    if len(valid) < 2:
        return []

    above = valid["Close"] > valid[col]
    transition = above.astype(int).diff()

    signals: List[SMACrossoverSignal] = []
    for date, delta in transition.items():
        if delta == 1:
            signals.append(SMACrossoverSignal(
                date=date, kind="bullish",
                price=float(valid.loc[date, "Close"]), sma_value=float(valid.loc[date, col]),
                sma_period=period,
            ))
        elif delta == -1:
            signals.append(SMACrossoverSignal(
                date=date, kind="bearish",
                price=float(valid.loc[date, "Close"]), sma_value=float(valid.loc[date, col]),
                sma_period=period,
            ))
    return signals


# ----- RSI ---------------------------------------------------------------

def compute_rsi(df: pd.DataFrame, period: int) -> pd.Series:
    """Relative Strength Index via Wilder's smoothing method — the standard
    used by TradingView and virtually every other charting platform (a
    naive simple-average-of-gains-and-losses RSI, which some simplified
    implementations use instead, does not match this).

        RS  = smoothed average gain / smoothed average loss
        RSI = 100 - (100 / (1 + RS))

    "Smoothed" here means Wilder's exponential moving average (alpha =
    1/period), implemented via `.ewm(alpha=1/period, adjust=False)` — the
    standard vectorized approximation of Wilder's original recursive
    formula, used by pandas_ta and most production RSI implementations for
    the same reason SMA uses `.rolling().mean()` instead of a Python loop.
    It converges to the same values as the textbook seeded-SMA-then-smooth
    method within roughly one `period` of bars past the warm-up point.

    Divide-by-zero handling (avg_loss == 0 makes RS undefined): Wilder's
    convention is RSI = 100 when there were gains and zero losses (maximum
    upward momentum), and RSI = 50 when there were neither gains nor losses
    at all — a genuinely flat market has no directional momentum to report,
    so neutral is the only defensible value, not an artifact of the
    formula's arithmetic.
    """
    delta = df["Close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # NaN (not a ZeroDivisionError) when avg_loss is 0 — division of a float
    # Series by 0.0 produces inf/NaN rather than raising; the explicit masks
    # below immediately replace those with Wilder's defined values (50/100).
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    flat_market = (avg_gain == 0) & (avg_loss == 0)
    all_gains_no_losses = (avg_loss == 0) & ~flat_market
    rsi = rsi.mask(flat_market, 50.0)
    rsi = rsi.mask(all_gains_no_losses, 100.0)
    return rsi


@dataclass
class RSIInterpretation:
    """Plain-English read of one RSI value against the configured
    overbought/oversold thresholds (config.TECHNICAL.rsi_overbought/
    rsi_oversold)."""
    value: float
    zone: str  # "overbought" | "oversold" | "neutral"
    label: str
    explanation: str


def interpret_rsi(value: Optional[float]) -> Optional[RSIInterpretation]:
    """None when `value` is missing/NaN (e.g. still in the warm-up period)
    — an absent interpretation, never a fabricated "neutral" default for
    data that simply isn't there yet."""
    if value is None or pd.isna(value):
        return None

    overbought, oversold = TECHNICAL.rsi_overbought, TECHNICAL.rsi_oversold
    if value >= overbought:
        return RSIInterpretation(
            value=value, zone="overbought", label="🔴 Overbought",
            explanation=(
                f"RSI is at {value:.1f}, at or above the overbought threshold of {overbought:.0f} — "
                "the recent price advance has been unusually strong and may be due for a pullback or consolidation."
            ),
        )
    if value <= oversold:
        return RSIInterpretation(
            value=value, zone="oversold", label="🟢 Oversold",
            explanation=(
                f"RSI is at {value:.1f}, at or below the oversold threshold of {oversold:.0f} — "
                "the recent decline has been unusually strong and may be due for a bounce or stabilization."
            ),
        )
    return RSIInterpretation(
        value=value, zone="neutral", label="🟡 Neutral",
        explanation=(
            f"RSI is at {value:.1f}, between the oversold ({oversold:.0f}) and overbought ({overbought:.0f}) "
            "thresholds — no extreme momentum signal in either direction."
        ),
    )


# ----- MACD ----------------------------------------------------------------

def _sma_seed(series: pd.Series, length: int) -> pd.Series:
    """`series`, with the first `length - 1` non-NaN values replaced by NaN
    and the value at position `length - 1` replaced by the SMA of the first
    `length` non-NaN values — the seeding step shared by every SMA-seeded
    exponential smoothing this module uses (_ema_sma_seeded() for MACD's
    span-based EMA, _rma_sma_seeded() for ATR's Wilder alpha=1/length EMA).
    TA-Lib and TradingView seed their recursive smoothing this way rather
    than treating the very first raw observation as an already-converged
    seed, which is what pandas' bare `.ewm(adjust=False)` does on its own
    and why a naive implementation measurably diverges from TradingView
    during the warm-up period (confirmed empirically against pandas_ta
    while building this — see compute_macd()'s and compute_atr()'s
    docstrings). Returns an all-NaN series, same index, if there isn't even
    `length` observations to seed from.
    """
    valid = series.dropna()
    if len(valid) < length:
        return pd.Series(index=series.index, dtype=float)
    sma_seed = valid.iloc[:length].mean()
    seeded = valid.copy()
    seeded.iloc[:length - 1] = float("nan")
    seeded.iloc[length - 1] = sma_seed
    return seeded


def _ema_sma_seeded(series: pd.Series, length: int) -> pd.Series:
    """SMA-seeded EMA using the standard span-based formula (alpha = 2 /
    (span + 1)) — MACD's convention. See _sma_seed() for the seeding
    rationale.

    Operates on `series` as given, so its own valid (non-NaN) range is used
    to find the seed window — pass the raw Close series for the fast/slow
    EMAs, or the MACD Line itself (which already carries leading NaN from
    the slow EMA's own warm-up) for the signal line, matching how MACD's
    signal line is conventionally seeded from the MACD line's own first
    valid value, not from the very start of the whole price series.
    """
    seeded = _sma_seed(series, length)
    if seeded.isna().all():
        return seeded
    return seeded.ewm(span=length, adjust=False).mean().reindex(series.index)


def _rma_sma_seeded(series: pd.Series, length: int) -> pd.Series:
    """SMA-seeded EMA using Wilder's alpha = 1/length formula ("RMA") — the
    convention shared by RSI's own smoothing *except* RSI's gain/loss
    averages are NOT SMA-seeded (confirmed by reading pandas_ta's rma()
    source: it's a bare `.ewm(alpha=1/length, adjust=False)` with no
    seeding step), while ATR's True Range average IS seeded (pandas_ta's
    atr() explicitly SMA-seeds True Range before calling the same rma()).
    Two indicators sharing a smoothing *family* doesn't mean they seed it
    the same way — see _sma_seed() for the seeding rationale and
    compute_atr() for where this is used.
    """
    seeded = _sma_seed(series, length)
    if seeded.isna().all():
        return seeded
    return seeded.ewm(alpha=1.0 / length, adjust=False).mean().reindex(series.index)


def compute_macd(
    df: pd.DataFrame,
    fast: Optional[int] = None,
    slow: Optional[int] = None,
    signal: Optional[int] = None,
) -> pd.DataFrame:
    """MACD Line, Signal Line, and Histogram, added as columns to a copy of
    `df`. Defaults to the standard 12/26/9 periods (config.TECHNICAL) when
    not overridden.

        Fast EMA    = EMA(Close, fast)
        Slow EMA    = EMA(Close, slow)
        MACD Line   = Fast EMA − Slow EMA
        Signal Line = EMA(MACD Line, signal)
        Histogram   = MACD Line − Signal Line

    Each EMA is SMA-seeded (see _ema_sma_seeded()) — TradingView's/TA-Lib's
    convention — deliberately different from RSI's Wilder smoothing
    (alpha = 1/period, no SMA seed) in compute_rsi(). These are two
    distinct, both-standard conventions: MACD has always been defined on
    ordinary EMAs, RSI on Wilder's specific variant. Matching each
    indicator's own convention is what makes both match TradingView, not
    picking one smoothing method and reusing it everywhere.
    """
    fast = fast or TECHNICAL.macd_fast_period
    slow = slow or TECHNICAL.macd_slow_period
    signal = signal or TECHNICAL.macd_signal_period

    result = df.copy()
    fast_ema = _ema_sma_seeded(result["Close"], fast)
    slow_ema = _ema_sma_seeded(result["Close"], slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema_sma_seeded(macd_line, signal)

    result["MACD_Line"] = macd_line
    result["MACD_Signal"] = signal_line
    result["MACD_Histogram"] = macd_line - signal_line
    return result


@dataclass
class MACDCrossoverSignal:
    """One crossover event: the day the MACD Line moved from one side of
    the Signal Line to the other. A single row per crossing, not a
    continuous per-day state — see detect_macd_crossovers()."""
    date: pd.Timestamp
    kind: str  # "bullish" (MACD crossed above Signal) or "bearish" (crossed below)
    macd_value: float
    signal_value: float

    @property
    def icon(self) -> str:
        return "🟢▲" if self.kind == "bullish" else "🔴▼"

    @property
    def label(self) -> str:
        return "Bullish Crossover" if self.kind == "bullish" else "Bearish Crossover"


def detect_macd_crossovers(df: pd.DataFrame) -> List[MACDCrossoverSignal]:
    """Every point where the MACD Line crosses the Signal Line —
    "bullish" when it moves from at-or-below to above, "bearish" for the
    reverse. `compute_macd(df)` must have been called first.

    Same edge-detection technique as detect_sma_crossovers(): a diff on the
    boolean "is MACD above Signal" series only changes value on the exact
    day of a crossing, so false/duplicate signals are ruled out by
    construction rather than filtered after the fact.
    """
    if "MACD_Line" not in df.columns or "MACD_Signal" not in df.columns:
        return []
    valid = df[["MACD_Line", "MACD_Signal"]].dropna()
    if len(valid) < 2:
        return []

    above = valid["MACD_Line"] > valid["MACD_Signal"]
    transition = above.astype(int).diff()

    signals: List[MACDCrossoverSignal] = []
    for date, delta in transition.items():
        if delta == 1:
            signals.append(MACDCrossoverSignal(
                date=date, kind="bullish",
                macd_value=float(valid.loc[date, "MACD_Line"]), signal_value=float(valid.loc[date, "MACD_Signal"]),
            ))
        elif delta == -1:
            signals.append(MACDCrossoverSignal(
                date=date, kind="bearish",
                macd_value=float(valid.loc[date, "MACD_Line"]), signal_value=float(valid.loc[date, "MACD_Signal"]),
            ))
    return signals


# ----- Bollinger Bands ------------------------------------------------------

def compute_bollinger_bands(df: pd.DataFrame, period: int, num_std: Optional[float] = None) -> pd.DataFrame:
    """Bollinger Bands (middle SMA, upper/lower bands), added as columns to
    a copy of `df`.

        Middle Band = SMA(Close, period)
        Upper Band  = Middle Band + num_std × rolling_std(Close, period)
        Lower Band  = Middle Band − num_std × rolling_std(Close, period)

    `num_std` defaults to the standard 2.0 (config.TECHNICAL.bollinger_num_std)
    when not overridden. The rolling standard deviation uses pandas' default
    sample convention (`ddof=1`, N-1 divisor) — matches pandas_ta's own
    non-TA-Lib bbands() path (confirmed empirically; TA-Lib itself, when
    present, uses population ddof=0 instead, a discrepancy worth knowing
    about if this environment ever gains a TA-Lib installation).

    Vectorized (pandas' `.rolling().mean()`/`.std()`, no Python loop) — the
    first `period - 1` rows of every column are genuinely NaN (insufficient
    observations), same convention as compute_sma_lines().
    """
    num_std = num_std if num_std is not None else TECHNICAL.bollinger_num_std
    result = df.copy()
    middle = result["Close"].rolling(window=period, min_periods=period).mean()
    std = result["Close"].rolling(window=period, min_periods=period).std()

    result["BB_Middle"] = middle
    result["BB_Upper"] = middle + num_std * std
    result["BB_Lower"] = middle - num_std * std
    return result


@dataclass
class BollingerBreakout:
    """One breakout event: the day Close first closed outside a Bollinger
    Band, having been inside (or on the opposite side) the prior day. A
    single row per new excursion, not a continuous per-day state while
    price remains outside — see detect_bollinger_breakouts()."""
    date: pd.Timestamp
    kind: str  # "upper" or "lower"
    price: float
    band_value: float

    @property
    def icon(self) -> str:
        return "🔴⬆" if self.kind == "upper" else "🟢⬇"

    @property
    def label(self) -> str:
        return "Upper Band Breakout" if self.kind == "upper" else "Lower Band Breakout"


def detect_bollinger_breakouts(df: pd.DataFrame) -> List[BollingerBreakout]:
    """Every point where Close first closes outside a Bollinger Band.
    `compute_bollinger_bands(df, period)` must have been called first.

    Close-based, not High/Low-based: a breakout requires the day's *Close*
    to be outside the band, not merely an intraday wick touching it — the
    same, less noisy convention the SMA/MACD crossover signals already use
    (computed off Close), and the deliberate choice for "false signals
    minimized" per this task's acceptance criterion.

    One signal per new excursion: each day is classified as "above" /
    "inside" / "below", and a breakout only fires on the day that
    classification *changes into* "above" or "below" — every subsequent
    day price remains outside the same band is not a new signal, so a
    multi-day breakout doesn't flood the list with duplicates.
    """
    cols = ["Close", "BB_Upper", "BB_Lower"]
    if not all(c in df.columns for c in cols):
        return []
    valid = df[cols].dropna()
    if len(valid) < 2:
        return []

    state = pd.Series(
        np.select(
            [valid["Close"] > valid["BB_Upper"], valid["Close"] < valid["BB_Lower"]],
            ["above", "below"],
            default="inside",
        ),
        index=valid.index,
    )
    prev_state = state.shift(1)

    signals: List[BollingerBreakout] = []
    for date in valid.index:
        cur, prev = state.loc[date], prev_state.loc[date]
        if prev is None or (isinstance(prev, float) and pd.isna(prev)):
            continue
        if cur == "above" and prev != "above":
            signals.append(BollingerBreakout(
                date=date, kind="upper",
                price=float(valid.loc[date, "Close"]), band_value=float(valid.loc[date, "BB_Upper"]),
            ))
        elif cur == "below" and prev != "below":
            signals.append(BollingerBreakout(
                date=date, kind="lower",
                price=float(valid.loc[date, "Close"]), band_value=float(valid.loc[date, "BB_Lower"]),
            ))
    return signals


# ----- ATR -------------------------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    """True Range = max(High − Low, |High − PrevClose|, |Low − PrevClose|)
    — the single-bar volatility measure ATR smooths. The first bar has no
    previous close, so its two gap-vs-prior-close terms are NaN; pandas'
    `.max(axis=1)` default (`skipna=True`) correctly degrades that bar to
    plain High − Low rather than propagating NaN, matching pandas_ta's own
    true_range() (confirmed by reading its source)."""
    prev_close = df["Close"].shift(1)
    ranges = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range — Wilder's original (1978) volatility indicator.

        True Range = max(High − Low, |High − PrevClose|, |Low − PrevClose|)
        ATR        = Wilder's smoothed (SMA-seeded, alpha=1/period) average
                     of True Range over `period` bars

    SMA-seeded via _rma_sma_seeded() — Wilder's original definition
    computes the first ATR value as a plain `period`-bar average of True
    Range, then smooths recursively from there; this is TA-Lib's/
    TradingView's convention too (confirmed by reading pandas_ta's atr()
    source, which explicitly seeds True Range this way before applying its
    "rma" smoothing — a different, additional step from RSI's plain,
    unseeded Wilder smoothing in compute_rsi()).

    Missing observations: the first `period - 1` rows are genuinely NaN
    (insufficient True Range history to seed from — the first bar's True
    Range itself is defined even without a previous close, so the count is
    period-1, not period), never backfilled — same convention as every
    other indicator in this module.
    """
    tr = _true_range(df)
    return _rma_sma_seeded(tr, period)


def suggested_stop_loss(current_price: Optional[float], current_atr: Optional[float], multiplier: Optional[float] = None) -> Optional[float]:
    """Volatility-adjusted long-position stop-loss: Current Price −
    multiplier × ATR (config.TECHNICAL.atr_stop_multiplier, default 2.0,
    when not overridden). None when either input is missing — never a
    fabricated stop level computed from partial data.

    Long-only, matching this app's framing throughout (Kelly Criterion,
    DCF, Quality Score all assume going long, never shorting) — this is a
    downside stop for a bought position, not a two-sided bracket.
    """
    if current_price is None or current_atr is None or pd.isna(current_atr):
        return None
    multiplier = multiplier if multiplier is not None else TECHNICAL.atr_stop_multiplier
    return current_price - multiplier * current_atr
