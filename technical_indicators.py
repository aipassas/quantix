"""Technical Analysis Engine for Quantix.

Every indicator computed from price history — Simple Moving Average, RSI,
MACD, Bollinger Bands, ATR, Stochastic Oscillator, Anchored VWAP, ADX, the
Ichimoku Cloud, and On-Balance Volume — is calculated here, in one place,
instead of inline in finance.py. This mirrors fundamental_analysis.py's
role for statement-derived ratios: finance.py consumes the results and
renders them; it performs no indicator arithmetic of its own.

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
        return "▲" if self.kind == "bullish" else "▼"

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
            value=value, zone="overbought", label="Overbought",
            explanation=(
                f"RSI is at {value:.1f}, at or above the overbought threshold of {overbought:.0f} — "
                "the recent price advance has been unusually strong and may be due for a pullback or consolidation."
            ),
        )
    if value <= oversold:
        return RSIInterpretation(
            value=value, zone="oversold", label="Oversold",
            explanation=(
                f"RSI is at {value:.1f}, at or below the oversold threshold of {oversold:.0f} — "
                "the recent decline has been unusually strong and may be due for a bounce or stabilization."
            ),
        )
    return RSIInterpretation(
        value=value, zone="neutral", label="Neutral",
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
        return "▲" if self.kind == "bullish" else "▼"

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
        return "▲" if self.kind == "upper" else "▼"

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


def _rma_unseeded(series: pd.Series, length: int) -> pd.Series:
    """Bare Wilder/"rma" smoothing (alpha = 1/length), NOT SMA-seeded and
    with NO min_periods constraint — matching pandas_ta's rma() exactly,
    which compute_adx() chains across two stages (+DM/-DM, then DX→ADX).

    Deliberately does NOT add a min_periods warmup mask the way
    compute_rsi() does inline for its own gain/loss averages: RSI only
    chains ONE smoothing stage, so a min_periods mask just delays when
    real numbers appear. ADX chains a SECOND stage (DX→ADX) on top of the
    first — adding min_periods to the first stage shifts which bar the
    second stage's unseeded `.ewm()` treats as its effective starting
    seed, which measurably diverges from pandas_ta (confirmed: ~0.49 max
    diff on ADX itself, caught by cross-validating the full valid range,
    not just the tail — the same class of bug, and the same order of
    magnitude, as the original MACD seeding issue). The natural NaN
    propagation from ADX's own seeded-ATR warm-up already produces
    genuinely-NaN early values without an extra mask — see compute_adx().
    """
    return series.ewm(alpha=1.0 / length, adjust=False).mean()


# ----- Stochastic Oscillator ------------------------------------------------

def compute_stochastic(
    df: pd.DataFrame,
    k_period: Optional[int] = None,
    d_period: Optional[int] = None,
    smooth_k: Optional[int] = None,
) -> pd.DataFrame:
    """Stochastic Oscillator (the "Slow Stochastic" — see note below), added
    as Stoch_K/Stoch_D columns to a copy of `df`.

        Raw %K = 100 × (Close − Lowest Low(k_period)) / (Highest High(k_period) − Lowest Low(k_period))
        %K     = SMA(Raw %K, smooth_k)
        %D     = SMA(%K, d_period)

    Defaults: k_period=14, d_period=3, smooth_k=3 (config.TECHNICAL).

    Convention note: the textbook "Fast Stochastic" is just Raw %K with
    %D = SMA(Raw %K, 3) — no smooth_k step. What TradingView (and every
    major charting platform) actually displays by default as "Stochastic"
    is the Slow variant above, which pre-smooths %K itself before %D is
    even computed — confirmed by reading pandas_ta's stoch() source rather
    than assuming the textbook formula, the same discipline that caught
    MACD's/ATR's seeding conventions in this module. Passing smooth_k=1
    recovers the Fast Stochastic if ever needed.
    """
    k_period = k_period or TECHNICAL.stochastic_k_period
    d_period = d_period or TECHNICAL.stochastic_d_period
    smooth_k = smooth_k or TECHNICAL.stochastic_smooth_k

    result = df.copy()
    lowest_low = result["Low"].rolling(window=k_period, min_periods=k_period).min()
    highest_high = result["High"].rolling(window=k_period, min_periods=k_period).max()
    raw_k = 100 * (result["Close"] - lowest_low) / (highest_high - lowest_low)

    stoch_k = raw_k.rolling(window=smooth_k, min_periods=smooth_k).mean() if smooth_k > 1 else raw_k
    stoch_d = stoch_k.rolling(window=d_period, min_periods=d_period).mean()

    result["Stoch_K"] = stoch_k
    result["Stoch_D"] = stoch_d
    return result


@dataclass
class StochasticCrossoverSignal:
    """One crossover event: the day %K moved from one side of %D to the
    other. A single row per crossing — see detect_stochastic_crossovers()."""
    date: pd.Timestamp
    kind: str  # "bullish" (%K crossed above %D) or "bearish" (crossed below)
    k_value: float
    d_value: float

    @property
    def icon(self) -> str:
        return "▲" if self.kind == "bullish" else "▼"

    @property
    def label(self) -> str:
        return "Bullish Crossover" if self.kind == "bullish" else "Bearish Crossover"


def detect_stochastic_crossovers(df: pd.DataFrame) -> List[StochasticCrossoverSignal]:
    """Every point where %K crosses %D. `compute_stochastic(df)` must have
    been called first. Same edge-detection technique as every other
    crossover detector in this module — a diff on the boolean "%K above %D"
    series only changes value on the exact day of a crossing."""
    if "Stoch_K" not in df.columns or "Stoch_D" not in df.columns:
        return []
    valid = df[["Stoch_K", "Stoch_D"]].dropna()
    if len(valid) < 2:
        return []

    above = valid["Stoch_K"] > valid["Stoch_D"]
    transition = above.astype(int).diff()

    signals: List[StochasticCrossoverSignal] = []
    for date, delta in transition.items():
        if delta == 1:
            signals.append(StochasticCrossoverSignal(
                date=date, kind="bullish",
                k_value=float(valid.loc[date, "Stoch_K"]), d_value=float(valid.loc[date, "Stoch_D"]),
            ))
        elif delta == -1:
            signals.append(StochasticCrossoverSignal(
                date=date, kind="bearish",
                k_value=float(valid.loc[date, "Stoch_K"]), d_value=float(valid.loc[date, "Stoch_D"]),
            ))
    return signals


# ----- Anchored VWAP ---------------------------------------------------------

def compute_anchored_vwap(df: pd.DataFrame, anchor_date: Optional[pd.Timestamp] = None) -> pd.Series:
    """Anchored VWAP — cumulative volume-weighted average price from
    `anchor_date` forward (defaults to the first date in `df`).

        Typical Price = (High + Low + Close) / 3
        VWAP          = cumulative(Typical Price × Volume) / cumulative(Volume)

    Deliberately NOT TradingView's intraday, session-resetting VWAP — this
    app only has daily bars, so a daily "session reset" would just be a
    single-bar VWAP (meaningless). An anchored VWAP, cumulative from a
    user-chosen start date, is the daily-bar equivalent that's actually
    informative: "the volume-weighted average price since this reference
    point." Callers should label this "Anchored VWAP" in the UI, not plain
    "VWAP", so it isn't mistaken for the intraday convention.

    Bars before `anchor_date` get NaN (not zero/backfilled) — VWAP is
    undefined before its own anchor point.
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical_price * df["Volume"]

    if anchor_date is None:
        anchor_date = df.index.min()
    anchor_date = pd.Timestamp(anchor_date)

    in_window = df.index >= anchor_date
    cum_pv = pv.where(in_window, 0.0).cumsum()
    cum_vol = df["Volume"].where(in_window, 0.0).cumsum()
    vwap = (cum_pv / cum_vol).where(in_window)
    return vwap


# ----- ADX (Average Directional Index) --------------------------------------

def _sma_seed_positional(series: pd.Series, length: int) -> pd.Series:
    """SMA-seed the first `length` POSITIONS of `series` — pandas' default
    skipna mean over that fixed positional window, even when some of those
    positions are NaN — rather than _sma_seed()'s "drop NaN first, then
    take the first `length` genuinely valid values" approach.

    The two are IDENTICAL whenever there's no leading NaN before the seed
    window (true for every other seeded indicator in this module: MACD,
    ATR's own standalone use). They diverge specifically when a leading
    NaN sits inside that fixed window — exactly _atr_for_adx()'s case
    (bar 0 forced to NaN). Confirmed by reading pandas_ta's atr() presma
    logic directly: `sma_nth = tr[0:length].mean()` — a positional slice,
    not a value-count-based one. Using _sma_seed() there instead would
    silently shift the seed window by one bar and reproduce the same
    warm-up divergence documented on _atr_for_adx().
    """
    seeded = series.copy()
    seed_value = series.iloc[:length].mean()  # positional slice, skipna default
    seeded.iloc[:length - 1] = float("nan")
    seeded.iloc[length - 1] = seed_value
    return seeded


def _atr_for_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """ATR variant used internally by ADX's denominator only — differs
    from compute_atr() in two compounding respects, both confirmed by
    reading pandas_ta's adx()/atr() source directly rather than assumed:

    1. The very first bar's True Range is forced to NaN before seeding
       (pandas_ta's adx() calls atr() with prenan=True; atr()'s own
       standalone default — and compute_atr()'s, matching it — is
       prenan=False).
    2. Because that forces a leading NaN inside the seed window, the seed
       itself must use _sma_seed_positional() (a fixed positional window),
       NOT _sma_seed()/_rma_sma_seeded() (a value-count window) — see
       _sma_seed_positional()'s docstring for why these two produce
       different results specifically in this situation.

    Getting either piece wrong shifts the effective seed window by one
    bar, which doesn't show up as a difference in Plus_DI/Minus_DI
    themselves (they still match pandas_ta exactly wherever both are
    non-NaN) but DOES show up once DX is smoothed into ADX — an unseeded
    `.ewm()` is sensitive to which bar it starts from, so a one-bar NaN
    misalignment upstream produces a ~0.5-1.3 max-diff divergence in ADX
    that decays away over time, the same class of bug (and comparable
    magnitude) as the MACD warm-up discrepancy this codebase already
    caught once — see TECHNICAL_ANALYSIS.md §4. Caught here by
    cross-validating the FULL valid range against pandas_ta, not just
    recent values, and by comparing intermediate series' first-valid-index
    directly rather than only their overlapping values.
    """
    tr = _true_range(df)
    tr.iloc[0] = float("nan")
    seeded = _sma_seed_positional(tr, period)
    return seeded.ewm(alpha=1.0 / period, adjust=False).mean()


def compute_adx(df: pd.DataFrame, period: Optional[int] = None) -> pd.DataFrame:
    """Average Directional Index — Wilder's (1978) trend-STRENGTH indicator
    (not direction), added as Plus_DI/Minus_DI/ADX columns to a copy of `df`.

        +DM = max(High − PrevHigh, 0) if (High − PrevHigh) > (PrevLow − Low) else 0
        -DM = max(PrevLow − Low, 0)   if (PrevLow − Low) > (High − PrevHigh) else 0
        +DI = 100 × WilderSmooth(+DM, period) / ATR(period)
        -DI = 100 × WilderSmooth(-DM, period) / ATR(period)
        DX  = 100 × |+DI − -DI| / (+DI + -DI)
        ADX = WilderSmooth(DX, period)

    Seeding convention — confirmed by reading pandas_ta's adx() source
    directly, not assumed from RSI's or ATR's convention despite ADX
    nominally sharing their Wilder-smoothing "family" (exactly the lesson
    this codebase already learned once — see TECHNICAL_ANALYSIS.md §6):
    pandas_ta's adx() calls its own atr() (SMA-seeded, prenan=True — see
    _atr_for_adx()) for the denominator, but smooths +DM/-DM and DX→ADX
    with its bare, UNSEEDED rma() — the same unseeded convention
    compute_rsi() uses, not the seeded one compute_atr() uses. One
    indicator, two different smoothing conventions for two different
    pieces of its own formula — reused via _rma_sma_seeded() (inside
    _atr_for_adx()) for the ATR piece and the new _rma_unseeded() for the
    +DM/-DM/DX pieces, rather than picking one and applying it everywhere.
    """
    period = period or TECHNICAL.adx_period
    result = df.copy()

    atr_series = _atr_for_adx(result, period)

    up_move = result["High"].diff()
    down_move = -result["Low"].diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    k = 100.0 / atr_series
    plus_di = k * _rma_unseeded(plus_dm, period)
    minus_di = k * _rma_unseeded(minus_dm, period)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _rma_unseeded(dx, period)

    result["Plus_DI"] = plus_di
    result["Minus_DI"] = minus_di
    result["ADX"] = adx
    return result


# ----- Ichimoku Cloud ---------------------------------------------------------

def _midprice(df: pd.DataFrame, period: int) -> pd.Series:
    """(Highest High + Lowest Low) / period-bar window, /2 — the building
    block every Ichimoku line (Tenkan/Kijun/Senkou B) is defined from."""
    return (df["High"].rolling(window=period, min_periods=period).max()
            + df["Low"].rolling(window=period, min_periods=period).min()) / 2.0


@dataclass
class IchimokuResult:
    """Ichimoku Cloud's 5 components, split into two DataFrames because the
    cloud (Senkou A/B) is genuinely forward-looking — plotting it "26
    periods ahead" means real dates beyond the last observed bar, which a
    fixed historical index can't hold.

    `historical`: index matches the input df's index. Columns: Tenkan,
    Kijun, SenkouA, SenkouB, Chikou. SenkouA/SenkouB here are already
    shifted `kijun_period - 1` bars forward in time (so the value visible
    on today's date is what was computed `kijun_period - 1` bars ago) —
    this is what makes the cloud visible over historical price action
    without extending the index; Chikou is Close shifted `-(kijun_period - 1)`
    bars (i.e. plotted in the past). Matches pandas_ta's ichimoku()
    historical DataFrame convention exactly (confirmed by direct source
    read, including the kijun_period - 1 offset — an easy off-by-one to
    get wrong).

    `forward`: `kijun_period` new business-day dates beyond the last date
    in the input, with SenkouA/SenkouB values already computed from the
    most recent bars — the genuinely-in-the-future part of the cloud.
    """
    historical: pd.DataFrame
    forward: pd.DataFrame


def compute_ichimoku(
    df: pd.DataFrame,
    tenkan_period: Optional[int] = None,
    kijun_period: Optional[int] = None,
    senkou_b_period: Optional[int] = None,
) -> IchimokuResult:
    """Ichimoku Kinkō Hyō ("one-glance equilibrium chart") — see
    IchimokuResult's docstring for why this returns two DataFrames.

        Tenkan-sen (Conversion) = midprice(High, Low, tenkan_period)
        Kijun-sen (Base)        = midprice(High, Low, kijun_period)
        Senkou Span A           = (Tenkan + Kijun) / 2, plotted kijun_period-1 bars ahead
        Senkou Span B           = midprice(High, Low, senkou_b_period), plotted kijun_period-1 bars ahead
        Chikou Span             = Close, plotted kijun_period-1 bars behind

    Defaults: 9/26/52 (config.TECHNICAL), the standard periods.
    """
    tenkan_period = tenkan_period or TECHNICAL.ichimoku_tenkan_period
    kijun_period = kijun_period or TECHNICAL.ichimoku_kijun_period
    senkou_b_period = senkou_b_period or TECHNICAL.ichimoku_senkou_b_period

    tenkan_sen = _midprice(df, tenkan_period)
    kijun_sen = _midprice(df, kijun_period)
    senkou_a_raw = (tenkan_sen + kijun_sen) / 2.0
    senkou_b_raw = _midprice(df, senkou_b_period)

    historical = pd.DataFrame({
        "Tenkan": tenkan_sen,
        "Kijun": kijun_sen,
        "SenkouA": senkou_a_raw.shift(kijun_period - 1),
        "SenkouB": senkou_b_raw.shift(kijun_period - 1),
        "Chikou": df["Close"].shift(-(kijun_period - 1)),
    }, index=df.index)

    # The forward-looking cloud: the most recent kijun_period raw values,
    # shifted onto kijun_period new future business dates — genuinely
    # beyond the last observed bar, not a fabricated price, a projection
    # (the entire point of plotting a cloud ahead of price).
    last_date = df.index.max()
    forward_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=kijun_period)
    forward = pd.DataFrame({
        "SenkouA": senkou_a_raw.iloc[-kijun_period:].shift(-1).values,
        "SenkouB": senkou_b_raw.iloc[-kijun_period:].shift(-1).values,
    }, index=forward_dates)

    return IchimokuResult(historical=historical, forward=forward)


# ----- On-Balance Volume (OBV) -----------------------------------------------

def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume — a cumulative running total of signed daily
    volume, the simplest indicator in this module (no smoothing/seeding
    convention to get wrong).

        OBV_t = OBV_(t-1) + Volume_t   if Close_t > Close_(t-1)
        OBV_t = OBV_(t-1) − Volume_t   if Close_t < Close_(t-1)
        OBV_t = OBV_(t-1)              if Close_t == Close_(t-1)

    The very first bar has no previous Close to compare against, so its
    direction is genuinely undefined — NaN there (not a fabricated +1/0
    assumption), consistent with this module's warm-up convention
    elsewhere. Confirmed this matches pandas_ta's obv() empirically (it
    produces the same NaN-first-bar behavior via a Series.diff()-based
    sign calculation, not because of any explicit "first bar" special case
    in either implementation).
    """
    price_change = df["Close"].diff()
    direction = pd.Series(np.nan, index=df.index)
    direction[price_change > 0] = 1.0
    direction[price_change < 0] = -1.0
    direction[price_change == 0] = 0.0
    return (direction * df["Volume"]).cumsum()


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
