# Quantix Technical Analysis Documentation

This document explains every indicator in [`technical_indicators.py`](technical_indicators.py)
— formulas, null handling, and (most importantly) the exact smoothing
conventions each one uses and how each was validated. It exists so a future
contributor doesn't have to re-derive, by trial and error, the same subtle
seeding discrepancies this module's development ran into more than once.

For the fundamentals side of the app, see [`FUNDAMENTALS.md`](FUNDAMENTALS.md).
For the data pipeline, see [`ARCHITECTURE.md`](ARCHITECTURE.md). This
document covers only price-derived technical indicators.

## 1. Scope and data flow

`technical_indicators.py` is the sole place indicator arithmetic happens —
`finance.py` calls into it and renders the results, the same separation
`fundamental_analysis.py` has from statement-derived ratios.

Every function here expects an **already-cleaned** OHLCV DataFrame — the
output of [`price_processing.py`](price_processing.py)'s
`process_price_data()`, never a raw Yahoo fetch directly:

```
data_loader.py            → raw OHLCV from Yahoo Finance
price_processing.py       → validated, deduplicated, gap-checked DataFrame
technical_indicators.py   → SMA, RSI, MACD, Bollinger Bands, ATR
finance.py                → renders the chart, reads the results
```

This guarantees every indicator is computed on a sorted, timezone-naive,
duplicate-free index with no structurally invalid bars — an indicator can
still return `NaN` for insufficient history, but never operates on corrupt
input.

## 2. Simple Moving Average (SMA)

**Formula**: `SMA = rolling_mean(Close, period)` — vectorized via pandas'
`.rolling(period, min_periods=period).mean()`.

**Configuration**: one custom period via the sidebar slider (default 20),
plus an optional fixed 20/50/200-day trio (`config.TECHNICAL.sma_trio_periods`)
— the three most widely-used periods across virtually every charting
platform, shown as a toggle so the chart isn't cluttered by default... it
actually defaults to on, since it's expected out of the box.

**Null handling**: the first `period - 1` bars are `NaN` — genuinely
insufficient observations, never backfilled.

**Crossover signals** (`detect_sma_crossovers`): a signal fires when Close
moves from at-or-below to above the SMA (bullish) or the reverse (bearish).
Implementation: `above = Close > SMA` (boolean), then `above.astype(int).diff()`
— this only changes value on the exact day of a crossing, which rules out
duplicate/repeated signals *by construction* rather than by filtering
afterward. Every other crossover/breakout detector in this module
(MACD, Bollinger) uses the same technique.

## 3. Relative Strength Index (RSI)

**Formula** (Wilder, 1978):

```
RS  = smoothed avg gain / smoothed avg loss
RSI = 100 − (100 / (1 + RS))
```

"Smoothed" is Wilder's own exponential moving average, alpha = 1/period,
implemented as `.ewm(alpha=1/period, adjust=False)` directly on the
gain/loss series — **no SMA seeding** (see §5 for why this matters and how
it differs from ATR, which also uses Wilder smoothing but does seed).

**Divide-by-zero handling** (avg_loss = 0): RSI = 100 when there were gains
and zero losses; RSI = 50 when there were neither gains nor losses at all (a
genuinely flat market has no directional momentum to report — neutral is
the only defensible value, not an arithmetic artifact).

**Thresholds**: fixed at the classic Wilder 70 (overbought) / 30 (oversold)
— `config.TECHNICAL.rsi_overbought` / `rsi_oversold`. `interpret_rsi()`
classifies the latest value into overbought / oversold / neutral with a
plain-English explanation; returns `None` (never a fabricated "neutral")
when the value itself is missing.

**Validated against**: pandas_ta's `rsi()` — matched to within
**floating-point noise** (~1e-14) across the full valid range on real AAPL
data.

## 4. MACD

**Formula**:

```
Fast EMA    = EMA(Close, 12)
Slow EMA    = EMA(Close, 26)
MACD Line   = Fast EMA − Slow EMA
Signal Line = EMA(MACD Line, 9)
Histogram   = MACD Line − Signal Line
```

Periods fixed at the universal 12/26/9 default (`config.TECHNICAL.macd_*`)
— unlike SMA/RSI, MACD's periods are rarely tuned in practice.

**Validated against**: pandas_ta's `macd()`.

### The seeding bug this task caught

The first implementation used pandas' plain `.ewm(span=N, adjust=False)`
directly on the raw Close series. Cross-checking against pandas_ta showed a
real, non-trivial discrepancy (max diff ≈0.49 on the MACD line — not
floating-point noise).

**Root cause**: TA-Lib and TradingView don't seed the EMA recursion with the
very first raw price. They seed it with a **simple average (SMA) of the
first `length` prices**, then switch to exponential smoothing from that
point forward. Confirmed by reading pandas_ta's own `ema()` source
(`presma=True` is its documented default: *"Initialize with SMA like TA
Lib"*). pandas' bare `.ewm(adjust=False)`, left to its own devices, treats
the very first observation as an already-converged seed — no warm-up
averaging at all — and the two conventions measurably diverge during the
warm-up period even though they converge for later bars as the seed's
influence decays exponentially. "Converges eventually" isn't "identical to
TradingView."

**Fix**: `_sma_seed()` — seed the input series with an SMA of its first
`length` values before handing it to `.ewm()`. Applied to the fast EMA, the
slow EMA, *and* the signal line (which pandas_ta seeds from the MACD
line's own first valid value, not from the start of the whole price
series — matched that too).

**Result**: MACD Line, Signal Line, and Histogram now match pandas_ta
**exactly** — `0.00e+00` difference across the entire overlapping valid
range on real AAPL data. Not "close." Identical.

**Crossover signals** (`detect_macd_crossovers`): same edge-detection
technique as SMA — bullish when the MACD Line crosses above the Signal
Line, bearish for the reverse, one event per crossing.

## 5. Bollinger Bands

**Formula**:

```
Middle Band = SMA(Close, period)
Upper Band  = Middle Band + num_std × rolling_std(Close, period)
Lower Band  = Middle Band − num_std × rolling_std(Close, period)
```

**Configuration**: the period reuses the same sidebar "SMA Length" slider
that drives the trend-following SMA line (the middle band *is* that same
SMA — one control instead of two, since they're mathematically the same
value). Width fixed at the standard 2.0σ (`config.TECHNICAL.bollinger_num_std`).

**A ddof gotcha checked before implementing, not after**: pandas_ta's
`bbands()` docstring is actually self-contradictory about its ddof
(degrees-of-freedom) default, and — more importantly — **TA-Lib uses
population ddof=0** for its internal deviation calculation when installed,
which would have been a silent mismatch against pandas' own `.std()`
default (`ddof=1`, sample deviation). Checked directly: this environment
does **not** have TA-Lib installed, so pandas_ta falls through to its own
Python path, which resolves to `ddof=1` — matching pandas' default and what
this implementation already uses. If this environment ever gains a TA-Lib
installation, re-verify this assumption.

**Validated against**: pandas_ta's `bbands()` — matched to within
floating-point noise (~1e-13) across the full valid range.

**Breakout detection** (`detect_bollinger_breakouts`): **Close-based**, not
High/Low-based — a breakout requires the day's *Close* to end outside a
band, not merely an intraday wick touching it (the deliberate, less noisy
choice for "false signals minimized"). Three-state classification (above /
inside / below), diffed the same way crossovers are — a breakout only fires
the day the state first changes *into* "above" or "below"; every
subsequent day price remains outside the same band is not a new signal, so
a multi-day breakout collapses to one event.

## 6. Average True Range (ATR)

**Formula** (Wilder, 1978):

```
True Range = max(High − Low, |High − PrevClose|, |Low − PrevClose|)
ATR        = Wilder-smoothed (SMA-seeded) average of True Range over `period` bars
```

The first bar has no previous close, so its two gap terms are `NaN`;
pandas' `.max(axis=1, skipna=True)` default correctly degrades that bar to
plain `High − Low` rather than propagating `NaN` — confirmed this matches
pandas_ta's own `true_range()`.

**Configuration**: period configurable via a dedicated "ATR Length" sidebar
slider (default 14, Wilder's standard).

### Why ATR needed the seeding fix but RSI didn't, despite both using Wilder smoothing

This was checked *before* implementing, applying the MACD lesson directly:
read pandas_ta's `atr()` source first. Both RSI and ATR nominally use
Wilder's "rma" smoothing family (alpha = 1/period), but:

- `rma()` **itself** — the function both RSI and ATR ultimately call — is
  a bare `.ewm(alpha=1/length, adjust=False)` with **no seeding step**.
- RSI's `rsi()` passes its gain/loss series straight into `rma()`, unseeded.
- ATR's `atr()` **explicitly SMA-seeds the True Range series itself**
  before ever calling `rma()` (`presma=True`, the same technique MACD's
  `ema()` uses, just combined with Wilder's alpha instead of a span-based
  EMA).

Two indicators sharing a smoothing *family* does not mean they seed it the
same way. The shared `_sma_seed()` helper (built for MACD) is reused here
via a new `_rma_sma_seeded()` wrapper — same seeding logic, Wilder's alpha
instead of span-based EMA.

**Result**: matched pandas_ta's ATR **exactly** (`0.0` difference) on the
first implementation attempt.

### ATR-based stop-loss

`suggested_stop_loss(price, atr, multiplier)` = `price − multiplier × atr`.
Multiplier fixed at the standard 2.0× (`config.TECHNICAL.atr_stop_multiplier`).
**Long-only** — this app's framing throughout (Kelly Criterion, DCF,
Quality Score) assumes going long, never shorting, so this is a downside
stop for a bought position, not a two-sided bracket. Returns `None` when
either input is missing — never a fabricated stop level. Displayed in the
existing "Execution & Position Sizing" section alongside the Kelly
Criterion allocation, not a separate panel — both answer "how do I manage
risk on this position."

## 7. Cross-validation methodology

Every indicator was checked against **pandas_ta** — a mature, widely-used
Python technical-analysis library whose non-TA-Lib code paths implement
the same formulas TA-Lib and TradingView use. This is the practical
substitute in an environment with no live TradingView access: not a
guess, but a cross-check against an independent, reputable implementation,
with the *source code itself* read whenever results diverged rather than
tweaking constants until numbers happened to match.

This caught two real, non-obvious bugs (MACD's and, pre-emptively, ATR's
EMA seeding — see §4 and §6) that a purely "eyeball the chart" validation
would have missed, since both errors are largest during the warm-up period
and shrink into apparent agreement over time.

| Indicator | Result | Match type |
|---|---|---|
| SMA | Exact | Element-wise equality vs. manual `rolling().mean()` |
| RSI | ~1e-14 max diff | Floating-point noise |
| MACD (Line/Signal/Histogram) | `0.00e+00` | Exact, after the seeding fix |
| Bollinger Bands | ~1e-13 max diff | Floating-point noise |
| ATR | `0.0` | Exact, first attempt |

Every indicator was also tested against a bank ticker (JPM) alongside a
normal company (AAPL) throughout this project — technical indicators don't
have the sector-structural gaps fundamentals do, so this mainly confirmed
no ticker-specific edge cases, rather than surfacing new ones the way it
did for the fundamentals validation reports.

## 8. Interactive dashboard

The price chart (`finance.py`) is built dynamically from which indicator
panels are toggled on in the sidebar:

- **Price panel** (row 1, always shown): candlesticks, the custom SMA line,
  the optional 20/50/200-day SMA trio, Bollinger Bands (shaded region
  between dotted upper/lower lines), SMA/Bollinger crossover and breakout
  markers, and **volume bars** — overlaid at the bottom of the price panel
  on a secondary y-axis (scaled to 4× the actual max so bars stay compact),
  the same space-efficient convention TradingView uses, rather than a
  dedicated 4th chart row.
- **RSI panel** (toggleable, "Show RSI Panel"): the RSI line, shaded
  overbought/oversold reference zones.
- **MACD panel** (toggleable, "Show MACD Panel"): histogram (colored by
  sign), MACD/Signal lines, crossover markers.

The chart is 1-3 rows depending on what's active — `row_heights` and each
trace's row number are computed from the active panel list, not hardcoded,
so toggling a panel off both shrinks the chart and reduces the number of
Plotly trace objects rendered (part of "optimize rendering," not just a
visual preference).

**Zoom, pan, and fullscreen** use Plotly's native modebar rather than a
custom-built control: `dragmode='zoom'` (drag-to-zoom on the chart itself)
and an explicit `config={'displayModeBar': True, 'scrollZoom': True,
'displaylogo': False}` on `st.plotly_chart()` — stated explicitly rather
than left to library defaults, so it's clear in the code that these
aren't accidentally disabled. The modebar's expand icon provides
fullscreen viewing.

Every indicator's current-value summary (the RSI badge, the SMA/MACD
crossover-signal tables, the Bollinger breakout table) stays visible
regardless of whether that indicator's *chart panel* is toggled on — hiding
a chart row declutters the visualization, it doesn't mean the underlying
information stops being useful.

## 9. Developer guide: adding a new indicator

1. Write the calculation as a function taking the cleaned OHLCV `df` (and
   any parameters) and returning either a new `pd.Series` or a copy of `df`
   with new columns — follow the existing `compute_*()` naming and
   docstring style (formula, null handling, any seeding/convention notes).
2. **Before writing the formula, check whether `pandas_ta` implements the
   same indicator, and read its source** — not just call it and compare
   outputs. Two indicators in the same "family" (Wilder smoothing, EMA)
   can seed their recursion differently; the only way to know is to read
   the actual code, as this file's own docstrings caught twice.
3. If the indicator needs crossover/breakout signals, follow the
   `detect_sma_crossovers()` pattern: classify each bar into states, diff a
   boolean/categorical series, and let the diff's transition points *be*
   the signals — this rules out duplicate/repeated firing by construction
   rather than requiring a separate deduplication step.
4. Any threshold, period default, or multiplier belongs in
   `config.TechnicalConfig` (or `ChartDefaults` for a sidebar
   slider's default/range) — never a magic number inline, the same
   convention the rest of this codebase follows.
5. Validate against pandas_ta (or another reputable independent
   implementation) on real data for at least one normal ticker (AAPL) —
   check the *entire* valid range's max difference, not just the most
   recent values, since a seeding discrepancy is largest during the
   warm-up period and can look like agreement if you only check the tail.

## 10. References

- Wilder, J. Welles Jr. *New Concepts in Technical Trading Systems.*
  Trend Research, 1978 — the original definition of RSI, ATR, and the
  smoothing technique ("Wilder's Moving Average" / RMA) both share.
- Appel, Gerald. *Technical Analysis: Power Tools for Active Investors.*
  Financial Times Prentice Hall, 2005 — MACD, developed by Appel in the
  late 1970s.
- Bollinger, John. *Bollinger on Bollinger Bands.* McGraw-Hill, 2001.
- [pandas_ta](https://github.com/twopirllc/pandas-ta) — the independent
  reference implementation used for cross-validation throughout this
  module's development; several of its own function source files (`ema()`,
  `rma()`, `atr()`, `bbands()`, `true_range()`) were read directly to
  confirm exact conventions rather than assumed from documentation alone.
