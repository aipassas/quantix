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
technical_indicators.py   → SMA, RSI, MACD, Bollinger Bands, ATR,
                             Stochastic, Anchored VWAP, ADX, Ichimoku, OBV
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

## 7. Stochastic Oscillator

**Formula** (the "Slow Stochastic" — see note below):

```
Raw %K = 100 × (Close − Lowest Low(k_period)) / (Highest High(k_period) − Lowest Low(k_period))
%K     = SMA(Raw %K, smooth_k)
%D     = SMA(%K, d_period)
```

**Fast vs. Slow Stochastic**: the original spec for this task described the
textbook "Fast Stochastic" (%D = SMA(3) of the *raw* %K directly). Reading
pandas_ta's `stoch()` source showed its actual default output — the one
TradingView also plots by default — is the **Slow Stochastic**: raw %K is
first smoothed by `SMA(smooth_k)` (default 3) into what this codebase calls
`Stoch_K`, and *that* smoothed line, not the raw one, feeds the `SMA(d_period)`
that produces `Stoch_D`. Implemented the Slow variant, since genuine
TradingView parity was the point, not the literal wording of the original
note.

**Configuration**: `k_period` (sidebar "Stochastic %K Length", default 14),
`d_period` (fixed at 3), `smooth_k` (fixed at 3) —
`config.TECHNICAL.stochastic_k_period/d_period/smooth_k`. Overbought/oversold
thresholds fixed at 80/20 (`stochastic_overbought`/`stochastic_oversold`),
the standard convention (distinct from RSI's 70/30).

**Crossover signals** (`detect_stochastic_crossovers`): same
dropna-first, diff-based edge-detection technique as every other crossover
detector in this module — %K crossing above %D is bullish, below is bearish.

**Validated against**: pandas_ta's `stoch()` — `Stoch_K`/`Stoch_D` matched to
within floating-point noise (~5.7e-14) across the full valid range on real
data. Crossover count cross-checked independently by re-deriving the
transition count from a dropna-first boolean diff (matched exactly) — an
early version of the *verification script itself* (not the product code)
double-counted one warm-up-boundary transition by computing `above` on the
full, not-yet-dropna'd series, where `NaN > NaN` evaluates to `False` rather
than `NaN`; fixed by dropping NaN first, mirroring what
`detect_stochastic_crossovers()` already did correctly throughout.

## 8. Anchored VWAP

**Formula**:

```
Typical Price = (High + Low + Close) / 3
VWAP          = cumulative(Typical Price × Volume) / cumulative(Volume), from anchor_date forward
```

**Deliberately not TradingView's intraday, session-resetting VWAP.** Quantix
only has daily bars — a daily "session reset" would produce a single-bar
VWAP, which is meaningless. An **anchored** VWAP, cumulative from a
user-chosen start date, is the daily-bar equivalent that's actually
informative: "the volume-weighted average price since this reference
point." Labeled "Anchored VWAP" in the UI specifically so it isn't mistaken
for the intraday convention.

**Configuration**: `anchor_date` via the sidebar "VWAP Anchor Date" picker
(only shown once "Show Anchored VWAP" is toggled on), defaulting to the
first date in the loaded price history if left blank.

**Null handling**: bars before `anchor_date` are `NaN` — VWAP is undefined
before its own anchor point, never backfilled or zero-filled.

**Validated against**: a hand-worked 3-bar manual calculation (both the
default anchor and a mid-series anchor, checking that pre-anchor bars are
correctly `NaN`) — matched to within `1e-9`.

## 9. Average Directional Index (ADX)

**Formula** (Wilder, 1978 — a trend-**strength** indicator, not direction):

```
+DM = max(High − PrevHigh, 0) if (High − PrevHigh) > (PrevLow − Low) else 0
-DM = max(PrevLow − Low, 0)   if (PrevLow − Low) > (High − PrevHigh) else 0
+DI = 100 × WilderSmooth(+DM, period) / ATR(period)
-DI = 100 × WilderSmooth(-DM, period) / ATR(period)
DX  = 100 × |+DI − -DI| / (+DI + -DI)
ADX = WilderSmooth(DX, period)
```

**Configuration**: period via the sidebar "ADX Length" slider (default 14).
Trend-strength threshold fixed at the standard 25 (`config.TECHNICAL.adx_trend_threshold`),
drawn as a reference line on the ADX panel (ADX above it: trending; below:
non-trending — direction still comes from comparing +DI/-DI, not ADX itself).

### The seeding bug this task caught — a genuinely harder sequel to MACD's

This was the standout technical challenge of the "Additional Technical
Indicators" work, and a two-part, *compounding* version of the same class of
bug MACD's seeding caught (§4) and ATR's seeding pre-empted (§6).

**First attempt**: reused this module's own `compute_atr()` (already
verified exact against pandas_ta standalone, §6) as ADX's internal ATR
denominator. Result: `Plus_DI`/`Minus_DI` matched pandas_ta *exactly*
(`0.00e+00`), but final `ADX` diverged by ≈0.49 — not floating-point noise.

**The trap**: DI matching exactly while ADX doesn't looks like it should be
impossible, since ADX is *derived from* DI (via DX). It isn't — DX itself
matched pandas_ta exactly on the overlapping non-NaN range; the divergence
was purely about **which bar each series' recursion effectively starts
from**, something that doesn't show up in a plain value-diff on the
overlapping range but does compound once a second, unseeded `.ewm()` stage
(DX → ADX) is chained on top.

**Root cause, found by reading pandas_ta's source directly** (not
guessed, not tuned to match — the same discipline as §4 and §6), had two
independent parts:

1. `pandas_ta.adx()` calls its own `atr()` internally with **`prenan=True`**
   — forcing the very first True Range bar to `NaN` before seeding — which
   differs from `atr()`'s own **standalone** default of `prenan=False` (the
   convention this codebase's own `compute_atr()` correctly matches for
   ATR-as-an-indicator, §6). Same function, different default depending on
   whether it's called standalone or as ADX's internal building block.
2. Given that leading `NaN`, `atr()`'s SMA-seed step (`presma=True`) uses a
   **positional** slice — `tr[0:length].mean()`, skipping the `NaN` via
   pandas' default `skipna` — not the "drop `NaN` first, then take the first
   `length` valid values" approach this module's existing `_sma_seed()`
   helper uses for MACD/ATR. The two approaches are **identical** whenever
   there's no leading `NaN` in the seed window (true for every other seeded
   indicator already in this codebase) — and diverge specifically when
   `prenan=True` introduces one, exactly ADX's internal case.

**Debugging trail** (documented in full in `_sma_seed_positional()` and
`_atr_for_adx()`'s docstrings in `technical_indicators.py`): removing
`min_periods` from the shared `_rma_unseeded()` helper was tried first and
made the divergence *worse* (0.49 → 1.28) — a genuine wrong turn, corrected
by manually replicating pandas_ta's algorithm step-by-step with this
codebase's own helpers (reproduced the same divergence, confirming the
algorithm structure was right and a building block was wrong), then
rebuilding the same chain with pandas_ta's *own* `atr()`/`ma()` functions
directly (got an exact `0.0` diff, isolating the bug to this module's ATR
substitute specifically), then comparing `first_valid_index()` across
intermediate series (found a genuine one-bar offset — pandas_ta's internal
series all started one trading day earlier than this module's).

**Fix**: `_atr_for_adx()` — a dedicated internal ATR helper (distinct from
the public `compute_atr()`) that forces `tr.iloc[0] = NaN`, then seeds via
the new `_sma_seed_positional()` helper, then applies bare
`.ewm(alpha=1/period, adjust=False)`. `_rma_unseeded()` — bare Wilder
smoothing with **no** `min_periods` constraint (the correct final state;
`compute_adx()` chains it across *two* stages, +DM/-DM then DX→ADX, and
adding a warm-up mask to the first stage was what shifted the second
stage's effective starting bar in the first place).

**Result**: `ADX`, `Plus_DI`, and `Minus_DI` all match pandas_ta **exactly**
— `0.00e+00` difference across the full overlapping valid range.

## 10. Ichimoku Cloud

**Formula** ("one-glance equilibrium chart"):

```
Tenkan-sen (Conversion) = midprice(High, Low, tenkan_period)
Kijun-sen (Base)        = midprice(High, Low, kijun_period)
Senkou Span A           = (Tenkan + Kijun) / 2, plotted kijun_period − 1 bars ahead
Senkou Span B           = midprice(High, Low, senkou_b_period), plotted kijun_period − 1 bars ahead
Chikou Span             = Close, plotted kijun_period − 1 bars behind
```

where `midprice(H, L, n) = (rolling_max(H, n) + rolling_min(L, n)) / 2`.

**Configuration**: fixed at the standard 9/26/52 periods
(`config.TECHNICAL.ichimoku_tenkan_period/kijun_period/senkou_b_period`) —
per the original task spec, these are not user-tunable via the sidebar the
way SMA/RSI/ADX periods are.

### Why this returns two DataFrames, not one

pandas_ta's own `ichimoku()` returns a `(historical, forward)` pair, and
`compute_ichimoku()` mirrors that exactly via the `IchimokuResult`
dataclass — confirmed by reading pandas_ta's source directly that this is
its actual, intentional design, not incidental:

- **`historical`** — index matches the input `df` exactly. Senkou A/B are
  shifted `kijun_period − 1` bars **backward**, so today's plotted cloud
  value is what was actually computed `kijun_period − 1` bars ago (the
  standard Ichimoku convention — the cloud you see today was "cast" in the
  past).
- **`forward`** — genuinely extends the date index **beyond** the last
  observed bar, by `kijun_period` new business days
  (`pd.bdate_range(start=last_date + 1 day, periods=kijun_period)`), holding
  the real forward-projected Senkou A/B values. A fixed, input-matching
  index literally cannot hold real future-dated cloud values, so a second
  DataFrame is the only way to represent them at all.

This is a standard, universally-understood **forward projection** — not
data fabrication in the sense this codebase otherwise strictly avoids
(inventing missing historical observations). The forward cloud is clearly
a projection of already-known High/Low history, plotted on real future
calendar dates, the same way any charting platform draws it.

**Validated against**: pandas_ta's `ichimoku()` — all 5 historical
components (Tenkan, Kijun, Senkou A, Senkou B, Chikou) and both forward
Senkou A/B values matched **exactly** (`0.00e+00`), and the forward
DataFrame's date index matched pandas_ta's own forward index element-for-element.

## 11. On-Balance Volume (OBV)

**Formula** (Granville, 1963):

```
OBV = cumulative sum of: +Volume (Close up), −Volume (Close down), 0 (Close unchanged)
```

**First-bar behavior — deliberately replicated, not "fixed"**: pandas_ta's
`obv()` produces `NaN` for the very first bar rather than an assumed
starting value, traced to a shadowed `initial` parameter in its internal
`signed_series()` helper that always resolves to `None` (→ `NaN`)
regardless of the `initial=1` argument `obv()` itself passes — arguably an
unintentional quirk in pandas_ta's own code, confirmed empirically (not
just theorized) via a direct small-scale `ta.obv()` test. This module
replicates it deliberately rather than "fixing" it to some other
convention, because it's actually philosophically **consistent** with this
codebase's own rule throughout: the first bar has no prior Close to compare
against, so its direction is genuinely undefined — `NaN`, never a fabricated
"+1" assumption.

**Configuration**: none — OBV has no period or threshold to tune.

**Validated against**: pandas_ta's `obv()` — matched exactly
(`0.00e+00`), including the matching `NaN` first-bar behavior on both sides.

## 12. Cross-validation methodology

Every indicator was checked against **pandas_ta** — a mature, widely-used
Python technical-analysis library whose non-TA-Lib code paths implement
the same formulas TA-Lib and TradingView use. This is the practical
substitute in an environment with no live TradingView access: not a
guess, but a cross-check against an independent, reputable implementation,
with the *source code itself* read whenever results diverged rather than
tweaking constants until numbers happened to match.

This caught three real, non-obvious seeding bugs (MACD's, ADX's, and,
pre-emptively, ATR's — see §4, §9, and §6) that a purely "eyeball the
chart" validation would have missed, since seeding errors are largest
during the warm-up period and shrink into apparent agreement over time.
ADX's was the hardest of the three: a two-part, compounding discrepancy
that left the intermediate +DI/-DI values matching exactly while the final
ADX still diverged — see §9 for the full debugging trail.

| Indicator | Result | Match type |
|---|---|---|
| SMA | Exact | Element-wise equality vs. manual `rolling().mean()` |
| RSI | ~1e-14 max diff | Floating-point noise |
| MACD (Line/Signal/Histogram) | `0.00e+00` | Exact, after the seeding fix |
| Bollinger Bands | ~1e-13 max diff | Floating-point noise |
| ATR | `0.0` | Exact, first attempt |
| Stochastic (%K/%D) | ~5.7e-14 max diff | Floating-point noise |
| Anchored VWAP | ~1e-9 max diff | Floating-point noise, vs. hand-worked manual calc |
| ADX / +DI / -DI | `0.00e+00` | Exact, after the seeding fix |
| Ichimoku (5 historical + 2 forward components) | `0.00e+00` | Exact |
| OBV | `0.00e+00` | Exact, including matching NaN first bar |

Every indicator was also tested against a bank ticker (JPM) alongside a
normal company (AAPL) throughout this project — technical indicators don't
have the sector-structural gaps fundamentals do, so this mainly confirmed
no ticker-specific edge cases, rather than surfacing new ones the way it
did for the fundamentals validation reports.

## 13. Interactive dashboard

The price chart (`finance.py`) is built dynamically from which indicator
panels are toggled on in the sidebar:

- **Price panel** (row 1, always shown): candlesticks, the custom SMA line,
  the optional 20/50/200-day SMA trio, Bollinger Bands (shaded region
  between dotted upper/lower lines), SMA/Bollinger crossover and breakout
  markers, the optional Anchored VWAP line and Ichimoku Cloud overlay
  (Tenkan-sen/Kijun-sen/Chikou Span lines plus the shaded Senkou A/B cloud,
  drawn continuously across the historical *and* forward-projected segments
  — see §10), and **volume bars** — overlaid at the bottom of the price
  panel on a secondary y-axis (scaled to 4× the actual max so bars stay
  compact), the same space-efficient convention TradingView uses, rather
  than a dedicated extra chart row.
- **RSI panel** (toggleable, "Show RSI Panel"): the RSI line, shaded
  overbought/oversold reference zones.
- **MACD panel** (toggleable, "Show MACD Panel"): histogram (colored by
  sign), MACD/Signal lines, crossover markers.
- **Stochastic panel** (toggleable, "Show Stochastic Panel"): %K/%D lines,
  shaded 80/20 overbought/oversold reference zones, crossover markers.
- **ADX panel** (toggleable, "Show ADX Panel"): ADX, +DI, -DI lines, a
  reference line at the trend-strength threshold (25).
- **OBV panel** (toggleable, "Show OBV Panel"): the OBV line.

All 5 newer panels/overlays (Stochastic, VWAP, Ichimoku, ADX, OBV) default
**off** — a deliberate choice so a first-time chart stays exactly as
uncluttered as it was before this indicator set existed; a user who wants
them opts in per-indicator.

The chart is 1-6 rows depending on what's active — `row_heights` and each
trace's row number are computed from the active panel list, not hardcoded.
The price panel always keeps half the total chart height; any active
oscillator rows (RSI/MACD/Stochastic/ADX/OBV) split the remaining half
evenly between however many of them are on. Toggling a panel off both
shrinks the chart and reduces the number of Plotly trace objects rendered
(part of "optimize rendering," not just a visual preference).

**Zoom, pan, and fullscreen** use Plotly's native modebar rather than a
custom-built control: `dragmode='zoom'` (drag-to-zoom on the chart itself)
and an explicit `config={'displayModeBar': True, 'scrollZoom': True,
'displaylogo': False}` on `st.plotly_chart()` — stated explicitly rather
than left to library defaults, so it's clear in the code that these
aren't accidentally disabled. The modebar's expand icon provides
fullscreen viewing.

Every indicator's current-value summary (the RSI badge, the SMA/MACD/
Stochastic crossover-signal tables, the Bollinger breakout table) stays
visible regardless of whether that indicator's *chart panel* is toggled on
— hiding a chart row declutters the visualization, it doesn't mean the
underlying information stops being useful.

## 14. Developer guide: adding a new indicator

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

## 15. References

- Wilder, J. Welles Jr. *New Concepts in Technical Trading Systems.*
  Trend Research, 1978 — the original definition of RSI, ATR, ADX/DMI, and
  the smoothing technique ("Wilder's Moving Average" / RMA) they share.
- Appel, Gerald. *Technical Analysis: Power Tools for Active Investors.*
  Financial Times Prentice Hall, 2005 — MACD, developed by Appel in the
  late 1970s.
- Bollinger, John. *Bollinger on Bollinger Bands.* McGraw-Hill, 2001.
- Lane, George C. — developer of the Stochastic Oscillator in the late
  1950s.
- Hosoda, Goichi. *Ichimoku Kinkō Hyō* — published under the pen name
  "Ichimoku Sanjin," 1969.
- Granville, Joseph. *Granville's New Key to Stock Market Profits.*
  Prentice-Hall, 1963 — On-Balance Volume.
- [pandas_ta](https://github.com/twopirllc/pandas-ta) — the independent
  reference implementation used for cross-validation throughout this
  module's development; several of its own function source files (`ema()`,
  `rma()`, `atr()`, `bbands()`, `true_range()`, `stoch()`, `adx()`,
  `ichimoku()`, `obv()`) were read directly to confirm exact conventions
  rather than assumed from documentation alone.
