# Quantix Risk Analytics Documentation

This document explains every metric in [`risk_analytics.py`](risk_analytics.py)
— formulas, return conventions, null handling, and how each was validated
— plus the Altman Z-Score (in [`fundamental_analysis.py`](fundamental_analysis.py),
since it's statement-derived rather than price-derived) and the composite
Risk Dashboard that synthesizes all of them into one view.

For technical indicators (SMA/RSI/MACD/Bollinger/ATR), see
[`TECHNICAL_ANALYSIS.md`](TECHNICAL_ANALYSIS.md). For fundamentals, see
[`FUNDAMENTALS.md`](FUNDAMENTALS.md). For the data pipeline, see
[`ARCHITECTURE.md`](ARCHITECTURE.md).

## 1. Scope and data flow

`risk_analytics.py` is the sole place risk arithmetic happens —
`finance.py` calls into it and renders the results, the same separation
`technical_indicators.py` has for chart indicators.

```
data_loader.py            → raw OHLCV from Yahoo Finance
price_processing.py       → validated, deduplicated, gap-checked DataFrame
risk_analytics.py         → volatility, VaR/CVaR, drawdown, Sharpe/Sortino/Calmar, composite score
fundamental_analysis.py   → Altman Z-Score (statement-derived, not price-derived)
finance.py                → renders every panel, reads the results
```

**Returns convention**: every function here uses **logarithmic returns**
(`ln(P_t / P_t-1)`), not simple `pct_change()` returns — matching
Bloomberg/TradingView historical-volatility methodology, and giving
time-additive returns (daily log returns sum exactly to the period's total
log return, which is what makes `compute_annualized_return()`'s
`mean × trading_days` formula exact rather than an approximation). One
remaining ad-hoc consumer in `finance.py` (the Kelly Criterion's win/loss
rate) still reads simple returns — a separate, still-unmigrated feature
outside this section's scope.

**Never-fabricate convention**: every function returns `None` (or `NaN` for
warm-up periods in a series) rather than a plausible-looking substitute
when there isn't enough data — a flat/degenerate price series, a
short-lookback window, or a company type (banks/financials) the underlying
model doesn't apply to. The UI shows "N/A" or an explicit info message in
these cases, never a silently wrong number.

## 2. Historical Volatility

**Formulas**:

```
Log returns          = ln(Close_t / Close_t-1)
Rolling volatility    = rolling_std(log returns, window) × √trading_days_per_year
Full-range volatility = std(all log returns) × √trading_days_per_year
```

`compute_rolling_volatility()` uses `min_periods=window`, so the first
`window` bars are genuinely `NaN`. `compute_annualized_volatility()` is the
full-sample figure over the entire selected date range — the two answer
different questions ("volatility right now" vs. "volatility over this
whole period") and both are shown side by side.

**Configuration**: rolling window via the sidebar "Volatility Window"
slider (default 21 trading days ≈ 1 month), `√252` annualization factor
in `config.RISK.trading_days_per_year`.

**Validated against**: a from-scratch numpy/pandas reimplementation
(`std(ddof=1) × sqrt(252)`) — exact match — plus a manual bar-by-bar
rolling loop for the windowed version, and real AAPL data cross-checked
against an independent `numpy.std()` call on the identical return series.

## 3. Value at Risk (VaR)

Two independent methodologies, computed side by side so the gap between
them is itself informative (see §7):

**Historical VaR** (`compute_historical_var`) — non-parametric, reads the
`(1 − confidence)` percentile straight off the empirical return
distribution (`log_returns.quantile()`, pandas' default linear
interpolation — the same method Excel's `PERCENTILE.INC` and numpy's
default use).

**Parametric VaR** (`compute_parametric_var`) — the classic
variance-covariance/RiskMetrics approach: fits a normal distribution to
the sample mean/stdev of the same log-return window, then reads the same
quantile off the fitted normal curve instead of the raw sample
(`μ + z·σ`, `z = scipy.stats.norm.ppf(1 − confidence)`).

**Configuration**: confidence level via sidebar selectbox (90/95/99%,
`config.RISK.var_confidence_levels`), lookback via sidebar slider (default
252 trading days ≈ 1 year, the RiskMetrics-style industry standard —
deliberately longer than the volatility window above, since a stable tail
percentile needs far more observations than a reactive short-term figure).

**Sign convention**: returned as a signed log return — negative for a
loss (e.g. `-0.02` = "the 5th-percentile day loses 2%") — not flipped to a
positive "loss magnitude."

**Minimum observations**: both return `None` below
`config.RISK.var_min_observations` (20) — a percentile or normal fit off a
handful of points isn't a meaningful tail estimate.

**Validated against**: `compute_historical_var()` — exact match vs. manual
`numpy.percentile(..., method="linear")` across 90/95/99% confidence.
`compute_parametric_var()` — exact match vs. manual `scipy.stats.norm.ppf()`,
plus a hand-worked textbook case (known mean/stdev) confirming the
closed-form answer to floating-point precision.

## 4. Expected Shortfall (CVaR)

**Formula**: the mean of every log return at or below the Historical VaR
threshold — `mean(log_returns[log_returns <= VaR_threshold])`, over the
same lookback/confidence as §3. This is strictly more informative than VaR
alone: VaR answers "what's the loss at the cutoff", CVaR answers "given
that the loss exceeds the cutoff, what's the *average* loss" — the reason
Basel III moved bank capital requirements from VaR to Expected Shortfall.

**Interpretation** (`interpret_tail_risk`): a ready-to-render sentence
quantifying the gap between VaR and CVaR in plain language.

**Visualization**: a histogram of the log-return distribution over the
lookback window, with vertical VaR/CVaR marker lines — the first histogram
in the app (everything else is a time series) since this is the one metric
where the *shape* of the distribution, not just a single number, is the
point.

**Validated against**: exact match vs. a manual tail-mean computation on a
deliberately fat-tailed (Student-t distributed, not normal) synthetic
sample, plus a hand-worked 5-observation textbook case. Structural
invariant (CVaR must always be ≤ VaR, further into the loss tail) checked
and held on every test case.

## 5. Maximum Drawdown

**Formula**:

```
Drawdown series = (price − running_peak) / running_peak
Max Drawdown     = min(drawdown series)
Recovery         = trading days from the trough until price first closes
                    back at/above the prior peak
```

**Scale invariance**: the formula works identically on a raw price series
or a cumulative-return/equity-curve index (`(1+returns).cumprod()`) — the
same function (`compute_max_drawdown`) computes both the buy-and-hold
drawdown shown in the Risk section *and* the mean-reversion Backtest
strategy's own drawdown (that inline calculation was refactored to call
this shared function rather than duplicating the `cummax()`/drawdown math
a second time).

**Recovery, not just magnitude**: `MaxDrawdownResult` carries peak/trough
dates and prices in addition to the percentage, since "detect peak" /
"detect trough" are the metric's whole point, not incidental detail. A
drawdown that hasn't recovered by the end of the selected date range
reports `recovered=False` / `recovery_days=None` — shown as "Ongoing" in
the UI — rather than a fabricated recovery date.

**Visualization**: an "underwater chart" — % decline from the running peak
over time, filled area, worst point marked with a diamond.

**Validated against**: a hand-constructed 8-point series with a
known-by-eye answer (peak 120 → trough 80 → recovers at 130 = exactly
-33.33%, 2-day recovery) — exact match. Edge cases: monotonically
increasing series → 0% drawdown; never-recovered series → `recovered=False`.
Real AAPL data (5-year history) correctly identified the Dec 2024 peak →
April 2025 selloff trough, a real, independently checkable -33.36% event.

## 6. Sharpe, Sortino, and Calmar Ratios

All three share one annualized-return numerator
(`compute_annualized_return()` = `mean(log returns) × trading_days` — the
log-return analogue of CAGR, exact because log returns telescope to
`ln(P_end/P_start)`), differing only in what they divide by:

| Ratio | Denominator | What it penalizes |
|---|---|---|
| **Sharpe** | Annualized volatility (§2) | All volatility, upside and downside equally |
| **Sortino** | Downside deviation (below) | Only returns below a target (0 by default) |
| **Calmar** | \|Maximum Drawdown\| (§5) | The single worst realized loss, not variance |

**Downside deviation** (`compute_downside_deviation`) — the textbook
Sortino/Fouse semi-deviation formula:

```
Downside deviation = sqrt(mean(min(R_i − target, 0)²)) × √trading_days
```

averaged over **every** observation (a day that beats the target
contributes exactly zero), not `std()` of only the negative-return subset
— a meaningfully different, non-standard formula the app used before this
section's Sortino task, which both narrowed the sample *and*
mean-centered on that narrowed subset rather than measuring shortfall
below a fixed target across the full sample.

**Configurable risk-free rate**: sidebar slider (default 4%, matching
`config.RISK.risk_free_rate`), feeding Sharpe/Sortino only — not the DCF's
CAPM cost-of-equity, which keeps its own fixed assumption from the same
config constant (touching that would be fundamentals-module scope creep).

**Undefined, not infinite**: all three return `None` when volatility /
downside deviation / drawdown is exactly zero, rather than `0` (which
reads identically to "no edge" instead of "couldn't be computed") or `inf`.

**Interpretation** (`interpret_sharpe_ratio`, `interpret_calmar_ratio`):
5-tier quality bands (Poor / Sub-optimal-or-Below-Average / Good / Very
Good / Exceptional-or-Excellent) with an explicit "verify" caveat on the
top tier — an unusually high ratio on a real asset more often signals a
short or low-volatility sample than genuinely exceptional performance.
Sharpe's interpretation also states its own textbook limitation (penalizes
upside/downside equally, assumes near-normal returns) and points at
Sortino as the downside-only complement.

**Validated against**: exact matches vs. manual numpy/scipy computations
across multiple risk-free rates, hand-worked textbook cases with known
closed-form answers, and a skew-sensitivity sanity check — on a
deliberately positively-skewed synthetic series, Sortino read
meaningfully higher than Sharpe on the identical data, exactly the
behavior it exists to produce.

## 7. Altman Z-Score (Financial Distress)

Lives in `fundamental_analysis.py`'s `altman_z_score()` (statement-derived,
not price-derived — see [`FUNDAMENTALS.md`](FUNDAMENTALS.md) for the rest
of that module). Documented here because it's this section's distress
metric, feeding both the composite Risk Score (§9) and its own dedicated
display.

**Formula** (original Altman 1968, public manufacturers):

```
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 0.999·X5
X1 = (Current Assets − Current Liabilities) / Total Assets
X2 = Retained Earnings / Total Assets
X3 = EBIT / Total Assets
X4 = Market Cap / Total Liabilities
X5 = Total Revenue / Total Assets
```

**Zones**: Safe (`Z > 2.99`) / Grey (`1.81 ≤ Z ≤ 2.99`) / Distress
(`Z < 1.81`) — `config.RISK.altman_safe_zone` / `altman_grey_zone`, the
original published thresholds.

**Scope decision**: this app deliberately does **not** implement the
alternate Z′/Z″ formulas (for private companies / non-manufacturers,
which drops the asset-turnover term). Companies without a classified
balance sheet — banks, financials — correctly return `None` /
"Insufficient Financial Data" rather than a different model silently
substituted in; that's an honest reflection of the original model's scope,
not a gap to paper over.

**Input hardening**: every required field is checked for `None`
(genuinely missing), *and* `Total Assets`/`Total Liabilities` — the
formula's two denominators — are checked for `≤ 0`, which is accounting-
invalid data (never a legitimate "not reported," unlike a bank's missing
current-asset breakdown) rather than merely absent. Both cases populate a
descriptive reason in `altman_missing_inputs` — previously, a zero (but
technically "present") Total Assets value produced an **empty** missing-fields
list alongside a generic "Insufficient Financial Data" verdict, which the
UI's `st.warning()` call only renders when the list is non-empty — so the
zero-denominator case silently showed *no explanation at all*. Fixed as
part of this section's validation task.

**Validated against**: a hand-worked formula check (exact match to a
manually computed Z from known X1-X5 inputs), zone-boundary tests
confirming the verdict text always matches which side of each threshold
the computed Z lands on, and real-ticker checks through the full pipeline
— AAPL (Z=11.87) and MSFT (Z=8.89) both land solidly in the Safe Zone,
consistent with what's publicly cited for large-cap tech balance sheets;
JPM correctly returns `None` (no classified balance sheet — expected, not
a bug).

## 8. Composite Risk Score

`compute_risk_score()` combines every metric above into one 0-100 figure
for the Risk Dashboard, using the same weighted-average-of-normalized-
signals pattern `data_quality.py`'s `assess_data_quality()` established
for the Data Quality Report.

**Normalization**: each raw metric is mapped to a 0-100 sub-score via
`config.RISK.risk_score_*_anchors` — a `(best, worst)` pair, linearly
interpolated and clamped, where "best" always maps to 100 regardless of
whether the metric's own natural direction is "higher is better" (Sharpe)
or "closer to zero is better" (VaR, drawdown).

| Factor | Weight | Best anchor | Worst anchor |
|---|---|---|---|
| Annualized Volatility | 15% | 15% | 60% |
| 1-Day Historical VaR | 15% | 0% | -8% |
| Expected Shortfall (CVaR) | 15% | 0% | -12% |
| Maximum Drawdown | 20% | 0% | -60% |
| Sharpe Ratio | 10% | 2.5 | 0.0 |
| Sortino Ratio | 10% | 2.5 | 0.0 |
| Calmar Ratio | 10% | 5.0 | 0.0 |
| Altman Z-Score | 5% | Safe zone (2.99) | 0.0 |

Maximum Drawdown carries the highest single weight — the most viscerally
consequential risk for a long-only holding — while Altman Z carries the
lowest, since it's frequently unavailable (banks/financials) and measures
a slower-moving, statement-based risk rather than the market-price risk
the other seven factors share.

**Missing-factor handling**: a factor that can't be computed (most often
Altman Z for a bank) is **excluded** from both the weighted numerator and
the weight-sum denominator, and the remaining weights implicitly
renormalize — the same "don't penalize what can't be checked" principle
`data_quality.py` uses for field completeness, never treated as a zero.

**Grade bands**: Low Risk (≥75) / Moderate Risk (≥50) / Elevated Risk
(≥30) / High Risk (<30).

**Validated against**: a hand-worked case with all 8 factors set to known
values landing at exact sub-score midpoints/anchors (100/0/50/50/50/100/0/100)
— composite score matched a manual weighted-average computation exactly.
A second case with Altman Z excluded confirmed the renormalization drops
the excluded weight from *both* the numerator and denominator (not zeroed
into just the numerator) by matching a manual recomputation over the
remaining 7 factors. Clamping verified: values beyond either anchor still
produce exactly 0 or 100, never overshoot. A realistic AAPL-like input set
(matching the live app's actual values) produced a sensible score in the
"Moderate Risk" band — appropriate for a single stock with strong
fundamentals but real market-price volatility, neither an artificially
rosy "Low Risk" nor an alarmist "High Risk."

## 9. Risk Dashboard

A summary panel placed at the top of the risk section (right after
"Interactive Price & Technicals," before the detailed per-metric panels
below it) — mirrors the existing Data Quality Report's "one grade up top,
details below" pattern.

- **Composite Risk Score gauge**: a Plotly `go.Indicator` gauge (0-100),
  colored by grade, with shaded threshold bands matching the grade
  boundaries.
- **8 factor cards**: one `st.metric()` per risk factor, showing the raw
  value, a color-coded zone icon (🟢/🟡/🔴, or ⚪ if not computable for
  this ticker), and its weight in the composite score.
- **Dynamic updates**: built entirely from the same reactive
  ticker/date-range/sidebar-driven variables every other panel uses — no
  separate caching or state, so it updates on every rerun exactly like the
  rest of the app.
- **Detail panels remain**: the Risk Dashboard is a synthesis view, not a
  replacement — every detailed panel below it (Volatility, VaR/CVaR
  histogram, Max Drawdown underwater chart, Sharpe/Sortino/Calmar with
  their own interpretation captions) stays fully intact for drill-down.

## 10. Developer guide: adding a new risk metric

1. Write the calculation as a function taking the cleaned OHLCV `df` (or,
   for a metric built from other metrics like Calmar, their already-
   computed values) and returning a plain value or a small `@dataclass` —
   follow the existing `compute_*()` naming and docstring style (formula,
   return/sign convention, null-handling behavior).
2. Use log returns (`compute_log_returns()`), not `pct_change()`, for
   internal consistency with every other function in this module.
3. Return `None` — never `0`, `inf`, or a fabricated fallback — when there
   isn't enough data or the metric is genuinely undefined (e.g. dividing
   by zero volatility/drawdown). Let the UI render "N/A" or an explicit
   info message.
4. Any threshold, anchor, or weight belongs in `config.RiskConfig` (or
   `ChartDefaults` for a sidebar slider's default/range) — never a magic
   number inline.
5. Validate against a hand-worked case with a known, checkable-by-eye
   answer, plus at least one edge case (insufficient data, a degenerate
   zero-variance series) and real ticker data — the same three-tier
   validation approach used for every metric in this document.
6. If the metric is meant to feed the composite Risk Score, add it to
   `compute_risk_score()`'s candidate list with sensible `(best, worst)`
   anchors and a weight, then confirm the total weight still sums to 1.0
   across whichever factors are actually available.

## 11. References

- Jorion, Philippe. *Value at Risk: The New Benchmark for Managing
  Financial Risk.* McGraw-Hill, 2006 — VaR/CVaR methodology, including the
  historical and variance-covariance approaches implemented here.
- Sortino, Frank A., and Lee N. Price. "Performance Measurement in a
  Downside Risk Framework." *Journal of Investing*, 1994 — the downside
  deviation / Sortino Ratio formula.
- Young, Terry W. "Calmar Ratio: A Smoother Tool." *Futures Magazine*,
  1991 — the Calmar Ratio's original context (managed futures / trend
  following).
- Altman, Edward I. "Financial Ratios, Discriminant Analysis and the
  Prediction of Corporate Bankruptcy." *Journal of Finance*, 1968 — the
  original Z-Score model implemented here (public manufacturers).
- Basel Committee on Banking Supervision. *Minimum Capital Requirements
  for Market Risk.* Bank for International Settlements, 2019 — the shift
  from VaR to Expected Shortfall in bank capital regulation, cited in §4.
