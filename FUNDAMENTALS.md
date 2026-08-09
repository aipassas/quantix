# Quantix Fundamental Analysis Documentation

This document explains every ratio, formula, and assumption inside
[`fundamental_analysis.py`](fundamental_analysis.py) — the module that turns
a company's financial statements into the Master Matrix, Strategic
Investment Scorecard, the four category validation reports, the Financial
Metrics Validation Report, the Company Quality Classification, Altman
Z-Score, and the DCF valuation. It exists so a future contributor can
understand *why* a number is calculated the way it is without reverse-
engineering it from the code, and so every non-obvious judgment call this
project has made is written down in one place instead of scattered across
commit history.

For how data gets *into* this module (fetching, validation, standardization,
quality scoring), see [`ARCHITECTURE.md`](ARCHITECTURE.md). This document
starts where that one ends: a `StandardizedFinancials` object (canonical
units, one field per concept) and an optional raw Yahoo `info` dict, both
already in hand.

## 1. Scope

`fundamental_analysis.py` covers statement-derived fundamentals only:
profitability, leverage, liquidity, cash-flow quality, valuation, distress
(Altman Z), and discounted cash flow — everything a company's balance sheet,
income statement, and cash flow statement can tell you.

**Explicitly out of scope**: any metric computed from *price returns*
rather than financial statements (Sharpe ratio, Sortino ratio, VaR, CVaR,
maximum drawdown, Hurst exponent, Kelly position sizing) belongs to the risk
analytics layer in `finance.py` and is not documented here. If a future
"Risk Analytics documentation" task is undertaken, it should cover those.

Every function in this module takes a `StandardizedFinancials` object (via
`self.std` on `FundamentalAnalysisEngine`) and returns `None` — never raises
— when a required input is missing. A `None` result renders as `N/A`
downstream; it is never silently treated as zero except where explicitly
documented (Total Debt, Retained Earnings, Inventory, Cash & Equivalents,
and Depreciation & Amortization all default to `0` on `StandardizedFinancials`
itself, upstream of this module — see `ARCHITECTURE.md` §4).

## 2. How to read this document

Every ratio below is documented the same way:

- **Formula** — the exact calculation, in the same notation used in code.
- **Inputs** — which `StandardizedFinancials` fields it reads.
- **Null handling** — what makes it return `None` instead of a number.
- **Assumptions** — any judgment call baked into the formula, and why.
- **Cross-check** — if a `validate_*()` method compares this metric against
  Yahoo's own independently-reported figure, what field it uses.

Where a formula reflects a specific accounting or valuation standard (Altman
Z-Score, CAPM, NOPAT), the standard is named explicitly in §8.

## 3. Profitability

Computed by `gross_margin_pct()`, `operating_margin_pct()`, `roa_pct()`,
`net_margin_pct_computed()`, `roe_pct_computed()`, `roic_pct()`. Cross-checked
as a group by `validate_profitability()`.

| Metric | Formula | Null when | Cross-check (Yahoo field) |
|---|---|---|---|
| Net Margin | Net Income / Total Revenue | Net Income or Total Revenue missing | `profitMargins` |
| Gross Margin | Gross Profit / Total Revenue | Gross Profit missing (absent for banks/financials — no cost of revenue) | `grossMargins` |
| Operating Margin | Operating Income / Total Revenue | Operating Income missing (absent for banks/financials) | `operatingMargins` |
| Return on Assets (ROA) | Net Income / Total Assets | Net Income or Total Assets missing | `returnOnAssets` |
| Return on Equity (ROE) | Net Income / Stockholders Equity | Net Income or Stockholders Equity missing | `returnOnEquity` |
| Return on Invested Capital (ROIC) | NOPAT / (Total Debt + Stockholders Equity) | EBIT or Stockholders Equity missing, or Total Debt + Equity = 0 | *(none — Yahoo doesn't report ROIC)* |

**ROE assumption**: a large or negative ROE is not treated as an error. A
company with small or negative book equity from sustained share buybacks
(Apple is the textbook example) will legitimately produce a triple-digit or
negative ROE — the formula is correct, the company's capital structure is
just unusual. `OUTLIER_BOUNDS.max_abs_roe_pct` (500%) is set well above this
to avoid flagging it as a data error (see §9).

**ROIC methodology — NOPAT, not raw EBIT**: `roic_pct()` tax-adjusts EBIT
before dividing by invested capital:

```
NOPAT = EBIT × (1 − effective tax rate)
ROIC  = NOPAT / (Total Debt + Stockholders Equity)
```

This is the standard institutional ROIC definition. Dividing raw, pre-tax
EBIT by invested capital (an earlier version of this engine did exactly
that) overstates the return, because it counts money that will be paid out
as tax as if it were available to capital providers.

**Effective tax rate** (`effective_tax_rate()`): Tax Provision / Pretax
Income when Pretax Income is positive and the resulting rate falls in
`[0%, 50%]`; otherwise falls back to the assumed statutory rate
(`config.DCF.tax_rate`, 21% — the US federal corporate rate). A company's
own reported rate is more accurate when available (multinational effective
rates routinely sit below the US statutory rate), but one-time tax items
can produce a nonsensical ratio, hence the sanity range. Whether the
fallback fired is exposed via `effective_tax_rate_used_fallback()` and
surfaced in the Financial Metrics Validation Report as an incomplete-
calculation flag (§9).

## 4. Liquidity

Computed by `current_ratio_computed()` and `quick_ratio_computed()`.
Cross-checked by `validate_liquidity()`. **Informational only** — neither
feeds the Scorecard or Master Matrix; Current Ratio there stays sourced
from Yahoo directly (`standardized.current_ratio`), and Quick Ratio isn't a
scoreboard flag at all.

| Metric | Formula | Null when | Cross-check (Yahoo field) |
|---|---|---|---|
| Current Ratio (computed) | Current Assets / Current Liabilities | Current Assets or Current Liabilities missing/zero | `currentRatio` |
| Quick Ratio (Acid-Test) | (Current Assets − Inventory) / Current Liabilities | Current Assets or Current Liabilities missing/zero | `quickRatio` |

**Inventory assumption**: Inventory defaults to `0` on `StandardizedFinancials`
when the balance sheet doesn't report it (banks, most software/services
companies genuinely carry none), so its absence never blocks the Quick
Ratio calculation the way a missing Current Liabilities figure does.

## 5. Leverage

Computed inline in `validate_leverage()`, plus `interest_coverage()`.

| Metric | Formula | Null when | Cross-check (Yahoo field) |
|---|---|---|---|
| Debt-to-Equity | Total Debt / Stockholders Equity | Stockholders Equity missing/zero | `debtToEquity` (scale-normalized — see below) |
| Total Debt (source check) | Balance-sheet-only value vs. Yahoo's info-dict value | Either source missing | `totalDebt` |
| Interest Coverage | EBIT / \|Interest Expense\| | EBIT or Interest Expense missing/zero | *(none — Yahoo doesn't report this)* |

**Debt-to-Equity is statement-computed, not a Yahoo passthrough** — this is
the one metric in the app whose *canonical* source (used everywhere: Master
Matrix, Scorecard, Watchlist screening, Company Quality) was deliberately
switched away from Yahoo. Yahoo's `debtToEquity` field has been observed at
inconsistent scales (a ratio like `0.78` on some tickers/versions, a
percent-like number such as `78.0` on others) — `normalize_debt_to_equity()`
in `financial_standardization.py` guesses the scale via a threshold
(`abs(raw) >= 5` ⇒ treat as percent-scaled), but this is a heuristic, not a
guarantee. The statement-computed ratio avoids the ambiguity entirely and is
used as the primary source; Yahoo's (normalized) figure is only a fallback
for the rare case Stockholders Equity itself isn't reported.

**Total Debt has two independent Yahoo sources** that can disagree — the
balance sheet's `Total Debt` line item and the info-dict's `totalDebt`
summary field. `financial_standardization.py` prefers the balance-sheet
value (more detailed, multi-period source) and falls through to the info
field only when the statement doesn't report it; that resolved value is
`standardized.total_debt`, used everywhere in this module. The Leverage
validation report additionally exposes the raw, *un-resolved* balance-sheet-
only figure (`standardized.total_debt_from_statement`) purely so a
disagreement between the two sources is visible instead of silently hidden
behind whichever one happened to be picked.

**Sector-adjusted threshold**: the Scorecard's Debt-to-Equity pass/fail
benchmark (not the ratio calculation itself) is looser for Financial
Services companies — see §10.

## 6. Valuation

Computed by `pe_ratio_computed()`, `price_to_book_computed()`,
`enterprise_value()`, `ebitda()`, `ev_to_ebitda_computed()`, `fcf_yield_pct()`.
Cross-checked by `validate_valuation()`.

| Metric | Formula | Null when |
|---|---|---|
| P/E (computed) | Market Cap / Net Income | Net Income ≤ 0, Market Cap missing |
| Price-to-Book (computed) | Market Cap / Stockholders Equity | Stockholders Equity ≤ 0, Market Cap missing |
| Enterprise Value (EV) | Market Cap + Total Debt − Cash & Equivalents | Market Cap missing |
| EBITDA | EBIT + Depreciation & Amortization | EBIT missing |
| EV / EBITDA | Enterprise Value / EBITDA | EBITDA ≤ 0, or either input missing |
| FCF Yield | Free Cash Flow / Market Cap | Free Cash Flow missing, Market Cap missing |
| PEG Ratio | P/E / (Earnings Growth expressed as a whole number, e.g. 15 for 15%) | P/E, PEG inputs, or Earnings Growth missing/non-positive |

**Negative-earnings handling**: a non-positive P/E or Price-to-Book is not a
meaningful valuation multiple — it doesn't mean "very cheap," it means the
company lost money or has negative book equity. Both the computed values
above *and* the canonical `standardized.pe_ratio` / `standardized.price_to_book`
fields (used by the Master Matrix/Scorecard) resolve to `None` rather than
passing through a misleading negative number. The same reasoning extends to
EV/EBITDA: a non-positive EBITDA makes the ratio meaningless, so it's `None`
rather than a nonsensical (often negative) multiple — EBITDA *itself* is
still reported as a real, informative negative number for a genuinely
distressed company; only the *ratio* is suppressed.

**PEG Ratio** is computed once, upstream, in `financial_standardization.py`
(`_compute_peg_ratio`) rather than in this module, since it's needed as a
canonical Scorecard/Matrix field: `P/E / (Earnings Growth × 100)`. Yahoo's
own `pegRatio` field is commonly `None` or based on forward-looking,
multi-year analyst growth estimates rather than trailing growth — a
disagreement between the two in the Valuation validation report usually
reflects that definitional difference, not a formula error.

**Canonical vs. cross-check source**: P/E and Price-to-Book stay
Yahoo-sourced as the values the Master Matrix, Scorecard, and Company
Quality Classification actually use (`standardized.pe_ratio`,
`standardized.price_to_book`) — unlike Debt-to-Equity, no systematic
scale/unit problem was found for these two, so switching the canonical
source wasn't warranted. The computed versions in this section exist purely
to verify Yahoo's numbers in `validate_valuation()`. EV/EBITDA has no prior
canonical value anywhere in the app — it's a new metric introduced
alongside its cross-check.

## 7. Distress: Altman Z-Score

`altman_z_score()` implements the **original 1968 Altman Z-Score for public
manufacturing companies** (Edward I. Altman, *"Financial Ratios,
Discriminant Analysis and the Prediction of Corporate Bankruptcy,"* Journal
of Finance, 1968):

```
Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 0.999·X5

X1 = (Current Assets − Current Liabilities) / Total Assets   (working capital / total assets)
X2 = Retained Earnings / Total Assets
X3 = EBIT / Total Assets
X4 = Market Cap / Total Liabilities (Net Minority Interest)
X5 = Total Revenue / Total Assets
```

Zone thresholds (`config.RISK.altman_safe_zone` = 2.99,
`config.RISK.altman_grey_zone` = 1.81):

| Z-Score | Verdict |
|---|---|
| Z > 2.99 | 🟢 Safe Zone |
| 1.81 ≤ Z ≤ 2.99 | 🟡 Grey Zone |
| Z < 1.81 | 🔴 Distress Zone (High Risk) |

**X3 deliberately uses raw EBIT, not NOPAT** — this is a departure from
ROIC's methodology (§3) and is intentional: the 1968 formula's coefficients
were empirically fit to raw, pre-tax EBIT/Total Assets. Tax-adjusting it
here would silently deviate from the published, validated model instead of
implementing it faithfully.

**Why Z is often `None`**: the model requires seven inputs (Total Assets,
Current Assets, Current Liabilities, EBIT, Total Revenue, Market Cap, Total
Liabilities). Companies without a classified balance sheet — banks and
other financials, notably — routinely lack Current Assets/Current
Liabilities and sometimes EBIT, since those concepts don't map cleanly onto
a bank's balance sheet structure. This is an expected, disclosed limitation
of applying a manufacturing-sector model universally, not a bug: `Z = None`
for such companies rather than a fabricated or forced value.

## 8. Valuation model: WACC and DCF

### 8.1 WACC (`wacc()`)

Weighted Average Cost of Capital via the **Capital Asset Pricing Model**
(CAPM) for the cost of equity, plus after-tax cost of debt:

```
Cost of Equity = risk_free_rate + β × (market_return − risk_free_rate)
Cost of Debt   = (|Interest Expense| / Total Debt) × (1 − tax_rate)

WACC = [E / (E+D)] × Cost of Equity + [D / (E+D)] × Cost of Debt
```

where `E` = Market Cap, `D` = Total Debt. Assumptions
(`config.RISK.risk_free_rate` = 4%, `config.DCF.market_return` = 10%,
`config.DCF.tax_rate` = 21%) are shared with other parts of the app (Sharpe/
Sortino also use `risk_free_rate`) so the app's implied cost-of-capital
assumptions stay internally consistent.

**Declared assumption, not a fabricated data value**: when `standardized.beta`
is `None`, `wacc()` assumes market beta (`β = 1.0`) rather than leaving the
whole WACC calculation undefined. This is a modeling convention (the market
portfolio's beta is 1 by definition), not a guess at the company's real
beta — it's documented here so it's clear the DCF still runs on a stated
assumption rather than silently failing.

### 8.2 Two-stage DCF (`intrinsic_price()`, `run_dcf()`)

```
Stage 1 (explicit projection, config.DCF.projection_years = 5 years):
    FCF_t = FCF_0 × (1 + growth_rate)^t                for t = 1..5
    PV(FCF_t) = FCF_t / (1 + discount_rate)^(t+1)

Stage 2 (Gordon Growth terminal value):
    Terminal Value = FCF_5 × (1 + terminal_growth_rate) / (discount_rate − terminal_growth_rate)
    PV(Terminal)    = Terminal Value / (1 + discount_rate)^5

Intrinsic Value per Share = [ Σ PV(FCF_t) + PV(Terminal) ] / Shares Outstanding
Margin of Safety           = (Intrinsic Value − Market Price) / Intrinsic Value
```

`growth_rate` and `discount_rate` (WACC) are both parameters — `growth_rate`
comes from the user's sidebar slider, `discount_rate` from `wacc()`. This is
why `intrinsic_price()` is exposed as a standalone method: the sensitivity
analysis grid in `finance.py` re-runs it across a range of growth/WACC
combinations without duplicating the model.

`config.DCF.terminal_growth_rate` (2%) is a standard long-run-GDP-growth
proxy for the terminal growth assumption, applied uniformly regardless of
company or sector.

**Why the DCF sometimes doesn't run** (`run_dcf()` returns `ok=False`):
- Missing Market Cap — cannot weight WACC's cost-of-equity/debt split.
- Free Cash Flow is `None`, non-positive, or shares outstanding is
  non-positive — a negative-FCF company cannot be meaningfully projected
  forward with a simple constant-growth model; this is a genuine modeling
  limitation of the 2-stage DCF approach, not a data problem to work around.
- `ZeroDivisionError` (handled in `finance.py`, not here) if `discount_rate`
  exactly equals `terminal_growth_rate` — the Gordon Growth denominator is
  undefined at that point by construction.

No fabricated defaults are substituted in any of these cases — the DCF
section shows a clear reason instead of a garbage number.

## 9. Cross-validation & the Financial Metrics Validation Report

Every ratio family above has a `validate_*()` counterpart
(`validate_profitability()`, `validate_liquidity()`, `validate_leverage()`,
`validate_valuation()`) that independently computes the metric from raw
statement data and compares it against Yahoo's own separately-reported
figure for the same concept. This is the practical substitute, in an
environment with no live access to real annual reports or a third-party
data provider, for reconciling a calculation against an external source.

**Agreement tolerance** (`_AGREEMENT_TOLERANCE` = 15%, relative): two
figures "agree" when they're within 15% of each other. This is deliberately
loose — Yahoo's reported ratios are often trailing-twelve-month while this
module's figures use the most recently reported annual period, and that
timing/basis difference alone can exceed a tighter tolerance without either
number being wrong. A ✅/⚠️/⚪ status on any check should be read as
"consistent with Yahoo, within normal reporting variance" — not "correct" —
and ⚠️ as "worth a second look," not "proven wrong."

`⚪` ("not evaluable") appears whenever either side of a comparison is
`None` — either Yahoo has no equivalent field (ROIC, Interest Coverage, FCF
Yield have no Yahoo counterpart at all) or the underlying statement line
item isn't reported for this company (Gross Margin for a bank).

### 9.1 Consolidated overview (`validate_all_metrics()`)

`MetricsValidationSummary` aggregates all four category reports into one
overview (the "Financial Metrics Validation Report" in the UI), and adds two
checks that don't belong to any single category:

**Outlier detection** (`_outlier_note()`, bounds in `config.OUTLIER_BOUNDS`):
flags a computed value whose *magnitude* exceeds a configured sanity bound,
independent of whether it agrees with Yahoo — a value can cross-check
cleanly and still be an outlier (both sources could share the same
underlying data error), or disagree with Yahoo while being perfectly
plausible. Every bound is a disclosed judgment call (no live external
industry-benchmark database exists in this environment to calibrate
against), deliberately set above every legitimate value already observed
against real Yahoo data during this module's development — see the
docstring on `config.OutlierBoundsConfig` for the exact figures and the
reasoning behind each. Not every metric has a bound: Interest Coverage and
Total Debt are intentionally excluded (a very high Interest Coverage is
always good news, not a red flag; Total Debt is a raw dollar figure, not a
bounded ratio).

**Incomplete-calculation flags** (`data_fallbacks` + effective-tax-rate
fallback): `StandardizedFinancials.data_fallbacks` is a list of plain-English
notes built during standardization whenever a value fell back from its
preferred source to a secondary one (Total Debt, Interest Expense, Current
Price, or Debt-to-Equity — see `financial_standardization.py`). This module
adds one more: whether `effective_tax_rate()` had to use the assumed
statutory rate instead of this company's own reported rate. Together these
answer "was this number built on a real, reported figure, or an estimate?"

## 10. Strategic Investment Scorecard

`_build_checks()` evaluates 9 metrics into `MetricCheck` objects — the
single source for both the Master Matrix (all 9, informational) and the
Scorecard (8 of the 9; FCF Yield is Matrix-only, `in_scorecard=False`).
Thresholds live in `config.SCORECARD`:

| Metric | Benchmark | Weight |
|---|---|---|
| Net Margin | > 10% | 1.5 |
| Debt-to-Equity | < 2.5 (< 4.0 for Financial Services — see below) | 1.5 |
| ROIC | > 10% | 1.5 |
| Interest Coverage | > 3.0x | 1.0 |
| Current Ratio | > 1.0 | 1.0 |
| P/E Ratio | 10–45 | 0.75 |
| PEG Ratio | 0 < PEG ≤ 2.5 | 0.75 |
| Beta | < 1.5 | 0.5 |
| *FCF Yield (Matrix only, not scored)* | > 4% | *n/a* |

**Sector adjustment**: `SCORECARD.max_debt_to_equity_for(sector)` returns
4.0 instead of the global 2.5 when `standardized.sector` is `"Financial
Services"` or `"Financials"` (Yahoo has used both spellings across
versions). Rationale: for a bank, deposits and borrowings *are* the
business model, not a discretionary leverage choice — the corporate-leverage
threshold that flags a non-financial company as risky isn't a meaningful
signal for one. This is the *only* sector-specific threshold in the app; a
full per-sector benchmark table across all ~11 GICS sectors was
deliberately not built, since there's no live external industry-benchmark
source to validate the other ten against, and Debt-to-Equity was the one
threshold with a concrete, demonstrated cross-sector problem.

**Weighted, not a simple average**: `FundamentalMetrics.score_pct` is a
weighted percentage over *evaluable* checks only (`passed is not None`) —
`Σ(weight × passed) / Σ(weight)`, not `green_flags / total_checks`. Core
financial-health signals (profitability, leverage, capital efficiency)
count for more than secondary considerations (valuation multiples,
volatility).

**Missing data is excluded, not penalized**: a check with no computable
value (`passed is None` — e.g. ROIC for a bank with no reported EBIT) is
excluded from *both* the numerator and denominator entirely, rather than
counted as a failure. This means `FundamentalMetrics.total_checks` can be
less than 8 and varies per ticker — a company in a sector with different
reporting norms isn't silently scored down for data that was never going to
be reported. `alignment_verdict` maps the resulting `score_pct` to
`"high"` (≥ 75%), `"moderate"` (≥ 40%), or `"low"` — thresholds in
`config.SCORECARD.high_alignment_pct` / `moderate_alignment_pct`.

## 11. Company Quality Classification

`classify_company_quality()` is a complementary, differently-framed view
from the Scorecard above — not a replacement. Where the Scorecard is a
flat pass/fail checklist, this blends five weighted **factors** into one
0–100 score and category. Bands and weights live in `config.QUALITY`.

| Factor | Weight | Inputs | Band (0 pts → 100 pts) |
|---|---|---|---|
| Profitability | 25% | Net Margin, Gross Margin, Operating Margin, ROA | (0%, 25%) / (0%, 60%) / (0%, 30%) / (0%, 15%) |
| Financial Stability | 25% | Debt-to-Equity, Current Ratio, Interest Coverage, Altman Z | (2× sector D/E threshold, 0) / (0, 3.0) / (0, 10x) / (0, 3.5) |
| Growth | 15% | Earnings/Revenue Growth (Yahoo-reported) | (−10%, 25%) |
| Valuation | 15% | P/E, PEG, Price-to-Book, EV/EBITDA | "ideal" center points — see below |
| Capital Efficiency | 20% | ROIC, ROE, Asset Turnover | (0%, 20%) / (0%, 30%) / (0x, 1.5x) |

Each metric scores 0–100 against its band (`_linear_score()`, clamped;
Debt-to-Equity's band is inverted since lower is better), a factor's score
is the average of its *evaluable* metrics, and the overall score is a
weighted average of *evaluable* factors — same "exclude, don't penalize"
principle as the Scorecard, applied one level up: a factor with zero
computable metrics (e.g. Growth with no Yahoo growth figure) is excluded
from the overall score entirely, and the remaining factors' weights are
implicitly renormalized.

**Valuation intentionally does not reward cheapness.** Standard
quality-investing methodology (e.g. the MSCI Quality Index) excludes
valuation from a quality score entirely, since excellent businesses often
justly trade at a premium — "expensive" is not the same as "bad," and
"cheap" is not the same as "good." Since documenting/classifying by
valuation was an explicit requirement, each valuation metric instead scores
100 at a "reasonably priced" center point and falls off proportionally to
*relative distance* from it in either direction (`_ideal_score()`):

```
score = 100 × (1 − |value − ideal| / |ideal|),  clamped to [0, 100]
```

Center points: P/E ≈ 20, PEG ≈ 1.0 (the textbook Peter Lynch "fairly
priced" PEG), Price-to-Book ≈ 3, EV/EBITDA ≈ 12. Both a P/E of 3 and a P/E
of 40 score low under this model — the first for being suspiciously cheap
(often a distress signal), the second for pricing in aggressive future
growth that may not materialize — while a P/E near 20 scores near 100.

**Sector awareness inherited from the Scorecard**: Financial Stability's
Debt-to-Equity sub-score reuses `SCORECARD.max_debt_to_equity_for(sector)`
as its scoring anchor (0 points at 2× that threshold), so Financials get a
proportionally fair band without a second, separately-maintained table.
Capital Efficiency's Asset Turnover band is a single global approximation,
by contrast — genuinely sector-dependent (asset-light software vs.
asset-heavy banks/utilities) but not sector-adjusted, since a bank's low
turnover reflects its balance-sheet structure rather than poor capital
discipline. This is disclosed in the UI wherever the factor is shown.

Category thresholds (`config.QUALITY`): Elite Quality (≥ 85), High Quality
(≥ 70), Average Quality (≥ 50), Below Average (≥ 30), Weak Quality (below).

## 12. Developer guide: extending this module

**Adding a new pass/fail Scorecard/Matrix metric**: add one `MetricCheck`
entry inside `_build_checks()`. It will automatically appear in the Master
Matrix (`in_matrix=True` by default) and count toward the weighted
Scorecard score (`in_scorecard=True` by default) — no changes needed
anywhere else, including `finance.py`. Give it a `weight` if it should count
more/less than the default 1.0 (add the key to `config.SCORECARD.weights`).

**Adding a new cross-checked ratio** (independently computed + compared
against Yahoo): add a calculation method near the relevant section (§3–§6),
then a `ProfitabilityCheck` entry inside the matching `validate_*()` method
— despite the name, `ProfitabilityCheck` is the shared type for every
category's cross-check rows (see its docstring for the `suffix` field,
which controls whether it renders as a percentage or a plain ratio/multiple).
It will automatically be picked up by `validate_all_metrics()` for the
consolidated report and outlier detection — add a bound to
`config.OutlierBoundsConfig` and `_OUTLIER_BOUNDS_TABLE` if one makes sense
for it (not every metric needs one; see §9.1).

**Adding a new Company Quality factor or metric**: add a `QualityFactorMetric`
inside the relevant `QualityFactor` in `classify_company_quality()`, with a
band or ideal-point in `config.QualityConfig`. Rebalance the other factor
weights if adding a new *factor* (they're expected to sum to 1.0, though the
weighted-average math in `CompanyQuality.overall_score` tolerates weights
that don't, since it divides by the evaluable factors' total weight rather
than assuming it's always 1.0).

**Every new threshold, band, or weight belongs in `config.py`**, not as a
magic number inline — this is a hard convention across the whole codebase,
not specific to this module (see `config.py`'s own module docstring).

**Testing a new metric against real data**: this module has no unit test
suite; verification throughout this project's development has been direct
comparison against real Yahoo Finance data for at least two structurally
different tickers — a normal company (e.g. AAPL) and a bank (e.g. JPM, which
lacks a classified balance sheet and therefore exercises almost every
`None`-handling path in this document). Banks are the single most useful
edge case in this codebase: if a new metric handles JPM gracefully, it
almost certainly handles every other missing-data scenario too.

## 13. References

- Altman, Edward I. *"Financial Ratios, Discriminant Analysis and the
  Prediction of Corporate Bankruptcy."* The Journal of Finance, Vol. 23,
  No. 4 (1968) — the original Z-Score model implemented in §7.
- Sharpe, William F. *"Capital Asset Prices: A Theory of Market Equilibrium
  under Conditions of Risk."* The Journal of Finance, Vol. 19, No. 3
  (1964) — the Capital Asset Pricing Model underlying the cost-of-equity
  term in §8.1.
- NOPAT / ROIC as EBIT × (1 − effective tax rate) is standard institutional
  equity-research and corporate-finance practice (see, e.g., any graduate
  corporate finance text's treatment of "invested capital" returns) rather
  than a single citable paper.
- MSCI Quality Index methodology (publicly documented by MSCI) is the
  reference point for §11's decision to exclude cheapness-based scoring
  from the Valuation quality factor.
