"""Plain-language explanations for every metric AND chart the app
displays — the single source of truth behind the `help=` tooltip on each
st.metric and the explanatory caption above each chart.

WHY CHARTS GET A CAPTION RATHER THAN A TOOLTIP: st.plotly_chart has no
`help=` parameter (checked against Streamlit 1.58 — its signature is
figure_or_data / use_container_width / width / height / theme / key /
on_select / selection_mode / config), so a hover tooltip simply isn't
available for a chart. A caption directly above it is the honest
equivalent, and it was already this app's own convention for the handful
of charts that had any explanation at all. Plotly's native hover readout
still covers "what is this data point"; these captions cover the
different question of "what is this chart telling me".

WHY A CENTRAL GLOSSARY RATHER THAN INLINE STRINGS: several metrics are
shown in more than one panel (Max Drawdown appears in the Risk Dashboard,
the Strategy Backtester and the Portfolio Backtester; Sharpe in two).
When the text lived inline, those drifted — Max Drawdown was explained in
one place and left bare in another, which is exactly the "consistent in
tone and format" problem this module exists to fix. Looking every tooltip
up by key means the same metric is explained the same way everywhere, and
a test can assert that no key is referenced that doesn't exist here.

HOUSE STYLE, enforced by tests in tests/test_metric_help.py:
  1. What it is, in plain language — no jargon that itself needs a
     lookup, and no formula unless the formula IS the intuition.
  2. How to read it — which direction is good, and the threshold that
     matters. A bare "1.4" tells a non-finance reader nothing without it.
Short: two sentences, comfortably under a paragraph.

Thresholds are interpolated from config rather than typed as literals, so
a tooltip can never quietly disagree with the number the app actually
uses to classify things (e.g. RISK.altman_grey_zone drives both the
Distress-zone verdict and the sentence describing it).
"""
from typing import Dict

from config import RISK, TECHNICAL

# --- Risk-adjusted return ---------------------------------------------------
_RETURN_QUALITY = {
    "sharpe": (
        "Return earned per unit of total risk taken — it asks whether the gains justified how "
        "bumpy the ride was. Higher is better; above 1 is generally considered solid, and below 0 "
        "means it did worse than simply holding cash."
    ),
    "sortino": (
        "Like the Sharpe ratio, but it only counts DOWNSIDE moves as risk — violent moves upward "
        "aren't penalised. Higher is better, and a large gap above the Sharpe ratio tells you most "
        "of the volatility was in your favour."
    ),
    "calmar": (
        "Annual return divided by the worst peak-to-trough fall — return per unit of pain endured. "
        "Higher is better; it specifically rewards growth that didn't come with a deep collapse "
        "along the way."
    ),
}

# --- Loss and volatility ----------------------------------------------------
_LOSS_RISK = {
    "volatility_rolling": (
        "How much the price swings around, annualised and measured over a moving window. Higher "
        "means a bumpier ride — this is the raw risk input the Sharpe and Sortino ratios divide by."
    ),
    "volatility_full_range": (
        "The same swing measure as rolling volatility, but computed once across the whole selected "
        "date range instead of a moving window. Useful as a steadier baseline when the rolling "
        "figure is jumping around."
    ),
    "var_historical": (
        f"The loss you would NOT expect to exceed on a typical day, at the confidence level shown, "
        f"based on how this stock actually moved. Shown as a negative percentage — closer to zero "
        f"is safer, and it needs at least {RISK.var_min_observations} observations to be computed."
    ),
    "var_parametric": (
        "The same one-day loss estimate as Historical VaR, but assuming returns follow a normal "
        "bell curve instead of using the moves that actually happened. A big gap between the two "
        "means this stock's real behaviour departs from that tidy assumption."
    ),
    "cvar": (
        "The AVERAGE loss on the bad days — specifically the days worse than the VaR threshold. It "
        "answers 'when it does go wrong, how wrong?', and is always at least as negative as VaR."
    ),
    "max_drawdown": (
        "The largest peak-to-trough fall over the period — the worst stretch somebody who bought at "
        "exactly the wrong moment had to sit through. Shown as a negative percentage; closer to "
        "zero is better."
    ),
    "peak_to_trough": (
        "The two dates bracketing that worst decline: when the price last peaked, and when it "
        "finally bottomed out."
    ),
    "recovery_period": (
        "How long it took to climb back to the previous peak after the worst decline. A long "
        "recovery matters as much as a deep fall — capital was tied up the whole time."
    ),
}

# --- Solvency and fundamentals ---------------------------------------------
_FUNDAMENTALS = {
    "altman_z": (
        f"A bankruptcy-risk score built from five balance-sheet and earnings ratios. Above "
        f"{RISK.altman_safe_zone} is the safe zone and below {RISK.altman_grey_zone} is the "
        f"distress zone, with a grey area in between where the signal is inconclusive."
    ),
    "intrinsic_value": (
        "What the discounted cash flow model estimates one share is fundamentally worth, based on "
        "projected future cash flows valued in today's money. It is only ever as good as its "
        "growth and discount-rate assumptions."
    ),
    "margin_of_safety": (
        "How far the current price sits below that estimated intrinsic value. Positive means the "
        "model thinks it's cheap; negative means the market is paying more than the model can "
        "justify."
    ),
    "market_price": "The latest traded share price, shown for comparison against the model's estimate.",
    "dividend_yield": (
        "The annual dividend as a percentage of the current share price — the income return if both "
        "the payout and the price hold. Higher isn't automatically better: an unusually high yield "
        "is often a falling share price rather than a generous company."
    ),
    "annual_dividend_share": "The total dividend paid per share over a year, in currency rather than percentage terms.",
    "lost_income_share": "The dividend income per share that would disappear under this scenario's assumed cut.",
    "base_intrinsic_value": "The model's estimated worth per share before this scenario's shock is applied.",
    "shocked_intrinsic_value": "The model's estimated worth per share after applying this scenario's assumptions.",
}

# --- Technicals -------------------------------------------------------------
_TECHNICALS = {
    "rsi": (
        f"Momentum on a 0–100 scale. Above {TECHNICAL.rsi_overbought:.0f} is conventionally called "
        f"'overbought' and below {TECHNICAL.rsi_oversold:.0f} 'oversold' — though a genuinely "
        f"strong trend can sit at an extreme reading for a long time without reversing."
    ),
    "atr": (
        "The average distance the price covers in a day, gaps included — a plain measure of how far "
        "it typically travels. Higher means wider swings, which is why stop distances are sized "
        "from it rather than from a fixed percentage."
    ),
    "stop_loss": (
        f"A suggested exit price set {TECHNICAL.atr_stop_multiplier:.0f}× the average daily range "
        f"below the current price, far enough out that ordinary day-to-day noise shouldn't trigger "
        f"it. A mechanical guide, not advice."
    ),
    "risk_per_share": (
        "The gap between the current price and that suggested stop — what a single share would lose "
        "if the stop were hit. Multiply by position size to see the trade's total risk."
    ),
    "price_z_score": (
        "How far today's price sits from its own recent average, counted in standard deviations. "
        "Around 0 is unremarkable; beyond ±2 is statistically unusual and often read as stretched."
    ),
    "hurst": (
        f"Whether price moves tend to keep going or snap back. Above {RISK.hurst_trending_above} "
        f"suggests a persistent trend, below {RISK.hurst_mean_reverting_below} suggests mean "
        f"reversion, and near 0.5 means it behaves indistinguishably from a random walk."
    ),
}

# --- Benchmark-relative performance ----------------------------------------
_RELATIVE = {
    "alpha_generated": (
        "Performance above the benchmark over the same window. Positive means this stock outpaced "
        "the market; negative means it lagged it."
    ),
    "beta_systematic": (
        "The slice of the excess return explained purely by riding the market at this stock's "
        "sensitivity to it — nothing company-specific. A large value means the return mostly came "
        "from market exposure you could have bought more cheaply with an index fund."
    ),
    "alpha_selection": (
        "What's left of the excess return once market exposure is accounted for — the part actually "
        "attributable to this specific stock. Positive means it beat what its market exposure alone "
        "would predict."
    ),
    "period_return": "Total price change across the selected date range, dividends excluded.",
    "excess_return_total": (
        "How much this stock returned above the benchmark in total, before splitting that into the "
        "market-driven and stock-specific parts."
    ),
}

# --- Strategy / portfolio results ------------------------------------------
_STRATEGY = {
    "win_rate": (
        "The share of periods that finished positive. On its own it says surprisingly little — a "
        "high win rate built on small wins and rare large losses can still lose money, so read it "
        "alongside the payoff ratio."
    ),
    "payoff_ratio": (
        "The average size of an up move divided by the average size of a down move. Above 1 means "
        "winners are bigger than losers, which is what lets even a sub-50% win rate come out ahead."
    ),
    "kelly_half": (
        f"A position-size suggestion derived from the win rate and payoff ratio, then divided by "
        f"{RISK.kelly_half_factor:.0f} to stay deliberately conservative. A rough heuristic and an "
        f"upper bound to think about — not a recommendation."
    ),
    "strategy_return_gross": "What the strategy returned before any trading costs or slippage are deducted.",
    "strategy_return_net": "What the strategy returned after deducting modelled trading costs and slippage — the figure that actually matters.",
    "buy_hold_baseline": (
        "What simply buying and holding over the same period would have returned. The strategy has "
        "to beat this to have earned its complexity."
    ),
    "trades": "How many round-trip trades the strategy took over the period.",
    "total_cost_paid": "Modelled commission and slippage paid across every trade, in return terms.",
    "oos_return": (
        "Return measured only on data the strategy was never fitted to — the honest test. "
        "In-sample results are almost always flattering by comparison."
    ),
    "windows": "How many separate train-then-test splits the walk-forward run evaluated.",
    "portfolio_return": "What the whole weighted, periodically rebalanced basket returned over the period.",
    "static_weight_reference": "What the same basket would have returned with weights set once and never rebalanced.",
    "rebalances": "How many times the basket was traded back to its target weights over the period.",
}

# --- Portfolio construction -------------------------------------------------
_PORTFOLIO = {
    "portfolio_volatility": (
        "The whole basket's annualised swing, accounting for the fact that holdings don't all move "
        "together. This is normally lower than the weighted average below it — that gap is the "
        "benefit of diversifying."
    ),
    "weighted_avg_volatility": (
        "What the basket's volatility would be if every holding moved in perfect lockstep — the "
        "no-diversification reference point, not something achievable."
    ),
    "diversification_benefit": (
        "How much volatility is avoided purely because the holdings don't move in lockstep. Larger "
        "is better; near zero means the positions are effectively duplicates of each other."
    ),
}

# --- Macro ------------------------------------------------------------------
_MACRO = {
    "vix": (
        "The market's expected volatility over the coming 30 days, widely known as the 'fear "
        "index'. Low readings suggest calm; a sustained spike signals broad market stress rather "
        "than anything specific to this stock."
    ),
    "treasury_10y": (
        "The yield on 10-year US government debt — the benchmark 'risk-free' return. When it rises "
        "it lifts the bar every other investment has to clear, which typically weighs on stock "
        "valuations."
    ),
}

# --- Data quality / model diagnostics --------------------------------------
_DIAGNOSTICS = {
    "metrics_evaluated": "How many of this company's metrics could actually be computed and cross-checked from the data available.",
    "yahoo_disagreements": (
        "Metrics where Quantix's own calculation differs materially from the figure Yahoo Finance "
        "reports. Not automatically an error — the two often simply define the metric differently."
    ),
    "extreme_outliers": (
        "Metrics whose magnitude falls outside the sanity bounds in config — flagged as worth a "
        "second look rather than assumed to be wrong."
    ),
    "incomplete_calculations": "Metrics that couldn't be computed at all because a required field was missing from the source data.",
    "green_flags": "How many of the Blueprint's institutional-quality checks this company passes.",
    "warning_signs": "How many operational red flags the Blueprint's checks raised for this company.",
    "model_accuracy": (
        "The share of test-set predictions the model got right. Judge it against the naive baseline "
        "beside it — accuracy alone can look impressive by always guessing the commoner outcome."
    ),
    "roc_auc": (
        "How well the model separates the two outcomes, from 0.5 (no better than a coin flip) to "
        "1.0 (perfect). Above roughly 0.6 is a weak but real signal; treat anything near 0.5 as noise."
    ),
}

GLOSSARY: Dict[str, str] = {
    "news_sentiment": (
        "How positive or negative recent coverage reads, averaged across headlines that "
        "actually mention this company. It measures the tone of the narrative, not whether "
        "the market has already priced it in — a strongly positive read on a stock that has "
        "already run is not a buy signal."
    ),

    **_RETURN_QUALITY, **_LOSS_RISK, **_FUNDAMENTALS, **_TECHNICALS,
    **_RELATIVE, **_STRATEGY, **_PORTFOLIO, **_MACRO, **_DIAGNOSTICS,
}


# --- Charts -----------------------------------------------------------------
# Same house style as the metric glossary above: what it shows, then how to
# read it. Rendered as a caption directly above each chart (see the module
# docstring for why a caption rather than a tooltip).
CHART_HELP: Dict[str, str] = {
    "portfolio_performance": (
        "Your portfolio's value day by day against the benchmark, rebased to the same starting "
        "value so the gap between the two lines is the real comparison: above it you are "
        "beating the index. Each holding joins on its own purchase date, and a step upward "
        "means money arriving rather than a gain."
    ),

    "price_technicals": (
        "Price with whichever overlays you enabled above, and any oscillator panels stacked "
        "beneath it. Crossover markers show where a signal actually fired — drag to zoom into a "
        "stretch, and double-click to reset."
    ),
    "relative_strength": (
        "This stock's cumulative return against the benchmark's over the same window. The gap "
        "between the lines IS the outperformance: above means it beat the market, below means it "
        "lagged."
    ),
    "risk_gauge": (
        f"The Composite Risk Score on a 0–100 scale, blending every risk metric below into one "
        f"number. Higher is safer, and the coloured bands mark the app's own risk bands."
    ),
    "var_distribution": (
        "How often each size of daily return actually occurred, with the VaR and CVaR cut-offs "
        "marked. The left tail is what those two measure — VaR is where it starts, CVaR is the "
        "average of everything beyond it."
    ),
    "drawdown_underwater": (
        "How far below its running peak the price sat on each day — it touches zero at every new "
        "high and dips in between. The deepest point is the Maximum Drawdown quoted above."
    ),
    "strategy_equity": (
        "What the strategy did to your capital over time, against simply buying and holding. Watch "
        "the gap between the lines, and whether the strategy actually dodged the drops rather than "
        "just finishing higher."
    ),
    "walk_forward_equity": (
        "In-sample performance (data the strategy was fitted on) against out-of-sample (data it "
        "never saw). A large drop from the first to the second is the classic signature of a "
        "strategy fitted to noise."
    ),
    "portfolio_equity": (
        "The periodically rebalanced basket against the same weights left untouched. The gap "
        "between them is what rebalancing actually added — or cost."
    ),
    "monte_carlo_paths": (
        "Many simulated future price paths from today's price, each one a different random draw. "
        "Read the SPREAD rather than any single line: its width is the range of outcomes the model "
        "considers plausible."
    ),
    "seasonality_surface": (
        "Monthly returns laid out year by year, so a month that repeatedly behaves the same way "
        "shows up as a ridge or valley running across years. Drag to rotate — a consistent ridge "
        "is a tendency, not a guarantee."
    ),
    "peer_radar": (
        "This stock's standing against its peers on each axis, so a larger enclosed shape means "
        "broadly stronger. Any axis pulled in toward the centre is where it trails the peer group."
    ),
    "correlation_matrix": (
        "How closely each pair of holdings moves together, from -1 (opposite) through 0 (unrelated) "
        "to 1 (identical). A grid full of high values means the basket is far less diversified than "
        "the number of names in it suggests."
    ),
    "efficient_frontier": (
        "Each point is a possible mix of these holdings, placed by the risk it carries against the "
        "return it earned. The upper-left edge is the efficient frontier — the most return "
        "available at each level of risk."
    ),
}


def help_for(key: str) -> str:
    """The tooltip text for a metric key.

    Raises KeyError on an unknown key rather than returning a placeholder
    or empty string: a typo here would otherwise render as a metric with
    no tooltip at all — silently reintroducing the exact gap this module
    exists to close. tests/test_metric_help.py additionally cross-checks
    every key finance.py actually references against this dict, so a typo
    fails the suite rather than waiting to be noticed in the browser.
    """
    return GLOSSARY[key]


def chart_help(key: str) -> str:
    """The explanatory caption for a chart. Raises KeyError on an unknown
    key for the same reason help_for() does."""
    return CHART_HELP[key]
