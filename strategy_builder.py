"""No-code strategy builder — compose entry/exit trading rules from the
indicators technical_indicators.py already computes, without writing code,
and run them through a generalized backtest engine.

Deliberately reuses every indicator/crossover function from
technical_indicators.py and the Max Drawdown engine from risk_analytics.py
rather than reimplementing any indicator math — this module only defines
how existing signals get COMPOSED into a strategy and how a composed
strategy's Position column is turned into equity-curve metrics.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from risk_analytics import compute_max_drawdown
from technical_indicators import (
    detect_bollinger_breakouts,
    detect_macd_crossovers,
    detect_sma_crossovers,
)

LOGIC_OPTIONS: Tuple[str, ...] = ("AND", "OR")

LEVEL_OPERATORS: Tuple[Tuple[str, str], ...] = (
    ("<", "<"), (">", ">"), ("<=", "<="), (">=", ">="),
)


@dataclass(frozen=True)
class ConditionSpec:
    """Metadata describing one entry in the condition library — enough for
    the UI to render the right controls (a threshold input for "level"
    conditions, an event-kind picker with no threshold for "event" ones)."""
    key: str
    label: str
    kind: str  # "level" (a per-bar comparison) or "event" (a discrete crossing)
    operators: Tuple[Tuple[str, str], ...]  # (value, display_label) pairs
    default_threshold: Optional[float] = None


def condition_library(sma_length: int, rsi_length: int) -> Dict[str, ConditionSpec]:
    """The full catalog of screenable conditions. A function (not a module
    constant) because two labels embed the user's current SMA/RSI period
    sidebar selections, so the dropdown always describes the indicator
    that's actually being evaluated."""
    specs = (
        ConditionSpec("rsi", f"RSI ({rsi_length})", "level", LEVEL_OPERATORS, 30.0),
        ConditionSpec("zscore", "Price Z-Score", "level", LEVEL_OPERATORS, -2.0),
        ConditionSpec("sma_cross", f"Price / SMA {sma_length} Crossover", "event", (
            ("bullish", "Bullish (price crosses above)"),
            ("bearish", "Bearish (price crosses below)"),
        )),
        ConditionSpec("macd_cross", "MACD Crossover", "event", (
            ("bullish", "Bullish (MACD crosses above Signal)"),
            ("bearish", "Bearish (MACD crosses below Signal)"),
        )),
        ConditionSpec("bollinger_breakout", "Bollinger Band Breakout", "event", (
            ("upper", "Breaks above upper band"),
            ("lower", "Breaks below lower band"),
        )),
    )
    return {s.key: s for s in specs}


@dataclass(frozen=True)
class StrategyCondition:
    indicator: str            # key into condition_library()
    operator: str             # comparison op ("level") or event kind ("event")
    threshold: Optional[float] = None  # only used by "level" conditions


@dataclass(frozen=True)
class StrategyRule:
    """A complete entry/exit strategy: two independently AND/OR-composed
    condition sets. Both the built-in preset and any user-built strategy
    from the UI are this same shape — see CLASSIC_MEAN_REVERSION below,
    which expresses the app's original hardcoded Z-Score strategy as an
    ordinary StrategyRule so it runs through the identical evaluator/engine
    as a custom one, rather than keeping two parallel code paths."""
    name: str
    entry_conditions: Tuple[StrategyCondition, ...]
    entry_logic: str  # "AND" or "OR"
    exit_conditions: Tuple[StrategyCondition, ...]
    exit_logic: str


def _events_to_bool_series(df: pd.DataFrame, events: List, kind: str) -> pd.Series:
    """True only on the exact bar(s) a crossover/breakout event of `kind`
    fired — a discrete one-bar-per-crossing signal, not a continuous
    per-day state (matches how detect_*_crossovers() itself works)."""
    matching_dates = {e.date for e in events if e.kind == kind}
    return pd.Series(df.index.isin(matching_dates), index=df.index)


LEVEL_OPS: Dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "<": lambda s, t: s < t,
    ">": lambda s, t: s > t,
    "<=": lambda s, t: s <= t,
    ">=": lambda s, t: s >= t,
}


def evaluate_condition(df: pd.DataFrame, condition: StrategyCondition, sma_length: int, rsi_length: int) -> pd.Series:
    """One condition -> one boolean Series aligned to `df.index`. NaN
    warm-up bars naturally compare False (pandas comparison operators don't
    propagate NaN the way arithmetic does), so a strategy never fires
    during an indicator's warm-up period.

    `df` must already have the relevant indicator columns computed
    (RSI_{rsi_length}, Z_Score, SMA_{sma_length}, MACD_Line/MACD_Signal,
    BB_Upper/BB_Lower) — this function only reads them, it never computes
    indicator math itself.
    """
    key = condition.indicator
    if key == "rsi":
        series = df[f"RSI_{rsi_length}"]
        return LEVEL_OPS[condition.operator](series, condition.threshold)
    if key == "zscore":
        series = df["Z_Score"]
        return LEVEL_OPS[condition.operator](series, condition.threshold)
    if key == "sma_cross":
        return _events_to_bool_series(df, detect_sma_crossovers(df, sma_length), condition.operator)
    if key == "macd_cross":
        return _events_to_bool_series(df, detect_macd_crossovers(df), condition.operator)
    if key == "bollinger_breakout":
        return _events_to_bool_series(df, detect_bollinger_breakouts(df), condition.operator)
    raise ValueError(f"Unknown strategy condition indicator: {key!r}")


def evaluate_condition_set(df: pd.DataFrame, conditions: Tuple[StrategyCondition, ...], logic: str, sma_length: int, rsi_length: int) -> pd.Series:
    """AND/OR-combine every condition in a set into one boolean Series. An
    empty condition set never fires (all False) rather than defaulting to
    always-True, which would silently turn "no exit rule" into "exit every
    bar" — the caller (a strategy with no exit conditions at all) handles
    that case explicitly instead."""
    if not conditions:
        return pd.Series(False, index=df.index)
    combined = evaluate_condition(df, conditions[0], sma_length, rsi_length)
    for c in conditions[1:]:
        next_series = evaluate_condition(df, c, sma_length, rsi_length)
        combined = (combined & next_series) if logic == "AND" else (combined | next_series)
    return combined


@dataclass
class BacktestResult:
    df: pd.DataFrame  # copy of the input with Signal/Position/Strategy_Returns/Cum_Strategy added
    entry_series: pd.Series
    exit_series: pd.Series
    total_strategy_return_pct: float
    total_buy_hold_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: Optional[float]
    trade_count: int


def run_backtest(df: pd.DataFrame, rule: StrategyRule, sma_length: int, rsi_length: int) -> BacktestResult:
    """Turn a StrategyRule into Position/equity-curve metrics — the same
    Signal -> forward-filled Position -> Strategy_Returns -> Cum_Strategy
    pipeline the app's original hardcoded Z-Score strategy used, now driven
    by an arbitrary entry/exit condition set instead of one fixed
    threshold. Long-only throughout (clip(lower=0)), matching this app's
    framing everywhere else (Kelly Criterion, DCF, ATR stop-loss).

    Same-bar precedence: if a bar satisfies both the entry and exit
    condition sets, the exit assignment runs second and wins — inherited
    unchanged from the original hardcoded strategy, where it never mattered
    because Z-Score entry/exit thresholds are mutually exclusive by
    construction. A custom strategy mixing indicator families (e.g. a
    trend-following entry with a mean-reversion exit) can genuinely hit
    this case; finance.py's live preview surfaces it explicitly rather than
    leaving "entry signal count doesn't match trade count" unexplained.
    """
    df = df.copy()
    entry_series = evaluate_condition_set(df, rule.entry_conditions, rule.entry_logic, sma_length, rsi_length)
    exit_series = evaluate_condition_set(df, rule.exit_conditions, rule.exit_logic, sma_length, rsi_length)

    df["Signal"] = 0
    df.loc[entry_series, "Signal"] = 1
    df.loc[exit_series, "Signal"] = -1  # exit wins on a same-bar conflict — see docstring

    df["Position"] = df["Signal"].replace(0, np.nan).ffill().fillna(0)
    df["Position"] = df["Position"].clip(lower=0)

    df["Strategy_Returns"] = df["Position"].shift(1) * df["Returns"]
    df["Cum_Strategy"] = (1 + df["Strategy_Returns"]).cumprod()
    df["Cum_Buy_Hold"] = (1 + df["Returns"]).cumprod()

    total_strategy_return_pct = (df["Cum_Strategy"].iloc[-1] - 1) * 100
    total_buy_hold_return_pct = (df["Cum_Buy_Hold"].iloc[-1] - 1) * 100

    dd_result = compute_max_drawdown(df["Cum_Strategy"])
    max_drawdown_pct = dd_result.max_drawdown * 100 if dd_result is not None else 0.0

    # Win rate: of the days actually held (Position was 1 on the PRIOR bar —
    # the same alignment Strategy_Returns itself uses), what fraction had a
    # positive return? Same "fraction of winning observations" definition
    # the existing Kelly Criterion win-rate uses (finance.py's Execution &
    # Position Sizing section), just scoped to strategy-active days instead
    # of every day in the window.
    active_returns = df["Returns"][df["Position"].shift(1) == 1]
    win_rate_pct = (active_returns > 0).mean() * 100 if len(active_returns) > 0 else None

    trade_count = int((df["Position"].diff() == 1).sum())

    return BacktestResult(
        df=df, entry_series=entry_series, exit_series=exit_series,
        total_strategy_return_pct=total_strategy_return_pct,
        total_buy_hold_return_pct=total_buy_hold_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        win_rate_pct=win_rate_pct, trade_count=trade_count,
    )


def classic_mean_reversion(buy_z_score: float, sell_z_score: float) -> StrategyRule:
    """The app's original, already-shipped strategy — buy when Z-Score
    drops below `buy_z_score`, sell when it rises above `sell_z_score` —
    re-expressed as an ordinary StrategyRule so it runs through the exact
    same evaluator/engine as a custom strategy, instead of a second,
    parallel implementation. See tests/test_strategy_builder.py for the
    exact-match regression check against the original hardcoded logic."""
    return StrategyRule(
        name="Classic Mean-Reversion",
        entry_conditions=(StrategyCondition("zscore", "<", buy_z_score),),
        entry_logic="AND",
        exit_conditions=(StrategyCondition("zscore", ">", sell_z_score),),
        exit_logic="AND",
    )
