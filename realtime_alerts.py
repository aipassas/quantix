"""Real-Time Alert Engine — continuously monitors user-defined rules
against a chosen ticker's price, technicals, and fundamentals, notifying
in-app the moment a condition is newly met.

SCOPE — three decisions made explicitly with the user before this was
built, when Quantix had no accounts, no database and no background
worker. Sign-in has since arrived (auth.py), which changes point 3
below; the other two still stand:

1. Monitoring model: IN-TAB POLLING via st.fragment(run_every=...) in
   finance.py, not a real background worker/process. Monitoring only runs
   while the browser tab stays open — the same limitation risk_alerts.py
   already documents for its own check-on-click alerts, just upgraded
   from "on click" to "every N seconds while the tab is open." A genuine
   background worker (a second always-running process, independent of
   whether anyone has the app open) is a distinct, much larger
   architecture change and was explicitly declined for this pass.

2. Delivery: in-app banner/toast only. Email/push were explicitly
   declined — they need real credentials (an SMTP account, or a push
   provider's API key) this app doesn't have and can't invent.

3. Persistence: rules and trigger history ARE persisted to a local JSON
   file (unlike every other piece of state in this app, which lives only
   in st.session_state and resets each fresh session). Since auth.py
   landed these rules ARE per-user in the literal sense the originating
   task meant: signed in, they live in that user's namespace; signed out,
   they fall back to the shared instance-wide store.

TRIGGER TYPES — each reuses an already-existing, tested calculation
rather than new arithmetic:
  - price_above / price_below: watchlist_panel's live quote, the same
    source the sidebar Watchlist and the symbol header already use.
  - sma_cross_bullish / sma_cross_bearish: technical_indicators.py's own
    SMA-crossover detector, at the app's existing default SMA length.
  - macd_bullish_cross / macd_bearish_cross: technical_indicators.py's
    own MACD-crossover detector, at the app's existing default periods.
  - rsi_overbought / rsi_oversold: technical_indicators.py's own RSI +
    interpret_rsi(), at the app's existing default RSI length and
    TECHNICAL's overbought/oversold thresholds.
  - fundamental: risk_alerts.py's own METRICS/OPERATORS and
    compute_watchlist_snapshots() — the exact same Composite Risk Score /
    Altman Z / VaR / CVaR / Max Drawdown metrics Smart Risk-Aware Alerts
    already uses, evaluated for one rule's ticker instead of across the
    fixed Institutional Watchlist.

EDGE-TRIGGERING: a rule produces a notification (and one persisted
history row) only on the transition from "not currently met" to
"currently met" — tracked in st.session_state, never persisted, since
it's about not repeating a banner every single poll while a condition
stays true, not about remembering state across restarts. On a session's
first poll there is nothing to compare against, so any already-true
condition counts as a transition — the same "tell me what's currently
triggered" behavior risk_alerts.py already has on every Check Alerts
click, just applied automatically on first load instead of a manual one.
"""
import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from config import CHART_DEFAULTS, REALTIME_ALERTS, TECHNICAL
from data_loader import load_ticker_bundle
from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_event, log_exception
from price_processing import process_price_data
from risk_alerts import (
    METRICS_BY_KEY as FUNDAMENTAL_METRICS_BY_KEY,
    OPERATORS as FUNDAMENTAL_OPERATORS,
    compute_watchlist_snapshots,
)
from technical_indicators import (
    compute_macd,
    compute_rsi,
    compute_sma_lines,
    detect_macd_crossovers,
    detect_sma_crossovers,
    interpret_rsi,
)
from watchlist_panel import load_quote_snapshots

logger = get_logger("realtime_alerts")

PRICE_TRIGGER_TYPES = ("price_above", "price_below")
INDICATOR_TRIGGER_TYPES = (
    "sma_cross_bullish", "sma_cross_bearish",
    "macd_bullish_cross", "macd_bearish_cross",
    "rsi_overbought", "rsi_oversold",
)
FUNDAMENTAL_TRIGGER_TYPE = "fundamental"
ALL_TRIGGER_TYPES = PRICE_TRIGGER_TYPES + INDICATOR_TRIGGER_TYPES + (FUNDAMENTAL_TRIGGER_TYPE,)

TRIGGER_LABELS: Dict[str, str] = {
    "price_above": "Price rises to or above",
    "price_below": "Price falls to or below",
    "sma_cross_bullish": f"Price crosses ABOVE its {CHART_DEFAULTS.sma_default}-period SMA (bullish)",
    "sma_cross_bearish": f"Price crosses BELOW its {CHART_DEFAULTS.sma_default}-period SMA (bearish)",
    "macd_bullish_cross": "MACD crosses above its Signal line (bullish)",
    "macd_bearish_cross": "MACD crosses below its Signal line (bearish)",
    "rsi_overbought": f"RSI({CHART_DEFAULTS.rsi_default}) enters overbought (≥ {TECHNICAL.rsi_overbought:.0f})",
    "rsi_oversold": f"RSI({CHART_DEFAULTS.rsi_default}) enters oversold (≤ {TECHNICAL.rsi_oversold:.0f})",
    "fundamental": "Fundamental / risk metric threshold",
}


@dataclass
class AlertRule:
    id: str
    ticker: str
    trigger_type: str
    threshold: Optional[float] = None   # price_above / price_below / fundamental
    metric: Optional[str] = None        # fundamental only: key into risk_alerts.METRICS_BY_KEY
    operator: Optional[str] = None      # fundamental only: key into risk_alerts.OPERATORS
    created_at: str = ""

    @property
    def label(self) -> str:
        if self.trigger_type in PRICE_TRIGGER_TYPES:
            # A plain, unescaped "$" on purpose: this value also feeds
            # st.table's Trigger History (plain text, not markdown), so it
            # must stay a literal dollar sign here. Any markdown-rendering
            # caller (st.markdown/st.caption/st.toast) is responsible for
            # escaping it at display time instead — see finance.py's
            # `_rt_md_escape_dollar()` and its call-site comment for why:
            # concatenating two labels/details that each carry one bare "$"
            # into a single markdown call forms a matched "$...$" pair,
            # which Streamlit renders as inline LaTeX math, not literal
            # text. Caught live: a rule's active-alert line rendered as a
            # raw, unparsed LaTeX box instead of "$1.00".
            return f"{self.ticker}: {TRIGGER_LABELS[self.trigger_type]} ${self.threshold:,.2f}"
        if self.trigger_type == FUNDAMENTAL_TRIGGER_TYPE:
            spec = FUNDAMENTAL_METRICS_BY_KEY[self.metric]
            thr = f"{self.threshold * 100:.{spec.decimals}f}%" if spec.unit == "%" else f"{self.threshold:.{spec.decimals}f}"
            return f"{self.ticker}: {spec.label} {self.operator} {thr}"
        return f"{self.ticker}: {TRIGGER_LABELS[self.trigger_type]}"


@dataclass
class TriggerEvent:
    rule_id: str
    ticker: str
    trigger_type: str
    detail: str
    triggered_at: str  # ISO timestamp, poll time


@dataclass
class EvaluationResult:
    rule_id: str
    is_met: bool
    detail: str
    status: str = "ok"  # "ok" | "insufficient_data" | "fetch_error"


def new_rule_id() -> str:
    return uuid.uuid4().hex[:12]


# --- Persistence -------------------------------------------------------------

def _default_store_path() -> Path:
    return store_path(REALTIME_ALERTS.store_filename)


def load_store(path: Optional[Path] = None) -> Tuple[List[AlertRule], List[TriggerEvent]]:
    """Never raises: a missing or corrupt store is treated as empty rather
    than crashing the app on load, since this file is local runtime state,
    not something the user directly edits."""
    path = path or _default_store_path()
    if not path.exists():
        return [], []
    try:
        raw = json.loads(path.read_text())
        rules = [AlertRule(**r) for r in raw.get("rules", [])]
        history = [TriggerEvent(**h) for h in raw.get("history", [])]
        return rules, history
    except Exception:
        log_exception(logger, "store.corrupt", section="realtime_alerts")
        return [], []


def save_store(rules: List[AlertRule], history: List[TriggerEvent], path: Optional[Path] = None) -> None:
    """Trims history to REALTIME_ALERTS.max_history and writes atomically
    (write-to-temp then rename) so a crash mid-write can never leave a
    truncated, unparseable store behind."""
    path = path or _default_store_path()
    payload = {
        "rules": [asdict(r) for r in rules],
        "history": [asdict(h) for h in history[-REALTIME_ALERTS.max_history:]],
    }
    atomic_write_text(path, json.dumps(payload, indent=2))


# --- Evaluation ----------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def _load_indicator_frame(ticker: str):
    """One ticker's price history plus the SMA/MACD/RSI columns every
    indicator-trigger rule needs, at the app's existing default lengths.
    5-minute TTL matches watchlist_panel's quote cadence — cheap enough to
    recompute on a poll's cache-miss, shared across every indicator rule
    for the same ticker within the window. Returns (df, "") on success or
    (None, detail) on failure — never raises."""
    try:
        bundle = load_ticker_bundle(ticker, deep=True)
    except Exception as e:
        log_exception(logger, "indicator_frame.error", section="realtime_alerts", ticker=ticker)
        return None, f"unexpected error: {type(e).__name__}: {e}"

    if not bundle.is_valid or bundle.price_history.empty:
        return None, "; ".join(bundle.errors) or "no price history returned"

    df = process_price_data(bundle.price_history, ticker=ticker).df
    df = compute_sma_lines(df, [CHART_DEFAULTS.sma_default])
    df = compute_macd(df)
    df["RSI"] = compute_rsi(df, CHART_DEFAULTS.rsi_default)
    return df, ""


def _evaluate_price_rule(rule: AlertRule, quote) -> EvaluationResult:
    if quote.status != "ok" or quote.price is None:
        return EvaluationResult(rule.id, False, quote.detail or "quote unavailable", status="fetch_error")
    if rule.trigger_type == "price_above":
        met = quote.price >= rule.threshold
    else:
        met = quote.price <= rule.threshold
    # Plain "$" on purpose — see AlertRule.label's comment.
    return EvaluationResult(rule.id, met, f"current price ${quote.price:,.2f}")


def _evaluate_indicator_rule(rule: AlertRule, df, err: str) -> EvaluationResult:
    if df is None or df.empty:
        return EvaluationResult(rule.id, False, err or "no data", status="insufficient_data")

    if rule.trigger_type in ("sma_cross_bullish", "sma_cross_bearish"):
        crossings = detect_sma_crossovers(df, CHART_DEFAULTS.sma_default)
        wanted = "bullish" if rule.trigger_type == "sma_cross_bullish" else "bearish"
        if not crossings:
            return EvaluationResult(rule.id, False, "no crossover in the loaded history")
        latest = crossings[-1]
        met = latest.kind == wanted and latest.date == df.index[-1]
        detail = f"latest crossover: {latest.kind} on {latest.date.date()}"
        return EvaluationResult(rule.id, met, detail)

    if rule.trigger_type in ("macd_bullish_cross", "macd_bearish_cross"):
        crossings = detect_macd_crossovers(df)
        wanted = "bullish" if rule.trigger_type == "macd_bullish_cross" else "bearish"
        if not crossings:
            return EvaluationResult(rule.id, False, "no crossover in the loaded history")
        latest = crossings[-1]
        met = latest.kind == wanted and latest.date == df.index[-1]
        detail = f"latest crossover: {latest.kind} on {latest.date.date()}"
        return EvaluationResult(rule.id, met, detail)

    if rule.trigger_type in ("rsi_overbought", "rsi_oversold"):
        latest_rsi = df["RSI"].iloc[-1]
        interp = interpret_rsi(latest_rsi if pd.notna(latest_rsi) else None)
        if interp is None:
            return EvaluationResult(rule.id, False, "RSI not yet computable (warm-up period)", status="insufficient_data")
        wanted_zone = "overbought" if rule.trigger_type == "rsi_overbought" else "oversold"
        met = interp.zone == wanted_zone
        return EvaluationResult(rule.id, met, f"RSI currently {interp.value:.1f} ({interp.zone})")

    return EvaluationResult(rule.id, False, f"unknown trigger type: {rule.trigger_type}", status="fetch_error")


def _evaluate_fundamental_rule(rule: AlertRule, snapshot) -> EvaluationResult:
    if snapshot.status != "ok":
        return EvaluationResult(rule.id, False, snapshot.detail or "metric unavailable", status="fetch_error")
    value = snapshot.values.get(rule.metric)
    if value is None:
        return EvaluationResult(rule.id, False, f"{FUNDAMENTAL_METRICS_BY_KEY[rule.metric].label} not computable for this ticker", status="insufficient_data")
    met = FUNDAMENTAL_OPERATORS[rule.operator](value, rule.threshold)
    spec = FUNDAMENTAL_METRICS_BY_KEY[rule.metric]
    value_display = f"{value * 100:.{spec.decimals}f}%" if spec.unit == "%" else f"{value:.{spec.decimals}f}"
    return EvaluationResult(rule.id, met, f"current {spec.label}: {value_display}")


def evaluate_all(rules: List[AlertRule]) -> Dict[str, EvaluationResult]:
    """Every rule's current state, batched by trigger family so each
    ticker's quote/indicator-frame/fundamental-snapshot is fetched once
    and shared across every rule on that ticker that needs it — same
    batching principle watchlist_panel.load_quote_snapshots() and
    risk_alerts.compute_watchlist_snapshots() already use."""
    if not rules:
        return {}

    results: Dict[str, EvaluationResult] = {}

    price_rules = [r for r in rules if r.trigger_type in PRICE_TRIGGER_TYPES]
    if price_rules:
        price_tickers = tuple(dict.fromkeys(r.ticker for r in price_rules))
        quotes = {q.ticker: q for q in load_quote_snapshots(price_tickers)}
        for r in price_rules:
            results[r.id] = _evaluate_price_rule(r, quotes[r.ticker])

    indicator_rules = [r for r in rules if r.trigger_type in INDICATOR_TRIGGER_TYPES]
    if indicator_rules:
        indicator_tickers = tuple(dict.fromkeys(r.ticker for r in indicator_rules))
        frames = {t: _load_indicator_frame(t) for t in indicator_tickers}
        for r in indicator_rules:
            df, err = frames[r.ticker]
            results[r.id] = _evaluate_indicator_rule(r, df, err)

    fundamental_rules = [r for r in rules if r.trigger_type == FUNDAMENTAL_TRIGGER_TYPE]
    if fundamental_rules:
        fundamental_tickers = tuple(dict.fromkeys(r.ticker for r in fundamental_rules))
        snapshots = {s.ticker: s for s in compute_watchlist_snapshots(fundamental_tickers)}
        for r in fundamental_rules:
            results[r.id] = _evaluate_fundamental_rule(r, snapshots[r.ticker])

    return results


def detect_new_triggers(results: Dict[str, EvaluationResult], previously_active: Dict[str, bool]) -> List[str]:
    """Rule ids that just transitioned from not-met to met this poll — see
    the module docstring's "EDGE-TRIGGERING" section for why an absent
    entry in `previously_active` (a session's first poll) counts as
    "was not active" rather than being skipped."""
    return [
        rule_id for rule_id, result in results.items()
        if result.is_met and not previously_active.get(rule_id, False)
    ]
