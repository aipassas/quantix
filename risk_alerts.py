"""Smart Risk-Aware Alerts — check-on-load alert conditions built from
Quantix's own computed risk metrics (Composite Risk Score, Altman Z-Score,
1-Day Historical VaR, Expected Shortfall/CVaR, Max Drawdown), evaluated
across the existing Institutional Watchlist universe.

Every metric is computed by risk_analytics.py / fundamental_analysis.py —
no new risk arithmetic here, only the alert schema, the per-ticker
fetch/compute pipeline (mirroring exactly how finance.py already builds
the single analyzed ticker's Risk Dashboard), and the evaluator.

Explicitly in-session / check-on-load, not real-time push: Quantix is a
stateless Streamlit process with no persistent background worker, so
"triggered" here means "this metric is currently past the threshold" at
the moment of evaluation — a snapshot check, not a historical
boundary-crossing event. Real-time push delivery (email/SMS) needs new
infrastructure (a background worker) and is a distinct follow-on task,
deliberately out of scope here (see realtime_alerts.py for the app's
actual answer to that — per-ticker auto-polling rules, in-app delivery).

The CONFIGURED RULES (which metrics/thresholds to check) DO persist
across restarts — see load_rules()/save_rules() below, the same
atomic-write local-file pattern every other cross-restart store in this
app uses. What still doesn't persist, deliberately, is any notion of a
past trigger EVENT: every click of "Check Alerts" is a fresh snapshot
against current data, never a replay of a historical crossing, so there
is nothing here resembling realtime_alerts.py's trigger history.
"""
import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import streamlit as st

from config import CHART_DEFAULTS, RISK, WATCHLIST
from data_loader import load_ticker_bundle
from financial_standardization import standardize_financials
from fundamental_analysis import FundamentalAnalysisEngine
from local_store import atomic_write_text
from logging_setup import get_logger, log_event, log_exception
from price_processing import process_price_data
from risk_analytics import (
    compute_calmar_ratio,
    compute_expected_shortfall,
    compute_historical_var,
    compute_max_drawdown,
    compute_risk_score,
    compute_rolling_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
)

logger = get_logger("risk_alerts")


@dataclass(frozen=True)
class AlertMetricSpec:
    key: str
    label: str
    unit: str
    decimals: int
    default_operator: str
    default_threshold: float


# Every metric named explicitly in the task notes: Composite Risk Score,
# Altman Z-Score, a VaR/CVaR breach, Max Drawdown. Default operator/
# threshold reflect this codebase's own sign conventions — VaR/CVaR/Max
# Drawdown are signed log returns (negative = a loss, see risk_analytics.py
# docstrings), so a "breach" alert is naturally a "<" comparison against a
# negative threshold. Altman Z's default threshold is the app's own
# already-configured Distress-zone boundary (RISK.altman_grey_zone), not a
# new invented cutoff.
METRICS: Tuple[AlertMetricSpec, ...] = (
    AlertMetricSpec("risk_score", "Composite Risk Score", "", 1, "<", 50.0),
    AlertMetricSpec("altman_z", "Altman Z-Score", "", 2, "<", RISK.altman_grey_zone),
    AlertMetricSpec("historical_var", "1-Day Historical VaR", "%", 2, "<", -0.05),
    AlertMetricSpec("expected_shortfall", "Expected Shortfall (CVaR)", "%", 2, "<", -0.08),
    AlertMetricSpec("max_drawdown", "Max Drawdown", "%", 2, "<", -0.20),
)
METRICS_BY_KEY: Dict[str, AlertMetricSpec] = {m.key: m for m in METRICS}

OPERATORS: Dict[str, Callable[[float, float], bool]] = {
    "<": lambda v, t: v < t,
    ">": lambda v, t: v > t,
    "<=": lambda v, t: v <= t,
    ">=": lambda v, t: v >= t,
}


@dataclass(frozen=True)
class AlertRule:
    metric: str    # key into METRICS_BY_KEY
    operator: str  # key into OPERATORS
    threshold: float


@dataclass
class TickerRiskSnapshot:
    ticker: str
    values: Dict[str, Optional[float]] = field(default_factory=dict)
    status: str = "ok"   # "ok" | "insufficient_data" | "fetch_error"
    detail: str = ""


@dataclass(frozen=True)
class TriggeredAlert:
    ticker: str
    rule: AlertRule
    value: float


_RULES_STORE_FILENAME = "risk_alert_rules_store.json"


def _rules_store_path() -> Path:
    return Path(__file__).resolve().parent / _RULES_STORE_FILENAME


def load_rules(path: Optional[Path] = None) -> Optional[List[dict]]:
    """The configured rule list (plain dicts — {"metric", "operator",
    "threshold"} — the exact shape finance.py already keeps in
    st.session_state, so no dataclass conversion is needed on either
    side). Returns None (not an empty list) when no store file exists
    yet, so the caller can distinguish "never configured, use the
    built-in default rules" from "configured down to zero rules on
    purpose" — an empty list IS a valid, deliberate configuration and
    must not be silently replaced with the default pair on every reload.
    A corrupt file also returns None, degrading to the same default
    rather than crashing the app on load.
    """
    path = path or _rules_store_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        log_exception(logger, "risk_alerts.rules_store_corrupt", section="risk_alerts")
        return None


def save_rules(rules: List[dict], path: Optional[Path] = None) -> None:
    """Atomic write (temp file + rename), same pattern as every other
    local store in this app."""
    path = path or _rules_store_path()
    atomic_write_text(path, json.dumps(rules, indent=2))


def watchlist_tickers() -> Tuple[str, ...]:
    """The existing Institutional Watchlist universe (both baskets),
    deduplicated — reused as-is per the task's "reusing the existing
    watchlist ticker list rather than a separate ticker-entry flow"
    requirement."""
    return tuple(dict.fromkeys(WATCHLIST.tech_basket + WATCHLIST.diversified_basket))


@st.cache_data(ttl=3600, show_spinner=False)
def _compute_ticker_risk_snapshot(ticker: str) -> TickerRiskSnapshot:
    """One ticker's full risk-metric bundle — the exact same pipeline
    finance.py's Risk Dashboard uses for the single analyzed ticker
    (compute_historical_var -> compute_expected_shortfall -> Sharpe/Sortino
    -> rolling volatility -> Max Drawdown -> Calmar -> Altman Z ->
    compute_risk_score), just run once per watchlist ticker instead of the
    one currently loaded in the sidebar. Uses config defaults for
    confidence/lookback/volatility-window/risk-free-rate rather than the
    sidebar's per-session slider values, since this scans a fixed watchlist
    independent of whatever ticker/settings are currently on screen.

    Needs a deep=True bundle for every ticker (unlike the Stock Screener's
    adaptive shallow/deep tiers) because Altman Z — and therefore the
    Composite Risk Score itself — always needs full statements.
    """
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=CHART_DEFAULTS.default_lookback_days)
        bundle = load_ticker_bundle(ticker, start, end, deep=True)
    except Exception as e:
        log_exception(logger, "calc.error", section="risk_alerts", ticker=ticker)
        return TickerRiskSnapshot(ticker=ticker, status="fetch_error", detail=f"Unexpected error: {type(e).__name__}: {e}")

    if not bundle.is_valid:
        return TickerRiskSnapshot(ticker=ticker, status="fetch_error", detail="; ".join(bundle.errors) or "Ticker could not be loaded")

    df = process_price_data(bundle.price_history, ticker=ticker).df
    df["Returns"] = df["Close"].pct_change()

    confidence = RISK.var_confidence_default
    lookback = CHART_DEFAULTS.var_lookback_default
    historical_var = compute_historical_var(df, confidence, lookback=lookback)
    expected_shortfall = compute_expected_shortfall(df, confidence, lookback=lookback)
    sharpe_ratio = compute_sharpe_ratio(df, RISK.risk_free_rate)
    sortino_ratio = compute_sortino_ratio(df, RISK.risk_free_rate)
    rolling_vol_series = compute_rolling_volatility(df, CHART_DEFAULTS.vol_window_default)
    current_rolling_vol = rolling_vol_series.dropna().iloc[-1] if rolling_vol_series.notna().any() else None
    max_dd_result = compute_max_drawdown(df["Close"])
    max_drawdown = max_dd_result.max_drawdown if max_dd_result is not None else None
    calmar_ratio = compute_calmar_ratio(df)

    standardized = standardize_financials(bundle)
    fundamentals_engine = FundamentalAnalysisEngine(standardized, raw_info=bundle.info)
    altman_z, _verdict, _missing = fundamentals_engine.altman_z_score()

    risk_score_result = compute_risk_score(
        rolling_volatility=current_rolling_vol, historical_var=historical_var,
        expected_shortfall=expected_shortfall, max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio, altman_z=altman_z,
    )

    values: Dict[str, Optional[float]] = {
        "risk_score": risk_score_result.score,
        "altman_z": altman_z,
        "historical_var": historical_var,
        "expected_shortfall": expected_shortfall,
        "max_drawdown": max_drawdown,
    }
    missing = [METRICS_BY_KEY[k].label for k, v in values.items() if v is None]
    status = "ok" if not missing else "insufficient_data"
    detail = f"Not computable: {', '.join(missing)}" if missing else ""
    return TickerRiskSnapshot(ticker=ticker, values=values, status=status, detail=detail)


@st.cache_data(ttl=3600, show_spinner=False)
def compute_watchlist_snapshots(tickers: Tuple[str, ...]) -> List[TickerRiskSnapshot]:
    log_event(logger, logging.INFO, "risk_alerts.snapshot_scan", watchlist_size=len(tickers))
    return [_compute_ticker_risk_snapshot(t) for t in tickers]


def evaluate_alerts(snapshots: List[TickerRiskSnapshot], rules: Tuple[AlertRule, ...]) -> List[TriggeredAlert]:
    """Every (ticker, rule) pair where the ticker's current value for that
    metric satisfies the rule. A ticker missing that specific metric is
    silently skipped for that rule (not a false trigger, not an error) —
    its snapshot's own status/detail already discloses why, surfaced
    separately in the UI."""
    triggered: List[TriggeredAlert] = []
    for snap in snapshots:
        for rule in rules:
            value = snap.values.get(rule.metric)
            if value is None:
                continue
            if OPERATORS[rule.operator](value, rule.threshold):
                triggered.append(TriggeredAlert(ticker=snap.ticker, rule=rule, value=value))
    return triggered
