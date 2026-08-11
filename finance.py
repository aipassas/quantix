import logging
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pandas as pd
import numpy as np
import zlib

from data_loader import load_ticker_bundle, load_macro_bundle, load_seasonality_history, load_price_history_only, clear_all_caches
from financial_standardization import standardize_financials
from price_processing import process_price_data
from technical_indicators import compute_sma_lines, detect_sma_crossovers, compute_rsi, interpret_rsi, compute_macd, detect_macd_crossovers, compute_bollinger_bands, detect_bollinger_breakouts, compute_atr, suggested_stop_loss, compute_stochastic, detect_stochastic_crossovers, compute_anchored_vwap, compute_adx, compute_ichimoku, compute_obv
from risk_analytics import compute_rolling_volatility, compute_annualized_volatility, compute_historical_var, compute_parametric_var, compute_expected_shortfall, interpret_tail_risk, compute_log_returns, compute_max_drawdown, compute_drawdown_series, compute_annualized_return, compute_sharpe_ratio, interpret_sharpe_ratio, compute_sortino_ratio, compute_downside_deviation, compute_calmar_ratio, interpret_calmar_ratio, compute_risk_score, compute_hurst_exponent
from portfolio_analytics import build_aligned_returns, compute_correlation_matrix, compute_portfolio_diversification, compute_capm_beta, compute_performance_attribution, compute_efficient_frontier
from report_export import generate_tear_sheet_pdf
from data_quality import assess_data_quality
from config import WATCHLIST, SCORECARD, DCF, RISK, MONTE_CARLO, CHART_DEFAULTS, PEER_DEFAULTS, TEAR_SHEET, TECHNICAL, WALK_FORWARD, BACKTEST_COST, WATCHLIST_PANEL, REALTIME_ALERTS, PORTFOLIO_BACKTEST, ML_PIPELINE
from fundamental_analysis import FundamentalAnalysisEngine
from logging_setup import setup_logging, get_logger, log_event, log_exception, recent_logs, log_file_path
from screener import METRICS as SCREENER_METRICS, METRICS_BY_KEY as SCREENER_METRICS_BY_KEY, OPERATORS as SCREENER_OPERATORS, MAX_UNIVERSE_SIZE as SCREENER_MAX_UNIVERSE_SIZE, ScreenCriterion, run_screen
from strategy_builder import LOGIC_OPTIONS, StrategyCondition, StrategyRule, classic_mean_reversion, condition_library, evaluate_condition_set, run_backtest, run_walk_forward_backtest
from portfolio_backtester import REBALANCE_FREQUENCIES, REBALANCE_FREQUENCY_LABELS, prepare_ticker_for_backtest, run_portfolio_backtest
from ml_pipeline import (
    load_history as load_ml_history,
    load_model as load_ml_model,
    predict_latest,
    save_model as save_ml_model,
    train_momentum_model,
    training_universe,
)
from sector_percentile import MIN_PEERS as SECTOR_MIN_PEERS, compute_sector_percentiles, format_percentile
from risk_alerts import METRICS as RISK_ALERT_METRICS, METRICS_BY_KEY as RISK_ALERT_METRICS_BY_KEY, OPERATORS as RISK_ALERT_OPERATORS, AlertRule, compute_watchlist_snapshots, evaluate_alerts, watchlist_tickers
from realtime_alerts import (
    ALL_TRIGGER_TYPES as RT_ALL_TRIGGER_TYPES,
    FUNDAMENTAL_TRIGGER_TYPE as RT_FUNDAMENTAL_TRIGGER_TYPE,
    PRICE_TRIGGER_TYPES as RT_PRICE_TRIGGER_TYPES,
    TRIGGER_LABELS as RT_TRIGGER_LABELS,
    AlertRule as RealtimeAlertRule,
    TriggerEvent as RealtimeTriggerEvent,
    detect_new_triggers as rt_detect_new_triggers,
    evaluate_all as rt_evaluate_all,
    load_store as rt_load_store,
    new_rule_id as rt_new_rule_id,
    save_store as rt_save_store,
)
from executive_digest import collect_flags
from monte_carlo import simulate_gbm_paths, simulate_bootstrap_paths, terminal_stats
from watchlist_panel import add_ticker, load_quote_snapshots, parse_tickers, record_recent, remove_ticker


def fmt_num(value, suffix="", decimals=2, prefix=""):
    """Format a number for display, or 'N/A' when the underlying field was missing."""
    return "N/A" if value is None else f"{prefix}{value:.{decimals}f}{suffix}"


# --- Logging ---
# The debug toggle lives in the sidebar (rendered further down), but the log
# level has to be set before any module logs anything. Streamlit persists
# widget values in session_state across reruns, so the previous run's toggle
# state is already available here on this run.
setup_logging(logging.DEBUG if st.session_state.get("debug_mode") else logging.INFO)
logger = get_logger("finance")

if not st.session_state.get("_session_logged"):
    log_event(logger, logging.INFO, "session.start", log_file=log_file_path())
    st.session_state["_session_logged"] = True


def log_input_changes(**current):
    """Log only genuine changes to the meaningful inputs.

    Streamlit re-runs this entire script on every widget interaction, so
    logging unconditionally would emit an entry per slider increment. Diffing
    against the previous run's values keeps the log to real user intent
    (ticker switch, date change, benchmark change, peer edit).
    """
    previous = st.session_state.get("_last_inputs")
    if previous is not None:
        changed = {key: value for key, value in current.items() if previous.get(key) != value}
        if changed:
            log_event(logger, logging.INFO, "user.input_changed", **changed)
    st.session_state["_last_inputs"] = current


# --- Page Configuration ---
st.set_page_config(page_title="Quantix", layout="wide", page_icon=None)
st.title("Quantix: Institutional-Grade Stock Analysis & Simulation Engine")

# Sticky symbol header slot. Reserved HERE, at the very top, so the
# ticker/price/day-change is on screen the moment the page loads — but it
# can only be FILLED after the data fetch further down, so it uses the
# same container-as-placeholder pattern executive_digest_container already
# uses (content written into a container later still renders at the
# container's position). See the "SYMBOL HEADER (fill)" block below.
symbol_header_container = st.container()

# --- Professional UI Injection (OLED Edition) ---
st.markdown("""
    <style>
    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ...but keep the sidebar re-open button visible. It lives inside the
       header, so hiding the header above would otherwise make collapsing the
       sidebar irreversible without reloading the page. stExpandSidebarButton
       is the Streamlit 1.58 test ID; the other two are older/newer aliases,
       matched so this keeps working across version upgrades. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        z-index: 999999;
    }
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {
        visibility: visible !important;
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Absolute OLED Black Background */
    .stApp {
        background-color: #000000 !important;
        color: #e2e8f0;
    }

    /* Headers in crisp white */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* OLED Metric Cards - High Contrast */
    div[data-testid="metric-container"] {
        background-color: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-left: 4px solid #00ea77; /* Neon Green Action Border */
        border-radius: 6px;
        padding: 15px 20px;
        transition: border-left-color 0.3s ease, transform 0.2s ease;
    }

    /* Hover state for absolute black theme */
    div[data-testid="metric-container"]:hover {
        border-left: 4px solid #ffffff;
        transform: scale(1.02);
    }

    /* Pop-out values for readability */
    [data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 700;
        color: #ffffff;
    }

    /* Adjust table styles for high contrast */
    thead tr th {
        background-color: #0a0a0a !important;
        color: #ffffff !important;
        border-bottom: 2px solid #333333 !important;
    }

    tbody tr td {
        background-color: #000000 !important;
        color: #cccccc !important;
        border-bottom: 1px solid #1a1a1a !important;
    }

    /* Ensure expanders remain legible */
    .streamlit-expanderHeader {
        background-color: #0a0a0a !important;
        border: 1px solid #1a1a1a !important;
        color: #ffffff !important;
    }

    /* --- Top-level panel navigation ---------------------------------
       The main page tabs are the primary navigation for the whole
       analysis, so they get real presence instead of Streamlit's default
       small text links: larger type, generous hit area, a visible rail
       under the strip, and a solid highlight + underline on the active
       panel. Scoped to .stTabs so it applies to nested tab groups too. */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #1f1f1f;
        padding-bottom: 0;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        height: 52px;
        padding: 0 16px;
        background-color: #0a0a0a;
        border: 1px solid #1a1a1a;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        white-space: nowrap;
    }
    /* Streamlit's default inactive-tab colour is near-black (#31333F), which
       is effectively invisible against this theme's black background — set
       an explicitly readable grey so unselected panels are still legible. */
    .stTabs [data-baseweb="tab"] p {
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
        color: #9ca3af !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #141414;
    }
    .stTabs [data-baseweb="tab"]:hover p {
        color: #e5e7eb !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #161616 !important;
        border-color: #2a2a2a !important;
    }
    .stTabs [aria-selected="true"] p {
        color: #ffffff !important;
    }
    /* Streamlit's own active-tab underline, thickened to match. */
    .stTabs [data-baseweb="tab-highlight"] {
        height: 3px;
    }

    /* The sidebar's control tabs share the strip styling but stay
       compact — they sit in a much narrower column. */
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 12px;
    }
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] p {
        font-size: 0.86rem !important;
    }

    /* --- Sticky symbol header --------------------------------------
       Streamlit has no sticky-header primitive. A sticky element can only
       travel within its PARENT's box, so this has to be applied to the
       outermost wrapper st.container() produces — the one whose parent is
       the tall page-level vertical block. Targeting anything further in
       (the markdown div, or even the stElementContainer) pins it inside a
       ~48px box, which looks identical to not being sticky at all; that
       was verified the hard way in-browser before landing on this
       selector. Also verified that the scroll container is
       section[data-testid="stMain"] and every ancestor between it and the
       content is overflow:visible, so sticky resolves against stMain's
       scrollport as intended.

       Both wrapper test-ids are targeted so a Streamlit version that
       renames or drops one still leaves a working rule; whichever
       resolves to a tall-parent element does the sticking, and a nested
       pair is harmless. */
    [data-testid="stLayoutWrapper"]:has(.quantix-symbol-header),
    [data-testid="stVerticalBlockBorderWrapper"]:has(.quantix-symbol-header) {
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .quantix-symbol-header {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 8px 18px;
        padding: 10px 18px;
        margin-bottom: 4px;
        background: #0a0a0a;
        border: 1px solid #1f1f1f;
        border-left: 4px solid #00ea77;
        border-radius: 8px;
        /* Opaque background plus a shadow so page content scrolling
           underneath never shows through or visually collides. */
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.85);
    }
    .quantix-symbol-header .qsh-ticker {
        font-size: 1.5rem; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;
    }
    .quantix-symbol-header .qsh-name {
        font-size: 0.95rem; color: #9ca3af; margin-right: auto;
    }
    .quantix-symbol-header .qsh-price {
        font-size: 1.5rem; font-weight: 700; color: #ffffff;
    }
    .quantix-symbol-header .qsh-change { font-size: 1.05rem; font-weight: 600; }
    .quantix-symbol-header .qsh-up { color: #22c55e; }
    .quantix-symbol-header .qsh-down { color: #ef4444; }
    .quantix-symbol-header .qsh-flat { color: #9ca3af; }
    .quantix-symbol-header .qsh-meta { font-size: 0.85rem; color: #6b7280; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# INSTITUTIONAL WATCHLIST SUGGESTIONS (CHRONOLOGICAL PORTFOLIOS)
# ==========================================
st.markdown("---")
st.header("Institutional Watchlist Suggestions")

@st.cache_data(ttl=3600)
def process_ticker_data(ticker):
    """Helper function to fetch and score a single ticker safely"""
    try:
        bundle = load_ticker_bundle(ticker, deep=False)
        if not bundle.is_valid:
            st.warning(f"Could not process ticker '{ticker}': {'; '.join(bundle.errors)}")
            return None
        std = standardize_financials(bundle)

        screened = FundamentalAnalysisEngine(std).screen_watchlist()
        if screened is None:
            return None

        return {
            "ticker": screened.ticker,
            "score": screened.score,
            "status": screened.status,
            "pe": screened.pe_ratio,
            "margin": screened.net_margin_pct,
        }
    except Exception as e:
        # load_ticker_bundle/standardize_financials already handle routine
        # data issues internally (returning bundle.errors / None fields)
        # without raising, so anything caught here is a genuinely unexpected
        # bug rather than ordinary bad data — log it fully and skip this ticker.
        log_exception(logger, "calc.error", section="watchlist_scan", ticker=ticker)
        st.warning(f"Unexpected error processing ticker '{ticker}': {type(e).__name__}: {e}")
        return None

@st.cache_data(ttl=3600)
def scan_split_watchlists():
    tech_results = []
    other_results = []

    for t in WATCHLIST.tech_basket:
        res = process_ticker_data(t)
        if res: tech_results.append(res)

    for t in WATCHLIST.diversified_basket:
        res = process_ticker_data(t)
        if res: other_results.append(res)

    # Sort both lists by best blueprint score first
    tech_results = sorted(tech_results, key=lambda x: x['score'], reverse=True)
    other_results = sorted(other_results, key=lambda x: x['score'], reverse=True)

    return tech_results, other_results

with st.spinner("Analyzing macro sectors and grouping asset classes..."):
    tech_picks, other_picks = scan_split_watchlists()

    # --- SECTION 1: TECH & GROWTH PORTFOLIO ---
    st.subheader("Top Tech & Growth Alignments")
    if tech_picks:
        t_cols = st.columns(WATCHLIST.cards_shown)
        for i, data in enumerate(tech_picks[:WATCHLIST.cards_shown]):
            with t_cols[i]:
                st.info(f"**{data['ticker']}**  \n  \nAlignment: {data['status']} ({data['score']:.0f}%)  \nP/E: {data['pe']:.1f}  \nMargin: {data['margin']:.1f}%")
    else:
        st.write("No tech leaders currently pass basic filters.")

    st.markdown("##") # Space out the sections cleanly

    # --- SECTION 2: DIVERSIFIED SECTOR PORTFOLIO ---
    st.subheader("Top Diversified Market Alignments")
    if other_picks:
        o_cols = st.columns(WATCHLIST.cards_shown)
        for i, data in enumerate(other_picks[:WATCHLIST.cards_shown]):
            with o_cols[i]:
                st.info(f"**{data['ticker']}**  \n  \nAlignment: {data['status']} ({data['score']:.0f}%)  \nP/E: {data['pe']:.1f}  \nMargin: {data['margin']:.1f}%")
    else:
        st.write("No diversified sector leaders currently pass basic filters.")

# ==========================================
# STOCK SCREENER
# ==========================================
st.markdown("---")
st.header("Stock Screener")
st.caption("Screen an arbitrary ticker universe against your own fundamental, technical, and risk criteria — not just the fixed thresholds above.")

_screener_default_universe = ", ".join(WATCHLIST.tech_basket + WATCHLIST.diversified_basket)
screener_universe_input = st.text_input(
    "Ticker Universe (comma-separated)", value=_screener_default_universe,
    help=f"Up to {SCREENER_MAX_UNIVERSE_SIZE} tickers — Yahoo Finance rate limits scale with universe size, so larger universes get truncated with a warning.",
)

if "screener_criteria" not in st.session_state:
    st.session_state["screener_criteria"] = [{"metric": "pe_ratio", "operator": "<", "threshold": 25.0}]

st.markdown("**Filter Criteria**")
_screener_metric_options = [m.key for m in SCREENER_METRICS]
_screener_operator_options = list(SCREENER_OPERATORS.keys())

_screener_remove_index = None
for _i, _crit in enumerate(st.session_state["screener_criteria"]):
    _c1, _c2, _c3, _c4 = st.columns([3, 1, 2, 1])
    with _c1:
        _crit["metric"] = st.selectbox(
            "Metric", _screener_metric_options, index=_screener_metric_options.index(_crit["metric"]),
            format_func=lambda k: SCREENER_METRICS_BY_KEY[k].label, key=f"screener_metric_{_i}",
            label_visibility="visible" if _i == 0 else "collapsed",
        )
    with _c2:
        _crit["operator"] = st.selectbox(
            "Op", _screener_operator_options, index=_screener_operator_options.index(_crit["operator"]),
            key=f"screener_operator_{_i}", label_visibility="visible" if _i == 0 else "collapsed",
        )
    with _c3:
        _crit["threshold"] = st.number_input(
            "Threshold", value=float(_crit["threshold"]), key=f"screener_threshold_{_i}",
            label_visibility="visible" if _i == 0 else "collapsed",
        )
    with _c4:
        if _i == 0:
            st.markdown("&nbsp;")  # aligns the remove button with the inputs, which have a label row above them on row 0
        if st.button("✕", key=f"screener_remove_{_i}", help="Remove this filter"):
            _screener_remove_index = _i

if _screener_remove_index is not None:
    st.session_state["screener_criteria"].pop(_screener_remove_index)
    st.rerun()

_screener_btn_col1, _screener_btn_col2 = st.columns([1, 1])
with _screener_btn_col1:
    if st.button("+ Add Filter"):
        st.session_state["screener_criteria"].append({"metric": "rsi", "operator": "<", "threshold": 30.0})
        st.rerun()
with _screener_btn_col2:
    screener_run_clicked = st.button("Run Screen", type="primary")

if screener_run_clicked:
    _screener_universe = [t.strip().upper() for t in screener_universe_input.split(",") if t.strip()]
    _screener_universe = list(dict.fromkeys(_screener_universe))  # dedupe, preserve order
    if len(_screener_universe) > SCREENER_MAX_UNIVERSE_SIZE:
        st.warning(f"Universe capped at {SCREENER_MAX_UNIVERSE_SIZE} tickers — dropped: {', '.join(_screener_universe[SCREENER_MAX_UNIVERSE_SIZE:])}")
        _screener_universe = _screener_universe[:SCREENER_MAX_UNIVERSE_SIZE]

    if not _screener_universe:
        st.warning("Enter at least one ticker in the Ticker Universe field.")
    elif not st.session_state["screener_criteria"]:
        st.warning("Add at least one filter criterion.")
    else:
        _screener_criteria_tuple = tuple(
            ScreenCriterion(metric=c["metric"], operator=c["operator"], threshold=float(c["threshold"]))
            for c in st.session_state["screener_criteria"]
        )
        with st.spinner(f"Screening {len(_screener_universe)} ticker(s)..."):
            _screener_results = run_screen(tuple(_screener_universe), _screener_criteria_tuple)
        st.session_state["screener_results_state"] = {"results": _screener_results, "criteria": _screener_criteria_tuple}
        log_event(logger, logging.INFO, "user.screener_run", universe_size=len(_screener_universe), criteria_count=len(_screener_criteria_tuple))

_screener_state = st.session_state.get("screener_results_state")
if _screener_state:
    _screener_results = _screener_state["results"]
    _screener_criteria = _screener_state["criteria"]

    _screener_pass_count = sum(1 for r in _screener_results if r.passed_all)
    _screener_insufficient_count = sum(1 for r in _screener_results if r.status == "insufficient_data")
    _screener_error_count = sum(1 for r in _screener_results if r.status == "fetch_error")
    _screener_summary = f"{_screener_pass_count} of {len(_screener_results)} passed all filters"
    if _screener_insufficient_count:
        _screener_summary += f" · {_screener_insufficient_count} with insufficient data"
    if _screener_error_count:
        _screener_summary += f" · {_screener_error_count} could not be loaded"
    st.caption(_screener_summary)

    _screener_rows = []
    for r in _screener_results:
        row = {"Ticker": r.ticker}
        for c in _screener_criteria:
            spec = SCREENER_METRICS_BY_KEY[c.metric]
            v = r.values.get(c.metric)
            row[f"{spec.label} ({c.operator} {c.threshold:g}{spec.unit})"] = round(v, spec.decimals) if v is not None else None
        if r.status == "fetch_error":
            row["Result"] = "🟡 Could Not Load"
        elif r.status == "insufficient_data":
            row["Result"] = "🟡 Insufficient Data"
        elif r.passed_all:
            row["Result"] = "🟢 Pass"
        else:
            row["Result"] = "🔴 Fail"
        row["Detail"] = r.detail
        _screener_rows.append(row)

    _screener_results_df = pd.DataFrame(_screener_rows)
    _screener_event = st.dataframe(
        _screener_results_df, width="stretch", hide_index=True,
        on_select="rerun", selection_mode="single-row", key="screener_results_table",
    )

    _screener_selected_rows = _screener_event.selection.rows if _screener_event and _screener_event.selection else []
    if _screener_selected_rows:
        _screener_selected_ticker = _screener_results_df.iloc[_screener_selected_rows[0]]["Ticker"]
        if st.button(f"Open full analysis for {_screener_selected_ticker} →"):
            st.session_state["ticker_input"] = _screener_selected_ticker
            st.rerun()

# ==========================================
# SMART RISK-AWARE ALERTS
# ==========================================
st.markdown("---")
st.header("Smart Risk-Aware Alerts")
st.caption(
    "Alerts built from Quantix's own risk engine (Composite Risk Score, Altman Z-Score, 1-Day VaR, Expected Shortfall, "
    "Max Drawdown) across your Institutional Watchlist — checked when you click \"Check Alerts\", not a real-time push "
    "notification. Quantix is a stateless app with no background worker, so this is an on-load snapshot check "
    "(\"triggered right now\"), not a historical crossing event; live push delivery (email/SMS) would need new "
    "infrastructure and isn't built here."
)

if "risk_alert_rules" not in st.session_state:
    st.session_state["risk_alert_rules"] = [
        {"metric": "risk_score", "operator": "<", "threshold": 50.0},
        {"metric": "altman_z", "operator": "<", "threshold": RISK.altman_grey_zone},
    ]

_alert_metric_options = [m.key for m in RISK_ALERT_METRICS]
_alert_operator_options = list(RISK_ALERT_OPERATORS.keys())

_alert_remove_index = None
for _ai, _rule in enumerate(st.session_state["risk_alert_rules"]):
    _ac1, _ac2, _ac3, _ac4 = st.columns([3, 1, 2, 1])
    with _ac1:
        _rule["metric"] = st.selectbox(
            "Metric", _alert_metric_options, index=_alert_metric_options.index(_rule["metric"]),
            format_func=lambda k: RISK_ALERT_METRICS_BY_KEY[k].label, key=f"alert_metric_{_ai}",
            label_visibility="visible" if _ai == 0 else "collapsed",
        )
    with _ac2:
        _rule["operator"] = st.selectbox(
            "Op", _alert_operator_options, index=_alert_operator_options.index(_rule["operator"]),
            key=f"alert_operator_{_ai}", label_visibility="visible" if _ai == 0 else "collapsed",
        )
    with _ac3:
        _rule["threshold"] = st.number_input(
            "Threshold", value=float(_rule["threshold"]), key=f"alert_threshold_{_ai}",
            label_visibility="visible" if _ai == 0 else "collapsed",
        )
    with _ac4:
        if _ai == 0:
            st.markdown("&nbsp;")
        if st.button("✕", key=f"alert_remove_{_ai}", help="Remove this alert rule"):
            _alert_remove_index = _ai

if _alert_remove_index is not None:
    st.session_state["risk_alert_rules"].pop(_alert_remove_index)
    st.rerun()

_alert_btn_col1, _alert_btn_col2 = st.columns([1, 1])
with _alert_btn_col1:
    if st.button("+ Add Alert Rule"):
        st.session_state["risk_alert_rules"].append({"metric": "max_drawdown", "operator": "<", "threshold": -0.20})
        st.rerun()
with _alert_btn_col2:
    check_alerts_clicked = st.button("Check Alerts", type="primary")

if check_alerts_clicked:
    if not st.session_state["risk_alert_rules"]:
        st.warning("Add at least one alert rule.")
    else:
        _alert_watchlist = watchlist_tickers()
        _alert_rules_tuple = tuple(
            AlertRule(metric=r["metric"], operator=r["operator"], threshold=float(r["threshold"]))
            for r in st.session_state["risk_alert_rules"]
        )
        with st.spinner(f"Checking risk metrics for {len(_alert_watchlist)} watchlist ticker(s)..."):
            _alert_snapshots = compute_watchlist_snapshots(_alert_watchlist)
        _alert_triggered = evaluate_alerts(_alert_snapshots, _alert_rules_tuple)
        st.session_state["risk_alerts_state"] = {"snapshots": _alert_snapshots, "triggered": _alert_triggered}
        log_event(logger, logging.INFO, "user.risk_alerts_check", watchlist_size=len(_alert_watchlist), rule_count=len(_alert_rules_tuple), triggered=len(_alert_triggered))

_alerts_state = st.session_state.get("risk_alerts_state")
if _alerts_state:
    _alert_snapshots = _alerts_state["snapshots"]
    _alert_triggered = _alerts_state["triggered"]
    _alert_error_count = sum(1 for s in _alert_snapshots if s.status == "fetch_error")
    _alert_insufficient_count = sum(1 for s in _alert_snapshots if s.status == "insufficient_data")

    if _alert_triggered:
        _alert_tickers_hit = sorted({t.ticker for t in _alert_triggered})
        st.error(f"{len(_alert_triggered)} alert(s) triggered across {len(_alert_tickers_hit)} ticker(s): {', '.join(_alert_tickers_hit)}")
        _alert_rows = []
        for t in _alert_triggered:
            spec = RISK_ALERT_METRICS_BY_KEY[t.rule.metric]
            value_display = f"{t.value * 100:.{spec.decimals}f}%" if spec.unit == "%" else f"{t.value:.{spec.decimals}f}"
            threshold_display = f"{t.rule.threshold * 100:.{spec.decimals}f}%" if spec.unit == "%" else f"{t.rule.threshold:.{spec.decimals}f}"
            _alert_rows.append({
                "Ticker": t.ticker, "Metric": spec.label, "Current Value": value_display,
                "Condition": f"{t.rule.operator} {threshold_display}",
            })
        st.table(pd.DataFrame(_alert_rows))
    else:
        st.success(f"🟢 No alerts triggered across your {len(_alert_snapshots)}-ticker watchlist right now.")

    if _alert_error_count or _alert_insufficient_count:
        st.caption(f"{_alert_error_count} ticker(s) could not be loaded · {_alert_insufficient_count} had at least one non-computable metric — never silently excluded from the check.")
        with st.expander("Data issues detail", expanded=False):
            _alert_issue_rows = [{"Ticker": s.ticker, "Status": s.status, "Detail": s.detail} for s in _alert_snapshots if s.status != "ok"]
            st.table(pd.DataFrame(_alert_issue_rows))

# ==========================================
# REAL-TIME ALERT ENGINE
# ==========================================
# See realtime_alerts.py's module docstring for the full reasoning behind
# each of these three scope decisions (agreed with the user before this
# was built, not silent simplifications):
def _rt_md_escape_dollar(text: str) -> str:
    """AlertRule.label and EvaluationResult.detail are kept as plain text
    with genuine, unescaped "$" characters (realtime_alerts.py's own
    trigger-history table renders them as-is, not through markdown). But a
    label carrying one "$" concatenated with a detail carrying another
    "$", once both land in the SAME st.markdown/st.caption/st.toast call,
    form a matched "$...$" pair — and Streamlit's markdown treats that as
    inline LaTeX math, not literal text. Caught live: a currently-active
    price rule rendered as a raw, unparsed LaTeX box instead of "$1.00".
    Every markdown-rendering call site below escapes through this first;
    the persisted TriggerEvent.detail (st.table, JSON) never does."""
    return text.replace("$", "\\$")


st.markdown("---")
st.header("Real-Time Alert Engine")
st.caption(
    "Per-ticker rules on price, technical crossovers, and fundamental/risk thresholds, rechecked automatically "
    f"every {REALTIME_ALERTS.poll_interval_seconds}s while this tab stays open. Not a background service: closing "
    "the tab stops monitoring, same as everything else in this stateless app. Delivery is in-app only (no "
    "email/SMS/push — this app has no messaging credentials to send them with). Rules and trigger history ARE "
    "saved to a local file and survive a restart, unlike the rest of this app's session-only state; Quantix has no "
    "user accounts, so that file is shared by whoever runs this instance rather than private per login."
)

if "rt_alert_rules" not in st.session_state:
    _rt_init_rules, _rt_init_history = rt_load_store()
    st.session_state["rt_alert_rules"] = _rt_init_rules
    st.session_state["rt_alert_history"] = _rt_init_history
    st.session_state["rt_alert_prev_active"] = {}

# Every widget below is seeded into session_state ONLY if the key is
# absent, then constructed with `key=` alone — no `value=` on top of an
# already-populated key. Passing both is the same anti-pattern the
# sidebar's own ticker_input comment already documents: on the very next
# rerun (e.g. the one "+ Add Rule" itself triggers), Streamlit re-applies
# the stale `value=` over whatever the user just typed, silently reverting
# a committed ticker back to "" or a threshold back to its default. Caught
# live: the Ticker field kept resetting to empty between typing it and
# clicking Add Rule.
if "rt_new_ticker" not in st.session_state:
    st.session_state["rt_new_ticker"] = ""
if "rt_new_price_threshold" not in st.session_state:
    st.session_state["rt_new_price_threshold"] = 100.0

_rt_r1c1, _rt_r1c2 = st.columns([1, 2])
with _rt_r1c1:
    _rt_new_ticker = st.text_input("Ticker", key="rt_new_ticker", placeholder="e.g. AAPL").strip().upper()
with _rt_r1c2:
    _rt_new_type = st.selectbox(
        "Trigger Condition", RT_ALL_TRIGGER_TYPES, format_func=lambda k: RT_TRIGGER_LABELS[k], key="rt_new_type",
    )

_rt_new_threshold = None
_rt_new_metric = None
_rt_new_operator = None

if _rt_new_type in RT_PRICE_TRIGGER_TYPES:
    _rt_new_threshold = st.number_input("Price ($)", min_value=0.0, key="rt_new_price_threshold")
elif _rt_new_type == RT_FUNDAMENTAL_TRIGGER_TYPE:
    _rt_r2c1, _rt_r2c2, _rt_r2c3 = st.columns([2, 1, 1.5])
    _rt_fund_metric_options = [m.key for m in RISK_ALERT_METRICS]
    with _rt_r2c1:
        _rt_new_metric = st.selectbox(
            "Metric", _rt_fund_metric_options, format_func=lambda k: RISK_ALERT_METRICS_BY_KEY[k].label, key="rt_new_fund_metric",
        )
    with _rt_r2c2:
        _rt_new_operator = st.selectbox("Op", list(RISK_ALERT_OPERATORS.keys()), key="rt_new_fund_operator")
    with _rt_r2c3:
        # Re-seeded only when the selected METRIC actually changes (not on
        # every rerun) — this is what lets the field start at a sensible
        # per-metric default while still letting the user's own edit stick
        # across reruns once they've touched it for that metric.
        if st.session_state.get("_rt_new_fund_threshold_for_metric") != _rt_new_metric:
            st.session_state["rt_new_fund_threshold"] = float(RISK_ALERT_METRICS_BY_KEY[_rt_new_metric].default_threshold)
            st.session_state["_rt_new_fund_threshold_for_metric"] = _rt_new_metric
        _rt_new_threshold = st.number_input("Threshold", key="rt_new_fund_threshold")
else:
    st.caption(f'Uses the app\'s existing defaults — see "{RT_TRIGGER_LABELS[_rt_new_type]}" above for the exact levels/periods.')

if st.button("+ Add Rule", key="rt_add_rule_btn"):
    if not _rt_new_ticker:
        st.warning("Enter a ticker symbol first.")
    elif len(st.session_state["rt_alert_rules"]) >= REALTIME_ALERTS.max_rules:
        st.warning(f"Rule limit reached ({REALTIME_ALERTS.max_rules} max) — remove one below first.")
    else:
        _rt_rule = RealtimeAlertRule(
            id=rt_new_rule_id(), ticker=_rt_new_ticker, trigger_type=_rt_new_type,
            threshold=_rt_new_threshold, metric=_rt_new_metric, operator=_rt_new_operator,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        st.session_state["rt_alert_rules"].append(_rt_rule)
        rt_save_store(st.session_state["rt_alert_rules"], st.session_state["rt_alert_history"])
        log_event(logger, logging.INFO, "user.realtime_alert_rule_added", ticker=_rt_new_ticker, trigger_type=_rt_new_type)
        # Deliberately NOT clearing rt_new_ticker here: Streamlit forbids
        # writing to a widget's session_state key in the same run that
        # widget was already instantiated in (raises StreamlitAPIException
        # — hit this live). Clearing it would need the same deferred
        # "stash a flag, consume it before the widget renders next run"
        # trick watchlist_panel's ticker-switch uses; not worth the added
        # complexity for a convenience the sidebar's own "Add ticker" input
        # already forgoes (it doesn't clear after adding either).
        st.rerun()

if st.session_state["rt_alert_rules"]:
    st.markdown("**Active Rules**")
    _rt_remove_id = None
    for _rt_existing_rule in st.session_state["rt_alert_rules"]:
        _rt_lc1, _rt_lc2 = st.columns([6, 1])
        with _rt_lc1:
            st.caption(_rt_md_escape_dollar(_rt_existing_rule.label))
        with _rt_lc2:
            if st.button("✕", key=f"rt_remove_{_rt_existing_rule.id}", help="Remove this rule"):
                _rt_remove_id = _rt_existing_rule.id
    if _rt_remove_id is not None:
        st.session_state["rt_alert_rules"] = [r for r in st.session_state["rt_alert_rules"] if r.id != _rt_remove_id]
        st.session_state["rt_alert_prev_active"].pop(_rt_remove_id, None)
        rt_save_store(st.session_state["rt_alert_rules"], st.session_state["rt_alert_history"])
        st.rerun()


@st.fragment(run_every=REALTIME_ALERTS.poll_interval_seconds)
def _render_realtime_alerts_fragment():
    """Re-evaluates every rule on the configured interval WITHOUT rerunning
    the rest of the page (st.fragment's whole purpose) — this is the "poll
    while the tab is open" mechanism itself. Everything this function needs
    to redraw (the active/cleared banner, the data-issues expander, the
    "last checked" timestamp) is rendered from inside it so a full page
    rerun is never required just to reflect a new poll result."""
    _rt_rules = st.session_state["rt_alert_rules"]
    if not _rt_rules:
        st.caption("No active rules — add one above to start monitoring.")
        return

    _rt_results = rt_evaluate_all(_rt_rules)
    _rt_active_now = {rid: res.is_met for rid, res in _rt_results.items()}
    _rt_newly_triggered = rt_detect_new_triggers(_rt_results, st.session_state.get("rt_alert_prev_active", {}))

    if _rt_newly_triggered:
        _rt_rules_by_id = {r.id: r for r in _rt_rules}
        _rt_now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        for _rt_rid in _rt_newly_triggered:
            _rt_rule = _rt_rules_by_id.get(_rt_rid)
            if _rt_rule is None:
                continue
            _rt_result = _rt_results[_rt_rid]
            st.toast(_rt_md_escape_dollar(f"{_rt_rule.label} — {_rt_result.detail}"), icon="🔔")
            st.session_state["rt_alert_history"].append(RealtimeTriggerEvent(
                rule_id=_rt_rid, ticker=_rt_rule.ticker, trigger_type=_rt_rule.trigger_type,
                detail=_rt_result.detail, triggered_at=_rt_now_iso,
            ))
            log_event(logger, logging.INFO, "user.realtime_alert_triggered", ticker=_rt_rule.ticker, trigger_type=_rt_rule.trigger_type)
        rt_save_store(st.session_state["rt_alert_rules"], st.session_state["rt_alert_history"])

    st.session_state["rt_alert_prev_active"] = _rt_active_now

    _rt_active_rules = [r for r in _rt_rules if _rt_active_now.get(r.id)]
    if _rt_active_rules:
        st.error(f"🔴 {len(_rt_active_rules)} alert(s) currently active")
        for _rt_r in _rt_active_rules:
            st.markdown(_rt_md_escape_dollar(f"- **{_rt_r.label}** — {_rt_results[_rt_r.id].detail}"))
    else:
        st.success("🟢 No alert conditions currently met.")

    _rt_issue_rules = [r for r in _rt_rules if _rt_results[r.id].status != "ok"]
    if _rt_issue_rules:
        with st.expander(f"{len(_rt_issue_rules)} rule(s) with data issues", expanded=False):
            for _rt_r in _rt_issue_rules:
                st.caption(_rt_md_escape_dollar(f"{_rt_r.label}: {_rt_results[_rt_r.id].detail}"))

    st.caption(
        f"Checked {datetime.datetime.now().strftime('%H:%M:%S')} · "
        f"rechecking every {REALTIME_ALERTS.poll_interval_seconds}s while this tab stays open."
    )


_render_realtime_alerts_fragment()

if st.session_state["rt_alert_history"]:
    with st.expander(f"Trigger history ({len(st.session_state['rt_alert_history'])})", expanded=False):
        _rt_hist_rows = [
            {
                "When": h.triggered_at, "Ticker": h.ticker,
                "Trigger": RT_TRIGGER_LABELS.get(h.trigger_type, h.trigger_type), "Detail": h.detail,
            }
            for h in reversed(st.session_state["rt_alert_history"])
        ]
        st.table(pd.DataFrame(_rt_hist_rows))


# --- Sidebar Controls ---
# TradingView-style control rail: only Ticker/Date (used on every
# interaction) stays always-visible; everything else groups into tabs so
# the sidebar shows one focused panel at a time instead of one long
# 8-header scroll. Streamlit keeps every widget's session_state live even
# while its tab isn't the active one, so this is a pure layout change —
# every widget below has the identical label/key/default/help it always
# had, just a different container.
st.sidebar.header("Target Configuration")
# No `value=` here deliberately: this widget is keyed so the Stock Screener's
# "Open full analysis" click-through can set st.session_state["ticker_input"]
# before this line runs on the next rerun. Passing both `value=` and a `key=`
# already present in session_state is a Streamlit anti-pattern (it warns) —
# so the default is seeded into session_state once, only if absent, instead.
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = CHART_DEFAULTS.default_ticker

# Deferred ticker switch. Streamlit forbids assigning to a widget's
# session_state key AFTER that widget has been instantiated this run, so
# the sidebar Watchlist (rendered below this line) can't set
# "ticker_input" directly the way the Stock Screener can — the Screener
# sits ABOVE this line in script order, the Watchlist necessarily doesn't.
# The Watchlist instead parks its choice in "_pending_ticker" and reruns;
# this applies it on the next run, before the widget exists.
_pending_ticker = st.session_state.pop("_pending_ticker", None)
if _pending_ticker:
    st.session_state["ticker_input"] = _pending_ticker

ticker_symbol = st.sidebar.text_input("Stock Ticker", key="ticker_input").upper()

today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=CHART_DEFAULTS.default_lookback_days)
start_date = st.sidebar.date_input("Start Date", one_year_ago)
end_date = st.sidebar.date_input("End Date", today)

# ==========================================
# WATCHLIST (quick symbol switching)
# ==========================================
# Deliberately OUTSIDE the control tabs below: this is navigation, not
# configuration, and it stays visible no matter which analysis panel or
# control tab is open — the "persistent" part of the panel.
#
# Scope note: "persistent" here means across reruns within a session
# (st.session_state), which is what this app has — Quantix is a stateless
# Streamlit process with no user accounts or backing store, so the list
# resets on a fresh session. Real cross-session persistence would need
# storage this app doesn't have, same honest limitation the Smart
# Risk-Aware Alerts section already documents for itself.
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")

if "watchlist_tickers" not in st.session_state:
    st.session_state["watchlist_tickers"] = tuple(WATCHLIST_PANEL.default_tickers)

_wl_add_col, _wl_btn_col = st.sidebar.columns([3, 1])
with _wl_add_col:
    _wl_new = st.text_input(
        "Add ticker", key="watchlist_add_input", label_visibility="collapsed",
        placeholder="Add ticker",
    )
with _wl_btn_col:
    _wl_add_clicked = st.button("Add", width="stretch")

if _wl_add_clicked:
    _wl_updated, _wl_error = add_ticker(
        st.session_state["watchlist_tickers"], _wl_new, WATCHLIST_PANEL.max_tickers,
    )
    if _wl_error:
        st.sidebar.warning(_wl_error)
    else:
        st.session_state["watchlist_tickers"] = _wl_updated
        log_event(logger, logging.INFO, "user.watchlist_add", tickers=_wl_new)
        st.rerun()

_wl_tickers = st.session_state["watchlist_tickers"]
if not _wl_tickers:
    st.sidebar.caption("No tickers yet — add one above to build a quick-switch list.")
else:
    _wl_snapshots = load_quote_snapshots(_wl_tickers)
    for _wl_snap in _wl_snapshots:
        _wl_row, _wl_remove = st.sidebar.columns([5, 1])
        _wl_is_active = _wl_snap.ticker == ticker_symbol
        with _wl_row:
            if _wl_snap.status == "ok":
                _wl_label = f"{_wl_snap.direction_icon} {_wl_snap.ticker} · {_wl_snap.change_pct:+.2f}%"
            else:
                _wl_label = f"{_wl_snap.direction_icon} {_wl_snap.ticker} · n/a"
            # The active ticker gets the accent style so it's obvious which
            # row you're looking at. Deliberately NOT disabled: a disabled
            # primary button renders as a washed-out pill that reads as
            # broken, and clicking your own current ticker is a harmless
            # no-op rerun anyway.
            if st.button(
                _wl_label, key=f"wl_go_{_wl_snap.ticker}", width="stretch",
                type="primary" if _wl_is_active else "secondary",
                help=None if _wl_snap.status == "ok" else _wl_snap.detail,
            ):
                st.session_state["_pending_ticker"] = _wl_snap.ticker
                log_event(logger, logging.INFO, "user.watchlist_switch", ticker=_wl_snap.ticker)
                st.rerun()
        with _wl_remove:
            if st.button("✕", key=f"wl_rm_{_wl_snap.ticker}", help=f"Remove {_wl_snap.ticker}"):
                st.session_state["watchlist_tickers"] = remove_ticker(_wl_tickers, _wl_snap.ticker)
                st.rerun()
        _wl_current = " · current" if _wl_is_active else ""
        if _wl_snap.status == "ok":
            _wl_pe = f"P/E {_wl_snap.pe_ratio:.1f}" if _wl_snap.pe_ratio else "P/E n/a"
            st.sidebar.caption(f"${_wl_snap.price:,.2f} · {_wl_pe}{_wl_current}")
        else:
            st.sidebar.caption(f"Quote unavailable{_wl_current}")

# `side_`-prefixed so these never shadow the main-page panel variables of
# the same concern (tab_risk) defined further down — the two are different
# containers and mixing them up would render a control into the wrong place.
# NOTE: there is deliberately no "Chart" tab here — every chart indicator
# control now lives in the Chart Workspace panel itself, directly above the
# chart it configures (see the "Chart Indicators & Overlays" expander), so
# the controls and their effect are in one place instead of split across
# the sidebar and the page.
side_valuation, side_risk, side_portfolio, side_system = st.sidebar.tabs(
    ["Valuation", "Risk", "Portfolio", "System"]
)

with side_valuation:
    benchmark_symbol = st.text_input("Market Benchmark", CHART_DEFAULTS.default_benchmark).upper()
    dcf_growth = st.slider("Expected FCF Growth (%)", min_value=CHART_DEFAULTS.dcf_growth_range_pct[0], max_value=CHART_DEFAULTS.dcf_growth_range_pct[1], value=CHART_DEFAULTS.dcf_growth_default_pct) / 100
    dcf_wacc = st.slider("Discount Rate / WACC (%)", min_value=CHART_DEFAULTS.dcf_wacc_range_pct[0], max_value=CHART_DEFAULTS.dcf_wacc_range_pct[1], value=CHART_DEFAULTS.dcf_wacc_default_pct) / 100

with side_risk:
    vol_window = st.slider("Volatility Window (days)", min_value=CHART_DEFAULTS.vol_window_range[0], max_value=CHART_DEFAULTS.vol_window_range[1], value=CHART_DEFAULTS.vol_window_default)
    var_confidence = st.selectbox("VaR Confidence Level", options=RISK.var_confidence_levels, index=list(RISK.var_confidence_levels).index(RISK.var_confidence_default), format_func=lambda c: f"{c:.0%}")
    var_lookback = st.slider("VaR Lookback (days)", min_value=CHART_DEFAULTS.var_lookback_range[0], max_value=CHART_DEFAULTS.var_lookback_range[1], value=CHART_DEFAULTS.var_lookback_default)
    risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=CHART_DEFAULTS.risk_free_rate_range_pct[0], max_value=CHART_DEFAULTS.risk_free_rate_range_pct[1], value=RISK.risk_free_rate * 100, step=0.25, help="Feeds the Sharpe/Sortino ratios below. The DCF's CAPM cost of equity uses its own fixed risk-free-rate assumption (RISK.risk_free_rate), unaffected by this slider.") / 100

with side_portfolio:
    portfolio_basket_input = st.text_input("Correlation Basket (comma-separated)", value=CHART_DEFAULTS.portfolio_default_basket, help="Tickers to include in the Portfolio Correlation & Diversification section further down the page. The main Stock Ticker above is always included automatically.")
    portfolio_lookback = st.slider("Portfolio Lookback (days)", min_value=CHART_DEFAULTS.portfolio_lookback_range[0], max_value=CHART_DEFAULTS.portfolio_lookback_range[1], value=CHART_DEFAULTS.portfolio_lookback_default)

with side_system:
    st.caption("Diagnostics")
    # Rendered BEFORE the Force Refresh button below on purpose. Streamlit
    # discards the session_state entry for any keyed widget that isn't rendered
    # during a run — and Force Refresh calls st.rerun(), which aborts the script
    # immediately. If this checkbox came after it, ticking Debug then hitting
    # Force Refresh would silently reset the toggle back to off.
    debug_mode = st.checkbox(
        "Debug logging", key="debug_mode",
        help="Raise the log level to DEBUG and show recent log entries in-page.",
    )
    st.caption(f"Log file: {log_file_path().name}")

    # Re-apply the level now that the checkbox's value for THIS run is known. The
    # call at the top of the script runs before this widget exists, so on the run
    # where the box is first ticked it would otherwise still be at INFO. Handler
    # setup is idempotent, so this only adjusts the level — and everything that
    # actually loads data happens below this point.
    setup_logging(logging.DEBUG if debug_mode else logging.INFO)
    log_event(logger, logging.DEBUG, "logging.level",
              level=logging.getLevelName(get_logger("data_loader").getEffectiveLevel()))

    st.markdown("---")
    st.caption("Data Cache")
    st.caption("Quotes: 30 min · Prices: 1 hr · Statements: 24 hr · Ownership: 12 hr")
    if st.button("Force Refresh Data", help="Bypass all cached data and refetch everything from Yahoo Finance now."):
        log_event(logger, logging.INFO, "user.force_refresh", ticker=ticker_symbol)
        clear_all_caches()
        st.rerun()

# Record meaningful input changes only (see log_input_changes docstring).
log_input_changes(
    ticker=ticker_symbol, start=str(start_date), end=str(end_date),
    benchmark=benchmark_symbol,
)

# --- Data Fetching ---
# All Yahoo Finance access for the selected ticker + macro context happens
# once here via data_loader, and the results are reused by every section
# below instead of each section re-fetching independently.
with st.spinner(f"Running deep audit on {ticker_symbol} & loading Macro Data..."):
    ticker_bundle = load_ticker_bundle(ticker_symbol, start_date, end_date, deep=True)
    # Every technical indicator below (SMA, RSI, and future MACD/Bollinger/
    # ATR) is computed on this processed frame, never the raw fetch directly
    # — see price_processing.py for exactly what's validated/cleaned.
    price_processing_result = process_price_data(ticker_bundle.price_history, ticker=ticker_symbol)
    df = price_processing_result.df
    if price_processing_result.issue_count:
        log_event(
            logger, logging.WARNING, "price_data.issues", ticker=ticker_symbol,
            duplicates_removed=price_processing_result.duplicate_rows_removed,
            invalid_rows_removed=price_processing_result.invalid_rows_removed,
            possible_gaps=len(price_processing_result.possible_gaps),
        )

    macro_bundle = load_macro_bundle(benchmark_symbol, start_date, end_date)
    bench_df = macro_bundle.benchmark_history
    vix_df = macro_bundle.vix_history
    tnx_df = macro_bundle.tnx_history

    # Canonical, unit-consistent view of this ticker's data (statements +
    # info-dict ratios). Every section below reads from this instead of
    # touching raw info/statement fields directly, so units, field names,
    # and missing-value handling are consistent across the whole app.
    standardized = standardize_financials(ticker_bundle)

# ==========================================
# SYMBOL HEADER (fill) — renders into the sticky slot reserved at the very
# top of the page, so it's on screen from first paint and stays pinned
# while scrolling through any panel. Fills even when df is empty, since
# knowing WHICH symbol failed to load is exactly when the header matters.
# ==========================================
with symbol_header_container:
    # Price/day-change come from the LIVE QUOTE (the same load_quote_snapshots
    # the sidebar watchlist uses), NOT from standardized.current_price.
    # That distinction is load-bearing, not incidental: standardized's price
    # is the last bar of the fetched price HISTORY, and that history lags the
    # live quote by a session — during market hours (and before the daily bar
    # lands) its "latest" close is yesterday's. Deriving a day change from it
    # therefore reports the PREVIOUS session's move as if it were today's,
    # and disagrees with the watchlist row for the same symbol. Observed
    # directly: history's last two bars gave AAPL -1.53% (Aug 7 -> Aug 10)
    # while the live quote gave -0.37% (Aug 10 close -> now).
    #
    # The analysis below deliberately still runs on the price history — the
    # DCF's "Market Price", ATR stop, and every chart are all built on that
    # one validated dataset. So this bar is the live quote and those are the
    # dataset's close; they can differ by a fraction of a percent intraday,
    # which is correct for what each is measuring.
    _hdr_quote = load_quote_snapshots((ticker_symbol,))[0]
    _hdr_name = standardized.long_name or ""

    if _hdr_quote.status == "ok":
        _hdr_change = _hdr_quote.price - _hdr_quote.previous_close
        _hdr_class = "qsh-up" if _hdr_change > 0 else ("qsh-down" if _hdr_change < 0 else "qsh-flat")
        _hdr_price_html = f'<span class="qsh-price">${_hdr_quote.price:,.2f}</span>'
        _hdr_change_html = (
            f'<span class="qsh-change {_hdr_class}">{_hdr_change:+,.2f} ({_hdr_quote.change_pct:+.2f}%)</span>'
        )
    elif _hdr_quote.price is not None or standardized.current_price is not None:
        # A price is known but there's no usable prior close to compare it
        # against — show the price and say so, rather than implying 0.00%.
        _hdr_fallback_price = _hdr_quote.price if _hdr_quote.price is not None else standardized.current_price
        _hdr_price_html = f'<span class="qsh-price">${_hdr_fallback_price:,.2f}</span>'
        _hdr_change_html = '<span class="qsh-change qsh-flat">day change unavailable</span>'
    else:
        _hdr_price_html = '<span class="qsh-price">—</span>'
        _hdr_change_html = '<span class="qsh-change qsh-flat">price unavailable</span>'

    _hdr_meta_bits = [b for b in (standardized.sector, ticker_bundle.info.get("currency")) if b]
    _hdr_meta = f'<span class="qsh-meta">{" · ".join(_hdr_meta_bits)}</span>' if _hdr_meta_bits else ""

    st.markdown(
        f'<div class="quantix-symbol-header">'
        f'<span class="qsh-ticker">{ticker_symbol}</span>'
        f'<span class="qsh-name">{_hdr_name}</span>'
        f'{_hdr_price_html}{_hdr_change_html}{_hdr_meta}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Recently viewed: one-click switching between symbols visited
    # this session. Rendered INSIDE symbol_header_container so it's part
    # of the same sticky block as the header above — the symbol you're on
    # and the symbols you can jump to stay together and stay on screen.
    #
    # Recorded here rather than at the sidebar input so it captures the
    # ticker actually analysed on this run, however it was chosen (typed,
    # watchlist click, screener click-through, or one of these chips).
    # record_recent() is idempotent, which matters because Streamlit
    # re-runs this whole script on every widget interaction.
    st.session_state["recent_tickers"] = record_recent(
        st.session_state.get("recent_tickers", ()), ticker_symbol,
        WATCHLIST_PANEL.max_recent_tickers,
    )
    _recents = st.session_state["recent_tickers"]

    if len(_recents) > 1:
        # Fixed-width chips: the columns list is padded to the configured
        # maximum with a trailing spacer, so a chip is the same size
        # whether two symbols have been visited or eight — rather than
        # two chips stretching across the whole page.
        _chip_cols = st.columns(
            [1] * len(_recents) + [max(1, WATCHLIST_PANEL.max_recent_tickers - len(_recents) + 1)]
        )
        for _chip_col, _recent in zip(_chip_cols, _recents):
            with _chip_col:
                _is_current = _recent == ticker_symbol
                if st.button(
                    _recent, key=f"recent_{_recent}", width="stretch",
                    type="primary" if _is_current else "secondary",
                    help="Currently analysed" if _is_current else f"Switch analysis to {_recent}",
                ):
                    st.session_state["_pending_ticker"] = _recent
                    log_event(logger, logging.INFO, "user.recent_switch", ticker=_recent)
                    st.rerun()

if df.empty:
    detail = " ".join(ticker_bundle.errors) if ticker_bundle.errors else "No data returned by Yahoo Finance."
    log_event(logger, logging.ERROR, "analysis.aborted", ticker=ticker_symbol, reason=detail)
    st.error(f"No reliable data found for '{ticker_symbol}'. {detail} Try again shortly or check the ticker symbol.")
else:
    # Top-level panel navigation. Every section below keeps its EXACT
    # original execution order (nothing is reordered) — only which
    # visual container each block renders into changes, exactly the same
    # trick executive_digest_container (right below) already uses: a tab
    # is a valid `with` context manager, so a block written later in the
    # script can still target a tab defined earlier, same as containers.
    # Chart Workspace sits second, right after Overview: it's the primary
    # charting surface, so it gets a prominent position rather than being
    # buried mid-list among the analysis panels.
    tab_overview, tab_chart_workspace, tab_fundamentals, tab_risk, tab_simulation, tab_smart_money, tab_tearsheet = st.tabs([
        "Overview", "Chart Workspace", "Fundamentals & Valuation", "Risk & Technicals",
        "Monte Carlo & Seasonality", "Smart Money & Peers", "CIO Tear Sheet",
    ])


    # ==========================================
    # EXECUTIVE DIGEST (placeholder — filled in after the DCF section below,
    # once every source signal it synthesizes has been computed, but
    # rendered HERE at the top via Streamlit's container-as-placeholder
    # pattern: content written into a container later in the script still
    # appears at the container's position in the page, not at the point in
    # execution where it was written. See the "EXECUTIVE DIGEST (fill)"
    # block after the DCF section for what actually goes in this slot.)
    # ==========================================
    with tab_overview:
        executive_digest_container = st.container()

        # ==========================================
        # DATA QUALITY REPORT
        # ==========================================
        # Combines field-level statement completeness (financial_validation.py),
        # data freshness (most recent reported quarter), and fetch reliability
        # (data_loader.py retries/warnings) into one score, run before any ratio
        # below is calculated — so it's clear up front how much to trust the
        # analysis instead of piecing it together from separate panels.
        quality = assess_data_quality(standardized, ticker_bundle, macro_bundle)

        st.subheader(f"{quality.grade_icon} Data Quality Report — {quality.score}/100 ({quality.grade})")
        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Required Fields", f"{quality.required_completeness_pct:.0f}%", help="% of required balance sheet / income statement / cash flow fields present.")
        dq2.metric("Optional Fields", f"{quality.optional_completeness_pct:.0f}%", help="% of optional statement fields present (e.g. Retained Earnings, Interest Expense).")
        freshness_label = "N/A" if quality.staleness_days is None else f"{quality.staleness_days}d old"
        dq3.metric("Data Freshness", freshness_label, delta="Stale" if quality.is_stale else "Fresh", delta_color="inverse" if quality.is_stale else "normal", help="Age of the most recently reported quarter. Flagged stale beyond 120 days.")
        dq4.metric("Fetch Reliability", f"{quality.fetch_reliability_score:.0f}%", help="Penalized for retried/failed downloads and empty optional datasets.")

        if quality.grade in ("Poor", "Fair"):
            st.warning(f"Data quality is {quality.grade.lower()} for {ticker_symbol} — treat derived metrics with extra caution and check the detail below.")

        detail_issue_count = len(quality.missing_required_fields) + len(quality.missing_optional_fields) + len(quality.fetch_warnings) + len(quality.fetch_errors)
        with st.expander(f"Data Quality Detail ({detail_issue_count} issue(s))", expanded=quality.grade in ("Poor", "Fair")):
            for stmt in standardized.validation.statements:
                status = "🟢 Complete" if stmt.is_valid else f"🔴 {len(stmt.missing_required)} required field(s) missing"
                st.markdown(f"**{stmt.statement_name}** — {status}")
                for check in stmt.checks:
                    icon = "🟢" if check.present else ("🔴" if check.required else "⚪")
                    label = check.name + (" (required)" if check.required else " (optional)")
                    st.markdown(f"&nbsp;&nbsp;{icon} {label}")

            if quality.most_recent_quarter is not None:
                st.markdown(f"**Freshness** — most recent reported quarter: {quality.most_recent_quarter.strftime('%B %d, %Y')} ({quality.staleness_days} days ago)")
            else:
                st.markdown("**Freshness** — most recent quarter date not reported by Yahoo Finance; freshness could not be verified.")

            if quality.fetch_errors or quality.fetch_warnings:
                st.markdown("**Fetch Reliability**")
                for w in quality.fetch_errors:
                    st.error(w)
                for w in quality.fetch_warnings:
                    st.warning(w)

        # ==========================================
        # NEW: MACRO REGIME FILTER
        # ==========================================
        st.header("Macro Regime & Systemic Risk", anchor="macro-regime")
        vix_current = vix_df['Close'].iloc[-1] if not vix_df.empty else 20.0
        tnx_current = tnx_df['Close'].iloc[-1] if not tnx_df.empty else 4.0
        macro_risk_flag = vix_current > RISK.vix_high_risk_threshold

        m1, m2 = st.columns(2)
        m1.metric("VIX (Fear Index)", f"{vix_current:.2f}", delta=f"High Risk (>{RISK.vix_high_risk_threshold:.0f})" if macro_risk_flag else "Stable Market", delta_color="inverse" if macro_risk_flag else "normal")
        m2.metric("10-Year Treasury Yield", f"{tnx_current:.2f}%")

        if macro_risk_flag:
            st.error("SYSTEMIC RISK WARNING: High VIX detected. The broader market is experiencing fear. Position sizing will be automatically penalized to protect capital.")
        st.markdown("---")

    # ==========================================
    # DATA EXTRACTION & CALCULATIONS
    # ==========================================
    # Every statement-derived ratio is calculated by the Fundamental Analysis
    # Engine — this file renders the results and performs no ratio arithmetic
    # of its own. Metrics that can't be computed come back as None and render
    # as "N/A" rather than a fabricated default.
    fundamentals_engine = FundamentalAnalysisEngine(standardized, raw_info=ticker_bundle.info)
    fundamentals = fundamentals_engine.analyze()

    # Sector-relative standing — computed once here, displayed alongside
    # (never instead of) the fixed-threshold Scorecard/Master Matrix rows
    # and the Quality Classification's Return on Equity metric below. None
    # when the sector is unknown or too few configured-universe peers share
    # it (see sector_percentile.py's MIN_PEERS guard) — an honest "N/A",
    # never a percentile computed from 1-2 companies.
    sector_percentiles = compute_sector_percentiles(standardized)

    net_margin = fundamentals.net_margin
    de_ratio = fundamentals.debt_to_equity
    pe_ratio = fundamentals.pe_ratio
    peg_ratio = fundamentals.peg_ratio
    beta = fundamentals.beta
    roic_val = fundamentals.roic_pct
    int_cov_val = fundamentals.interest_coverage
    fcf_yield_val = fundamentals.fcf_yield_pct
    fcf_raw = standardized.free_cash_flow  # Extracted for DCF later

    # ==========================================
    # FINANCIAL METRICS VALIDATION REPORT
    # ==========================================
    # One consolidated overview across every metric the engine validates —
    # Profitability, Liquidity, Leverage, Valuation — plus two checks that
    # don't belong to any single category: outliers (a value whose magnitude
    # exceeds a configured sanity bound, config.OUTLIER_BOUNDS, independent of
    # whether it agrees with Yahoo) and incomplete calculations (a fallback
    # data source or assumption was used). This does not replace the four
    # detailed category reports further down — it's the "check here first"
    # summary; scroll down to any category for the full breakdown.
    mv = fundamentals.metrics_validation
    with tab_fundamentals:
        st.markdown("---")
        st.header("Financial Metrics Validation Report")

        if mv.is_clean:
            st.success(f"🟢 No issues found across {len(mv.evaluated_checks)} evaluated metric(s) for {ticker_symbol}.")
        else:
            issue_parts = []
            if mv.disagreement_count:
                issue_parts.append(f"{mv.disagreement_count} disagree with Yahoo's own figure")
            if mv.outlier_count:
                issue_parts.append(f"{mv.outlier_count} exceed a sanity bound")
            if mv.fallback_count:
                issue_parts.append(f"{mv.fallback_count} used a fallback data source")
            st.warning(f"{mv.total_issues} issue(s) found across {len(mv.evaluated_checks)} evaluated metric(s) for {ticker_symbol}: " + "; ".join(issue_parts) + ".")

        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Metrics Evaluated", f"{len(mv.evaluated_checks)} / {len(mv.checks)}")
        vc2.metric("Yahoo Disagreements", mv.disagreement_count)
        vc3.metric("Extreme Outliers", mv.outlier_count)
        vc4.metric("Incomplete Calculations", mv.fallback_count)

        if mv.outliers:
            with st.expander(f"{mv.outlier_count} extreme outlier(s)", expanded=True):
                for o in mv.outliers:
                    st.error(f"**[{o.category}] {o.label}**: {o.display} — {o.note}")

        if mv.fallback_notes:
            with st.expander(f"{mv.fallback_count} incomplete calculation(s)", expanded=False):
                for note in mv.fallback_notes:
                    st.info(note)

        if mv.disagreements:
            with st.expander(f"{mv.disagreement_count} disagreement(s) with Yahoo's own reported figure", expanded=False):
                disagreement_data = {
                    "Category": [cat for cat, _ in mv.disagreements],
                    "Metric": [c.label for _, c in mv.disagreements],
                    "Computed": [c.computed_display for _, c in mv.disagreements],
                    "Yahoo Reported": [c.reference_display for _, c in mv.disagreements],
                }
                st.table(pd.DataFrame(disagreement_data))
                st.caption("See the category-specific reports below for the likely cause of each disagreement.")

        # ==========================================
        # SCOREBOARD
        # ==========================================
        # Flags and score come straight from the engine's evaluated checks. Both
        # are sector-aware: Debt-to-Equity uses a looser threshold for Financial
        # Services companies, and a metric with no computable value for this
        # company (common for banks) is excluded from the score entirely rather
        # than counted as a failure — so `total_checks` can be less than the 8
        # possible scorecard metrics, and the Blueprint Alignment % is a weighted
        # score over the evaluable ones (core health metrics count for more than
        # valuation/volatility — see config.SCORECARD.weights).
        green_flags = fundamentals.green_flags
        total_checks = fundamentals.total_checks
        possible_checks = len(fundamentals.scorecard_checks)
        score_pct = fundamentals.score_pct

        st.header("Strategic Investment Scorecard", anchor="scorecard")
        sector_note = f"Sector: {standardized.sector}" if standardized.sector else "Sector: Unknown"
        if standardized.sector in SCORECARD.financials_sector_names:
            sector_note += f" — Debt-to-Equity benchmark relaxed to < {SCORECARD.financials_max_debt_to_equity} (banks are structurally more leveraged as a business model)"
        st.caption(sector_note)

        c1, c2, c3 = st.columns(3)
        c1.metric("Institutional Green Flags", f"{green_flags} / {total_checks}")
        c2.metric("Operational Warning Signs", f"{total_checks - green_flags}")
        c3.metric("Blueprint Alignment", f"{score_pct:.0f}%", help="Weighted over evaluable metrics — see the sector/weighting note above.")

        if total_checks < possible_checks:
            st.caption(f"{possible_checks - total_checks} of {possible_checks} scorecard metric(s) not computable for {ticker_symbol} and excluded from scoring, rather than counted as a failure.")

        if fundamentals.alignment_verdict == "high": st.success("🟢 HIGH ALIGNMENT: Passes major filters.")
        elif fundamentals.alignment_verdict == "moderate": st.warning("🟡 MODERATE RISK: Proceed with caution.")
        else: st.error("🔴 ABORT RESEARCH: Fails safety benchmarks.")

        # ==========================================
        # COMPANY QUALITY CLASSIFICATION
        # ==========================================
        # A complementary, differently-framed view from the Scorecard above: five
        # weighted factors (Profitability, Financial Stability, Growth, Valuation,
        # Capital Efficiency) blended into one 0-100 quality score and category,
        # instead of a flat pass/fail checklist. Valuation deliberately does NOT
        # reward cheapness here — it scores how close each multiple sits to a
        # "reasonably priced" center point, since excellent businesses often
        # justly trade at a premium (standard quality-investing methodology
        # excludes valuation from "quality" for exactly this reason).
        cq = fundamentals.company_quality
        st.markdown("---")
        st.header("Company Quality Classification", anchor="quality-classification")

        if cq.overall_score is None:
            st.warning(f"⚪ Not enough data to classify {ticker_symbol}'s quality — every factor was missing all of its inputs.")
        else:
            qc1, qc2 = st.columns([1, 2])
            qc1.metric("Overall Quality Score", f"{cq.overall_score:.0f} / 100", help="Weighted average across evaluable factors — see config.QUALITY for every band and weight.")
            with qc2:
                st.markdown(f"### {cq.category_icon} {cq.category}")
                if len(cq.evaluable_factors) < len(cq.factors):
                    st.caption(f"{len(cq.evaluable_factors)} of {len(cq.factors)} factors had computable data for {ticker_symbol}; the rest were excluded rather than scored as 0.")

            factor_cols = st.columns(len(cq.factors))
            for col, factor in zip(factor_cols, cq.factors):
                with col:
                    if factor.score is None:
                        st.metric(factor.name, "N/A", help=f"Weight: {factor.weight:.0%}. No computable inputs for {ticker_symbol}.")
                    else:
                        st.metric(factor.name, f"{factor.score:.0f}", help=f"Weight: {factor.weight:.0%} of the overall score.")
                        st.progress(min(max(factor.score / 100, 0.0), 1.0))

            # Sector-relative ROE — a separate small callout rather than a new
            # column on the Capital Efficiency factor's table above, since that
            # table's "Return on Equity" row uses the statement-computed figure
            # (roe_pct_computed()) while peers here are only ever shallow-fetched,
            # so the percentile is against Yahoo's own reported ROE — a related
            # but distinct number, labeled explicitly so the two are never
            # conflated.
            if sector_percentiles is not None:
                if sector_percentiles.percentiles.get("roe") is not None:
                    st.caption(f"Sector-relative standing: Return on Equity (Yahoo-reported, {sector_percentiles.target_values['roe']*100:.1f}%) ranks at the {format_percentile(sector_percentiles.percentiles['roe'])} among {sector_percentiles.peer_count} {sector_percentiles.sector} peers.")
                else:
                    st.caption(f"Sector-relative Return on Equity standing unavailable — too few {sector_percentiles.sector} peers report this specific metric.")

        with st.expander("Quality methodology & full metric breakdown", expanded=False):
            st.caption(
                "Each metric scores 0-100 against a configured band (config.QUALITY), then a factor is the average "
                "of its evaluable metrics, and the overall score is a weighted average of evaluable factors. A metric "
                "or factor with no computable data is excluded rather than scored as 0. Every band/weight is a "
                "disclosed judgment call, not derived from a live external quality-rating source."
            )
            for factor in cq.factors:
                score_label = "N/A" if factor.score is None else f"{factor.score:.0f} / 100"
                st.markdown(f"**{factor.name}** (weight {factor.weight:.0%}) — {score_label}")
                factor_data = {
                    "Metric": [met.label for met in factor.metrics],
                    "Value": [met.display for met in factor.metrics],
                    "Sub-Score": ["N/A" if met.sub_score is None else f"{met.sub_score:.0f}" for met in factor.metrics],
                }
                st.table(pd.DataFrame(factor_data))
            st.caption("Asset Turnover (Capital Efficiency) uses one global band and is naturally sector-dependent — asset-heavy businesses like banks score low here structurally, not necessarily because of poor capital discipline.")

        # ==========================================
        # STEP 1: QUALITATIVE AUDIT
        # ==========================================
        st.markdown("---")
        st.header("Step 1: The Qualitative Business Audit")
        with st.expander("Run The 2-Sentence Revenue Test", expanded=True):
            st.subheader("Business Summary")
            st.write(standardized.business_summary or 'Description not found.')
            st.text_area("Your 2-Sentence Test (How do they make money?):")

        # ==========================================
        # STEP 2-5: THE MASTER MATRIX
        # ==========================================
        st.header("The Comprehensive Pass / Fail Master Matrix")

        # Rows are generated from the engine's evaluated checks, so adding a new
        # ratio in fundamental_analysis.py surfaces here automatically.
        matrix_rows = fundamentals.matrix_checks
        if sector_percentiles is not None:
            st.caption(f"Sector Percentile column ranks against {sector_percentiles.peer_count} {sector_percentiles.sector} peers from Quantix's configured watchlist/peer universe: {', '.join(sector_percentiles.peer_tickers)}.")
        else:
            reason = "sector unknown" if not standardized.sector else f"fewer than {SECTOR_MIN_PEERS} same-sector peers in the configured universe"
            st.caption(f"Sector Percentile column unavailable ({reason} for {ticker_symbol}) — never shown as a fabricated rank.")
        matrix_data = {
            "Category": [c.category for c in matrix_rows],
            "Metric": [c.label for c in matrix_rows],
            "Current Value": [c.display for c in matrix_rows],
            "Blueprint Benchmark": [c.benchmark for c in matrix_rows],
            "Status": [c.status_icon for c in matrix_rows],
            "Sector Percentile": [format_percentile(sector_percentiles.percentiles.get(c.key)) if sector_percentiles else "N/A" for c in matrix_rows],
        }
        matrix_df = pd.DataFrame(matrix_data)
        st.table(matrix_df)
        st.download_button(
            "Download Scorecard (CSV)",
            data=matrix_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker_symbol}_scorecard_{datetime.date.today().isoformat()}.csv",
            mime="text/csv",
            help="The Comprehensive Pass/Fail Master Matrix exactly as shown above.",
        )

        # ==========================================
        # PROFITABILITY VALIDATION REPORT
        # ==========================================
        # Every profitability ratio is independently computed from raw statement
        # data and cross-checked against Yahoo's own separately-reported ratio
        # for the same concept — the practical substitute here for reconciling
        # against a real annual report (no live 10-K access in this environment).
        # A 🟡 does not necessarily mean our formula is wrong: Yahoo's figure is
        # often trailing-twelve-month while ours uses the most recent annual
        # period, and that timing difference alone can exceed the tolerance.
        st.markdown("---")
        st.header("Profitability Validation Report")
        st.caption("Formula vs. Yahoo Finance's own reported ratio for the same concept · 🟢 agrees (within 15%) · 🟡 disagrees · ⚪ no independent reference / not applicable for this company")

        prof_rows = fundamentals.profitability_checks
        profitability_data = {
            "Metric": [c.label for c in prof_rows],
            "Formula": [c.formula for c in prof_rows],
            "Computed": [c.computed_display for c in prof_rows],
            "Yahoo Reported": [c.reference_display for c in prof_rows],
            "Status": [c.status_icon for c in prof_rows],
        }
        st.table(pd.DataFrame(profitability_data))

        disagreements = [c for c in prof_rows if c.agrees is False]
        if disagreements:
            with st.expander(f"{len(disagreements)} metric(s) diverge from Yahoo's reported figure", expanded=False):
                for c in disagreements:
                    st.info(
                        f"**{c.label}**: computed {c.computed_display} vs. Yahoo's {c.reference_display}. "
                        "Likely cause: Yahoo's ratio is typically trailing-twelve-month, while this figure uses "
                        "the most recently reported annual period — a timing/basis difference, not necessarily a formula error."
                    )

        not_applicable = [c for c in prof_rows if c.agrees is None and c.computed_pct is None]
        if not_applicable:
            st.caption(
                "Not computable for " + ticker_symbol + ": " +
                ", ".join(c.label for c in not_applicable) +
                " — the required statement line item isn't reported (common for banks/financials, which don't report cost of revenue)."
            )

        # ==========================================
        # LIQUIDITY VALIDATION REPORT
        # ==========================================
        # Current Ratio and Quick Ratio, independently computed from the balance
        # sheet and cross-checked against Yahoo's own separately-reported ratio.
        # Informational only — the Current Ratio shown in the Master Matrix above
        # stays sourced from Yahoo directly; this report exists purely to verify
        # that figure and to surface Quick Ratio, which has no scorecard flag.
        st.markdown("---")
        st.header("Liquidity Validation Report")
        st.caption("Formula vs. Yahoo Finance's own reported ratio for the same concept · 🟢 agrees (within 15%) · 🟡 disagrees · ⚪ no independent reference / not applicable for this company")

        liq_rows = fundamentals.liquidity_checks
        liquidity_data = {
            "Metric": [c.label for c in liq_rows],
            "Formula": [c.formula for c in liq_rows],
            "Computed": [c.computed_display for c in liq_rows],
            "Yahoo Reported": [c.reference_display for c in liq_rows],
            "Status": [c.status_icon for c in liq_rows],
        }
        st.table(pd.DataFrame(liquidity_data))

        liq_disagreements = [c for c in liq_rows if c.agrees is False]
        if liq_disagreements:
            with st.expander(f"{len(liq_disagreements)} metric(s) diverge from Yahoo's reported figure", expanded=False):
                for c in liq_disagreements:
                    st.info(
                        f"**{c.label}**: computed {c.computed_display} vs. Yahoo's {c.reference_display}. "
                        "Likely cause: Yahoo's ratio is typically trailing-twelve-month, while this figure uses "
                        "the most recently reported annual period — a timing/basis difference, not necessarily a formula error."
                    )

        liq_not_applicable = [c for c in liq_rows if c.agrees is None and c.computed_pct is None]
        if liq_not_applicable:
            st.caption(
                "Not computable for " + ticker_symbol + ": " +
                ", ".join(c.label for c in liq_not_applicable) +
                " — Current Assets, Current Liabilities, or Inventory isn't reported for this company (common for banks, which don't file a classified balance sheet)."
            )

        # ==========================================
        # LEVERAGE VALIDATION REPORT
        # ==========================================
        # Debt-to-Equity here IS the value shown in the Master Matrix above —
        # unlike Current Ratio, it's statement-computed (Total Debt / Stockholders
        # Equity) rather than a Yahoo passthrough, specifically because Yahoo's
        # debtToEquity field has been observed at inconsistent scales (ratio vs.
        # percent) across tickers. This report cross-checks that computed value
        # against Yahoo's own figure, and separately verifies Total Debt itself by
        # comparing its two independent Yahoo sources (balance sheet vs. info
        # dict) side by side, since the app silently prefers one over the other
        # everywhere else.
        st.markdown("---")
        st.header("Leverage Validation Report")
        st.caption("Formula vs. Yahoo Finance's own reported figure for the same concept · 🟢 agrees (within 15%) · 🟡 disagrees · ⚪ no independent reference / not applicable for this company")

        lev_rows = fundamentals.leverage_checks
        leverage_data = {
            "Metric": [c.label for c in lev_rows],
            "Formula": [c.formula for c in lev_rows],
            "Computed": [c.computed_display for c in lev_rows],
            "Yahoo Reported": [c.reference_display for c in lev_rows],
            "Status": [c.status_icon for c in lev_rows],
        }
        st.table(pd.DataFrame(leverage_data))

        lev_disagreements = [c for c in lev_rows if c.agrees is False]
        if lev_disagreements:
            with st.expander(f"{len(lev_disagreements)} metric(s) diverge from Yahoo's reported figure", expanded=False):
                for c in lev_disagreements:
                    if c.key == "total_debt":
                        st.info(
                            f"**{c.label}**: balance sheet reports {c.computed_display}, Yahoo's info feed reports {c.reference_display}. "
                            "Likely cause: for some companies (especially banks/financials) these two Yahoo sources use different underlying "
                            "definitions of \"debt\" — not a timing issue. The balance sheet figure is what the rest of the app uses."
                        )
                    else:
                        st.info(
                            f"**{c.label}**: computed {c.computed_display} vs. Yahoo's {c.reference_display}. "
                            "Likely cause: Yahoo's debtToEquity field has been observed at inconsistent scales (ratio vs. percent) "
                            "across tickers — the statement-computed figure above is what the rest of the app uses."
                        )

        lev_not_applicable = [c for c in lev_rows if c.agrees is None]
        if lev_not_applicable:
            st.caption(
                "No independent Yahoo reference for " + ticker_symbol + ": " +
                ", ".join(c.label for c in lev_not_applicable) +
                " — either Yahoo doesn't report an equivalent field (Interest Coverage), or this company doesn't report it (missing Stockholders Equity/Total Debt)."
            )

        # ==========================================
        # VALUATION VALIDATION REPORT
        # ==========================================
        # P/E and Price-to-Book stay Yahoo-sourced as the canonical Master Matrix
        # values (Yahoo showed no known scale/unit bug here, unlike Debt-to-Equity)
        # — this report cross-checks them only. EV/EBITDA is a brand-new metric
        # with no prior canonical value in the app.
        st.markdown("---")
        st.header("Valuation Validation Report")
        st.caption("Formula vs. Yahoo Finance's own reported figure for the same concept · 🟢 agrees (within 15%) · 🟡 disagrees · ⚪ no independent reference / not applicable for this company")

        val_rows = fundamentals.valuation_checks
        valuation_data = {
            "Metric": [c.label for c in val_rows],
            "Formula": [c.formula for c in val_rows],
            "Computed": [c.computed_display for c in val_rows],
            "Yahoo Reported": [c.reference_display for c in val_rows],
            "Status": [c.status_icon for c in val_rows],
        }
        st.table(pd.DataFrame(valuation_data))

        val_disagreement_reasons = {
            "price_to_book": "Yahoo's own Price-to-Book appears to use a different book-value basis (e.g. a separately reported/stale book value per share) than the Stockholders Equity line item used here — the ROE cross-check above agreeing with Yahoo suggests the equity figure itself is correct.",
            "peg_ratio": "Yahoo's pegRatio is typically based on forward-looking multi-year analyst growth estimates, while the figure above uses trailing earnings/revenue growth — a genuine definitional difference, not a formula error (Yahoo's own field is also frequently unavailable/deprecated).",
            "ev_ebitda": "This inherits the Total Debt disagreement from the Leverage Validation Report above — Enterprise Value here is built from the balance-sheet Total Debt figure, which can differ from whatever total debt Yahoo used to compute its own enterpriseValue.",
        }
        val_disagreements = [c for c in val_rows if c.agrees is False]
        if val_disagreements:
            with st.expander(f"{len(val_disagreements)} metric(s) diverge from Yahoo's reported figure", expanded=False):
                for c in val_disagreements:
                    reason = val_disagreement_reasons.get(c.key, "Likely a timing/basis difference between this figure and Yahoo's own calculation, not necessarily a formula error.")
                    st.info(f"**{c.label}**: computed {c.computed_display} vs. Yahoo's {c.reference_display}. {reason}")

        val_not_applicable = [c for c in val_rows if c.agrees is None]
        if val_not_applicable:
            st.caption(
                "No independent Yahoo reference for " + ticker_symbol + ": " +
                ", ".join(c.label for c in val_not_applicable) +
                " — Yahoo doesn't report an equivalent field (FCF Yield), doesn't report one for this company (PEG, EV/EBITDA), or a required input (EBIT, Net Income, Market Cap) is missing."
            )

    with tab_chart_workspace:
        # ==========================================
        # CHARTS & INDICATORS
        # ==========================================
        st.header("Interactive Price & Technicals", anchor="technicals")

        # Indicator controls live HERE, directly above the chart they
        # configure, rather than in the sidebar — one place to look instead
        # of two. Collapsed by default so the chart itself stays the
        # prominent thing in this panel; every widget keeps the exact
        # label/range/default/help it had in the old sidebar Chart tab, and
        # every one of these variables is first CONSUMED further down this
        # same block (sma_periods onward), so defining them here keeps the
        # original define-before-use order intact.
        with st.expander("Chart Indicators & Overlays", expanded=False):
            _ind_core, _ind_momentum, _ind_extra = st.columns(3)
            with _ind_core:
                st.caption("**Trend**")
                sma_length = st.slider("SMA Length", min_value=CHART_DEFAULTS.sma_range[0], max_value=CHART_DEFAULTS.sma_range[1], value=CHART_DEFAULTS.sma_default)
                show_sma_trio = st.checkbox(f"Show {'/'.join(str(p) for p in TECHNICAL.sma_trio_periods)}-day SMA Trio", value=True)
                show_bollinger_bands = st.checkbox(f"Show Bollinger Bands ({TECHNICAL.bollinger_num_std:.0f}σ, {sma_length}-period)", value=True)
                show_ichimoku = st.checkbox("Show Ichimoku Cloud", value=False, help=f"Tenkan {TECHNICAL.ichimoku_tenkan_period} / Kijun {TECHNICAL.ichimoku_kijun_period} / Senkou B {TECHNICAL.ichimoku_senkou_b_period} — fixed periods, includes a real forward-projected cloud.")
            with _ind_momentum:
                st.caption("**Momentum & Volatility**")
                rsi_length = st.slider("RSI Length", min_value=CHART_DEFAULTS.rsi_range[0], max_value=CHART_DEFAULTS.rsi_range[1], value=CHART_DEFAULTS.rsi_default)
                show_rsi_panel = st.checkbox("Show RSI Panel", value=True)
                show_macd_panel = st.checkbox("Show MACD Panel", value=True)
                atr_length = st.slider("ATR Length", min_value=CHART_DEFAULTS.atr_range[0], max_value=CHART_DEFAULTS.atr_range[1], value=CHART_DEFAULTS.atr_default)
                stochastic_k_length = st.slider("Stochastic %K Length", min_value=CHART_DEFAULTS.stochastic_k_range[0], max_value=CHART_DEFAULTS.stochastic_k_range[1], value=TECHNICAL.stochastic_k_period)
                show_stochastic_panel = st.checkbox("Show Stochastic Panel", value=False)
            with _ind_extra:
                st.caption("**Volume & Strength**")
                show_vwap = st.checkbox("Show Anchored VWAP", value=False, help="Cumulative Volume-Weighted Average Price from a chosen anchor date. Quantix only has daily bars, so this is an anchored VWAP rather than an intraday session VWAP.")
                vwap_anchor_date = None
                if show_vwap:
                    vwap_anchor_date = st.date_input("VWAP Anchor Date", value=None, help="Defaults to the start of the loaded price history if left blank.")
                adx_length = st.slider("ADX Length", min_value=CHART_DEFAULTS.adx_range[0], max_value=CHART_DEFAULTS.adx_range[1], value=TECHNICAL.adx_period)
                show_adx_panel = st.checkbox("Show ADX Panel", value=False)
                show_obv_panel = st.checkbox("Show OBV Panel", value=False)

        # Every indicator below reads from the processed, validated frame (see
        # price_processing.py) — this note is purely informational: it's rare
        # for real Yahoo data to trigger any of these, so it stays a quiet
        # caption rather than a prominent report unless something was found.
        ppr = price_processing_result
        if ppr.is_clean:
            st.caption("🟢 Price data validated — no duplicate timestamps, invalid bars, or likely gaps detected.")
        else:
            with st.expander(f"Price data required cleaning ({ppr.issue_count} issue(s)) — click for details", expanded=False):
                if ppr.duplicate_rows_removed:
                    st.warning(f"Removed {ppr.duplicate_rows_removed} duplicate timestamp(s).")
                if ppr.invalid_rows_removed:
                    st.warning(f"Removed {ppr.invalid_rows_removed} structurally invalid bar(s) (e.g. High below Low, non-positive price).")
                for gap in ppr.possible_gaps:
                    st.info(f"Possible missing observation: {gap}")
                for w in ppr.warnings:
                    st.info(w)

        # SMA lines and crossover signals are computed by technical_indicators.py
        # (never inline here) — the custom-length line (sidebar slider) plus,
        # optionally, the standard 20/50/200-day trio.
        sma_periods = [sma_length] + (list(TECHNICAL.sma_trio_periods) if show_sma_trio else [])
        df = compute_sma_lines(df, sma_periods)
        sma_signals = detect_sma_crossovers(df, sma_length)
        df[f"RSI_{rsi_length}"] = compute_rsi(df, rsi_length)
        df = compute_macd(df)
        macd_signals = detect_macd_crossovers(df)
        # Bollinger Bands reuse the SMA Length slider as their period (the
        # middle band IS that same SMA — already plotted above, so it isn't
        # redrawn separately here) — per your call, one period controls both.
        df = compute_bollinger_bands(df, sma_length)
        bb_breakouts = detect_bollinger_breakouts(df)
        df[f"ATR_{atr_length}"] = compute_atr(df, atr_length)

        # The 5 newer indicators are only computed when their sidebar toggle is
        # on — all default off (see config.py / sidebar section 4), so a fresh
        # chart stays exactly as it was before this section existed.
        stoch_signals = []
        if show_stochastic_panel:
            df = compute_stochastic(df, k_period=stochastic_k_length)
            stoch_signals = detect_stochastic_crossovers(df)
        if show_vwap:
            df["VWAP"] = compute_anchored_vwap(df, anchor_date=vwap_anchor_date)
        if show_adx_panel:
            df = compute_adx(df, period=adx_length)
        if show_obv_panel:
            df["OBV"] = compute_obv(df)
        ichimoku_result = compute_ichimoku(df) if show_ichimoku else None

        # Chart rows are built dynamically from which indicator panels are
        # toggled on — the price panel (candlesticks + SMA/Bollinger/VWAP/
        # Ichimoku overlays + volume) is always row 1; RSI, MACD, Stochastic,
        # ADX, and OBV each claim the next row only if their sidebar toggle is
        # on, so the chart is 1-6 rows depending on what the user actually wants
        # to see (fewer traces too, when panels are off — "optimize rendering"
        # isn't just visual, it's fewer Plotly objects).
        panels = (
            ["price"]
            + (["rsi"] if show_rsi_panel else [])
            + (["macd"] if show_macd_panel else [])
            + (["stochastic"] if show_stochastic_panel else [])
            + (["adx"] if show_adx_panel else [])
            + (["obv"] if show_obv_panel else [])
        )
        row_of = {name: i + 1 for i, name in enumerate(panels)}
        num_rows = len(panels)
        # Price panel keeps half the chart height; any oscillator rows split the
        # remaining half evenly between them. This generalizes the old
        # hand-picked {1,2,3}-row table (which this reduces to exactly for the
        # num_rows==3 case) to the now-possible 4-6 row layouts.
        row_heights = [1.0] if num_rows == 1 else [0.5] + [0.5 / (num_rows - 1)] * (num_rows - 1)
        # secondary_y on row 1 only, for the volume overlay below.
        specs = [[{"secondary_y": True}]] + [[{}] for _ in range(num_rows - 1)]

        fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights, specs=specs)
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
        # Volume bars, overlaid at the bottom of the price panel on a secondary
        # axis (the standard, space-efficient convention — TradingView does the
        # same) rather than a dedicated 4th chart row. The secondary axis range
        # is set to 4x the actual max so the bars stay compact and never
        # visually compete with the candlesticks.
        volume_colors = ['#22c55e' if c >= o else '#ef4444' for o, c in zip(df['Open'], df['Close'])]
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=volume_colors, name='Volume', opacity=0.3), row=1, col=1, secondary_y=True)
        fig.update_yaxes(range=[0, df['Volume'].max() * 4], showgrid=False, showticklabels=False, secondary_y=True, row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{sma_length}"], line=dict(color='orange', width=2), name=f'SMA {sma_length}'), row=1, col=1)
        if show_sma_trio:
            for period, color in zip(TECHNICAL.sma_trio_periods, TECHNICAL.sma_trio_colors):
                if period == sma_length:
                    continue  # already plotted as the primary orange line above — avoid an exact duplicate overlay
                fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{period}"], line=dict(color=color, width=1), name=f'SMA {period}'), row=1, col=1)
        if show_bollinger_bands:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper'], line=dict(color='rgba(148, 163, 184, 0.6)', width=1, dash='dot'), name='BB Upper',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower'], line=dict(color='rgba(148, 163, 184, 0.6)', width=1, dash='dot'), name='BB Lower',
                fill='tonexty', fillcolor='rgba(148, 163, 184, 0.08)',
            ), row=1, col=1)
            if bb_breakouts:
                bb_upper_breaks = [b for b in bb_breakouts if b.kind == "upper"]
                bb_lower_breaks = [b for b in bb_breakouts if b.kind == "lower"]
                fig.add_trace(go.Scatter(
                    x=[b.date for b in bb_upper_breaks], y=[b.price for b in bb_upper_breaks], mode='markers', name='BB Upper Breakout',
                    marker=dict(symbol='star', size=12, color='#ef4444', line=dict(width=1, color='white')),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[b.date for b in bb_lower_breaks], y=[b.price for b in bb_lower_breaks], mode='markers', name='BB Lower Breakout',
                    marker=dict(symbol='star', size=12, color='#22c55e', line=dict(width=1, color='white')),
                ), row=1, col=1)
        if sma_signals:
            bullish = [s for s in sma_signals if s.kind == "bullish"]
            bearish = [s for s in sma_signals if s.kind == "bearish"]
            fig.add_trace(go.Scatter(
                x=[s.date for s in bullish], y=[s.price for s in bullish], mode='markers', name='Bullish Crossover',
                marker=dict(symbol='triangle-up', size=11, color='#22c55e', line=dict(width=1, color='white')),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[s.date for s in bearish], y=[s.price for s in bearish], mode='markers', name='Bearish Crossover',
                marker=dict(symbol='triangle-down', size=11, color='#ef4444', line=dict(width=1, color='white')),
            ), row=1, col=1)
        if show_vwap:
            fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], line=dict(color='#e879f9', width=2, dash='dash'), name='Anchored VWAP'), row=1, col=1)
        if show_ichimoku and ichimoku_result is not None:
            hist, fwd = ichimoku_result.historical, ichimoku_result.forward
            # Senkou A/B plotted across the FULL horizon (historical + forward
            # projection) as one continuous pair of lines, with the cloud shaded
            # between them — the forward segment uses real future business dates
            # (see compute_ichimoku's docstring), not a fabricated extension of
            # historical price.
            senkou_a_full = pd.concat([hist['SenkouA'], fwd['SenkouA']])
            senkou_b_full = pd.concat([hist['SenkouB'], fwd['SenkouB']])
            fig.add_trace(go.Scatter(x=senkou_a_full.index, y=senkou_a_full, line=dict(color='rgba(34, 197, 94, 0.5)', width=1), name='Senkou A'), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=senkou_b_full.index, y=senkou_b_full, line=dict(color='rgba(239, 68, 68, 0.5)', width=1), name='Senkou B',
                fill='tonexty', fillcolor='rgba(148, 163, 184, 0.12)',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Tenkan'], line=dict(color='#38bdf8', width=1), name='Tenkan-sen'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Kijun'], line=dict(color='#facc15', width=1), name='Kijun-sen'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Chikou'], line=dict(color='#a3a3a3', width=1, dash='dot'), name='Chikou Span'), row=1, col=1)
        if show_rsi_panel:
            rsi_row = row_of["rsi"]
            fig.add_trace(go.Scatter(x=df.index, y=df[f"RSI_{rsi_length}"], line=dict(color='purple'), name='RSI'), row=rsi_row, col=1)
            # Overbought/oversold reference zones (config.TECHNICAL.rsi_overbought/rsi_oversold)
            fig.add_hrect(y0=TECHNICAL.rsi_overbought, y1=100, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, row=rsi_row, col=1)
            fig.add_hrect(y0=0, y1=TECHNICAL.rsi_oversold, fillcolor="rgba(34, 197, 94, 0.08)", line_width=0, row=rsi_row, col=1)
            fig.add_hline(y=TECHNICAL.rsi_overbought, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", row=rsi_row, col=1)
            fig.add_hline(y=TECHNICAL.rsi_oversold, line_dash="dash", line_color="rgba(34, 197, 94, 0.6)", row=rsi_row, col=1)
        if show_macd_panel:
            macd_row = row_of["macd"]
            # MACD: histogram bars (colored by sign) plus the MACD/Signal lines and crossover markers.
            hist_colors = ['#22c55e' if v >= 0 else '#ef4444' for v in df['MACD_Histogram'].fillna(0)]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], marker_color=hist_colors, name='MACD Histogram', opacity=0.5), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Line'], line=dict(color='#38bdf8', width=1.5), name='MACD Line'), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line=dict(color='#facc15', width=1.5), name='MACD Signal'), row=macd_row, col=1)
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", row=macd_row, col=1)
            if macd_signals:
                macd_bullish = [s for s in macd_signals if s.kind == "bullish"]
                macd_bearish = [s for s in macd_signals if s.kind == "bearish"]
                fig.add_trace(go.Scatter(
                    x=[s.date for s in macd_bullish], y=[s.macd_value for s in macd_bullish], mode='markers', name='MACD Bullish Crossover',
                    marker=dict(symbol='triangle-up', size=9, color='#22c55e', line=dict(width=1, color='white')),
                ), row=macd_row, col=1)
                fig.add_trace(go.Scatter(
                    x=[s.date for s in macd_bearish], y=[s.macd_value for s in macd_bearish], mode='markers', name='MACD Bearish Crossover',
                    marker=dict(symbol='triangle-down', size=9, color='#ef4444', line=dict(width=1, color='white')),
                ), row=macd_row, col=1)
        if show_stochastic_panel:
            stoch_row = row_of["stochastic"]
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_K'], line=dict(color='#38bdf8', width=1.5), name='%K'), row=stoch_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Stoch_D'], line=dict(color='#facc15', width=1.5), name='%D'), row=stoch_row, col=1)
            fig.add_hrect(y0=TECHNICAL.stochastic_overbought, y1=100, fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, row=stoch_row, col=1)
            fig.add_hrect(y0=0, y1=TECHNICAL.stochastic_oversold, fillcolor="rgba(34, 197, 94, 0.08)", line_width=0, row=stoch_row, col=1)
            fig.add_hline(y=TECHNICAL.stochastic_overbought, line_dash="dash", line_color="rgba(239, 68, 68, 0.6)", row=stoch_row, col=1)
            fig.add_hline(y=TECHNICAL.stochastic_oversold, line_dash="dash", line_color="rgba(34, 197, 94, 0.6)", row=stoch_row, col=1)
            if stoch_signals:
                stoch_bullish = [s for s in stoch_signals if s.kind == "bullish"]
                stoch_bearish = [s for s in stoch_signals if s.kind == "bearish"]
                fig.add_trace(go.Scatter(
                    x=[s.date for s in stoch_bullish], y=[s.k_value for s in stoch_bullish], mode='markers', name='Stochastic Bullish Crossover',
                    marker=dict(symbol='triangle-up', size=9, color='#22c55e', line=dict(width=1, color='white')),
                ), row=stoch_row, col=1)
                fig.add_trace(go.Scatter(
                    x=[s.date for s in stoch_bearish], y=[s.k_value for s in stoch_bearish], mode='markers', name='Stochastic Bearish Crossover',
                    marker=dict(symbol='triangle-down', size=9, color='#ef4444', line=dict(width=1, color='white')),
                ), row=stoch_row, col=1)
        if show_adx_panel:
            adx_row = row_of["adx"]
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='white', width=1.5), name='ADX'), row=adx_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], line=dict(color='#22c55e', width=1), name='+DI'), row=adx_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], line=dict(color='#ef4444', width=1), name='-DI'), row=adx_row, col=1)
            fig.add_hline(y=TECHNICAL.adx_trend_threshold, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=adx_row, col=1)
        if show_obv_panel:
            obv_row = row_of["obv"]
            fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], line=dict(color='#a78bfa', width=1.5), name='OBV'), row=obv_row, col=1)
        fig.update_layout(
        xaxis_rangeslider_visible=False,
        # Taller than the old mid-page chart (was 400 + 200/extra row): this
        # is now a dedicated workspace panel with the whole viewport to
        # itself, so the price panel gets the room a primary charting
        # surface warrants. Still scales with the number of oscillator rows.
        height=620 + 220 * (num_rows - 1),
        dragmode='zoom',
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
        # Explicit, not relying on Streamlit/Plotly defaults: the modebar (with
        # its built-in zoom/pan/fullscreen-expand controls) stays visible, mouse
        # scroll zooms directly, and the Plotly logo link is dropped as clutter.
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False})

        # Current-value interpretation (RSI Interpretation Engine) — the shaded
        # zones above cover the full history at a glance; this is the "what does
        # today's reading mean" readout for the latest bar specifically.
        rsi_series = df[f"RSI_{rsi_length}"]
        rsi_interpretation = interpret_rsi(rsi_series.iloc[-1]) if rsi_series.notna().any() else None
        if rsi_interpretation:
            ri1, ri2 = st.columns([1, 3])
            ri1.metric(f"RSI ({rsi_length})", f"{rsi_interpretation.value:.1f}", help=f"Overbought ≥ {TECHNICAL.rsi_overbought:.0f} · Oversold ≤ {TECHNICAL.rsi_oversold:.0f}")
            with ri2:
                st.markdown(f"**{rsi_interpretation.label}**")
                st.caption(rsi_interpretation.explanation)
        else:
            st.info(f"RSI ({rsi_length}) not yet available — the selected date range doesn't cover enough trading days to complete the warm-up period.")

        if sma_signals:
            with st.expander(f"SMA {sma_length} Crossover Signals ({len(sma_signals)} in range)", expanded=False):
                recent = sma_signals[-10:][::-1]  # most recent first
                signal_data = {
                    "Date": [s.date.strftime("%Y-%m-%d") for s in recent],
                    "Signal": [f"{s.icon} {s.label}" for s in recent],
                    "Price": [f"${s.price:.2f}" for s in recent],
                    f"SMA {sma_length}": [f"${s.sma_value:.2f}" for s in recent],
                }
                st.table(pd.DataFrame(signal_data))
                if len(sma_signals) > 10:
                    st.caption(f"Showing the 10 most recent of {len(sma_signals)} signals in the selected date range.")

        if macd_signals:
            with st.expander(f"MACD Crossover Signals ({len(macd_signals)} in range)", expanded=False):
                recent_macd = macd_signals[-10:][::-1]  # most recent first
                macd_signal_data = {
                    "Date": [s.date.strftime("%Y-%m-%d") for s in recent_macd],
                    "Signal": [f"{s.icon} {s.label}" for s in recent_macd],
                    "MACD Line": [f"{s.macd_value:.3f}" for s in recent_macd],
                    "Signal Line": [f"{s.signal_value:.3f}" for s in recent_macd],
                }
                st.table(pd.DataFrame(macd_signal_data))
                if len(macd_signals) > 10:
                    st.caption(f"Showing the 10 most recent of {len(macd_signals)} signals in the selected date range.")

        if stoch_signals:
            with st.expander(f"Stochastic Crossover Signals ({len(stoch_signals)} in range)", expanded=False):
                recent_stoch = stoch_signals[-10:][::-1]  # most recent first
                stoch_signal_data = {
                    "Date": [s.date.strftime("%Y-%m-%d") for s in recent_stoch],
                    "Signal": [f"{s.icon} {s.label}" for s in recent_stoch],
                    "%K": [f"{s.k_value:.1f}" for s in recent_stoch],
                    "%D": [f"{s.d_value:.1f}" for s in recent_stoch],
                }
                st.table(pd.DataFrame(stoch_signal_data))
                if len(stoch_signals) > 10:
                    st.caption(f"Showing the 10 most recent of {len(stoch_signals)} signals in the selected date range.")

        if bb_breakouts:
            with st.expander(f"Bollinger Band Breakouts ({len(bb_breakouts)} in range)", expanded=False):
                recent_bb = bb_breakouts[-10:][::-1]  # most recent first
                bb_data = {
                    "Date": [b.date.strftime("%Y-%m-%d") for b in recent_bb],
                    "Event": [f"{b.icon} {b.label}" for b in recent_bb],
                    "Close": [f"${b.price:.2f}" for b in recent_bb],
                    "Band": [f"${b.band_value:.2f}" for b in recent_bb],
                }
                st.table(pd.DataFrame(bb_data))
                if len(bb_breakouts) > 10:
                    st.caption(f"Showing the 10 most recent of {len(bb_breakouts)} breakouts in the selected date range.")

        st.download_button(
            "Download Price & Indicator Data (CSV)",
            data=df.to_csv().encode("utf-8"),
            file_name=f"{ticker_symbol}_price_indicators_{start_date}_{end_date}.csv",
            mime="text/csv",
            help="OHLCV plus every technical indicator column computed above (SMA/RSI/MACD/Bollinger/ATR, and Stochastic/VWAP/ADX/OBV if their panel is toggled on) — exactly what's plotted, one row per trading day. The Ichimoku Cloud's forward-projected segment isn't included, since it extends past the last trading day in this table.",
        )

        # ==========================================
        # NEW: ALPHA BENCHMARKING
        # ==========================================
        st.markdown("---")
        st.header("Relative Strength & Alpha Generation")

        # Computed here (not down in the DCF section) so both this section's
        # Performance Attribution and the DCF's WACC can reuse the same
        # regression instead of running it twice.
        beta_regression = compute_capm_beta(df, bench_df) if not bench_df.empty else None

        if not bench_df.empty:
            df['CumReturn'] = (df['Close'] / df['Close'].iloc[0]) - 1
            bench_df['CumReturn'] = (bench_df['Close'] / bench_df['Close'].iloc[0]) - 1

            ticker_return = df['CumReturn'].iloc[-1] * 100
            bench_return = bench_df['CumReturn'].iloc[-1] * 100
            alpha = ticker_return - bench_return

            a1, a2, a3 = st.columns(3)
            a1.metric(f"{ticker_symbol} Period Return", f"{ticker_return:.2f}%")
            a2.metric(f"{benchmark_symbol} Period Return", f"{bench_return:.2f}%")
            a3.metric("Generated Alpha", f"{alpha:.2f}%", help="Performance strictly above the market benchmark.")

            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(x=df.index, y=df['CumReturn']*100, name=ticker_symbol, line=dict(color='orange')))
            fig_alpha.add_trace(go.Scatter(x=bench_df.index, y=bench_df['CumReturn']*100, name=benchmark_symbol, line=dict(color='gray')))
            fig_alpha.update_layout(xaxis_rangeslider_visible=False, height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_alpha, width="stretch")

            # --- Performance Attribution: a deeper breakdown of the same Alpha
            # comparison above, alongside it rather than replacing it — how
            # much of the excess return came from simply being exposed to the
            # market (Systematic) vs. this specific stock (Selection).
            st.subheader("Performance Attribution")
            attribution_beta, attribution_beta_source = fundamentals_engine.beta_estimate(
                beta_regression.beta if beta_regression else None
            )
            period_days = (df.index[-1] - df.index[0]).days
            attribution = compute_performance_attribution(
                ticker_return_pct=ticker_return, benchmark_return_pct=bench_return,
                beta=attribution_beta, period_days=period_days,
            )
            beta_source_labels = {
                "regressed": f"regressed vs. {benchmark_symbol}",
                "yahoo_reported": "Yahoo-reported",
                "market_assumption": "1.0 market assumption",
            }
            st.caption(
                f"Decomposes {ticker_symbol}'s excess return over a {attribution.risk_free_period_pct:.2f}% period "
                f"risk-free rate into how much came from market exposure (beta × benchmark excess return) vs. "
                f"stock-specific selection. Beta: {attribution_beta:.2f} ({beta_source_labels[attribution_beta_source]}). "
                f"Systematic + Selection always reconstructs Total Excess Return exactly, by construction."
            )
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Excess Return", f"{attribution.total_excess_return_pct:.2f}%", help=f"{ticker_symbol}'s period return minus the period risk-free rate.")
            p2.metric("Systematic (Market Beta)", f"{attribution.systematic_pct:.2f}%", help="beta × the benchmark's excess return — the portion of return explained by simply being exposed to the market at this beta.")
            p3.metric("Selection (Residual)", f"{attribution.selection_pct:.2f}%", help="Total excess return minus the systematic component — the portion attributable to this specific stock, not market exposure.")

    with tab_risk:
        # --- QUANTITATIVE CALCULATIONS ---
        df['Returns'] = df['Close'].pct_change()

        # Centralized VaR/CVaR engine (log returns, user-selected confidence/lookback)
        # — see risk_analytics.py. None when the selected date range + lookback
        # don't provide enough observations for a stable tail estimate.
        historical_var = compute_historical_var(df, var_confidence, lookback=var_lookback)
        parametric_var = compute_parametric_var(df, var_confidence, lookback=var_lookback)
        expected_shortfall = compute_expected_shortfall(df, var_confidence, lookback=var_lookback)

        # Centralized annualized return/volatility/Sharpe engine (log returns,
        # not the simple pct_change() above, on both legs of the ratio) — see
        # risk_analytics.py. None when the selected date range is too short or
        # volatility is exactly zero.
        annual_return = compute_annualized_return(df)
        annual_vol = compute_annualized_volatility(df)
        sharpe_ratio = compute_sharpe_ratio(df, risk_free_rate)
        sharpe_interpretation = interpret_sharpe_ratio(sharpe_ratio)
        # Textbook Sortino: downside deviation is the semi-deviation below a 0
        # target over ALL observations (not std() of only the negative days —
        # a different, non-standard formula the old ad-hoc code used).
        downside_deviation = compute_downside_deviation(df)
        sortino_ratio = compute_sortino_ratio(df, risk_free_rate)
        rolling_vol = compute_rolling_volatility(df, vol_window)
        current_rolling_vol = rolling_vol.dropna().iloc[-1] if rolling_vol.notna().any() else None
        # Buy-and-hold Maximum Drawdown over the selected date range — see
        # risk_analytics.py. Reused below by the Backtest section too, so the
        # strategy's own drawdown and this one share one formula.
        max_dd_result = compute_max_drawdown(df['Close'])
        calmar_ratio = compute_calmar_ratio(df)
        calmar_interpretation = interpret_calmar_ratio(calmar_ratio)

        df['Mean'] = df['Close'].rolling(window=sma_length).mean()
        df['Std'] = df['Close'].rolling(window=sma_length).std()
        current_z_score = (df['Close'].iloc[-1] - df['Mean'].iloc[-1]) / df['Std'].iloc[-1]
        # Computed here (not inline in the Strategy Builder section below) so
        # both the Z-Score condition in strategy_builder.py's condition library
        # and the "Statistical Distance from Mean" chart further down read the
        # same column.
        df['Z_Score'] = (df['Close'] - df['Mean']) / df['Std']

        # Classical Rescaled Range (R/S) analysis on log returns — see
        # risk_analytics.compute_hurst_exponent()'s docstring for the method
        # and its cross-check against synthetic random-walk/trending/
        # mean-reverting series. Replaces the previous single-scale
        # sqrt(std(price differences)) approximation.
        hurst_exponent = compute_hurst_exponent(df)
        if hurst_exponent is None:
            st.warning("Could not calculate Hurst Exponent (insufficient price history for this date range to form multiple R/S window sizes), defaulting to 0.5 (Random Walk).")
            hurst_exponent = 0.5

        # Altman Z-Score is calculated by the engine (it's pure statement math);
        # this file only reports the outcome.
        altman_z = fundamentals.altman_z
        z_verdict = fundamentals.altman_verdict
        if fundamentals.altman_missing_inputs:
            st.warning(f"Could not calculate Altman Z-Score — missing: {', '.join(fundamentals.altman_missing_inputs)}.")

        # ==========================================
        # RISK DASHBOARD (composite summary of every metric below)
        # ==========================================
        risk_score_result = compute_risk_score(
            rolling_volatility=current_rolling_vol,
            historical_var=historical_var,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_dd_result.max_drawdown if max_dd_result is not None else None,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            altman_z=altman_z,
        )

        st.markdown("---")
        st.header("Risk Dashboard", anchor="risk-dashboard")
        st.caption("At-a-glance summary of every risk metric below — click through to the detailed panels further down this page for the full picture on any one of them.")

        gauge_color = {"🟢": "#22c55e", "🟡": "#eab308", "🟠": "#f97316", "🔴": "#ef4444"}[risk_score_result.grade_icon]
        fig_risk_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score_result.score,
            number={'suffix': " / 100"},
            title={'text': f"{risk_score_result.grade_icon} Composite Risk Score — {risk_score_result.grade}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(239,68,68,0.25)'},
                    {'range': [30, 50], 'color': 'rgba(249,115,22,0.25)'},
                    {'range': [50, 75], 'color': 'rgba(234,179,8,0.25)'},
                    {'range': [75, 100], 'color': 'rgba(34,197,94,0.25)'},
                ],
            },
        ))
        fig_risk_gauge.update_layout(height=280, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=60, b=20))
        st.plotly_chart(fig_risk_gauge, width="stretch")
        if risk_score_result.excluded_factors:
            st.caption(f"Not computable for {ticker_symbol} and excluded from the composite score (rather than counted as a failure): {', '.join(risk_score_result.excluded_factors)}.")

        factor_cols = st.columns(4)
        for i, factor in enumerate(risk_score_result.factors):
            factor_cols[i % 4].metric(
                factor.label,
                factor.value_display,
                delta=f"{factor.icon} weight {factor.weight:.0%}",
                delta_color="off",
                help=f"Sub-score: {factor.sub_score:.0f}/100" if factor.sub_score is not None else "Not computable for this ticker — excluded from the composite score.",
            )

        # --- UI DISPLAY FOR QUANT ---
        st.markdown("---")
        st.header("Advanced Quantitative Risk & Time-Series Engine")

        v1, v2 = st.columns(2)
        v1.metric(
            f"Annualized Volatility ({vol_window}d)",
            f"{current_rolling_vol * 100:.2f}%" if current_rolling_vol is not None else "N/A",
            help="Rolling annualized standard deviation of daily log returns — the same figure feeding the Sharpe/Sortino ratios below.",
        )
        v2.metric(
            f"Full-Range Annualized Volatility",
            f"{annual_vol * 100:.2f}%" if annual_vol is not None else "N/A",
            help="Annualized volatility over the entire selected date range, not just the rolling window.",
        )
        if current_rolling_vol is not None:
            st.caption(f"Rolling {vol_window}-day annualized volatility")
            st.line_chart(rolling_vol * 100)
        else:
            st.info(f"Rolling volatility needs at least {vol_window} trading days in the selected date range — widen the date range or shorten the window to see it.")

        st.markdown("##")
        st.subheader("Value at Risk")
        if historical_var is not None and parametric_var is not None:
            var1, var2, var3 = st.columns(3)
            var1.metric(
                f"1-Day Historical VaR ({var_confidence:.0%})",
                f"{historical_var * 100:.2f}%",
                help=f"Empirical {1 - var_confidence:.0%} percentile of daily log returns over the last {var_lookback} trading days — no distributional assumption.",
            )
            var2.metric(
                f"1-Day Parametric VaR ({var_confidence:.0%})",
                f"{parametric_var * 100:.2f}%",
                help=f"Same {var_confidence:.0%} confidence level, but assuming daily log returns are normally distributed (variance-covariance method) over the same {var_lookback}-day window.",
            )
            var3.metric(
                f"Expected Shortfall / CVaR ({var_confidence:.0%})",
                f"{expected_shortfall * 100:.2f}%" if expected_shortfall is not None else "N/A",
                help=f"Average loss across every day worse than the Historical VaR cutoff — the tail average, not just the cutoff itself. Increasingly preferred by regulators (Basel III) over VaR alone.",
            )
            var_gap = (historical_var - parametric_var) * 100
            if abs(var_gap) >= 0.1:
                fatter_tail = "fatter" if historical_var < parametric_var else "thinner"
                st.caption(f"Historical and Parametric VaR diverge by {abs(var_gap):.2f}pp — the empirical return distribution has a {fatter_tail} tail than a normal distribution would predict at this confidence level.")
            tail_risk_text = interpret_tail_risk(historical_var, expected_shortfall, var_confidence)
            if tail_risk_text:
                st.caption(tail_risk_text)

            lookback_returns = compute_log_returns(df).dropna().tail(var_lookback) * 100
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(x=lookback_returns, nbinsx=40, marker_color="#3b82f6", opacity=0.75, name="Daily log returns"))
            fig_var.add_vline(x=historical_var * 100, line_width=2, line_dash="dash", line_color="#f59e0b", annotation_text="VaR", annotation_position="top")
            if expected_shortfall is not None:
                fig_var.add_vline(x=expected_shortfall * 100, line_width=2, line_dash="dash", line_color="#ef4444", annotation_text="CVaR", annotation_position="top")
            fig_var.update_layout(
                height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Daily log return (%)", yaxis_title="Frequency", showlegend=False, margin=dict(t=30, b=30),
            )
            st.caption(f"Return distribution over the last {var_lookback} trading days, with VaR and CVaR marked")
            st.plotly_chart(fig_var, width="stretch")
        else:
            st.info(f"VaR needs at least {RISK.var_min_observations} trading days of returns in the selected lookback window — widen the date range or shorten the lookback to see it.")

        st.markdown("##")
        r1_c1, r1_c2 = st.columns(2)
        r1_c1.metric(
            "Sharpe Ratio TTM",
            f"{sharpe_ratio:.2f}" if sharpe_ratio is not None else "N/A",
            delta=sharpe_interpretation.label if sharpe_interpretation else None,
            delta_color="off",
            help=f"(Annualized return − {risk_free_rate:.2%} risk-free rate) / annualized volatility, both legs from daily log returns.",
        )
        r1_c2.metric(
            "Sortino Ratio TTM",
            f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A",
            help="(Annualized return − risk-free rate) / downside deviation — penalizes ONLY returns below the target, unlike Sharpe which penalizes all volatility equally.",
        )
        if sharpe_interpretation:
            st.caption(f"{sharpe_interpretation.explanation} {sharpe_interpretation.limitation}")

        st.markdown("##")
        r2_c1, r2_c2, r2_c3 = st.columns(3)

        r2_c1.metric("Current Price Z-Score", f"{current_z_score:.2f}", help="Distance from mean. >2 or <-2 indicates extreme deviation.")

        hurst_desc = "Random Walk"
        if hurst_exponent < RISK.hurst_mean_reverting_below: hurst_desc = "Mean-Reverting (Statistical Edge)"
        elif hurst_exponent > RISK.hurst_trending_above: hurst_desc = "Strongly Trending"
        r2_c2.metric("Hurst Exponent (H)", f"{hurst_exponent:.2f}", delta=hurst_desc, delta_color="normal" if RISK.hurst_mean_reverting_below <= hurst_exponent <= RISK.hurst_trending_above else "inverse")

        r2_c3.metric("Altman Z-Score", f"{altman_z:.2f}" if isinstance(altman_z, float) else "N/A", delta=z_verdict, delta_color="normal" if "Safe" in z_verdict else "inverse")

        st.subheader("Statistical Distance from Mean (Z-Score)")
        st.line_chart(df['Z_Score'])

        st.markdown("##")
        st.subheader("Maximum Drawdown")
        if max_dd_result is not None and max_dd_result.max_drawdown < 0:
            dd1, dd2, dd3 = st.columns(3)
            dd1.metric(
                "Max Drawdown",
                f"{max_dd_result.max_drawdown * 100:.2f}%",
                help=f"Peak: ${max_dd_result.peak_price:.2f} on {max_dd_result.peak_date.date()} → Trough: ${max_dd_result.trough_price:.2f} on {max_dd_result.trough_date.date()}.",
                delta_color="off",
            )
            dd2.metric("Peak → Trough", f"{max_dd_result.peak_date.date()} → {max_dd_result.trough_date.date()}")
            if max_dd_result.recovered:
                dd3.metric("Recovery Period", f"{max_dd_result.recovery_days} trading days", help=f"Recovered by {max_dd_result.recovery_date.date()}.")
            else:
                dd3.metric("Recovery Period", "Ongoing", help="Price has not yet closed back above the prior peak within the selected date range.", delta_color="off")

            drawdown_series = compute_drawdown_series(df['Close']) * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, fill='tozeroy', line=dict(color='#ef4444'), name='Drawdown'))
            fig_dd.add_trace(go.Scatter(x=[max_dd_result.trough_date], y=[max_dd_result.max_drawdown * 100], mode='markers', marker=dict(color='#f59e0b', size=10, symbol='diamond'), name='Trough'))
            fig_dd.update_layout(
                height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Drawdown (%)", showlegend=False, margin=dict(t=20, b=20),
            )
            st.caption("Underwater chart: % decline from the running peak over time, worst point marked")
            st.plotly_chart(fig_dd, width="stretch")

            st.markdown("##")
            calmar1, calmar2 = st.columns(2)
            calmar1.metric(
                "Calmar Ratio",
                f"{calmar_ratio:.2f}" if calmar_ratio is not None else "N/A",
                delta=calmar_interpretation.label if calmar_interpretation else None,
                delta_color="off",
                help="Annualized return ÷ |Maximum Drawdown| — return earned per unit of the single worst realized loss, rather than per unit of volatility.",
            )
            if calmar_interpretation:
                calmar2.caption(calmar_interpretation.explanation)

            risk_series_df = pd.DataFrame({
                f"Rolling_Volatility_{vol_window}d": rolling_vol,
                "Drawdown_Pct": drawdown_series,
            })
            st.download_button(
                "Download Risk Time-Series Data (CSV)",
                data=risk_series_df.to_csv().encode("utf-8"),
                file_name=f"{ticker_symbol}_risk_series_{start_date}_{end_date}.csv",
                mime="text/csv",
                help="Rolling annualized volatility and drawdown %, one row per trading day — the two per-row risk series above. Scalar metrics (VaR, Sharpe, Sortino, Calmar, Risk Score) are single values for the whole selected window, not per-row data, so they aren't in this file.",
            )
        else:
            st.info("No drawdown in the selected date range — price has been at or above every prior high throughout.")

        # ==========================================
        # EXECUTION & POSITION SIZING (KELLY-STYLE HEURISTIC)
        # ==========================================
        st.markdown("---")
        st.header("Execution & Position Sizing")
        st.caption(
            "A Kelly-STYLE sizing heuristic, not textbook Kelly applied to a validated trading edge. "
            "\"Win rate\" and \"payoff ratio\" below come from the UNCONDITIONAL fraction of positive-return "
            "days and the average up-day vs. down-day return over the selected date range — not a specific "
            "strategy's actual backtested trade-level performance. For an edge tied to a real trading rule, "
            "see the win rate reported in the Algorithmic Backtesting Simulator further down this page."
        )

        win_returns = df[df['Returns'] > 0]['Returns']
        loss_returns = df[df['Returns'] < 0]['Returns']
        final_allocation = 0.0

        if len(df['Returns'].dropna()) > 0 and len(loss_returns) > 0:
            win_prob = len(win_returns) / len(df['Returns'].dropna())
            win_loss_ratio = win_returns.mean() / abs(loss_returns.mean()) if abs(loss_returns.mean()) > 0 else 0
            # Kelly's formula (f* = p - (1-p)/b) applied to the UNCONDITIONAL daily
            # up/down statistics above — a simplification of Kelly's original
            # formulation, which assumes a discrete bet with a known, validated
            # edge and probability distribution, not "how often did the stock go
            # up on an arbitrary day." A declared simplifying heuristic, not a
            # fabricated number — see the caption above.
            kelly_pct = (win_prob - ((1 - win_prob) / win_loss_ratio)) if win_loss_ratio > 0 else 0
            if kelly_pct > 0:
                # Half-Kelly for risk management; quartered further when the macro risk flag is active
                half_kelly_pct = (kelly_pct / RISK.kelly_half_factor) * 100
                final_allocation = half_kelly_pct / RISK.kelly_macro_risk_extra_factor if macro_risk_flag else half_kelly_pct

            k1, k2, k3 = st.columns(3)
            k1.metric("Daily Win Rate", f"{win_prob * 100:.1f}%", help="Fraction of ALL trading days in the selected range with a positive return — an unconditional statistic, not a specific strategy's trade-level win rate.")
            k2.metric("Up/Down-Day Payoff Ratio", f"{win_loss_ratio:.2f}", help="Average positive-day return divided by the average magnitude of a negative-day return.")
            k3.metric("Heuristic Allocation (Half-Kelly)", f"{final_allocation:.2f}%", help="Half-Kelly position size from the simplified daily statistics above, penalized further during high VIX regimes. A rough sizing heuristic, not a rigorous Kelly application to a validated edge.")
        else:
            st.info("Insufficient return data to calculate this sizing heuristic.")

        # ATR-based volatility stop-loss — computed by technical_indicators.py,
        # not here (see suggested_stop_loss()). A complementary risk-management
        # figure alongside Kelly's bet-sizing above: how much to risk vs. where
        # to exit if the position moves against you.
        atr_series = df[f"ATR_{atr_length}"]
        current_atr = atr_series.iloc[-1] if atr_series.notna().any() else None
        stop_loss = suggested_stop_loss(standardized.current_price, current_atr)
        st.markdown("##")
        if stop_loss is not None:
            a1, a2, a3 = st.columns(3)
            a1.metric(f"ATR ({atr_length})", f"${current_atr:.2f}", help="Average True Range — the typical size of a full day's price movement, in dollars, over the selected period.")
            a2.metric(f"Suggested Stop-Loss ({TECHNICAL.atr_stop_multiplier:.0f}× ATR)", f"${stop_loss:.2f}", delta=f"{((stop_loss / standardized.current_price) - 1) * 100:.1f}% below current price", delta_color="off")
            a3.metric("Risk per Share", f"${standardized.current_price - stop_loss:.2f}")
            st.caption(f"Long-only, volatility-adjusted downside stop: current price − {TECHNICAL.atr_stop_multiplier:.0f}×ATR. Not a recommendation to hold a short position or a guarantee against gap-through losses.")
        else:
            st.info(f"ATR ({atr_length}) not yet available — the selected date range doesn't cover enough trading days to complete the warm-up period.")

    with tab_fundamentals:
        # ==========================================
        # PROFESSIONAL MULTI-STAGE DCF ENGINE
        # ==========================================
        st.markdown("---")
        st.header("Automated DCF Valuation Engine", anchor="dcf")

        # CAPM beta for the DCF's WACC: reuses the same regression computed
        # above for the Alpha/Performance Attribution section (against the
        # selected benchmark) rather than running it twice; wacc()/run_dcf()
        # fall back to Yahoo's reported beta, then a declared 1.0 market
        # assumption, when this comes back None.
        dcf_result = None  # stays None if the try block below raises before assignment — the Executive Digest checks this rather than assuming it's always set
        with st.expander("Professional Multi-Stage DCF Valuation", expanded=True):
            try:
                # The DCF model itself lives in the Fundamental Analysis Engine;
                # this block only renders its result.
                dcf_result = fundamentals_engine.run_dcf(
                    dcf_growth, fallback_price=df['Close'].iloc[-1],
                    regressed_beta=beta_regression.beta if beta_regression else None,
                    beta_r_squared=beta_regression.r_squared if beta_regression else None,
                )
                current_price = dcf_result.current_price

                if dcf_result.ok:
                    wacc = dcf_result.wacc
                    intrinsic_price = dcf_result.intrinsic_price
                    intrinsic_value = intrinsic_price  # Alias για το Executive Briefing
                    margin_of_safety = dcf_result.margin_of_safety_pct

                    # UI Metrics
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Market Price", f"${current_price:.2f}")
                    d2.metric("Intrinsic Value (2-Stage)", f"${intrinsic_price:.2f}")
                    d3.metric("Margin of Safety", f"{margin_of_safety:.2f}%", delta=dcf_result.status, delta_color=dcf_result.status_color)

                    beta_labels = {
                        "regressed": f"regressed vs. {benchmark_symbol}, R²={dcf_result.beta_r_squared:.2f}",
                        "yahoo_reported": "Yahoo-reported",
                        "market_assumption": "1.0 market assumption — no reported or regressible beta available",
                    }
                    st.info(
                        f"**WACC Calculated:** {wacc*100:.2f}% (CAPM & Debt Structure) | **Model:** 2-Stage Gordon Growth  \n"
                        f"**Beta:** {dcf_result.beta:.2f} ({beta_labels[dcf_result.beta_source]})"
                    )

                    # Visual Gauge
                    st.write("Safety Gauge:")
                    st.progress(min(max(margin_of_safety / 50, 0), 1.0))

                    # Sensitivity Analysis Heatmap
                    st.subheader("Sensitivity Analysis: WACC vs Terminal Growth")
                    wacc_range = np.linspace(max(0.01, wacc - DCF.sensitivity_wacc_delta), wacc + DCF.sensitivity_wacc_delta, DCF.sensitivity_steps)
                    growth_range = np.linspace(max(0.01, dcf_growth - DCF.sensitivity_growth_delta), dcf_growth + DCF.sensitivity_growth_delta, DCF.sensitivity_steps)

                    data = [[fundamentals_engine.intrinsic_price(g, w) for w in wacc_range] for g in growth_range]
                    df_heat = pd.DataFrame(data, columns=[f"{w*100:.1f}% WACC" for w in wacc_range], index=[f"{g*100:.1f}% Growth" for g in growth_range])
                    st.dataframe(df_heat.style.format("${:.2f}"))

                else:
                    if dcf_result.reason == "missing market cap":
                        st.warning("Missing market capitalization data. Cannot compute WACC/DCF reliably.")
                    else:
                        st.warning("Negative Free Cash Flow or missing shares. Cannot compute DCF reliably.")
                    intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0

            except ZeroDivisionError:
                # Terminal value is undefined when WACC exactly equals the terminal
                # growth rate used in the Gordon Growth model.
                st.error(f"DCF Engine Error: the discount rate (WACC) came out equal to the terminal growth rate ({DCF.terminal_growth_rate*100:.0f}%), which makes the terminal value mathematically undefined. Try adjusting the WACC or growth sliders.")
                intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0
            except Exception as e:
                log_exception(logger, "calc.error", section="dcf_engine", ticker=ticker_symbol)
                st.error(f"Unexpected DCF Engine error: {type(e).__name__}: {e}")
                intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0

    # ==========================================
    # EXECUTIVE DIGEST (fill) — every source signal this synthesizes
    # (Scorecard, Company Quality, Altman Z, Risk Score, DCF, buy-and-hold
    # Max Drawdown, VIX, RSI/SMA/MACD/Bollinger) has now been computed by
    # the sections above, so this is the earliest point in the script where
    # the digest can actually be built. It still renders at the TOP of the
    # page, in executive_digest_container, created before Data Quality
    # Report — see that container's own comment for why writing to it here
    # doesn't mean it displays here.
    # ==========================================
    with executive_digest_container:
        st.header("Executive Digest")
        st.caption("The top strengths and concerns auto-prioritized from every signal computed on this page — not a new analysis, a synthesis of what's already below. Click a line to jump to its source section.")

        # "Recent" = within the last 10 trading days of the selected range —
        # an SMA/MACD cross or Bollinger breakout from 6 months ago isn't a
        # current signal worth headlining here, even though it's still a
        # valid historical entry in its own section's table.
        _digest_recency_cutoff = df.index[-10] if len(df.index) >= 10 else df.index[0]
        _digest_recent_sma = sma_signals[-1] if sma_signals and sma_signals[-1].date >= _digest_recency_cutoff else None
        _digest_recent_macd = macd_signals[-1] if macd_signals and macd_signals[-1].date >= _digest_recency_cutoff else None
        _digest_recent_bb = bb_breakouts[-1] if bb_breakouts and bb_breakouts[-1].date >= _digest_recency_cutoff else None

        _digest_strengths, _digest_concerns = collect_flags(
            alignment_verdict=fundamentals.alignment_verdict,
            alignment_score_pct=fundamentals.score_pct,
            scorecard_checks=fundamentals.scorecard_checks,
            company_quality_category=cq.category,
            company_quality_score=cq.overall_score,
            altman_z=fundamentals.altman_z,
            altman_verdict=fundamentals.altman_verdict,
            risk_score=risk_score_result.score,
            risk_grade=risk_score_result.grade,
            risk_factors=risk_score_result.factors,
            dcf_ok=dcf_result.ok if dcf_result is not None else False,
            dcf_status=dcf_result.status if dcf_result is not None else None,
            dcf_margin_of_safety_pct=dcf_result.margin_of_safety_pct if dcf_result is not None else None,
            buy_hold_max_drawdown_pct=(max_dd_result.max_drawdown * 100) if max_dd_result is not None else None,
            macro_risk_flag=macro_risk_flag,
            vix_current=vix_current,
            rsi_interpretation=rsi_interpretation,
            recent_sma_signal=_digest_recent_sma,
            recent_macd_signal=_digest_recent_macd,
            recent_bollinger_breakout=_digest_recent_bb,
        )

        # Anchors that live outside this tab (Overview) can't be jumped to
        # with a plain #anchor link now that the page is split into
        # st.tabs() panels: Streamlit renders an inactive panel with
        # `hidden`, and browsers never scroll into something hidden — a
        # dead link that LOOKS clickable but silently does nothing would be
        # worse than no link at all. (A click-interception script was tried
        # and doesn't work either: st.markdown(unsafe_allow_html=True)
        # inserts via innerHTML, which browsers never execute <script>
        # through — confirmed live, window.__quantixTabJumpBound never got
        # set.) Named honestly instead: an anchor still inside THIS tab
        # (macro-regime) stays a real jump link; every other one names the
        # tab to check rather than pretending to jump there.
        _digest_anchor_tab = {
            "scorecard": "Fundamentals & Valuation",
            "quality-classification": "Fundamentals & Valuation",
            "dcf": "Fundamentals & Valuation",
            "technicals": "Chart Workspace",
            "risk-dashboard": "Risk & Technicals",
            # "macro-regime" deliberately absent — it's in this same
            # Overview tab as the Executive Digest, so the plain anchor
            # link already works natively.
        }

        def _digest_flag_line(flag):
            target_tab = _digest_anchor_tab.get(flag.anchor)
            if target_tab:
                return f"- {flag.text} *(see {target_tab} tab)*"
            return f"- [{flag.text}](#{flag.anchor})"

        if not _digest_strengths and not _digest_concerns:
            st.info("No standout strengths or concerns right now — every computed signal for this ticker currently reads as neutral, or too little data was available to classify one. See the sections below for the full picture.")
        else:
            _dcol1, _dcol2 = st.columns(2)
            with _dcol1:
                st.markdown("**🟢 Top Strengths**")
                if _digest_strengths:
                    for _flag in _digest_strengths:
                        st.markdown(_digest_flag_line(_flag))
                else:
                    st.caption("No standout strengths identified.")
            with _dcol2:
                st.markdown("**🔴 Top Concerns**")
                if _digest_concerns:
                    for _flag in _digest_concerns:
                        st.markdown(_digest_flag_line(_flag))
                else:
                    st.caption("No standout concerns identified.")

    with tab_simulation:
        # ==========================================
        # PATH 1: ALGORITHMIC BACKTESTING SIMULATOR
        # ==========================================
        st.markdown("---")
        st.header("Algorithmic Backtesting Simulator")
        st.caption("Compose an entry/exit strategy from indicators already on the chart — no code — or run the original Classic Mean-Reversion strategy.")

        strategy_preset = st.radio(
            "Strategy", ["Classic Mean-Reversion", "Custom"], horizontal=True, key="strategy_preset",
            help="Classic Mean-Reversion is the app's original Z-Score strategy. Custom lets you build entry/exit rules from RSI, SMA/price crossovers, MACD crossovers, Bollinger Band breakouts, and Z-Score thresholds.",
        )

        _strategy_library = condition_library(sma_length, rsi_length)

        if strategy_preset == "Classic Mean-Reversion":
            st.markdown(f"**Buy when Z-Score < {RISK.backtest_buy_z_score} | Sell when Z-Score > {RISK.backtest_sell_z_score}**")
            active_rule = classic_mean_reversion(RISK.backtest_buy_z_score, RISK.backtest_sell_z_score)
        else:
            st.session_state.setdefault("strategy_entry_conditions", [{"indicator": "zscore", "operator": "<", "threshold": -2.0}])
            st.session_state.setdefault("strategy_exit_conditions", [{"indicator": "zscore", "operator": ">", "threshold": 0.0}])

            def _render_condition_rows(state_key, key_prefix):
                remove_index = None
                for i, cond in enumerate(st.session_state[state_key]):
                    indicator_keys = list(_strategy_library.keys())
                    c1, c2, c3, c4 = st.columns([3, 3, 2, 1])
                    with c1:
                        cond["indicator"] = st.selectbox(
                            "Indicator", indicator_keys, index=indicator_keys.index(cond["indicator"]),
                            format_func=lambda k: _strategy_library[k].label, key=f"{key_prefix}_indicator_{i}",
                            label_visibility="visible" if i == 0 else "collapsed",
                        )
                    spec = _strategy_library[cond["indicator"]]
                    valid_ops = [op for op, _ in spec.operators]
                    # The indicator dropdown may have just changed to one with a
                    # different operator set (e.g. a level "<" to an event
                    # "bullish") — fall back to that indicator's first operator
                    # rather than keeping a now-invalid stored one.
                    if cond["operator"] not in valid_ops:
                        cond["operator"] = valid_ops[0]
                    op_labels = dict(spec.operators)
                    with c2:
                        cond["operator"] = st.selectbox(
                            "Condition", valid_ops, index=valid_ops.index(cond["operator"]),
                            format_func=lambda k: op_labels[k], key=f"{key_prefix}_operator_{i}",
                            label_visibility="visible" if i == 0 else "collapsed",
                        )
                    with c3:
                        if spec.kind == "level":
                            default_threshold = cond.get("threshold")
                            if default_threshold is None:
                                default_threshold = spec.default_threshold
                            cond["threshold"] = st.number_input(
                                "Threshold", value=float(default_threshold), key=f"{key_prefix}_threshold_{i}",
                                label_visibility="visible" if i == 0 else "collapsed",
                            )
                        else:
                            cond["threshold"] = None
                            if i == 0:
                                st.markdown("&nbsp;")
                    with c4:
                        if i == 0:
                            st.markdown("&nbsp;")
                        if st.button("✕", key=f"{key_prefix}_remove_{i}", help="Remove this condition"):
                            remove_index = i
                if remove_index is not None:
                    st.session_state[state_key].pop(remove_index)
                    st.rerun()

            st.markdown("**Entry Conditions**")
            entry_logic = st.radio("Combine with", LOGIC_OPTIONS, horizontal=True, key="strategy_entry_logic", label_visibility="collapsed")
            _render_condition_rows("strategy_entry_conditions", "strategy_entry")
            if st.button("+ Add Entry Condition"):
                st.session_state["strategy_entry_conditions"].append({"indicator": "rsi", "operator": "<", "threshold": 30.0})
                st.rerun()

            st.markdown("**Exit Conditions**")
            exit_logic = st.radio("Combine with", LOGIC_OPTIONS, horizontal=True, key="strategy_exit_logic", label_visibility="collapsed")
            _render_condition_rows("strategy_exit_conditions", "strategy_exit")
            if st.button("+ Add Exit Condition"):
                st.session_state["strategy_exit_conditions"].append({"indicator": "rsi", "operator": ">", "threshold": 70.0})
                st.rerun()

            active_rule = StrategyRule(
                name="Custom",
                entry_conditions=tuple(StrategyCondition(**c) for c in st.session_state["strategy_entry_conditions"]),
                entry_logic=entry_logic,
                exit_conditions=tuple(StrategyCondition(**c) for c in st.session_state["strategy_exit_conditions"]),
                exit_logic=exit_logic,
            )

            # Live preview — how many historical bars this rule set would fire
            # on, computed the same way the backtest itself will, so it's an
            # honest preview rather than an approximation. No "Run" button: this
            # is pure in-memory vectorized math on data already loaded for the
            # chart above, cheap enough to recompute on every edit.
            _preview_entries = evaluate_condition_set(df, active_rule.entry_conditions, active_rule.entry_logic, sma_length, rsi_length)
            _preview_exits = evaluate_condition_set(df, active_rule.exit_conditions, active_rule.exit_logic, sma_length, rsi_length)
            st.caption(f"Live preview: {int(_preview_entries.sum())} historical entry signal(s) · {int(_preview_exits.sum())} historical exit signal(s) in the selected date range.")
            if not active_rule.entry_conditions:
                st.info("Add at least one entry condition to generate a strategy.")
            # On a bar where entry AND exit both fire, the exit wins (see
            # run_backtest()'s Signal assignment order) — surfaced explicitly
            # here rather than left as a silent "why did my signal count not
            # match my trade count" surprise, since mixing indicator families
            # (e.g. a trend-following entry with a mean-reversion exit) makes
            # same-bar overlap genuinely possible, unlike the Classic preset
            # where Z-Score entry/exit thresholds are mutually exclusive by
            # construction.
            _preview_overlap = int((_preview_entries & _preview_exits).sum())
            if _preview_overlap:
                st.caption(f"{_preview_overlap} of those bars satisfy both entry and exit conditions at once — the exit takes precedence on a same-bar conflict, so those specific bars won't open a new position.")

        cost_bps = st.slider(
            "Transaction Cost (bps per trade leg)", min_value=0.0, max_value=BACKTEST_COST.max_cost_bps,
            value=BACKTEST_COST.default_cost_bps, step=1.0,
            help=(
                "A flat basis-point commission + slippage charge on EVERY entry AND every exit "
                "independently (not once per round trip) — a simplifying assumption, not a full "
                "market-impact model with size/liquidity/volatility dependence. Set to 0 to see the "
                "(unrealistic) cost-free gross result on its own."
            ),
        )
        backtest_result = run_backtest(df, active_rule, sma_length, rsi_length, cost_bps=cost_bps)

        bt1, bt2, bt3, bt4, bt5 = st.columns(5)
        bt1.metric("Strategy Return (Gross)", f"{backtest_result.total_strategy_return_pct:.2f}%", delta=f"{backtest_result.total_strategy_return_pct - backtest_result.total_buy_hold_return_pct:.2f}% vs Buy & Hold")
        bt2.metric("Buy & Hold Baseline", f"{backtest_result.total_buy_hold_return_pct:.2f}%")
        bt3.metric("Max Strategy Drawdown", f"{backtest_result.max_drawdown_pct:.2f}%", help="The deepest percentage drop your portfolio would have suffered using this algorithm (gross).", delta_color="inverse")
        bt4.metric("Win Rate", f"{backtest_result.win_rate_pct:.1f}%" if backtest_result.win_rate_pct is not None else "N/A", help="Of the days this strategy held a position, the fraction with a positive return.")
        bt5.metric("Trades", f"{backtest_result.trade_count}", help="Number of distinct times this strategy entered a position over the selected date range.")

        if cost_bps > 0:
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric(
                "Strategy Return (Net of Cost)", f"{backtest_result.total_net_strategy_return_pct:.2f}%",
                delta=f"{backtest_result.total_net_strategy_return_pct - backtest_result.total_strategy_return_pct:.2f}% vs gross",
                delta_color="inverse",
                help="Gross return minus every entry/exit's transaction cost charge — the more realistic net-of-cost outcome.",
            )
            nc2.metric("Net Max Drawdown", f"{backtest_result.net_max_drawdown_pct:.2f}%", delta_color="inverse")
            nc3.metric("Total Cost Paid", f"{backtest_result.total_cost_pct:.2f}%", help="Sum of every entry/exit's cost charge, as a percentage of starting capital.")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Buy_Hold'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
        fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Strategy'], name=f"{active_rule.name} (Gross)", line=dict(color='cyan', width=2)))
        if cost_bps > 0:
            fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Net_Strategy'], name=f"{active_rule.name} (Net of Cost)", line=dict(color='orange', width=2, dash='dash')))

        fig_bt.update_layout(
            title="Strategy Equity Curve vs Baseline",
            xaxis_rangeslider_visible=False,
            height=450,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        st.plotly_chart(fig_bt, width="stretch")

        # --- Walk-Forward Validation (optional, alongside the single-pass backtest above) ---
        walk_forward_enabled = st.checkbox(
            "Enable Walk-Forward Validation", value=False,
            help=(
                "Splits the selected date range into sequential train/test windows and reports "
                "out-of-sample-only performance, stitched together across every test window — alongside, "
                "never replacing, the single in-sample pass above. A large gap between the two curves is "
                "itself the useful signal: it means the strategy's apparent edge above depends on having "
                "already seen the period it's being judged against."
            ),
        )
        if walk_forward_enabled:
            wf_c1, wf_c2 = st.columns(2)
            train_days = int(wf_c1.number_input(
                "Train Window (trading days)", min_value=WALK_FORWARD.min_train_days,
                value=WALK_FORWARD.default_train_days, step=5,
                help="History each test window gets to warm up its indicators and carry forward any open position — not a parameter-fitting step, this strategy has no parameters to fit.",
            ))
            test_days = int(wf_c2.number_input(
                "Test Window (trading days)", min_value=WALK_FORWARD.min_test_days,
                value=WALK_FORWARD.default_test_days, step=5,
                help="Length of each out-of-sample segment. Windows roll forward by exactly this many days, never overlapping.",
            ))

            wf_result = run_walk_forward_backtest(df, active_rule, sma_length, rsi_length, train_days, test_days)

            if not wf_result.ok:
                st.warning(f"Walk-forward validation not available: {wf_result.reason}.")
            else:
                wf1, wf2, wf3, wf4, wf5 = st.columns(5)
                wf1.metric(
                    "Out-of-Sample Return", f"{wf_result.total_oos_return_pct:.2f}%",
                    delta=f"{wf_result.total_oos_return_pct - backtest_result.total_strategy_return_pct:.2f}% vs in-sample",
                    help="Stitched performance across every out-of-sample test window only — never a period the strategy was evaluated against before.",
                )
                wf2.metric("Windows", f"{wf_result.window_count}", help=f"{train_days} train / {test_days} test trading days per window, rolled forward non-overlapping.")
                wf3.metric("OOS Max Drawdown", f"{wf_result.max_drawdown_pct:.2f}%", delta_color="inverse")
                wf4.metric("OOS Win Rate", f"{wf_result.win_rate_pct:.1f}%" if wf_result.win_rate_pct is not None else "N/A")
                wf5.metric("OOS Trades", f"{wf_result.trade_count}")

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Strategy'], name="In-Sample (single pass)", line=dict(color='cyan', dash='dot')))
                fig_wf.add_trace(go.Scatter(x=wf_result.stitched_equity_curve.index, y=wf_result.stitched_equity_curve, name="Walk-Forward (out-of-sample)", line=dict(color='magenta', width=2)))
                fig_wf.update_layout(
                    title="In-Sample vs. Walk-Forward Out-of-Sample Equity",
                    xaxis_rangeslider_visible=False,
                    height=450,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                )
                st.plotly_chart(fig_wf, width="stretch")

                with st.expander(f"Per-window out-of-sample performance ({wf_result.window_count} windows)"):
                    wf_table = pd.DataFrame([
                        {
                            "Test Start": w.test_start.date(), "Test End": w.test_end.date(),
                            "Return": f"{w.strategy_return_pct:.2f}%", "Trades": w.trade_count,
                        }
                        for w in wf_result.windows
                    ])
                    st.dataframe(wf_table, hide_index=True, width="stretch")

        # ==========================================
        # PORTFOLIO BACKTESTER
        # ==========================================
        # Runs `active_rule` — the exact strategy configured above, whether
        # Classic Mean-Reversion or a Custom rule — across a multi-ticker
        # basket instead of the single loaded ticker. Deliberately NOT a
        # second strategy-definition UI: the basket is tested against
        # whatever's already configured, so editing the strategy above and
        # re-running this section always compares like-for-like.
        st.markdown("---")
        st.header("Portfolio Backtester")
        st.caption(
            "Runs the strategy configured above across a weighted, multi-ticker basket — with weight drift "
            "between rebalances and a configurable rebalance rule — instead of one ticker at a time. Each "
            "ticker's own signals are generated by the identical single-ticker engine above; this only "
            "combines the results into one portfolio equity curve."
        )

        pbt_tickers_raw = st.text_input(
            "Basket tickers (comma-separated)", value="", key="pbt_tickers_raw",
            placeholder=f"e.g. AAPL, MSFT, NVDA (max {PORTFOLIO_BACKTEST.max_tickers})",
        )
        pbt_tickers = parse_tickers(pbt_tickers_raw)[:PORTFOLIO_BACKTEST.max_tickers]
        if len(parse_tickers(pbt_tickers_raw)) > PORTFOLIO_BACKTEST.max_tickers:
            st.caption(f"Only the first {PORTFOLIO_BACKTEST.max_tickers} tickers are used.")

        pbt_weights = {}
        if pbt_tickers:
            st.markdown("**Target Weights** (rescaled automatically to sum to 100%)")
            pbt_weight_cols = st.columns(len(pbt_tickers))
            for pbt_col, pbt_t in zip(pbt_weight_cols, pbt_tickers):
                with pbt_col:
                    pbt_weights[pbt_t] = st.number_input(
                        pbt_t, min_value=0.0, value=round(100.0 / len(pbt_tickers), 1),
                        step=1.0, key=f"pbt_weight_{pbt_t}", help=f"Target allocation % for {pbt_t}.",
                    )

        pbt_c1, pbt_c2, pbt_c3 = st.columns(3)
        with pbt_c1:
            pbt_rebalance_freq = st.selectbox(
                "Rebalance Frequency", REBALANCE_FREQUENCIES, index=REBALANCE_FREQUENCIES.index(PORTFOLIO_BACKTEST.default_rebalance_frequency),
                format_func=lambda k: REBALANCE_FREQUENCY_LABELS[k], key="pbt_rebalance_freq",
                help="Periodic reset to target weights. Weights DRIFT with each ticker's own performance between rebalances — not reset every day.",
            )
        with pbt_c2:
            pbt_threshold_enabled = st.checkbox(
                "Also rebalance on drift threshold", value=False, key="pbt_threshold_enabled",
                help="An ADDITIONAL trigger: rebalance as soon as any ticker's weight drifts this many percentage points from its target, even between scheduled rebalances.",
            )
        with pbt_c3:
            pbt_threshold_pct = st.number_input(
                "Drift threshold (pp)", min_value=0.5, value=PORTFOLIO_BACKTEST.default_rebalance_threshold_pct,
                step=0.5, key="pbt_threshold_pct", disabled=not pbt_threshold_enabled,
            )

        pbt_run_clicked = st.button("Run Portfolio Backtest", type="primary", disabled=not pbt_tickers)
        if not pbt_tickers:
            st.caption("Enter at least one ticker to run a portfolio backtest.")

        if pbt_run_clicked:
            with st.spinner(f"Backtesting {len(pbt_tickers)} ticker(s)..."):
                pbt_input_dfs = {}
                pbt_prep_errors = {}
                for pbt_t in pbt_tickers:
                    pbt_df, pbt_err = prepare_ticker_for_backtest(pbt_t, start_date, end_date, sma_length, rsi_length)
                    if pbt_err:
                        pbt_prep_errors[pbt_t] = pbt_err
                    else:
                        pbt_input_dfs[pbt_t] = pbt_df

                pbt_result, pbt_run_err = run_portfolio_backtest(
                    pbt_input_dfs, active_rule, pbt_weights, sma_length, rsi_length,
                    rebalance_frequency=pbt_rebalance_freq,
                    rebalance_threshold_pct=pbt_threshold_pct if pbt_threshold_enabled else None,
                    cost_bps=cost_bps,
                )
                # Prep failures (couldn't even fetch/prepare the ticker) are a
                # DIFFERENT stage than run_portfolio_backtest's own exclusions
                # (which only sees tickers that made it past prep) — merged
                # here so the UI shows one complete list either way.
                if pbt_result is not None:
                    pbt_result.exclusion_reasons.update(pbt_prep_errors)
                    pbt_result.excluded_tickers = pbt_result.excluded_tickers + tuple(pbt_prep_errors.keys())
                st.session_state["pbt_last_result"] = pbt_result
                st.session_state["pbt_last_error"] = pbt_run_err if pbt_result is None else (
                    "; ".join(f"{t}: {r}" for t, r in pbt_prep_errors.items()) if pbt_prep_errors and not pbt_input_dfs else None
                )
                log_event(logger, logging.INFO, "user.portfolio_backtest_run", tickers=len(pbt_tickers), ok=pbt_result is not None)

        pbt_result = st.session_state.get("pbt_last_result")
        pbt_last_error = st.session_state.get("pbt_last_error")

        if pbt_last_error and pbt_result is None:
            st.error(f"Portfolio backtest could not run: {pbt_last_error}")
        elif pbt_result is not None:
            pb1, pb2, pb3, pb4, pb5 = st.columns(5)
            pb1.metric(
                "Portfolio Return", f"{pbt_result.total_return_pct:.2f}%",
                delta=f"{pbt_result.total_return_pct - pbt_result.total_buy_hold_return_pct:.2f}% vs static-weight reference",
            )
            pb2.metric(
                "Static-Weight Reference", f"{pbt_result.total_buy_hold_return_pct:.2f}%",
                help="Same target weights held with NO rebalancing at all — the reference line rebalancing is measured against.",
            )
            pb3.metric("Max Drawdown", f"{pbt_result.max_drawdown_pct:.2f}%", delta_color="inverse")
            pb4.metric("Sharpe Ratio", f"{pbt_result.sharpe_ratio:.2f}" if pbt_result.sharpe_ratio is not None else "N/A")
            pb5.metric("Rebalances", f"{len(pbt_result.rebalance_dates)}")

            fig_pbt = go.Figure()
            fig_pbt.add_trace(go.Scatter(x=pbt_result.df.index, y=pbt_result.df['Cum_Buy_Hold'], name='Static Weights (no rebalancing)', line=dict(color='gray', dash='dot')))
            fig_pbt.add_trace(go.Scatter(x=pbt_result.df.index, y=pbt_result.df['Cum_Portfolio'], name=f"Rebalanced Portfolio ({REBALANCE_FREQUENCY_LABELS[pbt_result.rebalance_frequency]})", line=dict(color='cyan', width=2)))
            fig_pbt.update_layout(
                title="Portfolio Equity Curve", xaxis_rangeslider_visible=False, height=420,
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
            )
            st.plotly_chart(fig_pbt, width="stretch")

            st.markdown("**Per-Ticker Contribution**")
            st.caption(
                "Contribution is each ticker's own weighted daily return, summed across the backtest — an "
                "additive decomposition that sums exactly to the sum of daily portfolio returns, which is "
                "close to but not exactly the compounded total above (the standard trade-off of simple "
                "return attribution, not a rounding error)."
            )
            pbt_contrib_rows = []
            for pbt_t in pbt_result.included_tickers:
                pbt_tb = pbt_result.ticker_backtests.get(pbt_t)
                pbt_contrib_rows.append({
                    "Ticker": pbt_t,
                    "Target Weight": f"{pbt_result.target_weights[pbt_t] * 100:.1f}%",
                    "Contribution to Return": f"{pbt_result.contribution_pct[pbt_t]:+.2f}%",
                    "Solo Strategy Return": f"{pbt_tb.total_net_strategy_return_pct:.2f}%" if pbt_tb else "N/A",
                    "Solo Trades": pbt_tb.trade_count if pbt_tb else "N/A",
                })
            st.dataframe(pd.DataFrame(pbt_contrib_rows), hide_index=True, width="stretch")

            if pbt_result.excluded_tickers:
                with st.expander(f"{len(pbt_result.excluded_tickers)} ticker(s) excluded from this run", expanded=False):
                    for pbt_t in pbt_result.excluded_tickers:
                        st.caption(f"{pbt_t}: {pbt_result.exclusion_reasons.get(pbt_t, 'unknown reason')}")

        # ==========================================
        # PATH 2: MONTE CARLO FUTURE PROBABILITY SIMULATOR
        # ==========================================
        st.markdown("---")
        st.header(" Path 2: Monte Carlo Future Probability Simulator")

        sim_method = st.radio(
            "Simulation Method",
            options=["Block Bootstrap (Historical Resampling)", "Geometric Brownian Motion (Normal)"],
            index=0,
            horizontal=True,
            help=(
                "Block Bootstrap resamples actual historical 5-day return blocks, preserving whatever real "
                "fat tails, skew, and volatility clustering the ticker's own history has. Geometric Brownian "
                "Motion instead draws shocks from a fitted normal distribution, which structurally cannot "
                "produce fatter-than-normal tails no matter what really happened historically."
            ),
        )
        is_bootstrap = sim_method.startswith("Block Bootstrap")

        st.markdown(
            f"Projecting 1,000 randomized future price paths over the next 60 trading days using **{sim_method}**."
        )

        # 1. Extract Historical Parameters from the Target Stock
        returns = df['Returns'].dropna()

        if len(returns) > MONTE_CARLO.min_history_days_for_bootstrap:
            # Simulation Parameters
            num_simulations = MONTE_CARLO.num_simulations
            forecast_days = MONTE_CARLO.forecast_days
            current_price = df['Close'].iloc[-1]

            # 2. Run the Simulation Matrix
            # Seed deterministically per-ticker so the simulation is stable across Streamlit reruns
            seed = zlib.crc32(ticker_symbol.encode()) % (2**32)

            if is_bootstrap:
                price_paths = simulate_bootstrap_paths(
                    df['Close'], current_price, num_simulations, forecast_days,
                    MONTE_CARLO.bootstrap_block_days, seed,
                )
            else:
                price_paths = simulate_gbm_paths(returns, current_price, num_simulations, forecast_days, seed)

            # 3. Process Statistical Thresholds
            pct_above_current, p10, p50, p90 = terminal_stats(price_paths, current_price)

            # 4. UI Metric Readout
            mc_c1, mc_c2, mc_c3 = st.columns(3)
            mc_c1.metric("Upside Probability", f"{pct_above_current:.1f}%", help="The percentage of simulated paths that ended higher than today's price.")
            mc_c2.metric("Median Target (P50)", f"${p50:.2f}", help="The 50th percentile median expected price outcome.")
            mc_c3.metric("Downside Floor (P10)", f"${p10:.2f}", help="90% of all simulations stayed ABOVE this price. Extreme statistical floor.")

            # 5. Plot the Probability Cloud using Plotly
            fig_mc = go.Figure()

            # Create a timeline for the next 60 trading days
            sim_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(forecast_days + 1)]

            # Plot a subset of lines to keep UI rendering fast while showing the cloud density
            for i in range(MONTE_CARLO.plotted_paths):
                fig_mc.add_trace(go.Scatter(
                    x=sim_dates,
                    y=price_paths[:, i],
                    mode='lines',
                    line=dict(color='rgba(0, 255, 240, 0.08)', width=1),
                    showlegend=False
                ))

            # Highlight key threshold lines
            fig_mc.add_trace(go.Scatter(x=sim_dates, y=[p90]*len(sim_dates), name="P90 Optimistic Target", line=dict(color="green", dash="dash")))
            fig_mc.add_trace(go.Scatter(x=sim_dates, y=[p50]*len(sim_dates), name="P50 Median Trend", line=dict(color="orange", width=2)))
            fig_mc.add_trace(go.Scatter(x=sim_dates, y=[p10]*len(sim_dates), name="P10 Downside Risk Floor", line=dict(color="red", dash="dash")))

            fig_mc.update_layout(
                xaxis_title="Simulation Timeline",
                yaxis_title="Stock Price ($)",
                height=500,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x"
            )
            st.plotly_chart(fig_mc, width="stretch")

            # 6. Feed the Monte Carlo insights down into the final verdict
            if pct_above_current > MONTE_CARLO.upside_bias_threshold_pct:
                st.success(f"Mathematical variance indicates a strong asymmetric upside bias. {pct_above_current:.1f}% of randomized models favor positive drift.")
            elif pct_above_current < MONTE_CARLO.downside_bias_threshold_pct:
                st.error(f"Negative drift bias detected. Less than {pct_above_current:.1f}% of randomized models manage to clear current price thresholds.")

        else:
            st.warning("Insufficient trading history to populate volatility parameters for Monte Carlo simulation.")

        # ==========================================
        # PATH 3: 3D SEASONALITY & TIME-SERIES SURFACE
        # ==========================================
        st.markdown("---")
        st.header("Path 3: 10-Year 3D Seasonality Surface")
        st.markdown("Mapping a decade of monthly performance to locate historical liquidity traps and explosive seasonal windows.")

        @st.cache_data(ttl=3600)
        def fetch_seasonality_data(ticker):
            try:
                # Fetch 10 years of monthly data to build the 3D grid
                hist = load_seasonality_history(ticker)
                if hist.empty: return pd.DataFrame()

                # Calculate monthly % change
                hist['Return'] = hist['Close'].pct_change() * 100

                # Extract Year and Month for our X and Y axes
                hist['Year'] = hist.index.year
                hist['Month'] = hist.index.strftime('%b')

                # Create a pivot table: Rows = Years, Cols = Months, Z-Values = Return
                months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                pivot = hist.pivot_table(values='Return', index='Year', columns='Month', aggfunc='sum')
                pivot = pivot.reindex(columns=months_order)

                return pivot.dropna(how='all')
            except (AttributeError, KeyError) as e:
                # load_seasonality_history always returns either an empty frame or
                # one with a proper DatetimeIndex, so this means Yahoo returned an
                # unexpected shape (e.g. a missing 'Close' column or non-datetime index).
                st.warning(f"Could not build seasonality surface for '{ticker}' (unexpected data shape): {e}")
                return pd.DataFrame()
            except Exception as e:
                log_exception(logger, "calc.error", section="seasonality_surface", ticker=ticker)
                st.warning(f"Unexpected error building seasonality surface for '{ticker}': {type(e).__name__}: {e}")
                return pd.DataFrame()

        with st.spinner("Generating 3D historical surface mesh..."):
            season_df = fetch_seasonality_data(ticker_symbol)

            if not season_df.empty:
                # Fill NaNs with 0 so the 3D surface renders as a continuous sheet
                z_data = season_df.fillna(0).values
                x_data = season_df.columns.tolist()
                y_data = season_df.index.tolist()

                # Build the 3D Plotly Surface
                fig_3d = go.Figure(data=[go.Surface(
                    z=z_data,
                    x=x_data,
                    y=y_data,
                    colorscale='RdYlGn', # Red = Loss, Yellow = Flat, Green = Massive Gain
                    cmin=-15, # Caps the color scale at +/- 15% so outliers don't wash out the map
                    cmax=15
                )])

                fig_3d.update_layout(
                    title=f"{ticker_symbol} Monthly Return Topography (%)",
                    autosize=True,
                    height=600,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(
                        xaxis_title='Month',
                        yaxis_title='Year',
                        zaxis_title='Return (%)',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)) # Angles the camera perfectly
                    )
                )

                # Split the layout to show the 3D graph and the exact data insights
                c1, c2 = st.columns([2.5, 1])
                with c1:
                    st.plotly_chart(fig_3d, width="stretch")

                with c2:
                    st.subheader("Seasonal Edge")
                    avg_monthly = season_df.mean()
                    best_month = avg_monthly.idxmax()
                    worst_month = avg_monthly.idxmin()

                    st.success(f"**Historical Best:** {best_month} ({avg_monthly[best_month]:.1f}%)")
                    st.error(f"**Historical Worst:** {worst_month} ({avg_monthly[worst_month]:.1f}%)")

                    st.markdown("### How to read the topology:")
                    st.markdown("- **Green Peaks:** Consistent institutional buying. This is your mathematical holding window.")
                    st.markdown("- **Red Valleys:** Historical capitulation. Wait for these drops to buy into the position.")
                    st.markdown("- **The Edge:** If you see a deep trench forming across the exact same month every single year, you are witnessing a structural liquidity drain.")
            else:
                st.warning("Insufficient multi-year history to generate a 3D seasonality surface.")

    with tab_smart_money:
        # ==========================================
        # PATH 4: SMART MONEY & INSTITUTIONAL FLOW
        # ==========================================
        st.markdown("---")
        st.header("Path 4: Smart Money & Insider Flow")
        st.markdown("Tracking the allocation of major hedge funds and the personal capital of C-Suite executives.")

        with st.spinner("Scraping SEC filings and institutional ownership data..."):
            try:
                # 1. Extract basic ownership metrics from the standardized object
                insider_pct = standardized.held_pct_insiders * 100 if standardized.held_pct_insiders is not None else None
                inst_pct = standardized.held_pct_institutions * 100 if standardized.held_pct_institutions is not None else None

                # Calculate the "Float" theoretically available to retail
                retail_pct = None if (insider_pct is None or inst_pct is None) else max(0, 100 - insider_pct - inst_pct)

                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Shares Held by Insiders", fmt_num(insider_pct, "%"), help="High insider ownership aligns management with shareholders.")
                sm2.metric("Shares Held by Institutions", fmt_num(inst_pct, "%"), help="Shows the level of hedge fund and mutual fund backing.")
                sm3.metric("Retail / Public Float", fmt_num(retail_pct, "%"), help="The remaining percentage of shares traded by the general public.")

                st.markdown("##")

                # 2. Reuse institutional/insider data already loaded in ticker_bundle
                inst_df = ticker_bundle.institutional_holders
                insider_df = ticker_bundle.insider_transactions

                col_inst, col_insider = st.columns(2)

                with col_inst:
                    st.subheader("Top Institutional Holders")
                    if inst_df is not None and not inst_df.empty:
                        # Clean up the dataframe for a pristine UI display
                        display_inst = inst_df[['Holder', 'Shares', 'Value']].head(10).copy()
                        display_inst['Shares'] = display_inst['Shares'].apply(lambda x: f"{x:,.0f}" if pd.notnull(x) else "0")
                        display_inst['Value'] = display_inst['Value'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0")
                        st.dataframe(display_inst, width="stretch", hide_index=True)
                    else:
                        st.info("No institutional holding data available.")

                with col_insider:
                    st.subheader("Recent Insider Transactions")
                    if insider_df is not None and not insider_df.empty:
                        # Dynamically check columns to prevent crashes based on yfinance version differences
                        cols_to_keep = [c for c in ['Start Date', 'Insider', 'Position', 'Transaction', 'Shares', 'Value'] if c in insider_df.columns]
                        display_insider = insider_df[cols_to_keep].head(10).copy()

                        if 'Shares' in display_insider.columns:
                            display_insider['Shares'] = pd.to_numeric(display_insider['Shares'], errors='coerce').fillna(0)
                            display_insider['Shares'] = display_insider['Shares'].apply(lambda x: f"{x:,.0f}")

                        st.dataframe(display_insider, width="stretch", hide_index=True)
                    else:
                        st.info("No recent insider transactions reported in SEC filings.")

            except KeyError as e:
                # This section only reads data already fetched (with retries) by
                # data_loader — it makes no network call of its own, so a KeyError
                # here means Yahoo's institutional_holders/insider_transactions
                # columns don't match what this section expects, not a timeout.
                st.error(f"Could not render institutional/insider data: Yahoo Finance returned an unexpected column layout (missing {e}). This is a known yfinance version-drift issue, not a network failure.")
            except Exception as e:
                log_exception(logger, "calc.error", section="smart_money", ticker=ticker_symbol)
                st.error(f"Unexpected error rendering Smart Money data: {type(e).__name__}: {e}")

        # ==========================================
        # PATH 5: PEER COMPETITOR MATRIX (RELATIVE VALUE)
        # ==========================================
        st.markdown("---")
        st.header("Path 5: Peer Competitor Matrix")
        st.markdown("Comparing relative valuation and fundamental health against top industry competitors.")

        # Smart default peers based on common targets to make testing feel seamless
        default_peers = PEER_DEFAULTS.for_ticker(ticker_symbol)

        peer_input = st.text_input("Enter up to 3 Competitor Tickers (comma-separated):", default_peers)

        def _competitor_row(t, std):
            return {
                "Ticker": t,
                "P/E Ratio": std.pe_ratio if std.pe_ratio is not None else np.nan,
                "PEG Ratio": std.peg_ratio if std.peg_ratio is not None else np.nan,
                "Price/Book": std.price_to_book if std.price_to_book is not None else np.nan,
                "ROE (%)": (std.return_on_equity * 100) if std.return_on_equity is not None else np.nan,
                "Net Margin (%)": (std.net_margin * 100) if std.net_margin is not None else np.nan,
                "Debt/Equity": std.debt_to_equity if std.debt_to_equity is not None else np.nan
            }

        @st.cache_data(ttl=3600)
        def fetch_competitor_data(target, target_std, peer_string):
            peers = [p.strip().upper() for p in peer_string.split(',')][:3] # Limit to top 3 peers

            # Target ticker's data was already standardized above, so reuse it here
            # instead of re-fetching it from Yahoo Finance a second time.
            metrics = [_competitor_row(target, target_std)]
            for t in peers:
                peer_bundle = load_ticker_bundle(t, deep=False)
                if not peer_bundle.is_valid:
                    st.warning(f"Could not fetch competitor data for '{t}': {'; '.join(peer_bundle.errors)}")
                    continue
                metrics.append(_competitor_row(t, standardize_financials(peer_bundle)))
            return pd.DataFrame(metrics)

        with st.spinner("Analyzing sector peers and building relative valuation matrix..."):
            comp_df = fetch_competitor_data(ticker_symbol, standardized, peer_input)

            if not comp_df.empty and len(comp_df) > 1:
                # Drop rows where critical data is totally missing
                comp_df = comp_df.dropna(subset=['P/E Ratio', 'Net Margin (%)'], how='all')

                # 1. Display the raw data matrix
                st.subheader("Fundamental Comparison Matrix")
                st.dataframe(comp_df.style.format(precision=2, na_rep="N/A"), width="stretch", hide_index=True)

                # --- 2. Normalize Data for Radar Chart ---
                radar_df = comp_df.set_index('Ticker').copy()
                radar_df = radar_df.fillna(radar_df.mean()) # Fill missing with average to prevent chart crashes

                # Define which metrics are "Higher is Better" vs "Lower is Better"
                higher_better = ['ROE (%)', 'Net Margin (%)']

                normalized_df = pd.DataFrame(index=radar_df.index)

                for col in radar_df.columns:
                    max_val = radar_df[col].max()
                    min_val = radar_df[col].min()

                    if max_val == min_val:
                        normalized_df[col] = 50 # Middle score if everyone is identical
                    else:
                        if col in higher_better:
                            # Standard normalization (0 to 100)
                            normalized_df[col] = ((radar_df[col] - min_val) / (max_val - min_val)) * 100
                        else:
                            # Inverted normalization for P/E and Debt (Lower = Closer to 100)
                            normalized_df[col] = (1 - ((radar_df[col] - min_val) / (max_val - min_val))) * 100

                # 3. Build Radar Chart
                st.subheader("Relative Edge Radar")
                st.markdown("*Note: Scores are normalized (0-100). A higher score means 'Best in Class' (e.g., a high score on P/E means the stock is the cheapest).*")

                categories = list(normalized_df.columns)
                fig_radar = go.Figure()

                # Specific colors: Target is Green, Peers are distinct colors
                colors = ['#00cc66', '#ffcc00', '#ff4b4b', '#3399ff']

                for i, ticker in enumerate(normalized_df.index):
                    # We append the first value to the end of the list to "close" the radar loop shape
                    r_values = normalized_df.loc[ticker].values.tolist()
                    r_values.append(r_values[0])
                    theta_values = categories + [categories[0]]

                    fig_radar.add_trace(go.Scatterpolar(
                        r=r_values,
                        theta=theta_values,
                        fill='toself' if i == 0 else 'none', # Only fill the target stock to make it pop
                        name=ticker,
                        line=dict(color=colors[i % len(colors)], width=3 if i == 0 else 1.5)
                    ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                        angularaxis=dict(tickfont=dict(size=13))
                    ),
                    showlegend=True,
                    height=550,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40)
                )

                st.plotly_chart(fig_radar, width="stretch")

            else:
                st.warning("Could not fetch sufficient competitor data. Please check the tickers and try again.")

    with tab_tearsheet:
        # ==========================================
        # PATH 6: DYNAMIC EXECUTIVE SYNTHESIS BRIEFING
        # ==========================================
        st.markdown("---")
        st.header("Path 6: Executive Synthesis Briefing")
        st.markdown("Automated algorithmic synthesis compiling core qualitative and quantitative pillars into a definitive narrative brief.")

        with st.spinner("Compiling investment narrative..."):
            try:
                # 1. Gather variables from previous sections dynamically to feed the logic
                # (Using fallback values if previous sections haven't fully executed)
                current_p = standardized.current_price if standardized.current_price is not None else 0
                target_v = intrinsic_value if 'intrinsic_value' in locals() else 0
                mos_val = margin_of_safety if 'margin_of_safety' in locals() else 0

                # Smart money proxies (reuse the values computed in Path 4 instead of re-deriving them)
                insider_own = insider_pct if 'insider_pct' in locals() and insider_pct is not None else 0
                inst_own = inst_pct if 'inst_pct' in locals() and inst_pct is not None else 0

                # Competitor context from Path 5 if available
                peer_text = ""
                if 'comp_df' in locals() and not comp_df.empty and len(comp_df) > 1:
                    top_peer = comp_df.iloc[1]['Ticker'] if len(comp_df) > 1 else "industry peers"
                    peer_text = f"When benchmarked against its direct peer group (specifically {top_peer}), "
                else:
                    peer_text = "Relative to its broader industry sector, "

                # 2. Algorithmic Narrative Logic Block
                # Valuation Pillar
                if mos_val > 0:
                    valuation_narrative = f"The asset displays a positive margin of safety ({mos_val:.1f}%), suggesting it is currently underpriced relative to its intrinsic value of ${target_v:.2f} derived via discounted free cash flows."
                else:
                    valuation_narrative = f"The asset trades at a premium relative to its calculated intrinsic value of ${target_v:.2f}. The current market price of ${current_p:.2f} reflects an baked-in growth premium, narrowing the statistical margin of safety ({mos_val:.1f}%)."

                # Liquidity & Smart Money Pillar
                if insider_own > TEAR_SHEET.high_insider_ownership_pct:
                    ownership_narrative = f"Management interests are fundamentally aligned with shareholders, underscored by a notable {insider_own:.2f}% insider ownership stake. Institutional backing remains robust at {inst_own:.2f}%, indicating deep liquidity and strong sponsorship from major fund complexes."
                else:
                    ownership_narrative = f"The equity profile is highly institutionalized with {inst_own:.2f}% controlled by major asset managers, while structural insider ownership sits at a lean {insider_own:.2f}%. Capital allocation decisions will be heavily policed by external institutional blocks."

                # Synthesis Construction
                briefing_text = f"""### **INVESTMENT MEMORANDUM**
                **Ticker Target:** {ticker_symbol} | **Generated:** {pd.Timestamp.now().strftime('%B %d, %Y')}

                ---

                #### **1. Core Valuation & Pricing Discrepancy**
                {valuation_narrative} {peer_text} the asset's structural return profile requires consistent capital efficiency to sustain its trading multiple. If the default projected cash flow compound annual growth rate holds true, current price levels represent a calculated entry window for risk-adjusted accounts.

                #### **2. Market Microstructure & Ownership Alignment**
                {ownership_narrative} Recent regulatory filings indicate that the supply dynamics are tightly controlled by long-term capital allocators, reducing the probability of erratic, retail-driven liquidity drawdowns.

                #### **3. Strategic Risk Execution Guidance**
                * **Capital Allocation:** If deploying capital under a conservative risk mandate, entries should ideally be scaled using dynamic positioning or dollar-cost averaging around historical low-volatility seasonal entry windows.
                * **Structural Invalidation Trigger:** A material degradation in the underlying free cash flow margins or a sudden spike in the structural debt-to-equity ratio relative to the peer group will automatically invalidate the current valuation thesis.
                """

                # 3. Render the output text cleanly in an institutional text layout
                st.info("### CIO Synthesis Brief")
                st.markdown(briefing_text)

                # Add a functional copy button for workflow convenience
                st.text_area("Raw Text Export (Click inside to select all and copy):", briefing_text, height=250)

            except (KeyError, IndexError) as e:
                # comp_df access (e.g. iloc[1]['Ticker']) is the one place here
                # that indexes into data built by an earlier section rather than
                # reading already-guarded standardized/locals() values.
                st.error(f"Could not construct the text briefing — unexpected shape in the peer comparison data: {e}")
            except Exception as e:
                log_exception(logger, "calc.error", section="executive_briefing", ticker=ticker_symbol)
                st.error(f"Unexpected error constructing the text briefing: {type(e).__name__}: {e}")

    with tab_risk:
        # ==========================================
        # PATH 7: PORTFOLIO CORRELATION & DIVERSIFICATION
        # ==========================================
        st.markdown("---")
        st.header("Path 7: Portfolio Correlation & Diversification")
        st.caption("How the current ticker and a user-defined basket move together — a real portfolio risk view, not just a single-ticker one.")

        basket_tickers = [t.strip().upper() for t in portfolio_basket_input.split(",") if t.strip()]
        if ticker_symbol not in basket_tickers:
            basket_tickers = [ticker_symbol] + basket_tickers

        if len(basket_tickers) < 2:
            st.info("Add at least one more ticker to the sidebar's \"Correlation Basket\" to see correlation and diversification metrics.")
        else:
            portfolio_price_histories = {}
            portfolio_fetch_errors = {}
            for basket_ticker in basket_tickers:
                if basket_ticker == ticker_symbol:
                    # Already fetched and cleaned for the main analysis — reuse it
                    # rather than fetching the same ticker's price history twice.
                    portfolio_price_histories[basket_ticker] = df
                    continue
                history, errors = load_price_history_only(basket_ticker, start_date, end_date)
                if errors:
                    portfolio_fetch_errors[basket_ticker] = "; ".join(errors)
                elif not history.empty:
                    # Run through the same cleaning pipeline as the main ticker
                    # (price_processing.py) — not just for consistency, but
                    # because it's what normalizes the tz-aware index yfinance
                    # returns into the tz-naive one `df` (the main ticker) is
                    # already on; without this, aligning returns across tickers
                    # raises on the mismatched index types.
                    history = process_price_data(history, ticker=basket_ticker).df
                portfolio_price_histories[basket_ticker] = history

            alignment = build_aligned_returns(portfolio_price_histories, lookback=portfolio_lookback)

            if alignment.excluded_tickers:
                reasons_text = "; ".join(f"{t}: {alignment.exclusion_reasons.get(t, 'unknown reason')}" for t in alignment.excluded_tickers)
                st.warning(f"Excluded from the basket (no usable data): {reasons_text}")

            if not alignment.sufficient_data or len(alignment.included_tickers) < 2:
                st.info(f"Not enough overlapping trading days across the basket to compute correlation/diversification reliably (need at least {RISK.correlation_min_observations}, have {alignment.observation_count}). Try a shorter lookback or a different basket.")
            else:
                corr_matrix = compute_correlation_matrix(alignment.returns)
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index,
                    colorscale='RdBu', zmid=0, zmin=-1, zmax=1,
                    text=corr_matrix.round(2).values, texttemplate="%{text}",
                    colorbar=dict(title="Correlation"),
                ))
                fig_corr.update_layout(height=350 + 20 * len(alignment.included_tickers), template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=30))
                st.caption(f"Pairwise correlation over the last {portfolio_lookback} trading days")
                st.plotly_chart(fig_corr, width="stretch")

                diversification = compute_portfolio_diversification(alignment.returns)
                if diversification is not None:
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Portfolio Volatility", f"{diversification.portfolio_volatility * 100:.2f}%", help="Equal-weighted portfolio's annualized volatility, accounting for how the holdings move together.")
                    d2.metric("Weighted-Average Volatility", f"{diversification.weighted_average_volatility * 100:.2f}%", help="What the portfolio's volatility would be if every holding moved completely independently — no diversification credit.")
                    d3.metric("Diversification Benefit", f"{diversification.diversification_benefit * 100:.2f}pp", help="Weighted-Average minus Portfolio Volatility — the risk reduction this basket actually gets from not being perfectly correlated.")
                    if diversification.diversification_ratio is not None:
                        st.caption(f"Diversification ratio: {diversification.diversification_ratio:.2f}× — a basket with zero correlation benefit would read 1.00×; this basket's correlation structure lowers portfolio risk to {1 / diversification.diversification_ratio:.0%} of what the un-diversified weighted average would be.")

                st.markdown("##")
                st.subheader("Efficient Frontier")
                st.caption(
                    "Markowitz mean-variance optimization over the same basket above: each point on the curve is the "
                    "lowest possible volatility achievable for that target return, long-only with weights summing to "
                    "100%. The white diamond is this basket's current equal-weighted portfolio — the exact same one "
                    "the Diversification metrics above evaluate — for direct reference against what's actually optimal."
                )
                frontier = compute_efficient_frontier(alignment.returns)
                if frontier is None:
                    st.info("Not enough data to compute the efficient frontier for this basket (the optimizer didn't converge).")
                else:
                    fig_frontier = go.Figure()
                    fig_frontier.add_trace(go.Scatter(
                        x=[v * 100 for v in frontier.frontier_volatilities], y=[r * 100 for r in frontier.frontier_returns],
                        mode='lines', name='Efficient Frontier', line=dict(color='cyan', width=2),
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[frontier.equal_weighted.volatility * 100], y=[frontier.equal_weighted.expected_return * 100],
                        mode='markers', name='Equal-Weighted (current basket)', marker=dict(color='white', size=13, symbol='diamond'),
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[frontier.max_sharpe.volatility * 100], y=[frontier.max_sharpe.expected_return * 100],
                        mode='markers', name='Max Sharpe', marker=dict(color='lime', size=13, symbol='star'),
                    ))
                    fig_frontier.add_trace(go.Scatter(
                        x=[frontier.min_variance.volatility * 100], y=[frontier.min_variance.expected_return * 100],
                        mode='markers', name='Min Variance', marker=dict(color='orange', size=13, symbol='square'),
                    ))
                    fig_frontier.update_layout(
                        xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)",
                        height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="closest",
                    )
                    st.plotly_chart(fig_frontier, width="stretch")

                    w1, w2 = st.columns(2)
                    with w1:
                        sharpe_label = f"{frontier.max_sharpe.sharpe_ratio:.2f}" if frontier.max_sharpe.sharpe_ratio is not None else "N/A"
                        st.markdown(f"**Max-Sharpe Portfolio** — Return {frontier.max_sharpe.expected_return*100:.2f}% · Vol {frontier.max_sharpe.volatility*100:.2f}% · Sharpe {sharpe_label}")
                        st.dataframe(
                            pd.DataFrame({"Weight": [f"{w*100:.1f}%" for w in frontier.max_sharpe.weights.values()]}, index=list(frontier.max_sharpe.weights.keys())),
                            width="stretch",
                        )
                    with w2:
                        st.markdown(f"**Min-Variance Portfolio** — Return {frontier.min_variance.expected_return*100:.2f}% · Vol {frontier.min_variance.volatility*100:.2f}%")
                        st.dataframe(
                            pd.DataFrame({"Weight": [f"{w*100:.1f}%" for w in frontier.min_variance.weights.values()]}, index=list(frontier.min_variance.weights.keys())),
                            width="stretch",
                        )

        # ==========================================
        # MACHINE LEARNING PIPELINE: MOMENTUM CONTINUATION SIGNAL
        # ==========================================
        # The one PREDICTIVE section in an otherwise entirely descriptive,
        # rule-based app. Kept visually and textually distinct from every
        # section above it for exactly that reason.
        st.markdown("---")
        st.header("Momentum Continuation Signal (Experimental)")
        st.caption(
            f"A baseline Logistic Regression trained on this app's own technical/risk features (RSI, MACD, SMA "
            f"structure, trailing returns, volatility, volume), estimating the probability {ticker_symbol}'s price "
            f"is higher {ML_PIPELINE.label_horizon_days} trading days from now. This is a backtested statistical "
            f"estimate from one specific, disclosed model — not investment advice, and not a claim of a proven "
            f"edge. Predicting price direction from public technical data alone is a genuinely hard problem; every "
            f"result below is shown next to the naive \"always guess the majority class\" baseline so you can "
            f"judge whether the model is actually adding anything."
        )

        ml_model = load_ml_model()
        ml_history = load_ml_history()

        if st.button("Train / Retrain Model", key="ml_train_button"):
            with st.spinner(f"Training on {len(training_universe())} ticker(s) — fetching history, engineering features, fitting..."):
                trained_model, training_result = train_momentum_model()
                if training_result.error:
                    st.error(f"Training did not produce a usable model: {training_result.error}")
                else:
                    save_ml_model(trained_model, training_result)
                    log_event(logger, logging.INFO, "user.ml_model_trained", test_accuracy=round(training_result.test_accuracy, 4), tickers=len(training_result.tickers_used))
                    st.rerun()

        if ml_model is None:
            st.info("No model has been trained yet. Click \"Train / Retrain Model\" above to train the first one.")
        else:
            _ml_latest = ml_history[-1] if ml_history else None
            if _ml_latest:
                _ml_beats_baseline = _ml_latest["test_accuracy"] > _ml_latest["majority_class_baseline_accuracy"]
                ml_c1, ml_c2, ml_c3 = st.columns(3)
                ml_c1.metric(
                    "Model Test Accuracy", f"{_ml_latest['test_accuracy']*100:.1f}%",
                    delta=f"{(_ml_latest['test_accuracy'] - _ml_latest['majority_class_baseline_accuracy'])*100:+.1f}pp vs naive baseline",
                    # "normal" (not "inverse"): a POSITIVE delta means the
                    # model beats the baseline (good, green) and a NEGATIVE
                    # delta means it underperforms (bad, red) — exactly
                    # st.metric's default semantics. Caught live: an
                    # earlier "inverse" here painted a -3.0pp
                    # underperformance GREEN, directly contradicting the
                    # warning text right below it.
                )
                ml_c2.metric("Naive Majority-Class Baseline", f"{_ml_latest['majority_class_baseline_accuracy']*100:.1f}%", help="Accuracy from simply always predicting whichever label (up/down) was more common in the test period — the bar a model has to clear to be adding anything at all.")
                ml_c3.metric("Test ROC-AUC", f"{_ml_latest['test_roc_auc']:.3f}" if _ml_latest.get('test_roc_auc') is not None else "N/A", help="0.5 = no better than random ranking; 1.0 = perfect. Shown alongside accuracy since accuracy alone can be misleading on imbalanced labels.")

                if not _ml_beats_baseline:
                    st.warning(
                        "This model's out-of-sample accuracy does NOT currently exceed the naive majority-class "
                        "baseline — on the most recent training run, it is not demonstrating a measurable edge. "
                        "The prediction below is shown anyway for transparency, not as a signal to act on."
                    )

                st.caption(
                    f"Last trained {_ml_latest['trained_at']} on {_ml_latest['train_rows']:,} rows "
                    f"({_ml_latest['train_start']} → {_ml_latest['train_end']}), tested on {_ml_latest['test_rows']:,} rows "
                    f"({_ml_latest['test_start']} → {_ml_latest['test_end']}) across {len(_ml_latest['tickers_used'])} ticker(s)."
                )

            _ml_prediction = predict_latest(ticker_symbol, df, ml_model)
            if _ml_prediction.status == "ok":
                _ml_prob_pct = _ml_prediction.probability_up * 100
                st.metric(
                    f"{ticker_symbol}: Probability Higher in {ML_PIPELINE.label_horizon_days} Trading Days",
                    f"{_ml_prob_pct:.1f}%",
                    help="As of the most recent complete bar. 50% means the model sees no directional lean either way for this specific ticker right now.",
                )
                with st.expander("Feature values behind this prediction", expanded=False):
                    st.table(pd.DataFrame({"Value": _ml_prediction.feature_values}))
            elif _ml_prediction.status == "insufficient_data":
                st.caption(f"No prediction for {ticker_symbol}: {_ml_prediction.detail}")

            if ml_history:
                with st.expander(f"Training history ({len(ml_history)} run(s))", expanded=False):
                    _ml_hist_rows = [
                        {
                            "Trained": h["trained_at"], "Test Accuracy": f"{h['test_accuracy']*100:.1f}%",
                            "Baseline": f"{h['majority_class_baseline_accuracy']*100:.1f}%",
                            "ROC-AUC": f"{h['test_roc_auc']:.3f}" if h.get("test_roc_auc") is not None else "N/A",
                            "Train Rows": h["train_rows"], "Test Rows": h["test_rows"], "Tickers": len(h["tickers_used"]),
                        }
                        for h in reversed(ml_history)
                    ]
                    st.dataframe(pd.DataFrame(_ml_hist_rows), hide_index=True, width="stretch")

    with tab_tearsheet:
        # ==========================================
        # PRINTABLE TEAR SHEET (HTML/CSS)
        # ==========================================
        st.markdown("---")
        st.header("Quantitative Tear Sheet")
        st.markdown("Press **Command + P** (Mac) or **Ctrl + P** (Windows) to save this final report as a PDF, or use the **Generate PDF** button below for a direct download.")

        # 1. Safely extract variables
        _intrinsic = intrinsic_price if 'intrinsic_price' in locals() else 0.0
        _mos = margin_of_safety if 'margin_of_safety' in locals() else 0.0
        _kelly = final_allocation if 'final_allocation' in locals() else 0.0
        _altman_display = fmt_num(altman_z)

        # 2. Determine the CIO Verdict
        if macro_risk_flag:
            verdict = "STRONG AVOID"
            verdict_color = "#dc2626"
            reason = f"Systemic market risk (VIX > {RISK.vix_high_risk_threshold:.0f}). Capital preservation prioritized over individual asset alpha."
        elif score_pct >= TEAR_SHEET.strong_buy_min_score_pct and _mos > TEAR_SHEET.strong_buy_min_margin_of_safety:
            verdict = "STRONG BUY"
            verdict_color = "#16a34a"
            reason = "Elite institutional fundamentals and trading at an attractive valuation."
        elif score_pct >= TEAR_SHEET.strong_buy_min_score_pct and _mos <= TEAR_SHEET.strong_buy_min_margin_of_safety:
            verdict = "HOLD (PREMIUM)"
            verdict_color = "#2563eb"
            reason = "Flawless underlying business, but the market is pricing it at a massive premium. Wait for a technical pullback."
        elif score_pct >= TEAR_SHEET.hold_watchlist_min_score_pct and _mos > TEAR_SHEET.hold_watchlist_min_margin_of_safety:
            verdict = "HOLD / WATCHLIST"
            verdict_color = "#ca8a04"
            reason = "Stable underlying business but currently commands a high market premium."
        else:
            verdict = "AVOID"
            verdict_color = "#dc2626"
            reason = "Fails fundamental safety checks, carries extreme debt, or is severely overvalued relative to cash flow."

        # --- DEFINE DATE AND LOGO ---
        report_date = datetime.date.today().strftime("%B %d, %Y")

        website = standardized.website or ''
        domain = website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
        logo_html = f'<img src="https://logo.clearbit.com/{domain}" onerror="this.style.display=\'none\'" style="height: 55px; width: 55px; object-fit: contain; margin-right: 20px; border-radius: 8px; border: 1px solid #e2e8f0; padding: 2px; background: white;">' if domain else ''

        # 3. Build the HTML Template
        tear_sheet_html = f"""
        <div class="tear-sheet">
            <div class="ts-top-accent"></div>
            <div class="ts-header">
                <div style="display: flex; align-items: center;">
                    {logo_html}
                    <div>
                        <div style="font-size: 0.8rem; font-weight: 700; color: #3b82f6; letter-spacing: 2px; margin-bottom: 4px;">POWERED BY QUANTIX</div>
                        <h1 style="margin:0; font-size: 2.5rem; color: #0f172a; letter-spacing: -1px;">{ticker_symbol}</h1>
                        <p style="margin:4px 0 0 0; color: #64748b; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 1px;">Institutional Tear Sheet • {report_date}</p>
                    </div>
                </div>
                <div style="text-align: right;">
                    <h2 style="margin:0; font-size: 2.2rem; color: #0f172a;">${current_price:.2f}</h2>
                    <span style="background-color: {verdict_color}; color: white; padding: 6px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-top: 8px; letter-spacing: 0.5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">{verdict}</span>
                </div>
            </div>

            <div class="ts-section">
                <h3 class="ts-title">Chief Investment Officer Thesis</h3>
                <p style="font-size: 1.05rem; color: #1e293b; line-height: 1.7;"><strong>Primary Driver:</strong> {reason}</p>
                <p style="font-size: 0.95rem; color: #475569; line-height: 1.6; border-left: 3px solid #e2e8f0; padding-left: 15px; margin-top: 15px;">{(standardized.business_summary or 'Business summary not available.')[:450]}...</p>
            </div>

            <div class="ts-grid">
                <div class="ts-card">
                    <h4>Valuation & DCF</h4>
                    <div class="ts-metric"><span class="ts-label">Intrinsic Value</span> <span class="ts-value">${_intrinsic:.2f}</span></div>
                    <div class="ts-metric"><span class="ts-label">Margin of Safety</span> <span class="ts-value" style="color: {'#16a34a' if _mos > 0 else '#dc2626'};">{_mos:.2f}%</span></div>
                    <div class="ts-metric"><span class="ts-label">FCF Yield</span> <span class="ts-value">{fmt_num(fcf_yield_val, "%")}</span></div>
                </div>
                <div class="ts-card">
                    <h4>Fundamental Health</h4>
                    <div class="ts-metric"><span class="ts-label">Blueprint Score</span> <span class="ts-value">{score_pct:.0f}%</span></div>
                    <div class="ts-metric"><span class="ts-label">Net Margin</span> <span class="ts-value">{fmt_num(None if net_margin is None else net_margin * 100, "%", decimals=1)}</span></div>
                    <div class="ts-metric"><span class="ts-label">Altman Z-Score</span> <span class="ts-value">{_altman_display}</span></div>
                </div>
                <div class="ts-card">
                    <h4>Execution & Risk</h4>
                    <div class="ts-metric"><span class="ts-label">Kelly-Style Sizing</span> <span class="ts-value">{_kelly:.2f}%</span></div>
                    <div class="ts-metric"><span class="ts-label">Z-Score (Trend)</span> <span class="ts-value">{current_z_score:.2f}</span></div>
                    <div class="ts-metric"><span class="ts-label">1-Day VaR ({var_confidence:.0%})</span> <span class="ts-value">{f"{historical_var * 100:.2f}%" if historical_var is not None else "N/A"}</span></div>
                </div>
            </div>

            <div class="ts-footer">
                <p>Generated by <strong>Quantix Terminal</strong> | Algorithmic execution carries inherent risk. Verify all execution parameters via broker.</p>
            </div>
        </div>

        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            .tear-sheet {{
                background-color: #ffffff;
                color: #0f172a;
                padding: 40px 50px;
                border-radius: 12px;
                margin-top: 20px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
                position: relative;
                overflow: hidden;
                border: 1px solid #e2e8f0;
            }}
            .ts-top-accent {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 6px;
                background: linear-gradient(90deg, #3b82f6, #0f172a); /* Updated gradient to match Quantix blue */
            }}
            .ts-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #f1f5f9;
                padding-bottom: 25px;
                margin-bottom: 25px;
            }}
            .ts-section {{ margin-bottom: 35px; }}
            .ts-title {{
                border-bottom: 2px solid #e2e8f0;
                padding-bottom: 10px;
                color: #0f172a;
                text-transform: uppercase;
                font-size: 0.85rem;
                letter-spacing: 1.5px;
                margin-bottom: 15px;
            }}
            .ts-grid {{
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 25px;
                margin-bottom: 30px;
            }}
            .ts-card {{
                background-color: #f8fafc;
                padding: 20px;
                border-top: 3px solid #0f172a;
                border-radius: 0 0 8px 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            }}
            .ts-card h4 {{
                margin-top: 0;
                color: #475569;
                margin-bottom: 15px;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 8px;
            }}
            .ts-metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                font-size: 0.95rem;
            }}
            .ts-label {{ color: #64748b; }}
            .ts-value {{ font-weight: 600; color: #0f172a; }}
            .ts-footer {{
                text-align: center;
                font-size: 0.75rem;
                color: #94a3b8;
                border-top: 1px solid #f1f5f9;
                padding-top: 20px;
                letter-spacing: 0.5px;
            }}
            .ts-footer strong {{ color: #3b82f6; font-weight: 700; }}

            @media print {{
                body * {{ visibility: hidden; }}
                .tear-sheet, .tear-sheet * {{ visibility: visible; }}
                .tear-sheet {{
                    position: absolute;
                    left: 0;
                    top: 0;
                    width: 100%;
                    padding: 0;
                    box-shadow: none;
                    border: none;
                }}
                .ts-top-accent {{ display: none; }}
                header, .stSidebar, .stApp > header, footer {{ display: none !important; }}
            }}
        </style>
        """
        st.html(tear_sheet_html)

        if st.button("Generate PDF Report"):
            with st.spinner("Rendering PDF..."):
                pdf_bytes, pdf_error = generate_tear_sheet_pdf(tear_sheet_html)
            if pdf_bytes is not None:
                st.session_state["_tear_sheet_pdf"] = {"ticker": ticker_symbol, "bytes": pdf_bytes}
            else:
                st.session_state.pop("_tear_sheet_pdf", None)
                st.warning(pdf_error)

        cached_pdf = st.session_state.get("_tear_sheet_pdf")
        if cached_pdf and cached_pdf["ticker"] == ticker_symbol:
            st.download_button(
                "Download PDF",
                data=cached_pdf["bytes"],
                file_name=f"{ticker_symbol}_tear_sheet_{datetime.date.today().isoformat()}.pdf",
                mime="application/pdf",
            )


# ==========================================
# DIAGNOSTICS: IN-APP LOG VIEWER
# ==========================================
# Deliberately outside the `if df.empty` branch above so it still renders when
# data loading fails — that's exactly when the log is most useful. Placed last
# so it captures every event emitted during this run.
if debug_mode:
    st.markdown("---")
    st.subheader("Diagnostics — Recent Log")
    st.caption(f"Newest last · also written to `{log_file_path()}`")
    entries = recent_logs(limit=200)
    if entries:
        st.code("\n".join(entries), language="log")
    else:
        st.info("No log entries captured yet this session.")
