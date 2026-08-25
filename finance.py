import dataclasses
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
import export_deck
import export_workbook
from email_report import is_email_configured, send_notification_email, send_report_email
from data_quality import assess_data_quality
import data_quality
from config import WATCHLIST, SCORECARD, DCF, RISK, MONTE_CARLO, CHART_DEFAULTS, PEER_DEFAULTS, TEAR_SHEET, TECHNICAL, WALK_FORWARD, BACKTEST_COST, WATCHLIST_PANEL, REALTIME_ALERTS, PORTFOLIO_BACKTEST, ML_PIPELINE, SCENARIO_MODELING, COMPETITIVE_BENCHMARKING, EMAIL_REPORT, FAVORITES, API_KEYS, SUPPORT, DIGEST, PORTFOLIO, NEWS_SENTIMENT, RECOMMENDATIONS
from metric_help import chart_help, help_for
from ticker_search import (
    build_universe as ts_build_universe,
    search_symbols as ts_search_symbols,
    suggest_alternatives as ts_suggest_alternatives,
    symbol_from_label as ts_symbol_from_label,
)
from historical_comparison import (
    available_range as hc_available_range,
    build_comparison as hc_build_comparison,
)
from collaboration import (
    add_member as collab_add_member,
    add_note as collab_add_note,
    delete_note as collab_delete_note,
    load_store as collab_load_store,
    mark_notified as collab_mark_notified,
    notes_for as collab_notes_for,
    notify_mentions as collab_notify_mentions,
    remove_member as collab_remove_member,
    save_store as collab_save_store,
)
from user_thresholds import (
    EDITABLE as THRESHOLD_SPECS,
    EDITABLE_BY_KEY as THRESHOLD_SPECS_BY_KEY,
    effective_risk,
    effective_scorecard,
    effective_sector_pe as threshold_effective_sector_pe,
    effective_values as threshold_effective_values,
    load_overrides as load_threshold_overrides,
    load_sector_pe as load_threshold_sector_pe,
    reset_all as reset_thresholds,
    validate as validate_thresholds,
    save_overrides as save_threshold_overrides,
    save_sector_pe as save_threshold_sector_pe,
)
from favorites import (
    is_favorite,
    load_store as load_quick_access,
    quick_access_chips,
    save_store as save_quick_access,
    toggle_favorite,
)
from fundamental_analysis import FundamentalAnalysisEngine
from logging_setup import setup_logging, get_logger, log_event, log_exception, recent_logs, log_file_path
from screener import METRICS as SCREENER_METRICS, METRICS_BY_KEY as SCREENER_METRICS_BY_KEY, OPERATORS as SCREENER_OPERATORS, MAX_UNIVERSE_SIZE as SCREENER_MAX_UNIVERSE_SIZE, ScreenCriterion, run_screen
import screener as screener_module
import screener_templates
import quick_stats
import profile_menu
import alignment_card
import ticker_discovery as td
import empty_states
import button_roles
import keyboard_shortcuts
import date_range
import notifications
import loading_states
import asset_class
import asset_views
import etf_analysis
import etf_comparison
import etf_pipeline
import etf_risk
import bond_data
import bond_market
import bond_screener
import etf_technicals
import etf_screener
import streamlit.components.v1 as components
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
from scenario_modeling import (
    SCENARIO_TYPE_LABELS,
    SCENARIO_TYPES,
    ScenarioDefinition,
    apply_risk_scenario,
    default_scenario,
    delete_scenario,
    dividend_cut_impact,
    load_scenarios,
    run_scenario,
    save_scenario,
)
from competitive_benchmarking import METRICS, build_benchmark_rows, build_peer_metrics
from onboarding import STEPS as ONBOARDING_STEPS, has_completed_onboarding, mark_onboarding_done
from sector_percentile import MIN_PEERS as SECTOR_MIN_PEERS, compute_sector_percentiles, format_percentile
from risk_alerts import effective_default_threshold as rt_effective_default_threshold, METRICS as RISK_ALERT_METRICS, METRICS_BY_KEY as RISK_ALERT_METRICS_BY_KEY, OPERATORS as RISK_ALERT_OPERATORS, AlertRule, compute_watchlist_snapshots, evaluate_alerts, load_rules, save_rules, watchlist_tickers
from realtime_alerts import (
    ALL_TRIGGER_TYPES as RT_ALL_TRIGGER_TYPES,
    FUNDAMENTAL_TRIGGER_TYPE as RT_FUNDAMENTAL_TRIGGER_TYPE,
    PRICE_TRIGGER_TYPES as RT_PRICE_TRIGGER_TYPES,
    TRIGGER_LABELS as RT_TRIGGER_LABELS,
    FIRST_ALERT_TRIGGER as RT_FIRST_ALERT_TRIGGER,
    CATEGORY_NAMES as RT_CATEGORY_NAMES,
    category_of as rt_category_of,
    triggers_in as rt_triggers_in,
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
from watchlist_panel import (
    add_ticker,
    create_watchlist,
    delete_watchlist,
    load_quote_snapshots,
    load_watchlist_store,
    parse_tickers,
    record_recent,
    remove_ticker,
    rename_watchlist,
    save_watchlist_store,
    set_active_watchlist,
    update_active_tickers,
)
from theme import PALETTES, load_theme, save_theme
# Importing auth is what switches every local store from the shared files
# to the signed-in user's namespace: auth.py registers itself with
# local_store on import. It must therefore be imported before anything
# reads a store, which the import block guarantees.
import auth
from branding import (
    apply_accent as apply_brand_accent,
    brand,
    configuration_notes as brand_config_notes,
    rebrand,
    summary as brand_summary,
)
from alert_watch import run as alert_watch_run
from slack_notify import unavailable_reason as slack_unavailable_reason
from realtime_alerts import load_store as load_rt_store
from recommendations import (
    Preferences as rc_Preferences,
    available_sectors as rc_available_sectors,
    criteria_for as rc_criteria_for,
    rank as rc_rank,
)
from news_sentiment import (
    accuracy_summary as ns_accuracy_summary,
    analyse as ns_analyse,
)
from portfolio_holdings import (
    add_holding as pf_add_holding,
    create_portfolio as pf_create_portfolio,
    delete_portfolio as pf_delete_portfolio,
    portfolio_names as pf_portfolio_names,
    rename_portfolio as pf_rename_portfolio,
    set_active_portfolio as pf_set_active_portfolio,
    build_performance as pf_build_performance,
    load_store as pf_load_store,
    remove_holding as pf_remove_holding,
    save_store as pf_save_store,
)
from digest import (
    DigestSettings,
    build_digest as digest_build,
    cron_line as digest_cron_line,
    save_settings as digest_save_settings,
    settings_for as digest_settings_for,
    validate as digest_validate,
)
from support import (
    build_index as support_build_index,
    compose_report as support_compose,
    diagnostics_snapshot as support_diagnostics,
    is_destination_configured as support_destination_configured,
    search as support_search,
    send_report as support_send,
)

# The questions shown before anyone types — the ones the app's own design
# provokes most often, rather than an arbitrary first-N slice of the FAQ.
SUPPORT_STARTERS = (
    "faq_unavailable_metric",
    "faq_settings_vanished",
    "faq_restart_needed",
    "faq_alignment_score",
    "faq_where_is_my_data",
)
from api_keys import (
    DEFAULT_SCOPES as API_KEY_DEFAULT_SCOPES,
    SCOPES as API_SCOPES,
    create_key as create_api_key,
    keys_for_owner as api_keys_for_owner,
    load_store as load_api_key_store,
    revoke_key as revoke_api_key,
    save_store as save_api_key_store,
)
import brand_assets

# --- Page Configuration ---
# First Streamlit call in the file, deliberately. Streamlit requires
# set_page_config to precede any command that affects the page, and while
# it tolerated the session_state read the logging setup does below, that
# is a tolerance rather than a guarantee — putting it here removes the
# question. Page title and icon are read at import time, so they are the
# first thing a browser tab shows rather than appearing a beat later.
#
# The icon is the mark with its white export card keyed out: a browser
# tab's background is the browser's colour, not ours, so a boxed variant
# shows as a white tile in dark mode. mark_image() returns a PIL image
# rather than a path — page_icon takes anything st.image does — so nothing
# derived gets written into the designer's source folder.
#
# Falls back to the raw file, then to None, which is exactly Streamlit's
# "use the default": a missing asset costs the favicon and nothing else.
_page_icon = brand_assets.mark_image()
if _page_icon is None and brand_assets.mark() is not None:
    _page_icon = str(brand_assets.mark())
st.set_page_config(
    page_title="Quantix | Institutional Analysis",
    page_icon=_page_icon,
    layout="wide",
)


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


# --- Page Configuration --- moved to the top of the file; see above.

# ==========================================
# IDENTITY SWITCH GUARD
# ==========================================
# Signing in or out changes which files every store reads (see auth.py),
# but a dozen stores are cached in session_state so they're loaded once
# per session rather than once per rerun. Without this, signing in would
# leave the previous profile's watchlist, theme, favourites, thresholds
# and alert rules on screen — showing one user another user's data, which
# is precisely the failure per-user scoping exists to prevent.
#
# Placed here, above every one of those `if key not in session_state`
# loads, so purging is enough on its own: each store simply reloads from
# the new namespace further down this same run. No rerun needed, and no
# window in which stale data is visible.
_AUTH_SCOPED_STATE = (
    "theme_choice", "watchlist_store", "quick_access_store", "scenario_saved",
    "risk_alert_rules", "rt_alert_rules", "rt_alert_history",
    "onboarding_active", "onboarding_step", "collab_store",
    # The key STORE is shared, but the panel lists only the current
    # owner's keys — so the cached copy still has to be re-read when the
    # identity changes, or you'd see the previous account's key list.
    "api_key_store",
    # Digest settings are per-owner records in a shared store, so the
    # cached copy still has to be re-read when the identity changes.
    "digest_settings",
    "portfolio_store",
)
_auth_user = auth.current_user()
_auth_namespace = _auth_user.key if _auth_user else ""
if st.session_state.get("_auth_namespace", _auth_namespace) != _auth_namespace:
    for _k in _AUTH_SCOPED_STATE:
        st.session_state.pop(_k, None)
    log_event(logger, logging.INFO, "auth.namespace_switched",
              signed_in=bool(_auth_namespace))
st.session_state["_auth_namespace"] = _auth_namespace

# ==========================================
# ONBOARDING (first-run walkthrough)
# ==========================================
# Rendered before anything ticker-specific, since it doesn't depend on any
# ticker data at all — a first-time visitor sees it before picking a
# symbol. See onboarding.py's module docstring for why this is a native
# step panel (Next/Back/Skip) rather than a spotlight-style tour: a real,
# already-proven Streamlit constraint in this codebase, not a stylistic
# choice.
if "onboarding_active" not in st.session_state:
    st.session_state["onboarding_active"] = not has_completed_onboarding()
if "onboarding_step" not in st.session_state:
    st.session_state["onboarding_step"] = 0

if st.session_state["onboarding_active"]:
    _ob_step_idx = st.session_state["onboarding_step"]
    _ob_step = ONBOARDING_STEPS[_ob_step_idx]
    with st.container(border=True):
        st.caption(f"Getting Started · Step {_ob_step_idx + 1} of {len(ONBOARDING_STEPS)}")
        st.subheader(_ob_step.title)
        st.markdown(_ob_step.body)
        st.progress((_ob_step_idx + 1) / len(ONBOARDING_STEPS))

        _ob_back_col, _ob_next_col, _ob_skip_col = st.columns([1, 1, 1])
        with _ob_back_col:
            if st.button("← Back", key="onboarding_back", disabled=(_ob_step_idx == 0), width="stretch"):
                st.session_state["onboarding_step"] -= 1
                st.rerun()
        with _ob_next_col:
            _ob_is_last = _ob_step_idx == len(ONBOARDING_STEPS) - 1
            if st.button("Finish" if _ob_is_last else "Next →", key="onboarding_next", type="primary", width="stretch"):
                if _ob_is_last:
                    mark_onboarding_done(skipped=False)
                    st.session_state["onboarding_active"] = False
                    log_event(logger, logging.INFO, "user.onboarding_finished")
                else:
                    st.session_state["onboarding_step"] += 1
                st.rerun()
        with _ob_skip_col:
            if st.button("Skip Tutorial", key="onboarding_skip", width="stretch"):
                mark_onboarding_done(skipped=True)
                st.session_state["onboarding_active"] = False
                log_event(logger, logging.INFO, "user.onboarding_skipped", at_step=_ob_step_idx)
                st.rerun()
    st.markdown("---")

# Sticky symbol header slot. Reserved HERE, at the very top, so the
# ticker/price/day-change is on screen the moment the page loads — but it
# can only be FILLED after the data fetch further down, so it uses the
# same container-as-placeholder pattern executive_digest_container already
# uses (content written into a container later still renders at the
# container's position). See the "SYMBOL HEADER (fill)" block below.
# st.empty(), NOT st.container(): a container APPENDS, so a skeleton
# written here would still be on screen underneath the real header. An
# empty slot holds one thing and replaces it, and .container() on it
# gives a replaceable slot that can hold the several elements the header
# needs. This slot sits visibly empty for the ~10s the ticker bundle
# takes, which is the gap the skeleton covers.
symbol_header_container = st.empty()

# --- Theme (dark/light) ---
# Loaded from the persisted local preference the first time this session
# touches it; the actual toggle widget lives in the sidebar's System tab,
# far below this point in the script — but Streamlit keeps a keyed
# widget's value in session_state across reruns, so seeding it here (the
# same "read before the widget renders" trick already used for
# debug_mode / onboarding_active elsewhere in this file) makes the loaded
# preference available for the CSS injection immediately below, on the
# very first render of a given session, not just after the widget itself
# has rendered once.
if "theme_choice" not in st.session_state:
    st.session_state["theme_choice"] = load_theme()
# A licensee's accent colour replaces the identity-carrying palette
# fields; the contrast-tuned remainder is untouched (see branding.py).
_theme = apply_brand_accent(PALETTES[st.session_state["theme_choice"]])
_plotly_template = _theme.plotly_template
_chart_fg = _theme.chart_fg
_chart_faint_line = _theme.chart_faint_line

# --- Professional UI Injection (OLED Edition, theme-aware) ---
st.markdown(f"""
    <style>
    /* Hide Streamlit default UI elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    /* ...but keep the sidebar re-open button visible. It lives inside the
       header, so hiding the header above would otherwise make collapsing the
       sidebar irreversible without reloading the page. stExpandSidebarButton
       is the Streamlit 1.58 test ID; the other two are older/newer aliases,
       matched so this keeps working across version upgrades. */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        z-index: 999999;
    }}
    [data-testid="stExpandSidebarButton"] *,
    [data-testid="stSidebarCollapsedControl"] *,
    [data-testid="collapsedControl"] * {{
        visibility: visible !important;
        color: {_theme.sidebar_icon} !important;
        fill: {_theme.sidebar_icon} !important;
    }}

    /* App background */
    .stApp {{
        background-color: {_theme.app_bg} !important;
        color: {_theme.app_text};
    }}

    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {_theme.header_text} !important;
        font-weight: 600 !important;
    }}

    /* Metric Cards - High Contrast */
    div[data-testid="metric-container"] {{
        background-color: {_theme.card_bg};
        border: 1px solid {_theme.card_border};
        border-left: 4px solid {_theme.card_accent}; /* Neon Green Action Border */
        border-radius: 6px;
        padding: 15px 20px;
        transition: border-left-color 200ms ease, transform 200ms ease;
    }}

    /* Hover state */
    div[data-testid="metric-container"]:hover {{
        border-left: 4px solid {_theme.card_hover_accent};
        transform: scale(1.02);
    }}

    /* Pop-out values for readability */
    [data-testid="stMetricValue"] {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {_theme.metric_value};
    }}

    /* Metric caption line. Streamlit's own base theme is LIGHT, so left
       alone this renders in its light-theme label colour (#31333F) —
       measured at 1.68:1 against this theme's black canvas, i.e. barely
       visible. Set explicitly per theme instead. */
    [data-testid="stMain"] [data-testid="stMetricLabel"],
    [data-testid="stMain"] [data-testid="stMetricLabel"] * {{
        color: {_theme.metric_label} !important;
        /* Measured 8.27:1 against this canvas, so contrast was never the
           problem — 14px at weight 400 simply disappeared beside a 1.85rem
           value. Size and weight are what make a caption readable at a
           glance; going brighter instead would have flattened the
           deliberate hierarchy between label and figure. */
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }}

    /* --- Secondary buttons in the main area --------------------------
       Same root cause as the metric label above, but worse: Streamlit's
       light base theme gives these a WHITE face, while the .stApp rule
       further up cascades near-white text (#e2e8f0) onto them — measured
       at 1.23:1 in-browser, which is why they read as blank boxes until
       hover swapped in Streamlit's own dark hover colour. Painting both
       the face and the text from the palette fixes the whole class at
       once (chips, ✕ removes, + Add Filter/Rule, Download CSV, ...).

       Scoped to stMain deliberately: the sidebar keeps Streamlit's light
       chrome, where these same buttons already measure ~11.9:1 and look
       correct — restyling them there would break what already works.

       Both the data-testid and the kind attribute are matched so a
       Streamlit version that renames or drops either still leaves a
       working rule. */
    [data-testid="stMain"] button[data-testid="stBaseButton-secondary"],
    [data-testid="stMain"] button[kind="secondary"] {{
        background-color: {_theme.button_bg} !important;
        border: 1px solid {_theme.button_border} !important;
        color: {_theme.button_text} !important;
    }}
    [data-testid="stMain"] button[data-testid="stBaseButton-secondary"] *,
    [data-testid="stMain"] button[kind="secondary"] * {{
        color: {_theme.button_text} !important;
    }}
    [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover,
    [data-testid="stMain"] button[kind="secondary"]:hover {{
        background-color: {_theme.button_hover_bg} !important;
        border-color: {_theme.button_hover_border} !important;
        color: {_theme.button_hover_text} !important;
    }}
    [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:hover *,
    [data-testid="stMain"] button[kind="secondary"]:hover * {{
        color: {_theme.button_hover_text} !important;
    }}
    /* A disabled button must still read as disabled rather than just
       looking like a normal one, so it keeps the face but dims. */
    [data-testid="stMain"] button[data-testid="stBaseButton-secondary"]:disabled,
    [data-testid="stMain"] button[kind="secondary"]:disabled {{
        opacity: 0.45 !important;
    }}

    /* --- The active ticker -------------------------------------------
       "You are here" was Streamlit's stock primary red (#FF4B4B), which
       in a financial app is already the colour of a loss. The active
       watchlist row therefore rendered as "▼ AAPL · -0.63%" on a red
       pill, where the red meant "selected" and the ▼ meant "down" and
       nothing distinguished them — while "Send report" and "Create key"
       wore the identical red for an entirely different meaning.

       Selection is now hueless: a bright border, a lifted surface and
       bolder text. Red and green are spoken for by loss and gain, so the
       third state borrows neither and the label keeps its own up/down
       colour underneath.

       Scoped by widget key rather than to button[kind="primary"] at
       large, because primary is doing two unrelated jobs in this app —
       selection state here, and genuine call-to-action on Run Screen,
       Save and Create key, where Streamlit's red is right. Streamlit
       stamps each widget's container with st-key-<key>, which is what
       makes the two separable. The chip key is qa_chip_ rather than the
       shorter quick_ it used to be: a [class*="st-key-quick_"] selector
       also matches quick_stats_save and quick_stats_reset in the stats
       picker, which are a different control entirely. */
    [class*="st-key-wl_go_"] button[kind="primary"],
    [class*="st-key-qa_chip_"] button[kind="primary"],
    [class*="st-key-peer_switch_"] button[kind="primary"] {{
        background-color: {_theme.tab_selected_bg} !important;
        border: 2px solid {_theme.header_text} !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.06) !important;
        font-weight: 700 !important;
    }}
    [class*="st-key-wl_go_"] button[kind="primary"] p,
    [class*="st-key-qa_chip_"] button[kind="primary"] p,
    [class*="st-key-peer_switch_"] button[kind="primary"] p {{
        font-weight: 700 !important;
    }}
    /* The ticker name stays bright; any :red[]/:green[] span in the label
       keeps its own colour, so direction still reads on a neutral chip. */
    [class*="st-key-wl_go_"] button[kind="primary"] p,
    [class*="st-key-qa_chip_"] button[kind="primary"] p {{
        color: {_theme.header_text} !important;
    }}
    [class*="st-key-wl_go_"] button[kind="primary"]:hover,
    [class*="st-key-qa_chip_"] button[kind="primary"]:hover,
    [class*="st-key-peer_switch_"] button[kind="primary"]:hover {{
        background-color: {_theme.button_hover_bg} !important;
        border-color: {_theme.header_text} !important;
    }}
    /* The peer switcher disables its own active chip. Left enabled-looking
       elsewhere on purpose (see the watchlist comment), so here the dim
       is removed rather than letting the current ticker read as broken. */
    [class*="st-key-peer_switch_"] button[kind="primary"]:disabled {{
        opacity: 1 !important;
    }}

    /* Quick-access chips are single short ticker labels in fixed-width
       columns, so a label that doesn't quite fit should ellipsise rather
       than break mid-word ("AAP / L"), which is what it did before the
       row was capped. Belt-and-braces alongside that cap.

       Scoped by the same :has() sticky-block selector used below rather
       than to all secondary buttons, because plenty of those elsewhere
       have genuinely long labels ("Download Price & Indicator Data
       (CSV)") that SHOULD be allowed to wrap. */
    [data-testid="stLayoutWrapper"]:has(.quantix-symbol-header) button p,
    [data-testid="stVerticalBlockBorderWrapper"]:has(.quantix-symbol-header) button p {{
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }}

    /* Adjust table styles for high contrast */
    thead tr th {{
        background-color: {_theme.table_head_bg} !important;
        color: {_theme.table_head_text} !important;
        border-bottom: 2px solid {_theme.table_head_border} !important;
    }}

    tbody tr td {{
        background-color: {_theme.table_body_bg} !important;
        color: {_theme.table_body_text} !important;
        border-bottom: 1px solid {_theme.table_body_border} !important;
    }}

    /* Expander headers.
       This rule used to target `.streamlit-expanderHeader`, a class that no
       longer exists in Streamlit 1.58 — grepping the shipped frontend
       bundle finds no such string, so the styling had silently done
       nothing since the upgrade and every expander title fell back to
       Streamlit's defaults. The current markup is a <summary> inside
       [data-testid="stExpander"]. */
    [data-testid="stExpander"] details {{
        background-color: {_theme.expander_bg} !important;
        border: 1px solid {_theme.expander_border} !important;
    }}
    [data-testid="stExpander"] summary {{
        color: {_theme.expander_text} !important;
        border-radius: 6px;
        transition: background-color 200ms ease, color 200ms ease,
                    box-shadow 200ms ease;
    }}
    /* Streamlit renders the label as a <p>, which inherits the body colour
       and weight rather than the summary's — so the colour above alone
       leaves the title looking like ordinary paragraph text. */
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span {{
        color: {_theme.expander_text} !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }}
    [data-testid="stExpander"] summary:hover p {{
        color: {_theme.card_accent} !important;
    }}

    /* --- Micro-interactions ------------------------------------------
       Three things the app was missing, and one it only half had.

       SIDEBAR PANELS ARE THE MENU. This app has no nav rail; the eight
       st.sidebar.expander panels (Branding, Slack Alerts, Email Digest,
       Help & Support, API Keys, Find a ticker, Manage Watchlists and the
       thresholds panel) are what "sidebar menu items" means here. They
       already changed the title's COLOUR on hover, which on a near-black
       ground is easy to miss — so the whole header now takes a surface
       and a left rail, which is legible peripherally.

       THE WATCHLIST ✕ RECEDES, IT DOES NOT DISAPPEAR. The brief asks to
       "hover to show edit/delete options". Delete already exists and is
       always visible; there is no edit operation on a watchlist row at
       all (the only per-row mutation in the model is remove). Hiding the
       one working control behind :hover would make it unreachable on
       touch — and this app's sidebar becomes a full-screen overlay on a
       phone, so that is not a hypothetical. It is de-emphasised at rest
       and comes fully forward on row hover or keyboard focus instead:
       the same "the row has actions" feel, without a control that only
       a mouse can find. Red only on the button's own hover, where the
       meaning is destruction rather than loss.

       200ms EVERYWHERE. The app had exactly one transition (the metric
       card, at 300ms/200ms); every other hover state snapped. These are
       property-scoped rather than `all`, which would animate layout and
       fight Plotly's own resizing. */
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus-visible {{
        background-color: {_theme.tab_hover_bg} !important;
        box-shadow: inset 3px 0 0 0 {_theme.card_accent};
    }}
    [data-testid="stExpander"] summary:focus-visible p {{
        color: {_theme.card_accent} !important;
    }}

    /* Watchlist rows. The chip nudges toward the pointer; the remove
       button fades up from 0.45. */
    [data-testid="stSidebar"] [class*="st-key-wl_go_"] button:hover {{
        transform: translateX(2px);
    }}
    /* The accent border is for UNSELECTED chips only. The active chip's
       "you are here" is deliberately hueless (see the block above), and
       an unscoped hover rule here out-specifies its white border and
       repaints it cyan — measured in-browser, where the current ticker
       stopped looking current the moment the pointer touched it. */
    [data-testid="stSidebar"] [class*="st-key-wl_go_"] button[kind="secondary"]:hover {{
        border-color: {_theme.card_accent} !important;
    }}
    /* The ✕'s resting fade and its red hover now come from
       button_roles.css(), which applies the same treatment to every
       destructive control in the app — keeping a second copy here is how
       the watchlist's remove button and the alert rule's would drift
       apart. What stays is the one rule that is genuinely specific to
       this row: :has() on the row wrapper is what makes hovering the
       TICKER light up its ✕, since the two are separate st.columns
       children with no shared hoverable ancestor otherwise. */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has([class*="st-key-wl_rm_"]):hover
        [class*="st-key-wl_rm_"] button {{
        opacity: 1;
    }}

    /* Every button and tab, main area and sidebar alike. */
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-primary"],
    button[kind="secondary"],
    button[kind="primary"] {{
        /* !important, so this is the ONE list that applies to a button —
           a per-widget `transition:` further down is silently shadowed by
           it. Every property any hover state changes must therefore be
           named here, including the watchlist chip's transform and the
           remove button's opacity. Verified in-browser via
           getComputedStyle, not by reading the rule. */
        transition: background-color 200ms ease, border-color 200ms ease,
                    color 200ms ease, box-shadow 200ms ease,
                    opacity 200ms ease, transform 200ms ease !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        transition: background-color 200ms ease, color 200ms ease,
                    border-color 200ms ease;
    }}

    /* Motion is decoration; the colour changes carry the meaning on
       their own for anyone who has asked the OS to stop it. */
    @media (prefers-reduced-motion: reduce) {{
        [data-testid="stExpander"] summary,
        [data-testid="stSidebar"] [class*="st-key-wl_go_"] button,
        [data-testid="stSidebar"] [class*="st-key-wl_rm_"] button,
        button[kind="secondary"], button[kind="primary"],
        .stTabs [data-baseweb="tab"],
        div[data-testid="metric-container"] {{
            transition-duration: 1ms !important;
        }}
        [data-testid="stSidebar"] [class*="st-key-wl_go_"] button:hover,
        div[data-testid="metric-container"]:hover {{
            transform: none !important;
        }}
    }}

    /* --- Top-level panel navigation ---------------------------------
       The main page tabs are the primary navigation for the whole
       analysis, so they get real presence instead of Streamlit's default
       small text links: larger type, generous hit area, a visible rail
       under the strip, and a solid highlight + underline on the active
       panel. Scoped to .stTabs so it applies to nested tab groups too. */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        border-bottom: 2px solid {_theme.tab_rail_border};
        padding-bottom: 0;
        overflow-x: auto;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 52px;
        padding: 0 16px;
        background-color: {_theme.tab_bg};
        border: 1px solid {_theme.tab_border};
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        white-space: nowrap;
    }}
    /* Streamlit's default inactive-tab colour is near-black (#31333F), which
       is effectively invisible against a dark background — set an
       explicitly readable colour so unselected panels stay legible in
       both themes. */
    .stTabs [data-baseweb="tab"] p {{
        font-size: 0.98rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
        color: {_theme.tab_inactive_text} !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: {_theme.tab_hover_bg};
    }}
    .stTabs [data-baseweb="tab"]:hover p {{
        color: {_theme.tab_hover_text} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {_theme.tab_selected_bg} !important;
        border-color: {_theme.tab_selected_border} !important;
    }}
    .stTabs [aria-selected="true"] p {{
        color: {_theme.tab_selected_text} !important;
    }}
    /* Streamlit's own active-tab underline, thickened to match. */
    .stTabs [data-baseweb="tab-highlight"] {{
        height: 3px;
    }}

    /* The sidebar's control tabs share the strip styling but stay
       compact — they sit in a much narrower column. */
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {{
        height: 38px;
        padding: 0 12px;
    }}
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] p {{
        font-size: 0.86rem !important;
    }}

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
    [data-testid="stVerticalBlockBorderWrapper"]:has(.quantix-symbol-header) {{
        position: sticky;
        top: 0;
        z-index: 100;
    }}
    .quantix-symbol-header {{
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 8px 18px;
        padding: 10px 18px;
        margin-bottom: 4px;
        background: {_theme.symbol_header_bg};
        border: 1px solid {_theme.symbol_header_border};
        border-left: 4px solid {_theme.card_accent};
        border-radius: 8px;
        /* Opaque background plus a shadow so page content scrolling
           underneath never shows through or visually collides. */
        box-shadow: 0 6px 18px {_theme.symbol_header_shadow};
    }}
    .quantix-symbol-header .qsh-ticker {{
        font-size: 1.5rem; font-weight: 700; color: {_theme.header_text}; letter-spacing: 0.5px;
    }}
    .quantix-symbol-header .qsh-name {{
        font-size: 0.95rem; color: {_theme.symbol_header_name}; margin-right: auto;
    }}
    .quantix-symbol-header .qsh-price {{
        font-size: 1.5rem; font-weight: 700; color: {_theme.header_text};
    }}
    .quantix-symbol-header .qsh-change {{ font-size: 1.05rem; font-weight: 600; }}
    .quantix-symbol-header .qsh-up {{ color: #22c55e; }}
    .quantix-symbol-header .qsh-down {{ color: #ef4444; }}
    .quantix-symbol-header .qsh-flat {{ color: {_theme.symbol_header_flat}; }}
    .quantix-symbol-header .qsh-meta {{ font-size: 0.85rem; color: {_theme.symbol_header_meta}; }}

    /* --- Responsive breakpoints (tablet / mobile) --------------------
       Streamlit already does a lot of this natively for free: the
       sidebar becomes a full-screen overlay and st.columns() stacks to
       one-per-row below its own internal ~640px breakpoint, and Plotly
       figures resize to their container. What's NOT handled natively is
       the "tablet squeeze" zone — roughly 641-900px, an open (non-overlay)
       sidebar plus a still-side-by-side st.columns(3/4/5) row leaves each
       column too narrow for its own content (verified in-browser: a
       4-column metric row at 768px viewport was squeezed to ~97px per
       column, narrow enough that "Alignment:" was wrapping mid-word).
       flex-wrap on the row + a floor on each column's width fixes every
       such row across the app at once (Institutional Watchlist cards,
       Risk Dashboard metrics, Data Quality Report fields, alert-rule
       inputs, etc.) without hand-tuning each one individually.

       Scoped to stMain, not the sidebar: the sidebar's own multi-column
       rows (e.g. a watchlist ticker pill next to its ✕ button) are
       intentionally narrow-by-design and already fit their own docked
       width fine — applying the same 150px floor there was verified
       in-browser to force the ✕ button onto its own line for no reason,
       a regression this scoping avoids. */
    @media (max-width: 900px) {{
        [data-testid="stMain"] [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
        }}
        [data-testid="stMain"] [data-testid="stColumn"] {{
            min-width: 150px !important;
            flex: 1 1 150px !important;
        }}
        /* The page title is the single biggest offender for "wastes the
           whole first screen on a phone" — verified in-browser at 375px
           it spanned six lines before any real content was visible. */
        h1 {{
            font-size: 1.9rem !important;
        }}
    }}
    @media (max-width: 640px) {{
        h1 {{
            font-size: 1.55rem !important;
        }}
        .quantix-symbol-header {{
            gap: 4px 12px;
            padding: 8px 12px;
        }}
        .quantix-symbol-header .qsh-ticker,
        .quantix-symbol-header .qsh-price {{
            font-size: 1.2rem;
        }}
    }}

    /* --- Readable type scale -----------------------------------------
       Streamlit's defaults put captions, body copy, every widget label
       and every input at 14px. That is tuned for a compact dashboard; on
       a large monitor, in a dense analytical app whose panels are mostly
       label-and-control pairs, it reads as fine print. Everything here is
       a modest step up — roughly one point — rather than a redesign, so
       column widths and table layouts are unaffected.

       Deliberately NOT touching table cell text: st.table and st.dataframe
       size their columns from content, and widening the type there pushes
       the wider scorecards into horizontal scroll. */
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {{
        font-size: 0.97rem !important;
        line-height: 1.55 !important;
    }}
    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] .stMarkdown li {{
        font-size: 1rem;
        line-height: 1.6;
    }}
    /* Widget labels — every widget type at once.
       Streamlit tags the label element with data-testid="stWidgetLabel"
       and renders the text in a nested <p> inside it. Targeting the
       per-widget classes (.stSelectbox label p and friends) matches
       nothing in this version and fails silently; the testid is the
       stable hook, and naming the <p> is required because it inherits
       from the markdown rules above rather than from the label. */
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p {{
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }}
    /* The values people actually read back and check. */
    [data-testid="stMain"] .stTextInput input,
    [data-testid="stMain"] .stNumberInput input,
    [data-testid="stMain"] .stDateInput input,
    [data-testid="stMain"] .stTextArea textarea,
    [data-testid="stMain"] div[data-baseweb="select"] {{
        font-size: 0.97rem !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# Button roles — primary / secondary / danger — in one stylesheet rather
# than scattered across the panels that own each button. Injected after
# the block above because it deliberately overrides Streamlit's stock
# primary; the hueless selection chips are excluded inside the rule
# itself, so this does not depend on which stylesheet lands last.
st.markdown(button_roles.css(_theme, _theme.card_accent), unsafe_allow_html=True)

# ==========================================
# AUTHENTICATION GATE
# ==========================================
# Everything below this line assumes a signed-in user. The gate sits here
# rather than at the top of the file because it needs st.set_page_config
# and the theme CSS to have run — a login page rendered before those is
# unstyled and flashes white — and above every st.* call that draws the
# app itself, so a signed-out visitor never sees a frame of it.
#
# require_sign_in() calls st.stop() when signed out, so nothing after this
# executes for an anonymous visitor. Both sign-in paths satisfy it: a
# local email/password account or the existing OIDC provider.
import login_page

login_page.require_sign_in()

# The skeleton for the slot reserved at the top of the page. Drawn HERE,
# not there, for two reasons this script's order makes unavoidable:
# _theme is not assigned until ~20 lines below the reservation (a
# NameError that fires on first paint, which is the only paint that
# matters for a loading state), and everything above the gate renders
# behind a signed-out visitor's login page.
with symbol_header_container.container():
    st.markdown(loading_states.css(_theme), unsafe_allow_html=True)
    st.markdown(
        loading_states.skeleton("Loading symbol", rows=(28, 62, 40), tall_first=True),
        unsafe_allow_html=True,
    )

# Only now does the app itself begin. The title sits below the gate so a
# signed-out visitor sees the login page alone rather than the app's
# masthead stacked on top of it.
# ==========================================
# KEYBOARD SHORTCUTS & COMMAND PALETTE
# ==========================================
# The listener is an invisible components.html iframe. That is NOT the
# same mechanism onboarding.py rules out: a <script> inserted through
# st.markdown never executes, but a component iframe is a real document
# whose scripts do — and it is same-origin here, so it can bind on
# window.parent.document and click things. Probed on this app before any
# of it was written; see keyboard_shortcuts' docstring.
#
# The hidden buttons below are the bridge back into Python. They are
# clipped rather than display:none so a synthetic .click() still lands.
st.markdown(
    """
    <style>
    [class*="st-key-kbd_trigger_"] {
        position: absolute !important;
        width: 1px; height: 1px;
        overflow: hidden; clip: rect(0 0 0 0);
        white-space: nowrap;
    }
    /* Both panels open ABOVE the sticky symbol header, which sits at
       z-index 100 and otherwise renders straight through them — the
       quick-stats chips landed on top of the palette's own search box.
       An opaque background is part of the fix, not decoration: without
       it the page still shows through a merely-raised panel. */
    [class*="st-key-kbd_palette_panel"],
    [class*="st-key-kbd_shortcuts_panel"] {
        position: relative;
        z-index: 200;
        background: var(--background-color, #000000);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# See the note where this is assigned: read with a default because
# asset_kind is not computed until far below this point.
_kbd_tabs = asset_views.tab_labels(
    st.session_state.get("asset_kind", asset_class.EQUITY))
_kbd_pending_tab = st.session_state.pop("kbd_pending_tab", None)
# Whichever panel just opened gets scrolled to and focused — see the
# module docstring: they render far below the fold, so without this ⌘K
# looked like it did nothing.
_kbd_focus = ("kbd_palette_panel" if st.session_state.get("kbd_palette_open")
              else "kbd_shortcuts_panel" if st.session_state.get("kbd_shortcuts_open")
              else None)
components.html(
    keyboard_shortcuts.listener_html(pending_tab=_kbd_pending_tab,
                                     focus_panel=_kbd_focus),
    height=0,
)

for _kbd_name, _kbd_key in keyboard_shortcuts.TRIGGERS.items():
    if st.button(_kbd_name, key=_kbd_key):
        if _kbd_name == "close":
            st.session_state["kbd_palette_open"] = False
            st.session_state["kbd_shortcuts_open"] = False
        elif _kbd_name == "palette":
            st.session_state["kbd_palette_open"] = True
            st.session_state["kbd_shortcuts_open"] = False
        elif _kbd_name == "shortcuts":
            st.session_state["kbd_shortcuts_open"] = not st.session_state.get(
                "kbd_shortcuts_open", False)
        elif _kbd_name == "help":
            st.session_state[profile_menu.OPEN_HELP_KEY] = True
        elif _kbd_name == "new_alert":
            st.session_state["kbd_new_alert_requested"] = True
        log_event(logger, logging.INFO, "user.keyboard_shortcut", action=_kbd_name)
        st.rerun()

# One hidden button per asset-class pill, for Alt+1..6. They are hidden by
# the same st-key-kbd_trigger_ rule as the others above. The pill itself
# cannot be clicked from JS reliably (Streamlit rebuilds it every run), so
# the shortcut goes through a real button and a deferred switch — the same
# route the watchlist uses for the ticker.
for _kbd_i, _kbd_view in enumerate(asset_views.VIEWS):
    if st.button(_kbd_view.pill,
                 key=f"{keyboard_shortcuts.ASSET_TRIGGER_PREFIX}{_kbd_i}"):
        st.session_state["_pending_asset_view"] = _kbd_view.key
        log_event(logger, logging.INFO, "user.keyboard_shortcut",
                  action=f"asset:{_kbd_view.key}")
        st.rerun()

st.title(brand().title)

# --- the palette and the shortcuts panel ---------------------------------
if st.session_state.get("kbd_palette_open"):
    with st.container(border=True, key="kbd_palette_panel"):
        st.caption("**Command palette** · type to filter, Esc to close")
        _kbd_query = st.text_input(
            "Command", key="kbd_palette_query", label_visibility="collapsed",
            placeholder="Jump to a tab, create an alert, open help…",
        )
        _kbd_hits = keyboard_shortcuts.search(
            _kbd_query, keyboard_shortcuts.commands(_kbd_tabs))
        if not _kbd_hits:
            st.caption(f'Nothing matches "{_kbd_query.strip()}".')
        for _kbd_cmd in _kbd_hits[:8]:
            _kbd_label = (f"{_kbd_cmd.label}  ·  {_kbd_cmd.hint}"
                          if _kbd_cmd.hint else _kbd_cmd.label)
            if st.button(_kbd_label, key=f"kbd_cmd_{_kbd_cmd.id}", width="stretch"):
                if _kbd_cmd.kind == "tab":
                    # st.tabs cannot be selected from Python, so the index
                    # is handed to the listener component, which clicks the
                    # tab when it mounts on the next run.
                    st.session_state["kbd_pending_tab"] = _kbd_cmd.payload
                elif _kbd_cmd.id == "action:help":
                    st.session_state[profile_menu.OPEN_HELP_KEY] = True
                elif _kbd_cmd.id == "action:shortcuts":
                    st.session_state["kbd_shortcuts_open"] = True
                elif _kbd_cmd.id == "action:new_alert":
                    st.session_state["kbd_new_alert_requested"] = True
                st.session_state["kbd_palette_open"] = False
                log_event(logger, logging.INFO, "user.palette_command",
                          command=_kbd_cmd.id)
                st.rerun()
        if st.button("Close", key="kbd_palette_close", width="stretch"):
            st.session_state["kbd_palette_open"] = False
            st.rerun()

if st.session_state.get("kbd_shortcuts_open"):
    with st.container(border=True, key="kbd_shortcuts_panel"):
        st.caption("**Keyboard shortcuts**")
        for _kbd_cat, _kbd_items in keyboard_shortcuts.shortcuts_by_category(
                _kbd_tabs).items():
            st.caption(f"**{_kbd_cat}**")
            for _kbd_sc in _kbd_items:
                st.markdown(
                    f"`{_kbd_sc.keys}` — {_kbd_sc.description}"
                    + (f"  \n<span style='opacity:.7;font-size:.86em'>{_kbd_sc.note}</span>"
                       if _kbd_sc.note else ""),
                    unsafe_allow_html=True,
                )
        st.caption(
            "Shortcuts are delivered by an invisible component iframe, so they "
            "work on this page only — not while a browser dialog has focus."
        )
        if st.button("Close", key="kbd_shortcuts_close", width="stretch"):
            st.session_state["kbd_shortcuts_open"] = False
            st.rerun()

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
            # Everything below is for the card's hover panel only. It costs
            # no extra request: the bundle above is already loaded and
            # cached, and these were simply being discarded. EPS and revenue
            # growth are not on StandardizedFinancials, so they come off the
            # raw info dict — .get() rather than [] because a thinly covered
            # ticker reports neither, and the panel says "Not reported".
            "eps": bundle.info.get("trailingEps"),
            "revenue_growth": bundle.info.get("revenueGrowth"),
            "earnings_growth": std.earnings_growth,
            "return_on_equity": std.return_on_equity,
            "dividend_yield_pct": std.dividend_yield_pct,
            "market_cap": std.market_cap,
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

    # The cards are markup rather than st.info() so they can carry a hover
    # panel — see alignment_card's docstring. Injected here rather than in
    # the global CSS block because the styles are meaningless on any page
    # that has not drawn a card.
    st.markdown(alignment_card.css(_theme, _theme.card_accent), unsafe_allow_html=True)

    # --- SECTION 1: TECH & GROWTH PORTFOLIO ---
    st.subheader("Top Tech & Growth Alignments")
    if tech_picks:
        t_cols = st.columns(WATCHLIST.cards_shown)
        for i, data in enumerate(tech_picks[:WATCHLIST.cards_shown]):
            with t_cols[i]:
                st.markdown(alignment_card.card_html(data), unsafe_allow_html=True)
    else:
        st.write("No tech leaders currently pass basic filters.")

    st.markdown("##") # Space out the sections cleanly

    # --- SECTION 2: DIVERSIFIED SECTOR PORTFOLIO ---
    st.subheader("Top Diversified Market Alignments")
    if other_picks:
        o_cols = st.columns(WATCHLIST.cards_shown)
        for i, data in enumerate(other_picks[:WATCHLIST.cards_shown]):
            with o_cols[i]:
                st.markdown(alignment_card.card_html(data), unsafe_allow_html=True)
    else:
        st.write("No diversified sector leaders currently pass basic filters.")

# ==========================================
# CUSTOM THRESHOLDS
# ==========================================
# Placed here, above the Screener and Alerts and before the analysis tabs,
# because it governs all of them: the same numbers drive the Scorecard's
# pass/fail checks, the Altman verdict the screener reports, and the
# default an alert rule starts at. Collapsed by default so it costs
# nothing until wanted. Deliberately NOT in the sidebar — 17 numeric
# inputs plus a sector table need horizontal room the ~300px sidebar
# doesn't have.
st.markdown("---")
with st.expander("Custom Thresholds", expanded=False):
    st.caption(
        "Your own valuation and risk cut-offs, applied consistently to the Scorecard, the Altman "
        "verdict the screener reports, and the default a new alert rule starts at. Only the "
        "pass/fail LINES are editable — the per-metric scoring weights and the Composite Risk "
        "Score's internal anchors deliberately aren't, since those change how a score is computed "
        "rather than where its threshold sits. Blank the store with Reset to return to the "
        "shipped defaults."
    )

    _thr_values = threshold_effective_values()
    # Pre-seed each widget's session_state entry, then render with key= only.
    # Passing BOTH value= and key= is the stomping bug this codebase has hit
    # before: once the key holds a value, a re-passed value= can silently
    # revert the user's edit on the next rerun.
    for _spec in THRESHOLD_SPECS:
        _wkey = f"thr_{_spec.key}"
        if _wkey not in st.session_state:
            st.session_state[_wkey] = float(_thr_values[_spec.key])

    _thr_groups = (
        ("Profitability & capital efficiency", ("min_net_margin", "min_roic_pct", "min_fcf_yield_pct")),
        ("Leverage & liquidity", ("max_debt_to_equity", "financials_max_debt_to_equity",
                                  "min_current_ratio", "min_interest_coverage")),
        ("Valuation & market", ("pe_range_low", "pe_range_high", "peg_range_low",
                                "peg_range_high", "max_beta")),
        ("Verdict bands", ("high_alignment_pct", "moderate_alignment_pct")),
        ("Risk", ("altman_safe_zone", "altman_grey_zone", "vix_high_risk_threshold")),
    )
    for _grp_label, _grp_keys in _thr_groups:
        st.markdown(f"**{_grp_label}**")
        _grp_cols = st.columns(len(_grp_keys))
        for _gc, _gkey in zip(_grp_cols, _grp_keys):
            _gspec = THRESHOLD_SPECS_BY_KEY[_gkey]
            with _gc:
                st.number_input(
                    _gspec.label, key=f"thr_{_gkey}", help=_gspec.helptext,
                    min_value=float(_gspec.minimum), max_value=float(_gspec.maximum),
                    step=float(_gspec.step),
                )

    st.markdown("---")
    st.markdown("**Sector P/E bands**")
    st.caption(
        "Overrides the global P/E band for a named sector — the task's own example of tech vs "
        "utilities. Sector names must match Yahoo's spelling exactly to take effect (both "
        "\"Financial Services\" and \"Financials\" are listed because Yahoo has used each). Any "
        "sector not listed here falls back to the global band above. Add or delete rows directly "
        "in the table; PEG is deliberately not sector-adjusted, since it already divides P/E by "
        "growth."
    )
    _sector_rows = [
        {"Sector": _s, "P/E Low": float(_lo), "P/E High": float(_hi)}
        for _s, (_lo, _hi) in sorted(threshold_effective_sector_pe().items())
    ]
    _sector_edited = st.data_editor(
        pd.DataFrame(_sector_rows, columns=["Sector", "P/E Low", "P/E High"]),
        num_rows="dynamic", width="stretch", key="thr_sector_editor",
        column_config={
            "Sector": st.column_config.TextColumn("Sector", help="Yahoo's sector name, spelled exactly."),
            "P/E Low": st.column_config.NumberColumn("P/E Low", min_value=0.0, max_value=500.0, step=1.0),
            "P/E High": st.column_config.NumberColumn("P/E High", min_value=0.0, max_value=500.0, step=1.0),
        },
    )

    _thr_save_col, _thr_reset_col, _thr_status_col = st.columns([1, 1, 4])
    with _thr_save_col:
        if st.button("Save thresholds", type="primary"):
            _thr_new = {s.key: float(st.session_state[f"thr_{s.key}"]) for s in THRESHOLD_SPECS}
            _thr_errors = []
            _thr_table = {}
            for _row in _sector_edited.to_dict("records"):
                _rs = str(_row.get("Sector") or "").strip()
                if not _rs:
                    continue
                try:
                    _rlo, _rhi = float(_row.get("P/E Low")), float(_row.get("P/E High"))
                except (TypeError, ValueError):
                    _thr_errors.append(f'Sector "{_rs}": P/E values must be numbers.')
                    continue
                _thr_table[_rs] = (_rlo, _rhi)

            # Cross-field checks live in user_thresholds.validate() so they're
            # unit-tested rather than buried in this script.
            _thr_errors.extend(validate_thresholds(_thr_new, _thr_table))
            if _thr_errors:
                for _e in _thr_errors:
                    st.warning(_e)
            else:
                save_threshold_overrides(_thr_new)
                save_threshold_sector_pe(_thr_table)
                log_event(logger, logging.INFO, "user.thresholds_saved",
                          changed=len(load_threshold_overrides()), sectors=len(_thr_table))
                st.success("Thresholds saved — Scorecard, alerts and screener now use them.")
    with _thr_reset_col:
        # Keyed so button_roles can mark it destructive — it discards the
        # user's customised thresholds. A key on a BUTTON is safe (unlike
        # the screener's criteria selectboxes, which must stay unkeyed):
        # a button holds no value between runs, so there is nothing for
        # Streamlit to restore over a fresh one.
        if st.button("Reset to defaults", key="thresholds_reset"):
            reset_thresholds()
            for _spec in THRESHOLD_SPECS:
                st.session_state.pop(f"thr_{_spec.key}", None)
            st.session_state.pop("thr_sector_editor", None)
            log_event(logger, logging.INFO, "user.thresholds_reset")
            st.rerun()
    with _thr_status_col:
        _thr_changed = load_threshold_overrides()
        _thr_sectors = load_threshold_sector_pe()
        if _thr_changed or _thr_sectors:
            st.caption(
                f"{len(_thr_changed)} threshold(s) and {len(_thr_sectors)} sector band(s) "
                "differ from the shipped defaults."
            )
        else:
            st.caption("Currently using the shipped defaults for everything.")

# ==========================================
# STOCK SCREENER
# ==========================================
st.markdown("---")
st.header("Stock Screener")
st.caption("Screen an arbitrary ticker universe against your own fundamental, technical, and risk criteria — not just the fixed thresholds above.")

_screener_default_universe = ", ".join(WATCHLIST.tech_basket + WATCHLIST.diversified_basket)

# The universe box is keyed so a saved screener can load its own ticker
# list into it. Seeded first and rendered with key= only: passing both
# value= and key= makes Streamlit restore the old value on the next run,
# silently undoing the template that was just applied.
if "screener_universe_text" not in st.session_state:
    st.session_state["screener_universe_text"] = _screener_default_universe

if "screener_criteria" not in st.session_state:
    st.session_state["screener_criteria"] = [{"metric": "pe_ratio", "operator": "<", "threshold": 25.0}]

# Deferred clear of the save-name box, executed before that widget renders.
# A compose box still holding what you just saved reads as "nothing
# happened" and gets clicked again.
if st.session_state.pop("screener_save_clear", False):
    st.session_state["screener_save_name"] = ""
_screener_saved_name = st.session_state.pop("screener_save_done", None)
if _screener_saved_name:
    st.success(f"Saved “{_screener_saved_name}”.")


def _screener_apply_template(_tpl) -> None:
    """Load a saved screener into the builder."""
    st.session_state["screener_criteria"] = [dict(c) for c in _tpl.criteria]
    if _tpl.universe:
        st.session_state["screener_universe_text"] = ", ".join(_tpl.universe)
    st.session_state["screener_applied_template"] = _tpl.name


# --- Saved screeners ---------------------------------------------------
_screener_templates = screener_templates.load()
if screener_templates.store_is_corrupt():
    # An empty list here would be a lie: there may be saved screens in a
    # file we cannot parse. Nothing is overwritten until it's resolved.
    st.warning(
        f"Your saved screeners file ({screener_templates.STORE_FILENAME}) can't be read, "
        "so none are listed and saving is disabled. Nothing has been overwritten — move "
        "or delete that file to start fresh."
    )
if _screener_templates:
    st.markdown("**Saved screeners**")
    st.caption(
        "One click loads a screen's filters and its ticker list. The list travels with "
        "the screen because this screener filters the universe you give it rather than "
        "searching the whole market."
    )
    _tpl_cols = st.columns(4)
    for _tpl_i, _tpl in enumerate(_screener_templates):
        with _tpl_cols[_tpl_i % 4]:
            if st.button(_tpl.name, key=f"screener_tpl_{_tpl_i}",
                         width="stretch", help=_tpl.summary):
                _screener_apply_template(_tpl)
                st.rerun()

_screener_applied = st.session_state.pop("screener_applied_template", None)
if _screener_applied:
    st.success(f"Loaded “{_screener_applied}”. Press Run Screen to execute it.")

with st.expander("Manage saved screeners", expanded=False):
    st.caption(
        "Reordering is up/down rather than drag-and-drop: Streamlit has no native "
        "dragging, and buttons work with a keyboard, which dragging does not."
    )
    for _tpl_i, _tpl in enumerate(_screener_templates):
        _m1, _m2, _m3, _m4 = st.columns([6, 1, 1, 1])
        with _m1:
            st.markdown(f"**{_tpl.name}**")
            st.caption(f"{_tpl.summary} · {len(_tpl.universe)} tickers")
            for _tpl_problem in _tpl.unknown_parts():
                st.warning(
                    f"This version can't evaluate {_tpl_problem}. The filter is kept in "
                    "the saved screen but will show as not evaluable rather than being "
                    "silently ignored."
                )
        with _m2:
            if st.button("↑", key=f"screener_tpl_up_{_tpl_i}", help="Move up",
                         disabled=_tpl_i == 0):
                screener_templates.move(_tpl.name, -1)
                st.rerun()
        with _m3:
            if st.button("↓", key=f"screener_tpl_down_{_tpl_i}", help="Move down",
                         disabled=_tpl_i == len(_screener_templates) - 1):
                screener_templates.move(_tpl.name, +1)
                st.rerun()
        with _m4:
            if st.button("Delete", key=f"screener_tpl_del_{_tpl_i}"):
                screener_templates.delete(_tpl.name)
                st.rerun()

    if not _screener_templates:
        st.caption("No saved screeners. Build a screen below and save it.")
        if st.button("Restore the built-in screeners", key="screener_tpl_reset"):
            screener_templates.reset_to_starters()
            st.rerun()

screener_universe_input = st.text_input(
    "Ticker Universe (comma-separated)", key="screener_universe_text",
    help=f"Up to {SCREENER_MAX_UNIVERSE_SIZE} tickers — Yahoo Finance rate limits scale with universe size, so larger universes get truncated with a warning.",
)

st.markdown("**Filter Criteria**")
_screener_metric_options = [m.key for m in SCREENER_METRICS]

_screener_remove_index = None
for _i, _crit in enumerate(st.session_state["screener_criteria"]):
    _c1, _c2, _c3, _c4 = st.columns([3, 1, 2, 1])
    _spec = SCREENER_METRICS_BY_KEY.get(_crit.get("metric"))
    _is_categorical = _spec is not None and _spec.kind == "categorical"
    _ops = list(screener_module.operators_for(_crit.get("metric", "")))

    # None of these three widgets take a key. The criteria list IS the
    # source of truth, and a keyed widget would fight it: applying a saved
    # screener changes metric/operator/threshold underneath the widget,
    # and Streamlit would restore the stored widget value on the next run
    # and undo it. Worse for the operator box, whose OPTIONS change when
    # the metric is categorical — a stored "<" is not in ["is", "is not"]
    # and selecting it raises.
    # Each row's labels carry the row index so the four widgets get
    # DISTINCT auto-generated IDs. Streamlit hashes (label, options, index,
    # help) to identify an unkeyed widget and does NOT include
    # label_visibility, so two rows whose operator box offered the same
    # options at the same index collided outright:
    # StreamlitDuplicateElementId, which took the whole screener down.
    # "+ Add Filter" appends rsi/"<"/30.0, so it collided with any existing
    # numeric "<" row — including the default P/E filter, i.e. the very
    # first click of Add Filter on a fresh screener.
    #
    # A key= would be the obvious fix and is the one thing that must NOT be
    # used here, for the two reasons spelled out above: the operator box's
    # options change with the metric, and a keyed widget restores its old
    # value over a just-applied saved screener. Renaming the label is
    # invisible instead — every row past the first has its label collapsed,
    # so nothing is displayed, and a screen reader gets "Op 2" rather than
    # a third identical "Op".
    _row_suffix = "" if _i == 0 else f" {_i + 1}"
    with _c1:
        _crit["metric"] = st.selectbox(
            f"Metric{_row_suffix}", _screener_metric_options,
            index=_screener_metric_options.index(_crit["metric"]) if _crit.get("metric") in _screener_metric_options else 0,
            format_func=lambda k: SCREENER_METRICS_BY_KEY[k].label,
            label_visibility="visible" if _i == 0 else "collapsed",
        )
        if _crit["metric"] != (_spec.key if _spec else None):
            # Metric just changed kind; reset the operator/threshold to
            # something valid for the new metric before they render.
            _new_spec = SCREENER_METRICS_BY_KEY[_crit["metric"]]
            _new_ops = list(screener_module.operators_for(_crit["metric"]))
            if _crit.get("operator") not in _new_ops:
                _crit["operator"] = _new_ops[0]
            if _new_spec.kind == "categorical" and not isinstance(_crit.get("threshold"), str):
                _crit["threshold"] = _new_spec.choices[0] if _new_spec.choices else ""
            elif _new_spec.kind != "categorical" and isinstance(_crit.get("threshold"), str):
                _crit["threshold"] = 0.0
            _spec, _is_categorical, _ops = _new_spec, _new_spec.kind == "categorical", _new_ops
    with _c2:
        _crit["operator"] = st.selectbox(
            f"Op{_row_suffix}", _ops,
            index=_ops.index(_crit["operator"]) if _crit.get("operator") in _ops else 0,
            label_visibility="visible" if _i == 0 else "collapsed",
        )
    with _c3:
        if _is_categorical:
            _choices = list(_spec.choices) or [str(_crit.get("threshold", ""))]
            _crit["threshold"] = st.selectbox(
                f"Value{_row_suffix}", _choices,
                index=_choices.index(_crit["threshold"]) if _crit.get("threshold") in _choices else 0,
                label_visibility="visible" if _i == 0 else "collapsed",
            )
        else:
            _crit["threshold"] = st.number_input(
                f"Threshold{_row_suffix}",
                value=float(_crit["threshold"]) if isinstance(_crit.get("threshold"), (int, float)) else 0.0,
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

# --- Save the current screen -------------------------------------------
with st.expander("Save this screen", expanded=False):
    st.caption(
        "Saves the filters above together with the ticker universe, so clicking it "
        "later reproduces this exact screen. Saving under an existing name overwrites "
        "it and keeps its position in the list."
    )
    _save_name_col, _save_btn_col = st.columns([3, 1])
    with _save_name_col:
        _screener_save_name = st.text_input(
            "Name", key="screener_save_name", placeholder="e.g. Cheap quality tech",
        )
    with _save_btn_col:
        st.markdown("&nbsp;")
        _screener_save_clicked = st.button("Save", key="screener_save_btn", width="stretch")

    if _screener_save_clicked:
        _save_universe = [t.strip().upper() for t in screener_universe_input.split(",") if t.strip()]
        _saved_ok, _saved_err = screener_templates.save(
            _screener_save_name, st.session_state["screener_criteria"], _save_universe)
        if _saved_ok:
            # Deferred clear: popping the widget's own key inside this
            # handler does nothing, because Streamlit restores it from the
            # widget-state layer on the next run. Set a flag, rerun, and
            # clear it at the top of the next run before the widget draws.
            st.session_state["screener_save_clear"] = True
            st.session_state["screener_save_done"] = _screener_save_name.strip()
            st.rerun()
        else:
            st.warning(_saved_err)

# The empty-state "remove this filter" action re-runs the screen without
# making the user find the button again.
if st.session_state.pop("_screener_rerun", False):
    screener_run_clicked = True

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
        def _screener_threshold(_c):
            """Categorical metrics carry a name, not a number.

            This used to coerce every threshold with float(), which is
            correct for all fourteen original metrics and raises the moment
            a Sector criterion reaches it.
            """
            _spec = SCREENER_METRICS_BY_KEY.get(_c.get("metric"))
            if _spec is not None and _spec.kind == "categorical":
                return str(_c["threshold"])
            return float(_c["threshold"])

        _screener_criteria_tuple = tuple(
            ScreenCriterion(metric=c["metric"], operator=c["operator"],
                            threshold=_screener_threshold(c))
            for c in st.session_state["screener_criteria"]
        )
        # A real bar, not a spinner: a sixteen-ticker screen takes about
        # fifteen seconds and a spinner says nothing about whether it is
        # halfway or wedged. On a cache hit the callback never fires and
        # the bar goes straight to complete, which is honest — there was
        # no work to report.
        _screener_bar = st.progress(0.0, text="Starting screen…")

        def _screener_progress(done, total, ticker):
            _screener_bar.progress(
                loading_states.progress_fraction(done, total),
                text="Screening " + loading_states.progress_text(done, total, ticker))

        _screener_results = run_screen(
            tuple(_screener_universe), _screener_criteria_tuple,
            _on_progress=_screener_progress)
        _screener_bar.empty()
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

    # Zero PASSING is not zero results — the table below still lists every
    # ticker with its own pass/fail, and that evidence stays on screen.
    # So rather than "try adjusting filters", name the filter that actually
    # did the rejecting, counted from the results themselves.
    if _screener_pass_count == 0:
        _screener_why, _screener_drop = empty_states.screener_guidance(
            _screener_results, _screener_criteria)
        _screener_acted = empty_states.render(
            "No stocks passed every filter",
            _screener_why,
            action_label=(f"Remove “{_screener_drop.text}” and re-run"
                          if _screener_drop else None),
            key="empty_screener_relax",
            help_text=("Drops just that one criterion and screens the same "
                       "universe again. The others stay as they are."),
        )
        if _screener_acted and _screener_drop is not None:
            # session_state["screener_criteria"] is the source of truth and
            # is positionally aligned with the ScreenCriterions that were
            # run, so the index carries across.
            _screener_kept = [
                c for i, c in enumerate(st.session_state["screener_criteria"])
                if i != _screener_drop.index
            ]
            st.session_state["screener_criteria"] = _screener_kept
            # Re-run on the next pass rather than duplicating the run block
            # here; the trigger below treats this exactly like a click.
            st.session_state["_screener_rerun"] = True
            empty_states.log_action("screener_filter_dropped",
                                    remaining=len(_screener_kept))
            st.rerun()

    _screener_rows = []
    for r in _screener_results:
        row = {"Ticker": r.ticker}
        for c in _screener_criteria:
            spec = SCREENER_METRICS_BY_KEY[c.metric]
            v = r.values.get(c.metric)
            # Both the column header and the cell assumed a number here.
            # A categorical metric carries a name in each: "{:g}" on
            # "Technology" and round() on a sector both raise.
            if spec.kind == "categorical":
                header = f"{spec.label} ({c.operator} {c.threshold})"
                row[header] = v if v is not None else None
            else:
                shown = f"${c.threshold:g}" if spec.unit == "$" else f"{c.threshold:g}{spec.unit}"
                header = f"{spec.label} ({c.operator} {shown})"
                row[header] = round(v, spec.decimals) if v is not None else None
        if r.status == "fetch_error":
            row["Result"] = "Could Not Load"
        elif r.status == "insufficient_data":
            row["Result"] = "Insufficient Data"
        elif r.passed_all:
            row["Result"] = "Pass"
        else:
            row["Result"] = "Fail"
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
# ETF SCREENER
# ==========================================
# A SEPARATE section from the Stock Screener above, because it works the
# other way round. That one filters a universe you supply, one deep fetch
# per ticker — its docstring is explicit that it does not scan the
# market. Yahoo publishes a whole ETF table in one request, so this one
# genuinely screens 250 funds you did not have to name in advance.
st.markdown("---")
# ==========================================
# BOND MARKET — YIELD CURVE (PHASE 2.4)
# ==========================================
st.markdown("---")
st.header("Treasury Yield Curve", anchor="yield-curve")
_yc_curve = bond_data.load_curve()
_yc_history, _yc_err = bond_market.load_curve_history("5y")
if _yc_err:
    st.caption(_yc_err)
if not _yc_curve.ok:
    st.info(_yc_curve.error or "The treasury curve could not be loaded.")
else:
    _yc_shape = bond_data.curve_shape(_yc_curve)
    _yc_slope = bond_market.slope_history(_yc_history)
    _yc1, _yc2, _yc3 = st.columns(3)
    _yc1.metric("Curve shape", _yc_shape.label,
                help="Measured between the shortest and longest maturities "
                     "loaded, both of which are named beneath it.")
    _yc1.caption(_yc_shape.detail)
    if _yc_slope.ok:
        _yc2.metric(f"{_yc_slope.short_label}-{_yc_slope.long_label} slope",
                    f"{_yc_slope.current_pp:+.2f}pp",
                    help="The classic recession signal. Negative means long "
                         "money is priced below short.")
        if _yc_slope.inverted_share_pct is not None:
            _yc2.caption(
                f"Inverted on {_yc_slope.inverted_share_pct:.0f}% of the "
                f"last {_yc_slope.total_days} trading days")
    else:
        _yc2.metric("Curve slope", "Unavailable",
                    help="Needs history for both ends of the curve.")
    _yc3.metric("Source", _yc_curve.source.upper(),
                help="FRED when a key is configured, Yahoo otherwise.")
    if _yc_curve.missing:
        _yc3.caption("Missing: " + ", ".join(_yc_curve.missing))

    _yc_points = sorted(_yc_curve.points, key=lambda p: p.months)
    _yc_fig = go.Figure()
    _yc_fig.add_trace(go.Scatter(
        x=[p.years for p in _yc_points], y=[p.yield_pct for p in _yc_points],
        mode="lines+markers", name="Today",
        text=[p.label for p in _yc_points]))
    # Overlaying where the curve WAS is what makes a shift visible; a
    # single line only ever shows today's shape.
    if _yc_history is not None and not _yc_history.empty:
        for _yc_label, _yc_days in (("1 month ago", 21), ("1 year ago", 252)):
            if len(_yc_history) <= _yc_days:
                continue
            _yc_past = _yc_history.iloc[-1 - _yc_days]
            _yc_xs, _yc_ys = [], []
            for _yc_p in _yc_points:
                _yc_v = _yc_past.get(_yc_p.label)
                if _yc_v is not None and _yc_v == _yc_v:
                    _yc_xs.append(_yc_p.years)
                    _yc_ys.append(float(_yc_v))
            if _yc_xs:
                _yc_fig.add_trace(go.Scatter(
                    x=_yc_xs, y=_yc_ys, mode="lines+markers",
                    name=_yc_label, line=dict(dash="dot")))
    _yc_fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Maturity (years)", yaxis_title="Yield (%)",
        template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(_yc_fig, width="stretch")
    st.caption(chart_help("treasury_yield_curve"))

    _yc_shifts = [sh for sh in bond_market.curve_shifts(_yc_history) if sh.ok]
    if _yc_shifts:
        _yc_table = pd.DataFrame([
            {"Since": sh.label, **{k: v for k, v in sh.changes_bps.items()},
             "Shape": sh.shape}
            for sh in _yc_shifts])
        for _yc_col in _yc_table.columns:
            if _yc_col not in ("Since", "Shape"):
                _yc_table[_yc_col] = pd.to_numeric(_yc_table[_yc_col],
                                                   errors="coerce")
        st.dataframe(
            _yc_table, width="stretch", hide_index=True,
            column_config={c: st.column_config.NumberColumn(c, format="%+.0fbp")
                           for c in _yc_table.columns
                           if c not in ("Since", "Shape")},
            key="yield_curve_shifts")
        st.caption(
            "Change per maturity, in basis points. A curve where the long "
            "end moved MORE than the short end is steepening, whichever "
            "direction rates went overall.")
    if not bond_data.fred_is_configured():
        st.caption(bond_data.FRED_UNCONFIGURED)

# ==========================================
# BOND FUND SCREENER (PHASE 2.5)
# ==========================================
st.markdown("---")
st.header("Bond Fund Screener", anchor="bond-screener")
st.caption(bond_screener.INDIVIDUAL_BONDS_NOT_SCREENED)
_bs_rows, _bs_err = bond_screener.load_bond_universe()
if _bs_err:
    st.warning(_bs_err)
if not _bs_rows:
    st.info("No bond funds could be loaded right now.")
else:
    st.caption(f"{len(_bs_rows)} bond funds loaded, each with a duration "
               "measured from its own price behaviour.")
    if "bond_criteria" not in st.session_state:
        st.session_state["bond_criteria"] = [
            {"metric": "duration", "operator": "<", "threshold": 5.0}]

    st.markdown("**Preset screens**")
    _bs_cols = st.columns(len(bond_screener.PRESETS))
    for _bs_i, _bs_preset in enumerate(bond_screener.PRESETS):
        with _bs_cols[_bs_i]:
            if st.button(_bs_preset.name, key=f"bond_preset_{_bs_i}",
                         width="stretch", help=_bs_preset.detail):
                st.session_state["bond_criteria"] = [
                    {"metric": c.metric, "operator": c.operator,
                     "threshold": c.threshold}
                    for c in _bs_preset.criteria]
                st.rerun()

    st.markdown("**Filters**")
    _bs_remove = None
    for _bs_i, _bs_c in enumerate(st.session_state["bond_criteria"]):
        # Labels are prefixed AND numbered for the same reason the ETF
        # screener's are: Streamlit hashes (label, options, index, help)
        # to identify an unkeyed widget and label_visibility is not in
        # that hash, so two identical rows collide outright.
        _bs_suffix = "" if _bs_i == 0 else f" {_bs_i + 1}"
        _bs_m, _bs_o, _bs_v, _bs_x = st.columns([3, 2, 3, 1])
        with _bs_m:
            _bs_keys = [m.key for m in bond_screener.METRICS]
            _bs_metric = st.selectbox(
                f"Bond metric{_bs_suffix}", _bs_keys,
                index=_bs_keys.index(_bs_c["metric"])
                if _bs_c["metric"] in _bs_keys else 0,
                format_func=lambda k: bond_screener.METRICS_BY_KEY[k].label,
                label_visibility="collapsed")
        _bs_ops = bond_screener.operators_for(_bs_metric)
        with _bs_o:
            _bs_op = st.selectbox(
                f"Bond op{_bs_suffix}", _bs_ops,
                index=_bs_ops.index(_bs_c["operator"])
                if _bs_c["operator"] in _bs_ops else 0,
                label_visibility="collapsed")
        with _bs_v:
            if bond_screener.METRICS_BY_KEY[_bs_metric].kind == "text":
                _bs_types = list(bond_screener.fund_types())
                _bs_threshold = st.selectbox(
                    f"Bond value{_bs_suffix}", _bs_types,
                    index=_bs_types.index(str(_bs_c["threshold"]))
                    if str(_bs_c["threshold"]) in _bs_types else 0,
                    label_visibility="collapsed")
            else:
                _bs_threshold = st.number_input(
                    f"Bond threshold{_bs_suffix}",
                    value=float(_bs_c["threshold"])
                    if isinstance(_bs_c["threshold"], (int, float)) else 0.0,
                    label_visibility="collapsed")
        with _bs_x:
            if st.button("✕", key=f"bond_remove_{_bs_i}",
                         help="Remove this filter"):
                _bs_remove = _bs_i
        st.session_state["bond_criteria"][_bs_i] = {
            "metric": _bs_metric, "operator": _bs_op,
            "threshold": _bs_threshold}

    if _bs_remove is not None:
        st.session_state["bond_criteria"].pop(_bs_remove)
        st.rerun()
    if st.button("+ Add bond filter", key="bond_add_filter"):
        st.session_state["bond_criteria"].append(
            {"metric": "yield_pct", "operator": ">", "threshold": 4.0})
        st.rerun()

    _bs_criteria = [bond_screener.BondCriterion(**c)
                    for c in st.session_state["bond_criteria"]]
    _bs_passed, _bs_unjudged = bond_screener.run(_bs_rows, _bs_criteria)
    st.caption(
        f"{len(_bs_passed)} of {len(_bs_rows)} funds match"
        + (f" · {len(_bs_unjudged)} set aside because they do not report "
           "every filtered metric" if _bs_unjudged else ""))

    if not _bs_passed:
        st.info("No fund passes every filter. Loosen one, or try a preset.")
    else:
        _bs_table = pd.DataFrame([
            {"Symbol": m.row.symbol, "Name": m.row.name,
             "Type": m.row.fund_type, "Yield %": m.row.yield_pct,
             "Duration": m.row.duration, "Spread bp": m.row.spread_bps,
             "ER %": m.row.expense_ratio_pct, "AUM": m.row.assets,
             "1Y %": m.row.return_1y_pct}
            for m in _bs_passed[:bond_screener.MAX_RESULTS_SHOWN]])
        for _bs_col in ("Yield %", "Duration", "Spread bp", "ER %", "AUM",
                        "1Y %"):
            _bs_table[_bs_col] = pd.to_numeric(_bs_table[_bs_col],
                                               errors="coerce")
        st.dataframe(
            _bs_table, width="stretch", hide_index=True,
            column_config={
                "Yield %": st.column_config.NumberColumn("Yield %",
                                                         format="%.2f%%"),
                # Streamlit draws its own muted "None" into any null
                # numeric cell and gives no API to reword it, so the
                # meaning goes in the column's help.
                "Duration": st.column_config.NumberColumn(
                    "Duration", format="%.2f",
                    help="Years, measured by regressing the fund's daily "
                         "return on the change in the 10-year yield. A "
                         "NEGATIVE figure is not an error: rate-hedged "
                         "funds short treasuries to strip duration out."),
                "Spread bp": st.column_config.NumberColumn(
                    "Spread bp", format="%+.0f",
                    help="Yield over the treasury curve at the fund's own "
                         "duration. Blank where that duration is at or "
                         "below zero — a rate-hedged fund has no maturity "
                         "on the curve to match."),
                "ER %": st.column_config.NumberColumn("ER %", format="%.2f%%"),
                "AUM": st.column_config.NumberColumn("AUM", format="compact"),
                "1Y %": st.column_config.NumberColumn("1Y %", format="%.1f%%"),
            },
            key="bond_results_table")
        if len(_bs_passed) > bond_screener.MAX_RESULTS_SHOWN:
            st.caption(f"Showing the first {bond_screener.MAX_RESULTS_SHOWN}.")
        st.download_button(
            "Download these results (CSV)",
            _bs_table.to_csv(index=False).encode("utf-8"),
            file_name=f"quantix_bond_screen_{datetime.date.today()}.csv",
            mime="text/csv", key="bond_csv")
        st.caption(bond_screener.LADDER_UNAVAILABLE)

st.header("ETF Screener")
st.caption(
    "Screens a market-wide table of funds — no ticker list required, "
    "unlike the stock screener above. Refreshed every "
    f"{etf_screener.CACHE_TTL_SECONDS // 60} minutes.")

with st.expander("What this screener cannot filter on", expanded=False):
    for _etfs_gap in etf_screener.UNSUPPORTED_FILTERS:
        st.caption(f"· {_etfs_gap}")

_etfs_rows, _etfs_error = etf_screener.load_universe()
if _etfs_error:
    st.warning(_etfs_error)
else:
    st.caption(f"{len(_etfs_rows)} funds loaded.")

    # --- search ---------------------------------------------------------
    _etfs_query = st.text_input(
        "Find a fund", key="etf_search",
        placeholder="Symbol or name — e.g. vgt, vanguard, dividend",
        help="Substring match over the loaded table. Instant, because the "
             "universe is already in memory — no search index needed for "
             "250 rows.")
    if _etfs_query.strip():
        _etfs_hits = etf_screener.search(_etfs_rows, _etfs_query)
        if not _etfs_hits:
            st.caption(f'No fund matches "{_etfs_query.strip()}".')
        for _etfs_hit in _etfs_hits:
            if st.button(f"{_etfs_hit.symbol} — {_etfs_hit.name}",
                         key=f"etf_hit_{_etfs_hit.symbol}", width="stretch"):
                st.session_state["_pending_ticker"] = _etfs_hit.symbol
                st.rerun()

    # --- presets --------------------------------------------------------
    if "etf_criteria" not in st.session_state:
        st.session_state["etf_criteria"] = list(etf_screener.PRESETS[0].criteria)

    st.markdown("**Preset screens**")
    _etfs_cols = st.columns(len(etf_screener.PRESETS))
    for _etfs_col, _etfs_preset in zip(_etfs_cols, etf_screener.PRESETS):
        with _etfs_col:
            if st.button(_etfs_preset.name, key=f"etf_preset_{_etfs_preset.name}",
                         width="stretch", help=_etfs_preset.description):
                st.session_state["etf_criteria"] = list(_etfs_preset.criteria)
                st.rerun()

    # --- criteria -------------------------------------------------------
    st.markdown("**Filters**")
    _etfs_remove = None
    for _etfs_i, _etfs_crit in enumerate(st.session_state["etf_criteria"]):
        _etfs_m, _etfs_o, _etfs_v, _etfs_x = st.columns([3, 1.4, 2, 0.6])
        _etfs_keys = [m.key for m in etf_screener.METRICS]
        with _etfs_m:
            # Labels vary per row for the reason documented on the stock
            # screener: Streamlit hashes (label, options, index, help) to
            # identify an unkeyed widget and ignores label_visibility, so
            # two rows with matching parameters collide outright.
            # Prefixed "ETF", not just numbered. The suffix alone left row
            # 0 labelled "Op" — identical to the STOCK screener's row 0
            # Op, same options, same index — and Streamlit's auto-ID
            # collided across the two screeners, taking the page down with
            # StreamlitDuplicateElementId. The prefix is also plainly
            # better for a screen reader now that the page has two
            # screeners on it.
            _etfs_suffix = "" if _etfs_i == 0 else f" {_etfs_i + 1}"
            _etfs_metric = st.selectbox(
                f"ETF metric{_etfs_suffix}", _etfs_keys,
                index=_etfs_keys.index(_etfs_crit.metric)
                if _etfs_crit.metric in _etfs_keys else 0,
                format_func=lambda k: etf_screener.METRICS_BY_KEY[k].label,
                label_visibility="visible" if _etfs_i == 0 else "collapsed")
        _etfs_ops = list(etf_screener.operators_for(_etfs_metric))
        with _etfs_o:
            _etfs_op = st.selectbox(
                f"ETF op{_etfs_suffix}", _etfs_ops,
                index=_etfs_ops.index(_etfs_crit.operator)
                if _etfs_crit.operator in _etfs_ops else 0,
                label_visibility="visible" if _etfs_i == 0 else "collapsed")
        with _etfs_v:
            _etfs_spec = etf_screener.METRICS_BY_KEY[_etfs_metric]
            if _etfs_spec.kind == "text":
                _etfs_threshold = st.text_input(
                    f"ETF value{_etfs_suffix}",
                    value=str(_etfs_crit.threshold)
                    if isinstance(_etfs_crit.threshold, str) else "",
                    label_visibility="visible" if _etfs_i == 0 else "collapsed")
            else:
                _etfs_threshold = st.number_input(
                    f"ETF threshold{_etfs_suffix}",
                    value=float(_etfs_crit.threshold)
                    if isinstance(_etfs_crit.threshold, (int, float)) else 0.0,
                    label_visibility="visible" if _etfs_i == 0 else "collapsed")
        with _etfs_x:
            if _etfs_i == 0:
                st.markdown("&nbsp;")
            if st.button("✕", key=f"etf_remove_{_etfs_i}", help="Remove this filter"):
                _etfs_remove = _etfs_i
        st.session_state["etf_criteria"][_etfs_i] = etf_screener.EtfCriterion(
            metric=_etfs_metric, operator=_etfs_op, threshold=_etfs_threshold)

    if _etfs_remove is not None:
        st.session_state["etf_criteria"].pop(_etfs_remove)
        st.rerun()

    if st.button("+ Add ETF filter", key="etf_add_filter"):
        st.session_state["etf_criteria"].append(
            etf_screener.EtfCriterion("expense_ratio_pct", "<", 0.5))
        st.rerun()

    # --- results --------------------------------------------------------
    _etfs_passed, _etfs_unjudged = etf_screener.run(
        _etfs_rows, st.session_state["etf_criteria"])
    st.caption(
        f"{len(_etfs_passed)} of {len(_etfs_rows)} funds match"
        + (f" · {len(_etfs_unjudged)} set aside because they do not report "
           "every filtered metric" if _etfs_unjudged else ""))

    if not _etfs_passed:
        st.info("No fund passes every filter. Loosen one, or try a preset.")
    else:
        _etfs_table = etf_screener.results_frame(
            _etfs_passed[:etf_screener.MAX_RESULTS_SHOWN])
        _etfs_event = st.dataframe(
            _etfs_table, width="stretch", hide_index=True,
            column_config=etf_screener.column_config(),
            on_select="rerun", selection_mode="single-row",
            key="etf_results_table")
        if len(_etfs_passed) > etf_screener.MAX_RESULTS_SHOWN:
            st.caption(f"Showing the first {etf_screener.MAX_RESULTS_SHOWN}.")

        _etfs_sel = (_etfs_event.selection.rows
                     if _etfs_event and _etfs_event.selection else [])
        if _etfs_sel:
            _etfs_pick = _etfs_table.iloc[_etfs_sel[0]]["Symbol"]
            if st.button(f"Open full analysis for {_etfs_pick} →",
                         key="etf_open_pick"):
                st.session_state["_pending_ticker"] = _etfs_pick
                st.rerun()

        st.download_button(
            "Download these results (CSV)",
            _etfs_table.to_csv(index=False).encode("utf-8"),
            file_name=f"quantix_etf_screen_{datetime.date.today()}.csv",
            mime="text/csv", key="etf_csv")

# ==========================================
# SMART RISK-AWARE ALERTS
# ==========================================
st.markdown("---")
st.header("Smart Risk-Aware Alerts")
# The limitation that changes what someone expects stays on screen; the
# rest of the explanation moves into the expander. Collapsing the whole
# caption would bury "this is not a push notification", which is the one
# sentence that stops someone relying on an alert that will never arrive.
st.caption(
    "Snapshot check across your Institutional Watchlist, run when you click **Check Alerts** — "
    "not a push notification."
)
with st.expander("How these alerts work", expanded=False):
    st.markdown(
        "Built from Quantix's own risk engine — Composite Risk Score, Altman Z-Score, 1-Day VaR, "
        "Expected Shortfall and Max Drawdown — evaluated across every ticker in your Institutional "
        "Watchlist.\n\n"
        "Quantix is a stateless app with no background worker, so this is an **on-load snapshot** "
        "(\"triggered right now\") rather than a historical crossing event: a threshold that was "
        "breached yesterday and has since recovered will not show. Live push delivery by email or "
        "SMS would need infrastructure that isn't built here.\n\n"
        "For continuous per-ticker monitoring while a tab is open, use the Real-Time Alert Engine "
        "below instead."
    )

if "risk_alert_rules" not in st.session_state:
    _persisted_alert_rules = load_rules()
    if _persisted_alert_rules is not None:
        st.session_state["risk_alert_rules"] = _persisted_alert_rules
    else:
        st.session_state["risk_alert_rules"] = [
            {"metric": "risk_score", "operator": "<", "threshold": 50.0},
            {"metric": "altman_z", "operator": "<", "threshold": effective_risk().altman_grey_zone},
        ]
        save_rules(st.session_state["risk_alert_rules"])

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

save_rules(st.session_state["risk_alert_rules"])

if _alert_remove_index is not None:
    st.session_state["risk_alert_rules"].pop(_alert_remove_index)
    save_rules(st.session_state["risk_alert_rules"])
    st.rerun()

_alert_btn_col1, _alert_btn_col2 = st.columns([1, 1])
with _alert_btn_col1:
    if st.button("+ Add Alert Rule"):
        st.session_state["risk_alert_rules"].append({"metric": "max_drawdown", "operator": "<", "threshold": -0.20})
        save_rules(st.session_state["risk_alert_rules"])
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
        _alert_bar = st.progress(0.0, text="Starting risk check…")

        def _alert_progress(done, total, ticker):
            _alert_bar.progress(
                loading_states.progress_fraction(done, total),
                text="Checking " + loading_states.progress_text(done, total, ticker))

        _alert_snapshots = compute_watchlist_snapshots(
            _alert_watchlist, _on_progress=_alert_progress)
        _alert_bar.empty()
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
        st.success(f"No alerts triggered across your {len(_alert_snapshots)}-ticker watchlist right now.")

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
def _hc_fmt(metric, value) -> str:
    """One historical-comparison cell. An unavailable value renders as an
    explicit "N/A" rather than a dash or a zero — the whole point of this
    panel is that it never implies a number it doesn't have."""
    if value is None:
        return "N/A"
    if metric.unit == "$":
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= cutoff:
                return f"${value / cutoff:,.2f}{suffix}"
        return f"${value:,.2f}"
    return f"{value:,.{metric.decimals}f}{metric.unit}"


def _hc_delta_text(metric) -> str:
    """The then->now change, with an arrow only where the metric has a
    meaningful direction. Price and market cap deliberately get no
    better/worse arrow — a higher price is not self-evidently good."""
    delta = metric.delta
    if delta is None:
        return "—"
    body = _hc_fmt(metric, delta) if metric.unit == "$" else f"{delta:+,.{metric.decimals}f}{metric.unit}"
    if metric.higher_is_better is None or delta == 0:
        return body
    improved = delta > 0 if metric.higher_is_better else delta < 0
    return body


def _pf_price_loader(ticker, start, end):
    """Closing prices for the portfolio dashboard, as a Series.

    Wraps the app's existing cached fetch rather than adding a second
    path to Yahoo, so a ticker already loaded elsewhere on the page is
    served from the same cache. Returns None rather than raising when
    there's no data — build_performance treats that as "exclude and say
    so", which is what keeps an unpriceable holding visible instead of
    silently dropped.
    """
    history, _errors = load_price_history_only(ticker, start, end)
    if history is None or history.empty or "Close" not in history:
        return None
    closes = history["Close"].dropna()
    return closes if not closes.empty else None


def _pf_money(value):
    """Currency-ish formatting that never fabricates a number: None stays
    visibly unavailable rather than rendering as 0.00."""
    if value is None:
        return "unavailable"
    return f"{value:,.2f}"


def _pf_pct(value):
    if value is None:
        return "unavailable"
    return f"{value:+.2f}%"


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
    f"Per-ticker rules rechecked every {REALTIME_ALERTS.poll_interval_seconds}s **while this tab stays open** — "
    "closing it stops monitoring. Delivery is in-app only."
)
with st.expander("How these alerts work", expanded=False):
    st.markdown(
        f"Rules cover price levels, technical crossovers and risk thresholds, rechecked automatically every "
        f"{REALTIME_ALERTS.poll_interval_seconds} seconds while this tab is open.\n\n"
        "**Not a background service.** Closing the tab stops monitoring, same as everything else in this "
        "stateless app. Delivery is in-app only — no email, SMS or push, because this app has no messaging "
        "credentials to send them with.\n\n"
        "Rules and trigger history **are** saved to a local file and survive a restart, unlike the rest of "
        "this app's session-only state. Signed in, those rules are private to your account; signed out "
        "they're shared by whoever runs this instance."
    )

if "rt_alert_rules" not in st.session_state:
    _rt_init_rules, _rt_init_history = rt_load_store()
    st.session_state["rt_alert_rules"] = _rt_init_rules
    st.session_state["rt_alert_history"] = _rt_init_history
    st.session_state["rt_alert_prev_active"] = {}

# ⌘⇧A, from the keyboard listener or the command palette. Consumed here
# because this is the first point where the rule store exists.
#
# The ticker comes from session_state rather than ticker_symbol, which
# the sidebar does not assign until several hundred lines below this —
# the same ordering trap the alerts empty state hit. CHART_DEFAULTS is
# the last resort because ticker_input is not seeded until then either.
if st.session_state.pop("kbd_new_alert_requested", False):
    _kbd_alert_ticker = (st.session_state.get("rt_new_ticker")
                         or st.session_state.get("ticker_input")
                         or CHART_DEFAULTS.default_ticker
                         or "").strip().upper()
    if not _kbd_alert_ticker:
        st.warning("No ticker to alert on yet — enter one first.")
    elif len(st.session_state["rt_alert_rules"]) >= REALTIME_ALERTS.max_rules:
        st.warning(f"Rule limit reached ({REALTIME_ALERTS.max_rules} max) — "
                   "remove one in the Real-Time Alert Engine first.")
    else:
        _kbd_rule = RealtimeAlertRule(
            id=rt_new_rule_id(), ticker=_kbd_alert_ticker,
            trigger_type=RT_FIRST_ALERT_TRIGGER,
            created_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        st.session_state["rt_alert_rules"].append(_kbd_rule)
        rt_save_store(st.session_state["rt_alert_rules"],
                      st.session_state["rt_alert_history"])
        log_event(logger, logging.INFO, "user.keyboard_new_alert",
                  ticker=_kbd_alert_ticker)
        # A keystroke that silently changes stored state is a bad
        # surprise, so say exactly what was created and where to undo it.
        st.toast(
            f"Alert created: {_kbd_alert_ticker} — "
            f"{RT_TRIGGER_LABELS[RT_FIRST_ALERT_TRIGGER]}. "
            "Remove it in the Real-Time Alert Engine.",
            icon=":material/notifications_active:",
        )

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

# Trigger choice is two steps: a category, then the triggers inside it.
# Nine flat options mixed "price hits X" with "MACD crosses" with "Altman
# Z drops" — three different questions someone arrives with.
#
# The category box keeps its key because its options never change. The
# trigger box must NOT have one: its options change with the category, and
# a stored value outside the new list raises on selection. The selected
# trigger is therefore held in session_state under a non-widget key and
# passed back as a computed index — the pattern the screener's criteria
# rows already use for the same reason.
if "rt_new_type_value" not in st.session_state:
    st.session_state["rt_new_type_value"] = RT_ALL_TRIGGER_TYPES[0]

_rt_r1c1, _rt_r1c2, _rt_r1c3 = st.columns([1, 1.2, 1.8])
with _rt_r1c1:
    _rt_new_ticker = st.text_input("Ticker", key="rt_new_ticker", placeholder="e.g. AAPL").strip().upper()
with _rt_r1c2:
    _rt_category = st.selectbox(
        "Alert type", RT_CATEGORY_NAMES,
        index=RT_CATEGORY_NAMES.index(rt_category_of(st.session_state["rt_new_type_value"])),
        key="rt_new_category",
        help="Narrows the trigger list below to one kind of alert.",
    )
with _rt_r1c3:
    _rt_choices = list(rt_triggers_in(_rt_category))
    _rt_current = st.session_state["rt_new_type_value"]
    _rt_new_type = st.selectbox(
        "Trigger Condition", _rt_choices,
        index=_rt_choices.index(_rt_current) if _rt_current in _rt_choices else 0,
        format_func=lambda k: RT_TRIGGER_LABELS[k],
    )
    st.session_state["rt_new_type_value"] = _rt_new_type

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
            st.session_state["rt_new_fund_threshold"] = rt_effective_default_threshold(_rt_new_metric)
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
        _rt_first_label = RT_TRIGGER_LABELS[RT_FIRST_ALERT_TRIGGER]
        # NOT ticker_symbol: this fragment runs at line ~1875, and the
        # sidebar assigns ticker_symbol several hundred lines below — a
        # NameError that only fires for a user with no rules yet, which
        # is to say every new user. Read from session_state instead of
        # the script-level names, because on a TIMER rerun Streamlit
        # re-executes this function alone and script locals are whatever
        # the last full run left behind. Prefers the rule form's own
        # ticker box, since that is the one the reader is looking at.
        # Third fallback matters: ticker_input is not seeded until line
        # ~2412, well below this fragment, so on the FIRST run of a
        # session neither session key exists and the button would be
        # missing exactly when a new user needs it. The constant is the
        # same value that seeding uses, and is also what the page is
        # already analysing at that moment, so the offer is accurate.
        _rt_seed_ticker = (st.session_state.get("rt_new_ticker")
                           or st.session_state.get("ticker_input")
                           or CHART_DEFAULTS.default_ticker
                           or "").strip().upper()
        if empty_states.render(
            "No alerts set",
            "Alerts watch a symbol while this tab stays open and tell you "
            "the moment a condition is met."
            + ("" if _rt_seed_ticker else
               " Enter a ticker above to create one."),
            # No ticker in either box means nothing to act on, so no
            # button — an action that cannot complete is worse than none.
            action_label=(f"Create your first alert for {_rt_seed_ticker} →"
                          if _rt_seed_ticker else None),
            key="empty_rt_first_alert",
            help_text=f"Creates one rule: {_rt_seed_ticker} — {_rt_first_label}. "
                      "No threshold to choose; it uses the app's own period. "
                      "Remove it with the ✕ beside it.",
        ):
            _rt_seed = RealtimeAlertRule(
                id=rt_new_rule_id(), ticker=_rt_seed_ticker,
                trigger_type=RT_FIRST_ALERT_TRIGGER,
                created_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )
            st.session_state["rt_alert_rules"].append(_rt_seed)
            rt_save_store(st.session_state["rt_alert_rules"],
                          st.session_state["rt_alert_history"])
            empty_states.log_action("first_realtime_alert", ticker=_rt_seed_ticker)
            # scope="app": the Active Rules list this adds to is rendered
            # OUTSIDE this fragment, so a fragment-scoped rerun would
            # redraw the empty state and leave the new rule invisible.
            st.rerun(scope="app")
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
            # A snoozed rule records nothing and raises no toast. This is
            # the only place snooze can meaningfully act: the event is what
            # the bell counts, so suppressing it here is what makes the
            # mute real rather than cosmetic. The rule still EVALUATES —
            # the active/cleared banner below still reflects reality — it
            # simply stops generating notifications until the mute lapses.
            if notifications.is_muted(_rt_rid):
                continue
            _rt_result = _rt_results[_rt_rid]
            st.toast(_rt_md_escape_dollar(f"{_rt_rule.label} — {_rt_result.detail}"))
            st.session_state["rt_alert_history"].append(RealtimeTriggerEvent(
                rule_id=_rt_rid, ticker=_rt_rule.ticker, trigger_type=_rt_rule.trigger_type,
                detail=_rt_result.detail, triggered_at=_rt_now_iso,
            ))
            log_event(logger, logging.INFO, "user.realtime_alert_triggered", ticker=_rt_rule.ticker, trigger_type=_rt_rule.trigger_type)
        rt_save_store(st.session_state["rt_alert_rules"], st.session_state["rt_alert_history"])

    st.session_state["rt_alert_prev_active"] = _rt_active_now

    _rt_active_rules = [r for r in _rt_rules if _rt_active_now.get(r.id)]
    if _rt_active_rules:
        st.error(f"{len(_rt_active_rules)} alert(s) currently active")
        for _rt_r in _rt_active_rules:
            st.markdown(_rt_md_escape_dollar(f"- **{_rt_r.label}** — {_rt_results[_rt_r.id].detail}"))
    else:
        st.success("No alert conditions currently met.")

    _rt_issue_rules = [r for r in _rt_rules if _rt_results[r.id].status != "ok"]
    if _rt_issue_rules:
        with st.expander(f"{len(_rt_issue_rules)} rule(s) with data issues", expanded=False):
            for _rt_r in _rt_issue_rules:
                st.caption(_rt_md_escape_dollar(f"{_rt_r.label}: {_rt_results[_rt_r.id].detail}"))

    # A pulse here is honest: this fragment really does re-run on a timer.
    # Nothing else on the page gets one — every other figure is as old as
    # the last rerun, and a pulse would imply a liveness this app, being a
    # stateless script with no background worker, does not have.
    st.markdown(
        loading_states.pulse(
            f"Checked {datetime.datetime.now().strftime('%H:%M:%S')} · "
            f"rechecking every {REALTIME_ALERTS.poll_interval_seconds}s "
            "while this tab stays open."),
        unsafe_allow_html=True,
    )


_render_realtime_alerts_fragment()

if st.session_state["rt_alert_history"]:
    # Opened by the bell's "See all" — the dropdown shows the most recent
    # few and this is the full list, rather than a third place that
    # renders the same events.
    with st.expander(f"Trigger history ({len(st.session_state['rt_alert_history'])})",
                     expanded=bool(st.session_state.pop("_notif_open_history", False))):
        _rt_hist_rows = [
            {
                "When": h.triggered_at, "Ticker": h.ticker,
                "Trigger": RT_TRIGGER_LABELS.get(h.trigger_type, h.trigger_type), "Detail": h.detail,
            }
            for h in reversed(st.session_state["rt_alert_history"])
        ]
        st.table(pd.DataFrame(_rt_hist_rows))


# No logo in the sidebar. A large boxed lockup at the top of the rail read
# as an advertisement inside the user's own workspace, and it pushed the
# controls people actually came for below the fold. The product is named
# on the login page and on every export that leaves the building; a
# working tool does not need to keep saying so.

# --- Sidebar: Account ---
# Top of the rail, above Target Configuration, because signing in changes
# what every panel below it shows. See auth.py for why there is no
# hand-written auth code here and why GitHub needs an OIDC broker.
_auth_reason = auth.unavailable_reason()
with st.sidebar.expander(
    f"{_auth_user.display_name}" if _auth_user else "Account",
    expanded=False,
):
    if _auth_user is not None:
        if _auth_user.email:
            st.caption(_auth_user.email)
        st.caption(
            "Your watchlists, favourites, theme, thresholds, alert rules and saved "
            "scenarios are private to this account on this instance. Team Notes stay "
            "shared — that's the point of them."
        )

        # First sign-in on an instance that already has data: without this
        # the app looks empty and empty is indistinguishable from lost.
        _auth_shared = auth.shared_data_files()
        if auth.adoption_pending(_auth_user.key):
            st.info(
                f"This instance has {len(_auth_shared)} saved "
                f"{'setting' if len(_auth_shared) == 1 else 'settings'} files from before you "
                "signed in. They're still there — signing out shows them again — but your "
                "account starts empty."
            )
            if st.button("Copy them into my account", key="auth_adopt"):
                _auth_copied, _auth_errs = auth.adopt_shared_data(_auth_user.key)
                for _auth_e in _auth_errs:
                    st.warning(f"Couldn't copy {_auth_e}")
                if _auth_copied:
                    for _k in _AUTH_SCOPED_STATE:
                        st.session_state.pop(_k, None)
                    st.success(f"Copied {len(_auth_copied)} files. The originals are untouched.")
                    st.rerun()
                elif not _auth_errs:
                    st.info("Nothing to copy.")
                    st.rerun()
            if st.button("No thanks", key="auth_adopt_decline"):
                # Declining is an answer too. Without a way to say no, the
                # offer would sit in the panel forever for anyone who
                # genuinely wants a clean account.
                auth.mark_adoption_handled(_auth_user.key, adopted=False)
                st.rerun()

        if st.button("Sign out", key="auth_logout"):
            # Two sign-in paths, two ways out. st.logout() only clears
            # Streamlit's own OIDC cookie and does nothing to a local
            # session, so a password user clicking Sign out would have
            # stayed signed in. Clear the local session either way, then
            # hand off to st.logout() only when OIDC is what's active.
            auth.sign_out_local()
            login_page.reset_state()
            if auth.is_logged_in():
                st.logout()
            else:
                st.rerun()
    elif _auth_reason:
        st.caption(
            "Sign in to keep your watchlists, favourites, theme, thresholds and alert "
            "rules private to you instead of shared with everyone using this instance — "
            "and to have your Team Notes carry a verified name rather than a typed one."
        )
        st.info(_auth_reason)
    else:
        st.caption(
            "Signing in gives you your own watchlists, favourites, theme, thresholds, "
            "alert rules and scenarios. Nothing you've saved so far is lost — it stays "
            "on the signed-out profile and can be copied across in one click."
        )
        for _auth_p in auth.configured_providers():
            if st.button(
                f"Sign in with {auth.provider_label(_auth_p)}",
                key=f"auth_login_{_auth_p or 'default'}",
                width="stretch",
            ):
                # st.login() with no argument is the unnamed-default-provider
                # form; auth.configured_providers() returns "" for that case.
                st.login(_auth_p) if _auth_p else st.login()

# --- Sidebar: Branding ---
with st.sidebar.expander("Branding", expanded=False):
    st.caption(brand_summary())
    for _br_note in brand_config_notes():
        st.warning(_br_note)
    st.caption(
        "Set a [branding] section in .streamlit/secrets.toml to run this instance under "
        "your own name, tagline and accent colour — see .streamlit/secrets.toml.example."
    )
    st.caption(
        "**Disclosures are not brandable.** \"Not investment advice\", the unavailable-data "
        "notices and the measured model accuracy stay whatever the branding says. They're "
        "why the numbers here can be relied on."
    )
    st.caption(
        "Licensing to more than one firm means one deployment each: Streamlit reads secrets "
        "once per process, so a single instance can't give two firms separate sign-in "
        "providers or mail senders."
    )

# --- Sidebar: Slack ---
# Posting happens on the SCHEDULED run, not in-tab: alerts only evaluate
# while a browser tab is open, and a Slack message about something
# already on your screen isn't the point. See alert_watch.py.
with st.sidebar.expander("Slack Alerts", expanded=False):
    st.caption(
        "Posts your triggered alert rules to a Slack channel, so they reach you when "
        "Quantix is closed. Alerts only — not the digest, not note mentions."
    )
    _sl_reason = slack_unavailable_reason()
    if _sl_reason:
        st.info(_sl_reason)
    else:
        st.success("Slack webhook configured.")

    _sl_rules, _sl_history = load_rt_store()
    st.caption(
        f"{len(_sl_rules)} alert rule(s) configured. Rules are created in the Real-Time "
        "Alerts panel on the Overview tab; this only controls where they're delivered."
    )

    if st.button("Check alerts now (posts nothing)", key="slack_dry_run"):
        with st.spinner("Evaluating rules…"):
            _sl_posted, _sl_messages = alert_watch_run(post=False)
        for _sl_m in _sl_messages:
            st.caption(_sl_m)

    st.markdown("---")
    st.caption(
        "Delivery runs on the same schedule as the digest — one crontab entry drives both. "
        "The line is shown in the Email Digest panel above. Nothing is posted until you "
        "install it; opening this panel doesn't start anything."
    )

# --- Sidebar: Email Digest ---
# The schedule lives in cron, not here — Streamlit runs nothing while the
# app is shut, which is exactly when a digest for an inactive user has to
# go out. See digest.py. This panel configures it and shows the line to
# install; installing it stays a deliberate act.
with st.sidebar.expander("Email Digest", expanded=False):
    _dg_owner = _auth_user.key if _auth_user else ""
    if "digest_settings" not in st.session_state:
        st.session_state["digest_settings"] = digest_settings_for(_dg_owner)
    _dg = st.session_state["digest_settings"]

    st.caption(
        "A periodic email summarising how your watchlist moved, which alerts fired, and "
        "which risk thresholds are breached. Quantix doesn't track holdings, so it covers "
        "what you watch rather than what you own."
    )
    if not is_email_configured():
        st.info(
            "Email isn't configured on this instance, so nothing can be sent yet. Preview "
            "works regardless. See .streamlit/secrets.toml.example to set up SMTP."
        )

    _dg_enabled = st.checkbox(
        "Send the digest on a schedule", value=_dg.enabled, key="digest_enabled",
        help="Only takes effect once you install the schedule line below — nothing sends on its own until then.",
    )
    _dg_recipient = st.text_input(
        "Send to", value=_dg.recipient, key="digest_recipient", placeholder="you@example.com",
    )
    _dg_period = st.number_input(
        "Cover the last (days)", min_value=DIGEST.min_period_days, max_value=DIGEST.max_period_days,
        value=_dg.period_days, step=1, key="digest_period",
        help="Also how often it sends — a 7-day digest goes out weekly.",
    )
    _dg_watch = st.checkbox("Watchlist movement", value=_dg.include_watchlist, key="digest_watch")
    _dg_alerts = st.checkbox("Alerts that fired", value=_dg.include_alerts, key="digest_alerts")
    _dg_risk = st.checkbox("Risk thresholds breached", value=_dg.include_risk, key="digest_risk")

    if st.button("Save digest settings", key="digest_save"):
        _dg_new = DigestSettings(
            owner_key=_dg_owner, recipient=_dg_recipient.strip(), enabled=_dg_enabled,
            period_days=int(_dg_period), include_watchlist=_dg_watch,
            include_alerts=_dg_alerts, include_risk=_dg_risk,
            last_sent_at=_dg.last_sent_at,
        )
        _dg_err = digest_validate(_dg_new)
        if _dg_err:
            st.warning(_dg_err)
        else:
            digest_save_settings(_dg_new)
            st.session_state["digest_settings"] = _dg_new
            log_event(logger, logging.INFO, "user.digest_settings_saved", enabled=_dg_new.enabled)
            st.success("Saved.")

    if st.button("Preview digest now", key="digest_preview"):
        # Preview never sends — it builds and renders the same text the
        # scheduled run would email, so it can be read before committing
        # to a schedule.
        with st.spinner("Building digest…"):
            _dg_preview = digest_build(DigestSettings(
                owner_key=_dg_owner, recipient=_dg_recipient.strip(),
                period_days=int(_dg_period), include_watchlist=_dg_watch,
                include_alerts=_dg_alerts, include_risk=_dg_risk,
            ))
        st.caption(f"Subject: {_dg_preview.subject()}")
        st.code(_dg_preview.as_text(), language=None)
        st.caption("Preview only — nothing was sent.")

    st.markdown("---")
    st.markdown("**Scheduling**")
    st.caption(
        "Streamlit runs nothing while the app is closed, which is exactly when a digest needs "
        "to go out. Install this line in your crontab (`crontab -e`) to run it weekly. "
        "It's yours to install — Quantix won't schedule mail on your behalf."
    )
    st.code(digest_cron_line(), language="bash")
    if _dg.last_sent_at:
        st.caption(f"Last sent: {_dg.last_sent_at.replace('T', ' ')}")

# --- Sidebar: Help & Support ---
# Search first, contact second. See support.py for why there is no live
# chat here despite the originating task asking for one: nobody is
# staffing a chat on a locally-run instance, and an unanswered box is a
# promise the app can't keep.
# expanded= is driven by the account menu's Help item: a popover cannot
# expand something inside the sidebar, so it sets a flag that this reads
# once. help_requested() consumes it, so the panel opens on that run and
# behaves normally afterwards rather than being pinned open.
with st.sidebar.expander("Help & Support", expanded=profile_menu.help_requested()):
    if "support_index" not in st.session_state:
        st.session_state["support_index"] = support_build_index()
    _sp_index = st.session_state["support_index"]

    _sp_query = st.text_input(
        "Search help", key="support_query",
        placeholder="e.g. sharpe, sign in, alerts, missing data",
        help=f"Searches {len(_sp_index)} articles: the FAQ plus every metric and chart explanation in the app.",
    )
    if _sp_query.strip():
        _sp_hits = support_search(_sp_query, _sp_index)
        if _sp_hits:
            st.caption(f"{len(_sp_hits)} result{'s' if len(_sp_hits) != 1 else ''}")
            for _sp_hit in _sp_hits:
                with st.expander(f"{_sp_hit.title}  ·  {_sp_hit.category}", expanded=False):
                    st.markdown(_rt_md_escape_dollar(_sp_hit.body))
        else:
            st.info(
                f'Nothing matched "{_sp_query}". Try a single word — every search term has to '
                "appear for an article to show. Or send a question below."
            )
    else:
        st.caption("Common questions")
        for _sp_start in SUPPORT_STARTERS:
            _sp_art = next((a for a in _sp_index if a.id == _sp_start), None)
            if _sp_art is not None:
                with st.expander(_sp_art.title, expanded=False):
                    st.markdown(_rt_md_escape_dollar(_sp_art.body))

    st.markdown("---")
    st.markdown("**Still stuck? Send a report**")
    if not support_destination_configured():
        st.caption(
            "No support address is configured on this instance — expected, since Quantix "
            "ships with no support desk behind it. Fill this in anyway and it produces a "
            "formatted report you can copy into an email or a GitHub issue."
        )

    if st.session_state.pop("_support_clear_form", False):
        st.session_state["support_subject"] = ""
        st.session_state["support_body"] = ""

    _sp_category = st.selectbox("Type", SUPPORT.categories, key="support_category")
    _sp_subject = st.text_input("Subject", key="support_subject", placeholder="One line")
    _sp_body = st.text_area(
        "What happened?", key="support_body",
        placeholder="What you did, what you expected, what happened instead.",
    )
    _sp_reply = st.text_input(
        "Your email (optional)", key="support_reply",
        placeholder="so a reply can reach you",
    )
    # Off by default and shown in full before sending — recent log lines
    # name the tickers you've been researching, which is yours to disclose
    # deliberately rather than discover afterwards.
    _sp_diag_on = st.checkbox(
        "Attach diagnostics", key="support_diagnostics", value=False,
        help="Environment details and recent log lines. Review exactly what would be sent below.",
    )
    _sp_diag = ""
    if _sp_diag_on:
        # NOT `ticker_symbol` — this panel renders ABOVE Target Configuration
        # in script order, so that name doesn't exist yet and referencing it
        # here raises NameError the moment the box is ticked. The widget's
        # session_state key survives from the previous run, which is what a
        # diagnostics snapshot actually wants anyway.
        _sp_diag = support_diagnostics(
            extra={"Current ticker": st.session_state.get("ticker_input", "(not set yet)")})
        with st.expander("Review what would be attached", expanded=False):
            st.caption(
                "This is the complete text that would be included. It names the tickers you "
                "have looked at recently."
            )
            st.code(_sp_diag, language=None)

    if st.button("Send report", type="primary", key="support_send"):
        _sp_report, _sp_err = support_compose(
            _sp_category, _sp_subject, _sp_body, _sp_reply, _sp_diag,
        )
        if _sp_err:
            st.warning(_sp_err)
        else:
            _sp_ok, _sp_send_err = (False, None)
            if support_destination_configured() and is_email_configured():
                _sp_ok, _sp_send_err = support_send(_sp_report)
            if _sp_ok:
                st.success("Report sent.")
                st.session_state["_support_clear_form"] = True
                log_event(logger, logging.INFO, "user.support_report_sent",
                          category=_sp_report.category, diagnostics=bool(_sp_diag))
                st.rerun()
            else:
                # No destination, or the send failed. Either way the user's
                # words must not be lost — show the composed report so it can
                # be copied somewhere that will actually be read.
                if _sp_send_err:
                    st.warning(_sp_send_err)
                elif not is_email_configured():
                    st.info(
                        "Email isn't configured on this instance, so nothing was sent. "
                        "Copy the report below."
                    )
                st.code(
                    f"Subject: [Quantix {_sp_report.category}] {_sp_report.subject}\n\n"
                    f"{_sp_report.as_email_body()}",
                    language=None,
                )

# --- Sidebar: API Keys ---
# Sits under Account because a key belongs to whoever created it. See
# api_keys.py for why the store is shared rather than namespaced, and
# api_server.py for what a key can actually reach (reads only).
with st.sidebar.expander("API Keys", expanded=False):
    if "api_key_store" not in st.session_state:
        st.session_state["api_key_store"] = load_api_key_store()
    _ak_store = st.session_state["api_key_store"]
    _ak_owner = _auth_user.key if _auth_user else ""

    st.caption(
        "Keys let scripts and other programs read your Quantix analysis without your "
        "login. The API is **read-only** — it has no endpoint that places trades or "
        "changes anything, because Quantix has no brokerage connection."
    )

    # A freshly-created key is shown exactly once. Held in session_state
    # across the rerun that refreshes the list, then dropped on dismiss —
    # it is never written anywhere, which is the whole point.
    _ak_fresh = st.session_state.get("_api_key_plaintext")
    if _ak_fresh:
        st.success("Key created — copy it now.")
        st.code(_ak_fresh, language=None)
        st.warning(
            "This is the only time this key is shown. Only its hash is stored, so it "
            "cannot be looked up again. If you lose it, revoke it and make another."
        )
        if st.button("I've copied it", key="api_key_dismiss"):
            st.session_state.pop("_api_key_plaintext", None)
            st.rerun()

    _ak_mine = api_keys_for_owner(_ak_store, _ak_owner)
    if _ak_mine:
        st.markdown("**Your keys**")
        for _ak in _ak_mine:
            # [5, 1] squeezed the action button to ~25px in the sidebar and
            # wrapped its label one letter per line. Matches the ✕ affordance
            # the watchlist and notes panels already use for the same reason.
            _ak_cols = st.columns([6, 1])
            with _ak_cols[0]:
                _ak_badge = {"active": "Active", "expired": "Expired", "revoked": "Revoked"}[_ak.status]
                st.markdown(f"{_ak_badge} **{_ak.name}** · `{_ak.id}` · {_ak.status}")
                _ak_bits = [", ".join(_ak.scopes) or "no scopes"]
                if _ak.expires_at:
                    _ak_bits.append(f"expires {_ak.expires_at[:10]}")
                _ak_bits.append(f"last used {_ak.last_used_at[:16]}" if _ak.last_used_at else "never used")
                st.caption(" · ".join(_ak_bits))
            with _ak_cols[1]:
                if not _ak.revoked and st.button(
                    "✕", key=f"api_key_revoke_{_ak.id}",
                    help=f"Revoke '{_ak.name}' — any script using it stops working immediately.",
                ):
                    _ak_store = revoke_api_key(_ak_store, _ak.id)
                    st.session_state["api_key_store"] = _ak_store
                    save_api_key_store(_ak_store)
                    st.rerun()
    else:
        st.caption("No keys yet.")

    st.markdown("---")
    # Deferred clear — assigning a widget's own key after it renders does
    # nothing (see CLAUDE.md and the Team Notes compose box).
    if st.session_state.pop("_api_key_clear_form", False):
        st.session_state["api_key_name"] = ""
    _ak_name = st.text_input(
        "New key name", key="api_key_name", placeholder="e.g. nightly-screener",
        help="Only for your own reference — it identifies the key in this list.",
    )
    _ak_scopes = st.multiselect(
        "Scopes", options=list(API_SCOPES.keys()), default=list(API_KEY_DEFAULT_SCOPES),
        key="api_key_scopes",
        help="What this key may read. Grant only what the script actually needs.",
        format_func=lambda s: s,
    )
    for _ak_s in _ak_scopes:
        st.caption(f"`{_ak_s}` — {API_SCOPES[_ak_s]}")
    _ak_expiry = st.number_input(
        "Expires in (days)", min_value=0, max_value=API_KEYS.max_expiry_days,
        value=API_KEYS.default_expiry_days, step=30, key="api_key_expiry",
        help="0 means the key never expires. A dated key limits the damage of one that leaks.",
    )
    if st.button("Create key", type="primary", key="api_key_create"):
        _ak_store, _ak_new, _ak_plain, _ak_err = create_api_key(
            _ak_store, _ak_name, tuple(_ak_scopes), owner_key=_ak_owner,
            expires_in_days=int(_ak_expiry),
        )
        if _ak_err:
            st.warning(_ak_err)
        else:
            st.session_state["api_key_store"] = _ak_store
            save_api_key_store(_ak_store)
            st.session_state["_api_key_plaintext"] = _ak_plain
            st.session_state["_api_key_clear_form"] = True
            log_event(logger, logging.INFO, "user.api_key_created", scopes=len(_ak_new.scopes))
            st.rerun()

    st.markdown("---")
    st.caption(
        f"The API is a separate process and is never started automatically. Run it with "
        f"`python3 api_server.py` — it listens on {API_KEYS.default_host}:{API_KEYS.default_port} "
        f"and `GET /v1` lists every endpoint."
    )

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

# --- Asset class selector -----------------------------------------------------
# A LENS, not a mode. It steers what the search suggests and which
# examples are offered; it never overrides what a symbol actually IS.
# Typing SPY with "Stocks" lit still analyses SPY as a fund, because the
# classification comes from the data (asset_class.classify) and not from
# this widget — a page that applied stock analysis to a fund because a
# pill was left selected would be confidently wrong.
#
# Rendered in the sidebar directly above the ticker box it steers, and
# BEFORE it, because the box's own placeholder depends on the selection.
if "asset_view" not in st.session_state:
    st.session_state["asset_view"] = asset_views.DEFAULT_VIEW
_pending_view = st.session_state.pop("_pending_asset_view", None)
if _pending_view in asset_views.VIEWS_BY_KEY:
    st.session_state["asset_view"] = _pending_view

_view_labels = list(asset_views.pill_labels())
_view_current = asset_views.view(st.session_state.get("asset_view"))
# No key= on this widget. The options are fixed here, but the same
# session_state entry is written by the Alt+N shortcut and by the
# deferred switch above, and a keyed widget restores its own stored value
# over an externally applied one on the next run — the trap the screener's
# criteria widgets already document.
_view_pick = st.sidebar.pills(
    "Asset class", _view_labels,
    default=_view_current.pill, label_visibility="collapsed",
    help="Steers the search suggestions and examples below. It does not "
         "change how a symbol is analysed — that follows what the symbol "
         "actually is.")
if _view_pick and asset_views.key_for_pill(_view_pick) != st.session_state["asset_view"]:
    st.session_state["asset_view"] = asset_views.key_for_pill(_view_pick)
    _view_current = asset_views.view(st.session_state["asset_view"])

ticker_symbol = st.sidebar.text_input(
    "Stock Ticker", key="ticker_input",
    placeholder=_view_current.placeholder).upper()
st.sidebar.caption(
    f"{_view_current.pill}: " + " · ".join(_view_current.examples))

# --- Ticker autocomplete -----------------------------------------------------
# Deliberately ADDITIVE: the free-text box above is untouched, because three
# separate flows drive it through st.session_state["ticker_input"] (the
# deferred _pending_ticker switch, watchlist row clicks, and the quick-access
# chips) and it is also the only way to reach an arbitrary symbol. Both
# controls here simply write to that same key and rerun, so every existing
# path keeps working unchanged.
#
# Neither control is per-keystroke — Streamlit doesn't round-trip on every
# keystroke and injected <script> doesn't execute (see ticker_search.py).
# The dropdown filters its options client-side as you type; the name search
# is one round-trip on submit.
with st.sidebar.expander("Find a ticker", expanded=False):
    _ts_known = list(WATCHLIST.tech_basket) + list(WATCHLIST.diversified_basket)
    if "watchlist_store" in st.session_state:
        for _ts_wl in st.session_state["watchlist_store"].lists.values():
            _ts_known += list(_ts_wl.tickers)
    if "quick_access_store" in st.session_state:
        _ts_qa = st.session_state["quick_access_store"]
        _ts_known += list(_ts_qa.favorites) + list(_ts_qa.recents)
    _ts_universe = ts_build_universe(_ts_known)

    _ts_labels = [m.label for m in _ts_universe]
    _ts_pick = st.selectbox(
        "Known tickers",
        options=_ts_labels,
        index=None,
        placeholder="Type a ticker or company name…",
        accept_new_options=True,
        key="ts_pick",
        help=(
            "Filters as you type across every ticker this app already knows — the institutional "
            "baskets, your watchlists, favorites and recently viewed. A symbol that isn't listed "
            "can still be typed in directly; the list is a shortcut, not a restriction."
        ),
    )
    if _ts_pick:
        _ts_symbol = ts_symbol_from_label(_ts_pick)
        # Two things are load-bearing here.
        #
        # 1. st.session_state["ts_pick"] is deliberately NOT cleared: a
        #    widget's own key cannot be assigned after that widget has been
        #    instantiated this run — Streamlit raises, which silently killed
        #    the whole handler before the rerun ever fired. Same family as
        #    the value=/key= footgun documented elsewhere in this app.
        #
        # 2. The trigger is "the dropdown CHANGED", not "the dropdown
        #    disagrees with the current ticker". Those look equivalent and
        #    aren't: with the latter, a stale selection left in the box
        #    re-fires on every subsequent run and drags the ticker back,
        #    so switching by any OTHER control (the name search below, a
        #    watchlist row, a quick-access chip) would silently revert.
        #    Caught live — a search pick of COIN was immediately undone by
        #    a leftover "V — Visa Inc." in this box.
        if _ts_pick != st.session_state.get("_ts_last_applied"):
            st.session_state["_ts_last_applied"] = _ts_pick
            if _ts_symbol and _ts_symbol != ticker_symbol:
                st.session_state["_pending_ticker"] = _ts_symbol
                log_event(logger, logging.INFO, "user.autocomplete_pick", ticker=_ts_symbol)
                st.rerun()

    st.markdown("---")
    _ts_query = st.text_input(
        "Search by company name",
        key="ts_query",
        placeholder="e.g. apple, nvid, vanguard",
        help="Looks the name up against Yahoo on Enter — one request, not one per keystroke.",
    )
    if _ts_query and len(_ts_query.strip()) >= 2:
        with st.spinner(f'Searching for "{_ts_query.strip()}"…'):
            _ts_results, _ts_err = ts_search_symbols(_ts_query)
        if _ts_err:
            st.caption(_ts_err)
        else:
            st.caption(
                f"{len(_ts_results)} match(es). Yahoo returns foreign cross-listings and tokenized "
                "proxies alongside the primary listing, so check the exchange before picking."
            )
            for _ts_m in _ts_results:
                if st.button(_ts_m.label, key=f"ts_hit_{_ts_m.symbol}", width="stretch",
                             help=_ts_m.detail or None):
                    # Same reason as above — ts_query is this run's widget key
                    # and assigning to it here would raise and abort the switch.
                    st.session_state["_pending_ticker"] = _ts_m.symbol
                    log_event(logger, logging.INFO, "user.autocomplete_search_pick", ticker=_ts_m.symbol)
                    st.rerun()

    # --- discovery -----------------------------------------------------
    # Everything below answers "I don't know what I'm looking for yet",
    # where the two controls above answer "I know, help me type it".
    #
    # These render as full-width stacked buttons rather than a row of
    # chips: the sidebar is ~300px, and the task's mock ([Tesla] [Nvidia]
    # [Apple]) assumes a page-width strip that does not exist here.
    #
    # None of them switches a button to type="primary" for the ticker
    # already on screen. That is how the watchlist rows, quick-access
    # chips and peer switcher mark "you are here", but their styling is
    # scoped by widget-key prefix in the CSS above — a fourth surface
    # using primary without being added to that scope renders in
    # Streamlit's stock red, which is exactly the bug the login page
    # shipped. A "· current" suffix says the same thing and cannot rot.
    def _ts_pick_button(_listing, _key_prefix: str, _note: str = "") -> None:
        _is_current = _listing.symbol == ticker_symbol
        _label = f"{_listing.symbol}  {_listing.change_text()}"
        if _is_current:
            _label += "  · current"
        _bits = [b for b in (_listing.name, _listing.exchange, _note) if b]
        if st.button(_label, key=f"{_key_prefix}{_listing.symbol}", width="stretch",
                     help=" · ".join(_bits) or None):
            st.session_state["_pending_ticker"] = _listing.symbol
            log_event(logger, logging.INFO, "user.discovery_pick",
                      ticker=_listing.symbol, source=_key_prefix)
            st.rerun()

    def _ts_render(_listings, _error, _key_prefix: str, _empty: str) -> None:
        if _error:
            st.caption(_error)
            return
        if not _listings:
            st.caption(_empty)
            return
        for _row in _listings:
            _ts_pick_button(_row, _key_prefix)

    # Recently viewed. The symbol on screen is skipped rather than shown
    # as the first of five: it is already in the header in 30px type, and
    # spending a slot to repeat it is the same waste the quick-stats
    # defaults were trimmed for.
    st.markdown("---")
    st.caption("**Recently viewed**")
    _ts_recents = ()
    if "quick_access_store" in st.session_state:
        _ts_recents = tuple(
            t for t in st.session_state["quick_access_store"].recents
            if t != ticker_symbol
        )[:td.RECENTS_LIMIT]
    if _ts_recents:
        for _ts_r in _ts_recents:
            if st.button(_ts_r, key=f"ts_recent_{_ts_r}", width="stretch",
                         help=f"Switch analysis to {_ts_r}"):
                st.session_state["_pending_ticker"] = _ts_r
                log_event(logger, logging.INFO, "user.discovery_pick",
                          ticker=_ts_r, source="ts_recent_")
                st.rerun()
    else:
        st.caption("Tickers you open appear here.")

    # Live movers. Deliberately NOT called "trending": these are ranked by
    # a stated, measurable quantity, and the label says which.
    st.markdown("---")
    _ts_screen_labels = [lbl for _, lbl, _ in td.TRENDING_SCREENS]
    _ts_screen_choice = st.radio(
        "Market movers", _ts_screen_labels, horizontal=True,
        key="ts_trend_screen",
        help="Yahoo's US screens, refreshed every ten minutes. A ranking, "
             "not a recommendation — a stock is not worth owning because "
             "it is heavily traded today.",
    )
    _ts_kind = next(k for k, lbl, _ in td.TRENDING_SCREENS if lbl == _ts_screen_choice)
    st.caption(f"Ranked {td.TRENDING_BASIS[_ts_kind]}.")
    with st.spinner("Loading movers…"):
        _ts_trend, _ts_trend_err = td.trending(_ts_kind)
    _ts_render(_ts_trend, _ts_trend_err, "ts_trend_", "No movers reported right now.")

    # Browse by sector, and the same machinery behind a description box.
    st.markdown("---")
    _ts_sector = st.selectbox(
        "Browse a sector", options=list(screener_module.SECTORS), index=None,
        placeholder="Pick a sector…", key="ts_sector",
        help="The largest US companies in that sector by market cap, from "
             "Yahoo's screener. Secondary OTC listings of foreign names are "
             "filtered out so the list is eight companies, not four and "
             "their shadows.",
    )
    if _ts_sector:
        with st.spinner(f"Loading {_ts_sector}…"):
            _ts_sec_rows, _ts_sec_err = td.by_sector(_ts_sector)
        _ts_render(_ts_sec_rows, _ts_sec_err, "ts_sect_",
                   f"No {_ts_sector} companies came back.")

    _ts_desc = st.text_input(
        "Or describe it",
        key="ts_describe",
        placeholder="e.g. biotech, semiconductors, banks",
        help="Matches KEYWORDS against a fixed table of sectors and "
             "industries — it does not interpret language. An unrecognised "
             "phrase says so rather than guessing.",
    )
    if _ts_desc and _ts_desc.strip():
        with st.spinner("Looking that up…"):
            _ts_d_rows, _ts_d_err, _ts_intent = td.for_description(_ts_desc)
        if _ts_intent is None:
            st.caption(
                f'No keyword in "{_ts_desc.strip()}" matches a sector or industry '
                "this understands. Try a plainer word (biotech, banks, energy) "
                "or use the sector list above."
            )
        else:
            st.caption(f'Read "{_ts_intent.matched}" as {_ts_intent.label}.')
            _ts_render(_ts_d_rows, _ts_d_err, "ts_desc_",
                       f"No {_ts_intent.label} companies came back.")

today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=CHART_DEFAULTS.default_lookback_days)

# --- Analysis date range ---------------------------------------------------
# ONE control, not two. st.date_input takes a (start, end) tuple and renders
# a single calendar where you click the start and then the end — the native
# form of "drag to select a range", and half the controls on a phone.
#
# Both controls were ALREADY calendar pickers, contrary to the task's
# premise; what was missing was the presets, a statement of how long the
# window is, and the range shape.
#
# The range lives in a non-widget session key and the calendar gets a
# computed `value=` with NO `key=`. That is the pattern this codebase
# arrived at the hard way: passing both value= and key= silently reverts
# the user's edit on the next rerun, and a keyed widget restores its old
# value over a just-applied preset.
if "_dr_range" not in st.session_state:
    st.session_state["_dr_range"] = (one_year_ago, today)

_dr_start, _dr_end = st.session_state["_dr_range"]

# Pills carry no key either, for the same reason: `default=` is computed
# from the range that is actually in force, so the row reflects reality
# rather than whatever was last clicked. A range typed by hand matches no
# preset and correctly shows none selected.
_dr_pick = st.sidebar.pills(
    "Quick range", date_range.PRESET_KEYS,
    default=date_range.matching_preset(_dr_start, _dr_end, today),
    selection_mode="single", label_visibility="collapsed",
    help="Presets end today. Max asks for "
         f"{date_range.MAX_LOOKBACK_YEARS} years and returns whatever the "
         "data source actually has for this symbol.",
)
if _dr_pick:
    _dr_resolved = date_range.resolve(_dr_pick, today)
    # Compared against the range in force rather than against the last
    # click: a stale selection would otherwise re-fire every run and drag
    # the dates back over any manual edit — the bug the ticker
    # autocomplete hit and documents above.
    if _dr_resolved and _dr_resolved != st.session_state["_dr_range"]:
        st.session_state["_dr_range"] = _dr_resolved
        log_event(logger, logging.INFO, "user.date_preset", preset=_dr_pick)
        st.rerun()

_dr_picked = st.sidebar.date_input(
    "Analysis range", value=st.session_state["_dr_range"],
    min_value=date_range.EARLIEST_SELECTABLE, max_value=today,
    help="Click a start date, then an end date. Drives every fetch on the "
         "page — prices, statements, benchmarks and backtests.",
)
# Between those two clicks the widget returns a ONE-element tuple.
# Unpacking that raises, and this control feeds every fetch on the page,
# so the half-made selection holds the previous range instead.
start_date, end_date = date_range.coerce(_dr_picked, st.session_state["_dr_range"])
st.session_state["_dr_range"] = (start_date, end_date)

st.sidebar.caption(date_range.describe(start_date, end_date, today))
for _dr_problem in date_range.problems(start_date, end_date, today):
    st.sidebar.warning(_dr_problem)

# ==========================================
# WATCHLIST (quick symbol switching) — multiple saved, named lists
# ==========================================
# Deliberately OUTSIDE the control tabs below: this is navigation, not
# configuration, and it stays visible no matter which analysis panel or
# control tab is open — the "persistent" part of the panel.
#
# Persisted to a local file (atomic write, same pattern as every other
# cross-restart store in this app) — genuinely survives an app restart,
# not just reruns within a session. Scoped per signed-in user when auth
# is configured, and shared instance-wide when it isn't — see auth.py.
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")

if "watchlist_store" not in st.session_state:
    st.session_state["watchlist_store"] = load_watchlist_store()
_wl_store = st.session_state["watchlist_store"]

# No `key=` on this selectbox deliberately: the list of NAMES it offers
# changes whenever a watchlist is created/renamed/deleted, and a keyed
# widget whose stored value falls out of its own `options` list on a
# later rerun raises — the same class of widget-state pitfall already hit
# (and fixed) elsewhere in this app for text/number inputs. `_wl_store.active`
# (persisted, not any Streamlit-internal widget state) is the single
# source of truth instead; the selectbox's return value is only compared
# against it to detect a user-initiated switch.
_wl_names = list(_wl_store.lists.keys())
_wl_selected_name = st.sidebar.selectbox("Active Watchlist", _wl_names, index=_wl_names.index(_wl_store.active))
if _wl_selected_name != _wl_store.active:
    _wl_store, _wl_switch_err = set_active_watchlist(_wl_store, _wl_selected_name)
    st.session_state["watchlist_store"] = _wl_store
    save_watchlist_store(_wl_store)
    log_event(logger, logging.INFO, "user.watchlist_active_switch", watchlist=_wl_selected_name)
    st.rerun()

with st.sidebar.expander("Manage Watchlists", expanded=False):
    st.caption(f"{len(_wl_store.lists)} of {WATCHLIST_PANEL.max_watchlists} watchlists")
    _wl_new_name = st.text_input("New watchlist name", key="watchlist_new_name_input", placeholder="e.g. Dividend Payers")
    if st.button("+ Create", key="watchlist_create_btn"):
        _wl_store, _wl_create_err = create_watchlist(_wl_store, _wl_new_name)
        if _wl_create_err:
            st.warning(_wl_create_err)
        else:
            st.session_state["watchlist_store"] = _wl_store
            save_watchlist_store(_wl_store)
            log_event(logger, logging.INFO, "user.watchlist_created", watchlist=_wl_new_name)
            st.rerun()

    st.markdown("---")
    _wl_rename_input = st.text_input("Rename active watchlist to", key="watchlist_rename_input", placeholder=_wl_store.active)
    if st.button("Rename", key="watchlist_rename_btn"):
        _wl_store, _wl_rename_err = rename_watchlist(_wl_store, _wl_store.active, _wl_rename_input)
        if _wl_rename_err:
            st.warning(_wl_rename_err)
        else:
            st.session_state["watchlist_store"] = _wl_store
            save_watchlist_store(_wl_store)
            log_event(logger, logging.INFO, "user.watchlist_renamed", to=_wl_rename_input)
            st.rerun()

    st.markdown("---")
    if st.button(f'Delete "{_wl_store.active}"', key="watchlist_delete_btn", disabled=(len(_wl_store.lists) <= 1)):
        _wl_store, _wl_delete_err = delete_watchlist(_wl_store, _wl_store.active)
        if _wl_delete_err:
            st.warning(_wl_delete_err)
        else:
            st.session_state["watchlist_store"] = _wl_store
            save_watchlist_store(_wl_store)
            log_event(logger, logging.INFO, "user.watchlist_deleted")
            st.rerun()
    if len(_wl_store.lists) <= 1:
        st.caption("Can't delete your last watchlist.")

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
        _wl_store.lists[_wl_store.active].tickers, _wl_new, WATCHLIST_PANEL.max_tickers,
    )
    if _wl_error:
        st.sidebar.warning(_wl_error)
    else:
        _wl_store = update_active_tickers(_wl_store, _wl_updated)
        st.session_state["watchlist_store"] = _wl_store
        save_watchlist_store(_wl_store)
        log_event(logger, logging.INFO, "user.watchlist_add", tickers=_wl_new, watchlist=_wl_store.active)
        st.rerun()

_wl_tickers = _wl_store.lists[_wl_store.active].tickers
if not _wl_tickers:
    # The action adds the symbol already on screen. Streamlit cannot focus
    # the "Add ticker" box above, so a button that merely points at it
    # would do nothing visible; this finishes the job in one click and is
    # undone by the ✕ that appears next to the row.
    if empty_states.render(
        "No tickers yet",
        "A watchlist is how you switch between symbols without retyping them.",
        action_label=f"Add {ticker_symbol} to get started",
        key="empty_watchlist_add",
        help_text=f"Adds {ticker_symbol}, the symbol currently being analysed.",
        container=st.sidebar,
    ):
        _wl_seed, _wl_seed_err = add_ticker(
            _wl_tickers, ticker_symbol, WATCHLIST_PANEL.max_tickers)
        if _wl_seed_err:
            st.sidebar.warning(_wl_seed_err)
        else:
            _wl_store = update_active_tickers(_wl_store, _wl_seed)
            st.session_state["watchlist_store"] = _wl_store
            save_watchlist_store(_wl_store)
            empty_states.log_action("watchlist_seeded", ticker=ticker_symbol)
            st.rerun()
else:
    _wl_snapshots = load_quote_snapshots(_wl_tickers)
    for _wl_snap in _wl_snapshots:
        _wl_row, _wl_remove = st.sidebar.columns([5, 1])
        _wl_is_active = _wl_snap.ticker == ticker_symbol
        # A watchlist has always been able to mix asset types — it stores
        # plain ticker strings — but every row looked identical, so a
        # bond fund among four stocks was indistinguishable. The badge is
        # the CLASSIFIED type, not whichever pill was lit when the ticker
        # was added, and it costs no fetch: the quote already loaded the
        # info dict that carries quoteType.
        _wl_badge = asset_views.badge(_wl_snap.asset_class_key)
        with _wl_row:
            if _wl_snap.status == "ok":
                # The direction is coloured on the LABEL rather than by the
                # chip, because the chip's colour now means "selected".
                # Button labels take markdown, so :green[]/:red[] survives
                # inside the neutral active pill and up/down still reads.
                _wl_move = f"{_wl_snap.direction_icon} {_wl_snap.change_pct:+.2f}%".strip()
                if _wl_snap.change_pct > 0:
                    _wl_move = f":green[{_wl_move}]"
                elif _wl_snap.change_pct < 0:
                    _wl_move = f":red[{_wl_move}]"
                _wl_label = f"{_wl_badge} {_wl_snap.ticker} · {_wl_move}"
            else:
                _wl_label = f"{_wl_badge} {_wl_snap.ticker} · n/a"
            # The active ticker gets the accent style so it's obvious which
            # row you're looking at. Deliberately NOT disabled: a disabled
            # primary button renders as a washed-out pill that reads as
            # broken, and clicking your own current ticker is a harmless
            # no-op rerun anyway.
            if st.button(
                _wl_label, key=f"wl_go_{_wl_snap.ticker}", width="stretch",
                type="primary" if _wl_is_active else "secondary",
                help=(f"{asset_views.badge_title(_wl_snap.asset_class_key)}"
                      if _wl_snap.status == "ok"
                      else f"{asset_views.badge_title(_wl_snap.asset_class_key)} — "
                           f"{_wl_snap.detail}"),
            ):
                st.session_state["_pending_ticker"] = _wl_snap.ticker
                log_event(logger, logging.INFO, "user.watchlist_switch", ticker=_wl_snap.ticker)
                st.rerun()
        with _wl_remove:
            if st.button("✕", key=f"wl_rm_{_wl_snap.ticker}", help=f"Remove {_wl_snap.ticker}"):
                _wl_store = update_active_tickers(_wl_store, remove_ticker(_wl_tickers, _wl_snap.ticker))
                st.session_state["watchlist_store"] = _wl_store
                save_watchlist_store(_wl_store)
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
    st.caption("Appearance")
    # The theme control MOVED to the account menu's Preferences section
    # (top right). It is a personal preference and that is where the brief
    # puts it — and it can only live in one place: two widgets sharing
    # key="theme_choice" raise DuplicateWidgetID, and a second widget on a
    # different key would need to write theme_choice after the first had
    # already instantiated it, which Streamlit refuses outright.
    st.caption(
        f"Theme is **{PALETTES[st.session_state['theme_choice']].label}** — change it in "
        "the account menu at the top right."
    )

    st.markdown("---")
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

    st.markdown("---")
    st.caption("Getting Started")
    if st.button("Replay Tutorial", help="Reopen the first-run walkthrough, starting from step 1."):
        st.session_state["onboarding_active"] = True
        st.session_state["onboarding_step"] = 0
        log_event(logger, logging.INFO, "user.onboarding_replayed")
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

    # What KIND of instrument is this? Nothing in the analysis path asked
    # before, so BTC-USD loaded as "valid" with pe=None and then got a
    # discounted cash flow, an eight-point company scorecard and a
    # sector-percentile ranking run against it. Classified once here and
    # honoured by the panels that only make sense for a company.
    asset_kind = asset_class.classify(ticker_bundle.info, ticker_symbol)
    # Parked for the blocks that run ABOVE this line. finance.py is a
    # script, so name order is execution order: the keyboard palette and
    # the shortcuts panel render ~2400 lines earlier and cannot see
    # asset_kind on this run. They read this with an equity default,
    # which is correct from the second run onward — and the panels are
    # only ever opened by a keypress, which is always a later run.
    st.session_state["asset_kind"] = asset_kind

# ==========================================
# SYMBOL HEADER (fill) — renders into the sticky slot reserved at the very
# top of the page, so it's on screen from first paint and stays pinned
# while scrolling through any panel. Fills even when df is empty, since
# knowing WHICH symbol failed to load is exactly when the header matters.
# ==========================================
# .container() REPLACES the skeleton written above rather than appending
# beneath it.
with symbol_header_container.container():
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

    # --- Quick stats strip ------------------------------------------------
    # One compact row of key metrics under the symbol, inside the same
    # sticky block. Each stat is an st.popover rather than styled text:
    # custom HTML in st.markdown cannot call back into Streamlit, so a
    # text-rendered metric can never satisfy the brief's "click to expand".
    #
    # Wrapped in st.fragment so the timer refreshes THIS strip instead of
    # rerunning the whole page — the same mechanism the real-time alert
    # engine uses above. The interval matches the quote cache's TTL:
    # polling faster would re-render an identical cached number and buy
    # nothing but Yahoo requests.
    # --- Account menu -------------------------------------------------
    # Top-right of the sticky header, which is as close to "top right
    # corner" as a Streamlit app gets: the framework owns the page chrome
    # and gives an app no bar of its own, and this block is the one thing
    # always on screen. A wide spacer column pushes it right.
    # --- Data quality badge -------------------------------------------
    # Computed HERE rather than in the Overview tab where the full report
    # lives, because this block is the one thing always on screen and the
    # question it answers — "can I trust the numbers I am looking at?" —
    # should not require finding a panel first. It is the same call the
    # detail section makes; the result is reused rather than recomputed.
    #
    # A POPOVER, not markup in the header line above. That header is a
    # single st.markdown string and custom HTML cannot call back into
    # Streamlit, so a badge rendered there could never satisfy "click to
    # expand" — the identical constraint the quick-stats strip below
    # already documents.
    #
    # There is no refresh timer: the report is recomputed from the same
    # bundle this page is built from on every run, so it is never staler
    # than the figures beside it. A fragment here would re-render a value
    # closed over from this run and only look live.
    # asset_kind is passed rather than left to default, so the badge and
    # the rest of the page cannot disagree about what this symbol is.
    data_quality_report = assess_data_quality(
        standardized, ticker_bundle, macro_bundle, klass=asset_kind)
    _dq_colour = data_quality.grade_colour(data_quality_report.grade)

    _dq_slot, _pm_spacer, _nb_slot, _pm_slot = st.columns([2, 6, 1.4, 1])
    with _dq_slot:
        st.markdown(
            # Qualified to (0,3,1). The obvious selector,
            # [class*="st-key-dq_badge"] button, is only (0,1,1) and loses
            # outright to finance.py's own (0,2,1) main-secondary rule no
            # matter how many !importants it carries — measured live, the
            # badge came back with the ordinary grey border. Same trap
            # button_roles documents.
            #
            # And it matches on kind= rather than the stBaseButton-secondary
            # testid that button_roles uses: a POPOVER trigger carries
            # data-testid="stPopoverButton", so the testid form would not
            # select it at all.
            f"""<style>
            [class*="st-key-dq_badge"] button,
            [data-testid="stMain"] [class*="st-key-dq_badge"] button[kind="secondary"] {{
                border: 1px solid {_dq_colour} !important;
                color: {_dq_colour} !important;
            }}
            [class*="st-key-dq_badge"] button p,
            [data-testid="stMain"] [class*="st-key-dq_badge"] button[kind="secondary"] p {{
                color: {_dq_colour} !important; font-weight: 700 !important;
            }}
            </style>""",
            unsafe_allow_html=True,
        )
        with st.popover(
            f"Data {data_quality_report.score:.0f}/100 · {data_quality_report.grade}",
            width="stretch", key="dq_badge",
            help="How complete and current the data behind this page is",
        ):
            st.markdown(
                f"**Data quality — {data_quality_report.score:.1f}/100 "
                f"({data_quality_report.grade})**")
            st.caption(data_quality.grade_meaning(
                data_quality_report.grade, data_quality_report.asset_class))
            # The rows are driven by the report's OWN dimensions, not a
            # hardcoded four. A fund is not graded on filings it never
            # makes, so listing "Required fields: 0%" underneath its score
            # would explain the number with a fact that carried no weight
            # in it — which is the misreport this panel used to make.
            for _dq_dim in data_quality_report.dimensions:
                _dq_value = getattr(data_quality_report, _dq_dim.key)
                st.caption(
                    f"{_dq_dim.label}: **{_dq_value:.0f}/100** "
                    f"({_dq_dim.weight:.0%} of the score)")
            if data_quality_report.asset_class != asset_class.EQUITY:
                st.caption(
                    f"Scored as {asset_class.with_article(data_quality_report.asset_class)}: "
                    f"{data_quality_report.scored_on}. Company filings are not "
                    "part of this score because this instrument does not make them.")
            if data_quality_report.staleness_days is not None:
                _dq_stale = (" — past the point where a quarterly filing should have landed"
                             if data_quality_report.is_stale else "")
                st.caption(
                    f"Most recent quarter: {data_quality_report.most_recent_quarter} "
                    f"({data_quality_report.staleness_days} days ago){_dq_stale}")
            elif data_quality_report.asset_class == asset_class.EQUITY:
                st.caption("No filing date reported, so freshness could not be measured.")
            _dq_gaps = data_quality_report.issue_count
            st.caption(
                f"{_dq_gaps} field-level issue(s). The full list is in "
                "Overview → Data Quality Report."
                if _dq_gaps else
                "No field-level issues. The full report is in Overview.")

    # --- Notification bell --------------------------------------------
    # Fed by the real-time engine's persisted TriggerEvents only. The
    # Smart Risk-Aware Alerts are an on-demand snapshot with no event log
    # and no timestamps, so counting them would mean inventing an
    # occurrence time — settled with the user rather than assumed.
    with _nb_slot:
        _nb_history = st.session_state.get("rt_alert_history", [])
        _nb_seen = notifications.last_seen()
        _nb_unread = notifications.unread(_nb_history, _nb_seen)
        _nb_badge = notifications.badge_text(len(_nb_unread))
        with st.popover(f"Alerts {_nb_badge}".strip(), width="stretch",
                        key="notif_bell",
                        help="Alerts your rules have fired, newest first"):
            if notifications.store_is_corrupt():
                st.caption(
                    "The notifications file on this instance can't be read, so "
                    "read state and snoozes are unavailable. Move or delete "
                    f"{notifications.STORE_FILENAME} and reload.")
            if not _nb_history:
                st.caption("No alerts yet. Rules that fire appear here.")
            else:
                _nb_unread_ids = {id(e) for e in _nb_unread}
                _nb_rules = {r.id: r for r in st.session_state.get("rt_alert_rules", [])}
                for _nb_event in list(reversed(_nb_history))[:notifications.DROPDOWN_LIMIT]:
                    _nb_new = "**NEW** · " if id(_nb_event) in _nb_unread_ids else ""
                    st.markdown(
                        f"{_nb_new}**{_nb_event.ticker}** — "
                        f"{RT_TRIGGER_LABELS.get(_nb_event.trigger_type, _nb_event.trigger_type)}")
                    st.caption(
                        f"{notifications.describe_age(_nb_event.triggered_at)} · "
                        + _rt_md_escape_dollar(_nb_event.detail or ""))
                    # Snooze acts on the RULE behind the event. The event
                    # has already happened; muting the rule is what stops
                    # it firing again.
                    if _nb_event.rule_id in _nb_rules:
                        _nb_until = notifications.mutes().get(_nb_event.rule_id)
                        if _nb_until:
                            st.caption(notifications.describe_mute(_nb_until))
                            if st.button("Unmute", key=f"notif_unmute_{_nb_event.rule_id}",
                                         width="stretch"):
                                notifications.unsnooze(_nb_event.rule_id)
                                st.rerun()
                        else:
                            _nb_choice = st.selectbox(
                                f"Snooze {_nb_event.ticker}",
                                [label for label, _ in notifications.SNOOZE_CHOICES],
                                index=None, placeholder="Snooze this rule…",
                                label_visibility="collapsed",
                                key=f"notif_snooze_{_nb_event.rule_id}")
                            if _nb_choice:
                                _nb_hours = dict(notifications.SNOOZE_CHOICES)[_nb_choice]
                                notifications.snooze(_nb_event.rule_id, _nb_hours)
                                st.rerun()
                    st.divider()

                if len(_nb_history) > notifications.DROPDOWN_LIMIT:
                    if st.button(f"See all {len(_nb_history)}", key="notif_see_all",
                                 width="stretch"):
                        st.session_state["_notif_open_history"] = True
                        st.rerun()
                if _nb_unread and st.button("Mark all read", key="notif_mark_read",
                                            width="stretch"):
                    notifications.mark_all_read()
                    st.rerun()
                if st.button("Clear history", key="notif_clear", width="stretch"):
                    st.session_state["rt_alert_history"] = []
                    rt_save_store(st.session_state["rt_alert_rules"], [])
                    notifications.clear_history()
                    st.rerun()

    with _pm_slot:
        profile_menu.render()

    # The user's saved choice is FILTERED to the stats that mean something
    # for this asset class, then topped up from that class's own list.
    # Before this, an ETF's header read "Market Cap · Not reported" beside
    # "Net Margin · Not reported" and "ROE · Not reported" — three
    # questions a fund cannot be asked, presented as data it failed to
    # supply. Filtering rather than replacing keeps the saved selection
    # intact: switching back to a stock restores it untouched.
    _qs_applicable = asset_views.header_stats(asset_kind)
    _qs_selected = tuple(k for k in quick_stats.selected() if k in _qs_applicable)
    if len(_qs_selected) < len(_qs_applicable):
        _qs_selected += tuple(k for k in _qs_applicable if k not in _qs_selected)
    _qs_selected = _qs_selected[:quick_stats.MAX_SELECTED]

    @st.fragment(run_every=quick_stats.REFRESH_SECONDS)
    def _render_quick_stats():
        _qs_quote = load_quote_snapshots((ticker_symbol,))[0]
        # Loaded only for a fund, and from the same cached profile the
        # Fund Decomposition panel uses, so the strip still adds no fetch.
        _qs_fund = (etf_analysis.load_profile(ticker_symbol)
                    if asset_class.supports(asset_kind, asset_class.HOLDINGS)
                    else None)
        _qs_specs = [quick_stats.STATS_BY_KEY[k] for k in _qs_selected
                     if k in quick_stats.STATS_BY_KEY]
        _qs_cols = st.columns(len(_qs_specs) + 1)

        for _qs_i, _qs_spec in enumerate(_qs_specs):
            _qs_text = quick_stats.display(_qs_spec, _qs_quote, standardized, _qs_fund)
            with _qs_cols[_qs_i]:
                with st.popover(f"{_qs_spec.label} · {_qs_text}", width="stretch"):
                    st.markdown(f"**{_qs_spec.label}** — {_qs_text}")
                    if _qs_spec.help_key:
                        st.caption(help_for(_qs_spec.help_key))
                    if _qs_spec.note:
                        st.caption(_qs_spec.note)
                    if _qs_text == quick_stats.NOT_REPORTED:
                        st.caption(
                            "This ticker doesn't report the figure. Shown as "
                            "unavailable rather than as a zero.")

        with _qs_cols[-1]:
            with st.popover("Customize", width="stretch"):
                st.caption(
                    f"Pick up to {quick_stats.MAX_SELECTED}. The order you choose is "
                    "the order they appear. Clearing them all hides the strip.")
                st.caption(
                    f"Showing the stats that apply to "
                    f"{asset_class.with_article(asset_kind)}. Your choice is "
                    "remembered across asset classes.")
                _qs_choice = st.multiselect(
                    "Stats shown", [s.key for s in quick_stats.STATS
                                    if s.key in _qs_applicable],
                    default=list(_qs_selected),
                    format_func=lambda k: quick_stats.STATS_BY_KEY[k].label,
                    label_visibility="collapsed",
                )
                _qs_save, _qs_reset = st.columns(2)
                with _qs_save:
                    if st.button("Save", key="quick_stats_save", width="stretch"):
                        # Folded back into the saved list rather than
                        # replacing it: the picker only offered this
                        # class's stats, so a verbatim save would erase
                        # every stat belonging to the others.
                        _qs_ok, _qs_err = quick_stats.set_selected(
                            quick_stats.merge_selection(
                                quick_stats.selected(), _qs_choice,
                                _qs_applicable))
                        if _qs_ok:
                            # scope="app": the strip's column count changes
                            # with the selection, and a fragment-only rerun
                            # would redraw the old layout.
                            st.rerun(scope="app")
                        else:
                            st.warning(_qs_err)
                with _qs_reset:
                    if st.button("Reset", key="quick_stats_reset", width="stretch"):
                        quick_stats.reset()
                        st.rerun(scope="app")

                st.markdown(
                    loading_states.pulse(
                        f"Refreshes about every {quick_stats.REFRESH_SECONDS // 60} "
                        "minutes, which is how often the underlying quote is "
                        "re-fetched. Not a tick-by-tick feed."),
                    unsafe_allow_html=True,
                )

    _render_quick_stats()

    # --- Quick access: favorites (starred, curated) + recently viewed
    # (automatic), merged into ONE row of one-click switch chips.
    # Rendered INSIDE symbol_header_container so it's part of the same
    # sticky block as the header above — the symbol you're on and the
    # symbols you can jump to stay together and stay on screen. One row
    # rather than two stacked ones precisely because it's sticky: every
    # extra row here costs vertical space on every screen forever.
    #
    # Both halves persist across restarts via favorites.py; see that
    # module's docstring for why favorites are deliberately separate from
    # the sidebar's named watchlists, and why recents are now durable
    # (they used to be session-only) with a Clear control as the escape
    # hatch.
    if "quick_access_store" not in st.session_state:
        st.session_state["quick_access_store"] = load_quick_access()
    _qa_store = st.session_state["quick_access_store"]

    # Recorded here rather than at the sidebar input so it captures the
    # ticker actually analysed on this run, however it was chosen (typed,
    # watchlist click, screener click-through, or one of these chips).
    # record_recent() is idempotent, which matters because Streamlit
    # re-runs this whole script on every widget interaction — and the
    # store is only WRITTEN when the result actually changed, so an
    # ordinary rerun (slider nudge, tab click) costs no disk write.
    _qa_new_recents = record_recent(
        _qa_store.recents, ticker_symbol, WATCHLIST_PANEL.max_recent_tickers,
    )
    if _qa_new_recents != _qa_store.recents:
        _qa_store = dataclasses.replace(_qa_store, recents=_qa_new_recents)
        st.session_state["quick_access_store"] = _qa_store
        save_quick_access(_qa_store)

    _qa_chips = quick_access_chips(_qa_store)
    _qa_pinned = is_favorite(_qa_store, ticker_symbol)

    # Fixed-width chips: the columns list is padded to the configured
    # maximum with a trailing spacer, so a chip is the same size whether
    # two symbols are listed or eight — rather than two chips stretching
    # across the whole page. The leading column is the star toggle for
    # the ticker currently on screen, given a fraction of a chip's width
    # since it holds a single glyph.
    _qa_cols = st.columns(
        [0.5] + [1] * len(_qa_chips) + [max(1, FAVORITES.max_chips - len(_qa_chips) + 1)]
    )
    with _qa_cols[0]:
        if st.button(
            "★" if _qa_pinned else "☆", key="favorite_toggle", width="stretch",
            help=f"Remove {ticker_symbol} from favorites" if _qa_pinned else f"Add {ticker_symbol} to favorites",
        ):
            _qa_store, _qa_error = toggle_favorite(_qa_store, ticker_symbol)
            if _qa_error:
                st.warning(_qa_error)
            else:
                st.session_state["quick_access_store"] = _qa_store
                save_quick_access(_qa_store)
                log_event(
                    logger, logging.INFO,
                    "user.favorite_removed" if _qa_pinned else "user.favorite_added",
                    ticker=ticker_symbol,
                )
                st.rerun()

    for _qa_col, (_qa_ticker, _qa_is_fav) in zip(_qa_cols[1:], _qa_chips):
        with _qa_col:
            _is_current = _qa_ticker == ticker_symbol
            _qa_label = f"★ {_qa_ticker}" if _qa_is_fav else _qa_ticker
            if st.button(
                _qa_label, key=f"qa_chip_{_qa_ticker}", width="stretch",
                type="primary" if _is_current else "secondary",
                help="Currently analysed" if _is_current else f"Switch analysis to {_qa_ticker}",
            ):
                st.session_state["_pending_ticker"] = _qa_ticker
                log_event(
                    logger, logging.INFO, "user.quick_access_switch",
                    ticker=_qa_ticker, favorite=_qa_is_fav,
                )
                st.rerun()

    if _qa_store.recents:
        if st.button(
            "Clear recents", key="clear_recents",
            help="Forget automatically-tracked recently-viewed symbols. Favorites are kept.",
        ):
            _qa_store = dataclasses.replace(_qa_store, recents=())
            st.session_state["quick_access_store"] = _qa_store
            save_quick_access(_qa_store)
            log_event(logger, logging.INFO, "user.recents_cleared")
            st.rerun()

if df.empty:
    detail = " ".join(ticker_bundle.errors) if ticker_bundle.errors else "No data returned by Yahoo Finance."
    log_event(logger, logging.ERROR, "analysis.aborted", ticker=ticker_symbol, reason=detail)
    st.error(f"No reliable data found for '{ticker_symbol}'. {detail}")

    # "Check the ticker symbol" is unhelpful when the symbol is RIGHT and
    # only lacks a venue. A European fund is quoted by its bare ticker
    # everywhere except this data source, which needs the exchange suffix:
    # VWCE does not resolve, VWCE.DE / VWCE.MI / VWCE.AS all do. Yahoo's
    # own search already maps one to the other — measured, searching
    # "VWCE" returns seven listings with VWCE.DE first — so the failure
    # path runs the search the sidebar already uses and offers what it
    # finds, rather than leaving the reader to guess a suffix.
    _fail_matches, _fail_err = ts_suggest_alternatives(ticker_symbol)
    if _fail_matches:
        st.markdown(
            f"**`{ticker_symbol}` is listed under a venue-specific symbol.** "
            "Funds outside the US are quoted by their bare ticker on most "
            "platforms, but this data source needs the exchange suffix. "
            "These are the listings it has:")
        for _fail_m in _fail_matches[:6]:
            if st.button(_fail_m.label, key=f"fail_hit_{_fail_m.symbol}",
                         width="stretch", help=_fail_m.detail or None):
                st.session_state["_pending_ticker"] = _fail_m.symbol
                log_event(logger, logging.INFO, "user.failed_ticker_recovery",
                          typed=ticker_symbol, picked=_fail_m.symbol)
                st.rerun()
        st.caption(
            "The same fund often lists on several exchanges in different "
            "currencies — check the venue before picking, because the "
            "price and the currency differ even though the holdings do not.")
    elif _fail_err:
        st.caption(_fail_err)
    else:
        st.caption("Try again shortly, or check the ticker symbol.")
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
    # Labels follow the asset class; POSITIONS never do. The eight panels
    # below are unpacked positionally and each is several hundred lines,
    # and ⌘1–⌘8 are bound to those positions — so a class re-labels its
    # tabs ("Holdings & Fund Profile" rather than "Fundamentals &
    # Valuation") and the one structural change is additive: a ninth
    # comparison tab appended for funds only, which leaves every existing
    # index untouched. See asset_views.
    _tab_labels = list(asset_views.tab_labels(asset_kind))
    _tab_objects = st.tabs(_tab_labels)
    (tab_overview, tab_chart_workspace, tab_fundamentals, tab_risk, tab_simulation,
     tab_smart_money, tab_portfolio, tab_tearsheet) = _tab_objects[:8]
    tab_comparison = (_tab_objects[8]
                      if asset_views.has_comparison(asset_kind) else None)


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
        executive_digest_container = st.empty()
        with executive_digest_container.container():
            st.markdown(
                loading_states.skeleton("Building executive digest", rows=(45, 80, 65, 30)),
                unsafe_allow_html=True,
            )

        # ==========================================
        # HISTORICAL COMPARISON
        # ==========================================
        # Replays the whole analysis as of an earlier date and shows it
        # beside today's. Lives on Overview because it's a statement about
        # this ticker's trajectory rather than about any one panel.
        st.markdown("---")
        with st.expander("Historical Comparison", expanded=False):
            _hc_earliest, _hc_latest = hc_available_range(ticker_bundle)
            if _hc_earliest is None:
                st.caption("No price history is loaded for this ticker, so there's nothing to replay.")
            else:
                st.caption(
                    "Re-runs the analysis using only what was knowable on the date you pick, and shows it "
                    "beside today's. Technicals and risk are recomputed from the price series truncated to "
                    "that date. Fundamentals come from the financial statement actually in force then — "
                    "Yahoo reports **annual** periods, so those step at roughly yearly boundaries rather "
                    "than daily. Today's figures are never carried backwards: anything that wasn't "
                    "knowable then is shown as unavailable rather than filled in."
                )
                _hc_default = max(_hc_earliest, min(_hc_latest, _hc_latest - datetime.timedelta(days=365)))
                _hc_col_date, _hc_col_btn = st.columns([2, 1])
                with _hc_col_date:
                    _hc_as_of = st.date_input(
                        "Replay the analysis as of",
                        value=_hc_default, min_value=_hc_earliest, max_value=_hc_latest,
                        key="hist_as_of",
                        help="Bounded by the price history currently loaded — widen Start Date in the sidebar to reach further back.",
                    )
                with _hc_col_btn:
                    st.markdown("&nbsp;")
                    _hc_run = st.button("Compare", type="primary", key="hist_run")

                if _hc_run:
                    with st.spinner(f"Replaying {ticker_symbol} as of {_hc_as_of:%d %b %Y}..."):
                        st.session_state["hist_result"] = hc_build_comparison(
                            ticker_bundle, _hc_as_of, risk_free_rate=risk_free_rate,
                        )
                        st.session_state["hist_result_ticker"] = ticker_symbol
                        log_event(logger, logging.INFO, "user.historical_comparison",
                                  ticker=ticker_symbol, as_of=str(_hc_as_of))

                _hc_res = st.session_state.get("hist_result")
                if _hc_res is not None and st.session_state.get("hist_result_ticker") == ticker_symbol:
                    if _hc_res.statement_period:
                        st.caption(
                            f"Fundamentals below are from the statement period ending "
                            f"**{_hc_res.statement_period:%d %b %Y}** — the filing in force on "
                            f"{_hc_res.as_of:%d %b %Y}. Technicals and risk use "
                            f"{_hc_res.price_bars:,} trading day{'' if _hc_res.price_bars == 1 else 's'} up to that date."
                        )
                    for _hc_w in _hc_res.warnings:
                        st.warning(_hc_w)

                    for _hc_group in ("Fundamentals", "Technicals", "Risk"):
                        _hc_rows = [m for m in _hc_res.metrics if m.group == _hc_group]
                        if not _hc_rows:
                            continue
                        st.markdown(f"**{_hc_group}**")
                        _hc_table = []
                        for _hc_m in _hc_rows:
                            _hc_table.append({
                                "Metric": _hc_m.label,
                                f"As of {_hc_res.as_of:%d %b %Y}": _hc_fmt(_hc_m, _hc_m.then),
                                "Today": _hc_fmt(_hc_m, _hc_m.now),
                                "Change": _hc_delta_text(_hc_m),
                            })
                        st.dataframe(pd.DataFrame(_hc_table), width="stretch", hide_index=True)

        # ==========================================
        # TEAM NOTES (per-ticker thread with @-mentions)
        # ==========================================
        # Lives on the Overview tab, attached to the ticker currently being
        # analysed — the note is about THIS stock, so it belongs beside the
        # analysis rather than in a separate area you'd have to navigate to.
        st.markdown("---")
        with st.expander(f"Team Notes — {ticker_symbol}", expanded=False):
            if "collab_store" not in st.session_state:
                st.session_state["collab_store"] = collab_load_store()
            _cl_store = st.session_state["collab_store"]

            _cl_caption = (
                "Notes attached to this ticker, shared by everyone using this Quantix instance — "
                "deliberately, since a thread only works if teammates can read each other. Anyone "
                "using this instance can read or delete any note. Mention a teammate with @ to "
                "email them; only people on the Team roster below can be mentioned, so a typo can "
                "never mail a stranger."
            )
            if _auth_user is not None:
                _cl_caption += (
                    f" You're signed in, so notes you post are attributed to **{_auth_user.display_name}** "
                    "as a verified identity."
                )
            else:
                _cl_caption += (
                    " **You're not signed in**, so the name you type is a self-declared label that "
                    "nothing verifies. Sign in from the Account panel to post under a verified name."
                )
            st.caption(_cl_caption)

            _cl_existing = collab_notes_for(_cl_store, ticker_symbol)
            if _cl_existing:
                for _cl_note in _cl_existing:
                    _cl_body_col, _cl_del_col = st.columns([12, 1])
                    with _cl_body_col:
                        _cl_when = _cl_note.created_at.replace("T", " ") if _cl_note.created_at else "unknown time"
                        _cl_badge = " (verified)" if _cl_note.authenticated else ""
                        _cl_meta = f"**{_cl_note.author}**{_cl_badge} · {_cl_when}"
                        if _cl_note.mentions:
                            _cl_sent = [m for m in _cl_note.mentions if m in _cl_note.notified]
                            _cl_unsent = [m for m in _cl_note.mentions if m not in _cl_note.notified]
                            _cl_bits = []
                            if _cl_sent:
                                _cl_bits.append("emailed " + ", ".join(_cl_sent))
                            if _cl_unsent:
                                _cl_bits.append("not emailed: " + ", ".join(_cl_unsent))
                            _cl_meta += " · " + "; ".join(_cl_bits)
                        st.markdown(_cl_meta)
                        st.markdown(_rt_md_escape_dollar(_cl_note.body))
                    with _cl_del_col:
                        if st.button("✕", key=f"collab_del_{_cl_note.id}", help="Delete this note"):
                            _cl_store = collab_delete_note(_cl_store, ticker_symbol, _cl_note.id)
                            st.session_state["collab_store"] = _cl_store
                            collab_save_store(_cl_store)
                            st.rerun()
            else:
                st.caption("No notes on this ticker yet.")

            st.markdown("---")
            if _auth_user is not None:
                # No text box: letting a signed-in user type a different name
                # would make the verified badge a lie.
                _cl_author = _auth_user.display_name
                st.caption(f"Posting as **{_cl_author}** (verified)")
            else:
                _cl_author_col, _cl_spacer = st.columns([2, 3])
                with _cl_author_col:
                    _cl_author = st.text_input(
                        "Your name", key="collab_author",
                        placeholder="e.g. Angelos",
                        help="Stored with the note as its author. Self-declared — nothing verifies it.",
                    )
            _cl_handles = ", ".join(f"@{m.handle}" for m in _cl_store.members) or "no teammates added yet"
            # Clear the compose box after a successful post — deferred to
            # the top of the NEXT run, before the widget is instantiated.
            #
            # The obvious version (pop the key in the button handler, then
            # rerun) does not work, and was shipped broken: popping a
            # text_area's key AFTER the widget has rendered raises nothing
            # and clears nothing, because Streamlit restores the widget's
            # value from its own widget-state layer on the next run rather
            # than from the session_state mirror that was popped. Verified
            # side by side in an isolated app — the popped box still read
            # "Test1" after posting; this one comes back empty.
            #
            # It mattered: a box that still holds the text you just posted
            # reads as "nothing happened", and the second click posts a
            # duplicate. That is exactly how it was found.
            if st.session_state.pop("_collab_clear_body", False):
                st.session_state["collab_body"] = ""
            _cl_body = st.text_area(
                "Add a note", key="collab_body",
                placeholder="Your thesis, a concern, a reminder… mention a teammate with @",
                help=f"Mentionable handles: {_cl_handles}",
            )
            if st.button("Post note", type="primary", key="collab_post"):
                _cl_store, _cl_note, _cl_err = collab_add_note(
                    _cl_store, ticker_symbol, _cl_author, _cl_body,
                    authenticated=_auth_user is not None,
                    issuer=_auth_user.issuer if _auth_user else "",
                )
                if _cl_err:
                    st.warning(_cl_err)
                else:
                    # Save FIRST, then attempt notification. A mail failure
                    # must never cost someone their written note.
                    st.session_state["collab_store"] = _cl_store
                    collab_save_store(_cl_store)
                    if _cl_note.mentions:
                        if is_email_configured():
                            _cl_sent, _cl_errs = collab_notify_mentions(
                                _cl_store, _cl_note, send_notification_email,
                            )
                            if _cl_sent:
                                _cl_store = collab_mark_notified(
                                    _cl_store, ticker_symbol, _cl_note.id, _cl_sent,
                                )
                                st.session_state["collab_store"] = _cl_store
                                collab_save_store(_cl_store)
                                st.success(f"Note posted — emailed {', '.join(_cl_sent)}.")
                            for _cl_e in _cl_errs:
                                st.warning(f"Couldn't notify {_cl_e}")
                            if not _cl_sent and not _cl_errs:
                                st.success("Note posted.")
                        else:
                            st.success("Note posted.")
                            st.info(
                                "Mentioned " + ", ".join(_cl_note.mentions) +
                                ", but email isn't configured on this instance so no notification "
                                "was sent. See .streamlit/secrets.toml.example to enable it."
                            )
                    else:
                        st.success("Note posted.")
                    log_event(logger, logging.INFO, "user.note_posted",
                              ticker=ticker_symbol, mentions=len(_cl_note.mentions))
                    st.session_state["_collab_clear_body"] = True
                    st.rerun()

            st.markdown("---")
            st.markdown("**Team roster**")
            st.caption(
                "Only these people can be @-mentioned, and only these addresses can ever receive "
                "a notification from this app."
            )
            for _cl_m in _cl_store.members:
                _cl_mc1, _cl_mc2 = st.columns([6, 1])
                with _cl_mc1:
                    st.caption(f"**{_cl_m.name}** · @{_cl_m.handle} · {_cl_m.email}")
                with _cl_mc2:
                    if st.button("✕", key=f"collab_rm_{_cl_m.handle}", help=f"Remove {_cl_m.name}"):
                        _cl_store = collab_remove_member(_cl_store, _cl_m.name)
                        st.session_state["collab_store"] = _cl_store
                        collab_save_store(_cl_store)
                        st.rerun()
            _cl_n1, _cl_n2, _cl_n3 = st.columns([2, 3, 1])
            with _cl_n1:
                _cl_new_name = st.text_input("Name", key="collab_new_name", placeholder="Ana Silva")
            with _cl_n2:
                _cl_new_email = st.text_input("Email", key="collab_new_email", placeholder="ana@example.com")
            with _cl_n3:
                st.markdown("&nbsp;")
                if st.button("Add", key="collab_add_member"):
                    _cl_store, _cl_merr = collab_add_member(_cl_store, _cl_new_name, _cl_new_email)
                    if _cl_merr:
                        st.warning(_cl_merr)
                    else:
                        st.session_state["collab_store"] = _cl_store
                        collab_save_store(_cl_store)
                        st.rerun()


        # ==========================================
        # DATA QUALITY REPORT
        # ==========================================
        # Combines field-level statement completeness (financial_validation.py),
        # data freshness (most recent reported quarter), and fetch reliability
        # (data_loader.py retries/warnings) into one score, run before any ratio
        # below is calculated — so it's clear up front how much to trust the
        # analysis instead of piecing it together from separate panels.
        # Computed once in the sticky header, where the badge lives.
        quality = data_quality_report

        _dq_is_equity = quality.asset_class == asset_class.EQUITY

        st.subheader(f"Data Quality Report — {quality.score}/100 ({quality.grade})")
        if not _dq_is_equity:
            # Say what was measured BEFORE showing the number. A reader who
            # assumes this score means the same thing for every symbol will
            # otherwise compare a fund's score against a stock's and think
            # the two were graded on the same evidence.
            st.caption(
                f"Scored as {asset_class.with_article(quality.asset_class)}, on "
                f"{quality.scored_on}. "
                f"{asset_class.spec(quality.asset_class).absence_reason} "
                "Its filings are not scored because it does not make any — an "
                "absent income statement here is a fact about the instrument, "
                "not a defect in the data.")

        # One column per dimension that actually carried weight.
        _dq_cols = st.columns(len(quality.dimensions))
        for _dq_col, _dq_dim in zip(_dq_cols, quality.dimensions):
            _dq_val = getattr(quality, _dq_dim.key)
            if _dq_dim.key == "freshness_score" and quality.staleness_days is not None:
                _dq_col.metric(
                    "Data Freshness", f"{quality.staleness_days}d old",
                    delta="Stale" if quality.is_stale else "Fresh",
                    delta_color="inverse" if quality.is_stale else "normal",
                    help=_dq_dim.help)
            else:
                _dq_col.metric(
                    _dq_dim.label, f"{_dq_val:.0f}%",
                    help=f"{_dq_dim.help} Worth {_dq_dim.weight:.0%} of the score.")

        if quality.grade in ("Poor", "Fair"):
            st.warning(f"Data quality is {quality.grade.lower()} for {ticker_symbol} — treat derived metrics with extra caution and check the detail below.")

        detail_issue_count = quality.issue_count
        with st.expander(f"Data Quality Detail ({detail_issue_count} issue(s))", expanded=quality.grade in ("Poor", "Fair")):
            if _dq_is_equity:
                for stmt in standardized.validation.statements:
                    status = "Complete" if stmt.is_valid else f"{len(stmt.missing_required)} required field(s) missing"
                    st.markdown(f"**{stmt.statement_name}** — {status}")
                    for check in stmt.checks:
                        icon = "Present" if check.present else ("Missing" if check.required else "Optional, absent")
                        label = check.name + (" (required)" if check.required else " (optional)")
                        st.markdown(f"&nbsp;&nbsp;{icon} {label}")

                if quality.most_recent_quarter is not None:
                    st.markdown(f"**Freshness** — most recent reported quarter: {quality.most_recent_quarter.strftime('%B %d, %Y')} ({quality.staleness_days} days ago)")
                else:
                    st.markdown("**Freshness** — most recent quarter date not reported by Yahoo Finance; freshness could not be verified.")
            else:
                st.markdown(
                    "**Statements** — not applicable. "
                    f"{asset_class.unavailable_note(quality.asset_class, asset_class.FUNDAMENTALS)}")
                if asset_class.supports(quality.asset_class, asset_class.HOLDINGS):
                    _dq_missing = quality.missing_fund_fields
                    st.markdown(
                        "**Fund profile** — "
                        + ("every reported field present."
                           if not _dq_missing else
                           f"{len(_dq_missing)} field(s) not reported: "
                           + ", ".join(_dq_missing)))
                _dq_age = quality.price_age_days
                st.markdown(
                    "**Price history** — "
                    + ("no price series returned."
                       if _dq_age is None and quality.price_history_score == 0 else
                       "present; last bar's date could not be read."
                       if _dq_age is None else
                       f"last bar {_dq_age} day(s) old."))

            if quality.fetch_errors or quality.fetch_warnings:
                st.markdown("**Fetch Reliability**")
                if not _dq_is_equity and any(
                        data_quality._is_statement_warning(w)
                        for w in quality.fetch_warnings):
                    # Otherwise the header says "0 issue(s)" above a list of
                    # three warnings, and the reader cannot tell whether
                    # they counted.
                    st.caption(
                        "The statement warnings below are recorded for "
                        "completeness and did NOT count against this score — "
                        "this instrument files no statements, so their "
                        "absence is expected rather than a data problem.")
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
        macro_risk_flag = vix_current > effective_risk().vix_high_risk_threshold

        m1, m2 = st.columns(2)
        m1.metric("VIX (Fear Index)", f"{vix_current:.2f}", delta=f"High Risk (>{effective_risk().vix_high_risk_threshold:.0f})" if macro_risk_flag else "Stable Market", delta_color="inverse" if macro_risk_flag else "normal", help=help_for("vix"))
        m2.metric("10-Year Treasury Yield", f"{tnx_current:.2f}%", help=help_for("treasury_10y"))

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
    # NameError-avoiding placeholders for the equity-only panels below.
    # A non-equity skips those panels entirely, but the Executive Digest
    # and the tear sheet read these names unconditionally — so they are
    # defined here rather than inside the gates that may not execute.
    # Every READ of the DCF outputs is already guarded on
    # `dcf_result is not None and dcf_result.ok`, which is what stops a
    # skipped valuation rendering as an intrinsic value of $0.00.
    dcf_result = None
    intrinsic_price = 0.0
    intrinsic_value = 0.0
    margin_of_safety = 0.0
    score_pct = fundamentals.score_pct
    # Hoisted OUT of the equity-only panel below: the Executive Digest
    # reads cq.category unconditionally, and leaving the assignment inside
    # the gate made it undefined for every non-equity — a NameError that
    # only fires once a crypto or currency is loaded. company_quality is
    # always present; for an instrument with no financials it reports
    # "Not Ratable", which is exactly what the digest should say.
    cq = fundamentals.company_quality

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
        # Equity-only. Nothing here has meaning for an instrument with
        # no issuer and no filings, and rendering it as "not reported"
        # would imply the question was sensible and the data merely
        # absent. See asset_class.py.
        if not asset_class.supports(asset_kind, asset_class.FUNDAMENTALS):
            st.info(asset_class.unavailable_note(asset_kind, asset_class.FUNDAMENTALS))
            for _ac_gap in asset_class.missing_sources(asset_kind):
                st.caption(f"Not sourced in this build: {_ac_gap}.")

            # A fund has no financials of its own, but it is not opaque:
            # what it HOLDS, what it costs and how it trades are all
            # answerable. This is the analysis that replaces the company
            # scorecard rather than an apology for its absence.
            if asset_class.supports(asset_kind, asset_class.HOLDINGS):
                _etf = etf_analysis.load_profile(ticker_symbol)
                if not _etf.ok:
                    st.warning(_etf.error)
                else:
                    st.markdown("---")
                    st.header("Fund Decomposition")
                    st.caption(
                        f"{_etf.category or 'Uncategorised'}"
                        + (f" · {_etf.family}" if _etf.family else "")
                        + (f" · {_etf.legal_type}" if _etf.legal_type else ""))

                    _etf_c1, _etf_c2, _etf_c3, _etf_c4 = st.columns(4)
                    with _etf_c1:
                        st.metric("Price / Earnings",
                                  f"{_etf.price_earnings:.1f}x" if _etf.price_earnings
                                  else "Not reported",
                                  help="Whole-fund figure. Yahoo reports this "
                                       "field as its reciprocal; it is inverted "
                                       "here — see etf_analysis.")
                    with _etf_c2:
                        st.metric("Price / Book",
                                  f"{_etf.price_book:.2f}x" if _etf.price_book
                                  else "Not reported",
                                  help="Whole-fund price-to-book, inverted from the "
                                       "reciprocal this source reports. Below ~2x leans "
                                       "value, above ~5x leans growth.")
                    with _etf_c3:
                        st.metric("Expense ratio",
                                  f"{_etf.expense_ratio_pct:.2f}%"
                                  if _etf.expense_ratio_pct is not None else "Not reported",
                                  delta=(f"{etf_analysis.expense_gap_pct(_etf):+.2f}pp vs category"
                                         if etf_analysis.expense_gap_pct(_etf) is not None
                                         else None),
                                  delta_color="inverse",
                                  help="The annual fee, as a percentage of assets, "
                                       "against the average for this fund's category. "
                                       "Charged whether the fund gains or loses.")
                    with _etf_c4:
                        st.metric("Style", etf_analysis.style_label(_etf),
                                  help="The provider's own category where it gives one, "
                                       "since it carries the size band too. Falls back "
                                       "to the fund's price-to-earnings only when no "
                                       "category is reported.")

                    if etf_analysis.expense_is_high(_etf):
                        st.warning(
                            f"This fund costs {etf_analysis.expense_gap_pct(_etf):.2f} "
                            "percentage points more than its category average.")

                    _etf_gap = etf_analysis.valuation_gap_pct(_etf)
                    if _etf_gap is not None:
                        st.caption(
                            f"Valuation gap vs category: {_etf_gap:+.1f}% "
                            f"({_etf.price_earnings:.1f}x against "
                            f"{_etf.category_price_earnings:.1f}x).")

                    # --- what it holds ---------------------------------
                    if _etf.top_holdings:
                        _etf_covered = etf_analysis.concentration_pct(_etf.top_holdings)
                        st.markdown("**Top holdings**")
                        st.caption(
                            f"These {len(_etf.top_holdings)} are {_etf_covered:.1f}% of "
                            "the fund. That is a CONCENTRATION figure — the "
                            "valuation above is computed across the whole "
                            "portfolio, not from this list, because weighting "
                            f"a P/E by {_etf_covered:.0f}% of the fund would "
                            "understate it by the rest.")
                        st.dataframe(
                            pd.DataFrame([
                                {"Ticker": h.symbol, "Name": h.name,
                                 "Weight %": round(h.weight_pct, 2)}
                                for h in _etf.top_holdings
                            ]),
                            width="stretch", hide_index=True)

                    if _etf.sector_weights:
                        _etf_sect = pd.DataFrame(
                            [{"Sector": k.replace("_", " ").title(),
                              "Weight %": round(v * 100.0, 2)}
                             for k, v in _etf.sector_weights.items() if v],
                        ).sort_values("Weight %", ascending=False)
                        # graph_objects, not express: px is not imported in
                        # this app and adding it for one chart would be a
                        # second plotting idiom for no gain.
                        _etf_fig = go.Figure(go.Pie(
                            labels=_etf_sect["Sector"], values=_etf_sect["Weight %"],
                            hole=0.45, sort=False))
                        _etf_fig.update_layout(
                            title="Sector allocation",
                            template=_theme.plotly_template,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font_color=_theme.chart_fg,
                            margin=dict(t=48, b=8, l=8, r=8),
                        )
                        st.plotly_chart(_etf_fig, width="stretch")
                        st.caption(chart_help("etf_sector_allocation"))

                    # --- what it costs ---------------------------------
                    st.markdown("**What the fee costs over time**")
                    _etf_drag_rows = []
                    for _etf_years in etf_analysis.DRAG_YEARS:
                        _etf_drag = etf_analysis.expense_drag(
                            _etf.expense_ratio_pct, _etf_years)
                        _etf_drag_rows.append({
                            "Horizon": f"{_etf_years} years",
                            "Given up to fees":
                                f"{_etf_drag:.2f}% of the gross outcome"
                                if _etf_drag is not None else "Not reported",
                        })
                    st.dataframe(pd.DataFrame(_etf_drag_rows),
                                 width="stretch", hide_index=True)
                    st.caption(
                        f"Illustrated against a declared "
                        f"{etf_analysis.DRAG_ASSUMED_GROSS_RETURN_PCT:.0f}% gross "
                        "annual return, compounded. That rate is an assumption "
                        "for showing what a fee costs, not a forecast of this "
                        "fund's return.")

                    # --- how good is it --------------------------------
                    _etf_parts, _etf_score, _etf_how = etf_analysis.quality_scorecard(_etf)
                    st.markdown("**Holdings quality scorecard**")
                    if _etf_score is None:
                        st.caption(_etf_how)
                    else:
                        st.metric("Score", f"{_etf_score:.1f} / 10",
                                  help="Averaged over the components that could be "
                                       "scored from the available data — the count is "
                                       "stated below, and unscored components are "
                                       "listed rather than quietly dropped.")
                        st.caption(_etf_how)
                    for _etf_part in _etf_parts:
                        _etf_shown = (f"{_etf_part.score:.1f}/10"
                                      if _etf_part.score is not None else "not scored")
                        st.caption(f"**{_etf_part.name}** — {_etf_shown}. {_etf_part.detail}")

                    # --- identity, lifecycle and performance ----------
                    # PHASE 1.1. Everything here is either absent from the
                    # panels above (ISIN, inception, dividend cadence) or
                    # deliberately recomputed, because the provider's own
                    # inception, beta and return fields are each wrong in
                    # a different way — see etf_pipeline's docstring.
                    _ident = etf_pipeline.load_identity(ticker_symbol)
                    if _ident.ok:
                        st.markdown("---")
                        st.subheader("Fund identity & record")
                        _id1, _id2, _id3 = st.columns(3)
                        _id1.metric(
                            "ISIN", _ident.isin or "Not reported",
                            help="The fund's international securities "
                                 "identifier. Not every listing reports "
                                 "one — QQQ and the European listings "
                                 "checked do not.")
                        _id2.metric(
                            "Distributions",
                            _ident.dividend_frequency or "None",
                            help="Derived from the median gap between "
                                 "payments rather than counting a "
                                 "trailing year, which over-counts "
                                 "whenever a boundary payment lands "
                                 "inside the window.")
                        _id3.metric(
                            "Beta vs S&P 500",
                            "Unavailable" if _ident.beta is None
                            else f"{_ident.beta:.2f}",
                            help="Regressed from daily returns, not taken "
                                 "from the provider's own field — that "
                                 "field reports 2.40 for a long treasury "
                                 "fund whose measured beta is near zero.")
                        if _ident.beta is not None and _ident.beta_r_squared is not None:
                            _id3.caption(
                                f"R² {_ident.beta_r_squared:.2f}"
                                + (" — barely explains this fund's moves, "
                                   "which is itself the point for "
                                   "something held to diversify away from "
                                   "equities."
                                   if _ident.beta_r_squared < 0.2 else ""))
                        st.caption(etf_pipeline.describe_inception(_ident))
                        if _ident.beta_disagrees_with_reported:
                            st.caption(
                                f"The data source reports a beta of "
                                f"{_ident.reported_beta:.2f} for this fund; "
                                f"regressed from its own prices it is "
                                f"{_ident.beta:.2f}. The measured figure is "
                                "the one shown.")

                        _perf_rows = [
                            {"Window": w.label,
                             "Return": (None if w.return_pct is None
                                        else w.return_pct),
                             "Basis": ("annualised" if w.annualised
                                       else "total" if w.return_pct is not None
                                       else "not enough history")}
                            for w in _ident.performance]
                        _perf_table = pd.DataFrame(_perf_rows)
                        _perf_table["Return"] = pd.to_numeric(
                            _perf_table["Return"], errors="coerce")
                        st.dataframe(
                            _perf_table, width="stretch", hide_index=True,
                            column_config={"Return": st.column_config.NumberColumn(
                                "Return", format="%.2f%%")},
                            key="etf_performance_windows")
                        st.caption(
                            "Computed from the price series. The provider's "
                            "own return fields mix units — its year-to-date "
                            "figure is a percent while its three- and "
                            "five-year figures are fractions, 100x apart in "
                            "the same response. Windows past a year are "
                            "annualised so they can be read beside the "
                            "shorter ones.")
                        st.caption(etf_pipeline.GEOGRAPHIC_ALLOCATION_UNAVAILABLE)
                        if not etf_pipeline.morningstar_is_configured():
                            st.caption(etf_pipeline.MORNINGSTAR_UNCONFIGURED)
        else:
            st.markdown("---")
            st.header("Financial Metrics Validation Report")

            if mv.is_clean:
                st.success(f"No issues found across {len(mv.evaluated_checks)} evaluated metric(s) for {ticker_symbol}.")
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
            vc1.metric("Metrics Evaluated", f"{len(mv.evaluated_checks)} / {len(mv.checks)}", help=help_for("metrics_evaluated"))
            vc2.metric("Yahoo Disagreements", mv.disagreement_count, help=help_for("yahoo_disagreements"))
            vc3.metric("Extreme Outliers", mv.outlier_count, help=help_for("extreme_outliers"))
            vc4.metric("Incomplete Calculations", mv.fallback_count, help=help_for("incomplete_calculations"))

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
            c1.metric("Institutional Green Flags", f"{green_flags} / {total_checks}", help=help_for("green_flags"))
            c2.metric("Operational Warning Signs", f"{total_checks - green_flags}", help=help_for("warning_signs"))
            c3.metric("Blueprint Alignment", f"{score_pct:.0f}%", help="Weighted over evaluable metrics — see the sector/weighting note above.")

            if total_checks < possible_checks:
                st.caption(f"{possible_checks - total_checks} of {possible_checks} scorecard metric(s) not computable for {ticker_symbol} and excluded from scoring, rather than counted as a failure.")

            if fundamentals.alignment_verdict == "high": st.success("HIGH ALIGNMENT: Passes major filters.")
            elif fundamentals.alignment_verdict == "moderate": st.warning("MODERATE RISK: Proceed with caution.")
            else: st.error("ABORT RESEARCH: Fails safety benchmarks.")

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
                st.warning(f"Not enough data to classify {ticker_symbol}'s quality — every factor was missing all of its inputs.")
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
            # A "Differs" does not necessarily mean our formula is wrong: Yahoo's figure is
            # often trailing-twelve-month while ours uses the most recent annual
            # period, and that timing difference alone can exceed the tolerance.
            st.markdown("---")
            st.header("Profitability Validation Report")
            st.caption("Formula vs. Yahoo Finance's own reported ratio for the same concept · Agrees (within 15%) · Differs · Not checked (no independent reference, or not applicable for this company)")

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
            st.caption("Formula vs. Yahoo Finance's own reported ratio for the same concept · Agrees (within 15%) · Differs · Not checked (no independent reference, or not applicable for this company)")

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
            st.caption("Formula vs. Yahoo Finance's own reported figure for the same concept · Agrees (within 15%) · Differs · Not checked (no independent reference, or not applicable for this company)")

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
            st.caption("Formula vs. Yahoo Finance's own reported figure for the same concept · Agrees (within 15%) · Differs · Not checked (no independent reference, or not applicable for this company)")

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
            st.caption("Price data validated — no duplicate timestamps, invalid bars, or likely gaps detected.")
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
                    marker=dict(symbol='star', size=12, color='#ef4444', line=dict(width=1, color=_chart_fg)),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=[b.date for b in bb_lower_breaks], y=[b.price for b in bb_lower_breaks], mode='markers', name='BB Lower Breakout',
                    marker=dict(symbol='star', size=12, color='#22c55e', line=dict(width=1, color=_chart_fg)),
                ), row=1, col=1)
        if sma_signals:
            bullish = [s for s in sma_signals if s.kind == "bullish"]
            bearish = [s for s in sma_signals if s.kind == "bearish"]
            fig.add_trace(go.Scatter(
                x=[s.date for s in bullish], y=[s.price for s in bullish], mode='markers', name='Bullish Crossover',
                marker=dict(symbol='triangle-up', size=11, color='#22c55e', line=dict(width=1, color=_chart_fg)),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=[s.date for s in bearish], y=[s.price for s in bearish], mode='markers', name='Bearish Crossover',
                marker=dict(symbol='triangle-down', size=11, color='#ef4444', line=dict(width=1, color=_chart_fg)),
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
            fig.add_hline(y=0, line_dash="dot", line_color=_chart_faint_line, row=macd_row, col=1)
            if macd_signals:
                macd_bullish = [s for s in macd_signals if s.kind == "bullish"]
                macd_bearish = [s for s in macd_signals if s.kind == "bearish"]
                fig.add_trace(go.Scatter(
                    x=[s.date for s in macd_bullish], y=[s.macd_value for s in macd_bullish], mode='markers', name='MACD Bullish Crossover',
                    marker=dict(symbol='triangle-up', size=9, color='#22c55e', line=dict(width=1, color=_chart_fg)),
                ), row=macd_row, col=1)
                fig.add_trace(go.Scatter(
                    x=[s.date for s in macd_bearish], y=[s.macd_value for s in macd_bearish], mode='markers', name='MACD Bearish Crossover',
                    marker=dict(symbol='triangle-down', size=9, color='#ef4444', line=dict(width=1, color=_chart_fg)),
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
                    marker=dict(symbol='triangle-up', size=9, color='#22c55e', line=dict(width=1, color=_chart_fg)),
                ), row=stoch_row, col=1)
                fig.add_trace(go.Scatter(
                    x=[s.date for s in stoch_bearish], y=[s.k_value for s in stoch_bearish], mode='markers', name='Stochastic Bearish Crossover',
                    marker=dict(symbol='triangle-down', size=9, color='#ef4444', line=dict(width=1, color=_chart_fg)),
                ), row=stoch_row, col=1)
        if show_adx_panel:
            adx_row = row_of["adx"]
            fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color=_chart_fg, width=1.5), name='ADX'), row=adx_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Plus_DI'], line=dict(color='#22c55e', width=1), name='+DI'), row=adx_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Minus_DI'], line=dict(color='#ef4444', width=1), name='-DI'), row=adx_row, col=1)
            fig.add_hline(y=TECHNICAL.adx_trend_threshold, line_dash="dash", line_color=_chart_faint_line, row=adx_row, col=1)
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
        template=_plotly_template,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
        # Explicit, not relying on Streamlit/Plotly defaults: the modebar (with
        # its built-in zoom/pan/fullscreen-expand controls) stays visible, mouse
        # scroll zooms directly, and the Plotly logo link is dropped as clutter.
        st.caption(chart_help("price_technicals"))
        st.plotly_chart(fig, width="stretch", config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False})

        # Current-value interpretation (RSI Interpretation Engine) — the shaded
        # zones above cover the full history at a glance; this is the "what does
        # today's reading mean" readout for the latest bar specifically.
        rsi_series = df[f"RSI_{rsi_length}"]
        rsi_interpretation = interpret_rsi(rsi_series.iloc[-1]) if rsi_series.notna().any() else None
        if rsi_interpretation:
            ri1, ri2 = st.columns([1, 3])
            ri1.metric(f"RSI ({rsi_length})", f"{rsi_interpretation.value:.1f}", help=help_for("rsi"))
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
            a1.metric(f"{ticker_symbol} Period Return", f"{ticker_return:.2f}%", help=help_for("period_return"))
            a2.metric(f"{benchmark_symbol} Period Return", f"{bench_return:.2f}%", help=help_for("period_return"))
            a3.metric("Generated Alpha", f"{alpha:.2f}%", help=help_for("alpha_generated"))

            fig_alpha = go.Figure()
            fig_alpha.add_trace(go.Scatter(x=df.index, y=df['CumReturn']*100, name=ticker_symbol, line=dict(color='orange')))
            fig_alpha.add_trace(go.Scatter(x=bench_df.index, y=bench_df['CumReturn']*100, name=benchmark_symbol, line=dict(color='gray')))
            fig_alpha.update_layout(xaxis_rangeslider_visible=False, height=400, template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.caption(chart_help("relative_strength"))
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
            p1.metric("Total Excess Return", f"{attribution.total_excess_return_pct:.2f}%", help=f'{help_for("excess_return_total")} Here: {ticker_symbol}\'s period return minus the period risk-free rate.')
            p2.metric("Systematic (Market Beta)", f"{attribution.systematic_pct:.2f}%", help=help_for("beta_systematic"))
            p3.metric("Selection (Residual)", f"{attribution.selection_pct:.2f}%", help=help_for("alpha_selection"))


        # ==========================================
        # FUND TECHNICALS & SECTOR MOMENTUM  (PHASE 1.4)
        # ==========================================
        # Placed in the Chart Workspace rather than beside Fund
        # Decomposition, because everything here is price-derived and
        # belongs next to the chart it describes.
        #
        # The indicator suite above is NOT rebuilt: the SMA trio, RSI and
        # MACD subplots and the up/down-coloured volume bars already
        # render for a fund, because none of them was ever gated on asset
        # class. What follows is the part that is specific to a basket.
        if asset_class.supports(asset_kind, asset_class.HOLDINGS):
            st.markdown("---")
            st.header("Fund Technicals & Sector Momentum", anchor="fund-technicals")

            # Computed here rather than read off `df`, deliberately. The
            # 50/200-day lines above exist only when the "Show 20/50/200
            # SMA Trio" checkbox is ticked, and a signal that reported
            # "unavailable" because a DISPLAY toggle was off would be
            # blaming the data for a UI state.
            _ft_sma = compute_sma_lines(df, (50, etf_technicals.SMA_LONG_PERIOD))
            _ft_rsi = compute_rsi(df, rsi_length)
            _ft_rsi_clean = _ft_rsi.dropna() if _ft_rsi is not None else None
            _ft_rsi_latest = (float(_ft_rsi_clean.iloc[-1])
                              if _ft_rsi_clean is not None and len(_ft_rsi_clean) else None)
            _ft_volume = etf_technicals.relative_volume(df)
            _ft_range = etf_technicals.range_position(df)
            _ft_verdict = etf_technicals.momentum_verdict(
                _ft_rsi_latest, _ft_sma, df, _ft_range)

            _ft_g1, _ft_g2, _ft_g3 = st.columns(3)
            _ft_g1.metric(
                "Momentum", _ft_verdict.label,
                help=("Net of the readings that could actually be taken over "
                      "the loaded range — RSI, price against its 50- and "
                      "200-day averages, and position in range. Scored out "
                      "of what was measurable, so a short range gives a "
                      "less confident reading rather than a falsely "
                      "decisive one."))
            _ft_g1.caption(
                f"{_ft_verdict.considered} reading(s) considered"
                if _ft_verdict.considered else "Not enough history to read.")
            if _ft_range.position_pct is None:
                _ft_g2.metric("Position in range", "Unavailable",
                              help="0% sits at the low of the loaded range, "
                                   "100% at the high. Not computable when the "
                                   "range has no span.")
                _ft_g2.caption("The loaded range has no high-low span.")
            else:
                _ft_g2.metric("Position in range", f"{_ft_range.position_pct:.0f}%",
                              help="0% sits at the low of the loaded range, "
                                   "100% at the high.")
                _ft_g2.caption(
                    f"{_ft_range.low:,.2f} – {_ft_range.high:,.2f} over "
                    f"{_ft_range.days_used} trading day(s)"
                    + ("" if _ft_range.sufficient else
                       " — shorter than a year, so this is the loaded range "
                       "and not a 52-week figure"))
            if _ft_volume.ok:
                _ft_g3.metric("Relative volume", f"{_ft_volume.ratio_pct:.0f}%",
                              help="Latest bar's volume as a percentage OF its "
                                   "trailing average. The average excludes the "
                                   "latest bar, so a doubling reads as 200%.")
                _ft_g3.caption(etf_technicals.describe_volume(_ft_volume))
            else:
                _ft_g3.metric("Relative volume", "Unavailable",
                              help="Latest bar's volume as a percentage OF its "
                                   "trailing average. Not computable without "
                                   "volume for the latest bar.")
                _ft_g3.caption("Volume was not reported for the latest bar.")

            if _ft_range.sufficient and _ft_range.at_new_high:
                st.success(f"{ticker_symbol} closed at a new 52-week high.")
            elif _ft_range.sufficient and _ft_range.at_new_low:
                st.warning(f"{ticker_symbol} closed at a new 52-week low.")

            # --- signals -------------------------------------------------
            st.markdown("**Entry / exit signals**")
            _ft_signals = etf_technicals.signals(df, _ft_sma, _ft_rsi, _ft_volume)
            _ft_state_labels = {
                etf_technicals.FIRED: "Fired",
                etf_technicals.NOT_FIRED: "Not fired",
                etf_technicals.UNAVAILABLE: "Not evaluated",
            }
            st.dataframe(
                pd.DataFrame([
                    {"Signal": _s.name,
                     "Status": _ft_state_labels[_s.state],
                     "Basis": _s.detail}
                    for _s in _ft_signals
                ]),
                width="stretch", hide_index=True, key="etf_signal_table")
            st.caption(
                "“Not evaluated” is not “not fired” — it means the loaded "
                "range is too short to check that rule at all. The "
                "200-day trend rule needs a year of history; a three-month "
                "range gives 63 trading days.")

            # --- sector momentum -----------------------------------------
            st.markdown("**Sector momentum**")
            _ft_weights = {}
            _ft_profile = etf_analysis.load_profile(ticker_symbol)
            if _ft_profile.ok:
                _ft_weights = _ft_profile.sector_weights or {}

            if not _ft_weights:
                st.info(
                    "This fund reports no equity sector weightings. That is "
                    "normal for a bond fund or a commodity trust — there are "
                    "no equity sectors to weight — rather than a gap in the "
                    "data.")
            else:
                _ft_sector_returns, _ft_sector_error = etf_technicals.load_sector_returns()
                if _ft_sector_error:
                    st.warning(_ft_sector_error)
                _ft_fund_move = etf_technicals._pct_change(
                    df["Close"], etf_technicals.MOMENTUM_LOOKBACK_DAYS)
                _ft_rows = etf_technicals.sector_momentum(
                    _ft_weights, _ft_sector_returns, _ft_fund_move)
                _ft_estimate = etf_technicals.estimated_fund_move(_ft_rows)

                st.caption(
                    f"Each sector's weight in {ticker_symbol} against that "
                    f"sector's own {etf_technicals.MOMENTUM_LOOKBACK_DAYS}-day "
                    "move, measured through the SPDR select sector ETFs. "
                    "Yahoo reports a fund's sector weights as a single "
                    "undated snapshot with no history, so the time dimension "
                    "comes from sector prices — the proxy is the sector as a "
                    "whole, not this fund's particular holdings within it.")

                _ft_t = pd.DataFrame([
                    {"Sector": _r.label,
                     "Proxy": _r.proxy,
                     "Weight %": _r.weight_pct,
                     f"{etf_technicals.MOMENTUM_LOOKBACK_DAYS}d %": _r.return_pct,
                     "Contribution (pp)": _r.contribution_pct,
                     "vs fund (pp)": _r.divergence_pct}
                    for _r in _ft_rows
                ])
                for _c in ("Weight %", f"{etf_technicals.MOMENTUM_LOOKBACK_DAYS}d %",
                           "Contribution (pp)", "vs fund (pp)"):
                    _ft_t[_c] = pd.to_numeric(_ft_t[_c], errors="coerce")
                st.dataframe(
                    _ft_t, width="stretch", hide_index=True,
                    column_config={
                        "Weight %": st.column_config.NumberColumn(
                            "Weight %", format="%.1f%%"),
                        f"{etf_technicals.MOMENTUM_LOOKBACK_DAYS}d %":
                            st.column_config.NumberColumn(
                                f"{etf_technicals.MOMENTUM_LOOKBACK_DAYS}d %",
                                format="%.2f%%",
                                help="The sector proxy ETF's own return."),
                        "Contribution (pp)": st.column_config.NumberColumn(
                            "Contribution (pp)", format="%.2f",
                            help="Weight x sector return, in percentage points "
                                 "of the fund's move."),
                        "vs fund (pp)": st.column_config.NumberColumn(
                            "vs fund (pp)", format="%+.2f",
                            help="Sector return minus the fund's own return "
                                 "over the same window."),
                    },
                    key="etf_sector_momentum_table")

                if _ft_estimate is not None and _ft_fund_move is not None:
                    st.caption(
                        f"Contributions sum to {_ft_estimate:+.2f}pp against "
                        f"{ticker_symbol}'s actual "
                        f"{etf_technicals.MOMENTUM_LOOKBACK_DAYS}-day move of "
                        f"{_ft_fund_move:+.2f}%. The two are close when the "
                        "fund tracks its sectors; a gap is the fund's own "
                        "security selection.")

                _ft_ahead, _ft_behind = etf_technicals.leaders_and_laggards(_ft_rows)
                if _ft_ahead or _ft_behind:
                    _ft_parts = []
                    if _ft_ahead:
                        _ft_parts.append(
                            "outperforming: " + ", ".join(
                                f"{_r.label} ({_r.divergence_pct:+.1f}pp)"
                                for _r in _ft_ahead))
                    if _ft_behind:
                        _ft_parts.append(
                            "underperforming: " + ", ".join(
                                f"{_r.label} ({_r.divergence_pct:+.1f}pp)"
                                for _r in _ft_behind))
                    st.caption(
                        f"Diverging from the fund by "
                        f"{etf_technicals.DIVERGENCE_FLAG_PCT:.0f}pp or more — "
                        + "; ".join(_ft_parts) + ".")

            # --- what is deliberately absent ------------------------------
            st.caption(etf_technicals.NAV_PREMIUM_UNAVAILABLE)

            # --- risk: tracking, capture, concentration (PHASE 1.3) ----
            st.markdown("---")
            st.subheader("Fund risk")
            _rk_bench = etf_comparison.load_prices(
                (ticker_symbol, benchmark_symbol), period="3y")[0]
            _rk_track = _rk_cap = None
            if (_rk_bench is not None and ticker_symbol in _rk_bench
                    and benchmark_symbol in _rk_bench):
                _rk_track = etf_risk.tracking(
                    _rk_bench[ticker_symbol], _rk_bench[benchmark_symbol],
                    benchmark_symbol)
                _rk_cap = etf_risk.capture_ratios(
                    _rk_bench[ticker_symbol], _rk_bench[benchmark_symbol])

            _rk1, _rk2, _rk3 = st.columns(3)
            if _rk_track is not None and _rk_track.ok:
                _rk1.metric(
                    f"Tracking error vs {benchmark_symbol}",
                    f"{_rk_track.tracking_error_pct:.2f}%",
                    help="Annualised standard deviation of the daily "
                         "return difference. Measured over three years.")
                _rk1.caption(_rk_track.band)
                _rk2.metric(
                    "Cumulative gap",
                    f"{_rk_track.cumulative_gap_pct:+.2f}%",
                    help="Total return over the same window, fund minus "
                         "benchmark. This is where a fee shows up; the "
                         "daily figure beside it is mostly closing-price "
                         "mismatch.")
                _rk2.caption(
                    f"{_rk_track.annualised_gap_pct:+.2f}% a year over "
                    f"{_rk_track.days} trading days")
            else:
                _rk1.metric(f"Tracking error vs {benchmark_symbol}",
                            "Unavailable",
                            help="Needs overlapping daily history for the "
                                 "fund and the benchmark.")
                _rk2.metric("Cumulative gap", "Unavailable",
                            help="Needs overlapping daily history for the "
                                 "fund and the benchmark.")
            if _rk_cap is not None and _rk_cap.ok:
                _rk3.metric(
                    "Up / down capture",
                    f"{_rk_cap.up_pct:.0f}% / {_rk_cap.down_pct:.0f}%",
                    help="Share of the benchmark's average up-day and "
                         "down-day move this fund captured. Measured on "
                         "the BENCHMARK's direction, which is what makes "
                         "it a capture ratio.")
                if _rk_cap.asymmetry is not None:
                    _rk3.caption(
                        f"{_rk_cap.asymmetry:+.0f}pp asymmetry — "
                        + ("more of the rise than the fall"
                           if _rk_cap.asymmetry > 0 else
                           "more of the fall than the rise"))
            else:
                _rk3.metric("Up / down capture", "Unavailable",
                            help="Needs overlapping daily history for the "
                                 "fund and the benchmark.")
            st.caption(etf_risk.BENCHMARK_IS_YOUR_CHOICE)
            st.caption(etf_risk.PRICE_INDEX_FLATTERS_A_FUND)

            # --- concentration and liquidity ---------------------------
            _rk_prof = etf_analysis.load_profile(ticker_symbol)
            _rk_conc = etf_risk.concentration(
                _rk_prof.top_holdings if _rk_prof.ok else ())
            _rk_info = ticker_bundle.info or {}
            _rk_liq = etf_risk.liquidity(
                price=_rk_info.get("regularMarketPrice"),
                volume=_rk_info.get("averageVolume") or _rk_info.get("volume"),
                assets=_rk_prof.net_assets if _rk_prof.ok else None,
                bid=_rk_info.get("bid"), ask=_rk_info.get("ask"))

            _rc1, _rc2, _rc3 = st.columns(3)
            if _rk_conc.ok:
                _rc1.metric("Concentration (HHI)",
                            f"{_rk_conc.herfindahl:.4f}",
                            help="Sum of squared weights over the "
                                 "disclosed holdings. 1.0 would be a "
                                 "single position.")
                _rc1.caption(
                    f"Effective {_rk_conc.effective_holdings:.0f} equally "
                    "weighted positions")
                _rc2.metric("Largest holding",
                            f"{_rk_conc.max_holding_pct:.2f}%",
                            help="The single biggest position among the "
                                 "disclosed holdings.")
                _rc2.caption(_rk_conc.max_holding_symbol or "")
            else:
                _rc1.metric("Concentration (HHI)", "Unavailable",
                            help="This fund discloses no holdings — normal "
                                 "for a bond or commodity fund.")
                _rc2.metric("Largest holding", "Unavailable",
                            help="This fund discloses no holdings.")
            if _rk_liq.turnover_pct is not None:
                _rc3.metric("Daily volume / AUM",
                            f"{_rk_liq.turnover_pct:.2f}%",
                            help="Average daily dollar volume against fund "
                                 "size — how much of the fund changes "
                                 "hands on an ordinary day.")
            else:
                _rc3.metric("Daily volume / AUM", "Unavailable",
                            help="Needs both a volume and a fund size.")
            _rc3.caption(_rk_liq.detail)
            if _rk_conc.ok:
                st.caption(etf_risk.TOP_TEN_CONCENTRATION_NOTE)
            st.caption(etf_risk.BID_ASK_MOSTLY_ABSENT)

            # --- bond funds: rate risk and credit (PHASE 2.3) ----------
            # Only for funds that actually hold bonds. A duration and a
            # credit spread on an equity fund would be arithmetic
            # performed on the wrong instrument.
            #
            # The fund's NAME comes from the bundle: EtfProfile carries a
            # symbol, category, family and legal type but NO name — an
            # assumption that took this panel down until it was checked.
            _bd_name = str((ticker_bundle.info or {}).get("longName")
                           or (ticker_bundle.info or {}).get("shortName") or "")
            _bd_is_bond = (
                bond_screener.looks_like_bond_fund(_bd_name)
                or bond_screener.looks_like_bond_fund(
                    _rk_prof.category if _rk_prof.ok else ""))
            if _bd_is_bond:
                st.markdown("---")
                st.subheader("Bond risk")
                _bd_fund = bond_data.load_bond_fund(ticker_symbol)
                _bd_curve = bond_data.load_curve()
                _bd_price = (ticker_bundle.info or {}).get("regularMarketPrice")
                _bd_raw_yield = (ticker_bundle.info or {}).get("yield")
                _bd_yield = (_bd_raw_yield * 100.0
                             if _bd_raw_yield is not None and _bd_raw_yield < 1
                             else _bd_raw_yield)
                _bd_dur = _bd_fund.empirical_duration if _bd_fund.ok else None

                _bd1, _bd2, _bd3 = st.columns(3)
                if _bd_dur is not None:
                    _bd1.metric("Duration (measured)", f"{_bd_dur:.2f} years",
                                help=bond_data.REPORTED_DURATION_UNUSABLE)
                    _bd1.caption(
                        f"A 100bp rate rise costs about {_bd_dur:.1f}%"
                        + (f" · R² {_bd_fund.duration_r_squared:.2f}"
                           if _bd_fund.duration_r_squared is not None else ""))
                    _bd_var = bond_market.value_at_risk(_bd_price, _bd_dur)
                    if _bd_var is not None:
                        _bd2.metric("1-day VaR (95%)", f"{_bd_var:.2f}%",
                                    help="From the MEASURED daily volatility "
                                         "of the 10-year yield (6.25bp), not "
                                         "an assumed one. A 2% figure is an "
                                         "ANNUAL volatility and overstates a "
                                         "one-day loss about 32-fold.")
                        _bd_dv01 = bond_market.dv01(_bd_price, _bd_dur)
                        if _bd_dv01 is not None:
                            _bd2.caption(
                                f"DV01 {_bd_dv01:.4f} per 100 of face — what "
                                "one basis point costs in money.")
                else:
                    _bd1.metric("Duration (measured)", "Unavailable",
                                help="Needs enough overlapping price history "
                                     "against the 10-year yield.")
                    _bd2.metric("1-day VaR (95%)", "Unavailable",
                                help="Needs a measured duration.")

                _bd_spread = bond_market.credit_spread(
                    ticker_symbol, _bd_yield, _bd_dur, _bd_curve)
                if _bd_spread.ok:
                    _bd3.metric("Credit spread",
                                f"{_bd_spread.spread_bps:+.0f}bp",
                                help="Yield over the treasury curve at this "
                                     "fund's OWN duration, not at a fixed "
                                     "ten years.")
                    _bd3.caption(
                        f"vs {_bd_spread.matched_treasury_pct:.2f}% at "
                        f"{_bd_spread.duration:.1f}y ({_bd_spread.method})")
                else:
                    _bd3.metric("Credit spread", "Unavailable",
                                help="Needs a yield and a positive measured "
                                     "duration inside the loaded curve.")
                st.caption(bond_market.SPREAD_UNDERSTATED)

                _bd_rows = bond_data.rating_rows(_bd_fund) if _bd_fund.ok else []
                if _bd_rows:
                    _bd_table = pd.DataFrame(
                        [{"Rating": label, "Weight %": pct}
                         for label, pct in _bd_rows])
                    _bd_table["Weight %"] = pd.to_numeric(
                        _bd_table["Weight %"], errors="coerce")
                    st.dataframe(
                        _bd_table, width="stretch", hide_index=True,
                        column_config={"Weight %": st.column_config.NumberColumn(
                            "Weight %", format="%.1f%%")},
                        key="bond_ratings_table")
                    _bd_ig = _bd_fund.investment_grade_pct
                    _bd_note = (f"{_bd_ig:.1f}% investment grade (AAA-BBB)."
                                if _bd_ig is not None else "")
                    if _bd_fund.government_pct:
                        _bd_note += (f" {_bd_fund.government_pct:.1f}% is "
                                     "government-issued — a SEPARATE axis "
                                     "from the letters, which sum to 100% on "
                                     "their own.")
                    st.caption(_bd_note)

                _bd_credit = _bd_fund.ok and (_bd_fund.government_pct or 0) < 80
                _bd_stress = bond_market.stress_test(_bd_dur, _bd_credit)
                if _bd_stress:
                    st.markdown("**Historical stress scenarios**")
                    _bd_st = pd.DataFrame([
                        {"Scenario": r.scenario.label,
                         "Move": (f"{r.scenario.shift_bps:+.0f}bp "
                                  f"{r.scenario.kind}"),
                         "Impact %": r.impact_pct,
                         "Note": r.detail}
                        for r in _bd_stress])
                    _bd_st["Impact %"] = pd.to_numeric(_bd_st["Impact %"],
                                                       errors="coerce")
                    st.dataframe(
                        _bd_st, width="stretch", hide_index=True,
                        column_config={"Impact %": st.column_config.NumberColumn(
                            "Impact %", format="%+.2f%%")},
                        key="bond_stress_table")
                    st.caption(
                        "A blank impact means the scenario does not apply: a "
                        "government fund RALLIES when credit spreads blow "
                        "out, so applying a widening to it would invert the "
                        "answer rather than estimate it.")

            # --- historical extremes and sector stress -----------------
            _rk_ext = etf_risk.historical_extremes(df["Close"])
            if _rk_ext.worst_day_pct is not None:
                _rk_worst = (f"Worst single day in the loaded range: "
                             f"{_rk_ext.worst_day_pct:.2f}%"
                             + (f" on {_rk_ext.worst_day_date}"
                                if _rk_ext.worst_day_date else ""))
                if _rk_ext.worst_month_pct is not None:
                    _rk_worst += (f" · worst month "
                                  f"{_rk_ext.worst_month_pct:.2f}%"
                                  f" in {_rk_ext.worst_month_label}")
                st.caption(_rk_worst + ". Historical, not a forecast — but "
                           "it needs no model, because it happened.")

            if _rk_prof.ok and _rk_prof.sector_weights:
                st.markdown("**Sector stress test**")
                _rk_shock = st.slider(
                    "Shock applied to each sector (%)", min_value=-50,
                    max_value=20, value=-30, step=5,
                    help="First-order only: it assumes every other sector "
                         "holds still, which a real rout never does.")
                _rk_rows = etf_risk.sector_stress(
                    _rk_prof.sector_weights, float(_rk_shock),
                    etf_technicals.SECTOR_LABELS)
                _rk_table = pd.DataFrame([
                    {"Sector": r.sector, "Weight %": r.weight_pct,
                     "Fund impact %": r.fund_impact_pct}
                    for r in _rk_rows])
                for _rk_col in ("Weight %", "Fund impact %"):
                    _rk_table[_rk_col] = pd.to_numeric(
                        _rk_table[_rk_col], errors="coerce")
                st.dataframe(
                    _rk_table, width="stretch", hide_index=True,
                    column_config={
                        "Weight %": st.column_config.NumberColumn(
                            "Weight %", format="%.1f%%"),
                        "Fund impact %": st.column_config.NumberColumn(
                            "Fund impact %", format="%+.2f%%")},
                    key="etf_sector_stress")
                _rk_total = etf_risk.total_shock_impact(_rk_rows)
                if _rk_total is not None:
                    st.caption(
                        f"Every sector moving {_rk_shock:+d}% at once would "
                        f"move the fund {_rk_total:+.2f}% — which is the "
                        "shock itself, because the weights sum to the whole "
                        "fund. The value of the table is the per-sector "
                        "split above it.")

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

        gauge_color = risk_score_result.grade_color
        fig_risk_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score_result.score,
            number={'suffix': " / 100"},
            title={'text': f"Composite Risk Score — {risk_score_result.grade}"},
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
        fig_risk_gauge.update_layout(height=280, template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=60, b=20))
        st.caption(chart_help("risk_gauge"))
        st.plotly_chart(fig_risk_gauge, width="stretch")
        if risk_score_result.excluded_factors:
            st.caption(f"Not computable for {ticker_symbol} and excluded from the composite score (rather than counted as a failure): {', '.join(risk_score_result.excluded_factors)}.")

        factor_cols = st.columns(4)
        for i, factor in enumerate(risk_score_result.factors):
            factor_cols[i % 4].metric(
                factor.label,
                factor.value_display,
                delta=f"{factor.status} · weight {factor.weight:.0%}",
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
            help=f'{help_for("volatility_rolling")} Computed as the rolling annualized standard deviation of daily log returns.',
        )
        v2.metric(
            f"Full-Range Annualized Volatility",
            f"{annual_vol * 100:.2f}%" if annual_vol is not None else "N/A",
            help=help_for("volatility_full_range"),
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
                help=f'{help_for("var_historical")} Here: the empirical {1 - var_confidence:.0%} percentile over the last {var_lookback} trading days, with no distributional assumption.',
            )
            var2.metric(
                f"1-Day Parametric VaR ({var_confidence:.0%})",
                f"{parametric_var * 100:.2f}%",
                help=f'{help_for("var_parametric")} Here: the variance-covariance method at {var_confidence:.0%} confidence over the same {var_lookback}-day window.',
            )
            var3.metric(
                f"Expected Shortfall / CVaR ({var_confidence:.0%})",
                f"{expected_shortfall * 100:.2f}%" if expected_shortfall is not None else "N/A",
                help=f'{help_for("cvar")} Increasingly preferred by regulators (Basel III) over VaR alone.',
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
                height=300, template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Daily log return (%)", yaxis_title="Frequency", showlegend=False, margin=dict(t=30, b=30),
            )
            st.caption(f'{chart_help("var_distribution")} Shown over the last {var_lookback} trading days.')
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
            help=f'{help_for("sharpe")} Computed as (annualized return − {risk_free_rate:.2%} risk-free rate) / annualized volatility.',
        )
        r1_c2.metric(
            "Sortino Ratio TTM",
            f"{sortino_ratio:.2f}" if sortino_ratio is not None else "N/A",
            help=help_for("sortino"),
        )
        if sharpe_interpretation:
            st.caption(f"{sharpe_interpretation.explanation} {sharpe_interpretation.limitation}")

        st.markdown("##")
        r2_c1, r2_c2, r2_c3 = st.columns(3)

        r2_c1.metric("Current Price Z-Score", f"{current_z_score:.2f}", help=help_for("price_z_score"))

        hurst_desc = "Random Walk"
        if hurst_exponent < RISK.hurst_mean_reverting_below: hurst_desc = "Mean-Reverting (Statistical Edge)"
        elif hurst_exponent > RISK.hurst_trending_above: hurst_desc = "Strongly Trending"
        r2_c2.metric("Hurst Exponent (H)", f"{hurst_exponent:.2f}", delta=hurst_desc, delta_color="normal" if RISK.hurst_mean_reverting_below <= hurst_exponent <= RISK.hurst_trending_above else "inverse", help=help_for("hurst"))

        r2_c3.metric("Altman Z-Score", f"{altman_z:.2f}" if isinstance(altman_z, float) else "N/A", delta=z_verdict, delta_color="normal" if "Safe" in z_verdict else "inverse", help=help_for("altman_z"))

        st.subheader("Statistical Distance from Mean (Z-Score)")
        st.line_chart(df['Z_Score'])

        st.markdown("##")
        st.subheader("Maximum Drawdown")
        if max_dd_result is not None and max_dd_result.max_drawdown < 0:
            dd1, dd2, dd3 = st.columns(3)
            dd1.metric(
                "Max Drawdown",
                f"{max_dd_result.max_drawdown * 100:.2f}%",
                help=f'{help_for("max_drawdown")} Here: peak ${max_dd_result.peak_price:.2f} on {max_dd_result.peak_date.date()} → trough ${max_dd_result.trough_price:.2f} on {max_dd_result.trough_date.date()}.',
                delta_color="off",
            )
            dd2.metric("Peak → Trough", f"{max_dd_result.peak_date.date()} → {max_dd_result.trough_date.date()}", help=help_for("peak_to_trough"))
            if max_dd_result.recovered:
                dd3.metric("Recovery Period", f"{max_dd_result.recovery_days} trading days", help=f'{help_for("recovery_period")} Recovered by {max_dd_result.recovery_date.date()}.')
            else:
                dd3.metric("Recovery Period", "Ongoing", help=f'{help_for("recovery_period")} Here the price has not yet closed back above the prior peak within the selected range.', delta_color="off")

            drawdown_series = compute_drawdown_series(df['Close']) * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, fill='tozeroy', line=dict(color='#ef4444'), name='Drawdown'))
            fig_dd.add_trace(go.Scatter(x=[max_dd_result.trough_date], y=[max_dd_result.max_drawdown * 100], mode='markers', marker=dict(color='#f59e0b', size=10, symbol='diamond'), name='Trough'))
            fig_dd.update_layout(
                height=300, template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title="Drawdown (%)", showlegend=False, margin=dict(t=20, b=20),
            )
            st.caption(chart_help("drawdown_underwater"))
            st.plotly_chart(fig_dd, width="stretch")

            st.markdown("##")
            calmar1, calmar2 = st.columns(2)
            calmar1.metric(
                "Calmar Ratio",
                f"{calmar_ratio:.2f}" if calmar_ratio is not None else "N/A",
                delta=calmar_interpretation.label if calmar_interpretation else None,
                delta_color="off",
                help=help_for("calmar"),
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
            k1.metric("Daily Win Rate", f"{win_prob * 100:.1f}%", help=f'{help_for("win_rate")} This one counts ALL trading days in the range, not a specific strategy\'s trades.')
            k2.metric("Up/Down-Day Payoff Ratio", f"{win_loss_ratio:.2f}", help=help_for("payoff_ratio"))
            k3.metric("Heuristic Allocation (Half-Kelly)", f"{final_allocation:.2f}%", help=f'{help_for("kelly_half")} Penalized further during high-VIX regimes, and derived from simplified daily statistics rather than a validated edge.')
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
            a1.metric(f"ATR ({atr_length})", f"${current_atr:.2f}", help=help_for("atr"))
            a2.metric(f"Suggested Stop-Loss ({TECHNICAL.atr_stop_multiplier:.0f}× ATR)", f"${stop_loss:.2f}", delta=f"{((stop_loss / standardized.current_price) - 1) * 100:.1f}% below current price", delta_color="off", help=help_for("stop_loss"))
            a3.metric("Risk per Share", f"${standardized.current_price - stop_loss:.2f}", help=help_for("risk_per_share"))
            st.caption(f"Long-only, volatility-adjusted downside stop: current price − {TECHNICAL.atr_stop_multiplier:.0f}×ATR. Not a recommendation to hold a short position or a guarantee against gap-through losses.")
        else:
            st.info(f"ATR ({atr_length}) not yet available — the selected date range doesn't cover enough trading days to complete the warm-up period.")

    with tab_fundamentals:
        # Equity-only. Nothing here has meaning for an instrument with
        # no issuer and no filings, and rendering it as "not reported"
        # would imply the question was sensible and the data merely
        # absent. See asset_class.py.
        if not asset_class.supports(asset_kind, asset_class.DCF):
            st.info(asset_class.unavailable_note(asset_kind, asset_class.DCF))
            for _ac_gap in asset_class.missing_sources(asset_kind):
                st.caption(f"Not sourced in this build: {_ac_gap}.")
        else:
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
                        d1.metric("Market Price", f"${current_price:.2f}", help=help_for("market_price"))
                        d2.metric("Intrinsic Value (2-Stage)", f"${intrinsic_price:.2f}", help=help_for("intrinsic_value"))
                        d3.metric("Margin of Safety", f"{margin_of_safety:.2f}%", delta=dcf_result.status, delta_color=dcf_result.status_color, help=help_for("margin_of_safety"))

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
                        # PLACEHOLDERS ONLY — these keep the names bound for code
                        # further down that would otherwise raise NameError. They are
                        # NOT a valuation. Every consumer must gate on
                        # `dcf_result is not None and dcf_result.ok` before reading
                        # them; rendering these zeroes is how the tear sheet once
                        # told a client Rivian's intrinsic value was $0.00.
                        intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0

                except ZeroDivisionError:
                    # Terminal value is undefined when WACC exactly equals the terminal
                    # growth rate used in the Gordon Growth model.
                    st.error(f"DCF Engine Error: the discount rate (WACC) came out equal to the terminal growth rate ({DCF.terminal_growth_rate*100:.0f}%), which makes the terminal value mathematically undefined. Try adjusting the WACC or growth sliders.")
                    # PLACEHOLDERS ONLY — these keep the names bound for code
                    # further down that would otherwise raise NameError. They are
                    # NOT a valuation. Every consumer must gate on
                    # `dcf_result is not None and dcf_result.ok` before reading
                    # them; rendering these zeroes is how the tear sheet once
                    # told a client Rivian's intrinsic value was $0.00.
                    intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0
                except Exception as e:
                    log_exception(logger, "calc.error", section="dcf_engine", ticker=ticker_symbol)
                    st.error(f"Unexpected DCF Engine error: {type(e).__name__}: {e}")
                    # PLACEHOLDERS ONLY — these keep the names bound for code
                    # further down that would otherwise raise NameError. They are
                    # NOT a valuation. Every consumer must gate on
                    # `dcf_result is not None and dcf_result.ok` before reading
                    # them; rendering these zeroes is how the tear sheet once
                    # told a client Rivian's intrinsic value was $0.00.
                    intrinsic_price, intrinsic_value, margin_of_safety = 0.0, 0.0, 0.0

            # ==========================================
            # SCENARIO MODELING
            # ==========================================
            # Wires hypothetical shocks into the SAME DCF/risk functions above
            # — engine.intrinsic_price() (the identical call the sensitivity
            # heatmap uses) and the unchanged risk_analytics VaR/CVaR/Sharpe/
            # Max Drawdown functions — never a second valuation or risk model.
            st.markdown("---")
            st.header("Scenario Modeling")
            st.caption(
                "Define a hypothetical event and see its before/after impact on THIS ticker's DCF valuation and risk "
                "metrics, computed by re-running the exact same engines above at shocked inputs — not a separate model. "
                "Dividend Cut is deliberately the odd one out: this app's DCF is built on unlevered free cash flow, "
                "which by the Modigliani-Miller theorem doesn't mechanically change with dividend POLICY — so a "
                "dividend cut's real, non-fabricated effect shown below is the lost cash income per share, not an "
                "invented DCF impact, unless you explicitly opt into a discount-rate add-on for it."
            )

            if "scenario_saved" not in st.session_state:
                st.session_state["scenario_saved"] = load_scenarios()

            sc_type = st.radio(
                "Scenario Type", SCENARIO_TYPES, format_func=lambda t: SCENARIO_TYPE_LABELS[t],
                horizontal=True, key="scenario_type_radio",
            )

            # Re-seeded only when the TYPE actually changes (not on every rerun)
            # — same pattern realtime_alerts.py's fundamental-threshold field
            # established, so editing a parameter sticks across reruns instead
            # of being silently reset back to the type's default every time.
            if st.session_state.get("_scenario_seeded_for_type") != sc_type:
                _seed = default_scenario(sc_type)
                st.session_state["scenario_growth_delta"] = _seed.growth_rate_delta * 100
                st.session_state["scenario_discount_delta"] = _seed.discount_rate_delta * 100
                st.session_state["scenario_vol_multiplier"] = _seed.volatility_multiplier
                st.session_state["scenario_mean_shift"] = _seed.mean_return_shift * 100
                st.session_state["scenario_dividend_cut_pct"] = _seed.dividend_cut_pct
                st.session_state["_scenario_seeded_for_type"] = sc_type

            sc_c1, sc_c2, sc_c3 = st.columns(3)
            with sc_c1:
                sc_growth_delta = st.number_input("Growth Rate Shock (pp)", key="scenario_growth_delta", step=0.5, help="Added to the DCF's current growth-rate slider (in percentage points).") / 100
                sc_discount_delta = st.number_input("Discount Rate Shock (pp)", key="scenario_discount_delta", step=0.25, help="Added to the DCF's calculated WACC (in percentage points).") / 100
            with sc_c2:
                sc_vol_multiplier = st.number_input("Volatility Multiplier", key="scenario_vol_multiplier", min_value=0.1, step=0.1, help="Scales daily returns before recomputing VaR/CVaR/Sharpe/Max Drawdown. 1.0 = no change.")
                sc_mean_shift = st.number_input("Daily Return Shift (pp)", key="scenario_mean_shift", step=0.05, help="Added to every daily log-return before recomputing risk metrics (in percentage points per day).") / 100
            with sc_c3:
                sc_dividend_cut = st.number_input("Dividend Cut (%)", key="scenario_dividend_cut_pct", min_value=0.0, max_value=100.0, step=5.0, disabled=(sc_type != "dividend_cut"), help="Only used for the Dividend Cut scenario type.")
                sc_investment = st.number_input("Illustrative Investment ($)", value=SCENARIO_MODELING.default_investment_amount, step=1000.0, key="scenario_investment", help="Applies the DCF's intrinsic-value % change to this dollar amount — illustrative only, not a price forecast.")

            sc_run_col, sc_save_col = st.columns([1, 1])
            with sc_run_col:
                sc_run_clicked = st.button("Run Scenario", type="primary", key="scenario_run_btn")
            with sc_save_col:
                sc_save_clicked = st.button("Save This Scenario", key="scenario_save_btn")

            _sc_definition = ScenarioDefinition(
                name=f"{SCENARIO_TYPE_LABELS[sc_type]} ({ticker_symbol})", scenario_type=sc_type,
                growth_rate_delta=sc_growth_delta, discount_rate_delta=sc_discount_delta,
                volatility_multiplier=sc_vol_multiplier, mean_return_shift=sc_mean_shift,
                dividend_cut_pct=sc_dividend_cut, created_at=datetime.datetime.now().isoformat(timespec="seconds"),
            )

            if sc_save_clicked:
                st.session_state["scenario_saved"] = save_scenario(_sc_definition)
                st.success(f"Saved \"{_sc_definition.name}\".")

            if sc_run_clicked:
                _sc_result, _sc_risk, _sc_dividend = None, None, None
                if dcf_result is not None and dcf_result.ok:
                    _sc_result = run_scenario(
                        fundamentals_engine, df, ticker_bundle.info, dcf_growth, dcf_result.wacc,
                        _sc_definition, confidence_level=var_confidence, lookback=var_lookback, investment_amount=sc_investment,
                    )
                else:
                    # DCF unavailable: still run the risk/dividend legs by
                    # calling the same functions directly rather than routing
                    # through run_scenario(), which requires a base discount
                    # rate to also attempt the DCF leg.
                    st.caption("DCF is unavailable for this ticker (see the DCF section above) — showing risk/dividend impact only.")
                    _sc_risk = apply_risk_scenario(df, _sc_definition, var_confidence, var_lookback)
                    _sc_dividend = dividend_cut_impact(ticker_bundle.info, standardized.current_price, sc_dividend_cut) if sc_type == "dividend_cut" else None

                log_event(logger, logging.INFO, "user.scenario_run", ticker=ticker_symbol, scenario_type=sc_type)

                if _sc_result is not None:
                    if _sc_result.dcf.ok:
                        st.markdown("**DCF Impact**")
                        dc1, dc2, dc3 = st.columns(3)
                        dc1.metric("Base Intrinsic Value", f"${_sc_result.dcf.base_intrinsic_price:.2f}", help=help_for("base_intrinsic_value"))
                        dc2.metric("Shocked Intrinsic Value", f"${_sc_result.dcf.shocked_intrinsic_price:.2f}", delta=f"{_sc_result.dcf.pct_change:+.2f}%", help=help_for("shocked_intrinsic_value"))
                        if _sc_result.implied_portfolio_value_change is not None:
                            dc3.metric(f"Illustrative Impact on ${sc_investment:,.0f}", f"${_sc_result.implied_portfolio_value_change:+,.2f}", help="Applies the intrinsic-value % change to your investment amount — assumes eventual price convergence to intrinsic value, not a forecast.")
                    else:
                        st.warning(f"DCF impact not available: {_sc_result.dcf.reason}")

                    if _sc_result.risk.ok:
                        st.markdown("**Risk Impact**")
                        _sc_risk_rows = pd.DataFrame({
                            "Base": [f"{_sc_result.risk.base_var_pct:.2f}%", f"{_sc_result.risk.base_cvar_pct:.2f}%", f"{_sc_result.risk.base_sharpe:.2f}" if _sc_result.risk.base_sharpe is not None else "N/A", f"{_sc_result.risk.base_max_drawdown_pct:.2f}%"],
                            "Shocked": [f"{_sc_result.risk.shocked_var_pct:.2f}%", f"{_sc_result.risk.shocked_cvar_pct:.2f}%", f"{_sc_result.risk.shocked_sharpe:.2f}" if _sc_result.risk.shocked_sharpe is not None else "N/A", f"{_sc_result.risk.shocked_max_drawdown_pct:.2f}%"],
                        }, index=[f"1-Day VaR ({var_confidence:.0%})", "Expected Shortfall (CVaR)", "Sharpe Ratio", "Max Drawdown"])
                        st.table(_sc_risk_rows)

                    if _sc_result.dividend.applicable:
                        st.markdown("**Dividend Impact**")
                        dv1, dv2, dv3 = st.columns(3)
                        dv1.metric("Annual Dividend / Share", f"${_sc_result.dividend.current_annual_dividend:.2f} → ${_sc_result.dividend.shocked_annual_dividend:.2f}", help=help_for("annual_dividend_share"))
                        dv2.metric("Dividend Yield", f"{_sc_result.dividend.current_yield_pct:.2f}% → {_sc_result.dividend.shocked_yield_pct:.2f}%", help=help_for("dividend_yield"))
                        dv3.metric("Lost Income / Share", f"${_sc_result.dividend.lost_annual_income_per_share:.2f}", help=help_for("lost_income_share"))
                    elif sc_type == "dividend_cut":
                        st.caption(f"Dividend impact not available: {_sc_result.dividend.detail}")
                elif _sc_risk is not None:
                    if _sc_risk.ok:
                        st.markdown("**Risk Impact**")
                        _sc_risk_rows = pd.DataFrame({
                            "Base": [f"{_sc_risk.base_var_pct:.2f}%", f"{_sc_risk.base_cvar_pct:.2f}%", f"{_sc_risk.base_sharpe:.2f}" if _sc_risk.base_sharpe is not None else "N/A", f"{_sc_risk.base_max_drawdown_pct:.2f}%"],
                            "Shocked": [f"{_sc_risk.shocked_var_pct:.2f}%", f"{_sc_risk.shocked_cvar_pct:.2f}%", f"{_sc_risk.shocked_sharpe:.2f}" if _sc_risk.shocked_sharpe is not None else "N/A", f"{_sc_risk.shocked_max_drawdown_pct:.2f}%"],
                        }, index=[f"1-Day VaR ({var_confidence:.0%})", "Expected Shortfall (CVaR)", "Sharpe Ratio", "Max Drawdown"])
                        st.table(_sc_risk_rows)
                    if _sc_dividend is not None and _sc_dividend.applicable:
                        st.markdown("**Dividend Impact**")
                        dv1, dv2, dv3 = st.columns(3)
                        dv1.metric("Annual Dividend / Share", f"${_sc_dividend.current_annual_dividend:.2f} → ${_sc_dividend.shocked_annual_dividend:.2f}", help=help_for("annual_dividend_share"))
                        dv2.metric("Dividend Yield", f"{_sc_dividend.current_yield_pct:.2f}% → {_sc_dividend.shocked_yield_pct:.2f}%", help=help_for("dividend_yield"))
                        dv3.metric("Lost Income / Share", f"${_sc_dividend.lost_annual_income_per_share:.2f}", help=help_for("lost_income_share"))

            if st.session_state["scenario_saved"]:
                with st.expander(f"Saved scenarios ({len(st.session_state['scenario_saved'])})", expanded=False):
                    for _saved in st.session_state["scenario_saved"]:
                        _sv_c1, _sv_c2 = st.columns([5, 1])
                        with _sv_c1:
                            st.caption(
                                f"**{_saved.name}** ({SCENARIO_TYPE_LABELS[_saved.scenario_type]}) — "
                                f"growth {_saved.growth_rate_delta*100:+.1f}pp, discount {_saved.discount_rate_delta*100:+.1f}pp, "
                                f"vol ×{_saved.volatility_multiplier:.1f}, saved {_saved.created_at}"
                            )
                        with _sv_c2:
                            if st.button("✕", key=f"scenario_delete_{_saved.name}", help="Delete this saved scenario"):
                                st.session_state["scenario_saved"] = delete_scenario(_saved.name)
                                st.rerun()

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
    with executive_digest_container.container():
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
                st.markdown("**Top Strengths**")
                if _digest_strengths:
                    for _flag in _digest_strengths:
                        st.markdown(_digest_flag_line(_flag))
                else:
                    st.caption("No standout strengths identified.")
            with _dcol2:
                st.markdown("**Top Concerns**")
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
        bt1.metric("Strategy Return (Gross)", f"{backtest_result.total_strategy_return_pct:.2f}%", delta=f"{backtest_result.total_strategy_return_pct - backtest_result.total_buy_hold_return_pct:.2f}% vs Buy & Hold", help=help_for("strategy_return_gross"))
        bt2.metric("Buy & Hold Baseline", f"{backtest_result.total_buy_hold_return_pct:.2f}%", help=help_for("buy_hold_baseline"))
        bt3.metric("Max Strategy Drawdown", f"{backtest_result.max_drawdown_pct:.2f}%", help="The deepest percentage drop your portfolio would have suffered using this algorithm (gross).", delta_color="inverse")
        bt4.metric("Win Rate", f"{backtest_result.win_rate_pct:.1f}%" if backtest_result.win_rate_pct is not None else "N/A", help="Of the days this strategy held a position, the fraction with a positive return.")
        bt5.metric("Trades", f"{backtest_result.trade_count}", help="Number of distinct times this strategy entered a position over the selected date range.")

        if cost_bps > 0:
            nc1, nc2, nc3 = st.columns(3)
            nc1.metric(
                "Strategy Return (Net of Cost)", f"{backtest_result.total_net_strategy_return_pct:.2f}%",
                delta=f"{backtest_result.total_net_strategy_return_pct - backtest_result.total_strategy_return_pct:.2f}% vs gross",
                delta_color="inverse",
                help=help_for("strategy_return_net"),
            )
            nc2.metric("Net Max Drawdown", f"{backtest_result.net_max_drawdown_pct:.2f}%", delta_color="inverse", help=help_for("max_drawdown"))
            nc3.metric("Total Cost Paid", f"{backtest_result.total_cost_pct:.2f}%", help=help_for("total_cost_paid"))

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Buy_Hold'], name='Buy & Hold', line=dict(color='gray', dash='dot')))
        fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Strategy'], name=f"{active_rule.name} (Gross)", line=dict(color='cyan', width=2)))
        if cost_bps > 0:
            fig_bt.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Net_Strategy'], name=f"{active_rule.name} (Net of Cost)", line=dict(color='orange', width=2, dash='dash')))

        fig_bt.update_layout(
            title="Strategy Equity Curve vs Baseline",
            xaxis_rangeslider_visible=False,
            height=450,
            template=_plotly_template,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        st.caption(chart_help("strategy_equity"))
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
                    help=help_for("oos_return"),
                )
                wf2.metric("Windows", f"{wf_result.window_count}", help=f'{help_for("windows")} Here: {train_days} train / {test_days} test trading days each, rolled forward without overlap.')
                wf3.metric("OOS Max Drawdown", f"{wf_result.max_drawdown_pct:.2f}%", delta_color="inverse", help=help_for("max_drawdown"))
                wf4.metric("OOS Win Rate", f"{wf_result.win_rate_pct:.1f}%" if wf_result.win_rate_pct is not None else "N/A", help=help_for("win_rate"))
                wf5.metric("OOS Trades", f"{wf_result.trade_count}", help=help_for("trades"))

                fig_wf = go.Figure()
                fig_wf.add_trace(go.Scatter(x=backtest_result.df.index, y=backtest_result.df['Cum_Strategy'], name="In-Sample (single pass)", line=dict(color='cyan', dash='dot')))
                fig_wf.add_trace(go.Scatter(x=wf_result.stitched_equity_curve.index, y=wf_result.stitched_equity_curve, name="Walk-Forward (out-of-sample)", line=dict(color='magenta', width=2)))
                fig_wf.update_layout(
                    title="In-Sample vs. Walk-Forward Out-of-Sample Equity",
                    xaxis_rangeslider_visible=False,
                    height=450,
                    template=_plotly_template,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    hovermode="x unified",
                )
                st.caption(chart_help("walk_forward_equity"))
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
            help=help_for("portfolio_return"),
            )
            pb2.metric(
                "Static-Weight Reference", f"{pbt_result.total_buy_hold_return_pct:.2f}%",
                help=help_for("static_weight_reference"),
            )
            pb3.metric("Max Drawdown", f"{pbt_result.max_drawdown_pct:.2f}%", delta_color="inverse", help=help_for("max_drawdown"))
            pb4.metric("Sharpe Ratio", f"{pbt_result.sharpe_ratio:.2f}" if pbt_result.sharpe_ratio is not None else "N/A", help=help_for("sharpe"))
            pb5.metric("Rebalances", f"{len(pbt_result.rebalance_dates)}", help=help_for("rebalances"))

            fig_pbt = go.Figure()
            fig_pbt.add_trace(go.Scatter(x=pbt_result.df.index, y=pbt_result.df['Cum_Buy_Hold'], name='Static Weights (no rebalancing)', line=dict(color='gray', dash='dot')))
            fig_pbt.add_trace(go.Scatter(x=pbt_result.df.index, y=pbt_result.df['Cum_Portfolio'], name=f"Rebalanced Portfolio ({REBALANCE_FREQUENCY_LABELS[pbt_result.rebalance_frequency]})", line=dict(color='cyan', width=2)))
            fig_pbt.update_layout(
                title="Portfolio Equity Curve", xaxis_rangeslider_visible=False, height=420,
                template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
            )
            st.caption(chart_help("portfolio_equity"))
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
                template=_plotly_template,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                hovermode="x"
            )
            st.caption(chart_help("monte_carlo_paths"))
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
                    template=_plotly_template,
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
                    st.caption(chart_help("seasonality_surface"))
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
        # Also serves the Competitive Benchmarking task: rather than build a
        # second, parallel comparison view answering the same question with
        # possibly different numbers, that task's two still-missing pieces
        # (growth/momentum metrics, outperform/laggard flagging — this
        # section already had custom peer selection and dynamic recompute)
        # are added directly here. See competitive_benchmarking.py's module
        # docstring for the full reasoning.
        st.markdown("---")
        st.header("Path 5: Peer Competitor Matrix")
        st.markdown("Comparing relative valuation, growth, and momentum against a peer group you define.")

        # Smart default peers based on common targets to make testing feel seamless
        default_peers = PEER_DEFAULTS.for_ticker(ticker_symbol)

        peer_input = st.text_input(f"Enter up to {COMPETITIVE_BENCHMARKING.max_peers} Competitor Tickers (comma-separated):", default_peers)

        def _competitor_row(t, std):
            return {
                "Ticker": t,
                "P/E Ratio": std.pe_ratio if std.pe_ratio is not None else np.nan,
                "PEG Ratio": std.peg_ratio if std.peg_ratio is not None else np.nan,
                "Price/Book": std.price_to_book if std.price_to_book is not None else np.nan,
                "ROE (%)": (std.return_on_equity * 100) if std.return_on_equity is not None else np.nan,
                "Net Margin (%)": (std.net_margin * 100) if std.net_margin is not None else np.nan,
                "Debt/Equity": std.debt_to_equity if std.debt_to_equity is not None else np.nan,
                "Earnings Growth (%)": (std.earnings_growth * 100) if std.earnings_growth is not None else np.nan,
            }

        def _momentum_pct(price_history: pd.DataFrame):
            """Identical calculation to the Chart Workspace's own Relative
            Strength headline return (Close[-1]/Close[0] - 1) over the
            SAME user-selected date range — not a separately-invented
            lookback window, so a peer's momentum number means the same
            thing the main ticker's own return number means elsewhere on
            this page."""
            if price_history.empty or len(price_history) < 2:
                return None
            return (price_history["Close"].iloc[-1] / price_history["Close"].iloc[0] - 1) * 100

        @st.cache_data(ttl=3600)
        def fetch_competitor_data(target, target_std, peer_string, range_start, range_end):
            peers = [p.strip().upper() for p in peer_string.split(',')][:COMPETITIVE_BENCHMARKING.max_peers]

            # Target ticker's data was already standardized above, so reuse it here
            # instead of re-fetching it from Yahoo Finance a second time. Its price
            # history for momentum is likewise the already-loaded/cleaned `df`.
            target_momentum = _momentum_pct(df)
            metrics = [_competitor_row(target, target_std)]
            peer_rows = [build_peer_metrics(target, target_std, is_target=True, momentum_pct=target_momentum)]

            for t in peers:
                if t == target:
                    continue  # already the target row above — don't double-count it as its own peer
                peer_bundle = load_ticker_bundle(t, deep=False)
                if not peer_bundle.is_valid:
                    st.warning(f"Could not fetch competitor data for '{t}': {'; '.join(peer_bundle.errors)}")
                    continue
                peer_std = standardize_financials(peer_bundle)
                metrics.append(_competitor_row(t, peer_std))
                peer_price_history, _ = load_price_history_only(t, range_start, range_end)
                peer_rows.append(build_peer_metrics(t, peer_std, is_target=False, momentum_pct=_momentum_pct(peer_price_history)))

            return pd.DataFrame(metrics), peer_rows

        with st.spinner("Analyzing sector peers and building relative valuation matrix..."):
            comp_df, peer_metrics_rows = fetch_competitor_data(ticker_symbol, standardized, peer_input, start_date, end_date)

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
                higher_better = ['ROE (%)', 'Net Margin (%)', 'Earnings Growth (%)']

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
                    template=_plotly_template,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=40, r=40, t=40, b=40)
                )

                st.caption(chart_help("peer_radar"))
                st.plotly_chart(fig_radar, width="stretch")

                # --- 4. Relative Performance: outperform/laggard flagging ---
                # Reuses the SAME peer_metrics_rows the matrix/radar above were
                # built from — one comparison, two views of it, never a second
                # independently-fetched dataset that could silently disagree.
                st.subheader("Relative Performance")
                st.caption(
                    f"Each metric is flagged against the GROUP AVERAGE (every ticker below, target included) — "
                    f"Outperform / Laggard once a value sits at least {COMPETITIVE_BENCHMARKING.outperform_threshold_pct:.0f}% "
                    f"away from that average in the favorable direction for that metric; In line otherwise, or n/a. "
                    f"This is a distance threshold, not a statistical test — a peer group this size (2-6 names) is too "
                    f"small for one to mean anything."
                )
                _benchmark_rows = build_benchmark_rows(peer_metrics_rows)
                _bench_table_rows = []
                for _br in _benchmark_rows:
                    _row_dict = {"Ticker": f"{_br.ticker} (Target)" if _br.metrics.is_target else _br.ticker, "Verdict": f"{_br.overall_icon} {_br.overall_verdict}"}
                    for _m in METRICS:
                        _flag = _br.flags[_m.key]
                        if _flag.value is None:
                            _row_dict[_m.label] = f"{_flag.icon} N/A"
                        else:
                            _row_dict[_m.label] = f"{_flag.icon} {_flag.value:.{_m.decimals}f}{_m.unit}"
                    _bench_table_rows.append(_row_dict)
                st.dataframe(pd.DataFrame(_bench_table_rows), hide_index=True, width="stretch")

                # Switch benchmark: reuses the SAME deferred-ticker-switch
                # mechanism the sidebar Watchlist and recent-tickers strip
                # already use (session_state["_pending_ticker"] + st.rerun()),
                # so clicking a peer here is identical to typing it into the
                # sidebar — satisfies "support switching the benchmark stock
                # within the comparison" without a second switching mechanism.
                st.caption("Make a peer the new primary analysis:")
                _switch_cols = st.columns(len(_benchmark_rows))
                for _sw_col, _br in zip(_switch_cols, _benchmark_rows):
                    with _sw_col:
                        if st.button(
                            _br.ticker, key=f"peer_switch_{_br.ticker}", width="stretch",
                            type="primary" if _br.metrics.is_target else "secondary",
                            disabled=_br.metrics.is_target,
                            help="Currently analyzed" if _br.metrics.is_target else f"Switch analysis to {_br.ticker}",
                        ):
                            st.session_state["_pending_ticker"] = _br.ticker
                            log_event(logger, logging.INFO, "user.peer_switch", ticker=_br.ticker)
                            st.rerun()

            else:
                st.warning("Could not fetch sufficient competitor data. Please check the tickers and try again.")

    # ==========================================
    # MATCHING STOCKS (the "AI recommendations" task)
    # ==========================================
    # Ranks how well candidates match STATED CRITERIA. It predicts
    # nothing: ml_pipeline's momentum model measured 0.4792 ROC AUC on
    # held-out data — below a coin flip — so ranking by its probabilities
    # would be noise dressed as intelligence on a screen used to decide
    # where money goes. See recommendations.py.
    with tab_overview:
        st.markdown("---")
        with st.expander("Find stocks matching your criteria", expanded=False):
            st.caption(
                "Tell it what you're looking for and it ranks the tickers Quantix knows "
                "about by how well they fit — using the same scorecard, leverage and risk "
                "figures shown elsewhere in the app. **It does not predict prices** and "
                "makes no buy or sell suggestion; it answers \"what matches what I asked "
                "for\", which is a different and much more answerable question."
            )

            _rc_universe = tuple(dict.fromkeys(
                list(WATCHLIST.tech_basket) + list(WATCHLIST.diversified_basket)
                + [t for entry in _wl_store.lists.values() for t in entry.tickers]
            ))
            _rc_sectors_available = st.session_state.get("_rc_sectors")
            if _rc_sectors_available is None:
                _rc_sectors_available = rc_available_sectors(_rc_universe)
                st.session_state["_rc_sectors"] = _rc_sectors_available

            _rc_a, _rc_b, _rc_c = st.columns(3)
            _rc_risk = _rc_a.selectbox(
                "Risk tolerance", list(RECOMMENDATIONS.risk_profiles),
                index=list(RECOMMENDATIONS.risk_profiles).index(
                    RECOMMENDATIONS.default_risk_profile),
                help="Scales the app's own configured thresholds — the exact numbers are shown below.",
            )
            _rc_val = _rc_b.selectbox(
                "Valuation leaning", list(RECOMMENDATIONS.valuation_preferences),
                help="Value adds P/E and price-to-book ceilings; growth uses PEG instead, so a "
                     "high multiple backed by matching earnings growth isn't excluded.",
            )
            _rc_profitable = _rc_c.checkbox(
                "Require profitability", value=True,
                help="Applies a net-margin floor. Switch off to include companies not yet profitable.",
            )
            _rc_chosen_sectors = st.multiselect(
                "Sectors (leave empty for any)", _rc_sectors_available,
                help="Only sectors actually present in the universe are offered.",
            )

            _rc_prefs = rc_Preferences(
                sectors=tuple(_rc_chosen_sectors), risk_profile=_rc_risk,
                valuation=_rc_val, require_profitable=_rc_profitable,
            )

            # Shown BEFORE any result: a preference whose effect is
            # invisible is indistinguishable from one that does nothing.
            with st.expander("What these preferences mean, exactly", expanded=False):
                st.dataframe(pd.DataFrame([{
                    "Criterion": o.label, "Metric": o.metric,
                    "Threshold": f"{o.operator} {o.threshold:g}",
                } for o in rc_criteria_for(_rc_prefs)]), width="stretch", hide_index=True)
                st.caption(
                    "These are the app's own configured thresholds, scaled by the risk "
                    "profile — not a separate set of numbers. Ceilings tighten and floors "
                    "rise together as the profile gets more conservative."
                )

            if st.button("Find matches", type="primary", key="rc_run"):
                with st.spinner(f"Evaluating {len(_rc_universe)} tickers…"):
                    st.session_state["_rc_results"] = rc_rank(_rc_prefs, _rc_universe)

            _rc_results = st.session_state.get("_rc_results")
            if _rc_results:
                _rc_ranked, _rc_notes = _rc_results
                if _rc_ranked:
                    st.markdown(f"**{len(_rc_ranked)} best matches**")
                    st.caption(
                        "Ordered with an adjustment for how much each match rests on, so a "
                        "company judged on four criteria doesn't outrank one judged on eight "
                        "at a similar rate — which is why a slightly lower percentage can "
                        "appear above a higher one. The count beside each row is the basis."
                    )
                    for _rc_s in _rc_ranked:
                        _rc_head = (
                            f"**{_rc_s.ticker}** · {_rc_s.sector or 'sector unknown'} · "
                            f"matches **{_rc_s.match_pct:.0f}%** "
                            f"({len(_rc_s.matched)} of {len(_rc_s.evaluable)} criteria)"
                        )
                        st.markdown(_rc_head)
                        _rc_bits = []
                        for _rc_o in _rc_s.matched:
                            _rc_bits.append(f"met: {_rc_o.label} {_rc_o.threshold:g} (is {_rc_o.value:.2f})")
                        for _rc_o in _rc_s.missed:
                            _rc_bits.append(f"not met: {_rc_o.label} {_rc_o.threshold:g} (is {_rc_o.value:.2f})")
                        for _rc_o in _rc_s.unavailable:
                            _rc_bits.append(f"not reported: {_rc_o.label}")
                        st.caption("  ·  ".join(_rc_bits))
                else:
                    st.info("Nothing matched those criteria. Loosening the risk profile is usually the quickest way to widen the field.")
                for _rc_n in _rc_notes:
                    st.caption(_rc_n)

            st.markdown("---")
            st.caption(
                "**Not investment advice, and not a prediction.** A high match means a "
                "company fits the thresholds you selected today — nothing more. Quantix "
                "does contain a momentum model, and it is deliberately NOT used here: "
                "measured on held-out data it scored 0.479 ROC AUC against 0.50 for a coin "
                "flip, so ranking by it would be presenting noise as insight. Every figure "
                "above is a published, backward-looking metric you can check yourself "
                "elsewhere in the app."
            )

    # ==========================================
    # NEWS SENTIMENT
    # ==========================================
    # Deliberately NOT wired into the Blueprint Alignment score — that
    # number comes from audited statements, and blending a
    # headline-derived signal into it would quietly degrade something
    # the reader has reason to trust. See news_sentiment.py.
    with tab_overview:
        st.markdown("---")
        with st.expander(f"News Sentiment — {ticker_symbol}", expanded=False):
            _ns_key = f"news_sentiment_{ticker_symbol}"
            if _ns_key not in st.session_state:
                with st.spinner("Reading recent coverage…"):
                    # ticker_bundle.info, not a bare `info` — that name
                    # doesn't exist here, and the company name is what
                    # the relevance filter matches headlines against.
                    st.session_state[_ns_key] = ns_analyse(
                        ticker_symbol, (ticker_bundle.info or {}).get("longName", ""))
            _ns = st.session_state[_ns_key]

            if _ns.has_score:
                _ns_cols = st.columns([1, 1, 2])
                _ns_cols[0].metric(
                    "Sentiment", _ns.label.title(),
                    help=help_for("news_sentiment"),
                )
                _ns_cols[1].metric(
                    "Score", f"{_ns.score:+.2f}",
                    help="Mean compound score across the scored headlines, from -1 to +1.",
                )
                _ns_cols[2].markdown(
                    f"**{_ns.positive}** positive · **{_ns.neutral}** neutral · "
                    f"**{_ns.negative}** negative  \n"
                    f"from {len(_ns.articles)} relevant article(s)"
                )
            else:
                st.info(f"No sentiment score for {ticker_symbol} right now.")

            for _ns_note in _ns.notes:
                st.caption(_ns_note)

            if _ns.articles:
                st.markdown("**Headlines it was computed from**")
                for _ns_a in _ns.articles:
                    _ns_badge = {"positive": "[positive]", "negative": "[negative]",
                                 "neutral": "[neutral]", "unscored": "[unscored]"}[_ns_a.label]
                    _ns_line = f"{_ns_badge} {_rt_md_escape_dollar(_ns_a.title)}"
                    if _ns_a.url:
                        _ns_line = f"{_ns_badge} [{_rt_md_escape_dollar(_ns_a.title)}]({_ns_a.url})"
                    st.markdown(_ns_line)
                    st.caption(
                        f"{_ns_a.provider or 'unknown source'} · {_ns_a.published_display}"
                        + (f" · {_ns_a.score:+.2f}" if _ns_a.score is not None else "")
                    )

            # The headlines are shown ABOVE this on purpose: the honest
            # use of a lexicon score is as a pointer to what to read, so
            # the reader can check it against the actual coverage rather
            # than taking the number on faith.
            st.markdown("---")
            st.caption(ns_accuracy_summary())
            st.caption(
                "This is a word-list score, not a language model. It sums the sentiment of "
                "individual words, so it cannot read clause structure — \"beats estimates but "
                "cuts guidance\" is scored by its words rather than by understanding that the "
                "second half dominates. It also can't detect sarcasm or judge whether news is "
                "already priced in.\n\n"
                "Article selection is imperfect too: a headline has to name the company, which "
                "keeps out stories about competitors, but still lets through pieces that merely "
                "mention it in passing — or that happen to contain the word as someone's "
                "surname. That is why every headline is listed above: if the score looks wrong, "
                "the articles it came from will usually show why. Treat it as a fast read on "
                "tone, not a forecast."
            )

    # ==========================================
    # PORTFOLIO PERFORMANCE
    # ==========================================
    # Ticker-independent content living in a ticker-scoped tab set: the
    # portfolio has nothing to do with the symbol in the sidebar. It sits
    # here because a tab is where someone will look for it, at the cost
    # of being unreachable if the current ticker fails to load entirely.
    with tab_portfolio:
        st.header("Portfolio Performance")
        if "portfolio_store" not in st.session_state:
            st.session_state["portfolio_store"] = pf_load_store()
        _pf_store = st.session_state["portfolio_store"]

        st.caption(
            "Your actual holdings measured against "
            f"**{PORTFOLIO.default_benchmark}**. Each position counts only from its own "
            "purchase date, so the chart shows what you really held rather than "
            "back-projecting today's portfolio onto the past."
        )

        # Switcher. No key= and a computed index=, with the store as the
        # source of truth — a keyed selectbox whose options change can
        # raise when its stored value falls outside the new list, which
        # is exactly what happens the moment a portfolio is renamed or
        # deleted. Same shape as the Active Watchlist control.
        _pf_names = list(pf_portfolio_names(_pf_store))
        _pf_active_idx = _pf_names.index(_pf_store.active) if _pf_store.active in _pf_names else 0
        _pf_switch_col, _pf_count_col = st.columns([2, 3])
        with _pf_switch_col:
            _pf_chosen = st.selectbox("Active portfolio", _pf_names, index=_pf_active_idx)
        with _pf_count_col:
            st.caption(
                f"\n\n{len(_pf_names)} portfolio(s) · "
                f"{len(_pf_store.holdings(_pf_chosen))} position(s) in this one"
            )
        if _pf_chosen != _pf_store.active:
            _pf_store, _pf_switch_err = pf_set_active_portfolio(_pf_store, _pf_chosen)
            if _pf_switch_err:
                st.warning(_pf_switch_err)
            else:
                st.session_state["portfolio_store"] = _pf_store
                pf_save_store(_pf_store)
                log_event(logger, logging.INFO, "user.portfolio_switched")
                st.rerun()

        with st.expander("Manage portfolios", expanded=False):
            st.caption(
                "Separate portfolios for separate people or mandates. Holdings, returns and "
                "the benchmark comparison are computed per portfolio — nothing is pooled."
            )
            _pf_new_col, _pf_rename_col, _pf_delete_col = st.columns(3)

            with _pf_new_col:
                st.markdown("**New**")
                if st.session_state.pop("_pf_clear_new_name", False):
                    st.session_state["pf_new_name"] = ""
                _pf_new_name = st.text_input(
                    "Name", key="pf_new_name", placeholder="e.g. Client A",
                    label_visibility="collapsed")
                if st.button("Create", key="pf_create"):
                    _pf_store, _pf_err = pf_create_portfolio(_pf_store, _pf_new_name)
                    if _pf_err:
                        st.warning(_pf_err)
                    else:
                        st.session_state["portfolio_store"] = _pf_store
                        pf_save_store(_pf_store)
                        st.session_state["_pf_clear_new_name"] = True
                        st.rerun()

            with _pf_rename_col:
                st.markdown("**Rename**")
                _pf_rename_to = st.text_input(
                    "New name", key="pf_rename_to", placeholder=_pf_store.active,
                    label_visibility="collapsed")
                if st.button("Rename", key="pf_rename"):
                    _pf_store, _pf_err = pf_rename_portfolio(
                        _pf_store, _pf_store.active, _pf_rename_to)
                    if _pf_err:
                        st.warning(_pf_err)
                    else:
                        st.session_state["portfolio_store"] = _pf_store
                        pf_save_store(_pf_store)
                        st.rerun()

            with _pf_delete_col:
                st.markdown("**Delete**")
                # Two-step, because deleting takes the holdings with it and
                # there is no undo. A single click that destroys a client's
                # positions is the wrong affordance.
                _pf_confirm = st.checkbox(
                    f"Delete “{_pf_store.active}” and its positions", key="pf_delete_confirm")
                if st.button("Delete", key="pf_delete", disabled=not _pf_confirm):
                    _pf_store, _pf_err = pf_delete_portfolio(_pf_store, _pf_store.active)
                    if _pf_err:
                        st.warning(_pf_err)
                    else:
                        st.session_state["portfolio_store"] = _pf_store
                        pf_save_store(_pf_store)
                        st.session_state["pf_delete_confirm"] = False
                        st.rerun()

        # Read AFTER the switcher above, not before it. Switching reruns,
        # so reading early would also work — but only because of a
        # st.rerun() three blocks away. Reading here makes the invariant
        # local instead of load-bearing at a distance.
        _pf_holdings = _pf_store.holdings()

        if not _pf_holdings:
            st.info(
                "No holdings yet. Add your first position below — ticker, how many shares, "
                "what you paid per share, and when you bought. Nothing is sent anywhere; it's "
                "stored locally like every other setting."
            )
        else:
            with st.spinner("Pricing your portfolio…"):
                _pf_perf = pf_build_performance(_pf_holdings, _pf_price_loader)

            _pf_cols = st.columns(4)
            _pf_cols[0].metric(
                "Market value", _pf_money(_pf_perf.market_value),
                help="Today's value of every position that could be priced.",
            )
            _pf_cols[1].metric(
                "Total gain", _pf_money(_pf_perf.total_gain),
                delta=_pf_pct(
                    (_pf_perf.total_gain / _pf_perf.cost_total * 100)
                    if _pf_perf.cost_total else None),
                help="Market value minus what you paid. Unrealised — nothing is sold.",
            )
            _pf_cols[2].metric(
                "Time-weighted return", _pf_pct(_pf_perf.twr_pct),
                help=(
                    "Return with the timing of your purchases stripped out — the figure it is "
                    "fair to compare against an index, and what fund factsheets quote. "
                    "Adding money never inflates it."
                ),
            )
            _pf_cols[3].metric(
                f"vs {PORTFOLIO.default_benchmark}", _pf_pct(_pf_perf.excess_vs_benchmark_pct),
                delta=_pf_pct(_pf_perf.benchmark_return_pct) + " benchmark"
                      if _pf_perf.benchmark_return_pct is not None else None,
                delta_color="off",
                help=(
                    "Your time-weighted return minus the benchmark's over the same period. "
                    "Positive means your selection beat simply buying the index."
                ),
            )

            if _pf_perf.mwr_pct is not None:
                st.caption(
                    f"Money-weighted return (IRR), annualised: **{_pf_pct(_pf_perf.mwr_pct)}** — "
                    "what your money actually did, including whether you happened to buy at good "
                    "moments. It differs from the time-weighted figure precisely because it "
                    "*does* count timing, which is why the benchmark comparison above doesn't "
                    "use it."
                )

            if not _pf_perf.value_series.empty:
                _pf_fig = go.Figure()
                _pf_fig.add_trace(go.Scatter(
                    x=_pf_perf.value_series.index, y=_pf_perf.value_series.values,
                    name="Portfolio", mode="lines", line=dict(width=2.5),
                ))
                if _pf_perf.benchmark_series is not None:
                    _pf_fig.add_trace(go.Scatter(
                        x=_pf_perf.benchmark_series.index, y=_pf_perf.benchmark_series.values,
                        name=f"{PORTFOLIO.default_benchmark} (rebased)", mode="lines",
                        line=dict(width=1.5, dash="dot"),
                    ))
                # Mark each purchase. Without these the step where money
                # arrived reads as a price spike, and a reader who skips
                # the caption draws exactly the wrong conclusion from the
                # most visually striking feature of the chart.
                #
                # add_shape + add_annotation rather than the tidier
                # add_vline: add_vline computes its annotation position by
                # averaging the axis values, which raises TypeError on a
                # date axis ("unsupported operand type(s) for +: 'int' and
                # 'datetime.date'"). It fails for ISO strings too, so there
                # is no one-liner form that works here.
                for _pf_h in _pf_perf.holdings:
                    if not _pf_h.ok:
                        continue
                    _pf_fig.add_shape(
                        type="line", x0=_pf_h.purchase_date, x1=_pf_h.purchase_date,
                        y0=0, y1=1, yref="paper",
                        line=dict(width=1, dash="dash", color=_chart_faint_line),
                    )
                    # Inside the plot, not above it: the legend already
                    # occupies y=1 in paper coordinates, so an annotation
                    # anchored there collides with it.
                    _pf_fig.add_annotation(
                        x=_pf_h.purchase_date, y=0.02, yref="paper", yanchor="bottom",
                        xanchor="left", text=f" bought {_pf_h.ticker}", showarrow=False,
                        font=dict(size=10, color=_chart_fg), opacity=0.75,
                    )
                _pf_fig.update_layout(
                    template=_plotly_template, height=380,
                    margin=dict(l=10, r=10, t=30, b=10),
                    yaxis_title="Value", xaxis_title=None,
                    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                )
                st.plotly_chart(_pf_fig, width="stretch")
                st.caption(chart_help("portfolio_performance"))

            st.markdown("**Positions**")
            st.dataframe(
                pd.DataFrame([{
                    "Ticker": h.ticker,
                    "Shares": h.shares,
                    "Cost basis": _pf_money(h.cost_basis),
                    "Bought": h.purchase_date.isoformat(),
                    "Price now": _pf_money(h.current_price) if h.ok else "unavailable",
                    "Value": _pf_money(h.market_value) if h.ok else "—",
                    "Gain": _pf_money(h.gain) if h.ok else "—",
                    "Gain %": _pf_pct(h.gain_pct) if h.ok else "—",
                } for h in _pf_perf.holdings]),
                width="stretch", hide_index=True,
            )

            for _pf_note in _pf_perf.notes:
                st.caption(_pf_note)

            _pf_remove_col, _ = st.columns([2, 3])
            with _pf_remove_col:
                _pf_labels = [
                    f"{i + 1}. {h.ticker} — {h.shares:g} @ {_pf_money(h.cost_basis)} ({h.purchase_date})"
                    for i, h in enumerate(_pf_holdings)
                ]
                # Positional, not by ticker: the same symbol can be held as
                # two lots at different cost bases, and matching on ticker
                # would delete the wrong one.
                _pf_choice = st.selectbox("Remove a position", _pf_labels, key="pf_remove_choice")
                if st.button("Remove", key="pf_remove"):
                    _pf_store = pf_remove_holding(_pf_store, _pf_labels.index(_pf_choice))
                    st.session_state["portfolio_store"] = _pf_store
                    pf_save_store(_pf_store)
                    st.rerun()

        st.markdown("---")
        st.markdown("**Add a position**")
        if st.session_state.pop("_pf_clear_form", False):
            st.session_state["pf_ticker"] = ""
        _pf_a, _pf_b, _pf_c, _pf_d = st.columns(4)
        _pf_new_ticker = _pf_a.text_input("Ticker", key="pf_ticker", placeholder="AAPL")
        _pf_new_shares = _pf_b.number_input("Shares", min_value=0.0, step=1.0, key="pf_shares")
        _pf_new_cost = _pf_c.number_input(
            "Cost per share", min_value=0.0, step=1.0, key="pf_cost",
            help="What you actually paid, not today's price.",
        )
        _pf_new_date = _pf_d.date_input(
            "Purchase date", value=datetime.date.today(), key="pf_date",
            max_value=datetime.date.today(),
            help="Each position counts from this date onward, which is what keeps the return honest.",
        )
        if st.button("Add position", type="primary", key="pf_add"):
            _pf_store, _pf_err = pf_add_holding(
                _pf_store, _pf_new_ticker, _pf_new_shares, _pf_new_cost, _pf_new_date)
            if _pf_err:
                st.warning(_pf_err)
            else:
                st.session_state["portfolio_store"] = _pf_store
                pf_save_store(_pf_store)
                st.session_state["_pf_clear_form"] = True
                log_event(logger, logging.INFO, "user.portfolio_holding_added")
                st.rerun()

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
                # Same rule as the tear sheet below: a DCF that could not be
                # computed must not become a $0.00 intrinsic value. It is
                # worse here than in a metric card, because this section
                # states it as prose — "its calculated intrinsic value of
                # $0.00" reads as a finding rather than as missing data.
                _briefing_dcf_ok = dcf_result is not None and dcf_result.ok
                target_v = intrinsic_value if (_briefing_dcf_ok and 'intrinsic_value' in locals()) else None
                mos_val = margin_of_safety if (_briefing_dcf_ok and 'margin_of_safety' in locals()) else None

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
                if mos_val is None:
                    valuation_narrative = (
                        f"No discounted cash flow valuation could be produced for this company, so "
                        f"there is no intrinsic value or margin of safety to compare the market "
                        f"price of ${current_p:.2f} against. This section therefore makes no claim "
                        f"about whether the asset is cheap or expensive."
                    )
                elif mos_val > 0:
                    valuation_narrative = f"The asset displays a positive margin of safety ({mos_val:.1f}%), suggesting it is currently underpriced relative to its intrinsic value of ${target_v:.2f} derived via discounted free cash flows."
                else:
                    valuation_narrative = f"The asset trades at a premium relative to its calculated intrinsic value of ${target_v:.2f}. The current market price of ${current_p:.2f} reflects an baked-in growth premium, narrowing the statistical margin of safety ({mos_val:.1f}%)."

                # Liquidity & Smart Money Pillar
                if insider_own > TEAR_SHEET.high_insider_ownership_pct:
                    ownership_narrative = f"Management interests are fundamentally aligned with shareholders, underscored by a notable {insider_own:.2f}% insider ownership stake. Institutional backing remains robust at {inst_own:.2f}%, indicating deep liquidity and strong sponsorship from major fund complexes."
                else:
                    ownership_narrative = f"The equity profile is highly institutionalized with {inst_own:.2f}% controlled by major asset managers, while structural insider ownership sits at a lean {insider_own:.2f}%. Capital allocation decisions will be heavily policed by external institutional blocks."

                # The entry-window claim is derived from the DCF's projected
                # cash flows. Without a DCF it has nothing behind it, so it is
                # dropped rather than restated with missing inputs.
                _entry_window_sentence = (
                    " If the default projected cash flow compound annual growth rate holds true, "
                    "current price levels represent a calculated entry window for risk-adjusted "
                    "accounts." if mos_val is not None else ""
                )

                # Synthesis Construction
                briefing_text = f"""### **INVESTMENT MEMORANDUM**
                **Ticker Target:** {ticker_symbol} | **Generated:** {pd.Timestamp.now().strftime('%B %d, %Y')}

                ---

                #### **1. Core Valuation & Pricing Discrepancy**
                {valuation_narrative} {peer_text} the asset's structural return profile requires consistent capital efficiency to sustain its trading multiple.{_entry_window_sentence}

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
                fig_corr.update_layout(height=350 + 20 * len(alignment.included_tickers), template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, b=30))
                st.caption(f'{chart_help("correlation_matrix")} Measured over the last {portfolio_lookback} trading days.')
                st.plotly_chart(fig_corr, width="stretch")

                diversification = compute_portfolio_diversification(alignment.returns)
                if diversification is not None:
                    d1, d2, d3 = st.columns(3)
                    d1.metric("Portfolio Volatility", f"{diversification.portfolio_volatility * 100:.2f}%", help=help_for("portfolio_volatility"))
                    d2.metric("Weighted-Average Volatility", f"{diversification.weighted_average_volatility * 100:.2f}%", help=help_for("weighted_avg_volatility"))
                    d3.metric("Diversification Benefit", f"{diversification.diversification_benefit * 100:.2f}pp", help=help_for("diversification_benefit"))
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
                        mode='markers', name='Equal-Weighted (current basket)', marker=dict(color=_chart_fg, size=13, symbol='diamond'),
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
                        height=450, template=_plotly_template, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        hovermode="closest",
                    )
                    st.caption(chart_help("efficient_frontier"))
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
                    help=help_for("model_accuracy"),
                )
                ml_c2.metric("Naive Majority-Class Baseline", f"{_ml_latest['majority_class_baseline_accuracy']*100:.1f}%", help="Accuracy from simply always predicting whichever label (up/down) was more common in the test period — the bar a model has to clear to be adding anything at all.")
                ml_c3.metric("Test ROC-AUC", f"{_ml_latest['test_roc_auc']:.3f}" if _ml_latest.get('test_roc_auc') is not None else "N/A", help=help_for("roc_auc"))

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

        # 1. Safely extract variables.
        #
        # A FAILED DCF MUST NOT RENDER AS $0.00. This block used to fall back
        # to 0.0 for intrinsic value and margin of safety, which printed a
        # fabricated valuation onto the most client-facing artefact this app
        # produces — and then fed that same fabricated 0.0 into the verdict
        # logic below, so a company with no valuation still got classified as
        # though it had one. Negative free cash flow is ordinary rather than
        # exotic, so this fired on real companies, not edge cases.
        # Availability is read off dcf_result exactly the way the Executive
        # Digest reads it (see the dcf_ok argument to collect_flags).
        _dcf_ok = dcf_result is not None and dcf_result.ok
        _intrinsic = intrinsic_price if (_dcf_ok and 'intrinsic_price' in locals()) else None
        _mos = margin_of_safety if (_dcf_ok and 'margin_of_safety' in locals()) else None
        _kelly = final_allocation if 'final_allocation' in locals() else None
        _altman_display = fmt_num(altman_z)

        def _money(value):
            return f"${value:,.2f}" if value is not None else "Not reported"

        def _pct(value, decimals=2):
            return f"{value:.{decimals}f}%" if value is not None else "Not reported"

        def _plain(value, decimals=2):
            return f"{value:.{decimals}f}" if value is not None else "Not reported"

        # Brand identity for the exported sheet. A white-label licensee's
        # report must not carry this app's name out of the building.
        from branding import brand as _brand
        from export_theme import palette as _export_palette
        _brand_now = _brand()
        _brand_name = _brand_now.name
        # The tear sheet is an exported document, so it takes the export
        # palette rather than the viewer's current theme (see export_theme).
        # It is the same palette the PowerPoint deck uses, so a PDF and a
        # deck of the same company look like the same document.
        _ec = _export_palette()
        _accent = _ec.css("accent")
        _ec_bg = _ec.css("background")
        _ec_surface = _ec.css("surface")
        _ec_text = _ec.css("text")
        _ec_strong = _ec.css("text_strong")
        _ec_muted = _ec.css("text_muted")
        _ec_border = _ec.css("border")
        # Lifted for contrast on black — the #16a34a/#dc2626 pair this sheet
        # used is tuned for a white page and reads as mud here.
        _ec_positive = _ec.css("positive")
        _ec_negative = _ec.css("negative")

        # 2. Determine the CIO Verdict
        if macro_risk_flag:
            verdict = "STRONG AVOID"
            verdict_color = "#dc2626"
            reason = f"Systemic market risk (VIX > {effective_risk().vix_high_risk_threshold:.0f}). Capital preservation prioritized over individual asset alpha."
        elif not _dcf_ok:
            # Every branch below compares a margin of safety against a
            # threshold. Without a DCF there is no margin of safety, so the
            # honest answer is to grade the business and say plainly that the
            # price has not been judged — not to grade it against zero.
            verdict = "NO VALUATION"
            verdict_color = "#64748b"
            reason = (
                f"Fundamental checks scored {score_pct:.0f}%, but a discounted cash flow "
                "valuation could not be computed for this company, so there is no margin of "
                "safety to judge the current price against. This assesses the business only."
            )
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
        # "23 Aug 2026", not "August 23, 2026". The masthead is tight now
        # that the logo is legible at 150px, and the long form wrapped onto
        # a second line beside it. The short form is also the unambiguous
        # one internationally, which a document that leaves the building
        # ought to prefer anyway.
        report_date = datetime.date.today().strftime("%d %b %Y")

        # The masthead carries OUR mark, not the analysed company's.
        #
        # It used to fetch the company logo from Clearbit by domain, which
        # was wrong twice over: it made rendering the sheet depend on a
        # third-party service being up and on the machine having outbound
        # network at print time, and it put another firm's trademark at the
        # top of a document Quantix signs. WeasyPrint renders a detached
        # HTML string with no base URL, so the file is inlined as base64
        # rather than referenced by path — a relative src resolves to
        # nothing there. Absent asset simply means no logo.
        # The DARK-ground variant, not the light one the brand brief named
        # for "documents". That instruction assumed a white page; this tear
        # sheet is deliberately black (the export palette — see
        # export_theme), so the white-ground file renders as a white tile
        # in the masthead. Verified by rasterising the PDF and looking at
        # it. Swap to "light" here if the sheet ever goes back to white.
        _ts_logo_uri = brand_assets.data_uri("dark")
        logo_html = (
            f'<img src="{_ts_logo_uri}" alt="{_brand_name}" '
            f'style="height: 150px; object-fit: contain; margin-right: 26px; '
            f'flex: 0 0 auto;">'
            # 150px. The artwork is a SQUARE lockup — mark stacked above
            # the wordmark — so the height is shared between the two and
            # only about a fifth of it is the "QUANTIX / INSTITUTIONAL
            # STOCK ANALYSIS" line. At 58px and again at 82px that line
            # was illegible in print. Chosen by rendering the masthead at
            # 82, 130 and 170 and looking at the pages: 130 is the floor
            # for legibility, 170 makes the logo taller than the ticker
            # block beside it.
            if _ts_logo_uri else ''
        )

        # 3. Build the HTML Template
        tear_sheet_html = f"""
        <div class="tear-sheet">
            <div class="ts-top-accent"></div>
            <div class="ts-header">
                <div style="display: flex; align-items: center; min-width: 0;">
                    {logo_html}
                    <div style="min-width: 0;">
                        <div style="font-size: 0.8rem; font-weight: 700; color: {_accent}; letter-spacing: 2px; margin-bottom: 4px;">POWERED BY {_brand_name.upper()}</div>
                        <h1 style="margin:0; font-size: 2.5rem; color: {_ec_strong}; letter-spacing: -1px;">{ticker_symbol}</h1>
                        <p style="margin:4px 0 0 0; color: {_ec_muted}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.6px;">Institutional Tear Sheet • {report_date}</p>
                    </div>
                </div>
                <div style="text-align: right; flex: 0 0 auto; padding-left: 18px;">
                    <h2 style="margin:0; font-size: 2.2rem; color: {_ec_strong};">${current_price:.2f}</h2>
                    <span style="background-color: {verdict_color}; color: white; padding: 6px 14px; border-radius: 6px; font-weight: 600; font-size: 0.9rem; display: inline-block; margin-top: 8px; letter-spacing: 0.5px; white-space: nowrap;">{verdict}</span>
                </div>
            </div>

            <div class="ts-section">
                <h3 class="ts-title">Chief Investment Officer Thesis</h3>
                <p style="font-size: 1.05rem; color: {_ec_text}; line-height: 1.7;"><strong>Primary Driver:</strong> {reason}</p>
                <p style="font-size: 0.95rem; color: {_ec_muted}; line-height: 1.6; border-left: 3px solid {_accent}; padding-left: 15px; margin-top: 15px;">{(standardized.business_summary or 'Business summary not available.')[:450]}...</p>
            </div>

            <div class="ts-grid">
                <div class="ts-card">
                    <h4>Valuation & DCF</h4>
                    <div class="ts-metric"><span class="ts-label">Intrinsic Value</span> <span class="ts-value">{_money(_intrinsic)}</span></div>
                    <div class="ts-metric"><span class="ts-label">Margin of Safety</span> <span class="ts-value" style="color: {_ec_muted if _mos is None else (_ec_positive if _mos > 0 else _ec_negative)};">{_pct(_mos)}</span></div>
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
                    <div class="ts-metric"><span class="ts-label">Kelly-Style Sizing</span> <span class="ts-value">{_pct(_kelly)}</span></div>
                    <div class="ts-metric"><span class="ts-label">Z-Score (Trend)</span> <span class="ts-value">{_plain(current_z_score)}</span></div>
                    <div class="ts-metric"><span class="ts-label">1-Day VaR ({var_confidence:.0%})</span> <span class="ts-value">{f"{historical_var * 100:.2f}%" if historical_var is not None else "N/A"}</span></div>
                </div>
            </div>

            <div class="ts-footer">
                <p>Generated by <strong>{_brand_name}</strong> | Not investment advice. Produced from public market data for research purposes. Algorithmic execution carries inherent risk; verify all execution parameters via broker.</p>
                <p style="margin-top:6px; font-size:0.78rem; color:{_ec_muted};">Figures shown as &ldquo;Not reported&rdquo; were unavailable at export time. They are never assumed to be zero.</p>
            </div>
        </div>

        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            .tear-sheet {{
                background-color: {_ec_bg};
                color: {_ec_text};
                padding: 40px 50px;
                border-radius: 12px;
                margin-top: 20px;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                box-shadow: 0 10px 30px rgba(0,0,0,0.45);
                position: relative;
                overflow: hidden;
                border: 1px solid {_ec_border};
                /* Without this the background is dropped when printing —
                   browsers strip backgrounds to save ink by default, which
                   would put this document's light text on white paper. */
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .ts-top-accent {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 6px;
                background: linear-gradient(90deg, {_accent}, #0f172a);  /* brand cyan into slate, per the brand spec */
            }}
            .ts-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid {_ec_border};
                padding-bottom: 25px;
                margin-bottom: 25px;
            }}
            .ts-section {{ margin-bottom: 35px; }}
            .ts-title {{
                border-bottom: 2px solid {_ec_border};
                padding-bottom: 10px;
                color: {_ec_strong};
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
                background-color: {_ec_surface};
                padding: 20px;
                border-top: 3px solid {_accent};
                border-radius: 0 0 8px 8px;
                border: 1px solid {_ec_border};
                border-top-width: 3px;
            }}
            .ts-card h4 {{
                margin-top: 0;
                color: {_ec_muted};
                margin-bottom: 15px;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                border-bottom: 1px solid {_ec_border};
                padding-bottom: 8px;
            }}
            .ts-metric {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
                font-size: 0.95rem;
            }}
            .ts-label {{ color: {_ec_muted}; }}
            .ts-value {{ font-weight: 600; color: {_ec_strong}; }}
            .ts-footer {{
                text-align: center;
                font-size: 0.75rem;
                color: {_ec_muted};
                border-top: 1px solid {_ec_border};
                padding-top: 20px;
                letter-spacing: 0.5px;
            }}
            .ts-footer strong {{ color: {_accent}; font-weight: 700; }}

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

        # --- EXPORTS -------------------------------------------------------
        # Three formats because they are for three different jobs, not three
        # skins on one: the PDF is what you send, the deck is what you
        # present and finish editing, the workbook is what you re-model.
        # Each is generated on demand — building all three on every rerun
        # would cost seconds of chart rendering for exports nobody clicked.
        st.markdown("---")
        st.subheader("Export this analysis")
        _export_cols = st.columns(3)

        with _export_cols[0]:
            st.caption("**PDF** — fixed layout, for sending.")
            if st.button("Generate PDF Report", key="_export_pdf_btn"):
                with st.spinner("Rendering PDF..."):
                    pdf_bytes, pdf_error = generate_tear_sheet_pdf(tear_sheet_html)
                if pdf_bytes is not None:
                    st.session_state["_tear_sheet_pdf"] = {"ticker": ticker_symbol, "bytes": pdf_bytes}
                else:
                    st.session_state.pop("_tear_sheet_pdf", None)
                    st.warning(pdf_error)

        with _export_cols[1]:
            st.caption("**PowerPoint** — native editable slides.")
            _deck_ok, _deck_reason = export_deck.is_available()
            if not _deck_ok:
                st.info(_deck_reason)
            elif st.button("Generate PowerPoint", key="_export_pptx_btn"):
                with st.spinner("Building deck..."):
                    _charts = []
                    for _title, _figure in (("Price & Technicals", locals().get("fig")),
                                            ("Risk Profile", locals().get("fig_risk_gauge"))):
                        if _figure is None:
                            continue
                        _png = export_deck.chart_png(_figure)
                        if _png:
                            _charts.append((_title, _png))

                    _deck_data = export_deck.DeckData(
                        ticker=ticker_symbol,
                        company_name=standardized.long_name or "",
                        sector=standardized.sector or "",
                        alignment_verdict=fundamentals.alignment_verdict,
                        alignment_pct=score_pct,
                        # The SAME ranked narrative the Executive Digest shows.
                        strengths=tuple(f.text for f in _digest_strengths),
                        concerns=tuple(f.text for f in _digest_concerns),
                        current_price=current_price,
                        intrinsic_price=_intrinsic,
                        margin_of_safety_pct=_mos,
                        dcf_status=(dcf_result.status if _dcf_ok else ""),
                        dcf_unavailable_reason=(
                            "" if _dcf_ok else
                            (dcf_result.reason if dcf_result is not None and dcf_result.reason
                             else "A discounted cash flow valuation could not be computed.")
                        ),
                        wacc=(dcf_result.wacc if _dcf_ok else None),
                        altman_z=fundamentals.altman_z,
                        altman_verdict=fundamentals.altman_verdict,
                        risk_grade=risk_score_result.grade,
                        # The scorecard, plus the three execution figures the
                        # brand brief names for the metrics slide. VaR and
                        # Kelly are not scorecard checks — they come from the
                        # risk panel — so they are appended rather than
                        # derived, and each is None-safe: an unavailable
                        # figure renders as "not reported", never as 0.00.
                        metrics=tuple(
                            export_deck.Metric(
                                c.label,
                                None if c.value is None else (
                                    c.value * 100 if c.key in ("net_margin",) else c.value),
                                "%" if c.key in ("net_margin", "roic", "fcf_yield") else "",
                            )
                            for c in fundamentals.scorecard_checks
                        ) + (
                            export_deck.Metric(
                                f"1-Day VaR ({var_confidence:.0%})",
                                None if historical_var is None else historical_var * 100,
                                "%",
                            ),
                            export_deck.Metric("Kelly-Style Sizing", _kelly, "%"),
                            export_deck.Metric("Trend Z-Score", current_z_score),
                        ),
                        charts=tuple(_charts),
                    )
                    _deck_bytes, _deck_error = export_deck.build_deck(_deck_data)
                if _deck_bytes is not None:
                    st.session_state["_tear_sheet_deck"] = {
                        "ticker": ticker_symbol, "bytes": _deck_bytes,
                        "charts": len(_charts),
                    }
                else:
                    st.session_state.pop("_tear_sheet_deck", None)
                    st.warning(_deck_error)

        with _export_cols[2]:
            st.caption("**Excel** — real numbers, re-modellable.")
            _wb_ok, _wb_reason = export_workbook.is_available()
            if not _wb_ok:
                st.info(_wb_reason)
            elif st.button("Generate Excel", key="_export_xlsx_btn"):
                with st.spinner("Building workbook..."):
                    _scorecard_rows = tuple(
                        export_workbook.Row(
                            label=c.label,
                            value=None if c.value is None else (
                                c.value * 100 if c.key in ("net_margin",) else c.value),
                            unit="%" if c.key in ("net_margin", "roic", "fcf_yield") else "",
                            percent=c.key in ("net_margin", "roic", "fcf_yield"),
                            detail=(
                                f"{'passes' if c.passed else 'fails' if c.passed is not None else 'not evaluable'}"
                                f" — benchmark: {c.benchmark}"
                            ),
                        )
                        for c in fundamentals.scorecard_checks
                    )
                    _valuation_rows = (
                        export_workbook.Row("Market price", current_price, "USD"),
                        export_workbook.Row("Intrinsic value (2-stage DCF)", _intrinsic, "USD",
                                            detail="" if _dcf_ok else "DCF could not be computed"),
                        export_workbook.Row("Margin of safety", _mos, "%", percent=True),
                        export_workbook.Row("WACC", (dcf_result.wacc * 100) if _dcf_ok else None,
                                            "%", percent=True),
                    )
                    _risk_rows = (
                        export_workbook.Row("Altman Z-Score", fundamentals.altman_z, "",
                                            detail=fundamentals.altman_verdict),
                        export_workbook.Row("Kelly-style position size", _kelly, "%", percent=True),
                        export_workbook.Row("Trend Z-Score", current_z_score, ""),
                    )
                    _wb_data = export_workbook.WorkbookData(
                        ticker=ticker_symbol,
                        company_name=standardized.long_name or "",
                        sector=standardized.sector or "",
                        summary_lines=(
                            f"Verdict: {verdict}",
                            reason,
                            f"Blueprint alignment: {score_pct:.0f}% of evaluable checks passed.",
                        ),
                        sheets=(
                            export_workbook.Sheet(
                                "Scorecard", rows=_scorecard_rows,
                                note="Benchmarks are those in effect at export time; "
                                     "a metric with no value was not reported by the filing."),
                            export_workbook.Sheet("Valuation", rows=_valuation_rows),
                            export_workbook.Sheet("Risk", rows=_risk_rows),
                        ),
                    )
                    _wb_bytes, _wb_error = export_workbook.build_workbook(_wb_data)
                if _wb_bytes is not None:
                    st.session_state["_tear_sheet_workbook"] = {
                        "ticker": ticker_symbol, "bytes": _wb_bytes}
                else:
                    st.session_state.pop("_tear_sheet_workbook", None)
                    st.warning(_wb_error)

        _cached_deck = st.session_state.get("_tear_sheet_deck")
        if _cached_deck and _cached_deck["ticker"] == ticker_symbol:
            st.download_button(
                "Download PowerPoint",
                data=_cached_deck["bytes"],
                file_name=export_deck.filename_for(ticker_symbol),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key="_export_pptx_dl",
            )
            if not _cached_deck["charts"]:
                st.caption(
                    "Chart slides were skipped — rendering Plotly figures to images needs the "
                    "`kaleido` package. The rest of the deck is complete.")

        _cached_wb = st.session_state.get("_tear_sheet_workbook")
        if _cached_wb and _cached_wb["ticker"] == ticker_symbol:
            st.download_button(
                "Download Excel",
                data=_cached_wb["bytes"],
                file_name=export_workbook.filename_for(ticker_symbol),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="_export_xlsx_dl",
            )

        cached_pdf = st.session_state.get("_tear_sheet_pdf")
        if cached_pdf and cached_pdf["ticker"] == ticker_symbol:
            _report_filename = f"{ticker_symbol}_tear_sheet_{datetime.date.today().isoformat()}.pdf"
            st.download_button(
                "Download PDF",
                data=cached_pdf["bytes"],
                file_name=_report_filename,
                mime="application/pdf",
            )

            st.markdown("---")
            st.caption("Email this report")
            if is_email_configured():
                _report_recipient = st.text_input(
                    "Recipient email", key="report_email_recipient", placeholder="client@example.com",
                )
                if st.button("Send Email"):
                    _report_date = datetime.date.today().isoformat()
                    _subject = EMAIL_REPORT.default_subject_template.format(ticker=ticker_symbol, date=_report_date)
                    _body = EMAIL_REPORT.default_body_template.format(ticker=ticker_symbol, date=_report_date)
                    with st.spinner(f"Emailing report to {_report_recipient}..."):
                        _sent, _send_error = send_report_email(
                            _report_recipient, _subject, _body, cached_pdf["bytes"], _report_filename,
                        )
                    if _sent:
                        st.success(f"Report emailed to {_report_recipient}.")
                        log_event(logger, logging.INFO, "user.report_emailed", ticker=ticker_symbol)
                    else:
                        st.warning(_send_error)
            else:
                st.caption(
                    "Not configured on this instance — set [smtp] host/port/username/password/from_address in "
                    ".streamlit/secrets.toml (see .streamlit/secrets.toml.example) to enable emailing reports "
                    "directly from the app. Meanwhile, use Download PDF above and attach it manually."
                )


    # ==========================================
    # FUND COMPARISON  (PHASE 1.6, funds only)
    # ==========================================
    # A ninth tab, APPENDED so every existing tab keeps its index and the
    # ⌘1–⌘8 bindings keep pointing at the panel they always did.
    if tab_comparison is not None:
        with tab_comparison:
            st.header("Compare funds side by side", anchor="fund-comparison")
            st.caption(
                f"Up to {etf_comparison.MAX_FUNDS} funds, {ticker_symbol} "
                "included. Everything below is computed over the analysis "
                "range in the sidebar, so the returns and the Sharpe are "
                "over that window and not the funds' own published periods.")

            _cmp_others = st.text_input(
                "Compare against", key="etf_compare_input",
                placeholder="e.g. QQQ, VTV",
                help="Comma-separated fund tickers.")
            _cmp_extra = [t.strip().upper()
                          for t in (_cmp_others or "").replace(";", ",").split(",")
                          if t.strip()]
            _cmp_symbols = [ticker_symbol]
            for _cmp_sym in _cmp_extra:
                if _cmp_sym not in _cmp_symbols:
                    _cmp_symbols.append(_cmp_sym)
            _cmp_dropped = _cmp_symbols[etf_comparison.MAX_FUNDS:]
            _cmp_symbols = _cmp_symbols[:etf_comparison.MAX_FUNDS]
            if _cmp_dropped:
                st.caption(
                    f"Comparing the first {etf_comparison.MAX_FUNDS}; "
                    f"ignored {', '.join(_cmp_dropped)}.")

            if len(_cmp_symbols) < 2:
                st.info(
                    "Add at least one other fund to compare against — try "
                    + ", ".join(asset_views.view(asset_class.ETF).examples) + ".")
            else:
                _cmp_closes, _cmp_err = etf_comparison.load_prices(
                    tuple(_cmp_symbols), period="1y")
                if _cmp_err:
                    st.warning(_cmp_err)
                _cmp_profiles = {sym: etf_analysis.load_profile(sym)
                                 for sym in _cmp_symbols}
                _cmp_bad = [sym for sym, prof in _cmp_profiles.items()
                            if not prof.ok]
                if _cmp_bad:
                    st.warning(
                        "No fund profile for " + ", ".join(_cmp_bad)
                        + " — those columns will be mostly blank. Check the "
                        "ticker is a fund.")
                _cmp_yields = {}
                for sym in _cmp_symbols:
                    try:
                        _cmp_info = load_ticker_bundle(sym, deep=False).info or {}
                    except Exception:
                        _cmp_info = {}
                    _cmp_yields[sym] = _cmp_info.get("dividendYield")

                _cmp_rows = etf_comparison.build_rows(
                    _cmp_profiles, _cmp_closes, _cmp_yields)

                # A plain table, not st.dataframe: the "best" mark is per
                # ROW and there is no column_config for that.
                _cmp_table = pd.DataFrame(
                    [{"Metric": _r.label,
                      **{sym: etf_comparison.format_value(_r, sym)
                         for sym in _cmp_symbols},
                      "Best": etf_comparison.best_symbol(_r) or "—"}
                     for _r in _cmp_rows])
                st.dataframe(_cmp_table, width="stretch", hide_index=True,
                             key="etf_comparison_grid")
                st.caption(
                    "“Best” is only marked where better and worse are "
                    "actually defined — a fund is not superior for holding "
                    "more assets, and a tie is left unmarked rather than "
                    "broken arbitrarily. Blank means the fund does not "
                    "report that figure.")
                st.caption(etf_comparison.TRACKING_ERROR_UNAVAILABLE)

                # --- performance overlay -----------------------------
                _cmp_rebased = etf_comparison.rebased(_cmp_closes)
                if _cmp_rebased is None:
                    st.info(
                        "Not enough overlapping price history to chart these "
                        "funds together.")
                else:
                    st.markdown("**Performance, rebased to 100**")
                    _cmp_fig = go.Figure()
                    for sym in _cmp_symbols:
                        if sym in _cmp_rebased.columns:
                            _cmp_fig.add_trace(go.Scatter(
                                x=_cmp_rebased.index, y=_cmp_rebased[sym],
                                mode="lines", name=sym))
                    # Same template/transparent-background treatment every
                    # other chart on the page gets, so the overlay follows
                    # the active theme instead of Plotly's default white.
                    _cmp_fig.update_layout(
                        height=380, margin=dict(l=0, r=0, t=10, b=0),
                        yaxis_title="Rebased (100 = start)",
                        template=_plotly_template,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(_cmp_fig, width="stretch")
                    st.caption(chart_help("etf_fund_comparison"))

                # --- holdings overlap --------------------------------
                st.markdown("**Holdings overlap**")
                st.caption(etf_comparison.HOLDINGS_CAVEAT)
                _cmp_base = _cmp_symbols[0]
                for _cmp_other in _cmp_symbols[1:]:
                    _cmp_ov = etf_comparison.holdings_overlap(
                        _cmp_profiles.get(_cmp_base),
                        _cmp_profiles.get(_cmp_other))
                    if not _cmp_ov.ok:
                        st.caption(
                            f"{_cmp_base} vs {_cmp_other}: neither fund "
                            "discloses top holdings (normal for a bond or "
                            "commodity fund).")
                        continue
                    st.caption(
                        f"**{_cmp_base} vs {_cmp_other}** — "
                        f"{len(_cmp_ov.shared)} shared "
                        f"({_cmp_ov.shared_weight_pct:.1f}% of {_cmp_base}'s "
                        f"disclosed top ten): "
                        + (", ".join(_cmp_ov.shared) or "none"))
                    st.caption(
                        f"Only in {_cmp_base}: "
                        + (", ".join(_cmp_ov.only_a) or "none")
                        + f" · Only in {_cmp_other}: "
                        + (", ".join(_cmp_ov.only_b) or "none"))


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
