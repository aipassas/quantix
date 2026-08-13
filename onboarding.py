"""Onboarding — step content and local persistence for the first-run
guided walkthrough. finance.py renders the actual panel (Next/Back/Skip
buttons, progress indicator); this module only holds what doesn't change
per-render: the step content and whether onboarding has been dismissed.

NATIVE STEP-BY-STEP PANEL, NOT A SPOTLIGHT-STYLE OVERLAY TOUR — a real,
already-proven constraint in this codebase, not a stylistic choice.
st.markdown(..., unsafe_allow_html=True) inserts HTML via innerHTML, and
browsers never execute a <script> tag inserted that way — confirmed
empirically earlier in this app's own development (the Executive Digest's
cross-tab jump links hit this exact wall and fell back to plain text for
the same reason). A spotlight tour that scrolls to and highlights
specific DOM elements needs JavaScript DOM manipulation, which isn't
available here short of building a full bidirectional Streamlit
component — disproportionate to what a first-run walkthrough needs. A
sequential native panel with Next/Back/Skip controls delivers the same
"walk a first-time user through every major module" goal without needing
anything Streamlit can't actually do.

"FIRST-TIME USER" DETECTION: a single local JSON flag (the same
atomic-write, gitignored-local-file pattern every other piece of
cross-restart state in this app already uses — see realtime_alerts.py /
ml_pipeline.py / scenario_modeling.py). Quantix has no accounts, so this
is genuinely "has onboarding been dismissed on this locally-run
instance," not per-visitor tracking — disclosed here and in the UI
rather than implied. "Replay Tutorial" in the sidebar's System tab is the
deliberate way to see it again on the same instance.
"""
import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from config import ONBOARDING
from logging_setup import get_logger, log_exception

logger = get_logger("onboarding")


@dataclass(frozen=True)
class OnboardingStep:
    title: str
    body: str  # markdown, rendered via st.markdown


STEPS: Tuple[OnboardingStep, ...] = (
    OnboardingStep(
        "Welcome to Quantix",
        "Quantix is an institutional-grade research tool for one ticker at a time — fundamentals, technicals, "
        "risk, simulation, and a machine-learning signal, all built from the same underlying data so every "
        "section agrees with every other. This short walkthrough introduces where each capability lives. "
        "You can skip it now and reopen it anytime from the sidebar's System tab.",
    ),
    OnboardingStep(
        "Choosing a Ticker",
        "The sidebar's **Target Configuration** sets the ticker and date range every panel on the page analyzes. "
        "Below it, your **Watchlist** shows live quotes for a small list of tickers you maintain — click one to "
        "switch the whole page to it instantly. A **Recently Viewed** strip appears under the ticker's price "
        "header once you've looked at more than one symbol, for the same one-click switching.",
    ),
    OnboardingStep(
        "Overview",
        "The first tab starts with the **Executive Digest** — an auto-prioritized summary of the strongest "
        "signals from every section below it (not a separate analysis of its own), followed by a **Data "
        "Quality Report** disclosing exactly how complete and fresh the underlying data is, and a **Macro "
        "Regime** snapshot (VIX, 10-Year Treasury) for broader market context.",
    ),
    OnboardingStep(
        "Chart Workspace",
        "An interactive price chart with configurable overlays (moving averages, Bollinger Bands, RSI, MACD, "
        "and more), plus **Relative Strength & Alpha Generation** — how the ticker has performed against a "
        "benchmark you choose, broken down into how much of that came from market exposure versus the stock "
        "itself.",
    ),
    OnboardingStep(
        "Fundamentals & Valuation",
        "Independent validation of every financial-statement metric, a sector-adjusted quality Scorecard, and "
        "a multi-stage **DCF valuation** with its own sensitivity heatmap. **Scenario Modeling** further down "
        "lets you shock that same DCF and risk numbers with a hypothetical dividend cut, recession, or sector "
        "re-rating and see the before/after.",
    ),
    OnboardingStep(
        "Risk & Technicals",
        "Value-at-Risk, Expected Shortfall, Sharpe/Sortino, Kelly Criterion position sizing, and Portfolio "
        "Correlation — every risk figure this app computes, all in one place. Near the bottom, an experimental "
        "**Momentum Continuation** signal shows a trained model's estimated probability the price is higher in "
        "10 trading days, always reported next to a naive baseline so you can judge whether it's actually "
        "adding anything.",
    ),
    OnboardingStep(
        "Simulation & Comparison",
        "**Monte Carlo & Seasonality** projects a range of future outcomes and a decade of monthly seasonal "
        "patterns; the **Portfolio Backtester** further down runs your configured trading strategy across a "
        "weighted, rebalanced multi-ticker basket instead of one ticker at a time. **Smart Money & Peers** adds "
        "institutional/insider flow and a peer comparison that flags which competitors are outperforming or "
        "lagging on valuation, growth, and momentum.",
    ),
    OnboardingStep(
        "Screener & Alerts",
        "Above the main analysis, the **Stock Screener** filters an arbitrary ticker universe against your own "
        "criteria. Below it, **Smart Risk-Aware Alerts** check your watchlist against risk thresholds on "
        "demand, and the **Real-Time Alert Engine** rechecks price, technical, and fundamental rules "
        "automatically every 60 seconds while the tab stays open.",
    ),
    OnboardingStep(
        "You're Ready",
        "One more stop: the **CIO Tear Sheet** tab synthesizes everything above into a single executive "
        "briefing and verdict. That's every major panel — pick a ticker in the sidebar and start exploring. "
        "You can reopen this walkthrough anytime from the sidebar's System tab.",
    ),
)


def _state_path() -> Path:
    return Path(__file__).resolve().parent / ONBOARDING.state_filename


def load_onboarding_state(path: Optional[Path] = None) -> dict:
    """Never raises: a missing or corrupt state file is treated as
    "not yet completed" rather than crashing the app on load."""
    path = path or _state_path()
    if not path.exists():
        return {"completed": False, "completed_at": None, "skipped": False}
    try:
        return json.loads(path.read_text())
    except Exception:
        log_exception(logger, "onboarding.state_corrupt", section="onboarding")
        return {"completed": False, "completed_at": None, "skipped": False}


def mark_onboarding_done(skipped: bool, path: Optional[Path] = None) -> None:
    """Atomic write (temp file + rename), same pattern as every other
    local store in this app."""
    path = path or _state_path()
    payload = {
        "completed": True, "skipped": skipped,
        "completed_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def has_completed_onboarding(path: Optional[Path] = None) -> bool:
    return bool(load_onboarding_state(path).get("completed"))
