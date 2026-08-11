"""Scenario Modeling — hypothetical discrete-event shocks wired into the
EXISTING DCF and risk-metric engines, not a parallel valuation model.

Every number a scenario changes comes from a real, already-tested
function: fundamental_analysis.FundamentalAnalysisEngine.intrinsic_price()
for valuation, and risk_analytics.compute_historical_var /
compute_expected_shortfall / compute_sharpe_ratio / compute_max_drawdown
for risk. This module's only job is turning a scenario definition into
the SAME two inputs the sensitivity-analysis heatmap already uses
(growth_rate, discount_rate) for valuation, and a shocked return series
for risk — never a second DCF or a second risk formula.

THREE REQUIRED SCENARIO TYPES AND WHY EACH IS MODELED THE WAY IT IS:

  - Recession: a growth-rate cut plus a widened discount rate (both
    already-accepted DCF parameters), plus a volatility multiplier and a
    negative mean-return shift applied to the historical return series
    feeding VaR/CVaR/Sharpe/Max Drawdown. This is the most direct
    mapping — a recession is exactly a growth/discount/volatility event.

  - Sector Multiple Compression: primarily a discount-rate increase. A
    lower market-clearing multiple and a higher required return are the
    same statement in DCF terms — raising the discount rate IS
    compressing the multiple, not an approximation of it.

  - Dividend Cut: the one that needed real thought. This app's DCF is
    built on unlevered free cash flow (NOPAT + D&A + Capex + ΔWorking
    Capital — see intrinsic_price()'s docstring), which by the
    Modigliani-Miller dividend irrelevance theorem does NOT mechanically
    change with a company's dividend POLICY — cutting the dividend
    doesn't touch the cash the business generates, only how much of it
    gets distributed versus retained. Silently wiring a "dividend cut %"
    into the DCF's growth or discount rate would be fabricating a causal
    link this app's own model doesn't actually have. Instead: the real,
    directly-computed, non-fabricated effect (lost dividend income per
    share, before/after yield — see dividend_cut_impact()) is shown on
    its own, and any DCF effect is an OPT-IN, clearly-labeled discount-
    rate add-on representing an assumed market repricing of risk after a
    cut announcement (default 0 — off unless the user deliberately adds
    it), never an automatic, unearned consequence of the cut itself.

RISK SHOCK METHODOLOGY: the historical daily log-return series is scaled
(volatility_multiplier) and shifted (mean_return_shift), then a synthetic
price path is reconstructed by compounding those shocked returns forward
from the same starting price — a standard stress-testing technique, not
a forecast. That synthetic series is fed through the UNCHANGED
risk_analytics functions, so "shocked VaR" is computed by the exact same
formula as ordinary VaR, just on stressed input.

PERSISTENCE: named scenario definitions save/load from a local JSON file
(same atomic-write pattern realtime_alerts.py and ml_pipeline.py already
established for this app's other local stores), so a user-built scenario
can be re-run later without redefining it.
"""
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DCF, SCENARIO_MODELING
from logging_setup import get_logger, log_event, log_exception
from risk_analytics import (
    compute_expected_shortfall,
    compute_historical_var,
    compute_log_returns,
    compute_max_drawdown,
    compute_sharpe_ratio,
)

logger = get_logger("scenario_modeling")

SCENARIO_TYPES: Tuple[str, ...] = ("dividend_cut", "recession", "sector_shift")
SCENARIO_TYPE_LABELS: Dict[str, str] = {
    "dividend_cut": "Dividend Cut",
    "recession": "Recession",
    "sector_shift": "Sector Multiple Compression",
}


@dataclass
class ScenarioDefinition:
    name: str
    scenario_type: str
    growth_rate_delta: float = 0.0        # added to the DCF's current growth-rate input
    discount_rate_delta: float = 0.0      # added to the DCF's current WACC
    volatility_multiplier: float = 1.0    # multiplies daily log-returns before recomputing risk metrics
    mean_return_shift: float = 0.0        # added to daily log-returns (per day, not annualized)
    dividend_cut_pct: float = 0.0         # 0-100; only meaningful for scenario_type == "dividend_cut"
    created_at: str = ""


def default_scenario(scenario_type: str) -> ScenarioDefinition:
    """One sensible, disclosed starting point per required scenario type —
    edited by the user before running, never forced through unedited."""
    now = datetime.datetime.now().isoformat(timespec="seconds")
    if scenario_type == "recession":
        return ScenarioDefinition(
            name="Recession", scenario_type="recession",
            growth_rate_delta=SCENARIO_MODELING.default_growth_rate_delta_recession,
            discount_rate_delta=SCENARIO_MODELING.default_discount_rate_delta_recession,
            volatility_multiplier=SCENARIO_MODELING.default_volatility_multiplier_recession,
            mean_return_shift=SCENARIO_MODELING.default_mean_return_shift_recession,
            created_at=now,
        )
    if scenario_type == "sector_shift":
        return ScenarioDefinition(
            name="Sector Multiple Compression", scenario_type="sector_shift",
            discount_rate_delta=SCENARIO_MODELING.default_discount_rate_delta_sector_shift,
            volatility_multiplier=SCENARIO_MODELING.default_volatility_multiplier_sector_shift,
            created_at=now,
        )
    if scenario_type == "dividend_cut":
        return ScenarioDefinition(
            name="Dividend Cut", scenario_type="dividend_cut",
            dividend_cut_pct=SCENARIO_MODELING.default_dividend_cut_pct,
            discount_rate_delta=SCENARIO_MODELING.default_discount_rate_delta_dividend_cut,
            created_at=now,
        )
    raise ValueError(f"Unknown scenario type: {scenario_type!r}")


# --- DCF impact ------------------------------------------------------------

@dataclass
class DCFScenarioResult:
    ok: bool
    reason: Optional[str] = None
    base_growth_rate: Optional[float] = None
    shocked_growth_rate: Optional[float] = None
    base_discount_rate: Optional[float] = None
    shocked_discount_rate: Optional[float] = None
    base_intrinsic_price: Optional[float] = None
    shocked_intrinsic_price: Optional[float] = None
    pct_change: Optional[float] = None


def apply_dcf_scenario(engine, base_growth_rate: float, base_discount_rate: float, scenario: ScenarioDefinition) -> DCFScenarioResult:
    """Re-runs engine.intrinsic_price() — the SAME function the DCF
    sensitivity heatmap already calls — at the scenario's shocked
    growth/discount rates. `engine` is a
    fundamental_analysis.FundamentalAnalysisEngine already constructed for
    the analyzed ticker (reused, not rebuilt).

    Refuses (ok=False) rather than returning nonsense when the shocked
    discount rate collapses onto or below the terminal growth rate — the
    Gordon Growth terminal-value formula divides by
    (discount_rate - terminal_growth_rate), which is undefined or negative
    there. This is a real edge case a large-enough discount-rate shock can
    actually hit, not a hypothetical guard.
    """
    shocked_growth = base_growth_rate + scenario.growth_rate_delta
    shocked_discount = base_discount_rate + scenario.discount_rate_delta

    if shocked_discount <= DCF.terminal_growth_rate:
        reason = (
            f"shocked discount rate ({shocked_discount*100:.2f}%) would fall at or below the terminal growth rate "
            f"({DCF.terminal_growth_rate*100:.2f}%) — the terminal value formula is undefined there, so this "
            f"scenario's DCF impact can't be computed as configured. Reduce the discount-rate shock."
        )
        return DCFScenarioResult(ok=False, reason=reason, base_growth_rate=base_growth_rate, base_discount_rate=base_discount_rate)

    try:
        base_intrinsic = engine.intrinsic_price(base_growth_rate, base_discount_rate)
        shocked_intrinsic = engine.intrinsic_price(shocked_growth, shocked_discount)
    except Exception as e:
        log_exception(logger, "scenario.dcf_error", section="scenario_modeling")
        return DCFScenarioResult(ok=False, reason=f"DCF recomputation failed: {type(e).__name__}: {e}")

    pct_change = ((shocked_intrinsic - base_intrinsic) / base_intrinsic) * 100 if base_intrinsic else None

    return DCFScenarioResult(
        ok=True, base_growth_rate=base_growth_rate, shocked_growth_rate=shocked_growth,
        base_discount_rate=base_discount_rate, shocked_discount_rate=shocked_discount,
        base_intrinsic_price=base_intrinsic, shocked_intrinsic_price=shocked_intrinsic,
        pct_change=pct_change,
    )


# --- Risk impact -------------------------------------------------------------

@dataclass
class RiskScenarioResult:
    ok: bool
    reason: Optional[str] = None
    base_var_pct: Optional[float] = None
    shocked_var_pct: Optional[float] = None
    base_cvar_pct: Optional[float] = None
    shocked_cvar_pct: Optional[float] = None
    base_sharpe: Optional[float] = None
    shocked_sharpe: Optional[float] = None
    base_max_drawdown_pct: Optional[float] = None
    shocked_max_drawdown_pct: Optional[float] = None


def _shocked_price_series(df: pd.DataFrame, scenario: ScenarioDefinition) -> pd.DataFrame:
    """A synthetic 'Close' series built by scaling+shifting the REAL
    historical daily log-returns, then compounding them forward from the
    same starting price — the standard return-shock stress-test
    technique. Never fabricates a future price path; this reshapes
    HISTORY, which is then run through the same risk functions as usual.

    Includes the untouched FIRST bar (index 0) as the compounding anchor,
    not just the post-return bars — every downstream risk function
    re-derives log-returns from this series via a diff, which always
    produces a leading NaN with nothing to diff the first bar against. Omit
    that anchor bar and the reconstruction silently loses exactly one
    return observation relative to the original series. Caught live: a
    scenario with volatility_multiplier=1.0 and mean_return_shift=0.0 (a
    true no-op) still moved Sharpe from 0.97 to 1.03 before this fix,
    because it was dropping the single oldest real return in the window
    every time. Verified after the fix that a no-op scenario now reproduces
    every metric to floating-point precision, as it must.
    """
    log_returns = compute_log_returns(df).dropna()
    shocked_returns = log_returns * scenario.volatility_multiplier + scenario.mean_return_shift
    start_price = df["Close"].iloc[0]
    compounded = start_price * np.exp(shocked_returns.cumsum())
    shocked_prices = pd.concat([pd.Series([start_price], index=[df.index[0]]), compounded])
    return pd.DataFrame({"Close": shocked_prices})


def apply_risk_scenario(
    df: pd.DataFrame,
    scenario: ScenarioDefinition,
    confidence_level: float,
    lookback: Optional[int] = None,
) -> RiskScenarioResult:
    """Recomputes VaR/CVaR/Sharpe/Max Drawdown on a shocked return series
    using the UNCHANGED risk_analytics functions — see
    _shocked_price_series() for how the shocked series is built."""
    if df.empty or len(df) < 2:
        return RiskScenarioResult(ok=False, reason="not enough price history to compute a risk shock")

    try:
        shocked_df = _shocked_price_series(df, scenario)
    except Exception as e:
        log_exception(logger, "scenario.risk_shock_error", section="scenario_modeling")
        return RiskScenarioResult(ok=False, reason=f"could not build the shocked return series: {type(e).__name__}: {e}")

    base_var = compute_historical_var(df, confidence_level, lookback)
    shocked_var = compute_historical_var(shocked_df, confidence_level, lookback)
    base_cvar = compute_expected_shortfall(df, confidence_level, lookback)
    shocked_cvar = compute_expected_shortfall(shocked_df, confidence_level, lookback)
    base_sharpe = compute_sharpe_ratio(df)
    shocked_sharpe = compute_sharpe_ratio(shocked_df)
    base_dd = compute_max_drawdown(df["Close"])
    shocked_dd = compute_max_drawdown(shocked_df["Close"])

    return RiskScenarioResult(
        ok=True,
        base_var_pct=base_var * 100 if base_var is not None else None,
        shocked_var_pct=shocked_var * 100 if shocked_var is not None else None,
        base_cvar_pct=base_cvar * 100 if base_cvar is not None else None,
        shocked_cvar_pct=shocked_cvar * 100 if shocked_cvar is not None else None,
        base_sharpe=base_sharpe, shocked_sharpe=shocked_sharpe,
        base_max_drawdown_pct=base_dd.max_drawdown * 100 if base_dd is not None else None,
        shocked_max_drawdown_pct=shocked_dd.max_drawdown * 100 if shocked_dd is not None else None,
    )


# --- Dividend Cut: the one real, directly-computed number -----------------

@dataclass
class DividendCutResult:
    applicable: bool
    detail: str = ""
    current_annual_dividend: Optional[float] = None
    shocked_annual_dividend: Optional[float] = None
    lost_annual_income_per_share: Optional[float] = None
    current_yield_pct: Optional[float] = None
    shocked_yield_pct: Optional[float] = None


def dividend_cut_impact(info: dict, current_price: Optional[float], cut_pct: float) -> DividendCutResult:
    """The genuine, non-fabricated effect of a dividend cut: less cash
    income per share. Yield is DERIVED here (dividendRate / price) rather
    than trusting Yahoo's own dividendYield field directly — that field's
    scale has been inconsistent across tickers/versions elsewhere in this
    app's own experience (see financial_standardization.py's debtToEquity
    handling for the same class of issue), and computing it from two
    values already in hand removes the ambiguity entirely.
    """
    dividend_rate = info.get("dividendRate")
    if not dividend_rate or dividend_rate <= 0:
        return DividendCutResult(applicable=False, detail="This ticker does not report a regular dividend — no dividend-cut impact to model.")
    if not current_price or current_price <= 0:
        return DividendCutResult(applicable=False, detail="Current price unavailable — cannot derive dividend yield.")

    shocked_dividend = dividend_rate * (1 - cut_pct / 100)
    lost = dividend_rate - shocked_dividend

    return DividendCutResult(
        applicable=True,
        current_annual_dividend=dividend_rate, shocked_annual_dividend=shocked_dividend,
        lost_annual_income_per_share=lost,
        current_yield_pct=(dividend_rate / current_price) * 100,
        shocked_yield_pct=(shocked_dividend / current_price) * 100,
    )


# --- Orchestration -----------------------------------------------------------

@dataclass
class ScenarioRunResult:
    scenario: ScenarioDefinition
    dcf: DCFScenarioResult
    risk: RiskScenarioResult
    dividend: DividendCutResult
    investment_amount: float
    implied_portfolio_value_change: Optional[float] = None  # illustrative only — see docstring


def run_scenario(
    engine,
    df: pd.DataFrame,
    info: dict,
    base_growth_rate: float,
    base_discount_rate: float,
    scenario: ScenarioDefinition,
    confidence_level: float,
    lookback: Optional[int] = None,
    investment_amount: Optional[float] = None,
) -> ScenarioRunResult:
    """Runs every applicable impact for one scenario and packages a single
    before/after result for the UI. `investment_amount`'s resulting
    `implied_portfolio_value_change` is explicitly illustrative — it
    applies the DCF's intrinsic-value % change to a user-supplied dollar
    amount, which assumes price eventually converges to intrinsic value;
    it is not a trade recommendation or a price forecast."""
    investment_amount = investment_amount if investment_amount is not None else SCENARIO_MODELING.default_investment_amount

    dcf_result = apply_dcf_scenario(engine, base_growth_rate, base_discount_rate, scenario)
    risk_result = apply_risk_scenario(df, scenario, confidence_level, lookback)

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    dividend_result = dividend_cut_impact(info, current_price, scenario.dividend_cut_pct) if scenario.scenario_type == "dividend_cut" else DividendCutResult(applicable=False, detail="Not a dividend-cut scenario.")

    implied_value_change = None
    if dcf_result.ok and dcf_result.pct_change is not None:
        implied_value_change = investment_amount * (dcf_result.pct_change / 100)

    log_event(
        logger, logging.INFO, "scenario.run", scenario_type=scenario.scenario_type, name=scenario.name,
        dcf_ok=dcf_result.ok, risk_ok=risk_result.ok,
    )

    return ScenarioRunResult(
        scenario=scenario, dcf=dcf_result, risk=risk_result, dividend=dividend_result,
        investment_amount=investment_amount, implied_portfolio_value_change=implied_value_change,
    )


# --- Persistence: local file, same pattern as realtime_alerts.py/ml_pipeline.py --

def _store_path() -> Path:
    return Path(__file__).resolve().parent / SCENARIO_MODELING.store_filename


def load_scenarios(path: Optional[Path] = None) -> List[ScenarioDefinition]:
    """Never raises: a missing or corrupt store is treated as empty."""
    path = path or _store_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return [ScenarioDefinition(**s) for s in raw]
    except Exception:
        log_exception(logger, "scenario.store_corrupt", section="scenario_modeling")
        return []


def save_scenarios(scenarios: List[ScenarioDefinition], path: Optional[Path] = None) -> None:
    path = path or _store_path()
    trimmed = scenarios[-SCENARIO_MODELING.max_saved_scenarios:]
    payload = [asdict(s) for s in trimmed]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def save_scenario(scenario: ScenarioDefinition, path: Optional[Path] = None) -> List[ScenarioDefinition]:
    """Adds/updates one named scenario (by name) and persists the whole
    list. Returns the updated list so the caller's session state stays in
    sync with what's on disk."""
    existing = load_scenarios(path)
    existing = [s for s in existing if s.name != scenario.name]
    existing.append(scenario)
    save_scenarios(existing, path)
    return existing


def delete_scenario(name: str, path: Optional[Path] = None) -> List[ScenarioDefinition]:
    remaining = [s for s in load_scenarios(path) if s.name != name]
    save_scenarios(remaining, path)
    return remaining
