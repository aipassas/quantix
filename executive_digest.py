"""Executive Digest — the top 3 strengths and top 3 concerns across the
ENTIRE analysis, auto-prioritized from every 🟢/🔴 signal already computed
elsewhere in the app (Scorecard checks, Company Quality classification,
Altman Z zone, the Composite Risk Score and its 8 factors, DCF margin of
safety, buy-and-hold Max Drawdown, VIX macro regime, and RSI/SMA/MACD/
Bollinger state).

Pure synthesis, per the task's own technical requirement: this module
performs zero indicator/statement/risk math of its own. Every threshold
used to classify a flag as a strength or a concern is the SAME threshold
already used to color that exact signal 🟢/🟡/🔴 elsewhere in finance.py
(e.g. a Risk Score factor's 70/40 sub-score cutoffs are risk_analytics.py's
own _factor_icon() boundaries, not a new scale invented here) — see each
branch's comment in collect_flags() for the specific source.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

STRENGTH = "strength"
CONCERN = "concern"


@dataclass(frozen=True)
class Flag:
    direction: str   # STRENGTH or CONCERN
    severity: int     # 3 = holistic/aggregate signal, 2 = component signal, 1 = individual checklist item — ranking only, never shown to the user
    category: str      # "Fundamentals" | "Risk" | "Valuation" | "Technical" | "Macro"
    text: str            # one-line plain-English explanation, standalone (no source-section context assumed)
    anchor: str            # matches the `anchor=` kwarg on the source st.header() call, for the jump-to-source link


def _select_top(flags: List[Flag], n: int = 3) -> List[Flag]:
    """Highest severity first; ties keep their original (insertion) order.
    Returns fewer than `n` when fewer than `n` genuine signals exist in
    that direction — never pads with a fabricated flag to hit a quota."""
    return sorted(flags, key=lambda f: -f.severity)[:n]


def collect_flags(
    *,
    alignment_verdict: str,
    alignment_score_pct: float,
    scorecard_checks: Sequence,          # MetricCheck-like: .label, .passed, .display, .benchmark
    company_quality_category: str,
    company_quality_score: Optional[float],
    altman_z: Optional[float],
    altman_verdict: str,
    risk_score: Optional[float],
    risk_grade: Optional[str],
    risk_factors: Sequence,               # RiskFactor-like: .label, .sub_score, .value_display
    dcf_ok: bool,
    dcf_status: Optional[str],
    dcf_margin_of_safety_pct: Optional[float],
    buy_hold_max_drawdown_pct: Optional[float],
    macro_risk_flag: bool,
    vix_current: Optional[float],
    rsi_interpretation,                    # RSIInterpretation-like: .zone, .value or None
    recent_sma_signal,                      # SMACrossoverSignal-like: .kind, .price, .sma_period or None
    recent_macd_signal,                     # MACDCrossoverSignal-like: .kind or None
    recent_bollinger_breakout,              # BollingerBreakout-like: .kind, .price, .band_value or None
) -> Tuple[List[Flag], List[Flag]]:
    """Every candidate flag from every already-computed signal, then the
    top 3 strengths and top 3 concerns. Returns (strengths, concerns),
    each already ranked and capped — nothing further for the caller to
    rank. An ambiguous/neutral reading (e.g. a "Moderate Risk" grade, an
    RSI in neither zone) contributes no flag at all in either direction,
    exactly like the underlying 🟡 signal it comes from."""
    strengths: List[Flag] = []
    concerns: List[Flag] = []

    # --- Scorecard (fundamental_analysis.py's Strategic Investment Scorecard) ---
    if alignment_verdict == "high":
        strengths.append(Flag(STRENGTH, 3, "Fundamentals", f"High Scorecard alignment — passes major Blueprint filters ({alignment_score_pct:.0f}% weighted score).", "scorecard"))
    elif alignment_verdict == "low":
        concerns.append(Flag(CONCERN, 3, "Fundamentals", f"Low Scorecard alignment — fails several Blueprint filters ({alignment_score_pct:.0f}% weighted score).", "scorecard"))

    for check in scorecard_checks:
        if check.passed is None:
            continue
        text = f"{check.label}: {check.display} ({'meets' if check.passed else 'misses'} the Blueprint benchmark of {check.benchmark})."
        (strengths if check.passed else concerns).append(Flag(STRENGTH if check.passed else CONCERN, 1, "Fundamentals", text, "scorecard"))

    # --- Company Quality Classification ---
    if company_quality_score is not None:
        if company_quality_category in ("Elite Quality", "High Quality"):
            strengths.append(Flag(STRENGTH, 3, "Fundamentals", f"{company_quality_category} classification ({company_quality_score:.0f}/100 across Profitability, Stability, Growth, Valuation, and Capital Efficiency).", "quality-classification"))
        elif company_quality_category in ("Weak Quality", "Below Average"):
            concerns.append(Flag(CONCERN, 3, "Fundamentals", f"{company_quality_category} classification ({company_quality_score:.0f}/100 across Profitability, Stability, Growth, Valuation, and Capital Efficiency).", "quality-classification"))

    # --- Altman Z-Score zone (config.RISK.altman_safe_zone / altman_grey_zone) ---
    if altman_z is not None:
        if altman_verdict == "Safe Zone":
            strengths.append(Flag(STRENGTH, 3, "Risk", f"Altman Z-Score of {altman_z:.2f} is in the Safe Zone — low near-term bankruptcy risk by this model.", "risk-dashboard"))
        elif altman_verdict == "Distress Zone":
            concerns.append(Flag(CONCERN, 3, "Risk", f"Altman Z-Score of {altman_z:.2f} is in the Distress Zone — elevated bankruptcy risk by this model.", "risk-dashboard"))

    # --- Composite Risk Score (risk_analytics.py's own _risk_grade() 75/50 boundaries) ---
    if risk_score is not None:
        if risk_score >= 75:
            strengths.append(Flag(STRENGTH, 3, "Risk", f"Composite Risk Score of {risk_score:.0f}/100 ({risk_grade}) — favorable across volatility, tail-risk, and drawdown factors.", "risk-dashboard"))
        elif risk_score < 50:
            concerns.append(Flag(CONCERN, 3, "Risk", f"Composite Risk Score of {risk_score:.0f}/100 ({risk_grade}) — weak across volatility, tail-risk, and drawdown factors.", "risk-dashboard"))

    # --- Individual Risk Score factors (risk_analytics.py's own _factor_icon() 70/40 boundaries) ---
    for factor in risk_factors:
        if factor.sub_score is None:
            continue
        if factor.sub_score >= 70:
            strengths.append(Flag(STRENGTH, 2, "Risk", f"{factor.label}: {factor.value_display} — a strong factor in the Composite Risk Score.", "risk-dashboard"))
        elif factor.sub_score < 40:
            concerns.append(Flag(CONCERN, 2, "Risk", f"{factor.label}: {factor.value_display} — a weak factor in the Composite Risk Score.", "risk-dashboard"))

    # --- DCF margin of safety verdict (fundamental_analysis.py's DCFResult.status) ---
    if dcf_ok and dcf_status is not None and dcf_margin_of_safety_pct is not None:
        if dcf_status in ("Strong Buy", "Buy"):
            strengths.append(Flag(STRENGTH, 2, "Valuation", f"DCF model shows a {dcf_margin_of_safety_pct:.0f}% margin of safety ({dcf_status}) — trading below its estimated intrinsic value.", "dcf"))
        elif dcf_status == "Overvalued":
            concerns.append(Flag(CONCERN, 2, "Valuation", f"DCF model shows a {dcf_margin_of_safety_pct:.0f}% margin of safety ({dcf_status}) — trading above its estimated intrinsic value.", "dcf"))

    # --- Buy-and-hold Max Drawdown over the selected period ---
    if buy_hold_max_drawdown_pct is not None:
        if buy_hold_max_drawdown_pct <= -30:
            concerns.append(Flag(CONCERN, 2, "Risk", f"Suffered a {buy_hold_max_drawdown_pct:.0f}% Max Drawdown over the selected period — a significant historical decline.", "risk-dashboard"))
        elif buy_hold_max_drawdown_pct >= -10:
            strengths.append(Flag(STRENGTH, 2, "Risk", f"Shallow {buy_hold_max_drawdown_pct:.0f}% Max Drawdown over the selected period — relatively stable price history.", "risk-dashboard"))

    # --- Macro regime (VIX, config.RISK.vix_high_risk_threshold) ---
    if macro_risk_flag and vix_current is not None:
        concerns.append(Flag(CONCERN, 1, "Macro", f"VIX at {vix_current:.1f} signals a high-fear market regime — broad conditions, not specific to this ticker.", "macro-regime"))

    # --- RSI zone (technical_indicators.py's interpret_rsi(), config.TECHNICAL.rsi_overbought/oversold) ---
    if rsi_interpretation is not None:
        if rsi_interpretation.zone == "oversold":
            strengths.append(Flag(STRENGTH, 2, "Technical", f"RSI at {rsi_interpretation.value:.1f} is oversold — a potential mean-reversion entry signal.", "technicals"))
        elif rsi_interpretation.zone == "overbought":
            concerns.append(Flag(CONCERN, 2, "Technical", f"RSI at {rsi_interpretation.value:.1f} is overbought — a potential mean-reversion exhaustion signal.", "technicals"))

    # --- Most recent SMA/price crossover, if within the caller's recency window ---
    if recent_sma_signal is not None:
        if recent_sma_signal.kind == "bullish":
            strengths.append(Flag(STRENGTH, 1, "Technical", f"Recent bullish SMA {recent_sma_signal.sma_period} crossover — price moved above its trend line.", "technicals"))
        else:
            concerns.append(Flag(CONCERN, 1, "Technical", f"Recent bearish SMA {recent_sma_signal.sma_period} crossover — price moved below its trend line.", "technicals"))

    # --- Most recent MACD crossover, if within the caller's recency window ---
    if recent_macd_signal is not None:
        if recent_macd_signal.kind == "bullish":
            strengths.append(Flag(STRENGTH, 1, "Technical", "Recent bullish MACD crossover — momentum turning positive.", "technicals"))
        else:
            concerns.append(Flag(CONCERN, 1, "Technical", "Recent bearish MACD crossover — momentum turning negative.", "technicals"))

    # --- Most recent Bollinger Band breakout, if within the caller's recency window
    # (same directional convention as the chart's own breakout markers: lower-band = green/strength, upper-band = red/concern) ---
    if recent_bollinger_breakout is not None:
        if recent_bollinger_breakout.kind == "lower":
            strengths.append(Flag(STRENGTH, 1, "Technical", f"Recent close broke below the lower Bollinger Band (${recent_bollinger_breakout.price:.2f}) — a potential oversold bounce setup.", "technicals"))
        else:
            concerns.append(Flag(CONCERN, 1, "Technical", f"Recent close broke above the upper Bollinger Band (${recent_bollinger_breakout.price:.2f}) — a potential overextension.", "technicals"))

    return _select_top(strengths), _select_top(concerns)
