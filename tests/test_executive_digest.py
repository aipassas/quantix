"""Tests for executive_digest.py — the flag-prioritization synthesis layer.

Pure logic over plain inputs, no network calls. Uses lightweight stand-ins
(SimpleNamespace) for the MetricCheck/RiskFactor/crossover-signal objects
this module reads duck-typed, rather than constructing full real objects
from other modules — this module doesn't care about their concrete types,
only their attributes.
"""
from types import SimpleNamespace

from executive_digest import CONCERN, STRENGTH, collect_flags


def _check(label, passed, display="X", benchmark="> Y"):
    return SimpleNamespace(label=label, passed=passed, display=display, benchmark=benchmark)


def _factor(label, sub_score, value_display="X"):
    return SimpleNamespace(label=label, sub_score=sub_score, value_display=value_display)


def _base_kwargs(**overrides):
    kwargs = dict(
        alignment_verdict="moderate", alignment_score_pct=60.0, scorecard_checks=[],
        company_quality_category="Average Quality", company_quality_score=55.0,
        altman_z=None, altman_verdict="Insufficient Financial Data",
        risk_score=60.0, risk_grade="Moderate Risk", risk_factors=[],
        dcf_ok=False, dcf_status=None, dcf_margin_of_safety_pct=None,
        buy_hold_max_drawdown_pct=-15.0, macro_risk_flag=False, vix_current=None,
        rsi_interpretation=None, recent_sma_signal=None, recent_macd_signal=None,
        recent_bollinger_breakout=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_all_neutral_inputs_produce_no_flags():
    strengths, concerns = collect_flags(**_base_kwargs())
    assert strengths == []
    assert concerns == []


def test_high_alignment_and_safe_altman_and_high_risk_score_are_strengths():
    strengths, concerns = collect_flags(**_base_kwargs(
        alignment_verdict="high", alignment_score_pct=92.0,
        altman_z=5.5, altman_verdict="Safe Zone",
        risk_score=82.0, risk_grade="Low Risk",
    ))
    texts = " ".join(f.text for f in strengths)
    assert "Scorecard" in texts and "Altman Z" in texts and "Composite Risk Score" in texts
    assert concerns == []


def test_low_alignment_and_distress_altman_and_low_risk_score_are_concerns():
    strengths, concerns = collect_flags(**_base_kwargs(
        alignment_verdict="low", alignment_score_pct=20.0,
        altman_z=0.9, altman_verdict="Distress Zone",
        risk_score=25.0, risk_grade="High Risk",
    ))
    texts = " ".join(f.text for f in concerns)
    assert "Scorecard" in texts and "Distress Zone" in texts and "Composite Risk Score" in texts
    assert strengths == []


def test_scorecard_checks_split_by_passed_with_none_skipped():
    strengths, concerns = collect_flags(**_base_kwargs(scorecard_checks=[
        _check("Net Margin", True), _check("Beta", False), _check("PEG Ratio", None),
    ]))
    assert len(strengths) == 1 and "Net Margin" in strengths[0].text
    assert len(concerns) == 1 and "Beta" in concerns[0].text


def test_risk_factors_split_by_subscore_with_middle_band_skipped():
    strengths, concerns = collect_flags(**_base_kwargs(risk_factors=[
        _factor("Sharpe Ratio", 85.0), _factor("Max Drawdown", 20.0), _factor("Volatility", 55.0),
    ]))
    assert len(strengths) == 1 and "Sharpe Ratio" in strengths[0].text
    assert len(concerns) == 1 and "Max Drawdown" in concerns[0].text


def test_dcf_only_flags_when_ok():
    strengths, _ = collect_flags(**_base_kwargs(
        dcf_ok=False, dcf_status="Strong Buy", dcf_margin_of_safety_pct=40.0,
    ))
    assert strengths == []  # dcf_ok=False must suppress the flag even though status/margin are present

    strengths, _ = collect_flags(**_base_kwargs(
        dcf_ok=True, dcf_status="Strong Buy", dcf_margin_of_safety_pct=40.0,
    ))
    assert len(strengths) == 1 and "margin of safety" in strengths[0].text


def test_never_pads_below_three_when_fewer_signals_exist():
    strengths, concerns = collect_flags(**_base_kwargs(
        alignment_verdict="high", alignment_score_pct=80.0,
    ))
    assert len(strengths) == 1
    assert len(concerns) == 0


def test_caps_at_three_and_ranks_severity_first():
    checks = [_check(f"Metric{i}", True) for i in range(5)]  # 5 severity-1 strengths
    strengths, _ = collect_flags(**_base_kwargs(
        alignment_verdict="high", alignment_score_pct=90.0,  # 1 severity-3 strength
        scorecard_checks=checks,
    ))
    assert len(strengths) == 3
    assert strengths[0].severity == 3  # the aggregate alignment flag ranks above the individual checks
    assert all(f.severity <= 3 for f in strengths)


def test_rsi_zones():
    strengths, concerns = collect_flags(**_base_kwargs(
        rsi_interpretation=SimpleNamespace(zone="oversold", value=22.0),
    ))
    assert len(strengths) == 1 and "oversold" in strengths[0].text

    _, concerns = collect_flags(**_base_kwargs(
        rsi_interpretation=SimpleNamespace(zone="overbought", value=81.0),
    ))
    assert len(concerns) == 1 and "overbought" in concerns[0].text


def test_bollinger_breakout_direction_matches_chart_color_convention():
    strengths, _ = collect_flags(**_base_kwargs(
        recent_bollinger_breakout=SimpleNamespace(kind="lower", price=100.0, band_value=102.0),
    ))
    assert len(strengths) == 1

    _, concerns = collect_flags(**_base_kwargs(
        recent_bollinger_breakout=SimpleNamespace(kind="upper", price=150.0, band_value=148.0),
    ))
    assert len(concerns) == 1


def test_every_flag_has_a_nonempty_anchor():
    strengths, concerns = collect_flags(**_base_kwargs(
        alignment_verdict="high", alignment_score_pct=90.0,
        altman_z=5.0, altman_verdict="Safe Zone",
        risk_score=80.0, risk_grade="Low Risk",
        risk_factors=[_factor("Sharpe Ratio", 90.0)],
        dcf_ok=True, dcf_status="Buy", dcf_margin_of_safety_pct=15.0,
        rsi_interpretation=SimpleNamespace(zone="oversold", value=25.0),
    ))
    for flag in strengths + concerns:
        assert flag.anchor
