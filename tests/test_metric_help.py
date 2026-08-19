"""Tests for metric_help.py — the central glossary behind every metric
tooltip.

The most valuable tests here are the two that read finance.py's source
directly: one asserts every metric on screen actually has a tooltip (the
task's first acceptance criterion, which is otherwise only checkable by
clicking through 88 metrics in a browser), and one asserts no tooltip
references a glossary key that doesn't exist (a typo would otherwise
render as a silently missing tooltip rather than an error).
"""
import ast
from pathlib import Path

import pytest

from config import RISK, TECHNICAL
from metric_help import GLOSSARY, help_for

FINANCE_PY = Path(__file__).resolve().parent.parent / "finance.py"


def _finance_tree():
    return ast.parse(FINANCE_PY.read_text())


def _call_name(node):
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _metric_calls(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _call_name(n).endswith("metric")]


def _help_for_keys(tree):
    return [n.args[0].value for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _call_name(n) == "help_for"
            and n.args and isinstance(n.args[0], ast.Constant)]


# --- the acceptance criteria, checked against finance.py's real source ---------

def test_every_metric_on_screen_has_a_tooltip():
    """Acceptance criterion: "Every advanced metric has an inline tooltip
    explanation." Encoded here so a metric added later without a help=
    fails the suite instead of quietly shipping bare."""
    bare = []
    for call in _metric_calls(_finance_tree()):
        if not any(k.arg == "help" for k in call.keywords):
            label = (call.args[0].value if call.args and isinstance(call.args[0], ast.Constant)
                     else "<dynamic label>")
            bare.append(f"finance.py:{call.lineno} {label}")
    assert bare == [], "st.metric calls with no help= tooltip:\n  " + "\n  ".join(bare)


def test_finance_only_references_glossary_keys_that_exist():
    """A typo in help_for("sharp") would raise at render time on whichever
    panel happens to contain it — caught here instead."""
    unknown = sorted({k for k in _help_for_keys(_finance_tree()) if k not in GLOSSARY})
    assert unknown == [], f"finance.py references unknown glossary keys: {unknown}"


def test_glossary_has_no_dead_entries():
    """Every definition written here should actually reach a user."""
    used = set(_help_for_keys(_finance_tree()))
    unused = sorted(set(GLOSSARY) - used)
    assert unused == [], f"glossary entries nothing references: {unused}"


def test_the_metrics_the_task_named_explicitly_are_covered():
    """The originating task called out Hurst exponent and Sortino ratio by
    name as the motivating examples, plus VaR/CVaR/Altman Z."""
    for key in ("hurst", "sortino", "var_historical", "cvar", "altman_z"):
        assert key in GLOSSARY, f"{key} missing from the glossary"


# --- house style, so tone stays consistent across panels ----------------------

def test_every_definition_is_a_nonempty_sentence():
    for key, text in GLOSSARY.items():
        assert text.strip(), f"{key}: empty"
        assert text[0].isupper(), f"{key}: does not start with a capital — {text[:40]!r}"
        assert text.rstrip().endswith("."), f"{key}: does not end with a full stop — {text[-40:]!r}"


def test_definitions_are_short_enough_for_a_tooltip():
    """"Short, plain-language definitions" per the task. Long enough to say
    something useful, short enough to read in a hover."""
    for key, text in GLOSSARY.items():
        assert 40 <= len(text) <= 340, f"{key}: {len(text)} chars — {text[:60]!r}"


def test_definitions_have_no_double_spaces_or_stray_whitespace():
    for key, text in GLOSSARY.items():
        assert "  " not in text, f"{key}: contains a double space"
        assert text == text.strip(), f"{key}: has leading/trailing whitespace"


def test_advanced_metrics_say_which_direction_is_good():
    """House rule 2: a bare number tells a non-finance reader nothing
    without knowing how to read it. Every risk-adjusted-return and
    loss metric must orient the reader."""
    orienting = ("higher", "lower", "closer to zero", "above", "below", "larger", "positive", "negative")
    for key in ("sharpe", "sortino", "calmar", "max_drawdown", "var_historical",
                "hurst", "altman_z", "diversification_benefit", "roc_auc"):
        text = GLOSSARY[key].lower()
        assert any(w in text for w in orienting), f"{key}: never says which direction is good"


# --- thresholds must track config, not drift from it --------------------------

def test_threshold_bearing_definitions_quote_the_configured_values():
    """These are interpolated from config on purpose: a tooltip claiming a
    distress zone below 1.81 while the app classifies against a different
    number would be worse than no tooltip at all."""
    assert str(RISK.altman_safe_zone) in GLOSSARY["altman_z"]
    assert str(RISK.altman_grey_zone) in GLOSSARY["altman_z"]
    assert str(RISK.hurst_trending_above) in GLOSSARY["hurst"]
    assert str(RISK.hurst_mean_reverting_below) in GLOSSARY["hurst"]
    assert f"{TECHNICAL.rsi_overbought:.0f}" in GLOSSARY["rsi"]
    assert f"{TECHNICAL.rsi_oversold:.0f}" in GLOSSARY["rsi"]


# --- accessor contract --------------------------------------------------------

def test_help_for_returns_the_glossary_text():
    assert help_for("sharpe") == GLOSSARY["sharpe"]


def test_help_for_raises_on_unknown_key_rather_than_returning_a_placeholder():
    """Failing loudly beats rendering a metric with no tooltip, which is
    exactly the gap this module exists to close."""
    with pytest.raises(KeyError):
        help_for("not_a_real_metric")
