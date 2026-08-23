"""Tests for the grouped alert-rule builder.

The acceptance criterion is "every trigger is reachable under exactly one
heading" — the kind of thing a hand-written list drifts out of the moment
someone adds a tenth trigger type. These read the module's own tables, so
a new trigger that nobody filed fails the suite rather than quietly
vanishing from the UI.

Also pins the two disclosures that must stay on screen. Collapsing the
long captions into expanders was the other half of this change, and the
sentences that stop someone relying on an alert that will never arrive
are exactly the ones that must not end up behind a click.
"""
import ast
import re
from pathlib import Path

import pytest

import realtime_alerts

FINANCE_PY = Path(__file__).resolve().parent.parent / "finance.py"
SOURCE = FINANCE_PY.read_text()


# --- the grouping is complete and unambiguous ---------------------------------

def test_every_trigger_appears_in_exactly_one_category():
    """A trigger filed nowhere is unreachable in the UI; filed twice, it
    shows up under two headings."""
    seen = []
    for _, triggers in realtime_alerts.TRIGGER_CATEGORIES:
        seen.extend(triggers)

    assert sorted(seen) == sorted(realtime_alerts.ALL_TRIGGER_TYPES)
    assert len(seen) == len(set(seen)), "a trigger is filed under two categories"


def test_every_category_has_at_least_one_trigger():
    """An empty heading is the exact failure this task's brief would have
    produced: it asked for a Fundamental section, and there is no
    fundamental trigger to put in one."""
    for name, triggers in realtime_alerts.TRIGGER_CATEGORIES:
        assert triggers, f"category {name!r} is empty"


def test_no_category_is_called_fundamental():
    """The "fundamental" trigger type reuses risk_alerts.METRICS, which is
    five RISK metrics. A heading called Fundamental holding Altman Z and
    VaR would misfile them."""
    for name in realtime_alerts.CATEGORY_NAMES:
        assert "fundamental" not in name.casefold(), name


def test_the_risk_category_holds_only_risk_metrics():
    import risk_alerts

    risk_triggers = realtime_alerts.triggers_in("Risk thresholds")
    assert risk_triggers == (realtime_alerts.FUNDAMENTAL_TRIGGER_TYPE,)
    # and the metrics behind it really are the risk engine's
    assert {m.key for m in risk_alerts.METRICS} == {
        "risk_score", "altman_z", "historical_var",
        "expected_shortfall", "max_drawdown"}


def test_category_of_round_trips_every_trigger():
    for trigger in realtime_alerts.ALL_TRIGGER_TYPES:
        category = realtime_alerts.category_of(trigger)
        assert trigger in realtime_alerts.triggers_in(category)


def test_an_unknown_trigger_is_not_orphaned():
    """Falls back to a real category rather than returning something the
    UI would then fail to render."""
    assert realtime_alerts.category_of("no_such_trigger") in realtime_alerts.CATEGORY_NAMES


def test_every_trigger_has_a_label():
    for trigger in realtime_alerts.ALL_TRIGGER_TYPES:
        assert realtime_alerts.TRIGGER_LABELS.get(trigger), trigger


# --- the UI wiring ------------------------------------------------------------

def test_the_trigger_selectbox_takes_no_streamlit_key():
    """Its options change with the chosen category, and a stored value
    outside the new list raises on selection — the footgun the screener's
    criteria rows already document."""
    block = SOURCE.split('"Trigger Condition"')[1][:400]
    assert "key=" not in block, "the trigger selectbox must stay a controlled widget"


def test_the_category_selectbox_does_keep_its_key():
    """Its options are fixed, so a key is safe and preserves the choice
    across reruns."""
    block = SOURCE.split('"Alert type"')[1][:400]
    assert 'key="rt_new_category"' in block


# --- the disclosures stay on screen -------------------------------------------

@pytest.mark.parametrize("phrase", [
    "not a push notification",
    "while this tab stays open",
    "closing it stops monitoring",
    "Delivery is in-app only",
])
def test_the_load_bearing_limitations_are_not_behind_a_click(phrase):
    """These sit in the always-visible caption, not the expander. Someone
    who never opens "How these alerts work" must still learn that these
    alerts do not push and do not run with the tab closed."""
    captions = re.findall(r"st\.caption\(\s*((?:[^()]|\([^()]*\))*)\)", SOURCE, re.S)
    assert any(phrase in caption for caption in captions), phrase


def test_the_full_explanation_is_still_reachable():
    assert SOURCE.count('with st.expander("How these alerts work"') == 2


def test_finance_still_parses():
    ast.parse(SOURCE)
