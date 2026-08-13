"""Tests for onboarding.py — step content and local persistence for the
first-run walkthrough.
"""
from onboarding import (
    STEPS,
    has_completed_onboarding,
    load_onboarding_state,
    mark_onboarding_done,
)


# --- STEPS content -------------------------------------------------------------

def test_steps_is_nonempty_and_short():
    """"Short step-by-step walkthrough" per the task — enough steps to
    cover every major module, few enough to actually be short."""
    assert 5 <= len(STEPS) <= 12


def test_every_step_has_a_title_and_nonempty_body():
    for step in STEPS:
        assert step.title.strip()
        assert step.body.strip()


def test_steps_introduce_every_major_module_named_in_the_task():
    """The originating task named these five explicitly: fundamentals,
    technicals, risk, screener, alerts. Every one must appear somewhere
    in the walkthrough — checked case-insensitively across all step text,
    not tied to one specific step, since some modules are introduced
    together."""
    all_text = " ".join(f"{s.title} {s.body}" for s in STEPS).lower()
    for required in ["fundamental", "technical", "risk", "screener", "alert"]:
        assert required in all_text, f"'{required}' not mentioned anywhere in the walkthrough"


def test_first_step_is_a_welcome():
    assert "welcome" in STEPS[0].title.lower()


# --- persistence -----------------------------------------------------------------

def test_fresh_instance_has_not_completed_onboarding(tmp_path):
    path = tmp_path / "state.json"
    assert has_completed_onboarding(path) is False


def test_mark_onboarding_done_persists_completion(tmp_path):
    path = tmp_path / "state.json"
    mark_onboarding_done(skipped=False, path=path)
    assert has_completed_onboarding(path) is True


def test_mark_onboarding_done_records_whether_skipped(tmp_path):
    path = tmp_path / "state.json"
    mark_onboarding_done(skipped=True, path=path)
    state = load_onboarding_state(path)
    assert state["completed"] is True
    assert state["skipped"] is True


def test_finishing_normally_records_skipped_false(tmp_path):
    path = tmp_path / "state.json"
    mark_onboarding_done(skipped=False, path=path)
    state = load_onboarding_state(path)
    assert state["skipped"] is False


def test_load_state_missing_file_returns_not_completed(tmp_path):
    state = load_onboarding_state(tmp_path / "nope.json")
    assert state["completed"] is False


def test_load_state_corrupt_file_degrades_to_not_completed(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not valid json")
    state = load_onboarding_state(path)
    assert state["completed"] is False


def test_mark_onboarding_done_writes_atomically_no_tmp_left_behind(tmp_path):
    path = tmp_path / "state.json"
    mark_onboarding_done(skipped=False, path=path)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_completed_at_is_recorded(tmp_path):
    path = tmp_path / "state.json"
    mark_onboarding_done(skipped=False, path=path)
    state = load_onboarding_state(path)
    assert state["completed_at"]  # non-empty timestamp
