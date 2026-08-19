"""Tests for user_thresholds.py — the editable valuation/risk thresholds.

The consequential tests here are the ones proving an override actually
reaches the scoring engine, since "reflected across scoring, alerts, and
screener" is the acceptance criterion that a store-only test would pass
while the app quietly ignored the user's numbers.
"""
import json

import pytest

from config import RISK, SCORECARD
import user_thresholds as ut
from user_thresholds import (
    EDITABLE,
    EDITABLE_BY_KEY,
    defaults,
    effective_risk,
    effective_scorecard,
    effective_sector_pe,
    effective_values,
    load_overrides,
    load_sector_pe,
    save_overrides,
    save_sector_pe,
)


# --- the whitelist ------------------------------------------------------------

def test_only_pass_fail_lines_are_editable_not_weights_or_anchors():
    """Scope agreed with the user: the thresholds move, the machinery that
    computes a score does not. A weight or risk-score anchor sneaking into
    EDITABLE would let someone make the score incoherent by accident."""
    keys = set(EDITABLE_BY_KEY)
    for forbidden in ("weights", "risk_score_weight_sharpe", "risk_score_var_anchors",
                      "risk_score_sharpe_anchors", "risk_score_weight_altman_z"):
        assert forbidden not in keys


def test_every_spec_bound_contains_its_own_default():
    """A default outside its own min/max would be rejected by load_overrides
    the moment the user saved it unchanged."""
    for spec in EDITABLE:
        d = ut.default_value(spec.key)
        assert spec.minimum <= d <= spec.maximum, f"{spec.key}: default {d} outside [{spec.minimum}, {spec.maximum}]"


def test_defaults_are_read_from_config_not_duplicated():
    """Reset must restore what config actually ships, so the defaults are
    derived rather than re-typed."""
    d = defaults()
    assert d["min_net_margin"] == SCORECARD.min_net_margin
    assert d["max_beta"] == SCORECARD.max_beta
    assert d["altman_grey_zone"] == RISK.altman_grey_zone
    assert d["pe_range_low"] == float(SCORECARD.pe_range[0])
    assert d["pe_range_high"] == float(SCORECARD.pe_range[1])


# --- persistence and validation ----------------------------------------------

def test_no_store_means_shipped_defaults(tmp_path):
    assert load_overrides(tmp_path / "nope.json") == {}
    assert effective_values(tmp_path / "nope.json") == defaults()


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "t.json"
    save_overrides({"min_net_margin": 0.25, "max_beta": 2.0}, path)
    assert load_overrides(path) == {"min_net_margin": 0.25, "max_beta": 2.0}


def test_only_deltas_are_persisted(tmp_path):
    """Storing every value would silently pin today's defaults forever, so a
    later revision to a shipped default would never reach the user."""
    path = tmp_path / "t.json"
    save_overrides({**defaults(), "max_beta": 2.0}, path)
    assert json.loads(path.read_text()) == {"max_beta": 2.0}


def test_corrupt_store_degrades_to_defaults(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{not json")
    assert load_overrides(path) == {}


def test_unknown_keys_are_dropped(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"bogus": 1.0, "max_beta": 2.0}))
    assert load_overrides(path) == {"max_beta": 2.0}


def test_out_of_range_and_non_numeric_values_are_dropped(tmp_path):
    """A hand-edited store must not be able to push a nonsense threshold into
    the scoring engine."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({
        "max_beta": 999.0,            # above its maximum
        "min_net_margin": -1.0,       # below its minimum
        "min_current_ratio": "abc",   # not a number
        "max_debt_to_equity": True,   # bool is not a threshold
        "min_roic_pct": 12.0,         # the one good value
    }))
    assert load_overrides(path) == {"min_roic_pct": 12.0}


# --- the effective config actually changes ------------------------------------

def test_effective_scorecard_applies_overrides(tmp_path):
    path = tmp_path / "t.json"
    save_overrides({"min_net_margin": 0.3, "pe_range_high": 30.0}, path)
    sc = effective_scorecard(path)
    assert sc.min_net_margin == 0.3
    assert sc.pe_range == (float(SCORECARD.pe_range[0]), 30.0)
    assert sc.max_beta == SCORECARD.max_beta  # untouched field keeps its default


def test_effective_risk_applies_overrides(tmp_path):
    path = tmp_path / "t.json"
    save_overrides({"altman_grey_zone": 2.5}, path)
    assert effective_risk(path).altman_grey_zone == 2.5


def test_config_singletons_are_never_mutated(tmp_path):
    """effective_* returns a COPY. Mutating the shared frozen singletons would
    leak one user's thresholds into every module that imported them."""
    path = tmp_path / "t.json"
    before_margin, before_pe = SCORECARD.min_net_margin, SCORECARD.pe_range
    save_overrides({"min_net_margin": 0.9}, path)
    effective_scorecard(path)
    assert SCORECARD.min_net_margin == before_margin
    assert SCORECARD.pe_range == before_pe


# --- per-sector ---------------------------------------------------------------

def test_sector_table_overrides_one_band_and_leaves_others(tmp_path):
    """save_sector_pe takes the COMPLETE table (what the data-editor hands
    back), so retuning one row must leave every other row on its shipped
    band rather than dropping it."""
    path = tmp_path / "t.json"
    full = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    save_sector_pe({**full, "Technology": (20.0, 40.0)}, path)
    table = effective_sector_pe(path)
    assert table["Technology"] == (20.0, 40.0)
    assert table["Utilities"] == tuple(float(x) for x in SCORECARD.sector_pe_ranges["Utilities"])


def test_user_can_add_a_sector_config_does_not_ship(tmp_path):
    path = tmp_path / "t.json"
    assert "Healthcare" not in SCORECARD.sector_pe_ranges
    full = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    save_sector_pe({**full, "Healthcare": (12.0, 30.0)}, path)
    assert effective_sector_pe(path)["Healthcare"] == (12.0, 30.0)


def test_effective_scorecard_resolves_sectors_through_the_user_table(tmp_path):
    """pe_range_for() on the returned object has to see the user's table, or
    the Scorecard would score against the shipped band regardless."""
    path = tmp_path / "t.json"
    full = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    save_sector_pe({**full, "Technology": (20.0, 40.0)}, path)
    sc = effective_scorecard(path)
    assert sc.pe_range_for("Technology") == (20.0, 40.0)
    assert sc.pe_range_for("Industrials") == sc.pe_range  # unlisted falls back


def test_malformed_sector_rows_are_dropped_individually(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"_sector_pe_ranges": {
        "Good": [10.0, 20.0],
        "Inverted": [30.0, 10.0],   # high <= low
        "Negative": [-5.0, 20.0],
        "TooShort": [10.0],
        "": [10.0, 20.0],
    }}))
    assert load_sector_pe(path) == {"Good": (10.0, 20.0)}


def test_saving_sectors_preserves_scalar_overrides(tmp_path):
    """Both live in one file; writing one half must not wipe the other."""
    path = tmp_path / "t.json"
    save_overrides({"max_beta": 2.0}, path)
    full = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    save_sector_pe({**full, "Technology": (20.0, 40.0)}, path)
    assert load_overrides(path) == {"max_beta": 2.0}
    assert load_sector_pe(path) == {"Technology": (20.0, 40.0)}


# --- reaches the scoring engine ------------------------------------------------

def test_scoring_engine_defaults_to_the_effective_thresholds(monkeypatch, tmp_path):
    """The acceptance criterion, tested at the point it could silently fail:
    the engine must pick up a retuned threshold WITHOUT the five call sites
    passing anything, which is what keeps scoring, alerts and the screener
    consistent with each other."""
    path = tmp_path / "t.json"
    save_overrides({"min_net_margin": 0.42}, path)
    monkeypatch.setattr(ut, "_store_path", lambda: path)

    import fundamental_analysis as fa
    monkeypatch.setattr(fa, "effective_scorecard", lambda: ut.effective_scorecard(path))

    engine = fa.FundamentalAnalysisEngine.__new__(fa.FundamentalAnalysisEngine)
    engine._sc = fa.effective_scorecard()
    assert engine._sc.min_net_margin == 0.42


def test_alert_default_for_altman_tracks_the_user_distress_zone(monkeypatch, tmp_path):
    path = tmp_path / "t.json"
    save_overrides({"altman_grey_zone": 2.4}, path)
    import risk_alerts as ra
    monkeypatch.setattr(ra, "effective_risk", lambda: ut.effective_risk(path))
    assert ra.effective_default_threshold("altman_z") == 2.4
    # a metric with no user-editable counterpart keeps its static spec default
    assert ra.effective_default_threshold("historical_var") == -0.05


def test_saving_the_unchanged_sector_table_stores_nothing(tmp_path):
    """Same reasoning as the scalar deltas test: writing back rows identical
    to the shipped table would pin today's bands forever."""
    path = tmp_path / "t.json"
    save_sector_pe({k: tuple(v) for k, v in SCORECARD.sector_pe_ranges.items()}, path)
    assert load_sector_pe(path) == {}
    assert effective_sector_pe(path) == {
        k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()
    }


def test_deleting_a_shipped_sector_sticks(tmp_path):
    """"No row" can't mean "unchanged" and "deleted" at once — a removed
    shipped sector has to be recorded explicitly or its band returns."""
    path = tmp_path / "t.json"
    kept = {k: tuple(v) for k, v in SCORECARD.sector_pe_ranges.items() if k != "Utilities"}
    save_sector_pe(kept, path)
    table = effective_sector_pe(path)
    assert "Utilities" not in table
    assert "Technology" in table


def test_restoring_a_deleted_sector_clears_the_deletion(tmp_path):
    path = tmp_path / "t.json"
    full = {k: tuple(v) for k, v in SCORECARD.sector_pe_ranges.items()}
    save_sector_pe({k: v for k, v in full.items() if k != "Utilities"}, path)
    assert "Utilities" not in effective_sector_pe(path)
    save_sector_pe(full, path)
    assert "Utilities" in effective_sector_pe(path)


# --- cross-field validation ----------------------------------------------------

def _ok_values(**overrides):
    v = defaults()
    v.update(overrides)
    return v


def test_validate_accepts_the_shipped_defaults():
    """The numbers the app ships with must themselves be a valid combination."""
    assert ut.validate(defaults(), {}) == []


def test_validate_rejects_an_inverted_pe_band():
    errs = ut.validate(_ok_values(pe_range_low=80.0, pe_range_high=45.0), {})
    assert any("P/E band" in e for e in errs)


def test_validate_rejects_an_inverted_peg_band():
    errs = ut.validate(_ok_values(peg_range_low=5.0, peg_range_high=2.5), {})
    assert any("PEG band" in e for e in errs)


def test_validate_rejects_a_distress_zone_above_the_safe_zone():
    """Otherwise the grey band between them is impossible to land in."""
    errs = ut.validate(_ok_values(altman_grey_zone=4.0, altman_safe_zone=2.99), {})
    assert any("distress zone" in e.lower() for e in errs)


def test_validate_rejects_moderate_above_high_alignment():
    errs = ut.validate(_ok_values(moderate_alignment_pct=90.0, high_alignment_pct=75.0), {})
    assert any("Moderate alignment" in e for e in errs)


def test_validate_rejects_an_inverted_sector_band_and_names_the_sector():
    errs = ut.validate(defaults(), {"Technology": (60.0, 20.0)})
    assert any("Technology" in e for e in errs)


def test_validate_reports_every_problem_at_once():
    """Fixing one mistake only to be shown the next is a worse experience
    than seeing all of them together."""
    errs = ut.validate(
        _ok_values(pe_range_low=80.0, pe_range_high=45.0, altman_grey_zone=4.0, altman_safe_zone=2.99),
        {"Technology": (60.0, 20.0)},
    )
    assert len(errs) >= 3
