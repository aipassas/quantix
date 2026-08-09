"""Tests for fundamental_analysis.py's Altman Z-Score (altman_z_score()).

Uses a lightweight types.SimpleNamespace stand-in for StandardizedFinancials
rather than the full dataclass — altman_z_score() only reads a handful of
attributes off self.std, so a namespace with just those attributes set is
enough to exercise it in isolation without constructing an entire
standardized-financials object.
"""
import types

import pytest

from fundamental_analysis import FundamentalAnalysisEngine
from config import RISK


def make_engine(**overrides):
    defaults = dict(
        ticker="TEST", total_assets=1000.0, current_assets=400.0,
        current_liabilities=250.0, ebit=120.0, total_revenue=900.0,
        market_cap=1500.0, total_liabilities=600.0, retained_earnings=200.0,
    )
    defaults.update(overrides)
    std = types.SimpleNamespace(**defaults)
    return FundamentalAnalysisEngine(std, raw_info={})


def test_altman_z_matches_hand_worked_formula():
    z, verdict, missing = make_engine().altman_z_score()

    x1 = (400.0 - 250.0) / 1000.0
    x2 = 200.0 / 1000.0
    x3 = 120.0 / 1000.0
    x4 = 1500.0 / 600.0
    x5 = 900.0 / 1000.0
    expected_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5

    assert z == pytest.approx(expected_z, abs=1e-9)
    assert missing == []
    assert "Safe" in verdict


@pytest.mark.parametrize("market_cap,expected_zone", [
    (3000, "Safe"),
    (600, "Grey"),
    (50, "Distress"),
])
def test_altman_z_zone_thresholds(market_cap, expected_zone):
    z, verdict, _ = make_engine(market_cap=market_cap).altman_z_score()
    if z > RISK.altman_safe_zone:
        assert expected_zone == "Safe"
    elif z >= RISK.altman_grey_zone:
        assert expected_zone == "Grey"
    else:
        assert expected_zone == "Distress"
    assert expected_zone in verdict


def test_altman_z_missing_required_field():
    z, verdict, missing = make_engine(ebit=None).altman_z_score()
    assert z is None
    assert "EBIT / Operating Income" in missing


def test_altman_z_zero_total_assets_produces_descriptive_reason():
    """Regression test: this used to return an EMPTY missing-fields list for
    a present-but-zero denominator, which meant finance.py's warning UI
    (only rendered when the list is non-empty) silently showed nothing at
    all. A zero/negative denominator must always populate a descriptive
    reason, same as a genuinely missing field."""
    z, verdict, missing = make_engine(total_assets=0.0).altman_z_score()
    assert z is None
    assert missing != [], "regression: missing list must not be empty for a zero-denominator case"
    assert any("Total Assets" in m for m in missing)


def test_altman_z_negative_total_liabilities_is_caught():
    z, verdict, missing = make_engine(total_liabilities=-50.0).altman_z_score()
    assert z is None
    assert any("Total Liabilities" in m for m in missing)
