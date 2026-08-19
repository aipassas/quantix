"""User-editable valuation and risk thresholds.

config.py holds the DEFAULTS — the numbers this app ships with and the
reasoning behind each one. This module holds the user's overrides of a
curated subset of them, and hands out the "effective" config every
consumer should actually read.

WHY AN EFFECTIVE-CONFIG ACCESSOR RATHER THAN MUTATING config: the config
objects are frozen dataclasses imported by name (`from config import
SCORECARD`) into eight modules at import time, so rebinding config.SCORECARD
later would not reach any of them, and mutating a frozen instance isn't
possible at all. effective_scorecard() instead returns a dataclasses.replace()
copy with the user's overrides applied, and the scoring engine defaults to
calling it — so scoring, alerts and the screener all pick up a changed
threshold without any of those five call sites having to remember to opt in.
That default is what makes "reflected across scoring, alerts, and screener"
true by construction rather than by discipline.

ONLY A CURATED SUBSET IS EDITABLE. The pass/fail lines ("what counts as
overvalued / high risk") are exposed; the per-metric scoring WEIGHTS and the
Composite Risk Score's internal anchor points are not. Those change how a
score is computed rather than where its threshold sits, and are easy to make
incoherent by accident. That split was agreed with the user before this was
built rather than assumed. EDITABLE below is the whitelist, and anything not
in it is rejected by load_overrides() even if it's present in the store file.

Persisted with the same atomic-write, gitignored-local-file pattern as every
other piece of cross-restart state here (see local_store.py). Quantix has no
accounts, so this is one shared set of thresholds for whoever runs this
instance, not per-user.
"""
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import RISK, SCORECARD, THRESHOLDS
from local_store import atomic_write_text
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("user_thresholds")


@dataclass(frozen=True)
class ThresholdSpec:
    """One editable number: which config object it belongs to, how to label
    it, and the bounds the UI enforces. `minimum`/`maximum` are sanity rails
    (a negative current ratio or a 900% margin is a typo, not a preference),
    not opinions about what a good threshold is."""
    key: str
    target: str          # "scorecard" | "risk"
    label: str
    minimum: float
    maximum: float
    step: float
    helptext: str


# The whitelist. Order here is the order the UI renders them in.
EDITABLE: Tuple[ThresholdSpec, ...] = (
    # --- Scorecard: profitability & capital efficiency ---
    ThresholdSpec("min_net_margin", "scorecard", "Min Net Margin", 0.0, 1.0, 0.01,
                  "A company passes the margin check at or above this. Stored as a fraction, so 0.10 is 10%."),
    ThresholdSpec("min_roic_pct", "scorecard", "Min ROIC (%)", 0.0, 100.0, 0.5,
                  "Return on invested capital a company must clear to pass the capital-efficiency check."),
    ThresholdSpec("min_fcf_yield_pct", "scorecard", "Min FCF Yield (%)", 0.0, 50.0, 0.5,
                  "Free cash flow as a percentage of market cap. Shown in the Master Matrix rather than scored."),
    # --- Scorecard: leverage & liquidity ---
    ThresholdSpec("max_debt_to_equity", "scorecard", "Max Debt/Equity", 0.0, 20.0, 0.1,
                  "Leverage ceiling for ordinary companies. Above this fails the leverage check."),
    ThresholdSpec("financials_max_debt_to_equity", "scorecard", "Max Debt/Equity — Financials", 0.0, 30.0, 0.1,
                  "The separate, higher ceiling applied to banks and other Financials, whose borrowings ARE the business model."),
    ThresholdSpec("min_current_ratio", "scorecard", "Min Current Ratio", 0.0, 10.0, 0.1,
                  "Current assets over current liabilities. Below this fails the short-term liquidity check."),
    ThresholdSpec("min_interest_coverage", "scorecard", "Min Interest Coverage", 0.0, 50.0, 0.5,
                  "How many times over operating income covers the interest bill."),
    # --- Scorecard: valuation & market ---
    ThresholdSpec("pe_range_low", "scorecard", "P/E Band — Low", 0.0, 200.0, 1.0,
                  "Bottom of the acceptable P/E band used when a sector has no override of its own."),
    ThresholdSpec("pe_range_high", "scorecard", "P/E Band — High", 0.0, 500.0, 1.0,
                  "Top of the acceptable P/E band used when a sector has no override of its own."),
    ThresholdSpec("peg_range_low", "scorecard", "PEG Band — Low", 0.0, 20.0, 0.1,
                  "Bottom of the acceptable PEG band. PEG is deliberately not sector-adjusted — it already divides P/E by growth."),
    ThresholdSpec("peg_range_high", "scorecard", "PEG Band — High", 0.0, 20.0, 0.1,
                  "Top of the acceptable PEG band."),
    ThresholdSpec("max_beta", "scorecard", "Max Beta", 0.0, 10.0, 0.1,
                  "Ceiling on market sensitivity. Above this fails the volatility check."),
    # --- Scorecard: verdict bands ---
    ThresholdSpec("high_alignment_pct", "scorecard", "High Alignment (%)", 0.0, 100.0, 1.0,
                  "Weighted score at or above which a company is labelled High alignment."),
    ThresholdSpec("moderate_alignment_pct", "scorecard", "Moderate Alignment (%)", 0.0, 100.0, 1.0,
                  "Weighted score at or above which a company is labelled Moderate rather than Low."),
    # --- Risk ---
    ThresholdSpec("altman_safe_zone", "risk", "Altman Z — Safe Zone", 0.0, 20.0, 0.01,
                  "Altman Z at or above which a company is in the Safe zone."),
    ThresholdSpec("altman_grey_zone", "risk", "Altman Z — Distress Zone", 0.0, 20.0, 0.01,
                  "Altman Z below which a company is in the Distress zone. Between the two is the grey area."),
    ThresholdSpec("vix_high_risk_threshold", "risk", "VIX High-Risk Level", 0.0, 100.0, 0.5,
                  "VIX at or above which the macro regime is flagged as high risk."),
)

EDITABLE_BY_KEY: Dict[str, ThresholdSpec] = {s.key: s for s in EDITABLE}

# pe_range/peg_range are tuples on the dataclass but two separate numbers in
# the UI, so they're stored flat and reassembled in effective_scorecard().
_PAIRED = {"pe_range": ("pe_range_low", "pe_range_high"),
           "peg_range": ("peg_range_low", "peg_range_high")}


def _store_path() -> Path:
    return Path(__file__).resolve().parent / THRESHOLDS.store_filename


def default_value(key: str) -> float:
    """The shipped default for an editable key, read off config rather than
    duplicated here — so a default changed in config.py can never disagree
    with what the reset button restores."""
    spec = EDITABLE_BY_KEY[key]
    for field, (lo, hi) in _PAIRED.items():
        if key == lo:
            return float(getattr(SCORECARD, field)[0])
        if key == hi:
            return float(getattr(SCORECARD, field)[1])
    source = SCORECARD if spec.target == "scorecard" else RISK
    return float(getattr(source, key))


def defaults() -> Dict[str, float]:
    return {spec.key: default_value(spec.key) for spec in EDITABLE}


def load_overrides(path: Optional[Path] = None) -> Dict[str, float]:
    """The user's saved overrides, keyed by spec key.

    Never raises, and never trusts the file: unknown keys, non-numeric
    values and out-of-range values are all dropped with a logged warning
    rather than allowed to reach the scoring engine. A hand-edited or
    stale store therefore degrades to the shipped defaults for the bad
    entries instead of producing a silently wrong Scorecard.
    """
    path = path or _store_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        log_exception(logger, "user_thresholds.store_corrupt", section="user_thresholds")
        return {}
    if not isinstance(raw, dict):
        return {}

    clean: Dict[str, float] = {}
    for key, value in raw.items():
        spec = EDITABLE_BY_KEY.get(key)
        if spec is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not (spec.minimum <= float(value) <= spec.maximum):
            continue
        clean[key] = float(value)
    return clean


def save_overrides(overrides: Dict[str, float], path: Optional[Path] = None) -> None:
    """Persist only the values that differ from the shipped defaults, so the
    store stays a record of what the user deliberately changed rather than a
    frozen copy of every default — which would silently pin old numbers if a
    default were ever revised."""
    path = path or _store_path()
    base = defaults()
    changed = {k: float(v) for k, v in overrides.items()
               if k in EDITABLE_BY_KEY and float(v) != base[k]}
    atomic_write_text(path, json.dumps(changed, indent=2, sort_keys=True))


def effective_values(path: Optional[Path] = None) -> Dict[str, float]:
    """Defaults with the user's valid overrides applied — what the UI shows
    in its inputs and what the effective_* builders below use."""
    values = defaults()
    values.update(load_overrides(path))
    return values


def effective_scorecard(path: Optional[Path] = None):
    """SCORECARD with the user's overrides applied. This is what the scoring
    engine reads by default, so the Scorecard, the watchlist screen and the
    screener's Altman verdict all move together."""
    values = effective_values(path)
    patch = {}
    for spec in EDITABLE:
        if spec.target != "scorecard" or spec.key in ("pe_range_low", "pe_range_high",
                                                      "peg_range_low", "peg_range_high"):
            continue
        patch[spec.key] = values[spec.key]
    patch["pe_range"] = (values["pe_range_low"], values["pe_range_high"])
    patch["peg_range"] = (values["peg_range_low"], values["peg_range_high"])
    # The per-sector band table rides along, so pe_range_for() on the returned
    # object resolves a sector the user retuned rather than the shipped one.
    patch["sector_pe_ranges"] = effective_sector_pe(path)
    return replace(SCORECARD, **patch)


def effective_risk(path: Optional[Path] = None):
    """RISK with the user's overrides applied."""
    values = effective_values(path)
    patch = {spec.key: values[spec.key] for spec in EDITABLE if spec.target == "risk"}
    return replace(RISK, **patch)


# --- per-sector P/E bands -----------------------------------------------------
# The sector mechanism config.py already has, made user-editable. Stored in the
# same file under a reserved key that isn't a valid spec key, so it can't
# collide with a scalar override.
_SECTOR_KEY = "_sector_pe_ranges"
_SECTOR_REMOVED_KEY = "_sector_pe_removed"


def load_sector_pe(path: Optional[Path] = None) -> Dict[str, Tuple[float, float]]:
    """The user's sector P/E table, or {} if they haven't set one. Never
    raises; malformed rows are dropped individually."""
    path = path or _store_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text()).get(_SECTOR_KEY, {})
    except Exception:
        log_exception(logger, "user_thresholds.sector_store_corrupt", section="user_thresholds")
        return {}
    out: Dict[str, Tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return {}
    for sector, band in raw.items():
        if not isinstance(sector, str) or not sector.strip():
            continue
        if not (isinstance(band, (list, tuple)) and len(band) == 2):
            continue
        try:
            lo, hi = float(band[0]), float(band[1])
        except (TypeError, ValueError):
            continue
        if lo < 0 or hi <= lo:
            continue
        out[sector] = (lo, hi)
    return out


def save_sector_pe(table: Dict[str, Tuple[float, float]], path: Optional[Path] = None) -> None:
    """Write the sector table alongside the scalar overrides, preserving
    whatever scalars are already stored.

    `table` is the COMPLETE desired table, not a patch — it mirrors what the
    data-editor hands back, so a shipped sector missing from it means the
    user deleted that row.

    Stores only rows that DIFFER from the shipped table (or name a sector it
    doesn't ship), for the same reason save_overrides() stores only changed
    scalars: writing back a row identical to the default would silently pin
    today's shipped band forever, so a later revision in config.py would
    never reach a user who had merely opened this panel once. A shipped
    sector the user DELETED is recorded explicitly, since "no row" would
    otherwise be indistinguishable from "unchanged" and the band would come
    straight back.
    """
    path = path or _store_path()
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}

    shipped = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    deltas = {s: [float(lo), float(hi)] for s, (lo, hi) in table.items()
              if shipped.get(s) != (float(lo), float(hi))}
    removed = sorted(set(shipped) - set(table))

    if deltas:
        existing[_SECTOR_KEY] = deltas
    else:
        existing.pop(_SECTOR_KEY, None)
    if removed:
        existing[_SECTOR_REMOVED_KEY] = removed
    else:
        existing.pop(_SECTOR_REMOVED_KEY, None)
    atomic_write_text(path, json.dumps(existing, indent=2, sort_keys=True))


def load_removed_sectors(path: Optional[Path] = None) -> Tuple[str, ...]:
    """Shipped sectors the user deleted from the table. Never raises."""
    path = path or _store_path()
    if not path.exists():
        return ()
    try:
        raw = json.loads(path.read_text()).get(_SECTOR_REMOVED_KEY, [])
    except Exception:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(s for s in raw if isinstance(s, str) and s)


def effective_sector_pe(path: Optional[Path] = None) -> Dict[str, Tuple[float, float]]:
    """The shipped sector P/E table with the user's edits layered on top.
    A user row replaces the shipped band for that sector; sectors the user
    hasn't touched keep theirs."""
    table = {k: tuple(float(x) for x in v) for k, v in SCORECARD.sector_pe_ranges.items()}
    for sector in load_removed_sectors(path):
        table.pop(sector, None)
    table.update(load_sector_pe(path))
    return table


def validate(values: Dict[str, float], sector_table: Dict[str, Tuple[float, float]]) -> list:
    """Cross-field checks the per-field min/max bounds can't express, returned
    as human-readable messages (empty list means OK).

    Lives here rather than inline in the UI so it can be tested directly:
    these are exactly the mistakes that would otherwise produce a silently
    incoherent Scorecard — an inverted band accepts nothing, and a distress
    zone above the safe zone makes the middle "grey" band impossible.
    """
    errors = []
    if values.get("pe_range_low", 0) >= values.get("pe_range_high", 0):
        errors.append("P/E band low must be below high.")
    if values.get("peg_range_low", 0) >= values.get("peg_range_high", 0):
        errors.append("PEG band low must be below high.")
    if values.get("altman_grey_zone", 0) >= values.get("altman_safe_zone", 0):
        errors.append("Altman distress zone must be below the safe zone.")
    if values.get("moderate_alignment_pct", 0) >= values.get("high_alignment_pct", 0):
        errors.append("Moderate alignment must be below High alignment.")
    for sector, band in sector_table.items():
        lo, hi = band
        if hi <= lo:
            errors.append(f'Sector "{sector}": P/E low must be below high.')
    return errors


def reset_all(path: Optional[Path] = None) -> None:
    """Forget every override, scalar and per-sector, returning the app to the
    numbers it ships with."""
    path = path or _store_path()
    atomic_write_text(path, json.dumps({}, indent=2))
    log_event(logger, logging.INFO, "user_thresholds.reset")
