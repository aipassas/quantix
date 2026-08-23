"""Saved screeners: named, reorderable filter sets you can re-run in one click.

A TEMPLATE SAVES THE UNIVERSE AS WELL AS THE CRITERIA, and that is the
decision the whole module turns on. This app's screener filters a ticker
list you supply; it does not scan the market. So "Tech Stocks Under $100"
cannot mean "go find tech stocks" — the tech tickers ARE part of the
template, and a saved screen that captured only the filters would produce
a different answer every time depending on whatever happened to be in the
universe box. Saving both is what makes clicking a template reproduce a
result rather than merely re-apply a filter.

THE STARTER SET IS SEEDED ONCE, THEN OWNED BY THE USER. The four examples
from the brief are written into the store on first use as ordinary
templates — renameable, reorderable, deletable — rather than kept as
read-only builtins the user is stuck with. A `seeded` flag records that
it happened, so deleting them all keeps them gone instead of having them
reappear on the next run.

ORDER IS THE LIST ORDER, moved with explicit up/down operations. Streamlit
has no drag-and-drop and adding a third-party component for it would put
an iframe-rendered dependency into an app that is otherwise entirely
native — so reordering is buttons. That is also the more accessible
choice: dragging is difficult or impossible with a keyboard.

CRITERIA ARE STORED AS PLAIN DICTS, not as ScreenCriterion objects, so a
template written by one version still loads in another. A criterion whose
metric or operator this version does not recognise is kept in the file and
reported rather than dropped — silently discarding part of a saved screen
would change what it means without saying so.
"""
import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from local_store import atomic_write_text, store_path
from logging_setup import get_logger, log_event, log_exception

logger = get_logger("screener_templates")

STORE_FILENAME = "screener_templates.json"

MAX_TEMPLATES = 40
MAX_NAME_LENGTH = 60


@dataclass
class Template:
    name: str
    criteria: Tuple[Dict[str, Any], ...] = ()
    universe: Tuple[str, ...] = ()
    created_at: Optional[str] = None

    @property
    def summary(self) -> str:
        """One line describing what this screen does, for the UI."""
        from screener import METRICS_BY_KEY

        parts = []
        for raw in self.criteria:
            spec = METRICS_BY_KEY.get(raw.get("metric", ""))
            label = spec.label if spec else raw.get("metric", "?")
            threshold = raw.get("threshold")
            unit = spec.unit if spec else ""
            if isinstance(threshold, (int, float)):
                # A currency unit is a prefix, not a suffix — "< $100",
                # never "< 100$".
                shown = f"${threshold:g}" if unit == "$" else f"{threshold:g}{unit}"
            else:
                shown = str(threshold)
            parts.append(f"{label} {raw.get('operator', '?')} {shown}")
        return "  ·  ".join(parts) or "No criteria"

    def unknown_parts(self) -> List[str]:
        """Metrics or operators this version cannot evaluate.

        Reported to the user rather than quietly skipped: a template that
        silently dropped one of its filters would return more matches than
        the screen it claims to be.
        """
        from screener import METRICS_BY_KEY, operators_for

        problems = []
        for raw in self.criteria:
            metric = raw.get("metric", "")
            spec = METRICS_BY_KEY.get(metric)
            if spec is None:
                problems.append(f"unknown metric “{metric}”")
                continue
            if raw.get("operator") not in operators_for(metric):
                problems.append(f"“{raw.get('operator')}” is not valid for {spec.label}")
        return problems


def starter_templates() -> Tuple[Template, ...]:
    """The four screens from the brief, as real, runnable filters.

    Each carries its own universe because the screener filters a list
    rather than searching the market — see the module docstring. The
    universes are deliberately small, recognisable names rather than an
    attempt at market coverage the 30-ticker cap could not honour anyway.
    """
    tech = ("AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "IBM",
            "QCOM", "TXN", "MU", "HPQ")
    broad = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "JPM", "V",
             "JNJ", "KO", "PG", "XOM", "WMT", "UNH", "HD")
    income = ("KO", "PG", "JNJ", "XOM", "CVX", "PFE", "VZ", "T", "MMM",
              "IBM", "PEP", "MRK")
    return (
        Template(
            name="Tech Stocks Under $100",
            criteria=({"metric": "sector", "operator": "is", "threshold": "Technology"},
                      {"metric": "price", "operator": "<", "threshold": 100.0}),
            universe=tech,
        ),
        Template(
            name="Value Plays: P/E < 15",
            criteria=({"metric": "pe_ratio", "operator": "<", "threshold": 15.0},),
            universe=broad,
        ),
        Template(
            name="Dividend Stocks: Yield > 2%",
            criteria=({"metric": "dividend_yield_pct", "operator": ">", "threshold": 2.0},),
            universe=income,
        ),
        Template(
            name="Growth: Revenue Growth > 20%",
            criteria=({"metric": "revenue_growth_pct", "operator": ">", "threshold": 20.0},),
            universe=broad,
        ),
    )


# --- storage ------------------------------------------------------------------

def _path():
    return store_path(STORE_FILENAME)


_EMPTY: Dict[str, Any] = {"templates": [], "seeded": False}


def _read() -> Tuple[Dict[str, Any], bool]:
    """(data, corrupt). `corrupt` means the file exists but could not be
    parsed — which is NOT the same as having no saved screeners.

    The distinction matters because seeding and saving both write. Treating
    an unreadable file as "empty" would overwrite it with the starter set
    on the very next load, destroying whatever the user had saved because
    of a single bad byte. Nothing here writes when corrupt is True.
    """
    path = _path()
    if not path.exists():
        return dict(_EMPTY), False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log_exception(logger, "screener_templates.unreadable", section="screener_templates")
        return dict(_EMPTY), True
    if not isinstance(data, dict) or not isinstance(data.get("templates"), list):
        log_event(logger, logging.WARNING, "screener_templates.unexpected_shape")
        return dict(_EMPTY), True
    return data, False


def store_is_corrupt() -> bool:
    """For the UI: the file is there but unreadable, so the list shown is
    empty for a reason worth stating rather than because nothing is saved."""
    return _read()[1]


def _write(data: Dict[str, Any]) -> None:
    atomic_write_text(_path(), json.dumps(data, indent=2))


def _to_template(record: Dict[str, Any]) -> Optional[Template]:
    try:
        name = str(record.get("name", "")).strip()
        if not name:
            return None
        criteria = tuple(c for c in record.get("criteria", []) if isinstance(c, dict))
        universe = tuple(str(t).strip().upper() for t in record.get("universe", []) if str(t).strip())
        return Template(name=name, criteria=criteria, universe=universe,
                        created_at=record.get("created_at"))
    except Exception:
        return None


def load() -> List[Template]:
    """Every saved screener, in display order. Seeds the starter set once."""
    data, corrupt = _read()
    if corrupt:
        # Show nothing rather than overwrite something. store_is_corrupt()
        # lets the UI say why the list is empty.
        return []
    if not data.get("seeded") and not data.get("templates"):
        starters = [asdict(t) for t in starter_templates()]
        for record in starters:
            record["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        data = {"templates": starters, "seeded": True}
        _write(data)
        log_event(logger, logging.INFO, "screener_templates.seeded", count=len(starters))

    out = []
    for record in data.get("templates", []):
        template = _to_template(record)
        if template:
            out.append(template)
    return out


def names() -> List[str]:
    return [t.name for t in load()]


def get(name: str) -> Optional[Template]:
    wanted = (name or "").strip().casefold()
    for template in load():
        if template.name.casefold() == wanted:
            return template
    return None


def save(name: str, criteria: Sequence[Dict[str, Any]],
         universe: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Create or overwrite a saved screener. Returns (ok, error)."""
    name = (name or "").strip()
    if not name:
        return False, "Give the screener a name."
    if len(name) > MAX_NAME_LENGTH:
        return False, f"Keep the name under {MAX_NAME_LENGTH} characters."
    criteria = [c for c in (criteria or []) if isinstance(c, dict)]
    if not criteria:
        return False, "Add at least one filter before saving."

    cleaned_universe = tuple(dict.fromkeys(
        str(t).strip().upper() for t in (universe or []) if str(t).strip()))

    data, corrupt = _read()
    if corrupt:
        return False, ("The saved-screener file on this instance can't be read, so "
                       "saving would overwrite it. Move or delete "
                       f"{STORE_FILENAME} and try again.")
    records = list(data.get("templates", []))
    record = {
        "name": name,
        "criteria": criteria,
        "universe": list(cleaned_universe),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    for index, existing in enumerate(records):
        if str(existing.get("name", "")).strip().casefold() == name.casefold():
            # Overwrite in place so re-saving keeps its position in the
            # order the user arranged, rather than jumping to the end.
            record["created_at"] = existing.get("created_at") or record["created_at"]
            records[index] = record
            data["templates"] = records
            data["seeded"] = True
            _write(data)
            log_event(logger, logging.INFO, "screener_templates.updated",
                      criteria=len(criteria), universe=len(cleaned_universe))
            return True, None

    if len(records) >= MAX_TEMPLATES:
        return False, f"You already have {MAX_TEMPLATES} saved screeners. Delete one first."

    records.append(record)
    data["templates"] = records
    data["seeded"] = True
    _write(data)
    log_event(logger, logging.INFO, "screener_templates.saved",
              criteria=len(criteria), universe=len(cleaned_universe))
    return True, None


def delete(name: str) -> bool:
    wanted = (name or "").strip().casefold()
    data, corrupt = _read()
    if corrupt:
        return False
    records = [r for r in data.get("templates", [])
               if str(r.get("name", "")).strip().casefold() != wanted]
    if len(records) == len(data.get("templates", [])):
        return False
    data["templates"] = records
    # Stays True so deleting every template does not re-seed the starters
    # on the next load. Someone who cleared the list meant to clear it.
    data["seeded"] = True
    _write(data)
    log_event(logger, logging.INFO, "screener_templates.deleted")
    return True


def move(name: str, delta: int) -> bool:
    """Move a screener up (-1) or down (+1) in the display order."""
    wanted = (name or "").strip().casefold()
    data, corrupt = _read()
    if corrupt:
        return False
    records = list(data.get("templates", []))
    index = next((i for i, r in enumerate(records)
                  if str(r.get("name", "")).strip().casefold() == wanted), None)
    if index is None:
        return False
    target = index + delta
    if not (0 <= target < len(records)):
        return False          # already at an end; not an error
    records[index], records[target] = records[target], records[index]
    data["templates"] = records
    _write(data)
    return True


def reset_to_starters() -> None:
    """Restore the built-in set, discarding what is saved."""
    records = [asdict(t) for t in starter_templates()]
    for record in records:
        record["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write({"templates": records, "seeded": True})
    log_event(logger, logging.INFO, "screener_templates.reset")
