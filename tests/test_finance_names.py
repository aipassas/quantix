"""finance.py uses no config constant it never imported.

WHY THIS EXISTS. finance.py is an 11,700-line script, so a name it uses
but never binds is a NameError that fires at RUNTIME, in whatever branch
happens to touch it — and every source-reading test in this suite passes
against it, because the text is right there in the file. That is exactly
how `STREAKS.calendar_days` and `CONTEST.benchmark` reached a committed
panel: the code read correctly, the tests asserted the right substrings,
and the expander would have raised the moment anyone opened it.

The check is deliberately narrow rather than a full scope analysis. Every
config constant in this app is ALL_CAPS and is reached by attribute
access, so "every ALL_CAPS name whose attribute is read must be bound
somewhere at module level" catches the whole class without needing to
model comprehension scopes or function locals. pyflakes would do more,
but it is not a declared dependency and this needs none.
"""
import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

APP_DIR = Path(__file__).resolve().parent.parent
FINANCE = APP_DIR / "finance.py"

# Names that look like constants but come from somewhere other than a
# module-level binding in finance.py.
ALLOWED = {"TYPE_CHECKING"}


def _tree() -> ast.Module:
    return ast.parse(FINANCE.read_text())


def _bound_names(tree: ast.Module) -> set:
    """Every name finance.py binds anywhere — imports, assignments,
    defs, loop and with targets, at any depth. Depth is ignored on
    purpose: a name bound inside an `if` at module level is still
    available afterwards in a script, and this check is about names that
    are bound NOWHERE."""
    import builtins
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    return bound


def _constant_reads(tree: ast.Module):
    """(name, lineno) for every `SOMETHING.attr` where SOMETHING is an
    ALL_CAPS bare name — the shape every config constant is used in."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and isinstance(node.value.ctx, ast.Load)):
            name = node.value.id
            if name.isupper() and len(name) > 1 and name not in ALLOWED:
                out.append((name, node.value.lineno))
    return out


def test_every_config_constant_used_is_also_imported():
    """The regression this file exists for. A panel using STREAKS or
    CONTEST without importing them reads perfectly and raises on open."""
    tree = _tree()
    bound = _bound_names(tree)
    missing = sorted({(name, line) for name, line in _constant_reads(tree)
                      if name not in bound})
    assert not missing, (
        "finance.py reads config constants it never imports — these are "
        f"NameErrors waiting for someone to open the panel: {missing}"
    )


def test_the_check_would_actually_catch_one():
    """A test that cannot fail proves nothing. Parse a snippet with a
    known-missing constant and confirm it is reported."""
    tree = ast.parse("from config import ALPHA\nx = ALPHA.a\ny = BETA.b\n")
    bound = _bound_names(tree)
    found = {name for name, _ in _constant_reads(tree) if name not in bound}
    assert found == {"BETA"}


def test_the_check_does_not_flag_a_locally_assigned_constant():
    tree = ast.parse("GAMMA = object()\nz = GAMMA.c\n")
    bound = _bound_names(tree)
    assert not {n for n, _ in _constant_reads(tree) if n not in bound}


def test_the_constants_this_session_added_are_imported():
    """Named explicitly so the failure message points at the right
    feature rather than at a set difference."""
    tree = _tree()
    bound = _bound_names(tree)
    for name in ("CONTEST", "STREAKS", "LEADERBOARD", "STOCK_OF_THE_WEEK",
                 "PEER_COMPARISON", "FOLLOWING"):
        used = any(n == name for n, _ in _constant_reads(tree))
        if used:
            assert name in bound, f"finance.py uses {name} without importing it"
