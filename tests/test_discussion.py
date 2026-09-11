"""Per-stock discussion: replies, ownership, and soft removal with an audit.

Team Notes already WAS a per-ticker discussion — shared threads,
verified-vs-typed authorship, roster-bounded mentions. What the ticket
adds is replies and moderation, and the moderation model chosen (by the
product owner, asked explicitly) is OWNERSHIP + AUDIT: only a note's
verified author may remove it, removal hides rather than destroys and
records who/when, and posting requires a sign-in so every new note has
an owner. There is deliberately no moderator role — that is the RBAC
ticket's territory.

Ownership is checked against the ACCOUNT KEY, never the display name.
Two accounts can both be "Ana".
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import collaboration as c

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


def _post(store, author, key, body, ticker="AAPL", parent_id=""):
    store, note, err = c.add_note(store, ticker, author, body,
                                  authenticated=True, author_key=key,
                                  parent_id=parent_id)
    assert err is None, err
    return store, note


# --- posting needs an owner ---------------------------------------------------

def test_a_signed_out_post_is_refused_with_the_reason():
    store, note, err = c.add_note(c.CollaborationStore(), "AAPL", "Ana", "hi")
    assert note is None
    assert err == c.SIGN_IN_TO_POST
    assert "Reading stays open" in err


def test_a_new_note_carries_its_owner_key():
    _, note = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    assert note.author_key == "k-ana"
    assert note.authenticated is True


# --- replies ------------------------------------------------------------------

def test_a_reply_points_at_its_parent():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, reply = _post(store, "Ben", "k-ben", "disagree", parent_id=top.id)
    assert reply.is_reply and reply.parent_id == top.id
    assert [r.id for r in c.replies_for(store, "AAPL", top.id)] == [reply.id]
    assert [t.id for t in c.top_level(store, "AAPL")] == [top.id]


def test_a_reply_to_a_reply_is_refused():
    """One level deep on purpose — a tree on a ticker page is a forum."""
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, reply = _post(store, "Ben", "k-ben", "disagree", parent_id=top.id)
    _, note, err = c.add_note(store, "AAPL", "Cy", "me too", authenticated=True,
                              author_key="k-cy", parent_id=reply.id)
    assert note is None and "original note" in err


def test_a_reply_to_a_missing_note_is_refused():
    _, note, err = c.add_note(c.CollaborationStore(), "AAPL", "Ben", "x",
                              authenticated=True, author_key="k-ben",
                              parent_id="nope")
    assert note is None and "no longer here" in err


def test_a_reply_to_a_hidden_note_is_refused():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, _ = c.hide_note(store, "AAPL", top.id, "k-ana")
    _, note, err = c.add_note(store, "AAPL", "Ben", "x", authenticated=True,
                              author_key="k-ben", parent_id=top.id)
    assert note is None and "removed" in err


def test_replies_stay_in_their_own_ticker():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    _, note, err = c.add_note(store, "MSFT", "Ben", "x", authenticated=True,
                              author_key="k-ben", parent_id=top.id)
    assert note is None, "a parent on AAPL is not a parent on MSFT"


# --- ownership ----------------------------------------------------------------

def test_only_the_author_can_hide():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store2, err = c.hide_note(store, "AAPL", top.id, "k-ben")
    assert err and "Only the person who wrote" in err
    assert store2 == store, "a refused hide must change nothing"


def test_ownership_is_the_account_key_not_the_display_name():
    """Two accounts can both be 'Ana'."""
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana-1", "thesis")
    other_ana = c.Note("x", "AAPL", "Ana", "y", "t", authenticated=True,
                       author_key="k-ana-2")
    assert c.can_moderate(top, "k-ana-2") is False
    assert c.can_moderate(other_ana, "k-ana-1") is False
    assert c.can_moderate(top, "k-ana-1") is True


def test_a_legacy_typed_name_note_has_no_owner():
    legacy = c.Note("old", "AAPL", "Typed", "old", "2026-01-01T00:00:00")
    store = c.CollaborationStore(notes={"AAPL": (legacy,)})
    assert c.can_moderate(legacy, "k-anyone") is False
    _, err = c.hide_note(store, "AAPL", "old", "k-anyone")
    assert "no owner" in err


def test_an_empty_key_never_owns_anything():
    """A note with author_key='' must not match a signed-out user's ''."""
    note = c.Note("n", "AAPL", "A", "b", "t", authenticated=True, author_key="")
    assert c.can_moderate(note, "") is False


# --- soft removal with audit --------------------------------------------------

def test_hiding_records_who_and_when_and_keeps_the_note():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, err = c.hide_note(store, "AAPL", top.id, "k-ana")
    assert err is None
    (kept,) = c.notes_for(store, "AAPL")
    assert kept.hidden is True
    assert kept.hidden_by == "k-ana"
    assert kept.hidden_at
    assert kept.body == "thesis", "hidden, not destroyed"


def test_a_hidden_note_leaves_the_visible_thread():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, _ = c.hide_note(store, "AAPL", top.id, "k-ana")
    assert c.visible_notes_for(store, "AAPL") == ()


def test_hiding_a_parent_hides_its_replies_from_view_but_keeps_them():
    """A reply with no visible parent reads as a non sequitur; restoring
    the parent must bring the conversation back whole."""
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, reply = _post(store, "Ben", "k-ben", "disagree", parent_id=top.id)
    store, _ = c.hide_note(store, "AAPL", top.id, "k-ana")
    assert c.visible_notes_for(store, "AAPL") == ()
    assert len(c.notes_for(store, "AAPL")) == 2
    store, _ = c.restore_note(store, "AAPL", top.id, "k-ana")
    assert [n.id for n in c.visible_notes_for(store, "AAPL")] == [top.id, reply.id]


def test_only_the_author_can_restore():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, _ = c.hide_note(store, "AAPL", top.id, "k-ana")
    _, err = c.restore_note(store, "AAPL", top.id, "k-ben")
    assert err and "restore" in err


def test_restoring_clears_the_audit_fields():
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, _ = c.hide_note(store, "AAPL", top.id, "k-ana")
    store, _ = c.restore_note(store, "AAPL", top.id, "k-ana")
    (n,) = c.notes_for(store, "AAPL")
    assert n.hidden is False and n.hidden_by == "" and n.hidden_at == ""


def test_the_author_sees_only_their_own_hidden_notes():
    store, a = _post(c.CollaborationStore(), "Ana", "k-ana", "one")
    store, b = _post(store, "Ben", "k-ben", "two")
    store, _ = c.hide_note(store, "AAPL", a.id, "k-ana")
    store, _ = c.hide_note(store, "AAPL", b.id, "k-ben")
    assert [n.id for n in c.hidden_by_author(store, "AAPL", "k-ana")] == [a.id]
    assert c.hidden_by_author(store, "AAPL", "") == ()


def test_hiding_a_missing_note_is_an_error_not_a_crash():
    _, err = c.hide_note(c.CollaborationStore(), "AAPL", "nope", "k")
    assert "no longer here" in err


# --- persistence --------------------------------------------------------------

def test_replies_ownership_and_audit_survive_a_round_trip(tmp_path):
    store, top = _post(c.CollaborationStore(), "Ana", "k-ana", "thesis")
    store, reply = _post(store, "Ben", "k-ben", "disagree", parent_id=top.id)
    store, _ = c.hide_note(store, "AAPL", reply.id, "k-ben")
    path = tmp_path / "c.json"
    c.save_store(store, path)

    loaded = {n.id: n for n in c.notes_for(c.load_store(path), "AAPL")}
    assert loaded[top.id].author_key == "k-ana"
    assert loaded[reply.id].parent_id == top.id
    assert loaded[reply.id].hidden is True
    assert loaded[reply.id].hidden_by == "k-ben"


def test_an_old_store_without_the_new_fields_still_loads(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"members": [], "notes": {"AAPL": [
        {"id": "a", "author": "A", "body": "old", "created_at": "t",
         "authenticated": True, "issuer": "x"}]}}))
    (n,) = c.notes_for(c.load_store(path), "AAPL")
    assert n.author_key == "" and n.parent_id == "" and n.hidden is False


# --- UI wiring ----------------------------------------------------------------

def _panel():
    src = FINANCE.read_text()
    start = src.index('with st.expander(f"Team Notes — {ticker_symbol}"')
    end = src.index('"user.note_posted"', start)
    # Comments stripped: the block's own explanation names the things it
    # avoids, and a substring check would match the warning, not the code.
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_ui_hides_rather_than_hard_deletes():
    panel = _panel()
    assert "collab_hide_note(" in panel
    assert "collab_delete_note(" not in panel


def test_the_remove_button_is_drawn_only_for_the_owner():
    """Drawing it for everyone and refusing on click would advertise a
    power the reader does not have."""
    panel = _panel()
    gate = panel.index("if collab_can_moderate(_n, _cl_me):")
    button = panel.index('key=f"collab_del_{_n.id}"')
    assert gate < button


def test_replies_render_under_their_parent():
    panel = _panel()
    assert "collab_top_level(" in panel
    assert "collab_replies_for(" in panel
    assert "parent_id=_cl_note.id" in panel


def test_posting_is_gated_on_sign_in_and_says_so():
    panel = _panel()
    assert "st.info(COLLAB_SIGN_IN_TO_POST)" in panel
    assert "if _auth_user is None:" in panel


def test_every_post_carries_the_owner_key():
    panel = _panel()
    calls = [ln for ln in panel.splitlines() if "author_key=" in ln]
    assert len(calls) >= 2, "both the top-level post and the reply must set author_key"


def test_the_author_can_restore_from_the_panel():
    panel = _panel()
    assert "collab_hidden_by_author(" in panel
    assert "collab_restore_note(" in panel


def test_the_caption_no_longer_claims_anyone_can_delete():
    src = FINANCE.read_text()
    assert "can read or delete any note" not in src


def test_the_remove_button_keeps_its_danger_role():
    import button_roles
    assert "collab_del_" in button_roles.DANGER_INLINE
