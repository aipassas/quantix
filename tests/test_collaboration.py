"""Tests for collaboration.py — per-ticker notes with @-mentions.

The mention-resolution tests carry the most weight here: because a mention
triggers a real email, the boundary that a mention can ONLY resolve to
someone on the curated roster is a safety property, not a convenience.
"""
import json

from collaboration import (
    CollaborationStore,
    Note,
    TeamMember,
    add_member,
    add_note,
    delete_note,
    load_store,
    mark_notified,
    member_by_handle,
    notes_for,
    notify_mentions,
    parse_mentions,
    remove_member,
    save_store,
)
from config import COLLABORATION


def _team():
    store = CollaborationStore()
    store, _ = add_member(store, "Ana Silva", "ana@example.com")
    store, _ = add_member(store, "Bob Chen", "bob@example.com")
    return store


class _FakeSender:
    """Stands in for email_report.send_notification_email."""

    def __init__(self, fail_for=()):
        self.sent = []
        self.fail_for = set(fail_for)

    def __call__(self, to_address, subject, body):
        if to_address in self.fail_for:
            return False, "SMTP exploded"
        self.sent.append((to_address, subject, body))
        return True, None


# --- roster -------------------------------------------------------------------

def test_add_member_and_derive_handle():
    store = _team()
    assert [m.name for m in store.members] == ["Ana Silva", "Bob Chen"]
    assert member_by_handle(store, "anasilva").email == "ana@example.com"


def test_handle_is_derived_not_stored_so_it_cannot_go_stale():
    """Renaming a member must not leave an old handle pointing at them."""
    assert TeamMember("Ana Silva", "a@b.com").handle == "anasilva"
    assert TeamMember("Ana Silvana", "a@b.com").handle == "anasilvana"


def test_member_requires_a_name_and_a_plausible_email():
    store = CollaborationStore()
    _, err = add_member(store, "", "a@b.com")
    assert err is not None
    _, err = add_member(store, "Ana", "not-an-email")
    assert err is not None


def test_duplicate_handles_are_rejected():
    """Two members resolving to the same @handle would make a mention
    ambiguous — and ambiguous means mailing the wrong person."""
    store = _team()
    _, err = add_member(store, "ana silva", "other@example.com")
    assert err is not None and "already" in err.lower()


def test_roster_is_capped():
    store = CollaborationStore()
    for i in range(COLLABORATION.max_members):
        store, err = add_member(store, f"P{i}", f"p{i}@example.com")
        assert err is None
    _, err = add_member(store, "OneTooMany", "x@example.com")
    assert err is not None and "full" in err.lower()


def test_remove_member():
    store = remove_member(_team(), "Ana Silva")
    assert [m.name for m in store.members] == ["Bob Chen"]


# --- mention resolution (the safety boundary) ---------------------------------

def test_mentions_resolve_against_the_roster():
    store = _team()
    assert parse_mentions("ping @AnaSilva and @BobChen", store.members) == ("Ana Silva", "Bob Chen")


def test_mentions_are_case_insensitive():
    store = _team()
    assert parse_mentions("@anasilva @BOBCHEN", store.members) == ("Ana Silva", "Bob Chen")


def test_unknown_handles_are_ignored_not_treated_as_addresses():
    """The safety property: an @ that isn't on the roster must never become
    a recipient. Otherwise a typo — or a deliberately typed outside
    address — would cause this app to mail a stranger."""
    store = _team()
    assert parse_mentions("@nobody @AnaSilva @ceo@bigcorp.com", store.members) == ("Ana Silva",)


def test_a_raw_email_address_in_the_body_is_never_a_mention():
    store = _team()
    assert parse_mentions("email me at someone@elsewhere.com", store.members) == ()


def test_mentions_are_deduplicated_in_first_appearance_order():
    store = _team()
    assert parse_mentions("@BobChen @AnaSilva @BobChen", store.members) == ("Bob Chen", "Ana Silva")


def test_no_roster_means_no_mentions_can_resolve():
    assert parse_mentions("@anyone @everyone", ()) == ()


# --- notes --------------------------------------------------------------------

def test_add_note_normalises_ticker_and_records_author():
    store, note, err = add_note(_team(), "aapl", "Angelos", "Looks cheap")
    assert err is None
    assert note.ticker == "AAPL" and note.author == "Angelos"
    assert notes_for(store, "AAPL") == (note,)
    assert notes_for(store, "aapl") == (note,)


def test_notes_are_appended_so_the_thread_reads_chronologically():
    store, _, _ = add_note(_team(), "AAPL", "A", "first")
    store, _, _ = add_note(store, "AAPL", "B", "second")
    assert [n.body for n in notes_for(store, "AAPL")] == ["first", "second"]


def test_note_requires_ticker_author_and_body():
    store = _team()
    for ticker, author, body in (("", "A", "x"), ("AAPL", "", "x"), ("AAPL", "A", "   ")):
        _, note, err = add_note(store, ticker, author, body)
        assert note is None and err is not None


def test_note_length_is_capped():
    store = _team()
    _, note, err = add_note(store, "AAPL", "A", "x" * (COLLABORATION.max_note_chars + 1))
    assert note is None and err is not None


def test_delete_note_removes_only_that_note():
    store, first, _ = add_note(_team(), "AAPL", "A", "first")
    store, second, _ = add_note(store, "AAPL", "B", "second")
    store = delete_note(store, "AAPL", first.id)
    assert [n.body for n in notes_for(store, "AAPL")] == ["second"]


def test_deleting_the_last_note_drops_the_empty_thread():
    store, note, _ = add_note(_team(), "AAPL", "A", "only")
    store = delete_note(store, "AAPL", note.id)
    assert "AAPL" not in store.notes


def test_threads_are_per_ticker():
    store, _, _ = add_note(_team(), "AAPL", "A", "apple note")
    store, _, _ = add_note(store, "MSFT", "A", "msft note")
    assert [n.body for n in notes_for(store, "AAPL")] == ["apple note"]
    assert [n.body for n in notes_for(store, "MSFT")] == ["msft note"]


# --- notification --------------------------------------------------------------

def test_notify_emails_every_mentioned_member():
    store = _team()
    store, note, _ = add_note(store, "AAPL", "Angelos", "@AnaSilva @BobChen thoughts?")
    sender = _FakeSender()
    notified, errors = notify_mentions(store, note, sender)
    assert notified == ("Ana Silva", "Bob Chen")
    assert errors == []
    assert sorted(a for a, _, _ in sender.sent) == ["ana@example.com", "bob@example.com"]


def test_notification_body_carries_the_note_and_discloses_the_lack_of_auth():
    store = _team()
    store, note, _ = add_note(store, "AAPL", "Angelos", "@AnaSilva please review")
    sender = _FakeSender()
    notify_mentions(store, note, sender)
    _, subject, body = sender.sent[0]
    assert "AAPL" in subject and "Angelos" in subject
    assert "please review" in body
    assert "no user accounts" in body  # the self-declared-identity caveat


def test_one_failed_recipient_does_not_stop_the_others():
    store = _team()
    store, note, _ = add_note(store, "AAPL", "Angelos", "@AnaSilva @BobChen")
    sender = _FakeSender(fail_for={"ana@example.com"})
    notified, errors = notify_mentions(store, note, sender)
    assert notified == ("Bob Chen",)
    assert len(errors) == 1 and "Ana Silva" in errors[0]


def test_a_note_with_no_mentions_sends_nothing():
    store = _team()
    store, note, _ = add_note(store, "AAPL", "Angelos", "just a private thought")
    sender = _FakeSender()
    assert notify_mentions(store, note, sender) == ((), [])
    assert sender.sent == []


def test_mark_notified_records_partial_delivery():
    """The UI has to be able to say who was actually reached rather than
    implying everyone mentioned got an email."""
    store = _team()
    store, note, _ = add_note(store, "AAPL", "Angelos", "@AnaSilva @BobChen")
    store = mark_notified(store, "AAPL", note.id, ("Bob Chen",))
    stored = notes_for(store, "AAPL")[0]
    assert stored.mentions == ("Ana Silva", "Bob Chen")
    assert stored.notified == ("Bob Chen",)


# --- persistence ---------------------------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "c.json"
    store = _team()
    store, _, _ = add_note(store, "AAPL", "Angelos", "@AnaSilva look at this")
    save_store(store, path)
    loaded = load_store(path)
    assert [m.name for m in loaded.members] == ["Ana Silva", "Bob Chen"]
    note = notes_for(loaded, "AAPL")[0]
    assert note.body == "@AnaSilva look at this"
    assert note.mentions == ("Ana Silva",)


def test_missing_store_is_empty(tmp_path):
    assert load_store(tmp_path / "nope.json") == CollaborationStore()


def test_corrupt_store_degrades_to_empty(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{not json")
    assert load_store(path) == CollaborationStore()


def test_one_malformed_note_does_not_discard_the_whole_file(tmp_path):
    """Losing everyone's notes because one row is bad would be a far worse
    failure than dropping the bad row."""
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "members": [{"name": "Ana Silva", "email": "ana@example.com"},
                    {"name": "Broken"}],
        "notes": {"AAPL": [
            {"id": "1", "author": "A", "body": "good note", "created_at": ""},
            {"id": "2", "author": "B", "body": "   "},
        ]},
    }))
    loaded = load_store(path)
    assert [m.name for m in loaded.members] == ["Ana Silva"]
    assert [n.body for n in notes_for(loaded, "AAPL")] == ["good note"]


def test_save_leaves_no_leftover_temp_file(tmp_path):
    path = tmp_path / "c.json"
    save_store(_team(), path)
    assert [p.name for p in tmp_path.iterdir()] == ["c.json"]
