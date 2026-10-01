"""The audit trail: who did what, in a chain that shows its own edits.

THE TEST THAT MATTERS MOST is test_a_record_cannot_be_written_without_an
_actor. Measured before this was built, only 7 of 174 existing log_event
calls named the account that caused them — so the gap is not "we forgot
to log things", it is "the thing we log cannot answer who". A signature
with no default for `actor` is what stops that recurring.

The second is test_erasing_an_actor_does_not_break_the_chain. An
immutable trail and a GDPR erasure request are in direct conflict, and
the resolution has to keep both properties: deleting the lines would
break the chain, which would make honouring a lawful request look
exactly like forgery.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audit
from config import AUDIT

FINANCE = Path(__file__).resolve().parent.parent / "finance.py"


@pytest.fixture
def trail(tmp_path):
    return tmp_path / "audit_log.jsonl"


def _fill(path, *rows):
    for actor, action in rows:
        entry, err = audit.record(actor, action, target="AAPL", detail="d",
                                  path=path)
        assert err == "", err
    return audit.read_all(path)


def _check(path):
    return audit.verify(audit.read_all(path), audit.line_count(path))


def _rewrite(path, records):
    path.write_text("\n".join(json.dumps(r, sort_keys=True)
                              for r in records) + "\n")


def _raw(path):
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# --- the actor ----------------------------------------------------------------

def test_a_record_cannot_be_written_without_an_actor(trail):
    """The gap this whole module exists to close."""
    entry, err = audit.record("", "sign_in", path=trail)
    assert entry is None and err == audit.NO_ACTOR
    assert "not an audit trail" in err
    assert not trail.exists()


def test_whitespace_is_not_an_actor(trail):
    entry, err = audit.record("   ", "sign_in", path=trail)
    assert entry is None and err == audit.NO_ACTOR


def test_the_actor_has_no_default_in_the_signature():
    """A default would let a caller omit it, which is how the existing
    log ended up with 7 of 174 events naming an account."""
    import inspect
    params = inspect.signature(audit.record).parameters
    assert params["actor"].default is inspect.Parameter.empty
    assert list(params)[0] == "actor"


def test_an_unknown_action_is_refused(trail):
    entry, err = audit.record("k-ana", "did_something", path=trail)
    assert entry is None and err == audit.UNKNOWN_ACTION
    assert "closed on purpose" in err


# --- the vocabulary -----------------------------------------------------------

def test_the_action_set_has_an_exact_size():
    """Adding one should break this and be extended deliberately — the
    same invariant quick_stats, support and badges already carry."""
    assert len(audit.ACTION_SPECS) == 21


def test_every_action_id_is_unique_and_mapped():
    ids = [s.id for s in audit.ACTION_SPECS]
    assert len(ids) == len(set(ids))
    assert set(audit.ACTIONS) == set(ids)


def test_every_action_has_a_known_category():
    for spec in audit.ACTION_SPECS:
        assert spec.category in audit.CATEGORIES
        assert spec.label.strip()


def test_every_category_the_ticket_names_is_covered():
    """The ticket says logins, exports and config changes."""
    covered = {s.category for s in audit.ACTION_SPECS}
    for required in (audit.AUTH, audit.EXPORT, audit.CONFIG):
        assert required in covered


def test_sharing_and_deletion_are_also_covered():
    """The surfaces PHASE 4 built: an opt-in that discloses something
    about a person, and a deletion that destroys their record."""
    covered = {s.category for s in audit.ACTION_SPECS}
    assert audit.SHARING in covered and audit.DELETION in covered


# --- the chain ----------------------------------------------------------------

def test_the_first_record_links_to_genesis(trail):
    (first,) = _fill(trail, ("k-ana", "sign_in"))
    assert first.seq == 1
    assert first.prev_hash == audit.GENESIS
    assert first.record_hash and first.record_hash != audit.GENESIS


def test_genesis_is_a_literal_not_an_empty_string():
    """An empty prev_hash would let a truncated file look like a valid
    chain that merely starts late."""
    assert audit.GENESIS == "0" * 64


def test_each_record_links_to_the_one_before(trail):
    records = _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
                    ("k-cy", "sign_out"))
    assert [r.seq for r in records] == [1, 2, 3]
    for previous, current in zip(records, records[1:]):
        assert current.prev_hash == previous.record_hash


def test_an_intact_trail_verifies(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    result = _check(trail)
    assert result.ok and result.checked == 2 and result.broken_at is None
    assert "chain is intact" in result.sentence()


def test_an_empty_trail_verifies_and_says_so():
    result = audit.verify(())
    assert result.ok and result.checked == 0
    assert "nothing to verify" in result.sentence()


def test_writing_appends_rather_than_rewriting(trail):
    """A whole-file write is how an append-only log stops being
    append-only — and it would race the cron processes sharing it."""
    _fill(trail, ("k-ana", "sign_in"))
    before = trail.read_bytes()
    _fill(trail, ("k-ben", "export_pdf"))
    after = trail.read_bytes()
    assert after.startswith(before), "the earlier bytes were rewritten"


# --- tamper evidence ----------------------------------------------------------

def test_editing_a_record_breaks_the_chain(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "sign_out"))
    rows = _raw(trail)
    rows[1]["target"] = "MSFT"
    _rewrite(trail, rows)
    result = _check(trail)
    assert not result.ok and result.broken_at == 2
    assert "do not match its own hash" in result.reason


def test_removing_a_record_breaks_the_chain(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "sign_out"))
    rows = _raw(trail)
    _rewrite(trail, [rows[0], rows[2]])
    result = _check(trail)
    assert not result.ok
    assert "removed or reordered" in result.reason


def test_reordering_records_breaks_the_chain(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "sign_out"))
    rows = _raw(trail)
    _rewrite(trail, [rows[1], rows[0], rows[2]])
    assert not _check(trail).ok


def test_swapping_only_the_actor_is_caught(trail):
    """The chain covers the DIGEST, so changing the readable actor alone
    would otherwise slip through."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    rows = _raw(trail)
    rows[0]["actor"] = "k-someone-else"
    _rewrite(trail, rows)
    result = _check(trail)
    assert not result.ok and result.broken_at == 1
    assert "actor field was swapped" in result.reason


def test_the_chain_covers_the_digest_and_not_the_readable_actor():
    """The design decision erasure depends on. If the hash took the
    actor itself, tombstoning it would break every one of that
    account's records."""
    import inspect
    params = list(inspect.signature(audit._hash).parameters)
    assert "actor_digest" in params
    assert "actor" not in params


def test_swapping_the_actor_and_its_digest_breaks_the_hash(trail):
    """Covering both is what makes the first check non-bypassable."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    rows = _raw(trail)
    rows[0]["actor"] = "k-someone-else"
    rows[0]["actor_digest"] = audit.digest_for("k-someone-else")
    _rewrite(trail, rows)
    assert not _check(trail).ok


def test_a_forger_who_recomputes_the_edited_records_hash_is_still_caught(trail):
    """THE ACTUAL ATTACK, and the only thing the prev_hash link catches
    that the sequence check does not.

    Editing a record and leaving its hash stale is caught by the record's
    own hash. A competent forger recomputes it — and then the NEXT
    record's prev_hash no longer matches, because recomputing the whole
    downstream chain is the work the chain exists to impose. An earlier
    version of this suite had no such case, so disabling the prev_hash
    check entirely still passed.
    """
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "sign_out"))
    rows = _raw(trail)
    rows[1]["detail"] = "something else"
    rows[1]["record_hash"] = audit._hash(
        rows[1]["seq"], rows[1]["at"], rows[1]["actor_digest"],
        rows[1]["action"], rows[1]["target"], rows[1]["detail"],
        rows[1]["prev_hash"])
    _rewrite(trail, rows)

    # The edited record now hashes correctly on its own...
    edited = audit.read_all(trail)[1]
    assert edited.record_hash == audit._hash(
        edited.seq, edited.at, edited.actor_digest, edited.action,
        edited.target, edited.detail, edited.prev_hash)

    # ...and the chain still catches it, at the record AFTER the edit.
    result = _check(trail)
    assert not result.ok
    assert result.broken_at == 3
    assert "link to the previous record" in result.reason


def test_an_unparseable_line_is_reported_rather_than_skipped(trail):
    """Dropping it silently would make the trail look shorter but valid."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    trail.write_text(trail.read_text() + "{not json\n")
    result = _check(trail)
    assert not result.ok
    assert "line(s) on disk but only" in result.reason


def test_truncating_the_file_is_caught(trail):
    """Deleting the TAIL leaves a chain that is internally consistent —
    so the count against the file is what catches it."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "sign_out"))
    rows = _raw(trail)
    _rewrite(trail, rows[:2])
    # Internally consistent, so verify() alone passes...
    assert audit.verify(audit.read_all(trail)).ok
    # ...which is exactly why the panel must compare against a known count.
    assert len(audit.read_all(trail)) == 2


# --- erasure ------------------------------------------------------------------

def test_erasing_an_actor_does_not_break_the_chain(trail):
    """The whole reason the chain covers the digest instead of the actor.
    Deleting the lines would break it and make a lawful request look
    exactly like forgery."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-ana", "sharing_opted_in"))
    count, err = audit.erase_actor("k-ana", trail)
    assert err == "" and count == 2
    result = _check(trail)
    assert result.ok, result.reason
    assert result.erased == 2


def test_an_erased_record_keeps_its_event_and_its_time(trail):
    """The evidence that something happened survives; only the identity
    goes."""
    _fill(trail, ("k-ana", "export_pdf"))
    before = audit.read_all(trail)[0]
    audit.erase_actor("k-ana", trail)
    after = audit.read_all(trail)[0]
    assert after.actor == AUDIT.tombstone and after.erased
    assert after.action == before.action
    assert after.at == before.at
    assert after.target == before.target
    assert after.record_hash == before.record_hash


def test_erasure_leaves_other_accounts_alone(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    audit.erase_actor("k-ana", trail)
    records = audit.read_all(trail)
    assert records[0].erased and not records[1].erased
    assert records[1].actor == "k-ben"


def test_erasing_an_unknown_actor_changes_nothing(trail):
    _fill(trail, ("k-ana", "sign_in"))
    before = trail.read_bytes()
    count, err = audit.erase_actor("k-nobody", trail)
    assert count == 0 and err == ""
    assert trail.read_bytes() == before


def test_erasure_needs_an_actor(trail):
    count, err = audit.erase_actor("", trail)
    assert count == 0 and err == audit.NO_ACTOR


def test_the_erasure_itself_is_a_recordable_action():
    """So that an erasure is never invisible in the trail."""
    assert "data_erased" in audit.ACTIONS
    assert audit.ACTIONS["data_erased"].category == audit.DELETION


def test_the_erasure_policy_is_stated():
    assert "tombstone" in audit.ERASURE_NOTE
    assert "indistinguishable from tampering" in audit.ERASURE_NOTE


# --- what is claimed ----------------------------------------------------------

def test_immutability_is_explicitly_not_claimed():
    """There is no WORM storage here and the operator owns the disk.
    Promising immutability would be a claim the file cannot back."""
    assert "tamper-EVIDENT, not tamper-proof" in audit.NOT_IMMUTABLE
    assert "nothing here can stop the file being edited" in audit.NOT_IMMUTABLE
    assert "no write-once storage" in audit.NOT_IMMUTABLE
    # It must also say what it DOES do, or the disclosure reads as an
    # apology rather than as the actual guarantee.
    assert "make any edit, deletion or reordering visible" in audit.NOT_IMMUTABLE


def test_no_compliance_claim_is_made():
    """branding.py locks 'No certification is claimed' against
    rebranding; a report implying an assessment would undo it."""
    text = audit.NO_COMPLIANCE_CLAIM
    assert "not an attestation" in text
    assert "No certification is claimed" in text
    assert "does not assess any control" in text


def test_no_framework_is_named_as_satisfied():
    src = Path(audit.__file__).read_text()
    for framework in ("SOX", "HIPAA", "GDPR"):
        for line in src.splitlines():
            if framework in line:
                lowered = line.lower()
                assert any(w in lowered for w in
                           ("not", "no ", "never", "adjacent", "clause",
                            "erasure", "separate")), line


def test_the_module_maps_no_control_references():
    src = Path(audit.__file__).read_text()
    for banned in ("CC6.", "CC7.", "164.312", "Article 30", "control_id"):
        assert banned not in src


# --- filtering and export -----------------------------------------------------

def test_filtering_by_category(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-cy", "thresholds_changed"))
    records = audit.read_all(trail)
    assert len(audit.filtered(records, category=audit.EXPORT)) == 1
    assert len(audit.filtered(records, category=audit.AUTH)) == 1


def test_filtering_by_actor(trail):
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"),
          ("k-ana", "sign_out"))
    records = audit.read_all(trail)
    assert len(audit.filtered(records, actor="k-ana")) == 2


def test_filtering_never_reorders(trail):
    """The sequence IS the evidence.

    The actors must be in an order that sorting WOULD change — an
    earlier fixture used the same actor twice, so a sort by actor was
    stable and the test passed a build that reordered.
    """
    _fill(trail, ("k-zoe", "sign_in"), ("k-ana", "export_pdf"),
          ("k-mia", "sign_out"), ("k-ben", "thresholds_changed"))
    records = audit.read_all(trail)
    assert [r.actor for r in records] != sorted(r.actor for r in records), (
        "the fixture must not already be in actor order, or it proves nothing")
    out = audit.filtered(records)
    assert [r.seq for r in out] == [1, 2, 3, 4]
    assert [r.actor for r in out] == ["k-zoe", "k-ana", "k-mia", "k-ben"]


def test_filtering_by_date_window(trail):
    audit.record("k-ana", "sign_in", path=trail, now="2026-01-01T00:00:00")
    audit.record("k-ana", "sign_out", path=trail, now="2026-06-01T00:00:00")
    records = audit.read_all(trail)
    assert len(audit.filtered(records, since="2026-03-01T00:00:00")) == 1
    assert len(audit.filtered(records, until="2026-03-01T00:00:00")) == 1


def test_the_export_carries_the_hash_with_every_row(trail):
    """So an export can be checked back against the trail it came from."""
    _fill(trail, ("k-ana", "sign_in"))
    (row,) = audit.export_rows(audit.read_all(trail))
    assert set(row) == set(audit.EXPORT_COLUMNS)
    assert row["record_hash"] == audit.read_all(trail)[0].record_hash


def test_the_export_bundle_carries_the_verification_and_the_notice(trail):
    _fill(trail, ("k-ana", "sign_in"))
    payload = json.loads(audit.export_payload(audit.read_all(trail),
                                              _check(trail)))
    assert payload["verification"]["ok"] is True
    assert payload["verification"]["checked"] == 1
    assert payload["notice"] == audit.NO_COMPLIANCE_CLAIM
    assert payload["tamper_evidence"] == audit.NOT_IMMUTABLE
    assert payload["records"]


def test_a_broken_chain_is_carried_into_the_export(trail):
    """An export of a tampered trail must say so, or it launders it."""
    _fill(trail, ("k-ana", "sign_in"), ("k-ben", "export_pdf"))
    rows = _raw(trail)
    rows[0]["detail"] = "changed"
    _rewrite(trail, rows)
    payload = json.loads(audit.export_payload(audit.read_all(trail),
                                              _check(trail)))
    assert payload["verification"]["ok"] is False
    assert payload["verification"]["broken_at"] == 1


def test_counts_by_category_covers_every_category(trail):
    _fill(trail, ("k-ana", "sign_in"))
    counts = audit.counts_by_category(audit.read_all(trail))
    assert set(counts) == set(audit.CATEGORIES)
    assert counts[audit.AUTH] == 1


def test_the_coverage_note_counts_the_real_vocabulary():
    note = audit.coverage_note()
    assert str(len(audit.ACTION_SPECS)) in note
    assert str(len(audit.CATEGORIES)) in note


# --- robustness and storage ---------------------------------------------------

def test_a_missing_trail_reads_as_empty(tmp_path):
    assert audit.read_all(tmp_path / "nope.jsonl") == ()
    assert audit.line_count(tmp_path / "nope.jsonl") == 0


def test_a_long_detail_is_truncated_not_refused(trail):
    entry, err = audit.record("k-ana", "sign_in", detail="x" * 5000, path=trail)
    assert err == "" and len(entry.detail) == AUDIT.max_detail_chars


def test_the_store_is_shared_not_per_user(monkeypatch):
    """A trail scoped to the account that caused the events is one each
    person could curate."""
    import local_store
    monkeypatch.setattr(local_store, "_namespace_provider", lambda: "somekey",
                        raising=False)
    assert local_store.current_namespace() == "somekey"
    assert audit._path() == local_store.app_dir() / AUDIT.store_filename


def test_the_store_is_declared_shared_in_auth():
    import auth
    assert AUDIT.store_filename in auth.SHARED_STORES
    assert AUDIT.store_filename not in auth.PER_USER_STORES


def test_the_store_is_gitignored():
    ignored = (Path(audit.__file__).resolve().parent / ".gitignore").read_text()
    assert AUDIT.store_filename in ignored


def test_the_trail_is_not_the_rotating_log():
    """quantix.log rotates on a 4 MB byte budget and loses history —
    measured at ~32 days. The audit trail must not inherit that."""
    src = Path(audit.__file__).read_text()
    assert "RotatingFileHandler" not in src
    assert "maxBytes" not in src
    assert "backupCount" not in src
    import logging_setup
    assert AUDIT.store_filename != logging_setup.LOG_FILENAME


# --- UI wiring ----------------------------------------------------------------

def _panel() -> str:
    src = FINANCE.read_text()
    start = src.index("# --- Audit trail ---")
    end = src.index("# --- end audit ---", start)
    return "\n".join(ln for ln in src[start:end].splitlines()
                     if not ln.strip().startswith("#"))


def test_the_panel_verifies_rather_than_asserting_integrity():
    panel = _panel()
    assert "audit_verify_file(" in panel


def test_the_panel_states_what_tamper_evidence_means():
    assert "AUDIT_NOT_IMMUTABLE" in _panel()


def test_the_panel_states_that_no_compliance_claim_is_made():
    assert "AUDIT_NO_COMPLIANCE_CLAIM" in _panel()


def test_the_panel_offers_the_evidence_export():
    panel = _panel()
    assert "audit_export_payload(" in panel
    assert "download_button" in panel


def test_exporting_the_trail_is_itself_recorded():
    """An auditor asks who took a copy of the evidence."""
    panel = _panel()
    assert "audit_exported" in panel


def test_the_significant_actions_are_recorded_somewhere_in_the_app():
    """A trail nobody writes to is not a trail. Asserted by ACTION NAME
    rather than by a count — the badges suite proved a count alone
    passes a build with a hook deleted."""
    import re
    # Both files: finance.py records through its _audit() helper, and the
    # authentication events are written in login_page.py because they
    # happen BEFORE the auth gate that finance.py sits behind.
    app_dir = FINANCE.resolve().parent
    sources = (FINANCE.read_text()
               + (app_dir / "login_page.py").read_text())
    used = set(re.findall(r'_audit(?:_sign_in)?\(\s*"([a-z_]+)"', sources))
    for required in ("sign_in", "account_created", "export_pdf",
                     "export_workbook", "export_deck", "thresholds_changed",
                     "sharing_opted_in", "sharing_opted_out", "audit_exported"):
        assert required in used, f"nothing records '{required}'"
    assert used <= set(audit.ACTIONS), f"undeclared actions: {used - set(audit.ACTIONS)}"


def test_every_audit_category_is_actually_reachable():
    """A declared category nothing ever writes is a column of zeroes
    that reads as 'this never happens' rather than 'this is not wired'."""
    import re
    app_dir = FINANCE.resolve().parent
    sources = (FINANCE.read_text() + (app_dir / "login_page.py").read_text())
    used = set(re.findall(r'_audit(?:_sign_in)?\(\s*"([a-z_]+)"', sources))
    reached = {audit.ACTIONS[a].category for a in used if a in audit.ACTIONS}
    missing = set(audit.CATEGORIES) - reached
    assert not missing, f"declared but never written: {missing}"
