"""Tests for local_store.py — the shared atomic-write helper every local
JSON store in this app (onboarding, theme, watchlists, scenarios, ML
training history, real-time/risk alert rules, ...) uses.

The concurrency test reproduces, and confirms the fix for, a real crash
seen in production: FileNotFoundError from tmp.replace(path) when two
writers to the same store raced on a temp filename that used to be
fixed (path.with_suffix(".tmp")) rather than unique per write.
"""
import json
import threading

from local_store import atomic_write_text


def test_write_then_read_round_trips(tmp_path):
    path = tmp_path / "store.json"
    atomic_write_text(path, json.dumps({"a": 1}))
    assert json.loads(path.read_text()) == {"a": 1}


def test_overwrites_existing_file(tmp_path):
    path = tmp_path / "store.json"
    atomic_write_text(path, json.dumps({"a": 1}))
    atomic_write_text(path, json.dumps({"a": 2}))
    assert json.loads(path.read_text()) == {"a": 2}


def test_no_leftover_tmp_file_after_a_normal_write(tmp_path):
    path = tmp_path / "store.json"
    atomic_write_text(path, "{}")
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []


def test_concurrent_writers_to_the_same_path_never_crash(tmp_path):
    """Regression test for the exact production crash: FileNotFoundError
    from two writers racing on a shared, fixed temp filename. Twelve
    threads hammering the same store concurrently must all succeed, the
    file must end up containing valid JSON from one of them (whichever
    replace() ran last), and no orphaned temp file should be left behind."""
    path = tmp_path / "store.json"
    errors = []

    def writer(n):
        try:
            for i in range(50):
                atomic_write_text(path, json.dumps({"writer": n, "i": i}))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert json.loads(path.read_text())  # valid, non-empty JSON from some writer
    leftovers = [p for p in tmp_path.iterdir() if p != path]
    assert leftovers == []
