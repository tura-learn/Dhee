"""Forgetting depends on time, not on how often the decay pass runs.

`calculate_decayed_strength` fades the CURRENT strength by the time since the
memory was last accessed. Applied once, that is right. Applied again on the
next pass, it fades the already-faded strength by the whole elapsed time again,
so the result compounds with every run. Tura runs the pass every five minutes;
measured with this module's own function, a memory meant to last 115 days was
forgotten in under one, and on production (2026-10-07) none of the 12 students
who had studied in the previous fortnight had a single memory left.
"""

import math
import os
from datetime import datetime, timedelta, timezone

import pytest

from dhee import CoreMemory

def _ago(days):
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _memory(factory, data_dir):
    # A store of its own: a shared one dedupes the text and keeps old DECAY events.
    os.environ["DHEE_DATA_DIR"] = str(data_dir)
    if factory == "core":
        return CoreMemory(preset="minimal")
    from dhee.memory.main import FullMemory

    return FullMemory(preset="minimal")


def _aged(memory, text, days):
    try:
        added = memory.add(text, user_id="student", infer=False)
    except TypeError:  # CoreMemory never infers, and takes no such argument
        added = memory.add(text, user_id="student")
    memory_id = added["results"][0]["id"]
    memory.db.update_memory(memory_id, {"last_accessed": _ago(days), "strength": 1.0})
    return memory_id


def _strength_after(tmp_path, factory, passes, days):
    memory = _memory(factory, tmp_path / f"{factory}-{passes}-{days}")
    try:
        memory_id = _aged(memory, "Prefers derivations before worked examples", days)
        for _ in range(passes):
            memory.apply_decay(scope={"user_id": "student"})
        row = memory.db.get_memory(memory_id)
        return None if row is None or row.get("tombstone") else float(row["strength"])
    finally:
        memory.close()


@pytest.mark.parametrize("factory", ["core", "full"])
def test_thirty_passes_fade_a_memory_as_much_as_one(tmp_path, monkeypatch, factory):
    monkeypatch.setenv("DHEE_DATA_DIR", str(tmp_path))
    # Old enough to fade, young enough that one pass keeps it.
    once = _strength_after(tmp_path, factory, 1, days=2)
    assert once is not None and once < 1.0, "the fixture must fade, and survive one pass"
    thirty = _strength_after(tmp_path, factory, 30, days=2)
    assert thirty is not None, "thirty passes forgot what one pass keeps"
    assert math.isclose(thirty, once, abs_tol=0.01), f"one pass leaves {once:.3f}, thirty leave {thirty:.3f}"


def test_a_memory_still_fades_with_time(tmp_path, monkeypatch):
    """The fix must not stop forgetting: a pass long after the last one fades it."""
    monkeypatch.setenv("DHEE_DATA_DIR", str(tmp_path))
    assert (_strength_after(tmp_path, "core", 1, days=2) or 0.0) < 1.0
    assert (_strength_after(tmp_path, "full", 1, days=2) or 0.0) < 1.0


def _hold_fact(memory, memory_id, value="21 October"):
    """A fact extracted from `memory_id`, as the extractor would have stored it."""
    import uuid

    fact_id = str(uuid.uuid4())
    with memory.db._get_connection() as conn:
        conn.execute(
            "INSERT INTO engram_facts (id, memory_id, subject, predicate, value, canonical_key) "
            "VALUES (?, ?, 'user', 'exam_on', ?, ?)",
            (fact_id, memory_id, f"computational stats: {value}", f"user|exam_on|{value}"),
        )
    return fact_id


def _facts_from(memory, memory_id):
    with memory.db._get_connection() as conn:
        return conn.execute(
            "SELECT count(*) FROM engram_facts WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]


@pytest.mark.parametrize("factory", ["core", "full"])
def test_a_memory_holding_a_current_fact_is_not_forgotten(tmp_path, monkeypatch, factory):
    """Forgetting a memory deletes its facts; decay must not take a current one.

    Measured on production 2026-09-30: a student's "exam date is 21 october"
    was extracted and kept by the gate, then deleted with its conversation
    when decay forgot it. One of 38 student stores held any fact a week later.
    """
    monkeypatch.setenv("DHEE_DATA_DIR", str(tmp_path))
    memory = _memory(factory, tmp_path / factory)
    try:
        memory_id = _aged(memory, "The subject is computational stats and the exam is on 21 October", 2000)
        _hold_fact(memory, memory_id)
        memory.apply_decay(scope={"user_id": "student"})
        row = memory.db.get_memory(memory_id)
        assert row is not None and not row.get("tombstone"), "a memory holding a current fact was forgotten"
        assert _facts_from(memory, memory_id) == 1
        assert float(row["strength"]) == pytest.approx(memory.fade_config.forgetting_threshold)

        # Once the fact is replaced, the memory fades like any other. Time
        # passes for the last decay too: the clock now starts there.
        with memory.db._get_connection() as conn:
            conn.execute("UPDATE engram_facts SET superseded_by_id = 'newer' WHERE memory_id = ?", (memory_id,))
            conn.execute("UPDATE memory_history SET timestamp = ? WHERE memory_id = ?", (_ago(2000), memory_id))
        memory.apply_decay(scope={"user_id": "student"})
        row = memory.db.get_memory(memory_id)
        assert row is None or row.get("tombstone"), "a memory whose facts were replaced was kept"
    finally:
        memory.close()


def test_a_memory_with_no_facts_is_still_forgotten(tmp_path, monkeypatch):
    monkeypatch.setenv("DHEE_DATA_DIR", str(tmp_path))
    memory = _memory("full", tmp_path / "plain")
    try:
        memory_id = _aged(memory, "Said hello and left", 2000)
        memory.apply_decay(scope={"user_id": "student"})
        row = memory.db.get_memory(memory_id)
        assert row is None or row.get("tombstone")
    finally:
        memory.close()
