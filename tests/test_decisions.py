"""The decision layer: the fact gate, the read gate, the Jev client, and the
distillation ledger that stops a sleep cycle re-distilling the same day.

Decisions come from a scripted decider here; what is pinned is what the
pipeline does with them — and that every missing or failed decision leaves
behaviour exactly as it was without one.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dhee.configs.base import DecisionConfig, DistillationConfig
from dhee.core.distillation import ReplayDistiller
from dhee.core.resolvers import ContextResolver
from dhee.db.sqlite import FullSQLiteManager, SQLiteManager
from dhee.decisions import (
    FactGate,
    JevDecider,
    decider_from_config,
    fact_gate_from_config,
    select_relevant,
)
from dhee.decisions.facts import NEW_FACT, NONE_OF_THESE

VOCABULARY = {
    "finds_hard": {"what": "a topic or skill the person struggles with", "many": True},
    "preparing_for": "an exam or goal they are working towards",
    "prefers": {"what": "how they like to be taught or to work", "many": True},
}


class Scripted:
    """A decider that answers from a function of the question id."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        if self.answer is None:
            return None
        return {qid: self.answer(qid, q) for qid, q in questions.items()}


def choice(best, p=0.9, rest=()):
    probabilities = {best: p}
    for other in rest:
        probabilities[other] = round((1 - p) / max(1, len(rest)), 4)
    return {"type": "choice", "choice": best, "confidence": p, "probabilities": probabilities}


def fact(subject, predicate, value):
    return SimpleNamespace(
        subject=subject,
        predicate=predicate,
        value=value,
        value_numeric=None,
        value_unit=None,
        time=None,
        valid_from=None,
        valid_until=None,
        qualifier=None,
        canonical_key="",
        confidence=1.0,
        is_derived=False,
    )


# ----- the fact gate ---------------------------------------------------------


def test_off_subject_facts_are_dropped_and_labels_join_the_vocabulary():
    def answer(qid, q):
        if qid == "about_0":
            return {"noul": 0.94}
        if qid == "about_1":
            return {"noul": 0.06}  # "rest | is | relative" — textbook, not the student
        if qid == "label_0":
            return choice("finds_hard", 0.91, [NONE_OF_THESE])
        return choice(NONE_OF_THESE, 0.8, ["finds_hard"])

    gate = FactGate(Scripted(answer), vocabulary=VOCABULARY, subjects=["the student"])
    facts = [fact("Student", "struggling_with", "tension in strings"), fact("rest", "is", "relative")]
    kept, report = gate.apply("…", facts)
    assert [(f.subject, f.predicate, f.value) for f in kept] == [("user", "finds_hard", "tension in strings")]
    assert report.dropped_off_subject == 1 and report.relabelled == 1 and report.decided


def test_a_fact_that_fits_no_relation_is_not_stored_when_a_vocabulary_is_set():
    gate = FactGate(
        Scripted(lambda qid, q: {"noul": 0.9} if qid.startswith("about") else choice(NONE_OF_THESE, 0.9, ["prefers"])),
        vocabulary=VOCABULARY,
        subjects=["the student"],
    )
    kept, report = gate.apply("…", [fact("user", "tracks", "correct answers out of attempts")])
    assert kept == [] and report.dropped_unlabelled == 1


def test_an_unsure_label_keeps_a_vocabulary_word_the_extractor_already_used():
    gate = FactGate(
        Scripted(
            lambda qid, q: {"noul": 0.9}
            if qid.startswith("about")
            else {"choice": "prefers", "probabilities": {"prefers": 0.4, "finds_hard": 0.35, NONE_OF_THESE: 0.25}}
        ),
        vocabulary=VOCABULARY,
        subjects=["the student"],
    )
    kept, _ = gate.apply("…", [fact("user", "preparing_for", "JEE Main")])
    assert [f.predicate for f in kept] == ["preparing_for"]


def test_no_decision_means_the_facts_are_stored_exactly_as_extracted():
    gate = FactGate(Scripted(None), vocabulary=VOCABULARY, subjects=["the student"])
    facts = [fact("rest", "is", "relative")]
    kept, report = gate.apply("…", facts)
    assert kept == facts and not report.decided


def test_a_restated_fact_points_at_the_stored_one():
    def answer(qid, q):
        if qid.startswith("about"):
            return {"noul": 0.9}
        if qid.startswith("label"):
            return choice("finds_hard", 0.9, [NONE_OF_THESE])
        return choice("stored_1", 0.88, [NEW_FACT, "stored_0"])

    stored = [("vectors", "user|finds_hard|vectors"), ("tension in strings", "user|finds_hard|tension in strings")]
    gate = FactGate(Scripted(answer), vocabulary=VOCABULARY, subjects=["the student"])
    kept, report = gate.apply(
        "…",
        [fact("user", "has_difficulty_with", "string tension problems")],
        existing=lambda s, p: stored,
    )
    assert kept[0].value == "tension in strings"
    assert kept[0].canonical_key == "user|finds_hard|tension in strings"
    assert report.merged == 1


def test_a_single_valued_predicate_is_updated_not_merged():
    """ "class 12" is not a restatement of "class 11"; it replaces it."""
    decider = Scripted(lambda qid, q: {"noul": 0.9} if not qid.startswith("label") else choice("preparing_for", 0.9))
    gate = FactGate(decider, vocabulary=VOCABULARY, subjects=["the student"])
    kept, report = gate.apply(
        "…", [fact("user", "preparing_for", "JEE Advanced")], existing=lambda s, p: [("JEE Main", "k")]
    )
    assert kept[0].value == "JEE Advanced" and report.merged == 0
    assert not any(qid.startswith("same") for qid in decider.calls[-1][1])


def test_identical_values_merge_without_asking():
    decider = Scripted(lambda qid, q: {"noul": 0.9} if qid.startswith("about") else choice("finds_hard", 0.9))
    gate = FactGate(decider, vocabulary=VOCABULARY, subjects=["the student"])
    kept, _ = gate.apply(
        "…",
        [fact("user", "finds_hard", "Vectors.")],
        existing=lambda s, p: [("vectors", "user|finds_hard|vectors")],
    )
    assert kept[0].value == "vectors"
    assert len(decider.calls) == 1, "no second call for an exact match"


def test_the_gate_and_storage_together_reaffirm_instead_of_duplicating():
    """The measured bug, end to end: eight copies of one fact."""
    tmp = tempfile.mkdtemp()
    db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    with db._get_connection() as conn:
        for mid in ("m1", "m2"):
            conn.execute("INSERT INTO memories (id, memory, user_id) VALUES (?, ?, ?)", (mid, mid, "u1"))
    resolver = ContextResolver(db)

    def answer(qid, q):
        if qid.startswith("about"):
            return {"noul": 0.95}
        if qid.startswith("label"):
            return choice("finds_hard", 0.93, [NONE_OF_THESE])
        return choice("stored_0", 0.9, [NEW_FACT])

    gate = FactGate(Scripted(answer), vocabulary=VOCABULARY, subjects=["the student"])

    def existing(subject, predicate):
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT value, canonical_key FROM engram_facts WHERE subject=? AND predicate=? "
                "AND superseded_by_id IS NULL",
                (subject, predicate),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    for memory_id, said in (("m1", "tension in strings"), ("m2", "tension in strings (physics concept)")):
        kept, _ = gate.apply("…", [fact("user", "struggling_with", said)], existing)
        resolver.store_engram(_engram(kept), memory_id)

    with db._get_connection() as conn:
        rows = conn.execute("SELECT value, reaffirmed_count FROM engram_facts").fetchall()
    assert [(r[0], r[1]) for r in rows] == [("tension in strings", 1)]


def _engram(facts):
    return SimpleNamespace(
        context=SimpleNamespace(has_context=lambda: False),
        scene=SimpleNamespace(setting=None, people_present=[]),
        facts=facts,
        entities=[],
        links=[],
    )


def test_a_gate_is_built_only_from_an_enabled_config_with_a_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert decider_from_config(DecisionConfig()) is None
    assert decider_from_config(DecisionConfig(enabled=True)) is None, "no key, no decider"
    decider = decider_from_config(DecisionConfig(enabled=True, api_key="sk-test"))
    assert isinstance(decider, JevDecider)
    assert fact_gate_from_config(DecisionConfig(enabled=True, fact_vocabulary=VOCABULARY), decider)
    monkeypatch.setenv("DHEE_DECISIONS", "off")
    assert decider_from_config(DecisionConfig(enabled=True, api_key="sk-test")) is None


# ----- the read gate ---------------------------------------------------------


def test_only_what_helps_this_turn_is_selected_most_useful_first():
    scores = {"m0": 0.2, "m1": 0.91, "m2": 0.66, "m3": 0.05}
    chosen = select_relevant(
        Scripted(lambda qid, q: {"noul": scores[qid]}),
        "why does the string tension change?",
        ["exam on 30 October", "finds tension in strings hard", "prefers step-by-step", "likes cricket"],
        floor=0.5,
    )
    assert [c.text for c in chosen] == ["finds tension in strings hard", "prefers step-by-step"]


def test_a_failed_read_decision_is_none_not_an_empty_selection():
    assert select_relevant(Scripted(None), "q", ["a"]) is None
    assert select_relevant(None, "q", ["a"]) is None
    assert select_relevant(Scripted(lambda qid, q: {"noul": 0.9}), "q", ["", "  "]) == []


# ----- the client ------------------------------------------------------------


def test_the_client_never_raises_and_chunks_large_batches(monkeypatch):
    import requests

    sent = []

    class Response:
        status_code = 200

        def __init__(self, questions):
            self._questions = questions

        def json(self):
            return {"answers": {qid: {"noul": 0.5} for qid in self._questions}}

    def post(url, json=None, headers=None, timeout=None):
        sent.append(len(json["questions"]))
        return Response(json["questions"])

    monkeypatch.setattr(requests, "post", post)
    decider = JevDecider(api_key="sk-test")
    answers = decider.decide({"x": 1}, {f"q{i}": {"type": "noul"} for i in range(100)})
    assert len(answers) == 100 and sent == [48, 48, 4]

    def broken(*args, **kwargs):
        raise ConnectionError("down")

    monkeypatch.setattr(requests, "post", broken)
    assert decider.decide({}, {"q": {"type": "noul"}}) is None
    assert decider.stats["failures"] == 1


# ----- distillation ----------------------------------------------------------


def test_a_day_is_distilled_once_however_often_the_cycle_runs():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = SQLiteManager(path)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    for i in range(5):
        now = f"{yesterday}T1{i}:00:00"
        db.add_memory({"memory": f"episode {i}", "user_id": "u", "memory_type": "episodic",
                       "created_at": now, "updated_at": now, "layer": "sml", "strength": 1.0})
    llm = MagicMock()
    llm.generate.return_value = json.dumps({"semantic_facts": [{"content": "fact", "importance": "high"}]})
    add = MagicMock(return_value={"results": [{"id": "sem", "event": "ADD"}]})
    config = DistillationConfig(enable_distillation=True, distillation_batch_size=10,
                                distillation_min_episodes=2, max_semantic_per_batch=3)
    distiller = ReplayDistiller(db, llm, config)

    first = distiller.run("u", date_str=yesterday, memory_add_fn=add)
    second = distiller.run("u", date_str=yesterday, memory_add_fn=add)

    assert first.get("skipped") is not True and first["semantic_created"] == 1
    assert second["skipped"] is True and second["reason"] == "insufficient new episodes"
    assert second["already_distilled"] == 5
    assert llm.generate.call_count == 1
    db.close()
    os.unlink(path)


# ----- regating a store that grew without the gate -----------------------------


def test_an_old_store_is_regated_once_and_nothing_is_deleted():
    from dhee.decisions.regate import regate_facts

    tmp = tempfile.mkdtemp()
    db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    with db._get_connection() as conn:
        for mid, text in (("m1", "struggled with tension"), ("m2", "tension again")):
            conn.execute("INSERT INTO memories (id, memory, user_id) VALUES (?, ?, ?)", (mid, text, "u"))
        rows = [
            ("f1", "m1", "user", "struggling_with", "tension in strings", "2026-09-01"),
            ("f2", "m1", "rest", "is", "relative", "2026-09-01"),
            ("f3", "m2", "user", "has_difficulty_with", "tension in a string", "2026-09-02"),
        ]
        for fid, mid, s, p, v, at in rows:
            conn.execute(
                "INSERT INTO engram_facts (id, memory_id, subject, predicate, value, canonical_key, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, mid, s, p, v, f"{s}|{p}|{v}", at),
            )

    def answer(qid, q):
        if qid.startswith("about"):
            return {"noul": 0.02 if "relative" in q["instructions"]["fact"] else 0.95}
        if qid.startswith("label"):
            return choice("finds_hard", 0.95, [NONE_OF_THESE])
        return choice("stored_0", 0.9, [NEW_FACT])

    gate = FactGate(Scripted(answer), vocabulary=VOCABULARY, subjects=["the student"])

    dry = regate_facts(db, gate, dry_run=True)
    assert (dry.retired, dry.relabelled, dry.merged) == (1, 1, 1)
    with db._get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM engram_facts WHERE predicate='finds_hard'").fetchone()[0] == 0

    regate_facts(db, gate, dry_run=False)
    with db._get_connection() as conn:
        facts = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM engram_facts").fetchall()}
    assert len(facts) == 3, "nothing is deleted"
    assert (facts["f1"]["predicate"], facts["f1"]["reaffirmed_count"]) == ("finds_hard", 1)
    assert facts["f2"]["valid_until"] and facts["f2"]["tier"] == "avoid"
    assert facts["f3"]["superseded_by_id"] == "f1"


def test_a_dated_fact_does_not_wipe_a_predicate_that_holds_many_values():
    """Measured on a student store: one dated "finds_hard: vectors" retired ten
    other difficulties, each "superseded by vectors"."""
    tmp = tempfile.mkdtemp()
    db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    with db._get_connection() as conn:
        for mid in ("m1", "m2"):
            conn.execute("INSERT INTO memories (id, memory, user_id) VALUES (?, ?, ?)", (mid, mid, "u"))
    resolver = ContextResolver(db)
    vocabulary = {
        "finds_hard": {"what": "something they struggle with", "many": True},
        "prefers": {"what": "how they like to learn", "many": True},
        "in_class": "their class",
    }
    labels = {"tension": "finds_hard", "vectors": "finds_hard", "11": "in_class", "12": "in_class",
              "short answers": "prefers", "worked examples": "prefers"}

    def answer(qid, q):
        if qid.startswith("about"):
            return {"noul": 0.95}
        if qid.startswith("label"):
            value = q["instructions"]["fact"].split(" | ")[-1]
            return choice(labels[value], 0.95, [NONE_OF_THESE])
        return choice(NEW_FACT, 0.95)

    gate = FactGate(Scripted(answer), vocabulary=vocabulary, subjects=["the student"])
    for memory_id, facts in (
        ("m1", [fact("user", "struggles_with", "tension"), fact("user", "class", "11"),
                fact("user", "likes", "short answers")]),
        ("m2", [fact("user", "struggles_with", "vectors"), fact("user", "class", "12"),
                fact("user", "likes", "worked examples")]),
    ):
        for f in facts:
            f.valid_from = "2026-09-2" + memory_id[-1]
        kept, _ = gate.apply("…", facts)
        resolver.store_engram(_engram(kept), memory_id)

    with db._get_connection() as conn:
        active = conn.execute(
            "SELECT predicate, value FROM engram_facts WHERE superseded_by_id IS NULL ORDER BY predicate, value"
        ).fetchall()
    assert [(r[0], r[1]) for r in active] == [
        ("finds_hard", "tension"),
        ("finds_hard", "vectors"),
        ("in_class", "12"),
        # `prefers` is on the built-in single-valued list; `many` still wins.
        ("prefers", "short answers"),
        ("prefers", "worked examples"),
    ]


def test_one_slot_per_idea_but_a_shared_label_is_not_a_shared_idea():
    lines = [
        "What they find hard: vectors",
        "What they find hard: resolving components",
        "Has difficulty with: resolving components into x and y",
    ]
    chosen = select_relevant(Scripted(lambda qid, q: {"noul": 0.9}), "signs of components", lines, limit=5)
    assert [c.text for c in chosen] == lines[:2]


def test_getting_better_at_something_retires_finding_it_hard():
    """Measured on a simulated term: "vectors are fine now" left
    `finds_hard: vectors` standing beside `finds_easy: vectors`."""
    tmp = tempfile.mkdtemp()
    db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    with db._get_connection() as conn:
        for mid in ("m1", "m2"):
            conn.execute("INSERT INTO memories (id, memory, user_id) VALUES (?, ?, ?)", (mid, mid, "u"))
    resolver = ContextResolver(db)
    vocabulary = {
        "finds_hard": {"what": "hard for them", "many": True, "retires": "finds_easy"},
        "finds_easy": {"what": "clicked", "many": True, "retires": "finds_hard"},
    }

    def answer(qid, q):
        if qid.startswith("about"):
            return {"noul": 0.95}
        if qid.startswith("label"):
            said = q["instructions"]["fact"]
            return choice("finds_easy" if "fine" in said else "finds_hard", 0.95, [NONE_OF_THESE])
        if qid.startswith("undo"):
            return {"noul": 0.92 if "vectors" in q["instructions"]["stored_fact"] else 0.05}
        return choice(NEW_FACT, 0.95)

    gate = FactGate(Scripted(answer), vocabulary=vocabulary, subjects=["the student"])

    def existing(subject, predicate):
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT value, canonical_key FROM engram_facts WHERE subject=? AND predicate=? "
                "AND superseded_by_id IS NULL",
                (subject, predicate),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    kept, _ = gate.apply(
        "…",
        [fact("user", "struggles_with", "vectors"), fact("user", "struggles_with", "friction"),
         fact("user", "struggles_with", "vectors (resolving into components)")],
        existing,
    )
    resolver.store_engram(_engram(kept), "m1")
    kept, report = gate.apply("…", [fact("user", "fine_with", "vector components")], existing)
    resolver.store_engram(_engram(kept), "m2")

    with db._get_connection() as conn:
        active = conn.execute(
            "SELECT predicate, value FROM engram_facts WHERE superseded_by_id IS NULL ORDER BY predicate"
        ).fetchall()
    assert [(r[0], r[1]) for r in active] == [("finds_easy", "vector components"), ("finds_hard", "friction")]
    assert report.retiring == 1


def test_a_category_is_picked_by_decision_before_any_llm_is_asked():
    from dhee.core.category import CategoryProcessor

    llm = MagicMock()
    processor = CategoryProcessor(llm=llm, embedder=None, config={"use_llm": True})
    assert processor.categories, "the default categories load"
    target = next(iter(processor.categories))
    processor.decider_fn = lambda: Scripted(lambda qid, q: choice(target, 0.92, ["new_category"]))
    match = processor.detect_category("zzqx unmatched words", use_llm=True)
    assert match.category_id == target
    llm.generate.assert_not_called()

    # "None fits" hands the case to the LLM, which can name a new category.
    processor.decider_fn = lambda: Scripted(lambda qid, q: choice("new_category", 0.9, [target]))
    llm.generate.return_value = "{}"
    processor.detect_category("zzqx unmatched words", use_llm=True)
    llm.generate.assert_called()


def test_a_memory_is_extracted_once_whoever_asks_again():
    """The enrichment pass re-extracted memories the write had already
    extracted: twice the cost, and a paraphrased duplicate each time."""
    from dhee.memory.write_pipeline import MemoryWritePipeline as WritePipeline

    tmp = tempfile.mkdtemp()
    pipeline = WritePipeline.__new__(WritePipeline)
    pipeline._db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    calls = []

    class Extractor:
        def extract(self, **kwargs):
            calls.append(kwargs["content"])
            return SimpleNamespace(facts=[], prospective_scenes=[])

    pipeline._engram_extractor_fn = lambda: Extractor()
    pipeline._config = SimpleNamespace(decisions=None, prospective_scene=SimpleNamespace(enable_prospective_scenes=False))
    pipeline._context_resolver_fn = None
    for _ in range(3):
        attempted, succeeded, _ = pipeline._run_engram_extraction(
            memory_id="m1", content="I find vectors hard", mem_metadata={}, user_id="u"
        )
        assert attempted and succeeded
    assert calls == ["I find vectors hard"]


def test_a_fact_filed_under_a_thing_is_asked_again_as_the_persons():
    """ "JEE Main | scheduled_in | April": the subject is the exam, the fact is
    the student's exam date. Measured: about-the-person 0.04, label exam_on."""
    def answer(qid, q):
        if qid.startswith(("about", "support")):
            return {"noul": 0.04}
        if qid.startswith("label"):
            return choice("exam_on", 0.9, [NONE_OF_THESE])
        if qid.startswith("again"):
            assert q["instructions"]["fact"] == "user | exam_on | April"
            return {"noul": 0.9}
        return choice(NEW_FACT, 0.9)

    vocabulary = dict(VOCABULARY, exam_on="the date of their exam")
    gate = FactGate(Scripted(answer), vocabulary=vocabulary, subjects=["the student"])
    kept, report = gate.apply("my JEE got moved to April", [fact("JEE Main", "scheduled_in", "April")])
    assert [(f.subject, f.predicate, f.value) for f in kept] == [("user", "exam_on", "April")]


def test_one_date_per_exam_not_one_date_per_student():
    """Measured over a simulated hundred days: `exam_on` held one value, so the
    JEE date (April) replaced the boards date (March)."""
    tmp = tempfile.mkdtemp()
    db = FullSQLiteManager(os.path.join(tmp, "t.db"))
    with db._get_connection() as conn:
        for mid in ("m1", "m2", "m3"):
            conn.execute("INSERT INTO memories (id, memory, user_id) VALUES (?, ?, ?)", (mid, mid, "u"))
    resolver = ContextResolver(db)
    vocabulary = {"exam_on": {"what": "the date of an exam, as '<exam>: <date>'", "many": True, "per": "exam"}}

    def answer(qid, q):
        if qid.startswith(("about", "support")):
            return {"noul": 0.95}
        if qid.startswith("label"):
            return choice("exam_on", 0.95, [NONE_OF_THESE])
        if qid.startswith("same"):
            new = q["instructions"]["new_value"].split(":")[0]
            for option, value in q["criteria"].items():
                if option.startswith("stored") and value.split(":")[0] == new:
                    return choice(option, 0.95, [NEW_FACT])
            return choice(NEW_FACT, 0.95)
        return {"noul": 0.05}

    gate = FactGate(Scripted(answer), vocabulary=vocabulary, subjects=["the student"])

    def existing(subject, predicate):
        with db._get_connection() as conn:
            rows = conn.execute(
                "SELECT value, canonical_key FROM engram_facts WHERE subject=? AND predicate=? "
                "AND superseded_by_id IS NULL",
                (subject, predicate),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    for memory_id, value in (("m1", "JEE Main: January"), ("m2", "CBSE boards: March"), ("m3", "JEE Main: April")):
        f = fact("user", "exam", value)
        f.valid_from = "2026-06-0" + memory_id[-1]
        kept, _ = gate.apply("…", [f], existing)
        resolver.store_engram(_engram(kept), memory_id)

    with db._get_connection() as conn:
        active = [r[0] for r in conn.execute(
            "SELECT value FROM engram_facts WHERE superseded_by_id IS NULL ORDER BY value").fetchall()]
    assert active == ["CBSE boards: March", "JEE Main: April"]
