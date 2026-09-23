"""The fact gate: three closed decisions between extraction and storage.

An extractor reads a passage and proposes facts. Before this gate, every one of
them was stored, and three things went wrong in production — measured on
Tura's student stores, 2026-09-23:

* **Facts that are not about the person.** 81 of 186 stored facts were
  general knowledge lifted from the text ("rest is relative"), bookkeeping
  ("quiz has_questions 2") or about the assistant.
* **Labels nobody can join on.** The application reads nine predicates; 2 of
  105 facts about the student used one of them. ``struggles_with``,
  ``struggling_with``, ``has_difficulty_with`` and ``finds_challenging`` all
  meant ``finds_hard`` and none of them reached a reader.
* **The same fact, stored again.** Reaffirmation matched value strings exactly,
  so "tension in strings" and "tension in strings (physics concept)" were two
  facts, and a week of study produced eight copies of one.

Each is a choice among known options, which is what a decision model answers:
a Noul for "about them?", a Choice over the vocabulary for the label, a Choice
over the values already stored for "restates which, or new?". The words stay
the extractor's; only the judgement moves.

Every step falls back to what the pipeline did before: a failed or disabled
decision leaves the facts exactly as extracted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from dhee.decisions.jev import Decider, noul, ranked

logger = logging.getLogger(__name__)

NONE_OF_THESE = "none_of_these"
NEW_FACT = "new_fact"

#: At most this many stored values are offered when judging "restates which?".
MAX_EXISTING = 12

#: How sure a retirement must be. Set from nine measured pairs: every
#: must-retire pair scored 0.66 or more, every must-not pair 0.05 or less, and
#: the one genuinely ambiguous pair (torque easy vs rotational motion hard) 0.40.
RETIRE_FLOOR = 0.6

#: How sure a label must be before a fact whose subject is a thing — "JEE Main |
#: scheduled_in | April" — is asked about again as a fact about the person.
RELABEL_SUBJECT_FLOOR = 0.8


@dataclass
class GateReport:
    """What the gate did to one extraction, for logs and tests."""

    kept: int = 0
    dropped_off_subject: int = 0
    relabelled: int = 0
    dropped_unlabelled: int = 0
    merged: int = 0
    retiring: int = 0
    decided: bool = False
    notes: List[str] = field(default_factory=list)


#: ``existing(subject, predicate) -> [(value, canonical_key), ...]``, newest first.
ExistingLookup = Callable[[str, str], List[Tuple[str, str]]]


class FactGate:
    def __init__(
        self,
        decider: Decider,
        *,
        vocabulary: Optional[Dict[str, Any]] = None,
        subjects: Sequence[str] = (),
        canonical_subject: str = "user",
        about_floor: float = 0.5,
        label_floor: float = 0.5,
        same_floor: float = 0.7,
    ) -> None:
        self.decider = decider
        self.vocabulary = dict(vocabulary or {})
        self.subjects = [s for s in subjects if str(s).strip()]
        self.canonical_subject = canonical_subject
        self.about_floor = about_floor
        self.label_floor = label_floor
        self.same_floor = same_floor

    # ------------------------------------------------------------------ api

    def apply(
        self,
        content: str,
        facts: List[Any],
        existing: Optional[ExistingLookup] = None,
    ) -> Tuple[List[Any], GateReport]:
        report = GateReport()
        facts = [f for f in facts if getattr(f, "subject", "") and getattr(f, "predicate", "") and getattr(f, "value", "")]
        if not facts:
            return facts, report

        judged = self._judge(content, facts)
        if judged is None:
            report.kept = len(facts)
            report.notes.append("decision unavailable; facts stored as extracted")
            return facts, report
        report.decided = True

        kept: List[Any] = []
        for index, fact in enumerate(facts):
            about, label = judged[index]
            if self.subjects and about is not None and about < self.about_floor:
                report.dropped_off_subject += 1
                continue
            if self.subjects:
                fact.subject = self.canonical_subject
            if self.vocabulary:
                if label is None:
                    report.dropped_unlabelled += 1
                    continue
                if fact.predicate != label:
                    report.relabelled += 1
                    fact.predicate = label
                meaning = self.vocabulary.get(label)
                if isinstance(meaning, dict) and meaning.get("many"):
                    # Storage must not treat a new value as replacing the old.
                    fact.multi_valued = True
                # The extractor's key was built from its own label; a stale key
                # would file the fact under a predicate it no longer has.
                fact.canonical_key = ""
            kept.append(fact)

        if existing is not None and kept:
            report.merged = self._merge_restatements(kept, existing)
            report.retiring = self._mark_retirements(kept, existing)
        report.kept = len(kept)
        return kept, report

    # ------------------------------------------------------------ decisions

    def _judge(
        self, content: str, facts: List[Any]
    ) -> Optional[List[Tuple[Optional[float], Optional[str]]]]:
        """(about-probability, vocabulary label) for every fact, in one call."""
        questions: Dict[str, Any] = {}
        for index, fact in enumerate(facts):
            stated = f"{fact.subject} | {fact.predicate} | {fact.value}"
            if self.subjects:
                questions[f"about_{index}"] = {
                    "type": "noul",
                    "instructions": {
                        "fact_id": str(index),
                        "fact": stated,
                        "person": ", ".join(self.subjects),
                        "question": (
                            "Is THIS FACT (not the text around it) a durable fact about the "
                            "person — who they are, what they study or are preparing for, what "
                            "they find hard or easy, how they prefer to learn or work, what they "
                            "plan, what they scored?"
                        ),
                        "false_when": [
                            "its subject is a topic, a resource or the world rather than the person",
                            "it is general knowledge or a definition taken from the text",
                            "it is about the assistant, a tool, or someone else",
                            "it is bookkeeping: counts of attempts, identifiers, sources, progress tracking",
                        ],
                    },
                }
                # Asked separately from "about the person", because a fact can be
                # about them and still not be something they said. Measured: two
                # questions a student asked out of curiosity ("why is tension the
                # same throughout the string?") came back from the extractor as
                # `finds_hard`, passed "about the person", and scored 0.30 here —
                # against 0.95 for "friction problems are killing me".
                questions[f"support_{index}"] = {
                    "type": "noul",
                    "instructions": {
                        "fact_id": str(index),
                        "fact": stated,
                        "question": (
                            "Does the text show THIS FACT about the person — they said it, or "
                            "their own words or answers clearly show it?"
                        ),
                        "false_when": [
                            "the person only asked a question about the topic, out of curiosity or to understand it",
                            "it is something someone else explained, not something the person said or showed",
                            "it reads more into their words than they said",
                        ],
                    },
                }
            if self.vocabulary:
                criteria = {
                    name: (
                        {k: v for k, v in meaning.items() if k not in ("many", "retires", "per")}
                        if isinstance(meaning, dict)
                        else meaning
                    )
                    for name, meaning in self.vocabulary.items()
                }
                criteria[NONE_OF_THESE] = "The fact fits none of these relations."
                questions[f"label_{index}"] = {
                    "type": "choice",
                    "instructions": {
                        "fact_id": str(index),
                        "fact": stated,
                        "question": "Which relation does THIS FACT express, if any?",
                    },
                    "criteria": criteria,
                }
        if not questions:
            return [(None, fact.predicate) for fact in facts]

        # The facts go in the state as well as the questions. Measured: with only
        # the source text as state, a question about one fact was answered about
        # the text — a student who struggles made "resource | identifier | …"
        # read as `finds_hard`. With the facts in the state, the same questions
        # dropped every off-subject fact at <=0.05 and labelled the rest at >=0.98.
        state = {
            "facts": {str(i): f"{f.subject} | {f.predicate} | {f.value}" for i, f in enumerate(facts)},
            "text": _clip(content, 3000),
        }
        answers = self.decider.decide(state, questions)
        if answers is None:
            return None

        out: List[Tuple[Optional[float], Optional[str]]] = []
        for index, fact in enumerate(facts):
            about = noul(answers, f"about_{index}") if self.subjects else None
            support = noul(answers, f"support_{index}") if self.subjects else None
            if about is not None and support is not None:
                # Both must hold; the weaker of the two decides.
                about = min(about, support)
            label: Optional[str] = fact.predicate
            if self.vocabulary:
                top = ranked(answers, f"label_{index}")
                if top and top[0][0] != NONE_OF_THESE and top[0][1] >= self.label_floor:
                    label = top[0][0]
                elif fact.predicate in self.vocabulary:
                    # The extractor already used a vocabulary word and the
                    # decision was merely unsure: keep the extractor's label.
                    label = fact.predicate
                else:
                    label = None
            out.append((about, label))
        return self._second_look(content, facts, out, answers)

    def _second_look(self, content, facts, judged, answers):
        """Ask again about facts filed under a thing but labelled as the person's.

        The extractor writes "JEE Main | scheduled_in | April" for "my JEE got
        moved to April". Asked whether that is a fact about the person, the
        model says no (0.04, measured) — its subject is an exam — though the
        label it gives it with confidence is `exam_on`. Restated as the
        person's fact, it is asked once more. Only such facts pay for this.
        """
        if not self.subjects or not self.vocabulary:
            return judged
        retry: Dict[int, str] = {}
        for index, (about, label) in enumerate(judged):
            if about is None or about >= self.about_floor or not label:
                continue
            top = ranked(answers, f"label_{index}")
            if top and top[0][0] == label and top[0][1] >= RELABEL_SUBJECT_FLOOR:
                meaning = self.vocabulary.get(label)
                if isinstance(meaning, dict) and meaning.get("per"):
                    # "JEE Main | scheduled_in | April" is the date of the exam
                    # its subject names; keep the name, or the value is a bare
                    # month that no longer says which exam.
                    facts[index].value = f"{facts[index].subject}: {facts[index].value}"
                retry[index] = f"{self.canonical_subject} | {label} | {facts[index].value}"
        if not retry:
            return judged
        questions = {}
        for index, stated in retry.items():
            questions[f"again_{index}"] = {
                "type": "noul",
                "instructions": {
                    "fact": stated,
                    "question": "Does the text show THIS FACT about the person — they said it, or their own words clearly show it?",
                    "false_when": ["the person only asked a question about the topic", "someone else said it"],
                },
            }
        again = self.decider.decide({"facts": retry, "text": _clip(content, 3000)}, questions)
        if again is None:
            return judged
        out = list(judged)
        for index in retry:
            p = noul(again, f"again_{index}")
            if p is not None and p >= self.about_floor:
                out[index] = (p, judged[index][1])
        return out

    def _merge_restatements(self, facts: List[Any], existing: ExistingLookup) -> int:
        """Point a restated fact at the stored one, so storage reaffirms it."""
        questions: Dict[str, Any] = {}
        state_items: Dict[str, Any] = {}
        offered: Dict[int, List[Tuple[str, str]]] = {}
        keyed: Dict[int, List[Tuple[str, str]]] = {}
        for index, fact in enumerate(facts):
            try:
                stored = existing(fact.subject, fact.predicate)[:MAX_EXISTING]
            except Exception:  # noqa: BLE001
                stored = []
            if not stored:
                continue
            same = [pair for pair in stored if _norm(pair[0]) == _norm(fact.value)]
            if same:
                # Identical up to case and spacing: arithmetic, not a decision.
                fact.value, fact.canonical_key = same[0][0], same[0][1] or fact.canonical_key
                continue
            meaning = self.vocabulary.get(fact.predicate)
            if self.vocabulary and not (isinstance(meaning, dict) and meaning.get("many")):
                # A predicate with one value at a time: a different value is an
                # update, and storage supersedes the old one. Asking "same
                # topic?" here merged "class 12" into "class 11" — measured.
                continue
            per = meaning.get("per") if isinstance(meaning, dict) else None
            if per:
                # One value per thing — per exam, say. The question is not
                # "same topic?" but "same exam?", and a match is an update of
                # that exam's value, not a restatement. Measured without this:
                # the JEE date (April) replaced the boards date (March).
                keyed[index] = stored
                criteria = {f"stored_{k}": value for k, (value, _key) in enumerate(stored)}
                criteria[NEW_FACT] = f"A different {per} from every stored value."
                questions[f"same_{index}"] = {
                    "type": "choice",
                    "instructions": {
                        "relation": f"{fact.subject} {fact.predicate}",
                        "new_value": fact.value,
                        "question": (
                            f"Is the new value for the same {per} as one of the stored values, "
                            f"whatever else differs? Pick that stored value, or new_fact for a "
                            f"different {per}."
                        ),
                    },
                    "criteria": criteria,
                }
                state_items[str(index)] = {
                    "relation": f"{fact.subject} {fact.predicate}",
                    "new_value": fact.value,
                    "stored_values": criteria,
                }
                continue
            offered[index] = stored
            criteria = {f"stored_{k}": value for k, (value, _key) in enumerate(stored)}
            criteria[NEW_FACT] = "A different topic from every stored value."
            questions[f"same_{index}"] = {
                "type": "choice",
                "instructions": {
                    "relation": f"{fact.subject} {fact.predicate}",
                    "new_value": fact.value,
                    # Measured on "friction on inclines" against a stored
                    # "friction problems": asked whether the new value adds
                    # anything, the model said new (0.90) — it does add a
                    # detail — and a term of study grew one row per sub-case.
                    # Asked whether it is the same topic, it merged (0.93).
                    "question": (
                        "Is the new value about the SAME topic as one of the stored values — "
                        "the same thing reworded, or the same topic named more or less "
                        "specifically (a sub-case, a situation, an example of it)? Pick that "
                        "stored value; pick new_fact only for a different topic."
                    ),
                },
                "criteria": criteria,
            }
            state_items[str(index)] = {
                "relation": f"{fact.subject} {fact.predicate}",
                "new_value": fact.value,
                "stored_values": {f"stored_{k}": value for k, (value, _key) in enumerate(stored)},
            }
        if not questions:
            return 0
        # As with labels: what is being compared goes in the state, or the
        # decision reads the question's wording instead of the values.
        answers = self.decider.decide({"new_facts": state_items}, questions)
        if answers is None:
            return 0
        merged = 0
        for index, stored in keyed.items():
            top = ranked(answers, f"same_{index}")
            if not top or top[0][0] == NEW_FACT or top[0][1] < self.same_floor:
                continue
            try:
                pick = int(top[0][0].split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if 0 <= pick < len(stored):
                value, key = stored[pick]
                fact = facts[index]
                # The same thing with a new value replaces the old value.
                _add_retirement(fact, key or f"{fact.subject}|{fact.predicate}|{value}")
        for index, stored in offered.items():
            top = ranked(answers, f"same_{index}")
            if not top or top[0][0] == NEW_FACT or top[0][1] < self.same_floor:
                continue
            try:
                pick = int(top[0][0].split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            if 0 <= pick < len(stored):
                value, key = stored[pick]
                facts[index].value = value
                facts[index].canonical_key = key or facts[index].canonical_key
                merged += 1
        return merged


    def _mark_retirements(self, facts: List[Any], existing: ExistingLookup) -> int:
        """Mark stored facts a new one says are no longer true.

        A vocabulary can say that one predicate undoes another — "finds_easy"
        retires "finds_hard". Measured before this: a student who said
        "vectors are fine now" kept `finds_hard: vectors` beside
        `finds_easy: vectors`, and a tutor was handed both. The new fact carries
        the canonical keys it retires; storage stamps them superseded by it.

        One yes/no per stored value, not one pick among them: a student store
        held "vectors", "resolving into components" and "vectors (resolving
        into components)", and a single choice retired the first and left the
        other two in front of the tutor for the rest of the term.
        """
        questions: Dict[str, Any] = {}
        pending: Dict[str, Tuple[int, str]] = {}
        state: Dict[str, Any] = {}
        for index, fact in enumerate(facts):
            meaning = self.vocabulary.get(fact.predicate)
            targets = meaning.get("retires") if isinstance(meaning, dict) else None
            if not targets:
                continue
            for target in [targets] if isinstance(targets, str) else list(targets):
                try:
                    stored = existing(fact.subject, target)[:MAX_EXISTING]
                except Exception:  # noqa: BLE001
                    stored = []
                for k, (value, key) in enumerate(stored):
                    key = key or f"{fact.subject}|{target}|{value}"
                    if _norm(value) == _norm(fact.value):
                        _add_retirement(fact, key)
                        continue
                    qid = f"undo_{index}_{target}_{k}"
                    pending[qid] = (index, key)
                    state[qid] = {
                        "new_fact": f"{fact.subject} {fact.predicate} {fact.value}",
                        "stored_fact": f"{fact.subject} {target} {value}",
                    }
                    questions[qid] = {
                        "type": "noul",
                        "instructions": {
                            "new_fact": f"{fact.subject} {fact.predicate} {fact.value}",
                            "stored_fact": f"{fact.subject} {target} {value}",
                            # Measured over nine pairs: must-retire pairs 0.66-0.93,
                            # must-not pairs 0.03-0.05. The earlier wording ("no
                            # longer true, now the other way round?") scored the
                            # clearest must-retire pair 0.22.
                            "question": (
                                "The person now finds the new fact's topic easy (or hard). Is "
                                "the stored fact about that same topic or a part of it, so that "
                                "it no longer holds?"
                            ),
                        },
                    }
        if questions:
            answers = self.decider.decide(state, questions)
            floor = RETIRE_FLOOR
            if answers is not None:
                for qid, (index, key) in pending.items():
                    p = noul(answers, qid)
                    if p is not None and p >= floor:
                        _add_retirement(facts[index], key)
        return sum(1 for f in facts if getattr(f, "retires_keys", None))


def _add_retirement(fact: Any, canonical_key: str) -> None:
    keys = list(getattr(fact, "retires_keys", None) or [])
    if canonical_key and canonical_key not in keys:
        keys.append(canonical_key)
    fact.retires_keys = keys


def fact_gate_from_config(config: Any, decider: Optional[Decider]) -> Optional[FactGate]:
    if decider is None or config is None:
        return None
    vocabulary = dict(getattr(config, "fact_vocabulary", {}) or {})
    subjects = list(getattr(config, "fact_subjects", []) or [])
    return FactGate(
        decider,
        vocabulary=vocabulary,
        subjects=subjects,
        canonical_subject=getattr(config, "canonical_subject", "user") or "user",
        about_floor=float(getattr(config, "about_floor", 0.5)),
        label_floor=float(getattr(config, "label_floor", 0.5)),
        same_floor=float(getattr(config, "same_floor", 0.7)),
    )


_ARTICLES = {"a", "an", "the"}


def _norm(text: str) -> str:
    """Case, spacing, underscores, punctuation and articles do not make a new fact."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text or "").lower().replace("_", " "))
    return " ".join(word for word in cleaned.split() if word not in _ARTICLES)


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"
