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


@dataclass
class GateReport:
    """What the gate did to one extraction, for logs and tests."""

    kept: int = 0
    dropped_off_subject: int = 0
    relabelled: int = 0
    dropped_unlabelled: int = 0
    merged: int = 0
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
                # The extractor's key was built from its own label; a stale key
                # would file the fact under a predicate it no longer has.
                fact.canonical_key = ""
            kept.append(fact)

        if existing is not None and kept:
            report.merged = self._merge_restatements(kept, existing)
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
            if self.vocabulary:
                criteria = {name: meaning for name, meaning in self.vocabulary.items()}
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
        return out

    def _merge_restatements(self, facts: List[Any], existing: ExistingLookup) -> int:
        """Point a restated fact at the stored one, so storage reaffirms it."""
        questions: Dict[str, Any] = {}
        state_items: Dict[str, Any] = {}
        offered: Dict[int, List[Tuple[str, str]]] = {}
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
            offered[index] = stored
            criteria = {f"stored_{k}": value for k, (value, _key) in enumerate(stored)}
            criteria[NEW_FACT] = "Something none of the stored values already says."
            questions[f"same_{index}"] = {
                "type": "choice",
                "instructions": {
                    "relation": f"{fact.subject} {fact.predicate}",
                    "new_value": fact.value,
                    "question": (
                        "Is the new value already said by one of the stored values — the same "
                        "thing reworded, abbreviated, or only slightly more or less specific — "
                        "so that someone reading the stored list learns nothing new from it? "
                        "Pick that stored value, or new_fact if it adds something."
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
