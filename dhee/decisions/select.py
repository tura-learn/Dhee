"""Choose what a turn should see: memory as candidates, a decision per item.

Retrieval ranks by similarity, which answers "what looks like this query?". An
agent needs "what would change my answer to this query?" — the student's exam
date matters to a question that never mentions it, and a near-duplicate of the
question is worth nothing. So the harness gathers candidates cheaply (facts,
profile, a vector search), and one decision call scores every candidate as a
Noul: would knowing this help the next reply? Only what clears the floor goes
into the prompt.

One request carries every candidate, because the questions are evaluated in
parallel against the same state — the call costs about as much for twenty
candidates as for one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from dhee.decisions.jev import Decider, noul


@dataclass(frozen=True)
class Selected:
    index: int
    text: str
    p: float


def select_relevant(
    decider: Optional[Decider],
    query: str,
    candidates: Sequence[str],
    *,
    situation: Optional[Dict[str, Any]] = None,
    floor: float = 0.6,
    limit: int = 8,
    distinct: float = 0.6,
) -> Optional[List[Selected]]:
    """The candidates worth putting in front of the model, most useful first.

    ``None`` means no decision was made (no decider, a timeout, an error) and
    the caller should fall back to its own ordering. An empty list is a real
    answer: nothing here helps.
    """
    if decider is None:
        return None
    texts = [str(c or "").strip() for c in candidates]
    indexed = [(i, t) for i, t in enumerate(texts) if t]
    if not indexed:
        return []
    questions: Dict[str, Any] = {}
    for i, text in indexed:
        questions[f"m{i}"] = {
            "type": "noul",
            "instructions": {
                "memory": text[:600],
                "question": (
                    "Would knowing this change or improve the assistant's next reply to the "
                    "message — how to pitch it, what to include or avoid, what to connect it to?"
                ),
                "false_when": [
                    "it only repeats what the message itself already says",
                    "it is about a different subject with no bearing on this reply",
                ],
            },
        }
    state: Dict[str, Any] = {"message": str(query or "")[:2000]}
    if situation:
        state["situation"] = situation
    answers = decider.decide(state, questions)
    if answers is None:
        return None
    chosen: List[Selected] = []
    for i, text in indexed:
        p = noul(answers, f"m{i}")
        if p is not None and p >= floor:
            chosen.append(Selected(index=i, text=text, p=p))
    chosen.sort(key=lambda s: s.p, reverse=True)
    # One slot per idea. A store that grew before its writes were deduplicated
    # says one difficulty six ways, and all six score alike — measured on a
    # student store, the top six picks for a tension question were four
    # phrasings of the same struggle. Word overlap is enough to catch that; a
    # different idea that happens to share words scores lower and still lands.
    kept: List[Selected] = []
    for item in chosen:
        words = _content_words(item.text)
        if distinct and any(_overlap(words, _content_words(k.text)) >= distinct for k in kept):
            continue
        kept.append(item)
        if len(kept) >= max(0, limit):
            break
    return kept


_STOP = {"a", "an", "the", "of", "in", "on", "to", "and", "with", "for", "is", "are", "their", "they"}


def _content_words(text: str) -> set:
    cleaned = "".join(ch if ch.isalnum() else " " for ch in str(text).lower().replace("_", " "))
    return {w.rstrip("s") for w in cleaned.split() if w not in _STOP and len(w) > 2}


def _overlap(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))
