"""Put a store that grew without the fact gate through it, once.

The gate only sees facts as they are written. Every store that existed before
it keeps what it collected — measured on Tura's student stores on 2026-09-23:
81 of 186 facts not about the student, the student's own facts under labels no
reader joins on, and one difficulty stored eight ways. Left alone those stay
forever and a read gate has to step round them on every turn.

This walks the active facts, grouped by the memory they were extracted from so
each decision sees the text it came from, and applies the same three decisions:

* **Not about the subject** — retired: ``valid_until`` stamped, ``tier`` set to
  ``avoid``. Nothing is deleted; the row stays explorable.
* **Under another label** — relabelled in place, with a fresh canonical key.
* **Already said by a kept fact** — merged: the stored fact's
  ``reaffirmed_count`` goes up and the restatement is superseded by it.

A dry run reports what would change and writes nothing.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from dhee.decisions.facts import FactGate

logger = logging.getLogger(__name__)

RETIRED = "gate:not-about-subject"


@dataclass
class RegateReport:
    facts: int = 0
    retired: int = 0
    relabelled: int = 0
    merged: int = 0
    undecided_groups: int = 0
    changes: List[Dict[str, Any]] = field(default_factory=list)


def regate_facts(db: Any, gate: FactGate, *, dry_run: bool = True, limit: int = 2000) -> RegateReport:
    """Apply ``gate`` to the store's active facts. Returns what changed.

    ``db`` is Dhee's SQLite manager (anything with ``_get_connection``). A group
    whose decision fails is left exactly as it was and counted, so a partial
    run is safe to repeat.
    """
    report = RegateReport()
    with db._get_connection() as conn:
        rows = conn.execute(
            """SELECT f.id, f.memory_id, f.subject, f.predicate, f.value, f.canonical_key,
                      coalesce(m.memory, '') AS text
               FROM engram_facts f LEFT JOIN memories m ON m.id = f.memory_id
               WHERE f.superseded_by_id IS NULL AND f.valid_until IS NULL
               ORDER BY f.created_at ASC LIMIT ?""",
            (int(limit),),
        ).fetchall()

    groups: "OrderedDict[str, Tuple[str, List[Any]]]" = OrderedDict()
    for row in rows:
        fact = SimpleNamespace(
            id=row["id"],
            subject=row["subject"] or "",
            predicate=row["predicate"] or "",
            value=row["value"] or "",
            canonical_key=row["canonical_key"] or "",
        )
        groups.setdefault(row["memory_id"] or row["id"], (row["text"], []))[1].append(fact)
    report.facts = len(rows)

    #: The facts kept so far, by (subject, predicate): what a later group's
    #: restatement is compared against. Built as the walk goes, oldest first, so
    #: the first way a thing was said is the one that stays.
    kept_so_far: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

    def existing(subject: str, predicate: str) -> List[Tuple[str, str]]:
        return [(value, key) for value, key, _id in reversed(kept_so_far.get((subject, predicate), []))]

    for _memory_id, (text, facts) in groups.items():
        before = {f.id: (f.subject, f.predicate, f.value) for f in facts}
        originals = [SimpleNamespace(**vars(f)) for f in facts]
        kept, gate_report = gate.apply(text, originals, existing)
        if not gate_report.decided:
            report.undecided_groups += 1
            for f in facts:
                kept_so_far.setdefault((f.subject, f.predicate), []).append((f.value, f.canonical_key, f.id))
            continue
        kept_ids = {f.id for f in kept}
        for fact in facts:
            if fact.id not in kept_ids:
                report.retired += 1
                report.changes.append({"id": fact.id, "action": "retire", "fact": before[fact.id]})
        for fact in kept:
            old = before[fact.id]
            match = next(
                (item for item in kept_so_far.get((fact.subject, fact.predicate), []) if item[0] == fact.value),
                None,
            )
            if match is not None and match[2] != fact.id:
                report.merged += 1
                report.changes.append(
                    {"id": fact.id, "action": "merge", "into": match[2], "fact": old, "as": fact.value}
                )
                continue
            if (fact.subject, fact.predicate) != old[:2]:
                report.relabelled += 1
                report.changes.append(
                    {"id": fact.id, "action": "relabel", "fact": old, "to": (fact.subject, fact.predicate)}
                )
            key = fact.canonical_key or f"{fact.subject}|{fact.predicate}|{fact.value}"
            kept_so_far.setdefault((fact.subject, fact.predicate), []).append((fact.value, key, fact.id))
            fact.canonical_key = key

    if dry_run or not report.changes:
        return report

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with db._get_connection() as conn:
        for change in report.changes:
            if change["action"] == "retire":
                conn.execute(
                    "UPDATE engram_facts SET valid_until = ?, tier = 'avoid', qualifier = "
                    "coalesce(qualifier || '; ', '') || ? WHERE id = ?",
                    (now, RETIRED, change["id"]),
                )
            elif change["action"] == "merge":
                conn.execute(
                    "UPDATE engram_facts SET superseded_by_id = ?, valid_until = ?, tier = 'avoid' WHERE id = ?",
                    (change["into"], now, change["id"]),
                )
                conn.execute(
                    "UPDATE engram_facts SET reaffirmed_count = coalesce(reaffirmed_count, 0) + 1, "
                    "last_reaffirmed_at = ? WHERE id = ?",
                    (time.time(), change["into"]),
                )
            elif change["action"] == "relabel":
                subject, predicate = change["to"]
                value = change["fact"][2]
                conn.execute(
                    "UPDATE engram_facts SET subject = ?, predicate = ?, canonical_key = ? WHERE id = ?",
                    (subject, predicate, f"{subject}|{predicate}|{value}", change["id"]),
                )
    logger.info(
        "Regated %d facts: %d retired, %d relabelled, %d merged (%d groups undecided)",
        report.facts,
        report.retired,
        report.relabelled,
        report.merged,
        report.undecided_groups,
    )
    return report
