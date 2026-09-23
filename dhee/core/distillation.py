"""Replay-driven semantic distillation (CLS consolidation).

During sleep cycles, the ReplayDistiller samples recent episodic memories,
groups them by scene or time window, and uses an LLM to extract durable
semantic facts. This models the hippocampus-to-neocortex transfer in
Complementary Learning Systems theory.

v3 addition: DistillationStore and DistillationCandidate for the
event-sourced candidate promotion pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dhee.memory.utils import strip_code_fences
from dhee.utils.prompts import DISTILLATION_PROMPT

if TYPE_CHECKING:
    from dhee.configs.base import DistillationConfig
    from dhee.db.sqlite import SQLiteManager
    from dhee.llms.base import BaseLLM

logger = logging.getLogger(__name__)


# ===========================================================================
# v2 ReplayDistiller (used by dhee.memory.main sleep_cycle)
# ===========================================================================


class ReplayDistiller:
    """Extracts semantic knowledge from episodic memory batches."""

    def __init__(
        self,
        db: "SQLiteManager",
        llm: "BaseLLM",
        config: "DistillationConfig",
    ):
        self.db = db
        self.llm = llm
        self.config = config

    def run(
        self,
        user_id: str,
        date_str: Optional[str] = None,
        memory_add_fn: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run one distillation cycle for a user.

        Args:
            user_id: The user whose episodic memories to distill.
            date_str: Target date (defaults to yesterday).
            memory_add_fn: Callable to add a memory (typically Memory.add).
                           Required for actual distillation; if None, dry-run only.

        Returns:
            Stats dict with episodes_sampled, semantic_created, etc.
        """
        if not self.config.enable_distillation:
            return {"skipped": True, "reason": "distillation disabled"}

        target_date = date_str or (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).date().isoformat()

        window_hours = self.config.distillation_time_window_hours
        created_after = f"{target_date}T00:00:00"
        created_before = f"{target_date}T23:59:59.999999"

        # Sample recent episodic memories
        episodes = self.db.get_episodic_memories(
            user_id,
            created_after=created_after,
            created_before=created_before,
            limit=self.config.distillation_batch_size * 5,
        )

        # Only episodes no earlier run has read. The cycle runs every few
        # minutes and targets a whole day, so without this it distils the same
        # day on every run — measured on a student store: the same 9 episodes
        # distilled 11 times in a morning into 98 near-identical memories.
        already = set()
        seen_fn = getattr(self.db, "distilled_episode_ids", None)
        if callable(seen_fn) and episodes:
            try:
                already = seen_fn([ep.get("id") for ep in episodes])
            except Exception as e:  # noqa: BLE001 - an unreadable ledger is an empty one
                logger.warning("Could not read the distilled-episode ledger: %s", e)
        fresh = [ep for ep in episodes if ep.get("id") not in already]
        if len(fresh) < self.config.distillation_min_episodes:
            return {
                "skipped": True,
                "reason": "insufficient episodes" if not already else "insufficient new episodes",
                "episodes_found": len(fresh),
                "already_distilled": len(already),
                "min_required": self.config.distillation_min_episodes,
            }
        episodes = fresh

        run_marker = str(uuid.uuid4())
        # Group into batches
        batches = self._group_episodes(episodes)

        total_created = 0
        total_dedup = 0
        total_errors = 0

        for batch in batches:
            try:
                created, dedup = self._distill_batch(
                    user_id=user_id,
                    batch=batch,
                    memory_add_fn=memory_add_fn,
                )
                total_created += created
                total_dedup += dedup
                mark_fn = getattr(self.db, "mark_episodes_distilled", None)
                if callable(mark_fn) and memory_add_fn is not None:
                    mark_fn([ep.get("id") for ep in batch], run_marker)
            except Exception as e:
                logger.warning("Distillation batch failed: %s", e)
                total_errors += 1

        # Log the run
        run_id = self.db.log_distillation_run(
            user_id=user_id,
            episodes_sampled=len(episodes),
            semantic_created=total_created,
            semantic_deduplicated=total_dedup,
            errors=total_errors,
        )

        return {
            "run_id": run_id,
            "episodes_sampled": len(episodes),
            "batches_processed": len(batches),
            "semantic_created": total_created,
            "semantic_deduplicated": total_dedup,
            "errors": total_errors,
        }

    def _group_episodes(
        self, episodes: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group episodes by scene_id or into time-window chunks."""
        if self.config.distillation_scene_grouping:
            # Group by scene_id first
            scene_groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for ep in episodes:
                scene_id = ep.get("scene_id")
                scene_groups.setdefault(scene_id, []).append(ep)

            batches = []
            for scene_id, group in scene_groups.items():
                # Split large scene groups into sub-batches
                batch_size = self.config.distillation_batch_size
                for i in range(0, len(group), batch_size):
                    batches.append(group[i : i + batch_size])
            return batches

        # Fallback: chunk by batch_size
        batch_size = self.config.distillation_batch_size
        return [
            episodes[i : i + batch_size]
            for i in range(0, len(episodes), batch_size)
        ]

    def _distill_batch(
        self,
        user_id: str,
        batch: List[Dict[str, Any]],
        memory_add_fn: Optional[Any],
    ) -> tuple:
        """Distill a single batch of episodes. Returns (created, deduplicated)."""
        # Build the episodes text for the prompt
        episode_texts = []
        episode_ids = []
        for ep in batch:
            ep_id = ep.get("id", "unknown")
            episode_ids.append(ep_id)
            content = ep.get("memory", "")
            created_at = ep.get("created_at", "")
            episode_texts.append(f"[{ep_id}] ({created_at}): {content}")

        episodes_str = "\n".join(episode_texts)
        prompt = DISTILLATION_PROMPT.format(
            episodes=episodes_str,
            max_facts=self.config.max_semantic_per_batch,
        )

        raw_response = self.llm.generate(prompt)
        cleaned = strip_code_fences(raw_response)

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Distillation LLM returned invalid JSON: %.200s", raw_response)
            return (0, 0)

        facts = parsed.get("semantic_facts", [])
        if not isinstance(facts, list):
            return (0, 0)

        created = 0
        deduplicated = 0

        for fact in facts[: self.config.max_semantic_per_batch]:
            content = fact.get("content", "").strip()
            if not content:
                continue

            importance = fact.get("importance", "medium")
            source_eps = fact.get("source_episodes", episode_ids)

            if memory_add_fn is not None:
                result = memory_add_fn(
                    content,
                    user_id=user_id,
                    infer=False,
                    initial_layer="lml",
                    initial_strength=0.8,
                    metadata={
                        "is_distilled": True,
                        "distillation_source_count": len(source_eps),
                        "importance": importance,
                        "memory_type": "semantic",
                    },
                )

                # Check if it was deduplicated (NOOP/SUBSUMED)
                results = result.get("results", [])
                if results:
                    first = results[0]
                    event = first.get("event", "ADD")
                    if event in ("NOOP", "SUBSUMED"):
                        deduplicated += 1
                    else:
                        created += 1
                        # Record provenance
                        semantic_id = first.get("id")
                        if semantic_id:
                            try:
                                self.db.add_distillation_provenance(
                                    semantic_memory_id=semantic_id,
                                    episodic_memory_ids=source_eps,
                                    run_id=str(uuid.uuid4()),
                                )
                            except Exception as e:
                                logger.warning("Failed to record provenance: %s", e)

        return (created, deduplicated)


# ===========================================================================
# v3 Distillation: episodic-to-semantic candidate creation
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Current distillation algorithm version. Bump when extraction logic changes.
DERIVATION_VERSION = 1


def compute_idempotency_key(
    source_event_ids: List[str],
    derivation_version: int,
    canonical_key: str,
) -> str:
    """Deterministic key from sorted source IDs + version + canonical key."""
    payload = (
        "|".join(sorted(source_event_ids))
        + f"|v{derivation_version}"
        + f"|{canonical_key}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


@dataclass
class DistillationCandidate:
    """A proposed derived object awaiting promotion."""

    candidate_id: str
    source_event_ids: List[str]
    target_type: str  # belief, policy, insight, heuristic
    canonical_key: str  # human-readable dedup key
    payload: Dict[str, Any]  # type-specific data for the derived object
    confidence: float = 0.5
    derivation_version: int = DERIVATION_VERSION
    idempotency_key: str = ""
    status: str = "pending_validation"

    def __post_init__(self):
        if not self.idempotency_key:
            self.idempotency_key = compute_idempotency_key(
                self.source_event_ids,
                self.derivation_version,
                self.canonical_key,
            )


class DistillationStore:
    """Manages distillation candidates in the database."""

    def __init__(self, conn: "sqlite3.Connection", lock: "threading.RLock"):
        import sqlite3
        import threading
        self._conn = conn
        self._lock = lock

    def submit(self, candidate: DistillationCandidate) -> Optional[str]:
        """Submit a candidate. Returns candidate_id if new, None if duplicate."""
        now = _utcnow_iso()

        with self._lock:
            try:
                # Idempotency check — if same key exists and is not rejected, skip
                existing = self._conn.execute(
                    """SELECT candidate_id, status FROM distillation_candidates
                       WHERE idempotency_key = ? AND status != 'rejected'
                       LIMIT 1""",
                    (candidate.idempotency_key,),
                ).fetchone()

                if existing:
                    logger.debug(
                        "Candidate dedup hit: %s (existing=%s, status=%s)",
                        candidate.idempotency_key,
                        existing["candidate_id"],
                        existing["status"],
                    )
                    return None

                self._conn.execute(
                    """INSERT INTO distillation_candidates
                       (candidate_id, source_event_ids, derivation_version,
                        confidence, canonical_key, idempotency_key,
                        target_type, payload_json, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate.candidate_id,
                        json.dumps(candidate.source_event_ids),
                        candidate.derivation_version,
                        candidate.confidence,
                        candidate.canonical_key,
                        candidate.idempotency_key,
                        candidate.target_type,
                        json.dumps(candidate.payload),
                        candidate.status,
                        now,
                    ),
                )
                self._conn.commit()
                return candidate.candidate_id

            except Exception:
                self._conn.rollback()
                raise

    def get_pending(
        self,
        target_type: Optional[str] = None,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get pending candidates for promotion."""
        query = """SELECT * FROM distillation_candidates
                   WHERE status = 'pending_validation'"""
        params: list = []
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)
        query += " ORDER BY confidence DESC, created_at ASC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def set_status(
        self,
        candidate_id: str,
        status: str,
        *,
        promoted_id: Optional[str] = None,
    ) -> bool:
        """Update candidate status (promoted, rejected, quarantined)."""
        with self._lock:
            try:
                if promoted_id:
                    result = self._conn.execute(
                        """UPDATE distillation_candidates
                           SET status = ?, promoted_id = ?
                           WHERE candidate_id = ?""",
                        (status, promoted_id, candidate_id),
                    )
                else:
                    result = self._conn.execute(
                        """UPDATE distillation_candidates
                           SET status = ?
                           WHERE candidate_id = ?""",
                        (status, candidate_id),
                    )
                self._conn.commit()
                return result.rowcount > 0
            except Exception:
                self._conn.rollback()
                raise

    def get(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM distillation_candidates WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        source_ids = row["source_event_ids"]
        if isinstance(source_ids, str):
            try:
                source_ids = json.loads(source_ids)
            except (json.JSONDecodeError, TypeError):
                source_ids = []

        payload = row["payload_json"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                payload = {}

        return {
            "candidate_id": row["candidate_id"],
            "source_event_ids": source_ids,
            "derivation_version": row["derivation_version"],
            "confidence": row["confidence"],
            "canonical_key": row["canonical_key"],
            "idempotency_key": row["idempotency_key"],
            "target_type": row["target_type"],
            "payload": payload,
            "status": row["status"],
            "promoted_id": row["promoted_id"],
            "created_at": row["created_at"],
        }


def distill_belief_from_events(
    events: List[Dict[str, Any]],
    *,
    user_id: str,
    domain: str = "general",
) -> Optional[DistillationCandidate]:
    """Create a belief candidate from a set of corroborating events.

    This is a rule-based distillation. No LLM call.
    Finds recurring factual claims across events and proposes them as beliefs.
    """
    if not events:
        return None

    # Simple: use the first event's content as the claim
    # In a fuller implementation, this would extract common facts
    source_ids = [e["event_id"] for e in events]
    claim = events[0].get("content", "")
    if not claim:
        return None

    canonical_key = f"belief:{user_id}:{domain}:{claim[:80]}"

    return DistillationCandidate(
        candidate_id=str(uuid.uuid4()),
        source_event_ids=source_ids,
        target_type="belief",
        canonical_key=canonical_key,
        confidence=min(0.3 + 0.1 * len(events), 0.9),
        payload={
            "user_id": user_id,
            "claim": claim,
            "domain": domain,
            "source_memory_ids": source_ids,
        },
    )
