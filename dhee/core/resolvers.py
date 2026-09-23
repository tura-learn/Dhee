"""Context-first resolver — hierarchical context filtering before content matching.

Implements the human memory model: narrow by era/place/time first, then match content.
Runs BEFORE the existing vector search pipeline.

Integration point: engram/memory/main.py search()
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Preference classification (M2.2) ────────────────────────────────
# Preference-shaped predicates get a parallel row in ``engram_preferences``
# carrying stance, topic, and lineage. They still land in engram_facts so
# existing resolvers keep working; retrieval (M2.4) will prefer the
# preferences row when both exist.

_PREFERENCE_PREDICATES = {
    "prefers",
    "likes",
    "loves",
    "enjoys",
    "admires",
    "favors",
    "dislikes",
    "hates",
    "avoids",
    "refuses",
    "switched_to",
    "switched_away_from",
    "uses_editor",
    "uses_language",
    "subscribes_to",
}
_POSITIVE_STANCES = {
    "prefers", "likes", "loves", "enjoys", "admires", "favors",
    "switched_to", "uses_editor", "uses_language", "subscribes_to",
}
_NEGATIVE_STANCES = {
    "dislikes", "hates", "avoids", "refuses", "switched_away_from",
}


def _normalize_predicate(pred: str) -> str:
    return (pred or "").lower().replace(" ", "_")


def _classify_preference(pred: str) -> Optional[Tuple[str, str]]:
    """Return (topic, stance) if this predicate encodes a preference.

    ``topic`` is a normalised form of the predicate (what the preference is
    about) and ``stance`` is one of ``positive`` / ``negative`` / ``neutral``.
    Returns None for non-preference predicates.
    """
    p = _normalize_predicate(pred)
    if p.startswith("favorite_") or p.startswith("favourite_"):
        return p, "positive"
    if p in _POSITIVE_STANCES:
        return p, "positive"
    if p in _NEGATIVE_STANCES:
        return p, "negative"
    if p in _PREFERENCE_PREDICATES:
        return p, "neutral"
    return None

# ── Query intent classification patterns (zero-LLM, deterministic) ──

_COUNT_RE = re.compile(
    r"\bhow many\b(?!.*\b(?:days?|weeks?|months?|years?|hours?|minutes?)\b)",
    re.I,
)
_LATEST_RE = re.compile(
    r"\b(?:current(?:ly)?|latest|most recent(?:ly)?|right now|now(?:adays)?|at the moment"
    r"|what (?:is|are) (?:my|the)\b.*\bnow"
    # Current-habit queries: "what time do I [habit]" → asking current value,
    # not a timeline. "when do I [habit]" covers present-tense habituals too.
    r"|what time do (?:i|you)\b"
    r"|when do (?:i|you)\b)\b",
    re.I,
)
_SUM_RE = re.compile(
    r"\b(?:total|sum|how much.*(?:spend|spent|cost|pay|paid)|altogether)\b",
    re.I,
)
_TEMPORAL_RE = re.compile(
    r"\b(?:when did|what (?:time|date|day|year)|in what year|how long ago|chronolog|timeline|history of)\b",
    re.I,
)
_SET_RE = re.compile(
    r"\b(?:which|what|list|name|all the|enumerate)\b.*\b(?:have I|did I|do I|I (?:have|did|do))\b",
    re.I,
)

# Predicate extraction patterns — maps query phrases to fact predicates
_PREDICATE_PATTERNS = [
    (re.compile(r"\b(?:countries?|places?|cities?|locations?)\b.*\b(?:visit|been|travel|go)\b", re.I), "visited"),
    (re.compile(r"\b(?:visit|been to|travel(?:ed|ing)?\s+to|went to|go(?:ne)?\s+to)\b", re.I), "visited"),
    (re.compile(r"\b(?:movie|film|show)s?\b.*\b(?:watch|seen|saw)\b", re.I), "watched_movie"),
    (re.compile(r"\b(?:watch|seen|saw)\b.*\b(?:movie|film|show)s?\b", re.I), "watched_movie"),
    (re.compile(r"\b(?:book|novel)s?\b.*\b(?:read|finish)\b", re.I), "read_book"),
    (re.compile(r"\b(?:read|finish)\b.*\b(?:book|novel)s?\b", re.I), "read_book"),
    (re.compile(r"\b(?:sport|game)s?\b.*\b(?:play|compet)\b", re.I), "played_sport"),
    (re.compile(r"\b(?:play|compet)\b.*\b(?:sport|game)s?\b", re.I), "played_sport"),
    (re.compile(r"\b(?:restaurant|eat|din|ate)\b", re.I), "ate_at"),
    (re.compile(r"\b(?:recipe|cook|bak)\b", re.I), "cooked"),
    (re.compile(r"\b(?:editor|ide|code editor)\b", re.I), "uses_editor"),
    (re.compile(r"\b(?:language|programming)\b.*\b(?:use|learn|know|write)\b", re.I), "uses_language"),
    (re.compile(r"\b(?:subscribe|subscription|membership)\b", re.I), "subscribes_to"),
    (re.compile(r"\b(?:prefer|favorite|favourite)\b", re.I), "prefers"),
    (re.compile(r"\b(?:job|work(?:ed)?(?:\s+(?:at|for))?|employ|company|position)\b", re.I), "works_at"),
    (re.compile(r"\b(?:pet|dog|cat|animal)\b", re.I), "has_pet"),
    (re.compile(r"\b(?:hobby|hobbies|pastime|interest)\b", re.I), "has_hobby"),
    (re.compile(r"\b(?:buy|bought|purchas)\b", re.I), "bought"),
    (re.compile(r"\b(?:learn|stud|course|class)\b", re.I), "learned"),
    (re.compile(r"\b(?:award|won|prize|achievement)\b", re.I), "won_award"),
]


@dataclass
class ResolverResult:
    """Result from deterministic resolution."""
    answer: Optional[str] = None
    facts: List[Dict[str, Any]] = field(default_factory=list)
    memory_ids: List[str] = field(default_factory=list)
    resolver_path: str = ""                # "context->sql", "chain->derived"
    confidence: float = 1.0
    is_deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "facts": self.facts,
            "memory_ids": self.memory_ids,
            "resolver_path": self.resolver_path,
            "confidence": self.confidence,
            "is_deterministic": self.is_deterministic,
        }

    def grounded_memory_ids(self) -> List[str]:
        return list(dict.fromkeys(memory_id for memory_id in self.memory_ids if memory_id))

    def has_grounding(self) -> bool:
        return bool(self.grounded_memory_ids())


@dataclass
class QueryPlan:
    """Parsed query with intent and context filters."""
    intent: str = "freeform"               # count|latest|set_members|sum|temporal|freeform
    context_filters: Dict[str, Any] = field(default_factory=dict)
    search_terms: List[str] = field(default_factory=list)
    subject: Optional[str] = None
    predicate: Optional[str] = None
    chain_request: bool = False            # needs associative chain traversal


class ContextResolver:
    """Hierarchical context-first retrieval + deterministic fact resolution."""

    def __init__(self, db):
        """Initialize with a SQLiteManager instance."""
        self.db = db

    @staticmethod
    def _dedupe_memory_ids(memory_ids: List[str]) -> List[str]:
        return list(dict.fromkeys(memory_id for memory_id in memory_ids if memory_id))

    @staticmethod
    def _split_grouped_memory_ids(raw_value: Optional[str]) -> List[str]:
        if not raw_value:
            return []
        return ContextResolver._dedupe_memory_ids(raw_value.split(","))

    def _fact_query_parts(
        self,
        *,
        user_id: Optional[str] = None,
        predicate: Optional[str] = None,
        subject: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
        valid_only: bool = False,
    ) -> Tuple[str, List[Any]]:
        conditions = ["m.id = f.memory_id", "m.tombstone = 0"]
        params: List[Any] = []

        if user_id:
            conditions.append("m.user_id = ?")
            params.append(user_id)
        if subject:
            conditions.append("f.subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("f.predicate = ?")
            params.append(predicate)
        if context_ids:
            placeholders = ",".join("?" for _ in context_ids)
            conditions.append(f"f.memory_id IN ({placeholders})")
            params.extend(context_ids)
        if valid_only:
            # M2 substrate semantics: "valid" now means (a) not temporally
            # closed via valid_until, (b) not superseded by a newer row, and
            # (c) not demoted to the 'avoid' tier. These three signals are
            # substrate-native — no score fusion, no reranker bolt-on.
            conditions.append("f.valid_until IS NULL")
            conditions.append("(f.superseded_by_id IS NULL)")
            conditions.append("(f.tier IS NULL OR f.tier != 'avoid')")

        from_clause = " FROM engram_facts f JOIN memories m ON m.id = f.memory_id"
        if conditions:
            from_clause += " WHERE " + " AND ".join(conditions)
        return from_clause, params

    def get_fact_status(
        self,
        memory_ids: List[str],
        *,
        user_id: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Batch-return active vs superseded engram_fact counts per memory_id.

        Used by the search pipeline to make fact-level supersede visible at
        rank time. A memory with a superseded fact for (subject, predicate)
        is not authoritative for that question — the newer memory holding
        the non-superseded fact should outrank it. This helper exposes the
        signal; scoring is the caller's decision.

        Returns {memory_id: {"active": int, "superseded": int}} — memory_ids
        with no matching fact rows default to {"active": 0, "superseded": 0}.
        """
        status: Dict[str, Dict[str, int]] = {
            memory_id: {"active": 0, "superseded": 0} for memory_id in memory_ids if memory_id
        }
        if not status or not self._has_engram_tables():
            return status

        try:
            with self.db._get_connection() as conn:
                placeholders = ",".join("?" for _ in status)
                conditions = [f"f.memory_id IN ({placeholders})", "m.tombstone = 0"]
                params: List[Any] = list(status.keys())
                if user_id:
                    conditions.append("m.user_id = ?")
                    params.append(user_id)
                if subject:
                    conditions.append("f.subject = ?")
                    params.append(subject)
                if predicate:
                    conditions.append("f.predicate = ?")
                    params.append(predicate)
                sql = (
                    "SELECT f.memory_id, "
                    "SUM(CASE WHEN f.valid_until IS NULL THEN 1 ELSE 0 END) AS active, "
                    "SUM(CASE WHEN f.valid_until IS NOT NULL THEN 1 ELSE 0 END) AS superseded "
                    "FROM engram_facts f JOIN memories m ON m.id = f.memory_id "
                    "WHERE " + " AND ".join(conditions) + " GROUP BY f.memory_id"
                )
                for row in conn.execute(sql, params).fetchall():
                    status[row["memory_id"]] = {
                        "active": int(row["active"] or 0),
                        "superseded": int(row["superseded"] or 0),
                    }
        except Exception as e:
            logger.debug("get_fact_status skipped: %s", e)
        return status

    def resolve(
        self,
        query: str,
        query_plan: Optional[QueryPlan] = None,
        user_id: str = "default",
    ) -> Optional[ResolverResult]:
        """Try deterministic resolution.

        1. Auto-classify intent if no plan provided
        2. Context filtering: era -> place -> time_range -> activity
        3. Fact resolution: deterministic SQL over engram_facts
        4. Returns structured answer or None (fall through to vector search)
        """
        if not self._has_engram_tables():
            return None

        plan = query_plan or self._classify_query(query, user_id=user_id)

        # Apply context filters to narrow candidate memories
        context_ids = None
        if plan.context_filters:
            context_ids = self.filter_hierarchical(user_id=user_id, **plan.context_filters)
            if context_ids is not None and not context_ids:
                return None  # Context filter matched nothing

        # Try deterministic resolution based on intent
        if plan.intent == "count":
            return self._resolve_count(plan, context_ids, user_id)
        elif plan.intent == "latest":
            return self._resolve_latest(plan, context_ids, user_id)
        elif plan.intent == "set_members":
            return self._resolve_set_members(plan, context_ids, user_id)
        elif plan.intent == "sum":
            return self._resolve_sum(plan, context_ids, user_id)
        elif plan.intent == "temporal":
            return self._resolve_temporal(plan, context_ids, user_id)

        # For freeform, try fact-based resolution
        if plan.subject or plan.predicate:
            return self._resolve_fact_lookup(plan, context_ids, user_id)

        return None  # Fall through to vector search

    # ── Auto Query Classification ──

    def _classify_query(self, query: str, user_id: str = "default") -> QueryPlan:
        """Classify query intent and extract predicate from natural language.
        Zero-LLM, deterministic pattern matching."""
        plan = QueryPlan()

        if not query or not query.strip():
            return plan

        # Classify intent
        if _COUNT_RE.search(query):
            plan.intent = "count"
        elif _SUM_RE.search(query):
            plan.intent = "sum"
        elif _LATEST_RE.search(query):
            plan.intent = "latest"
            plan.subject = "user"
        elif _TEMPORAL_RE.search(query):
            plan.intent = "temporal"
            plan.subject = "user"
        elif _SET_RE.search(query):
            plan.intent = "set_members"

        # Extract predicate
        for pattern, predicate in _PREDICATE_PATTERNS:
            if pattern.search(query):
                plan.predicate = predicate
                break

        # Infer predicate from engram_facts when no regex match fired.
        # Running this on freeform queries too lets the resolver ground fact
        # lookups for any user question where we have matching stored facts
        # (e.g., "what time do I wake up?" → predicate `wake_time` if stored).
        # Safe because _resolve_fact_lookup filters valid_only=True — it can
        # only surface non-superseded facts.
        if not plan.predicate:
            plan.predicate = self._infer_predicate_from_db(query, user_id=user_id)

        # Default subject to "user" for any query — most user questions are
        # self-referential ("I", "my", "me"). If a fact lookup returns nothing
        # we fall back to vector search regardless.
        if not plan.subject:
            plan.subject = "user"

        return plan

    def _infer_predicate_from_db(self, query: str, user_id: Optional[str] = None) -> Optional[str]:
        """Try to match query keywords against existing predicates in engram_facts."""
        try:
            with self.db._get_connection() as conn:
                params: List[Any] = []
                sql = (
                    "SELECT DISTINCT f.predicate FROM engram_facts f "
                    "JOIN memories m ON m.id = f.memory_id "
                    "WHERE m.tombstone = 0"
                )
                if user_id:
                    sql += " AND m.user_id = ?"
                    params.append(user_id)
                rows = conn.execute(sql, params).fetchall()
                predicates = [r["predicate"] for r in rows]

            if not predicates:
                return None

            # Score each predicate by keyword overlap with query
            query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
            best_pred = None
            best_score = 0
            for pred in predicates:
                pred_words = set(pred.lower().replace("_", " ").split())
                overlap = len(query_words & pred_words)
                if overlap > best_score:
                    best_score = overlap
                    best_pred = pred

            return best_pred if best_score > 0 else None
        except Exception:
            return None

    # ── Context Filtering ──

    def filter_by_era(self, era: str, user_id: Optional[str] = None) -> List[str]:
        """Get memory IDs matching an era."""
        with self.db._get_connection() as conn:
            params: List[Any] = [era]
            sql = (
                "SELECT c.memory_id FROM engram_context c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE c.era = ? AND m.tombstone = 0"
            )
            if user_id:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            rows = conn.execute(sql, params).fetchall()
            return [r["memory_id"] for r in rows]

    def filter_by_place(self, place: str, user_id: Optional[str] = None) -> List[str]:
        """Get memory IDs matching a place."""
        with self.db._get_connection() as conn:
            params: List[Any] = [place, f"%{place}%"]
            sql = (
                "SELECT c.memory_id FROM engram_context c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE (c.place = ? OR c.place_detail LIKE ?) AND m.tombstone = 0"
            )
            if user_id:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            rows = conn.execute(sql, params).fetchall()
            return [r["memory_id"] for r in rows]

    def filter_by_time_range(
        self,
        start: str,
        end: str,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """Get memory IDs within a time range."""
        with self.db._get_connection() as conn:
            params: List[Any] = [start, end]
            sql = (
                "SELECT c.memory_id FROM engram_context c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE c.time_absolute BETWEEN ? AND ? AND m.tombstone = 0"
            )
            if user_id:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            rows = conn.execute(sql, params).fetchall()
            return [r["memory_id"] for r in rows]

    def filter_by_activity(self, activity: str, user_id: Optional[str] = None) -> List[str]:
        """Get memory IDs matching an activity."""
        with self.db._get_connection() as conn:
            params: List[Any] = [activity]
            sql = (
                "SELECT c.memory_id FROM engram_context c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE c.activity = ? AND m.tombstone = 0"
            )
            if user_id:
                sql += " AND m.user_id = ?"
                params.append(user_id)
            rows = conn.execute(sql, params).fetchall()
            return [r["memory_id"] for r in rows]

    def filter_hierarchical(
        self,
        era: Optional[str] = None,
        place: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None,
        activity: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[List[str]]:
        """Compound context filtering — narrows progressively."""
        conditions = []
        params = []

        if era:
            conditions.append("c.era = ?")
            params.append(era)
        if place:
            conditions.append("(c.place = ? OR c.place_detail LIKE ?)")
            params.extend([place, f"%{place}%"])
        if time_range and len(time_range) == 2:
            conditions.append("c.time_absolute BETWEEN ? AND ?")
            params.extend(time_range)
        if activity:
            conditions.append("c.activity = ?")
            params.append(activity)

        if not conditions:
            return None  # No filters applied

        query = (
            "SELECT DISTINCT c.memory_id FROM engram_context c "
            "JOIN memories m ON m.id = c.memory_id "
            f"WHERE {' AND '.join(conditions)} AND m.tombstone = 0"
        )
        if user_id:
            query += " AND m.user_id = ?"
            params.append(user_id)
        with self.db._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [r["memory_id"] for r in rows]

    # ── Deterministic Fact Resolution ──

    def resolve_count(
        self,
        predicate: str,
        context_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """COUNT(DISTINCT canonical_key) with optional context filter."""
        resolved = self._resolve_count_aggregate(predicate, context_ids, user_id)
        return resolved["count"] if resolved else 0

    def resolve_latest(
        self,
        subject: str,
        predicate: str,
        user_id: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Most recent valid fact.

        For preference-shaped predicates we try the engram_preferences
        store first — it is the substrate's first-class home for stance.
        If no active preference row exists, we fall back to engram_facts
        so the resolver stays truthful for mixed-history users.
        """
        pref = _classify_preference(predicate) if predicate else None
        if pref is not None and subject:
            topic, _ = pref
            pref_row = self.resolve_preference(
                subject=subject, topic=topic, user_id=user_id
            )
            if pref_row is not None:
                return pref_row

        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                subject=subject,
                predicate=predicate,
                context_ids=context_ids,
                valid_only=True,
            )
            # Tier priority ordering: canonical > high > medium > low.
            # 'avoid' is already filtered out by valid_only. Rows with NULL
            # tier (legacy) rank alongside 'medium'. Time breaks ties.
            tier_rank = (
                "CASE COALESCE(f.tier, 'medium') "
                "WHEN 'canonical' THEN 0 "
                "WHEN 'high' THEN 1 "
                "WHEN 'medium' THEN 2 "
                "WHEN 'low' THEN 3 "
                "ELSE 4 END"
            )
            row = conn.execute(
                "SELECT f.*" + from_clause
                + f" ORDER BY {tier_rank}, "
                + "COALESCE(f.valid_from, f.created_at) DESC LIMIT 1",
                params,
            ).fetchone()
            if row:
                return dict(row)
            return None

    def resolve_preference(
        self,
        subject: str,
        topic: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the active preference row for (user, subject, topic).

        Returns the row as a dict in the same shape as a fact row would
        be, so callers can treat the two sources uniformly. The row's
        ``predicate`` field is set to ``topic`` for symmetry.
        """
        try:
            with self.db._get_connection() as conn:
                conditions = ["superseded_by_id IS NULL",
                              "(tier IS NULL OR tier != 'avoid')",
                              "subject = ?", "topic = ?"]
                params: List[Any] = [subject, topic]
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                sql = (
                    "SELECT id, memory_id, user_id, subject, topic AS predicate, "
                    "stance, value, canonical_key, confidence, tier, "
                    "reaffirmed_count, last_reaffirmed_at, valid_from, "
                    "valid_until, created_at "
                    "FROM engram_preferences WHERE "
                    + " AND ".join(conditions)
                    + " ORDER BY CASE tier "
                    "WHEN 'canonical' THEN 0 WHEN 'high' THEN 1 "
                    "WHEN 'medium' THEN 2 WHEN 'low' THEN 3 ELSE 4 END, "
                    "COALESCE(valid_from, created_at) DESC LIMIT 1"
                )
                row = conn.execute(sql, params).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def resolve_set_members(
        self,
        predicate: str,
        context_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> List[str]:
        """DISTINCT values for a predicate."""
        rows = self._resolve_set_member_rows(predicate, context_ids, user_id)
        return [r["value"] for r in rows]

    def resolve_sum(
        self,
        predicate: str,
        unit: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> float:
        """SUM(value_numeric) with unit filtering."""
        resolved = self._resolve_sum_aggregate(predicate, unit, context_ids, user_id)
        return resolved["total"] if resolved else 0.0

    def resolve_temporal_sequence(
        self,
        subject: str,
        predicate: str,
        user_id: Optional[str] = None,
        context_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """All values ordered by time — shows change over time."""
        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                subject=subject,
                predicate=predicate,
                context_ids=context_ids,
            )
            rows = conn.execute(
                "SELECT f.*" + from_clause + " ORDER BY COALESCE(f.time, f.valid_from, f.created_at) ASC",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Associative Chain Traversal ──

    def walk_chain(
        self,
        start_canonical_key: str,
        link_type: Optional[str] = None,
        max_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        """Walk associative links from a memory.
        Returns chain of linked facts/memories."""
        visited = set()
        chain = []
        self._walk_chain_recursive(
            start_canonical_key, link_type, max_depth, 0, visited, chain
        )
        return chain

    def _walk_chain_recursive(
        self,
        canonical_key: str,
        link_type: Optional[str],
        max_depth: int,
        depth: int,
        visited: set,
        chain: list,
    ) -> None:
        if depth >= max_depth or canonical_key in visited:
            return
        visited.add(canonical_key)

        with self.db._get_connection() as conn:
            # Find the source fact/memory
            fact_row = conn.execute(
                "SELECT * FROM engram_facts WHERE canonical_key = ? LIMIT 1",
                (canonical_key,),
            ).fetchone()

            if fact_row:
                chain.append({
                    "depth": depth,
                    "canonical_key": canonical_key,
                    "fact": dict(fact_row),
                })

                # Find outgoing links
                if link_type:
                    link_rows = conn.execute(
                        """SELECT * FROM engram_links
                        WHERE source_memory_id = ? AND link_type = ?""",
                        (fact_row["memory_id"], link_type),
                    ).fetchall()
                else:
                    link_rows = conn.execute(
                        "SELECT * FROM engram_links WHERE source_memory_id = ?",
                        (fact_row["memory_id"],),
                    ).fetchall()

                for link in link_rows:
                    target_key = link["target_canonical_key"]
                    if target_key not in visited:
                        self._walk_chain_recursive(
                            target_key, link_type, max_depth,
                            depth + 1, visited, chain,
                        )

    def derive_time(self, canonical_key: str) -> Optional[str]:
        """Derive a memory's time from its associative chain."""
        chain = self.walk_chain(canonical_key, link_type="temporal_sequence", max_depth=5)
        for entry in chain:
            fact = entry.get("fact", {})
            time_val = fact.get("time") or fact.get("valid_from")
            if time_val:
                return time_val
        return None

    def find_co_occurring(self, canonical_key: str) -> List[Dict[str, Any]]:
        """Find memories that co-occurred with this one."""
        return self.walk_chain(canonical_key, link_type="co_occurring", max_depth=2)

    def reconstruct_scene(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Reconstruct the visual scene around a memory."""
        if not self._has_engram_tables():
            return None

        with self.db._get_connection() as conn:
            # Get scene snapshot
            scene_row = conn.execute(
                "SELECT * FROM engram_scenes WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()

            # Get context anchor
            ctx_row = conn.execute(
                "SELECT * FROM engram_context WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()

            if not scene_row and not ctx_row:
                return None

            result = {}
            if scene_row:
                result["scene"] = {
                    "setting": scene_row["setting"],
                    "people_present": json.loads(scene_row["people_present"] or "[]"),
                    "self_state": scene_row["self_state"],
                    "emotional_tone": scene_row["emotional_tone"],
                    "sensory_cues": json.loads(scene_row["sensory_cues"] or "[]"),
                }
            if ctx_row:
                result["context"] = {
                    "era": ctx_row["era"],
                    "place": ctx_row["place"],
                    "place_detail": ctx_row["place_detail"],
                    "activity": ctx_row["activity"],
                    "time_absolute": ctx_row["time_absolute"],
                }

            return result

    # ── Internal Resolution Helpers ──

    def _resolve_count_aggregate(
        self,
        predicate: str,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                predicate=predicate,
                context_ids=context_ids,
            )
            row = conn.execute(
                "SELECT COUNT(DISTINCT f.canonical_key) as cnt, "
                "GROUP_CONCAT(DISTINCT f.memory_id) as memory_ids" + from_clause,
                params,
            ).fetchone()
            if not row:
                return None
            memory_ids = self._split_grouped_memory_ids(row["memory_ids"])
            if not memory_ids:
                return None
            return {"count": int(row["cnt"] or 0), "memory_ids": memory_ids}

    def _resolve_set_member_rows(
        self,
        predicate: str,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                predicate=predicate,
                context_ids=context_ids,
            )
            rows = conn.execute(
                "SELECT DISTINCT f.value, f.memory_id" + from_clause + " ORDER BY f.value ASC",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

    def _resolve_sum_aggregate(
        self,
        predicate: str,
        unit: Optional[str],
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                predicate=predicate,
                context_ids=context_ids,
            )
            conditions = ["f.value_numeric IS NOT NULL"]
            if unit:
                conditions.append("f.value_unit = ?")
                params.append(unit)
            sql = (
                "SELECT SUM(f.value_numeric) as total, "
                "GROUP_CONCAT(DISTINCT f.memory_id) as memory_ids"
                + from_clause
            )
            if conditions:
                joiner = " AND " if " WHERE " in from_clause else " WHERE "
                sql += joiner + " AND ".join(conditions)
            row = conn.execute(sql, params).fetchone()
            if not row:
                return None
            memory_ids = self._split_grouped_memory_ids(row["memory_ids"])
            if not memory_ids or row["total"] is None:
                return None
            return {"total": float(row["total"]), "memory_ids": memory_ids}

    def _resolve_count(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        if not plan.predicate:
            return None
        resolved = self._resolve_count_aggregate(plan.predicate, context_ids, user_id)
        if not resolved:
            return None
        return ResolverResult(
            answer=str(resolved["count"]),
            resolver_path="context->sql->count",
            memory_ids=resolved["memory_ids"],
        )

    def _resolve_latest(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        subject = plan.subject or "user"
        predicate = plan.predicate
        if not predicate:
            return None
        fact = self.resolve_latest(subject, predicate, user_id=user_id, context_ids=context_ids)
        if not fact:
            return None
        return ResolverResult(
            answer=fact.get("value"),
            facts=[fact],
            memory_ids=[fact.get("memory_id", "")],
            resolver_path="context->sql->latest",
        )

    def _resolve_set_members(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        if not plan.predicate:
            return None
        rows = self._resolve_set_member_rows(plan.predicate, context_ids, user_id)
        if not rows:
            return None
        members = [row["value"] for row in rows]
        return ResolverResult(
            answer=", ".join(members),
            facts=[{"predicate": plan.predicate, "value": row["value"], "memory_id": row["memory_id"]} for row in rows],
            memory_ids=self._dedupe_memory_ids([row["memory_id"] for row in rows]),
            resolver_path="context->sql->set_members",
        )

    def _resolve_sum(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        if not plan.predicate:
            return None
        resolved = self._resolve_sum_aggregate(plan.predicate, None, context_ids, user_id)
        if not resolved:
            return None
        return ResolverResult(
            answer=str(resolved["total"]),
            memory_ids=resolved["memory_ids"],
            resolver_path="context->sql->sum",
        )

    def _resolve_temporal(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        subject = plan.subject or "user"
        predicate = plan.predicate
        if not predicate:
            return None
        sequence = self.resolve_temporal_sequence(
            subject,
            predicate,
            user_id=user_id,
            context_ids=context_ids,
        )
        if not sequence:
            return None
        formatted = [
            f"{s.get('value')} ({s.get('time', 'unknown time')})"
            for s in sequence
        ]
        return ResolverResult(
            answer=" -> ".join(formatted),
            facts=sequence,
            memory_ids=self._dedupe_memory_ids([s.get("memory_id", "") for s in sequence]),
            resolver_path="context->sql->temporal",
        )

    def _resolve_fact_lookup(
        self,
        plan: QueryPlan,
        context_ids: Optional[List[str]],
        user_id: Optional[str],
    ) -> Optional[ResolverResult]:
        """General fact lookup by subject and/or predicate.

        When both subject and predicate are specified, this is a "what's my
        current X?" query. We apply read-time latest-wins: the memory with
        the highest valid_from for (subject, predicate) is the authoritative
        grounding. This gives knowledge-update questions a supersede answer
        even when the write-side _SINGLE_VALUED_PREDICATES whitelist didn't
        catch the predicate — any fact with a newer valid_from outranks older
        ones at read time.
        """
        with self.db._get_connection() as conn:
            from_clause, params = self._fact_query_parts(
                user_id=user_id,
                subject=plan.subject,
                predicate=plan.predicate,
                context_ids=context_ids,
                valid_only=True,
            )
            if " WHERE " not in from_clause:
                return None

            query = "SELECT f.*" + from_clause + " ORDER BY COALESCE(f.valid_from, f.created_at) DESC LIMIT 10"
            rows = conn.execute(query, params).fetchall()
            if not rows:
                return None

            facts = [dict(r) for r in rows]

            # Read-time latest-wins: if subject+predicate were both specified
            # and multiple active facts exist, ground on the single latest one.
            # This implements fact-level supersede at read time for predicates
            # outside the write-side whitelist.
            latest_wins = bool(plan.subject and plan.predicate and len(facts) > 1)
            if latest_wins:
                grounding = facts[:1]
            else:
                grounding = facts

            return ResolverResult(
                answer=grounding[0].get("value") if len(grounding) == 1 else None,
                facts=grounding,
                memory_ids=self._dedupe_memory_ids([f.get("memory_id", "") for f in grounding]),
                resolver_path="context->sql->fact_lookup"
                    + ("->latest_wins" if latest_wins else ""),
            )

    def _has_engram_tables(self) -> bool:
        """Check if engram v3 tables exist."""
        try:
            with self.db._get_connection() as conn:
                conn.execute("SELECT 1 FROM engram_facts LIMIT 0")
                return True
        except Exception:
            return False

    # ── Engram Storage ──

    def _upsert_preference_row(
        self,
        conn,
        *,
        fact,
        memory_id: str,
        user_id: str,
        canonical: str,
        now_epoch: float,
    ) -> None:
        """Route preference-shaped facts to engram_preferences.

        Same substrate semantics as engram_facts (reaffirm / supersede /
        tier='medium' on new) but scoped to (user_id, subject, topic). A
        non-preference predicate returns without side effects.
        """
        pref = _classify_preference(fact.predicate)
        if pref is None:
            return
        topic, stance = pref
        pref_canonical = f"{user_id}|{fact.subject}|{topic}|{fact.value}"

        # Reaffirmation — exact value match on active row.
        existing = conn.execute(
            """SELECT id, reaffirmed_count FROM engram_preferences
            WHERE canonical_key = ?
              AND superseded_by_id IS NULL
              AND value = ?
            ORDER BY created_at ASC LIMIT 1""",
            (pref_canonical, fact.value),
        ).fetchone()
        if existing is not None:
            prev = int(existing["reaffirmed_count"] or 0)
            conn.execute(
                """UPDATE engram_preferences
                SET reaffirmed_count = ?,
                    last_reaffirmed_at = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?""",
                (prev + 1, now_epoch, existing["id"]),
            )
            return

        # Supersede — same (user, subject, topic) with different value.
        new_pref_id = str(uuid.uuid4())
        try:
            conn.execute(
                """UPDATE engram_preferences
                SET superseded_by_id = ?,
                    tier = 'avoid',
                    valid_until = COALESCE(valid_until, ?),
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND subject = ? AND topic = ?
                  AND value != ?
                  AND superseded_by_id IS NULL""",
                (
                    new_pref_id,
                    fact.valid_from or fact.time or "superseded",
                    user_id,
                    fact.subject,
                    topic,
                    fact.value,
                ),
            )
        except Exception:
            pass

        conn.execute(
            """INSERT INTO engram_preferences
            (id, memory_id, user_id, subject, topic, stance, value,
             canonical_key, confidence, tier, superseded_by_id,
             reaffirmed_count, last_reaffirmed_at,
             valid_from, valid_until, schema_v)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'medium', NULL,
                    0, NULL, ?, ?, 1)""",
            (
                new_pref_id,
                memory_id,
                user_id,
                fact.subject,
                topic,
                stance,
                fact.value,
                pref_canonical,
                fact.confidence if fact.confidence is not None else 1.0,
                fact.valid_from,
                fact.valid_until,
            ),
        )

    def store_engram(self, engram, memory_id: str) -> None:
        """Store engram structured data into the v3 tables.

        Args:
            engram: UniversalEngram instance
            memory_id: The memory ID from the memories table
        """
        if not self._has_engram_tables():
            return

        with self.db._get_connection() as conn:
            # Store context anchor
            if engram.context.has_context():
                conn.execute(
                    """INSERT OR REPLACE INTO engram_context
                    (memory_id, era, place, place_type, place_detail,
                     time_absolute, time_markers, time_range_start, time_range_end,
                     time_derivation, activity, session_id, session_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        engram.context.era,
                        engram.context.place,
                        engram.context.place_type,
                        engram.context.place_detail,
                        engram.context.time_absolute,
                        json.dumps(engram.context.time_markers),
                        engram.context.time_range_start,
                        engram.context.time_range_end,
                        engram.context.time_derivation,
                        engram.context.activity,
                        engram.context.session_id,
                        engram.context.session_position,
                    ),
                )

            # Store scene snapshot
            if engram.scene.setting or engram.scene.people_present:
                conn.execute(
                    """INSERT OR REPLACE INTO engram_scenes
                    (memory_id, setting, people_present, self_state,
                     emotional_tone, sensory_cues)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        memory_id,
                        engram.scene.setting,
                        json.dumps(engram.scene.people_present),
                        engram.scene.self_state,
                        engram.scene.emotional_tone,
                        json.dumps(engram.scene.sensory_cues),
                    ),
                )

            # Store facts (skip any with NULL required fields).
            #
            # M2 substrate semantics:
            #
            #   * Reaffirmation — the incoming fact is value-identical to an
            #     existing active row under the same canonical_key. Increment
            #     ``reaffirmed_count`` + stamp ``last_reaffirmed_at`` on the
            #     existing row and do NOT insert a duplicate. This is the
            #     load-bearing bit for tier promotion in M3.
            #
            #   * Supersede — the incoming fact contradicts a single-valued
            #     predicate (user moved cities, switched editors, etc.). The
            #     old row is not deleted: we set ``superseded_by_id = new.id``,
            #     demote its tier to ``'avoid'``, and stamp ``valid_until``.
            #     The new row lands at ``tier='medium'``. The chain stays
            #     explorable via ``dhee_why``.
            #
            #   * Preference routing — preference-shaped predicates also land
            #     in ``engram_preferences`` with stance + topic. The fact row
            #     remains for backwards compatibility; M2.4 teaches retrieval
            #     to prefer the preferences row when both exist.
            _user_id_row = conn.execute(
                "SELECT user_id FROM memories WHERE id = ? LIMIT 1",
                (memory_id,),
            ).fetchone()
            _user_id = (_user_id_row["user_id"] if _user_id_row else None) or "default"

            _SINGLE_VALUED_PREDICATES = {
                "lives_in", "works_at", "uses_editor", "prefers",
                "switched_to", "current_status", "has_title",
                "has_email", "has_phone", "has_address", "has_role",
                "uses_language", "subscribes_to",
            }

            for fact in engram.facts:
                if not fact.subject or not fact.predicate or not fact.value:
                    continue
                canonical = fact.canonical_key or f"{fact.subject}|{fact.predicate}|{fact.value}"
                now_iso = fact.valid_from or fact.time or ""
                now_epoch = time.time()

                # --- Reaffirmation path -------------------------------------
                reaffirmed = conn.execute(
                    """SELECT id, reaffirmed_count FROM engram_facts
                    WHERE canonical_key = ?
                      AND superseded_by_id IS NULL
                      AND value = ?
                    ORDER BY created_at ASC LIMIT 1""",
                    (canonical, fact.value),
                ).fetchone()
                if reaffirmed is not None:
                    prev = int(reaffirmed["reaffirmed_count"] or 0)
                    conn.execute(
                        """UPDATE engram_facts
                        SET reaffirmed_count = ?, last_reaffirmed_at = ?
                        WHERE id = ?""",
                        (prev + 1, now_epoch, reaffirmed["id"]),
                    )
                    # Still route preference rows so the preferences store
                    # sees the reaffirmation too.
                    self._upsert_preference_row(
                        conn,
                        fact=fact,
                        memory_id=memory_id,
                        user_id=_user_id,
                        canonical=canonical,
                        now_epoch=now_epoch,
                    )
                    continue

                # --- Supersede path -----------------------------------------
                pred_lower = _normalize_predicate(fact.predicate)
                # A dated fact is treated as the new value of its predicate —
                # unless the predicate is declared to hold many values at once.
                # Without that exception a dated "finds_hard: vectors" retired
                # every other difficulty the person had: measured on a student
                # store, ten of them, each "superseded by vectors".
                is_single_valued = pred_lower in _SINGLE_VALUED_PREDICATES or (
                    bool(fact.valid_from) and not getattr(fact, "multi_valued", False)
                )
                new_fact_id = str(uuid.uuid4())
                if is_single_valued and not fact.valid_until:
                    try:
                        conn.execute(
                            """UPDATE engram_facts
                            SET valid_until = ?,
                                superseded_by_id = ?,
                                tier = 'avoid'
                            WHERE subject = ? AND predicate = ?
                              AND valid_until IS NULL
                              AND memory_id != ?
                              AND value != ?""",
                            (
                                now_iso or "superseded",
                                new_fact_id,
                                fact.subject,
                                fact.predicate,
                                memory_id,
                                fact.value,
                            ),
                        )
                    except Exception:
                        pass

                # --- New row insert ----------------------------------------
                conn.execute(
                    """INSERT INTO engram_facts
                    (id, memory_id, subject, predicate, value,
                     value_numeric, value_unit, time, valid_from, valid_until,
                     qualifier, canonical_key, confidence, is_derived,
                     tier, reaffirmed_count, last_reaffirmed_at, schema_v)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            'medium', 0, NULL, 1)""",
                    (
                        new_fact_id,
                        memory_id,
                        fact.subject,
                        fact.predicate,
                        fact.value,
                        fact.value_numeric,
                        fact.value_unit,
                        fact.time,
                        fact.valid_from,
                        fact.valid_until,
                        fact.qualifier,
                        canonical,
                        fact.confidence,
                        1 if fact.is_derived else 0,
                    ),
                )

                # --- Preference routing ------------------------------------
                self._upsert_preference_row(
                    conn,
                    fact=fact,
                    memory_id=memory_id,
                    user_id=_user_id,
                    canonical=canonical,
                    now_epoch=now_epoch,
                )

            # Store entities
            for entity in engram.entities:
                entity_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO engram_entities
                    (id, memory_id, name, entity_type, state, relationships)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        entity_id,
                        memory_id,
                        entity.name,
                        entity.entity_type,
                        entity.state,
                        json.dumps(entity.relationships),
                    ),
                )

            # Store associative links
            for link in engram.links:
                link_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO engram_links
                    (id, source_memory_id, target_memory_id, target_canonical_key,
                     link_type, direction, qualifier)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        link_id,
                        memory_id,
                        link.target_memory_id,
                        link.target_canonical_key,
                        link.link_type,
                        link.direction,
                        link.qualifier,
                    ),
                )
