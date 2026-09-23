"""Memory write pipeline: processing, enrichment, indexing.

Extracted from memory/main.py — centralizes the full write path:
_process_single_memory, _process_single_memory_lite, and supporting
helpers (resolve_memory_metadata, encode_memory, extract_memories,
classify_memory_type, select_primary_text).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from dhee.core.conflict import resolve_conflict
from dhee.core.echo import EchoDepth, EchoResult
from dhee.core.traces import initialize_traces
from dhee.memory.cost import estimate_token_count, estimate_output_tokens
from dhee.memory.episodic import index_episodic_events_for_memory as _index_episodic
from dhee.memory.admission import (
    admission_expiration_date,
    evaluate_memory_candidate,
    sanitize_admitted_content,
)
from dhee.memory.quality import (
    apply_memory_quality_contract,
    enforce_quality_layer,
    enforce_quality_strength,
    is_protected_canonical_memory,
)
from dhee.memory.retrieval_helpers import (
    attach_bitemporal_metadata,
    normalize_bitemporal_value,
)
from dhee.memory.utils import (
    build_filters_and_metadata,
    normalize_categories,
    parse_messages,
    strip_code_fences,
)
from dhee.memory.vectors import build_index_vectors
from dhee.utils.prompts import AGENT_MEMORY_EXTRACTION_PROMPT, MEMORY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_MAX_ENRICHMENT_ATTEMPTS = 3


class MemoryWritePipeline:
    """Handles the full memory write path: processing, enrichment, indexing.

    Receives all dependencies via constructor so the class carries no hidden
    coupling to FullMemory internals.  Each ``self.*_fn`` callback is a thin
    reference to the corresponding method on the owning FullMemory instance.
    """

    def __init__(
        self,
        *,
        db,
        embedder,
        llm,
        config,
        vector_store=None,
        echo_processor_fn: Optional[Callable] = None,
        category_processor_fn: Optional[Callable] = None,
        graph_fn: Optional[Callable] = None,
        scene_processor_fn: Optional[Callable] = None,
        profile_processor_fn: Optional[Callable] = None,
        unified_enrichment_fn: Optional[Callable] = None,
        engram_extractor_fn: Optional[Callable] = None,
        context_resolver_fn: Optional[Callable] = None,
        evolution_layer_fn: Optional[Callable] = None,
        buddhi_layer_fn: Optional[Callable] = None,
        scope_resolver=None,
        executor=None,
        record_cost_fn: Optional[Callable] = None,
        forget_by_query_fn: Optional[Callable] = None,
        demote_existing_fn: Optional[Callable] = None,
        nearest_memory_fn: Optional[Callable] = None,
        assign_to_scene_fn: Optional[Callable] = None,
        update_profiles_fn: Optional[Callable] = None,
        store_prospective_scenes_fn: Optional[Callable] = None,
        persist_categories_fn: Optional[Callable] = None,
    ):
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._config = config
        self._vector_store = vector_store

        # Lazy-property callables — call to get the current processor instance.
        self._echo_processor_fn = echo_processor_fn
        self._category_processor_fn = category_processor_fn
        self._graph_fn = graph_fn
        self._scene_processor_fn = scene_processor_fn
        self._profile_processor_fn = profile_processor_fn
        self._unified_enrichment_fn = unified_enrichment_fn
        self._engram_extractor_fn = engram_extractor_fn
        self._context_resolver_fn = context_resolver_fn
        self._evolution_layer_fn = evolution_layer_fn
        self._buddhi_layer_fn = buddhi_layer_fn

        self._scope_resolver = scope_resolver
        self._executor = executor

        # Callback hooks into owning FullMemory.
        self._record_cost_fn = record_cost_fn
        self._forget_by_query_fn = forget_by_query_fn
        self._demote_existing_fn = demote_existing_fn
        self._nearest_memory_fn = nearest_memory_fn
        self._assign_to_scene_fn = assign_to_scene_fn
        self._update_profiles_fn = update_profiles_fn
        self._store_prospective_scenes_fn = store_prospective_scenes_fn
        self._persist_categories_fn = persist_categories_fn

    # ------------------------------------------------------------------
    # Convenience accessors for lazy processors
    # ------------------------------------------------------------------

    @property
    def _echo_processor(self):
        return self._echo_processor_fn() if self._echo_processor_fn else None

    @property
    def _category_processor(self):
        return self._category_processor_fn() if self._category_processor_fn else None

    @property
    def _graph(self):
        return self._graph_fn() if self._graph_fn else None

    @property
    def _scene_processor(self):
        return self._scene_processor_fn() if self._scene_processor_fn else None

    @property
    def _profile_processor(self):
        return self._profile_processor_fn() if self._profile_processor_fn else None

    @property
    def _unified_enrichment(self):
        return self._unified_enrichment_fn() if self._unified_enrichment_fn else None

    @property
    def _engram_extractor(self):
        return self._engram_extractor_fn() if self._engram_extractor_fn else None

    @property
    def _context_resolver(self):
        return self._context_resolver_fn() if self._context_resolver_fn else None

    @property
    def _evolution_layer(self):
        return self._evolution_layer_fn() if self._evolution_layer_fn else None

    @property
    def _buddhi_layer(self):
        return self._buddhi_layer_fn() if self._buddhi_layer_fn else None

    # ------------------------------------------------------------------
    # Config sub-sections (read-through to self._config)
    # ------------------------------------------------------------------

    @property
    def _fade_config(self):
        return self._config.fade

    @property
    def _echo_config(self):
        return self._config.echo

    @property
    def _category_config(self):
        return self._config.category

    @property
    def _graph_config(self):
        return self._config.graph

    @property
    def _distillation_config(self):
        return getattr(self._config, "distillation", None)

    @property
    def _parallel_config(self):
        return getattr(self._config, "parallel", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_cost(self, **kwargs) -> None:
        if self._record_cost_fn:
            self._record_cost_fn(**kwargs)

    def _normalize_agent_category(self, category):
        return self._scope_resolver.normalize_agent_category(category) if self._scope_resolver else category

    def _normalize_connector_id(self, connector_id):
        return self._scope_resolver.normalize_connector_id(connector_id) if self._scope_resolver else connector_id

    def _infer_scope(self, **kwargs):
        return self._scope_resolver.infer_scope(**kwargs) if self._scope_resolver else "agent"

    def _persist_categories(self) -> None:
        if self._persist_categories_fn:
            self._persist_categories_fn()

    def _render_existing_categories(self, limit: int = 30) -> Optional[str]:
        cat_proc = self._category_processor
        if not cat_proc:
            return None
        cats = cat_proc.get_all_categories()
        if not cats:
            return None
        return "\n".join(
            f"- {c['id']}: {c['name']} — {c.get('description', '')}"
            for c in cats[:limit]
        )

    def _insert_fact_vectors(
        self,
        *,
        fact_entries: List[Dict[str, Any]],
        warning_prefix: str,
    ) -> float:
        if not fact_entries or not self._vector_store:
            return 0.0

        embed_calls = 0.0
        fact_embeddings: List[List[float]] = []
        try:
            fact_texts = [entry["fact_text"] for entry in fact_entries]
            for start in range(0, len(fact_texts), 50):
                sub = fact_texts[start:start + 50]
                fact_embeddings.extend(
                    self._embedder.embed_batch(sub, memory_action="add")
                )
                embed_calls += 1.0

            fact_vectors: List[List[float]] = []
            fact_payloads: List[Dict[str, Any]] = []
            fact_ids: List[str] = []
            for entry, fact_embedding in zip(fact_entries, fact_embeddings):
                fact_vectors.append(fact_embedding)
                fact_payloads.append(
                    {
                        "memory_id": entry["memory_id"],
                        "is_fact": True,
                        "fact_index": entry["fact_index"],
                        "fact_text": entry["fact_text"],
                        "user_id": entry.get("user_id"),
                        "agent_id": entry.get("agent_id"),
                    }
                )
                fact_ids.append(
                    f"{entry['memory_id']}__fact_{entry['fact_index']}"
                )
            if fact_vectors:
                self._vector_store.insert(
                    vectors=fact_vectors,
                    payloads=fact_payloads,
                    ids=fact_ids,
                )
        except Exception as exc:
            logger.warning("%s: %s", warning_prefix, exc)
            return 0.0

        return embed_calls

    def _apply_entity_updates(
        self,
        *,
        memory_id: str,
        entities: Optional[List[Any]],
        warning_prefix: str,
        auto_link: bool = True,
    ) -> None:
        knowledge_graph = self._graph
        if not knowledge_graph or not entities:
            return
        try:
            for entity in entities:
                existing_ent = knowledge_graph._get_or_create_entity(
                    entity.name, entity.entity_type,
                )
                existing_ent.memory_ids.add(memory_id)
            knowledge_graph.memory_entities[memory_id] = {
                entity.name for entity in entities
            }
            if auto_link and self._graph_config.auto_link_entities:
                knowledge_graph.link_by_shared_entities(memory_id)
        except Exception as exc:
            logger.warning("%s for %s: %s", warning_prefix, memory_id, exc)

    def _apply_profile_updates_batch(
        self,
        *,
        memory_id: str,
        user_id: str,
        profile_updates: Optional[List[Any]],
        warning_prefix: str,
    ) -> None:
        profile_proc = self._profile_processor
        if not profile_proc or not profile_updates:
            return
        try:
            for profile_update in profile_updates:
                profile_proc.apply_update(
                    profile_update=profile_update,
                    memory_id=memory_id,
                    user_id=user_id,
                )
        except Exception as exc:
            logger.warning("%s for %s: %s", warning_prefix, memory_id, exc)

    @staticmethod
    def _context_messages_from_memory(memory: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        raw = memory.get("conversation_context")
        if isinstance(raw, list):
            return [item for item in raw if isinstance(item, dict)]
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, list):
            return None
        return [item for item in parsed if isinstance(item, dict)]

    @staticmethod
    def _enrichment_attempts(metadata: Dict[str, Any]) -> int:
        try:
            return max(0, int(metadata.get("enrichment_attempts") or 0))
        except (TypeError, ValueError):
            return 0

    def _run_engram_extraction(
        self,
        *,
        memory_id: str,
        content: str,
        mem_metadata: Dict[str, Any],
        user_id: Optional[str],
        context_messages: Optional[List[Dict[str, Any]]] = None,
        strict_llm_failure: bool = False,
    ) -> Tuple[bool, bool, Optional[Any]]:
        """Run structured engram extraction and store resolver rows.

        Returns ``(attempted, succeeded, engram)`` so callers can distinguish a
        disabled extractor from an extractor that failed.
        """
        engram_extractor = self._engram_extractor
        if not engram_extractor:
            return False, False, None
        if self._already_extracted(memory_id):
            # The write already extracted this memory; the enrichment pass used
            # to extract it again. Measured on a simulated student: every
            # conversation paid for extraction twice, and the second pass wrote
            # a paraphrase of a fact the first had stored.
            return True, True, None

        try:
            session_ctx = None
            if context_messages:
                session_ctx = {"recent_messages": context_messages[-5:]}
            decisions = getattr(self._config, "decisions", None)
            engram_extractor.vocabulary = (
                dict(decisions.fact_vocabulary)
                if decisions is not None and getattr(decisions, "enabled", False)
                else {}
            )
            engram = engram_extractor.extract(
                content=content,
                session_context=session_ctx,
                existing_metadata=mem_metadata,
                user_id=user_id or "default",
                strict_llm_failure=strict_llm_failure,
            )
            context_resolver = self._context_resolver
            if engram is not None and getattr(engram, "facts", None):
                engram.facts = self._gate_facts(content, list(engram.facts), context_resolver)
            if context_resolver and engram:
                context_resolver.store_engram(engram, memory_id)
            if (
                engram.prospective_scenes
                and self._config.prospective_scene.enable_prospective_scenes
                and self._store_prospective_scenes_fn
            ):
                self._store_prospective_scenes_fn(
                    engram.prospective_scenes,
                    memory_id,
                    user_id or "default",
                )
            self._mark_extracted(memory_id)
            return True, True, engram
        except Exception as exc:
            logger.warning("Engram extraction failed for %s: %s", memory_id, exc)
            return True, False, None

    def _already_extracted(self, memory_id: Optional[str]) -> bool:
        if not memory_id:
            return False
        try:
            with self._db._get_connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_extractions "
                    "(memory_id TEXT PRIMARY KEY, extracted_at TEXT DEFAULT CURRENT_TIMESTAMP)"
                )
                return (
                    conn.execute(
                        "SELECT 1 FROM engram_extractions WHERE memory_id = ?", (memory_id,)
                    ).fetchone()
                    is not None
                )
        except Exception:  # noqa: BLE001 - an unreadable ledger never blocks a write
            return False

    def _mark_extracted(self, memory_id: Optional[str]) -> None:
        if not memory_id:
            return
        try:
            with self._db._get_connection() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS engram_extractions "
                    "(memory_id TEXT PRIMARY KEY, extracted_at TEXT DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT OR IGNORE INTO engram_extractions (memory_id) VALUES (?)", (memory_id,)
                )
        except Exception:  # noqa: BLE001
            logger.debug("could not record extraction of %s", memory_id)

    def _fact_gate(self):
        """The configured fact gate, or None. Built once per decisions config."""
        config = getattr(self._config, "decisions", None)
        if config is None or not getattr(config, "enabled", False):
            return None
        marker = id(config), config.model_dump_json() if hasattr(config, "model_dump_json") else ""
        cached = getattr(self, "_fact_gate_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]
        from dhee.decisions import decider_from_config, fact_gate_from_config

        gate = fact_gate_from_config(config, decider_from_config(config))
        self._fact_gate_cache = (marker, gate)
        return gate

    def _gate_facts(self, content: str, facts: List[Any], context_resolver: Any) -> List[Any]:
        """Run extracted facts through the decision gate before they are stored.

        Never loses a write: with no gate, or a gate that cannot decide, the
        facts go to storage exactly as the extractor wrote them.
        """
        gate = self._fact_gate()
        if gate is None or not facts:
            return facts

        def existing(subject: str, predicate: str) -> List[Tuple[str, str]]:
            db = getattr(context_resolver, "db", None)
            if db is None:
                return []
            with db._get_connection() as conn:
                rows = conn.execute(
                    """SELECT value, canonical_key FROM engram_facts
                    WHERE subject = ? AND predicate = ? AND superseded_by_id IS NULL
                    ORDER BY created_at DESC LIMIT 24""",
                    (subject, predicate),
                ).fetchall()
            seen: set = set()
            out: List[Tuple[str, str]] = []
            for row in rows:
                value = str(row[0] or "")
                if value and value not in seen:
                    seen.add(value)
                    out.append((value, str(row[1] or "")))
            return out

        try:
            kept, report = gate.apply(content, facts, existing)
        except Exception as exc:  # noqa: BLE001 - a gate never costs a write
            logger.warning("Fact gate failed; storing facts as extracted: %s", exc)
            return facts
        if report.decided:
            logger.info(
                "Fact gate: kept %d of %d (off-subject %d, unlabelled %d, relabelled %d, restated %d)",
                report.kept,
                len(facts),
                report.dropped_off_subject,
                report.dropped_unlabelled,
                report.relabelled,
                report.merged,
            )
        return kept

    def _store_operational_event(
        self,
        *,
        content: str,
        mem_metadata: Dict[str, Any],
        user_id: Optional[str],
        agent_id: Optional[str],
        source_app: Optional[str],
        memory_id: Optional[str],
    ) -> Dict[str, Any]:
        """Store transport/tool signal as an episodic event without a memory row."""
        owner_user_id = user_id or str(mem_metadata.get("user_id") or "default")
        event_time = str(mem_metadata.get("event_time") or datetime.now(timezone.utc).isoformat())
        event_seed = "|".join(
            [
                owner_user_id,
                str(mem_metadata.get("source_event_id") or ""),
                str(mem_metadata.get("kind") or ""),
                content,
            ]
        )
        digest = hashlib.sha256(event_seed.encode("utf-8")).hexdigest()
        event_memory_id = memory_id or f"operational:{digest[:32]}"
        tool_name = str(mem_metadata.get("tool") or mem_metadata.get("tool_name") or "").strip()
        path = str(mem_metadata.get("path") or mem_metadata.get("source_path") or "").strip()
        actor_id = str(
            mem_metadata.get("actor_id")
            or agent_id
            or tool_name
            or source_app
            or mem_metadata.get("source_app")
            or "agent"
        ).strip()
        actor_role = str(mem_metadata.get("actor_role") or ("tool" if tool_name else "agent")).strip()
        event_kind = str(mem_metadata.get("kind") or mem_metadata.get("type") or "operational_event").strip()
        entity_key = path or tool_name or actor_id
        canonical_key = f"operational:{actor_id}:{event_kind}:{entity_key or digest[:12]}"
        value_norm = " ".join(content.lower().split())[:500]

        self._db.add_episodic_events(
            [
                {
                    "id": f"{event_memory_id}:event",
                    "memory_id": event_memory_id,
                    "user_id": owner_user_id,
                    "conversation_id": mem_metadata.get("conversation_id"),
                    "session_id": mem_metadata.get("session_id"),
                    "turn_id": mem_metadata.get("turn_id", 0),
                    "actor_id": actor_id,
                    "actor_role": actor_role,
                    "event_time": event_time,
                    "event_type": "operational_event",
                    "canonical_key": canonical_key,
                    "value_text": content,
                    "value_num": None,
                    "value_unit": None,
                    "currency": None,
                    "normalized_time_start": event_time,
                    "normalized_time_end": event_time,
                    "time_granularity": "instant",
                    "entity_key": entity_key,
                    "value_norm": value_norm,
                    "confidence": mem_metadata.get("confidence", 0.8),
                    "superseded_by": None,
                }
            ]
        )
        self._record_cost(
            phase="write",
            user_id=owner_user_id,
            llm_calls=0.0,
            input_tokens=0.0,
            output_tokens=0.0,
            embed_calls=0.0,
        )
        return {
            "id": event_memory_id,
            "memory": content,
            "event": "EVENT",
            "layer": "episodic",
            "strength": 0.0,
            "namespace": "operational",
            "memory_type": "operational_event",
            "episodic": True,
            "stored_as": "episodic_event",
        }

    # ------------------------------------------------------------------
    # Extracted public methods
    # ------------------------------------------------------------------

    def resolve_memory_metadata(
        self,
        *,
        content: str,
        mem_metadata: Dict[str, Any],
        explicit_remember: bool,
        agent_id: Optional[str],
        run_id: Optional[str],
        app_id: Optional[str],
        effective_filters: Dict[str, Any],
        agent_category: Optional[str],
        connector_id: Optional[str],
        scope: Optional[str],
        source_app: Optional[str],
    ) -> tuple:
        """Resolve store identifiers, scope, and metadata for a single memory."""
        store_agent_id = agent_id
        store_run_id = run_id
        store_app_id = app_id
        store_filters = dict(effective_filters)
        if "user_id" in store_filters or "agent_id" in store_filters:
            store_filters.pop("run_id", None)

        if explicit_remember:
            store_agent_id = None
            store_run_id = None
            store_app_id = None
            store_filters.pop("agent_id", None)
            store_filters.pop("run_id", None)
            store_filters.pop("app_id", None)
            mem_metadata.pop("agent_id", None)
            mem_metadata.pop("run_id", None)
            mem_metadata.pop("app_id", None)
            mem_metadata["policy_scope"] = "user"
        else:
            mem_metadata["policy_scope"] = "agent"

        mem_metadata["policy_explicit"] = explicit_remember
        resolved_agent_category = self._normalize_agent_category(
            agent_category or mem_metadata.get("agent_category")
        )
        resolved_connector_id = self._normalize_connector_id(
            connector_id or mem_metadata.get("connector_id")
        )
        resolved_scope = self._infer_scope(
            scope=scope or mem_metadata.get("scope"),
            connector_id=resolved_connector_id,
            agent_category=resolved_agent_category,
            policy_explicit=explicit_remember,
            agent_id=store_agent_id,
        )
        mem_metadata["scope"] = resolved_scope
        if resolved_agent_category:
            mem_metadata["agent_category"] = resolved_agent_category
        if resolved_connector_id:
            mem_metadata["connector_id"] = resolved_connector_id
        if source_app or mem_metadata.get("source_app"):
            mem_metadata["source_app"] = source_app or mem_metadata.get("source_app")

        return store_agent_id, store_run_id, store_app_id, store_filters

    def encode_memory(
        self,
        content: str,
        echo_depth: Optional[str],
        mem_categories: List[str],
        mem_metadata: Dict[str, Any],
        initial_strength: float,
    ) -> tuple:
        """Run echo encoding + embedding.

        Returns ``(echo_result, effective_strength, mem_categories, embedding)``.
        """
        echo_result = None
        effective_strength = initial_strength
        echo_proc = self._echo_processor
        if echo_proc and self._echo_config.enable_echo:
            depth_override = EchoDepth(echo_depth) if echo_depth else None
            echo_result = echo_proc.process(content, depth=depth_override)
            effective_strength = initial_strength * echo_result.strength_multiplier
            mem_metadata.update(echo_result.to_metadata())
            if not mem_categories and echo_result.category:
                mem_categories = [echo_result.category]

        primary_text = self.select_primary_text(content, echo_result)
        embedding = self._embedder.embed(primary_text, memory_action="add")
        return echo_result, effective_strength, mem_categories, embedding

    def process_single_memory(
        self,
        *,
        mem: Dict[str, Any],
        processed_metadata: Dict[str, Any],
        effective_filters: Dict[str, Any],
        categories: Optional[List[str]],
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        app_id: Optional[str],
        agent_category: Optional[str],
        connector_id: Optional[str],
        scope: Optional[str],
        source_app: Optional[str],
        immutable: bool,
        expiration_date: Optional[str],
        initial_layer: str,
        initial_strength: float,
        echo_depth: Optional[str],
        memory_id: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process and store a single memory item. Returns result dict or None if skipped."""
        # Late import to avoid circular dep — these are module-level functions in main.py.
        from dhee.memory.main import detect_explicit_intent, detect_sensitive_categories, is_ephemeral, looks_high_confidence

        content = mem.get("content", "").strip()
        if not content:
            return None

        write_llm_calls = 0.0
        write_embed_calls = 0.0
        write_input_tokens = 0.0
        write_output_tokens = 0.0

        def _add_llm_cost(input_tokens: float) -> None:
            nonlocal write_llm_calls, write_input_tokens, write_output_tokens
            tokens = max(0.0, float(input_tokens or 0.0))
            write_llm_calls += 1.0
            write_input_tokens += tokens
            write_output_tokens += estimate_output_tokens(tokens)

        mem_categories = normalize_categories(categories or mem.get("categories"))
        mem_metadata = dict(processed_metadata)
        mem_metadata.update(mem.get("metadata", {}))
        if app_id:
            mem_metadata["app_id"] = app_id

        role = mem_metadata.get("role", "user")
        explicit_intent = detect_explicit_intent(content) if role == "user" else None
        explicit_action = explicit_intent.action if explicit_intent else None
        explicit_remember = bool(mem_metadata.get("explicit_remember")) or explicit_action == "remember"
        explicit_forget = bool(mem_metadata.get("explicit_forget")) or explicit_action == "forget"

        if explicit_forget:
            query = explicit_intent.content if explicit_intent else ""
            forget_filters = {"user_id": user_id} if user_id else dict(effective_filters)
            forget_result = self._forget_by_query_fn(query, forget_filters)
            return {
                "event": "FORGET",
                "query": query,
                "deleted_count": forget_result.get("deleted_count", 0),
                "deleted_ids": forget_result.get("deleted_ids", []),
            }

        if explicit_remember and explicit_intent and explicit_intent.content:
            content = explicit_intent.content

        mem_metadata, quality = apply_memory_quality_contract(
            content,
            mem_metadata,
            mem_categories,
            explicit_remember=explicit_remember,
        )
        if quality.layer == "lml" and initial_layer == "auto":
            initial_layer = "lml"
        if quality.layer == "sml":
            initial_layer = "sml"
        initial_strength = enforce_quality_strength(initial_strength, quality)
        if quality.is_canonical:
            expiration_date = None

        if quality.is_operational:
            return self._store_operational_event(
                content=content,
                mem_metadata=mem_metadata,
                user_id=user_id,
                agent_id=agent_id,
                source_app=source_app,
                memory_id=memory_id,
            )

        admission = evaluate_memory_candidate(
            content,
            mem_metadata,
            explicit_remember=explicit_remember,
        )
        if admission.applies:
            mem_metadata["dhee_admission"] = admission.to_metadata()
            mem_metadata["dhee_passive_observation"] = True
            if not explicit_remember:
                mem_metadata["dhee_lite_path"] = True
            mem_metadata["retention_policy"] = admission.retention_policy
            mem_metadata["confidence"] = admission.confidence
            mem_metadata["admission_score"] = admission.score
            mem_metadata["admission_promotion_reason"] = admission.promotion_reason
            if not admission.should_store:
                return {
                    "event": "SKIP",
                    "reason": admission.skip_reason or "admission_rejected",
                    "memory": content,
                    "admission": admission.to_metadata(),
                }
            if expiration_date is None:
                expiration_date = admission_expiration_date(admission.retention_policy)
            if admission.retention_policy != "durable":
                initial_strength = min(initial_strength, max(0.2, admission.confidence))
            content = sanitize_admitted_content(content, admission)
            mem_metadata, quality = apply_memory_quality_contract(
                content,
                mem_metadata,
                mem_categories,
                explicit_remember=explicit_remember,
            )
            initial_strength = enforce_quality_strength(initial_strength, quality)
            if quality.is_canonical:
                expiration_date = None
            if quality.is_operational:
                return self._store_operational_event(
                    content=content,
                    mem_metadata=mem_metadata,
                    user_id=user_id,
                    agent_id=agent_id,
                    source_app=source_app,
                    memory_id=memory_id,
                )

        blocked = detect_sensitive_categories(content)
        # allow_sensitive: explicit caller opt-in, or caller explicitly provided
        # the content (infer=False / user_provided=True).  PII detection is a
        # guardrail for agent-inferred memories from raw conversation, not for
        # bulk corpus ingestion where the caller owns the content decision.
        allow_sensitive = (
            bool(mem_metadata.get("allow_sensitive"))
            or bool(mem_metadata.get("user_provided"))
        )
        if blocked and not allow_sensitive:
            return {
                "event": "BLOCKED",
                "reason": "sensitive",
                "blocked_categories": blocked,
                "memory": content,
            }

        is_task_or_note = (mem_metadata or {}).get("memory_type") in ("task", "note")
        is_user_provided = bool(mem_metadata.get("user_provided"))
        if not explicit_remember and not is_task_or_note and not is_user_provided and is_ephemeral(content):
            return {
                "event": "SKIP",
                "reason": "ephemeral",
                "memory": content,
            }

        # --- Deferred enrichment: lite path (0 LLM calls) ---
        enrichment_config = getattr(self._config, "enrichment", None)
        use_lite_path = bool(
            (enrichment_config and enrichment_config.defer_enrichment)
            or mem_metadata.get("dhee_lite_path")
        )
        if use_lite_path:
            return self.process_single_memory_lite(
                content=content,
                mem_metadata=mem_metadata,
                mem_categories=mem_categories,
                context_messages=context_messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                app_id=app_id,
                effective_filters=effective_filters,
                agent_category=agent_category,
                connector_id=connector_id,
                scope=scope,
                source_app=source_app,
                immutable=immutable,
                expiration_date=expiration_date,
                initial_layer=initial_layer,
                initial_strength=initial_strength,
                explicit_remember=explicit_remember,
                memory_id=memory_id,
            )

        # Resolve store identifiers and scope metadata.
        store_agent_id, store_run_id, store_app_id, store_filters = self.resolve_memory_metadata(
            content=content,
            mem_metadata=mem_metadata,
            explicit_remember=explicit_remember,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            effective_filters=effective_filters,
            agent_category=agent_category,
            connector_id=connector_id,
            scope=scope,
            source_app=source_app,
        )

        high_confidence = explicit_remember or looks_high_confidence(content, mem_metadata)
        policy_repeated = False
        low_confidence = False

        # Determine if we should auto-categorize
        cat_proc = self._category_processor
        _should_categorize = (
            cat_proc
            and self._category_config.auto_categorize
            and not mem_categories
        )

        # Pre-extracted data from unified enrichment
        _unified_entities = None
        _unified_profiles = None
        _unified_facts = None

        # Determine echo depth for unified path check
        echo_proc = self._echo_processor
        _depth_for_echo = EchoDepth(echo_depth) if echo_depth else None
        if _depth_for_echo is None and echo_proc and hasattr(echo_proc, '_assess_depth'):
            try:
                _depth_for_echo = echo_proc._assess_depth(content)
            except Exception:
                _depth_for_echo = EchoDepth.MEDIUM

        # Site 0: Unified enrichment (single LLM call for echo+category+entities+profiles)
        unified = self._unified_enrichment
        _use_unified = (
            unified is not None
            and self._echo_config.enable_echo
            and _depth_for_echo != EchoDepth.SHALLOW
        )

        if _use_unified:
            enrichment_config = getattr(self._config, "enrichment", None)
            existing_cats = None
            if cat_proc:
                cats = cat_proc.get_all_categories()
                if cats:
                    existing_cats = "\n".join(
                        f"- {c['id']}: {c['name']} — {c.get('description', '')}"
                        for c in cats[:30]
                    )

            unified_input_tokens = estimate_token_count(content) + estimate_token_count(existing_cats)
            _add_llm_cost(unified_input_tokens)

            enrichment = unified.enrich(
                content=content,
                depth=_depth_for_echo or EchoDepth.MEDIUM,
                existing_categories=existing_cats,
                include_entities=enrichment_config.include_entities if enrichment_config else True,
                include_profiles=enrichment_config.include_profiles if enrichment_config else True,
            )

            # Apply echo result
            echo_result = enrichment.echo_result
            if echo_result:
                effective_strength = initial_strength * echo_result.strength_multiplier
                mem_metadata.update(echo_result.to_metadata())
                if not mem_categories and echo_result.category:
                    mem_categories = [echo_result.category]
            else:
                effective_strength = initial_strength

            # Apply category result
            if enrichment.category_match and not mem_categories:
                mem_categories = [enrichment.category_match.category_id]
                mem_metadata["category_confidence"] = enrichment.category_match.confidence
                mem_metadata["category_auto"] = True

            # Stash entities + profiles + facts for post-store hooks
            _unified_entities = enrichment.entities
            _unified_profiles = enrichment.profile_updates
            _unified_facts = enrichment.facts

            # Generate embedding
            primary_text = self.select_primary_text(content, echo_result)
            embedding = self._embedder.embed(primary_text, memory_action="add")
            write_embed_calls += 1.0

        else:
            # Site 1: Parallel echo encoding + category detection
            _use_parallel = (
                self._executor is not None
                and self._parallel_config
                and self._parallel_config.parallel_add
                and _should_categorize
                and echo_proc
                and self._echo_config.enable_echo
            )

            if _use_parallel:
                depth_for_parallel = EchoDepth(echo_depth) if echo_depth else (_depth_for_echo or EchoDepth(self._echo_config.default_depth))
                if self._echo_config.enable_echo and depth_for_parallel != EchoDepth.SHALLOW:
                    _add_llm_cost(estimate_token_count(content))
                if _should_categorize and self._category_config.use_llm_categorization:
                    _add_llm_cost(estimate_token_count(content))

                def _do_echo():
                    depth_override = EchoDepth(echo_depth) if echo_depth else None
                    return echo_proc.process(content, depth=depth_override)

                def _do_category():
                    return cat_proc.detect_category(
                        content,
                        metadata=mem_metadata,
                        use_llm=self._category_config.use_llm_categorization,
                    )

                echo_result_p, category_match = self._executor.run_parallel([
                    (_do_echo, ()),
                    (_do_category, ()),
                ])

                # Apply echo result
                effective_strength = initial_strength * echo_result_p.strength_multiplier
                mem_metadata.update(echo_result_p.to_metadata())
                if not mem_categories and echo_result_p.category:
                    mem_categories = [echo_result_p.category]

                # Apply category result
                mem_categories = [category_match.category_id]
                mem_metadata["category_confidence"] = category_match.confidence
                mem_metadata["category_auto"] = True

                # Generate embedding (depends on echo result, must be serial)
                primary_text = self.select_primary_text(content, echo_result_p)
                embedding = self._embedder.embed(primary_text, memory_action="add")
                write_embed_calls += 1.0
                echo_result = echo_result_p
            else:
                # Sequential path (original behavior)
                if _should_categorize:
                    if self._category_config.use_llm_categorization:
                        _add_llm_cost(estimate_token_count(content))
                    category_match = cat_proc.detect_category(
                        content,
                        metadata=mem_metadata,
                        use_llm=self._category_config.use_llm_categorization,
                    )
                    mem_categories = [category_match.category_id]
                    mem_metadata["category_confidence"] = category_match.confidence
                    mem_metadata["category_auto"] = True

                # Encode memory (echo + embedding).
                depth_for_encode = EchoDepth(echo_depth) if echo_depth else (_depth_for_echo or EchoDepth(self._echo_config.default_depth))
                if self._echo_config.enable_echo and depth_for_encode != EchoDepth.SHALLOW:
                    _add_llm_cost(estimate_token_count(content))
                echo_result, effective_strength, mem_categories, embedding = self.encode_memory(
                    content, echo_depth, mem_categories, mem_metadata, initial_strength,
                )
                write_embed_calls += 1.0

        nearest, similarity = self._nearest_memory_fn(embedding, store_filters)
        repeated_threshold = max(self._fade_config.conflict_similarity_threshold - 0.05, 0.7)
        if similarity >= repeated_threshold:
            policy_repeated = True
            high_confidence = True

        if not explicit_remember and not high_confidence:
            low_confidence = True

        # Conflict resolution against nearest memory in scope.
        event = "ADD"
        existing = None
        resolution = None
        deferred_demotion: Optional[Tuple[Dict[str, Any], str]] = None
        if nearest and similarity >= self._fade_config.conflict_similarity_threshold:
            existing = nearest

        if existing and self._fade_config.enable_forgetting:
            conflict_input_tokens = estimate_token_count(existing.get("memory", "")) + estimate_token_count(content)
            _add_llm_cost(conflict_input_tokens)
            resolution = resolve_conflict(existing, content, self._llm, self._config.custom_conflict_prompt)

            if resolution.classification == "CONTRADICTORY":
                if is_protected_canonical_memory(existing) and not quality.is_canonical:
                    mem_metadata["conflicts_with_memory_id"] = existing.get("id")
                    mem_metadata["conflict_resolution"] = "CONTRADICTORY"
                    mem_metadata["supersede_blocked_reason"] = "protected_canonical_existing"
                    event = "ADD"
                else:
                    deferred_demotion = (existing, "CONTRADICTORY")
                    event = "UPDATE"
            elif resolution.classification == "SUBSUMES":
                content = resolution.merged_content or content
                mem_metadata, quality = apply_memory_quality_contract(
                    content,
                    mem_metadata,
                    mem_categories,
                    explicit_remember=explicit_remember,
                )
                if is_protected_canonical_memory(existing) and not quality.is_canonical:
                    mem_metadata["conflicts_with_memory_id"] = existing.get("id")
                    mem_metadata["conflict_resolution"] = "SUBSUMES"
                    mem_metadata["supersede_blocked_reason"] = "protected_canonical_existing"
                    event = "ADD"
                else:
                    deferred_demotion = (existing, "SUBSUMES")
                    event = "UPDATE"
            elif resolution.classification == "SUBSUMED":
                boosted_strength = min(1.0, float(existing.get("strength", 1.0)) + 0.05)
                self._db.update_memory(existing["id"], {"strength": boosted_strength})
                self._db.increment_access(existing["id"])
                self._record_cost(
                    phase="write",
                    user_id=user_id,
                    llm_calls=write_llm_calls,
                    input_tokens=write_input_tokens,
                    output_tokens=write_output_tokens,
                    embed_calls=write_embed_calls,
                )
                return {
                    "id": existing["id"],
                    "memory": existing.get("memory", ""),
                    "event": "NOOP",
                    "layer": existing.get("layer", "sml"),
                    "strength": boosted_strength,
                }

        if existing and event == "UPDATE" and resolution and resolution.classification == "SUBSUMES":
            # Re-encode merged content.
            depth_for_encode = EchoDepth(echo_depth) if echo_depth else (_depth_for_echo or EchoDepth(self._echo_config.default_depth))
            if self._echo_config.enable_echo and depth_for_encode != EchoDepth.SHALLOW:
                _add_llm_cost(estimate_token_count(content))
            echo_result, _, mem_categories, embedding = self.encode_memory(
                content, echo_depth, mem_categories, mem_metadata, initial_strength,
            )
            write_embed_calls += 1.0

        if policy_repeated:
            mem_metadata["policy_repeated"] = True
        if low_confidence:
            mem_metadata["policy_low_confidence"] = True
            effective_strength = min(effective_strength, 0.4)

        effective_strength = enforce_quality_strength(effective_strength, quality)
        layer = initial_layer
        if layer == "auto":
            layer = "sml"
        if low_confidence:
            layer = "sml"
        layer = enforce_quality_layer(layer, quality)

        confidentiality_scope = str(
            mem_metadata.get("confidentiality_scope")
            or mem_metadata.get("privacy_scope")
            or "work"
        ).lower()
        source_type = (
            mem_metadata.get("source_type")
            or ("cli" if (source_app or "").lower() == "cli" else "mcp")
        )
        namespace_value = str(mem_metadata.get("namespace", "default") or "default").strip() or "default"

        # Gap 1: Classify memory type (episodic vs semantic)
        memory_type = self.classify_memory_type(mem_metadata, role)

        # Gap 4: Initialize multi-trace strength
        s_fast_val = None
        s_mid_val = None
        s_slow_val = None
        distillation_config = self._distillation_config
        if distillation_config and distillation_config.enable_multi_trace:
            s_fast_val, s_mid_val, s_slow_val = initialize_traces(effective_strength, is_new=True)

        # Metamemory: compute confidence score if enabled
        if self._config.metamemory.enable_confidence:
            try:
                from engram_metamemory.confidence import compute_confidence as _mm_confidence
                mem_metadata["mm_confidence"] = _mm_confidence(
                    metadata=mem_metadata,
                    strength=effective_strength,
                    access_count=0,
                    created_at=None,
                )
            except ImportError:
                pass

        effective_memory_id = memory_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        mem_metadata = attach_bitemporal_metadata(mem_metadata, observed_time=now)
        memory_data = {
            "id": effective_memory_id,
            "memory": content,
            "user_id": user_id,
            "agent_id": store_agent_id,
            "run_id": store_run_id,
            "app_id": store_app_id,
            "metadata": mem_metadata,
            "categories": mem_categories,
            "immutable": immutable,
            "expiration_date": expiration_date,
            "created_at": now,
            "updated_at": now,
            "layer": layer,
            "strength": effective_strength,
            "access_count": 0,
            "last_accessed": now,
            "embedding": embedding,
            "confidentiality_scope": confidentiality_scope,
            "source_type": source_type,
            "source_app": source_app or mem_metadata.get("source_app"),
            "source_event_id": mem_metadata.get("source_event_id"),
            "decay_lambda": mem_metadata.get("decay_lambda", self._fade_config.sml_decay_rate),
            "status": "active",
            "importance": mem_metadata.get("importance", 0.5),
            "sensitivity": mem_metadata.get("sensitivity", "normal"),
            "namespace": namespace_value,
            "memory_type": memory_type,
            "s_fast": s_fast_val,
            "s_mid": s_mid_val,
            "s_slow": s_slow_val,
        }

        vectors, payloads, vector_ids = build_index_vectors(
            memory_id=effective_memory_id,
            content=content,
            primary_text=self.select_primary_text(content, echo_result),
            embedding=embedding,
            echo_result=echo_result,
            metadata=mem_metadata,
            categories=mem_categories,
            user_id=user_id,
            agent_id=store_agent_id,
            run_id=store_run_id,
            app_id=store_app_id,
            embedder=self._embedder,
        )

        self._db.add_memory(memory_data)
        if vectors:
            try:
                self._vector_store.insert(vectors=vectors, payloads=payloads, ids=vector_ids)
            except Exception as e:
                logger.error(
                    "Vector insert failed for memory %s, rolling back DB record: %s",
                    effective_memory_id, e,
                )
                try:
                    self._db.delete_memory(effective_memory_id, use_tombstone=False)
                except Exception as rollback_err:
                    logger.critical(
                        "CRITICAL: DB rollback also failed for memory %s — manual cleanup required: %s",
                        effective_memory_id, rollback_err,
                    )
                raise

        if deferred_demotion and self._demote_existing_fn:
            demote_memory, demote_reason = deferred_demotion
            self._demote_existing_fn(
                demote_memory,
                reason=demote_reason,
                superseded_by=effective_memory_id,
                force=quality.is_canonical,
            )

        # Fact decomposition
        if _unified_facts:
            valid_facts = []
            for i, fact_text in enumerate(_unified_facts[:8]):
                fact_text = fact_text.strip()
                if fact_text and len(fact_text) >= 10:
                    valid_facts.append((i, fact_text))

            if valid_facts:
                try:
                    fact_texts = [ft for _, ft in valid_facts]
                    fact_embeddings = self._embedder.embed_batch(fact_texts, memory_action="add")
                    write_embed_calls += 1.0
                    fact_vectors = []
                    fact_payloads = []
                    fact_ids = []
                    for (i, fact_text), fact_embedding in zip(valid_facts, fact_embeddings):
                        fact_id = f"{effective_memory_id}__fact_{i}"
                        fact_vectors.append(fact_embedding)
                        fact_payloads.append({
                            "memory_id": effective_memory_id,
                            "is_fact": True,
                            "fact_index": i,
                            "fact_text": fact_text,
                            "user_id": user_id,
                            "agent_id": store_agent_id,
                        })
                        fact_ids.append(fact_id)
                    if fact_vectors:
                        self._vector_store.insert(vectors=fact_vectors, payloads=fact_payloads, ids=fact_ids)
                except Exception as e:
                    logger.warning("Fact embedding/insert failed for %s: %s", effective_memory_id, e)

        # Post-store hooks.
        if cat_proc and mem_categories:
            for cat_id in mem_categories:
                cat_proc.update_category_stats(
                    cat_id, effective_strength, is_addition=True
                )

        knowledge_graph = self._graph
        if knowledge_graph:
            if _unified_entities is not None:
                for entity in _unified_entities:
                    existing_ent = knowledge_graph._get_or_create_entity(
                        entity.name, entity.entity_type,
                    )
                    existing_ent.memory_ids.add(effective_memory_id)
                knowledge_graph.memory_entities[effective_memory_id] = {
                    e.name for e in _unified_entities
                }
            else:
                if self._graph_config.use_llm_extraction:
                    _add_llm_cost(estimate_token_count(content))
                knowledge_graph.extract_entities(
                    content=content,
                    memory_id=effective_memory_id,
                    use_llm=self._graph_config.use_llm_extraction,
                )
            if self._graph_config.auto_link_entities:
                knowledge_graph.link_by_shared_entities(effective_memory_id)

        if self._scene_processor:
            try:
                self._assign_to_scene_fn(effective_memory_id, content, embedding, user_id, now)
            except Exception as e:
                logger.warning("Scene assignment failed for %s: %s", effective_memory_id, e)

        if self._profile_processor:
            try:
                if _unified_profiles is not None and _unified_profiles:
                    profile_proc = self._profile_processor
                    for profile_update in _unified_profiles:
                        profile_proc.apply_update(
                            profile_update=profile_update,
                            memory_id=effective_memory_id,
                            user_id=user_id or "default",
                        )
                else:
                    if self._config.profile.use_llm_extraction:
                        _add_llm_cost(estimate_token_count(content))
                    self._update_profiles_fn(effective_memory_id, content, mem_metadata, user_id)
            except Exception as e:
                logger.warning("Profile update failed for %s: %s", effective_memory_id, e)

        _index_episodic(
            db=self._db,
            config=self._config,
            memory_id=effective_memory_id,
            user_id=user_id,
            content=content,
            metadata=mem_metadata,
        )

        # Dhee: Universal Engram extraction
        _engram_attempted, _engram_succeeded, engram = self._run_engram_extraction(
            memory_id=effective_memory_id,
            content=content,
            mem_metadata=mem_metadata,
            user_id=user_id,
            context_messages=context_messages,
            strict_llm_failure=False,
        )

        # Dhee: Self-evolution — record extraction quality signal
        evolution_layer = self._evolution_layer
        if evolution_layer:
            try:
                engram_facts = None
                engram_context = None
                if _engram_succeeded and engram:
                    engram_facts = [f.to_dict() if hasattr(f, 'to_dict') else f for f in getattr(engram, 'facts', [])]
                    engram_context = getattr(engram, 'context', None)
                    if engram_context and hasattr(engram_context, '__dict__'):
                        engram_context = engram_context.__dict__
                evolution_layer.on_memory_stored(
                    memory_id=effective_memory_id,
                    content=content,
                    facts=engram_facts,
                    context=engram_context,
                    user_id=user_id or "default",
                )
            except Exception as e:
                logger.debug("Evolution write hook skipped: %s", e)

        # Buddhi write hook: detect intentions in stored content
        buddhi_layer = self._buddhi_layer
        if buddhi_layer:
            try:
                buddhi_layer.on_memory_stored(
                    content=content,
                    user_id=user_id or "default",
                )
            except Exception as e:
                logger.debug("Buddhi write hook skipped: %s", e)

        self._record_cost(
            phase="write",
            user_id=user_id,
            llm_calls=write_llm_calls,
            input_tokens=write_input_tokens,
            output_tokens=write_output_tokens,
            embed_calls=write_embed_calls,
        )

        return {
            "id": effective_memory_id,
            "memory": content,
            "event": event,
            "layer": layer,
            "strength": effective_strength,
            "echo_depth": echo_result.echo_depth.value if echo_result else None,
            "categories": mem_categories,
            "namespace": namespace_value,
            "vector_nodes": len(vectors),
            "memory_type": memory_type,
            "admission": mem_metadata.get("dhee_admission"),
        }

    def process_single_memory_lite(
        self,
        *,
        content: str,
        mem_metadata: Dict[str, Any],
        mem_categories: List[str],
        context_messages: Optional[List[Dict[str, str]]],
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        app_id: Optional[str],
        effective_filters: Dict[str, Any],
        agent_category: Optional[str],
        connector_id: Optional[str],
        scope: Optional[str],
        source_app: Optional[str],
        immutable: bool,
        expiration_date: Optional[str],
        initial_layer: str,
        initial_strength: float,
        explicit_remember: bool,
        memory_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Lite processing path for deferred enrichment -- 0 LLM calls.

        Stores the memory with regex-extracted keywords, context-enriched
        embedding, and enrichment_status='pending'.  All heavy LLM processing
        (echo, category, conflict, entities, profiles) is deferred to
        enrich_pending().
        """
        from dhee.memory.main import (
            looks_high_confidence,
            _NAME_HINT_RE,
            _PREFERENCE_HINT_RE,
            _ROUTINE_HINT_RE,
            _GOAL_HINT_RE,
        )

        # Resolve store identifiers and scope metadata.
        store_agent_id, store_run_id, store_app_id, store_filters = self.resolve_memory_metadata(
            content=content,
            mem_metadata=mem_metadata,
            explicit_remember=explicit_remember,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            effective_filters=effective_filters,
            agent_category=agent_category,
            connector_id=connector_id,
            scope=scope,
            source_app=source_app,
        )

        high_confidence = explicit_remember or looks_high_confidence(content, mem_metadata)
        mem_metadata, quality = apply_memory_quality_contract(
            content,
            mem_metadata,
            mem_categories,
            explicit_remember=explicit_remember,
        )

        # --- Regex keyword extraction (0 LLM calls) ---
        extracted_keywords: List[str] = []
        content_lower = content.lower()

        for regex, tag in [
            (_PREFERENCE_HINT_RE, "preference"),
            (_ROUTINE_HINT_RE, "routine"),
            (_GOAL_HINT_RE, "goal"),
        ]:
            if regex.search(content):
                extracted_keywords.append(tag)

        _STOPWORDS = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "nor", "not", "so", "yet", "both", "either", "neither", "each",
            "every", "all", "any", "few", "more", "most", "other", "some", "such",
            "no", "only", "own", "same", "than", "too", "very", "just", "i", "me",
            "my", "we", "our", "you", "your", "he", "she", "it", "they", "them",
            "this", "that", "these", "those", "am", "his", "her", "its",
        }
        words = re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", content_lower)
        word_freq: Dict[str, int] = {}
        for w in words:
            if w not in _STOPWORDS:
                word_freq[w] = word_freq.get(w, 0) + 1
        top_words = sorted(word_freq, key=lambda w: word_freq[w], reverse=True)[:15]
        extracted_keywords.extend(top_words)

        name_match = _NAME_HINT_RE.search(content)
        if name_match:
            extracted_keywords.append(f"name:{name_match.group(1).strip()}")

        mem_metadata["echo_keywords"] = extracted_keywords
        mem_metadata["enrichment_status"] = "pending"

        # --- Build rich embedding text (content + context summary) ---
        context_window = getattr(self._config.enrichment, "context_window_turns", 10)
        context_summary = ""
        if context_messages:
            recent = context_messages[-context_window:]
            context_lines = [
                f"{m.get('role', 'user')}: {str(m.get('content', ''))[:200]}"
                for m in recent
            ]
            context_summary = " | ".join(context_lines)

        embed_text = content
        if context_summary:
            embed_text += f" [Context: {context_summary[:500]}]"

        # --- Generate embedding (1 API call, NOT an LLM call) ---
        embedding = self._embedder.embed(embed_text, memory_action="add")

        # --- Confidence and layer ---
        effective_strength = initial_strength
        if not explicit_remember and not high_confidence:
            mem_metadata["policy_low_confidence"] = True
            effective_strength = min(effective_strength, 0.4)
        effective_strength = enforce_quality_strength(effective_strength, quality)

        layer = initial_layer
        if layer == "auto":
            layer = "sml"
        layer = enforce_quality_layer(layer, quality)
        if quality.is_canonical:
            expiration_date = None

        # --- Metadata ---
        confidentiality_scope = str(
            mem_metadata.get("confidentiality_scope")
            or mem_metadata.get("privacy_scope")
            or "work"
        ).lower()
        source_type = (
            mem_metadata.get("source_type")
            or ("cli" if (source_app or "").lower() == "cli" else "mcp")
        )
        namespace_value = str(mem_metadata.get("namespace", "default") or "default").strip() or "default"
        memory_type = self.classify_memory_type(mem_metadata, mem_metadata.get("role", "user"))

        # Multi-trace strength
        s_fast_val = s_mid_val = s_slow_val = None
        distillation_config = self._distillation_config
        if distillation_config and distillation_config.enable_multi_trace:
            s_fast_val, s_mid_val, s_slow_val = initialize_traces(effective_strength, is_new=True)

        # Content hash for dedup
        from dhee.memory.core import _content_hash
        ch = _content_hash(content)
        existing = self._db.get_memory_by_content_hash(ch, user_id) if hasattr(self._db, 'get_memory_by_content_hash') else None
        if existing:
            self._db.increment_access(existing["id"])
            return {
                "id": existing["id"],
                "memory": existing.get("memory", ""),
                "event": "DEDUPLICATED",
                "layer": existing.get("layer", "sml"),
                "strength": existing.get("strength", 1.0),
            }

        effective_memory_id = memory_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        mem_metadata = attach_bitemporal_metadata(mem_metadata, observed_time=now)

        # Serialize conversation context
        context_json = None
        if context_messages:
            recent = context_messages[-context_window:]
            context_json = json.dumps(recent)

        memory_data = {
            "id": effective_memory_id,
            "memory": content,
            "user_id": user_id,
            "agent_id": store_agent_id,
            "run_id": store_run_id,
            "app_id": store_app_id,
            "metadata": mem_metadata,
            "categories": mem_categories,
            "immutable": immutable,
            "expiration_date": expiration_date,
            "created_at": now,
            "updated_at": now,
            "layer": layer,
            "strength": effective_strength,
            "access_count": 0,
            "last_accessed": now,
            "embedding": embedding,
            "confidentiality_scope": confidentiality_scope,
            "source_type": source_type,
            "source_app": source_app or mem_metadata.get("source_app"),
            "source_event_id": mem_metadata.get("source_event_id"),
            "decay_lambda": mem_metadata.get("decay_lambda", self._fade_config.sml_decay_rate),
            "status": "active",
            "importance": mem_metadata.get("importance", 0.5),
            "sensitivity": mem_metadata.get("sensitivity", "normal"),
            "namespace": namespace_value,
            "memory_type": memory_type,
            "s_fast": s_fast_val,
            "s_mid": s_mid_val,
            "s_slow": s_slow_val,
            "content_hash": ch,
            "conversation_context": context_json,
            "enrichment_status": "pending",
        }

        # Build vector index (single primary vector, no echo nodes)
        base_payload = {
            "memory_id": effective_memory_id,
            "user_id": user_id,
            "agent_id": store_agent_id,
            "run_id": store_run_id,
            "app_id": store_app_id,
            "categories": mem_categories,
            "text": embed_text,
            "type": "primary",
            "memory": content,
        }
        vectors = [embedding]
        payloads = [base_payload]
        vector_ids = [effective_memory_id]

        self._db.add_memory(memory_data)
        try:
            self._vector_store.insert(vectors=vectors, payloads=payloads, ids=vector_ids)
        except Exception as e:
            logger.error("Vector insert failed for memory %s (lite), rolling back: %s", effective_memory_id, e)
            try:
                self._db.delete_memory(effective_memory_id, use_tombstone=False)
            except Exception as rollback_err:
                logger.critical("DB rollback also failed for %s: %s", effective_memory_id, rollback_err)
            raise

        # Scene assignment still works (embedding-based, no LLM)
        if self._scene_processor:
            try:
                self._assign_to_scene_fn(effective_memory_id, content, embedding, user_id, now)
            except Exception as e:
                logger.warning("Scene assignment failed for %s (lite): %s", effective_memory_id, e)

        _index_episodic(
            db=self._db,
            config=self._config,
            memory_id=effective_memory_id,
            user_id=user_id,
            content=content,
            metadata=mem_metadata,
        )
        self._record_cost(
            phase="write",
            user_id=user_id,
            llm_calls=0.0,
            input_tokens=0.0,
            output_tokens=0.0,
            embed_calls=1.0,
        )

        return {
            "id": effective_memory_id,
            "memory": content,
            "event": "ADD",
            "layer": layer,
            "strength": effective_strength,
            "echo_depth": None,
            "categories": mem_categories,
            "namespace": namespace_value,
            "vector_nodes": 1,
            "memory_type": memory_type,
            "enrichment_status": "pending",
            "admission": mem_metadata.get("dhee_admission"),
        }

    def process_memory_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        app_id: Optional[str],
        metadata: Optional[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
        initial_strength: float,
        echo_depth: Optional[str],
        batch_config,
        **common_kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Process a batch of memory items with batched echo/embed/DB."""
        contents: List[str] = []
        item_metadata_list: List[Dict[str, Any]] = []
        for item in items:
            content = item.get("content") or item.get("messages", "")
            if isinstance(content, list):
                content = " ".join(
                    m.get("content", "") for m in content if isinstance(m, dict)
                )
            contents.append(str(content).strip())
            item_meta = dict(metadata or {})
            item_meta.update(item.get("metadata") or {})
            item_metadata_list.append(item_meta)

        batch_llm_calls_total = 0.0
        batch_embed_calls_total = 0.0
        batch_input_tokens_total = 0.0
        batch_output_tokens_total = 0.0

        echo_results = [None] * len(contents)
        category_results = [None] * len(contents)
        enrichment_results = [None] * len(contents)

        enrichment_config = getattr(self._config, "enrichment", None)
        echo_proc = self._echo_processor
        cat_proc = self._category_processor
        unified = self._unified_enrichment
        use_unified = (
            unified is not None
            and self._echo_config.enable_echo
            and batch_config.batch_echo
        )

        if use_unified:
            try:
                depth_override = (
                    EchoDepth(echo_depth)
                    if echo_depth
                    else EchoDepth(self._echo_config.default_depth)
                )
                existing_cats = self._render_existing_categories()
                enrich_batch_size = (
                    enrichment_config.max_batch_size if enrichment_config else 10
                )
                for start in range(0, len(contents), enrich_batch_size):
                    end = min(start + enrich_batch_size, len(contents))
                    sub_contents = contents[start:end]
                    sub_results = unified.enrich_batch(
                        sub_contents,
                        depth=depth_override,
                        existing_categories=existing_cats,
                        include_entities=(
                            enrichment_config.include_entities
                            if enrichment_config
                            else True
                        ),
                        include_profiles=(
                            enrichment_config.include_profiles
                            if enrichment_config
                            else True
                        ),
                    )
                    sub_input_tokens = sum(
                        estimate_token_count(c) for c in sub_contents
                    )
                    sub_input_tokens += estimate_token_count(existing_cats)
                    batch_llm_calls_total += 1.0
                    batch_input_tokens_total += sub_input_tokens
                    batch_output_tokens_total += estimate_output_tokens(
                        sub_input_tokens
                    )
                    for offset, enrichment in enumerate(sub_results):
                        idx = start + offset
                        if enrichment.echo_result:
                            echo_results[idx] = enrichment.echo_result
                        if enrichment.category_match:
                            category_results[idx] = enrichment.category_match
                        enrichment_results[idx] = enrichment
                logger.info(
                    "Unified batch enrichment completed for %d memories",
                    len(contents),
                )
            except Exception as exc:
                logger.warning(
                    "Unified batch enrichment failed, falling back to separate: %s",
                    exc,
                )
                echo_results = [None] * len(contents)
                category_results = [None] * len(contents)
                enrichment_results = [None] * len(contents)
                use_unified = False

        if not use_unified:
            if echo_proc and self._echo_config.enable_echo and batch_config.batch_echo:
                depth_override = (
                    EchoDepth(echo_depth)
                    if echo_depth
                    else EchoDepth(self._echo_config.default_depth)
                )
                if depth_override != EchoDepth.SHALLOW:
                    echo_input_tokens = sum(
                        estimate_token_count(c) for c in contents if c
                    )
                    non_empty_count = sum(1 for c in contents if c)
                    batch_llm_calls_total += float(non_empty_count)
                    batch_input_tokens_total += echo_input_tokens
                    batch_output_tokens_total += estimate_output_tokens(
                        echo_input_tokens
                    )
                try:
                    echo_results = echo_proc.process_batch(
                        contents, depth=depth_override
                    )
                except Exception as exc:
                    logger.warning(
                        "Batch echo failed, processing individually: %s",
                        exc,
                    )
                    for idx, content in enumerate(contents):
                        if not content:
                            continue
                        try:
                            depth_override = (
                                EchoDepth(echo_depth) if echo_depth else None
                            )
                            echo_results[idx] = echo_proc.process(
                                content, depth=depth_override
                            )
                        except Exception as fallback_exc:
                            logger.debug(
                                "Individual echo fallback failed for batch item %d: %s",
                                idx,
                                fallback_exc,
                            )

            if (
                cat_proc
                and self._category_config.auto_categorize
                and batch_config.batch_category
            ):
                if self._category_config.use_llm_categorization:
                    cat_input_tokens = sum(
                        estimate_token_count(c) for c in contents if c
                    )
                    non_empty_count = sum(1 for c in contents if c)
                    batch_llm_calls_total += float(non_empty_count)
                    batch_input_tokens_total += cat_input_tokens
                    batch_output_tokens_total += estimate_output_tokens(
                        cat_input_tokens
                    )
                try:
                    category_results = cat_proc.detect_categories_batch(
                        contents,
                        use_llm=self._category_config.use_llm_categorization,
                    )
                except Exception as exc:
                    logger.warning("Batch category failed: %s", exc)

        primary_texts: List[str] = []
        for idx, content in enumerate(contents):
            primary_texts.append(
                self.select_primary_text(content, echo_results[idx])
            )

        if batch_config.batch_embed:
            try:
                embeddings: List[List[float]] = []
                for start in range(0, len(primary_texts), 50):
                    sub = primary_texts[start:start + 50]
                    embeddings.extend(
                        self._embedder.embed_batch(sub, memory_action="add")
                    )
                    batch_embed_calls_total += 1.0
            except Exception as exc:
                logger.warning(
                    "Batch embed failed, falling back to sequential: %s",
                    exc,
                )
                embeddings = [
                    self._embedder.embed(text, memory_action="add")
                    for text in primary_texts
                ]
                batch_embed_calls_total += float(len(primary_texts))
        else:
            embeddings = [
                self._embedder.embed(text, memory_action="add")
                for text in primary_texts
            ]
            batch_embed_calls_total += float(len(primary_texts))

        echo_node_texts: List[str] = []
        for idx, content in enumerate(contents):
            echo_result = echo_results[idx]
            primary_text = primary_texts[idx]
            if primary_text != content:
                cleaned = content.strip()
                if cleaned:
                    echo_node_texts.append(cleaned)
            if echo_result:
                for paraphrase in echo_result.paraphrases:
                    cleaned = str(paraphrase).strip()
                    if cleaned:
                        echo_node_texts.append(cleaned)
                for question in echo_result.questions:
                    cleaned = str(question).strip()
                    if cleaned:
                        echo_node_texts.append(cleaned)

        embedding_cache: Dict[str, List[float]] = {}
        if echo_node_texts:
            unique_texts = list(dict.fromkeys(echo_node_texts))
            try:
                all_echo_embeddings: List[List[float]] = []
                for start in range(0, len(unique_texts), 50):
                    sub = unique_texts[start:start + 50]
                    all_echo_embeddings.extend(
                        self._embedder.embed_batch(sub, memory_action="add")
                    )
                    batch_embed_calls_total += 1.0
                for text, emb in zip(unique_texts, all_echo_embeddings):
                    embedding_cache[text] = emb
                logger.info(
                    "Batch-embedded %d echo node texts in %d API calls",
                    len(unique_texts),
                    (len(unique_texts) + 49) // 50,
                )
            except Exception as exc:
                logger.warning(
                    "Batch echo node embedding failed, will embed individually: %s",
                    exc,
                )

        processed_metadata_base, effective_filters = build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
            input_filters=filters,
        )
        if app_id:
            processed_metadata_base["app_id"] = app_id

        now = datetime.now(timezone.utc).isoformat()
        memory_records: List[Dict[str, Any]] = []
        record_entries: List[Dict[str, Any]] = []
        episodic_rows: List[Tuple[str, Optional[str], str, Dict[str, Any]]] = []
        vector_batch: List[Tuple[List[List[float]], List[Dict[str, Any]], List[str]]] = []
        results: List[Dict[str, Any]] = []

        for idx, content in enumerate(contents):
            if not content:
                continue

            owner_user_id = items[idx].get("user_id") or user_id
            memory_id = str(uuid.uuid4())
            mem_metadata = dict(processed_metadata_base)
            mem_metadata.update(item_metadata_list[idx])

            echo_result = echo_results[idx]
            effective_strength = initial_strength
            mem_categories = list(items[idx].get("categories") or [])
            explicit_batch_remember = bool(mem_metadata.get("explicit_remember") or mem_metadata.get("policy_explicit"))
            mem_metadata, quality = apply_memory_quality_contract(
                content,
                mem_metadata,
                mem_categories,
                explicit_remember=explicit_batch_remember,
            )
            mem_metadata = attach_bitemporal_metadata(
                mem_metadata, observed_time=now
            )

            if echo_result:
                effective_strength = (
                    initial_strength * echo_result.strength_multiplier
                )
                mem_metadata.update(echo_result.to_metadata())
                if not mem_categories and echo_result.category:
                    mem_categories = [echo_result.category]

            cat_match = category_results[idx]
            if cat_match and not mem_categories:
                mem_categories = [cat_match.category_id]
                mem_metadata["category_confidence"] = cat_match.confidence
                mem_metadata["category_auto"] = True

            effective_strength = enforce_quality_strength(effective_strength, quality)
            batch_layer = enforce_quality_layer("sml", quality)
            embedding = embeddings[idx]
            namespace_value = str(
                mem_metadata.get("namespace", "default") or "default"
            ).strip() or "default"
            memory_type = self.classify_memory_type(
                mem_metadata, mem_metadata.get("role", "user")
            )

            s_fast_val = s_mid_val = s_slow_val = None
            distillation_config = self._distillation_config
            if distillation_config and distillation_config.enable_multi_trace:
                s_fast_val, s_mid_val, s_slow_val = initialize_traces(
                    effective_strength, is_new=True
                )

            memory_data = {
                "id": memory_id,
                "memory": content,
                "user_id": owner_user_id,
                "agent_id": items[idx].get("agent_id") or agent_id,
                "run_id": items[idx].get("run_id") or run_id,
                "app_id": items[idx].get("app_id") or app_id,
                "metadata": mem_metadata,
                "categories": mem_categories,
                "immutable": items[idx].get("immutable", False),
                "expiration_date": items[idx].get("expiration_date"),
                "created_at": now,
                "updated_at": now,
                "layer": batch_layer,
                "strength": effective_strength,
                "access_count": 0,
                "last_accessed": now,
                "embedding": embedding,
                "confidentiality_scope": "work",
                "source_type": "mcp",
                "source_app": items[idx].get("source_app"),
                "source_event_id": mem_metadata.get("source_event_id"),
                "decay_lambda": mem_metadata.get("decay_lambda", self._fade_config.sml_decay_rate),
                "status": "active",
                "importance": mem_metadata.get("importance", 0.5),
                "sensitivity": mem_metadata.get("sensitivity", "normal"),
                "namespace": namespace_value,
                "memory_type": memory_type,
                "s_fast": s_fast_val,
                "s_mid": s_mid_val,
                "s_slow": s_slow_val,
            }
            memory_records.append(memory_data)
            record_entries.append(
                {
                    "source_index": idx,
                    "record": memory_data,
                    "user_id": owner_user_id,
                    "content": content,
                }
            )
            episodic_rows.append((memory_id, owner_user_id, content, mem_metadata))

            vectors, payloads, vector_ids = build_index_vectors(
                memory_id=memory_id,
                content=content,
                primary_text=primary_texts[idx],
                embedding=embedding,
                echo_result=echo_result,
                metadata=mem_metadata,
                categories=mem_categories,
                user_id=owner_user_id,
                agent_id=items[idx].get("agent_id") or agent_id,
                run_id=items[idx].get("run_id") or run_id,
                app_id=items[idx].get("app_id") or app_id,
                embedder=self._embedder,
                embedding_cache=embedding_cache if embedding_cache else None,
            )
            if vectors:
                vector_batch.append((vectors, payloads, vector_ids))

            results.append(
                {
                    "id": memory_id,
                    "memory": content,
                    "event": "ADD",
                    "layer": batch_layer,
                    "strength": effective_strength,
                    "echo_depth": (
                        echo_result.echo_depth.value if echo_result else None
                    ),
                    "categories": mem_categories,
                    "namespace": namespace_value,
                    "memory_type": memory_type,
                }
            )

        if memory_records:
            try:
                self._db.add_memories_batch(memory_records)
            except Exception as exc:
                logger.error(
                    "Batch DB insert failed, falling back to sequential: %s",
                    exc,
                )
                for record in memory_records:
                    self._db.add_memory(record)

        for vectors, payloads, vector_ids in vector_batch:
            try:
                self._vector_store.insert(
                    vectors=vectors,
                    payloads=payloads,
                    ids=vector_ids,
                )
            except Exception as exc:
                logger.error("Vector insert failed in batch: %s", exc)

        for memory_id, owner_user_id, content, mem_metadata in episodic_rows:
            _index_episodic(
                db=self._db,
                config=self._config,
                memory_id=memory_id,
                user_id=owner_user_id,
                content=content,
                metadata=mem_metadata,
            )

        if cat_proc:
            for entry in record_entries:
                record = entry["record"]
                if record.get("categories"):
                    for cat_id in record["categories"]:
                        cat_proc.update_category_stats(
                            cat_id,
                            record["strength"],
                            is_addition=True,
                        )

        fact_entries: List[Dict[str, Any]] = []
        for entry in record_entries:
            idx = entry["source_index"]
            enrichment = enrichment_results[idx]
            if not enrichment or not enrichment.facts:
                continue
            for fact_index, fact_text in enumerate(enrichment.facts[:8]):
                cleaned = fact_text.strip()
                if cleaned and len(cleaned) >= 10:
                    fact_entries.append(
                        {
                            "memory_id": entry["record"]["id"],
                            "fact_index": fact_index,
                            "fact_text": cleaned,
                            "user_id": entry["user_id"],
                            "agent_id": entry["record"].get("agent_id"),
                        }
                    )
        batch_embed_calls_total += self._insert_fact_vectors(
            fact_entries=fact_entries,
            warning_prefix="Batch fact embedding/insert failed",
        )

        for entry in record_entries:
            idx = entry["source_index"]
            record = entry["record"]
            enrichment = enrichment_results[idx]
            if enrichment:
                self._apply_entity_updates(
                    memory_id=record["id"],
                    entities=enrichment.entities,
                    warning_prefix="Entity linking failed",
                )
                self._apply_profile_updates_batch(
                    memory_id=record["id"],
                    user_id=record.get("user_id") or user_id or "default",
                    profile_updates=enrichment.profile_updates,
                    warning_prefix="Profile update failed",
                )

            self._run_engram_extraction(
                memory_id=record["id"],
                content=record.get("memory", ""),
                mem_metadata=record.get("metadata") or {},
                user_id=record.get("user_id") or user_id,
                context_messages=None,
                strict_llm_failure=False,
            )

        if episodic_rows:
            sample_count = float(len(episodic_rows))
            llm_calls_per_memory = batch_llm_calls_total / sample_count
            input_tokens_per_memory = batch_input_tokens_total / sample_count
            output_tokens_per_memory = batch_output_tokens_total / sample_count
            embed_calls_per_memory = batch_embed_calls_total / sample_count
            for _, owner_user_id, _, _ in episodic_rows:
                self._record_cost(
                    phase="write",
                    user_id=owner_user_id,
                    llm_calls=llm_calls_per_memory,
                    input_tokens=input_tokens_per_memory,
                    output_tokens=output_tokens_per_memory,
                    embed_calls=embed_calls_per_memory,
                )

        return results

    def enrich_pending(
        self,
        *,
        user_id: str = "default",
        batch_size: int = 10,
        max_batches: int = 5,
    ) -> Dict[str, Any]:
        """Batch-enrich memories that were stored with deferred enrichment."""
        limit = batch_size * max_batches
        pending = self._db.get_pending_enrichment(user_id=user_id, limit=limit)
        if not pending:
            return {
                "enriched_count": 0,
                "failed_count": 0,
                "degraded_count": 0,
                "batches": 0,
                "remaining": 0,
            }

        enriched_count = 0
        failed_count = 0
        degraded_count = 0
        batches_processed = 0
        unified = self._unified_enrichment

        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            contents = [memory.get("memory", "") for memory in batch]

            enrichment_results = None
            if unified is not None:
                try:
                    enrichment_results = unified.enrich_batch(
                        contents,
                        depth=EchoDepth.MEDIUM,
                        existing_categories=self._render_existing_categories(),
                        include_entities=True,
                        include_profiles=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "Unified batch enrichment failed in enrich_pending: %s",
                        exc,
                    )
                    enrichment_results = None

            if enrichment_results is None:
                enrichment_results = []
                for content in contents:
                    if unified is not None:
                        try:
                            enrichment_results.append(
                                unified.enrich(content, depth=EchoDepth.MEDIUM)
                            )
                        except Exception as exc:
                            logger.debug(
                                "Single-memory enrichment fallback failed: %s",
                                exc,
                            )
                            enrichment_results.append(None)
                    else:
                        enrichment_results.append(None)

            db_updates: List[Dict[str, Any]] = []
            fact_entries: List[Dict[str, Any]] = []

            for memory, enrichment in zip(batch, enrichment_results):
                mem_id = memory["id"]
                mem_meta = memory.get("metadata", {}) or {}
                mem_cats = memory.get("categories", []) or []
                attempts = self._enrichment_attempts(mem_meta) + 1
                mem_meta["enrichment_attempts"] = attempts
                mem_meta["last_enrichment_attempt_at"] = datetime.now(timezone.utc).isoformat()

                if enrichment:
                    if enrichment.echo_result:
                        mem_meta.update(enrichment.echo_result.to_metadata())
                        if not mem_cats and enrichment.echo_result.category:
                            mem_cats = [enrichment.echo_result.category]

                    if enrichment.category_match and not mem_cats:
                        mem_cats = [enrichment.category_match.category_id]
                        mem_meta["category_confidence"] = (
                            enrichment.category_match.confidence
                        )
                        mem_meta["category_auto"] = True

                    if enrichment.facts:
                        mem_meta["enrichment_facts"] = enrichment.facts[:8]

                    self._apply_entity_updates(
                        memory_id=mem_id,
                        entities=enrichment.entities,
                        warning_prefix="Entity linking failed during enrichment",
                    )
                    self._apply_profile_updates_batch(
                        memory_id=mem_id,
                        user_id=user_id,
                        profile_updates=enrichment.profile_updates,
                        warning_prefix="Profile update failed during enrichment",
                    )

                    if enrichment.facts:
                        for fact_index, fact_text in enumerate(enrichment.facts[:8]):
                            cleaned = fact_text.strip()
                            if cleaned and len(cleaned) >= 10:
                                fact_entries.append(
                                    {
                                        "memory_id": mem_id,
                                        "fact_index": fact_index,
                                        "fact_text": cleaned,
                                        "user_id": user_id,
                                        "agent_id": memory.get("agent_id"),
                                    }
                                )

                extraction_attempted, extraction_succeeded, _engram = self._run_engram_extraction(
                    memory_id=mem_id,
                    content=memory.get("memory", ""),
                    mem_metadata=mem_meta,
                    user_id=memory.get("user_id") or user_id,
                    context_messages=self._context_messages_from_memory(memory),
                    strict_llm_failure=True,
                )
                if extraction_attempted:
                    status_ready = extraction_succeeded
                else:
                    status_ready = enrichment is not None

                if status_ready:
                    mem_meta["enrichment_status"] = "complete"
                    mem_meta.pop("last_enrichment_error", None)
                    next_status = "complete"
                    enriched_count += 1
                else:
                    if attempts >= _MAX_ENRICHMENT_ATTEMPTS:
                        next_status = "degraded"
                        degraded_count += 1
                    else:
                        next_status = "failed"
                        failed_count += 1
                    mem_meta["enrichment_status"] = next_status
                    mem_meta["last_enrichment_error"] = "deferred_enrichment_produced_no_structured_result"

                db_updates.append(
                    {
                        "id": mem_id,
                        "metadata": mem_meta,
                        "categories": mem_cats,
                        "enrichment_status": next_status,
                    }
                )

            self._insert_fact_vectors(
                fact_entries=fact_entries,
                warning_prefix="Fact embedding failed during enrichment",
            )
            self._db.update_enrichment_bulk(db_updates)
            batches_processed += 1

        remaining_count = len(
            self._db.get_pending_enrichment(user_id=user_id, limit=1)
        )

        return {
            "enriched_count": enriched_count,
            "failed_count": failed_count,
            "degraded_count": degraded_count,
            "batches": batches_processed,
            "remaining": remaining_count,
        }

    def extract_memories(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        prompt: Optional[str] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract structured memories from a conversation using LLM."""
        conversation = parse_messages(messages)
        existing = self._db.get_all_memories(
            user_id=metadata.get("user_id"),
            agent_id=metadata.get("agent_id"),
            run_id=metadata.get("run_id"),
            app_id=metadata.get("app_id"),
        )
        existing_text = "\n".join([m.get("memory", "") for m in existing])

        if prompt or self._config.custom_fact_extraction_prompt:
            extraction_prompt = prompt or self._config.custom_fact_extraction_prompt
        else:
            if self._should_use_agent_memory_extraction(messages, metadata):
                extraction_prompt = AGENT_MEMORY_EXTRACTION_PROMPT
            else:
                extraction_prompt = MEMORY_EXTRACTION_PROMPT
        prompt_text = extraction_prompt.format(conversation=conversation, existing_memories=existing_text)

        try:
            response = self._llm.generate(prompt_text)
            data = strip_code_fences(response)
            if not data:
                return []
            parsed = json.loads(data)
            memories = parsed.get("memories", [])
            extracted = [
                {
                    "content": m.get("content", ""),
                    "categories": [m.get("category")] if m.get("category") else [],
                    "metadata": {"importance": m.get("importance"), "confidence": m.get("confidence")},
                }
                for m in memories
                if isinstance(m, dict)
            ]
            if includes:
                extracted = [m for m in extracted if includes.lower() in m.get("content", "").lower()]
            if excludes:
                extracted = [m for m in extracted if excludes.lower() not in m.get("content", "").lower()]
            return extracted
        except Exception as exc:
            logger.warning("Memory extraction failed (LLM or JSON error): %s", exc)
            return []

    @staticmethod
    def _should_use_agent_memory_extraction(messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> bool:
        has_agent_id = metadata.get("agent_id") is not None
        has_assistant_messages = any(msg.get("role") == "assistant" for msg in messages)
        return has_agent_id and has_assistant_messages

    def classify_memory_type(self, metadata: Dict[str, Any], role: str) -> str:
        """Classify a memory as 'episodic' or 'semantic' (Gap 1).

        When enable_memory_types is False, everything stays 'semantic' (backward compat).
        """
        distillation_config = self._distillation_config
        if not distillation_config or not distillation_config.enable_memory_types:
            return distillation_config.default_memory_type if distillation_config else "semantic"

        explicit = metadata.get("memory_type")
        if explicit in ("episodic", "semantic", "task", "note", "procedural",
                       "project", "project_status", "project_tag",
                       "warroom", "warroom_message", "test_fixture",
                       "operational_event",
                       "tool_observation", "trajectory", "utterance",
                       "screen_observation", "profile", "goal", "constraint",
                       "preference", "decision", "style", "product",
                       "product_philosophy"):
            return explicit

        if metadata.get("is_distilled"):
            return "semantic"

        if role in ("user", "assistant"):
            return "episodic"

        if metadata.get("source_type") == "active_signal":
            return "semantic"

        return "semantic"

    def select_primary_text(self, content: str, echo_result: Optional[EchoResult]) -> str:
        """Select the best text for embedding given optional echo enrichment."""
        if not echo_result:
            return content

        if self._echo_config.use_echo_augmented_embedding:
            parts = [content[:1500]]
            if echo_result.question_form:
                parts.append(echo_result.question_form)
            if echo_result.keywords:
                parts.append("Keywords: " + ", ".join(echo_result.keywords[:10]))
            if echo_result.paraphrases:
                parts.append(echo_result.paraphrases[0])
            return "\n".join(parts)

        if self._echo_config.use_question_embedding and echo_result.question_form:
            return echo_result.question_form
        return content
