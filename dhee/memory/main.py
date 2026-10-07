from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, date, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from dhee.configs.base import MemoryConfig
from dhee.core.decay import calculate_decayed_strength, decay_since, should_forget, should_promote
from dhee.core.conflict import resolve_conflict
from dhee.core.distillation import ReplayDistiller
from dhee.core.echo import EchoProcessor, EchoDepth, EchoResult
from dhee.core.forgetting import HomeostaticNormalizer, InterferencePruner, RedundancyCollapser
from dhee.core.fusion import fuse_memories
from dhee.core.intent import QueryIntent, classify_intent
from dhee.core.retrieval import composite_score, tokenize, HybridSearcher
from dhee.core.traces import (
    boost_fast_trace,
    cascade_traces,
    compute_effective_strength,
    decay_traces,
    initialize_traces,
)
from dhee.core.category import CategoryProcessor, CategoryMatch
from dhee.core.graph import KnowledgeGraph
from dhee.core.scene import SceneProcessor
from dhee.core.profile import ProfileProcessor
from dhee.db.sqlite import SQLiteManager
from dhee.exceptions import FadeMemValidationError
from dhee.memory.base import MemoryBase
from dhee.memory.utils import (
    build_filters_and_metadata,
    matches_filters,
    normalize_categories,
    normalize_messages,
    parse_messages,
    strip_code_fences,
)
from dhee.memory.parallel import ParallelExecutor
from dhee.memory.smart import SmartMemory
from dhee.memory.cost import CostTracker, estimate_token_count, estimate_output_tokens, stable_hash_text
from dhee.memory.scoping import ScopeResolver, MemoryScope as _ScopeEnum, is_shareable_memory as _is_shareable
from dhee.memory.vectors import VectorOps, build_index_vectors, resolve_memory_id, collapse_vector_results
from dhee.memory.retrieval_helpers import (
    ECHO_STOP_WORDS,
    normalize_bitemporal_value,
    parse_bitemporal_datetime,
    attach_bitemporal_metadata,
    query_prefers_recency,
    query_is_transactional,
    compute_temporal_boost,
    calculate_echo_boost,
    truncate_rerank_text,
    term_overlap_count,
    build_rerank_snippet,
)
from dhee.memory.episodic import (
    index_episodic_events_for_memory as _index_episodic,
    search_episodes as _search_episodes,
    lookup_entity_aggregates as _lookup_aggregates,
)
from dhee.memory.scene_profile import SceneProfileMixin
from dhee.memory.orchestration import OrchestrationEngine
from dhee.memory.search_pipeline import SearchPipeline
from dhee.memory.write_pipeline import MemoryWritePipeline
from dhee.observability import metrics
from dhee.utils.factory import EmbedderFactory, LLMFactory, VectorStoreFactory
from dhee.utils.prompts import AGENT_MEMORY_EXTRACTION_PROMPT, MEMORY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

_SCENE_BENCHMARK_NOISE_RE = re.compile(
    r"\b(?:you are fixing a real bug|benchmark prompt|swe-bench|fail[- ]to[- ]pass)\b",
    re.IGNORECASE,
)
_SCENE_OPERATIONAL_VERB_RE = re.compile(
    r"^\s*(?:edited|wrote|created|updated|modified|touched)\s+(.{1,240})\s*$",
    re.IGNORECASE,
)
_SCENE_PATH_HINT_RE = re.compile(
    r"(?:^|\s)(?:/|~/|\./|[\w.-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|rb|php|sh|sql|toml|yaml|yml|json|md)\b)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Inline helpers (formerly in deleted core/acceptance and core/policy modules)
# ---------------------------------------------------------------------------

def _metadata_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        current = dict(value)
        for _ in range(4):
            raw = current.get("legacy_metadata_raw")
            if not isinstance(raw, str):
                break
            try:
                parsed = json.loads(raw)
            except Exception:
                break
            if not isinstance(parsed, dict):
                break
            overlay = {
                key: item
                for key, item in current.items()
                if key not in {"legacy_metadata_raw", "legacy_metadata_type"}
            }
            current = {**parsed, **overlay}
        return current
    if value not in (None, "", [], {}):
        return {
            "legacy_metadata_raw": value,
            "legacy_metadata_type": type(value).__name__,
        }
    return {}


@dataclass
class ExplicitIntent:
    action: Optional[str]
    content: str


_REMEMBER_PATTERNS = [
    r"^\s*(?:please\s+)?remember\b(?: that)?\s*[:,-]?\s*(.+)$",
    r"\b(?:don't|do not)\s+forget\b(?: to)?\s*[:,-]?\s*(.+)$",
    r"\bmake sure to remember\b(?: that)?\s*[:,-]?\s*(.+)$",
]

_FORGET_PATTERNS = [
    r"^\s*(?:forget|delete|remove|erase)\b(?: about| that)?\s*[:,-]?\s*(.+)$",
    r"^\s*(?:don't|do not)\s+remember\b(?: that)?\s*[:,-]?\s*(.+)$",
]


def detect_explicit_intent(text: str) -> ExplicitIntent:
    cleaned = text.strip()
    for pattern in _REMEMBER_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            return ExplicitIntent(action="remember", content=content or cleaned)
    for pattern in _FORGET_PATTERNS:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            return ExplicitIntent(action="forget", content=content or "")
    return ExplicitIntent(action=None, content=cleaned)


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+\w+(?:\s+\w+){0,4}\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct)\b",
    re.IGNORECASE,
)
_NAME_HINT_RE = re.compile(
    r"\b(?:my name is|call me|i am|i'm)\s+([A-Za-z][A-Za-z'\\-]+(?:\s+[A-Za-z][A-Za-z'\\-]+)?)\b",
    re.IGNORECASE,
)
_ID_HINT_RE = re.compile(
    r"\b(passport|driver'?s license|license number|id number|social security|ssn)\b",
    re.IGNORECASE,
)
_HEALTH_HINT_RE = re.compile(
    r"\b(diagnosed|diagnosis|medication|prescription|doctor|clinic|therapy|symptom|allergy|allergic|sick|illness|disease|mental health|depression|anxiety|adhd|diabetes|asthma|blood pressure|migraine)\b",
    re.IGNORECASE,
)
_FINANCE_HINT_RE = re.compile(
    r"\b(bank account|account number|routing number|iban|swift code|credit card|debit card|cvv|salary|income|mortgage|loan amount|tax id|tax return)\b",
    re.IGNORECASE,
)
_EPHEMERAL_HINT_RE = re.compile(
    r"\b(remind me|pick up|todo|to-do)\b"
    r"|\b(today|tomorrow|tonight)\s+(?:i\s|we\s|I\s)"
    r"|(?:this|next)\s+(?:morning|afternoon|evening|week)\b"
    r"|\bin\s+\d+\s*(?:minutes|hours|days)\b",
    re.IGNORECASE,
)
_PREFERENCE_HINT_RE = re.compile(
    r"\b(prefer|favorite|always|never|like to|love|hate|avoid|must|can't|cannot)\b",
    re.IGNORECASE,
)
_ROUTINE_HINT_RE = re.compile(
    r"\b(every day|every morning|every night|every week|weekly|monthly|on weekends|each week|every weekday)\b",
    re.IGNORECASE,
)
_GOAL_HINT_RE = re.compile(
    r"\b(my goal is|i want to|i plan to|i'm working on|i am working on|long[- ]term)\b",
    re.IGNORECASE,
)
_TEMPORAL_RECENT_QUERY_RE = re.compile(
    r"\b(latest|most recent|currently|current|as of|recent|newest|last)\b",
    re.IGNORECASE,
)
_TEMPORAL_RANGE_QUERY_RE = re.compile(
    r"\b(past|in the past|within the last|last month|last week|last year)\b",
    re.IGNORECASE,
)
_TEMPORAL_TRANSACTIONAL_QUERY_RE = re.compile(
    r"\b(spent|spend|cost|price|payment|paid|bought|purchase|transaction|grocery|amount|money|dollars?|usd)\b",
    re.IGNORECASE,
)


def detect_sensitive_categories(text: str) -> List[str]:
    reasons: List[str] = []
    if _EMAIL_RE.search(text):
        reasons.append("email")
    if _PHONE_RE.search(text):
        reasons.append("phone")
    if _SSN_RE.search(text):
        reasons.append("ssn")
    if _ADDRESS_RE.search(text):
        reasons.append("address")
    if _ID_HINT_RE.search(text):
        reasons.append("id")
    name_match = _NAME_HINT_RE.search(text)
    if name_match:
        candidate = name_match.group(1).strip()
        if candidate and candidate[0].isupper():
            reasons.append("name")
    if _HEALTH_HINT_RE.search(text):
        reasons.append("health")
    if _FINANCE_HINT_RE.search(text):
        reasons.append("finance")
    return sorted(set(reasons))


def is_ephemeral(text: str) -> bool:
    return _EPHEMERAL_HINT_RE.search(text) is not None


def looks_high_confidence(content: str, metadata: Optional[Dict[str, object]] = None) -> bool:
    metadata = metadata or {}
    # Content explicitly provided by user (infer=False) is always high confidence.
    if metadata.get("user_provided"):
        return True
    confidence = _coerce_float(metadata.get("confidence"))
    importance = _coerce_float(metadata.get("importance"))
    if confidence is not None and confidence >= 0.7:
        return True
    if importance is not None and importance >= 0.7:
        return True
    if metadata.get("confirmed") or metadata.get("user_confirmed"):
        return True
    if _PREFERENCE_HINT_RE.search(content):
        return True
    if _ROUTINE_HINT_RE.search(content):
        return True
    if _GOAL_HINT_RE.search(content):
        return True
    return False


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def feature_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------

SHAREABLE_CATEGORY_IDS = {
    "preferences",
    "procedures",
    "corrections",
}

SHAREABLE_CATEGORY_HINTS = (
    "preference",
    "workflow",
    "procedure",
    "coding",
    "code",
    "style",
    "tooling",
    "editor",
)

SCOPE_VALUES = {"agent", "connector", "category", "global"}
DEFAULT_SCOPE_WEIGHTS = {
    "agent": 1.0,
    "connector": 0.97,
    "category": 0.94,
    "global": 0.92,
}


class MemoryScope(str, Enum):
    AGENT = "agent"
    CONNECTOR = "connector"
    CATEGORY = "category"
    GLOBAL = "global"


class FullMemory(SmartMemory, SceneProfileMixin):
    """Full-featured Memory class with orchestration, cognition, and subsystems.

    Extends SmartMemory with:
    - Orchestrated search (map-reduce, episodic index, hierarchical anchors)
    - SceneProcessor / ProfileProcessor (via SceneProfileMixin)
    - Buddhi cognition engine
    - Cost guardrails (via CostTracker)
    - Scope/visibility control (via ScopeResolver)
    - VectorOps for multi-node indexing
    - Trajectory recording and skill mining

    All base features (echo, categories, graph) are inherited from SmartMemory.
    Extracted modules: scoping, cost, vectors, retrieval_helpers, episodic, scene_profile.
    """

    def __init__(self, config: Optional[MemoryConfig] = None, preset: Optional[str] = None):
        # Use default full() config if neither config nor preset provided
        if config is None and preset is None:
            config = MemoryConfig.full()
        # Initialize parent SmartMemory (handles db, llm, embedder, etc.)
        super().__init__(config=config, preset=preset)
        self._runtime_root_dir = self._resolve_runtime_root_dir()
        self._buddhi_data_dir = os.path.join(self._runtime_root_dir, "buddhi")
        # Only FullMemory-specific lazy init
        self._scene_processor: Optional[SceneProcessor] = None
        self._profile_processor: Optional[ProfileProcessor] = None
        self._task_manager: Optional[Any] = None
        self._project_manager: Optional[Any] = None
        # Neural reranker (lazy init)
        self._reranker: Optional[Any] = None
        # Trajectory recording and skill mining
        self._trajectory_store: Optional[Any] = None
        self._skill_miner: Optional[Any] = None
        self._active_recorders: Dict[str, Any] = {}
        # Orchestration engine (lazy — created on first use)
        self.__orchestration_engine: Optional[OrchestrationEngine] = None
        # Parallel executor (lazy: created only when config enables it)
        self._executor: Optional[ParallelExecutor] = None
        if self.config.parallel.enable_parallel:
            self._executor = ParallelExecutor(max_workers=self.config.parallel.max_workers)
        # Dhee: Universal Engram extraction + context-first resolver + cognition
        self._engram_extractor: Optional[Any] = None
        self._context_resolver: Optional[Any] = None
        self._cognition_engine: Optional[Any] = None
        # Dhee: Self-evolution layer (samskara + viveka + alaya + nididhyasana)
        self._evolution_layer: Optional[Any] = None
        # Dhee: Buddhi — proactive cognition (HyperAgent layer)
        self._buddhi_layer: Optional[Any] = None
        # Scope resolver (delegates to extracted scoping module)
        self._scope_resolver = ScopeResolver(self.scope_config)
        # Write pipeline (lazy — created on first use)
        self.__write_pipeline: Optional[MemoryWritePipeline] = None
        # Search pipeline (lazy — created on first use)
        self.__search_pipeline: Optional[SearchPipeline] = None

    def _resolve_runtime_root_dir(self) -> str:
        """Pick a stable runtime root for cognition sidecars.

        FullMemory may be backed by a temporary or custom data dir even when the
        vector store itself is in-memory. Use the configured on-disk paths first
        so helper layers do not silently spill into ~/.dhee.
        """
        candidate_paths: List[object] = [getattr(self.config, "history_db_path", None)]
        vector_config = getattr(getattr(self.config, "vector_store", None), "config", {})
        if isinstance(vector_config, dict):
            candidate_paths.append(vector_config.get("path"))

        for raw_path in candidate_paths:
            if raw_path is None:
                continue
            path = str(raw_path).strip()
            if not path or path == ":memory:":
                continue
            root = os.path.dirname(os.path.abspath(os.path.expanduser(path)))
            if root:
                os.makedirs(root, exist_ok=True)
                return root

        fallback = os.path.join(os.path.expanduser("~"), ".dhee")
        os.makedirs(fallback, exist_ok=True)
        return fallback

    @property
    def _write_pipeline(self) -> MemoryWritePipeline:
        """Lazy-initialized write pipeline that delegates heavy write-path logic."""
        if self.__write_pipeline is None:
            self.__write_pipeline = MemoryWritePipeline(
                db=self.db,
                embedder=self.embedder,
                llm=self.llm,
                config=self.config,
                vector_store=self.vector_store,
                echo_processor_fn=lambda: self.echo_processor,
                category_processor_fn=lambda: self.category_processor,
                graph_fn=lambda: self.knowledge_graph,
                scene_processor_fn=lambda: self.scene_processor,
                profile_processor_fn=lambda: self.profile_processor,
                unified_enrichment_fn=lambda: self.unified_enrichment,
                engram_extractor_fn=lambda: self.engram_extractor,
                context_resolver_fn=lambda: self.context_resolver,
                evolution_layer_fn=lambda: self.evolution_layer,
                buddhi_layer_fn=lambda: self.buddhi_layer,
                scope_resolver=self._scope_resolver,
                executor=self._executor,
                record_cost_fn=self._record_cost_counter,
                forget_by_query_fn=self._forget_by_query,
                demote_existing_fn=self._demote_existing,
                nearest_memory_fn=self._nearest_memory,
                assign_to_scene_fn=self._assign_to_scene,
                update_profiles_fn=self._update_profiles,
                store_prospective_scenes_fn=self._store_prospective_scenes,
                persist_categories_fn=self._persist_categories,
            )
        return self.__write_pipeline

    @property
    def _search_pipeline(self) -> SearchPipeline:
        """Lazy-initialized search pipeline that delegates the full search path."""
        if self.__search_pipeline is None:
            self.__search_pipeline = SearchPipeline(
                db=self.db,
                embedder=self.embedder,
                config=self.config,
                vector_store=self.vector_store,
                echo_processor_fn=lambda: self.echo_processor,
                category_processor_fn=lambda: self.category_processor,
                reranker_fn=lambda: self.reranker,
                scope_resolver=self._scope_resolver,
                context_resolver_fn=lambda: self.context_resolver,
                evolution_layer_fn=lambda: self.evolution_layer,
                buddhi_layer_fn=lambda: self.buddhi_layer,
                knowledge_graph_fn=lambda: self.knowledge_graph,
                executor=self._executor,
                record_cost_fn=self._record_cost_counter,
                check_promotion_fn=self._check_promotion,
                persist_categories_fn=self._persist_categories,
                is_expired_fn=self._is_expired,
                update_vectors_for_memory_fn=self._update_vectors_for_memory,
            )
        return self.__search_pipeline

    @property
    def _orchestration_engine(self) -> OrchestrationEngine:
        """Lazy-initialized orchestration engine for search_orchestrated."""
        if self.__orchestration_engine is None:
            self.__orchestration_engine = OrchestrationEngine(
                config=self.config,
                db=self.db,
                search_fn=self.search,
                search_episodes_fn=self.search_episodes,
                lookup_aggregates_fn=self.lookup_entity_aggregates,
                intent_coverage_threshold_fn=self._intent_coverage_threshold,
                record_cost_fn=self._record_cost_counter,
                scene_processor_fn=lambda: self.scene_processor,
                profile_processor_fn=lambda: self.profile_processor,
                evolution_layer_fn=lambda: self.evolution_layer,
                llm_fn=lambda: self.llm,
            )
        return self.__orchestration_engine

    @property
    def scene_processor(self) -> Optional[SceneProcessor]:
        """Lazy-initialized SceneProcessor (only if scenes enabled in config)."""
        if self._scene_processor is None and self.config.scene.enable_scenes:
            self._scene_processor = SceneProcessor(
                db=self.db,
                embedder=self.embedder,
                llm=self.llm,
                config={
                    "scene_time_gap_minutes": self.config.scene.scene_time_gap_minutes,
                    "scene_topic_threshold": self.config.scene.scene_topic_threshold,
                    "auto_close_inactive_minutes": self.config.scene.auto_close_inactive_minutes,
                    "max_scene_memories": self.config.scene.max_scene_memories,
                    "use_llm_summarization": self.config.scene.use_llm_summarization,
                    "summary_regenerate_threshold": self.config.scene.summary_regenerate_threshold,
                },
            )
        return self._scene_processor

    @property
    def profile_processor(self) -> Optional[ProfileProcessor]:
        """Lazy-initialized ProfileProcessor (only if profiles enabled in config)."""
        if self._profile_processor is None and self.config.profile.enable_profiles:
            self._profile_processor = ProfileProcessor(
                db=self.db,
                embedder=self.embedder,
                llm=self.llm,
                config={
                    "auto_detect_profiles": self.config.profile.auto_detect_profiles,
                    "use_llm_extraction": self.config.profile.use_llm_extraction,
                    "narrative_regenerate_threshold": self.config.profile.narrative_regenerate_threshold,
                    "self_profile_auto_create": self.config.profile.self_profile_auto_create,
                    "max_facts_per_profile": self.config.profile.max_facts_per_profile,
                },
            )
        return self._profile_processor

    @property
    def engram_extractor(self):
        """Lazy-initialized EngramExtractor for structured memory extraction."""
        if self._engram_extractor is None and self.config.engram_extraction.enable_extraction:
            from dhee.core.engram_extractor import EngramExtractor
            llm = None
            if self.config.engram_extraction.use_llm_extraction:
                # Create a SEPARATE LLM instance for extraction with shorter timeout.
                # The main LLM (120s timeout, 3 retries) blocks too long when rate-limited.
                # Extraction must fail fast and fall back to rule-based extraction.
                from dhee.utils.factory import LLMFactory
                extraction_llm_config = dict(self.config.llm.config)
                extraction_llm_config["timeout"] = 30          # 30s vs 120s default
                extraction_llm_config["app_retries"] = 1       # 1 retry vs 3 default
                extraction_llm_config["max_retries"] = 0       # no OpenAI client retries
                extraction_llm_config["max_tokens"] = 2048     # extraction output is smaller
                try:
                    llm = LLMFactory.create(self.config.llm.provider, extraction_llm_config)
                    logger.info("EngramExtractor: created dedicated LLM (timeout=30s, retries=1)")
                except Exception as e:
                    logger.warning("EngramExtractor: failed to create dedicated LLM, using main: %s", e)
                    llm = self.llm
            self._engram_extractor = EngramExtractor(llm=llm)
        return self._engram_extractor

    @property
    def context_resolver(self):
        """Lazy-initialized ContextResolver for deterministic fact resolution."""
        if self._context_resolver is None:
            from dhee.core.resolvers import ContextResolver
            self._context_resolver = ContextResolver(db=self.db)
        return self._context_resolver

    @property
    def cognition_engine(self):
        """Lazy-initialized CognitionEngine for memory-grounded reasoning."""
        if self._cognition_engine is None and self.config.cognition.enable_cognition:
            from dhee.core.cognition import CognitionEngine
            self._cognition_engine = CognitionEngine(
                memory=self,
                external_llm=self.llm,
                max_depth=self.config.cognition.max_depth,
                max_sub_questions=self.config.cognition.max_sub_questions,
                store_solutions=self.config.cognition.store_solutions,
            )
        return self._cognition_engine

    @property
    def evolution_layer(self):
        """Lazy-initialized self-evolution layer (samskara + viveka + alaya)."""
        if self._evolution_layer is None:
            try:
                from dhee.core.evolution import EvolutionLayer
                self._evolution_layer = EvolutionLayer(
                    data_dir=self._runtime_root_dir,
                )
                # M4.2: wire the engram DB so answer acceptance can stamp
                # last_verified_at and bump tiers on downstream success.
                try:
                    self._evolution_layer.attach_substrate(self.db)
                except Exception as exc:
                    logger.debug("attach_substrate skipped: %s", exc)
            except Exception as e:
                logger.debug("Evolution layer init skipped: %s", e)
        return self._evolution_layer

    @property
    def buddhi_layer(self):
        """Lazy-initialized Buddhi — proactive cognition layer (HyperAgent)."""
        if self._buddhi_layer is None:
            try:
                from dhee.core.buddhi import Buddhi
                self._buddhi_layer = Buddhi(
                    data_dir=self._buddhi_data_dir,
                )
            except Exception as e:
                logger.debug("Buddhi layer init skipped: %s", e)
        return self._buddhi_layer

    @property
    def trajectory_store(self):
        """Lazy-initialized TrajectoryStore for persisting agent trajectories."""
        if self._trajectory_store is None:
            from dhee.skills.trajectory import TrajectoryStore
            self._trajectory_store = TrajectoryStore(
                db=self.db,
                embedder=self.embedder,
                vector_store=self.vector_store,
            )
        return self._trajectory_store

    @property
    def skill_miner(self):
        """Lazy-initialized SkillMiner for extracting skills from trajectories."""
        skill_cfg = getattr(self.config, "skill", None)
        if self._skill_miner is None and skill_cfg and skill_cfg.enable_mining:
            from dhee.skills.miner import SkillMiner
            self._skill_miner = SkillMiner(
                trajectory_store=self.trajectory_store,
                skill_store=self.skill_store,
                llm=self.llm,
                embedder=self.embedder,
                mutation_rate=skill_cfg.mutation_rate,
            )
        return self._skill_miner

    @property
    def reranker(self):
        """Lazy-initialized neural reranker (only if enabled in config)."""
        rerank_cfg = getattr(self.config, "rerank", None)
        if self._reranker is None and rerank_cfg and rerank_cfg.enable_rerank:
            from dhee.memory.reranker import create_reranker
            self._reranker = create_reranker({
                "provider": rerank_cfg.provider,
                "model": rerank_cfg.model,
                "api_key_env": rerank_cfg.api_key_env,
                **rerank_cfg.config,
            })
        return self._reranker

    def start_trajectory(
        self,
        task_description: str,
        user_id: str = "default",
        agent_id: str = "default",
    ) -> str:
        """Start recording a new trajectory for the given task.

        Returns the recorder ID to be used with record_trajectory_step()
        and complete_trajectory().
        """
        from dhee.skills.trajectory import TrajectoryRecorder
        recorder = TrajectoryRecorder(
            task_description=task_description,
            user_id=user_id,
            agent_id=agent_id,
        )
        self._active_recorders[recorder.id] = recorder
        return recorder.id

    def record_trajectory_step(
        self,
        recorder_id: str,
        action: str,
        tool: str = "",
        args: Optional[Dict[str, Any]] = None,
        result_summary: str = "",
        error: Optional[str] = None,
        slot_values: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Record a step in an active trajectory.

        If slot_values are provided, they are stored in state_snapshot
        for later structural mining.
        """
        recorder = self._active_recorders.get(recorder_id)
        if recorder is None:
            return {"error": f"No active recorder: {recorder_id}"}

        state_snapshot = None
        if slot_values:
            state_snapshot = {"slot_values": slot_values}

        step = recorder.record_step(
            action=action,
            tool=tool,
            args=args,
            result_summary=result_summary,
            error=error,
            state_snapshot=state_snapshot,
        )
        return {
            "recorder_id": recorder_id,
            "step_count": len(recorder.steps),
            "action": action,
            "tool": tool,
        }

    def complete_trajectory(
        self,
        recorder_id: str,
        success: bool,
        outcome_summary: str = "",
    ) -> Dict[str, Any]:
        """Finalize a trajectory recording and persist it.

        Returns the trajectory data.
        """
        recorder = self._active_recorders.pop(recorder_id, None)
        if recorder is None:
            return {"error": f"No active recorder: {recorder_id}"}

        trajectory = recorder.finalize(
            success=success,
            outcome_summary=outcome_summary,
        )
        self.trajectory_store.save(trajectory)

        return {
            "trajectory_id": trajectory.id,
            "task_description": trajectory.task_description,
            "step_count": len(trajectory.steps),
            "success": success,
            "outcome_summary": outcome_summary,
            "trajectory_hash": trajectory.trajectory_hash_val,
        }

    def mine_skills(
        self,
        task_query: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a skill mining cycle.

        Analyzes successful trajectories and extracts reusable skills.
        Returns info about mined skills.
        """
        if self.skill_miner is None:
            return {"error": "Skill mining not enabled", "skills_mined": 0}

        mined = self.skill_miner.mine(
            task_query=task_query,
            user_id=user_id,
        )
        return {
            "skills_mined": len(mined),
            "skills": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "confidence": s.confidence,
                    "source": s.source,
                    "tags": s.tags,
                }
                for s in mined
            ],
        }

    def get_skill_stats(self) -> Dict[str, Any]:
        """Get statistics about skills and trajectories."""
        skills = self.skill_store.list_all() if self.skill_store else []
        trajectories = self.trajectory_store.find_successful(limit=1000) if self._trajectory_store else []

        total_skills = len(skills)
        authored = sum(1 for s in skills if s.source == "authored")
        mined = sum(1 for s in skills if s.source == "mined")
        imported = sum(1 for s in skills if s.source == "imported")
        avg_confidence = sum(s.confidence for s in skills) / max(1, total_skills)

        return {
            "total_skills": total_skills,
            "authored_skills": authored,
            "mined_skills": mined,
            "imported_skills": imported,
            "avg_confidence": round(avg_confidence, 4),
            "total_successful_trajectories": len(trajectories),
            "active_recorders": len(self._active_recorders),
        }

    def close(self) -> None:
        """Release all resources held by the Memory instance."""
        errors = []

        # Flush self-evolution state before shutdown
        if self._evolution_layer is not None:
            try:
                self._evolution_layer.flush()
            except Exception as exc:
                logger.exception("FullMemory close failed for evolution.flush")
                errors.append(
                    f"evolution.flush: {type(exc).__name__}: {exc}"
                )

        # Shutdown parallel executor if it was created
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as exc:
                logger.exception("FullMemory close failed for executor.shutdown")
                errors.append(
                    f"executor.shutdown: {type(exc).__name__}: {exc}"
                )
            finally:
                self._executor = None

        # Release vector store
        if self.vector_store is not None:
            try:
                self.vector_store.close()
            except Exception as exc:
                logger.exception("FullMemory close failed for vector_store.close")
                errors.append(
                    f"vector_store.close: {type(exc).__name__}: {exc}"
                )
            finally:
                self.vector_store = None

        # Release database
        if self.db is not None:
            try:
                self.db.close()
            except Exception as exc:
                logger.exception("FullMemory close failed for db.close")
                errors.append(f"db.close: {type(exc).__name__}: {exc}")
            finally:
                self.db = None

        if errors:
            raise RuntimeError(
                "Failed to close FullMemory resources: " + "; ".join(errors)
            )

    def __repr__(self) -> str:
        return f"FullMemory(db={self.db!r}, echo={self.config.echo.enable_echo}, scenes={self.config.scene.enable_scenes})"

    # _cached_embed inherited from SmartMemory

    # from_config inherited from SmartMemory

    def _record_cost_counter(
        self,
        *,
        phase: str,
        user_id: Optional[str],
        llm_calls: float = 0.0,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        embed_calls: float = 0.0,
    ) -> None:
        cost_cfg = getattr(self.config, "cost_guardrail", None)
        if not cost_cfg or not cost_cfg.enable_cost_counters:
            return
        try:
            self.db.record_cost_counter(
                phase=phase,
                user_id=user_id,
                llm_calls=llm_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                embed_calls=embed_calls,
            )
            if str(phase) == "write":
                self._orchestration_engine._enforce_write_cost_guardrail(user_id=user_id)
        except Exception as e:
            logger.debug("Cost counter record failed: %s", e)

    _estimate_token_count = staticmethod(estimate_token_count)
    _estimate_output_tokens = staticmethod(estimate_output_tokens)

    def _intent_coverage_threshold(self, intent_value: str, fallback: float) -> float:
        orch_cfg = getattr(self.config, "orchestration", None)
        thresholds = getattr(orch_cfg, "intent_coverage_thresholds", {}) or {}
        key = str(intent_value or "freeform").strip().lower()
        value = thresholds.get(key, fallback)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return max(0.0, min(1.0, float(fallback)))

    _stable_hash_text = staticmethod(stable_hash_text)

    def _index_episodic_events_for_memory(self, *, memory_id, user_id, content, metadata):
        return _index_episodic(
            db=self.db, config=self.config,
            memory_id=memory_id, user_id=user_id, content=content, metadata=metadata,
        )

    def search_episodes(self, *, query, user_id, intent=None, actor_id=None,
                        time_anchor=None, entity_hints=None, min_coverage=None, limit=80):
        return _search_episodes(
            db=self.db, config=self.config, query=query, user_id=user_id,
            intent=intent, actor_id=actor_id, time_anchor=time_anchor,
            entity_hints=entity_hints, min_coverage=min_coverage, limit=limit,
            intent_coverage_threshold_fn=self._intent_coverage_threshold,
        )

    def lookup_entity_aggregates(self, query, user_id, intent=None):
        return _lookup_aggregates(db=self.db, query=query, user_id=user_id, intent=intent)

    def search_orchestrated(
        self,
        *,
        query: str,
        user_id: str,
        question_type: str = "",
        question_date: str = "",
        repo: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        limit: int = 10,
        orchestration_mode: str = "hybrid",
        base_search_limit: Optional[int] = None,
        base_context_limit: int = 10,
        search_cap: Optional[int] = None,
        context_cap: Optional[int] = None,
        map_max_candidates: Optional[int] = None,
        map_max_chars: Optional[int] = None,
        keyword_search: bool = True,
        hybrid_alpha: float = 0.7,
        include_evidence: bool = True,
        evidence_strategy: str = "full",
        evidence_max_chars: int = 3500,
        evidence_context_lines: int = 1,
        max_context_chars: int = 28000,
        rerank: bool = True,
        orchestrator_llm: Optional[Any] = None,
        reflection_max_hops: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._orchestration_engine.search_orchestrated(
            query=query,
            user_id=user_id,
            question_type=question_type,
            question_date=question_date,
            repo=repo,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            filters=filters,
            categories=categories,
            limit=limit,
            orchestration_mode=orchestration_mode,
            base_search_limit=base_search_limit,
            base_context_limit=base_context_limit,
            search_cap=search_cap,
            context_cap=context_cap,
            map_max_candidates=map_max_candidates,
            map_max_chars=map_max_chars,
            keyword_search=keyword_search,
            hybrid_alpha=hybrid_alpha,
            include_evidence=include_evidence,
            evidence_strategy=evidence_strategy,
            evidence_max_chars=evidence_max_chars,
            evidence_context_lines=evidence_context_lines,
            max_context_chars=max_context_chars,
            rerank=rerank,
            orchestrator_llm=orchestrator_llm,
            reflection_max_hops=reflection_max_hops,
        )

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        filters: Dict[str, Any] = None,
        categories: List[str] = None,
        immutable: bool = False,
        expiration_date: Optional[str] = None,
        infer: bool = True,
        prompt: Optional[str] = None,
        includes: Optional[str] = None,
        excludes: Optional[str] = None,
        initial_layer: str = "auto",
        initial_strength: float = 1.0,
        echo_depth: Optional[str] = None,  # EchoMem: override echo depth (shallow/medium/deep)
        agent_category: Optional[str] = None,
        connector_id: Optional[str] = None,
        scope: Optional[str] = None,
        source_app: Optional[str] = None,
        memory_id: Optional[str] = None,
        context_messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        processed_metadata, effective_filters = build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_metadata=metadata,
            input_filters=filters,
        )

        messages_list = normalize_messages(messages)

        if infer:
            memories_to_add = self._extract_memories(
                messages_list,
                processed_metadata,
                prompt=prompt,
                includes=includes,
                excludes=excludes,
            )
        else:
            memories_to_add = []
            for msg in messages_list:
                role = msg.get("role")
                if role == "system":
                    continue
                content = msg.get("content")
                if not content:
                    continue
                mem_meta = dict(processed_metadata)
                mem_meta["role"] = role
                # When infer=False, the caller explicitly provides content to store.
                # Treat as high confidence to avoid the low_confidence strength cap.
                mem_meta["user_provided"] = True
                if msg.get("name"):
                    mem_meta["actor_id"] = msg.get("name")
                memories_to_add.append({"content": content, "metadata": mem_meta})

        results: List[Dict[str, Any]] = []
        for mem in memories_to_add:
            result = self._process_single_memory(
                mem=mem,
                processed_metadata=processed_metadata,
                effective_filters=effective_filters,
                categories=categories,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                app_id=app_id,
                agent_category=agent_category,
                connector_id=connector_id,
                scope=scope,
                source_app=source_app,
                immutable=immutable,
                expiration_date=expiration_date,
                initial_layer=initial_layer,
                initial_strength=initial_strength,
                echo_depth=echo_depth,
                memory_id=memory_id,
                context_messages=context_messages,
            )
            if result is not None:
                results.append(result)

        # Persist categories after batch
        if self.category_processor:
            self._persist_categories()

        return {"results": results}

    def add_batch(
        self,
        items: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        filters: Dict[str, Any] = None,
        initial_strength: float = 1.0,
        echo_depth: Optional[str] = None,
        **common_kwargs: Any,
    ) -> Dict[str, Any]:
        """Add multiple memories in a batch, minimizing LLM/embedding/DB calls.

        Each item in *items* is a dict with at least a ``content`` key (or
        ``messages``). Items may also carry per-item ``user_id``, ``metadata``,
        ``categories``, etc.

        Batch optimization is only used when ``config.batch.enable_batch`` is
        True. Otherwise this is equivalent to calling ``add()`` in a loop.

        Returns ``{"results": [...]}``.
        """
        batch_config = getattr(self.config, "batch", None)
        use_batch = batch_config and batch_config.enable_batch

        if not use_batch or not items:
            all_results = []
            for item in items:
                content = item.get("content") or item.get("messages", "")
                item_meta = dict(metadata or {})
                item_meta.update(item.get("metadata") or {})
                result = self.add(
                    messages=content,
                    user_id=item.get("user_id") or user_id,
                    agent_id=item.get("agent_id") or agent_id,
                    run_id=item.get("run_id") or run_id,
                    app_id=item.get("app_id") or app_id,
                    metadata=item_meta,
                    filters=filters,
                    categories=item.get("categories"),
                    initial_strength=initial_strength,
                    echo_depth=echo_depth,
                    infer=False,
                    **common_kwargs,
                )
                all_results.extend(result.get("results", []))
            return {"results": all_results}

        max_batch = batch_config.max_batch_size
        all_results: List[Dict[str, Any]] = []
        for start in range(0, len(items), max_batch):
            chunk = items[start:start + max_batch]
            all_results.extend(
                self._process_memory_batch(
                    chunk,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    app_id=app_id,
                    metadata=metadata,
                    filters=filters,
                    initial_strength=initial_strength,
                    echo_depth=echo_depth,
                    batch_config=batch_config,
                    **common_kwargs,
                )
            )

        if self.category_processor:
            self._persist_categories()

        return {"results": all_results}

    def _process_memory_batch(
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
        return self._write_pipeline.process_memory_batch(
            items,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            metadata=metadata,
            filters=filters,
            initial_strength=initial_strength,
            echo_depth=echo_depth,
            batch_config=batch_config,
            **common_kwargs,
        )

    def _resolve_memory_metadata(self, **kwargs) -> tuple:
        return self._write_pipeline.resolve_memory_metadata(**kwargs)

    def _encode_memory(self, content, echo_depth, mem_categories, mem_metadata, initial_strength):
        return self._write_pipeline.encode_memory(content, echo_depth, mem_categories, mem_metadata, initial_strength)

    def _process_single_memory(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self._write_pipeline.process_single_memory(**kwargs)

    def _process_single_memory_lite(self, **kwargs) -> Optional[Dict[str, Any]]:
        return self._write_pipeline.process_single_memory_lite(**kwargs)

    def enrich_pending(
        self,
        user_id: str = "default",
        batch_size: int = 10,
        max_batches: int = 5,
    ) -> Dict[str, Any]:
        """Batch-enrich memories that were stored with deferred enrichment."""
        result = self._write_pipeline.enrich_pending(
            user_id=user_id,
            batch_size=batch_size,
            max_batches=max_batches,
        )
        if self.scene_processor:
            result["scene_summaries"] = self.scene_processor.summarize_unsummarized(
                user_id=user_id,
                limit=max(1, min(int(batch_size) * int(max_batches), 500)),
            )
        return result

    def reextract(
        self,
        user_id: str = "default",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Requeue complete memories that have no linked engram facts."""
        if not hasattr(self.db, "requeue_complete_without_engram_facts"):
            return {"requeued_count": 0, "reason": "db_requeue_unavailable"}
        return self.db.requeue_complete_without_engram_facts(
            user_id=user_id,
            limit=limit,
        )

    _normalize_bitemporal_value = staticmethod(normalize_bitemporal_value)
    _parse_bitemporal_datetime = classmethod(lambda cls, v: parse_bitemporal_datetime(v))
    _attach_bitemporal_metadata = classmethod(
        lambda cls, metadata, observed_time: attach_bitemporal_metadata(metadata, observed_time)
    )
    _query_prefers_recency = staticmethod(query_prefers_recency)
    _query_is_transactional = staticmethod(query_is_transactional)

    def _compute_temporal_boost(self, *, query, metadata, query_intent=None):
        return compute_temporal_boost(query=query, metadata=metadata, query_intent=query_intent)

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        filters: Dict[str, Any] = None,
        categories: List[str] = None,
        agent_category: Optional[str] = None,
        connector_ids: Optional[List[str]] = None,
        scope_filter: Optional[Union[str, List[str]]] = None,
        limit: int = 100,
        rerank: bool = True,
        keyword_search: bool = False,
        hybrid_alpha: float = 0.7,
        min_strength: float = 0.1,
        boost_on_access: bool = True,
        use_echo_rerank: bool = True,
        use_category_boost: bool = True,
        include_evidence: bool = False,
        evidence_strategy: str = "vector_or_snippet",
        evidence_max_chars: int = 900,
        evidence_context_lines: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._search_pipeline.search(
            query=query, user_id=user_id, agent_id=agent_id, run_id=run_id,
            app_id=app_id, filters=filters, categories=categories,
            agent_category=agent_category, connector_ids=connector_ids,
            scope_filter=scope_filter, limit=limit, rerank=rerank,
            keyword_search=keyword_search, hybrid_alpha=hybrid_alpha,
            min_strength=min_strength, boost_on_access=boost_on_access,
            use_echo_rerank=use_echo_rerank, use_category_boost=use_category_boost,
            include_evidence=include_evidence, evidence_strategy=evidence_strategy,
            evidence_max_chars=evidence_max_chars, evidence_context_lines=evidence_context_lines,
            **kwargs,
        )

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        memory = self.db.get_memory(memory_id)
        if memory:
            self.db.increment_access(memory_id)
        return memory

    # Hard cap to prevent unbounded result sets even if callers pass a huge limit.
    _GET_ALL_MAX_LIMIT = 10_000

    def get_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        filters: Dict[str, Any] = None,
        categories: List[str] = None,
        limit: int = 100,
        layer: Optional[str] = None,
        min_strength: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Clamp limit to a sensible maximum to avoid unbounded result sets.
        limit = max(1, min(limit, self._GET_ALL_MAX_LIMIT))

        _, effective_filters = build_filters_and_metadata(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            input_filters=filters,
        )
        if app_id:
            effective_filters["app_id"] = app_id

        # When metadata filters are present, fetch extra rows to account
        # for post-hoc filtering (same pattern as TaskManager uses limit*3).
        fetch_limit = limit
        if filters:
            fetch_limit = max(limit * 5, 200)

        memories = self.db.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            app_id=app_id,
            layer=layer,
            min_strength=min_strength,
            limit=fetch_limit,
        )

        if categories:
            memories = [m for m in memories if any(c in m.get("categories", []) for c in categories)]

        if filters:
            memories = [m for m in memories if matches_filters({**m, **m.get("metadata", {})}, filters)]

        memories = [m for m in memories if not self._is_expired(m)]
        return {"results": memories[:limit]}

    def update(self, memory_id: str, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        memory = self.db.get_memory(memory_id)
        if not memory:
            missing_memory = data.get("content") if isinstance(data, dict) else data
            return {"id": memory_id, "memory": missing_memory, "event": "ERROR"}

        content: Optional[str]
        metadata_updates: Optional[Dict[str, Any]] = None
        categories_updates: Optional[List[str]] = None

        if isinstance(data, dict):
            content = data.get("content") or data.get("memory")
            metadata_updates = data.get("metadata")
            if "categories" in data:
                categories_updates = normalize_categories(data.get("categories"))
        else:
            content = data

        if content is None and metadata_updates is None and categories_updates is None:
            return {"id": memory_id, "memory": memory.get("memory", ""), "event": "ERROR"}

        metadata = _metadata_dict(memory.get("metadata"))
        categories = list(memory.get("categories", []) or [])
        existing_content = memory.get("memory", "")
        echo_result = None

        content_changed = content is not None and content != existing_content
        if content is None:
            content = existing_content

        if content_changed and self.echo_processor and self.echo_config.enable_echo:
            depth_override = None
            current_depth = metadata.get("echo_depth")
            if current_depth:
                try:
                    depth_override = EchoDepth(current_depth)
                except ValueError:
                    depth_override = None
            echo_result = self.echo_processor.process(content, depth=depth_override)
            metadata.update(echo_result.to_metadata())
            if not categories and echo_result.category:
                categories = [echo_result.category]

        if metadata_updates:
            metadata.update(metadata_updates)
        if categories_updates is not None:
            categories = categories_updates

        if content_changed:
            primary_text = self._select_primary_text(content, echo_result)
            new_embedding = self.embedder.embed(primary_text, memory_action="update")
            success = self.db.update_memory(
                memory_id,
                {"memory": content, "embedding": new_embedding, "metadata": metadata, "categories": categories},
            )
            if success:
                self._delete_vectors_for_memory(memory_id)
                vectors, payloads, vector_ids = self._build_index_vectors(
                    memory_id=memory_id,
                    content=content,
                    primary_text=primary_text,
                    embedding=new_embedding,
                    echo_result=echo_result,
                    metadata=metadata,
                    categories=categories,
                    user_id=memory.get("user_id"),
                    agent_id=memory.get("agent_id"),
                    run_id=memory.get("run_id"),
                    app_id=memory.get("app_id"),
                )
                try:
                    self.vector_store.insert(vectors=vectors, payloads=payloads, ids=vector_ids)
                except Exception as e:
                    logger.error(
                        "Vector re-insert failed during update for memory %s: %s. "
                        "DB was updated but vector index is stale — will be rebuilt on next update.",
                        memory_id, e,
                    )
        else:
            success = self.db.update_memory(
                memory_id,
                {"metadata": metadata, "categories": categories},
            )
            if success:
                payload_updates = dict(metadata)
                payload_updates["categories"] = categories
                try:
                    self._update_vectors_for_memory(memory_id, payload_updates)
                except Exception as e:
                    logger.error(
                        "Vector payload update failed for memory %s: %s. "
                        "DB is authoritative — vector metadata may be stale.",
                        memory_id, e,
                    )

        if success:
            self._index_episodic_events_for_memory(
                memory_id=memory_id,
                user_id=memory.get("user_id"),
                content=content,
                metadata=metadata,
            )
            self._record_cost_counter(
                phase="write",
                user_id=memory.get("user_id"),
                llm_calls=0.0,
                input_tokens=0.0,
                output_tokens=0.0,
                embed_calls=1.0 if content_changed else 0.0,
            )

        return {"id": memory_id, "memory": content, "event": "UPDATE" if success else "ERROR"}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        logger.info("Deleting memory %s (tombstone=%s)", memory_id, self.fade_config.use_tombstone_deletion)
        memory = self.db.get_memory(memory_id)
        self.db.delete_memory(memory_id, use_tombstone=self.fade_config.use_tombstone_deletion)
        self._delete_vectors_for_memory(memory_id)
        self._record_cost_counter(
            phase="write",
            user_id=(memory or {}).get("user_id"),
            llm_calls=0.0,
            input_tokens=0.0,
            output_tokens=0.0,
            embed_calls=0.0,
        )
        return {"id": memory_id, "deleted": True}

    def delete_all(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        filters: Dict[str, Any] = None,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not any([user_id, agent_id, run_id, app_id, filters]):
            raise FadeMemValidationError(
                "At least one filter is required to delete all memories. Use reset() to clear everything.",
                error_code="VALIDATION_004",
            )
        memories = self.db.get_all_memories(user_id=user_id, agent_id=agent_id, run_id=run_id, app_id=app_id)
        if filters:
            memories = [m for m in memories if matches_filters({**m, **m.get("metadata", {})}, filters)]

        if dry_run:
            return {"deleted_count": 0, "would_delete": len(memories), "dry_run": True}

        logger.warning(
            "delete_all: deleting %d memories (user_id=%s, agent_id=%s, filters=%s)",
            len(memories), user_id, agent_id, filters,
        )
        count = 0
        for memory in memories:
            self.delete(memory["id"])
            count += 1

        # Clean up entity aggregates for benchmark isolation.
        if hasattr(self.db, "delete_entity_aggregates_for_user") and user_id:
            self.db.delete_entity_aggregates_for_user(user_id)

        # Cascade-clean v3 structured tables.
        # delete() handles per-memory cleanup, but tombstoned memories leave
        # orphan rows.  Belt-and-suspenders: purge by user_id via JOIN.
        if user_id:
            try:
                with self.db._get_connection() as conn:
                    for table in ("engram_facts", "engram_context", "engram_scenes",
                                  "engram_entities", "engram_links"):
                        col = "source_memory_id" if table == "engram_links" else "memory_id"
                        try:
                            conn.execute(
                                f"DELETE FROM {table} WHERE {col} IN "
                                f"(SELECT id FROM memories WHERE user_id = ?)",
                                (user_id,),
                            )
                        except Exception:
                            pass  # Table may not exist yet
            except Exception as e:
                logger.debug("v3 table cleanup skipped: %s", e)

        return {"deleted_count": count}

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        return self.db.get_history(memory_id)

    def reset(self) -> None:
        """Delete ALL memories including tombstoned. This is IRREVERSIBLE."""
        memories = self.db.get_all_memories(include_tombstoned=True)
        logger.warning("reset: permanently deleting ALL %d memories", len(memories))
        for mem in memories:
            self.delete(mem["id"])
        if hasattr(self.vector_store, "reset"):
            self.vector_store.reset()

    # FadeMem-specific methods
    def apply_decay(self, scope: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.fade_config.enable_forgetting:
            return {"decayed": 0, "forgotten": 0, "promoted": 0}

        stale_refs_removed = 0

        memories = self.db.get_all_memories(
            user_id=scope.get("user_id") if scope else None,
            agent_id=scope.get("agent_id") if scope else None,
            run_id=scope.get("run_id") if scope else None,
            app_id=scope.get("app_id") if scope else None,
        )

        decayed = 0
        forgotten = 0
        promoted = 0
        # Each memory's clock starts at its own last decay, not its last
        # access: the stored strength is what that pass left. See `decay_since`.
        last_decayed = (
            self.db.latest_event_times("DECAY")
            if callable(getattr(self.db, "latest_event_times", None))
            else {}
        )
        # Forgetting a memory deletes the facts extracted from it, so one that
        # is still the source of a current fact fades only to the threshold.
        holding_facts = (
            self.db.memory_ids_holding_current_facts()
            if callable(getattr(self.db, "memory_ids_holding_current_facts", None))
            else set()
        )

        for memory in memories:
            if memory.get("immutable"):
                continue
            since = decay_since(
                memory.get("last_accessed", memory.get("created_at")),
                last_decayed.get(str(memory.get("id"))),
            )

            # Shruti-tier memories are immune to decay
            _tier_md = memory.get("metadata") or {}
            if isinstance(_tier_md, str):
                import json as _tjson
                try:
                    _tier_md = _tjson.loads(_tier_md)
                except (ValueError, TypeError):
                    _tier_md = {}
            if _tier_md.get("tier") == "shruti":
                continue

            # Task-aware decay: active tasks don't decay
            if memory.get("memory_type") == "task":
                _md = memory.get("metadata") or {}
                if isinstance(_md, str):
                    import json as _json
                    try:
                        _md = _json.loads(_md)
                    except (ValueError, TypeError):
                        _md = {}
                _ts = _md.get("task_status", "inbox")
                if _ts in ("inbox", "assigned", "active", "review", "blocked"):
                    continue  # skip decay for active tasks

            ref_aware = feature_enabled("ENGRAM_V2_REF_AWARE_DECAY", default=False)
            ref_state = {"strong": 0, "weak": 0}
            if ref_aware:
                ref_state = self.db.get_memory_refcount(memory["id"])
                if int(ref_state.get("strong", 0)) > 0:
                    # Strong references pause decay/deletion.
                    metrics.record_ref_protected_skip(1)
                    continue

            # Gap 4: Multi-trace decay (if enabled and traces are initialized)
            use_multi_trace = (
                self.distillation_config
                and self.distillation_config.enable_multi_trace
                and memory.get("s_fast") is not None
            )

            if use_multi_trace:
                s_f, s_m, s_s = decay_traces(
                    s_fast=float(memory.get("s_fast", 0.0)),
                    s_mid=float(memory.get("s_mid", 0.0)),
                    s_slow=float(memory.get("s_slow", 0.0)),
                    last_accessed=since,
                    access_count=memory.get("access_count", 0),
                    config=self.distillation_config,
                )
                new_strength = compute_effective_strength(s_f, s_m, s_s, self.distillation_config)
            else:
                new_strength = calculate_decayed_strength(
                    current_strength=memory.get("strength", 1.0),
                    last_accessed=since,
                    access_count=memory.get("access_count", 0),
                    layer=memory.get("layer", "sml"),
                    config=self.fade_config,
                )

            if ref_aware and int(ref_state.get("weak", 0)) > 0:
                weak = min(int(ref_state.get("weak", 0)), 10)
                dampening = 1.0 + weak * 0.15
                retained_floor = memory.get("strength", 1.0) * (1.0 - 0.03 / dampening)
                new_strength = max(new_strength, retained_floor)

            forget_threshold = self.fade_config.forgetting_threshold
            if ref_aware and int(ref_state.get("weak", 0)) > 0:
                weak = min(int(ref_state.get("weak", 0)), 10)
                forget_threshold = forget_threshold / (1.0 + weak * 0.25)

            if new_strength < forget_threshold and str(memory.get("id")) in holding_facts:
                new_strength = forget_threshold
            if new_strength < forget_threshold:
                self.delete(memory["id"])
                forgotten += 1
                continue

            if new_strength != memory.get("strength"):
                if use_multi_trace:
                    self.db.update_multi_trace(memory["id"], s_f, s_m, s_s, new_strength)
                else:
                    self.db.update_memory(memory["id"], {"strength": new_strength})
                self.db.log_event(memory["id"], "DECAY", old_strength=memory.get("strength"), new_strength=new_strength)
                decayed += 1

            if should_promote(
                memory.get("layer", "sml"),
                memory.get("access_count", 0),
                new_strength,
                self.fade_config,
            ):
                from dhee.memory.quality import memory_quality_from_record

                quality = memory_quality_from_record(memory)
                if quality.layer == "lml" and not quality.suppress_from_default_recall:
                    self.db.update_memory(memory["id"], {"layer": "lml"})
                    self.db.log_event(memory["id"], "PROMOTE", old_layer="sml", new_layer="lml")
                    promoted += 1

        if self.fade_config.use_tombstone_deletion:
            self.db.purge_tombstoned()

        # Gap 3: Advanced forgetting mechanisms
        interference_stats = {"checked": 0, "demoted": 0}
        redundancy_stats = {"groups_fused": 0, "memories_fused": 0}
        homeostasis_stats = {"namespaces_over_budget": 0, "pressured": 0, "forgotten": 0}

        if self.distillation_config:
            user_id = scope.get("user_id") if scope else None

            _do_interference = self.distillation_config.enable_interference_pruning
            _do_redundancy = self.distillation_config.enable_redundancy_collapse

            # Site 3: Parallel interference + redundancy during apply_decay
            _use_parallel_decay = (
                self._executor is not None
                and self.parallel_config
                and self.parallel_config.parallel_decay
                and _do_interference
                and _do_redundancy
            )

            if _use_parallel_decay:
                pruner = InterferencePruner(
                    db=self.db,
                    config=self.distillation_config,
                    fade_config=self.fade_config,
                    resolve_conflict_fn=resolve_conflict,
                    search_fn=self.vector_store.search,
                    llm=self.llm,
                )
                collapser = RedundancyCollapser(
                    db=self.db,
                    config=self.distillation_config,
                    fuse_fn=self.fuse_memories,
                    search_fn=self.vector_store.search,
                )
                def _run_pruner():
                    return pruner.run(memories, user_id=user_id)

                def _run_collapser():
                    return collapser.run(memories, user_id=user_id)

                interference_stats, redundancy_stats = self._executor.run_parallel([
                    (_run_pruner, ()),
                    (_run_collapser, ()),
                ])
            else:
                if _do_interference:
                    pruner = InterferencePruner(
                        db=self.db,
                        config=self.distillation_config,
                        fade_config=self.fade_config,
                        resolve_conflict_fn=resolve_conflict,
                        search_fn=self.vector_store.search,
                        llm=self.llm,
                    )
                    interference_stats = pruner.run(memories, user_id=user_id)

                if _do_redundancy:
                    collapser = RedundancyCollapser(
                        db=self.db,
                        config=self.distillation_config,
                        fuse_fn=self.fuse_memories,
                        search_fn=self.vector_store.search,
                    )
                    redundancy_stats = collapser.run(memories, user_id=user_id)

            if self.distillation_config.enable_homeostasis and user_id:
                normalizer = HomeostaticNormalizer(
                    db=self.db,
                    config=self.distillation_config,
                    fade_config=self.fade_config,
                    delete_fn=self.delete,
                )
                homeostasis_stats = normalizer.run(user_id)

        # Distillation: episodic → semantic consolidation (the "sleep cycle")
        # This is the most impactful fusion operation per FadeMem ablation (53.7% F1 drop without it).
        # Runs after decay so it works on surviving memories only.
        distillation_stats = {}
        if (
            user_id
            and self.distillation_config
            and self.distillation_config.enable_distillation
            and self.llm
        ):
            try:
                distiller = ReplayDistiller(
                    db=self.db,
                    llm=self.llm,
                    config=self.distillation_config,
                )
                distillation_stats = distiller.run(
                    user_id=user_id,
                    memory_add_fn=self.add,
                )
                logger.info("Distillation: %s", distillation_stats)
            except Exception as e:
                logger.warning("Distillation failed: %s", e)
                distillation_stats = {"error": str(e)}

        self.db.log_decay(decayed, forgotten, promoted)
        return {
            "decayed": decayed,
            "forgotten": forgotten,
            "promoted": promoted,
            "stale_refs_removed": stale_refs_removed,
            "interference": interference_stats,
            "redundancy": redundancy_stats,
            "homeostasis": homeostasis_stats,
            "distillation": distillation_stats,
        }

    def sleep_cycle(self, user_id: str = "default") -> Dict[str, Any]:
        """Run a full sleep cycle: decay + distillation + fusion.

        Models the hippocampus-to-neocortex transfer in Complementary Learning
        Systems theory. Call this periodically (e.g. daily) to:
        1. Decay weak memories (Ebbinghaus forgetting curve)
        2. Fuse redundant memories (compression without information loss)
        3. Distill episodic → semantic knowledge (consolidation)
        4. Enforce memory budgets (homeostasis)
        """
        return self.apply_decay(scope={"user_id": user_id})

    def fuse_memories(self, memory_ids: List[str], user_id: Optional[str] = None) -> Dict[str, Any]:
        memories = [self.db.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m]
        if len(memories) < 2:
            return {"error": "Need at least 2 memories to fuse"}

        fused = fuse_memories(memories, self.llm, self.config.custom_fusion_prompt)
        result = self.add(
            fused.content,
            user_id=user_id or memories[0].get("user_id"),
            agent_id=memories[0].get("agent_id"),
            run_id=memories[0].get("run_id"),
            app_id=memories[0].get("app_id"),
            initial_layer=fused.layer,
            initial_strength=fused.strength,
            infer=False,
        )

        for mid in memory_ids:
            self.delete(mid)

        fused_id = result.get("results", [{}])[0].get("id") if result.get("results") else None
        return {"fused_id": fused_id, "source_ids": memory_ids, "fused_memory": fused.content}

    # ── Dhee: Cognition + Prospective Scenes ──

    def think(
        self,
        question: str,
        user_id: str = "default",
        max_depth: Optional[int] = None,
        repo: Optional[str] = None,
        ask_user_fn=None,
    ):
        """Cognitive decomposition loop — memory-grounded reasoning.

        Decomposes the question, searches memory for each sub-question,
        grounds facts, and synthesizes an answer from verified facts.
        """
        if not self.cognition_engine:
            # Direct search if cognition engine not configured
            return self.search(query=question, user_id=user_id)
        return self.cognition_engine.think(
            question=question,
            user_id=user_id,
            max_depth=max_depth,
            repo=repo,
            ask_user_fn=ask_user_fn,
        )

    def get_prospective_scenes(
        self,
        user_id: str = "default",
        now_iso: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get predicted scenes that should be triggered now.

        Returns upcoming events with relevant past context — the memory
        engine proactively surfaces what you'll need before you ask.
        """
        if now_iso is None:
            from datetime import datetime, timezone
            now_iso = datetime.now(timezone.utc).isoformat()

        try:
            with self.db._get_connection() as conn:
                rows = conn.execute(
                    """SELECT * FROM engram_prospective_scenes
                    WHERE user_id = ? AND status = 'predicted'
                    ORDER BY predicted_time ASC""",
                    (user_id,),
                ).fetchall()

                due_scenes = []
                for row in rows:
                    predicted_time = row["predicted_time"]
                    trigger_hours = row["trigger_window_hours"] or 24
                    if predicted_time:
                        try:
                            from datetime import datetime as dt
                            predicted = dt.fromisoformat(predicted_time.replace("Z", "+00:00"))
                            now = dt.fromisoformat(now_iso.replace("Z", "+00:00"))
                            delta_hours = (predicted - now).total_seconds() / 3600
                            if 0 <= delta_hours <= trigger_hours:
                                scene_data = dict(row)
                                # Parse JSON fields
                                import json
                                for json_field in ["participants", "predicted_needs", "relevant_past_scene_ids"]:
                                    val = scene_data.get(json_field, "[]")
                                    if isinstance(val, str):
                                        try:
                                            scene_data[json_field] = json.loads(val)
                                        except (json.JSONDecodeError, TypeError):
                                            scene_data[json_field] = []

                                # Enrich with relevant past memories
                                past_ids = scene_data.get("relevant_past_scene_ids", [])
                                if past_ids:
                                    past_memories = self.db.get_memories_bulk(past_ids)
                                    scene_data["past_context"] = [
                                        {"id": mid, "memory": m.get("memory", "")}
                                        for mid, m in past_memories.items()
                                    ]

                                due_scenes.append(scene_data)
                                # Mark as triggered
                                conn.execute(
                                    "UPDATE engram_prospective_scenes SET status = 'triggered' WHERE id = ?",
                                    (row["id"],),
                                )
                        except (ValueError, TypeError):
                            continue

                return due_scenes
        except Exception as e:
            logger.debug("Prospective scene check failed: %s", e)
            return []

    def _store_prospective_scenes(
        self,
        scenes,
        memory_id: str,
        user_id: str,
    ) -> None:
        """Store prospective scenes (predicted future events) in DB."""
        import json
        import uuid
        try:
            with self.db._get_connection() as conn:
                for scene in scenes:
                    scene_id = str(uuid.uuid4())

                    # Find relevant past scenes by searching for similar activities/participants
                    relevant_past_ids = []
                    if scene.participants or scene.event_type:
                        search_query = " ".join(scene.participants or [])
                        if scene.event_type:
                            search_query += f" {scene.event_type}"
                        if search_query.strip():
                            try:
                                past_results = self.search(
                                    query=search_query, user_id=user_id, limit=5,
                                )
                                relevant_past_ids = [
                                    r.get("id", "") for r in past_results.get("results", [])
                                    if r.get("id")
                                ]
                            except Exception:
                                pass
                    scene.relevant_past_scene_ids = relevant_past_ids

                    conn.execute(
                        """INSERT INTO engram_prospective_scenes
                        (id, memory_id, user_id, predicted_time, trigger_window_hours,
                         event_type, participants, predicted_setting, predicted_needs,
                         relevant_past_scene_ids, status, prediction_basis)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            scene_id,
                            memory_id,
                            user_id,
                            scene.predicted_time,
                            scene.trigger_window_hours,
                            scene.event_type,
                            json.dumps(scene.participants),
                            scene.predicted_setting,
                            json.dumps(scene.predicted_needs),
                            json.dumps(scene.relevant_past_scene_ids),
                            scene.status,
                            scene.prediction_basis,
                        ),
                    )
        except Exception as e:
            logger.debug("Failed to store prospective scene: %s", e)

    def _get_quality_health_memories(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        min_strength: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return rows that make up an agent's effective memory-health view.

        Agent-specific passive observations should stay scoped to that agent,
        but canonical personal/project knowledge is deliberately user-scoped so
        every assistant for that user can rely on the same durable facts. Health
        checks therefore combine exact agent rows with shared canonical/project
        rows instead of reporting a passive-only slice for agents such as Chotu.
        """
        memories = self.db.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            min_strength=min_strength,
            limit=limit,
        )
        if not (user_id and agent_id):
            return memories

        from dhee.memory.quality import memory_quality_from_record

        seen = {str(memory.get("id")) for memory in memories if memory.get("id")}
        shared_candidates = self.db.get_all_memories(
            user_id=user_id,
            min_strength=min_strength,
            limit=limit,
        )
        for memory in shared_candidates:
            memory_id = str(memory.get("id") or "")
            if not memory_id or memory_id in seen:
                continue
            if memory.get("agent_id") not in (None, ""):
                continue
            quality = memory_quality_from_record(memory)
            if quality.memory_class not in {"canonical_personal", "project_context"}:
                continue
            memories.append(memory)
            seen.add(memory_id)
            if limit and len(memories) >= limit:
                break
        return memories

    @staticmethod
    def _scene_noise_kind(scene: Dict[str, Any]) -> Optional[str]:
        title = str(scene.get("title") or "").strip()
        summary = str(scene.get("summary") or "").strip()
        topic = str(scene.get("topic") or "").strip()
        text = "\n".join(part for part in (title, summary, topic) if part)
        if not text:
            return None
        if _SCENE_BENCHMARK_NOISE_RE.search(text):
            return "benchmark_prompt_scene"
        operational = _SCENE_OPERATIONAL_VERB_RE.match(title)
        if operational and _SCENE_PATH_HINT_RE.search(operational.group(1)):
            return "operational_scene"
        return None

    def _scene_noise_candidates(
        self,
        *,
        user_id: Optional[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 50_000))
        query = """
            SELECT id, user_id, title, summary, topic, start_time, end_time
            FROM scenes
            WHERE tombstone = 0
        """
        params: List[Any] = []
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(bounded_limit)
        with self.db._get_connection() as conn:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]

        candidates: List[Dict[str, Any]] = []
        for row in rows:
            kind = self._scene_noise_kind(row)
            if not kind:
                continue
            row["noise_kind"] = kind
            candidates.append(row)
        return candidates

    def _backup_history_db(self, reason: str) -> Optional[str]:
        raw_path = getattr(self.config, "history_db_path", None)
        if not raw_path:
            return None
        db_path = Path(str(raw_path)).expanduser().resolve()
        if not db_path.exists():
            return None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{db_path.stem}-{reason}-{timestamp}{db_path.suffix}"
        shutil.copy2(db_path, backup_path)
        return str(backup_path)

    def _tombstone_scene_noise(
        self,
        candidates: List[Dict[str, Any]],
        *,
        dry_run: bool,
    ) -> Dict[str, Any]:
        counts = {
            "scene_noise_tombstoned": 0,
            "operational_scenes_tombstoned": 0,
            "benchmark_scenes_tombstoned": 0,
            "scene_noise_events_written": 0,
        }
        if not candidates:
            return {**counts, "scene_noise_backup": None}

        for scene in candidates:
            if scene["noise_kind"] == "operational_scene":
                counts["operational_scenes_tombstoned"] += 1
            elif scene["noise_kind"] == "benchmark_prompt_scene":
                counts["benchmark_scenes_tombstoned"] += 1
        counts["scene_noise_tombstoned"] = len(candidates)
        if dry_run:
            return {**counts, "scene_noise_backup": None}

        backup_path = self._backup_history_db("scene-noise")
        now = datetime.now(timezone.utc).isoformat()
        events: List[Dict[str, Any]] = []
        for scene in candidates:
            scene_id = str(scene["id"])
            event_time = str(scene.get("start_time") or scene.get("end_time") or now)
            event_type = (
                "operational_event"
                if scene["noise_kind"] == "operational_scene"
                else "test_fixture"
            )
            value_text = str(scene.get("title") or scene.get("summary") or scene_id)
            events.append(
                {
                    "id": f"scene-noise:{scene_id}",
                    "memory_id": f"scene:{scene_id}",
                    "user_id": scene.get("user_id") or "default",
                    "conversation_id": None,
                    "session_id": None,
                    "turn_id": 0,
                    "actor_id": "dhee_scene_repair",
                    "actor_role": "maintenance",
                    "event_time": event_time,
                    "event_type": event_type,
                    "canonical_key": f"scene_noise:{scene['noise_kind']}:{scene_id}",
                    "value_text": value_text,
                    "value_num": None,
                    "value_unit": None,
                    "currency": None,
                    "normalized_time_start": event_time,
                    "normalized_time_end": event_time,
                    "time_granularity": "instant",
                    "entity_key": str(scene.get("topic") or scene.get("title") or scene_id)[:240],
                    "value_norm": " ".join(value_text.lower().split())[:500],
                    "confidence": 1.0,
                    "superseded_by": None,
                }
            )
        counts["scene_noise_events_written"] = self.db.add_episodic_events(events)
        scene_ids = [(str(scene["id"]),) for scene in candidates]
        with self.db._get_connection() as conn:
            conn.executemany(
                "UPDATE scenes SET tombstone = 1, strength = 0.0 WHERE id = ?",
                scene_ids,
            )
            try:
                conn.executemany(
                    "UPDATE memories SET scene_id = NULL WHERE scene_id = ?",
                    scene_ids,
                )
            except sqlite3.OperationalError as exc:
                if "no such column" not in str(exc).lower():
                    raise
        with self.db._get_connection() as conn:
            conn.execute("VACUUM")
        return {**counts, "scene_noise_backup": backup_path}

    def get_stats(self, user_id: Optional[str] = None, agent_id: Optional[str] = None) -> Dict[str, Any]:
        memories = self._get_quality_health_memories(user_id=user_id, agent_id=agent_id)
        sml_count = sum(1 for m in memories if m.get("layer") == "sml")
        lml_count = sum(1 for m in memories if m.get("layer") == "lml")
        strengths = [m.get("strength", 1.0) for m in memories]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
        from dhee.memory.quality import memory_quality_from_record

        by_memory_class: Dict[str, int] = {}
        by_namespace: Dict[str, int] = {}
        by_memory_type: Dict[str, int] = {}
        durable_count = 0
        degraded_or_queued_count = 0
        superseded_count = 0
        shared_personal_count = 0
        for m in memories:
            metadata = m.get("metadata", {}) or {}
            quality = memory_quality_from_record(m)
            by_memory_class[quality.memory_class] = by_memory_class.get(quality.memory_class, 0) + 1
            namespace = str(m.get("namespace") or metadata.get("namespace") or "default")
            by_namespace[namespace] = by_namespace.get(namespace, 0) + 1
            mem_type = str(m.get("memory_type") or metadata.get("memory_type") or "semantic")
            by_memory_type[mem_type] = by_memory_type.get(mem_type, 0) + 1
            if agent_id and m.get("agent_id") in (None, "") and quality.memory_class in {"canonical_personal", "project_context"}:
                shared_personal_count += 1
            if str(metadata.get("retention_policy") or "").lower() == "durable":
                durable_count += 1
            if metadata.get("degraded") or metadata.get("queued") or str(m.get("enrichment_status") or "").lower() in {"queued", "degraded", "failed"}:
                degraded_or_queued_count += 1
            if metadata.get("superseded"):
                superseded_count += 1

        # EchoMem stats
        echo_stats = {"shallow": 0, "medium": 0, "deep": 0, "none": 0}
        for m in memories:
            metadata = m.get("metadata", {})
            depth = metadata.get("echo_depth", "none")
            if depth in echo_stats:
                echo_stats[depth] += 1
            else:
                echo_stats["none"] += 1

        try:
            write_cost = self.db.aggregate_cost_counters(phase="write", user_id=user_id)
        except Exception as exc:
            write_cost = {"phase": "write", "samples": 0, "error": f"{type(exc).__name__}: {exc}"}
        try:
            query_cost = self.db.aggregate_cost_counters(phase="query", user_id=user_id)
        except Exception as exc:
            query_cost = {"phase": "query", "samples": 0, "error": f"{type(exc).__name__}: {exc}"}

        return {
            "total": len(memories),
            "sml_count": sml_count,
            "lml_count": lml_count,
            "avg_strength": round(avg_strength, 3),
            "quality": {
                "by_memory_class": by_memory_class,
                "by_namespace": by_namespace,
                "by_memory_type": by_memory_type,
                "canonical_personal_count": by_memory_class.get("canonical_personal", 0),
                "passive_screen_count": by_memory_class.get("passive_screen", 0),
                "test_fixture_count": by_memory_class.get("test_fixture", 0),
                "operational_event_count": by_memory_class.get("operational_event", 0),
                "durable_count": durable_count,
                "degraded_or_queued_count": degraded_or_queued_count,
                "superseded_count": superseded_count,
                "shared_personal_count": shared_personal_count,
                "warnings": [
                    warning
                    for warning in (
                        "canonical_personal_count is zero" if by_memory_class.get("canonical_personal", 0) == 0 else "",
                        "degraded_or_queued writes need repair" if degraded_or_queued_count else "",
                        "test_fixture memories are present but isolated" if by_memory_class.get("test_fixture", 0) else "",
                    )
                    if warning
                ],
            },
            "echo_stats": echo_stats,
            "echo_enabled": self.echo_config.enable_echo if self.echo_config else False,
            "cost_counters": {
                "write": write_cost,
                "query": query_cost,
            },
        }

    def audit_memory_quality(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10_000,
        profile_keyword: Optional[str] = None,
        require_personal_model: bool = True,
    ) -> Dict[str, Any]:
        """Return an operator-grade readiness report for the memory vault.

        The audit is read-only and intentionally deterministic. It answers the
        Chotu-class question directly: can Dhee carry a dependable personal
        model, or are canonical facts, repair queues, or noise still broken?
        """
        from dhee.memory.quality import (
            apply_memory_quality_contract,
            enforce_quality_layer,
            enforce_quality_strength,
            memory_quality_from_record,
            sanitize_profile_for_recall,
        )

        memories = self._get_quality_health_memories(
            user_id=user_id,
            agent_id=agent_id,
            min_strength=0.0,
            limit=limit,
        )
        profile_l = (profile_keyword or "").strip().lower()
        by_memory_class: Dict[str, int] = {}
        by_namespace: Dict[str, int] = {}
        by_memory_type: Dict[str, int] = {}
        counts = {
            "total": len(memories),
            "canonical_personal": 0,
            "canonical_profile_matches": 0,
            "project_context": 0,
            "passive_screen": 0,
            "test_fixture": 0,
            "operational_event": 0,
            "evidence_artifact": 0,
            "lml": 0,
            "durable": 0,
            "degraded_or_queued": 0,
            "superseded": 0,
            "unresolved_test_noise": 0,
            "unresolved_passive_noise": 0,
            "unresolved_operational_noise": 0,
            "unapproved_lml": 0,
            "unpromoted_canonical": 0,
            "damaged_canonical": 0,
            "evidence_without_distillation": 0,
            "contract_repairs_available": 0,
            "profile_contamination": 0,
            "scene_noise": 0,
            "operational_scene_noise": 0,
            "benchmark_scene_noise": 0,
        }
        samples: Dict[str, List[Dict[str, Any]]] = {
            "unresolved_test_noise": [],
            "unresolved_passive_noise": [],
            "unresolved_operational_noise": [],
            "unapproved_lml": [],
            "unpromoted_canonical": [],
            "damaged_canonical": [],
            "evidence_without_distillation": [],
            "profile_contamination": [],
            "scene_noise": [],
        }

        def _sample(memory: Dict[str, Any], issue: str, updates: List[str]) -> None:
            bucket = samples.setdefault(issue, [])
            if len(bucket) >= 8:
                return
            text = " ".join(str(memory.get("memory") or "").split())
            bucket.append(
                {
                    "id": memory.get("id"),
                    "agent_id": memory.get("agent_id"),
                    "namespace": memory.get("namespace"),
                    "memory_type": memory.get("memory_type"),
                    "layer": memory.get("layer"),
                    "strength": round(float(memory.get("strength") or 0.0), 3),
                    "preview": text[:160],
                    "updates_needed": updates,
                }
            )

        for memory in memories:
            raw_metadata = memory.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata = raw_metadata
            elif raw_metadata not in (None, {}, []):
                metadata = {
                    "legacy_metadata_raw": raw_metadata,
                    "legacy_metadata_type": type(raw_metadata).__name__,
                }
            else:
                metadata = {}
            categories = memory.get("categories") if isinstance(memory.get("categories"), list) else []
            quality = memory_quality_from_record(memory)
            by_memory_class[quality.memory_class] = by_memory_class.get(quality.memory_class, 0) + 1
            namespace = str(memory.get("namespace") or metadata.get("namespace") or "default")
            by_namespace[namespace] = by_namespace.get(namespace, 0) + 1
            memory_type = str(memory.get("memory_type") or metadata.get("memory_type") or "semantic")
            by_memory_type[memory_type] = by_memory_type.get(memory_type, 0) + 1

            if quality.memory_class in counts:
                counts[quality.memory_class] += 1
            if memory.get("layer") == "lml":
                counts["lml"] += 1
            if str(metadata.get("retention_policy") or "").lower() == "durable":
                counts["durable"] += 1
            enrichment_status = str(memory.get("enrichment_status") or "").lower()
            if metadata.get("degraded") or metadata.get("queued") or enrichment_status in {"queued", "degraded", "failed"}:
                counts["degraded_or_queued"] += 1
            if metadata.get("superseded"):
                counts["superseded"] += 1

            content_l = str(memory.get("memory") or "").lower()
            metadata_l = json.dumps(metadata, sort_keys=True, default=str).lower()
            if quality.memory_class == "canonical_personal" and profile_l and (profile_l in content_l or profile_l in metadata_l):
                counts["canonical_profile_matches"] += 1

            updated_metadata, target_quality = apply_memory_quality_contract(
                str(memory.get("memory") or ""),
                metadata,
                categories,
                explicit_remember=bool(metadata.get("policy_explicit") or metadata.get("explicit_remember")),
            )
            updates_needed: List[str] = []
            if updated_metadata != metadata:
                updates_needed.append("metadata")
            if str(memory.get("namespace") or "default") != target_quality.namespace:
                updates_needed.append("namespace")
            if str(memory.get("memory_type") or "semantic") != target_quality.memory_type:
                updates_needed.append("memory_type")
            target_layer = enforce_quality_layer(str(memory.get("layer") or "sml"), target_quality)
            if target_layer != memory.get("layer"):
                updates_needed.append("layer")
            target_strength = enforce_quality_strength(float(memory.get("strength") or 0.0), target_quality)
            if abs(target_strength - float(memory.get("strength") or 0.0)) > 1e-9:
                updates_needed.append("strength")
            if enrichment_status in {"queued", "degraded", "failed"}:
                updates_needed.append("enrichment_status")

            if updates_needed:
                counts["contract_repairs_available"] += 1

            if target_quality.memory_class == "test_fixture" and updates_needed:
                counts["unresolved_test_noise"] += 1
                _sample(memory, "unresolved_test_noise", updates_needed)
            if target_quality.memory_class == "passive_screen" and updates_needed:
                counts["unresolved_passive_noise"] += 1
                _sample(memory, "unresolved_passive_noise", updates_needed)
            if target_quality.memory_class == "operational_event" and updates_needed:
                counts["unresolved_operational_noise"] += 1
                _sample(memory, "unresolved_operational_noise", updates_needed)
            if memory.get("layer") == "lml" and target_quality.layer != "lml":
                counts["unapproved_lml"] += 1
                _sample(memory, "unapproved_lml", updates_needed or ["layer"])
            if target_quality.memory_class == "canonical_personal" and updates_needed:
                counts["unpromoted_canonical"] += 1
                _sample(memory, "unpromoted_canonical", updates_needed)
            if target_quality.memory_class in {"evidence_artifact", "passive_screen"} and not metadata.get("evidence_distillation"):
                counts["evidence_without_distillation"] += 1
                _sample(memory, "evidence_without_distillation", updates_needed or ["evidence_distillation"])

            looks_canonical = (
                quality.memory_class == "canonical_personal"
                or target_quality.memory_class == "canonical_personal"
                or namespace == "canonical_personal"
                or bool(metadata.get("canonical_personal"))
            )
            if looks_canonical and (
                metadata.get("superseded")
                or float(memory.get("strength") or 0.0) < 0.2
                or memory.get("layer") != "lml"
                or enrichment_status in {"queued", "degraded", "failed"}
            ):
                counts["damaged_canonical"] += 1
                _sample(memory, "damaged_canonical", updates_needed or ["canonical_integrity"])

        try:
            profiles = self.db.get_all_profiles(user_id=user_id)
        except Exception:
            profiles = []
        for profile in profiles or []:
            cleaned_profile, issue = sanitize_profile_for_recall(profile)
            if not issue:
                continue
            counts["profile_contamination"] += 1
            counts["contract_repairs_available"] += 1
            bucket = samples.setdefault("profile_contamination", [])
            if len(bucket) < 8:
                bucket.append(
                    {
                        "id": profile.get("id"),
                        "name": profile.get("name"),
                        "profile_type": profile.get("profile_type"),
                        "strength": round(float(profile.get("strength") or 0.0), 3),
                        "removed": issue.get("removed", {}),
                        "suppressed_entire_profile": cleaned_profile is None,
                    }
                )

        scene_noise = self._scene_noise_candidates(user_id=user_id, limit=limit)
        counts["scene_noise"] = len(scene_noise)
        for scene in scene_noise:
            if scene["noise_kind"] == "operational_scene":
                counts["operational_scene_noise"] += 1
            elif scene["noise_kind"] == "benchmark_prompt_scene":
                counts["benchmark_scene_noise"] += 1
            bucket = samples.setdefault("scene_noise", [])
            if len(bucket) < 8:
                bucket.append(
                    {
                        "id": scene.get("id"),
                        "user_id": scene.get("user_id"),
                        "noise_kind": scene.get("noise_kind"),
                        "title": scene.get("title"),
                    }
                )
        if scene_noise:
            counts["contract_repairs_available"] += len(scene_noise)

        checks: List[Dict[str, Any]] = []

        def _check(name: str, passed: bool, severity: str, detail: str) -> None:
            checks.append(
                {
                    "name": name,
                    "passed": bool(passed),
                    "severity": severity,
                    "detail": detail,
                }
            )

        _check("stats_available", True, "critical", f"Scanned {len(memories)} active memories.")
        if require_personal_model:
            _check(
                "canonical_personal_present",
                counts["canonical_personal"] > 0,
                "critical",
                f"{counts['canonical_personal']} canonical personal memories found.",
            )
            if profile_l:
                _check(
                    "profile_keyword_grounded",
                    counts["canonical_profile_matches"] > 0,
                    "critical",
                    f"{counts['canonical_profile_matches']} canonical memories mention {profile_keyword!r}.",
                )
            _check(
                "lml_personal_layer_present",
                counts["lml"] > 0,
                "critical",
                f"{counts['lml']} long-memory-layer records found.",
            )
        _check(
            "no_unresolved_test_noise",
            counts["unresolved_test_noise"] == 0,
            "critical",
            f"{counts['unresolved_test_noise']} test-like memories still need isolation.",
        )
        _check(
            "no_unresolved_passive_noise",
            counts["unresolved_passive_noise"] == 0,
            "critical",
            f"{counts['unresolved_passive_noise']} passive observations still need isolation.",
        )
        _check(
            "no_unresolved_operational_noise",
            counts["unresolved_operational_noise"] == 0,
            "critical",
            f"{counts['unresolved_operational_noise']} tool/session transport memories still need isolation.",
        )
        _check(
            "no_unapproved_lml",
            counts["unapproved_lml"] == 0,
            "critical",
            f"{counts['unapproved_lml']} non-canonical memories are still in the long-memory layer.",
        )
        _check(
            "canonical_integrity",
            counts["damaged_canonical"] == 0,
            "critical",
            f"{counts['damaged_canonical']} canonical-looking memories are superseded, weak, or queued.",
        )
        _check(
            "repair_queue_clear",
            counts["degraded_or_queued"] == 0,
            "critical",
            f"{counts['degraded_or_queued']} memories are degraded, queued, or failed.",
        )
        _check(
            "evidence_distilled",
            counts["evidence_without_distillation"] == 0,
            "warning",
            f"{counts['evidence_without_distillation']} evidence/passive records lack structured distillation.",
        )
        _check(
            "no_profile_contamination",
            counts["profile_contamination"] == 0,
            "critical",
            f"{counts['profile_contamination']} derived profile anchors contain tool/test/doc noise.",
        )
        _check(
            "no_scene_noise",
            counts["scene_noise"] == 0,
            "critical",
            f"{counts['scene_noise']} operational/test prompt scenes remain active.",
        )

        failed_critical = [check for check in checks if check["severity"] == "critical" and not check["passed"]]
        failed_warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
        penalty = len(failed_critical) * 18 + len(failed_warnings) * 6
        penalty += min(20, counts["contract_repairs_available"] // 25)
        score = max(0, min(100, 100 - penalty))

        recommended_actions: List[str] = []
        if counts["contract_repairs_available"]:
            recommended_actions.append("Run repair_memory_quality with dry_run=false, then rerun this audit.")
        if counts["degraded_or_queued"]:
            recommended_actions.append("Run enrich_pending after repair to reconcile queued/degraded rows.")
        if counts["scene_noise"]:
            recommended_actions.append("Run repair_memory_quality with dry_run=false to tombstone operational/test prompt scenes.")
        if require_personal_model and counts["canonical_personal"] == 0:
            recommended_actions.append("Ingest explicit goals, preferences, decisions, style, and product philosophy as canonical personal memories.")
        if profile_l and counts["canonical_profile_matches"] == 0:
            recommended_actions.append(f"Ingest or promote canonical memories grounded to {profile_keyword!r}.")

        status = "ready" if not failed_critical else "needs_repair"
        if require_personal_model and counts["canonical_personal"] == 0:
            status = "needs_canonical_intake"

        return {
            "status": status,
            "ready": not failed_critical,
            "score": score,
            "scope": {
                "user_id": user_id,
                "agent_id": agent_id,
                "limit": limit,
                "profile_keyword": profile_keyword,
                "require_personal_model": require_personal_model,
            },
            "counts": counts,
            "by_memory_class": by_memory_class,
            "by_namespace": by_namespace,
            "by_memory_type": by_memory_type,
            "readiness_checks": checks,
            "sample_issues": {key: value for key, value in samples.items() if value},
            "recommended_actions": recommended_actions,
        }

    def repair_memory_quality(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10_000,
        dry_run: bool = True,
        reindex_vectors: bool = False,
    ) -> Dict[str, Any]:
        """Reconcile existing memories with the current quality contract.

        This is the production repair path for old Chotu/Dhee vaults: it
        promotes canonical personal truths, isolates passive/test noise, and
        reopens degraded queued enrichment rows without rewriting memory text.
        Set ``reindex_vectors`` when repairing a vault that may contain DB-only
        rows from previous vector-write failures.
        """
        from dhee.memory.quality import (
            apply_memory_quality_contract,
            enforce_quality_layer,
            enforce_quality_strength,
            sanitize_profile_for_recall,
        )

        memories = self.db.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            min_strength=0.0,
            limit=limit,
        )
        repaired: List[Dict[str, Any]] = []
        counts = {
            "canonical_promoted": 0,
            "passive_isolated": 0,
            "test_isolated": 0,
            "operational_isolated": 0,
            "ordinary_demoted_from_lml": 0,
            "queued_reopened": 0,
            "malformed_metadata_normalized": 0,
            "profiles_cleaned": 0,
            "scene_noise_tombstoned": 0,
            "operational_scenes_tombstoned": 0,
            "benchmark_scenes_tombstoned": 0,
            "scene_noise_events_written": 0,
        }
        for memory in memories:
            raw_metadata = memory.get("metadata")
            if isinstance(raw_metadata, dict):
                metadata = dict(raw_metadata)
                if "legacy_metadata_raw" in metadata and "legacy_metadata_type" in metadata:
                    counts["malformed_metadata_normalized"] += 1
            elif raw_metadata not in (None, {}, []):
                metadata = {
                    "legacy_metadata_raw": raw_metadata,
                    "legacy_metadata_type": type(raw_metadata).__name__,
                }
                counts["malformed_metadata_normalized"] += 1
            else:
                metadata = {}
            categories = memory.get("categories") if isinstance(memory.get("categories"), list) else []
            updated_metadata, quality = apply_memory_quality_contract(
                str(memory.get("memory") or ""),
                metadata,
                categories,
                explicit_remember=bool(metadata.get("policy_explicit") or metadata.get("explicit_remember")),
            )
            updates: Dict[str, Any] = {}
            if updated_metadata != metadata:
                updates["metadata"] = updated_metadata
            if str(memory.get("namespace") or "default") != quality.namespace:
                updates["namespace"] = quality.namespace
            if str(memory.get("memory_type") or "semantic") != quality.memory_type:
                updates["memory_type"] = quality.memory_type
            repaired_layer = enforce_quality_layer(str(memory.get("layer") or "sml"), quality)
            if repaired_layer != memory.get("layer"):
                updates["layer"] = repaired_layer
            repaired_strength = enforce_quality_strength(float(memory.get("strength") or 0.0), quality)
            if abs(repaired_strength - float(memory.get("strength") or 0.0)) > 1e-9:
                updates["strength"] = repaired_strength
            current_enrichment = str(memory.get("enrichment_status") or "").lower()
            if current_enrichment in {"queued", "degraded", "failed"}:
                updates["enrichment_status"] = "pending"
                counts["queued_reopened"] += 1

            if not updates:
                continue
            if quality.memory_class == "canonical_personal":
                counts["canonical_promoted"] += 1
            elif quality.memory_class == "passive_screen":
                counts["passive_isolated"] += 1
            elif quality.memory_class == "test_fixture":
                counts["test_isolated"] += 1
            elif quality.memory_class == "operational_event":
                counts["operational_isolated"] += 1
            elif quality.memory_class == "ordinary" and "layer" in updates:
                counts["ordinary_demoted_from_lml"] += 1

            item = {
                "id": memory.get("id"),
                "memory_class": quality.memory_class,
                "updates": sorted(updates.keys()),
            }
            repaired.append(item)
            if not dry_run:
                self.db.update_memory(str(memory["id"]), updates)
                payload_updates: Dict[str, Any] = {}
                if "metadata" in updates:
                    payload_updates.update(updates["metadata"])
                for key in ("namespace", "memory_type", "layer", "strength"):
                    if key in updates:
                        payload_updates[key] = updates[key]
                if payload_updates:
                    self._update_vectors_for_memory(str(memory["id"]), payload_updates)

        try:
            profiles = self.db.get_all_profiles(user_id=user_id)
        except Exception:
            profiles = []
        for profile in profiles or []:
            cleaned_profile, issue = sanitize_profile_for_recall(profile)
            if not issue:
                continue
            updates: Dict[str, Any] = {}
            if cleaned_profile is None:
                updates = {
                    "facts": [],
                    "preferences": [],
                    "aliases": [],
                    "narrative": None,
                    "profile_summary": None,
                    "strength": 0.0,
                }
            else:
                for key in ("name", "facts", "preferences", "aliases", "narrative", "profile_summary"):
                    if cleaned_profile.get(key) != profile.get(key):
                        updates[key] = cleaned_profile.get(key)
                if not any(
                    [
                        cleaned_profile.get("facts"),
                        cleaned_profile.get("preferences"),
                        cleaned_profile.get("aliases"),
                        cleaned_profile.get("narrative"),
                        cleaned_profile.get("profile_summary"),
                    ]
                ):
                    updates["strength"] = 0.0
            if not updates:
                continue
            counts["profiles_cleaned"] += 1
            repaired.append(
                {
                    "id": profile.get("id"),
                    "memory_class": "profile_anchor",
                    "updates": sorted(updates.keys()),
                    "name": profile.get("name"),
                }
            )
            if not dry_run:
                self.db.update_profile(str(profile["id"]), updates)

        scene_noise = self._scene_noise_candidates(user_id=user_id, limit=limit)
        for scene in scene_noise:
            repaired.append(
                {
                    "id": scene.get("id"),
                    "memory_class": scene.get("noise_kind"),
                    "updates": ["tombstone", "episodic_event"],
                    "title": scene.get("title"),
                }
            )
        scene_repair = self._tombstone_scene_noise(scene_noise, dry_run=dry_run)
        for key in (
            "scene_noise_tombstoned",
            "operational_scenes_tombstoned",
            "benchmark_scenes_tombstoned",
            "scene_noise_events_written",
        ):
            counts[key] = int(scene_repair.get(key, 0) or 0)

        result = {
            "dry_run": dry_run,
            "scanned_count": len(memories),
            "repair_count": len(repaired),
            **counts,
            "scene_noise_backup": scene_repair.get("scene_noise_backup"),
            "repairs": repaired[:200],
            "truncated_repairs": max(0, len(repaired) - 200),
        }
        if reindex_vectors:
            result["vector_repair"] = self.repair_memory_vectors(
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
                dry_run=dry_run,
            )
        return result

    def _primary_vector_payload_for_memory(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        metadata = _metadata_dict(memory.get("metadata"))
        categories = list(memory.get("categories") or [])
        content = str(memory.get("memory") or "")
        payload = dict(metadata)
        payload.update(
            {
                "memory_id": memory.get("id"),
                "user_id": memory.get("user_id"),
                "agent_id": memory.get("agent_id"),
                "run_id": memory.get("run_id"),
                "app_id": memory.get("app_id"),
                "categories": categories,
                "text": content,
                "type": "primary",
                "memory": content,
                "layer": memory.get("layer", "sml"),
                "strength": memory.get("strength", 1.0),
                "namespace": memory.get("namespace") or metadata.get("namespace") or "default",
                "memory_type": memory.get("memory_type") or metadata.get("memory_type") or "semantic",
                "confidentiality_scope": memory.get("confidentiality_scope", "work"),
                "source_type": memory.get("source_type"),
                "source_app": memory.get("source_app"),
                "source_event_id": memory.get("source_event_id"),
            }
        )
        return payload

    @staticmethod
    def _vector_payload_is_current(payload: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        keys = (
            "memory_id",
            "user_id",
            "agent_id",
            "run_id",
            "app_id",
            "categories",
            "text",
            "type",
            "memory",
            "layer",
            "strength",
            "namespace",
            "memory_type",
            "confidentiality_scope",
            "source_type",
            "source_app",
            "source_event_id",
            "dhee_memory_class",
            "canonical_kind",
            "retention_policy",
        )
        for key in keys:
            if payload.get(key) != expected.get(key):
                return False
        return True

    @staticmethod
    def _merge_vector_payload(
        current: Dict[str, Any],
        expected: Dict[str, Any],
        *,
        primary: bool,
    ) -> Dict[str, Any]:
        """Merge authoritative DB metadata without flattening echo-node text."""
        if primary:
            return dict(expected)
        node_specific = {"text", "type", "subtype", "memory", "category", "fact_text", "fact_index", "is_fact"}
        merged = dict(current or {})
        for key, value in expected.items():
            if key in node_specific and key in merged:
                continue
            merged[key] = value
        return merged

    def repair_memory_vectors(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 10_000,
        dry_run: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Reconcile the vector index with authoritative DB memory rows.

        This is intentionally model-free for existing rows: it uses the stored
        DB embedding and rebuilds the primary vector payload. That makes it safe
        for production repair queues where API keys or provider extras may be
        unavailable, while restoring semantic recall for DB-only memories.
        """
        memories = self.db.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            min_strength=0.0,
            limit=limit,
        )
        try:
            vector_info = self.vector_store.col_info() if self.vector_store else {}
        except Exception:
            vector_info = {}
        vector_backend = getattr(self.vector_store, "__class__", type("", (), {})).__name__ if self.vector_store else None
        expected_dims = vector_info.get("vector_size") or vector_info.get("dimension")

        repaired: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        counts = {
            "missing_vector": 0,
            "payload_stale": 0,
            "forced_reindex": 0,
            "missing_embedding": 0,
            "dimension_mismatch": 0,
            "vector_errors": 0,
        }

        for memory in memories:
            memory_id = str(memory.get("id") or "")
            if not memory_id:
                continue
            embedding = memory.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                counts["missing_embedding"] += 1
                continue
            if expected_dims:
                try:
                    if len(embedding) != int(expected_dims):
                        counts["dimension_mismatch"] += 1
                        errors.append(
                            {
                                "id": memory_id,
                                "error": "embedding_dimension_mismatch",
                                "embedding_dims": len(embedding),
                                "vector_store_dims": int(expected_dims),
                            }
                        )
                        continue
                except (TypeError, ValueError):
                    pass

            payload = self._primary_vector_payload_for_memory(memory)
            try:
                vectors = self.vector_store.list(filters={"memory_id": memory_id}) if self.vector_store else []
            except Exception as exc:
                counts["vector_errors"] += 1
                errors.append({"id": memory_id, "error": f"list_failed: {exc}"})
                continue

            primary = None
            try:
                primary = self.vector_store.get(memory_id) if self.vector_store else None
            except Exception:
                primary = None
            primary_payload = getattr(primary, "payload", None) or {}
            has_primary = primary is not None
            has_linked_vectors = bool(vectors)
            payload_current = (
                has_primary
                and self._vector_payload_is_current(primary_payload, payload)
                and all(
                    self._vector_payload_is_current(getattr(vec, "payload", None) or {}, payload)
                    for vec in vectors
                    if getattr(vec, "id", None) == memory_id
                )
            )

            action = None
            if force:
                counts["forced_reindex"] += 1
                action = "force_reindex"
            elif not has_primary and not has_linked_vectors:
                counts["missing_vector"] += 1
                action = "insert_primary"
            elif not has_primary:
                counts["missing_vector"] += 1
                action = "insert_primary"
            elif not payload_current:
                counts["payload_stale"] += 1
                action = "update_payload"

            if not action:
                continue

            repaired.append(
                {
                    "id": memory_id,
                    "action": action,
                    "vector_count": len(vectors) + (1 if has_primary and not has_linked_vectors else 0),
                }
            )
            if dry_run:
                continue

            try:
                if force:
                    self._delete_vectors_for_memory(memory_id)
                    self.vector_store.insert(vectors=[embedding], payloads=[payload], ids=[memory_id])
                elif action == "insert_primary":
                    self.vector_store.insert(vectors=[embedding], payloads=[payload], ids=[memory_id])
                    for vec in vectors:
                        vec_payload = getattr(vec, "payload", None) or {}
                        merged_payload = self._merge_vector_payload(
                            vec_payload,
                            payload,
                            primary=getattr(vec, "id", None) == memory_id,
                        )
                        self.vector_store.update(vec.id, payload=merged_payload)
                else:
                    targets = vectors or ([primary] if primary else [])
                    if primary and all(getattr(vec, "id", None) != memory_id for vec in targets):
                        targets.append(primary)
                    for vec in targets:
                        vec_payload = getattr(vec, "payload", None) or {}
                        merged_payload = self._merge_vector_payload(
                            vec_payload,
                            payload,
                            primary=getattr(vec, "id", None) == memory_id,
                        )
                        self.vector_store.update(vec.id, payload=merged_payload)
            except Exception as exc:
                counts["vector_errors"] += 1
                errors.append({"id": memory_id, "error": str(exc)})

        return {
            "dry_run": dry_run,
            "force": force,
            "scanned_count": len(memories),
            "repair_count": len(repaired),
            "vector_store": {
                "backend": vector_backend,
                "info": vector_info,
            },
            **counts,
            "repairs": repaired[:200],
            "truncated_repairs": max(0, len(repaired) - 200),
            "errors": errors[:50],
            "truncated_errors": max(0, len(errors) - 50),
        }

    def snapshot_cost_baseline(self, *, output_path: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Persist write/query counter aggregates for guardrail baselining."""
        write_cost = self.db.aggregate_cost_counters(phase="write", user_id=user_id)
        query_cost = self.db.aggregate_cost_counters(phase="query", user_id=user_id)

        write_samples = max(1, int(write_cost.get("samples", 0) or 0))
        write_llm_calls_per_memory = float(write_cost.get("llm_calls", 0.0) or 0.0) / float(write_samples)
        write_tokens_per_memory = (
            float(write_cost.get("input_tokens", 0.0) or 0.0)
            + float(write_cost.get("output_tokens", 0.0) or 0.0)
        ) / float(write_samples)

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "write": write_cost,
            "query": query_cost,
            "write_llm_calls_per_memory": round(write_llm_calls_per_memory, 6),
            "write_tokens_per_memory": round(write_tokens_per_memory, 6),
        }

        out_path = str(output_path or "").strip()
        if out_path:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        return payload

    def promote(self, memory_id: str) -> Dict[str, Any]:
        return {"success": self.db.update_memory(memory_id, {"layer": "lml"})}

    def demote(self, memory_id: str) -> Dict[str, Any]:
        return {"success": self.db.update_memory(memory_id, {"layer": "sml"})}

    # Internal helpers
    def _extract_memories(self, messages, metadata, prompt=None, includes=None, excludes=None):
        return self._write_pipeline.extract_memories(messages, metadata, prompt=prompt, includes=includes, excludes=excludes)

    def _should_use_agent_memory_extraction(self, messages, metadata):
        return MemoryWritePipeline._should_use_agent_memory_extraction(messages, metadata)

    def _classify_memory_type(self, metadata, role):
        return self._write_pipeline.classify_memory_type(metadata, role)

    def _select_primary_text(self, content, echo_result=None):
        return self._write_pipeline.select_primary_text(content, echo_result)

    def _resolve_memory_id(self, vector_result):
        return resolve_memory_id(vector_result)

    def _collapse_vector_results(self, vector_results):
        return collapse_vector_results(vector_results)

    def _normalize_scope(self, scope):
        return self._scope_resolver.normalize_scope(scope)

    def _normalize_agent_category(self, category):
        return self._scope_resolver.normalize_agent_category(category)

    def _normalize_connector_id(self, connector_id):
        return self._scope_resolver.normalize_connector_id(connector_id)

    def _infer_scope(self, **kwargs):
        return self._scope_resolver.infer_scope(**kwargs)

    def _resolve_scope(self, memory):
        return self._scope_resolver.resolve_scope(memory)

    def _get_scope_weight(self, scope):
        return self._scope_resolver.get_scope_weight(scope)

    def _allows_scope(self, memory, *, user_id=None, agent_id=None, agent_category=None, connector_ids=None):
        return self._scope_resolver.allows_scope(
            memory, user_id=user_id, agent_id=agent_id,
            agent_category=agent_category, connector_ids=connector_ids,
        )

    def _build_index_vectors(self, *, embedding_cache=None, **kwargs):
        return build_index_vectors(
            **kwargs, embedder=self.embedder, embedding_cache=embedding_cache,
        )

    def _delete_vectors_for_memory(self, memory_id: str) -> None:
        try:
            vectors = self.vector_store.list(filters={"memory_id": memory_id})
            if not vectors:
                self.vector_store.delete(memory_id)
                return
            for vec in vectors:
                self.vector_store.delete(vec.id)
        except Exception as e:
            logger.error(
                "Failed to delete vectors for memory %s: %s. "
                "Orphaned vector entries may exist.",
                memory_id, e,
            )

    def _update_vectors_for_memory(self, memory_id: str, payload_updates: Dict[str, Any]) -> None:
        try:
            vectors = self.vector_store.list(filters={"memory_id": memory_id})
        except Exception as e:
            logger.error("Failed to list vectors for memory %s: %s", memory_id, e)
            return
        if not vectors:
            try:
                existing = self.vector_store.get(memory_id)
                if existing:
                    payload = existing.payload or {}
                    payload.update(payload_updates)
                    self.vector_store.update(memory_id, payload=payload)
            except Exception as e:
                logger.error("Failed to update vector payload for memory %s: %s", memory_id, e)
            return
        for vec in vectors:
            payload = vec.payload or {}
            payload.update(payload_updates)
            try:
                self.vector_store.update(vec.id, payload=payload)
            except Exception as e:
                logger.error("Failed to update vector %s for memory %s: %s", vec.id, memory_id, e)

    def _nearest_memory(self, embedding: List[float], filters: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], float]:
        results = self.vector_store.search(query=None, vectors=embedding, limit=1, filters=filters)
        if not results:
            return None, 0.0
        memory_id = self._resolve_memory_id(results[0])
        memory = self.db.get_memory(memory_id)
        if not memory:
            return None, 0.0
        return memory, float(results[0].score)

    def _is_shareable_memory(self, memory):
        return _is_shareable(memory)

    @staticmethod
    def _belief_conflict_id(left_id: str, right_id: str) -> str:
        ordered = sorted([str(left_id or "").strip(), str(right_id or "").strip()])
        return f"belief::{ordered[0]}::{ordered[1]}"

    @staticmethod
    def _parse_belief_conflict_id(conflict_id: str) -> Tuple[str, str]:
        raw = str(conflict_id or "").strip()
        if not raw.startswith("belief::"):
            raise ValueError("Unsupported conflict id")
        parts = raw.split("::")
        if len(parts) != 3 or not parts[1] or not parts[2]:
            raise ValueError("Invalid belief conflict id")
        return parts[1], parts[2]

    def _belief_store(self):
        buddhi = self.buddhi_layer
        return getattr(buddhi, "beliefs", None) if buddhi is not None else None

    def _belief_evidence_preview(self, store: Any, belief_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            rows = list(store.get_belief_evidence(belief_id, limit=limit) or [])
        except Exception:
            return []
        return [
            {
                "id": row.get("id"),
                "content": row.get("content"),
                "supports": bool(row.get("supports", True)),
                "source": row.get("source"),
                "confidence": row.get("confidence"),
                "created_at": row.get("created_at"),
                "memory_id": row.get("memory_id"),
                "episode_id": row.get("episode_id"),
            }
            for row in rows
        ]

    def _belief_history_preview(self, store: Any, belief_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        try:
            rows = list(store.get_belief_history(belief_id, limit=limit) or [])
        except Exception:
            return []
        return [
            {
                "event_type": row.get("event_type"),
                "reason": row.get("reason"),
                "actor": row.get("actor"),
                "created_at": row.get("created_at"),
                "payload": row.get("payload") or {},
            }
            for row in rows
        ]

    def _belief_to_conflict_payload(self, store: Any, left: Any, right: Any) -> Dict[str, Any]:
        pair_gap = abs(float(getattr(left, "confidence", 0.0)) - float(getattr(right, "confidence", 0.0)))
        severity = "low"
        if pair_gap <= 0.15:
            severity = "high"
        elif pair_gap <= 0.35:
            severity = "medium"

        def pack(belief: Any) -> Dict[str, Any]:
            return {
                "id": belief.id,
                "content": belief.claim,
                "confidence": float(belief.confidence),
                "created": datetime.fromtimestamp(float(belief.created_at), tz=timezone.utc).isoformat()
                if getattr(belief, "created_at", None)
                else "",
                "source": getattr(belief, "origin", "belief"),
                "tier": "canonical" if float(getattr(belief, "confidence", 0.0)) >= 0.75 else "medium",
                "domain": getattr(belief, "domain", "general"),
                "freshness": getattr(getattr(belief, "freshness_status", None), "value", None),
                "lifecycle": getattr(getattr(belief, "lifecycle_status", None), "value", None),
                "truthStatus": getattr(getattr(belief, "truth_status", None), "value", None),
                "sourceMemoryIds": list(getattr(belief, "source_memory_ids", []) or []),
                "evidence": self._belief_evidence_preview(store, belief.id),
                "history": self._belief_history_preview(store, belief.id),
            }

        return {
            "id": self._belief_conflict_id(left.id, right.id),
            "kind": "belief",
            "severity": severity,
            "reason": f"Contradictory beliefs in {getattr(left, 'domain', 'general')}",
            "resolutionOptions": ["KEEP A", "KEEP B", "MERGE", "ARCHIVE BOTH"],
            "belief_a": pack(left),
            "belief_b": pack(right),
        }

    def _annotate_memory_resolution(
        self,
        memory_id: str,
        *,
        conflict_id: str,
        resolution: str,
        role: str,
        reason: Optional[str] = None,
        superseded_by: Optional[str] = None,
        demote: bool = False,
        tombstone: bool = False,
    ) -> None:
        memory = self.db.get_memory(memory_id)
        if not memory:
            return
        if demote:
            self._demote_existing(memory, reason=f"manual:{resolution}")
            memory = self.db.get_memory(memory_id) or memory
        metadata = _metadata_dict(memory.get("metadata"))
        metadata["manual_conflict_resolution"] = {
            "conflict_id": conflict_id,
            "resolution": resolution,
            "role": role,
            "reason": reason or "",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
            "superseded_by": superseded_by,
        }
        self.db.update_memory(memory_id, {"metadata": metadata})
        if tombstone:
            self.delete(memory_id)

    def get_conflicts(self, user_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        store = self._belief_store()
        if store is None:
            return []
        pairs = list(store.get_contradictions(user_id) or [])[: max(1, int(limit))]
        return [self._belief_to_conflict_payload(store, left, right) for left, right in pairs]

    def resolve_conflict(
        self,
        conflict_id: str,
        action: str,
        merged_content: Optional[str] = None,
        reason: Optional[str] = None,
        actor: str = "user",
    ) -> Dict[str, Any]:
        store = self._belief_store()
        if store is None:
            raise ValueError("Buddhi belief store is not available")

        left_id, right_id = self._parse_belief_conflict_id(conflict_id)
        left = store.get_belief(left_id)
        right = store.get_belief(right_id)
        if left is None or right is None:
            raise ValueError("Conflict beliefs were not found")

        normalized = str(action or "").strip().upper()
        if normalized not in {"KEEP A", "KEEP B", "MERGE", "ARCHIVE BOTH"}:
            raise ValueError("Unsupported conflict action")

        if normalized == "KEEP A":
            store.pin_belief(left.id, reason=reason or "Kept via manual conflict resolution", actor=actor)
            store.tombstone_belief(right.id, reason=reason or f"Rejected in favor of '{left.claim[:120]}'", actor=actor)
            for memory_id in list(getattr(right, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="loser",
                    reason=reason,
                    superseded_by=left.id,
                    tombstone=True,
                )
            for memory_id in list(getattr(left, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="winner",
                    reason=reason,
                )
            return {"action": normalized, "winner_belief_id": left.id, "archived_belief_id": right.id}

        if normalized == "KEEP B":
            store.pin_belief(right.id, reason=reason or "Kept via manual conflict resolution", actor=actor)
            store.tombstone_belief(left.id, reason=reason or f"Rejected in favor of '{right.claim[:120]}'", actor=actor)
            for memory_id in list(getattr(left, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="loser",
                    reason=reason,
                    superseded_by=right.id,
                    tombstone=True,
                )
            for memory_id in list(getattr(right, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="winner",
                    reason=reason,
                )
            return {"action": normalized, "winner_belief_id": right.id, "archived_belief_id": left.id}

        if normalized == "ARCHIVE BOTH":
            store.tombstone_belief(left.id, reason=reason or "Archived via manual conflict resolution", actor=actor)
            store.tombstone_belief(right.id, reason=reason or "Archived via manual conflict resolution", actor=actor)
            for memory_id in list(getattr(left, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="archived",
                    reason=reason,
                    tombstone=True,
                )
            for memory_id in list(getattr(right, "source_memory_ids", []) or []):
                self._annotate_memory_resolution(
                    memory_id,
                    conflict_id=conflict_id,
                    resolution=normalized,
                    role="archived",
                    reason=reason,
                    tombstone=True,
                )
            return {"action": normalized, "archived_belief_ids": [left.id, right.id]}

        merged = str(merged_content or "").strip()
        if not merged:
            raise ValueError("merged_content is required for MERGE")
        winner = left if float(getattr(left, "confidence", 0.0)) >= float(getattr(right, "confidence", 0.0)) else right
        loser = right if winner.id == left.id else left
        corrected = store.correct_belief(
            winner.id,
            merged,
            reason=reason or "Merged via manual conflict resolution",
            actor=actor,
        )
        if corrected is None:
            raise ValueError("Failed to create merged belief")
        _, merged_belief = corrected
        store.merge_beliefs(
            merged_belief.id,
            loser.id,
            reason=reason or "Merged conflicting beliefs",
            actor=actor,
        )
        merged_result = self.add(
            merged,
            user_id=getattr(winner, "user_id", "default"),
            metadata={
                "manual_conflict_resolution": {
                    "conflict_id": conflict_id,
                    "resolution": normalized,
                    "reason": reason or "",
                    "source_belief_ids": [left.id, right.id],
                    "merged_belief_id": merged_belief.id,
                }
            },
            infer=False,
            initial_strength=1.0,
        )
        merged_memory_id = ((merged_result.get("results") or [{}])[0]).get("id")
        for memory_id in list(getattr(left, "source_memory_ids", []) or []):
            self._annotate_memory_resolution(
                memory_id,
                conflict_id=conflict_id,
                resolution=normalized,
                role="merged-source",
                reason=reason,
                superseded_by=merged_belief.id,
                tombstone=True,
            )
        for memory_id in list(getattr(right, "source_memory_ids", []) or []):
            self._annotate_memory_resolution(
                memory_id,
                conflict_id=conflict_id,
                resolution=normalized,
                role="merged-source",
                reason=reason,
                superseded_by=merged_belief.id,
                tombstone=True,
            )
        if merged_memory_id:
            self._annotate_memory_resolution(
                merged_memory_id,
                conflict_id=conflict_id,
                resolution=normalized,
                role="merged-winner",
                reason=reason,
            )
        return {
            "action": normalized,
            "merged_belief_id": merged_belief.id,
            "merged_memory_id": merged_memory_id,
        }

    def _demote_existing(
        self,
        memory: Dict[str, Any],
        reason: str,
        *,
        superseded_by: Optional[str] = None,
        force: bool = False,
    ) -> None:
        if not memory:
            return
        from dhee.memory.quality import is_protected_canonical_memory

        if is_protected_canonical_memory(memory) and not force:
            self.db.log_event(
                memory["id"],
                "DEMOTE_BLOCKED_CANONICAL",
                old_strength=memory.get("strength"),
                new_strength=memory.get("strength"),
                old_layer=memory.get("layer"),
                new_layer=memory.get("layer"),
            )
            return
        old_strength = float(memory.get("strength", 1.0))
        old_layer = memory.get("layer", "sml")
        new_strength = min(old_strength, 0.05)
        metadata = _metadata_dict(memory.get("metadata"))
        metadata["superseded"] = True
        metadata["superseded_reason"] = reason
        metadata["superseded_at"] = datetime.now(timezone.utc).isoformat()
        if superseded_by:
            metadata["superseded_by_id"] = superseded_by

        self.db.update_memory(
            memory["id"],
            {
                "strength": new_strength,
                "layer": "sml",
                "metadata": metadata,
            },
        )

        self._update_vectors_for_memory(memory["id"], metadata)

        self.db.log_event(
            memory["id"],
            "DEMOTE",
            old_strength=old_strength,
            new_strength=new_strength,
            old_layer=old_layer,
            new_layer="sml",
        )

    def _forget_by_query(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = (query or "").strip()
        if not cleaned:
            return {"deleted_count": 0, "deleted_ids": []}

        threshold = max(self.fade_config.conflict_similarity_threshold, 0.85)
        query_embedding = self.embedder.embed(cleaned, memory_action="forget")
        results = self.vector_store.search(query=None, vectors=query_embedding, limit=20, filters=filters)

        deleted_ids: List[str] = []
        candidates: Dict[str, float] = {}
        for result in results:
            if float(result.score) < threshold:
                continue
            memory_id = self._resolve_memory_id(result)
            best = candidates.get(memory_id)
            if best is None or float(result.score) > best:
                candidates[memory_id] = float(result.score)

        for memory_id in candidates:
            memory = self.db.get_memory(memory_id)
            if not memory:
                continue
            self.delete(memory_id)
            deleted_ids.append(memory_id)

        return {"deleted_count": len(deleted_ids), "deleted_ids": deleted_ids}

    def _find_similar(self, embedding: List[float], filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        memory, similarity = self._nearest_memory(embedding, filters)
        if memory and similarity >= self.fade_config.conflict_similarity_threshold:
            return memory
        return None

    def _check_promotion(self, memory_id: str) -> None:
        memory = self.db.get_memory(memory_id)
        if not memory:
            return
        from dhee.memory.quality import memory_quality_from_record

        quality = memory_quality_from_record(memory)
        if quality.layer != "lml" or quality.suppress_from_default_recall:
            return
        if memory and should_promote(
            memory.get("layer", "sml"),
            memory.get("access_count", 0),
            memory.get("strength", 1.0),
            self.fade_config,
        ):
            self.db.update_memory(memory_id, {"layer": "lml"})
            self.db.log_event(memory_id, "PROMOTE", old_layer="sml", new_layer="lml")

    def _is_expired(self, memory: Dict[str, Any]) -> bool:
        expiration = memory.get("expiration_date")
        if not expiration:
            return False
        try:
            exp_date = date.fromisoformat(expiration)
        except Exception:
            return False
        return date.today() > exp_date

    # CategoryMem methods
    def _persist_categories(self) -> None:
        """Persist category state to database."""
        if not self.category_processor:
            return
        categories = self.category_processor.get_all_categories()
        self.db.save_all_categories(categories)

    def get_categories(self) -> List[Dict[str, Any]]:
        """Get all categories."""
        if not self.category_processor:
            return []
        return self.category_processor.get_all_categories()

    def get_category(self, category_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific category by ID."""
        if not self.category_processor:
            return None
        cat = self.category_processor.get_category(category_id)
        return cat.to_dict() if cat else None

    def get_category_summary(self, category_id: str, regenerate: bool = False) -> str:
        """
        Get or generate summary for a category.

        Args:
            category_id: Category ID
            regenerate: Force regenerate even if cached

        Returns:
            Summary text
        """
        if not self.category_processor:
            return ""

        cat = self.category_processor.get_category(category_id)
        if not cat:
            return "Category not found."

        # Return cached if available and not forcing regenerate
        if cat.summary and not regenerate:
            return cat.summary

        # Get memories in this category
        memories = self.db.get_memories_by_category(category_id, limit=20)

        return self.category_processor.generate_summary(category_id, memories)

    def get_all_summaries(self) -> Dict[str, str]:
        """
        Get summaries for all categories with memories.

        Returns category-level summaries with dynamic,
        evolving content based on stored memories.

        Returns:
            Dict mapping category name to summary
        """
        if not self.category_processor:
            return {}

        summaries = {}
        for cat in self.category_processor.categories.values():
            if cat.memory_count > 0:
                if not cat.summary:
                    memories = self.db.get_memories_by_category(cat.id, limit=20)
                    self.category_processor.generate_summary(cat.id, memories)
                summaries[cat.name] = cat.summary or f"{cat.memory_count} memories"

        self._persist_categories()
        return summaries

    def get_category_tree(self) -> List[Dict[str, Any]]:
        """
        Get hierarchical category tree.

        Returns:
            List of root categories with nested children
        """
        if not self.category_processor:
            return []

        def node_to_dict(node) -> Dict[str, Any]:
            return {
                "id": node.category.id,
                "name": node.category.name,
                "description": node.category.description,
                "memory_count": node.category.memory_count,
                "strength": node.category.strength,
                "depth": node.depth,
                "children": [node_to_dict(child) for child in node.children],
            }

        tree_nodes = self.category_processor.get_category_tree()
        return [node_to_dict(node) for node in tree_nodes]

    def apply_category_decay(self) -> Dict[str, Any]:
        """
        Apply decay to categories

        Unused categories weaken and may merge with similar ones.

        Returns:
            Stats about decayed/merged/deleted categories
        """
        if not self.category_processor or not self.category_config.enable_category_decay:
            return {"decayed": 0, "merged": 0, "deleted": 0}

        result = self.category_processor.apply_category_decay(
            decay_rate=self.category_config.category_decay_rate
        )

        self._persist_categories()
        return result

    def get_category_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the category layer.

        Returns:
            Category statistics
        """
        if not self.category_processor:
            return {"enabled": False}

        stats = self.category_processor.get_category_stats()
        stats["enabled"] = True
        stats["config"] = {
            "auto_categorize": self.category_config.auto_categorize,
            "enable_decay": self.category_config.enable_category_decay,
            "boost_weight": self.category_config.category_boost_weight,
        }
        return stats

    def search_by_category(
        self,
        category_id: str,
        limit: int = 50,
        min_strength: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Get memories in a specific category.

        Args:
            category_id: Category ID
            limit: Maximum results
            min_strength: Minimum memory strength

        Returns:
            Dict with results list
        """
        if not self.category_processor:
            return {"results": [], "category": None}

        cat = self.category_processor.get_category(category_id)
        if not cat:
            return {"results": [], "category": None, "error": "Category not found"}

        # Record access
        self.category_processor.access_category(category_id)

        memories = self.db.get_memories_by_category(
            category_id, limit=limit, min_strength=min_strength
        )

        self._persist_categories()

        return {
            "results": memories,
            "category": cat.to_dict(),
            "total": len(memories),
        }

    # =========================================================================
    # Knowledge Graph Methods
    # =========================================================================

    def get_related_memories(
        self,
        memory_id: str,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Get memories related to a given memory via the knowledge graph.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum graph traversal depth

        Returns:
            Dict with related memories and relationship paths
        """
        if not self.knowledge_graph:
            return {"results": [], "graph_enabled": False}

        related = self.knowledge_graph.get_related_memories(
            memory_id=memory_id,
            max_depth=max_depth,
        )

        results = []
        for other_id, depth, path in related:
            memory = self.db.get_memory(other_id)
            if memory:
                results.append({
                    "id": other_id,
                    "memory": memory.get("memory", ""),
                    "depth": depth,
                    "path": [
                        {
                            "type": r.relation_type.value,
                            "entity": r.entity,
                            "weight": r.weight,
                        }
                        for r in path
                    ],
                })

        return {"results": results, "total": len(results)}

    def get_memory_entities(self, memory_id: str) -> Dict[str, Any]:
        """
        Get entities extracted from a specific memory.

        Args:
            memory_id: Memory ID

        Returns:
            Dict with entity information
        """
        if not self.knowledge_graph:
            return {"entities": [], "graph_enabled": False}

        entity_names = self.knowledge_graph.memory_entities.get(memory_id, set())
        entities = []
        for name in entity_names:
            entity = self.knowledge_graph.entities.get(name)
            if entity:
                entities.append(entity.to_dict())

        return {"entities": entities, "total": len(entities)}

    def get_entity_memories(self, entity_name: str) -> Dict[str, Any]:
        """
        Get all memories containing a specific entity.

        Args:
            entity_name: Entity name to search for

        Returns:
            Dict with memories containing the entity
        """
        if not self.knowledge_graph:
            return {"results": [], "graph_enabled": False}

        memory_ids = self.knowledge_graph.get_entity_memories(entity_name)
        results = []
        for memory_id in memory_ids:
            memory = self.db.get_memory(memory_id)
            if memory:
                results.append({
                    "id": memory_id,
                    "memory": memory.get("memory", ""),
                    "strength": memory.get("strength", 1.0),
                    "layer": memory.get("layer", "sml"),
                })

        return {"results": results, "entity": entity_name, "total": len(results)}

    def get_memory_graph(self, memory_id: str) -> Dict[str, Any]:
        """
        Get graph visualization data centered on a memory.

        Args:
            memory_id: Center memory ID

        Returns:
            Dict with nodes and edges for visualization
        """
        if not self.knowledge_graph:
            return {"nodes": [], "edges": [], "graph_enabled": False}

        return self.knowledge_graph.get_memory_graph(memory_id)

    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get knowledge graph statistics.

        Returns:
            Dict with graph statistics
        """
        if not self.knowledge_graph:
            return {"enabled": False}

        stats = self.knowledge_graph.stats()
        stats["enabled"] = True
        return stats

    # =========================================================================
    # Scene Methods
    # =========================================================================

    def _assign_to_scene(
        self,
        memory_id: str,
        content: str,
        embedding: Optional[List[float]],
        user_id: Optional[str],
        timestamp: str,
    ) -> None:
        """Assign a memory to an existing or new scene."""
        if not self.scene_processor or not user_id:
            return

        # Auto-close stale scenes first
        self.scene_processor.auto_close_stale(user_id)

        current_scene = self.db.get_open_scene(user_id)
        memory_row = self.db.get_memory(memory_id) or {}
        namespace = str(memory_row.get("namespace", "default") or "default").strip() or "default"
        if (
            current_scene
            and str(current_scene.get("namespace", "default") or "default").strip() != namespace
        ):
            detection = self.scene_processor.detect_boundary(
                content=content,
                timestamp=timestamp,
                current_scene=None,
                embedding=embedding,
            )
        else:
            detection = self.scene_processor.detect_boundary(
                content=content,
                timestamp=timestamp,
                current_scene=current_scene,
                embedding=embedding,
            )

        if detection.is_new_scene:
            # Close old scene if open
            if current_scene:
                self.scene_processor.close_scene(current_scene["id"], timestamp)

            # Detect topic from content (first 60 chars as fallback)
            topic = content[:60].strip()
            location = detection.detected_location

            self.scene_processor.create_scene(
                first_memory_id=memory_id,
                user_id=user_id,
                timestamp=timestamp,
                topic=topic,
                location=location,
                embedding=embedding,
                namespace=namespace,
            )
        else:
            if current_scene:
                self.scene_processor.add_memory_to_scene(
                    scene_id=current_scene["id"],
                    memory_id=memory_id,
                    embedding=embedding,
                    timestamp=timestamp,
                    namespace=namespace,
                )

    def _update_profiles(
        self,
        memory_id: str,
        content: str,
        metadata: Dict[str, Any],
        user_id: Optional[str],
    ) -> None:
        """Extract and apply profile updates from memory content."""
        if not self.profile_processor or not user_id:
            return
        try:
            from dhee.memory.quality import classify_memory_quality

            quality = classify_memory_quality(
                content,
                metadata,
                None,
                explicit_remember=bool(metadata.get("policy_explicit") or metadata.get("explicit_remember")),
            )
            if quality.memory_class in {"test_fixture", "operational_event", "passive_screen", "evidence_artifact"}:
                return
        except Exception as exc:
            logger.debug("Profile extraction quality gate skipped: %s", exc)

        updates: List[Any] = []
        if hasattr(self.profile_processor, "extract_profile_mentions_from_speakers"):
            try:
                updates.extend(
                    self.profile_processor.extract_profile_mentions_from_speakers(
                        content=content,
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.debug("Speaker-based profile extraction failed: %s", e)

        updates.extend(
            self.profile_processor.extract_profile_mentions(
            content=content,
            metadata=metadata,
            user_id=user_id,
            )
        )

        # Merge duplicate profile updates before applying to reduce churn.
        merged_updates: Dict[Tuple[str, str], Any] = {}
        for update in updates:
            key = (str(update.profile_name or "").strip(), str(update.profile_type or "").strip())
            existing = merged_updates.get(key)
            if existing is None:
                merged_updates[key] = update
                continue
            for fact in list(getattr(update, "new_facts", []) or []):
                if fact not in existing.new_facts:
                    existing.new_facts.append(fact)
            for pref in list(getattr(update, "new_preferences", []) or []):
                if pref not in existing.new_preferences:
                    existing.new_preferences.append(pref)
            for rel in list(getattr(update, "new_relationships", []) or []):
                if rel not in existing.new_relationships:
                    existing.new_relationships.append(rel)

        for update in merged_updates.values():
            self.profile_processor.apply_update(
                profile_update=update,
                memory_id=memory_id,
                user_id=user_id,
            )

    def get_scene(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific scene by ID."""
        return self.db.get_scene(scene_id)

    def get_scenes(
        self,
        user_id: Optional[str] = None,
        topic: Optional[str] = None,
        start_after: Optional[str] = None,
        start_before: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List scenes chronologically."""
        return self.db.get_scenes(
            user_id=user_id,
            topic=topic,
            start_after=start_after,
            start_before=start_before,
            namespace=namespace,
            limit=limit,
        )

    def search_scenes(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Semantic search over scene summaries."""
        if not self.scene_processor:
            return []
        return self.scene_processor.search_scenes(query=query, user_id=user_id, limit=limit)

    def get_scene_timeline(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get scenes in chronological order."""
        if not self.scene_processor:
            return []
        return self.scene_processor.get_scene_timeline(user_id=user_id, limit=limit)

    def get_scene_memories(self, scene_id: str) -> List[Dict[str, Any]]:
        """Get all memories in a scene."""
        return self.db.get_scene_memories(scene_id)

    # =========================================================================
    # Profile Methods
    # =========================================================================

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get a character profile by ID."""
        return self.db.get_profile(profile_id)

    def get_all_profiles(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all profiles for a user."""
        return self.db.get_all_profiles(user_id=user_id)

    def get_self_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the self-profile for a user."""
        return self.db.get_profile_by_name("self", user_id=user_id)

    def search_profiles(self, query: str, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search profiles by name or description."""
        if not self.profile_processor:
            return []
        return self.profile_processor.search_profiles(query=query, user_id=user_id, limit=limit)

    def update_profile(self, profile_id: str, updates: Dict[str, Any]) -> bool:
        """Update a profile."""
        return self.db.update_profile(profile_id, updates)

    def get_profile_memories(self, profile_id: str) -> List[Dict[str, Any]]:
        """Get memories linked to a profile."""
        return self.db.get_profile_memories(profile_id)

    # =========================================================================
    # Dashboard / Visualization Methods
    # =========================================================================

    def get_constellation_data(self, user_id: Optional[str] = None, limit: int = 200) -> Dict[str, Any]:
        """Get graph nodes + edges for the constellation force layout."""
        return self.db.get_constellation_data(user_id=user_id, limit=limit)

    def get_decay_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decay history for dashboard sparkline."""
        return self.db.get_decay_log_entries(limit=limit)


# Historical alias kept for legacy ``engram.memory.main.Memory`` imports.
Memory = FullMemory
