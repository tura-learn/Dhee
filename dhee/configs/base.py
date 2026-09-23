import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from dhee.configs.active import ActiveMemoryConfig
from dhee.provider_defaults import (
    DEFAULT_COLLECTION,
    DEFAULT_NVIDIA_EMBEDDER_MODEL,
    DEFAULT_NVIDIA_LLM_MODEL,
    DEFAULT_NVIDIA_EMBEDDING_DIMS,
)


def resolve_dhee_data_dir(path: str | os.PathLike[str]) -> str:
    """Return an absolute Dhee data directory with user markers expanded."""
    return os.path.abspath(os.path.expanduser(os.fspath(path)))


def _dhee_data_dir() -> str:
    """Resolve data directory: DHEE_DATA_DIR > ~/.dhee."""
    env = os.environ.get("DHEE_DATA_DIR")
    if env:
        return resolve_dhee_data_dir(env)
    return resolve_dhee_data_dir(Path.home() / ".dhee")


_VALID_VECTOR_PROVIDERS = {"memory", "sqlite_vec", "zvec"}
_VALID_LLM_PROVIDERS = {"anthropic", "gemini", "openai", "nvidia", "ollama", "mock", "dhee"}
_VALID_EMBEDDER_PROVIDERS = {"gemini", "openai", "nvidia", "ollama", "simple", "qwen"}


class VectorStoreConfig(BaseModel):
    provider: str = Field(default="zvec")
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "path": os.path.join(_dhee_data_dir(), "zvec"),
            "collection_name": DEFAULT_COLLECTION,
        }
    )

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in _VALID_VECTOR_PROVIDERS:
            raise ValueError(f"Unknown vector store provider '{v}'. Valid: {sorted(_VALID_VECTOR_PROVIDERS)}")
        return v


class LLMConfig(BaseModel):
    provider: str = Field(default="nvidia")
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": DEFAULT_NVIDIA_LLM_MODEL,
            "temperature": 0.2,
            "max_tokens": 4096,
        }
    )

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in _VALID_LLM_PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{v}'. Valid: {sorted(_VALID_LLM_PROVIDERS)}")
        return v


class EmbedderConfig(BaseModel):
    provider: str = Field(default="nvidia")
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "model": DEFAULT_NVIDIA_EMBEDDER_MODEL,
            "embedding_dims": DEFAULT_NVIDIA_EMBEDDING_DIMS,
        }
    )

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in _VALID_EMBEDDER_PROVIDERS:
            raise ValueError(f"Unknown embedder provider '{v}'. Valid: {sorted(_VALID_EMBEDDER_PROVIDERS)}")
        return v


class GraphStoreConfig(BaseModel):
    provider: Optional[str] = Field(default=None)
    config: Optional[Dict[str, Any]] = Field(default=None)


class KnowledgeGraphConfig(BaseModel):
    """Configuration for knowledge graph entity extraction and linking."""
    enable_graph: bool = True  # Enable knowledge graph
    use_llm_extraction: bool = False  # Use LLM for entity extraction (slower but more accurate)
    auto_link_entities: bool = True  # Automatically link memories by shared entities
    max_traversal_depth: int = 2  # Maximum depth for graph traversal in search
    graph_boost_weight: float = 0.1  # Boost for graph-related memories in search


class EchoMemConfig(BaseModel):
    """Configuration for EchoMem multi-modal encoding."""
    enable_echo: bool = True
    auto_depth: bool = True  # Auto-detect echo depth based on content importance
    default_depth: str = "medium"  # shallow, medium, deep
    reecho_on_access: bool = False  # Re-process on retrieval (expensive but strengthening)
    reecho_threshold: int = 3  # Re-echo after N accesses
    # Strength multipliers for each depth
    shallow_multiplier: float = 1.0
    medium_multiplier: float = 1.3
    deep_multiplier: float = 1.6
    # Use question_form embedding for primary vector (better query matching)
    use_question_embedding: bool = True
    # Echo-augmented embedding: compose primary text from content + echo data
    # (question_form, keywords, first paraphrase) for richer retrieval vectors
    use_echo_augmented_embedding: bool = True

    @field_validator("default_depth")
    @classmethod
    def _valid_depth(cls, v: str) -> str:
        allowed = {"shallow", "medium", "deep"}
        v = str(v).strip().lower()
        if v not in allowed:
            return "medium"
        return v

    @field_validator("shallow_multiplier", "medium_multiplier", "deep_multiplier")
    @classmethod
    def _positive_multiplier(cls, v: float) -> float:
        return max(0.1, float(v))


class CategoryMemConfig(BaseModel):
    """
    Configuration for CategoryMem hierarchical category layer.

    Unlike traditional static approaches, CategoryMem provides:
    - Dynamic auto-discovered categories
    - Hierarchical structure with parent/child relationships
    - Category summaries that evolve with memories
    - Category decay (unused categories merge/fade)
    - Category-aware retrieval boosting
    """
    enable_categories: bool = True  # Enable category layer
    auto_categorize: bool = True  # Automatically categorize new memories
    use_llm_categorization: bool = True  # Use LLM for ambiguous categorization

    # Category decay (bio-inspired, like engram)
    enable_category_decay: bool = True
    category_decay_rate: float = 0.05  # Decay rate per cycle
    merge_weak_categories: bool = True  # Merge weak categories automatically
    weak_category_threshold: float = 0.3  # Strength below this triggers merge consideration

    # Summary generation
    auto_generate_summaries: bool = True  # Generate summaries for categories
    summary_update_threshold: int = 5  # Regenerate summary after N new memories

    # Retrieval boosting
    category_boost_weight: float = 0.15  # Boost for matching category in search
    cross_category_boost: float = 0.05  # Boost for related categories

    # Hierarchy
    max_category_depth: int = 3  # Maximum nesting depth
    auto_create_subcategories: bool = True  # Allow dynamic subcategory creation

    @field_validator(
        "category_decay_rate", "weak_category_threshold",
        "category_boost_weight", "cross_category_boost",
    )
    @classmethod
    def _clamp_unit_float(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("max_category_depth")
    @classmethod
    def _clamp_depth(cls, v: int) -> int:
        return min(10, max(1, int(v)))


class SceneConfig(BaseModel):
    """Configuration for episodic scene grouping."""
    enable_scenes: bool = True
    scene_time_gap_minutes: int = 30       # gap > this = new scene
    scene_topic_threshold: float = 0.55    # cosine sim below this = topic shift
    auto_close_inactive_minutes: int = 120
    max_scene_memories: int = 50
    use_llm_summarization: bool = True
    summary_regenerate_threshold: int = 5


class ProfileConfig(BaseModel):
    """Configuration for character profile tracking."""
    enable_profiles: bool = True
    auto_detect_profiles: bool = True
    use_llm_extraction: bool = True
    narrative_regenerate_threshold: int = 10
    self_profile_auto_create: bool = True
    max_facts_per_profile: int = 100


class OrchestrationConfig(BaseModel):
    """Configuration for event-first retrieval + answer orchestration."""
    enable_orchestrated_search: bool = True
    enable_episodic_index: bool = True
    enable_hierarchical_retrieval: bool = True
    orchestrator_model: str = "meta/llama-3.1-8b-instruct"
    reflection_max_hops: int = 1
    map_max_candidates: int = 8
    map_candidate_max_chars: int = 1200
    map_reduce_coverage_threshold: float = 0.6
    # Strict query-time guardrail: at most N orchestration LLM calls per query.
    max_query_llm_calls: int = 2
    # Intent-specific minimum coverage thresholds for map/reduce gating.
    intent_coverage_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "count": 0.65,
            "money_sum": 0.6,
            "duration": 0.6,
            "latest": 0.55,
            "set_members": 0.65,
            "freeform": 0.6,
        }
    )
    # Cache deterministic reducer outputs to avoid repeat map calls.
    reducer_cache_ttl_seconds: int = 900
    reducer_cache_max_entries: int = 2048
    context_cap: int = 20
    search_cap: int = 30

    @field_validator("reflection_max_hops")
    @classmethod
    def _valid_reflection_hops(cls, v: int) -> int:
        return max(0, min(3, int(v)))

    @field_validator("map_max_candidates", "context_cap", "search_cap", "reducer_cache_ttl_seconds", "reducer_cache_max_entries")
    @classmethod
    def _valid_positive_int(cls, v: int) -> int:
        return max(1, int(v))

    @field_validator("max_query_llm_calls")
    @classmethod
    def _valid_query_budget_calls(cls, v: int) -> int:
        return max(0, min(4, int(v)))

    @field_validator("map_candidate_max_chars")
    @classmethod
    def _valid_map_chars(cls, v: int) -> int:
        return max(200, int(v))

    @field_validator("map_reduce_coverage_threshold")
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("intent_coverage_thresholds")
    @classmethod
    def _valid_intent_thresholds(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not isinstance(v, dict):
            return {}
        cleaned: Dict[str, float] = {}
        for key, value in v.items():
            intent_key = str(key or "").strip().lower()
            if not intent_key:
                continue
            cleaned[intent_key] = max(0.0, min(1.0, float(value)))
        return cleaned


class CostGuardrailConfig(BaseModel):
    """Configuration for write/query unit-economics guardrails."""
    enable_cost_counters: bool = True
    strict_write_path_cap: bool = True
    auto_disable_on_violation: bool = True
    baseline_write_llm_calls_per_memory: float = 0.0
    baseline_write_tokens_per_memory: float = 0.0

    @field_validator("baseline_write_llm_calls_per_memory", "baseline_write_tokens_per_memory")
    @classmethod
    def _non_negative_float(cls, v: float) -> float:
        return max(0.0, float(v))


class HandoffConfig(BaseModel):
    """Configuration for cross-agent session handoff."""
    enable_handoff: bool = True
    auto_enrich: bool = True          # LLM-enrich digests with linked memories
    max_sessions_per_user: int = 100  # retain last N sessions
    handoff_backend: str = "hosted"   # hosted|local
    strict_handoff_auth: bool = True
    allow_auto_trusted_bootstrap: bool = False
    auto_session_bus: bool = True
    auto_checkpoint_events: List[str] = Field(
        default_factory=lambda: ["tool_complete", "agent_pause", "agent_end"]
    )
    lane_inactivity_minutes: int = 240
    max_lanes_per_user: int = 50
    max_checkpoints_per_lane: int = 200
    resume_statuses: List[str] = Field(default_factory=lambda: ["active", "paused"])
    auto_trusted_agents: List[str] = Field(
        default_factory=lambda: [
            "pm",
            "design",
            "frontend",
            "backend",
            "claude-code",
            "codex",
            "chatgpt",
        ]
    )


class ScopeConfig(BaseModel):
    """Configuration for scope-aware sharing weights."""
    agent_weight: float = 1.0
    connector_weight: float = 0.97
    category_weight: float = 0.94
    global_weight: float = 0.92


class DistillationConfig(BaseModel):
    """Configuration for CLS Distillation Memory (hippocampus-neocortex consolidation)."""

    # Gap 1: Episodic/Semantic separation
    enable_memory_types: bool = True
    default_memory_type: str = "semantic"

    # Gap 2: Replay distillation
    enable_distillation: bool = True
    distillation_batch_size: int = 20
    distillation_min_episodes: int = 5
    distillation_scene_grouping: bool = True
    distillation_time_window_hours: int = 24
    max_semantic_per_batch: int = 5

    # Gap 3: Advanced forgetting
    enable_interference_pruning: bool = True
    enable_redundancy_collapse: bool = True
    enable_homeostasis: bool = True
    homeostasis_budget_per_namespace: int = 5000
    homeostasis_pressure_factor: float = 0.1
    redundancy_collapse_threshold: float = 0.85

    # Gap 4: Multi-trace strength
    enable_multi_trace: bool = True
    s_fast_weight: float = 0.2
    s_mid_weight: float = 0.3
    s_slow_weight: float = 0.5
    s_fast_decay_rate: float = 0.20
    s_mid_decay_rate: float = 0.05
    s_slow_decay_rate: float = 0.005
    cascade_fast_to_mid: float = 0.1
    cascade_mid_to_slow: float = 0.05

    # Gap 5: Intent routing
    enable_intent_routing: bool = True
    episodic_boost: float = 0.15
    semantic_boost: float = 0.15
    intersection_boost: float = 0.1

    @field_validator("default_memory_type")
    @classmethod
    def _valid_memory_type(cls, v: str) -> str:
        allowed = {"episodic", "semantic"}
        v = str(v).strip().lower()
        if v not in allowed:
            return "semantic"
        return v

    @field_validator(
        "homeostasis_pressure_factor", "redundancy_collapse_threshold",
        "s_fast_weight", "s_mid_weight", "s_slow_weight",
        "s_fast_decay_rate", "s_mid_decay_rate", "s_slow_decay_rate",
        "cascade_fast_to_mid", "cascade_mid_to_slow",
        "episodic_boost", "semantic_boost", "intersection_boost",
    )
    @classmethod
    def _clamp_unit_float(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("homeostasis_budget_per_namespace", "distillation_batch_size",
                     "distillation_min_episodes", "distillation_time_window_hours",
                     "max_semantic_per_batch")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        return max(1, int(v))


class MetamemoryInlineConfig(BaseModel):
    """Inline config for metamemory features in core (no engram-metamemory dependency)."""
    enable_confidence: bool = True
    confidence_in_search_results: bool = True
    auto_log_gaps: bool = True


class ProspectiveInlineConfig(BaseModel):
    """Inline config for prospective memory features in core."""
    enable_prospective: bool = True


class ProceduralInlineConfig(BaseModel):
    """Inline config for procedural memory features in core."""
    enable_procedural: bool = True
    automaticity_boost_in_search: bool = True
    automaticity_boost_in_search_weight: float = 0.20


class ReconsolidationInlineConfig(BaseModel):
    """Inline config for reconsolidation features in core."""
    enable_auto_reconsolidation: bool = True


class FailureInlineConfig(BaseModel):
    """Inline config for failure learning features in core."""
    enable_failure_warnings: bool = True


class WorkingMemoryInlineConfig(BaseModel):
    """Inline config for working memory features in core."""
    enable_working_memory: bool = True


class SalienceInlineConfig(BaseModel):
    """Inline config for salience tagging in core."""
    enable_salience: bool = True
    use_llm_salience: bool = False
    salience_boost_weight: float = 0.15
    salience_decay_modifier: bool = True


class CausalInlineConfig(BaseModel):
    """Inline config for causal reasoning in core."""
    enable_causal: bool = True
    auto_detect_causal_language: bool = True


class ResolverBoostConfig(BaseModel):
    """Wire the deterministic ContextResolver into the retrieval read path.

    The ContextResolver already runs on write to supersede single-valued facts
    (subject|predicate in _SINGLE_VALUED_PREDICATES get valid_until set when
    a newer fact arrives). On read, resolver.resolve(query) returns the
    memory_ids that ground the current, non-superseded answer. This config
    controls whether those memory_ids get a composite-score boost in search
    results, making fact-aware supersede observable at rank 1.
    """
    enable_resolver_boost: bool = True
    resolver_boost_weight: float = 0.7  # applied as combined *= (1 + weight)
    emit_fact_status: bool = True       # include fact counts + resolver fields in results


class DheeModelConfig(BaseModel):
    """Configuration for the DheeModel (fine-tuned Qwen3.5 family)."""
    model_path: str = "auto"               # auto-download or local path
    base_model: str = "Qwen3.5-2B"
    quantization: str = "Q4_K_M"
    embedding_model: str = "Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen3-Reranker-0.6B"
    n_ctx: int = 4096
    n_threads: int = 4
    temperature: float = 0.1
    max_tokens: int = 2048


class CognitionConfig(BaseModel):
    """Configuration for the cognitive decomposition engine."""
    enable_cognition: bool = True
    max_depth: int = 3
    max_sub_questions: int = 10
    store_solutions: bool = True           # learn from cognitive results


class ProspectiveSceneConfig(BaseModel):
    """Configuration for prospective scene prediction."""
    enable_prospective_scenes: bool = True
    default_trigger_window_hours: int = 24  # surface N hours before event
    max_predicted_scenes: int = 50          # per user
    auto_link_past_scenes: bool = True      # find similar past events
    surface_on_context_load: bool = True    # check on session start


class DecisionConfig(BaseModel):
    """Typed decisions: the closed choices Dhee makes without writing prose.

    Several steps in the write path are not generation at all — is this fact
    about the person, which label does it take, does it restate something
    already stored — and asking an LLM for them means parsing a verdict out of
    free text. A decision model (TypeSafe's Jev) answers each as a calibrated
    probability in well under a second. Off by default: with no decider the
    pipeline behaves exactly as before, and every decider failure falls back
    to that behaviour too.
    """
    enabled: bool = False
    #: ``openrouter`` (``/api/alpha/decisions``) or ``typesafe`` (``/v1/systemone``).
    provider: str = "openrouter"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    model: Optional[str] = None
    timeout_seconds: float = 6.0
    #: Controlled vocabulary for facts about the subject: predicate -> meaning.
    #: A meaning is a sentence, or an object such as ``{"what": ..., "not_for":
    #: ...}`` when a boundary between two predicates needs stating. Empty means
    #: predicates are left as the extractor wrote them.
    fact_vocabulary: Dict[str, Any] = Field(default_factory=dict)
    #: Who the facts are about. When set, facts judged not to be about them
    #: (general knowledge, the assistant, bookkeeping) are not stored.
    fact_subjects: List[str] = Field(default_factory=list)
    #: The subject name stored for facts about them, whatever the extractor wrote.
    canonical_subject: str = "user"
    about_floor: float = 0.5
    label_floor: float = 0.5
    same_floor: float = 0.7


class EngramExtractionConfig(BaseModel):
    """Configuration for structured engram extraction."""
    enable_extraction: bool = True
    use_llm_extraction: bool = True         # use LLM for extraction (vs rule-based)
    extract_prospective: bool = True        # detect future plans -> ProspectiveScene
    extract_context_anchors: bool = True    # era/place/time/activity
    extract_scene_snapshots: bool = True    # visual scene reconstruction


class SkillConfig(BaseModel):
    """Configuration for the skill-learning agent memory system."""
    enable_skills: bool = True
    skill_collection_name: str = "dhee_skills"
    min_confidence_for_auto_apply: float = 0.3
    enable_mining: bool = True
    min_trajectory_steps: int = 3
    mutation_rate: float = 0.05
    # Structural intelligence
    enable_structural: bool = True
    use_llm_decomposition: bool = True
    structural_similarity_threshold: float = 0.4
    auto_decompose_on_mine: bool = True
    auto_decompose_on_import: bool = True

    @field_validator("min_confidence_for_auto_apply", "mutation_rate",
                     "structural_similarity_threshold")
    @classmethod
    def _clamp_unit_float(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("min_trajectory_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        return max(1, int(v))


class TaskConfig(BaseModel):
    """Configuration for tasks as first-class Engram memories."""
    enable_tasks: bool = True
    task_namespace: str = "tasks"
    default_priority: str = "normal"
    active_task_decay_rate: float = 0.0       # active tasks don't decay
    completed_task_decay_rate: float = 0.30   # done tasks decay 2x faster
    archived_task_decay_rate: float = 0.15    # normal rate
    task_category_prefix: str = "tasks"
    auto_archive_completed_days: int = 7

    @field_validator("default_priority")
    @classmethod
    def _valid_priority(cls, v: str) -> str:
        allowed = {"low", "normal", "high", "urgent"}
        v = str(v).strip().lower()
        if v not in allowed:
            return "normal"
        return v


class RerankConfig(BaseModel):
    """Configuration for neural reranking (cross-encoder second stage)."""
    enable_rerank: bool = False
    provider: str = "nvidia"
    model: str = "nvidia/llama-nemotron-rerank-vl-1b-v2"
    api_key_env: str = "NVIDIA_API_KEY"  # Env var name for API key
    top_n: int = 0  # Number of results to return after reranking (0 = return all, re-sorted)
    config: Dict[str, Any] = Field(default_factory=dict)


class EnrichmentConfig(BaseModel):
    """Configuration for unified enrichment (single LLM call for echo+category+entities+profiles)."""
    enable_unified: bool = False        # Off by default for backward compat
    fallback_to_individual: bool = True  # On parse failure, fall back to individual calls
    include_entities: bool = True        # Include entity extraction in unified call
    include_profiles: bool = True        # Include profile extraction in unified call
    max_batch_size: int = 10             # Max memories per unified batch call
    # Deferred enrichment: store with 0 LLM calls, enrich later in batch
    defer_enrichment: bool = False       # When True: 0 LLM calls at ingestion
    context_window_turns: int = 10       # Store last N conversation turns with each memory
    enrich_on_access: bool = False       # Auto-enrich pending memories when retrieved in search

    @field_validator("max_batch_size")
    @classmethod
    def _clamp_batch_size(cls, v: int) -> int:
        return min(50, max(1, int(v)))


class BatchConfig(BaseModel):
    """Configuration for batch memory operations."""
    enable_batch: bool = False    # off by default
    max_batch_size: int = 20
    batch_echo: bool = True
    batch_embed: bool = True
    batch_category: bool = True

    @field_validator("max_batch_size")
    @classmethod
    def _clamp_batch_size(cls, v: int) -> int:
        return min(100, max(1, int(v)))


class ParallelConfig(BaseModel):
    """Configuration for parallel I/O execution (ThreadPoolExecutor)."""
    enable_parallel: bool = False   # off by default
    max_workers: int = 4
    parallel_add: bool = True       # echo + category in parallel during add()
    parallel_reecho: bool = True    # parallel re-echo during search()
    parallel_decay: bool = True     # parallel interference + redundancy during apply_decay()

    @field_validator("max_workers")
    @classmethod
    def _clamp_workers(cls, v: int) -> int:
        return min(32, max(1, int(v)))


class FadeMemConfig(BaseModel):
    enable_forgetting: bool = True
    sml_decay_rate: float = 0.15
    lml_decay_rate: float = 0.02
    access_dampening_factor: float = 0.5
    promotion_access_threshold: int = 3
    promotion_strength_threshold: float = 0.7
    forgetting_threshold: float = 0.1
    access_strength_boost: float = 0.02
    conflict_similarity_threshold: float = 0.85
    fusion_similarity_threshold: float = 0.90
    enable_fusion: bool = True
    use_tombstone_deletion: bool = True

    @field_validator(
        "sml_decay_rate", "lml_decay_rate", "access_dampening_factor",
        "promotion_strength_threshold", "forgetting_threshold",
        "access_strength_boost", "conflict_similarity_threshold",
        "fusion_similarity_threshold",
    )
    @classmethod
    def _clamp_unit_float(cls, v: float) -> float:
        return min(1.0, max(0.0, float(v)))

    @field_validator("promotion_access_threshold")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        return max(1, int(v))


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedder: EmbedderConfig = Field(default_factory=EmbedderConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    history_db_path: str = Field(
        default_factory=lambda: os.path.join(_dhee_data_dir(), "history.db")
    )
    collection_name: str = DEFAULT_COLLECTION
    embedding_model_dims: int = DEFAULT_NVIDIA_EMBEDDING_DIMS
    version: str = "v1.4"  # Updated for CLS Distillation Memory
    custom_fact_extraction_prompt: Optional[str] = None
    custom_conflict_prompt: Optional[str] = None
    custom_fusion_prompt: Optional[str] = None
    custom_echo_prompt: Optional[str] = None
    custom_category_prompt: Optional[str] = None
    fade: FadeMemConfig = Field(default_factory=FadeMemConfig)
    echo: EchoMemConfig = Field(default_factory=EchoMemConfig)
    category: CategoryMemConfig = Field(default_factory=CategoryMemConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
    scene: SceneConfig = Field(default_factory=SceneConfig)
    profile: ProfileConfig = Field(default_factory=ProfileConfig)
    handoff: HandoffConfig = Field(default_factory=HandoffConfig)
    active: ActiveMemoryConfig = Field(default_factory=ActiveMemoryConfig)
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    cost_guardrail: CostGuardrailConfig = Field(default_factory=CostGuardrailConfig)
    skill: SkillConfig = Field(default_factory=SkillConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    metamemory: MetamemoryInlineConfig = Field(default_factory=MetamemoryInlineConfig)
    prospective: ProspectiveInlineConfig = Field(default_factory=ProspectiveInlineConfig)
    procedural: ProceduralInlineConfig = Field(default_factory=ProceduralInlineConfig)
    reconsolidation: ReconsolidationInlineConfig = Field(default_factory=ReconsolidationInlineConfig)
    failure: FailureInlineConfig = Field(default_factory=FailureInlineConfig)
    working_memory: WorkingMemoryInlineConfig = Field(default_factory=WorkingMemoryInlineConfig)
    salience: SalienceInlineConfig = Field(default_factory=SalienceInlineConfig)
    causal: CausalInlineConfig = Field(default_factory=CausalInlineConfig)
    resolver_boost: ResolverBoostConfig = Field(default_factory=ResolverBoostConfig)
    # Dhee: Cognition as a Service
    dhee_model: DheeModelConfig = Field(default_factory=DheeModelConfig)
    cognition: CognitionConfig = Field(default_factory=CognitionConfig)
    prospective_scene: ProspectiveSceneConfig = Field(default_factory=ProspectiveSceneConfig)
    engram_extraction: EngramExtractionConfig = Field(default_factory=EngramExtractionConfig)
    decisions: DecisionConfig = Field(default_factory=DecisionConfig)

    @field_validator("embedding_model_dims")
    @classmethod
    def _valid_dims(cls, v: int) -> int:
        v = int(v)
        if v < 1 or v > 65536:
            raise ValueError(f"embedding_model_dims must be 1-65536, got {v}")
        return v

    # ---- Preset factory methods ----

    @classmethod
    def minimal(cls) -> "MemoryConfig":
        """Zero-config: hash embedder, in-memory vector store, basic decay. No API key."""
        from dhee.configs.presets import minimal_config
        return minimal_config()

    @classmethod
    def smart(cls) -> "MemoryConfig":
        """Auto-detect best available provider + echo + categories."""
        from dhee.configs.presets import smart_config
        return smart_config()

    @classmethod
    def full(cls) -> "MemoryConfig":
        """Everything: scenes, profiles, graph, tasks."""
        from dhee.configs.presets import full_config
        return full_config()
