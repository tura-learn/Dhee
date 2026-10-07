"""CoreMemory — lightweight memory: add/search/delete with decay. No LLM required.

This is the zero-config, zero-API-key entry point. Uses hash-based embeddings
and in-memory vector store by default. Supports content-hash deduplication
and query embedding cache.

Dependencies: SQLiteManager, Embedder, VectorStore, dhee_accel (for cosine sim).
NO LLM, NO echo, NO categories, NO scenes, NO profiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dhee.configs.base import MemoryConfig
from dhee.core.decay import calculate_decayed_strength, decay_since, should_forget, should_promote
from dhee.core.retrieval import composite_score
from dhee.core.traces import (
    boost_fast_trace,
    compute_effective_strength,
    initialize_traces,
)
from dhee.db.sqlite import SQLiteManager
from dhee.memory.quality import (
    apply_memory_quality_contract,
    enforce_quality_layer,
    enforce_quality_strength,
    memory_quality_from_record,
)
from dhee.skills.hashing import content_hash as _content_hash
from dhee.utils.factory import EmbedderFactory, VectorStoreFactory
from dhee.utils.math import cosine_similarity_batch

logger = logging.getLogger(__name__)


class CoreMemory:
    """Lightweight memory: add/search/delete with decay. No LLM required.

    Usage:
        m = CoreMemory()  # zero-config, no API key
        m.add("I like Python")
        results = m.search("programming preferences")
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        preset: Optional[str] = None,
    ):
        if config is None and preset is None:
            config = MemoryConfig.minimal()
        elif preset:
            config = getattr(MemoryConfig, preset)()
        self.config = config

        # Ensure vector store config has dims/collection
        self.config.vector_store.config.setdefault("collection_name", self.config.collection_name)
        self.config.vector_store.config.setdefault("embedding_model_dims", self.config.embedding_model_dims)

        self.db = SQLiteManager(self.config.history_db_path)
        self.embedder = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.fade_config = self.config.fade
        self.distillation_config = getattr(self.config, "distillation", None)

        # Query embedding LRU cache
        self._query_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._query_cache_max = 128

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, "vector_store") and self.vector_store is not None:
            self.vector_store.close()
        if hasattr(self, "db") and self.db is not None:
            self.db.close()

    def __repr__(self) -> str:
        return f"CoreMemory(db={self.db!r})"

    # ---- Core API ----

    def add(
        self,
        content: str,
        user_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
        categories: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        source_app: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add a memory. Simple: content in, result out.

        Returns dict with 'results' list containing the stored memory info.
        Automatically deduplicates by content hash.
        """
        content = str(content).strip()
        if not content:
            return {"results": []}

        user_id = user_id or "default"
        metadata = dict(metadata or {})
        if source_app and not metadata.get("source_app"):
            metadata["source_app"] = source_app
        categories = list(categories or [])
        metadata, quality = apply_memory_quality_contract(
            content,
            metadata,
            categories,
            explicit_remember=bool(metadata.get("explicit_remember") or metadata.get("policy_explicit")),
        )

        # Content-hash dedup
        ch = _content_hash(content)
        existing = self.db.get_memory_by_content_hash(ch, user_id)
        if existing:
            # Re-encountering = spaced repetition = stronger
            self.db.increment_access(existing["id"])
            # Boost fast trace if multi-trace is enabled
            if self.distillation_config and self.distillation_config.enable_multi_trace:
                s_fast = existing.get("s_fast") or 0.0
                boosted = boost_fast_trace(s_fast, self.fade_config.access_strength_boost)
                self.db.update_memory(existing["id"], {"s_fast": boosted})
            return {
                "results": [{
                    "id": existing["id"],
                    "memory": existing.get("memory", ""),
                    "event": "DEDUPLICATED",
                    "layer": existing.get("layer", "sml"),
                    "strength": existing.get("strength", 1.0),
                }]
            }

        # Embed
        embedding = self.embedder.embed(content, memory_action="add")

        # Initialize multi-trace strength
        initial_strength = enforce_quality_strength(1.0, quality)
        s_fast_val = s_mid_val = s_slow_val = None
        if self.distillation_config and self.distillation_config.enable_multi_trace:
            s_fast_val, s_mid_val, s_slow_val = initialize_traces(initial_strength, is_new=True)

        now = datetime.now(timezone.utc).isoformat()
        memory_id = str(uuid.uuid4())
        layer = enforce_quality_layer("sml", quality)
        namespace = str(quality.namespace or metadata.get("namespace", "default") or "default").strip() or "default"
        memory_type = str(quality.memory_type or metadata.get("memory_type") or "semantic").strip() or "semantic"

        memory_data = {
            "id": memory_id,
            "memory": content,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata,
            "categories": categories,
            "created_at": now,
            "updated_at": now,
            "layer": layer,
            "strength": initial_strength,
            "access_count": 0,
            "last_accessed": now,
            "embedding": embedding,
            "confidentiality_scope": metadata.get("confidentiality_scope", "work"),
            "source_type": "mcp",
            "source_app": source_app,
            "decay_lambda": quality.decay_lambda if quality.decay_lambda is not None else self.fade_config.sml_decay_rate,
            "status": "active",
            "importance": metadata.get("importance", 0.5),
            "sensitivity": metadata.get("sensitivity", "normal"),
            "namespace": namespace,
            "memory_type": memory_type,
            "s_fast": s_fast_val,
            "s_mid": s_mid_val,
            "s_slow": s_slow_val,
            "content_hash": ch,
        }

        # Store in DB
        self.db.add_memory(memory_data)

        # Store in vector index
        payload = {
            "memory_id": memory_id,
            "user_id": user_id,
            "memory": content,
            "namespace": namespace,
            "memory_type": memory_type,
            "layer": layer,
            "metadata": metadata,
        }
        if agent_id:
            payload["agent_id"] = agent_id
        try:
            self.vector_store.insert(
                vectors=[embedding],
                payloads=[payload],
                ids=[memory_id],
            )
        except Exception as e:
            logger.warning("Vector insert failed: %s", e)

        return {
            "results": [{
                "id": memory_id,
                "memory": content,
                "event": "ADD",
                "layer": layer,
                "strength": initial_strength,
                "categories": categories,
                "namespace": namespace,
                "memory_type": memory_type,
                "memory_class": quality.memory_class,
            }]
        }

    def search(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 10,
        agent_id: Optional[str] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Search memories. Returns ranked results with scores."""
        query = str(query).strip()
        if not query:
            return {"results": []}

        # Cached embed
        embedding = self._cached_embed(query)

        # Vector search
        filters = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id
        try:
            vector_results = self.vector_store.search(
                query=None,
                vectors=embedding,
                limit=limit * 3,  # oversample for filtering
                filters=filters,
            )
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            vector_results = []

        if not vector_results:
            return {"results": []}

        # Fetch full memory data and score
        results = []
        for vr in vector_results:
            # Handle both dict and MemoryResult objects
            if hasattr(vr, "id"):
                memory_id = vr.id or (vr.payload or {}).get("memory_id")
                similarity = vr.score
            else:
                memory_id = vr.get("id") or (vr.get("payload", {}) or {}).get("memory_id")
                similarity = vr.get("score", 0.0)
            if not memory_id:
                continue
            mem = self.db.get_memory(memory_id)
            if not mem:
                continue
            if mem.get("tombstone"):
                continue
            strength = float(mem.get("strength", 1.0))
            score = composite_score(similarity, strength)

            # Category filter
            if categories:
                mem_cats = mem.get("categories", [])
                if isinstance(mem_cats, str):
                    try:
                        mem_cats = json.loads(mem_cats)
                    except (json.JSONDecodeError, TypeError):
                        mem_cats = []
                if not any(c in mem_cats for c in categories):
                    continue

            results.append({
                "id": mem["id"],
                "memory": mem.get("memory", ""),
                "score": round(score, 4),
                "composite_score": round(score, 4),
                "similarity": round(similarity, 4),
                "strength": round(strength, 4),
                "layer": mem.get("layer", "sml"),
                "categories": mem.get("categories", []),
                "created_at": mem.get("created_at"),
                "access_count": mem.get("access_count", 0),
            })

        results.sort(key=lambda r: r["composite_score"], reverse=True)
        return {"results": results[:limit]}

    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory by ID."""
        mem = self.db.get_memory(memory_id)
        if mem:
            self.db.increment_access(memory_id)
        return mem

    def get_all(
        self,
        user_id: str = "default",
        agent_id: Optional[str] = None,
        layer: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get all memories for a user."""
        memories = self.db.get_all_memories(
            user_id=user_id,
            agent_id=agent_id,
            layer=layer,
            limit=limit,
        )
        return {"results": memories}

    def update(self, memory_id: str, data: Any) -> Dict[str, Any]:
        """Update a memory's content or metadata."""
        if isinstance(data, str):
            # Simple content update
            content = data
            existing = self.db.get_memory(memory_id) or {}
            existing_metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
            existing_categories = existing.get("categories")
            if isinstance(existing_categories, str):
                try:
                    existing_categories = json.loads(existing_categories)
                except (json.JSONDecodeError, TypeError):
                    existing_categories = []
            if not isinstance(existing_categories, list):
                existing_categories = []
            metadata, quality = apply_memory_quality_contract(
                content,
                existing_metadata,
                existing_categories,
                explicit_remember=bool(
                    existing_metadata.get("explicit_remember")
                    or existing_metadata.get("policy_explicit")
                ),
            )
            embedding = self.embedder.embed(content, memory_action="add")
            ch = _content_hash(content)
            strength = enforce_quality_strength(float(existing.get("strength", 1.0)), quality)
            layer = enforce_quality_layer(str(existing.get("layer") or "sml"), quality)
            update = {
                "memory": content,
                "embedding": embedding,
                "content_hash": ch,
                "metadata": metadata,
                "namespace": quality.namespace,
                "memory_type": quality.memory_type,
                "layer": layer,
                "strength": strength,
                "importance": metadata.get("importance", existing.get("importance", 0.5)),
            }
            if quality.decay_lambda is not None:
                update["decay_lambda"] = quality.decay_lambda
            self.db.update_memory(memory_id, update)
            # Update vector store
            payload = {
                "memory_id": memory_id,
                "user_id": existing.get("user_id") or "default",
                "memory": content,
                "namespace": quality.namespace,
                "memory_type": quality.memory_type,
                "layer": layer,
                "metadata": metadata,
            }
            if existing.get("agent_id"):
                payload["agent_id"] = existing.get("agent_id")
            try:
                self.vector_store.delete(memory_id)
                self.vector_store.insert(
                    vectors=[embedding], payloads=[payload], ids=[memory_id]
                )
            except Exception as e:
                logger.warning("Vector update failed: %s", e)
            return {"id": memory_id, "event": "UPDATE", "memory": content}
        elif isinstance(data, dict):
            self.db.update_memory(memory_id, data)
            return {"id": memory_id, "event": "UPDATE"}
        return {"error": "Invalid update data"}

    def delete(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory (tombstone)."""
        self.db.delete_memory(memory_id)
        try:
            self.vector_store.delete(memory_id)
        except Exception as e:
            logger.warning("Vector delete failed: %s", e)
        return {"id": memory_id, "event": "DELETE"}

    def _log_decay(self, mem: Dict[str, Any], new_strength: float) -> None:
        """Record the pass, which is what the next pass starts its clock from."""
        if callable(getattr(self.db, "log_event", None)):
            self.db.log_event(
                mem["id"], "DECAY", old_strength=mem.get("strength"), new_strength=new_strength
            )

    def apply_decay(
        self,
        user_id: Optional[str] = None,
        scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply FadeMem decay to all memories."""
        scope = scope or {}
        target_user = user_id or scope.get("user_id")
        memories = self.db.get_all_memories(
            user_id=target_user,
            agent_id=scope.get("agent_id"),
            include_tombstoned=False,
        )

        decayed = 0
        forgotten = 0
        promoted = 0
        # Each memory's clock starts at its own last decay. See `decay_since`.
        last_decayed = (
            self.db.latest_event_times("DECAY")
            if callable(getattr(self.db, "latest_event_times", None))
            else {}
        )
        # A memory still the source of a current fact fades only to the
        # threshold: forgetting it would delete the fact with it.
        holding_facts = (
            self.db.memory_ids_holding_current_facts()
            if callable(getattr(self.db, "memory_ids_holding_current_facts", None))
            else set()
        )

        for mem in memories:
            if mem.get("immutable"):
                continue

            # Shruti-tier memories are immune to decay
            mem_meta = mem.get("metadata") or {}
            if isinstance(mem_meta, str):
                try:
                    mem_meta = json.loads(mem_meta)
                except Exception:
                    mem_meta = {}
            if mem_meta.get("tier") == "shruti":
                continue
            quality = memory_quality_from_record({**mem, "metadata": mem_meta})

            new_strength = calculate_decayed_strength(
                current_strength=float(mem.get("strength", 1.0)),
                last_accessed=decay_since(
                    mem.get("last_accessed", mem.get("created_at", "")),
                    last_decayed.get(str(mem.get("id"))),
                ),
                access_count=int(mem.get("access_count", 0)),
                layer=mem.get("layer", "sml"),
                config=self.fade_config,
            )
            new_strength = enforce_quality_strength(new_strength, quality)
            if should_forget(new_strength, self.fade_config) and str(mem.get("id")) in holding_facts:
                new_strength = self.fade_config.forgetting_threshold

            if should_forget(new_strength, self.fade_config):
                access_count = int(mem.get("access_count", 0))
                # Vasana: memories recalled 3+ times compress instead of dying
                if quality.suppress_from_default_recall:
                    if self.fade_config.use_tombstone_deletion:
                        self.db.update_memory(mem["id"], {"tombstone": 1, "strength": new_strength})
                    else:
                        self.db.delete_memory(mem["id"])
                        try:
                            self.vector_store.delete(mem["id"])
                        except Exception:
                            pass
                    forgotten += 1
                elif access_count >= 3:
                    content = mem.get("memory", mem.get("content", ""))
                    # Compress to first 100 chars + keep keywords
                    compressed = content[:100].rstrip() + "..." if len(content) > 100 else content
                    update = {
                        "strength": self.fade_config.forgetting_threshold + 0.01,
                        "memory": compressed,
                        "metadata": json.dumps({**mem_meta, "tier": "vasana"}),
                    }
                    self.db.update_memory(mem["id"], update)
                    decayed += 1
                elif self.fade_config.use_tombstone_deletion:
                    self.db.update_memory(mem["id"], {"tombstone": 1, "strength": new_strength})
                    forgotten += 1
                else:
                    self.db.delete_memory(mem["id"])
                    try:
                        self.vector_store.delete(mem["id"])
                    except Exception:
                        pass
                    forgotten += 1
            elif should_promote(
                mem.get("layer", "sml"),
                int(mem.get("access_count", 0)),
                new_strength,
                self.fade_config,
            ) and quality.layer == "lml" and not quality.suppress_from_default_recall:
                self.db.update_memory(mem["id"], {"strength": new_strength, "layer": "lml"})
                self._log_decay(mem, new_strength)
                promoted += 1
            elif new_strength != float(mem.get("strength", 1.0)):
                self.db.update_memory(mem["id"], {"strength": new_strength})
                self._log_decay(mem, new_strength)

            decayed += 1

        return {
            "decayed": decayed,
            "forgotten": forgotten,
            "promoted": promoted,
        }

    def get_stats(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        memories = self.db.get_all_memories(user_id=user_id, agent_id=agent_id)
        sml_count = sum(1 for m in memories if m.get("layer") == "sml")
        lml_count = sum(1 for m in memories if m.get("layer") == "lml")
        return {
            "total": len(memories),
            "sml_count": sml_count,
            "lml_count": lml_count,
        }

    def history(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get history for a memory."""
        return self.db.get_memory_history(memory_id)

    # ---- Internal helpers ----

    def _cached_embed(self, query: str) -> List[float]:
        """Embed a query with LRU caching."""
        key = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            return self._query_cache[key]
        embedding = self.embedder.embed(query, memory_action="search")
        self._query_cache[key] = embedding
        if len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)
        return embedding
