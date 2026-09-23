"""
CategoryMem - Dynamic hierarchical category layer for engram.

Unlike traditional static 3-layer approaches (Resource→Item→Category),
CategoryMem provides:

1. Dynamic Categories - Auto-discovered from content, not predefined
2. Hierarchical Structure - Nested categories (preferences > coding > languages)
3. Evolving Summaries - LLM-generated summaries that update with new memories
4. Cross-Category Links - Related categories are semantically linked
5. Category Decay - Unused categories merge/fade (bio-inspired, like engram)
6. Category Embeddings - Categories have their own vectors for semantic matching
7. Category-Aware Retrieval - Boost search results from relevant categories

The key insight: categories themselves are subject to biological memory principles.
Frequently accessed categories strengthen, unused ones fade and merge.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dhee.utils.math import cosine_similarity as _cosine_similarity

logger = logging.getLogger(__name__)


class CategoryType(str, Enum):
    """Types of memory categories."""
    # Core categories (built-in)
    PREFERENCE = "preference"      # User preferences and likes/dislikes
    FACT = "fact"                  # Factual information
    CONTEXT = "context"            # Contextual/situational info
    PROCEDURE = "procedure"        # How to do things
    CORRECTION = "correction"      # Learned corrections/mistakes

    # Dynamic categories (auto-discovered)
    DYNAMIC = "dynamic"            # Auto-generated from content


@dataclass
class Category:
    """A memory category with hierarchical structure."""
    id: str
    name: str
    description: str
    category_type: CategoryType = CategoryType.DYNAMIC

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Statistics
    memory_count: int = 0
    total_strength: float = 0.0  # Sum of all memory strengths
    access_count: int = 0
    last_accessed: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Semantic representation
    embedding: Optional[List[float]] = None  # Category's semantic vector
    keywords: List[str] = field(default_factory=list)

    # Summary
    summary: Optional[str] = None
    summary_updated_at: Optional[str] = None

    # Cross-category links (related categories)
    related_ids: List[str] = field(default_factory=list)

    # Decay tracking
    strength: float = 1.0  # Category strength (decays like memories)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category_type": self.category_type.value,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "memory_count": self.memory_count,
            "total_strength": self.total_strength,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "embedding": self.embedding,
            "keywords": self.keywords,
            "summary": self.summary,
            "summary_updated_at": self.summary_updated_at,
            "related_ids": self.related_ids,
            "strength": self.strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Category":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            category_type=CategoryType(data.get("category_type", "dynamic")),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            memory_count=data.get("memory_count", 0),
            total_strength=data.get("total_strength", 0.0),
            access_count=data.get("access_count", 0),
            last_accessed=data.get("last_accessed"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            embedding=data.get("embedding"),
            keywords=data.get("keywords", []),
            summary=data.get("summary"),
            summary_updated_at=data.get("summary_updated_at"),
            related_ids=data.get("related_ids", []),
            strength=data.get("strength", 1.0),
        )

    @property
    def avg_strength(self) -> float:
        """Average strength of memories in this category."""
        if self.memory_count == 0:
            return 0.0
        return self.total_strength / self.memory_count


@dataclass
class CategoryMatch:
    """Result of matching content to a category."""
    category_id: str
    category_name: str
    confidence: float  # 0-1
    is_new: bool = False
    suggested_parent_id: Optional[str] = None


@dataclass
class CategoryTreeNode:
    """Node in the category hierarchy tree."""
    category: Category
    children: List["CategoryTreeNode"] = field(default_factory=list)
    depth: int = 0


# Prompts for category operations
CATEGORY_DETECTION_PROMPT = """Analyze this memory content and determine its category.

Memory Content: {content}

Existing Categories:
{existing_categories}

Instructions:
1. If the content fits an existing category, return that category's ID
2. If it fits a sub-category of an existing one, suggest creating a child category
3. If it's entirely new, suggest a new category name and description

Return JSON:
{{
    "action": "use_existing" | "create_child" | "create_new",
    "category_id": "existing_category_id or null",
    "new_category": {{
        "name": "Category Name (2-4 words)",
        "description": "Brief description",
        "keywords": ["keyword1", "keyword2", "keyword3"],
        "parent_id": "parent_category_id or null"
    }},
    "confidence": 0.0-1.0
}}
"""

CATEGORY_SUMMARY_PROMPT = """Generate a concise summary for this memory category.

Category: {category_name}
Description: {category_description}

Memories in this category:
{memories}

Create a 2-3 sentence summary that captures:
1. The key information stored in this category
2. Common themes or patterns
3. Most important/frequently accessed items

Summary:"""

CATEGORY_MERGE_PROMPT = """Analyze if these two categories should be merged.

Category 1: {cat1_name}
- Description: {cat1_desc}
- Keywords: {cat1_keywords}
- Memory count: {cat1_count}
- Strength: {cat1_strength}

Category 2: {cat2_name}
- Description: {cat2_desc}
- Keywords: {cat2_keywords}
- Memory count: {cat2_count}
- Strength: {cat2_strength}

Should these categories be merged? Consider:
- Semantic overlap
- Whether one subsumes the other
- Combined usefulness

Return JSON:
{{
    "should_merge": true | false,
    "reason": "Brief explanation",
    "merged_name": "Name for merged category (if merging)",
    "merged_description": "Description for merged category (if merging)",
    "merged_keywords": ["combined", "keywords"]
}}
"""


class CategoryProcessor:
    """
    Processes and manages the category layer.

    This is the brain of CategoryMem - it handles:
    - Auto-categorization of new memories
    - Category hierarchy management
    - Summary generation and updates
    - Category decay and merging
    - Cross-category linking
    """

    # Built-in root categories
    ROOT_CATEGORIES = [
        ("preferences", "User Preferences", "Personal preferences, likes, dislikes, and choices", CategoryType.PREFERENCE),
        ("facts", "Facts & Knowledge", "Factual information and learned knowledge", CategoryType.FACT),
        ("context", "Context & Situations", "Situational context, projects, environments", CategoryType.CONTEXT),
        ("procedures", "Procedures & How-To", "Instructions, workflows, and procedures", CategoryType.PROCEDURE),
        ("corrections", "Corrections & Lessons", "Mistakes, corrections, and learned lessons", CategoryType.CORRECTION),
    ]

    def __init__(
        self,
        llm,
        embedder,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the category processor.

        Args:
            llm: Language model for category detection and summarization
            embedder: Embedding model for category vectors
            config: Configuration options
        """
        self.llm = llm
        self.embedder = embedder
        self.config = config or {}
        #: Returns a decider (see `dhee.decisions`) or None. Set by the memory
        #: that owns this processor, and read on every call so a decisions
        #: config applied after construction takes effect.
        self.decider_fn: Optional[Callable[[], Any]] = None

        # In-memory category cache (persisted to DB by Memory class)
        self.categories: Dict[str, Category] = {}

        # Initialize root categories
        self._init_root_categories()

    def _init_root_categories(self):
        """Initialize built-in root categories."""
        for cat_id, name, desc, cat_type in self.ROOT_CATEGORIES:
            if cat_id not in self.categories:
                self.categories[cat_id] = Category(
                    id=cat_id,
                    name=name,
                    description=desc,
                    category_type=cat_type,
                    keywords=name.lower().split() + desc.lower().split()[:5],
                )

    def load_categories(self, categories_data: List[Dict[str, Any]]):
        """Load categories from database."""
        for data in categories_data:
            cat = Category.from_dict(data)
            self.categories[cat.id] = cat

        # Ensure root categories exist
        self._init_root_categories()

    def detect_category(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        use_llm: bool = True,
    ) -> CategoryMatch:
        """
        Detect the best category for a piece of content.

        Uses a hybrid approach:
        1. Fast keyword/embedding matching first
        2. LLM for ambiguous cases or new categories

        Args:
            content: Memory content to categorize
            metadata: Optional metadata with hints
            use_llm: Whether to use LLM for detection

        Returns:
            CategoryMatch with category info
        """
        content_lower = content.lower()

        # Phase 1: Quick keyword matching
        best_match = None
        best_score = 0.0

        for cat in self.categories.values():
            score = self._keyword_match_score(content_lower, cat)
            if score > best_score:
                best_score = score
                best_match = cat

        # If strong keyword match, use it
        if best_match and best_score >= 0.7:
            return CategoryMatch(
                category_id=best_match.id,
                category_name=best_match.name,
                confidence=best_score,
            )

        # Phase 2: Embedding similarity (if available)
        if self.embedder:
            content_embedding = self.embedder.embed(content, memory_action="categorize")

            for cat in self.categories.values():
                if cat.embedding:
                    sim = self._cosine_similarity(content_embedding, cat.embedding)
                    if sim > best_score:
                        best_score = sim
                        best_match = cat

        # If good embedding match, use it
        if best_match and best_score >= 0.6:
            return CategoryMatch(
                category_id=best_match.id,
                category_name=best_match.name,
                confidence=best_score,
            )

        # Phase 3: a decision among the categories that already exist. Picking
        # one of a known list is a closed choice, not writing; the LLM is kept
        # for the case the decision says none fits, where a new category has
        # to be named.
        decided = self._decide_category(content)
        if decided is not None:
            return decided

        # Phase 4: Use LLM for detection/creation
        if use_llm and self.llm:
            return self._llm_detect_category(content, metadata)

        # Fallback: use best match or default to 'context'
        if best_match:
            return CategoryMatch(
                category_id=best_match.id,
                category_name=best_match.name,
                confidence=best_score,
            )

        return CategoryMatch(
            category_id="context",
            category_name="Context & Situations",
            confidence=0.3,
        )

    def detect_categories_batch(
        self,
        contents: List[str],
        use_llm: bool = True,
    ) -> List[CategoryMatch]:
        """Batch category detection for multiple contents.

        Uses keyword matching first for strong matches, then batches
        ambiguous items into a single LLM call.
        """
        if not contents:
            return []
        if len(contents) == 1:
            return [self.detect_category(contents[0], use_llm=use_llm)]

        results: List[Optional[CategoryMatch]] = [None] * len(contents)
        ambiguous_indices: List[int] = []

        # Phase 1: Fast keyword matching for each content
        for i, content in enumerate(contents):
            content_lower = content.lower()
            best_match = None
            best_score = 0.0

            for cat in self.categories.values():
                score = self._keyword_match_score(content_lower, cat)
                if score > best_score:
                    best_score = score
                    best_match = cat

            if best_match and best_score >= 0.7:
                results[i] = CategoryMatch(
                    category_id=best_match.id,
                    category_name=best_match.name,
                    confidence=best_score,
                )
            else:
                ambiguous_indices.append(i)

        # Phase 2: Batch LLM for ambiguous items (or sequential fallback)
        for idx in ambiguous_indices:
            results[idx] = self.detect_category(
                contents[idx], use_llm=use_llm,
            )

        return [r for r in results]  # type: ignore[misc]

    def _keyword_match_score(self, content_lower: str, category: Category) -> float:
        """Calculate keyword match score between content and category."""
        if not category.keywords:
            return 0.0

        matches = sum(1 for kw in category.keywords if kw.lower() in content_lower)
        return min(1.0, matches / max(3, len(category.keywords) * 0.5))

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return _cosine_similarity(vec1, vec2)

    def _decide_category(self, content: str) -> Optional["CategoryMatch"]:
        decider = self.decider_fn() if self.decider_fn else None
        if decider is None or not self.categories:
            return None
        from dhee.decisions.jev import ranked

        criteria = {
            cat.id: f"{cat.name}: {cat.description}"[:200] for cat in list(self.categories.values())[:200]
        }
        criteria["new_category"] = "None of these fits; it needs a category of its own."
        answers = decider.decide(
            {"memory": content[:1500]},
            {
                "category": {
                    "type": "choice",
                    "instructions": "Which category does this memory belong in?",
                    "criteria": criteria,
                }
            },
        )
        top = ranked(answers, "category")
        if not top or top[0][0] == "new_category" or top[0][1] < 0.6:
            return None
        cat = self.categories.get(top[0][0])
        if cat is None:
            return None
        return CategoryMatch(category_id=cat.id, category_name=cat.name, confidence=top[0][1])

    def _llm_detect_category(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CategoryMatch:
        """Use LLM to detect or create category."""
        # Format existing categories for prompt
        existing_cats = "\n".join([
            f"- {cat.id}: {cat.name} - {cat.description}"
            for cat in self.categories.values()
        ])

        prompt = CATEGORY_DETECTION_PROMPT.format(
            content=content[:500],  # Truncate for efficiency
            existing_categories=existing_cats,
        )

        try:
            response = self.llm.generate(prompt)

            # Strip <think>...</think> blocks (Qwen 3.x thinking models)
            response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE).strip()
            response = re.sub(r"<think>[\s\S]*$", "", response, flags=re.IGNORECASE).strip()

            # Parse JSON response — use raw_decode to ignore trailing LLM text
            json_start = response.find("{")
            if json_start >= 0:
                data, _ = json.JSONDecoder().raw_decode(response, json_start)

                action = data.get("action", "use_existing")
                confidence = float(data.get("confidence", 0.5))

                if action == "use_existing" and data.get("category_id"):
                    cat_id = data["category_id"]
                    if cat_id in self.categories:
                        return CategoryMatch(
                            category_id=cat_id,
                            category_name=self.categories[cat_id].name,
                            confidence=confidence,
                        )

                if action in ("create_child", "create_new") and data.get("new_category"):
                    new_cat = data["new_category"]
                    cat_id = self._create_category(
                        name=new_cat.get("name", "Unnamed"),
                        description=new_cat.get("description", ""),
                        keywords=new_cat.get("keywords", []),
                        parent_id=new_cat.get("parent_id"),
                    )
                    return CategoryMatch(
                        category_id=cat_id,
                        category_name=new_cat.get("name", "Unnamed"),
                        confidence=confidence,
                        is_new=True,
                        suggested_parent_id=new_cat.get("parent_id"),
                    )

        except Exception as e:
            logger.warning(f"LLM category detection failed: {e}")

        # Fallback
        return CategoryMatch(
            category_id="context",
            category_name="Context & Situations",
            confidence=0.3,
        )

    def _create_category(
        self,
        name: str,
        description: str,
        keywords: List[str] = None,
        parent_id: str = None,
    ) -> str:
        """Create a new category."""
        cat_id = f"cat_{uuid.uuid4().hex[:8]}"

        # Generate embedding for category
        embedding = None
        if self.embedder:
            embedding_text = f"{name}. {description}"
            embedding = self.embedder.embed(embedding_text, memory_action="categorize")

        category = Category(
            id=cat_id,
            name=name,
            description=description,
            category_type=CategoryType.DYNAMIC,
            parent_id=parent_id,
            keywords=keywords or [],
            embedding=embedding,
        )

        self.categories[cat_id] = category

        # Update parent's children list
        if parent_id and parent_id in self.categories:
            self.categories[parent_id].children_ids.append(cat_id)

        logger.info(f"Created new category: {cat_id} - {name}")
        return cat_id

    def update_category_stats(
        self,
        category_id: str,
        memory_strength: float,
        is_addition: bool = True,
    ):
        """Update category statistics when memory is added/removed."""
        if category_id not in self.categories:
            return

        cat = self.categories[category_id]

        if is_addition:
            cat.memory_count += 1
            cat.total_strength += memory_strength
        else:
            cat.memory_count = max(0, cat.memory_count - 1)
            cat.total_strength = max(0, cat.total_strength - memory_strength)

        # Invalidate summary
        cat.summary = None
        cat.summary_updated_at = None

    def access_category(self, category_id: str):
        """Record access to a category."""
        if category_id not in self.categories:
            return

        cat = self.categories[category_id]
        cat.access_count += 1
        cat.last_accessed = datetime.now(timezone.utc).isoformat()

        # Strengthen category on access (bio-inspired)
        cat.strength = min(1.0, cat.strength + 0.02)

    def generate_summary(self, category_id: str, memories: List[Dict[str, Any]]) -> str:
        """Generate or update summary for a category."""
        if category_id not in self.categories:
            return ""

        cat = self.categories[category_id]

        if not memories:
            return f"Empty category: {cat.description}"

        # Format memories for prompt
        memories_text = "\n".join([
            f"- {m.get('memory', '')[:200]}"
            for m in memories[:20]  # Limit to 20 for efficiency
        ])

        prompt = CATEGORY_SUMMARY_PROMPT.format(
            category_name=cat.name,
            category_description=cat.description,
            memories=memories_text,
        )

        try:
            summary = self.llm.generate(prompt)
            cat.summary = summary.strip()
            cat.summary_updated_at = datetime.now(timezone.utc).isoformat()
            return cat.summary
        except Exception as e:
            logger.warning(f"Summary generation failed for {category_id}: {e}")
            return f"Category with {len(memories)} memories about {cat.description}"

    def apply_category_decay(self, decay_rate: float = 0.05) -> Dict[str, Any]:
        """
        Apply decay to categories - bio-inspired like engram.

        Unused categories weaken over time, potentially merging
        with similar categories.

        Args:
            decay_rate: Rate of decay per cycle

        Returns:
            Stats about decayed/merged categories
        """
        decayed = 0
        merged = 0
        deleted = 0

        # Calculate decay for each dynamic category
        weak_categories = []

        for cat in list(self.categories.values()):
            if cat.category_type != CategoryType.DYNAMIC:
                continue  # Don't decay root categories

            # Decay based on time since last access
            if cat.last_accessed:
                try:
                    last_access = datetime.fromisoformat(cat.last_accessed)
                    if last_access.tzinfo is None:
                        last_access = last_access.replace(tzinfo=timezone.utc)
                    days_since = (datetime.now(timezone.utc) - last_access).days
                    decay_amount = decay_rate * (days_since / 7)  # Weekly decay
                    cat.strength = max(0.1, cat.strength - decay_amount)
                    decayed += 1
                except (ValueError, TypeError) as e:
                    logger.debug(f"Category decay calculation failed for {cat.id}: {e}")
                    continue

            # Track weak categories for potential merging
            if cat.strength < 0.3 and cat.memory_count < 3:
                weak_categories.append(cat)

        # Try to merge weak categories
        for cat in weak_categories:
            if cat.id not in self.categories:
                continue  # Already merged

            merge_target = self._find_merge_target(cat)
            if merge_target:
                self._merge_categories(cat.id, merge_target.id)
                merged += 1
            elif cat.memory_count == 0 and cat.strength < 0.15:
                # Delete empty, very weak categories
                del self.categories[cat.id]
                deleted += 1

        return {"decayed": decayed, "merged": merged, "deleted": deleted}

    def _find_merge_target(self, weak_cat: Category) -> Optional[Category]:
        """Find a suitable category to merge a weak one into."""
        best_target = None
        best_similarity = 0.0

        for cat in self.categories.values():
            if cat.id == weak_cat.id:
                continue
            if cat.category_type == CategoryType.DYNAMIC and cat.strength < 0.5:
                continue  # Don't merge into another weak category

            # Check embedding similarity
            if weak_cat.embedding and cat.embedding:
                sim = self._cosine_similarity(weak_cat.embedding, cat.embedding)
                if sim > best_similarity and sim > 0.7:
                    best_similarity = sim
                    best_target = cat

            # Check keyword overlap
            if weak_cat.keywords and cat.keywords:
                overlap = len(set(weak_cat.keywords) & set(cat.keywords))
                kw_count = len(weak_cat.keywords)
                if overlap >= 2 and kw_count > 0:
                    if not best_target or overlap / kw_count > best_similarity:
                        best_target = cat

        return best_target

    def _merge_categories(self, source_id: str, target_id: str):
        """Merge source category into target."""
        if source_id not in self.categories or target_id not in self.categories:
            return

        source = self.categories[source_id]
        target = self.categories[target_id]

        # Transfer stats
        target.memory_count += source.memory_count
        target.total_strength += source.total_strength
        target.access_count += source.access_count

        # Merge keywords (deduplicate)
        target.keywords = list(set(target.keywords + source.keywords))

        # Merge children
        for child_id in source.children_ids:
            if child_id in self.categories:
                self.categories[child_id].parent_id = target_id
                target.children_ids.append(child_id)

        # Invalidate summary
        target.summary = None

        # Remove source
        del self.categories[source_id]

        logger.info(f"Merged category {source_id} into {target_id}")

    def find_related_categories(self, category_id: str, limit: int = 3) -> List[str]:
        """Find categories related to the given one.

        Note: This is O(N * D) where N is the number of categories and D is the
        embedding dimensionality, because it computes cosine similarity against
        every category. For very large category counts this could become a
        bottleneck; consider caching related_ids or using an approximate
        nearest-neighbor index if N grows large.
        """
        if category_id not in self.categories:
            return []

        cat = self.categories[category_id]
        related = []

        for other in self.categories.values():
            if other.id == category_id:
                continue

            score = 0.0

            # Embedding similarity
            if cat.embedding and other.embedding:
                score = self._cosine_similarity(cat.embedding, other.embedding)

            # Keyword overlap bonus
            if cat.keywords and other.keywords:
                overlap = len(set(cat.keywords) & set(other.keywords))
                score += overlap * 0.1

            if score > 0.4:
                related.append((other.id, score))

        # Sort by score and return top N
        related.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in related[:limit]]

    def get_category_tree(self) -> List[CategoryTreeNode]:
        """Get hierarchical tree of categories."""
        # Find root categories (no parent)
        roots = [cat for cat in self.categories.values() if not cat.parent_id]

        def build_tree(cat: Category, depth: int = 0) -> CategoryTreeNode:
            children = [
                build_tree(self.categories[child_id], depth + 1)
                for child_id in cat.children_ids
                if child_id in self.categories
            ]
            return CategoryTreeNode(category=cat, children=children, depth=depth)

        return [build_tree(root) for root in roots]

    def get_all_categories(self) -> List[Dict[str, Any]]:
        """Get all categories as dicts for persistence."""
        return [cat.to_dict() for cat in self.categories.values()]

    def get_category(self, category_id: str) -> Optional[Category]:
        """Get a category by ID."""
        return self.categories.get(category_id)

    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about categories."""
        total = len(self.categories)
        root_count = len([c for c in self.categories.values() if not c.parent_id])
        dynamic_count = len([c for c in self.categories.values() if c.category_type == CategoryType.DYNAMIC])

        total_memories = sum(c.memory_count for c in self.categories.values())
        avg_strength = sum(c.strength for c in self.categories.values()) / total if total > 0 else 0

        # Find most active categories
        top_categories = sorted(
            self.categories.values(),
            key=lambda c: c.access_count,
            reverse=True
        )[:5]

        return {
            "total_categories": total,
            "root_categories": root_count,
            "dynamic_categories": dynamic_count,
            "total_memories_categorized": total_memories,
            "avg_category_strength": round(avg_strength, 3),
            "top_categories": [
                {"id": c.id, "name": c.name, "access_count": c.access_count}
                for c in top_categories
            ],
        }
