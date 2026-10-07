import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .sqlite_analytics import SQLiteAnalyticsMixin
from .sqlite_artifacts import SQLiteArtifactMixin
from .sqlite_common import (
    VALID_MEMORY_COLUMNS,
    VALID_PROFILE_COLUMNS,
    VALID_SCENE_COLUMNS,
    _utcnow_iso,
)
from .sqlite_domains import SQLiteDomainMixin

logger = logging.getLogger(__name__)


class _SQLiteBase:
    """Base class for SQLite managers with common functionality."""

    def latest_event_times(self, event: str) -> Dict[str, str]:
        """When each memory last had `event` logged, as the history table wrote it.

        Read once per decay pass, so the pass can start each memory's clock at
        its own last decay (see `core.decay.decay_since`).
        """
        with self._get_connection() as conn:
            if not getattr(self, "_history_event_index", False):
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_history_event_memory "
                    "ON memory_history(event, memory_id)"
                )
                self._history_event_index = True
            rows = conn.execute(
                "SELECT memory_id, MAX(timestamp) FROM memory_history WHERE event = ? GROUP BY memory_id",
                (event,),
            ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows if row[1]}

    def memory_ids_holding_current_facts(self) -> Set[str]:
        """Memories that are still the source of a fact nothing has replaced.

        Forgetting a memory deletes the facts extracted from it (see
        `_delete_memory_query_artifacts_conn`). Decay asks this first, so a
        conversation can fade without taking "exam on 21 October" with it.
        """
        with self._get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'engram_facts'"
            ).fetchone()
            if not exists:
                return set()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(engram_facts)")}
            where, params = [], []
            if "superseded_by_id" in columns:
                where.append("superseded_by_id IS NULL")
            if "valid_until" in columns:
                where.append("(valid_until IS NULL OR valid_until = '' OR valid_until > ?)")
                params.append(datetime.now(timezone.utc).isoformat())
            rows = conn.execute(
                "SELECT DISTINCT memory_id FROM engram_facts"
                + (" WHERE " + " AND ".join(where) if where else ""),
                params,
            ).fetchall()
        return {str(row[0]) for row in rows if row[0]}

    def __init__(self, db_path: str):
        self.db_path = db_path
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # Phase 1: Persistent connection with WAL mode.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA cache_size=-8000")  # 8MB cache
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

    def close(self) -> None:
        """Close the persistent connection for clean shutdown."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_path={self.db_path!r})"

    @contextmanager
    def _get_connection(self):
        """Yield the persistent connection under the thread lock."""
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _is_migration_applied(self, conn: sqlite3.Connection, version: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            (version,),
        ).fetchone()
        return row is not None

    # Phase 5: Allowed table names for ALTER TABLE to prevent SQL injection.
    _ALLOWED_TABLES = frozenset({
        "memories", "scenes", "profiles", "categories",
    })

    def _migrate_add_column_conn(
        self,
        conn: sqlite3.Connection,
        table: str,
        column: str,
        col_type: str,
    ) -> None:
        """Add a column using an existing connection, if missing."""
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"Invalid table for migration: {table!r}")
        # Validate column name: must be alphanumeric/underscore only.
        if not column.replace("_", "").isalnum():
            raise ValueError(f"Invalid column name: {column!r}")
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass

    @staticmethod
    def _parse_json_value(value: Any, default: Any) -> Any:
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            parsed = json.loads(value)
        except Exception:
            return default
        if isinstance(default, dict) and not isinstance(parsed, dict):
            return dict(default)
        if isinstance(default, list) and not isinstance(parsed, list):
            return list(default)
        return parsed

    @staticmethod
    def _parse_json_dict(value: Any, *, preserve_legacy: bool = False) -> Dict[str, Any]:
        """Decode a JSON object field without allowing scalar poison through.

        Some early Dhee and downstream integrations stored metadata as JSON
        strings/lists. Returning those raw scalars makes callers that expect a
        mapping fail much later in unrelated context tools. Keep the boundary
        strict here so every row leaves the DB with mapping-shaped metadata.
        """
        if value in (None, "", {}, []):
            return {}
        if isinstance(value, dict):
            return dict(value)
        parsed: Any = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = value
        if isinstance(parsed, dict):
            return dict(parsed)
        if preserve_legacy:
            return {
                "legacy_metadata_raw": parsed,
                "legacy_metadata_type": type(parsed).__name__,
            }
        return {}

    @staticmethod
    def _parse_json_list(value: Any) -> List[Any]:
        if value in (None, "", []):
            return []
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        parsed: Any = value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except Exception:
                return []
        return list(parsed) if isinstance(parsed, list) else []


class CoreSQLiteManager(_SQLiteBase):
    """Minimal SQLite manager for CoreMemory - only essential tables.

    Tables created:
        - memories: core memory storage with content_hash for deduplication
        - memory_history: audit trail for memory operations
        - decay_log: decay cycle metrics
        - schema_migrations: migration tracking
    """

    def __init__(self, db_path: str):
        super().__init__(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize minimal schema for CoreMemory."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    run_id TEXT,
                    app_id TEXT,
                    metadata TEXT DEFAULT '{}',
                    categories TEXT DEFAULT '[]',
                    immutable INTEGER DEFAULT 0,
                    expiration_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    layer TEXT DEFAULT 'sml' CHECK (layer IN ('sml', 'lml')),
                    strength REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
                    embedding TEXT,
                    related_memories TEXT DEFAULT '[]',
                    source_memories TEXT DEFAULT '[]',
                    tombstone INTEGER DEFAULT 0,
                    content_hash TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_user_layer ON memories(user_id, layer);
                CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength DESC);
                CREATE INDEX IF NOT EXISTS idx_tombstone ON memories(tombstone);

                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    old_strength REAL,
                    new_strength REAL,
                    old_layer TEXT,
                    new_layer TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS decay_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    memories_decayed INTEGER,
                    memories_forgotten INTEGER,
                    memories_promoted INTEGER,
                    storage_before_mb REAL,
                    storage_after_mb REAL
                );
                """
            )
            # Migrate content_hash column + index for pre-existing DBs
            self._ensure_content_hash_column(conn)

    def _ensure_content_hash_column(self, conn: sqlite3.Connection) -> None:
        """Add content_hash column + index for SHA-256 dedup (idempotent)."""
        if self._is_migration_applied(conn, "v2_content_hash"):
            return
        self._migrate_add_column_conn(conn, "memories", "content_hash", "TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash, user_id)"
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_content_hash')"
        )

    # Core memory operations
    def add_memory(self, memory_data: Dict[str, Any], *, log_history: bool = True) -> str:
        memory_id = memory_data.get("id", str(uuid.uuid4()))
        now = _utcnow_iso()
        metadata = self._parse_json_dict(memory_data.get("metadata"), preserve_legacy=True)
        categories = self._parse_json_list(memory_data.get("categories"))
        embedding = self._parse_json_list(memory_data.get("embedding"))
        related_memories = self._parse_json_list(memory_data.get("related_memories"))
        source_memories = self._parse_json_list(memory_data.get("source_memories"))

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, memory, user_id, agent_id, run_id, app_id,
                    metadata, categories, immutable, expiration_date,
                    created_at, updated_at, layer, strength, access_count,
                    last_accessed, embedding, related_memories, source_memories, tombstone,
                    content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    memory_data.get("memory", ""),
                    memory_data.get("user_id"),
                    memory_data.get("agent_id"),
                    memory_data.get("run_id"),
                    memory_data.get("app_id"),
                    json.dumps(metadata),
                    json.dumps(categories),
                    1 if memory_data.get("immutable", False) else 0,
                    memory_data.get("expiration_date"),
                    memory_data.get("created_at", now),
                    memory_data.get("updated_at", now),
                    memory_data.get("layer", "sml"),
                    memory_data.get("strength", 1.0),
                    memory_data.get("access_count", 0),
                    memory_data.get("last_accessed", now),
                    json.dumps(embedding),
                    json.dumps(related_memories),
                    json.dumps(source_memories),
                    1 if memory_data.get("tombstone", False) else 0,
                    memory_data.get("content_hash"),
                ),
            )
            # Log the add event
            if log_history:
                conn.execute(
                    """
                    INSERT INTO memory_history (
                        memory_id, event, old_value, new_value,
                        old_strength, new_strength, old_layer, new_layer
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, "ADD", None, memory_data.get("memory"), None, None, None, None),
                )
        return memory_id

    def get_memory(self, memory_id: str, include_tombstoned: bool = False) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM memories WHERE id = ?"
        params = [memory_id]
        if not include_tombstoned:
            query += " AND tombstone = 0"

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def get_memory_by_content_hash(
        self, content_hash: str, user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Find an existing memory by content hash (for deduplication)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? AND user_id = ? AND tombstone = 0 LIMIT 1",
                (content_hash, user_id),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def get_all_memories(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        layer: Optional[str] = None,
        namespace: Optional[str] = None,
        min_strength: float = 0.0,
        include_tombstoned: bool = False,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memories WHERE strength >= ?"
        params: List[Any] = [min_strength]

        if not include_tombstoned:
            query += " AND tombstone = 0"
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if app_id:
            query += " AND app_id = ?"
            params.append(app_id)
        if layer:
            query += " AND layer = ?"
            params.append(layer)

        query += " ORDER BY strength DESC"

        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        set_clauses = []
        params: List[Any] = []
        for key, value in updates.items():
            if key not in VALID_MEMORY_COLUMNS:
                raise ValueError(f"Invalid memory column: {key!r}")
            if key == "metadata":
                value = json.dumps(self._parse_json_dict(value, preserve_legacy=True))
            elif key in {"categories", "embedding", "related_memories", "source_memories"}:
                value = json.dumps(self._parse_json_list(value))
            set_clauses.append(f"{key} = ?")
            params.append(value)

        set_clauses.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.append(memory_id)

        with self._get_connection() as conn:
            old_row = conn.execute(
                "SELECT memory, strength, layer FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if not old_row:
                return False

            conn.execute(
                f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

            # Log the update event
            conn.execute(
                """
                INSERT INTO memory_history (
                    memory_id, event, old_value, new_value,
                    old_strength, new_strength, old_layer, new_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    "UPDATE",
                    old_row["memory"],
                    updates.get("memory"),
                    old_row["strength"],
                    updates.get("strength"),
                    old_row["layer"],
                    updates.get("layer"),
                ),
            )
        return True

    def delete_memory(self, memory_id: str, use_tombstone: bool = True) -> bool:
        if use_tombstone:
            return self.update_memory(memory_id, {"tombstone": 1})
        with self._get_connection() as conn:
            # Cascade delete v3 structured tables BEFORE deleting the memory row.
            for table in ("engram_facts", "engram_context", "engram_scenes",
                          "engram_entities", "engram_links"):
                try:
                    col = "source_memory_id" if table == "engram_links" else "memory_id"
                    conn.execute(f"DELETE FROM {table} WHERE {col} = ?", (memory_id,))
                except Exception:
                    pass  # Table may not exist yet
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._log_event(memory_id, "DELETE")
        return True

    def increment_access(self, memory_id: str) -> None:
        now = _utcnow_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                """,
                (now, memory_id),
            )

    def increment_access_bulk(self, memory_ids: List[str]) -> None:
        """Increment access count for multiple memories in a single transaction."""
        if not memory_ids:
            return
        now = _utcnow_iso()
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders})
                """,
                [now] + list(memory_ids),
            )

    def get_memories_bulk(
        self, memory_ids: List[str], include_tombstoned: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple memories by ID in a single query."""
        if not memory_ids:
            return {}
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            query = f"SELECT * FROM memories WHERE id IN ({placeholders})"
            if not include_tombstoned:
                query += " AND tombstone = 0"
            rows = conn.execute(query, memory_ids).fetchall()
            return {row["id"]: self._row_to_dict(row) for row in rows}

    def update_strength_bulk(self, updates: Dict[str, float]) -> None:
        """Batch-update strength for multiple memories."""
        if not updates:
            return
        now = _utcnow_iso()
        with self._get_connection() as conn:
            conn.executemany(
                "UPDATE memories SET strength = ?, updated_at = ? WHERE id = ?",
                [(strength, now, memory_id) for memory_id, strength in updates.items()],
            )

    _MEMORY_JSON_FIELDS = ("metadata", "categories", "related_memories", "source_memories")

    def _row_to_dict(self, row: sqlite3.Row, *, skip_embedding: bool = False) -> Dict[str, Any]:
        data = dict(row)
        if "metadata" in data:
            data["metadata"] = self._parse_json_dict(data.get("metadata"), preserve_legacy=True)
        for key in ("categories", "related_memories", "source_memories"):
            if key in data:
                data[key] = self._parse_json_list(data.get(key))
        # Embedding is the largest JSON field (~30-50KB for 3072-dim vectors).
        if skip_embedding:
            data.pop("embedding", None)
        elif "embedding" in data:
            data["embedding"] = self._parse_json_list(data.get("embedding"))
        data["immutable"] = bool(data.get("immutable", 0))
        data["tombstone"] = bool(data.get("tombstone", 0))
        return data

    def _log_event(self, memory_id: str, event: str, **kwargs: Any) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_history (
                    memory_id, event, old_value, new_value,
                    old_strength, new_strength, old_layer, new_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    event,
                    kwargs.get("old_value"),
                    kwargs.get("new_value"),
                    kwargs.get("old_strength"),
                    kwargs.get("new_strength"),
                    kwargs.get("old_layer"),
                    kwargs.get("new_layer"),
                ),
            )

    def log_event(self, memory_id: str, event: str, **kwargs: Any) -> None:
        """Public wrapper for logging custom events like DECAY or FUSE."""
        self._log_event(memory_id, event, **kwargs)

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY timestamp DESC",
                (memory_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # Alias for CoreMemory compatibility
    get_memory_history = get_history

    def log_decay(
        self,
        decayed: int,
        forgotten: int,
        promoted: int,
        storage_before_mb: Optional[float] = None,
        storage_after_mb: Optional[float] = None,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO decay_log (memories_decayed, memories_forgotten, memories_promoted, storage_before_mb, storage_after_mb)
                VALUES (?, ?, ?, ?, ?)
                """,
                (decayed, forgotten, promoted, storage_before_mb, storage_after_mb),
            )

    def purge_tombstoned(self) -> int:
        """Permanently delete all tombstoned memories."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT id, user_id, memory FROM memories WHERE tombstone = 1"
            ).fetchall()
            count = len(rows)
            if count > 0:
                for row in rows:
                    self._log_event(row["id"], "PURGE", old_value=row["memory"])
                conn.execute("DELETE FROM memories WHERE tombstone = 1")
            return count


class FullSQLiteManager(
    SQLiteAnalyticsMixin,
    SQLiteArtifactMixin,
    SQLiteDomainMixin,
    CoreSQLiteManager,
):
    def __repr__(self) -> str:
        return f"SQLiteManager(db_path={self.db_path!r})"

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory TEXT NOT NULL,
                    user_id TEXT,
                    agent_id TEXT,
                    run_id TEXT,
                    app_id TEXT,
                    metadata TEXT DEFAULT '{}',
                    categories TEXT DEFAULT '[]',
                    immutable INTEGER DEFAULT 0,
                    expiration_date TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    layer TEXT DEFAULT 'sml' CHECK (layer IN ('sml', 'lml')),
                    strength REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT DEFAULT CURRENT_TIMESTAMP,
                    embedding TEXT,
                    related_memories TEXT DEFAULT '[]',
                    source_memories TEXT DEFAULT '[]',
                    tombstone INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_user_layer ON memories(user_id, layer);
                CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength DESC);
                CREATE INDEX IF NOT EXISTS idx_tombstone ON memories(tombstone);

                CREATE TABLE IF NOT EXISTS memory_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    event TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    old_strength REAL,
                    new_strength REAL,
                    old_layer TEXT,
                    new_layer TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS decay_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    memories_decayed INTEGER,
                    memories_forgotten INTEGER,
                    memories_promoted INTEGER,
                    storage_before_mb REAL,
                    storage_after_mb REAL
                );

                -- CategoryMem tables
                CREATE TABLE IF NOT EXISTS categories (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category_type TEXT DEFAULT 'dynamic',
                    parent_id TEXT,
                    children_ids TEXT DEFAULT '[]',
                    memory_count INTEGER DEFAULT 0,
                    total_strength REAL DEFAULT 0.0,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    embedding TEXT,
                    keywords TEXT DEFAULT '[]',
                    summary TEXT,
                    summary_updated_at TEXT,
                    related_ids TEXT DEFAULT '[]',
                    strength REAL DEFAULT 1.0,
                    FOREIGN KEY (parent_id) REFERENCES categories(id)
                );

                CREATE INDEX IF NOT EXISTS idx_category_type ON categories(category_type);
                CREATE INDEX IF NOT EXISTS idx_category_parent ON categories(parent_id);
                CREATE INDEX IF NOT EXISTS idx_category_strength ON categories(strength DESC);

                -- Episodic scenes
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    summary TEXT,
                    topic TEXT,
                    location TEXT,
                    participants TEXT DEFAULT '[]',
                    memory_ids TEXT DEFAULT '[]',
                    start_time TEXT,
                    end_time TEXT,
                    embedding TEXT,
                    strength REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    tombstone INTEGER DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_scene_user ON scenes(user_id);
                CREATE INDEX IF NOT EXISTS idx_scene_start ON scenes(start_time DESC);

                -- Scene-Memory junction
                CREATE TABLE IF NOT EXISTS scene_memories (
                    scene_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    position INTEGER DEFAULT 0,
                    PRIMARY KEY (scene_id, memory_id),
                    FOREIGN KEY (scene_id) REFERENCES scenes(id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                );

                -- Character profiles
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    name TEXT NOT NULL,
                    profile_type TEXT DEFAULT 'contact' CHECK (profile_type IN ('self', 'contact', 'entity')),
                    narrative TEXT,
                    facts TEXT DEFAULT '[]',
                    preferences TEXT DEFAULT '[]',
                    relationships TEXT DEFAULT '[]',
                    sentiment TEXT,
                    theory_of_mind TEXT DEFAULT '{}',
                    aliases TEXT DEFAULT '[]',
                    embedding TEXT,
                    strength REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_profile_user ON profiles(user_id);
                CREATE INDEX IF NOT EXISTS idx_profile_name ON profiles(name);
                CREATE INDEX IF NOT EXISTS idx_profile_type ON profiles(profile_type);

                -- Profile-Memory junction
                CREATE TABLE IF NOT EXISTS profile_memories (
                    profile_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    role TEXT DEFAULT 'mentioned' CHECK (role IN ('subject', 'mentioned', 'about')),
                    PRIMARY KEY (profile_id, memory_id),
                    FOREIGN KEY (profile_id) REFERENCES profiles(id),
                    FOREIGN KEY (memory_id) REFERENCES memories(id)
                );
                """
            )
            # Legacy migration: add scene_id column to memories if missing.
            self._migrate_add_column_conn(conn, "memories", "scene_id", "TEXT")
            # v2 schema + idempotent migrations.
            self._ensure_v2_schema(conn)

    def _ensure_v2_schema(self, conn: sqlite3.Connection) -> None:
        """Create and migrate Engram v2 schema in-place (idempotent)."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        migrations: Dict[str, str] = {
            "v2_013": """
                CREATE TABLE IF NOT EXISTS distillation_provenance (
                    id TEXT PRIMARY KEY,
                    semantic_memory_id TEXT NOT NULL,
                    episodic_memory_id TEXT NOT NULL,
                    distillation_run_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_distill_prov_semantic ON distillation_provenance(semantic_memory_id);
                CREATE INDEX IF NOT EXISTS idx_distill_prov_episodic ON distillation_provenance(episodic_memory_id);
                CREATE INDEX IF NOT EXISTS idx_distill_prov_run ON distillation_provenance(distillation_run_id);

                CREATE TABLE IF NOT EXISTS distillation_log (
                    id TEXT PRIMARY KEY,
                    run_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT,
                    episodes_sampled INTEGER DEFAULT 0,
                    semantic_created INTEGER DEFAULT 0,
                    semantic_deduplicated INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_distill_log_user ON distillation_log(user_id, run_at DESC);
            """,
        }

        for version, ddl in migrations.items():
            if not self._is_migration_applied(conn, version):
                conn.executescript(ddl)
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (version) VALUES (?)",
                    (version,),
                )

        # Phase 3: Skip column migrations + backfills if already complete.
        if self._is_migration_applied(conn, "v2_columns_complete"):
            # CLS Distillation Memory columns (idempotent).
            self._ensure_cls_columns(conn)
            self._ensure_episodic_tables(conn)
            self._ensure_episodic_v2_columns(conn)
            self._ensure_cost_counter_tables(conn)
            self._ensure_entity_aggregates_table(conn)
            self._ensure_v3_universal_engram(conn)
            self._ensure_engram_fact_tiering(conn)
            self._ensure_engram_preferences(conn)
            self._ensure_engram_verification(conn)
            self._ensure_engram_cold_archive(conn)
            self._ensure_artifact_tables(conn)
            self._ensure_harness_stream_tables(conn)
            self._ensure_shared_task_tables(conn)
            self._ensure_workspace_hierarchy_tables(conn)
            self._ensure_thread_state_table(conn)
            self._ensure_route_decision_tables(conn)
            self._ensure_narrative_scene_tables(conn)
            return

        # v2 columns on existing canonical tables.
        self._migrate_add_column_conn(conn, "memories", "confidentiality_scope", "TEXT DEFAULT 'work'")
        self._migrate_add_column_conn(conn, "memories", "source_type", "TEXT")
        self._migrate_add_column_conn(conn, "memories", "source_app", "TEXT")
        self._migrate_add_column_conn(conn, "memories", "source_event_id", "TEXT")
        self._migrate_add_column_conn(conn, "memories", "decay_lambda", "REAL DEFAULT 0.12")
        self._migrate_add_column_conn(conn, "memories", "status", "TEXT DEFAULT 'active'")
        self._migrate_add_column_conn(conn, "memories", "importance", "REAL DEFAULT 0.5")
        self._migrate_add_column_conn(conn, "memories", "sensitivity", "TEXT DEFAULT 'normal'")
        self._migrate_add_column_conn(conn, "memories", "namespace", "TEXT DEFAULT 'default'")

        self._migrate_add_column_conn(conn, "scenes", "layer", "TEXT DEFAULT 'sml'")
        self._migrate_add_column_conn(conn, "scenes", "scene_strength", "REAL DEFAULT 1.0")
        self._migrate_add_column_conn(conn, "scenes", "topic_embedding_ref", "TEXT")
        self._migrate_add_column_conn(conn, "scenes", "namespace", "TEXT DEFAULT 'default'")

        self._migrate_add_column_conn(conn, "profiles", "role_bias", "TEXT")
        self._migrate_add_column_conn(conn, "profiles", "profile_summary", "TEXT")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_user_source_event
            ON memories(user_id, source_event_id, namespace, created_at DESC)
            """
        )

        # Backfills.
        conn.execute(
            """
            UPDATE memories
            SET confidentiality_scope = 'work'
            WHERE confidentiality_scope IS NULL OR confidentiality_scope = ''
            """
        )
        conn.execute(
            """
            UPDATE memories
            SET status = 'active'
            WHERE status IS NULL OR status = ''
            """
        )
        conn.execute(
            """
            UPDATE memories
            SET namespace = 'default'
            WHERE namespace IS NULL OR namespace = ''
            """
        )
        conn.execute(
            """
            UPDATE scenes
            SET namespace = 'default'
            WHERE namespace IS NULL OR namespace = ''
            """
        )
        conn.execute(
            """
            UPDATE memories
            SET decay_lambda = 0.12
            WHERE decay_lambda IS NULL
            """
        )
        conn.execute(
            """
            UPDATE memories
            SET importance = COALESCE(
                CASE
                    WHEN json_extract(metadata, '$.importance') IS NOT NULL
                    THEN json_extract(metadata, '$.importance')
                    ELSE importance
                END,
                0.5
            )
            """
        )
        conn.execute(
            """
            UPDATE memories
            SET sensitivity = CASE
                WHEN lower(memory) LIKE '%password%' OR lower(memory) LIKE '%api key%' OR lower(memory) LIKE '%token%'
                    THEN 'secret'
                WHEN lower(memory) LIKE '%health%' OR lower(memory) LIKE '%medical%'
                    THEN 'sensitive'
                WHEN lower(memory) LIKE '%bank%' OR lower(memory) LIKE '%salary%' OR lower(memory) LIKE '%credit card%'
                    THEN 'sensitive'
                ELSE COALESCE(NULLIF(sensitivity, ''), 'normal')
            END
            """
        )

        # Phase 3: Mark column migrations + backfills as complete.
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_columns_complete')"
        )

        # CLS Distillation Memory columns (idempotent).
        self._ensure_cls_columns(conn)

        # Content-hash dedup column (idempotent).
        self._ensure_content_hash_column(conn)

        # Deferred enrichment columns (idempotent).
        self._ensure_deferred_enrichment_columns(conn)
        self._ensure_episodic_tables(conn)
        self._ensure_episodic_v2_columns(conn)
        self._ensure_cost_counter_tables(conn)
        self._ensure_entity_aggregates_table(conn)
        self._ensure_v3_universal_engram(conn)
        self._ensure_engram_fact_tiering(conn)
        self._ensure_engram_preferences(conn)
        self._ensure_engram_verification(conn)
        self._ensure_engram_cold_archive(conn)
        self._ensure_artifact_tables(conn)
        self._ensure_harness_stream_tables(conn)
        self._ensure_shared_task_tables(conn)
        self._ensure_workspace_hierarchy_tables(conn)
        self._ensure_thread_state_table(conn)
        self._ensure_route_decision_tables(conn)
        self._ensure_narrative_scene_tables(conn)

    def _ensure_narrative_scene_tables(self, conn: sqlite3.Connection) -> None:
        """Dhee v1 narrative scene intelligence schema.

        The legacy ``scenes`` table remains the canonical scene table. This
        migration only adds v1 columns and normalized retrieval tables around it.
        """
        scene_columns = {
            "episode_id": "TEXT",
            "agent_id": "TEXT",
            "agent_category": "TEXT",
            "source_app": "TEXT",
            "hero_character_id": "TEXT",
            "hero_focus": "TEXT",
            "setting": "TEXT",
            "intent_type": "TEXT",
            "action_lane": "TEXT DEFAULT 'answer'",
            "action": "TEXT",
            "obstacle": "TEXT",
            "result": "TEXT",
            "emotional_weight": "REAL DEFAULT 0.0",
            "next_possible_moves_json": "TEXT DEFAULT '[]'",
            "outcome": "TEXT",
            "outcome_status": "TEXT DEFAULT 'partial'",
            "story_progress_delta": "TEXT",
            "importance": "REAL DEFAULT 0.5",
            "confidence": "REAL DEFAULT 0.5",
            "visibility_scope": "TEXT DEFAULT 'private'",
            "privacy_class": "TEXT DEFAULT 'user_private'",
            "consolidated_card_id": "TEXT",
            "consolidated_card_json": "TEXT DEFAULT '{}'",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        for column, col_type in scene_columns.items():
            self._migrate_add_column_conn(conn, "scenes", column, col_type)

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS series (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                theme TEXT NOT NULL,
                ultimate_goal TEXT,
                hero_identity TEXT,
                purpose TEXT NOT NULL,
                desired_identity TEXT,
                core_values_json TEXT NOT NULL DEFAULT '[]',
                long_term_conflicts_json TEXT NOT NULL DEFAULT '[]',
                current_active_season TEXT,
                arc_summary TEXT NOT NULL DEFAULT '',
                active_tensions_json TEXT NOT NULL DEFAULT '[]',
                latest_season_signal TEXT,
                deterministic_rollup_json TEXT NOT NULL DEFAULT '{}',
                llm_rollup_json TEXT NOT NULL DEFAULT '{}',
                rollup_model TEXT,
                rollup_prompt_version TEXT,
                rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]',
                rollup_input_hash TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_series_user_status
                ON series(user_id, status, namespace);

            CREATE TABLE IF NOT EXISTS seasons (
                id TEXT PRIMARY KEY,
                series_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                theme TEXT NOT NULL,
                major_goal TEXT,
                dominant_struggle TEXT,
                transformation_expected TEXT,
                open_threads_json TEXT NOT NULL DEFAULT '[]',
                arc_summary TEXT NOT NULL,
                deterministic_rollup_json TEXT NOT NULL DEFAULT '{}',
                llm_rollup_json TEXT NOT NULL DEFAULT '{}',
                rollup_model TEXT,
                rollup_prompt_version TEXT,
                rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]',
                rollup_input_hash TEXT,
                period_start TEXT,
                period_end TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                confidence REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (series_id) REFERENCES series(id)
            );
            CREATE INDEX IF NOT EXISTS idx_seasons_active
                ON seasons(user_id, namespace, series_id, status);

            CREATE TABLE IF NOT EXISTS story_characters (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                namespace TEXT NOT NULL,
                name TEXT NOT NULL,
                character_type TEXT NOT NULL,
                stable_identity_ref TEXT,
                description TEXT NOT NULL,
                skills_json TEXT NOT NULL DEFAULT '[]',
                influence REAL NOT NULL DEFAULT 0.5,
                trust_level REAL NOT NULL DEFAULT 0.5,
                lessons_learned_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_story_char_unique
                ON story_characters(user_id, namespace, lower(name), character_type);

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                series_id TEXT,
                season_id TEXT,
                user_id TEXT NOT NULL,
                local_date TEXT NOT NULL,
                timezone TEXT NOT NULL,
                namespace TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                primary_hero_id TEXT,
                goal TEXT,
                conflict TEXT,
                key_decisions_json TEXT NOT NULL DEFAULT '[]',
                outcome TEXT,
                lesson TEXT,
                unresolved_threads_json TEXT NOT NULL DEFAULT '[]',
                story_progress TEXT,
                category_summaries_json TEXT NOT NULL DEFAULT '{}',
                deterministic_rollup_json TEXT NOT NULL DEFAULT '{}',
                llm_rollup_json TEXT NOT NULL DEFAULT '{}',
                rollup_model TEXT,
                rollup_prompt_version TEXT,
                rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]',
                rollup_input_hash TEXT,
                agent_ids_json TEXT NOT NULL DEFAULT '[]',
                scene_ids_json TEXT NOT NULL DEFAULT '[]',
                open_loops_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, namespace, local_date, timezone),
                FOREIGN KEY (series_id) REFERENCES series(id),
                FOREIGN KEY (season_id) REFERENCES seasons(id)
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_lookup
                ON episodes(user_id, namespace, local_date, timezone);

            CREATE TABLE IF NOT EXISTS episode_characters (
                episode_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                role TEXT NOT NULL,
                relationship_to_hero TEXT,
                salience REAL NOT NULL DEFAULT 0.5,
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (episode_id, character_id, role),
                FOREIGN KEY (episode_id) REFERENCES episodes(id),
                FOREIGN KEY (character_id) REFERENCES story_characters(id)
            );

            CREATE TABLE IF NOT EXISTS scene_characters (
                scene_id TEXT NOT NULL,
                character_id TEXT NOT NULL,
                role TEXT NOT NULL,
                contribution TEXT NOT NULL,
                salience REAL NOT NULL DEFAULT 0.5,
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (scene_id, character_id, role),
                FOREIGN KEY (scene_id) REFERENCES scenes(id),
                FOREIGN KEY (character_id) REFERENCES story_characters(id)
            );

            CREATE TABLE IF NOT EXISTS scene_events (
                id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                evidence_ref TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (scene_id) REFERENCES scenes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_events_scene
                ON scene_events(scene_id, created_at);

            CREATE TABLE IF NOT EXISTS scene_cards (
                id TEXT PRIMARY KEY,
                scene_id TEXT NOT NULL UNIQUE,
                episode_id TEXT,
                user_id TEXT NOT NULL,
                agent_id TEXT,
                agent_category TEXT,
                namespace TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                summary TEXT NOT NULL,
                staleness_policy TEXT NOT NULL DEFAULT 'valid_until_superseded',
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.5,
                reuse_policy TEXT NOT NULL DEFAULT 'private',
                visibility_scope TEXT NOT NULL DEFAULT 'private',
                privacy_class TEXT NOT NULL DEFAULT 'user_private',
                retrieval_tags_json TEXT NOT NULL DEFAULT '[]',
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                do_not_use_for_json TEXT NOT NULL DEFAULT '[]',
                durable_facts_json TEXT NOT NULL DEFAULT '[]',
                decisions_json TEXT NOT NULL DEFAULT '[]',
                procedures_json TEXT NOT NULL DEFAULT '[]',
                success_patterns_json TEXT NOT NULL DEFAULT '[]',
                failure_patterns_json TEXT NOT NULL DEFAULT '[]',
                open_loops_json TEXT NOT NULL DEFAULT '[]',
                entities_json TEXT NOT NULL DEFAULT '[]',
                artifacts_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (scene_id) REFERENCES scenes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_cards_lookup
                ON scene_cards(user_id, namespace, visibility_scope, privacy_class, created_at DESC);

            CREATE TABLE IF NOT EXISTS scene_card_claims (
                id TEXT PRIMARY KEY,
                scene_card_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                claim TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                valid_until TEXT,
                supersedes_json TEXT NOT NULL DEFAULT '[]',
                contradicted_by_json TEXT NOT NULL DEFAULT '[]',
                evidence_refs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (scene_card_id) REFERENCES scene_cards(id)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_card_claims_card
                ON scene_card_claims(scene_card_id, kind);

            CREATE TABLE IF NOT EXISTS scene_categories (
                scene_id TEXT NOT NULL,
                category_id TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL,
                PRIMARY KEY (scene_id, category_id)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_categories_cat
                ON scene_categories(category_id, weight DESC);

            CREATE TABLE IF NOT EXISTS scene_markers (
                scene_id TEXT NOT NULL,
                marker_key TEXT NOT NULL,
                marker_value TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL,
                PRIMARY KEY (scene_id, marker_key, marker_value)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_markers_key_value
                ON scene_markers(marker_key, marker_value);

            CREATE TABLE IF NOT EXISTS scene_edges (
                from_scene_id TEXT NOT NULL,
                to_scene_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (from_scene_id, to_scene_id, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_scene_edges_to
                ON scene_edges(to_scene_id, edge_type);
            """
        )
        for statement in (
            "ALTER TABLE series ADD COLUMN arc_summary TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE series ADD COLUMN active_tensions_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE series ADD COLUMN latest_season_signal TEXT",
            "ALTER TABLE series ADD COLUMN deterministic_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE series ADD COLUMN llm_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE series ADD COLUMN rollup_model TEXT",
            "ALTER TABLE series ADD COLUMN rollup_prompt_version TEXT",
            "ALTER TABLE series ADD COLUMN rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE series ADD COLUMN rollup_input_hash TEXT",
            "ALTER TABLE seasons ADD COLUMN deterministic_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE seasons ADD COLUMN llm_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE seasons ADD COLUMN rollup_model TEXT",
            "ALTER TABLE seasons ADD COLUMN rollup_prompt_version TEXT",
            "ALTER TABLE seasons ADD COLUMN rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE seasons ADD COLUMN rollup_input_hash TEXT",
            "ALTER TABLE episodes ADD COLUMN deterministic_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE episodes ADD COLUMN llm_rollup_json TEXT NOT NULL DEFAULT '{}'",
            "ALTER TABLE episodes ADD COLUMN rollup_model TEXT",
            "ALTER TABLE episodes ADD COLUMN rollup_prompt_version TEXT",
            "ALTER TABLE episodes ADD COLUMN rollup_source_scene_card_ids_json TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE episodes ADD COLUMN rollup_input_hash TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass

        now = _utcnow_iso()
        conn.execute(
            """
            UPDATE scenes
            SET action_lane = COALESCE(NULLIF(action_lane, ''), 'answer'),
                outcome_status = COALESCE(NULLIF(outcome_status, ''), 'partial'),
                visibility_scope = COALESCE(NULLIF(visibility_scope, ''), 'private'),
                privacy_class = COALESCE(NULLIF(privacy_class, ''), 'user_private'),
                importance = COALESCE(importance, scene_strength, strength, 0.5),
                confidence = COALESCE(confidence, 0.5),
                created_at = COALESCE(created_at, start_time, ?),
                updated_at = COALESCE(updated_at, end_time, start_time, ?)
            WHERE action_lane IS NULL OR action_lane = ''
               OR outcome_status IS NULL OR outcome_status = ''
               OR visibility_scope IS NULL OR visibility_scope = ''
               OR privacy_class IS NULL OR privacy_class = ''
               OR importance IS NULL OR confidence IS NULL
               OR created_at IS NULL OR updated_at IS NULL
            """,
            (now, now),
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v5_narrative_scene_v1')"
        )

    def _ensure_engram_verification(self, conn: sqlite3.Connection) -> None:
        """M3 Epistemic Control Loop — ``last_verified_at`` on facts + prefs.

        Distinct from ``last_reaffirmed_at``: verification means the fact
        was cited by a policy / context bundle and the downstream task
        ran without user correction. It is the "still true as of T" signal.
        Reaffirmation is "user restated this"; verification is "world
        didn't push back". Agents use it to decide whether to epistemically
        re-check a load-bearing fact before acting on it.
        """
        if self._is_migration_applied(conn, "v4_engram_verification"):
            return
        for table in ("engram_facts", "engram_preferences"):
            try:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN last_verified_at REAL"
                )
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES "
            "('v4_engram_verification')"
        )

    def _ensure_engram_cold_archive(self, conn: sqlite3.Connection) -> None:
        """Cold-archive tables for lineage-preserving forgetting.

        Movement 3 forgetting never deletes — it moves the row into an
        archive table so ``dhee_why`` / lineage queries can still surface
        it on demand. A JSON blob holds the original row verbatim so
        future schema changes on the live table don't break the archive.
        """
        if self._is_migration_applied(conn, "v4_engram_cold_archive"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS engram_fact_archive (
                id TEXT PRIMARY KEY,
                canonical_key TEXT,
                memory_id TEXT,
                subject TEXT,
                predicate TEXT,
                value TEXT,
                payload TEXT NOT NULL,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_efa_canonical
                ON engram_fact_archive(canonical_key);
            CREATE INDEX IF NOT EXISTS idx_efa_subject_predicate
                ON engram_fact_archive(subject, predicate);

            CREATE TABLE IF NOT EXISTS engram_preference_archive (
                id TEXT PRIMARY KEY,
                canonical_key TEXT,
                memory_id TEXT,
                user_id TEXT,
                subject TEXT,
                topic TEXT,
                value TEXT,
                payload TEXT NOT NULL,
                archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
                reason TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_epa_canonical
                ON engram_preference_archive(canonical_key);
            CREATE INDEX IF NOT EXISTS idx_epa_subject_topic
                ON engram_preference_archive(subject, topic);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES "
            "('v4_engram_cold_archive')"
        )

    def _ensure_engram_preferences(self, conn: sqlite3.Connection) -> None:
        """First-class preference store.

        Preferences aren't just facts — they carry stance (like/dislike/
        neutral), decay differently, and survive supersede as a lineage
        chain rather than as simple overwrite. The same tier vocabulary
        used for facts applies here.
        """
        if self._is_migration_applied(conn, "v4_engram_preferences"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS engram_preferences (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id),
                user_id TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                stance TEXT NOT NULL,
                value TEXT,
                canonical_key TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                tier TEXT DEFAULT 'medium',
                superseded_by_id TEXT,
                reaffirmed_count INTEGER DEFAULT 0,
                last_reaffirmed_at REAL,
                valid_from TEXT,
                valid_until TEXT,
                schema_v INTEGER DEFAULT 1,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_ep_user_canonical
                ON engram_preferences(user_id, canonical_key);
            CREATE INDEX IF NOT EXISTS idx_ep_subject_topic
                ON engram_preferences(subject, topic);
            CREATE INDEX IF NOT EXISTS idx_ep_tier
                ON engram_preferences(tier);
            CREATE INDEX IF NOT EXISTS idx_ep_superseded
                ON engram_preferences(superseded_by_id);
            CREATE INDEX IF NOT EXISTS idx_ep_active_canonical
                ON engram_preferences(canonical_key)
                WHERE superseded_by_id IS NULL;
            CREATE INDEX IF NOT EXISTS idx_ep_memory
                ON engram_preferences(memory_id);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES "
            "('v4_engram_preferences')"
        )

    def _ensure_engram_fact_tiering(self, conn: sqlite3.Connection) -> None:
        """Add M2 tiering + supersede + reaffirmation columns to engram_facts.

        SQLite ``ALTER TABLE ADD COLUMN`` cannot attach a ``CHECK`` constraint
        on an existing table, so tier values are validated at the write-path
        layer (see dhee/core/resolvers.py). All columns are nullable-safe
        with sane defaults so existing rows upgrade cleanly.
        """
        if self._is_migration_applied(conn, "v4_engram_fact_tiering"):
            return

        additions = (
            ("tier", "TEXT DEFAULT 'medium'"),
            ("superseded_by_id", "TEXT"),
            ("reaffirmed_count", "INTEGER DEFAULT 0"),
            ("last_reaffirmed_at", "REAL"),
            ("schema_v", "INTEGER DEFAULT 1"),
        )
        for col, col_type in additions:
            try:
                conn.execute(
                    f"ALTER TABLE engram_facts ADD COLUMN {col} {col_type}"
                )
            except sqlite3.OperationalError:
                pass

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ef_tier ON engram_facts(tier)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ef_superseded "
            "ON engram_facts(superseded_by_id)"
        )
        # Active rows are the overwhelming majority of reads; index them for
        # the "return current, expose chain on demand" retrieval pattern.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ef_active_canonical "
            "ON engram_facts(canonical_key) "
            "WHERE superseded_by_id IS NULL"
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES "
            "('v4_engram_fact_tiering')"
        )

    def _ensure_deferred_enrichment_columns(self, conn: sqlite3.Connection) -> None:
        """Add conversation_context and enrichment_status columns for deferred enrichment."""
        if self._is_migration_applied(conn, "v2_deferred_enrichment"):
            return
        self._migrate_add_column_conn(conn, "memories", "conversation_context", "TEXT")
        self._migrate_add_column_conn(conn, "memories", "enrichment_status", "TEXT DEFAULT 'complete'")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_enrichment_status ON memories(enrichment_status)"
        )
        # Backfill: existing memories are already enriched.
        conn.execute(
            "UPDATE memories SET enrichment_status = 'complete' WHERE enrichment_status IS NULL"
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_deferred_enrichment')"
        )

    def _ensure_episodic_tables(self, conn: sqlite3.Connection) -> None:
        """Add deterministic episodic event index tables."""
        if self._is_migration_applied(conn, "v2_episodic_events"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodic_events (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                session_id TEXT,
                turn_id INTEGER DEFAULT 0,
                actor_id TEXT,
                actor_role TEXT,
                event_time TEXT,
                event_type TEXT NOT NULL,
                canonical_key TEXT NOT NULL,
                value_text TEXT,
                value_num REAL,
                value_unit TEXT,
                currency TEXT,
                normalized_time_start TEXT,
                normalized_time_end TEXT,
                time_granularity TEXT,
                entity_key TEXT,
                value_norm TEXT,
                confidence REAL DEFAULT 0.0,
                superseded_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodic_user_actor_time
                ON episodic_events(user_id, actor_id, event_time DESC);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_key
                ON episodic_events(user_id, canonical_key);
            CREATE INDEX IF NOT EXISTS idx_episodic_memory
                ON episodic_events(memory_id);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_type_time
                ON episodic_events(user_id, event_type, event_time DESC);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_entity_time
                ON episodic_events(user_id, entity_key, normalized_time_start DESC);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_type_entity
                ON episodic_events(user_id, event_type, entity_key);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_episodic_events')"
        )

    def _ensure_episodic_v2_columns(self, conn: sqlite3.Connection) -> None:
        """Add v2 episodic fields for deterministic temporal/entity normalization."""
        if self._is_migration_applied(conn, "v2_episodic_events_v2"):
            return
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episodic_events'"
        ).fetchone()
        if not table_exists:
            return

        # Backward-safe for existing DBs that already created episodic_events.
        try:
            conn.execute("ALTER TABLE episodic_events ADD COLUMN normalized_time_start TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE episodic_events ADD COLUMN normalized_time_end TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE episodic_events ADD COLUMN time_granularity TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE episodic_events ADD COLUMN entity_key TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE episodic_events ADD COLUMN value_norm TEXT")
        except sqlite3.OperationalError:
            pass

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_episodic_user_entity_time
                ON episodic_events(user_id, entity_key, normalized_time_start DESC);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_type_entity
                ON episodic_events(user_id, event_type, entity_key);
            CREATE INDEX IF NOT EXISTS idx_episodic_user_norm_time
                ON episodic_events(user_id, normalized_time_start DESC);
            """
        )

        # Minimal backfill so existing rows stay queryable by new filters.
        conn.execute(
            """
            UPDATE episodic_events
            SET normalized_time_start = COALESCE(normalized_time_start, event_time),
                normalized_time_end = COALESCE(normalized_time_end, event_time),
                time_granularity = COALESCE(NULLIF(time_granularity, ''), 'instant'),
                entity_key = COALESCE(NULLIF(entity_key, ''), NULLIF(actor_id, ''), NULLIF(actor_role, ''), 'unknown'),
                value_norm = COALESCE(NULLIF(value_norm, ''), NULLIF(value_text, ''), CAST(value_num AS TEXT))
            """
        )

        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_episodic_events_v2')"
        )

    def _ensure_cost_counter_tables(self, conn: sqlite3.Connection) -> None:
        """Add write/query cost counter table for unit-economics guardrails."""
        if self._is_migration_applied(conn, "v2_cost_counters"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS cost_counters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                phase TEXT NOT NULL CHECK (phase IN ('write', 'query')),
                llm_calls REAL DEFAULT 0,
                input_tokens REAL DEFAULT 0,
                output_tokens REAL DEFAULT 0,
                embed_calls REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_cost_counters_phase_ts
                ON cost_counters(phase, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_cost_counters_user_ts
                ON cost_counters(user_id, ts DESC);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_cost_counters')"
        )

    def _ensure_entity_aggregates_table(self, conn: sqlite3.Connection) -> None:
        """Add entity_aggregates table for write-time accumulation of counts/sums."""
        if self._is_migration_applied(conn, "v2_entity_aggregates"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entity_aggregates (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                agg_type TEXT NOT NULL,
                value_num REAL DEFAULT 0.0,
                value_unit TEXT,
                item_set TEXT,
                contributing_sessions TEXT,
                contributing_memory_ids TEXT,
                last_updated TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entity_agg_lookup
                ON entity_aggregates(user_id, agg_type, entity_key);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_entity_aggregates')"
        )

    def _ensure_v3_universal_engram(self, conn: sqlite3.Connection) -> None:
        """Add Universal Engram v3 tables (additive — zero breaking changes).

        Tables: engram_context, engram_scenes, engram_facts, engram_links,
                engram_entities, engram_prospective_scenes.
        """
        if self._is_migration_applied(conn, "v3_universal_engram"):
            return
        conn.executescript(
            """
            -- Context anchors (hierarchical retrieval: era -> place -> time -> activity)
            CREATE TABLE IF NOT EXISTS engram_context (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id),
                era TEXT,
                place TEXT,
                place_type TEXT,
                place_detail TEXT,
                time_absolute TEXT,
                time_markers TEXT DEFAULT '[]',
                time_range_start TEXT,
                time_range_end TEXT,
                time_derivation TEXT,
                activity TEXT,
                session_id TEXT,
                session_position INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_ec_era ON engram_context(era);
            CREATE INDEX IF NOT EXISTS idx_ec_place ON engram_context(place);
            CREATE INDEX IF NOT EXISTS idx_ec_time ON engram_context(time_absolute);
            CREATE INDEX IF NOT EXISTS idx_ec_activity ON engram_context(activity);
            CREATE INDEX IF NOT EXISTS idx_ec_era_place ON engram_context(era, place);

            -- Scene snapshots (visual reconstruction)
            CREATE TABLE IF NOT EXISTS engram_scenes (
                memory_id TEXT PRIMARY KEY REFERENCES memories(id),
                setting TEXT,
                people_present TEXT DEFAULT '[]',
                self_state TEXT,
                emotional_tone TEXT,
                sensory_cues TEXT DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_es_setting ON engram_scenes(setting);
            CREATE INDEX IF NOT EXISTS idx_es_tone ON engram_scenes(emotional_tone);

            -- Structured facts (deterministic query resolution)
            CREATE TABLE IF NOT EXISTS engram_facts (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id),
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value TEXT NOT NULL,
                value_numeric REAL,
                value_unit TEXT,
                time TEXT,
                valid_from TEXT,
                valid_until TEXT,
                qualifier TEXT,
                canonical_key TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                is_derived INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_ef_canonical ON engram_facts(canonical_key);
            CREATE INDEX IF NOT EXISTS idx_ef_subject ON engram_facts(subject);
            CREATE INDEX IF NOT EXISTS idx_ef_predicate ON engram_facts(predicate);
            CREATE INDEX IF NOT EXISTS idx_ef_valid ON engram_facts(valid_from, valid_until);
            CREATE INDEX IF NOT EXISTS idx_ef_memory ON engram_facts(memory_id);
            CREATE INDEX IF NOT EXISTS idx_ef_subject_pred ON engram_facts(subject, predicate);

            -- Associative links (causal/temporal/emotional chains between memories)
            CREATE TABLE IF NOT EXISTS engram_links (
                id TEXT PRIMARY KEY,
                source_memory_id TEXT NOT NULL REFERENCES memories(id),
                target_memory_id TEXT,
                target_canonical_key TEXT NOT NULL,
                link_type TEXT NOT NULL,
                direction TEXT NOT NULL,
                qualifier TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_el_source ON engram_links(source_memory_id);
            CREATE INDEX IF NOT EXISTS idx_el_target ON engram_links(target_memory_id);
            CREATE INDEX IF NOT EXISTS idx_el_target_key ON engram_links(target_canonical_key);
            CREATE INDEX IF NOT EXISTS idx_el_type ON engram_links(link_type);

            -- Entity references (knowledge graph backing)
            CREATE TABLE IF NOT EXISTS engram_entities (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id),
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                state TEXT,
                relationships TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_ee_name ON engram_entities(name);
            CREATE INDEX IF NOT EXISTS idx_ee_type ON engram_entities(entity_type);

            -- Prospective scenes (predicted future events — memory-driven anticipation)
            CREATE TABLE IF NOT EXISTS engram_prospective_scenes (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id),
                user_id TEXT NOT NULL,
                predicted_time TEXT,
                trigger_window_hours INTEGER DEFAULT 24,
                event_type TEXT,
                participants TEXT DEFAULT '[]',
                predicted_setting TEXT,
                predicted_needs TEXT DEFAULT '[]',
                relevant_past_scene_ids TEXT DEFAULT '[]',
                status TEXT DEFAULT 'predicted' CHECK (status IN ('predicted', 'triggered', 'occurred', 'cancelled')),
                prediction_basis TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_eps_user_status ON engram_prospective_scenes(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_eps_time ON engram_prospective_scenes(predicted_time);
            CREATE INDEX IF NOT EXISTS idx_eps_event ON engram_prospective_scenes(event_type);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v3_universal_engram')"
        )

    def _ensure_cls_columns(self, conn: sqlite3.Connection) -> None:
        """Add CLS Distillation Memory columns to memories table (idempotent)."""
        if self._is_migration_applied(conn, "v2_cls_columns_complete"):
            return

        self._migrate_add_column_conn(conn, "memories", "memory_type", "TEXT DEFAULT 'semantic'")
        self._migrate_add_column_conn(conn, "memories", "s_fast", "REAL DEFAULT NULL")
        self._migrate_add_column_conn(conn, "memories", "s_mid", "REAL DEFAULT NULL")
        self._migrate_add_column_conn(conn, "memories", "s_slow", "REAL DEFAULT NULL")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type, user_id)"
        )

        # Backfill: set memory_type to 'semantic' for existing memories.
        conn.execute(
            "UPDATE memories SET memory_type = 'semantic' WHERE memory_type IS NULL"
        )

        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v2_cls_columns_complete')"
        )

    def add_memory(self, memory_data: Dict[str, Any], *, log_history: bool = True) -> str:
        memory_id = memory_data.get("id", str(uuid.uuid4()))
        now = _utcnow_iso()
        metadata = self._parse_json_dict(memory_data.get("metadata"), preserve_legacy=True)
        categories = self._parse_json_list(memory_data.get("categories"))
        embedding = self._parse_json_list(memory_data.get("embedding"))
        related_memories = self._parse_json_list(memory_data.get("related_memories"))
        source_memories = self._parse_json_list(memory_data.get("source_memories"))
        source_app = memory_data.get("source_app") or memory_data.get("app_id") or metadata.get("source_app")

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, memory, user_id, agent_id, run_id, app_id,
                    metadata, categories, immutable, expiration_date,
                    created_at, updated_at, layer, strength, access_count,
                    last_accessed, embedding, related_memories, source_memories, tombstone,
                    confidentiality_scope, namespace, source_type, source_app, source_event_id, decay_lambda,
                    status, importance, sensitivity,
                    memory_type, s_fast, s_mid, s_slow, content_hash,
                    conversation_context, enrichment_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    memory_data.get("memory", ""),
                    memory_data.get("user_id"),
                    memory_data.get("agent_id"),
                    memory_data.get("run_id"),
                    memory_data.get("app_id"),
                    json.dumps(metadata),
                    json.dumps(categories),
                    1 if memory_data.get("immutable", False) else 0,
                    memory_data.get("expiration_date"),
                    memory_data.get("created_at", now),
                    memory_data.get("updated_at", now),
                    memory_data.get("layer", "sml"),
                    memory_data.get("strength", 1.0),
                    memory_data.get("access_count", 0),
                    memory_data.get("last_accessed", now),
                    json.dumps(embedding),
                    json.dumps(related_memories),
                    json.dumps(source_memories),
                    1 if memory_data.get("tombstone", False) else 0,
                    memory_data.get("confidentiality_scope", "work"),
                    memory_data.get("namespace", metadata.get("namespace", "default")),
                    memory_data.get("source_type") or metadata.get("source_type") or "mcp",
                    source_app,
                    memory_data.get("source_event_id") or metadata.get("source_event_id"),
                    memory_data.get("decay_lambda", 0.12),
                    memory_data.get("status", "active"),
                    memory_data.get("importance", metadata.get("importance", 0.5)),
                    memory_data.get("sensitivity", metadata.get("sensitivity", "normal")),
                    memory_data.get("memory_type", "semantic"),
                    memory_data.get("s_fast"),
                    memory_data.get("s_mid"),
                    memory_data.get("s_slow"),
                    memory_data.get("content_hash"),
                    memory_data.get("conversation_context"),
                    memory_data.get("enrichment_status", "complete"),
                ),
            )

            # Log within the same transaction -- atomic with the insert.
            if log_history:
                conn.execute(
                    """
                    INSERT INTO memory_history (
                        memory_id, event, old_value, new_value,
                        old_strength, new_strength, old_layer, new_layer
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (memory_id, "ADD", None, memory_data.get("memory"), None, None, None, None),
                )

        return memory_id

    def add_memories_batch(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple memories in a single transaction (atomic).

        Returns list of memory IDs in the same order as input.
        """
        if not memories:
            return []
        now = _utcnow_iso()
        ids: List[str] = []
        insert_rows = []
        history_rows = []

        for memory_data in memories:
            memory_id = memory_data.get("id", str(uuid.uuid4()))
            ids.append(memory_id)
            metadata = self._parse_json_dict(memory_data.get("metadata"), preserve_legacy=True)
            categories = self._parse_json_list(memory_data.get("categories"))
            embedding = self._parse_json_list(memory_data.get("embedding"))
            related_memories = self._parse_json_list(memory_data.get("related_memories"))
            source_memories = self._parse_json_list(memory_data.get("source_memories"))
            source_app = memory_data.get("source_app") or memory_data.get("app_id") or metadata.get("source_app")

            insert_rows.append((
                memory_id,
                memory_data.get("memory", ""),
                memory_data.get("user_id"),
                memory_data.get("agent_id"),
                memory_data.get("run_id"),
                memory_data.get("app_id"),
                json.dumps(metadata),
                json.dumps(categories),
                1 if memory_data.get("immutable", False) else 0,
                memory_data.get("expiration_date"),
                memory_data.get("created_at", now),
                memory_data.get("updated_at", now),
                memory_data.get("layer", "sml"),
                memory_data.get("strength", 1.0),
                memory_data.get("access_count", 0),
                memory_data.get("last_accessed", now),
                json.dumps(embedding),
                json.dumps(related_memories),
                json.dumps(source_memories),
                1 if memory_data.get("tombstone", False) else 0,
                memory_data.get("confidentiality_scope", "work"),
                memory_data.get("namespace", metadata.get("namespace", "default")),
                memory_data.get("source_type") or metadata.get("source_type") or "mcp",
                source_app,
                memory_data.get("source_event_id") or metadata.get("source_event_id"),
                memory_data.get("decay_lambda", 0.12),
                memory_data.get("status", "active"),
                memory_data.get("importance", metadata.get("importance", 0.5)),
                memory_data.get("sensitivity", metadata.get("sensitivity", "normal")),
                memory_data.get("memory_type", "semantic"),
                memory_data.get("s_fast"),
                memory_data.get("s_mid"),
                memory_data.get("s_slow"),
                memory_data.get("conversation_context"),
                memory_data.get("enrichment_status", "complete"),
            ))
            history_rows.append((
                memory_id, "ADD", None, memory_data.get("memory"), None, None, None, None,
            ))

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT INTO memories (
                    id, memory, user_id, agent_id, run_id, app_id,
                    metadata, categories, immutable, expiration_date,
                    created_at, updated_at, layer, strength, access_count,
                    last_accessed, embedding, related_memories, source_memories, tombstone,
                    confidentiality_scope, namespace, source_type, source_app, source_event_id, decay_lambda,
                    status, importance, sensitivity,
                    memory_type, s_fast, s_mid, s_slow,
                    conversation_context, enrichment_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            conn.executemany(
                """
                INSERT INTO memory_history (
                    memory_id, event, old_value, new_value,
                    old_strength, new_strength, old_layer, new_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                history_rows,
            )

        return ids

    def get_memory(self, memory_id: str, include_tombstoned: bool = False) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM memories WHERE id = ?"
        params = [memory_id]
        if not include_tombstoned:
            query += " AND tombstone = 0"

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_memory_by_content_hash(
        self, content_hash: str, user_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """Find an existing memory by content hash (for deduplication)."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE content_hash = ? AND user_id = ? AND tombstone = 0 LIMIT 1",
                (content_hash, user_id),
            ).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_memory_by_source_event(
        self,
        *,
        user_id: str,
        source_event_id: str,
        namespace: Optional[str] = None,
        source_app: Optional[str] = None,
        include_tombstoned: bool = False,
    ) -> Optional[Dict[str, Any]]:
        normalized_event = str(source_event_id or "").strip()
        if not normalized_event:
            return None
        query = """
            SELECT *
            FROM memories
            WHERE user_id = ?
              AND source_event_id = ?
        """
        params: List[Any] = [user_id, normalized_event]
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        if source_app:
            query += " AND source_app = ?"
            params.append(source_app)
        if not include_tombstoned:
            query += " AND tombstone = 0"
        query += " ORDER BY created_at DESC LIMIT 1"

        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_dict(row)
        return None

    def get_all_memories(
        self,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        app_id: Optional[str] = None,
        layer: Optional[str] = None,
        namespace: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_strength: float = 0.0,
        include_tombstoned: bool = False,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM memories WHERE strength >= ?"
        params: List[Any] = [min_strength]

        if not include_tombstoned:
            query += " AND tombstone = 0"
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        if app_id:
            query += " AND app_id = ?"
            params.append(app_id)
        if layer:
            query += " AND layer = ?"
            params.append(layer)
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        if created_after:
            query += " AND created_at >= ?"
            params.append(created_after)
        if created_before:
            query += " AND created_at <= ?"
            params.append(created_before)

        query += " ORDER BY strength DESC"

        # Apply SQL-level LIMIT to avoid fetching unbounded rows into memory.
        if limit is not None and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        set_clauses = []
        params: List[Any] = []
        for key, value in updates.items():
            if key not in VALID_MEMORY_COLUMNS:
                raise ValueError(f"Invalid memory column: {key!r}")
            if key == "metadata":
                value = json.dumps(self._parse_json_dict(value, preserve_legacy=True))
            elif key in {"categories", "embedding", "related_memories", "source_memories"}:
                value = json.dumps(self._parse_json_list(value))
            set_clauses.append(f"{key} = ?")
            params.append(value)

        set_clauses.append("updated_at = ?")
        params.append(_utcnow_iso())
        params.append(memory_id)

        with self._get_connection() as conn:
            # Read old values and update in a single transaction.
            old_row = conn.execute(
                "SELECT memory, strength, layer FROM memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            if not old_row:
                return False

            conn.execute(
                f"UPDATE memories SET {', '.join(set_clauses)} WHERE id = ?",
                params,
            )

            # Log within the same transaction.
            conn.execute(
                """
                INSERT INTO memory_history (
                    memory_id, event, old_value, new_value,
                    old_strength, new_strength, old_layer, new_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    "UPDATE",
                    old_row["memory"],
                    updates.get("memory"),
                    old_row["strength"],
                    updates.get("strength"),
                    old_row["layer"],
                    updates.get("layer"),
                ),
            )
        return True

    def _table_exists_conn(self, conn: sqlite3.Connection, table_name: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        return row is not None

    def _delete_memory_query_artifacts_conn(
        self,
        conn: sqlite3.Connection,
        memory_ids: List[str],
    ) -> None:
        active_memory_ids = [memory_id for memory_id in memory_ids if memory_id]
        if not active_memory_ids:
            return

        placeholders = ",".join("?" for _ in active_memory_ids)
        memory_tables = (
            "engram_context",
            "engram_scenes",
            "engram_facts",
            "engram_entities",
            "engram_prospective_scenes",
            "episodic_events",
        )
        for table_name in memory_tables:
            if not self._table_exists_conn(conn, table_name):
                continue
            conn.execute(
                f"DELETE FROM {table_name} WHERE memory_id IN ({placeholders})",
                active_memory_ids,
            )

        if self._table_exists_conn(conn, "engram_links"):
            link_params = active_memory_ids + active_memory_ids
            conn.execute(
                f"DELETE FROM engram_links WHERE source_memory_id IN ({placeholders}) "
                f"OR target_memory_id IN ({placeholders})",
                link_params,
            )

    def delete_memory(self, memory_id: str, use_tombstone: bool = True) -> bool:
        if use_tombstone:
            success = self.update_memory(memory_id, {"tombstone": 1})
            if not success:
                return False
            with self._get_connection() as conn:
                self._delete_memory_query_artifacts_conn(conn, [memory_id])
            return True
        with self._get_connection() as conn:
            self._delete_memory_query_artifacts_conn(conn, [memory_id])
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self._log_event(memory_id, "DELETE")
        return True

    def increment_access(self, memory_id: str) -> None:
        now = _utcnow_iso()
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
                """,
                (now, memory_id),
            )

    # Phase 2: Batch operations to eliminate N+1 queries in search.

    def get_memories_bulk(self, memory_ids: List[str], include_tombstoned: bool = False) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple memories by ID in a single query. Returns {id: memory_dict}."""
        if not memory_ids:
            return {}
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            query = f"SELECT * FROM memories WHERE id IN ({placeholders})"
            if not include_tombstoned:
                query += " AND tombstone = 0"
            rows = conn.execute(query, memory_ids).fetchall()
            return {row["id"]: self._row_to_dict(row) for row in rows}

    def increment_access_bulk(self, memory_ids: List[str]) -> None:
        """Increment access count for multiple memories in a single transaction."""
        if not memory_ids:
            return
        now = _utcnow_iso()
        with self._get_connection() as conn:
            placeholders = ",".join("?" for _ in memory_ids)
            conn.execute(
                f"""
                UPDATE memories
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id IN ({placeholders})
                """,
                [now] + list(memory_ids),
            )

    def update_strength_bulk(self, updates: Dict[str, float]) -> None:
        """Batch-update strength for multiple memories. updates = {memory_id: new_strength}."""
        if not updates:
            return
        now = _utcnow_iso()
        with self._get_connection() as conn:
            conn.executemany(
                "UPDATE memories SET strength = ?, updated_at = ? WHERE id = ?",
                [(strength, now, memory_id) for memory_id, strength in updates.items()],
            )

    def get_pending_enrichment(self, user_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return pending and retryable failed enrichment rows, oldest first."""
        fetch_limit = max(1, int(limit)) * 4
        with self._get_connection() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE enrichment_status IN ('pending', 'failed') AND user_id = ? "
                    "AND tombstone = 0 ORDER BY created_at ASC LIMIT ?",
                    (user_id, fetch_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE enrichment_status IN ('pending', 'failed') "
                    "AND tombstone = 0 ORDER BY created_at ASC LIMIT ?",
                    (fetch_limit,),
                ).fetchall()
            retryable: List[Dict[str, Any]] = []
            for row in rows:
                memory = self._row_to_dict(row)
                status = str(memory.get("enrichment_status") or "").lower()
                metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
                attempts_raw = metadata.get("enrichment_attempts", 0)
                try:
                    attempts = int(attempts_raw or 0)
                except (TypeError, ValueError):
                    attempts = 0
                if status == "pending" or attempts < 3:
                    retryable.append(memory)
                if len(retryable) >= max(1, int(limit)):
                    break
            return retryable

    def requeue_complete_without_engram_facts(
        self,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Reopen complete semantic memories that have no linked engram facts."""
        try:
            with self._get_connection() as conn:
                conn.execute("SELECT 1 FROM engram_facts LIMIT 0")
        except Exception:
            return {"requeued_count": 0, "reason": "engram_facts_unavailable"}

        params: List[Any] = []
        query = (
            "SELECT m.* FROM memories m "
            "LEFT JOIN engram_facts f ON f.memory_id = m.id "
            "WHERE m.tombstone = 0 "
            "AND m.enrichment_status = 'complete' "
            "AND COALESCE(m.memory_type, 'semantic') != 'operational_event' "
            "GROUP BY m.id HAVING COUNT(f.id) = 0 "
            "ORDER BY m.created_at ASC LIMIT ?"
        )
        if user_id:
            query = (
                "SELECT m.* FROM memories m "
                "LEFT JOIN engram_facts f ON f.memory_id = m.id "
                "WHERE m.tombstone = 0 "
                "AND m.enrichment_status = 'complete' "
                "AND COALESCE(m.memory_type, 'semantic') != 'operational_event' "
                "AND m.user_id = ? "
                "GROUP BY m.id HAVING COUNT(f.id) = 0 "
                "ORDER BY m.created_at ASC LIMIT ?"
            )
            params.append(user_id)
        params.append(max(1, int(limit)))

        rows: List[Dict[str, Any]] = []
        with self._get_connection() as conn:
            for row in conn.execute(query, params).fetchall():
                rows.append(self._row_to_dict(row))

        updates: List[Dict[str, Any]] = []
        for memory in rows:
            metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
            metadata = dict(metadata)
            metadata["enrichment_attempts"] = 0
            metadata.pop("last_enrichment_error", None)
            metadata["reextract_queued_at"] = _utcnow_iso()
            updates.append(
                {
                    "id": memory["id"],
                    "metadata": metadata,
                    "enrichment_status": "pending",
                }
            )
        self.update_enrichment_bulk(updates)
        return {
            "requeued_count": len(updates),
            "memory_ids": [item["id"] for item in updates],
        }

    def update_enrichment_status(self, memory_id: str, status: str) -> None:
        """Mark a memory's enrichment_status (e.g. 'complete')."""
        now = _utcnow_iso()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE memories SET enrichment_status = ?, updated_at = ? WHERE id = ?",
                (status, now, memory_id),
            )

    def update_enrichment_bulk(self, updates: List[Dict[str, Any]]) -> None:
        """Batch-update enrichment results for multiple memories.

        Each dict: {id, metadata, categories, enrichment_status}.
        """
        if not updates:
            return
        now = _utcnow_iso()
        with self._get_connection() as conn:
            for upd in updates:
                mid = upd["id"]
                sets = ["updated_at = ?"]
                params: list = [now]
                if "metadata" in upd:
                    sets.append("metadata = ?")
                    params.append(json.dumps(self._parse_json_dict(upd["metadata"], preserve_legacy=True)))
                if "categories" in upd:
                    sets.append("categories = ?")
                    params.append(json.dumps(self._parse_json_list(upd["categories"])))
                if "enrichment_status" in upd:
                    sets.append("enrichment_status = ?")
                    params.append(upd["enrichment_status"])
                params.append(mid)
                conn.execute(
                    f"UPDATE memories SET {', '.join(sets)} WHERE id = ?",
                    params,
                )

    _MEMORY_JSON_FIELDS = ("metadata", "categories", "related_memories", "source_memories")

    def _row_to_dict(self, row: sqlite3.Row, *, skip_embedding: bool = False) -> Dict[str, Any]:
        data = dict(row)
        if "metadata" in data:
            data["metadata"] = self._parse_json_dict(data.get("metadata"), preserve_legacy=True)
        for key in ("categories", "related_memories", "source_memories"):
            if key in data:
                data[key] = self._parse_json_list(data.get(key))
        # Embedding is the largest JSON field (~30-50KB for 3072-dim vectors).
        # Skip deserialization when the caller doesn't need it.
        if skip_embedding:
            data.pop("embedding", None)
        elif "embedding" in data:
            data["embedding"] = self._parse_json_list(data.get("embedding"))
        data["immutable"] = bool(data.get("immutable", 0))
        data["tombstone"] = bool(data.get("tombstone", 0))
        return data

    def _log_event(self, memory_id: str, event: str, **kwargs: Any) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memory_history (
                    memory_id, event, old_value, new_value,
                    old_strength, new_strength, old_layer, new_layer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    event,
                    kwargs.get("old_value"),
                    kwargs.get("new_value"),
                    kwargs.get("old_strength"),
                    kwargs.get("new_strength"),
                    kwargs.get("old_layer"),
                    kwargs.get("new_layer"),
                ),
            )

    def log_event(self, memory_id: str, event: str, **kwargs: Any) -> None:
        """Public wrapper for logging custom events like DECAY or FUSE."""
        self._log_event(memory_id, event, **kwargs)

    def get_history(self, memory_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM memory_history WHERE memory_id = ? ORDER BY timestamp DESC",
                (memory_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_decay(self, decayed: int, forgotten: int, promoted: int, storage_before_mb: Optional[float] = None, storage_after_mb: Optional[float] = None) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO decay_log (memories_decayed, memories_forgotten, memories_promoted, storage_before_mb, storage_after_mb)
                VALUES (?, ?, ?, ?, ?)
                """,
                (decayed, forgotten, promoted, storage_before_mb, storage_after_mb),
            )

    def purge_tombstoned(self) -> int:
        """Permanently delete all tombstoned memories. This is IRREVERSIBLE."""
        with self._get_connection() as conn:
            # Log what will be purged before deletion for audit trail.
            rows = conn.execute(
                "SELECT id, user_id, memory FROM memories WHERE tombstone = 1"
            ).fetchall()
            count = len(rows)
            if count > 0:
                ids = [row["id"] for row in rows]
                logger.warning(
                    "purge_tombstoned: permanently deleting %d memories: %s",
                    count,
                    ids,
                )
                for row in rows:
                    conn.execute(
                        """INSERT INTO memory_history (memory_id, event, old_value, new_value,
                           old_strength, new_strength, old_layer, new_layer)
                           VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL)""",
                        (row["id"], "PURGE", row["memory"]),
                    )
                self._delete_memory_query_artifacts_conn(conn, ids)
                conn.execute("DELETE FROM memories WHERE tombstone = 1")
            return count

    # Domain- and analytics-specific APIs live in focused mixins.


# Backward compatibility alias
# Keep SQLiteManager mapped to the full-capability manager so legacy call sites
# that expect category/scene/profile APIs continue to work.
SQLiteManager = FullSQLiteManager
