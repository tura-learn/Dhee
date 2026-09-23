import hashlib
import json
import sqlite3
import uuid
from datetime import timedelta
from typing import Any, Dict, List, Optional

from .sqlite_common import _utcnow, _utcnow_iso


class SQLiteAnalyticsMixin:
    """Distillation, episodic indexing, counters, and aggregate APIs."""

    @staticmethod
    def _add_column_if_missing(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        col_type: str,
    ) -> None:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        except sqlite3.OperationalError:
            pass

    def _ensure_harness_stream_tables(self, conn: sqlite3.Connection) -> None:
        """Track incremental harness stream ingestion cursors."""
        if self._is_migration_applied(conn, "v4_harness_stream_cursors"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS harness_stream_cursors (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                harness TEXT NOT NULL,
                stream_id TEXT NOT NULL,
                byte_offset INTEGER NOT NULL DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, harness, stream_id)
            );
            CREATE INDEX IF NOT EXISTS idx_harness_stream_cursors_user
                ON harness_stream_cursors(user_id, harness, updated_at DESC);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v4_harness_stream_cursors')"
        )

    def _ensure_shared_task_tables(self, conn: sqlite3.Connection) -> None:
        """Add the repo/task-scoped collaboration bus tables."""
        if self._is_migration_applied(conn, "v4_shared_tasks"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS shared_tasks (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                repo TEXT,
                workspace_id TEXT,
                folder_path TEXT,
                title TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'paused', 'completed', 'closed')),
                created_by TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                closed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_shared_tasks_user_updated
                ON shared_tasks(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_tasks_repo
                ON shared_tasks(user_id, repo, workspace_id, updated_at DESC);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_shared_tasks_one_active
                ON shared_tasks(user_id, repo, workspace_id)
                WHERE status = 'active';

            CREATE TABLE IF NOT EXISTS shared_task_results (
                id TEXT PRIMARY KEY,
                result_key TEXT NOT NULL UNIQUE,
                shared_task_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                repo TEXT,
                workspace_id TEXT,
                folder_path TEXT,
                packet_kind TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                result_status TEXT NOT NULL DEFAULT 'completed'
                    CHECK (result_status IN ('in_flight', 'completed', 'abandoned')),
                source_event_id TEXT,
                source_path TEXT,
                ptr TEXT,
                artifact_id TEXT,
                digest TEXT,
                metadata TEXT DEFAULT '{}',
                session_id TEXT,
                thread_id TEXT,
                harness TEXT,
                agent_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_shared_task_results_task_created
                ON shared_task_results(shared_task_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_task_results_path
                ON shared_task_results(shared_task_id, source_path, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_task_results_status
                ON shared_task_results(shared_task_id, result_status, updated_at DESC);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v4_shared_tasks')"
        )

    def _ensure_project_graph_tables(self, conn: sqlite3.Connection) -> None:
        """Add project/workspace/session graph tables and extend shared-task scope."""
        if self._is_migration_applied(conn, "v5_project_graph"):
            return
        self._ensure_shared_task_tables(conn)
        self._ensure_route_decision_tables(conn)
        conn.executescript(
            """
            DROP INDEX IF EXISTS idx_shared_tasks_one_active;

            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_projects_user_updated
                ON projects(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS project_workspaces (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                workspace_path TEXT NOT NULL,
                label TEXT NOT NULL,
                folder_path TEXT,
                is_primary INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, workspace_path)
            );
            CREATE INDEX IF NOT EXISTS idx_project_workspaces_project
                ON project_workspaces(project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_project_workspaces_user_path
                ON project_workspaces(user_id, workspace_path);

            CREATE TABLE IF NOT EXISTS agent_sessions (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                workspace_id TEXT,
                user_id TEXT NOT NULL,
                runtime_id TEXT NOT NULL,
                native_session_id TEXT,
                task_id TEXT,
                title TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'recent',
                model TEXT,
                cwd TEXT,
                rollout_path TEXT,
                permission_mode TEXT,
                started_at TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}',
                UNIQUE(runtime_id, native_session_id)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_workspace
                ON agent_sessions(workspace_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_project
                ON agent_sessions(project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_sessions_runtime_native
                ON agent_sessions(runtime_id, native_session_id);

            CREATE TABLE IF NOT EXISTS session_assets (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                workspace_id TEXT,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                artifact_id TEXT,
                storage_path TEXT NOT NULL,
                name TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_session_assets_session
                ON session_assets(session_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_session_assets_workspace
                ON session_assets(workspace_id, updated_at DESC);
            """
        )

        for table, column, col_type in [
            ("shared_tasks", "project_id", "TEXT"),
            ("shared_tasks", "session_id", "TEXT"),
            ("shared_tasks", "thread_id", "TEXT"),
            ("shared_tasks", "runtime_id", "TEXT"),
            ("shared_tasks", "native_session_id", "TEXT"),
            ("shared_task_results", "project_id", "TEXT"),
            ("route_decisions", "project_id", "TEXT"),
            ("route_decisions", "session_id", "TEXT"),
            ("route_decisions", "thread_id", "TEXT"),
            ("route_decisions", "runtime_id", "TEXT"),
            ("route_decisions", "agent_id", "TEXT"),
        ]:
            self._add_column_if_missing(conn, table, column, col_type)

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_shared_tasks_project
                ON shared_tasks(user_id, project_id, workspace_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_tasks_native_session
                ON shared_tasks(user_id, runtime_id, native_session_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_tasks_session_thread
                ON shared_tasks(user_id, session_id, thread_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_shared_task_results_project
                ON shared_task_results(project_id, workspace_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_route_decisions_runtime_session
                ON route_decisions(runtime_id, session_id, created_at DESC);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v5_project_graph')"
        )

    def _ensure_workspace_hierarchy_tables(self, conn: sqlite3.Connection) -> None:
        """Add workspace-first hierarchy, path scope rules and collaboration line tables."""
        applied_v6 = self._is_migration_applied(conn, "v6_workspace_hierarchy")
        if applied_v6:
            self._ensure_workspace_line_dedup_migration(conn)
            return
        self._ensure_project_graph_tables(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                root_path TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_workspaces_user_updated
                ON workspaces(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_mounts (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                mount_path TEXT NOT NULL,
                label TEXT,
                is_primary INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, mount_path)
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_mounts_workspace
                ON workspace_mounts(workspace_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_projects (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                default_runtime TEXT DEFAULT 'codex',
                color TEXT,
                icon TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_projects_workspace
                ON workspace_projects(workspace_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_project_scope_rules (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                path_prefix TEXT NOT NULL,
                label TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, path_prefix)
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_project_scope_rules_project
                ON workspace_project_scope_rules(project_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS workspace_line_messages (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                project_id TEXT,
                target_project_id TEXT,
                user_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                session_id TEXT,
                task_id TEXT,
                message_kind TEXT NOT NULL,
                title TEXT,
                body TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_line_messages_workspace
                ON workspace_line_messages(workspace_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_workspace_line_messages_project
                ON workspace_line_messages(project_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_workspace_line_messages_target_project
                ON workspace_line_messages(target_project_id, created_at DESC);
            """
        )

        project_rows = conn.execute("SELECT * FROM projects").fetchall()
        legacy_project_ids = {str(row["id"]) for row in project_rows}
        workspace_rows = conn.execute("SELECT * FROM project_workspaces").fetchall()
        legacy_workspace_ids = {str(row["id"]) for row in workspace_rows}

        for row in project_rows:
            workspace_id = str(row["id"])
            metadata = self._parse_json_value(row["metadata"], {})
            root_path = str(metadata.get("root_repo") or "").strip() or None
            conn.execute(
                """
                INSERT OR IGNORE INTO workspaces (
                    id, user_id, name, description, root_path, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace_id,
                    row["user_id"],
                    row["name"],
                    row["description"],
                    root_path,
                    row["metadata"] or "{}",
                    row["created_at"],
                    row["updated_at"],
                ),
            )

        for row in workspace_rows:
            workspace_id = str(row["project_id"] or "")
            project_id = str(row["id"] or "")
            if not workspace_id or not project_id:
                continue
            metadata = self._parse_json_value(row["metadata"], {})
            conn.execute(
                """
                INSERT OR IGNORE INTO workspace_projects (
                    id, workspace_id, user_id, name, description, default_runtime, color, icon,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    workspace_id,
                    row["user_id"],
                    row["label"],
                    metadata.get("description"),
                    metadata.get("default_runtime") or "codex",
                    metadata.get("color"),
                    metadata.get("icon"),
                    row["metadata"] or "{}",
                    row["created_at"],
                    row["updated_at"],
                ),
            )

            primary_path = str(row["workspace_path"] or "").strip()
            if primary_path:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_mounts (
                        id, workspace_id, user_id, mount_path, label, is_primary,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 1, '{}', ?, ?)
                    """,
                    (
                        f"{workspace_id}:{primary_path}",
                        workspace_id,
                        row["user_id"],
                        primary_path,
                        row["label"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_project_scope_rules (
                        id, project_id, user_id, path_prefix, label, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        f"{project_id}:{primary_path}",
                        project_id,
                        row["user_id"],
                        primary_path,
                        row["folder_path"] or row["label"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )

            extra_folders = metadata.get("folders") or []
            for item in extra_folders:
                if isinstance(item, str):
                    mount_path = item
                    label = ""
                elif isinstance(item, dict):
                    mount_path = str(item.get("path") or "").strip()
                    label = str(item.get("label") or "").strip()
                else:
                    continue
                if not mount_path:
                    continue
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_mounts (
                        id, workspace_id, user_id, mount_path, label, is_primary,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, 0, '{}', ?, ?)
                    """,
                    (
                        f"{workspace_id}:{mount_path}",
                        workspace_id,
                        row["user_id"],
                        mount_path,
                        label or row["label"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_project_scope_rules (
                        id, project_id, user_id, path_prefix, label, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
                    """,
                    (
                        f"{project_id}:{mount_path}",
                        project_id,
                        row["user_id"],
                        mount_path,
                        label or row["folder_path"] or row["label"],
                        row["created_at"],
                        row["updated_at"],
                    ),
                )

        for table in ("agent_sessions", "shared_tasks", "shared_task_results", "session_assets", "route_decisions"):
            rows = conn.execute(
                f"SELECT id, project_id, workspace_id FROM {table}"
            ).fetchall()
            for row in rows:
                current_project_id = str(row["project_id"] or "").strip()
                current_workspace_id = str(row["workspace_id"] or "").strip()
                next_workspace_id = current_workspace_id
                next_project_id = current_project_id
                if current_project_id in legacy_project_ids:
                    next_workspace_id = current_project_id
                if current_workspace_id in legacy_workspace_ids:
                    next_project_id = current_workspace_id
                if next_workspace_id == current_workspace_id and next_project_id == current_project_id:
                    continue
                conn.execute(
                    f"UPDATE {table} SET project_id = ?, workspace_id = ? WHERE id = ?",
                    (next_project_id or None, next_workspace_id or None, row["id"]),
                )

        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v6_workspace_hierarchy')"
        )
        self._ensure_workspace_line_dedup_migration(conn)

    def _ensure_workspace_line_dedup_migration(self, conn: sqlite3.Connection) -> None:
        """Idempotency key on workspace_line_messages so agent emitters can retry safely.

        Partial unique index — dedup_key is NULL for human-composed messages
        (they never collide), non-null for agent-emitted ones where we want
        at-most-once semantics per (workspace, event).
        """
        if self._is_migration_applied(conn, "v7_workspace_line_dedup"):
            self._ensure_project_assets_migration(conn)
            return
        self._add_column_if_missing(conn, "workspace_line_messages", "dedup_key", "TEXT")
        try:
            conn.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_workspace_line_messages_dedup
                    ON workspace_line_messages(workspace_id, dedup_key)
                    WHERE dedup_key IS NOT NULL;
                """
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v7_workspace_line_dedup')"
        )
        self._ensure_project_assets_migration(conn)

    def _ensure_project_assets_migration(self, conn: sqlite3.Connection) -> None:
        """Workspace/project-scoped asset store.

        Distinct from ``session_assets`` which lives and dies with a single
        agent session. ``project_assets`` rows outlive sessions — they are
        the "drop a spec PDF / design export / schema doc into the project,
        every agent in the workspace sees it forever" feature from the
        pitch deck. Deduped by SHA-256 within a (workspace, project)
        scope.
        """
        if self._is_migration_applied(conn, "v8_project_assets"):
            self._ensure_workspace_line_receipts_migration(conn)
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS project_assets (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                project_id TEXT,
                user_id TEXT NOT NULL,
                artifact_id TEXT,
                folder TEXT,
                storage_path TEXT NOT NULL,
                name TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER DEFAULT 0,
                checksum TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_project_assets_workspace
                ON project_assets(workspace_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_project_assets_project
                ON project_assets(project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_project_assets_storage_path
                ON project_assets(storage_path);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_project_assets_checksum_scope
                ON project_assets(workspace_id, COALESCE(project_id, ''), checksum)
                WHERE checksum IS NOT NULL;
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v8_project_assets')"
        )
        self._ensure_workspace_line_receipts_migration(conn)

    def _ensure_workspace_line_receipts_migration(self, conn: sqlite3.Connection) -> None:
        """Track which active agent consumers have seen live line messages."""
        if self._is_migration_applied(conn, "v9_workspace_line_receipts"):
            return
        self._ensure_project_assets_migration_without_receipts(conn)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS workspace_line_receipts (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                consumer_id TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                read_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(workspace_id, message_id, user_id, consumer_id)
            );
            CREATE INDEX IF NOT EXISTS idx_workspace_line_receipts_consumer
                ON workspace_line_receipts(workspace_id, user_id, consumer_id, read_at DESC);
            CREATE INDEX IF NOT EXISTS idx_workspace_line_receipts_message
                ON workspace_line_receipts(message_id, consumer_id);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v9_workspace_line_receipts')"
        )

    def _ensure_project_assets_migration_without_receipts(self, conn: sqlite3.Connection) -> None:
        """Compatibility helper for v9 bootstrap without recursive migration calls."""
        if self._is_migration_applied(conn, "v8_project_assets"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS project_assets (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                project_id TEXT,
                user_id TEXT NOT NULL,
                artifact_id TEXT,
                folder TEXT,
                storage_path TEXT NOT NULL,
                name TEXT NOT NULL,
                mime_type TEXT,
                size_bytes INTEGER DEFAULT 0,
                checksum TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_project_assets_workspace
                ON project_assets(workspace_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_project_assets_project
                ON project_assets(project_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_project_assets_storage_path
                ON project_assets(storage_path);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_project_assets_checksum_scope
                ON project_assets(workspace_id, COALESCE(project_id, ''), checksum)
                WHERE checksum IS NOT NULL;
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v8_project_assets')"
        )

    def _ensure_thread_state_table(self, conn: sqlite3.Connection) -> None:
        """Add lightweight thread-native continuity state."""
        if self._is_migration_applied(conn, "v4_thread_state"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS thread_states (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                repo TEXT,
                workspace_id TEXT,
                folder_path TEXT,
                status TEXT DEFAULT 'active',
                summary TEXT,
                current_goal TEXT,
                current_step TEXT,
                session_id TEXT,
                handoff_session_id TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, thread_id)
            );
            CREATE INDEX IF NOT EXISTS idx_thread_states_user_updated
                ON thread_states(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_thread_states_repo
                ON thread_states(user_id, repo, updated_at DESC);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v4_thread_state')"
        )

    def _ensure_route_decision_tables(self, conn: sqlite3.Connection) -> None:
        """Add critical-surface route decision table."""
        if self._is_migration_applied(conn, "v4_route_decisions"):
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS route_decisions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                source_event_id TEXT,
                packet_kind TEXT NOT NULL,
                route TEXT NOT NULL,
                depth_score REAL DEFAULT 0.0,
                semantic_fit REAL DEFAULT 0.0,
                structural_fit REAL DEFAULT 0.0,
                novelty REAL DEFAULT 0.0,
                confidence REAL DEFAULT 0.0,
                locality_scope TEXT DEFAULT 'global',
                workspace_id TEXT,
                folder_path TEXT,
                source_path TEXT,
                token_delta INTEGER DEFAULT 0,
                outcome_alignment REAL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_route_decisions_user_created
                ON route_decisions(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_route_decisions_packet_kind
                ON route_decisions(packet_kind, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_route_decisions_route
                ON route_decisions(route, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_route_decisions_source_path
                ON route_decisions(source_path);
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations (version) VALUES ('v4_route_decisions')"
        )

    def _thread_state_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _stream_cursor_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def get_harness_stream_cursor(
        self,
        *,
        user_id: str = "default",
        harness: str,
        stream_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_harness_stream_tables(conn)
            row = conn.execute(
                """
                SELECT *
                FROM harness_stream_cursors
                WHERE user_id = ? AND harness = ? AND stream_id = ?
                LIMIT 1
                """,
                (user_id, harness, stream_id),
            ).fetchone()
        if not row:
            return None
        return self._stream_cursor_row_to_dict(row)

    def upsert_harness_stream_cursor(self, cursor: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(cursor.get("user_id") or "default")
        harness = str(cursor.get("harness") or "").strip()
        stream_id = str(cursor.get("stream_id") or "").strip()
        if not harness:
            raise ValueError("harness is required")
        if not stream_id:
            raise ValueError("stream_id is required")
        try:
            byte_offset = max(0, int(cursor.get("byte_offset", 0)))
        except (TypeError, ValueError):
            byte_offset = 0
        metadata = json.dumps(cursor.get("metadata") or {})
        now = _utcnow_iso()

        with self._get_connection() as conn:
            self._ensure_harness_stream_tables(conn)
            existing = conn.execute(
                """
                SELECT id, created_at
                FROM harness_stream_cursors
                WHERE user_id = ? AND harness = ? AND stream_id = ?
                LIMIT 1
                """,
                (user_id, harness, stream_id),
            ).fetchone()
            if existing:
                cursor_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE harness_stream_cursors
                    SET byte_offset = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (byte_offset, metadata, now, cursor_id),
                )
            else:
                cursor_id = str(cursor.get("id") or uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO harness_stream_cursors (
                        id, user_id, harness, stream_id, byte_offset,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor_id,
                        user_id,
                        harness,
                        stream_id,
                        byte_offset,
                        metadata,
                        now,
                        now,
                    ),
                )
        return self.get_harness_stream_cursor(
            user_id=user_id,
            harness=harness,
            stream_id=stream_id,
        ) or {}

    def upsert_thread_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update a live thread continuity state."""
        user_id = str(state.get("user_id") or "default")
        thread_id = str(state.get("thread_id") or "").strip()
        if not thread_id:
            raise ValueError("thread_id is required")
        now = _utcnow_iso()
        metadata = json.dumps(state.get("metadata") or {})

        with self._get_connection() as conn:
            self._ensure_thread_state_table(conn)
            existing = conn.execute(
                """
                SELECT id, created_at FROM thread_states
                WHERE user_id = ? AND thread_id = ?
                LIMIT 1
                """,
                (user_id, thread_id),
            ).fetchone()

            state_id = str(state.get("id") or (existing["id"] if existing else uuid.uuid4()))
            created_at = str(state.get("created_at") or (existing["created_at"] if existing else now))
            if existing:
                conn.execute(
                    """
                    UPDATE thread_states
                    SET repo = ?, workspace_id = ?, folder_path = ?, status = ?,
                        summary = ?, current_goal = ?, current_step = ?,
                        session_id = ?, handoff_session_id = ?, metadata = ?,
                        updated_at = ?
                    WHERE user_id = ? AND thread_id = ?
                    """,
                    (
                        state.get("repo"),
                        state.get("workspace_id"),
                        state.get("folder_path"),
                        state.get("status", "active"),
                        state.get("summary"),
                        state.get("current_goal"),
                        state.get("current_step"),
                        state.get("session_id"),
                        state.get("handoff_session_id"),
                        metadata,
                        now,
                        user_id,
                        thread_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO thread_states (
                        id, user_id, thread_id, repo, workspace_id, folder_path,
                        status, summary, current_goal, current_step,
                        session_id, handoff_session_id, metadata,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        state_id,
                        user_id,
                        thread_id,
                        state.get("repo"),
                        state.get("workspace_id"),
                        state.get("folder_path"),
                        state.get("status", "active"),
                        state.get("summary"),
                        state.get("current_goal"),
                        state.get("current_step"),
                        state.get("session_id"),
                        state.get("handoff_session_id"),
                        metadata,
                        created_at,
                        now,
                    ),
                )
        return self.get_thread_state(user_id=user_id, thread_id=thread_id) or {}

    def get_thread_state(
        self,
        *,
        user_id: str = "default",
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_thread_state_table(conn)
            row = conn.execute(
                """
                SELECT *
                FROM thread_states
                WHERE user_id = ? AND thread_id = ?
                LIMIT 1
                """,
                (user_id, thread_id),
            ).fetchone()
        if not row:
            return None
        return self._thread_state_row_to_dict(row)

    def list_thread_states(
        self,
        *,
        user_id: str = "default",
        repo: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM thread_states WHERE user_id = ?"
        params: List[Any] = [user_id]
        if repo:
            query += " AND repo = ?"
            params.append(repo)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_thread_state_table(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._thread_state_row_to_dict(row) for row in rows]

    def delete_thread_state(self, *, user_id: str = "default", thread_id: str) -> bool:
        with self._get_connection() as conn:
            self._ensure_thread_state_table(conn)
            cursor = conn.execute(
                "DELETE FROM thread_states WHERE user_id = ? AND thread_id = ?",
                (user_id, thread_id),
            )
        return bool(cursor.rowcount)

    def _shared_task_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _shared_task_result_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _project_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _workspace_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        data["is_primary"] = bool(data.get("is_primary"))
        return data

    def _agent_session_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _session_asset_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _project_asset_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _workspace_root_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _workspace_mount_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        data["is_primary"] = bool(data.get("is_primary"))
        return data

    def _workspace_project_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _workspace_scope_rule_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def _workspace_line_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def upsert_project(self, project: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(project.get("user_id") or "default")
        name = str(project.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        now = _utcnow_iso()
        metadata = json.dumps(project.get("metadata") or {})
        project_id = str(project.get("id") or "").strip()

        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            existing = None
            if project_id:
                existing = conn.execute(
                    "SELECT * FROM projects WHERE id = ? AND user_id = ? LIMIT 1",
                    (project_id, user_id),
                ).fetchone()
            if existing is None:
                existing = conn.execute(
                    "SELECT * FROM projects WHERE user_id = ? AND name = ? LIMIT 1",
                    (user_id, name),
                ).fetchone()
            if existing:
                project_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE projects
                    SET name = ?, description = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        project.get("description"),
                        metadata,
                        now,
                        project_id,
                    ),
                )
            else:
                project_id = project_id or str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO projects (
                        id, user_id, name, description, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        user_id,
                        name,
                        project.get("description"),
                        metadata,
                        project.get("created_at") or now,
                        now,
                    ),
                )
        return self.get_project(project_id, user_id=user_id) or {}

    def get_project(
        self,
        project_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ? AND user_id = ? LIMIT 1",
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._project_row_to_dict(row)

    def list_projects(
        self,
        *,
        user_id: str = "default",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM projects
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, max(1, int(limit))),
            ).fetchall()
        return [self._project_row_to_dict(row) for row in rows]

    def upsert_project_workspace(self, workspace: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(workspace.get("user_id") or "default")
        project_id = str(workspace.get("project_id") or "").strip()
        workspace_path = str(workspace.get("workspace_path") or "").strip()
        label = str(workspace.get("label") or "").strip()
        if not project_id:
            raise ValueError("project_id is required")
        if not workspace_path:
            raise ValueError("workspace_path is required")
        if not label:
            raise ValueError("label is required")
        now = _utcnow_iso()
        metadata = json.dumps(workspace.get("metadata") or {})
        workspace_id = str(workspace.get("id") or "").strip()

        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            existing = None
            if workspace_id:
                existing = conn.execute(
                    """
                    SELECT * FROM project_workspaces
                    WHERE id = ? AND user_id = ?
                    LIMIT 1
                    """,
                    (workspace_id, user_id),
                ).fetchone()
            if existing is None:
                existing = conn.execute(
                    """
                    SELECT * FROM project_workspaces
                    WHERE project_id = ? AND workspace_path = ?
                    LIMIT 1
                    """,
                    (project_id, workspace_path),
                ).fetchone()
            if existing:
                workspace_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE project_workspaces
                    SET workspace_path = ?, label = ?, folder_path = ?,
                        is_primary = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        workspace_path,
                        label,
                        workspace.get("folder_path"),
                        1 if workspace.get("is_primary") else 0,
                        metadata,
                        now,
                        workspace_id,
                    ),
                )
            else:
                workspace_id = workspace_id or str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO project_workspaces (
                        id, project_id, user_id, workspace_path, label, folder_path,
                        is_primary, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        project_id,
                        user_id,
                        workspace_path,
                        label,
                        workspace.get("folder_path"),
                        1 if workspace.get("is_primary") else 0,
                        metadata,
                        workspace.get("created_at") or now,
                        now,
                    ),
                )
        return self.get_project_workspace(workspace_id, user_id=user_id) or {}

    def get_project_workspace(
        self,
        workspace_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM project_workspaces
                WHERE id = ? AND user_id = ?
                LIMIT 1
                """,
                (workspace_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._workspace_row_to_dict(row)

    def list_project_workspaces(
        self,
        *,
        user_id: str = "default",
        project_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM project_workspaces WHERE user_id = ?"
        params: List[Any] = [user_id]
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        query += " ORDER BY is_primary DESC, updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._workspace_row_to_dict(row) for row in rows]

    def upsert_workspace(self, workspace: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(workspace.get("user_id") or "default")
        name = str(workspace.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        now = _utcnow_iso()
        workspace_id = str(workspace.get("id") or "").strip()
        metadata = json.dumps(workspace.get("metadata") or {})
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            existing = None
            if workspace_id:
                existing = conn.execute(
                    "SELECT * FROM workspaces WHERE id = ? AND user_id = ? LIMIT 1",
                    (workspace_id, user_id),
                ).fetchone()
            if existing is None:
                existing = conn.execute(
                    "SELECT * FROM workspaces WHERE user_id = ? AND name = ? LIMIT 1",
                    (user_id, name),
                ).fetchone()
            if existing:
                workspace_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE workspaces
                    SET name = ?, description = ?, root_path = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        workspace.get("description"),
                        workspace.get("root_path"),
                        metadata,
                        now,
                        workspace_id,
                    ),
                )
            else:
                workspace_id = workspace_id or str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO workspaces (
                        id, user_id, name, description, root_path, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workspace_id,
                        user_id,
                        name,
                        workspace.get("description"),
                        workspace.get("root_path"),
                        metadata,
                        workspace.get("created_at") or now,
                        now,
                    ),
                )
        return self.get_workspace(workspace_id, user_id=user_id) or {}

    def get_workspace(
        self,
        workspace_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                "SELECT * FROM workspaces WHERE id = ? AND user_id = ? LIMIT 1",
                (workspace_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._workspace_root_row_to_dict(row)

    def list_workspaces(
        self,
        *,
        user_id: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM workspaces
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (user_id, max(1, int(limit))),
            ).fetchall()
        return [self._workspace_root_row_to_dict(row) for row in rows]

    def upsert_workspace_mount(self, mount: Dict[str, Any]) -> Dict[str, Any]:
        workspace_id = str(mount.get("workspace_id") or "").strip()
        user_id = str(mount.get("user_id") or "default")
        mount_path = str(mount.get("mount_path") or "").strip()
        if not workspace_id:
            raise ValueError("workspace_id is required")
        if not mount_path:
            raise ValueError("mount_path is required")
        now = _utcnow_iso()
        mount_id = str(mount.get("id") or "").strip() or f"{workspace_id}:{mount_path}"
        metadata = json.dumps(mount.get("metadata") or {})
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            existing = conn.execute(
                """
                SELECT * FROM workspace_mounts
                WHERE workspace_id = ? AND mount_path = ?
                LIMIT 1
                """,
                (workspace_id, mount_path),
            ).fetchone()
            if existing:
                mount_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE workspace_mounts
                    SET label = ?, is_primary = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        mount.get("label"),
                        1 if mount.get("is_primary") else 0,
                        metadata,
                        now,
                        mount_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO workspace_mounts (
                        id, workspace_id, user_id, mount_path, label, is_primary,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        mount_id,
                        workspace_id,
                        user_id,
                        mount_path,
                        mount.get("label"),
                        1 if mount.get("is_primary") else 0,
                        metadata,
                        mount.get("created_at") or now,
                        now,
                    ),
                )
        return self.get_workspace_mount(mount_id, user_id=user_id) or {}

    def get_workspace_mount(
        self,
        mount_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM workspace_mounts
                WHERE id = ? AND user_id = ?
                LIMIT 1
                """,
                (mount_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._workspace_mount_row_to_dict(row)

    def list_workspace_mounts(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM workspace_mounts
                WHERE workspace_id = ? AND user_id = ?
                ORDER BY is_primary DESC, updated_at DESC
                """,
                (workspace_id, user_id),
            ).fetchall()
        return [self._workspace_mount_row_to_dict(row) for row in rows]

    def delete_workspace_cascade(
        self,
        workspace_id: str,
        *,
        user_id: str = "default",
    ) -> bool:
        """Delete a workspace and every row transitively scoped to it.

        Ordering matters: drop the dependents first, then the workspace
        row. We collect project IDs up-front so asset rows (which key
        on project_id) get cleaned in the same transaction.
        """
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            project_ids = [
                str(row["id"])
                for row in conn.execute(
                    "SELECT id FROM workspace_projects WHERE workspace_id = ? AND user_id = ?",
                    (workspace_id, user_id),
                ).fetchall()
            ]
            # Project-scoped rows (session_assets are session-linked, not
            # project-linked, but still live under sessions that we're
            # about to prune).
            for table in ("workspace_project_scope_rules",):
                conn.execute(
                    f"DELETE FROM {table} WHERE user_id = ? AND project_id IN (SELECT id FROM workspace_projects WHERE workspace_id = ? AND user_id = ?)",
                    (user_id, workspace_id, user_id),
                )
            conn.execute(
                "DELETE FROM project_assets WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
            conn.execute(
                "DELETE FROM workspace_line_messages WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
            conn.execute(
                "DELETE FROM workspace_mounts WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
            conn.execute(
                "DELETE FROM workspace_projects WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
            conn.execute(
                "DELETE FROM agent_sessions WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
            cursor = conn.execute(
                "DELETE FROM workspaces WHERE id = ? AND user_id = ?",
                (workspace_id, user_id),
            )
        return bool(cursor.rowcount)

    def delete_workspace_project_cascade(
        self,
        project_id: str,
        *,
        user_id: str = "default",
    ) -> bool:
        """Delete a project and its assets, scope rules, and line links.

        Sessions associated with the project are detached (project_id set
        to NULL) rather than deleted — they represent live or recent
        agent runs that may still be running against the workspace.
        """
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            conn.execute(
                "DELETE FROM workspace_project_scope_rules WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            conn.execute(
                "DELETE FROM project_assets WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            # Null out dangling references on line messages instead of
            # wiping the history — the workspace line is an append-only
            # audit trail.
            conn.execute(
                "UPDATE workspace_line_messages SET project_id = NULL WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            conn.execute(
                "UPDATE workspace_line_messages SET target_project_id = NULL WHERE target_project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            conn.execute(
                "UPDATE agent_sessions SET project_id = NULL WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            cursor = conn.execute(
                "DELETE FROM workspace_projects WHERE id = ? AND user_id = ?",
                (project_id, user_id),
            )
        return bool(cursor.rowcount)

    def delete_workspace_mount(
        self,
        *,
        workspace_id: str,
        mount_path: str,
        user_id: str = "default",
    ) -> bool:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            cursor = conn.execute(
                """
                DELETE FROM workspace_mounts
                WHERE workspace_id = ? AND mount_path = ? AND user_id = ?
                """,
                (workspace_id, mount_path, user_id),
            )
        return bool(cursor.rowcount)

    def upsert_workspace_project(self, project: Dict[str, Any]) -> Dict[str, Any]:
        workspace_id = str(project.get("workspace_id") or "").strip()
        user_id = str(project.get("user_id") or "default")
        name = str(project.get("name") or "").strip()
        if not workspace_id:
            raise ValueError("workspace_id is required")
        if not name:
            raise ValueError("name is required")
        now = _utcnow_iso()
        project_id = str(project.get("id") or "").strip()
        metadata = json.dumps(project.get("metadata") or {})
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            existing = None
            if project_id:
                existing = conn.execute(
                    "SELECT * FROM workspace_projects WHERE id = ? AND user_id = ? LIMIT 1",
                    (project_id, user_id),
                ).fetchone()
            if existing is None:
                existing = conn.execute(
                    """
                    SELECT * FROM workspace_projects
                    WHERE workspace_id = ? AND user_id = ? AND name = ?
                    LIMIT 1
                    """,
                    (workspace_id, user_id, name),
                ).fetchone()
            if existing:
                project_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE workspace_projects
                    SET name = ?, description = ?, default_runtime = ?, color = ?, icon = ?,
                        metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        name,
                        project.get("description"),
                        project.get("default_runtime") or "codex",
                        project.get("color"),
                        project.get("icon"),
                        metadata,
                        now,
                        project_id,
                    ),
                )
            else:
                project_id = project_id or str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO workspace_projects (
                        id, workspace_id, user_id, name, description, default_runtime,
                        color, icon, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        workspace_id,
                        user_id,
                        name,
                        project.get("description"),
                        project.get("default_runtime") or "codex",
                        project.get("color"),
                        project.get("icon"),
                        metadata,
                        project.get("created_at") or now,
                        now,
                    ),
                )
        return self.get_workspace_project(project_id, user_id=user_id) or {}

    def get_workspace_project(
        self,
        project_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM workspace_projects
                WHERE id = ? AND user_id = ?
                LIMIT 1
                """,
                (project_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._workspace_project_row_to_dict(row)

    def list_workspace_projects(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM workspace_projects
                WHERE workspace_id = ? AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (workspace_id, user_id, max(1, int(limit))),
            ).fetchall()
        return [self._workspace_project_row_to_dict(row) for row in rows]

    def replace_workspace_project_scope_rules(
        self,
        *,
        project_id: str,
        rules: List[Dict[str, Any]],
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        now = _utcnow_iso()
        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rule in rules:
            path_prefix = str(rule.get("path_prefix") or "").strip()
            if not path_prefix or path_prefix in seen:
                continue
            seen.add(path_prefix)
            normalized.append(
                {
                    "id": str(rule.get("id") or f"{project_id}:{path_prefix}"),
                    "project_id": project_id,
                    "user_id": user_id,
                    "path_prefix": path_prefix,
                    "label": rule.get("label"),
                    "metadata": json.dumps(rule.get("metadata") or {}),
                    "created_at": rule.get("created_at") or now,
                    "updated_at": now,
                }
            )

        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            conn.execute(
                "DELETE FROM workspace_project_scope_rules WHERE project_id = ? AND user_id = ?",
                (project_id, user_id),
            )
            for rule in normalized:
                conn.execute(
                    """
                    INSERT INTO workspace_project_scope_rules (
                        id, project_id, user_id, path_prefix, label, metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule["id"],
                        project_id,
                        user_id,
                        rule["path_prefix"],
                        rule.get("label"),
                        rule["metadata"],
                        rule["created_at"],
                        rule["updated_at"],
                    ),
                )
        return self.list_workspace_project_scope_rules(project_id=project_id, user_id=user_id)

    def list_workspace_project_scope_rules(
        self,
        *,
        project_id: str,
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM workspace_project_scope_rules
                WHERE project_id = ? AND user_id = ?
                ORDER BY length(path_prefix) DESC, updated_at DESC
                """,
                (project_id, user_id),
            ).fetchall()
        return [self._workspace_scope_rule_row_to_dict(row) for row in rows]

    def add_workspace_line_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert a line message. Returns the row, or None if dedup_key collided.

        Idempotent on (workspace_id, dedup_key) when dedup_key is provided —
        retries from agent emitters silently drop the duplicate.
        """
        workspace_id = str(message.get("workspace_id") or "").strip()
        user_id = str(message.get("user_id") or "default")
        channel = str(message.get("channel") or "").strip() or "workspace"
        message_kind = str(message.get("message_kind") or "").strip() or "update"
        if not workspace_id:
            raise ValueError("workspace_id is required")
        now = _utcnow_iso()
        message_id = str(message.get("id") or uuid.uuid4())
        dedup_key = message.get("dedup_key")
        if dedup_key is not None:
            dedup_key = str(dedup_key).strip() or None
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            if dedup_key is not None:
                existing = conn.execute(
                    "SELECT id FROM workspace_line_messages WHERE workspace_id = ? AND dedup_key = ? LIMIT 1",
                    (workspace_id, dedup_key),
                ).fetchone()
                if existing:
                    return None
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO workspace_line_messages (
                    id, workspace_id, project_id, target_project_id, user_id, channel,
                    session_id, task_id, message_kind, title, body, metadata,
                    dedup_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    workspace_id,
                    message.get("project_id"),
                    message.get("target_project_id"),
                    user_id,
                    channel,
                    message.get("session_id"),
                    message.get("task_id"),
                    message_kind,
                    message.get("title"),
                    message.get("body"),
                    json.dumps(message.get("metadata") or {}),
                    dedup_key,
                    message.get("created_at") or now,
                    now,
                ),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_workspace_line_message(message_id)

    def get_workspace_line_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                "SELECT * FROM workspace_line_messages WHERE id = ? LIMIT 1",
                (message_id,),
            ).fetchone()
        if not row:
            return None
        return self._workspace_line_row_to_dict(row)

    def list_workspace_line_messages(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
        project_id: Optional[str] = None,
        channel: Optional[str] = None,
        cursor: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM workspace_line_messages WHERE workspace_id = ? AND user_id = ?"
        params: List[Any] = [workspace_id, user_id]
        if project_id:
            query += " AND (project_id = ? OR target_project_id = ?)"
            params.extend([project_id, project_id])
        if channel:
            query += " AND channel = ?"
            params.append(channel)
        if cursor:
            try:
                created_at, message_id = cursor.split("|", 1)
            except ValueError:
                created_at, message_id = cursor, ""
            query += " AND (created_at < ? OR (created_at = ? AND id < ?))"
            params.extend([created_at, created_at, message_id])
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._workspace_line_row_to_dict(row) for row in rows]

    def list_workspace_line_unread(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
        consumer_id: str,
        project_id: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List live line messages not yet acknowledged by this consumer."""
        consumer_id = str(consumer_id or "").strip()
        if not consumer_id:
            raise ValueError("consumer_id is required")
        query = """
            SELECT m.*
            FROM workspace_line_messages m
            LEFT JOIN workspace_line_receipts r
              ON r.workspace_id = m.workspace_id
             AND r.message_id = m.id
             AND r.user_id = m.user_id
             AND r.consumer_id = ?
            WHERE m.workspace_id = ? AND m.user_id = ? AND r.message_id IS NULL
        """
        params: List[Any] = [consumer_id, workspace_id, user_id]
        if project_id:
            query += " AND (m.project_id = ? OR m.target_project_id = ?)"
            params.extend([project_id, project_id])
        if channel:
            query += " AND m.channel = ?"
            params.append(channel)
        query += " ORDER BY m.created_at DESC, m.id DESC LIMIT ?"
        try:
            cap = max(1, min(200, int(limit) * 5))
        except (TypeError, ValueError):
            cap = 100
        params.append(cap)
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            self._ensure_workspace_line_receipts_migration(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._workspace_line_row_to_dict(row) for row in rows]

    def mark_workspace_line_messages_read(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
        consumer_id: str,
        message_ids: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Mark a set of live line messages as read for one consumer."""
        consumer_id = str(consumer_id or "").strip()
        if not consumer_id:
            raise ValueError("consumer_id is required")
        ids = [str(mid).strip() for mid in (message_ids or []) if str(mid or "").strip()]
        if not ids:
            return 0
        now = _utcnow_iso()
        meta = json.dumps(metadata or {})
        wrote = 0
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            self._ensure_workspace_line_receipts_migration(conn)
            for message_id in ids:
                cur = conn.execute(
                    """
                    INSERT OR IGNORE INTO workspace_line_receipts (
                        id, workspace_id, message_id, user_id, consumer_id, metadata, read_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        workspace_id,
                        message_id,
                        user_id,
                        consumer_id,
                        meta,
                        now,
                    ),
                )
                wrote += int(cur.rowcount or 0)
        return wrote

    def upsert_agent_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(session.get("user_id") or "default")
        runtime_id = str(session.get("runtime_id") or "").strip()
        title = str(session.get("title") or "").strip()
        if not runtime_id:
            raise ValueError("runtime_id is required")
        if not title:
            raise ValueError("title is required")
        now = _utcnow_iso()
        metadata = json.dumps(session.get("metadata") or {})
        session_id = str(session.get("id") or "").strip()
        native_session_id = str(session.get("native_session_id") or "").strip() or None

        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            existing = None
            if session_id:
                existing = conn.execute(
                    "SELECT * FROM agent_sessions WHERE id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
            if existing is None and native_session_id:
                existing = conn.execute(
                    """
                    SELECT * FROM agent_sessions
                    WHERE runtime_id = ? AND native_session_id = ?
                    LIMIT 1
                    """,
                    (runtime_id, native_session_id),
                ).fetchone()
            if existing:
                session_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE agent_sessions
                    SET project_id = ?, workspace_id = ?, user_id = ?, task_id = ?,
                        title = ?, state = ?, model = ?, cwd = ?, rollout_path = ?,
                        permission_mode = ?, started_at = ?, updated_at = ?, metadata = ?
                    WHERE id = ?
                    """,
                    (
                        session.get("project_id"),
                        session.get("workspace_id"),
                        user_id,
                        session.get("task_id"),
                        title,
                        session.get("state") or "recent",
                        session.get("model"),
                        session.get("cwd"),
                        session.get("rollout_path"),
                        session.get("permission_mode"),
                        session.get("started_at"),
                        session.get("updated_at") or now,
                        metadata,
                        session_id,
                    ),
                )
            else:
                session_id = session_id or str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO agent_sessions (
                        id, project_id, workspace_id, user_id, runtime_id, native_session_id,
                        task_id, title, state, model, cwd, rollout_path, permission_mode,
                        started_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        session.get("project_id"),
                        session.get("workspace_id"),
                        user_id,
                        runtime_id,
                        native_session_id,
                        session.get("task_id"),
                        title,
                        session.get("state") or "recent",
                        session.get("model"),
                        session.get("cwd"),
                        session.get("rollout_path"),
                        session.get("permission_mode"),
                        session.get("started_at"),
                        session.get("updated_at") or now,
                        metadata,
                    ),
                )
        return self.get_agent_session(session_id) or {}

    def get_agent_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return self._agent_session_row_to_dict(row)

    def find_agent_session(
        self,
        *,
        runtime_id: Optional[str] = None,
        native_session_id: Optional[str] = None,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Resolve an agent session by its native (host-runtime) id."""
        if not runtime_id or not native_session_id:
            return None
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM agent_sessions
                WHERE runtime_id = ? AND native_session_id = ? AND user_id = ?
                LIMIT 1
                """,
                (runtime_id, native_session_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._agent_session_row_to_dict(row)

    def list_agent_sessions(
        self,
        *,
        user_id: str = "default",
        project_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        runtime_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM agent_sessions WHERE user_id = ?"
        params: List[Any] = [user_id]
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if runtime_id:
            query += " AND runtime_id = ?"
            params.append(runtime_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._agent_session_row_to_dict(row) for row in rows]

    def add_session_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        session_id = str(asset.get("session_id") or "").strip()
        storage_path = str(asset.get("storage_path") or "").strip()
        name = str(asset.get("name") or "").strip()
        if not session_id:
            raise ValueError("session_id is required")
        if not storage_path:
            raise ValueError("storage_path is required")
        if not name:
            raise ValueError("name is required")
        now = _utcnow_iso()
        asset_id = str(asset.get("id") or uuid.uuid4())
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            conn.execute(
                """
                INSERT INTO session_assets (
                    id, project_id, workspace_id, session_id, user_id, artifact_id,
                    storage_path, name, mime_type, size_bytes, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    asset.get("project_id"),
                    asset.get("workspace_id"),
                    session_id,
                    str(asset.get("user_id") or "default"),
                    asset.get("artifact_id"),
                    storage_path,
                    name,
                    asset.get("mime_type"),
                    int(asset.get("size_bytes") or 0),
                    json.dumps(asset.get("metadata") or {}),
                    asset.get("created_at") or now,
                    now,
                ),
            )
        return self.get_session_asset(asset_id) or {}

    def get_session_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                "SELECT * FROM session_assets WHERE id = ? LIMIT 1",
                (asset_id,),
            ).fetchone()
        if not row:
            return None
        return self._session_asset_row_to_dict(row)

    def list_session_assets(
        self,
        *,
        session_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM session_assets
                WHERE session_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (session_id, max(1, int(limit))),
            ).fetchall()
        return [self._session_asset_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # project_assets (PR 3) — workspace/project-scoped assets shared across
    # sessions. Independent from session_assets.
    # ------------------------------------------------------------------
    def upsert_project_asset(self, asset: Dict[str, Any]) -> Dict[str, Any]:
        workspace_id = str(asset.get("workspace_id") or "").strip()
        if not workspace_id:
            raise ValueError("workspace_id is required")
        storage_path = str(asset.get("storage_path") or "").strip()
        name = str(asset.get("name") or "").strip()
        if not storage_path:
            raise ValueError("storage_path is required")
        if not name:
            raise ValueError("name is required")
        user_id = str(asset.get("user_id") or "default")
        project_id = str(asset.get("project_id") or "").strip() or None
        checksum = str(asset.get("checksum") or "").strip() or None
        asset_id = str(asset.get("id") or uuid.uuid4())
        now = _utcnow_iso()
        metadata = json.dumps(asset.get("metadata") or {})
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            # Dedup by (workspace, project, checksum) — returns the pre-existing
            # row if the same file has been uploaded before in this scope.
            if checksum:
                row = conn.execute(
                    """
                    SELECT * FROM project_assets
                    WHERE workspace_id = ? AND COALESCE(project_id, '') = COALESCE(?, '')
                      AND checksum = ?
                    LIMIT 1
                    """,
                    (workspace_id, project_id, checksum),
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE project_assets SET updated_at = ? WHERE id = ?",
                        (now, row["id"]),
                    )
                    return self.get_project_asset(str(row["id"])) or {}
            conn.execute(
                """
                INSERT INTO project_assets (
                    id, workspace_id, project_id, user_id, artifact_id, folder,
                    storage_path, name, mime_type, size_bytes, checksum, metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    workspace_id,
                    project_id,
                    user_id,
                    asset.get("artifact_id"),
                    asset.get("folder"),
                    storage_path,
                    name,
                    asset.get("mime_type"),
                    int(asset.get("size_bytes") or 0),
                    checksum,
                    metadata,
                    asset.get("created_at") or now,
                    now,
                ),
            )
        return self.get_project_asset(asset_id) or {}

    def get_project_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                "SELECT * FROM project_assets WHERE id = ? LIMIT 1",
                (asset_id,),
            ).fetchone()
        if not row:
            return None
        return self._project_asset_row_to_dict(row)

    def list_project_assets(
        self,
        *,
        project_id: str,
        user_id: str = "default",
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(
                """
                SELECT * FROM project_assets
                WHERE project_id = ? AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (project_id, user_id, max(1, int(limit))),
            ).fetchall()
        return [self._project_asset_row_to_dict(row) for row in rows]

    def list_workspace_assets(
        self,
        *,
        workspace_id: str,
        user_id: str = "default",
        include_project_assets: bool = True,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """List assets visible at workspace scope.

        When ``include_project_assets`` is true (default) the result
        includes project-scoped assets as well, so the workspace "all
        assets" view is a single query. Set to false to see only
        workspace-level (project_id IS NULL) items.
        """
        query = "SELECT * FROM project_assets WHERE workspace_id = ? AND user_id = ?"
        params: List[Any] = [workspace_id, user_id]
        if not include_project_assets:
            query += " AND project_id IS NULL"
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._project_asset_row_to_dict(row) for row in rows]

    def find_project_asset_by_storage_path(
        self,
        storage_path: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        if not storage_path:
            return None
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM project_assets
                WHERE storage_path = ? AND user_id = ?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (storage_path, user_id),
            ).fetchone()
        if not row:
            return None
        return self._project_asset_row_to_dict(row)

    def delete_project_asset(self, asset_id: str, *, user_id: str = "default") -> bool:
        with self._get_connection() as conn:
            self._ensure_workspace_hierarchy_tables(conn)
            cursor = conn.execute(
                "DELETE FROM project_assets WHERE id = ? AND user_id = ?",
                (asset_id, user_id),
            )
        return bool(cursor.rowcount)

    def find_shared_task(
        self,
        *,
        user_id: str = "default",
        shared_task_id: Optional[str] = None,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        native_session_id: Optional[str] = None,
        runtime_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            if shared_task_id:
                row = conn.execute(
                    "SELECT * FROM shared_tasks WHERE id = ? AND user_id = ? LIMIT 1",
                    (shared_task_id, user_id),
                ).fetchone()
                return self._shared_task_row_to_dict(row) if row else None
            for candidate in [
                (
                    """
                    SELECT * FROM shared_tasks
                    WHERE user_id = ? AND runtime_id = ? AND native_session_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (user_id, runtime_id, native_session_id),
                )
                if runtime_id and native_session_id
                else None,
                (
                    """
                    SELECT * FROM shared_tasks
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (user_id, session_id),
                )
                if session_id
                else None,
                (
                    """
                    SELECT * FROM shared_tasks
                    WHERE user_id = ? AND thread_id = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (user_id, thread_id),
                )
                if thread_id
                else None,
            ]:
                if not candidate:
                    continue
                query, params = candidate
                row = conn.execute(query, params).fetchone()
                if row:
                    return self._shared_task_row_to_dict(row)
        return None

    def upsert_shared_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(task.get("user_id") or "default")
        title = str(task.get("title") or "").strip()
        if not title:
            raise ValueError("title is required")
        now = _utcnow_iso()
        repo = task.get("repo")
        workspace_id = task.get("workspace_id")
        status = str(task.get("status") or "active")
        task_id = str(task.get("id") or "").strip()
        metadata = json.dumps(task.get("metadata") or {})

        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            existing = None
            if task_id:
                existing = conn.execute(
                    "SELECT * FROM shared_tasks WHERE id = ? AND user_id = ? LIMIT 1",
                    (task_id, user_id),
                ).fetchone()
            if existing is None:
                existing = self.find_shared_task(
                    user_id=user_id,
                    session_id=task.get("session_id"),
                    thread_id=task.get("thread_id"),
                    native_session_id=task.get("native_session_id"),
                    runtime_id=task.get("runtime_id"),
                )
            if existing:
                task_id = str(existing["id"]) if isinstance(existing, dict) else str(existing["id"])
                conn.execute(
                    """
                    UPDATE shared_tasks
                    SET project_id = ?, repo = ?, workspace_id = ?, folder_path = ?,
                        session_id = ?, thread_id = ?, runtime_id = ?, native_session_id = ?, title = ?,
                        status = ?, created_by = COALESCE(?, created_by),
                        metadata = ?, updated_at = ?, closed_at = ?
                    WHERE id = ?
                    """,
                    (
                        task.get("project_id"),
                        repo,
                        workspace_id,
                        task.get("folder_path"),
                        task.get("session_id"),
                        task.get("thread_id"),
                        task.get("runtime_id"),
                        task.get("native_session_id"),
                        title,
                        status,
                        task.get("created_by"),
                        metadata,
                        now,
                        now if status in {"completed", "closed"} else None,
                        task_id,
                    ),
                )
            else:
                task_id = task_id or str(uuid.uuid4())
                try:
                    conn.execute(
                        """
                        INSERT INTO shared_tasks (
                            id, user_id, project_id, repo, workspace_id, folder_path,
                            session_id, thread_id, runtime_id, native_session_id, title,
                            status, created_by, metadata, created_at, updated_at, closed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            task_id,
                            user_id,
                            task.get("project_id"),
                            repo,
                            workspace_id,
                            task.get("folder_path"),
                            task.get("session_id"),
                            task.get("thread_id"),
                            task.get("runtime_id"),
                            task.get("native_session_id"),
                            title,
                            status,
                            task.get("created_by"),
                            metadata,
                            task.get("created_at") or now,
                            now,
                            now if status in {"completed", "closed"} else None,
                        ),
                    )
                except sqlite3.IntegrityError:
                    conn.execute(
                        """
                        UPDATE shared_tasks
                        SET user_id = ?, project_id = ?, repo = ?, workspace_id = ?, folder_path = ?,
                            session_id = ?, thread_id = ?, runtime_id = ?, native_session_id = ?, title = ?,
                            status = ?, created_by = COALESCE(?, created_by),
                            metadata = ?, updated_at = ?, closed_at = ?
                        WHERE id = ?
                        """,
                        (
                            user_id,
                            task.get("project_id"),
                            repo,
                            workspace_id,
                            task.get("folder_path"),
                            task.get("session_id"),
                            task.get("thread_id"),
                            task.get("runtime_id"),
                            task.get("native_session_id"),
                            title,
                            status,
                            task.get("created_by"),
                            metadata,
                            now,
                            now if status in {"completed", "closed"} else None,
                            task_id,
                        ),
                    )
        return self.get_shared_task(task_id, user_id=user_id) or {}

    def get_shared_task(
        self,
        shared_task_id: str,
        *,
        user_id: str = "default",
    ) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                """
                SELECT * FROM shared_tasks
                WHERE id = ? AND user_id = ?
                LIMIT 1
                """,
                (shared_task_id, user_id),
            ).fetchone()
        if not row:
            return None
        return self._shared_task_row_to_dict(row)

    def list_shared_tasks(
        self,
        *,
        user_id: str = "default",
        project_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        repo: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM shared_tasks WHERE user_id = ?"
        params: List[Any] = [user_id]
        if project_id:
            query += " AND project_id = ?"
            params.append(project_id)
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if repo:
            query += " AND repo = ?"
            params.append(repo)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._shared_task_row_to_dict(row) for row in rows]

    def close_shared_task(
        self,
        shared_task_id: str,
        *,
        user_id: str = "default",
        status: str = "completed",
        prune_results: bool = True,
    ) -> bool:
        now = _utcnow_iso()
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            cur = conn.execute(
                """
                UPDATE shared_tasks
                SET status = ?, updated_at = ?, closed_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (status, now, now, shared_task_id, user_id),
            )
            if prune_results:
                conn.execute(
                    "DELETE FROM shared_task_results WHERE shared_task_id = ?",
                    (shared_task_id,),
                )
        return bool(cur.rowcount)

    def close_stale_shared_tasks(
        self,
        *,
        user_id: str = "default",
        max_age_hours: int = 24,
        status: str = "closed",
    ) -> int:
        # Bulk close active tasks whose updated_at is older than the cutoff.
        # ISO-8601 strings in `updated_at` sort lexicographically, so a string
        # comparison is safe and avoids per-row datetime parsing.
        cutoff = (_utcnow() - timedelta(hours=max(0, int(max_age_hours)))).isoformat()
        now = _utcnow_iso()
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            cur = conn.execute(
                """
                UPDATE shared_tasks
                SET status = ?, updated_at = ?, closed_at = ?
                WHERE user_id = ? AND status = 'active' AND updated_at < ?
                """,
                (status, now, now, user_id, cutoff),
            )
        return int(cur.rowcount or 0)

    def save_shared_task_result(self, result: Dict[str, Any]) -> str:
        shared_task_id = str(result.get("shared_task_id") or "").strip()
        result_key = str(result.get("result_key") or "").strip()
        packet_kind = str(result.get("packet_kind") or "").strip()
        tool_name = str(result.get("tool_name") or "").strip()
        if not shared_task_id:
            raise ValueError("shared_task_id is required")
        if not result_key:
            raise ValueError("result_key is required")
        if not packet_kind:
            raise ValueError("packet_kind is required")
        if not tool_name:
            raise ValueError("tool_name is required")

        now = _utcnow_iso()
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            existing = conn.execute(
                """
                SELECT id FROM shared_task_results
                WHERE result_key = ?
                LIMIT 1
                """,
                (result_key,),
            ).fetchone()
            if existing:
                result_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE shared_task_results
                    SET project_id = ?, result_status = ?, source_event_id = ?, source_path = ?,
                        ptr = ?, artifact_id = ?, digest = ?, metadata = ?,
                        session_id = ?, thread_id = ?, harness = ?, agent_id = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        result.get("project_id"),
                        result.get("result_status") or "completed",
                        result.get("source_event_id"),
                        result.get("source_path"),
                        result.get("ptr"),
                        result.get("artifact_id"),
                        result.get("digest"),
                        json.dumps(result.get("metadata") or {}),
                        result.get("session_id"),
                        result.get("thread_id"),
                        result.get("harness"),
                        result.get("agent_id"),
                        now,
                        result_id,
                    ),
                )
                return result_id

            result_id = str(result.get("id") or uuid.uuid4())
            conn.execute(
                """
                INSERT INTO shared_task_results (
                    id, result_key, shared_task_id, user_id, project_id, repo, workspace_id,
                    folder_path, packet_kind, tool_name, result_status,
                    source_event_id, source_path, ptr, artifact_id, digest,
                    metadata, session_id, thread_id, harness, agent_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    result_key,
                    shared_task_id,
                    str(result.get("user_id") or "default"),
                    result.get("project_id"),
                    result.get("repo"),
                    result.get("workspace_id"),
                    result.get("folder_path"),
                    packet_kind,
                    tool_name,
                    result.get("result_status") or "completed",
                    result.get("source_event_id"),
                    result.get("source_path"),
                    result.get("ptr"),
                    result.get("artifact_id"),
                    result.get("digest"),
                    json.dumps(result.get("metadata") or {}),
                    result.get("session_id"),
                    result.get("thread_id"),
                    result.get("harness"),
                    result.get("agent_id"),
                    now,
                    now,
                ),
            )
        return result_id

    def get_shared_task_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            row = conn.execute(
                "SELECT * FROM shared_task_results WHERE id = ? LIMIT 1",
                (result_id,),
            ).fetchone()
        if not row:
            return None
        return self._shared_task_result_row_to_dict(row)

    def list_shared_task_results(
        self,
        *,
        shared_task_id: str,
        limit: int = 20,
        result_status: Optional[str] = None,
        packet_kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM shared_task_results WHERE shared_task_id = ?"
        params: List[Any] = [shared_task_id]
        if result_status:
            query += " AND result_status = ?"
            params.append(result_status)
        if packet_kind:
            query += " AND packet_kind = ?"
            params.append(packet_kind)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._shared_task_result_row_to_dict(row) for row in rows]

    def list_shared_task_results_for_path(
        self,
        *,
        user_id: str = "default",
        workspace_id: Optional[str] = None,
        source_path: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM shared_task_results WHERE user_id = ? AND source_path = ?"
        params: List[Any] = [user_id, source_path]
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._shared_task_result_row_to_dict(row) for row in rows]

    def get_episodic_memories(
        self,
        user_id: str,
        *,
        scene_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        limit: int = 100,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch episodic-type memories for a user, optionally filtered."""
        query = (
            "SELECT * FROM memories WHERE user_id = ? "
            "AND memory_type = 'episodic' AND tombstone = 0"
        )
        params: List[Any] = [user_id]
        if scene_id:
            query += " AND scene_id = ?"
            params.append(scene_id)
        if created_after:
            query += " AND created_at >= ?"
            params.append(created_after)
        if created_before:
            query += " AND created_at <= ?"
            params.append(created_before)
        if namespace:
            query += " AND namespace = ?"
            params.append(namespace)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_dict(row) for row in rows]

    def add_distillation_provenance(
        self,
        semantic_memory_id: str,
        episodic_memory_ids: List[str],
        run_id: str,
    ) -> None:
        """Record which episodic memories contributed to a semantic memory."""
        with self._get_connection() as conn:
            for ep_id in episodic_memory_ids:
                conn.execute(
                    """
                    INSERT INTO distillation_provenance (
                        id, semantic_memory_id, episodic_memory_id,
                        distillation_run_id
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), semantic_memory_id, ep_id, run_id),
                )

    def _ensure_distilled_episodes_table(self, conn) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS distilled_episodes (
                episodic_memory_id TEXT PRIMARY KEY,
                distillation_run_id TEXT,
                distilled_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def distilled_episode_ids(self, episodic_memory_ids: List[str]) -> set:
        """Which of these episodes a distillation has already read.

        An episode counts as read when a batch containing it finished — whether
        or not that batch produced anything — or when a semantic memory already
        names it as a source. Without this, a sleep cycle that runs every few
        minutes distils the same day again on every run, and each run stores
        its paraphrases as new memories.
        """
        ids = [str(i) for i in episodic_memory_ids if i]
        if not ids:
            return set()
        found: set = set()
        with self._get_connection() as conn:
            self._ensure_distilled_episodes_table(conn)
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                marks = ",".join("?" for _ in chunk)
                for table in ("distilled_episodes", "distillation_provenance"):
                    rows = conn.execute(
                        f"SELECT DISTINCT episodic_memory_id FROM {table} "
                        f"WHERE episodic_memory_id IN ({marks})",
                        chunk,
                    ).fetchall()
                    found.update(str(row[0]) for row in rows)
        return found

    def mark_episodes_distilled(self, episodic_memory_ids: List[str], run_id: str) -> None:
        with self._get_connection() as conn:
            self._ensure_distilled_episodes_table(conn)
            conn.executemany(
                "INSERT OR IGNORE INTO distilled_episodes (episodic_memory_id, distillation_run_id) "
                "VALUES (?, ?)",
                [(str(i), run_id) for i in episodic_memory_ids if i],
            )

    def get_distillation_sources(
        self, semantic_memory_id: str
    ) -> List[Dict[str, Any]]:
        """Return the episodic memories that were distilled into this semantic one.

        Emits ``{episodic_memory_id, distillation_run_id, created_at}`` per
        row, ordered oldest-first. Empty list when the semantic memory was
        not produced by distillation (e.g. written directly by the user).
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT episodic_memory_id, distillation_run_id, created_at
                FROM distillation_provenance
                WHERE semantic_memory_id = ?
                ORDER BY created_at ASC
                """,
                (semantic_memory_id,),
            ).fetchall()
            return [
                {
                    "episodic_memory_id": r["episodic_memory_id"],
                    "distillation_run_id": r["distillation_run_id"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def get_distillation_derivatives(
        self, episodic_memory_id: str
    ) -> List[Dict[str, Any]]:
        """Return the semantic memories derived from this episodic one."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT semantic_memory_id, distillation_run_id, created_at
                FROM distillation_provenance
                WHERE episodic_memory_id = ?
                ORDER BY created_at ASC
                """,
                (episodic_memory_id,),
            ).fetchall()
            return [
                {
                    "semantic_memory_id": r["semantic_memory_id"],
                    "distillation_run_id": r["distillation_run_id"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def get_distillation_source_counts(
        self, semantic_memory_ids: List[str]
    ) -> Dict[str, int]:
        """Bulk count of episodic sources for a batch of semantic memory IDs.

        Used by the search pipeline to annotate distilled memories with
        ``provenance_source_count`` so the retrieval surface knows which
        returned rows are synthesis (backed by N episodes) vs. raw writes.
        """
        if not semantic_memory_ids:
            return {}
        placeholders = ",".join("?" for _ in semantic_memory_ids)
        with self._get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT semantic_memory_id, COUNT(*) AS n
                FROM distillation_provenance
                WHERE semantic_memory_id IN ({placeholders})
                GROUP BY semantic_memory_id
                """,
                tuple(semantic_memory_ids),
            ).fetchall()
            return {r["semantic_memory_id"]: int(r["n"]) for r in rows}

    def log_distillation_run(
        self,
        user_id: str,
        episodes_sampled: int,
        semantic_created: int,
        semantic_deduplicated: int = 0,
        errors: int = 0,
    ) -> str:
        """Log a distillation run and return the run ID."""
        run_id = str(uuid.uuid4())
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO distillation_log (
                    id, user_id, episodes_sampled, semantic_created,
                    semantic_deduplicated, errors
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_id,
                    episodes_sampled,
                    semantic_created,
                    semantic_deduplicated,
                    errors,
                ),
            )
        return run_id

    def get_memory_count_by_namespace(self, user_id: str) -> Dict[str, int]:
        """Return {namespace: count} for active memories of a user."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT COALESCE(namespace, 'default') AS ns, COUNT(*) AS cnt
                FROM memories
                WHERE user_id = ? AND tombstone = 0
                GROUP BY ns
                """,
                (user_id,),
            ).fetchall()
            return {row["ns"]: row["cnt"] for row in rows}

    def update_multi_trace(
        self,
        memory_id: str,
        s_fast: float,
        s_mid: float,
        s_slow: float,
        effective_strength: float,
    ) -> bool:
        """Update multi-trace columns and effective strength for a memory."""
        return self.update_memory(
            memory_id,
            {
                "s_fast": s_fast,
                "s_mid": s_mid,
                "s_slow": s_slow,
                "strength": effective_strength,
            },
        )

    def delete_episodic_events_for_memory(self, memory_id: str) -> int:
        with self._get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM episodic_events WHERE memory_id = ?",
                (memory_id,),
            )
            return int(cur.rowcount or 0)

    def add_episodic_events(self, events: List[Dict[str, Any]]) -> int:
        if not events:
            return 0
        rows = []
        for event in events:
            rows.append(
                (
                    event.get("id"),
                    event.get("memory_id"),
                    event.get("user_id"),
                    event.get("conversation_id"),
                    event.get("session_id"),
                    int(event.get("turn_id", 0) or 0),
                    event.get("actor_id"),
                    event.get("actor_role"),
                    event.get("event_time"),
                    event.get("event_type"),
                    event.get("canonical_key"),
                    event.get("value_text"),
                    event.get("value_num"),
                    event.get("value_unit"),
                    event.get("currency"),
                    event.get("normalized_time_start"),
                    event.get("normalized_time_end"),
                    event.get("time_granularity"),
                    event.get("entity_key"),
                    event.get("value_norm"),
                    event.get("confidence", 0.0),
                    event.get("superseded_by"),
                )
            )
        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO episodic_events (
                    id, memory_id, user_id, conversation_id, session_id, turn_id,
                    actor_id, actor_role, event_time, event_type, canonical_key,
                    value_text, value_num, value_unit, currency,
                    normalized_time_start, normalized_time_end, time_granularity,
                    entity_key, value_norm, confidence, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_episodic_events(
        self,
        *,
        user_id: str,
        actor_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        time_anchor: Optional[str] = None,
        entity_hints: Optional[List[str]] = None,
        include_superseded: bool = False,
        limit: int = 300,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM episodic_events WHERE user_id = ?"
        params: List[Any] = [user_id]
        if actor_id:
            query += " AND actor_id = ?"
            params.append(actor_id)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        if time_anchor:
            query += " AND COALESCE(normalized_time_start, event_time) <= ?"
            params.append(str(time_anchor))
        normalized_hints = [
            str(h).strip().lower()
            for h in (entity_hints or [])
            if str(h).strip()
        ]
        if normalized_hints:
            clauses = []
            for hint in normalized_hints:
                wildcard = f"%{hint}%"
                clauses.append(
                    "("
                    "LOWER(COALESCE(entity_key, '')) LIKE ? "
                    "OR LOWER(COALESCE(actor_id, '')) LIKE ? "
                    "OR LOWER(COALESCE(actor_role, '')) LIKE ?"
                    ")"
                )
                params.extend([wildcard, wildcard, wildcard])
            query += " AND (" + " OR ".join(clauses) + ")"
        if not include_superseded:
            query += " AND (superseded_by IS NULL OR superseded_by = '')"
        query += (
            " ORDER BY COALESCE(normalized_time_start, event_time) DESC, "
            "turn_id DESC LIMIT ?"
        )
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def record_cost_counter(
        self,
        *,
        phase: str,
        user_id: Optional[str] = None,
        llm_calls: float = 0.0,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        embed_calls: float = 0.0,
    ) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO cost_counters (
                    user_id, phase, llm_calls, input_tokens, output_tokens,
                    embed_calls
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    str(phase),
                    float(llm_calls or 0.0),
                    float(input_tokens or 0.0),
                    float(output_tokens or 0.0),
                    float(embed_calls or 0.0),
                ),
            )

    def aggregate_cost_counters(
        self,
        *,
        phase: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        query = """
            SELECT
                COUNT(*) AS samples,
                COALESCE(SUM(llm_calls), 0) AS llm_calls,
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(embed_calls), 0) AS embed_calls
            FROM cost_counters
            WHERE phase = ?
        """
        params: List[Any] = [str(phase)]
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        with self._get_connection() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return {
                "phase": phase,
                "samples": 0,
                "llm_calls": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "embed_calls": 0.0,
            }
        return {
            "phase": phase,
            "samples": int(row["samples"] or 0),
            "llm_calls": float(row["llm_calls"] or 0.0),
            "input_tokens": float(row["input_tokens"] or 0.0),
            "output_tokens": float(row["output_tokens"] or 0.0),
            "embed_calls": float(row["embed_calls"] or 0.0),
        }

    def get_constellation_data(
        self,
        user_id: Optional[str] = None,
        limit: int = 200,
    ) -> Dict[str, Any]:
        """Build graph data for the constellation visualizer."""
        with self._get_connection() as conn:
            mem_query = (
                "SELECT id, memory, strength, layer, categories, created_at "
                "FROM memories WHERE tombstone = 0"
            )
            params: List[Any] = []
            if user_id:
                mem_query += " AND user_id = ?"
                params.append(user_id)
            mem_query += " ORDER BY strength DESC LIMIT ?"
            params.append(limit)
            mem_rows = conn.execute(mem_query, params).fetchall()

            nodes = []
            node_ids = set()
            for row in mem_rows:
                nodes.append(
                    {
                        "id": row["id"],
                        "memory": (row["memory"] or "")[:120],
                        "strength": row["strength"],
                        "layer": row["layer"],
                        "categories": self._parse_json_value(
                            row["categories"], []
                        ),
                        "created_at": row["created_at"],
                    }
                )
                node_ids.add(row["id"])

            edges: List[Dict[str, Any]] = []
            if node_ids:
                placeholders = ",".join("?" for _ in node_ids)
                scene_rows = conn.execute(
                    f"""
                    SELECT a.memory_id AS source, b.memory_id AS target, a.scene_id
                    FROM scene_memories a
                    JOIN scene_memories b
                        ON a.scene_id = b.scene_id
                        AND a.memory_id < b.memory_id
                    WHERE a.memory_id IN ({placeholders})
                        AND b.memory_id IN ({placeholders})
                    """,
                    list(node_ids) + list(node_ids),
                ).fetchall()
                for row in scene_rows:
                    edges.append(
                        {
                            "source": row["source"],
                            "target": row["target"],
                            "type": "scene",
                        }
                    )

                profile_rows = conn.execute(
                    f"""
                    SELECT a.memory_id AS source, b.memory_id AS target, a.profile_id
                    FROM profile_memories a
                    JOIN profile_memories b
                        ON a.profile_id = b.profile_id
                        AND a.memory_id < b.memory_id
                    WHERE a.memory_id IN ({placeholders})
                        AND b.memory_id IN ({placeholders})
                    """,
                    list(node_ids) + list(node_ids),
                ).fetchall()
                for row in profile_rows:
                    edges.append(
                        {
                            "source": row["source"],
                            "target": row["target"],
                            "type": "profile",
                        }
                    )

        return {"nodes": nodes, "edges": edges}

    def get_decay_log_entries(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent decay log entries for the dashboard sparkline."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM decay_log ORDER BY run_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _ensure_entity_table(self, conn: sqlite3.Connection) -> None:
        """Lazily ensure entity_aggregates table exists."""
        conn.execute(
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
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entity_agg_lookup
                ON entity_aggregates(user_id, agg_type, entity_key)
            """
        )

    def upsert_entity_aggregate(
        self,
        user_id: str,
        entity_key: str,
        agg_type: str,
        value_delta: float,
        value_unit: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> None:
        """Increment an entity aggregate and append provenance metadata."""
        agg_id = hashlib.sha256(
            f"{user_id}|{agg_type}|{entity_key}".encode()
        ).hexdigest()
        now = _utcnow_iso()
        with self._get_connection() as conn:
            self._ensure_entity_table(conn)
            existing = conn.execute(
                """
                SELECT value_num, contributing_sessions, contributing_memory_ids
                FROM entity_aggregates WHERE id = ?
                """,
                (agg_id,),
            ).fetchone()
            if existing:
                cur_val = float(existing["value_num"] or 0)
                sessions = self._parse_json_value(
                    existing["contributing_sessions"], []
                )
                memories = self._parse_json_value(
                    existing["contributing_memory_ids"], []
                )
                if session_id and session_id not in sessions:
                    sessions.append(session_id)
                if memory_id and memory_id not in memories:
                    memories.append(memory_id)
                conn.execute(
                    """
                    UPDATE entity_aggregates
                    SET value_num = ?, value_unit = COALESCE(?, value_unit),
                        contributing_sessions = ?, contributing_memory_ids = ?,
                        last_updated = ?
                    WHERE id = ?
                    """,
                    (
                        cur_val + value_delta,
                        value_unit,
                        json.dumps(sessions),
                        json.dumps(memories),
                        now,
                        agg_id,
                    ),
                )
            else:
                sessions = [session_id] if session_id else []
                memories = [memory_id] if memory_id else []
                conn.execute(
                    """
                    INSERT INTO entity_aggregates (
                        id, user_id, entity_key, agg_type, value_num, value_unit,
                        item_set, contributing_sessions, contributing_memory_ids,
                        last_updated, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?)
                    """,
                    (
                        agg_id,
                        user_id,
                        entity_key,
                        agg_type,
                        value_delta,
                        value_unit,
                        json.dumps(sessions),
                        json.dumps(memories),
                        now,
                        now,
                    ),
                )

    def upsert_entity_set_member(
        self,
        user_id: str,
        entity_key: str,
        item_value: str,
        session_id: Optional[str] = None,
        memory_id: Optional[str] = None,
    ) -> None:
        """Add a unique item to an item_set aggregate and increment count."""
        agg_id = hashlib.sha256(
            f"{user_id}|item_set|{entity_key}".encode()
        ).hexdigest()
        now = _utcnow_iso()
        with self._get_connection() as conn:
            self._ensure_entity_table(conn)
            existing = conn.execute(
                """
                SELECT value_num, item_set, contributing_sessions,
                       contributing_memory_ids
                FROM entity_aggregates WHERE id = ?
                """,
                (agg_id,),
            ).fetchone()
            if existing:
                items = self._parse_json_value(existing["item_set"], [])
                sessions = self._parse_json_value(
                    existing["contributing_sessions"], []
                )
                memories = self._parse_json_value(
                    existing["contributing_memory_ids"], []
                )
                if item_value not in items:
                    items.append(item_value)
                if session_id and session_id not in sessions:
                    sessions.append(session_id)
                if memory_id and memory_id not in memories:
                    memories.append(memory_id)
                conn.execute(
                    """
                    UPDATE entity_aggregates
                    SET value_num = ?, item_set = ?,
                        contributing_sessions = ?, contributing_memory_ids = ?,
                        last_updated = ?
                    WHERE id = ?
                    """,
                    (
                        len(items),
                        json.dumps(items),
                        json.dumps(sessions),
                        json.dumps(memories),
                        now,
                        agg_id,
                    ),
                )
            else:
                sessions = [session_id] if session_id else []
                memories = [memory_id] if memory_id else []
                conn.execute(
                    """
                    INSERT INTO entity_aggregates (
                        id, user_id, entity_key, agg_type, value_num,
                        value_unit, item_set, contributing_sessions,
                        contributing_memory_ids, last_updated, created_at
                    ) VALUES (?, ?, ?, 'item_set', 1, NULL, ?, ?, ?, ?, ?)
                    """,
                    (
                        agg_id,
                        user_id,
                        entity_key,
                        json.dumps([item_value]),
                        json.dumps(sessions),
                        json.dumps(memories),
                        now,
                        now,
                    ),
                )

    def get_entity_aggregates(
        self,
        user_id: str,
        agg_type: Optional[str] = None,
        entity_hints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query entity aggregates with optional fuzzy match on entity_key."""
        with self._get_connection() as conn:
            self._ensure_entity_table(conn)
            if agg_type and entity_hints:
                conditions = " OR ".join(
                    ["entity_key LIKE ?" for _ in entity_hints]
                )
                params: List[Any] = [user_id, agg_type] + [
                    f"%{hint}%"
                    for hint in entity_hints
                ]
                rows = conn.execute(
                    f"""
                    SELECT * FROM entity_aggregates
                    WHERE user_id = ? AND agg_type = ? AND ({conditions})
                    """,
                    params,
                ).fetchall()
            elif agg_type:
                rows = conn.execute(
                    """
                    SELECT * FROM entity_aggregates
                    WHERE user_id = ? AND agg_type = ?
                    """,
                    (user_id, agg_type),
                ).fetchall()
            elif entity_hints:
                conditions = " OR ".join(
                    ["entity_key LIKE ?" for _ in entity_hints]
                )
                params = [user_id] + [f"%{hint}%" for hint in entity_hints]
                rows = conn.execute(
                    f"""
                    SELECT * FROM entity_aggregates
                    WHERE user_id = ? AND ({conditions})
                    """,
                    params,
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entity_aggregates WHERE user_id = ?",
                    (user_id,),
                ).fetchall()
            return [dict(row) for row in rows]

    def delete_entity_aggregates_for_user(self, user_id: str) -> int:
        """Delete all entity aggregates for a user."""
        with self._get_connection() as conn:
            self._ensure_entity_table(conn)
            cursor = conn.execute(
                "DELETE FROM entity_aggregates WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount or 0)

    def _route_decision_row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["metadata"] = self._parse_json_value(data.get("metadata"), {})
        return data

    def record_route_decision(self, decision: Dict[str, Any]) -> str:
        """Persist a critical-surface routing decision."""
        packet_kind = str(decision.get("packet_kind") or "").strip()
        route = str(decision.get("route") or "").strip()
        if not packet_kind:
            raise ValueError("packet_kind is required")
        if not route:
            raise ValueError("route is required")

        decision_id = str(decision.get("id") or uuid.uuid4())
        now = str(decision.get("created_at") or _utcnow_iso())
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            conn.execute(
                """
                INSERT INTO route_decisions (
                    id, user_id, source_event_id, packet_kind, route,
                    depth_score, semantic_fit, structural_fit, novelty,
                    confidence, locality_scope, project_id, workspace_id, folder_path,
                    session_id, thread_id, runtime_id, agent_id,
                    source_path, token_delta, outcome_alignment, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    str(decision.get("user_id") or "default"),
                    decision.get("source_event_id"),
                    packet_kind,
                    route,
                    float(decision.get("depth_score") or 0.0),
                    float(decision.get("semantic_fit") or 0.0),
                    float(decision.get("structural_fit") or 0.0),
                    float(decision.get("novelty") or 0.0),
                    float(decision.get("confidence") or 0.0),
                    str(decision.get("locality_scope") or "global"),
                    decision.get("project_id"),
                    decision.get("workspace_id"),
                    decision.get("folder_path"),
                    decision.get("session_id"),
                    decision.get("thread_id"),
                    decision.get("runtime_id"),
                    decision.get("agent_id"),
                    decision.get("source_path"),
                    int(decision.get("token_delta") or 0),
                    (
                        None
                        if decision.get("outcome_alignment") is None
                        else float(decision.get("outcome_alignment"))
                    ),
                    json.dumps(decision.get("metadata") or {}),
                    now,
                ),
            )
        return decision_id

    def list_route_decisions(
        self,
        *,
        user_id: str = "default",
        packet_kind: Optional[str] = None,
        route: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM route_decisions WHERE user_id = ?"
        params: List[Any] = [user_id]
        if packet_kind:
            query += " AND packet_kind = ?"
            params.append(packet_kind)
        if route:
            query += " AND route = ?"
            params.append(route)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._get_connection() as conn:
            self._ensure_project_graph_tables(conn)
            rows = conn.execute(query, params).fetchall()
        return [self._route_decision_row_to_dict(row) for row in rows]

    def summarize_route_decisions(
        self,
        *,
        user_id: str = "default",
        limit: int = 5000,
    ) -> Dict[str, Any]:
        rows = self.list_route_decisions(user_id=user_id, limit=limit)
        if not rows:
            return {
                "total_decisions": 0,
                "by_route": {},
                "by_packet_kind": {},
                "by_locality_scope": {},
                "avg_depth_score": 0.0,
                "avg_semantic_fit": 0.0,
                "avg_structural_fit": 0.0,
                "avg_novelty": 0.0,
                "avg_confidence": 0.0,
                "avg_outcome_alignment": None,
                "total_token_delta": 0,
                "read_saved_tokens": 0,
                "bash_saved_tokens": 0,
                "grep_saved_tokens": 0,
                "artifact_reuse_saved_tokens": 0,
            }

        by_route: Dict[str, int] = {}
        by_packet_kind: Dict[str, int] = {}
        by_locality_scope: Dict[str, int] = {}
        depth_total = semantic_total = structural_total = 0.0
        novelty_total = confidence_total = 0.0
        outcome_total = 0.0
        outcome_count = 0
        token_total = 0
        read_saved = 0
        bash_saved = 0
        grep_saved = 0
        artifact_reuse_saved = 0

        for row in rows:
            route = str(row.get("route") or "unknown")
            packet_kind = str(row.get("packet_kind") or "unknown")
            locality_scope = str(row.get("locality_scope") or "global")
            by_route[route] = by_route.get(route, 0) + 1
            by_packet_kind[packet_kind] = by_packet_kind.get(packet_kind, 0) + 1
            by_locality_scope[locality_scope] = by_locality_scope.get(locality_scope, 0) + 1

            depth_total += float(row.get("depth_score") or 0.0)
            semantic_total += float(row.get("semantic_fit") or 0.0)
            structural_total += float(row.get("structural_fit") or 0.0)
            novelty_total += float(row.get("novelty") or 0.0)
            confidence_total += float(row.get("confidence") or 0.0)

            token_delta = int(row.get("token_delta") or 0)
            token_total += token_delta
            if packet_kind == "routed_read":
                read_saved += token_delta
            elif packet_kind == "routed_bash":
                bash_saved += token_delta
            elif packet_kind == "routed_grep":
                grep_saved += token_delta
            elif packet_kind == "artifact_reuse":
                artifact_reuse_saved += token_delta

            if row.get("outcome_alignment") is not None:
                outcome_total += float(row.get("outcome_alignment"))
                outcome_count += 1

        count = len(rows)
        return {
            "total_decisions": count,
            "by_route": by_route,
            "by_packet_kind": by_packet_kind,
            "by_locality_scope": by_locality_scope,
            "avg_depth_score": round(depth_total / count, 4),
            "avg_semantic_fit": round(semantic_total / count, 4),
            "avg_structural_fit": round(structural_total / count, 4),
            "avg_novelty": round(novelty_total / count, 4),
            "avg_confidence": round(confidence_total / count, 4),
            "avg_outcome_alignment": (
                round(outcome_total / outcome_count, 4) if outcome_count else None
            ),
            "total_token_delta": token_total,
            "read_saved_tokens": read_saved,
            "bash_saved_tokens": bash_saved,
            "grep_saved_tokens": grep_saved,
            "artifact_reuse_saved_tokens": artifact_reuse_saved,
        }

    def delete_route_decisions_for_user(self, user_id: str) -> int:
        with self._get_connection() as conn:
            self._ensure_route_decision_tables(conn)
            cursor = conn.execute(
                "DELETE FROM route_decisions WHERE user_id = ?",
                (user_id,),
            )
            return int(cursor.rowcount or 0)
