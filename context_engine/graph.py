"""Knowledge graph CRUD operations backed by SQLite."""

import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from context_engine.models import (
    Entity, Thread, Relationship,
    EntityType, ThreadStatus, RelationshipType,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class KnowledgeGraph:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # --- Entity operations ---

    def add_entity(self, entity: Entity) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO entities (entity_id, type, content, confidence, "
                "created_at, last_referenced, source_session, source_agent, "
                "thread_id, tags, local_name, role) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entity.entity_id, entity.type.value, entity.content,
                 entity.confidence, entity.created_at, entity.last_referenced,
                 entity.source_session, entity.source_agent,
                 entity.thread_id, json.dumps(entity.tags),
                 entity.local_name, entity.role),
            )
        return entity.entity_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("SELECT * FROM entities LIMIT 0").description]
            d = dict(zip(cols, row))
            d["tags"] = json.loads(d["tags"])
            return Entity.from_dict(d)

    def update_entity(self, entity_id: str, **kwargs) -> None:
        allowed = {"content", "confidence", "thread_id", "tags", "last_referenced"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"])
        updates["last_referenced"] = _now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entity_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE entities SET {set_clause} WHERE entity_id = ?", values)

    def reinforce_entity(self, entity_id: str, boost: float = 0.05) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE entities SET confidence = MIN(1.0, confidence + ?), "
                "last_referenced = ? WHERE entity_id = ?",
                (boost, _now_iso(), entity_id),
            )

    def search(self, query: str, limit: int = 20) -> List[Entity]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT e.* FROM entities e "
                "JOIN entities_fts f ON e.entity_id = f.entity_id "
                "WHERE entities_fts MATCH ? LIMIT ?",
                (query, limit),
            ).fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM entities LIMIT 0").description]
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                d["tags"] = json.loads(d["tags"])
                results.append(Entity.from_dict(d))
            return results

    def list_entities(self, type_filter: Optional[EntityType] = None,
                      session_filter: Optional[str] = None,
                      limit: int = 100) -> List[Entity]:
        query = "SELECT * FROM entities"
        params = []
        conditions = []
        if type_filter:
            conditions.append("type = ?")
            params.append(type_filter.value)
        if session_filter:
            conditions.append("source_session = ?")
            params.append(session_filter)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY last_referenced DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM entities LIMIT 0").description]
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                d["tags"] = json.loads(d["tags"])
                results.append(Entity.from_dict(d))
            return results

    # --- Thread operations ---

    def add_thread(self, thread: Thread) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO threads (thread_id, title, trigger_entity_id, "
                "status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?, ?)",
                (thread.thread_id, thread.title, thread.trigger_entity_id,
                 thread.status.value, thread.created_at, thread.resolved_at),
            )
        return thread.thread_id

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM threads WHERE thread_id = ?", (thread_id,)
            ).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("SELECT * FROM threads LIMIT 0").description]
            return Thread.from_dict(dict(zip(cols, row)))

    def list_threads(self, status: Optional[ThreadStatus] = None) -> List[Thread]:
        query = "SELECT * FROM threads"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM threads LIMIT 0").description]
            return [Thread.from_dict(dict(zip(cols, row))) for row in rows]

    def update_thread(self, thread_id: str, **kwargs) -> None:
        allowed = {"title", "status", "resolved_at", "trigger_entity_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "status" in updates:
            if isinstance(updates["status"], ThreadStatus):
                updates["status"] = updates["status"].value
            if updates["status"] == "resolved" and "resolved_at" not in updates:
                updates["resolved_at"] = _now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [thread_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE threads SET {set_clause} WHERE thread_id = ?", values)

    def get_thread_entities(self, thread_id: str) -> List[Entity]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM entities WHERE thread_id = ? ORDER BY created_at",
                (thread_id,),
            ).fetchall()
            cols = [d[0] for d in conn.execute("SELECT * FROM entities LIMIT 0").description]
            results = []
            for row in rows:
                d = dict(zip(cols, row))
                d["tags"] = json.loads(d["tags"])
                results.append(Entity.from_dict(d))
            return results

    # --- Relationship operations ---

    def add_relationship(self, rel: Relationship) -> str:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO relationships (relationship_id, from_id, to_id, "
                "type, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (rel.relationship_id, rel.from_id, rel.to_id,
                 rel.type.value, rel.confidence, rel.created_at),
            )
        return rel.relationship_id

    def get_relationships(self, entity_id: str, direction: str = "both") -> List[Relationship]:
        with self._conn() as conn:
            cols = [d[0] for d in conn.execute("SELECT * FROM relationships LIMIT 0").description]
            if direction == "outgoing":
                query = "SELECT * FROM relationships WHERE from_id = ?"
            elif direction == "incoming":
                query = "SELECT * FROM relationships WHERE to_id = ?"
            else:
                query = "SELECT * FROM relationships WHERE from_id = ? OR to_id = ?"
                rows = conn.execute(query, (entity_id, entity_id)).fetchall()
                return [Relationship.from_dict(dict(zip(cols, row))) for row in rows]
            rows = conn.execute(query, (entity_id,)).fetchall()
            return [Relationship.from_dict(dict(zip(cols, row))) for row in rows]

    def get_entity_with_relationships(self, entity_id: str) -> Tuple[Optional[Entity], List[Relationship]]:
        entity = self.get_entity(entity_id)
        if not entity:
            return None, []
        rels = self.get_relationships(entity_id)
        return entity, rels

    # --- Session operations ---

    def start_session(self, session_id: str, agent: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
                (session_id, agent, _now_iso()),
            )

    def end_session(self, session_id: str, summary: str = None, complexity_score: float = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET ended_at = ?, summary = ?, complexity_score = ? "
                "WHERE session_id = ?",
                (_now_iso(), summary, complexity_score, session_id),
            )

    def get_session(self, session_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("SELECT * FROM sessions LIMIT 0").description]
            return dict(zip(cols, row))

    def get_last_ended_session(self) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE ended_at IS NOT NULL ORDER BY ended_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("SELECT * FROM sessions LIMIT 0").description]
            return dict(zip(cols, row))
