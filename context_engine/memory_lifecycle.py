"""Memory lifecycle management for agent working memory."""

import hashlib
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone, date


class MemoryLifecycle:

    def __init__(self, db_path: str, config: dict):
        self.db_path = db_path
        self.config = config

    def sync_agent(self, agent_id: str) -> dict:
        agent_cfg = self.config.get("agents", {}).get(agent_id)
        if not agent_cfg:
            return {"error": f"Unknown agent: {agent_id}"}

        memory_path = agent_cfg["memory_path"]
        if not os.path.exists(memory_path):
            return {"error": f"Memory file not found: {memory_path}"}

        content = open(memory_path).read()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        current_chars = len(content)
        max_chars = agent_cfg["max_chars"]

        memory_dir = agent_cfg.get("memory_dir")
        if memory_dir:
            entries = self._parse_directory_entries(memory_dir)
        else:
            entries = self._parse_section_entries(content)

        aging_days = self.config.get("aging_days", 30)
        threshold = self.config.get("capacity_threshold", 0.8)
        today = date.today()

        aging_candidates = []
        for entry in entries:
            entry_date = self._detect_entry_date(entry.get("content", ""))
            if entry_date:
                try:
                    d = date.fromisoformat(entry_date)
                    age = (today - d).days
                    if age > aging_days:
                        aging_candidates.append({
                            "name": entry["name"],
                            "category": entry.get("category"),
                            "date": entry_date,
                            "age_days": age,
                        })
                except ValueError:
                    pass

        aging_candidates.sort(key=lambda c: c["age_days"], reverse=True)

        utilization = current_chars / max_chars if max_chars > 0 else 0
        if utilization > 0.95:
            status = "critical"
        elif utilization > threshold:
            status = "warning"
        else:
            status = "healthy"

        capacity_warning = None
        if status != "healthy":
            capacity_warning = (
                f"Working memory at {int(utilization * 100)}% capacity. "
                f"{len(aging_candidates)} entries may be ready for archival."
            )

        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO agent_memory (agent_id, memory_path, memory_dir, max_chars, "
            "current_chars, entry_count, content_hash, last_synced, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(agent_id) DO UPDATE SET "
            "current_chars=?, entry_count=?, content_hash=?, last_synced=?",
            (agent_id, memory_path, memory_dir, max_chars,
             current_chars, len(entries), content_hash, now, now,
             current_chars, len(entries), content_hash, now),
        )
        conn.commit()
        conn.close()

        return {
            "agent_id": agent_id,
            "utilization": round(utilization, 3),
            "current_chars": current_chars,
            "max_chars": max_chars,
            "entry_count": len(entries),
            "aging_candidates": aging_candidates,
            "capacity_warning": capacity_warning,
            "status": status,
            "content_hash": content_hash,
            "synced_at": now,
        }

    def get_health(self, agent_id: str) -> dict | None:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT current_chars, max_chars, entry_count, last_synced, content_hash "
            "FROM agent_memory WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        conn.close()

        if not row:
            return None

        current_chars, max_chars, entry_count, last_synced, content_hash = row
        utilization = current_chars / max_chars if max_chars > 0 else 0
        threshold = self.config.get("capacity_threshold", 0.8)

        if utilization > 0.95:
            status = "critical"
        elif utilization > threshold:
            status = "warning"
        else:
            status = "healthy"

        return {
            "agent_id": agent_id,
            "utilization": round(utilization, 3),
            "current_chars": current_chars,
            "max_chars": max_chars,
            "entry_count": entry_count,
            "last_synced": last_synced,
            "status": status,
        }

    def archive_entries(self, agent_id: str, entries: list[dict]) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        archived = 0
        names = []

        for entry in entries:
            archive_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO memory_archive "
                "(archive_id, agent_id, entry_name, content, category, "
                "original_path, entry_created_at, archived_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (archive_id, agent_id, entry["name"], entry["content"],
                 entry.get("category"), entry.get("original_path"),
                 entry.get("entry_created_at"), now),
            )
            archived += 1
            names.append(entry["name"])

        conn.commit()
        conn.close()

        return {
            "archived": archived,
            "entries": names,
        }

    def recall(self, agent_id: str, query: str, limit: int = 10) -> list[dict]:
        if not query.strip():
            return []

        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT a.archive_id, a.agent_id, a.entry_name, a.content, "
            "a.category, a.archived_at, a.entry_created_at "
            "FROM memory_archive_fts f "
            "JOIN memory_archive a ON f.archive_id = a.archive_id "
            "WHERE memory_archive_fts MATCH ? AND a.agent_id = ? "
            "ORDER BY rank LIMIT ?",
            (query, agent_id, limit),
        ).fetchall()
        conn.close()

        return [
            {
                "archive_id": r[0],
                "agent_id": r[1],
                "entry_name": r[2],
                "content": r[3],
                "category": r[4],
                "archived_at": r[5],
                "entry_created_at": r[6],
            }
            for r in rows
        ]

    def get_registered_agents(self) -> list[str]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT agent_id FROM agent_memory").fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _parse_section_entries(self, content: str) -> list[dict]:
        sections = re.split(r'\n(?=## )', content)
        entries = []
        for section in sections:
            match = re.match(r'## (.+)', section)
            if not match:
                continue
            title = match.group(1).strip()
            body = section[match.end():].strip()
            entries.append({
                "name": re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-'),
                "title": title,
                "content": body,
                "category": None,
            })
        return entries

    def _parse_directory_entries(self, memory_dir: str) -> list[dict]:
        entries = []
        for fname in os.listdir(memory_dir):
            if not fname.endswith('.md') or fname == 'MEMORY.md':
                continue
            path = os.path.join(memory_dir, fname)
            content = open(path).read()

            name = fname.replace('.md', '')
            category = None
            fm = re.match(r'^---\n(.+?)\n---', content, re.DOTALL)
            if fm:
                fm_text = fm.group(1)
                nm = re.search(r'^name:\s*(.+)$', fm_text, re.MULTILINE)
                if nm:
                    name = nm.group(1).strip()
                tp = re.search(r'type:\s*(.+)$', fm_text, re.MULTILINE)
                if tp:
                    category = tp.group(1).strip()
                content = content[fm.end():].strip()

            entries.append({
                "name": name,
                "content": content,
                "category": category,
                "path": path,
            })
        return entries

    def _detect_entry_date(self, content: str) -> str | None:
        dates = re.findall(r'\b(20\d{2}-\d{2}-\d{2})\b', content)
        if not dates:
            return None
        dates.sort(reverse=True)
        return dates[0]
