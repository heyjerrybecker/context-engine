import sqlite3
import pytest
from context_engine.schema import init_db


class TestSchema:
    def test_creates_entities_table(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "entities" in tables
        conn.close()

    def test_creates_threads_table(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "threads" in tables
        conn.close()

    def test_creates_relationships_table(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "relationships" in tables
        conn.close()

    def test_creates_sessions_table(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "sessions" in tables
        conn.close()

    def test_creates_fts_index(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "entities_fts" in tables
        conn.close()

    def test_idempotent(self, tmp_db):
        init_db(tmp_db)
        init_db(tmp_db)  # second call should not raise
        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        assert count == 0
        conn.close()

    def test_fts_search_works(self, tmp_db):
        init_db(tmp_db)
        conn = sqlite3.connect(tmp_db)
        conn.execute(
            "INSERT INTO entities (entity_id, type, content, confidence, "
            "created_at, last_referenced, source_session, source_agent, tags) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("e1", "decision", "Adopted asymmetric scoring ratio", 0.5,
             "2026-06-05T00:00:00", "2026-06-05T00:00:00", "s1", "claude", "[]"),
        )
        conn.commit()
        results = conn.execute(
            "SELECT entity_id FROM entities_fts WHERE entities_fts MATCH ?",
            ("asymmetric",),
        ).fetchall()
        assert len(results) == 1
        assert results[0][0] == "e1"
        conn.close()
