"""SQLite schema for the Context Engine knowledge graph."""

import sqlite3


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            entity_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            last_referenced TEXT NOT NULL,
            source_session TEXT NOT NULL DEFAULT '',
            source_agent TEXT NOT NULL DEFAULT '',
            thread_id TEXT,
            tags TEXT NOT NULL DEFAULT '[]',
            local_name TEXT,
            role TEXT
        );

        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            trigger_entity_id TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS relationships (
            relationship_id TEXT PRIMARY KEY,
            from_id TEXT NOT NULL,
            to_id TEXT NOT NULL,
            type TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (from_id) REFERENCES entities(entity_id),
            FOREIGN KEY (to_id) REFERENCES entities(entity_id)
        );

        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            agent TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            summary TEXT,
            complexity_score REAL
        );

        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
        CREATE INDEX IF NOT EXISTS idx_entities_thread ON entities(thread_id);
        CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence);
        CREATE INDEX IF NOT EXISTS idx_entities_last_ref ON entities(last_referenced);
        CREATE INDEX IF NOT EXISTS idx_threads_status ON threads(status);
        CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_id);
        CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_id);
        CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(type);
        CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent);

        CREATE TABLE IF NOT EXISTS agent_memory (
            agent_id TEXT PRIMARY KEY,
            memory_path TEXT NOT NULL,
            memory_dir TEXT,
            max_chars INTEGER NOT NULL DEFAULT 4000,
            current_chars INTEGER NOT NULL DEFAULT 0,
            entry_count INTEGER NOT NULL DEFAULT 0,
            content_hash TEXT,
            last_synced TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_archive (
            archive_id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            entry_name TEXT NOT NULL,
            content TEXT NOT NULL,
            category TEXT,
            original_path TEXT,
            entry_created_at TEXT,
            archived_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_archive_agent ON memory_archive(agent_id);
    """)

    # FTS5 virtual table for full-text search on entity content
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts
        USING fts5(entity_id, content, tokenize='porter')
    """)

    # Triggers to keep FTS in sync
    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS entities_fts_insert
        AFTER INSERT ON entities BEGIN
            INSERT INTO entities_fts(entity_id, content)
            VALUES (new.entity_id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS entities_fts_update
        AFTER UPDATE OF content ON entities BEGIN
            DELETE FROM entities_fts WHERE entity_id = old.entity_id;
            INSERT INTO entities_fts(entity_id, content)
            VALUES (new.entity_id, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS entities_fts_delete
        AFTER DELETE ON entities BEGIN
            DELETE FROM entities_fts WHERE entity_id = old.entity_id;
        END;
    """)

    # FTS5 for memory archive search
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_archive_fts
        USING fts5(archive_id, entry_name, content, tokenize='porter')
    """)

    conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS archive_fts_insert
        AFTER INSERT ON memory_archive BEGIN
            INSERT INTO memory_archive_fts(archive_id, entry_name, content)
            VALUES (new.archive_id, new.entry_name, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS archive_fts_update
        AFTER UPDATE OF content ON memory_archive BEGIN
            DELETE FROM memory_archive_fts WHERE archive_id = old.archive_id;
            INSERT INTO memory_archive_fts(archive_id, entry_name, content)
            VALUES (new.archive_id, new.entry_name, new.content);
        END;

        CREATE TRIGGER IF NOT EXISTS archive_fts_delete
        AFTER DELETE ON memory_archive BEGIN
            DELETE FROM memory_archive_fts WHERE archive_id = old.archive_id;
        END;
    """)

    conn.commit()
    conn.close()
