import sqlite3
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.metrics_db import init_db, insert_session, insert_labels


@pytest.fixture
def conn(tmp_path):
    return init_db(str(tmp_path / "test.db"))


def test_schema_creates_tables(conn):
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    names = {r[0] for r in tables}
    assert "session_metrics" in names
    assert "message_labels" in names


def test_insert_session(conn):
    insert_session(conn, {
        "session_id": "abc123",
        "date": "2026-06-05",
        "session_type": "cold",
        "agent": "scotty",
        "total_messages": 10,
        "total_tokens": 500,
        "context_messages": 3,
        "context_tokens": 150,
        "warmup_message_count": 4,
        "correction_count": 1,
        "first_new_work_message_index": 4,
        "cor": 0.30,
        "cr": 0.10,
        "created_at": "2026-06-05T12:00:00+00:00",
    })
    row = conn.execute("SELECT session_id, cor FROM session_metrics WHERE session_id='abc123'").fetchone()
    assert row[0] == "abc123"
    assert abs(row[1] - 0.30) < 0.001


def test_insert_session_is_idempotent(conn):
    data = {
        "session_id": "abc123", "date": "2026-06-05", "session_type": "cold",
        "agent": "scotty", "total_messages": 10, "total_tokens": 500,
        "context_messages": 3, "context_tokens": 150, "warmup_message_count": 4,
        "correction_count": 1, "first_new_work_message_index": 4,
        "cor": 0.30, "cr": 0.10, "created_at": "2026-06-05T12:00:00+00:00",
    }
    insert_session(conn, data)
    insert_session(conn, data)  # second insert should not raise
    count = conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0]
    assert count == 1


def test_insert_labels(conn):
    insert_session(conn, {
        "session_id": "abc123", "date": "2026-06-05", "session_type": "cold",
        "agent": "scotty", "total_messages": 2, "total_tokens": 100,
        "context_messages": 1, "context_tokens": 50, "warmup_message_count": 1,
        "correction_count": 0, "first_new_work_message_index": 1,
        "cor": 0.50, "cr": 0.0, "created_at": "2026-06-05T12:00:00+00:00",
    })
    insert_labels(conn, "abc123", [
        {"index": 0, "role": "user", "label": "context-restoration", "token_count": 50, "is_correction": False},
        {"index": 1, "role": "assistant", "label": "new-work", "token_count": 50, "is_correction": False},
    ])
    rows = conn.execute("SELECT label FROM message_labels WHERE session_id='abc123' ORDER BY message_index").fetchall()
    assert [r[0] for r in rows] == ["context-restoration", "new-work"]


def test_insert_labels_enforces_foreign_key(conn):
    import sqlite3 as sqlite3_module
    with pytest.raises(sqlite3_module.IntegrityError):
        insert_labels(conn, "nonexistent_session", [
            {"index": 0, "role": "user", "label": "test", "token_count": 50, "is_correction": False}
        ])
