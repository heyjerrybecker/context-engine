import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.tagger import determine_session_type
from context_engine.schema import init_db


def _ts(offset_seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def graph_db(tmp_path):
    db_path = str(tmp_path / "graph.db")
    init_db(db_path)
    return db_path


def _add_session(db_path, session_id, started_at):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, "claude-code", started_at),
    )
    conn.commit()
    conn.close()


def _add_entity(db_path, created_at):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO entities (entity_id, type, content, confidence, created_at, last_referenced, source_agent, source_session) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (f"e-{created_at}", "decision", "some decision", 0.5, created_at, created_at, "claude-code", "old-session"),
    )
    conn.commit()
    conn.close()


def test_cold_no_prior_entities(graph_db):
    _add_session(graph_db, "sess-1", _ts())
    assert determine_session_type(graph_db, "sess-1") == "cold"


def test_warm_five_or_more_prior_entities(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-2", session_start)
    for i in range(6):
        _add_entity(graph_db, _ts(-100 - i))  # entities before session start
    assert determine_session_type(graph_db, "sess-2") == "warm"


def test_partial_one_to_four_prior_entities(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-3", session_start)
    _add_entity(graph_db, _ts(-50))
    assert determine_session_type(graph_db, "sess-3") == "partial"


def test_entities_after_session_start_dont_count(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-4", session_start)
    for i in range(10):
        _add_entity(graph_db, _ts(10 + i))  # entities AFTER session start
    assert determine_session_type(graph_db, "sess-4") == "cold"


def test_unknown_session_id_returns_cold(graph_db):
    assert determine_session_type(graph_db, "nonexistent-session") == "cold"


def test_unknown_string_session_id_returns_cold(graph_db):
    assert determine_session_type(graph_db, "unknown") == "cold"
