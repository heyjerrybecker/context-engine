import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.classifier import (
    find_transcript, compute_metrics, classify_session_async
)
from context_engine.schema import init_db
from lib.metrics_db import init_db as init_metrics_db


def test_find_transcript_returns_none_for_unknown(tmp_path):
    result = find_transcript("unknown", str(tmp_path))
    assert result is None


def test_find_transcript_returns_none_for_empty(tmp_path):
    result = find_transcript("", str(tmp_path))
    assert result is None


def test_find_transcript_returns_none_for_missing_session(tmp_path):
    result = find_transcript("00000000-0000-0000-0000-000000000000", str(tmp_path))
    assert result is None


def test_find_transcript_finds_existing_jsonl(tmp_path):
    session_id = "abc12345-0000-0000-0000-000000000000"
    proj = tmp_path / "project1"
    proj.mkdir()
    jsonl = proj / f"{session_id}.jsonl"
    jsonl.write_text('{"type": "user", "message": {"role": "user", "content": "hi"}}\n')
    result = find_transcript(session_id, str(tmp_path))
    assert result == jsonl


def test_compute_metrics_all_context():
    messages = [
        {"index": 0, "role": "user", "text": "re-explain", "token_count": 100},
        {"index": 1, "role": "assistant", "text": "clarify", "token_count": 100},
    ]
    labels = [{"index": 0, "label": "context-restoration"}, {"index": 1, "label": "agent-clarifying"}]
    result = compute_metrics(messages, labels)
    assert result["cor"] == 1.0
    assert result["context_messages"] == 2


def test_compute_metrics_no_new_work():
    messages = [{"index": 0, "role": "user", "text": "hi", "token_count": 10}]
    labels = [{"index": 0, "label": "other"}]
    result = compute_metrics(messages, labels)
    assert result["first_new_work_message_index"] is None
    assert result["warmup_message_count"] == 1


def test_classify_session_async_skips_unknown_session(tmp_path):
    db = str(tmp_path / "graph.db")
    init_db(db)
    mdb = str(tmp_path / "metrics.db")
    init_metrics_db(mdb)
    classify_session_async("unknown", db, mdb, projects_dir=str(tmp_path))
    conn = sqlite3.connect(mdb)
    assert conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0] == 0


def test_classify_session_async_skips_missing_transcript(tmp_path):
    db = str(tmp_path / "graph.db")
    init_db(db)
    mdb = str(tmp_path / "metrics.db")
    init_metrics_db(mdb)
    classify_session_async("session-not-found", db, mdb, projects_dir=str(tmp_path))
    conn = sqlite3.connect(mdb)
    assert conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0] == 0
