"""Tests for the background extraction worker."""
import sys
import time
import queue as queue_module
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import context_engine.worker as worker_module


@pytest.fixture(autouse=True)
def reset_worker():
    """Reset module-level state between tests."""
    worker_module._queue = queue_module.Queue()
    worker_module._worker_thread = None
    yield
    worker_module._queue = queue_module.Queue()
    worker_module._worker_thread = None


def test_enqueue_adds_to_queue():
    worker_module.enqueue("hello", "world", "sess-1")
    assert not worker_module._queue.empty()
    item = worker_module._queue.get_nowait()
    assert item["user_msg"] == "hello"
    assert item["session_id"] == "sess-1"


def test_start_worker_creates_daemon_thread(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)
    worker_module.start_worker(db_path)
    assert worker_module._worker_thread is not None
    assert worker_module._worker_thread.is_alive()
    assert worker_module._worker_thread.daemon


def test_start_worker_does_not_duplicate(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)
    worker_module.start_worker(db_path)
    first_thread = worker_module._worker_thread
    worker_module.start_worker(db_path)
    assert worker_module._worker_thread is first_thread


def test_worker_processes_queue_item(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)

    extracted = [{"type": "decision", "content": "Use SQLite", "confidence": 0.5}]
    with patch("context_engine.lenses.extract_from_turn", return_value=extracted):
        worker_module.start_worker(db_path)
        worker_module.enqueue("Let's use SQLite.", "Good.", "sess-1")
        time.sleep(0.5)  # let the worker thread process

    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert count == 1


def test_worker_survives_extraction_error(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)

    with patch("context_engine.lenses.extract_from_turn", side_effect=RuntimeError("boom")):
        worker_module.start_worker(db_path)
        worker_module.enqueue("crash this", "please", "sess-err")
        time.sleep(0.3)

    assert worker_module._worker_thread.is_alive()
