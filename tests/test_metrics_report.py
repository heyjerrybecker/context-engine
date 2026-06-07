import sqlite3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.report import generate_report
from lib.metrics_db import init_db, insert_session


@pytest.fixture
def metrics_db(tmp_path):
    db_path = str(tmp_path / "metrics.db")
    conn = init_db(db_path)
    # Insert 2 cold sessions
    for i in range(2):
        insert_session(conn, {
            "session_id": f"cold-{i}", "date": "2026-06-01",
            "session_type": "cold", "agent": "scotty",
            "total_messages": 20, "total_tokens": 1000,
            "context_messages": 6, "context_tokens": 300,
            "warmup_message_count": 5, "correction_count": 2,
            "first_new_work_message_index": 5,
            "cor": 0.30, "cr": 0.10,
            "is_marathon": False, "length_bucket": "10-49",
            "created_at": "2026-06-01T10:00:00+00:00",
        })
    # Insert 1 warm session
    insert_session(conn, {
        "session_id": "warm-0", "date": "2026-06-06",
        "session_type": "warm", "agent": "scotty",
        "total_messages": 8, "total_tokens": 400,
        "context_messages": 1, "context_tokens": 50,
        "warmup_message_count": 1, "correction_count": 0,
        "first_new_work_message_index": 1,
        "cor": 0.125, "cr": 0.0,
        "is_marathon": False, "length_bucket": "10-49",
        "created_at": "2026-06-06T10:00:00+00:00",
    })
    return db_path


def test_report_contains_anxiety_map_section(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "ANXIETY MAP" in report
    assert "Marathon Session Rate" in report
    assert "Marathon Token Share" in report


def test_report_contains_efficiency_section(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "EFFICIENCY" in report
    assert "Context Overhead Ratio" in report
    assert "Warmup Message Count" in report
    assert "Correction Rate" in report


def test_report_shows_session_counts(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "2 cold" in report
    assert "1 warm" in report


def test_report_missing_baseline_returns_message(metrics_db):
    report = generate_report(["nonexistent", "warm"], metrics_db, cost_per_token=0.000015)
    assert "No sessions" in report


def test_report_cor_values_are_correct(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    # cold COR avg = 30.0%
    assert "30.0%" in report


def test_report_sld_histogram_present(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "10-49" in report  # length bucket in SLD
