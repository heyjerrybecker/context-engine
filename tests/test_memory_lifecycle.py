"""Tests for memory lifecycle management."""

import os
from datetime import datetime, timedelta, timezone

import pytest

from context_engine.schema import init_db
from context_engine.memory_lifecycle import MemoryLifecycle


@pytest.fixture
def lifecycle(tmp_db, tmp_path):
    """Create a MemoryLifecycle instance with temp DB and memory files."""
    init_db(tmp_db)

    # Create Iris-style MEMORY.md (self-contained sections)
    iris_memory = tmp_path / "iris_memory"
    iris_memory.mkdir()
    iris_md = iris_memory / "MEMORY.md"
    iris_md.write_text(
        "# Memory Index\n\n"
        "## Strategic Context\n"
        "Gaia platform in active development. (2026-05-15)\n\n"
        "## Current Work\n"
        "Phase 1 memory cleanup active. (2026-06-10)\n\n"
        "## Architecture\n"
        "SOMA ecosystem with 4 scanner agents. (2026-04-01)\n"
    )

    # Create Scotty-style memory dir with satellite files
    scotty_memory = tmp_path / "scotty_memory"
    scotty_memory.mkdir()
    scotty_index = scotty_memory / "MEMORY.md"
    scotty_index.write_text(
        "# Memory Index\n\n"
        "## Projects\n"
        "- [Trust Cage](project_trust_cage.md) -- 5-layer security (2026-05-15)\n"
        "- [Gaia Vision](project_gaia_vision.md) -- North star vision (2026-06-01)\n\n"
        "## Feedback\n"
        "- [Git Push](feedback_git_push.md) -- Always confirm before push\n"
    )
    (scotty_memory / "project_trust_cage.md").write_text(
        "---\n"
        "name: trust-cage-built\n"
        "description: 5-layer zero-trust security\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        "Trust Cage shipped 2026-05-15. 94 tests, all passing.\n"
    )
    (scotty_memory / "project_gaia_vision.md").write_text(
        "---\n"
        "name: gaia-north-star\n"
        "description: Gaia north star vision\n"
        "metadata:\n"
        "  type: project\n"
        "---\n\n"
        "Reveal how Red Hat actually operates. (2026-06-01)\n"
    )
    (scotty_memory / "feedback_git_push.md").write_text(
        "---\n"
        "name: git-push-confirmation\n"
        "description: Always confirm before pushing\n"
        "metadata:\n"
        "  type: feedback\n"
        "---\n\n"
        "Always confirm before git push. Include repo name.\n"
    )

    config = {
        "agents": {
            "iris": {
                "memory_path": str(iris_md),
                "memory_dir": None,
                "max_chars": 200,
            },
            "scotty": {
                "memory_path": str(scotty_index),
                "memory_dir": str(scotty_memory),
                "max_chars": 2000,
            },
        },
        "capacity_threshold": 0.8,
        "aging_days": 30,
    }
    return MemoryLifecycle(tmp_db, config)


class TestParsing:
    def test_parse_iris_style_sections(self, lifecycle):
        result = lifecycle.sync_agent("iris")
        assert result["entry_count"] == 3
        assert result["current_chars"] > 0

    def test_parse_scotty_style_directory(self, lifecycle):
        result = lifecycle.sync_agent("scotty")
        assert result["entry_count"] == 3

    def test_unknown_agent_returns_error(self, lifecycle):
        result = lifecycle.sync_agent("unknown")
        assert "error" in result

    def test_missing_file_returns_error(self, lifecycle, tmp_path):
        lifecycle.config["agents"]["ghost"] = {
            "memory_path": str(tmp_path / "nonexistent.md"),
            "memory_dir": None,
            "max_chars": 1000,
        }
        result = lifecycle.sync_agent("ghost")
        assert "error" in result


class TestDateExtraction:
    def test_extract_iso_date(self, lifecycle):
        date = lifecycle._detect_entry_date("shipped 2026-06-04 successfully")
        assert date == "2026-06-04"

    def test_extract_parenthetical_date(self, lifecycle):
        date = lifecycle._detect_entry_date("Trust Cage built (2026-05-15)")
        assert date == "2026-05-15"

    def test_no_date_returns_none(self, lifecycle):
        date = lifecycle._detect_entry_date("no date here at all")
        assert date is None

    def test_most_recent_date_wins(self, lifecycle):
        date = lifecycle._detect_entry_date("started 2026-01-01, finished 2026-06-10")
        assert date == "2026-06-10"


class TestCapacityAnalysis:
    def test_healthy_utilization(self, lifecycle):
        result = lifecycle.sync_agent("scotty")
        assert result["status"] == "healthy"
        assert result["capacity_warning"] is None

    def test_over_capacity(self, lifecycle):
        result = lifecycle.sync_agent("iris")
        assert result["utilization"] > 0.8
        assert result["status"] in ("warning", "critical")
        assert result["capacity_warning"] is not None


class TestAgingAnalysis:
    def test_identifies_old_entries(self, lifecycle):
        result = lifecycle.sync_agent("iris")
        old_entries = [c for c in result.get("aging_candidates", [])
                       if c.get("age_days", 0) > 30]
        assert len(old_entries) >= 1

    def test_recent_entries_not_flagged(self, lifecycle):
        result = lifecycle.sync_agent("iris")
        recent = [c for c in result.get("aging_candidates", [])
                  if c.get("age_days", 0) < 2]
        assert len(recent) == 0


class TestArchiveAndRecall:
    def test_archive_stores_entry(self, lifecycle):
        lifecycle.sync_agent("iris")
        result = lifecycle.archive_entries("iris", [
            {"name": "old-project", "content": "Old project details about trust cage", "category": "project"}
        ])
        assert result["archived"] == 1

    def test_recall_by_query(self, lifecycle):
        lifecycle.sync_agent("iris")
        lifecycle.archive_entries("iris", [
            {"name": "trust-cage", "content": "5-layer zero-trust security architecture", "category": "project"},
            {"name": "gaia-vision", "content": "North star vision for the platform", "category": "project"},
        ])
        results = lifecycle.recall("iris", "trust cage")
        assert len(results) >= 1
        assert "trust" in results[0]["entry_name"].lower() or "trust" in results[0]["content"].lower()

    def test_recall_filters_by_agent(self, lifecycle):
        lifecycle.archive_entries("iris", [
            {"name": "iris-entry", "content": "Iris specific data", "category": "project"}
        ])
        lifecycle.archive_entries("scotty", [
            {"name": "scotty-entry", "content": "Scotty specific data", "category": "project"}
        ])
        results = lifecycle.recall("iris", "specific data")
        assert all(r["agent_id"] == "iris" for r in results)

    def test_recall_empty_query(self, lifecycle):
        results = lifecycle.recall("iris", "")
        assert results == []


class TestHealthCache:
    def test_get_health_returns_cached(self, lifecycle):
        lifecycle.sync_agent("iris")
        health = lifecycle.get_health("iris")
        assert health is not None
        assert "utilization" in health

    def test_get_health_before_sync(self, lifecycle):
        health = lifecycle.get_health("iris")
        assert health is None

    def test_sync_idempotent_on_unchanged(self, lifecycle):
        r1 = lifecycle.sync_agent("iris")
        r2 = lifecycle.sync_agent("iris")
        assert r1["content_hash"] == r2["content_hash"]
        assert r1["current_chars"] == r2["current_chars"]
