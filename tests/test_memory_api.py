"""Tests for memory lifecycle API endpoints."""

import pytest
from context_engine.schema import init_db
from context_engine.server import create_app


@pytest.fixture
def client_with_memory(tmp_db, tmp_path):
    """Flask test client with agent memory configured."""
    init_db(tmp_db)

    # Create config dir with agent memory config
    config_dir = tmp_path / ".context-engine"
    config_dir.mkdir()

    iris_memory = tmp_path / "iris_memory.md"
    iris_memory.write_text(
        "# Memory\n\n"
        "## Old Project\n"
        "Trust cage shipped (2026-04-01)\n\n"
        "## Recent Work\n"
        "Memory cleanup in progress (2026-06-10)\n"
    )

    config = config_dir / "config.yaml"
    config.write_text(
        f"identity:\n"
        f"  soul: ''\n"
        f"  user_model: ''\n"
        f"agents:\n"
        f"  test-agent:\n"
        f"    memory_path: {iris_memory}\n"
        f"    max_chars: 100\n"
        f"memory:\n"
        f"  capacity_threshold: 0.8\n"
        f"  aging_days: 30\n"
    )

    app = create_app(db_path=tmp_db, config_dir=str(config_dir))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestMemorySyncAPI:
    def test_sync_returns_health(self, client_with_memory):
        resp = client_with_memory.post(
            "/context/memory/sync",
            json={"agent_id": "test-agent"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["agent_id"] == "test-agent"
        assert "utilization" in data
        assert "entry_count" in data
        assert data["entry_count"] == 2

    def test_sync_unknown_agent_returns_404(self, client_with_memory):
        resp = client_with_memory.post(
            "/context/memory/sync",
            json={"agent_id": "nonexistent"},
        )
        assert resp.status_code == 404


class TestMemoryArchiveAPI:
    def test_archive_entries(self, client_with_memory):
        resp = client_with_memory.post(
            "/context/memory/archive",
            json={
                "agent_id": "test-agent",
                "entries": [
                    {"name": "old-project", "content": "Trust cage details", "category": "project"}
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["archived"] == 1

    def test_archive_empty_list(self, client_with_memory):
        resp = client_with_memory.post(
            "/context/memory/archive",
            json={"agent_id": "test-agent", "entries": []},
        )
        assert resp.status_code == 200
        assert resp.get_json()["archived"] == 0


class TestMemoryRecallAPI:
    def test_recall_search(self, client_with_memory):
        client_with_memory.post(
            "/context/memory/archive",
            json={
                "agent_id": "test-agent",
                "entries": [
                    {"name": "trust-cage", "content": "5-layer zero-trust security", "category": "project"}
                ],
            },
        )
        resp = client_with_memory.get(
            "/context/memory/recall?agent_id=test-agent&q=trust+security"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] >= 1

    def test_recall_no_results(self, client_with_memory):
        resp = client_with_memory.get(
            "/context/memory/recall?agent_id=test-agent&q=nonexistent+thing"
        )
        assert resp.status_code == 200
        assert resp.get_json()["count"] == 0


class TestMemoryHealthAPI:
    def test_health_after_sync(self, client_with_memory):
        client_with_memory.post(
            "/context/memory/sync",
            json={"agent_id": "test-agent"},
        )
        resp = client_with_memory.get(
            "/context/memory/health?agent_id=test-agent"
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] in ("healthy", "warning", "critical")

    def test_health_before_sync_returns_404(self, client_with_memory):
        resp = client_with_memory.get(
            "/context/memory/health?agent_id=test-agent"
        )
        assert resp.status_code == 404


class TestBriefingIncludesMemoryHealth:
    def test_briefing_shows_capacity_warning(self, client_with_memory):
        client_with_memory.post(
            "/context/memory/sync",
            json={"agent_id": "test-agent"},
        )
        resp = client_with_memory.post(
            "/context/session/start",
            json={"session_id": "test-session", "agent": "test-agent"},
        )
        assert resp.status_code == 200
        briefing = resp.get_json()["briefing"]
        assert "Memory Health" in briefing
        assert "test-agent" in briefing
