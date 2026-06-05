"""End-to-end integration test: simulates a full session lifecycle."""

import pytest
from context_engine.schema import init_db
from context_engine.server import create_app


@pytest.fixture
def client(tmp_db, tmp_config_dir):
    init_db(tmp_db)
    app = create_app(db_path=tmp_db, config_dir=tmp_config_dir)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFullSessionLifecycle:
    def test_session_lifecycle(self, client):
        # 1. Start session — get briefing (empty graph)
        resp = client.post("/context/session/start", json={
            "agent": "claude-code", "session_id": "session-001"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "Nothing in flight" in data["briefing"]
        assert "Test Agent" in data["identity"]["soul"]

        # 2. Create a thread
        t_resp = client.post("/context/thread", json={
            "title": "Context Engine Phase 1"
        })
        thread_id = t_resp.get_json()["thread_id"]

        # 3. Add entities to the thread
        e1 = client.post("/context/entity", json={
            "type": "decision", "content": "Use SQLite with FTS5",
            "source_agent": "claude-code", "source_session": "session-001",
            "thread_id": thread_id, "confidence": 0.6,
        }).get_json()["entity_id"]

        e2 = client.post("/context/entity", json={
            "type": "decision", "content": "Flask for the HTTP API",
            "source_agent": "claude-code", "source_session": "session-001",
            "thread_id": thread_id, "confidence": 0.5,
        }).get_json()["entity_id"]

        e3 = client.post("/context/entity", json={
            "type": "insight", "content": "Confidence decay prevents graph bloat",
            "source_agent": "claude-code", "source_session": "session-001",
            "thread_id": thread_id, "tags": ["architecture"],
        }).get_json()["entity_id"]

        # 4. Add a relationship
        client.post("/context/relationship", json={
            "from_id": e2, "to_id": e1, "type": "preceded_by",
        })

        # 5. Search should find entities
        search_resp = client.get("/context/search?q=SQLite")
        assert len(search_resp.get_json()["results"]) == 1

        # 6. Thread should have all entities
        thread_resp = client.get(f"/context/thread/{thread_id}")
        assert len(thread_resp.get_json()["entities"]) == 3

        # 7. Entity should have relationships
        entity_resp = client.get(f"/context/entity/{e2}")
        assert len(entity_resp.get_json()["relationships"]) == 1

        # 8. End session
        end_resp = client.post("/context/session/end", json={
            "session_id": "session-001",
            "summary": "Built the Context Engine foundation"
        })
        assert end_resp.get_json()["briefing_written"] is True

        # 9. Start new session — briefing should include previous work
        resp2 = client.post("/context/session/start", json={
            "agent": "hermes", "session_id": "session-002"
        })
        briefing = resp2.get_json()["briefing"]
        assert "Context Engine Phase 1" in briefing
        assert "SQLite" in briefing

        # 10. Health check
        health = client.get("/context/health")
        assert health.get_json()["status"] == "ok"
