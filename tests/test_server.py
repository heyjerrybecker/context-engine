import json
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


class TestHealth:
    def test_health_endpoint(self, client):
        resp = client.get("/context/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"


class TestIdentity:
    def test_identity_endpoint(self, client):
        resp = client.get("/context/identity")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "Test Agent" in data["soul"]
        assert "Test User" in data["user_model"]


class TestSessionAPI:
    def test_start_session(self, client):
        resp = client.post("/context/session/start", json={
            "agent": "claude-code", "session_id": "test-session-1"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "briefing" in data
        assert "active_threads" in data

    def test_end_session(self, client):
        client.post("/context/session/start", json={
            "agent": "claude-code", "session_id": "test-session-1"
        })
        resp = client.post("/context/session/end", json={
            "session_id": "test-session-1", "summary": "Did some work"
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["briefing_written"] is True


class TestEntityAPI:
    def test_create_entity(self, client):
        client.post("/context/session/start", json={"agent": "test", "session_id": "s1"})
        resp = client.post("/context/entity", json={
            "type": "decision", "content": "Use SQLite for storage",
            "source_agent": "test", "source_session": "s1",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "entity_id" in data

    def test_get_entity(self, client):
        client.post("/context/session/start", json={"agent": "test", "session_id": "s1"})
        create_resp = client.post("/context/entity", json={
            "type": "decision", "content": "Use SQLite",
            "source_agent": "test", "source_session": "s1",
        })
        entity_id = create_resp.get_json()["entity_id"]
        resp = client.get(f"/context/entity/{entity_id}")
        assert resp.status_code == 200
        assert resp.get_json()["entity"]["content"] == "Use SQLite"

    def test_search_entities(self, client):
        client.post("/context/session/start", json={"agent": "test", "session_id": "s1"})
        client.post("/context/entity", json={
            "type": "decision", "content": "Adopted asymmetric scoring",
            "source_agent": "test", "source_session": "s1",
        })
        resp = client.get("/context/search?q=asymmetric")
        assert resp.status_code == 200
        assert len(resp.get_json()["results"]) == 1


class TestThreadAPI:
    def test_create_thread(self, client):
        resp = client.post("/context/thread", json={
            "title": "Context Engine Design"
        })
        assert resp.status_code == 200
        assert "thread_id" in resp.get_json()

    def test_list_active_threads(self, client):
        client.post("/context/thread", json={"title": "Active Thread"})
        resp = client.get("/context/threads?status=active")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["threads"]) == 1

    def test_get_thread_with_entities(self, client):
        t_resp = client.post("/context/thread", json={"title": "My Thread"})
        thread_id = t_resp.get_json()["thread_id"]
        client.post("/context/entity", json={
            "type": "decision", "content": "Thread entity",
            "source_agent": "test", "source_session": "s1",
            "thread_id": thread_id,
        })
        resp = client.get(f"/context/thread/{thread_id}")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["thread"]["title"] == "My Thread"
        assert len(data["entities"]) == 1


class TestRelationshipAPI:
    def test_create_relationship(self, client):
        client.post("/context/session/start", json={"agent": "test", "session_id": "s1"})
        e1 = client.post("/context/entity", json={
            "type": "event", "content": "Meeting happened",
            "source_agent": "test", "source_session": "s1",
        }).get_json()["entity_id"]
        e2 = client.post("/context/entity", json={
            "type": "decision", "content": "Decision made",
            "source_agent": "test", "source_session": "s1",
        }).get_json()["entity_id"]
        resp = client.post("/context/relationship", json={
            "from_id": e2, "to_id": e1, "type": "caused_by",
        })
        assert resp.status_code == 200
        assert "relationship_id" in resp.get_json()


class TestBriefingAPI:
    def test_get_briefing(self, client):
        resp = client.get("/context/briefing")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "briefing" in data


class TestIngestAPI:
    def test_ingest_extracts_decision(self, client):
        client.post("/context/session/start", json={"agent": "test", "session_id": "s1"})
        resp = client.post("/context/ingest", json={
            "user_message": "Let's go with SQLite for storage.",
            "assistant_message": "Good choice.",
            "session_id": "s1",
            "source_agent": "claude-code",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["entities_extracted"] >= 1

    def test_ingest_boring_turn_extracts_nothing(self, client):
        resp = client.post("/context/ingest", json={
            "user_message": "Hi",
            "assistant_message": "Hello!",
            "session_id": "s1",
            "source_agent": "test",
        })
        assert resp.status_code == 200
        assert resp.get_json()["entities_extracted"] == 0
