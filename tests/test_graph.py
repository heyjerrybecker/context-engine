import pytest
from context_engine.models import Entity, Thread, Relationship, EntityType, RelationshipType, ThreadStatus
from context_engine.graph import KnowledgeGraph
from context_engine.schema import init_db


@pytest.fixture
def graph(tmp_db):
    init_db(tmp_db)
    return KnowledgeGraph(tmp_db)


class TestEntityCRUD:
    def test_add_and_get_entity(self, graph):
        e = Entity(type=EntityType.DECISION, content="Use Flask", source_agent="test", source_session="s1")
        graph.add_entity(e)
        result = graph.get_entity(e.entity_id)
        assert result is not None
        assert result.content == "Use Flask"

    def test_get_missing_entity_returns_none(self, graph):
        assert graph.get_entity("nonexistent") is None

    def test_update_entity(self, graph):
        e = Entity(type=EntityType.DECISION, content="Use Flask", source_agent="test", source_session="s1")
        graph.add_entity(e)
        graph.update_entity(e.entity_id, content="Use FastAPI", confidence=0.7)
        result = graph.get_entity(e.entity_id)
        assert result.content == "Use FastAPI"
        assert result.confidence == 0.7

    def test_reinforce_entity(self, graph):
        e = Entity(type=EntityType.DECISION, content="Use Flask", source_agent="test", source_session="s1", confidence=0.5)
        graph.add_entity(e)
        graph.reinforce_entity(e.entity_id, boost=0.1)
        result = graph.get_entity(e.entity_id)
        assert result.confidence == pytest.approx(0.6)

    def test_reinforce_caps_at_1(self, graph):
        e = Entity(type=EntityType.DECISION, content="Sure thing", source_agent="test", source_session="s1", confidence=0.95)
        graph.add_entity(e)
        graph.reinforce_entity(e.entity_id, boost=0.1)
        result = graph.get_entity(e.entity_id)
        assert result.confidence == 1.0

    def test_search_entities_fts(self, graph):
        graph.add_entity(Entity(type=EntityType.DECISION, content="Adopted asymmetric scoring", source_agent="test", source_session="s1"))
        graph.add_entity(Entity(type=EntityType.INSIGHT, content="Morning sessions faster", source_agent="test", source_session="s1"))
        results = graph.search("asymmetric")
        assert len(results) == 1
        assert "asymmetric" in results[0].content

    def test_list_entities_by_type(self, graph):
        graph.add_entity(Entity(type=EntityType.DECISION, content="D1", source_agent="test", source_session="s1"))
        graph.add_entity(Entity(type=EntityType.INSIGHT, content="I1", source_agent="test", source_session="s1"))
        graph.add_entity(Entity(type=EntityType.DECISION, content="D2", source_agent="test", source_session="s1"))
        decisions = graph.list_entities(type_filter=EntityType.DECISION)
        assert len(decisions) == 2


class TestThreadCRUD:
    def test_add_and_get_thread(self, graph):
        t = Thread(title="Epistemic Architecture")
        graph.add_thread(t)
        result = graph.get_thread(t.thread_id)
        assert result is not None
        assert result.title == "Epistemic Architecture"

    def test_list_active_threads(self, graph):
        graph.add_thread(Thread(title="Active One"))
        graph.add_thread(Thread(title="Resolved", status=ThreadStatus.RESOLVED))
        active = graph.list_threads(status=ThreadStatus.ACTIVE)
        assert len(active) == 1
        assert active[0].title == "Active One"

    def test_update_thread_status(self, graph):
        t = Thread(title="WIP")
        graph.add_thread(t)
        graph.update_thread(t.thread_id, status=ThreadStatus.RESOLVED)
        result = graph.get_thread(t.thread_id)
        assert result.status == ThreadStatus.RESOLVED
        assert result.resolved_at is not None

    def test_get_thread_entities(self, graph):
        t = Thread(title="My Thread")
        graph.add_thread(t)
        e1 = Entity(type=EntityType.DECISION, content="First", source_agent="test", source_session="s1", thread_id=t.thread_id)
        e2 = Entity(type=EntityType.INSIGHT, content="Second", source_agent="test", source_session="s1", thread_id=t.thread_id)
        graph.add_entity(e1)
        graph.add_entity(e2)
        entities = graph.get_thread_entities(t.thread_id)
        assert len(entities) == 2


class TestRelationshipCRUD:
    def test_add_and_get_relationships(self, graph):
        e1 = Entity(type=EntityType.EVENT, content="E1", source_agent="test", source_session="s1")
        e2 = Entity(type=EntityType.DECISION, content="D1", source_agent="test", source_session="s1")
        graph.add_entity(e1)
        graph.add_entity(e2)
        r = Relationship(from_id=e2.entity_id, to_id=e1.entity_id, type=RelationshipType.CAUSED_BY)
        graph.add_relationship(r)
        rels = graph.get_relationships(e2.entity_id)
        assert len(rels) == 1
        assert rels[0].type == RelationshipType.CAUSED_BY

    def test_get_entity_with_relationships(self, graph):
        e1 = Entity(type=EntityType.DECISION, content="D1", source_agent="test", source_session="s1")
        e2 = Entity(type=EntityType.PROJECT, content="P1", source_agent="test", source_session="s1")
        graph.add_entity(e1)
        graph.add_entity(e2)
        graph.add_relationship(Relationship(from_id=e1.entity_id, to_id=e2.entity_id, type=RelationshipType.AFFECTS))
        entity, rels = graph.get_entity_with_relationships(e1.entity_id)
        assert entity.content == "D1"
        assert len(rels) == 1


class TestSessionCRUD:
    def test_start_and_end_session(self, graph):
        graph.start_session("s1", "claude-code")
        graph.end_session("s1", summary="Fixed a bug")
        session = graph.get_session("s1")
        assert session["agent"] == "claude-code"
        assert session["ended_at"] is not None
        assert session["summary"] == "Fixed a bug"

    def test_get_last_session(self, graph):
        graph.start_session("s1", "claude-code")
        graph.end_session("s1")
        graph.start_session("s2", "hermes")
        last = graph.get_last_ended_session()
        assert last["session_id"] == "s1"
