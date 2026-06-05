import pytest
from context_engine.models import (
    Entity, Thread, Relationship,
    EntityType, RelationshipType, ThreadStatus,
    CAUSAL_RELATIONSHIP_TYPES,
)


class TestEntity:
    def test_create_decision(self):
        e = Entity(
            type=EntityType.DECISION,
            content="Adopted 2:1 asymmetric scoring",
            source_agent="claude-code",
            source_session="session-001",
        )
        assert e.entity_id  # UUID auto-generated
        assert e.type == EntityType.DECISION
        assert e.confidence == 0.5  # default for new entities
        assert e.created_at is not None
        assert e.last_referenced is not None
        assert e.thread_id is None
        assert e.tags == []

    def test_create_agent_entity(self):
        e = Entity(
            type=EntityType.AGENT,
            content="Scotty",
            source_agent="manual",
            source_session="session-001",
            local_name="Scotty",
            role="technical-executor",
        )
        assert e.local_name == "Scotty"
        assert e.role == "technical-executor"
        assert e.export_as == "technical-executor"

    def test_entity_to_dict_roundtrip(self):
        e = Entity(
            type=EntityType.INSIGHT,
            content="Morning sessions are faster",
            source_agent="hermes",
            source_session="session-002",
            tags=["productivity", "timing"],
        )
        d = e.to_dict()
        assert d["type"] == "insight"
        assert d["tags"] == ["productivity", "timing"]
        e2 = Entity.from_dict(d)
        assert e2.entity_id == e.entity_id
        assert e2.content == e.content

    def test_invalid_entity_type_raises(self):
        with pytest.raises(ValueError):
            Entity(
                type="invalid_type",
                content="test",
                source_agent="test",
                source_session="test",
            )


class TestThread:
    def test_create_thread(self):
        t = Thread(title="Token cost → context architecture")
        assert t.thread_id
        assert t.status == ThreadStatus.ACTIVE
        assert t.resolved_at is None

    def test_thread_to_dict_roundtrip(self):
        t = Thread(title="Test thread", status=ThreadStatus.RESOLVED)
        d = t.to_dict()
        t2 = Thread.from_dict(d)
        assert t2.thread_id == t.thread_id
        assert t2.status == ThreadStatus.RESOLVED


class TestRelationship:
    def test_create_relationship(self):
        r = Relationship(
            from_id="entity-1",
            to_id="entity-2",
            type=RelationshipType.PRECEDED_BY,
        )
        assert r.relationship_id
        assert r.confidence == 1.0  # structural relationships default to 1.0

    def test_caused_by_is_causal(self):
        assert RelationshipType.CAUSED_BY in CAUSAL_RELATIONSHIP_TYPES

    def test_relates_to_is_not_causal(self):
        assert RelationshipType.RELATES_TO not in CAUSAL_RELATIONSHIP_TYPES
