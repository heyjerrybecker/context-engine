import math
import pytest
from datetime import datetime, timezone, timedelta
from context_engine.models import Entity, EntityType
from context_engine.schema import init_db
from context_engine.graph import KnowledgeGraph
from context_engine.confidence import decay_confidence, compute_decay


class TestDecayMath:
    def test_no_decay_when_recently_referenced(self):
        factor = compute_decay(days_since_referenced=0, half_life_days=14)
        assert factor == pytest.approx(1.0)

    def test_half_decay_at_half_life(self):
        factor = compute_decay(days_since_referenced=14, half_life_days=14)
        assert factor == pytest.approx(0.5, abs=0.01)

    def test_quarter_at_two_half_lives(self):
        factor = compute_decay(days_since_referenced=28, half_life_days=14)
        assert factor == pytest.approx(0.25, abs=0.01)

    def test_no_decay_within_grace_period(self):
        factor = compute_decay(days_since_referenced=5, half_life_days=14, grace_days=7)
        assert factor == pytest.approx(1.0)


class TestDecayEngine:
    def test_decay_reduces_old_entity_confidence(self, tmp_db):
        init_db(tmp_db)
        graph = KnowledgeGraph(tmp_db)
        old_time = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
        e = Entity(
            type=EntityType.INSIGHT, content="Old insight",
            source_agent="test", source_session="s1",
            confidence=0.8, last_referenced=old_time,
        )
        graph.add_entity(e)
        archived = decay_confidence(tmp_db, half_life_days=14, archive_threshold=0.1)
        result = graph.get_entity(e.entity_id)
        assert result.confidence < 0.8

    def test_decay_does_not_touch_recent_entities(self, tmp_db):
        init_db(tmp_db)
        graph = KnowledgeGraph(tmp_db)
        e = Entity(
            type=EntityType.DECISION, content="Recent decision",
            source_agent="test", source_session="s1",
            confidence=0.6,
        )
        graph.add_entity(e)
        decay_confidence(tmp_db, half_life_days=14, archive_threshold=0.1, grace_days=7)
        result = graph.get_entity(e.entity_id)
        assert result.confidence == pytest.approx(0.6)

    def test_decay_archives_below_threshold(self, tmp_db):
        init_db(tmp_db)
        graph = KnowledgeGraph(tmp_db)
        very_old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        e = Entity(
            type=EntityType.STATE, content="Ancient state",
            source_agent="test", source_session="s1",
            confidence=0.2, last_referenced=very_old,
        )
        graph.add_entity(e)
        archived = decay_confidence(tmp_db, half_life_days=14, archive_threshold=0.1)
        assert len(archived) == 1
        assert archived[0] == e.entity_id
