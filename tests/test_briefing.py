import pytest
from context_engine.models import Entity, Thread, EntityType, ThreadStatus
from context_engine.schema import init_db
from context_engine.graph import KnowledgeGraph
from context_engine.briefing import generate_briefing


@pytest.fixture
def graph(tmp_db):
    init_db(tmp_db)
    return KnowledgeGraph(tmp_db)


class TestBriefing:
    def test_empty_graph_produces_minimal_briefing(self, graph):
        briefing = generate_briefing(graph)
        assert "## Right Now" in briefing
        assert "Nothing in flight" in briefing

    def test_active_threads_in_right_now(self, graph):
        t = Thread(title="Building Context Engine")
        graph.add_thread(t)
        e = Entity(type=EntityType.DECISION, content="Use SQLite", source_agent="test",
                    source_session="s1", thread_id=t.thread_id)
        graph.add_entity(e)
        briefing = generate_briefing(graph)
        assert "Building Context Engine" in briefing

    def test_last_session_entities_included(self, graph):
        graph.start_session("s1", "claude-code")
        graph.add_entity(Entity(type=EntityType.DECISION, content="Shipped the proxy",
                                source_agent="test", source_session="s1"))
        graph.end_session("s1", summary="Got Hermes working")
        briefing = generate_briefing(graph)
        assert "Shipped the proxy" in briefing

    def test_open_threads_section(self, graph):
        t = Thread(title="Unresolved architecture question")
        graph.add_thread(t)
        briefing = generate_briefing(graph)
        assert "Unresolved architecture question" in briefing

    def test_resolved_threads_not_in_right_now(self, graph):
        t = Thread(title="Done thing", status=ThreadStatus.RESOLVED)
        graph.add_thread(t)
        briefing = generate_briefing(graph)
        assert "Done thing" not in briefing.split("## Open Threads")[0]
