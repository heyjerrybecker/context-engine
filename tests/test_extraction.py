import pytest
from context_engine.extraction import extract_decisions, extract_thread_signals, extract_entities


class TestDecisionExtraction:
    def test_explicit_decision_lets_go_with(self):
        results = extract_decisions("Let's go with the 2:1 asymmetric ratio for scoring.")
        assert len(results) == 1
        assert "2:1 asymmetric ratio" in results[0]["content"]
        assert results[0]["confidence"] == 0.5

    def test_explicit_decision_decided(self):
        results = extract_decisions("I've decided we should use Flask instead of FastAPI.")
        assert len(results) == 1
        assert "Flask" in results[0]["content"]

    def test_explicit_decision_lock_it_in(self):
        results = extract_decisions("Lock it in — SQLite with FTS5.")
        assert len(results) == 1

    def test_explicit_decision_well_use(self):
        results = extract_decisions("We'll use the global region for Vertex AI.")
        assert len(results) == 1

    def test_no_decision_in_question(self):
        results = extract_decisions("Should we use Flask or FastAPI?")
        assert len(results) == 0

    def test_no_decision_in_description(self):
        results = extract_decisions("Flask is a lightweight web framework.")
        assert len(results) == 0


class TestThreadExtraction:
    def test_topic_change_detected(self):
        results = extract_thread_signals("Can we switch to talking about the Context Engine?")
        assert len(results) == 1
        assert results[0]["signal"] == "new_topic"
        assert "Context Engine" in results[0]["title"]

    def test_topic_resumption(self):
        results = extract_thread_signals("Going back to the Lightwell discussion.")
        assert len(results) == 1
        assert results[0]["signal"] == "resume"

    def test_no_thread_signal_in_normal_message(self):
        results = extract_thread_signals("The test suite has 62 tests passing.")
        assert len(results) == 0


class TestCombinedExtraction:
    def test_extract_entities_from_turn(self):
        user_msg = "Let's go with SQLite for the Context Engine storage."
        assistant_msg = "Good choice. SQLite with FTS5 gives us full-text search."
        entities = extract_entities(user_msg, assistant_msg, source_session="s1", source_agent="claude-code")
        decisions = [e for e in entities if e["type"] == "decision"]
        assert len(decisions) >= 1
        assert decisions[0]["source_session"] == "s1"
        assert decisions[0]["source_agent"] == "claude-code"

    def test_extract_entities_returns_empty_for_boring_turn(self):
        entities = extract_entities("Hi", "Hello! How can I help?", source_session="s1", source_agent="test")
        assert len(entities) == 0
