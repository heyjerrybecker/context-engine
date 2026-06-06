import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.lenses import extract_from_turn, extract_from_batch, _entities_from_result


def _mock_client(response_text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


def test_entities_from_result_decision():
    data = {"decision": {"content": "Use Redis for caching", "confidence": 0.6}, "thread": None}
    entities = _entities_from_result(data, "")
    assert len(entities) == 1
    assert entities[0]["type"] == "decision"
    assert entities[0]["content"] == "Use Redis for caching"
    assert entities[0]["confidence"] == 0.6


def test_entities_from_result_thread():
    data = {"decision": None, "thread": {"title": "Redis caching", "signal": "new_topic"}}
    entities = _entities_from_result(data, "")
    assert len(entities) == 1
    assert entities[0]["type"] == "thread_signal"
    assert entities[0]["signal"] == "new_topic"


def test_entities_from_result_both_null():
    entities = _entities_from_result({"decision": None, "thread": None}, "")
    assert entities == []


def test_entities_from_result_empty_content_skipped():
    data = {"decision": {"content": "", "confidence": 0.5}, "thread": None}
    assert _entities_from_result(data, "") == []


def test_extract_from_turn_returns_decision():
    client = _mock_client('{"decision": {"content": "Go with SQLite", "confidence": 0.5}, "thread": null}')
    with patch("context_engine.lenses._make_client", return_value=client):
        result = extract_from_turn("Let's go with SQLite.", "Good choice.")
    assert len(result) == 1
    assert result[0]["type"] == "decision"


def test_extract_from_turn_bad_json_returns_empty():
    client = _mock_client("Sorry, I cannot classify this.")
    with patch("context_engine.lenses._make_client", return_value=client):
        result = extract_from_turn("Hello", "Hi!")
    assert result == []


def test_extract_from_batch_returns_entities():
    client = _mock_client(
        '[{"turn": 0, "decision": {"content": "Use Haiku", "confidence": 0.6}, "thread": null},'
        ' {"turn": 1, "decision": null, "thread": null}]'
    )
    with patch("context_engine.lenses._make_client", return_value=client):
        result = extract_from_batch([
            ("Let's use Haiku.", "Good."),
            ("How are you?", "Fine."),
        ])
    assert any(e["type"] == "decision" for e in result)


def test_extract_from_batch_empty_returns_empty():
    with patch("context_engine.lenses._make_client", return_value=MagicMock()):
        result = extract_from_batch([])
    assert result == []
