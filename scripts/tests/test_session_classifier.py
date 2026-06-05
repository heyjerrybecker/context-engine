import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.session_classifier import classify_session, VALID_LABELS


def make_client(response_text: str):
    client = MagicMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=response_text)]
    client.messages.create.return_value = msg
    return client


def test_classify_returns_one_label_per_message():
    client = make_client('[{"index": 0, "label": "new-work"}, {"index": 1, "label": "context-restoration"}]')
    messages = [
        {"index": 0, "role": "user", "text": "Let's build the cache layer today."},
        {"index": 1, "role": "user", "text": "So just to recap — last session we chose Redis."},
    ]
    result = classify_session(client, messages)
    assert len(result) == 2
    assert result[0]["label"] == "new-work"
    assert result[1]["label"] == "context-restoration"


def test_classify_all_labels_are_valid():
    response = '[{"index": 0, "label": "new-work"}, {"index": 1, "label": "agent-clarifying"}]'
    client = make_client(response)
    messages = [
        {"index": 0, "role": "user", "text": "Implement the endpoint."},
        {"index": 1, "role": "assistant", "text": "What was the current state of the API again?"},
    ]
    result = classify_session(client, messages)
    for r in result:
        assert r["label"] in VALID_LABELS


def test_classify_fallback_on_bad_json():
    client = make_client("Sorry, I cannot classify these messages.")
    messages = [{"index": 0, "role": "user", "text": "Hello"}]
    result = classify_session(client, messages)
    assert len(result) == 1
    assert result[0]["label"] == "other"


def test_classify_chunks_large_sessions():
    responses = [
        '[' + ','.join(f'{{"index": {i}, "label": "new-work"}}' for i in range(30)) + ']',
        '[' + ','.join(f'{{"index": {i+30}, "label": "other"}}' for i in range(10)) + ']',
    ]
    call_count = [0]
    def side_effect(**kwargs):
        msg = MagicMock()
        msg.content = [MagicMock(text=responses[call_count[0]])]
        call_count[0] += 1
        return msg

    client = MagicMock()
    client.messages.create.side_effect = side_effect

    messages = [{"index": i, "role": "user", "text": f"message {i}"} for i in range(40)]
    result = classify_session(client, messages)
    assert len(result) == 40
    assert client.messages.create.call_count == 2  # chunked into 30 + 10


def test_empty_session_returns_empty():
    client = MagicMock()
    result = classify_session(client, [])
    assert result == []
    client.messages.create.assert_not_called()
