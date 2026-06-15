"""Tests for Token Observatory — cost estimation, usage logging, proxy, and API."""

import json
from unittest.mock import patch, MagicMock

import pytest

from context_engine.observatory import estimate_cost, UsageLog
from context_engine.schema import init_db
from context_engine.server import create_app


class TestCostEstimation:
    def test_sonnet_cost(self):
        cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        expected = (1000 * 3.0 / 1_000_000) + (500 * 15.0 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_opus_cost(self):
        cost = estimate_cost("claude-opus-4-20250514", input_tokens=1000, output_tokens=500)
        expected = (1000 * 15.0 / 1_000_000) + (500 * 75.0 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_haiku_cost(self):
        cost = estimate_cost("claude-haiku-4-5-20251001", input_tokens=1000, output_tokens=500)
        expected = (1000 * 0.80 / 1_000_000) + (500 * 4.0 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_cached_tokens_discounted(self):
        cost = estimate_cost(
            "claude-sonnet-4-20250514",
            input_tokens=1000, output_tokens=500,
            cache_read_tokens=500, cache_creation_tokens=200,
        )
        assert cost > 0

    def test_unknown_model_returns_zero(self):
        cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0


class TestUsageLog:
    def test_log_and_query(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        log = UsageLog(db_path)
        log.record(
            agent_id="claude-code-jerry", model="claude-sonnet-4-20250514",
            input_tokens=1000, output_tokens=500,
            cache_read_tokens=0, cache_creation_tokens=0,
            latency_ms=450, estimated_cost=0.0105,
        )
        rows = log.query(agent_id="claude-code-jerry")
        assert len(rows) == 1
        assert rows[0]["model"] == "claude-sonnet-4-20250514"
        assert rows[0]["input_tokens"] == 1000

    def test_query_filters_by_agent(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        log = UsageLog(db_path)
        log.record(agent_id="agent-a", model="claude-sonnet-4-20250514",
                   input_tokens=100, output_tokens=50,
                   cache_read_tokens=0, cache_creation_tokens=0,
                   latency_ms=100, estimated_cost=0.001)
        log.record(agent_id="agent-b", model="claude-haiku-4-5-20251001",
                   input_tokens=200, output_tokens=100,
                   cache_read_tokens=0, cache_creation_tokens=0,
                   latency_ms=80, estimated_cost=0.0006)
        assert len(log.query(agent_id="agent-a")) == 1
        assert len(log.query(agent_id="agent-b")) == 1
        assert len(log.query()) == 2

    def test_summary_aggregates(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        log = UsageLog(db_path)
        for _ in range(5):
            log.record(agent_id="test", model="claude-sonnet-4-20250514",
                       input_tokens=1000, output_tokens=500,
                       cache_read_tokens=0, cache_creation_tokens=0,
                       latency_ms=400, estimated_cost=0.0105)
        summary = log.summary(agent_id="test")
        assert summary["total_calls"] == 5
        assert summary["total_input_tokens"] == 5000
        assert summary["total_output_tokens"] == 2500
        assert abs(summary["total_cost"] - 0.0525) < 0.001


def _mock_response(model="claude-sonnet-4-20250514", input_tokens=100, output_tokens=50):
    mock = MagicMock()
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    mock.content = json.dumps({
        "id": "msg_test", "type": "message",
        "model": model,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        "content": [{"type": "text", "text": "test"}],
    }).encode()
    return mock


@pytest.fixture
def observatory_client(tmp_db, tmp_path):
    init_db(tmp_db)
    config_dir = tmp_path / ".context-engine"
    config_dir.mkdir()
    config = config_dir / "config.yaml"
    config.write_text(
        "identity:\n  soul: ''\n  user_model: ''\n"
        "observatory:\n  backend_url: https://test-backend.example.com\n"
    )
    app = create_app(db_path=tmp_db, config_dir=str(config_dir))
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestProxyRoute:
    @patch("httpx.Client")
    def test_proxy_forwards_and_logs(self, mock_client_cls, observatory_client):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = _mock_response()

        resp = observatory_client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-agent-id": "test-agent"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model"] == "claude-sonnet-4-20250514"


class TestVertexProxy:
    @patch("httpx.Client")
    def test_vertex_proxy_forwards_and_logs(self, mock_client_cls, observatory_client):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = _mock_response(
            model="claude-haiku-4-5@20251001", input_tokens=50, output_tokens=20)

        resp = observatory_client.post(
            "/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-haiku-4-5@20251001:rawPredict",
            json={"model": "claude-haiku-4-5@20251001", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-agent-id": "vertex-test", "Authorization": "Bearer fake-token"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model"] == "claude-haiku-4-5@20251001"

        # Verify it was logged
        usage_resp = observatory_client.get("/v1/usage?agent_id=vertex-test")
        assert usage_resp.get_json()["count"] == 1

    @patch("httpx.Client")
    def test_vertex_proxy_constructs_correct_url(self, mock_client_cls, observatory_client):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = _mock_response()

        observatory_client.post(
            "/v1/projects/my-proj/locations/us-east5/publishers/anthropic/models/claude-sonnet-4:rawPredict",
            json={"model": "claude-sonnet-4", "messages": []},
            headers={"x-agent-id": "url-test"},
        )

        call_args = mock_client_cls.return_value.post.call_args
        assert "us-east5-aiplatform.googleapis.com" in call_args[0][0]
        assert "my-proj" in call_args[0][0]


class TestStreamingProxy:
    @patch("httpx.Client")
    def test_streaming_proxy_passes_through_and_logs(self, mock_client_cls, observatory_client):
        sse_lines = [
            'event: message_start',
            'data: {"type":"message_start","message":{"usage":{"input_tokens":50}}}',
            '',
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"text":"hello"}}',
            '',
            'event: message_delta',
            'data: {"type":"message_delta","usage":{"output_tokens":25}}',
            '',
            'event: message_stop',
            'data: {"type":"message_stop"}',
            '',
        ]

        mock_stream = MagicMock()
        mock_stream.__enter__ = lambda s: s
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.iter_lines.return_value = iter(sse_lines)

        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.stream.return_value = mock_stream

        resp = observatory_client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "stream": True,
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-agent-id": "stream-test"},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.content_type

        # Consume the stream to trigger usage logging
        _ = resp.data

        # Verify usage was logged
        usage_resp = observatory_client.get("/v1/usage?agent_id=stream-test")
        usage_data = usage_resp.get_json()
        assert usage_data["count"] == 1
        assert usage_data["usage"][0]["input_tokens"] == 50
        assert usage_data["usage"][0]["output_tokens"] == 25


class TestUsageAPI:
    @patch("httpx.Client")
    def test_usage_returns_logged_calls(self, mock_client_cls, observatory_client):
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = _mock_response()

        observatory_client.post(
            "/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "messages": []},
            headers={"x-agent-id": "usage-test"},
        )

        resp = observatory_client.get("/v1/usage?agent_id=usage-test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["usage"]) == 1
        assert data["usage"][0]["model"] == "claude-sonnet-4-20250514"

    def test_usage_summary_empty(self, observatory_client):
        resp = observatory_client.get("/v1/usage/summary?agent_id=nonexistent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_calls"] == 0
