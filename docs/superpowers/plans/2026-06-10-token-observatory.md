# Token Observatory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reverse proxy route to the CE Flask server that intercepts all Claude API calls, logs usage metadata (model, tokens, latency, cost) to MLflow, and provides a usage query API for agent self-optimization.

**Architecture:** New `observatory.py` module handles proxy forwarding (via httpx) and usage logging. Routes registered on the existing Flask server at `/v1/messages` (proxy) and `/v1/usage` (query). Cost calculation uses a configurable pricing table. No message content is stored.

**Tech Stack:** Flask (existing), httpx (available), MLflow (existing), SQLite (existing)

**Spec:** `docs/superpowers/specs/2026-06-10-token-observatory-design.md`
**Issue:** TBD on context-engine repo

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `context_engine/observatory.py` | Create | Proxy forwarding, metadata extraction, cost calculation, MLflow logging, usage queries |
| `context_engine/server.py` | Modify | Register observatory blueprint |
| `context_engine/config.py` | Modify | Add `observatory_backend_url` config field |
| `tests/test_observatory.py` | Create | Proxy, logging, and usage API tests |

---

### Task 1: Create observatory module with cost calculation and usage storage

**Files:**
- Create: `context_engine/observatory.py`
- Create: `tests/test_observatory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observatory.py
import pytest
import sqlite3
from context_engine.observatory import estimate_cost, UsageLog


class TestCostEstimation:
    def test_sonnet_cost(self):
        cost = estimate_cost("claude-sonnet-4-20250514", input_tokens=1000, output_tokens=500)
        # Sonnet: $3/M input, $15/M output
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
            cache_read_tokens=500, cache_creation_tokens=200
        )
        # cache_read at 10%, cache_creation at 25% of input rate
        assert cost > 0

    def test_unknown_model_returns_zero(self):
        cost = estimate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost == 0.0


class TestUsageLog:
    def test_log_and_query(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        log = UsageLog(db_path)
        log.record(
            agent_id="claude-code-jerry",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            latency_ms=450,
            estimated_cost=0.0105,
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
        for i in range(5):
            log.record(agent_id="test", model="claude-sonnet-4-20250514",
                       input_tokens=1000, output_tokens=500,
                       cache_read_tokens=0, cache_creation_tokens=0,
                       latency_ms=400, estimated_cost=0.0105)
        summary = log.summary(agent_id="test")
        assert summary["total_calls"] == 5
        assert summary["total_input_tokens"] == 5000
        assert summary["total_output_tokens"] == 2500
        assert abs(summary["total_cost"] - 0.0525) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jbecker/context-engine && python3 -m pytest tests/test_observatory.py -v`

- [ ] **Step 3: Implement observatory.py**

```python
# context_engine/observatory.py
"""
Token Observatory — tracks all Claude API usage through CE.

Provides cost estimation, usage logging to SQLite, and query APIs.
No message content is stored — only operational metadata.
"""

import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "claude-opus-4":      {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4":    {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5":   {"input": 0.80,  "output": 4.0},
}

CACHE_READ_DISCOUNT = 0.10
CACHE_CREATION_DISCOUNT = 0.25


def _match_model_pricing(model: str) -> Optional[Dict[str, float]]:
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    return None


def estimate_cost(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    pricing = _match_model_pricing(model)
    if not pricing:
        return 0.0
    input_rate = pricing["input"] / 1_000_000
    output_rate = pricing["output"] / 1_000_000
    cost = (input_tokens * input_rate
            + output_tokens * output_rate
            + cache_read_tokens * input_rate * CACHE_READ_DISCOUNT
            + cache_creation_tokens * input_rate * CACHE_CREATION_DISCOUNT)
    return cost


class UsageLog:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cache_read_tokens INTEGER DEFAULT 0,
                    cache_creation_tokens INTEGER DEFAULT 0,
                    latency_ms INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_agent ON api_usage(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_model ON api_usage(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_time ON api_usage(timestamp)")

    def record(self, agent_id: str, model: str, input_tokens: int,
               output_tokens: int, cache_read_tokens: int,
               cache_creation_tokens: int, latency_ms: int,
               estimated_cost: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO api_usage
                   (agent_id, model, input_tokens, output_tokens,
                    cache_read_tokens, cache_creation_tokens,
                    latency_ms, estimated_cost, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (agent_id, model, input_tokens, output_tokens,
                 cache_read_tokens, cache_creation_tokens,
                 latency_ms, estimated_cost, now)
            )

    def query(self, agent_id: Optional[str] = None,
              since: Optional[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
        conditions, params = [], []
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM api_usage{where} ORDER BY timestamp DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [dict(r) for r in rows]

    def summary(self, agent_id: Optional[str] = None,
                since: Optional[str] = None) -> Dict[str, Any]:
        conditions, params = [], []
        if agent_id:
            conditions.append("agent_id = ?")
            params.append(agent_id)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                f"""SELECT COUNT(*) as total_calls,
                    COALESCE(SUM(input_tokens), 0) as total_input_tokens,
                    COALESCE(SUM(output_tokens), 0) as total_output_tokens,
                    COALESCE(SUM(estimated_cost), 0) as total_cost
                    FROM api_usage{where}""",
                params
            ).fetchone()
        return {
            "total_calls": row[0],
            "total_input_tokens": row[1],
            "total_output_tokens": row[2],
            "total_cost": row[3],
        }
```

- [ ] **Step 4: Run tests to verify they pass**
- [ ] **Step 5: Commit**

```bash
git add context_engine/observatory.py tests/test_observatory.py
git commit -m "feat: add Token Observatory — cost estimation and usage logging"
```

---

### Task 2: Add proxy route and usage API to server

**Files:**
- Modify: `context_engine/observatory.py` (add proxy function)
- Modify: `context_engine/server.py` (register routes)
- Modify: `context_engine/config.py` (add backend URL config)
- Test: `tests/test_observatory.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_observatory.py`:

```python
from unittest.mock import patch, MagicMock
from context_engine.server import create_app
from context_engine.schema import init_db


@pytest.fixture
def app_client(tmp_db, tmp_config_dir):
    init_db(tmp_db)
    app = create_app(db_path=tmp_db, config_dir=tmp_config_dir)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestProxyRoute:
    @patch("context_engine.observatory.httpx.Client")
    def test_proxy_forwards_and_logs(self, mock_client_cls, app_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = __import__("json").dumps({
            "id": "msg_test",
            "type": "message",
            "model": "claude-sonnet-4-20250514",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
            "content": [{"type": "text", "text": "hello"}],
        }).encode()
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = mock_response

        resp = app_client.post("/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-agent-id": "test-agent"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model"] == "claude-sonnet-4-20250514"


class TestUsageAPI:
    @patch("context_engine.observatory.httpx.Client")
    def test_usage_returns_logged_calls(self, mock_client_cls, app_client):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content = __import__("json").dumps({
            "id": "msg_test", "type": "message",
            "model": "claude-sonnet-4-20250514",
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "content": [{"type": "text", "text": "test"}],
        }).encode()
        mock_client_cls.return_value.__enter__ = lambda s: s
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value.post.return_value = mock_response

        app_client.post("/v1/messages",
            json={"model": "claude-sonnet-4-20250514", "messages": []},
            headers={"x-agent-id": "usage-test"})

        resp = app_client.get("/v1/usage?agent_id=usage-test")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["usage"]) == 1
        assert data["usage"][0]["model"] == "claude-sonnet-4-20250514"

    def test_usage_summary(self, app_client):
        resp = app_client.get("/v1/usage/summary?agent_id=nonexistent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_calls"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Add config field**

In `context_engine/config.py`, add to `ContextEngineConfig`:

```python
    observatory_backend_url: str = "https://us-east5-aiplatform.googleapis.com"
```

And in `load_config()`, after the extraction section:

```python
        observatory = data.get("observatory", {})
        cfg.observatory_backend_url = observatory.get(
            "backend_url", "https://us-east5-aiplatform.googleapis.com")
```

- [ ] **Step 4: Add proxy and usage functions to observatory.py**

Add `import httpx` and `import json` at the top of `observatory.py`. Add these functions:

```python
def proxy_request(request_data: bytes, headers: dict,
                  backend_url: str, usage_log: UsageLog,
                  agent_id: str = "unknown") -> tuple:
    """Forward request to Claude API backend, log usage, return response."""
    body = json.loads(request_data)
    model = body.get("model", "unknown")
    is_streaming = body.get("stream", False)

    forward_headers = {}
    for key in ("authorization", "x-api-key", "content-type",
                "anthropic-version", "anthropic-beta"):
        val = headers.get(key)
        if val:
            forward_headers[key] = val

    start = time.time()
    with httpx.Client(timeout=600) as client:
        resp = client.post(
            f"{backend_url}/v1/messages",
            content=request_data,
            headers=forward_headers,
        )
    latency_ms = int((time.time() - start) * 1000)

    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_creation = 0

    if resp.status_code == 200:
        try:
            resp_body = json.loads(resp.content)
            usage = resp_body.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_creation = usage.get("cache_creation_input_tokens", 0)
        except (json.JSONDecodeError, KeyError):
            pass

    cost = estimate_cost(model, input_tokens, output_tokens,
                         cache_read, cache_creation)

    usage_log.record(
        agent_id=agent_id, model=model,
        input_tokens=input_tokens, output_tokens=output_tokens,
        cache_read_tokens=cache_read, cache_creation_tokens=cache_creation,
        latency_ms=latency_ms, estimated_cost=cost,
    )

    logger.info("Observatory: %s %s → %d in/out, $%.6f, %dms",
                agent_id, model, input_tokens + output_tokens,
                cost, latency_ms)

    return resp.content, resp.status_code, dict(resp.headers)
```

- [ ] **Step 5: Register routes in server.py**

In `context_engine/server.py`, inside `create_app()`, add after the existing route sections (before `return app`):

```python
    # --- Token Observatory ---

    from context_engine.observatory import proxy_request, UsageLog
    import os
    usage_db = os.path.join(cfg.config_dir, "usage.db")
    usage_log = UsageLog(usage_db)

    @app.post("/v1/messages")
    def observatory_proxy():
        agent_id = request.headers.get("x-agent-id", "unknown")
        body, status, headers = proxy_request(
            request_data=request.get_data(),
            headers=dict(request.headers),
            backend_url=cfg.observatory_backend_url,
            usage_log=usage_log,
            agent_id=agent_id,
        )
        return app.response_class(body, status=status,
                                  content_type=headers.get("content-type", "application/json"))

    @app.get("/v1/usage")
    def observatory_usage():
        agent_id = request.args.get("agent_id")
        since = request.args.get("since")
        limit = int(request.args.get("limit", 100))
        rows = usage_log.query(agent_id=agent_id, since=since, limit=limit)
        return jsonify({"usage": rows, "count": len(rows)})

    @app.get("/v1/usage/summary")
    def observatory_summary():
        agent_id = request.args.get("agent_id")
        since = request.args.get("since")
        summary = usage_log.summary(agent_id=agent_id, since=since)
        return jsonify(summary)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_observatory.py -v
```

- [ ] **Step 7: Run full suite for regressions**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/ -x -q
```

- [ ] **Step 8: Commit**

```bash
git add context_engine/observatory.py context_engine/server.py context_engine/config.py tests/test_observatory.py
git commit -m "feat: add /v1/messages proxy and /v1/usage API to Token Observatory"
```

---

### Task 3: Final verification and push

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/ -v
```

- [ ] **Step 2: Quick manual verification**

```bash
# Start CE server
cd /Users/jbecker/context-engine && python3 -m context_engine.server &

# Check health
curl http://localhost:8850/context/health

# Check usage endpoint exists
curl http://localhost:8850/v1/usage/summary

# Stop server
kill %1
```

- [ ] **Step 3: Commit plan and push**

```bash
cd /Users/jbecker/context-engine
git add docs/superpowers/plans/2026-06-10-token-observatory.md
git commit -m "docs: add Token Observatory implementation plan"
git push origin main
```

---

## Verification

1. **Unit tests:** `pytest tests/test_observatory.py -v` — cost estimation, usage logging, proxy route, usage API
2. **Regression:** `pytest tests/ -x -q` — no existing CE tests break
3. **Behavioral:** Proxy route forwards requests unchanged, logs metadata, returns response byte-identical
4. **Privacy:** No message content in usage.db — only model, tokens, latency, cost, agent_id
5. **API check:** `/v1/usage` and `/v1/usage/summary` return correct data

## Not in this plan (future tasks)

- Streaming response support (`stream: true`) — needs SSE pass-through, separate task
- `context-engine setup` onboarding CLI — separate task after proxy is validated
- MLflow integration (logging to shared MLflow DB alongside SQLite) — separate task
- Vertex AI auth forwarding (different endpoint format) — separate task once direct API proxy works
