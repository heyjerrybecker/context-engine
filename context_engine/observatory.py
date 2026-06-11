"""Token Observatory — tracks all Claude API usage through CE.

No message content is stored — only operational metadata.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MODEL_PRICING = {
    "claude-opus-4":    {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4":  {"input": 3.0,   "output": 15.0},
    "claude-haiku-4-5": {"input": 0.80,  "output": 4.0},
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
    return (input_tokens * input_rate
            + output_tokens * output_rate
            + cache_read_tokens * input_rate * CACHE_READ_DISCOUNT
            + cache_creation_tokens * input_rate * CACHE_CREATION_DISCOUNT)


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
                 latency_ms, estimated_cost, now),
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
                params + [limit],
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
                params,
            ).fetchone()
        return {
            "total_calls": row[0],
            "total_input_tokens": row[1],
            "total_output_tokens": row[2],
            "total_cost": round(row[3], 6),
        }


def proxy_request(request_data: bytes, headers: dict,
                  backend_url: str, usage_log: UsageLog,
                  agent_id: str = "unknown",
                  target_url: str = None) -> tuple:
    import httpx

    body = json.loads(request_data)
    model = body.get("model", "unknown")

    forward_headers = {}
    for key in ("authorization", "x-api-key", "content-type",
                "anthropic-version", "anthropic-beta"):
        val = headers.get(key)
        if val:
            forward_headers[key] = val

    url = target_url or f"{backend_url}/v1/messages"

    start = time.time()
    with httpx.Client(timeout=600) as client:
        resp = client.post(url, content=request_data, headers=forward_headers)
    latency_ms = int((time.time() - start) * 1000)

    input_tokens = output_tokens = cache_read = cache_creation = 0

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

    logger.info("Observatory: %s %s in=%d out=%d $%.6f %dms",
                agent_id, model, input_tokens, output_tokens,
                cost, latency_ms)

    return resp.content, resp.status_code, dict(resp.headers)
