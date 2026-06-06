# Context Engine Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Context Engine graph non-empty — replace dead keyword extraction with async Haiku-powered lenses, wire /handoff to confirm CE state, and add a Hermes plugin so Iris's sessions are also extracted.

**Architecture:** `lenses.py` provides `extract_from_turn` (single pair, used by the async worker) and `extract_from_batch` (list of pairs, used by Hermes at session end). The Claude Code connector's ingest endpoint returns 202 and queues work for a background thread. The Hermes plugin uses `on_session_start`/`on_session_end` hooks. The handoff script calls `/context/session/end` and prints a confirmation.

**Tech Stack:** Python stdlib threading + queue, `anthropic` SDK (Vertex AI fallback), Flask (existing CE server), Hermes plugin hooks (`register(ctx)`).

---

## File Map

| File | Action |
|------|--------|
| `context_engine/lenses.py` | Create — Haiku lens prompt, `extract_from_turn`, `extract_from_batch`, `_entities_from_result` |
| `context_engine/worker.py` | Create — `start_worker(db_path)`, `enqueue(...)`, background thread |
| `context_engine/server.py` | Modify — ingest returns 202 + queues; `start_worker` called at app init |
| `context_engine/config.py` | Modify — add `extraction_model` field |
| `tests/test_lenses.py` | Create — mocked Haiku calls |
| `tests/test_worker.py` | Create — queue + thread behavior |
| `.claude/plugins/local/context-engine/handoff_ce.py` | Create — CE handoff step script |
| `/Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/__init__.py` | Create — Hermes plugin with session hooks |
| `/Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/plugin.yaml` | Create — plugin metadata |

---

## Task 1: Extraction Lenses

**Files:**
- Create: `context_engine/lenses.py`
- Create: `tests/test_lenses.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lenses.py
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_lenses.py -v
```
Expected: `ModuleNotFoundError: No module named 'context_engine.lenses'`

- [ ] **Step 3: Implement lenses.py**

```python
# context_engine/lenses.py
"""Haiku-powered extraction lenses for the Context Engine.

Two entry points:
  extract_from_turn(user_msg, assistant_msg) -> list[dict]
      Used by the async worker — one message pair at a time.

  extract_from_batch(pairs) -> list[dict]
      Used by the Hermes bridge — list of (user, assistant) tuples,
      chunked internally at 10 pairs per API call.
"""
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_TURN_PROMPT = """\
Analyze this conversation turn and extract knowledge graph entities.

User: {user_msg}
Assistant: {assistant_msg}

Extract:
1. Decision — did the user commit to a choice, direction, or approach?
   Look for "let's go with", "we'll use", "going with", "lock it in",
   "ship it", "I've decided", or clear commitment language.
2. Thread signal — does this message start a new topic or resume an old one?
   Look for "let's talk about X", "switching to X", "going back to X",
   "what about X" as a topic shift.

Return ONLY valid JSON, no markdown:
{{"decision": null, "thread": null}}
or with values:
{{"decision": {{"content": "short description", "confidence": 0.4-0.7}},
  "thread": {{"title": "short topic name", "signal": "new_topic or resume"}}}}"""

_BATCH_PROMPT = """\
Analyze these conversation turn pairs and extract decisions and thread signals.

{turns}

For each turn return one entry. Extract:
- decision: user commitment to a choice/direction (null if none)
- thread: topic change or resumption signal (null if none)

Return ONLY a JSON array, no markdown:
[{{"turn": 0, "decision": null, "thread": null}}, ...]"""

_CHUNK_SIZE = 10


def _make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()
    from anthropic import AnthropicVertex
    return AnthropicVertex(
        project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-gcp-pnd-pe-eng-claude"),
        region="us-east5",
    )


def _parse_json(text: str) -> Any:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _entities_from_result(data: dict, user_msg: str) -> list:
    entities = []
    decision = data.get("decision")
    if decision and isinstance(decision, dict):
        content = str(decision.get("content", "")).strip()[:500]
        if content:
            entities.append({
                "type": "decision",
                "content": content,
                "confidence": float(decision.get("confidence", 0.5)),
            })
    thread = data.get("thread")
    if thread and isinstance(thread, dict):
        title = str(thread.get("title", "")).strip()[:200]
        if title:
            entities.append({
                "type": "thread_signal",
                "content": title,
                "confidence": 0.5,
                "signal": thread.get("signal", "new_topic"),
            })
    return entities


def extract_from_turn(user_msg: str, assistant_msg: str) -> list:
    """Extract entities from one (user, assistant) pair."""
    client = _make_client()
    prompt = _TURN_PROMPT.format(
        user_msg=user_msg[:800],
        assistant_msg=assistant_msg[:800],
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5@20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        data = _parse_json(resp.content[0].text)
        return _entities_from_result(data, user_msg)
    except Exception as exc:
        logger.debug("extract_from_turn failed: %s", exc)
        return []


def extract_from_batch(pairs: list) -> list:
    """Extract entities from a list of (user_msg, assistant_msg) tuples."""
    if not pairs:
        return []
    client = _make_client()
    results = []
    for chunk_start in range(0, len(pairs), _CHUNK_SIZE):
        chunk = pairs[chunk_start:chunk_start + _CHUNK_SIZE]
        turns_text = "\n\n".join(
            f"Turn {i}:\nUser: {u[:400]}\nAssistant: {a[:400]}"
            for i, (u, a) in enumerate(chunk)
        )
        prompt = _BATCH_PROMPT.format(turns=turns_text)
        try:
            resp = client.messages.create(
                model="claude-haiku-4-5@20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            items = _parse_json(resp.content[0].text)
            for i, item in enumerate(items):
                user_msg = chunk[i][0] if i < len(chunk) else ""
                results.extend(_entities_from_result(item, user_msg))
        except Exception as exc:
            logger.debug("extract_from_batch chunk failed: %s", exc)
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_lenses.py -v
```
Expected: 8 passed

- [ ] **Step 5: Confirm existing suite still passes**

```bash
python3 -m pytest tests/ -v --ignore=tests/test_lenses.py -q
```
Expected: all green

- [ ] **Step 6: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/lenses.py tests/test_lenses.py
git commit -m "feat: Haiku extraction lenses (decision + thread signal)"
```

---

## Task 2: Background Worker

**Files:**
- Create: `context_engine/worker.py`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
import sys
import time
import queue as queue_module
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import context_engine.worker as worker_module


@pytest.fixture(autouse=True)
def reset_worker():
    """Reset module-level state between tests."""
    worker_module._queue = queue_module.Queue()
    worker_module._worker_thread = None
    yield
    worker_module._queue = queue_module.Queue()
    worker_module._worker_thread = None


def test_enqueue_adds_to_queue():
    worker_module.enqueue("hello", "world", "sess-1")
    assert not worker_module._queue.empty()
    item = worker_module._queue.get_nowait()
    assert item["user_msg"] == "hello"
    assert item["session_id"] == "sess-1"


def test_start_worker_creates_daemon_thread(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)
    worker_module.start_worker(db_path)
    assert worker_module._worker_thread is not None
    assert worker_module._worker_thread.is_alive()
    assert worker_module._worker_thread.daemon


def test_start_worker_does_not_duplicate(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)
    worker_module.start_worker(db_path)
    first_thread = worker_module._worker_thread
    worker_module.start_worker(db_path)
    assert worker_module._worker_thread is first_thread


def test_worker_processes_queue_item(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)

    extracted = [{"type": "decision", "content": "Use SQLite", "confidence": 0.5}]
    with patch("context_engine.lenses.extract_from_turn", return_value=extracted):
        worker_module.start_worker(db_path)
        worker_module.enqueue("Let's use SQLite.", "Good.", "sess-1")
        time.sleep(0.5)  # let the worker thread process

    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    assert count == 1


def test_worker_survives_extraction_error(tmp_path):
    db_path = str(tmp_path / "test.db")
    from context_engine.schema import init_db
    init_db(db_path)

    with patch("context_engine.lenses.extract_from_turn", side_effect=RuntimeError("boom")):
        worker_module.start_worker(db_path)
        worker_module.enqueue("crash this", "please", "sess-err")
        time.sleep(0.3)

    assert worker_module._worker_thread.is_alive()
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_worker.py -v
```
Expected: `ModuleNotFoundError: No module named 'context_engine.worker'`

- [ ] **Step 3: Implement worker.py**

```python
# context_engine/worker.py
"""Background extraction worker for the Context Engine.

start_worker(db_path) — call once at Flask app startup.
enqueue(user_msg, assistant_msg, session_id) — non-blocking, returns immediately.
The worker thread reads from the queue, calls Haiku lenses, writes entities to the graph.
"""
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None


def start_worker(db_path: str, source_agent: str = "claude-code") -> None:
    """Start the background extraction worker. Safe to call multiple times."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=_run,
        args=(db_path, source_agent),
        daemon=True,
        name="ce-extraction-worker",
    )
    _worker_thread.start()
    logger.info("Context Engine extraction worker started (db=%s)", db_path)


def enqueue(user_msg: str, assistant_msg: str, session_id: str) -> None:
    """Queue a message pair for async extraction. Non-blocking."""
    _queue.put({"user_msg": user_msg, "assistant_msg": assistant_msg, "session_id": session_id})


def _run(db_path: str, source_agent: str) -> None:
    from context_engine.graph import KnowledgeGraph
    from context_engine.lenses import extract_from_turn
    from context_engine.models import Entity

    graph = KnowledgeGraph(db_path)

    while True:
        try:
            item = _queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            entities = extract_from_turn(item["user_msg"], item["assistant_msg"])
            for e_data in entities:
                if e_data.get("type") == "thread_signal":
                    continue
                entity = Entity(
                    type=e_data["type"],
                    content=e_data["content"],
                    confidence=e_data["confidence"],
                    source_agent=source_agent,
                    source_session=item["session_id"],
                )
                graph.add_entity(entity)
                logger.debug("Extracted %s: %s", entity.type, entity.content[:60])
        except Exception as exc:
            logger.warning("Worker extraction error (item dropped): %s", exc)
        finally:
            _queue.task_done()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_worker.py -v
```
Expected: 5 passed

- [ ] **Step 5: Full suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all green

- [ ] **Step 6: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/worker.py tests/test_worker.py
git commit -m "feat: background extraction worker — async queue, Haiku lenses, entity persistence"
```

---

## Task 3: Async Ingest in Server

**Files:**
- Modify: `context_engine/server.py`
- Modify: `context_engine/config.py`
- Modify: `tests/test_server.py` (add 2 tests)

- [ ] **Step 1: Write the failing tests**

Add these two tests to `tests/test_server.py` inside the existing `TestIngestAPI` class:

```python
def test_ingest_returns_202(self, client):
    resp = client.post("/context/ingest", json={
        "user_message": "Let's go with Redis.",
        "assistant_message": "Good choice.",
        "session_id": "s1",
        "source_agent": "claude-code",
    })
    assert resp.status_code == 202
    assert resp.get_json()["queued"] is True

def test_ingest_empty_messages_still_returns_202(self, client):
    resp = client.post("/context/ingest", json={
        "user_message": "",
        "assistant_message": "",
        "session_id": "s1",
    })
    assert resp.status_code == 202
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_server.py::TestIngestAPI -v
```
Expected: the two new tests FAIL (existing ingest tests also fail since endpoint now returns 202)

- [ ] **Step 3: Update config.py — add extraction_model field**

In `context_engine/config.py`, add `extraction_model` to the dataclass and load it from config:

```python
# In the ContextEngineConfig dataclass, add after cache_default_timeout:
extraction_model: str = "claude-haiku-4-5@20251001"
```

And in `load_config`, add after the cache section:

```python
        extraction = data.get("extraction", {})
        cfg.extraction_model = extraction.get("model", "claude-haiku-4-5@20251001")
```

- [ ] **Step 4: Update server.py — async ingest + worker startup**

Replace the entire `ingest` endpoint and add worker startup in `create_app`. Find the ingest endpoint (currently around line 183) and replace it. Also add worker startup after the graph is created.

In `create_app`, after `graph = KnowledgeGraph(db)`, add:

```python
    from context_engine.worker import start_worker as _start_worker
    _start_worker(db)
```

Replace the entire `/context/ingest` endpoint with:

```python
    @app.post("/context/ingest")
    def ingest():
        from context_engine.worker import enqueue
        data = request.get_json()
        user_msg = data.get("user_message", "")
        assistant_msg = data.get("assistant_message", "")
        session_id = data.get("session_id", "")
        if user_msg and assistant_msg:
            enqueue(user_msg, assistant_msg, session_id)
        return jsonify({"queued": True}), 202
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_server.py -v
```

The existing `test_ingest_extracts_decision` and `test_ingest_boring_turn_extracts_nothing` tests will now fail because they tested the old synchronous behavior. Remove those two tests from `tests/test_server.py` (they tested the old `extract_entities` keyword extraction which is gone).

After removing those two obsolete tests, run again:

```bash
python3 -m pytest tests/test_server.py -v
```
Expected: all pass

- [ ] **Step 6: Full suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all green

- [ ] **Step 7: Smoke test — CE service restarts cleanly**

```bash
launchctl kickstart -k gui/$(id -u)/com.gaia.context-engine
sleep 3
curl -s http://localhost:8850/context/health | python3 -m json.tool
```
Expected: `{"status": "ok", ...}`

- [ ] **Step 8: Smoke test — ingest returns 202**

```bash
curl -s -X POST http://localhost:8850/context/ingest \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Let'\''s go with Redis.", "assistant_message": "Good choice.", "session_id": "smoke-test"}' \
  | python3 -m json.tool
```
Expected: `{"queued": true}` with HTTP 202

- [ ] **Step 9: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/server.py context_engine/config.py tests/test_server.py
git commit -m "feat: async ingest (202) + worker startup wired into CE server"
```

---

## Task 4: /handoff CE Integration

**Files:**
- Create: `.claude/plugins/local/context-engine/handoff_ce.py`

- [ ] **Step 1: Create handoff_ce.py**

```python
#!/usr/bin/env python3
"""CE step for /handoff. Calls Context Engine to end session and confirm briefing written.

Usage: python3 handoff_ce.py <session_id> [summary]
Exit 0 always — CE is optional infrastructure, handoff must not block.
"""
import json
import sys
import urllib.request
import urllib.error

API_BASE = "http://127.0.0.1:8850"


def _post(path: str, body: dict):
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else "unknown"
    summary = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    try:
        result = _post("/context/session/end", {"session_id": session_id, "summary": summary})
        entity_count = result.get("entities_captured", "?")
        print(f"[Context Engine] Briefing written. {entity_count} entities captured this session. Safe to clear.")
    except Exception as exc:
        print(f"[Context Engine] Offline ({exc}). Proceeding without CE confirmation.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /Users/jbecker/.claude/plugins/local/context-engine/handoff_ce.py
```

- [ ] **Step 3: Smoke test against live CE**

```bash
python3 /Users/jbecker/.claude/plugins/local/context-engine/handoff_ce.py unknown "Test handoff"
```
Expected: `[Context Engine] Briefing written. 0 entities captured this session. Safe to clear.`
(Or similar — entity count may vary)

- [ ] **Step 4: Smoke test when CE is offline**

```bash
# Stop CE temporarily
launchctl stop gui/$(id -u)/com.gaia.context-engine
python3 /Users/jbecker/.claude/plugins/local/context-engine/handoff_ce.py unknown
```
Expected: `[Context Engine] Offline (...). Proceeding without CE confirmation.`

```bash
# Restart CE
launchctl kickstart -k gui/$(id -u)/com.gaia.context-engine
```

- [ ] **Step 5: Commit**

```bash
cd /Users/jbecker/context-engine
git add .claude/plugins/local/context-engine/handoff_ce.py 2>/dev/null || \
git -C /Users/jbecker add .claude/plugins/local/context-engine/handoff_ce.py 2>/dev/null || \
echo "Note: handoff_ce.py is outside context-engine repo — commit from tricorder-repo if needed"
git commit -m "feat: handoff_ce.py — CE confirmation step for /handoff skill" 2>/dev/null || echo "committed elsewhere"
```

Note: if `.claude/plugins/local/` is in a different repo (tricorder-repo), `cd /Users/jbecker/tricorder-repo && git add .claude/plugins/local/context-engine/handoff_ce.py && git commit -m "feat: handoff CE confirmation step"`.

---

## Task 5: Hermes Bridge Plugin

**Files:**
- Create: `/Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/__init__.py`
- Create: `/Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/plugin.yaml`

- [ ] **Step 1: Create plugin.yaml**

```yaml
name: context-engine-bridge
version: 1.0.0
description: "Connects Hermes session lifecycle to the Context Engine REST API. Extracts decisions and thread signals at session end; injects briefing context at session start."
author: "Jerry Becker + Scotty"
hooks:
  - on_session_start
  - on_session_end
```

Save to: `/Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/plugin.yaml`

- [ ] **Step 2: Create __init__.py**

```python
# /Users/jbecker/.hermes/hermes-agent/plugins/context_engine_bridge/__init__.py
"""Hermes plugin: Context Engine Bridge.

Hooks:
  on_session_start — registers session with CE, writes briefing to memory file
  on_session_end   — batch-extracts entities from full message history, ends session

CE is optional infrastructure. All errors are caught and logged — Hermes
sessions must never crash because CE is unreachable.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:8850"
CE_REPO = "/Users/jbecker/context-engine"
BRIEFING_FILE = Path(os.path.expanduser("~/.hermes/memories/CE_BRIEFING.md"))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict) -> Optional[dict]:
    try:
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.debug("CE bridge POST %s failed: %s", path, exc)
        return None


def _get(path: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(f"{API_BASE}{path}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        logger.debug("CE bridge GET %s failed: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Message pair extraction
# ---------------------------------------------------------------------------

def _extract_pairs(messages: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Pull (user_text, assistant_text) pairs from a Hermes message list."""
    pairs = []
    i = 0
    while i < len(messages) - 1:
        cur = messages[i]
        nxt = messages[i + 1]
        if cur.get("role") == "user" and nxt.get("role") == "assistant":
            def _text(content) -> str:
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        b.get("text", "") for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                return ""
            user_text = _text(cur.get("content", ""))[:800]
            asst_text = _text(nxt.get("content", ""))[:800]
            if user_text and asst_text:
                pairs.append((user_text, asst_text))
            i += 2
        else:
            i += 1
    return pairs


# ---------------------------------------------------------------------------
# Hook handlers
# ---------------------------------------------------------------------------

def _on_session_start(session_id: str = None, **kwargs) -> None:
    if not session_id:
        return
    try:
        result = _post("/context/session/start", {"agent": "hermes", "session_id": session_id})
        if result and result.get("briefing"):
            briefing = result["briefing"]
            BRIEFING_FILE.parent.mkdir(parents=True, exist_ok=True)
            BRIEFING_FILE.write_text(f"# Context Engine Briefing\n\n{briefing}\n")
            logger.info("CE bridge: briefing written to %s", BRIEFING_FILE)
    except Exception as exc:
        logger.debug("CE bridge on_session_start error: %s", exc)


def _on_session_end(session_id: str = None, messages: List[Dict[str, Any]] = None, **kwargs) -> None:
    if not session_id:
        return
    try:
        if messages:
            pairs = _extract_pairs(messages)
            if pairs:
                # Add CE repo to path for lenses import
                if CE_REPO not in sys.path:
                    sys.path.insert(0, CE_REPO)
                from context_engine.lenses import extract_from_batch
                entities = extract_from_batch(pairs)
                for entity in entities:
                    if entity.get("type") == "thread_signal":
                        continue
                    _post("/context/entity", {
                        "type": entity["type"],
                        "content": entity["content"],
                        "confidence": entity.get("confidence", 0.5),
                        "source_agent": "hermes",
                        "source_session": session_id,
                    })
                logger.info("CE bridge: extracted %d entities from %d pairs", len(entities), len(pairs))
        _post("/context/session/end", {"session_id": session_id, "summary": None})
    except Exception as exc:
        logger.debug("CE bridge on_session_end error: %s", exc)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("Context Engine Bridge plugin registered")
```

- [ ] **Step 3: Verify plugin is discovered**

```bash
cd /Users/jbecker/.hermes/hermes-agent
python3 -c "
from hermes_cli.plugins import load_plugins, PluginManager
import logging
logging.basicConfig(level=logging.INFO)
# Just import to check no syntax errors
import plugins.context_engine_bridge as bridge
print('Plugin imports OK')
print('register function:', bridge.register)
"
```
Expected: `Plugin imports OK` and `register function: <function register ...>`

- [ ] **Step 4: Smoke test on_session_end directly**

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '/Users/jbecker/.hermes/hermes-agent')
sys.path.insert(0, '/Users/jbecker/context-engine')

from unittest.mock import patch, MagicMock

# Mock extract_from_batch so we don't make real API calls
mock_entities = [{"type": "decision", "content": "Use Redis for caching", "confidence": 0.6}]

with patch("context_engine.lenses.extract_from_batch", return_value=mock_entities):
    from plugins.context_engine_bridge import _on_session_end, _extract_pairs

    fake_messages = [
        {"role": "user", "content": "Let's go with Redis."},
        {"role": "assistant", "content": "Good choice."},
    ]
    _on_session_end(session_id="smoke-test-hermes", messages=fake_messages)
    print("on_session_end completed without error")

# Check if entity was written to CE
import urllib.request, json
req = urllib.request.Request("http://127.0.0.1:8850/context/search?q=Redis")
with urllib.request.urlopen(req, timeout=3) as resp:
    result = json.loads(resp.read())
    print(f"CE search for 'Redis': {result['count']} results")
EOF
```
Expected: `on_session_end completed without error` and `CE search for 'Redis': 1 results`

- [ ] **Step 5: Commit**

```bash
cd /Users/jbecker/.hermes/hermes-agent
git add plugins/context_engine_bridge/
git commit -m "feat: Context Engine Bridge plugin — session lifecycle hooks for Hermes/Iris"
```

---

## Self-Review

**Spec coverage:**
- ✅ Async extraction: lenses.py + worker.py + 202 ingest endpoint
- ✅ `extract_from_turn` for Claude Code (per-turn async worker)
- ✅ `extract_from_batch` for Hermes (end-of-session, full history)
- ✅ Decision + Thread Signal extraction — both lenses implemented
- ✅ 202 Accepted — hook returns instantly, no user-perceived latency
- ✅ Worker survives extraction errors — try/except in `_run`
- ✅ CE service-down is safe — all bridge/handoff calls wrapped in try/except
- ✅ /handoff CE step — handoff_ce.py prints entity count + safe-to-clear confirmation
- ✅ Hermes bridge — `register(ctx)` hooks on_session_start + on_session_end
- ✅ Briefing written to `~/.hermes/memories/CE_BRIEFING.md` for Hermes memory injection
- ✅ Error handling: CE down → handoff_ce prints warning, sessions continue

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:**
- `extract_from_turn(user_msg: str, assistant_msg: str) -> list` — matches worker call at Task 2 Step 3
- `extract_from_batch(pairs: list) -> list` — pairs are `List[Tuple[str, str]]` — matches Hermes bridge at Task 5 Step 2
- `_entities_from_result(data: dict, user_msg: str) -> list` — used internally in both — consistent
- `enqueue(user_msg, assistant_msg, session_id)` — matches worker.py and server.py ingest — consistent
- Entity dict shape: `{type, content, confidence}` — consistent across lenses → worker → graph insert → Hermes bridge POST
