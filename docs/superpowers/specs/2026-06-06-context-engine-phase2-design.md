# Context Engine Phase 2 — Design Spec

**Date:** 2026-06-06  
**Status:** Approved, ready for implementation planning  
**Goal:** Make the Context Engine actually useful — fix extraction, wire /handoff, bridge Hermes

---

## Problem Statement

Phase 1 shipped a complete knowledge graph + REST API. Phase 2 connectors are partially wired (Claude Code hooks exist, sessions are logged) but **the graph has 0 entities** because keyword extraction is too narrow to catch real conversation. This phase fixes that and completes the two remaining connectors.

---

## Three Components

### 1. Async LLM Extraction

**What:** Replace dead keyword extraction with Haiku-powered Decision + Thread lenses. Run async so there's no user-perceived latency.

**How it works:**

The CE service starts a background worker thread on startup. The existing `POST /context/ingest` endpoint now queues the message pair and returns `202 Accepted` immediately (~100ms). The worker pulls from the queue and calls Haiku.

Single Haiku call per turn with a combined lens prompt:

```
Given this user message and assistant response, extract:
1. Decision: did the user commit to a choice/direction? If yes: {content, confidence (0.4-0.7)}
2. Thread signal: topic change or resumption? If yes: {title, signal: new_topic|resume}

User: {user_msg[:800]}
Assistant: {assistant_msg[:800]}

Return JSON: {"decision": null|{content, confidence}, "thread": null|{title, signal}}
If neither: {"decision": null, "thread": null}
```

Extracted entities POST to `/context/entity`. Thread signals POST to `/context/thread` (new) or update thread status (resume).

**Why skip keyword extraction:** Proven ineffective — 0 entities extracted from 164 real sessions. LLM extraction with Haiku costs ~$0.00001/turn and is accurate enough for Phase 2.

**Queue behavior:** In-memory `queue.Queue`. If CE restarts mid-session, queued items are lost. Acceptable — each lost turn is a small gap, not a full session loss.

**`lenses.py` exposes two functions with the same entity output format:**
- `extract_from_turn(user_msg, assistant_msg) -> list[dict]` — single pair, used by the async worker
- `extract_from_batch(pairs: list[tuple]) -> list[dict]` — list of (user, assistant) tuples, used by the Hermes bridge for end-of-session extraction; chunked internally at 10 pairs per Haiku call

**New files:**
- `context_engine/lenses.py` — lens prompt + Haiku call, both single and batch extraction
- `context_engine/worker.py` — background worker thread, queue management

**Modified files:**
- `context_engine/server.py` — ingest endpoint returns 202, pushes to queue; worker starts with app
- `context_engine/config.py` — add `extraction_model` field (default: `claude-haiku-4-5@20251001`)

---

### 2. /handoff Skill

**What:** Extend the existing `/handoff` skill to call CE before confirming it's safe to clear context.

**How it works:**

When `/handoff` is invoked, after existing memory file updates, a CE step runs:

```python
# handoff_ce.py
POST /context/session/end {"session_id": ..., "summary": <one-line from Claude>}
→ Returns: {briefing_written: true, entities_captured: N}
→ Prints: "Context Engine: briefing written. N entities captured this session. Safe to clear."
```

If CE is unreachable (service down): prints a warning but doesn't block the handoff. The existing skill still works.

**Phase 2 scope:** Simple mode only. No interactive handoff (Navy watch model is Phase 5).

**New files:**
- `.claude/plugins/local/context-engine/handoff_ce.py` — CE handoff script

**Modified files:**
- Existing `/handoff` skill — add a call to `handoff_ce.py` as a step

---

### 3. Hermes Bridge

**What:** A Python class implementing Hermes's context engine interface that proxies to our REST API at port 8850.

**Interface Hermes expects** (from `run_agent.py` `_transition_context_engine_session`):

```python
class ContextEngineBridge:
    def on_session_start(self, session_id, *, old_session_id=None,
                         carry_over_context=False, platform=None, **kwargs) -> None:
        # Called at session start. Fetch briefing, optionally inject into system prompt.

    def on_session_end(self, session_id, messages: list) -> None:
        # Called at session end with full message history.
        # Run batch extraction, then POST /context/session/end.

    def on_session_reset(self) -> None:
        # No-op for CE.

    def carry_over_new_session_context(self, old_session_id, new_session_id) -> None:
        # Session continues under new ID. Start new session with carry-over.
```

**Batch extraction at session end:**

`on_session_end` receives the full `messages` list. We extract message pairs (user + next assistant), chunk into groups of 10, and call Haiku once per chunk with the same lens prompt as the Claude Code connector. This gives Iris higher-quality extraction than per-turn (full conversation context available).

```python
def on_session_end(self, session_id, messages):
    pairs = _extract_pairs(messages)       # [(user_text, assistant_text), ...]
    for chunk in _chunks(pairs, size=10):
        entities = _batch_extract(chunk)   # one Haiku call
        for entity in entities:
            _post("/context/entity", entity)
    _post("/context/session/end", {"session_id": session_id})
```

**Briefing injection:**

`on_session_start` fetches `/context/briefing`. Hermes injects context engine output into the system prompt via a return value or by writing to a shared context object — exact mechanism confirmed during implementation by reading `hermes_cli/plugins.py`.

**Error handling:**

All HTTP calls in the bridge are wrapped in try/except. If CE is unreachable, the bridge logs a warning and returns cleanly — Hermes sessions must never crash because CE is down. CE is optional infrastructure, not a hard dependency.

**Registration:**

Hermes sets `context_compressor` as an attribute on the agent object. Registration mechanism confirmed during implementation — likely via `plugins.py` or Hermes config. The bridge is instantiated once and assigned.

**New files:**
- `~/.hermes/hermes-agent/plugins/context_engine_bridge.py` — ~150 lines

---

## Data Flow Summary

```
Claude Code session:
  UserPromptSubmit → connector.py → POST /context/ingest (202) → queue
                                                                      ↓
                                                              worker thread
                                                                      ↓
                                                             Haiku lens call
                                                                      ↓
                                                          POST /context/entity
  /handoff → handoff_ce.py → POST /context/session/end → "N entities. Safe to clear."

Hermes/Iris session:
  session start → on_session_start → GET /context/briefing → inject into system prompt
  session end   → on_session_end(messages) → batch Haiku extraction → POST entities
                                           → POST /context/session/end
```

---

## What Phase 2 Does NOT Include

- Insight, State, Artifact, Agent, Trigger lenses (Phase 3)
- Emergent Decision Lens sliding window (Phase 3)
- Interactive handoff / Navy watch model (Phase 5)
- Graph traversal / on-demand context loading (Phase 4)
- Clarification queue (Phase 6)

---

## Success Criteria

1. Start a Claude Code session → briefing loads (may be sparse first session, richer after a few)
2. Have a conversation with at least one explicit decision ("let's go with X") → entity appears in graph
3. `/handoff` → confirms entities captured and briefing written
4. Start Iris in Hermes → briefing from CE loads at session start
5. End Iris session → batch extraction runs, entities appear in graph

---

## File Map

| File | Action | Notes |
|------|--------|-------|
| `context_engine/lenses.py` | Create | Haiku lens prompt + extraction logic |
| `context_engine/worker.py` | Create | Background queue worker thread |
| `context_engine/server.py` | Modify | Ingest returns 202, starts worker |
| `context_engine/config.py` | Modify | Add `extraction_model` field |
| `.claude/plugins/local/context-engine/handoff_ce.py` | Create | CE handoff step |
| Existing `/handoff` skill | Modify | Add CE step |
| `~/.hermes/hermes-agent/plugins/context_engine_bridge.py` | Create | Hermes bridge class |
