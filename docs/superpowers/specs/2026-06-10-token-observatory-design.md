# Token Observatory — Universal Claude Usage Tracking via CE Proxy

**Date:** 2026-06-10
**Issue:** TBD (create on context-engine repo)

## Problem

Context Engine's MLflow dashboard only shows Haiku because `mlflow.anthropic.autolog()` is per-process — it only captures API calls made by the CE extraction worker. Claude Code sessions (Sonnet, Opus), Iris, and multi-agent builds are all invisible. Associates onboarding to CE with their own agents will have the same blind spot.

## Solution

Add a reverse proxy route to the CE Flask server that intercepts all Claude API calls, logs usage metadata to the shared MLflow DB, and forwards requests to the real backend unchanged. No message content is stored — only operational metadata.

## Architecture

### Three-layer separation of concerns

1. **CE (infrastructure)** — collects and stores usage data. The proxy is a route on the existing CE server. Associates point their agents at CE's URL during onboarding.
2. **Each agent (self-awareness)** — queries CE's usage API to read its own history. Uses this to tailor its relationship with the associate around how they think and work.
3. **Gaia (meta-intelligence)** — reads the MLflow DB directly to discover cross-agent behavioral patterns, team-level usage anomalies, and model choice correlations.

### Data flow

```
Agent (Claude Code, Cursor, custom)
  → CE server localhost:8850/v1/messages
    → Extract metadata from request (model, agent_id)
    → Forward request unchanged (with agent's own auth) → Vertex AI / Anthropic
    → Extract metadata from response (tokens, latency)
    → Log to MLflow DB
  ← Return response unchanged to agent
```

### What gets logged per call

| Field | Source | Purpose |
|-------|--------|---------|
| `agent_id` | Request header or setup config | Identify which agent/associate |
| `model` | Request body `model` field | Model breakdown in dashboard |
| `input_tokens` | Response `usage.input_tokens` | Token tracking |
| `output_tokens` | Response `usage.output_tokens` | Token tracking |
| `cache_read_tokens` | Response `usage.cache_read_input_tokens` | Cache efficiency |
| `cache_creation_tokens` | Response `usage.cache_creation_input_tokens` | Cache efficiency |
| `latency_ms` | Measured at proxy | Performance tracking |
| `timestamp` | Proxy clock | Time series |
| `estimated_cost` | Computed from model + tokens | Cost visibility |

### What does NOT get logged

- **Message content** — the proxy never reads or stores prompt/response text. Privacy by design.
- **Auth tokens** — passed through to the backend, never stored.

## Components

### 1. Proxy route (`/v1/messages`)

Added to the existing CE Flask server in `server.py`. Accepts POST requests in Anthropic Messages API format, forwards to the configured backend (Vertex AI or direct Anthropic API), logs metadata, returns response unchanged.

Must handle:
- Streaming responses (`stream: true`) — log tokens from the final `message_delta` event
- Non-streaming responses — log tokens from the response body `usage` field
- Auth pass-through — forward `Authorization`, `x-api-key`, and Vertex AI headers unchanged
- Backend detection — route to Vertex AI or Anthropic API based on request headers or agent config

### 2. Usage API (`/v1/usage`)

REST endpoint that agents query to read their own usage data:
- `GET /v1/usage?agent_id=X` — returns usage history for agent X
- `GET /v1/usage?agent_id=X&since=2026-06-01` — time-filtered
- `GET /v1/usage/summary?agent_id=X` — aggregated: total tokens, cost, model breakdown

This is how agents self-optimize — they read their own patterns.

### 3. Onboarding integration

`context-engine setup` detects the agent type and writes the appropriate config:
- **Claude Code:** Sets `ANTHROPIC_BASE_URL` in shell profile or Claude Code settings
- **Cursor:** Updates Cursor's API settings
- **Custom Python:** Prints the one-liner to configure the Anthropic SDK

### 4. Cost calculation

Model pricing table (updatable via config):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|----------------------|
| claude-opus-4 | $15.00 | $75.00 |
| claude-sonnet-4 | $3.00 | $15.00 |
| claude-haiku-4-5 | $0.80 | $4.00 |

Cached input tokens are priced at 10% of the input rate.

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `context_engine/observatory.py` | Create | Proxy logic: forward, log, cost calculation |
| `context_engine/server.py` | Modify | Register `/v1/messages` and `/v1/usage` routes |
| `context_engine/config.py` | Modify | Add observatory config (backend URL, pricing) |
| `context_engine/tracing.py` | Modify | Add observatory-specific MLflow logging (not autolog) |
| `tests/test_observatory.py` | Create | Proxy and usage API tests |
| `scripts/setup_agent.py` | Create or modify | Auto-configure agent to use CE proxy |

## Onboarding UX

```bash
$ context-engine setup
Detected: Claude Code (Vertex AI)
Configuring proxy at http://localhost:8850/v1/messages...

Add this to your shell profile:
  export ANTHROPIC_BASE_URL=http://localhost:8850

Or run: context-engine setup --apply (writes it for you)

Done. All Claude API calls now flow through Context Engine.
Your MLflow dashboard will show all models within minutes.
```

## Success criteria

1. MLflow dashboard shows all models (Sonnet, Opus, Haiku) across all tools
2. Cost breakdown is accurate per model, per agent, per time period
3. Agents can query their own usage via `/v1/usage`
4. Zero impact on API call behavior — responses are byte-identical to direct calls
5. Streaming responses work without added latency (proxy streams through, logs after)
6. `context-engine setup` configures an agent in one command
7. Message content is never stored

## What this enables (future)

- **Agent self-optimization:** Each agent reads its own usage patterns to adapt to the associate's work style
- **Gaia behavioral patterns:** Cross-agent analysis of model choice, token efficiency, session patterns
- **Token intelligence dashboard:** The executive visibility layer from the token intelligence proposal
- **EU AI Act compliance:** Per-interaction audit trail without content storage
