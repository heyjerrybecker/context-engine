# Context Engine

**Persistent memory for AI coding sessions.**

Context Engine gives your AI assistant memory that spans across sessions. It watches your conversations, extracts decisions and context, and loads relevant history into every new session automatically. No manual bookkeeping — your AI just remembers.

## Quick Start

```bash
git clone https://github.com/heyjerrybecker/context-engine.git
cd context-engine
python3 setup.py
```

The setup script asks three questions (name, role, communication style), installs dependencies, hooks into Claude Code, and starts the server. Takes about 30 seconds.

**Prerequisites:** Python 3.9+, Claude Code installed.

## What It Does

**Session briefings.** When you start a new Claude Code session, Context Engine injects a briefing with what you were working on, decisions you made, and open threads — so the AI picks up where you left off.

**Automatic extraction.** A background worker analyzes each conversation turn using Claude Haiku and extracts decisions ("let's go with X") and thread signals ("switching to topic Y"). These accumulate in a local knowledge graph.

**Confidence decay.** Entities that aren't referenced lose confidence over time (14-day half-life, 7-day grace period). Old context fades naturally so briefings stay relevant.

**Memory lifecycle.** Monitors working memory capacity for AI agents, flags when memory files are getting too large, and provides an archive/recall API for deep storage.

**Token Observatory.** A reverse proxy that intercepts Claude API calls and logs usage metadata — model, tokens, latency, cost — without storing message content. Gives you visibility into what all your AI tools are costing.

## How It Works

```
Claude Code session starts
  → SessionStart hook fires
  → CE generates briefing from knowledge graph
  → Briefing injected into session context

You have a conversation
  → Each turn sent to CE via hooks
  → Background worker extracts entities (Haiku)
  → Entities stored in knowledge graph

Next session starts
  → Briefing includes yesterday's decisions and open threads
  → AI naturally continues where you left off
```

All data stays local in `~/.context-engine/graph.db`. Nothing leaves your machine.

## Architecture

```
~/.context-engine/
├── config.yaml         # Server + identity + agent config
├── graph.db            # Knowledge graph (entities, threads, relationships)
├── usage.db            # Token Observatory usage logs
├── SOUL.md             # AI assistant identity (generated during setup)
├── USER.md             # User profile (generated during setup)
└── cursors/            # Transcript position tracking per session
```

**Server:** Flask on `127.0.0.1:8850`
**Extraction:** Claude Haiku 4.5 via background worker thread
**Hooks:** Claude Code SessionStart, UserPromptSubmit, Stop

## API

### Knowledge Graph

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/context/session/start` | Start session, get briefing |
| POST | `/context/session/end` | End session, trigger classification |
| POST | `/context/entity` | Create entity |
| GET | `/context/search?q=` | Full-text search (FTS5) |
| POST | `/context/ingest` | Queue message pair for extraction |
| GET | `/context/entities/recent` | Recent entities (for federation) |

### Memory Lifecycle

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/context/memory/sync` | Sync agent working memory state |
| POST | `/context/memory/archive` | Archive entries to deep storage |
| GET | `/context/memory/recall?agent_id=&q=` | Search archived entries |
| GET | `/context/memory/health?agent_id=` | Check memory utilization |

### Token Observatory

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/v1/messages` | Proxy for direct Anthropic API |
| POST | `/v1/projects/.../models/...:rawPredict` | Proxy for Vertex AI |
| GET | `/v1/usage?agent_id=` | Usage history |
| GET | `/v1/usage/summary?agent_id=` | Aggregated cost + token summary |

## Configuration

`~/.context-engine/config.yaml`:

```yaml
identity:
  soul: ~/.context-engine/SOUL.md
  user_model: ~/.context-engine/USER.md

server:
  port: 8850
  host: 127.0.0.1

confidence:
  decay_half_life_days: 14
  archive_threshold: 0.1

# Optional: agent memory monitoring
agents:
  claude-code:
    memory_path: ~/.claude/projects/.../memory/MEMORY.md
    memory_dir: ~/.claude/projects/.../memory
    max_chars: 12000

memory:
  capacity_threshold: 0.8
  aging_days: 30

# Optional: Token Observatory backend
observatory:
  backend_url: https://us-east5-aiplatform.googleapis.com
```

## Running

The setup script installs a macOS LaunchAgent that starts CE automatically on login. To manage manually:

```bash
# Start
launchctl load ~/Library/LaunchAgents/com.context-engine.server.plist

# Stop
launchctl unload ~/Library/LaunchAgents/com.context-engine.server.plist

# Check health
curl http://localhost:8850/context/health
```

## Tests

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

150 tests covering knowledge graph, extraction, confidence decay, memory lifecycle, observatory, and API endpoints.

## Privacy

- All data stored locally (`~/.context-engine/`)
- Message content extracted into short entity summaries only — full transcripts are not stored
- Token Observatory logs usage metadata (model, tokens, cost) — never message content
- No external services, no cloud sync, no telemetry

## License

Internal project — not yet licensed for external use.
