# Native Metrics Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `context_engine/metrics/` — native instrumentation that tags every CE session as cold/warm/partial, classifies messages post-session, and generates the before/after comparison report that proves Context Engine's value.

**Architecture:** Five modules under `context_engine/metrics/`: tagger (determines session type from pre-existing entity count), classifier (finds transcript, runs Haiku classification async, writes COR/WMC/CR), report (queries metrics DB, formats two-tier output), `__main__` (CLI entry). Server wires session_end to trigger async classification. Reuses the existing `~/.context-engine/session_metrics.db` and `scripts/lib/` session loader/classifier from Target A.

**Tech Stack:** Python stdlib (sqlite3, threading, argparse, pathlib), `anthropic` SDK (Vertex AI fallback), existing `scripts/lib/session_loader.py` + `scripts/lib/session_classifier.py` (imported via sys.path).

---

## Context

- **Metrics DB:** `~/.context-engine/session_metrics.db` — already has 164 cold rows from Target A baseline run. Schema: `session_metrics` (session_id, date, session_type, agent, total_messages, total_tokens, is_marathon, length_bucket, context_messages, context_tokens, warmup_message_count, correction_count, first_new_work_message_index, cor, cr, created_at) + `message_labels` (id, session_id, message_index, role, label, token_count, is_correction).
- **Graph DB:** `~/.context-engine/graph.db` — `sessions` table has (session_id, agent, started_at, ended_at, summary, complexity_score). `entities` table has `created_at`.
- **Session type logic:** count entities with `created_at < session.started_at`. ≥5 = warm, 1-4 = partial, 0 = cold.
- **Target A scripts (reused):** `scripts/lib/session_loader.py` (load_session, find_session_files), `scripts/lib/session_classifier.py` (classify_session — calls Haiku), `scripts/lib/metrics_db.py` (init_db, insert_session, insert_labels).
- **compute_metrics** — defined inline in classifier.py (30 lines, same logic as baseline analyzer).

---

## File Map

| File | Action |
|------|--------|
| `context_engine/metrics/__init__.py` | Create — empty package marker |
| `context_engine/metrics/tagger.py` | Create — `determine_session_type(graph_db_path, session_id) -> str` |
| `context_engine/metrics/classifier.py` | Create — `classify_session_async(session_id, graph_db_path, metrics_db_path)` |
| `context_engine/metrics/report.py` | Create — `generate_report(compare, metrics_db_path, cost_per_token) -> str` |
| `context_engine/metrics/__main__.py` | Create — CLI: `python3 -m context_engine.metrics report --compare cold warm` |
| `context_engine/config.py` | Modify — add `metrics_db_path` property |
| `context_engine/server.py` | Modify — session_end triggers `classify_session_async` in daemon thread |
| `tests/test_metrics_tagger.py` | Create |
| `tests/test_metrics_classifier.py` | Create |
| `tests/test_metrics_report.py` | Create |

---

## Task 1: Package Scaffold + Config

**Files:**
- Create: `context_engine/metrics/__init__.py`
- Modify: `context_engine/config.py`

- [ ] **Step 1: Create the empty package marker**

```python
# context_engine/metrics/__init__.py
# (empty)
```

- [ ] **Step 2: Add metrics_db_path property to config.py**

In `context_engine/config.py`, add this property to the `ContextEngineConfig` dataclass, after the `db_path` property:

```python
    @property
    def metrics_db_path(self) -> str:
        return os.path.join(self.config_dir, "session_metrics.db")
```

- [ ] **Step 3: Verify config test still passes**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_config.py -v
```
Expected: all pass

- [ ] **Step 4: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/metrics/__init__.py context_engine/config.py
git commit -m "feat: metrics package scaffold + metrics_db_path config property"
```

---

## Task 2: Session Tagger

**Files:**
- Create: `context_engine/metrics/tagger.py`
- Create: `tests/test_metrics_tagger.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_tagger.py
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.tagger import determine_session_type
from context_engine.schema import init_db


def _ts(offset_seconds=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def graph_db(tmp_path):
    db_path = str(tmp_path / "graph.db")
    init_db(db_path)
    return db_path


def _add_session(db_path, session_id, started_at):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO sessions (session_id, agent, started_at) VALUES (?, ?, ?)",
        (session_id, "claude-code", started_at),
    )
    conn.commit()
    conn.close()


def _add_entity(db_path, created_at):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO entities (entity_id, type, content, confidence, created_at, source_agent, source_session) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (f"e-{created_at}", "decision", "some decision", 0.5, created_at, "claude-code", "old-session"),
    )
    conn.commit()
    conn.close()


def test_cold_no_prior_entities(graph_db):
    _add_session(graph_db, "sess-1", _ts())
    assert determine_session_type(graph_db, "sess-1") == "cold"


def test_warm_five_or_more_prior_entities(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-2", session_start)
    for i in range(6):
        _add_entity(graph_db, _ts(-100 - i))  # entities before session start
    assert determine_session_type(graph_db, "sess-2") == "warm"


def test_partial_one_to_four_prior_entities(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-3", session_start)
    _add_entity(graph_db, _ts(-50))
    assert determine_session_type(graph_db, "sess-3") == "partial"


def test_entities_after_session_start_dont_count(graph_db):
    session_start = _ts()
    _add_session(graph_db, "sess-4", session_start)
    for i in range(10):
        _add_entity(graph_db, _ts(10 + i))  # entities AFTER session start
    assert determine_session_type(graph_db, "sess-4") == "cold"


def test_unknown_session_id_returns_cold(graph_db):
    assert determine_session_type(graph_db, "nonexistent-session") == "cold"


def test_unknown_string_session_id_returns_cold(graph_db):
    assert determine_session_type(graph_db, "unknown") == "cold"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_tagger.py -v
```
Expected: `ModuleNotFoundError: No module named 'context_engine.metrics.tagger'`

- [ ] **Step 3: Implement tagger.py**

```python
# context_engine/metrics/tagger.py
"""Session type tagger — determines if a session was cold, warm, or partial.

cold    = no entities existed before this session started (no prior context)
partial = 1-4 entities existed (some context, sparse)
warm    = 5+ entities existed (meaningful context available)
"""
import logging
import sqlite3

logger = logging.getLogger(__name__)

WARM_THRESHOLD = 5


def determine_session_type(graph_db_path: str, session_id: str) -> str:
    """Return 'cold', 'partial', or 'warm' based on pre-existing entity count."""
    if not session_id or session_id == "unknown":
        return "cold"
    try:
        conn = sqlite3.connect(graph_db_path)
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return "cold"
        started_at = row[0]
        pre_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE created_at < ?", (started_at,)
        ).fetchone()[0]
        conn.close()
        if pre_count >= WARM_THRESHOLD:
            return "warm"
        if pre_count > 0:
            return "partial"
        return "cold"
    except Exception as exc:
        logger.debug("determine_session_type failed: %s", exc)
        return "cold"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_tagger.py -v
```
Expected: 6 passed

- [ ] **Step 5: Full suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all green

- [ ] **Step 6: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/metrics/tagger.py tests/test_metrics_tagger.py
git commit -m "feat: session tagger — cold/warm/partial based on pre-existing entity count"
```

---

## Task 3: Post-Session Classifier

**Files:**
- Create: `context_engine/metrics/classifier.py`
- Create: `tests/test_metrics_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_classifier.py
import sqlite3
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.classifier import (
    find_transcript, compute_metrics, classify_session_async
)
from context_engine.schema import init_db
from scripts.lib.metrics_db import init_db as init_metrics_db


def test_find_transcript_returns_none_for_unknown(tmp_path):
    result = find_transcript("unknown", str(tmp_path))
    assert result is None


def test_find_transcript_returns_none_for_missing_session(tmp_path):
    result = find_transcript("00000000-0000-0000-0000-000000000000", str(tmp_path))
    assert result is None


def test_find_transcript_finds_existing_jsonl(tmp_path):
    session_id = "abc12345-0000-0000-0000-000000000000"
    proj = tmp_path / "project1"
    proj.mkdir()
    jsonl = proj / f"{session_id}.jsonl"
    jsonl.write_text('{"type": "user", "message": {"role": "user", "content": "hi"}}\n')
    result = find_transcript(session_id, str(tmp_path))
    assert result == jsonl


def test_compute_metrics_all_context():
    messages = [
        {"index": 0, "role": "user", "text": "re-explain", "token_count": 100},
        {"index": 1, "role": "assistant", "text": "clarify", "token_count": 100},
    ]
    labels = [{"index": 0, "label": "context-restoration"}, {"index": 1, "label": "agent-clarifying"}]
    result = compute_metrics(messages, labels)
    assert result["cor"] == 1.0
    assert result["context_messages"] == 2


def test_compute_metrics_no_new_work():
    messages = [{"index": 0, "role": "user", "text": "hi", "token_count": 10}]
    labels = [{"index": 0, "label": "other"}]
    result = compute_metrics(messages, labels)
    assert result["first_new_work_message_index"] is None
    assert result["warmup_message_count"] == 1


def test_classify_session_async_skips_unknown_session(tmp_path):
    db = str(tmp_path / "graph.db")
    init_db(db)
    mdb = str(tmp_path / "metrics.db")
    init_metrics_db(mdb)
    # Should not raise, should not write anything
    classify_session_async("unknown", db, mdb, projects_dir=str(tmp_path))
    conn = sqlite3.connect(mdb)
    assert conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0] == 0


def test_classify_session_async_skips_missing_transcript(tmp_path):
    db = str(tmp_path / "graph.db")
    init_db(db)
    mdb = str(tmp_path / "metrics.db")
    init_metrics_db(mdb)
    classify_session_async("session-not-found", db, mdb, projects_dir=str(tmp_path))
    conn = sqlite3.connect(mdb)
    assert conn.execute("SELECT COUNT(*) FROM session_metrics").fetchone()[0] == 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_classifier.py -v
```
Expected: `ModuleNotFoundError: No module named 'context_engine.metrics.classifier'`

- [ ] **Step 3: Implement classifier.py**

```python
# context_engine/metrics/classifier.py
"""Post-session message classifier for Context Engine metrics.

classify_session_async() — call in a daemon thread at session end.
Finds the JSONL transcript, classifies messages with Haiku, computes
COR/WMC/CR, writes a row to session_metrics + rows to message_labels.
"""
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Add scripts dir so we can reuse session_loader, session_classifier, metrics_db
_scripts_dir = str(Path(__file__).parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from lib.session_loader import load_session
from lib.session_classifier import classify_session as _haiku_classify
from lib.metrics_db import init_db as init_metrics_db, insert_session, insert_labels
from context_engine.metrics.tagger import determine_session_type


def _make_client():
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()
    from anthropic import AnthropicVertex
    return AnthropicVertex(
        project_id=os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-gcp-pnd-pe-eng-claude"),
        region="us-east5",
    )


def find_transcript(session_id: str, projects_dir: str = None) -> Optional[Path]:
    """Search ~/.claude/projects/**/{session_id}.jsonl for the session transcript."""
    if not session_id or session_id in ("unknown", ""):
        return None
    base = Path(projects_dir) if projects_dir else Path.home() / ".claude" / "projects"
    for path in base.rglob(f"{session_id}.jsonl"):
        return path
    return None


def compute_metrics(messages: list, labels: list) -> dict:
    """Compute COR, WMC, CR from classified messages."""
    label_map = {l["index"]: l["label"] for l in labels}
    total_messages = len(messages)
    total_tokens = sum(m.get("token_count", 1) for m in messages)
    context_tokens = context_messages = correction_count = user_messages = 0
    first_new_work = total_messages

    for m in messages:
        label = label_map.get(m["index"], "other")
        tokens = m.get("token_count", 1)
        if label in ("context-restoration", "agent-clarifying"):
            context_messages += 1
            context_tokens += tokens
        if label == "correction":
            correction_count += 1
        if label == "new-work" and m["index"] < first_new_work:
            first_new_work = m["index"]
        if m["role"] == "user":
            user_messages += 1

    cor = context_tokens / total_tokens if total_tokens > 0 else 0.0
    cr = correction_count / user_messages if user_messages > 0 else 0.0
    return {
        "total_messages": total_messages,
        "total_tokens": total_tokens,
        "context_messages": context_messages,
        "context_tokens": context_tokens,
        "warmup_message_count": first_new_work,
        "correction_count": correction_count,
        "first_new_work_message_index": first_new_work if first_new_work < total_messages else None,
        "cor": cor,
        "cr": cr,
    }


def _length_bucket(n: int) -> str:
    if n >= 1000:
        return "1000+"
    if n >= 200:
        return "200-999"
    if n >= 50:
        return "50-199"
    if n >= 10:
        return "10-49"
    return "<10"


def classify_session_async(
    session_id: str,
    graph_db_path: str,
    metrics_db_path: str,
    projects_dir: str = None,
) -> None:
    """Classify a session transcript and write metrics. Safe to run in a daemon thread."""
    try:
        path = find_transcript(session_id, projects_dir)
        if not path:
            logger.debug("No transcript for session %s — skipping metrics", session_id)
            return

        session = load_session(path)
        messages = session.get("messages", [])
        if not messages:
            return

        for m in messages:
            m["token_count"] = max(1, len(m["text"]) // 4)

        client = _make_client()
        labels_raw = _haiku_classify(client, messages)
        label_map = {l["index"]: l["label"] for l in labels_raw}

        db_labels = [
            {
                "index": m["index"],
                "role": m["role"],
                "label": label_map.get(m["index"], "other"),
                "token_count": m["token_count"],
                "is_correction": label_map.get(m["index"]) == "correction",
            }
            for m in messages
        ]

        metrics = compute_metrics(messages, labels_raw)
        n = metrics["total_messages"]
        session_type = determine_session_type(graph_db_path, session_id)
        agent = session.get("agent", "unknown")

        conn = init_metrics_db(metrics_db_path)
        insert_session(conn, {
            "session_id": session_id,
            "date": session.get("date", datetime.now(timezone.utc).date().isoformat()),
            "session_type": session_type,
            "agent": agent,
            "is_marathon": n >= 200,
            "length_bucket": _length_bucket(n),
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        })
        insert_labels(conn, session_id, db_labels)

        logger.info(
            "[Metrics] Session %s (%s/%s): COR=%.1f%% WMC=%d CR=%.1f%%",
            session_id[:8], session_type, agent,
            metrics["cor"] * 100, metrics["warmup_message_count"], metrics["cr"] * 100,
        )
    except Exception as exc:
        logger.warning("classify_session_async failed for %s: %s", session_id, exc)
```

- [ ] **Step 4: Add scripts to sys.path for tests**

The tests import `from scripts.lib.metrics_db import init_db`. Add the scripts path to conftest.py (or add it inside the test file). Edit `tests/conftest.py` to add:

```python
import sys
from pathlib import Path

# Make scripts/lib importable in tests
_scripts = str(Path(__file__).parent.parent / "scripts")
if _scripts not in sys.path:
    sys.path.insert(0, _scripts)
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_classifier.py -v
```
Expected: 6 passed

- [ ] **Step 6: Full suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all green

- [ ] **Step 7: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/metrics/classifier.py tests/test_metrics_classifier.py tests/conftest.py
git commit -m "feat: post-session classifier — async COR/WMC/CR computation and metrics DB write"
```

---

## Task 4: Comparison Report + CLI

**Files:**
- Create: `context_engine/metrics/report.py`
- Create: `context_engine/metrics/__main__.py`
- Create: `tests/test_metrics_report.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics_report.py
import sqlite3
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from context_engine.metrics.report import generate_report
from scripts.lib.metrics_db import init_db, insert_session


@pytest.fixture
def metrics_db(tmp_path):
    db_path = str(tmp_path / "metrics.db")
    conn = init_db(db_path)
    # Insert 2 cold sessions
    for i in range(2):
        insert_session(conn, {
            "session_id": f"cold-{i}", "date": "2026-06-01",
            "session_type": "cold", "agent": "scotty",
            "total_messages": 20, "total_tokens": 1000,
            "context_messages": 6, "context_tokens": 300,
            "warmup_message_count": 5, "correction_count": 2,
            "first_new_work_message_index": 5,
            "cor": 0.30, "cr": 0.10,
            "is_marathon": False, "length_bucket": "10-49",
            "created_at": "2026-06-01T10:00:00+00:00",
        })
    # Insert 1 warm session
    insert_session(conn, {
        "session_id": "warm-0", "date": "2026-06-06",
        "session_type": "warm", "agent": "scotty",
        "total_messages": 8, "total_tokens": 400,
        "context_messages": 1, "context_tokens": 50,
        "warmup_message_count": 1, "correction_count": 0,
        "first_new_work_message_index": 1,
        "cor": 0.125, "cr": 0.0,
        "is_marathon": False, "length_bucket": "10-49",
        "created_at": "2026-06-06T10:00:00+00:00",
    })
    return db_path


def test_report_contains_anxiety_map_section(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "ANXIETY MAP" in report
    assert "Marathon Session Rate" in report
    assert "Marathon Token Share" in report


def test_report_contains_efficiency_section(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "EFFICIENCY" in report
    assert "Context Overhead Ratio" in report
    assert "Warmup Message Count" in report
    assert "Correction Rate" in report


def test_report_shows_session_counts(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "2 cold" in report
    assert "1 warm" in report


def test_report_missing_baseline_returns_message(metrics_db):
    report = generate_report(["nonexistent", "warm"], metrics_db, cost_per_token=0.000015)
    assert "No sessions" in report


def test_report_cor_values_are_correct(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    # cold COR avg = 30.0%
    assert "30.0%" in report


def test_report_sld_histogram_present(metrics_db):
    report = generate_report(["cold", "warm"], metrics_db, cost_per_token=0.000015)
    assert "10-49" in report  # length bucket in SLD
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_report.py -v
```
Expected: `ModuleNotFoundError: No module named 'context_engine.metrics.report'`

- [ ] **Step 3: Implement report.py**

```python
# context_engine/metrics/report.py
"""Comparison report generator for Context Engine session metrics."""
import sqlite3
from typing import List

BUCKETS = ["<10", "10-49", "50-199", "200-999", "1000+"]


def _stats(conn: sqlite3.Connection, session_type: str) -> list:
    return conn.execute(
        """SELECT total_messages, total_tokens, is_marathon, length_bucket,
                  cor, warmup_message_count, cr, correction_count, agent
           FROM session_metrics WHERE session_type = ?""",
        (session_type,),
    ).fetchall()


def _avg(rows: list, idx: int) -> float:
    vals = [r[idx] for r in rows if r[idx] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _marathon_rate(rows: list) -> float:
    return sum(1 for r in rows if r[2]) / len(rows) * 100 if rows else 0.0


def _token_share(rows: list) -> float:
    total = sum(r[1] or 0 for r in rows)
    marathon = sum(r[1] or 0 for r in rows if r[2])
    return marathon / total * 100 if total else 0.0


def _sld(rows: list) -> dict:
    n = len(rows)
    return {b: f"{sum(1 for r in rows if r[3] == b) / n * 100:.0f}%" if n else "0%" for b in BUCKETS}


def generate_report(
    compare: List[str],
    metrics_db_path: str,
    cost_per_token: float = 0.000015,
) -> str:
    baseline_type, treatment_type = compare
    conn = sqlite3.connect(metrics_db_path)

    baseline = _stats(conn, baseline_type)
    treatment = _stats(conn, treatment_type)

    if not baseline:
        return f"No sessions with session_type='{baseline_type}' found in {metrics_db_path}"

    b_msr = _marathon_rate(baseline)
    t_msr = _marathon_rate(treatment)
    b_mts = _token_share(baseline)
    t_mts = _token_share(treatment)
    b_sld = _sld(baseline)
    t_sld = _sld(treatment) if treatment else {b: "—" for b in BUCKETS}

    b_cor = _avg(baseline, 4) * 100
    t_cor = _avg(treatment, 4) * 100
    b_wmc = _avg(baseline, 5)
    t_wmc = _avg(treatment, 5)
    b_cr = _avg(baseline, 6) * 100
    t_cr = _avg(treatment, 6) * 100

    b_cost = _avg(baseline, 1) * cost_per_token
    t_cost = _avg(treatment, 1) * cost_per_token

    n_b, n_t = len(baseline), len(treatment)
    W = 54

    lines = [
        "Context Engine Impact Report",
        "",
        f"Sessions analyzed: {n_b} {baseline_type}, {n_t} {treatment_type}",
        "",
        "── THE ANXIETY MAP " + "─" * (W - 19),
        "",
        f"{'Marathon Session Rate (MSR)':<32}{baseline_type}: {b_msr:.1f}%   {treatment_type}: {t_msr:.1f}%   {t_msr - b_msr:+.1f}pp",
        f"{'Marathon Token Share  (MTS)':<32}{baseline_type}: {b_mts:.1f}%   {treatment_type}: {t_mts:.1f}%   {t_mts - b_mts:+.1f}pp",
        "",
        "Session Length Distribution:",
    ]
    for bucket in BUCKETS:
        lines.append(f"  {bucket:>8}   {baseline_type}: {b_sld[bucket]:>5}    {treatment_type}: {t_sld[bucket]:>5}")
    lines += [
        "",
        "── EFFICIENCY GAINS (once fear leaves) " + "─" * (W - 39),
        "",
        f"{'':32}{baseline_type:<14}{treatment_type:<14}Delta",
        f"{'Context Overhead Ratio (COR)':<32}{b_cor:.1f}%{'':10}{t_cor:.1f}%{'':10}{t_cor - b_cor:+.1f}pp",
        f"{'Warmup Message Count  (WMC)':<32}{b_wmc:.1f} msg{'':7}{t_wmc:.1f} msg{'':7}{t_wmc - b_wmc:+.1f}",
        f"{'Correction Rate       (CR)':<32}{b_cr:.1f}%{'':10}{t_cr:.1f}%{'':10}{t_cr - b_cr:+.1f}pp",
        "",
        f"Token cost (avg/session):        ${b_cost:.4f}{'':10}${t_cost:.4f}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Implement __main__.py**

```python
# context_engine/metrics/__main__.py
"""CLI entry point: python3 -m context_engine.metrics report --compare cold warm"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from context_engine.metrics.report import generate_report

DEFAULT_DB = os.path.expanduser("~/.context-engine/session_metrics.db")


def main():
    parser = argparse.ArgumentParser(
        prog="python3 -m context_engine.metrics",
        description="Context Engine session metrics CLI",
    )
    sub = parser.add_subparsers(dest="command")

    rp = sub.add_parser("report", help="Generate cold vs warm comparison report")
    rp.add_argument(
        "--compare", nargs=2, default=["cold", "warm"],
        metavar=("BASELINE", "TREATMENT"),
    )
    rp.add_argument("--db", default=DEFAULT_DB, help="Path to session_metrics.db")
    rp.add_argument(
        "--cost-per-token", type=float, default=0.000015,
        help="Output token cost in USD (default: Sonnet 4.6 output price)",
    )

    args = parser.parse_args()

    if args.command == "report":
        print(generate_report(args.compare, args.db, args.cost_per_token))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_metrics_report.py -v
```
Expected: 6 passed

- [ ] **Step 6: Smoke test the CLI**

```bash
cd /Users/jbecker/context-engine
python3 -m context_engine.metrics report --compare cold warm
```
Expected: prints the two-tier report with 164 cold sessions and 0 warm sessions (no warm sessions yet — that's the point, they'll accumulate).

- [ ] **Step 7: Full suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all green

- [ ] **Step 8: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/metrics/report.py context_engine/metrics/__main__.py tests/test_metrics_report.py
git commit -m "feat: comparison report CLI — two-tier anxiety map + efficiency metrics"
```

---

## Task 5: Server Integration

**Files:**
- Modify: `context_engine/server.py`

- [ ] **Step 1: Write the failing test**

Add this test to `tests/test_server.py` inside the `TestSessionAPI` class:

```python
def test_session_end_returns_200(self, client):
    client.post("/context/session/start", json={"agent": "test", "session_id": "s-end-test"})
    resp = client.post("/context/session/end", json={
        "session_id": "s-end-test", "summary": "Did some work"
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["briefing_written"] is True
```

- [ ] **Step 2: Run to verify it currently passes** (baseline — endpoint already works)

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/test_server.py::TestSessionAPI -v
```
Expected: existing tests pass. Note the test above should also pass (just validates the endpoint still works after our changes).

- [ ] **Step 3: Modify server.py — wire async classification at session end**

In `context_engine/server.py`, find the `session_end` endpoint and add async classification after `graph.end_session`. Also add the `metrics_db_path` variable at the top of `create_app` (after `db = db_path or cfg.db_path`).

Add after `db = db_path or cfg.db_path`:
```python
    metrics_db = cfg.metrics_db_path
```

Replace the `session_end` endpoint with:

```python
    @app.post("/context/session/end")
    def session_end():
        import threading
        from context_engine.metrics.classifier import classify_session_async
        data = request.get_json()
        session_id = data.get("session_id", "")
        summary = data.get("summary")
        graph.end_session(session_id, summary=summary)
        cache.clear()
        generate_briefing(graph)
        # Trigger async classification — non-blocking
        if session_id and session_id != "unknown":
            threading.Thread(
                target=classify_session_async,
                args=(session_id, db, metrics_db),
                daemon=True,
            ).start()
        return jsonify({"briefing_written": True, "summary": summary})
```

- [ ] **Step 4: Run the full test suite**

```bash
cd /Users/jbecker/context-engine && python3 -m pytest tests/ -q
```
Expected: all green (classify_session_async won't fire in tests because session_id="unknown" or tests skip it)

- [ ] **Step 5: Restart CE and smoke test**

```bash
launchctl kickstart -k gui/$(id -u)/com.gaia.context-engine
sleep 3
curl -s http://localhost:8850/context/health | python3 -m json.tool
```
Expected: `{"status": "ok", ...}`

- [ ] **Step 6: End a session and check metrics DB**

```bash
# Simulate a session end via the API
curl -s -X POST http://localhost:8850/context/session/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "unknown", "summary": "smoke test"}' \
  | python3 -m json.tool
```
Expected: `{"briefing_written": true, "summary": "smoke test"}`

- [ ] **Step 7: Commit**

```bash
cd /Users/jbecker/context-engine
git add context_engine/server.py tests/test_server.py
git commit -m "feat: wire session_end → async metrics classification"
```

---

## Self-Review

**Spec coverage:**
- ✅ Session tagger: cold/warm/partial at session end — tagger.py, Task 2
- ✅ Post-session classifier: Haiku classification, COR/WMC/CR computed — classifier.py, Task 3
- ✅ Metrics DB write: session_metrics + message_labels — classifier.py uses scripts/lib/metrics_db, Task 3
- ✅ Log line: `[Metrics] Session {id}: COR={x}%, WMC={n}, CR={x}%` — classifier.py, Task 3
- ✅ No real-time overhead: classification in daemon thread — Task 5
- ✅ Comparison report CLI: `python3 -m context_engine.metrics report --compare cold warm` — Tasks 4 + CLI
- ✅ Two-tier output: Anxiety Map (MSR, MTS, SLD) then Efficiency (COR, WMC, CR) — report.py
- ✅ Server integration: session_end triggers async — Task 5
- ✅ is_marathon: ≥200 messages — classifier.py `_length_bucket` and `is_marathon` field
- ✅ length_bucket: five buckets per spec — `_length_bucket()` in classifier.py

**Placeholder scan:** None found.

**Type consistency:**
- `classify_session_async(session_id, graph_db_path, metrics_db_path, projects_dir=None)` — matches Task 3 definition and Task 5 call site `(session_id, db, metrics_db)`.
- `determine_session_type(graph_db_path, session_id)` — matches Task 2 definition and Task 3 call site.
- `generate_report(compare, metrics_db_path, cost_per_token)` — matches Task 4 definition and `__main__.py` call site.
- `compute_metrics(messages, labels)` — same signature as in baseline analyzer; tested in Task 3.
- `insert_session(conn, row)` — `row` dict keys match the existing `session_metrics` schema exactly (verified against DB).
