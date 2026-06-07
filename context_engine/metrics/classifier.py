"""Post-session message classifier for Context Engine metrics.

classify_session_async() — call in a daemon thread at session end.
Finds the JSONL transcript, classifies messages with Haiku, computes
COR/WMC/CR, and writes a row to session_metrics + rows to message_labels.
"""
import logging
import os
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
