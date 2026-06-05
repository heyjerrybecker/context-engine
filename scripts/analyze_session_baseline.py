#!/usr/bin/env python3
"""
Analyze existing Claude Code JSONL session files to establish cold-session
baseline metrics (COR, WMC, CR) before Context Engine ships warm sessions.

Usage:
    python3 analyze_session_baseline.py
    python3 analyze_session_baseline.py --limit 10 --agent iris
    python3 analyze_session_baseline.py --projects-dir ~/.claude/projects --db /tmp/metrics.db
"""
import sys
import os
import argparse
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib.session_loader import find_session_files, load_session
from lib.session_classifier import classify_session
from lib.metrics_db import init_db, insert_session, insert_labels
from lib.metrics_reporter import print_summary

DEFAULT_DB = os.path.expanduser("~/.context-engine/session_metrics.db")
DEFAULT_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
MIN_MESSAGES = 4

VERTEX_PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "itpc-gcp-pnd-pe-eng-claude")
VERTEX_REGION = "us-east5"


def _make_client():
    """Return an Anthropic client. Prefers Vertex AI; falls back to direct API key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        return anthropic.Anthropic()

    from anthropic import AnthropicVertex
    return AnthropicVertex(project_id=VERTEX_PROJECT, region=VERTEX_REGION)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def compute_metrics(messages: list, labels: list) -> dict:
    label_map = {l["index"]: l["label"] for l in labels}
    total_messages = len(messages)
    total_tokens = sum(m.get("token_count", 1) for m in messages)
    context_tokens = 0
    context_messages = 0
    correction_count = 0
    user_messages = 0
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB, help="Path to output SQLite DB")
    parser.add_argument("--projects-dir", default=DEFAULT_PROJECTS_DIR)
    parser.add_argument("--limit", type=int, help="Max sessions to process (default: all)")
    parser.add_argument("--min-messages", type=int, default=MIN_MESSAGES,
                        help="Skip sessions with fewer messages than this")
    args = parser.parse_args()

    client = _make_client()
    conn = init_db(args.db)

    files = find_session_files(args.projects_dir)
    if args.limit:
        files = files[:args.limit]

    print(f"Found {len(files)} session files in {args.projects_dir}")
    print(f"Writing metrics to {args.db}\n")

    processed = 0
    skipped = 0

    for path in files:
        session = load_session(path)
        messages = session["messages"]

        if len(messages) < args.min_messages:
            skipped += 1
            continue

        print(f"[{processed + 1}] {session['session_id'][:12]}... "
              f"({len(messages)} msgs, agent={session['agent']})")

        for m in messages:
            m["token_count"] = estimate_tokens(m["text"])

        labels_raw = classify_session(client, messages)
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

        insert_session(conn, {
            "session_id": session["session_id"],
            "date": session["date"],
            "session_type": "cold",
            "agent": session["agent"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metrics,
        })
        insert_labels(conn, session["session_id"], db_labels)
        processed += 1

    print(f"\nDone. Processed {processed} sessions, skipped {skipped} (too short).")
    print_summary(conn)


if __name__ == "__main__":
    main()
