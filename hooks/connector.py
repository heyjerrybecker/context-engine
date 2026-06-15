#!/usr/bin/env python3
"""
Claude Code → Context Engine connector.

Hooks: SessionStart, UserPromptSubmit, Stop
Calls Context Engine REST API on localhost:8850.

Field names: Claude Code uses snake_case (session_id, transcript_path, etc.)
"""

import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime

API_BASE = "http://127.0.0.1:8850"
DEBUG_LOG = "/tmp/ce_hook_payloads.log"
_CURSOR_DIR = os.path.expanduser("~/.context-engine/cursors")


def _debug(event, data):
    try:
        os.makedirs(os.path.dirname(DEBUG_LOG), exist_ok=True)
        with open(DEBUG_LOG, "a") as f:
            f.write(f"\n--- {datetime.now().isoformat()} | {event} ---\n")
            f.write(json.dumps(data, indent=2, default=str)[:3000] + "\n")
    except Exception:
        pass


def _post(path, body):
    try:
        req = urllib.request.Request(
            f"{API_BASE}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _get(path):
    try:
        req = urllib.request.Request(f"{API_BASE}{path}")
        with urllib.request.urlopen(req, timeout=4) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _extract_text(content):
    """Extract plain text from Claude Code message content (string or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _get_cursor_path(session_id):
    os.makedirs(_CURSOR_DIR, exist_ok=True)
    return os.path.join(_CURSOR_DIR, f"{session_id}.cursor")


def _read_cursor(session_id):
    path = _get_cursor_path(session_id)
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def _write_cursor(session_id, position):
    path = _get_cursor_path(session_id)
    try:
        with open(path, "w") as f:
            f.write(str(position))
    except Exception:
        pass


def _read_new_pairs(transcript_path, session_id):
    """Read new (user, assistant) pairs from transcript JSONL since last cursor."""
    if not transcript_path or not os.path.exists(transcript_path):
        return []

    cursor = _read_cursor(session_id)
    pairs = []
    last_user = None
    line_num = 0

    try:
        with open(transcript_path) as f:
            for line in f:
                line_num += 1
                if line_num <= cursor:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if obj.get("type") not in ("user", "assistant"):
                    continue

                msg = obj.get("message", {})
                text = _extract_text(msg.get("content", ""))
                if not text or len(text) < 5:
                    continue

                if obj["type"] == "user":
                    last_user = text
                elif obj["type"] == "assistant" and last_user:
                    pairs.append((last_user[:2000], text[:2000]))
                    last_user = None

        _write_cursor(session_id, line_num)
    except Exception:
        pass

    return pairs


def handle_session_start(payload):
    session_id = payload.get("session_id", "unknown")
    _debug("session-start", {"session_id": session_id, "keys": sorted(payload.keys())})

    result = _post("/context/session/start", {
        "agent": "claude-code",
        "session_id": session_id,
    })

    if result and result.get("briefing"):
        briefing = result["briefing"]
        if "Nothing in flight" not in briefing:
            _debug("session-start-injecting", {"briefing_len": len(briefing)})
            return {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": f"[Context Engine Briefing]\n\n{briefing}",
                }
            }
    return {}


def handle_user_prompt(payload):
    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")
    _debug("user-prompt", {"session_id": session_id, "transcript_path": transcript_path})

    pairs = _read_new_pairs(transcript_path, session_id)
    if pairs:
        for user_msg, asst_msg in pairs:
            _post("/context/ingest", {
                "user_message": user_msg,
                "assistant_message": asst_msg,
                "session_id": session_id,
            })
        _debug("user-prompt-ingested", {"pairs": len(pairs)})

    return {}


def handle_stop(payload):
    """Ingest new message pairs on each Stop (fires after every Claude response).

    Does NOT call session/end here — Stop fires on every turn, not just
    at session close. Session boundaries are handled by session/start
    (each new session implicitly closes the previous one).
    """
    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")
    _debug("stop", {"session_id": session_id, "keys": sorted(payload.keys())})

    pairs = _read_new_pairs(transcript_path, session_id)
    if pairs:
        for user_msg, asst_msg in pairs:
            _post("/context/ingest", {
                "user_message": user_msg,
                "assistant_message": asst_msg,
                "session_id": session_id,
            })
        _debug("stop-ingested", {"pairs": len(pairs)})

    return {}


def main():
    try:
        event = sys.argv[1] if len(sys.argv) > 1 else "unknown"
        payload = json.load(sys.stdin)

        if event == "session-start":
            result = handle_session_start(payload)
        elif event == "user-prompt":
            result = handle_user_prompt(payload)
        elif event == "stop":
            result = handle_stop(payload)
        else:
            result = {}

        print(json.dumps(result))

    except Exception:
        print(json.dumps({}))
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
