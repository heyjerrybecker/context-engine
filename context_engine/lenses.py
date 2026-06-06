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
