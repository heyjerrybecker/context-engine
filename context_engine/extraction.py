"""Basic keyword extraction lenses for the Context Engine.

Phase 2: simple pattern matching. No LLM calls.
Phase 3 upgrades these to LLM-powered extraction.
"""

import re
from typing import List, Dict

DECISION_PATTERNS = [
    (r"(?i)\blet'?s go with\b(.{5,80}?)(?:\.|$)", 0.5),
    (r"(?i)\bi'?ve decided\b(.{5,80}?)(?:\.|$)", 0.5),
    (r"(?i)\block it in\b(.{5,80}?)(?:\.|$)", 0.6),
    (r"(?i)\bwe'?ll use\b(.{5,80}?)(?:\.|$)", 0.5),
    (r"(?i)\bgoing with\b(.{5,80}?)(?:\.|$)", 0.4),
    (r"(?i)\bdecided to\b(.{5,80}?)(?:\.|$)", 0.5),
    (r"(?i)\bthe decision is\b(.{5,80}?)(?:\.|$)", 0.6),
    (r"(?i)\bwe'?re doing\b(.{5,80}?)(?:\.|$)", 0.4),
]

QUESTION_PATTERNS = [
    r"(?i)^should we\b",
    r"(?i)^what if\b",
    r"(?i)^could we\b",
    r"(?i)^would it\b",
    r"\?$",
]

TOPIC_CHANGE_PATTERNS = [
    (r"(?i)\b(?:can we|let'?s)\s+(?:switch|move|talk|turn)\s+to\s+(?:talking about\s+)?(.{5,60}?)(?:\?|\.|$)", "new_topic"),
    (r"(?i)\b(?:now|next),?\s+(?:let'?s|we should)\s+(?:look at|work on|discuss)\s+(.{5,60}?)(?:\.|$)", "new_topic"),
    (r"(?i)\bgoing back to\s+(?:the\s+)?(.{5,60}?)(?:\.|$)", "resume"),
    (r"(?i)\b(?:back to|returning to)\s+(?:the\s+)?(.{5,60}?)(?:\.|$)", "resume"),
    (r"(?i)\bpicking up\s+(?:where we left off|on)\s+(.{5,60}?)(?:\.|$)", "resume"),
]


def extract_decisions(text: str) -> List[Dict]:
    is_question = any(re.search(p, text) for p in QUESTION_PATTERNS)
    if is_question:
        return []

    results = []
    for pattern, confidence in DECISION_PATTERNS:
        match = re.search(pattern, text)
        if match:
            content = match.group(0).strip().rstrip(".")
            results.append({
                "type": "decision",
                "content": content,
                "confidence": confidence,
            })
            break

    return results


def extract_thread_signals(text: str) -> List[Dict]:
    results = []
    for pattern, signal_type in TOPIC_CHANGE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            title = match.group(1).strip().rstrip(".")
            results.append({
                "signal": signal_type,
                "title": title,
            })
            break

    return results


def extract_entities(user_msg: str, assistant_msg: str,
                     source_session: str = "", source_agent: str = "") -> List[Dict]:
    entities = []

    decisions = extract_decisions(user_msg)
    for d in decisions:
        d["source_session"] = source_session
        d["source_agent"] = source_agent
        entities.append(d)

    return entities
