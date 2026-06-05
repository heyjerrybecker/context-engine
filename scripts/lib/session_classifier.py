import json
import re

VALID_LABELS = {"context-restoration", "agent-clarifying", "new-work", "correction", "other"}
CHUNK_SIZE = 30

SYSTEM_PROMPT = """You are classifying messages from AI assistant sessions to measure context overhead.

Assign ONE label per message:
- context-restoration (user msg): user re-establishes prior context instead of contributing new info
- agent-clarifying (assistant msg): agent asks for info it should know, or makes wrong assumptions about prior state
- new-work (user or assistant): genuine forward progress — new code, design, decisions, insights
- correction (user msg): user redirects agent due to missing/wrong context ("No, we decided...", "Remember that...", "That's not right...")
- other: acknowledgments, confirmations, pleasantries, meta-conversation

Respond ONLY with a JSON array: [{"index": N, "label": "label-name"}, ...]"""


def _classify_chunk(client, messages: list) -> list:
    formatted = "\n\n".join(
        f"[{m['index']}] {m['role'].upper()}: {m['text'][:400]}"
        for m in messages
    )
    prompt = f"Classify each message:\n\n{formatted}"

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    # Greedy match: captures from first '[' to last ']', handles multi-line arrays.
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return [{"index": m["index"], "label": "other"} for m in messages]

    try:
        parsed = json.loads(match.group())
        for item in parsed:
            if item.get("label") not in VALID_LABELS:
                item["label"] = "other"
        return parsed
    except (json.JSONDecodeError, KeyError):
        return [{"index": m["index"], "label": "other"} for m in messages]


def classify_session(client, messages: list) -> list:
    if not messages:
        return []

    results = []
    for i in range(0, len(messages), CHUNK_SIZE):
        chunk = messages[i:i + CHUNK_SIZE]
        results.extend(_classify_chunk(client, chunk))

    return results
