import json
from pathlib import Path


def extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(parts)
    return ""


def infer_agent(path: Path) -> str:
    project = path.parent.name
    if "iris" in project:
        return "iris"
    return "scotty"


def load_session(path: Path, max_messages: int = 100) -> dict:
    messages = []
    session_id = path.stem
    date = None

    with open(path) as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("type") not in ("user", "assistant"):
                continue

            content = record.get("message", {}).get("content", "")
            text = extract_text(content)
            if not text.strip():
                continue

            ts = record.get("timestamp", "")
            if date is None and ts:
                date = ts[:10]  # YYYY-MM-DD

            messages.append({
                "role": record["type"],
                "text": text,
                "timestamp": ts,
                "index": len(messages),
            })

            if len(messages) >= max_messages:
                break

    return {
        "session_id": session_id,
        "file": str(path),
        "date": date or "",
        "agent": infer_agent(path),
        "messages": messages,
    }


def find_session_files(projects_dir: str) -> list:
    base = Path(projects_dir).expanduser()
    return sorted(base.rglob("*.jsonl"))
