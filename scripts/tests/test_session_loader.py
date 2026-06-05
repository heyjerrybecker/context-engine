import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.session_loader import extract_text, load_session, find_session_files, infer_agent


def test_extract_text_from_string():
    assert extract_text("Hello world") == "Hello world"


def test_extract_text_from_blocks():
    content = [
        {"type": "thinking", "thinking": "internal"},
        {"type": "text", "text": "Hello"},
        {"type": "text", "text": "World"},
    ]
    assert extract_text(content) == "Hello\nWorld"


def test_extract_text_empty():
    assert extract_text([]) == ""
    assert extract_text("") == ""


def test_load_session(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    session = load_session(fixture)
    assert session["session_id"] == "sample_session"
    assert len(session["messages"]) == 4  # fixture has 4 non-empty messages
    assert session["messages"][0]["role"] == "user"
    assert session["messages"][0]["index"] == 0
    assert "timestamp" in session["messages"][0]
    assert session["date"] is not None


def test_load_session_skips_empty_messages(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "sample_session.jsonl"
    session = load_session(fixture)
    for m in session["messages"]:
        assert m["text"].strip() != ""


def test_infer_agent_iris():
    path = Path("/Users/jbecker/.claude/projects/-Users-jbecker-iris-workspace/abc.jsonl")
    assert infer_agent(path) == "iris"


def test_infer_agent_scotty():
    path = Path("/Users/jbecker/.claude/projects/-Users-jbecker-tricorder-repo/abc.jsonl")
    assert infer_agent(path) == "scotty"


def test_find_session_files(tmp_path):
    (tmp_path / "proj1").mkdir()
    (tmp_path / "proj1" / "abc.jsonl").write_text("{}")
    (tmp_path / "proj1" / "other.txt").write_text("ignore")
    files = find_session_files(str(tmp_path))
    assert len(files) == 1
    assert files[0].suffix == ".jsonl"
