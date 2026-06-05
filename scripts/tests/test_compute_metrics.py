import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def _make_msgs(*pairs):
    """pairs: (role, token_count) tuples, indexed from 0"""
    return [{"index": i, "role": r, "token_count": t, "text": "x"} for i, (r, t) in enumerate(pairs)]


def test_cor_all_context():
    msgs = _make_msgs(("user", 100), ("assistant", 100))
    labels = [{"index": 0, "label": "context-restoration"}, {"index": 1, "label": "agent-clarifying"}]
    result = compute_metrics(msgs, labels)
    assert result["cor"] == 1.0
    assert result["context_messages"] == 2


def test_cor_no_context():
    msgs = _make_msgs(("user", 100), ("assistant", 100))
    labels = [{"index": 0, "label": "new-work"}, {"index": 1, "label": "new-work"}]
    result = compute_metrics(msgs, labels)
    assert result["cor"] == 0.0


def test_wmc_first_new_work_at_index_3():
    msgs = _make_msgs(("user", 10), ("assistant", 10), ("user", 10), ("assistant", 10))
    labels = [
        {"index": 0, "label": "context-restoration"},
        {"index": 1, "label": "agent-clarifying"},
        {"index": 2, "label": "context-restoration"},
        {"index": 3, "label": "new-work"},
    ]
    result = compute_metrics(msgs, labels)
    assert result["warmup_message_count"] == 3
    assert result["first_new_work_message_index"] == 3


def test_wmc_no_new_work():
    msgs = _make_msgs(("user", 10), ("assistant", 10))
    labels = [{"index": 0, "label": "other"}, {"index": 1, "label": "other"}]
    result = compute_metrics(msgs, labels)
    assert result["warmup_message_count"] == 2  # total message count
    assert result["first_new_work_message_index"] is None


def test_cr_with_corrections():
    msgs = _make_msgs(("user", 10), ("user", 10), ("user", 10))
    labels = [
        {"index": 0, "label": "new-work"},
        {"index": 1, "label": "correction"},
        {"index": 2, "label": "other"},
    ]
    result = compute_metrics(msgs, labels)
    assert abs(result["cr"] - 1/3) < 0.001
    assert result["correction_count"] == 1


def test_empty_session():
    result = compute_metrics([], [])
    assert result["cor"] == 0.0
    assert result["cr"] == 0.0
    assert result["total_messages"] == 0
