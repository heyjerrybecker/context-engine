"""Background extraction worker for the Context Engine.

start_worker(db_path) — call once at Flask app startup.
enqueue(user_msg, assistant_msg, session_id) — non-blocking, returns immediately.
The worker thread reads from the queue, calls Haiku lenses, writes entities to the graph.
"""
import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None


def start_worker(db_path: str, source_agent: str = "claude-code") -> None:
    """Start the background extraction worker. Safe to call multiple times."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _worker_thread = threading.Thread(
        target=_run,
        args=(db_path, source_agent),
        daemon=True,
        name="ce-extraction-worker",
    )
    _worker_thread.start()
    logger.info("Context Engine extraction worker started (db=%s)", db_path)


def enqueue(user_msg: str, assistant_msg: str, session_id: str) -> None:
    """Queue a message pair for async extraction. Non-blocking."""
    _queue.put({"user_msg": user_msg, "assistant_msg": assistant_msg, "session_id": session_id})


def _run(db_path: str, source_agent: str) -> None:
    import sys
    from context_engine.graph import KnowledgeGraph
    from context_engine.lenses import extract_from_turn
    from context_engine.models import Entity

    graph = KnowledgeGraph(db_path)
    print(f"[CE worker] started, db={db_path}", file=sys.stderr, flush=True)

    while True:
        try:
            item = _queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            print(f"[CE worker] extracting (session={item['session_id']})...", file=sys.stderr, flush=True)
            entities = extract_from_turn(item["user_msg"], item["assistant_msg"])
            print(f"[CE worker] got {len(entities)} entities", file=sys.stderr, flush=True)
            for e_data in entities:
                if e_data.get("type") == "thread_signal":
                    continue
                entity = Entity(
                    type=e_data["type"],
                    content=e_data["content"],
                    confidence=e_data["confidence"],
                    source_agent=source_agent,
                    source_session=item["session_id"],
                )
                graph.add_entity(entity)
                print(f"[CE worker] saved: {entity.type} — {entity.content[:80]}", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"[CE worker] ERROR: {exc}", file=sys.stderr, flush=True)
        finally:
            _queue.task_done()
