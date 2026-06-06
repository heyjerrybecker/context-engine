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
    from context_engine.graph import KnowledgeGraph
    from context_engine.lenses import extract_from_turn
    from context_engine.models import Entity

    graph = KnowledgeGraph(db_path)

    while True:
        try:
            item = _queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            entities = extract_from_turn(item["user_msg"], item["assistant_msg"])
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
                logger.debug("Extracted %s: %s", entity.type, entity.content[:60])
        except Exception as exc:
            logger.warning("Worker extraction error (item dropped): %s", exc)
        finally:
            _queue.task_done()
