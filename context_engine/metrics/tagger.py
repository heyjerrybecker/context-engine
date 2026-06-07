"""Session type tagger — determines if a session was cold, warm, or partial.

cold    = no entities existed before this session started (no prior context)
partial = 1-4 entities existed (some context, sparse)
warm    = 5+ entities existed (meaningful context available)
"""
import logging
import sqlite3

logger = logging.getLogger(__name__)

WARM_THRESHOLD = 5


def determine_session_type(graph_db_path: str, session_id: str) -> str:
    """Return 'cold', 'partial', or 'warm' based on pre-existing entity count."""
    if not session_id or session_id == "unknown":
        return "cold"
    try:
        conn = sqlite3.connect(graph_db_path)
        row = conn.execute(
            "SELECT started_at FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if not row:
            return "cold"
        started_at = row[0]
        pre_count = conn.execute(
            "SELECT COUNT(*) FROM entities WHERE created_at < ?", (started_at,)
        ).fetchone()[0]
        conn.close()
        if pre_count >= WARM_THRESHOLD:
            return "warm"
        if pre_count > 0:
            return "partial"
        return "cold"
    except Exception as exc:
        logger.debug("determine_session_type failed: %s", exc)
        return "cold"
