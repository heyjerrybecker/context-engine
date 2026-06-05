"""Confidence decay engine for the Context Engine knowledge graph.

Living Claim rules: entities that aren't referenced decay over time.
Referenced entities get reinforced. Below-threshold entities get archived.
"""

import math
import sqlite3
from datetime import datetime, timezone
from typing import List


def compute_decay(days_since_referenced: float, half_life_days: float = 14,
                  grace_days: float = 0) -> float:
    """Compute exponential decay factor based on time since last reference.

    Args:
        days_since_referenced: Days since entity was last referenced
        half_life_days: Days for confidence to decay to 50%
        grace_days: Days before decay starts

    Returns:
        Decay factor between 0.0 and 1.0
    """
    if days_since_referenced <= grace_days:
        return 1.0
    effective_days = days_since_referenced - grace_days
    return math.pow(0.5, effective_days / half_life_days)


def decay_confidence(db_path: str, half_life_days: float = 14,
                     archive_threshold: float = 0.1,
                     grace_days: float = 7) -> List[str]:
    """Apply confidence decay to all entities in the knowledge graph.

    Entities not referenced within grace period decay exponentially.
    Entities below archive_threshold are set to 0.0 confidence (archived).

    Args:
        db_path: Path to SQLite database
        half_life_days: Days for confidence to decay to 50%
        archive_threshold: Confidence below which entities are archived
        grace_days: Days before decay starts

    Returns:
        List of entity IDs that were archived
    """
    now = datetime.now(timezone.utc)
    archived_ids = []

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT entity_id, confidence, last_referenced FROM entities "
        "WHERE confidence > 0"
    ).fetchall()

    for entity_id, confidence, last_referenced in rows:
        try:
            ref_dt = datetime.fromisoformat(last_referenced)
            if ref_dt.tzinfo is None:
                ref_dt = ref_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        days_since = (now - ref_dt).total_seconds() / 86400
        factor = compute_decay(days_since, half_life_days, grace_days)
        new_confidence = confidence * factor

        if new_confidence < archive_threshold:
            new_confidence = 0.0
            archived_ids.append(entity_id)

        if abs(new_confidence - confidence) > 0.001:
            conn.execute(
                "UPDATE entities SET confidence = ? WHERE entity_id = ?",
                (round(new_confidence, 4), entity_id),
            )

    conn.commit()
    conn.close()
    return archived_ids
