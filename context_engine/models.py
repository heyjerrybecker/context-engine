"""Data models for the Context Engine knowledge graph."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class EntityType(str, Enum):
    DECISION = "decision"
    PROJECT = "project"
    INSIGHT = "insight"
    THREAD = "thread"
    STATE = "state"
    EVENT = "event"
    AGENT = "agent"
    ARTIFACT = "artifact"


class ThreadStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    DORMANT = "dormant"
    MERGED = "merged"


class RelationshipType(str, Enum):
    PRECEDED_BY = "preceded_by"
    CAUSED_BY = "caused_by"
    AFFECTS = "affects"
    BLOCKS = "blocks"
    RELATES_TO = "relates_to"
    CONVERGES_WITH = "converges_with"
    SUPERSEDES = "supersedes"
    INVOLVES = "involves"
    PRODUCED = "produced"
    TRIGGERED = "triggered"


CAUSAL_RELATIONSHIP_TYPES = {
    RelationshipType.CAUSED_BY,
    RelationshipType.TRIGGERED,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class Entity:
    type: EntityType
    content: str
    source_agent: str
    source_session: str
    entity_id: str = field(default_factory=_new_id)
    confidence: float = 0.5
    created_at: str = field(default_factory=_now_iso)
    last_referenced: str = field(default_factory=_now_iso)
    thread_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    local_name: Optional[str] = None
    role: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.type, str):
            try:
                self.type = EntityType(self.type)
            except ValueError:
                raise ValueError(f"Invalid entity type: {self.type}")

    @property
    def export_as(self) -> Optional[str]:
        return self.role if self.type == EntityType.AGENT else None

    def to_dict(self) -> dict:
        d = {
            "entity_id": self.entity_id,
            "type": self.type.value,
            "content": self.content,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_referenced": self.last_referenced,
            "source_session": self.source_session,
            "source_agent": self.source_agent,
            "thread_id": self.thread_id,
            "tags": self.tags,
        }
        if self.type == EntityType.AGENT:
            d["local_name"] = self.local_name
            d["role"] = self.role
            d["export_as"] = self.export_as
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(
            entity_id=d["entity_id"],
            type=d["type"],
            content=d["content"],
            confidence=d.get("confidence", 0.5),
            created_at=d.get("created_at", _now_iso()),
            last_referenced=d.get("last_referenced", _now_iso()),
            source_session=d.get("source_session", ""),
            source_agent=d.get("source_agent", ""),
            thread_id=d.get("thread_id"),
            tags=d.get("tags", []),
            local_name=d.get("local_name"),
            role=d.get("role"),
        )


@dataclass
class Thread:
    title: str
    thread_id: str = field(default_factory=_new_id)
    trigger_entity_id: Optional[str] = None
    status: ThreadStatus = ThreadStatus.ACTIVE
    created_at: str = field(default_factory=_now_iso)
    resolved_at: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.status, str):
            self.status = ThreadStatus(self.status)

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "title": self.title,
            "trigger_entity_id": self.trigger_entity_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Thread":
        return cls(
            thread_id=d["thread_id"],
            title=d["title"],
            trigger_entity_id=d.get("trigger_entity_id"),
            status=d.get("status", "active"),
            created_at=d.get("created_at", _now_iso()),
            resolved_at=d.get("resolved_at"),
        )


@dataclass
class Relationship:
    from_id: str
    to_id: str
    type: RelationshipType
    relationship_id: str = field(default_factory=_new_id)
    confidence: float = 1.0
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RelationshipType(self.type)

    def to_dict(self) -> dict:
        return {
            "relationship_id": self.relationship_id,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "type": self.type.value,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Relationship":
        return cls(
            relationship_id=d["relationship_id"],
            from_id=d["from_id"],
            to_id=d["to_id"],
            type=d["type"],
            confidence=d.get("confidence", 1.0),
            created_at=d.get("created_at", _now_iso()),
        )
