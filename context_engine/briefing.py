"""Session briefing generation from the knowledge graph."""

from typing import List
from context_engine.models import Entity, Thread, ThreadStatus
from context_engine.graph import KnowledgeGraph


def generate_briefing(graph: KnowledgeGraph, lifecycle=None) -> str:
    sections = []

    # Right Now — active threads with their entities
    active_threads = graph.list_threads(status=ThreadStatus.ACTIVE)
    sections.append("## Right Now")
    if active_threads:
        for t in active_threads[:5]:
            entities = graph.get_thread_entities(t.thread_id)
            entity_summary = ", ".join(e.content[:60] for e in entities[:3])
            line = f"- **{t.title}**"
            if entity_summary:
                line += f" — {entity_summary}"
            sections.append(line)
    else:
        sections.append("Nothing in flight.")

    # Last Session — entities from most recent ended session
    sections.append("")
    sections.append("## Last Session")
    last_session = graph.get_last_ended_session()
    if last_session:
        session_entities = graph.list_entities(session_filter=last_session["session_id"], limit=10)
        if last_session.get("summary"):
            sections.append(f"Summary: {last_session['summary']}")
        if session_entities:
            for e in session_entities[:5]:
                sections.append(f"- [{e.type.value}] {e.content[:80]}")
        else:
            sections.append("No entities captured.")
    else:
        sections.append("No previous session recorded.")

    # Open Threads — active threads that might need attention
    sections.append("")
    sections.append("## Open Threads")
    if active_threads:
        for t in active_threads[:5]:
            sections.append(f"- {t.title}")
    else:
        sections.append("No open threads.")

    if lifecycle:
        has_warnings = False
        for agent_id in lifecycle.get_registered_agents():
            health = lifecycle.get_health(agent_id)
            if not health:
                continue
            util_pct = int(health["utilization"] * 100)
            if util_pct > 80:
                if not has_warnings:
                    sections.append("")
                    sections.append("## Memory Health")
                    has_warnings = True
                if util_pct > 95:
                    sections.append(f"- **{agent_id}**: {util_pct}% capacity (CRITICAL)")
                else:
                    sections.append(f"- **{agent_id}**: {util_pct}% capacity — consider archiving older entries")

    return "\n".join(sections)
