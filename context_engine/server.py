"""Flask HTTP API for the Context Engine."""

from flask import Flask, request, jsonify

from context_engine.config import load_config, ContextEngineConfig
from context_engine.schema import init_db
from context_engine.graph import KnowledgeGraph
from context_engine.briefing import generate_briefing
from context_engine.models import Entity, Thread, Relationship


def create_app(db_path: str = None, config_dir: str = None) -> Flask:
    app = Flask(__name__)

    cfg = load_config(config_dir) if config_dir else load_config()
    db = db_path or cfg.db_path
    init_db(db)
    graph = KnowledgeGraph(db)

    @app.get("/context/health")
    def health():
        return jsonify({
            "status": "ok",
            "db": db,
            "port": cfg.port,
        })

    @app.get("/context/identity")
    def identity():
        return jsonify({
            "soul": cfg.load_soul(),
            "user_model": cfg.load_user_model(),
        })

    @app.get("/context/briefing")
    def briefing():
        text = generate_briefing(graph)
        active = graph.list_threads()
        return jsonify({
            "briefing": text,
            "generated_at": __import__("datetime").datetime.now(
                __import__("datetime").timezone.utc).isoformat(),
            "active_threads": len(active),
        })

    # --- Session management ---

    @app.post("/context/session/start")
    def session_start():
        data = request.get_json()
        session_id = data.get("session_id", "")
        agent = data.get("agent", "unknown")
        graph.start_session(session_id, agent)
        text = generate_briefing(graph)
        active = graph.list_threads()
        return jsonify({
            "briefing": text,
            "identity": {
                "soul": cfg.load_soul(),
                "user_model": cfg.load_user_model(),
            },
            "active_threads": [t.to_dict() for t in active],
        })

    @app.post("/context/session/end")
    def session_end():
        data = request.get_json()
        session_id = data.get("session_id", "")
        summary = data.get("summary")
        graph.end_session(session_id, summary=summary)
        text = generate_briefing(graph)
        return jsonify({
            "briefing_written": True,
            "summary": summary,
        })

    # --- Entity CRUD ---

    @app.post("/context/entity")
    def create_entity():
        data = request.get_json()
        entity = Entity(
            type=data["type"],
            content=data["content"],
            source_agent=data.get("source_agent", "unknown"),
            source_session=data.get("source_session", ""),
            confidence=data.get("confidence", 0.5),
            thread_id=data.get("thread_id"),
            tags=data.get("tags", []),
            local_name=data.get("local_name"),
            role=data.get("role"),
        )
        entity_id = graph.add_entity(entity)
        return jsonify({"entity_id": entity_id, "thread_assigned": entity.thread_id})

    @app.get("/context/entity/<entity_id>")
    def get_entity(entity_id):
        entity, rels = graph.get_entity_with_relationships(entity_id)
        if not entity:
            return jsonify({"error": "not found"}), 404
        return jsonify({
            "entity": entity.to_dict(),
            "relationships": [r.to_dict() for r in rels],
        })

    @app.put("/context/entity/<entity_id>")
    def update_entity(entity_id):
        data = request.get_json()
        graph.update_entity(entity_id, **data)
        return jsonify({"entity_id": entity_id, "updated_fields": list(data.keys())})

    @app.get("/context/search")
    def search():
        q = request.args.get("q", "")
        if not q:
            return jsonify({"results": [], "count": 0})
        results = graph.search(q)
        return jsonify({
            "results": [e.to_dict() for e in results],
            "count": len(results),
        })

    # --- Thread CRUD ---

    @app.post("/context/thread")
    def create_thread():
        data = request.get_json()
        thread = Thread(
            title=data["title"],
            trigger_entity_id=data.get("trigger_entity_id"),
            status=data.get("status", "active"),
        )
        thread_id = graph.add_thread(thread)
        return jsonify({"thread_id": thread_id})

    @app.get("/context/threads")
    def list_threads():
        status = request.args.get("status")
        from context_engine.models import ThreadStatus
        status_filter = ThreadStatus(status) if status else None
        threads = graph.list_threads(status=status_filter)
        return jsonify({
            "threads": [t.to_dict() for t in threads],
            "count": len(threads),
        })

    @app.get("/context/thread/<thread_id>")
    def get_thread(thread_id):
        thread = graph.get_thread(thread_id)
        if not thread:
            return jsonify({"error": "not found"}), 404
        entities = graph.get_thread_entities(thread_id)
        return jsonify({
            "thread": thread.to_dict(),
            "entities": [e.to_dict() for e in entities],
        })

    @app.put("/context/thread/<thread_id>")
    def update_thread(thread_id):
        data = request.get_json()
        graph.update_thread(thread_id, **data)
        return jsonify({"thread_id": thread_id, "updated_status": data.get("status")})

    # --- Relationship CRUD ---

    @app.post("/context/relationship")
    def create_relationship():
        data = request.get_json()
        rel = Relationship(
            from_id=data["from_id"],
            to_id=data["to_id"],
            type=data["type"],
            confidence=data.get("confidence", 1.0),
        )
        rel_id = graph.add_relationship(rel)
        return jsonify({"relationship_id": rel_id})

    return app


def main():
    cfg = load_config()
    init_db(cfg.db_path)
    app = create_app()
    app.run(host=cfg.host, port=cfg.port, debug=False)


if __name__ == "__main__":
    main()
