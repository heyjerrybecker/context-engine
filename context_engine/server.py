"""Flask HTTP API for the Context Engine."""

import os

from flask import Flask, request, jsonify
from flask_caching import Cache

from context_engine.config import load_config
from context_engine.schema import init_db
from context_engine.graph import KnowledgeGraph
from context_engine.briefing import generate_briefing
from context_engine.models import Entity, Thread, Relationship
from context_engine.memory_lifecycle import MemoryLifecycle


def create_app(db_path: str = None, config_dir: str = None) -> Flask:
    app = Flask(__name__)

    from context_engine.tracing import init_tracing
    init_tracing()

    cfg = load_config(config_dir) if config_dir else load_config()
    db = db_path or cfg.db_path
    metrics_db = cfg.metrics_db_path
    init_db(db)
    graph = KnowledgeGraph(db)

    from context_engine.worker import start_worker as _start_worker
    _start_worker(db)

    lifecycle = MemoryLifecycle(db, {
        "agents": cfg.agents,
        "capacity_threshold": cfg.memory_capacity_threshold,
        "aging_days": cfg.memory_aging_days,
    })

    if cfg.redis_url:
        app.config["CACHE_TYPE"] = "RedisCache"
        app.config["CACHE_REDIS_URL"] = cfg.redis_url
    else:
        app.config["CACHE_TYPE"] = "SimpleCache"
    app.config["CACHE_DEFAULT_TIMEOUT"] = cfg.cache_default_timeout
    cache = Cache(app)

    @app.get("/context/health")
    def health():
        return jsonify({
            "status": "ok",
            "db": db,
            "port": cfg.port,
        })

    @app.get("/context/identity")
    @cache.cached(timeout=300)
    def identity():
        return jsonify({
            "soul": cfg.load_soul(),
            "user_model": cfg.load_user_model(),
        })

    @app.get("/context/briefing")
    @cache.cached(timeout=60)
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
        if agent in cfg.agents:
            lifecycle.sync_agent(agent)
        cache.clear()
        text = generate_briefing(graph, lifecycle=lifecycle)
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
        import threading
        from context_engine.metrics.classifier import classify_session_async
        data = request.get_json()
        session_id = data.get("session_id", "")
        summary = data.get("summary")
        graph.end_session(session_id, summary=summary)
        cache.clear()
        generate_briefing(graph)
        if session_id and session_id != "unknown":
            threading.Thread(
                target=classify_session_async,
                args=(session_id, db, metrics_db),
                daemon=True,
            ).start()
        return jsonify({"briefing_written": True, "summary": summary})

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
        cache.clear()
        return jsonify({"entity_id": entity_id, "thread_assigned": entity.thread_id})

    @app.get("/context/entity/<entity_id>")
    @cache.cached(timeout=60)
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
        cache.clear()
        return jsonify({"entity_id": entity_id, "updated_fields": list(data.keys())})

    @app.get("/context/search")
    @cache.cached(timeout=30, query_string=True)
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
        cache.clear()
        return jsonify({"thread_id": thread_id})

    @app.get("/context/threads")
    @cache.cached(timeout=30, query_string=True)
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
    @cache.cached(timeout=60)
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
        cache.clear()
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
        cache.clear()
        return jsonify({"relationship_id": rel_id})

    # --- Federation: recent entities for team Gaia ingestion ---

    @app.get("/context/entities/recent")
    def recent_entities():
        since = request.args.get("since", "")
        limit = min(int(request.args.get("limit", 50)), 200)
        entity_type = request.args.get("type")

        import sqlite3
        conn = sqlite3.connect(graph.db_path)
        conditions = ["1=1"]
        params = []
        if since:
            conditions.append("created_at > ?")
            params.append(since)
        if entity_type:
            conditions.append("type = ?")
            params.append(entity_type)
        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT entity_id, type, content, confidence, created_at, source_agent, source_session "
            f"FROM entities WHERE {where} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        conn.close()

        entities = [
            {"entity_id": r[0], "type": r[1], "content": r[2], "confidence": r[3],
             "created_at": r[4], "source_agent": r[5], "source_session": r[6]}
            for r in rows
        ]
        return jsonify({"entities": entities, "count": len(entities)})

    # --- Ingest (extraction from conversation turns) ---

    @app.post("/context/ingest")
    def ingest():
        from context_engine.worker import enqueue
        data = request.get_json()
        user_msg = data.get("user_message", "")
        assistant_msg = data.get("assistant_message", "")
        session_id = data.get("session_id", "")
        if user_msg and assistant_msg:
            enqueue(user_msg, assistant_msg, session_id)
        return jsonify({"queued": True}), 202

    # --- Token Observatory ---

    from context_engine.observatory import proxy_request, UsageLog
    usage_db = os.path.join(cfg.config_dir, "usage.db")
    usage_log = UsageLog(usage_db)

    @app.post("/v1/messages")
    def observatory_proxy():
        agent_id = request.headers.get("x-agent-id", "unknown")
        body, status, headers = proxy_request(
            request_data=request.get_data(),
            headers=dict(request.headers),
            backend_url=cfg.observatory_backend_url,
            usage_log=usage_log,
            agent_id=agent_id,
        )
        return app.response_class(body, status=status,
                                  content_type=headers.get("content-type", "application/json"))

    @app.post("/v1/projects/<project>/locations/<region>/publishers/anthropic/models/<model_spec>")
    def observatory_vertex_proxy(project, region, model_spec):
        agent_id = request.headers.get("x-agent-id", "unknown")
        real_url = (f"https://{region}-aiplatform.googleapis.com"
                    f"/v1/projects/{project}/locations/{region}"
                    f"/publishers/anthropic/models/{model_spec}")
        body, status, headers = proxy_request(
            request_data=request.get_data(),
            headers=dict(request.headers),
            backend_url=cfg.observatory_backend_url,
            usage_log=usage_log,
            agent_id=agent_id,
            target_url=real_url,
        )
        return app.response_class(body, status=status,
                                  content_type=headers.get("content-type", "application/json"))

    @app.get("/v1/usage")
    def observatory_usage():
        agent_id = request.args.get("agent_id")
        since = request.args.get("since")
        limit = int(request.args.get("limit", 100))
        rows = usage_log.query(agent_id=agent_id, since=since, limit=limit)
        return jsonify({"usage": rows, "count": len(rows)})

    @app.get("/v1/usage/summary")
    def observatory_summary():
        agent_id = request.args.get("agent_id")
        since = request.args.get("since")
        summary = usage_log.summary(agent_id=agent_id, since=since)
        return jsonify(summary)

    # --- Memory lifecycle ---

    @app.post("/context/memory/sync")
    def memory_sync():
        data = request.get_json()
        agent_id = data.get("agent_id", "")
        result = lifecycle.sync_agent(agent_id)
        if "error" in result:
            return jsonify(result), 404
        cache.clear()
        return jsonify(result)

    @app.post("/context/memory/archive")
    def memory_archive():
        data = request.get_json()
        agent_id = data.get("agent_id", "")
        entries = data.get("entries", [])
        result = lifecycle.archive_entries(agent_id, entries)
        return jsonify(result)

    @app.get("/context/memory/recall")
    def memory_recall():
        agent_id = request.args.get("agent_id", "")
        q = request.args.get("q", "")
        limit = min(int(request.args.get("limit", 10)), 50)
        results = lifecycle.recall(agent_id, q, limit)
        return jsonify({"results": results, "count": len(results)})

    @app.get("/context/memory/health")
    def memory_health():
        agent_id = request.args.get("agent_id", "")
        health = lifecycle.get_health(agent_id)
        if not health:
            return jsonify({"error": f"No data for agent: {agent_id}"}), 404
        return jsonify(health)

    return app


def main():
    cfg = load_config()
    init_db(cfg.db_path)
    app = create_app()
    app.run(host=cfg.host, port=cfg.port, debug=False)


if __name__ == "__main__":
    main()
