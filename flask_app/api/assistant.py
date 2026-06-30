"""AI Assistant API — streaming chat with Claude."""

import json
import logging

from flask import Blueprint, request, jsonify, Response, stream_with_context, g

from flask_app.auth.routes import login_required

assistant_bp = Blueprint("assistant", __name__)
logger = logging.getLogger(__name__)


def _ensure_chat_table():
    """Create chat_history table if it doesn't exist."""
    from flask_app.db import get_engine
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            __import__("sqlalchemy").text(
                "CREATE TABLE IF NOT EXISTS chat_history ("
                "  user_id INTEGER PRIMARY KEY,"
                "  messages TEXT NOT NULL,"
                "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
        )
        conn.commit()


@assistant_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """Chat with the AI assistant. Streams SSE events.

    Request body: { "messages": [{"role": "user", "content": "..."}, ...] }
    Response: Server-Sent Events stream with JSON payloads.
    """
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    page_context = body.get("page_context", {})

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Validate message format
    for msg in messages:
        if msg.get("role") not in ("user", "assistant"):
            return jsonify({"error": "Messages must have role 'user' or 'assistant'"}), 400
        if not msg.get("content"):
            return jsonify({"error": "Messages must have content"}), 400

    try:
        from flask_app.services.assistant_service import chat_completion

        def generate():
            try:
                for event in chat_completion(messages, stream=True, page_context=page_context):
                    yield f"data: {json.dumps(event)}\n\n"
            except ValueError as e:
                # Missing API key
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            except Exception as e:
                logger.exception("Assistant chat error")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {str(e)}'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except Exception as e:
        logger.exception("Assistant endpoint error")
        return jsonify({"error": str(e)}), 500


@assistant_bp.route("/status", methods=["GET"])
@login_required
def status():
    """Check if the AI assistant is configured (API key present)."""
    import os
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    return jsonify({"available": has_key})


@assistant_bp.route("/history", methods=["GET"])
@login_required
def get_history():
    """Load saved chat history for the current user."""
    try:
        import sqlalchemy as sa
        _ensure_chat_table()
        user_id = g.current_user["id"]
        from flask_app.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            row = conn.execute(
                sa.text("SELECT messages FROM chat_history WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchone()
        if row:
            return jsonify({"messages": json.loads(row[0])})
        return jsonify({"messages": []})
    except Exception as e:
        logger.exception("Error loading chat history")
        return jsonify({"messages": []})


@assistant_bp.route("/history", methods=["PUT"])
@login_required
def save_history():
    """Save chat history for the current user."""
    try:
        import sqlalchemy as sa
        _ensure_chat_table()
        user_id = g.current_user["id"]
        body = request.get_json(force=True)
        messages = body.get("messages", [])
        messages_json = json.dumps(messages)

        from flask_app.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            # Upsert
            existing = conn.execute(
                sa.text("SELECT 1 FROM chat_history WHERE user_id = :uid"),
                {"uid": user_id},
            ).fetchone()
            if existing:
                conn.execute(
                    sa.text("UPDATE chat_history SET messages = :msgs, updated_at = CURRENT_TIMESTAMP WHERE user_id = :uid"),
                    {"msgs": messages_json, "uid": user_id},
                )
            else:
                conn.execute(
                    sa.text("INSERT INTO chat_history (user_id, messages) VALUES (:uid, :msgs)"),
                    {"msgs": messages_json, "uid": user_id},
                )
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Error saving chat history")
        return jsonify({"error": str(e)}), 500


@assistant_bp.route("/history", methods=["DELETE"])
@login_required
def clear_history():
    """Clear chat history for the current user."""
    try:
        import sqlalchemy as sa
        _ensure_chat_table()
        user_id = g.current_user["id"]
        from flask_app.db import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                sa.text("DELETE FROM chat_history WHERE user_id = :uid"),
                {"uid": user_id},
            )
            conn.commit()
        return jsonify({"ok": True})
    except Exception as e:
        logger.exception("Error clearing chat history")
        return jsonify({"error": str(e)}), 500
