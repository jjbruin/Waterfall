"""AI Assistant API — streaming chat with Claude."""

import json
import logging

from flask import Blueprint, request, jsonify, Response, stream_with_context

from flask_app.auth.routes import login_required

assistant_bp = Blueprint("assistant", __name__)
logger = logging.getLogger(__name__)


@assistant_bp.route("/chat", methods=["POST"])
@login_required
def chat():
    """Chat with the AI assistant. Streams SSE events.

    Request body: { "messages": [{"role": "user", "content": "..."}, ...] }
    Response: Server-Sent Events stream with JSON payloads.
    """
    body = request.get_json(force=True)
    messages = body.get("messages", [])

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
                for event in chat_completion(messages, stream=True):
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
