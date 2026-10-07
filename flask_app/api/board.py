"""Board section API (Phase 0). Every path is behind the Board section, which is
OPT-IN (``auth/sections.py``): the section gate refuses a user who has not been
granted it before any route here runs. Inside, what may be DONE is a permission
by username (``auth/permissions.py``), checked per route:

    read anything here ............ the Board section (schedule views included)
    meeting schedules, narratives . board_edit
    create a meeting, rename it ... board_build
    grants and the access log ..... the admin USERNAME only

The admin ROLE confers none of it -- the reason none of these use role_required.
"""
from __future__ import annotations

from functools import wraps

from flask import Blueprint, g, jsonify, request

from flask_app.auth import audit, permissions
from flask_app.auth.routes import login_required
from flask_app.auth.sections import (SUPERUSER, denied_sections, section_grant_ends,
                                     set_user_sections)
from flask_app.services import board_service as svc

board_bp = Blueprint("board", __name__, url_prefix="/api/board")


def _actor() -> dict:
    return getattr(g, "current_user", None) or {}


def need(perm: str):
    def deco(f):
        @wraps(f)
        def inner(*a, **k):
            if not permissions.has(_actor(), perm):
                label = next(p["label"] for p in permissions.PERMISSIONS if p["key"] == perm)
                return jsonify({"error": "Forbidden",
                                "message": "This needs the %s permission, which the "
                                           "'%s' account grants." % (label, SUPERUSER)}), 403
            return f(*a, **k)
        return inner
    return deco


def superuser_only(f):
    @wraps(f)
    def inner(*a, **k):
        if _actor().get("username") != SUPERUSER:
            return jsonify({"error": "Forbidden",
                            "message": "Only the '%s' account manages Board access." % SUPERUSER}), 403
        return f(*a, **k)
    return inner


def _bad(e: Exception, status=400):
    return jsonify({"error": str(e)}), status


@board_bp.route("/me", methods=["GET"])
@login_required
def me():
    u = _actor()
    return jsonify({"username": u.get("username"),
                    "permissions": sorted(permissions.effective(u)),
                    "manages_access": u.get("username") == SUPERUSER})


@board_bp.route("/catalog", methods=["GET"])
@login_required
def catalog():
    return jsonify({"schedules": list(svc.SCHEDULES), "narratives": list(svc.NARRATIVES),
                    "permissions": [{k: v for k, v in p.items() if k != "implies"} | {"implies": list(p["implies"])}
                                    for p in permissions.PERMISSIONS]})


@board_bp.route("/meetings", methods=["GET"])
@login_required
def meetings():
    return jsonify({"meetings": svc.list_meetings()})


@board_bp.route("/meetings", methods=["POST"])
@login_required
@need("board_build")
def create_meeting():
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(svc.create_meeting(b.get("title"), b.get("meeting_date"),
                                          b.get("default_as_of"), _actor()["username"])), 201
    except ValueError as e:
        return _bad(e)


@board_bp.route("/meetings/<int:mid>", methods=["GET"])
@login_required
def meeting(mid):
    m = svc.get_meeting(mid)
    return jsonify(m) if m else (jsonify({"error": "Meeting not found"}), 404)


@board_bp.route("/meetings/<int:mid>/schedules/<key>/view", methods=["GET"])
@login_required
def schedule_view(mid, key):
    """One schedule's figures, at the as-of date THIS MEETING carries for it.

    Read-only, so the Board section is the whole gate (like reading the meeting).
    Every figure comes from an engine the app already owns -- see
    ``services/board_views_service.py``; a schedule whose phase has not built
    its view yet answers 404 with that said.
    """
    from flask_app.serializers import safe_json
    from flask_app.services import board_views_service as views

    m = svc.get_meeting(mid)
    if not m:
        return jsonify({"error": "Meeting not found"}), 404
    s = next((x for x in m["schedules"] if x["key"] == key), None)
    if s is None:
        return jsonify({"error": "Unknown schedule %r" % key}), 404
    if key not in views.VIEW_KEYS:
        return jsonify({"error": "The %s schedule has no view yet (phase %s)." % (s["title"], s["phase"])}), 404
    as_of = svc.parse_date(s["as_of"], "As-of date")
    out = views.build_view(key, as_of)
    return jsonify(safe_json({**out, "schedule": {k: s[k] for k in ("key", "title", "pages", "as_of")}}))


@board_bp.route("/meetings/<int:mid>", methods=["PUT"])
@login_required
@need("board_build")
def update_meeting(mid):
    b = request.get_json(silent=True) or {}
    fields = {k: b[k] for k in ("title", "meeting_date", "default_as_of") if k in b}
    try:
        return jsonify(svc.update_meeting(mid, fields, _actor()["username"]))
    except LookupError as e:
        return _bad(e, 404)
    except PermissionError as e:
        return _bad(e, 409)
    except ValueError as e:
        return _bad(e)


@board_bp.route("/meetings/<int:mid>/schedules", methods=["PUT"])
@login_required
@need("board_edit")
def put_schedules(mid):
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(svc.set_schedules(mid, b.get("schedules") or [], _actor()["username"]))
    except LookupError as e:
        return _bad(e, 404)
    except PermissionError as e:
        return _bad(e, 409)
    except ValueError as e:
        return _bad(e)


@board_bp.route("/meetings/<int:mid>/narratives/<key>", methods=["PUT"])
@login_required
@need("board_edit")
def put_narrative(mid, key):
    b = request.get_json(silent=True) or {}
    try:
        return jsonify(svc.save_narrative(mid, key, b.get("body") or "", _actor()["username"]))
    except LookupError as e:
        return _bad(e, 404)
    except PermissionError as e:
        return _bad(e, 409)
    except ValueError as e:
        return _bad(e)


# ── Access: the admin username only ──────────────────────────────────

@board_bp.route("/access", methods=["GET"])
@login_required
@superuser_only
def access():
    from flask_app.auth.models import list_users
    grants = permissions.all_stored()
    ends = section_grant_ends()
    users = []
    for u in list_users():
        if u["username"] == SUPERUSER:
            continue
        users.append({"id": u["id"], "username": u["username"], "role": u["role"],
                      "email": u.get("email"),
                      "board": "board" not in denied_sections(u["id"]),
                      "board_until": ends.get(u["id"], {}).get("board"),
                      "permissions": sorted(grants.get(u["id"], []))})
    return jsonify({"users": users, "permissions": [
        {"key": p["key"], "label": p["label"], "describe": p["describe"],
         "implies": list(p["implies"])} for p in permissions.PERMISSIONS]})


@board_bp.route("/access/<int:user_id>", methods=["PUT"])
@login_required
@superuser_only
def put_access(user_id):
    """Body: {board?: bool, board_until?: 'YYYY-MM-DD'|null, permissions?: {key: bool}}"""
    from flask_app.auth.models import get_user_by_id
    target = get_user_by_id(user_id)
    if target is None:
        return jsonify({"error": "User not found"}), 404
    if target["username"] == SUPERUSER:
        return jsonify({"error": "The '%s' account always has everything." % SUPERUSER}), 400
    b = request.get_json(silent=True) or {}
    actor = _actor()["username"]
    try:
        if "board" in b or "board_until" in b:
            on = bool(b.get("board", "board" not in denied_sections(user_id)))
            set_user_sections(user_id, {"board": on}, actor,
                              expires={"board": b.get("board_until")} if on else None)
        if b.get("permissions"):
            permissions.set_permissions(user_id, {k: bool(v) for k, v in b["permissions"].items()},
                                        actor)
    except ValueError as e:
        return _bad(e)
    return jsonify({"user_id": user_id,
                    "board": "board" not in denied_sections(user_id),
                    "board_until": section_grant_ends().get(user_id, {}).get("board"),
                    "permissions": sorted(permissions.stored(user_id))})


@board_bp.route("/audit", methods=["GET"])
@login_required
@superuser_only
def access_log():
    return jsonify({"entries": audit.recent(int(request.args.get("limit", 500)))})
