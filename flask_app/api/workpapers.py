"""Accounting workpaper package endpoints.

Cycles and deadlines are the CFO's; packages, steps and exhibits are the
preparer's; the approvals are the chain between them. Read access is open to
any signed-in user -- accounting needs to see where the close stands without
being able to change it -- and every write records who did it.
"""
import logging

from flask import Blueprint, g, jsonify, request, send_file
from io import BytesIO

from flask_app.auth.routes import (ACCOUNTING_ROLES, CLOSE_CYCLE_ROLES,
                                   login_required, roles_exactly)
from flask_app.db import get_engine
from flask_app.serializers import safe_json
from flask_app.services import workpaper_service as ws
from flask_app.services import workpaper_data as wd
from flask_app.services import workpaper_excel as wx
from flask_app.services import workpaper_tracker as wt

# THE ACCOUNTING SECTION BELONGS TO ACCOUNTING. Jim, Sep 17 2026: "Only the
# accountants, accounting manager, and cfo should be able to edit anything in
# the accounting section of the app generally. Me as admin, can edit only so I
# can help them get something fixed while we are building and testing the
# model."
#
# SO THE GATE IS MEMBERSHIP, NOT LEVEL. `role_required` compares LEVELS, and
# analyst, accountant, accounting_manager and cfo are all level 1 -- a level
# comparison naming any one of them admits all four. That was fine while the
# job was GRANTING the CFO access (Sep 16); it cannot express "not an analyst",
# which is what is wanted now, and Jim's own day-to-day login is an analyst one
# he wants read-only here. `roles_exactly` checks the name.
#
# Reads stay open to every signed-in user: accounting has to be visible to the
# people who depend on it without being editable by them.

logger = logging.getLogger(__name__)

workpapers_bp = Blueprint("workpapers", __name__, url_prefix="/api/workpapers")

MAX_EXHIBIT_BYTES = 25 * 1024 * 1024


def _user() -> str:
    return (getattr(g, "current_user", None) or {}).get("username", "unknown")


def _app_role() -> str:
    return (getattr(g, "current_user", None) or {}).get("role", "")


def _fail(e: Exception, what: str, code: int = 400):
    logger.error(f"{what} failed: {e}", exc_info=True)
    return jsonify({"error": str(e)[:300]}), code


# ── Cycles ───────────────────────────────────────────────────────────────

@workpapers_bp.route("/cycles", methods=["GET"])
@login_required
def list_cycles():
    try:
        return jsonify({"cycles": ws.list_cycles(), "steps": ws.STEP_TEMPLATE,
                        "slots": ws.EXHIBIT_SLOTS})
    except Exception as e:
        return _fail(e, "list_cycles", 500)


@workpapers_bp.route("/cycles", methods=["POST"])
@login_required
@roles_exactly(*CLOSE_CYCLE_ROLES)
def create_cycle():
    # NARROWER THAN THE REST OF THE SECTION, on purpose. Jim, Sep 17 2026:
    # "starting a close cycle should belong to the CFO, anyone on the
    # accounting team can sync entities." Opening a cycle decides the period
    # the firm reports on; syncing entities into one is preparation.
    body = request.get_json(silent=True) or {}
    label = (body.get("period_label") or "").strip()
    end = (body.get("period_end") or "").strip()
    if not label or not end:
        return jsonify({"error": "period_label and period_end are required"}), 400
    try:
        return jsonify(ws.create_cycle(label, end, _user())), 201
    except Exception as e:
        return _fail(e, "create_cycle")


@workpapers_bp.route("/cycles/<int:cycle_id>/steps", methods=["PUT"])
@login_required
@roles_exactly(*CLOSE_CYCLE_ROLES)
def set_due_date(cycle_id):
    """The CFO's deadline for one step of the close.

    NO SCREEN CALLS THIS as of Sep 17 2026 -- the two-tab split replaced the
    per-step deadline UI with the tracker's per-deliverable target dates, and
    `setDue()` in WorkpapersView.vue is left over and unreferenced. Gated with
    the target date anyway: it is the same decision, it is still reachable, and
    a rule that covers only the endpoint that happens to have a button is a
    rule that breaks the next time somebody adds one.
    """
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ws.set_step_due_date(
            cycle_id, body.get("step_key"), body.get("due_date"), _user()))
    except Exception as e:
        return _fail(e, "set_due_date")


@workpapers_bp.route("/cycles/<int:cycle_id>/sync", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def sync(cycle_id):
    """Create packages for any REP entity that has none in this cycle."""
    try:
        return jsonify(ws.sync_packages(cycle_id))
    except Exception as e:
        return _fail(e, "sync_packages")


@workpapers_bp.route("/cycles/<int:cycle_id>/tracker", methods=["GET"])
@login_required
def tracker(cycle_id):
    try:
        return jsonify(safe_json(ws.tracker(cycle_id)))
    except Exception as e:
        return _fail(e, "tracker", 500)


@workpapers_bp.route("/statements/batch", methods=["POST"])
@login_required
def statements_batch():
    """Every statement for one entity or for many, in a single request.

    THE SHAPE IS THE ONE PAGER'S, deliberately (Jim, Sep 16 2026: "we have a pdf
    reporting function within the one pager that handles individual and
    batches"). The server assembles the whole batch and the client stacks the
    pages with a page break between them, so one print produces one PDF whether
    it is one entity or fifty-eight.

    PER-ENTITY `error`, WHICH IS THE PART THAT MATTERS AT THIS SCALE. One entity
    whose statements fail to build must not take the other fifty-seven with it;
    it comes back as a page carrying its reason, and the batch still prints.

    Body: { entity_ids: [...], period_end: "2026-06-30", cycle_id?: n }
    Passing `cycle_id` and no `entity_ids` prints the whole cycle in the CFO's
    own tracker order, which is the order the close is worked in.
    """
    from flask_app.services import statement_service as ss
    body = request.get_json(silent=True) or {}
    period_end = (body.get("period_end") or "").strip()
    ids = [str(x).strip().upper() for x in (body.get("entity_ids") or []) if str(x).strip()]
    cycle_id = body.get("cycle_id")

    if cycle_id and not ids:
        try:
            g = wt.grid(int(cycle_id))
            if g.get("error"):
                return jsonify({"error": g["error"]}), 400
            ids = [r["entityid"] for r in g["rows"]]
            period_end = period_end or (g.get("cycle") or {}).get("period_end") or ""
        except Exception as e:
            return _fail(e, "statements_batch cycle", 400)

    if not ids:
        return jsonify({"error": "entity_ids or cycle_id is required"}), 400
    if not period_end:
        return jsonify({"error": "period_end is required"}), 400

    engine = get_engine()
    names = {}
    try:
        import pandas as pd
        import sqlalchemy as sa
        with engine.connect() as conn:
            ent = pd.read_sql(sa.text("SELECT * FROM entities"), conn)
        em = {str(c).lower(): c for c in ent.columns}
        if em.get("entityid") and em.get("name"):
            names = {str(r[em["entityid"]]).strip().upper(): str(r[em["name"]]).strip()
                     for _, r in ent.iterrows()}
    except Exception:
        names = {}

    pages = []
    for eid in ids:
        page = {"entityid": eid, "name": names.get(eid) or eid}
        try:
            both = ss.build(eid, period_end, "both", engine=engine)
            page["balance_sheet"] = both.get("balance_sheet")
            page["income_statement"] = both.get("income_statement")
            page["note"] = both.get("note")
            page["members_capital"] = ss.build_members_capital(
                eid, period_end, engine=engine)
            page["cash_flow"] = ss.build_cash_flow(eid, period_end, engine=engine)
            page["soi"] = ss.build_schedule_of_investments(
                eid, period_end, engine=engine)
        except Exception as e:
            # Recorded on the page, not raised. See the docstring.
            logger.exception("statements_batch failed for %s", eid)
            page["error"] = str(e)[:300]
        pages.append(page)

    return jsonify(safe_json({
        "period_end": period_end,
        "count": len(pages),
        "failed": sum(1 for p in pages if p.get("error")),
        "pages": pages,
    }))


@workpapers_bp.route("/preparers", methods=["GET"])
@login_required
def preparers():
    """Who can be assigned. See `wp_preparers` for why this is not `users`."""
    try:
        return jsonify(safe_json({"preparers": wt.list_preparers()}))
    except Exception as e:
        return _fail(e, "preparers", 500)


@workpapers_bp.route("/preparers", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def add_preparer():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.add_preparer(
            body.get("initials"), body.get("name"),
            body.get("wp_role"), body.get("username")))
    except Exception as e:
        return _fail(e, "add_preparer")


@workpapers_bp.route("/preparers/<initials>", methods=["DELETE"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def drop_preparer(initials):
    try:
        return jsonify(wt.remove_preparer(initials))
    except Exception as e:
        return _fail(e, "remove_preparer")


@workpapers_bp.route("/cycles/<int:cycle_id>/schedule/properties",
                     methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_properties(cycle_id):
    """Fill the Property column from the deal each entity holds.

    Never overwrites a typed value unless asked: a value the CFO typed is a
    decision and a derived one is a guess.
    """
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(safe_json(wt.apply_derived_properties(
            cycle_id, bool(body.get("overwrite")))))
    except Exception as e:
        return _fail(e, "schedule_properties")


# -- The close schedule: the CFO's tracker across every entity ------------
#
# Distinct from `/tracker` above, which is the per-package STEP checklist. This
# is the grid the CFO keeps in the reporting calendar workbook: one row per
# reporting entity, his order, his target dates, and a sign-off per stage.
#
# Reads are open to any signed-in user; accounting has to be able to see where
# the close stands. Writes divide in two:
#   * ORDER, TARGET DATES, RENUMBER and CARRY FORWARD are the CFO's arrangement
#     of the quarter, gated like the other cycle-level settings above.
#   * A SIGN-OFF is open to any signed-in user, because the preparer, the
#     manager and the CFO each record their own, and every one stores the name
#     typed into it as well as the account that saved it.

@workpapers_bp.route("/cycles/<int:cycle_id>/schedule", methods=["GET"])
@login_required
def schedule(cycle_id):
    try:
        return jsonify(safe_json(wt.grid(cycle_id)))
    except Exception as e:
        return _fail(e, "schedule", 500)


@workpapers_bp.route("/packages/<int:package_id>/schedule/order", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_order(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.set_order(package_id, body.get("order"), _user()))
    except Exception as e:
        return _fail(e, "schedule_order")


@workpapers_bp.route("/packages/<int:package_id>/schedule/target", methods=["PUT"])
@login_required
@roles_exactly(*CLOSE_CYCLE_ROLES)
def schedule_target(package_id):
    # THE DEADLINE ON THE TRACKER, and the CFO's alone (Jim, Sep 17 2026:
    # "deadlines should be CFO only too"). The sign-offs beside it are the
    # team's: they record what was done against the date, they do not set it.
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.set_target(package_id, body.get("deliverable", ""),
                                     body.get("target_date"), _user()))
    except Exception as e:
        return _fail(e, "schedule_target")


@workpapers_bp.route("/packages/<int:package_id>/schedule/signoff", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_signoff(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.set_signoff(
            package_id, body.get("deliverable", ""), body.get("stage", ""),
            body.get("signed_by"), body.get("signed_on"),
            body.get("note"), _user()))
    except Exception as e:
        return _fail(e, "schedule_signoff")


@workpapers_bp.route("/packages/<int:package_id>/schedule/preparer", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_preparer(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.set_preparer(package_id, body.get("preparer"), _user()))
    except Exception as e:
        return _fail(e, "schedule_preparer")


@workpapers_bp.route("/packages/<int:package_id>/schedule/property", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_property(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.set_property(package_id, body.get("property_name"),
                                       _user()))
    except Exception as e:
        return _fail(e, "schedule_property")


@workpapers_bp.route("/cycles/<int:cycle_id>/schedule/renumber", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_renumber(cycle_id):
    try:
        return jsonify(wt.renumber(cycle_id, _user()))
    except Exception as e:
        return _fail(e, "schedule_renumber")


@workpapers_bp.route("/cycles/<int:cycle_id>/schedule/carry-forward",
                     methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def schedule_carry_forward(cycle_id):
    """Bring the PREVIOUS quarter's order, preparers and properties forward.

    Never the sign-offs or the target dates -- those are facts about a quarter
    and copying them would assert work that has not been done.
    """
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(wt.carry_forward(int(body.get("from_cycle_id") or 0),
                                        cycle_id, _user()))
    except Exception as e:
        return _fail(e, "schedule_carry_forward")


# ── Packages ─────────────────────────────────────────────────────────────

@workpapers_bp.route("/packages/<int:package_id>", methods=["GET"])
@login_required
def package(package_id):
    try:
        return jsonify(safe_json(ws.package_detail(package_id)))
    except Exception as e:
        return _fail(e, "package_detail", 404)


@workpapers_bp.route("/packages/<int:package_id>/steps", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def set_step(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ws.set_step_done(
            package_id, body.get("step_key"), bool(body.get("done")),
            _user(), body.get("note", "")))
    except Exception as e:
        return _fail(e, "set_step")


@workpapers_bp.route("/packages/<int:package_id>/transition", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
# The approval CHAIN still checks who may take this action (ws.transition
# raises PermissionError). This gate is the outer one: you must be in
# accounting at all before the chain is asked which step is yours.
def transition(package_id):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ws.transition(package_id, body.get("action", ""), _user(),
                                     _app_role(), body.get("note", "")))
    except PermissionError as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return _fail(e, "transition")


@workpapers_bp.route("/packages/<int:package_id>/assign", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def assign(package_id):
    body = request.get_json(silent=True) or {}
    import sqlalchemy as sa
    try:
        with get_engine().begin() as conn:
            conn.execute(sa.text(
                "UPDATE wp_packages SET preparer = :p, reviewer = :r, notes = :n, "
                "updated_at = :t WHERE id = :i"),
                {"p": body.get("preparer"), "r": body.get("reviewer"),
                 "n": body.get("notes"), "t": ws._now(), "i": package_id})
        return jsonify({"status": "ok"})
    except Exception as e:
        return _fail(e, "assign")


# ── Exhibits ─────────────────────────────────────────────────────────────

@workpapers_bp.route("/packages/<int:package_id>/exhibits", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def upload_exhibit(package_id):
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    content = f.read()
    if not content:
        return jsonify({"error": "Empty file"}), 400
    if len(content) > MAX_EXHIBIT_BYTES:
        return jsonify({"error": f"File exceeds {MAX_EXHIBIT_BYTES // (1024*1024)}MB"}), 400
    try:
        return jsonify(ws.add_exhibit(
            package_id, request.form.get("slot_key", "other"), f.filename,
            content, f.content_type or "application/octet-stream",
            request.form.get("caption", ""), _user())), 201
    except Exception as e:
        return _fail(e, "upload_exhibit")


@workpapers_bp.route("/exhibits/<int:exhibit_id>", methods=["GET"])
@login_required
def download_exhibit(exhibit_id):
    try:
        e = ws.get_exhibit(exhibit_id)
        if not e:
            return jsonify({"error": "Not found"}), 404
        return send_file(BytesIO(e["content"]), as_attachment=True,
                         download_name=e["filename"],
                         mimetype=e.get("content_type") or "application/octet-stream")
    except Exception as e:
        return _fail(e, "download_exhibit", 500)


@workpapers_bp.route("/exhibits/<int:exhibit_id>", methods=["DELETE"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def delete_exhibit(exhibit_id):
    try:
        return jsonify(ws.delete_exhibit(exhibit_id, _user()))
    except Exception as e:
        return _fail(e, "delete_exhibit")


# ── The package itself ───────────────────────────────────────────────────

@workpapers_bp.route("/packages/<int:package_id>/download", methods=["GET"])
@login_required
def download_package(package_id):
    try:
        detail = ws.package_detail(package_id)
        pkg = detail["package"]
        content = wx.build_package(package_id)
        name = f"{pkg['entityid']} - WP - {pkg['period_end']}.xlsx"
        return send_file(BytesIO(content), as_attachment=True, download_name=name,
                         mimetype="application/vnd.openxmlformats-officedocument."
                                  "spreadsheetml.sheet")
    except Exception as e:
        return _fail(e, "download_package", 500)


@workpapers_bp.route("/packages/<int:package_id>/statements", methods=["GET"])
@login_required
def package_statements(package_id):
    """Every drafted statement with its tie-out — the thing being validated."""
    from flask_app.services import workpaper_workbench as wbench
    try:
        return jsonify(safe_json(wbench.statements_summary(package_id)))
    except Exception as e:
        return _fail(e, "package_statements", 500)


@workpapers_bp.route("/packages/<int:package_id>/steps/<step_key>/evidence", methods=["GET"])
@login_required
def step_evidence(package_id, step_key):
    """What the preparer needs in front of them to finish this step."""
    from flask_app.services import workpaper_workbench as wbench
    try:
        return jsonify(safe_json(wbench.step_evidence(package_id, step_key)))
    except Exception as e:
        return _fail(e, "step_evidence", 500)


@workpapers_bp.route("/packages/<int:package_id>/preview", methods=["GET"])
@login_required
def preview(package_id):
    """What the package contains, without building the workbook — so the
    preparer can see the figures and the unmapped accounts on screen."""
    try:
        pkg = ws.package_detail(package_id)["package"]
        ent, pe = pkg["entityid"], pkg["period_end"]
        tb = wd.trial_balance(ent, pe)
        return jsonify(safe_json({
            "entity": ent, "period_end": pe,
            "periods": tb["periods"],
            "trial_balance": tb["rows"][:500],
            "trial_balance_rows": len(tb["rows"]),
            "unmapped_accounts": tb["unmapped"],
            "statements": __import__("flask_app.services.statement_service",
                fromlist=["build"]).build(ent, pe),
            "ia_rollforward": wd.ia_rollforward(ent, pe),
        }))
    except Exception as e:
        return _fail(e, "preview", 500)


# ── Account -> FS line mapping ───────────────────────────────────────────

@workpapers_bp.route("/fs-map", methods=["GET"])
@login_required
def get_fs_map():
    """The mapping, plus every account the GL actually uses, so the screen can
    show what is still untagged rather than only what is."""
    try:
        import pandas as pd
        import sqlalchemy as sa
        with get_engine().connect() as conn:
            try:
                accts = pd.read_sql(sa.text(
                    'SELECT DISTINCT "ACCTNUM", "ACCTNAME" FROM gl_detail '
                    'ORDER BY "ACCTNUM"'), conn).to_dict("records")
            except Exception:
                accts = []
        return jsonify({"mapping": ws.get_fs_map(), "accounts": accts})
    except Exception as e:
        return _fail(e, "get_fs_map", 500)


@workpapers_bp.route("/fs-map", methods=["PUT"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def put_fs_map():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ws.set_fs_map(body.get("entries") or [], _user()))
    except Exception as e:
        return _fail(e, "set_fs_map")


# ── Financial statements ─────────────────────────────────────────────────
# Deliberately keyed by entity + period rather than by package: the same
# engine serves a workpaper tab, a standalone entity statement, and whatever
# consolidation comes later. A package is one caller, not the owner.

@workpapers_bp.route("/statements", methods=["GET"])
@login_required
def statements():
    from flask_app.services import statement_service as ss
    entity = (request.args.get("entity") or "").strip()
    period_end = (request.args.get("period_end") or "").strip()
    if not entity or not period_end:
        return jsonify({"error": "entity and period_end are required"}), 400
    which = request.args.get("statement", "both")
    bases = [b for b in (request.args.get("bases") or "").split(",") if b] or None
    try:
        return jsonify(safe_json(ss.build(entity, period_end, which, bases)))
    except Exception as e:
        return _fail(e, "statements", 500)


@workpapers_bp.route("/schedule-of-investments", methods=["GET"])
@login_required
def schedule_of_investments():
    from flask_app.services import statement_service as ss
    entity = (request.args.get("entity") or "").strip()
    period_end = (request.args.get("period_end") or "").strip()
    if not entity or not period_end:
        return jsonify({"error": "entity and period_end are required"}), 400
    try:
        return jsonify(safe_json(ss.build_schedule_of_investments(entity, period_end)))
    except Exception as e:
        return _fail(e, "schedule_of_investments", 500)


@workpapers_bp.route("/members-capital", methods=["GET"])
@login_required
def members_capital():
    from flask_app.services import statement_service as ss
    entity = (request.args.get("entity") or "").strip()
    period_end = (request.args.get("period_end") or "").strip()
    if not entity or not period_end:
        return jsonify({"error": "entity and period_end are required"}), 400
    try:
        return jsonify(safe_json(ss.build_members_capital(entity, period_end)))
    except Exception as e:
        return _fail(e, "members_capital", 500)


@workpapers_bp.route("/cash-flow", methods=["GET"])
@login_required
def cash_flow():
    from flask_app.services import statement_service as ss
    entity = (request.args.get("entity") or "").strip()
    period_end = (request.args.get("period_end") or "").strip()
    if not entity or not period_end:
        return jsonify({"error": "entity and period_end are required"}), 400
    try:
        return jsonify(safe_json(ss.build_cash_flow(entity, period_end)))
    except Exception as e:
        return _fail(e, "cash_flow", 500)


@workpapers_bp.route("/fs-map/consolidated", methods=["GET"])
@login_required
def consolidated_mapping():
    """Every account mapped into accounting's own 56 statement lines."""
    from flask_app.services import statement_service as ss
    import collections
    try:
        rows = ss.consolidated_mapping()
        return jsonify({
            "mapping": rows,
            "line_count": len({r["fs_line"] for r in rows}),
            "by_origin": dict(collections.Counter(r["origin"] for r in rows)),
            "by_section": dict(collections.Counter(r["statement"] for r in rows)),
        })
    except Exception as e:
        return _fail(e, "consolidated_mapping", 500)


@workpapers_bp.route("/fs-map/suggest", methods=["GET"])
@login_required
def suggest_mapping():
    """Proposed account -> line mapping from each account's own name.
    Suggestions only — nothing is applied until someone saves it."""
    from flask_app.services import statement_service as ss
    try:
        return jsonify({"suggestions": ss.seed_mapping_from_names()})
    except Exception as e:
        return _fail(e, "suggest_mapping", 500)


@workpapers_bp.route("/roles", methods=["GET"])
@login_required
def roles():
    try:
        return jsonify({"roles": ws.get_roles()})
    except Exception as e:
        return _fail(e, "roles", 500)
