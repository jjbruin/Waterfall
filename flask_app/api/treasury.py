"""Treasury endpoints: bank accounts, imported activity, and reconciliation.

Reads are open to any signed-in user -- an accountant has to be able to see
where a reconciliation stands. Writes that change the ARRANGEMENT (which entity
an account belongs to, which cash account it posts to, closing a period) are
gated like the rest of the accounting section, which is the CFO's.

IMPORTS ARE OPEN TO EVERY ACCOUNTING ROLE, not just the CFO, because importing
a bank export is the accountant's daily work and it is not destructive:
activity is identified by a row hash, so re-importing the same file adds
nothing. A viewer is still refused -- read-only should mean read-only.

A NOTE ON WHAT THIS GATE CAN EXPRESS. `role_required` compares LEVELS, and
analyst, accountant, accounting_manager and cfo are all level 1. Naming
("admin", "cfo", "analyst") therefore admits every accounting role and excludes
viewers. It cannot single the CFO out, and no arrangement of names would make
it; separating him would take a change to the role model itself.
"""
import io
import logging

from flask import Blueprint, g, jsonify, request

from flask_app.auth.routes import login_required, role_required
from flask_app.serializers import safe_json
from flask_app.services import treasury_service as ts

logger = logging.getLogger(__name__)

treasury_bp = Blueprint("treasury", __name__, url_prefix="/api/treasury")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _user() -> str:
    return (getattr(g, "current_user", None) or {}).get("username", "unknown")


def _fail(e: Exception, what: str, code: int = 400):
    logger.error("%s failed: %s", what, e, exc_info=True)
    return jsonify({"error": str(e)[:300]}), code


@treasury_bp.route("/accounts", methods=["GET"])
@login_required
def accounts():
    try:
        return jsonify(safe_json({
            "accounts": ts.accounts(),
            "cash_accounts": list(ts.CASH_ACCOUNTS),
            "default_cash_account": ts.DEFAULT_CASH_ACCOUNT,
        }))
    except Exception as e:
        return _fail(e, "accounts", 500)


@treasury_bp.route("/accounts/<account_number>", methods=["PUT"])
@login_required
@role_required("admin", "cfo", "analyst")
def update_account(account_number):
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ts.set_account(
            account_number,
            entityid=body.get("entityid"),
            gl_cash_account=body.get("gl_cash_account"),
            active=body.get("active"),
            user=_user()))
    except Exception as e:
        return _fail(e, "update_account")


@treasury_bp.route("/import/activity", methods=["POST"])
@login_required
@role_required("admin", "cfo", "analyst")
def import_activity():
    """A PNC activity export. Re-importing the same file adds nothing."""
    import pandas as pd
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file was sent."}), 400
    raw = f.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File is larger than 25MB."}), 400
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": "Could not read that as a CSV: %s"
                                 % str(e)[:160]}), 400
    parsed = ts.parse_activity(df, source_file=f.filename or "")
    if parsed.get("error"):
        return jsonify({"error": parsed["error"]}), 400
    try:
        res = ts.import_activity(parsed["rows"])
    except Exception as e:
        return _fail(e, "import_activity", 500)
    # Rows that could not be read are RETURNED, not logged and forgotten: a
    # transaction dropped in silence makes a period tie for the wrong reason.
    res["skipped"] = parsed["skipped"]
    res["skipped_count"] = len(parsed["skipped"])
    return jsonify(safe_json(res))


@treasury_bp.route("/import/statement", methods=["POST"])
@login_required
@role_required("admin", "cfo", "analyst")
def import_statement():
    """A PNC statement PDF, for its beginning and ending balances."""
    f = request.files.get("file")
    account_number = (request.form.get("account_number") or "").strip()
    if not f:
        return jsonify({"error": "No file was sent."}), 400
    if not account_number:
        return jsonify({"error": "account_number is required."}), 400
    raw = f.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File is larger than 25MB."}), 400
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
    except Exception as e:
        return jsonify({"error": "Could not open that PDF: %s"
                                 % str(e)[:160]}), 400
    parsed = ts.parse_statement_text(txt, source_file=f.filename or "")
    res = ts.import_statement(parsed, account_number)
    if res.get("error"):
        # The parse is returned alongside the refusal so the screen can show
        # what WAS found -- a scanned statement is a different problem from a
        # statement whose figures disagree.
        return jsonify({"error": res["error"], "parsed": safe_json(parsed)}), 400
    return jsonify(safe_json({**res, "parsed": parsed}))


@treasury_bp.route("/reconcile", methods=["GET"])
@login_required
def reconcile():
    acct = (request.args.get("account_number") or "").strip()
    period = (request.args.get("period") or "").strip()
    gl_net = request.args.get("gl_net")
    if not acct or not period:
        return jsonify({"error": "account_number and period are required."}), 400
    try:
        return jsonify(safe_json(ts.reconcile(
            acct, period, gl_net=(float(gl_net) if gl_net not in (None, "") else None))))
    except Exception as e:
        return _fail(e, "reconcile", 500)


@treasury_bp.route("/match", methods=["GET"])
@login_required
def match():
    acct = (request.args.get("account_number") or "").strip()
    period = (request.args.get("period") or "").strip()
    if not acct or not period:
        return jsonify({"error": "account_number and period are required."}), 400
    try:
        return jsonify(safe_json(ts.match(acct, period)))
    except Exception as e:
        return _fail(e, "match", 500)


@treasury_bp.route("/match", methods=["PUT"])
@login_required
@role_required("admin", "cfo", "analyst")
def set_match():
    """Pin a pairing by hand, or clear one with a null gl_item."""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ts.set_match(
            (body.get("account_number") or "").strip(),
            (body.get("period") or "").strip(),
            int(body.get("bank_id")),
            body.get("gl_item"), _user()))
    except Exception as e:
        return _fail(e, "set_match")


@treasury_bp.route("/close", methods=["POST"])
@login_required
@role_required("admin", "cfo", "analyst")
def close():
    """Record a period's computed ending so the next opens from it."""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ts.close_period(
            (body.get("account_number") or "").strip(),
            (body.get("period") or "").strip(), _user()))
    except Exception as e:
        return _fail(e, "close_period")


@treasury_bp.route("/seed-opening", methods=["POST"])
@login_required
@role_required("admin", "cfo", "analyst")
def seed_opening():
    """Start the chain for an account's first period."""
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(ts.seed_opening(
            (body.get("account_number") or "").strip(),
            (body.get("period") or "").strip(),
            body.get("amount"), _user()))
    except Exception as e:
        return _fail(e, "seed_opening")
