"""Treasury endpoints: bank accounts, imported activity, and reconciliation.

Reads are open to any signed-in user -- an accountant has to be able to see
where a reconciliation stands. Writes that change the ARRANGEMENT (which entity
an account belongs to, which cash account it posts to, closing a period) are
gated like the rest of the accounting section, which is the CFO's.

IMPORTS ARE OPEN TO EVERY ACCOUNTING ROLE, not just the CFO, because importing
a bank export is the accountant's daily work and it is not destructive:
activity is identified by a row hash, so re-importing the same file adds
nothing. A viewer is still refused -- read-only should mean read-only.

WHO MAY EDIT: `ACCOUNTING_ROLES` -- accountant, accounting_manager, cfo, and
admin so Jim can fix something while this is being built. Checked BY NAME
(`roles_exactly`) rather than by level, because analyst sits at the same level
as every accounting role and a level comparison cannot exclude it.
"""
import io
import logging

from flask import Blueprint, Response, g, jsonify, request

from flask_app.auth.routes import (ACCOUNTING_ROLES, login_required,
                                   roles_exactly)
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
@roles_exactly(*ACCOUNTING_ROLES)
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
@roles_exactly(*ACCOUNTING_ROLES)
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
@roles_exactly(*ACCOUNTING_ROLES)
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
@roles_exactly(*ACCOUNTING_ROLES)
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
@roles_exactly(*ACCOUNTING_ROLES)
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
@roles_exactly(*ACCOUNTING_ROLES)
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


# ------------------------------------------------- the MRI upload files
#
# STATELESS ON PURPOSE. The accountant's coding lives on the screen until they
# download; nothing is stored, because a half-coded journal entry is a draft,
# not a record. What IS stored is the reconciliation it was built from.

@treasury_bp.route("/split", methods=["GET"])
@login_required
def split():
    """Propose how a distribution divides between investors.

    A PROPOSAL, not an answer (Jim, Sep 17 2026: "compute it and show it as an
    editable proposal"). Computed from commitment AMOUNTS -- see
    `treasury_upload.SPLIT_BASIS` for why not the stored percentages.
    """
    from flask_app.services import treasury_upload as tu
    amount = request.args.get("amount")
    try:
        amt = float(amount)
    except (TypeError, ValueError):
        return jsonify({"error": "A numeric amount is required."}), 400
    try:
        return jsonify(safe_json(tu.propose_investor_split(
            (request.args.get("entityid") or "").strip(), amt,
            as_of=(request.args.get("as_of") or "").strip())))
    except Exception as e:
        return _fail(e, "split", 500)


@treasury_bp.route("/upload/gl", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def upload_gl():
    """The GL journal entry, as the CSV MRI's uploader takes."""
    from flask_app.services import treasury_upload as tu
    body = request.get_json(silent=True) or {}
    lines = body.get("lines") or []
    v = tu.validate_gl(lines)
    if v["errors"]:
        # The refusal carries the whole validation, so the screen can show
        # every problem at once rather than one per attempt.
        return jsonify({"error": v["errors"][0], "validation": safe_json(v)}), 400
    try:
        text = tu.build_gl_csv(lines)
    except Exception as e:
        return _fail(e, "upload_gl")
    name = "%s %s GL Upload.csv" % (v.get("entityid") or "GL",
                                    v.get("period") or "")
    return Response(text, mimetype="text/csv", headers={
        "Content-Disposition": 'attachment; filename="%s"' % name.strip()})


@treasury_bp.route("/upload/ia", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def upload_ia():
    """The investor transactions, in a copy of MRI's own template."""
    from flask_app.services import treasury_upload as tu
    body = request.get_json(silent=True) or {}
    rows = body.get("rows") or []
    lines = body.get("lines")
    acct = (body.get("ia_account") or "").strip()
    v = tu.validate_ia(rows, gl_lines=lines, ia_account=acct)
    if v["errors"]:
        return jsonify({"error": v["errors"][0], "validation": safe_json(v)}), 400
    try:
        data = tu.build_ia_xlsx(rows, gl_lines=lines, ia_account=acct)
    except Exception as e:
        return _fail(e, "upload_ia")
    name = "%s IA Upload.xlsx" % (rows[0].get("investmentid") or "IA")
    return Response(data, mimetype=(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        headers={"Content-Disposition": 'attachment; filename="%s"' % name})


@treasury_bp.route("/upload/preview", methods=["POST"])
@login_required
def upload_preview():
    """What the accountant sees BEFORE downloading either file.

    The three figures that made the August set verifiable: the entry balances,
    its cash lines equal the bank's movement, and the investor rows tie to the
    GL account they mirror.
    """
    from flask_app.services import treasury_upload as tu
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(safe_json(tu.summarise(
            body.get("lines") or [], ia_rows=body.get("rows"),
            cash_account=(body.get("cash_account") or ts.DEFAULT_CASH_ACCOUNT),
            ia_account=(body.get("ia_account") or ""))))
    except Exception as e:
        return _fail(e, "upload_preview", 500)


@treasury_bp.route("/activity", methods=["GET"])
@login_required
def activity():
    """The imported bank transactions for one account and period.

    The coding tab's own list. `/match` also carries bank rows, but it refuses
    when the account is unmapped or the GL feed is empty -- coding has to be
    possible before either is true.
    """
    import pandas as pd
    from sqlalchemy import text
    from flask_app.db import get_engine
    acct = (request.args.get("account_number") or "").strip()
    period = (request.args.get("period") or "").strip()
    if not acct or len(period) != 6:
        return jsonify({"error": "account_number and a YYYYMM period are "
                                 "required."}), 400
    try:
        eng = get_engine()
        ts.ensure_tables(eng)
        with eng.connect() as conn:
            df = pd.read_sql(text(
                "SELECT id, as_of_date, amount, direction, signed_amount, "
                "       reference, description, transaction_type "
                "  FROM tr_activity WHERE account_number = :a "
                "   AND as_of_date LIKE :p ORDER BY as_of_date, id"),
                conn, params={"a": acct, "p": "%s-%s%%" % (period[:4], period[4:])})
            meta = conn.execute(text(
                "SELECT entityid, gl_cash_account FROM tr_accounts "
                " WHERE account_number = :a"), {"a": acct}).fetchone()
    except Exception as e:
        return _fail(e, "activity", 500)
    return jsonify(safe_json({
        "account_number": acct, "period": period,
        "entityid": (meta[0] if meta else None),
        "gl_cash_account": ((meta[1] if meta else None)
                            or ts.DEFAULT_CASH_ACCOUNT),
        "transactions": df.to_dict("records"),
        "count": int(len(df)),
        "net": (float(df["signed_amount"].sum()) if len(df) else 0.0),
    }))


@treasury_bp.route("/accounts-list", methods=["GET"])
@login_required
def gl_accounts():
    """MRI's chart of accounts, for the offset dropdown.

    Returned with the TYPE so the screen can say what an account is. Empty with
    a reason rather than an error when the chart has not been loaded -- the
    accountant can still type an account number.
    """
    import pandas as pd
    from sqlalchemy import text
    from flask_app.db import get_engine
    try:
        with get_engine().connect() as conn:
            df = pd.read_sql(text(
                'SELECT "ACCTNUM", "ACCTNAME", "TYPE" FROM gl_accounts '
                ' ORDER BY "ACCTNUM"'), conn)
    except Exception as e:
        logger.warning("gl_accounts unavailable: %s", e)
        return jsonify({"accounts": [], "note":
                        "MRI's chart of accounts has not been loaded, so there "
                        "is no list to choose from. Account numbers can still "
                        "be typed."})
    return jsonify(safe_json({
        "accounts": [{"acctnum": str(r["ACCTNUM"]).strip(),
                      "name": str(r["ACCTNAME"] or "").strip(),
                      "type": str(r["TYPE"] or "").strip()}
                     for _, r in df.iterrows()],
        "count": int(len(df))}))


@treasury_bp.route("/import/statements", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def import_statements():
    """Several statement PDFs at once, each routed to its own account.

    PNC prints a MASKED account number (``XX-XXXX-5765``), so the last four
    digits are all there is to route on. Across the 50 accounts in the
    September export all fifty last-four groups are unique, but the service
    refuses an ambiguous suffix rather than picking one -- a statement filed
    against the wrong account corrupts a reconciliation silently.

    EVERY FILE GETS ITS OWN RESULT. One unreadable PDF must not take the other
    forty-nine with it, and "which ones did not land" is the only question
    worth asking after a bulk import.
    """
    from flask_app.services import treasury_service as _ts
    files = request.files.getlist("files") or []
    if not files:
        return jsonify({"error": "No files were sent."}), 400
    if len(files) > 200:
        return jsonify({"error": "Too many files in one go (limit 200)."}), 400

    try:
        import pdfplumber
    except Exception as e:
        return _fail(e, "pdfplumber", 500)

    results, filed, skipped = [], 0, 0
    for f in files:
        name = f.filename or "(unnamed)"
        row = {"file": name}
        try:
            raw = f.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                row["error"] = "Larger than 25MB."
                results.append(row); skipped += 1; continue
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception as e:
            row["error"] = "Could not open the PDF: %s" % str(e)[:120]
            results.append(row); skipped += 1; continue

        parsed = _ts.parse_statement_text(txt, source_file=name)
        row["period_end"] = parsed.get("period_end")
        row["ending_balance"] = parsed.get("ending_balance")
        row["suffix"] = parsed.get("account_suffix")

        hit = _ts.match_account_by_suffix(parsed.get("account_suffix"))
        if hit.get("error"):
            row["error"] = hit["error"]
            results.append(row); skipped += 1; continue
        row["account_number"] = hit["account_number"]

        res = _ts.import_statement(parsed, hit["account_number"])
        if res.get("error"):
            row["error"] = res["error"]
            skipped += 1
        else:
            row["filed"] = True
            filed += 1
        results.append(row)

    return jsonify(safe_json({
        "filed": filed, "skipped": skipped, "count": len(files),
        "results": results}))


@treasury_bp.route("/seed-from-statement", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def seed_from_statement():
    """Open an account's chain from the prior month's filed statement.

    Refused once a period has actually been reconciled -- see the service note.
    With `all: true`, does it for every account that can be seeded and reports
    the ones that cannot, which is the point of doing fifty at once.
    """
    from flask_app.services import treasury_service as _ts
    body = request.get_json(silent=True) or {}
    period = (body.get("period") or "").strip()
    if len(period) != 6 or not period.isdigit():
        return jsonify({"error": "A period like 202607 is required."}), 400

    if not body.get("all"):
        acct = (body.get("account_number") or "").strip()
        if not acct:
            return jsonify({"error": "account_number is required."}), 400
        return jsonify(safe_json(
            _ts.seed_from_statement(acct, period, _user())))

    out, seeded = [], 0
    for a in _ts.accounts():
        r = _ts.seed_from_statement(a["account_number"], period, _user())
        row = {"account_number": a["account_number"],
               "entityid": a.get("entityid")}
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row["opening"] = r.get("amount")
            row["source"] = r.get("source")
            seeded += 1
        out.append(row)
    return jsonify(safe_json({"seeded": seeded, "period": period,
                              "count": len(out), "results": out}))


@treasury_bp.route("/accounts", methods=["POST"])
@login_required
@roles_exactly(*ACCOUNTING_ROLES)
def create_account():
    """Register an account by hand, for one PNC will not serve activity for.

    The full number is required and never inferred from the statement's mask —
    see the service note.
    """
    body = request.get_json(silent=True) or {}
    try:
        res = ts.create_account(
            (body.get("account_number") or "").strip(),
            entityid=(body.get("entityid") or ""),
            gl_cash_account=(body.get("gl_cash_account") or ""),
            account_name=(body.get("account_name") or ""),
            user=_user())
    except Exception as e:
        return _fail(e, "create_account", 500)
    return (jsonify(res), 400) if res.get("error") else jsonify(res)
