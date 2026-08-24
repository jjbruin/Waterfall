"""Portfolio Snapshot — editable persistence + approval pipeline (Step 2).

Isolated by design: new tables, new code, nothing imported from the One Pager's
review layer. The pipeline below is a **copy** of ``review_service``'s state
machine, not a reuse of it, so a change to One Pager review can never alter
Portfolio Snapshot behaviour and vice versa.

Copied from review_service.py (read 2026-08-24):
  steps    0 draft / 1 pending_head_am / 2 pending_president / 3 pending_cco /
           4 pending_ceo / 5 approved, plus the out-of-band 'returned' at step 0
  submit   from draft|returned only, asset_manager role, -> step 1
  approve  steps 1..4 only, requires that step's role, -> next step
  reject   == return_to_draft: step >= 1, that step's role, note REQUIRED,
           -> 'returned' at step 0, remembering returned_to_step
  gating   is_editable == "no rows yet, or status != 'approved'". Editing stays
           open through the whole chain; only final approval locks.

One deliberate structural difference, documented rather than hidden: the One
Pager keeps status on a separate ``review_submissions`` row and stores none on
``one_pager_comments``. The Step 2 schema puts ``status``/``approver`` on each
element row. Both are honoured here — the columns live on the rows as specified,
and the transitions are applied at the document level ``(investor_code,
quarter)`` across all three tables in one transaction, so element rows can never
disagree about where the page sits in review.

Three element kinds share one save/approve code path (see ``_ELEMENT_SPECS``):
  comment   free text   — page 1 narrative (scope='report') and pages 3-4
                          per-deal comments (scope='deal', field=operating|loan)
  footnote  anchored    — page 2, auto-numbered from row order
  value     number      — manual Net ROE / ITD Distributions per deal
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import text

log = logging.getLogger(__name__)

# ── Pipeline definition (copied from review_service.REVIEW_STEPS) ──────────

SNAPSHOT_STEPS = [
    {"step": 0, "role": "asset_manager", "status": "draft",             "label": "Draft"},
    {"step": 1, "role": "head_am",       "status": "pending_head_am",   "label": "Head of AM"},
    {"step": 2, "role": "president",     "status": "pending_president", "label": "President"},
    {"step": 3, "role": "cco",           "status": "pending_cco",       "label": "CCO"},
    {"step": 4, "role": "ceo",           "status": "pending_ceo",       "label": "CEO"},
    {"step": 5, "role": None,            "status": "approved",          "label": "Approved"},
]
RETURNED_STATUS = "returned"
SNAPSHOT_ROLE_NAMES = [s["role"] for s in SNAPSHOT_STEPS if s["role"]]

TABLES = (
    "portfolio_snapshot_comments",
    "portfolio_snapshot_footnotes",
    "portfolio_snapshot_values",
)


def _step_for(step_num: int) -> dict:
    for s in SNAPSHOT_STEPS:
        if s["step"] == step_num:
            return s
    return SNAPSHOT_STEPS[0]


def _step_for_status(status: str) -> dict:
    if status == RETURNED_STATUS:
        return {"step": 0, "role": "asset_manager",
                "status": RETURNED_STATUS, "label": "Returned"}
    for s in SNAPSHOT_STEPS:
        if s["status"] == status:
            return s
    return SNAPSHOT_STEPS[0]


# ── Element specs — the shared editable-element pattern ───────────────────
#
# Every kind is described once: its table, the columns that identify a row
# within (investor_code, quarter), and the column holding the payload. save,
# load and the status transitions are then a single code path per kind.

_ELEMENT_SPECS = {
    "comment": {
        "table": "portfolio_snapshot_comments",
        "keys": ("scope", "scope_key", "field"),
        "payload": "comment_text",
        "payload_kind": "text",
    },
    "footnote": {
        "table": "portfolio_snapshot_footnotes",
        # A footnote is identified by its own id, not a natural key: the same
        # anchor may carry more than one footnote.
        "keys": (),
        "payload": "text",
        "payload_kind": "text",
    },
    "value": {
        "table": "portfolio_snapshot_values",
        "keys": ("deal_vcode", "field"),
        "payload": "value",
        "payload_kind": "number",
    },
}

VALUE_FIELDS = ("net_roe", "itd")
COMMENT_SCOPES = ("report", "deal")


class NotEditable(RuntimeError):
    """Raised when a write is attempted on an approved (locked) document."""


# ── Engine access (indirected so the self-test can point elsewhere) ───────

def _engine():
    from flask_app.db import get_engine
    return get_engine()


def _is_postgres() -> bool:
    try:
        from flask_app.db import is_postgres
        return bool(is_postgres())
    except Exception:
        return _engine().dialect.name == "postgresql"


def _ensure_tables() -> None:
    """Create the three Portfolio Snapshot tables if absent. Idempotent."""
    pk = "SERIAL PRIMARY KEY" if _is_postgres() else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with _engine().begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS portfolio_snapshot_comments (
                id {pk},
                investor_code TEXT NOT NULL,
                quarter TEXT NOT NULL,
                scope TEXT NOT NULL,
                scope_key TEXT NOT NULL DEFAULT '',
                field TEXT NOT NULL,
                comment_text TEXT,
                status TEXT NOT NULL DEFAULT 'draft',
                current_step INTEGER NOT NULL DEFAULT 0,
                returned_to_step INTEGER,
                approver TEXT,
                approved_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT,
                UNIQUE(investor_code, quarter, scope, scope_key, field)
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS portfolio_snapshot_footnotes (
                id {pk},
                investor_code TEXT NOT NULL,
                quarter TEXT NOT NULL,
                anchor TEXT NOT NULL,
                seq INTEGER NOT NULL DEFAULT 0,
                text TEXT,
                status TEXT NOT NULL DEFAULT 'draft',
                current_step INTEGER NOT NULL DEFAULT 0,
                returned_to_step INTEGER,
                approver TEXT,
                approved_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """))
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS portfolio_snapshot_values (
                id {pk},
                investor_code TEXT NOT NULL,
                quarter TEXT NOT NULL,
                deal_vcode TEXT NOT NULL,
                field TEXT NOT NULL,
                value DOUBLE PRECISION,
                status TEXT NOT NULL DEFAULT 'draft',
                current_step INTEGER NOT NULL DEFAULT 0,
                returned_to_step INTEGER,
                approver TEXT,
                approved_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT,
                UNIQUE(investor_code, quarter, deal_vcode, field)
            )
        """))


# ── Roles ─────────────────────────────────────────────────────────────────

def _user_roles(user_id: int, roles: Optional[list] = None) -> list[str]:
    """Review roles for a user.

    ``roles`` short-circuits the lookup (used by the self-test and by callers
    that already hold them). Otherwise this reads the existing ``review_roles``
    table **read-only** — role assignment is an app-wide concern and duplicating
    the assignment table would be worse than sharing the read. No write to it
    ever happens from this module.
    """
    if roles is not None:
        return list(roles)
    try:
        with _engine().connect() as conn:
            rows = conn.execute(
                text("SELECT review_role FROM review_roles WHERE user_id = :u"),
                {"u": user_id},
            ).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


# ── Document status ───────────────────────────────────────────────────────

def document_status(investor_code: str, quarter: str) -> dict:
    """Where this (investor, quarter) page sits in review.

    Derived from the element rows. With no rows at all the page is a draft, the
    same convention the One Pager uses for a missing submission row. If rows
    somehow disagree the *least* advanced status wins, so a stray approved row
    can never lock a page that is still being worked on.
    """
    _ensure_tables()
    found = []
    with _engine().connect() as conn:
        for tbl in TABLES:
            rows = conn.execute(
                text(f"""SELECT status, current_step, returned_to_step, approver,
                                approved_at
                         FROM {tbl}
                         WHERE investor_code = :i AND quarter = :q"""),
                {"i": investor_code, "q": quarter},
            ).fetchall()
            found.extend(rows)

    if not found:
        s = _step_for(0)
        return {"investor_code": investor_code, "quarter": quarter,
                "status": s["status"], "current_step": 0, "label": s["label"],
                "returned_to_step": None, "approver": None, "approved_at": None,
                "element_count": 0, "mixed": False}

    def rank(r):
        return -1 if r[0] == RETURNED_STATUS else int(r[1] or 0)

    least = min(found, key=rank)
    statuses = {r[0] for r in found}
    step_def = _step_for_status(least[0])
    return {
        "investor_code": investor_code, "quarter": quarter,
        "status": least[0],
        "current_step": int(least[1] or 0),
        "label": step_def["label"],
        "returned_to_step": least[2],
        "approver": least[3],
        "approved_at": least[4],
        "element_count": len(found),
        "mixed": len(statuses) > 1,
    }


def is_editable(investor_code: str, quarter: str) -> bool:
    """Editable unless the page has been finally approved.

    Copied gating: open through the entire review chain, locked only at
    'approved'. A page with no rows yet is editable.
    """
    return document_status(investor_code, quarter)["status"] != "approved"


def _require_editable(investor_code: str, quarter: str) -> None:
    if not is_editable(investor_code, quarter):
        raise NotEditable(
            f"{investor_code} {quarter} is approved; elements are locked")


# ── Shared save / load ────────────────────────────────────────────────────

def _spec(kind: str) -> dict:
    if kind not in _ELEMENT_SPECS:
        raise ValueError(f"unknown element kind '{kind}'")
    return _ELEMENT_SPECS[kind]


def save_element(kind: str, investor_code: str, quarter: str, payload,
                 updated_by: str = "", **keys) -> dict:
    """Insert or update one element, honouring the current lock.

    The single write path for all three kinds. A saved element is reset to the
    page's current status so a late edit cannot carry a stale approval forward.
    """
    spec = _spec(kind)
    _ensure_tables()
    _require_editable(investor_code, quarter)

    missing = [k for k in spec["keys"] if k not in keys]
    if missing:
        raise ValueError(f"{kind}: missing key(s) {missing}")
    if kind == "comment" and keys.get("scope") not in COMMENT_SCOPES:
        raise ValueError(f"comment scope must be one of {COMMENT_SCOPES}")
    if kind == "value" and keys.get("field") not in VALUE_FIELDS:
        raise ValueError(f"value field must be one of {VALUE_FIELDS}")
    if spec["payload_kind"] == "number" and payload is not None:
        payload = float(payload)

    doc = document_status(investor_code, quarter)
    tbl, col = spec["table"], spec["payload"]
    params = {"i": investor_code, "q": quarter, "p": payload,
              "u": updated_by, "st": doc["status"], "cs": doc["current_step"]}
    where = " AND ".join([f"{k} = :k_{k}" for k in spec["keys"]])
    params.update({f"k_{k}": keys[k] for k in spec["keys"]})
    clause = f"investor_code = :i AND quarter = :q" + (f" AND {where}" if where else "")

    with _engine().begin() as conn:
        existing = conn.execute(
            text(f"SELECT id FROM {tbl} WHERE {clause}"), params).fetchone()
        if existing:
            conn.execute(text(f"""
                UPDATE {tbl} SET {col} = :p, status = :st, current_step = :cs,
                       approver = NULL, approved_at = NULL,
                       updated_at = CURRENT_TIMESTAMP, updated_by = :u
                WHERE id = :id
            """), {**params, "id": existing[0]})
            row_id = existing[0]
        else:
            cols = ["investor_code", "quarter", col, "status", "current_step",
                    "updated_by"] + list(spec["keys"])
            vals = [":i", ":q", ":p", ":st", ":cs", ":u"] + \
                   [f":k_{k}" for k in spec["keys"]]
            conn.execute(text(
                f"INSERT INTO {tbl} ({', '.join(cols)}) "
                f"VALUES ({', '.join(vals)})"), params)
            row_id = conn.execute(
                text(f"SELECT id FROM {tbl} WHERE {clause}"), params).fetchone()[0]
    return {"kind": kind, "id": row_id, "status": doc["status"], **keys}


def get_elements(kind: str, investor_code: str, quarter: str,
                 **filters) -> list[dict]:
    """Load elements of one kind. Footnotes come back ordered and numbered."""
    spec = _spec(kind)
    _ensure_tables()
    tbl = spec["table"]
    clause = "investor_code = :i AND quarter = :q"
    params = {"i": investor_code, "q": quarter}
    for k, v in filters.items():
        clause += f" AND {k} = :f_{k}"
        params[f"f_{k}"] = v
    order = "seq, id" if kind == "footnote" else "id"

    with _engine().connect() as conn:
        rows = conn.execute(
            text(f"SELECT * FROM {tbl} WHERE {clause} ORDER BY {order}"),
            params).mappings().fetchall()
    out = [dict(r) for r in rows]
    if kind == "footnote":
        # `number` is derived from position, never stored, so anchor markers and
        # the bottom list cannot drift apart after an add or a remove.
        for i, r in enumerate(out, 1):
            r["number"] = i
    return out


# ── Footnotes: add / remove / re-sequence ─────────────────────────────────

def _resequence_footnotes(conn, investor_code: str, quarter: str) -> None:
    rows = conn.execute(
        text("""SELECT id FROM portfolio_snapshot_footnotes
                WHERE investor_code = :i AND quarter = :q
                ORDER BY seq, id"""),
        {"i": investor_code, "q": quarter}).fetchall()
    for i, (row_id,) in enumerate(rows, 1):
        conn.execute(
            text("UPDATE portfolio_snapshot_footnotes SET seq = :s WHERE id = :id"),
            {"s": i, "id": row_id})


def add_footnote(investor_code: str, quarter: str, anchor: str, footnote_text: str,
                 updated_by: str = "") -> dict:
    """Append a footnote to an anchor; it takes the next number."""
    _ensure_tables()
    _require_editable(investor_code, quarter)
    doc = document_status(investor_code, quarter)
    with _engine().begin() as conn:
        nxt = conn.execute(
            text("""SELECT COALESCE(MAX(seq), 0) + 1
                    FROM portfolio_snapshot_footnotes
                    WHERE investor_code = :i AND quarter = :q"""),
            {"i": investor_code, "q": quarter}).fetchone()[0]
        conn.execute(text("""
            INSERT INTO portfolio_snapshot_footnotes
                (investor_code, quarter, anchor, seq, text, status, current_step,
                 updated_by)
            VALUES (:i, :q, :a, :s, :t, :st, :cs, :u)
        """), {"i": investor_code, "q": quarter, "a": anchor, "s": nxt,
               "t": footnote_text, "st": doc["status"],
               "cs": doc["current_step"], "u": updated_by})
        _resequence_footnotes(conn, investor_code, quarter)
    notes = get_elements("footnote", investor_code, quarter)
    return next((n for n in notes if n["seq"] == nxt), notes[-1] if notes else {})


def remove_footnote(investor_code: str, quarter: str, footnote_id: int) -> list[dict]:
    """Delete a footnote and re-sequence the rest so numbering stays contiguous."""
    _ensure_tables()
    _require_editable(investor_code, quarter)
    with _engine().begin() as conn:
        conn.execute(
            text("""DELETE FROM portfolio_snapshot_footnotes
                    WHERE id = :id AND investor_code = :i AND quarter = :q"""),
            {"id": footnote_id, "i": investor_code, "q": quarter})
        _resequence_footnotes(conn, investor_code, quarter)
    return get_elements("footnote", investor_code, quarter)


# ── Pipeline transitions (copied state machine) ───────────────────────────

def _set_status(investor_code: str, quarter: str, status: str, step: int,
                returned_to_step: Optional[int] = None,
                approver: Optional[str] = None) -> None:
    """Move every element of this page to one status, in a single transaction."""
    approved = status == "approved"
    with _engine().begin() as conn:
        for tbl in TABLES:
            conn.execute(text(f"""
                UPDATE {tbl}
                SET status = :s, current_step = :step, returned_to_step = :ret,
                    approver = CASE WHEN :appr IS NULL THEN approver ELSE :appr END,
                    approved_at = CASE WHEN :is_appr = 1
                                       THEN CURRENT_TIMESTAMP ELSE NULL END,
                    updated_at = CURRENT_TIMESTAMP
                WHERE investor_code = :i AND quarter = :q
            """), {"s": status, "step": step, "ret": returned_to_step,
                   "appr": approver, "is_appr": 1 if approved else 0,
                   "i": investor_code, "q": quarter})


def submit_for_review(investor_code: str, quarter: str, user_id: int,
                      username: str, roles: Optional[list] = None) -> dict:
    """draft|returned -> pending_head_am. Asset manager only."""
    _ensure_tables()
    doc = document_status(investor_code, quarter)
    if doc["element_count"] == 0:
        raise ValueError("Nothing to submit: no saved elements for this page")
    if doc["status"] not in ("draft", RETURNED_STATUS):
        raise ValueError(f"Cannot submit: current status is '{doc['status']}'")
    if "asset_manager" not in _user_roles(user_id, roles):
        raise PermissionError("Only asset managers can submit for review")

    nxt = SNAPSHOT_STEPS[1]
    _set_status(investor_code, quarter, nxt["status"], nxt["step"])
    return document_status(investor_code, quarter)


def approve(investor_code: str, quarter: str, user_id: int, username: str,
            roles: Optional[list] = None) -> dict:
    """Approve at the current step and advance. Step 4 -> 'approved' locks."""
    _ensure_tables()
    doc = document_status(investor_code, quarter)
    step_num = doc["current_step"]
    if doc["status"] == RETURNED_STATUS or step_num < 1 or step_num > 4:
        raise ValueError(f"Nothing to approve at step {step_num} "
                         f"(status '{doc['status']}')")
    step_def = _step_for(step_num)
    if step_def["role"] not in _user_roles(user_id, roles):
        raise PermissionError(
            f"You need the '{step_def['role']}' role to approve at this step")

    nxt = SNAPSHOT_STEPS[step_num + 1]
    _set_status(investor_code, quarter, nxt["status"], nxt["step"],
                approver=username if nxt["status"] == "approved" else None)
    return document_status(investor_code, quarter)


def reject(investor_code: str, quarter: str, user_id: int, username: str,
           note_text: str, roles: Optional[list] = None) -> dict:
    """Return the page to the asset manager. Note required (copied rule)."""
    _ensure_tables()
    if not note_text or not str(note_text).strip():
        raise ValueError("A note is required when returning a page")
    doc = document_status(investor_code, quarter)
    step_num = doc["current_step"]
    if doc["status"] == RETURNED_STATUS or step_num < 1:
        raise ValueError("Page is already with the asset manager")
    step_def = _step_for(step_num)
    if step_def["role"] not in _user_roles(user_id, roles):
        raise PermissionError(
            f"You need the '{step_def['role']}' role to return at this step")

    _set_status(investor_code, quarter, RETURNED_STATUS, 0,
                returned_to_step=step_num)
    return document_status(investor_code, quarter)


# ── Convenience wrappers ──────────────────────────────────────────────────

def save_comment(investor_code: str, quarter: str, scope: str, field: str,
                 comment_text: str, scope_key: str = "",
                 updated_by: str = "") -> dict:
    return save_element("comment", investor_code, quarter, comment_text,
                        updated_by=updated_by, scope=scope,
                        scope_key=scope_key, field=field)


def save_value(investor_code: str, quarter: str, deal_vcode: str, field: str,
               value, updated_by: str = "") -> dict:
    return save_element("value", investor_code, quarter, value,
                        updated_by=updated_by, deal_vcode=deal_vcode,
                        field=field)


def load_page(investor_code: str, quarter: str) -> dict:
    """Everything editable for one page, plus its review status."""
    return {
        "status": document_status(investor_code, quarter),
        "editable": is_editable(investor_code, quarter),
        "comments": get_elements("comment", investor_code, quarter),
        "footnotes": get_elements("footnote", investor_code, quarter),
        "values": get_elements("value", investor_code, quarter),
    }


# ── Self-test ─────────────────────────────────────────────────────────────

def _selftest():                                    # pragma: no cover
    """Exercise persistence, the pipeline and footnote re-sequencing.

    Runs against a throwaway SQLite file so no application database is touched.
    """
    import os
    import sys
    import tempfile
    import sqlalchemy

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_step2_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    g = sys.modules[__name__]
    g._engine = lambda: eng                          # type: ignore[assignment]
    g._is_postgres = lambda: False                   # type: ignore[assignment]
    print(f"scratch db: {tmp}\n")

    INV, Q = "TGAM", "2026-Q2"
    AM = {"roles": ["asset_manager"]}
    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print(f"    [{'PASS' if cond else 'FAIL'}] {label}")

    print("1. COMMENT save / load / transitions")
    save_comment(INV, Q, "report", "narrative_1", "Portfolio held steady.",
                 updated_by="cbui")
    save_comment(INV, Q, "deal", "operating", "Occupancy improving.",
                 scope_key="P0000030", updated_by="cbui")
    save_comment(INV, Q, "deal", "loan", "Fixed through 2032.",
                 scope_key="P0000030", updated_by="cbui")
    cs = get_elements("comment", INV, Q)
    chk("3 comments stored", len(cs) == 3)
    one = get_elements("comment", INV, Q, scope="deal", scope_key="P0000030",
                       field="operating")
    chk("loads back by (investor, quarter, scope, key, field)",
        len(one) == 1 and one[0]["comment_text"] == "Occupancy improving.")
    save_comment(INV, Q, "deal", "operating", "Occupancy improving; +3pp.",
                 scope_key="P0000030", updated_by="cbui")
    again = get_elements("comment", INV, Q, scope="deal",
                         scope_key="P0000030", field="operating")
    chk("re-save updates in place (UNIQUE holds)",
        len(again) == 1 and again[0]["comment_text"].endswith("+3pp."))
    chk("same deal keeps operating and loan as separate fields",
        len(get_elements("comment", INV, Q, scope="deal",
                         scope_key="P0000030")) == 2)

    print("\n2. FOOTNOTES auto-number and re-sequence")
    f1 = add_footnote(INV, Q, "itd_distributions", "Net of fee allocation.",
                      updated_by="cbui")
    f2 = add_footnote(INV, Q, "net_roe", "Weighted by dollars and time.",
                      updated_by="cbui")
    f3 = add_footnote(INV, Q, "total_commitment", "Excludes unfunded.",
                      updated_by="cbui")
    fn = get_elements("footnote", INV, Q)
    chk("3 footnotes numbered 1,2,3",
        [x["number"] for x in fn] == [1, 2, 3])
    chk("numbers follow insertion order",
        [x["anchor"] for x in fn] ==
        ["itd_distributions", "net_roe", "total_commitment"])
    after = remove_footnote(INV, Q, f2["id"])
    chk("removing #2 re-sequences to 1,2 contiguous",
        [x["number"] for x in after] == [1, 2])
    chk("survivors keep relative order, anchors intact",
        [x["anchor"] for x in after] == ["itd_distributions", "total_commitment"])
    f4 = add_footnote(INV, Q, "ltv", "Debt over concluded value.",
                      updated_by="cbui")
    chk("next add takes number 3 (no gap, no reuse)",
        f4["number"] == 3 and
        [x["number"] for x in get_elements("footnote", INV, Q)] == [1, 2, 3])

    print("\n3. VALUES (manual Net ROE / ITD)")
    save_value(INV, Q, "P0000030", "net_roe", 0.0912, updated_by="cbui")
    save_value(INV, Q, "P0000030", "itd", 1250000.0, updated_by="cbui")
    save_value(INV, Q, "P0000075", "net_roe", 0.0788, updated_by="cbui")
    vs = get_elements("value", INV, Q)
    chk("3 values stored per (investor, quarter, deal, field)", len(vs) == 3)
    nr = get_elements("value", INV, Q, deal_vcode="P0000030", field="net_roe")
    chk("Net ROE round-trips as a number",
        len(nr) == 1 and abs(nr[0]["value"] - 0.0912) < 1e-12)
    try:
        save_value(INV, Q, "P0000030", "bogus_field", 1.0)
        chk("rejects an unknown value field", False)
    except ValueError:
        chk("rejects an unknown value field", True)

    print("\n4. PIPELINE draft -> pending -> approved -> locked")
    chk("starts in draft", document_status(INV, Q)["status"] == "draft")
    chk("editable while draft", is_editable(INV, Q))
    try:
        approve(INV, Q, 1, "cbui", roles=["head_am"])
        chk("cannot approve from draft", False)
    except ValueError:
        chk("cannot approve from draft", True)
    try:
        submit_for_review(INV, Q, 1, "cbui", roles=["cco"])
        chk("non-asset-manager cannot submit", False)
    except PermissionError:
        chk("non-asset-manager cannot submit", True)

    st = submit_for_review(INV, Q, 1, "cbui", **AM)
    chk("submit -> pending_head_am", st["status"] == "pending_head_am")
    chk("still editable during review", is_editable(INV, Q))
    try:
        approve(INV, Q, 2, "jim", roles=["ceo"])
        chk("wrong role cannot approve this step", False)
    except PermissionError:
        chk("wrong role cannot approve this step", True)

    st = reject(INV, Q, 2, "jim", "Fix the ITD footnote.", roles=["head_am"])
    chk("reject -> returned, remembers step",
        st["status"] == "returned" and st["returned_to_step"] == 1)
    chk("editable after reject", is_editable(INV, Q))
    try:
        reject(INV, Q, 2, "jim", "", roles=["head_am"])
        chk("reject requires a note", False)
    except ValueError:
        chk("reject requires a note", True)

    submit_for_review(INV, Q, 1, "cbui", **AM)
    for role, expect in [("head_am", "pending_president"),
                         ("president", "pending_cco"),
                         ("cco", "pending_ceo"),
                         ("ceo", "approved")]:
        st = approve(INV, Q, 9, "approver_" + role, roles=[role])
        chk(f"{role} approves -> {expect}", st["status"] == expect)

    chk("approved: NOT editable", not is_editable(INV, Q))
    chk("approver recorded", document_status(INV, Q)["approver"] == "approver_ceo")
    for label, fn_call in [
            ("comment save blocked when approved",
             lambda: save_comment(INV, Q, "report", "narrative_1", "late edit")),
            ("footnote add blocked when approved",
             lambda: add_footnote(INV, Q, "ltv", "late")),
            ("footnote remove blocked when approved",
             lambda: remove_footnote(INV, Q, f4["id"])),
            ("value save blocked when approved",
             lambda: save_value(INV, Q, "P0000030", "net_roe", 0.5))]:
        try:
            fn_call()
            chk(label, False)
        except NotEditable:
            chk(label, True)

    print("\n5. ISOLATION of a second page")
    save_comment(INV, "2026-Q1", "report", "narrative_1", "Prior quarter.",
                 updated_by="cbui")
    chk("other quarter unaffected by the approval",
        is_editable(INV, "2026-Q1") and
        document_status(INV, "2026-Q1")["status"] == "draft")
    chk("approved quarter still locked", not is_editable(INV, Q))
    page = load_page(INV, Q)
    chk("load_page returns all three kinds + status",
        len(page["comments"]) == 3 and len(page["footnotes"]) == 3
        and len(page["values"]) == 3 and page["editable"] is False)

    ok = all(c for _, c in checks)
    print(f"\n  {sum(1 for _, c in checks if c)}/{len(checks)} checks passed — "
          f"{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":                          # pragma: no cover
    raise SystemExit(_selftest())
