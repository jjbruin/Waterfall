"""Portfolio Snapshot — snapshot freeze.

An approved report must show what was **approved**, not a fresh computation. Live
MRI data moves: 45th & Main's look-through went 100% -> 90% on 2026-08-24 when a
PMX ownership row was corrected, which would silently have changed an already
approved 26Q1 report. Freezing removes that class of drift entirely.

Mirrors ``review_service._save_snapshot`` / ``get_snapshot`` (the One Pager's
freeze) in every respect except one, called out below.

  storage    ``portfolio_snapshot_frozen``, UNIQUE(investor_code, quarter),
             payload as a JSON text blob, plus approved_by / approved_at and the
             data version in force at freeze time.
  trigger    the FINAL transition into ``approved`` only (CEO), from
             ``portfolio_snapshot_persistence.approve``.
  upsert     DELETE-then-INSERT in one transaction — cross-DB (SQLite locally,
             PostgreSQL on Azure) rather than ON CONFLICT. This *is* the
             re-approval mechanism: a second approval overwrites, so a stale
             frozen payload can never outlive a legitimate re-approval.
  failure    the whole freeze is wrapped by its caller so a write failure never
             blocks the approval. Losing the user's approval because a snapshot
             write failed would be worse than a missing snapshot.
  unfreeze   nothing to do. ``_set_status`` already sets ``approved_at = NULL``
             on any non-approved transition, so a report reopened to draft stops
             reading ``status == 'approved'`` and the read path falls back to
             live by itself.

**THE ONE DELIBERATE DIVERGENCE FROM THE ONE PAGER.** The One Pager keeps *live*
as its default and exposes the frozen copy behind a manual "View Approved
Version" toggle (``financials.py`` ``/one-pager/snapshot`` plus a
``has_snapshot`` flag). The Portfolio Snapshot does the opposite: an approved
report serves the **frozen payload by default**, and every payload carries
``source: "frozen" | "live"`` so the UI can say so. Creator decision — do not
"align" the two tabs without re-reading this note, they are intentionally
different.

WHY THE ASSEMBLY ORCHESTRATION LIVES HERE. ``assemble_full_report`` is the single
path used by *both* the freeze and the live read. If the freeze assembled the
report any differently from the live path, a frozen payload would differ from
live even when nothing changed, and every comparison between them would be
noise. One function, both callers, no drift by construction.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from sqlalchemy import text

log = logging.getLogger(__name__)

#: Marks which path produced a payload. The UI branches on this.
SOURCE_FROZEN = "frozen"
SOURCE_LIVE = "live"

_TABLE = "portfolio_snapshot_frozen"


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def _is_postgres() -> bool:
    try:
        return _engine().dialect.name == "postgresql"
    except Exception:
        return False


def _ensure_table() -> None:
    """Create the frozen-payload table if absent. Idempotent."""
    pk = ("SERIAL PRIMARY KEY" if _is_postgres()
          else "INTEGER PRIMARY KEY AUTOINCREMENT")
    with _engine().begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
                id {pk},
                investor_code TEXT NOT NULL,
                quarter TEXT NOT NULL,
                payload TEXT NOT NULL,
                approved_by TEXT,
                approved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_version TEXT,
                UNIQUE(investor_code, quarter)
            )
        """))


# ── assembly (shared by the freeze and the live read) ─────────────────────

def _one_pager_provider(data: dict) -> Callable:
    """(vcode, quarter) -> One Pager payload, memoised per call.

    ``full_data`` is deliberately omitted: passing it runs a deal-analysis
    waterfall per deal, 30+ per report. The snapshot reads only ``cap_stack``
    and ``property_performance``, neither of which needs that enrichment.
    """
    from flask_app.services.financials_service import get_one_pager_data

    cache: dict = {}

    def provider(vcode: str, quarter: str) -> dict:
        key = (vcode, quarter)
        if key not in cache:
            cache[key] = get_one_pager_data(
                vcode, quarter, data["inv"], data["isbs_raw"],
                data["mri_loans_raw"], data["mri_val"],
                data["wf"], data["acct"],
                occupancy_raw=data["occupancy_raw"],
                budget_econ_occ=data.get("budget_econ_occ"),
                deal_terms=data.get("deal_terms_raw"),
                at_close_noi=data.get("at_close_noi_raw"),
                event_dates=data.get("event_dates_raw"),
                relationships=data.get("relationships_raw"),
                mri_loans_all=data.get("mri_loans_all"),
            )
        return cache[key]

    return provider


def _quarterly_noi_provider(data: dict) -> Callable:
    """(vcode, quarter) -> that quarter's periodic NOI, or None.

    Reads the same pipeline Property Financials uses, so the Loan subtab's Debt
    Yield cannot drift from the chart. A quarter missing any of its three months
    returns None and the subtab flags it rather than annualising a stub.
    """
    import pandas as pd
    from flask_app.services.financials_service import get_performance_chart_data

    cache: dict = {}

    def provider(vcode: str, quarter: str):
        key = (vcode, quarter)
        if key in cache:
            return cache[key]
        val = None
        try:
            year = int(str(quarter).split("-Q")[0])
            qn = int(str(quarter).split("Q")[1])
            q_end = (pd.Timestamp(year=year, month=qn * 3, day=1)
                     + pd.offsets.MonthEnd(0))
            chart = get_performance_chart_data(
                data["isbs_raw"], data["occupancy_raw"], vcode,
                freq="Quarterly", periods=12, period_end=str(q_end.date()),
            ) or {}
            for lbl, actual in zip(chart.get("periods") or [],
                                   chart.get("actual_noi") or []):
                if lbl == f"Q{qn} {year}" and actual is not None:
                    val = float(actual)
                    break
        except Exception:
            val = None
        cache[key] = val
        return val

    return provider


def build_subtab(name: str, investor: str, quarter: str, data: dict,
                 resolved: dict,
                 one_pager_provider: Optional[Callable] = None,
                 quarterly_noi_provider: Optional[Callable] = None) -> dict:
    """One subtab, by name. The only place an assembly is invoked.

    PASS THE PROVIDERS IN when building more than one subtab. They memoise, and
    a provider built here is scoped to this call — so four subtabs each getting
    their own would compute every deal's One Pager four times. Measured on
    TIAA/26Q1: 128 ``get_one_pager_data`` calls across 32 deals instead of 32.
    ``assemble_full_report`` builds them once and threads them through.
    """
    op = one_pager_provider or _one_pager_provider(data)

    if name == "summary":
        from flask_app.services.portfolio_snapshot_summary import assemble_summary
        return assemble_summary(investor, quarter, resolved=resolved,
                                one_pager_provider=op)
    if name == "financial":
        from flask_app.services.portfolio_snapshot_financial import assemble_financial
        return assemble_financial(investor, quarter, resolved=resolved,
                                  one_pager_provider=op)
    if name == "operating":
        from flask_app.services.portfolio_snapshot_operating import assemble_operating
        return assemble_operating(investor, quarter, resolved=resolved,
                                  one_pager_provider=op)
    if name == "loan":
        from flask_app.services.portfolio_snapshot_loan import assemble_loan
        return assemble_loan(
            investor, quarter, resolved=resolved, one_pager_provider=op,
            loans=data.get("mri_loans_raw"), valuations=data.get("mri_val"),
            inv=data["inv"],
            quarterly_noi_provider=(quarterly_noi_provider
                                    or _quarterly_noi_provider(data)))
    raise ValueError(f"unknown subtab {name!r}")


SUBTABS = ("summary", "financial", "operating", "loan")


def assemble_full_report(investor: str, quarter: str,
                         data: Optional[dict] = None) -> dict:
    """The complete report: resolution + all four subtabs.

    A per-subtab failure lands in ``errors`` rather than failing the whole
    report — three good subtabs beat none, and the shell renders the error in
    place of that tab body.
    """
    from flask_app.services import data_service
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals

    if data is None:
        data = data_service.get_data()

    resolved = resolve_investor_deals(
        investor, quarter, data.get("relationships_raw"), data["inv"])

    # ONE provider pair for all four subtabs. They memoise per instance, so
    # building them per subtab computed each deal's One Pager four times over
    # (128 calls for 32 deals on TIAA/26Q1). Output is byte-identical either
    # way — verified on all four subtabs — this is purely the cost.
    one_pager = _one_pager_provider(data)
    quarterly_noi = _quarterly_noi_provider(data)

    subtabs: dict = {}
    errors: dict = {}
    for name in SUBTABS:
        try:
            subtabs[name] = build_subtab(
                name, investor, quarter, data, resolved,
                one_pager_provider=one_pager,
                quarterly_noi_provider=quarterly_noi)
        except Exception as exc:
            log.exception("subtab %s failed for %s %s", name, investor, quarter)
            errors[name] = str(exc)

    return {
        "subtabs": subtabs,
        "errors": errors,
        "resolution": {
            "investor_name": resolved.get("investor_name"),
            "quarter_end": resolved.get("quarter_end"),
            "diagnostics": resolved.get("diagnostics"),
            "flagged": resolved.get("flagged"),
            "excluded_sold": resolved.get("excluded_sold"),
            "excluded_not_acquired": resolved.get("excluded_not_acquired"),
            "excluded_children": resolved.get("excluded_children"),
        },
        "_resolved": resolved,
    }


def _data_version() -> str:
    """Whatever identifies the data behind a freeze, best-effort.

    Recorded so a frozen payload can be traced back to the build and cutoff that
    produced it. Never allowed to break a freeze.
    """
    try:
        from flask import current_app
        build = current_app.config.get("BUILD_HASH", "?")
        actuals = current_app.config.get("ACTUALS_THROUGH", "?")
        return f"build={build};actuals_through={actuals}"
    except Exception:
        return "unknown"


# ── freeze / read ─────────────────────────────────────────────────────────

def freeze(investor_code: str, quarter: str, approved_by: str,
           assembler: Optional[Callable] = None,
           elements_loader: Optional[Callable] = None) -> dict:
    """Capture the complete report as approved, and store it.

    Freezes BOTH halves of the report: the four assembled subtabs (computed
    metrics) and the approved editable content (comments, footnotes, the manual
    Net ROE / ITD values) exactly as they stood at approval.

    ``assembler`` / ``elements_loader`` exist for the self-test; production uses
    the defaults. Raises on failure — the CALLER wraps, so that an approval is
    never lost to a snapshot write.
    """
    from flask_app.serializers import safe_json

    assemble = assembler or assemble_full_report
    load_elements = elements_loader
    if load_elements is None:
        from flask_app.services.portfolio_snapshot_persistence import load_page
        load_elements = load_page

    report = assemble(investor_code, quarter) or {}
    report.pop("_resolved", None)          # not part of the frozen contract
    elements = load_elements(investor_code, quarter) or {}

    payload = safe_json({
        "subtabs": report.get("subtabs") or {},
        "errors": report.get("errors") or {},
        "resolution": report.get("resolution") or {},
        # The approved editable content, frozen alongside the metrics.
        "elements": {
            "comments": elements.get("comments") or [],
            "footnotes": elements.get("footnotes") or [],
            "values": elements.get("values") or [],
        },
    })
    blob = json.dumps(payload)
    version = _data_version()

    _ensure_table()
    with _engine().begin() as conn:
        # DELETE-then-INSERT: cross-DB, and re-approval overwrites by design.
        conn.execute(text(f"DELETE FROM {_TABLE} "
                          f"WHERE investor_code = :i AND quarter = :q"),
                     {"i": investor_code, "q": quarter})
        conn.execute(text(f"""
            INSERT INTO {_TABLE}
                (investor_code, quarter, payload, approved_by, data_version)
            VALUES (:i, :q, :p, :by, :v)
        """), {"i": investor_code, "q": quarter, "p": blob,
               "by": approved_by, "v": version})

    log.info("Froze Portfolio Snapshot for %s %s (%s)",
             investor_code, quarter, version)
    return {"investor_code": investor_code, "quarter": quarter,
            "approved_by": approved_by, "data_version": version,
            "bytes": len(blob)}


def get_frozen(investor_code: str, quarter: str) -> Optional[dict]:
    """The frozen payload for a page, or None."""
    try:
        _ensure_table()
        with _engine().connect() as conn:
            row = conn.execute(text(f"""
                SELECT payload, approved_by, approved_at, data_version
                FROM {_TABLE}
                WHERE investor_code = :i AND quarter = :q
            """), {"i": investor_code, "q": quarter}).mappings().fetchone()
    except Exception as exc:
        log.exception("reading frozen snapshot failed")
        return None
    if not row:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:
        log.exception("frozen payload for %s %s is not valid JSON",
                      investor_code, quarter)
        return None
    approved_at = row["approved_at"]
    return {
        "payload": payload,
        "approved_by": row["approved_by"],
        "approved_at": (approved_at.isoformat()
                        if hasattr(approved_at, "isoformat")
                        else (str(approved_at) if approved_at else None)),
        "data_version": row["data_version"],
    }


def delete_frozen(investor_code: str, quarter: str) -> None:
    """Drop a frozen payload. Not used by the pipeline — reopening relies on
    ``status != 'approved'`` making it unreachable — but available for cleanup."""
    _ensure_table()
    with _engine().begin() as conn:
        conn.execute(text(f"DELETE FROM {_TABLE} "
                          f"WHERE investor_code = :i AND quarter = :q"),
                     {"i": investor_code, "q": quarter})


def load_report(investor_code: str, quarter: str,
                status: Optional[str] = None,
                assembler: Optional[Callable] = None,
                frozen_getter: Optional[Callable] = None,
                status_getter: Optional[Callable] = None) -> dict:
    """The report to serve, frozen when approved and live otherwise.

    Returns the report dict with ``source`` set to ``"frozen"`` or ``"live"``,
    plus ``approved_by`` / ``approved_at`` / ``data_version`` when frozen.

    An approved page with no frozen payload (approved before this feature
    existed, or a freeze that failed) falls back to live and says so in
    ``source_note`` rather than showing an empty report.
    """
    get_frozen_fn = frozen_getter or get_frozen
    assemble = assembler or assemble_full_report

    if status is None:
        if status_getter is None:
            from flask_app.services.portfolio_snapshot_persistence import (
                document_status)
            status_getter = lambda i, q: (document_status(i, q) or {}).get("status")
        try:
            status = status_getter(investor_code, quarter)
        except Exception:
            status = None

    if status == "approved":
        frozen = get_frozen_fn(investor_code, quarter)
        if frozen:
            out = dict(frozen["payload"])
            out["source"] = SOURCE_FROZEN
            out["source_note"] = (
                f"Approved snapshot — frozen at approval, not recomputed. "
                f"Approved by {frozen.get('approved_by') or 'unknown'}"
                + (f" on {str(frozen['approved_at'])[:10]}"
                   if frozen.get("approved_at") else ""))
            out["approved_by"] = frozen.get("approved_by")
            out["approved_at"] = frozen.get("approved_at")
            out["data_version"] = frozen.get("data_version")
            return out
        out = assemble(investor_code, quarter) or {}
        out.pop("_resolved", None)
        out["source"] = SOURCE_LIVE
        out["source_note"] = (
            "This report is approved but has no frozen payload, so it is being "
            "recomputed live and may not match what was approved.")
        return out

    out = assemble(investor_code, quarter) or {}
    out.pop("_resolved", None)
    out["source"] = SOURCE_LIVE
    out["source_note"] = "In progress — computed live from current data."
    return out


# ── Self-test ─────────────────────────────────────────────────────────────

def _selftest():                                    # pragma: no cover
    """Prove an approved report cannot move when live data moves.

    Runs entirely on a scratch SQLite database with an injected assembler, so
    the drift test is deterministic: 'live data changed' is modelled by the
    assembler returning a different number on the next call, which is exactly
    what a corrected MRI ownership row does in production.
    """
    import os
    import tempfile

    import sqlalchemy

    from flask_app.services import portfolio_snapshot_persistence as P

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print("    [" + ("PASS" if cond else "FAIL") + "] " + label)

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_freeze_"), "t.db")
    eng = sqlalchemy.create_engine("sqlite:///" + tmp)

    # Run via `python -m`, this file is __main__ — so importing it by its real
    # dotted name yields a SECOND module object, and that is the one
    # persistence's lazy `from ...freeze import freeze` resolves. Everything the
    # test touches must therefore go through `F`, not through __main__'s copy,
    # or the freeze writes to one engine while the test reads another.
    import flask_app.services.portfolio_snapshot_freeze as F

    F._engine = lambda: eng                          # type: ignore[assignment]
    F._is_postgres = lambda: False                   # type: ignore[assignment]
    P._engine = lambda: eng                          # type: ignore[assignment]
    P._is_postgres = lambda: False                   # type: ignore[assignment]

    get_frozen = F.get_frozen
    load_report = F.load_report
    delete_frozen = F.delete_frozen
    SOURCE_FROZEN, SOURCE_LIVE = F.SOURCE_FROZEN, F.SOURCE_LIVE

    INV, Q = "TGAM", "2026-Q1"
    ROLES = ["asset_manager", "head_am", "president", "cco", "ceo"]

    # A mutable stand-in for live data. `pct` is what a corrected MRI ownership
    # row changes; 1.00 -> 0.90 is exactly the 45th & Main move.
    live = {"pct": 1.00, "funded": 18_550_000.0}

    def assembler(investor, quarter):
        return {
            "subtabs": {
                "summary": {"asset_allocation": {
                    "total_funded": live["funded"] * live["pct"]}},
                "financial": {"groups": {"TGA24": {"deals": [
                    {"vcode": "P0000089", "name": "45th & Main",
                     "pct_of_pref": live["pct"],
                     "invested": live["funded"] * live["pct"]}]}}},
                "operating": {"groups": {}},
                "loan": {"groups": {}},
            },
            "errors": {},
            "resolution": {"investor_name": "TIAA", "quarter_end": quarter},
        }

    def elements_loader(investor, quarter):
        return {"comments": [{"field": "narrative_1",
                              "comment_text": "Approved commentary."}],
                "footnotes": [{"number": 1, "anchor": "invested",
                               "text": "Net of the fee allocation."}],
                "values": [{"deal_vcode": "P0000089", "field": "net_roe",
                            "value": 0.0912}]}

    def status_of(i, q):
        return (P.document_status(i, q) or {}).get("status")

    def load(**kw):
        return load_report(INV, Q, assembler=assembler,
                           status_getter=status_of, **kw)

    def _raises(fn) -> bool:
        try:
            fn()
            return False
        except Exception:
            return True

    # Persistence's approve() resolves `freeze` off the module at call time, so
    # patching it here makes the PRODUCTION trigger path run with the test
    # assembler. The freeze is exercised through approve(), never called direct.
    _real_freeze = F.freeze
    F.freeze = lambda i, q, by, **kw: _real_freeze(
        i, q, by, assembler=assembler, elements_loader=elements_loader)

    print("=" * 100)
    print("SNAPSHOT FREEZE - an approved report must not move when live data does")
    print("=" * 100)

    # Seed one editable element so the page exists and can be submitted.
    P.save_comment(INV, Q, "report", "narrative_1", "Approved commentary.",
                   updated_by="am")

    # ---- draft computes live ----
    rep = load()
    chk("draft report computes LIVE", rep["source"] == SOURCE_LIVE)
    chk("draft carries a source note", bool(rep.get("source_note")))
    chk("draft shows the current pct (1.00)",
        rep["subtabs"]["financial"]["groups"]["TGA24"]["deals"][0]
        ["pct_of_pref"] == 1.00)

    # ---- approve: freeze fires on the final transition only ----
    P.submit_for_review(INV, Q, 1, "am", ROLES)
    chk("mid-pipeline report is still LIVE", load()["source"] == SOURCE_LIVE)
    for _ in range(3):                     # head_am, president, cco
        P.approve(INV, Q, 1, "u", ROLES)
    chk("no frozen payload before the final approval",
        get_frozen(INV, Q) is None)

    # the CEO step — approve() must fire the freeze itself
    P.approve(INV, Q, 1, "ceo-user", ROLES)
    chk("status is approved", status_of(INV, Q) == "approved")
    chk("approve() itself fired the freeze (production trigger path)",
        get_frozen(INV, Q) is not None)

    frozen = get_frozen(INV, Q)
    chk("a frozen payload exists after approval", frozen is not None)
    chk("frozen payload captures all four subtabs",
        set((frozen or {}).get("payload", {}).get("subtabs", {})) ==
        {"summary", "financial", "operating", "loan"})
    chk("frozen payload captures the approved comments",
        bool(frozen["payload"]["elements"]["comments"]))
    chk("frozen payload captures the approved footnotes",
        bool(frozen["payload"]["elements"]["footnotes"]))
    chk("frozen payload captures the approved Net ROE / ITD values",
        frozen["payload"]["elements"]["values"][0]["value"] == 0.0912)
    chk("frozen payload records the data version",
        bool(frozen.get("data_version")))

    rep = load()
    chk("approved report serves FROZEN by default", rep["source"] == SOURCE_FROZEN)
    chk("frozen report reports its approver", rep.get("approved_by") == "ceo-user")

    # ================= THE KEY TEST =================
    print("\n  --- live data now MOVES: pct 1.00 -> 0.90 "
          "(the 45th & Main correction) ---")
    live["pct"] = 0.90

    live_now = assembler(INV, Q)["subtabs"]["financial"]["groups"]["TGA24"]["deals"][0]
    print(f"      live would now say pct={live_now['pct_of_pref']} "
          f"invested={live_now['invested']:,.0f}")

    rep = load()
    served = rep["subtabs"]["financial"]["groups"]["TGA24"]["deals"][0]
    print(f"      approved report serves pct={served['pct_of_pref']} "
          f"invested={served['invested']:,.0f}  (source={rep['source']})")

    chk("KEY: approved report still serves the APPROVED pct (1.00), not 0.90",
        served["pct_of_pref"] == 1.00)
    chk("KEY: approved report still serves the APPROVED invested figure",
        served["invested"] == 18_550_000.0)
    chk("KEY: approved report did NOT drift with live data",
        rep["source"] == SOURCE_FROZEN and
        rep["subtabs"]["summary"]["asset_allocation"]["total_funded"]
        == 18_550_000.0)

    # ---- reopen: automatic unfreeze via approved_at = NULL ----
    # reject() CANNOT do this: at the approved step _step_for(5)["role"] is
    # None, so its role check raises. reopen() was added for exactly this.
    chk("reject() refuses to reopen an approved page (reopen() is required)",
        _raises(lambda: P.reject(INV, Q, 1, "head", note_text="x",
                                 roles=ROLES)))
    chk("reopen() requires the ceo role",
        _raises(lambda: P.reopen(INV, Q, 1, "am", "note", ["asset_manager"])))
    chk("reopen() requires a note",
        _raises(lambda: P.reopen(INV, Q, 1, "ceo", "", ROLES)))

    P.reopen(INV, Q, 1, "ceo-user", "Reopening for a correction.", roles=ROLES)
    chk("reopened report status is 'returned'", status_of(INV, Q) == "returned")
    rep = load()
    chk("reopened report falls back to LIVE automatically",
        rep["source"] == SOURCE_LIVE)
    chk("reopened report now shows the CHANGED live pct (0.90)",
        rep["subtabs"]["financial"]["groups"]["TGA24"]["deals"][0]
        ["pct_of_pref"] == 0.90)
    chk("the old frozen row still exists but is unreachable while not approved",
        get_frozen(INV, Q) is not None)

    # ---- re-approve: new freeze overwrites the old ----
    P.submit_for_review(INV, Q, 1, "am", ROLES)
    for _ in range(3):
        P.approve(INV, Q, 1, "u", ROLES)
    P.approve(INV, Q, 1, "ceo-user-2", ROLES)
    refrozen = get_frozen(INV, Q)
    chk("re-approval overwrites the frozen payload (one row, new approver)",
        refrozen["approved_by"] == "ceo-user-2")
    chk("the NEW freeze captured the changed value (0.90)",
        refrozen["payload"]["subtabs"]["financial"]["groups"]["TGA24"]
        ["deals"][0]["pct_of_pref"] == 0.90)
    with eng.connect() as conn:
        n = conn.execute(sqlalchemy.text(
            "SELECT COUNT(*) FROM portfolio_snapshot_frozen "
            "WHERE investor_code = :i AND quarter = :q"),
            {"i": INV, "q": Q}).scalar()
    chk("exactly one frozen row per (investor, quarter)", n == 1)

    rep = load()
    chk("re-approved report serves the NEW frozen values",
        rep["source"] == SOURCE_FROZEN and
        rep["subtabs"]["financial"]["groups"]["TGA24"]["deals"][0]
        ["pct_of_pref"] == 0.90)

    # ---- a freeze failure must not block an approval ----
    P.reopen(INV, Q, 1, "ceo-user", "again", roles=ROLES)
    P.submit_for_review(INV, Q, 1, "am", ROLES)
    for _ in range(3):
        P.approve(INV, Q, 1, "u", ROLES)

    def exploding(i, q, by, **kw):
        raise RuntimeError("injected freeze failure")

    patched = F.freeze
    F.freeze = exploding
    try:
        P.approve(INV, Q, 1, "ceo-user-3", ROLES)      # must NOT raise
        approved_despite = status_of(INV, Q) == "approved"
    except Exception:
        approved_despite = False
    finally:
        F.freeze = patched
    chk("a freeze failure does NOT block the approval", approved_despite)

    # the stale frozen row is still there; the read path serves it and would
    # otherwise fall back to live with a warning note
    rep = load()
    chk("after a failed freeze the report still renders",
        rep.get("source") in (SOURCE_FROZEN, SOURCE_LIVE))

    # ---- approved with NO frozen payload falls back to live, loudly ----
    delete_frozen(INV, Q)
    rep = load()
    chk("approved-but-unfrozen falls back to LIVE", rep["source"] == SOURCE_LIVE)
    chk("...and says so in source_note",
        "no frozen payload" in (rep.get("source_note") or ""))

    # ---- a corrupt payload must not crash the read ----
    with eng.begin() as conn:
        conn.execute(sqlalchemy.text(
            "INSERT INTO portfolio_snapshot_frozen "
            "(investor_code, quarter, payload, approved_by) "
            "VALUES (:i, :q, 'not json', 'x')"), {"i": INV, "q": "2099-Q9"})
    chk("a corrupt frozen payload returns None rather than raising",
        get_frozen(INV, "2099-Q9") is None)

    F.freeze = _real_freeze          # never leave the module patched

    print("\n" + "=" * 100)
    passed = sum(1 for _, ok in checks if ok)
    print("RESULT: " + str(passed) + "/" + str(len(checks)) + " checks passed")
    for label, ok in checks:
        if not ok:
            print("  FAILED: " + label)
    print("=" * 100)
    return passed == len(checks)


if __name__ == "__main__":                          # pragma: no cover
    import sys
    sys.exit(0 if _selftest() else 1)
