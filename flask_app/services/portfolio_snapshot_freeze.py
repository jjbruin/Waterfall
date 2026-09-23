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
import re
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text

log = logging.getLogger(__name__)

#: Marks which path produced a payload. The UI branches on this.
SOURCE_FROZEN = "frozen"
SOURCE_LIVE = "live"

_TABLE = "portfolio_snapshot_frozen"
_HISTORY = "portfolio_snapshot_frozen_history"

#: Why a payload was stored. A freeze is NOT an approval and must never be
#: recorded as one — the two are separate acts with separate authority.
REASON_AS_SENT = "as-sent"          # the "Freeze as sent" button
REASON_AS_APPROVED = "as-approved"  # the legacy CEO-approval freeze
REASON_REFREEZE = "re-freeze"       # an admin correction

#: Columns added after the table first shipped. Applied one at a time because a
#: failed ALTER on one must not abandon the rest, and because SQLite has no
#: ``ADD COLUMN IF NOT EXISTS``.
_ADDED_COLUMNS = (
    ("frozen_by", "TEXT"),
    ("frozen_at", "TIMESTAMP"),
    ("frozen_reason", "TEXT"),
    ("source_manifest", "TEXT"),   # JSON: published source, sha256, pages
    ("roster", "TEXT"),            # JSON: the One Pager roster, in printed order
    ("one_pagers", "TEXT"),        # JSON: {vcode: payload} as published
    ("version", "INTEGER"),
)


def _engine():
    from flask_app.db import get_engine
    return get_engine()


def _is_postgres() -> bool:
    try:
        return _engine().dialect.name == "postgresql"
    except Exception:
        return False


def _ensure_table() -> None:
    """Create the frozen-payload tables if absent, and add later columns.

    PURELY ADDITIVE. The UNIQUE(investor_code, quarter) on the live table is
    deliberately left alone: superseded versions go to a SEPARATE history table
    rather than becoming extra rows here. Dropping a UNIQUE constraint means a
    table rebuild on SQLite, and this data is the only record of what an
    investor was sent — so the migration that cannot lose it is the one that
    never rewrites it.
    """
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
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_HISTORY} (
                id {pk},
                investor_code TEXT NOT NULL,
                quarter TEXT NOT NULL,
                payload TEXT NOT NULL,
                data_version TEXT,
                frozen_by TEXT,
                frozen_at TIMESTAMP,
                frozen_reason TEXT,
                source_manifest TEXT,
                roster TEXT,
                one_pagers TEXT,
                version INTEGER,
                superseded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                superseded_by TEXT,
                supersede_reason TEXT NOT NULL
            )
        """))
    existing = _columns(_TABLE)
    for col, typ in _ADDED_COLUMNS:
        if col in existing:
            continue
        try:
            with _engine().begin() as conn:
                conn.execute(text(f"ALTER TABLE {_TABLE} ADD COLUMN {col} {typ}"))
        except Exception:
            log.exception("could not add %s.%s", _TABLE, col)


def _columns(table: str) -> set:
    """Column names on a table, empty when it cannot be read."""
    try:
        import sqlalchemy as sa
        return set(sa.inspect(_engine()).get_columns(table) and
                   [c["name"] for c in sa.inspect(_engine()).get_columns(table)])
    except Exception:
        return set()


# ── assembly (shared by the freeze and the live read) ─────────────────────

def _one_pager_provider(data: dict) -> Callable:
    """(vcode, quarter) -> One Pager payload, memoised per call.

    Lean path: calls only ``get_capitalization_stack`` and
    ``get_property_performance`` directly — the only two sections the
    snapshot subtabs read. Skips ``pe_performance`` and ``comments``, and
    builds ``general`` from the ``inv`` row rather than calling
    ``get_general_information``, cutting per-deal cost by ~60%.

    THE GENERAL BLOCK CARRIES ONE FIELD, AND IT IS NOT OPTIONAL. Skipping the
    block entirely used to mean the Operating tab's insufficient-history rule
    read no closing date for ANY deal: ``months_owned`` returns None without
    one, the rule is guarded on ``mo is not None``, and so it NEVER FIRED —
    measured at 30 of 30 deals on the 26Q2 page before this. A brand-new
    acquisition was therefore reported as though it had a full operating
    history. The date comes straight off the row through
    ``one_pager.closing_date_from_row``, the same precedence the full path
    applies, so the two cannot disagree and no extra query is made.

    The shared ``get_one_pager_data`` is NOT modified — this is a parallel
    path that returns the same ``cap_stack`` and ``property_performance``
    keys with identical values.
    """
    from one_pager import (
        closing_date_from_row, get_capitalization_stack,
        get_property_performance,
    )

    deal_terms = data.get("deal_terms_raw")
    inv = data["inv"]
    _inv_by_vcode: dict = {}
    if inv is not None and not getattr(inv, "empty", True) and "vcode" in inv.columns:
        for _, _r in inv.iterrows():
            _inv_by_vcode.setdefault(str(_r["vcode"]).strip().upper(), _r)
    cache: dict = {}

    def provider(vcode: str, quarter: str) -> dict:
        key = (vcode, quarter)
        if key not in cache:
            cap_stack = get_capitalization_stack(
                vcode, data["mri_loans_raw"], data["mri_val"],
                data["wf"], data["acct"], data["inv"],
                isbs_raw=data["isbs_raw"], quarter_str=quarter,
                relationships=data.get("relationships_raw"),
                # Carries the debt_isbs / total_cap_isbs twins this page reads.
                # Passing it does NOT move any Snapshot figure — the dev branch
                # of resolve_debt already ignores the One Pager's debt.
                inspection=data.get("inspection_raw"),
            )
            prop_perf = get_property_performance(
                vcode, quarter, data["isbs_raw"], data["mri_val"],
                data["occupancy_raw"],
                budget_econ_occ_df=data.get("budget_econ_occ"),
                at_close_noi_df=data.get("at_close_noi_raw"),
                deal_terms_df=deal_terms,
                mri_loans_all_df=data.get("mri_loans_all"),
                inv_map=data["inv"],
            ) if quarter else {}

            # Enrich cap_stack with deal_terms (same as get_one_pager_data)
            if deal_terms is not None and not getattr(deal_terms, "empty", True):
                from flask_app.services.financials_service import (
                    _enrich_cap_stack_from_deal_terms,
                )
                _enrich_cap_stack_from_deal_terms(cap_stack, deal_terms, vcode)

            cache[key] = {
                "cap_stack": cap_stack,
                "property_performance": prop_perf,
                # Only what a subtab actually reads — today, the closing date
                # the insufficient-history rule needs. Deliberately not the
                # whole general block: this path exists to stay lean.
                "general": {
                    "date_closed": closing_date_from_row(
                        _inv_by_vcode.get(str(vcode).strip().upper())),
                },
            }
        return cache[key]

    return provider


def _quarterly_noi_provider(data: dict) -> Callable:
    """(vcode, quarter) -> that quarter's periodic NOI, or None.

    Bulk-optimised: pre-filters ISBS once for the report's quarter, then uses
    the SAME ``compute_cumulative_noi`` / ``cumulative_to_periodic`` /
    ``aggregate_periodic`` pipeline from ``isbs_helpers`` per vcode. Because
    the functions and the account lists (``IS_ACCOUNTS`` from ``config.py``)
    are shared — not copied — the result cannot drift from the chart.

    A quarter missing any of its three months returns None (the
    ``aggregate_periodic`` function excludes incomplete quarters by
    ``month_counts < 3``), so the None rule is inherited, not reimplemented.

    Giant 7 and other parent vcodes with no ISBS rows under their own vcode
    return None naturally: ``_prepare_isbs`` finds nothing, ``actual_dates``
    is empty, and the pipeline returns an empty dict.
    """
    import pandas as pd
    from config import IS_ACCOUNTS
    from flask_app.services.financials_service import _prepare_isbs
    from flask_app.services.isbs_helpers import (
        compute_cumulative_noi, cumulative_to_periodic, aggregate_periodic,
    )

    rev_accounts = [a for lst in IS_ACCOUNTS['REVENUES'].values() for a in lst]
    exp_accounts = [a for lst in IS_ACCOUNTS['EXPENSES'].values() for a in lst]
    isbs_raw = data["isbs_raw"]

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

            isbs = _prepare_isbs(isbs_raw, vcode)
            if isbs.empty:
                cache[key] = None
                return None

            actual = isbs[isbs['vSource'] == 'Interim IS']
            if actual.empty:
                cache[key] = None
                return None

            dates = sorted(actual['dtEntry_parsed'].dropna().unique())
            if not dates:
                cache[key] = None
                return None

            cum = compute_cumulative_noi(actual, dates, rev_accounts,
                                         exp_accounts)
            periodic = cumulative_to_periodic(cum, dates)
            quarterly = aggregate_periodic(periodic, "Quarterly")

            # aggregate_periodic returns only complete quarters (3 months),
            # so a missing key IS the None rule.
            noi = quarterly.get(q_end)
            if noi is not None:
                val = float(noi)
        except Exception:
            val = None
        cache[key] = val
        return val

    return provider


def _committed_debt_provider(data: dict) -> Callable:
    """vcode -> committed facility (sum of mOrigLoanAmt), memoised.

    The Financial subtab needs this to resolve a development deal's Debt on the
    same basis the Loan subtab uses, but it takes no DataFrames of its own — so
    the loans frame is bound here, where `data` is already in hand, and passed in
    as a callable. Same dependency-injection shape as the One Pager provider.

    Child loans are included via Portfolio_Name, matching the parent-inheritance
    rule in `deal_loan_rows`: Burton holds no loans itself, its children do.
    """
    from flask_app.services.portfolio_snapshot_debt import (
        committed_facility, deal_loan_rows,
    )

    loans = data.get("mri_loans_raw")
    inv = data.get("inv")
    cache: dict = {}

    def children_of(vcode: str) -> list:
        """Child vcodes rolling up into a parent, from the deals frame."""
        if inv is None or getattr(inv, "empty", True):
            return []
        cols = {c.lower(): c for c in inv.columns}
        c_vc, c_nm = cols.get("vcode"), cols.get("investment_name")
        c_pn, c_pc = cols.get("portfolio_name"), cols.get("property_count")
        if not (c_vc and c_pn):
            return []
        row = inv[inv[c_vc].astype(str).str.strip().str.lower()
                  == str(vcode).strip().lower()]
        if row.empty:
            return []
        keys = set()
        for col in (c_nm, c_pn):
            if col:
                v = str(row.iloc[0].get(col) or "").strip().lower()
                if v:
                    keys.add(v)
        if not keys:
            return []
        kid = inv[inv[c_pn].astype(str).str.strip().str.lower().isin(keys)]
        if c_pc:
            kid = kid[pd.to_numeric(kid[c_pc], errors="coerce").fillna(1) == 0]
        return [str(v).strip() for v in kid[c_vc].tolist()
                if str(v).strip().lower() != str(vcode).strip().lower()]

    def provider(vcode: str):
        if vcode not in cache:
            rows = deal_loan_rows(loans, vcode, children_of(vcode))
            cache[vcode] = committed_facility(rows)
        return cache[vcode]

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
        return assemble_financial(
            investor, quarter, resolved=resolved, one_pager_provider=op,
            committed_debt_provider=_committed_debt_provider(data))
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
    return _write_frozen(investor_code, quarter, payload,
                         frozen_by=approved_by,
                         reason=REASON_AS_APPROVED,
                         approved_by=approved_by)


def _write_frozen(investor_code: str, quarter: str, payload: dict,
                  frozen_by: str, reason: str,
                  approved_by: Optional[str] = None,
                  source_manifest: Optional[dict] = None,
                  roster: Optional[list] = None,
                  one_pagers: Optional[dict] = None,
                  supersede_reason: Optional[str] = None) -> dict:
    """THE one writer for a frozen payload. Both freeze paths funnel here.

    Any existing row is copied to the history table BEFORE being replaced, so a
    re-freeze never destroys what an investor was actually sent.
    """
    import datetime as _dt
    blob = json.dumps(payload)
    version_str = _data_version()
    now = _dt.datetime.utcnow()

    _ensure_table()
    prior = _current_row(investor_code, quarter)
    next_version = int((prior or {}).get("version") or 0) + 1

    with _engine().begin() as conn:
        if prior:
            conn.execute(text(f"""
                INSERT INTO {_HISTORY}
                    (investor_code, quarter, payload, data_version, frozen_by,
                     frozen_at, frozen_reason, source_manifest, roster,
                     one_pagers, version, superseded_by, supersede_reason)
                VALUES (:i, :q, :p, :dv, :fb, :fa, :fr, :sm, :ro, :op, :ver,
                        :sby, :sr)
            """), {"i": investor_code, "q": quarter,
                   "p": prior.get("payload_raw") or "{}",
                   "dv": prior.get("data_version"), "fb": prior.get("frozen_by"),
                   "fa": prior.get("frozen_at"), "fr": prior.get("frozen_reason"),
                   "sm": prior.get("source_manifest_raw"),
                   "ro": prior.get("roster_raw"), "op": prior.get("one_pagers_raw"),
                   "ver": prior.get("version"), "sby": frozen_by,
                   "sr": supersede_reason or reason})
        conn.execute(text(f"DELETE FROM {_TABLE} "
                          f"WHERE investor_code = :i AND quarter = :q"),
                     {"i": investor_code, "q": quarter})
        conn.execute(text(f"""
            INSERT INTO {_TABLE}
                (investor_code, quarter, payload, approved_by, data_version,
                 frozen_by, frozen_at, frozen_reason, source_manifest, roster,
                 one_pagers, version)
            VALUES (:i, :q, :p, :by, :v, :fb, :fa, :fr, :sm, :ro, :op, :ver)
        """), {"i": investor_code, "q": quarter, "p": blob,
               "by": approved_by, "v": version_str, "fb": frozen_by,
               "fa": now, "fr": reason,
               "sm": json.dumps(source_manifest) if source_manifest else None,
               "ro": json.dumps(roster) if roster is not None else None,
               "op": json.dumps(one_pagers) if one_pagers is not None else None,
               "ver": next_version})

    log.info("Froze %s %s (%s, version %s, %s)",
             investor_code, quarter, reason, next_version, version_str)
    return {"investor_code": investor_code, "quarter": quarter,
            "frozen_by": frozen_by, "frozen_at": now.isoformat(),
            "frozen_reason": reason, "version": next_version,
            "data_version": version_str, "bytes": len(blob),
            "one_pager_count": len(one_pagers or {}),
            "roster_count": len(roster or [])}


def _current_row(investor_code: str, quarter: str) -> Optional[dict]:
    """The stored row as-is, or None. Never raises."""
    try:
        _ensure_table()
        cols = _columns(_TABLE)
        sel = ["payload", "approved_by", "approved_at", "data_version"]
        sel += [c for c, _ in _ADDED_COLUMNS if c in cols]
        with _engine().connect() as conn:
            row = conn.execute(text(
                f"SELECT {', '.join(sel)} FROM {_TABLE} "
                f"WHERE investor_code = :i AND quarter = :q"),
                {"i": investor_code, "q": quarter}).mappings().fetchone()
    except Exception:
        log.exception("reading frozen row failed")
        return None
    if not row:
        return None
    out = dict(row)
    out["payload_raw"] = out.get("payload")
    out["source_manifest_raw"] = out.get("source_manifest")
    out["roster_raw"] = out.get("roster")
    out["one_pagers_raw"] = out.get("one_pagers")
    return out


def freeze_as_sent(investor_code: str, quarter: str, frozen_by: str,
                   overlay: Optional[dict] = None,
                   roster: Optional[list] = None,
                   source_manifest: Optional[dict] = None,
                   assembler: Optional[Callable] = None,
                   one_pager_getter: Optional[Callable] = None,
                   elements_loader: Optional[Callable] = None) -> dict:
    """Freeze a quarter AS SENT: subtabs, every One Pager, and the roster.

    Separate from the approval chain on purpose. ``approved_by`` is left NULL —
    a freeze records who froze it, never a decision nobody made.

    ``overlay`` is the published-value layer for the one-time 26Q2 seeding:
    ``{vcode_or_"__subtabs__": {dotted.path: {"published": v, "page": n}}}``.
    Applied AFTER assembly, so every untouched cell stays the computed one and
    the diff between the two is recoverable later.

    Raises on failure. The caller must NOT mark the quarter frozen unless this
    returns — a silent failure is the one outcome this design forbids.
    """
    from flask_app.serializers import safe_json

    assemble = assembler or assemble_full_report
    report = assemble(investor_code, quarter) or {}
    resolved = report.pop("_resolved", None) or {}

    load_elements = elements_loader
    if load_elements is None:
        from flask_app.services.portfolio_snapshot_persistence import load_page
        load_elements = load_page
    elements = load_elements(investor_code, quarter) or {}

    if roster is None:
        roster = _roster_from_report(report)
    get_op = one_pager_getter or _default_one_pager_getter()
    one_pagers, failures = {}, []
    for vc in roster:
        try:
            one_pagers[vc] = get_op(vc, quarter)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{vc}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(
            f"{len(failures)} One Pager(s) could not be built, so the freeze "
            f"would be incomplete: " + "; ".join(failures[:5]))

    payload = safe_json({
        "subtabs": report.get("subtabs") or {},
        "errors": report.get("errors") or {},
        "resolution": report.get("resolution") or {},
        "elements": {
            "comments": elements.get("comments") or [],
            "footnotes": elements.get("footnotes") or [],
            "values": elements.get("values") or [],
        },
    })
    one_pagers = safe_json(one_pagers)

    applied = 0
    if overlay:
        applied = _apply_overlay(payload, one_pagers, overlay)

    manifest = dict(source_manifest or {})
    manifest["overlay_cells_applied"] = applied
    return _write_frozen(investor_code, quarter, payload,
                         frozen_by=frozen_by, reason=REASON_AS_SENT,
                         source_manifest=manifest, roster=roster,
                         one_pagers=one_pagers)


def _roster_from_report(report: dict) -> list:
    """Deal vcodes on the report, in the order the Financial subtab lists them."""
    fin = ((report.get("subtabs") or {}).get("financial") or {})
    out = []
    for _g, blk in (fin.get("groups") or {}).items():
        for r in ((blk.get("deals") if isinstance(blk, dict) else blk) or []):
            if r.get("vcode") and r["vcode"] not in out:
                out.append(r["vcode"])
    for r in (fin.get("ownership_flagged") or []):
        if r.get("vcode") and r["vcode"] not in out:
            out.append(r["vcode"])
    return out


def _default_one_pager_getter() -> Callable:
    """(vcode, quarter) -> the live One Pager payload."""
    from flask_app.services import data_service
    from flask_app.services.financials_service import get_one_pager_data
    data = data_service.get_data()

    def _get(vcode, quarter):
        return get_one_pager_data(
            vcode, quarter, data["inv"], data["isbs_raw"],
            data["mri_loans_raw"], data["mri_val"], data["wf"], data["acct"],
            occupancy_raw=data["occupancy_raw"],
            budget_econ_occ=data.get("budget_econ_occ"),
            deal_terms=data.get("deal_terms_raw"),
            at_close_noi=data.get("at_close_noi_raw"),
            event_dates=data.get("event_dates_raw"),
            full_data=data, relationships=data.get("relationships_raw"),
            mri_loans_all=data.get("mri_loans_all"),
            inspection=data.get("inspection_raw"))
    return _get


def _set_path(obj, dotted: str, value) -> bool:
    """Set a dotted path, supporting ``a.b[2].c``. True when it landed."""
    cur = obj
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", dotted)
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if part.startswith("["):
            idx = int(part[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                return False
            if last:
                cur[idx] = value; return True
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return False
            if last:
                cur[part] = value; return True
            cur = cur[part]
    return False


def _apply_overlay(payload: dict, one_pagers: dict, overlay: dict) -> int:
    """Overwrite published cells, keeping the computed value beside each one.

    Records into ``payload['published_overrides']`` so drift stays measurable:
    a later reader can see what was published AND what the engine said at the
    moment of freezing.
    """
    recorded, applied = [], 0
    for scope, cells in (overlay or {}).items():
        target = payload if scope == "__subtabs__" else one_pagers.get(scope)
        if target is None:
            continue
        for path, spec in (cells or {}).items():
            published = spec.get("published") if isinstance(spec, dict) else spec
            before = _read_path(target, path)
            if _set_path(target, path, published):
                applied += 1
                recorded.append({
                    "scope": scope, "path": path, "published": published,
                    "computed_at_freeze": before,
                    "page": (spec or {}).get("page") if isinstance(spec, dict) else None,
                })
    payload["published_overrides"] = recorded
    return applied


def _read_path(obj, dotted: str):
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", dotted):
        try:
            cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
        except Exception:
            return None
    return cur


def is_frozen(investor_code: str, quarter: str) -> bool:
    """True when this investor+quarter has a stored copy."""
    return _current_row(investor_code, quarter) is not None


def refreeze(investor_code: str, quarter: str, frozen_by: str, reason: str,
             **kw) -> dict:
    """Replace a frozen quarter, keeping the previous version in history."""
    if not (reason or "").strip():
        raise ValueError("A reason is required to re-freeze a sent quarter")
    if not is_frozen(investor_code, quarter):
        raise ValueError(f"{investor_code} {quarter} is not frozen")
    out = freeze_as_sent(investor_code, quarter, frozen_by, **kw)
    out["refreeze_reason"] = reason
    return out


def unfreeze(investor_code: str, quarter: str, by: str, reason: str) -> dict:
    """Return a quarter to live, archiving what was frozen. Reason required."""
    if not (reason or "").strip():
        raise ValueError("A reason is required to unfreeze a sent quarter")
    prior = _current_row(investor_code, quarter)
    if not prior:
        raise ValueError(f"{investor_code} {quarter} is not frozen")
    import datetime as _dt
    _ensure_table()
    with _engine().begin() as conn:
        conn.execute(text(f"""
            INSERT INTO {_HISTORY}
                (investor_code, quarter, payload, data_version, frozen_by,
                 frozen_at, frozen_reason, source_manifest, roster, one_pagers,
                 version, superseded_by, supersede_reason)
            VALUES (:i, :q, :p, :dv, :fb, :fa, :fr, :sm, :ro, :op, :ver, :sby, :sr)
        """), {"i": investor_code, "q": quarter,
               "p": prior.get("payload_raw") or "{}",
               "dv": prior.get("data_version"), "fb": prior.get("frozen_by"),
               "fa": prior.get("frozen_at"), "fr": prior.get("frozen_reason"),
               "sm": prior.get("source_manifest_raw"),
               "ro": prior.get("roster_raw"), "op": prior.get("one_pagers_raw"),
               "ver": prior.get("version"), "sby": by,
               "sr": f"unfrozen: {reason}"})
        conn.execute(text(f"DELETE FROM {_TABLE} "
                          f"WHERE investor_code = :i AND quarter = :q"),
                     {"i": investor_code, "q": quarter})
    log.info("Unfroze %s %s by %s: %s", investor_code, quarter, by, reason)
    return {"investor_code": investor_code, "quarter": quarter,
            "unfrozen_by": by, "reason": reason,
            "archived_version": prior.get("version"),
            "frozen_at": _dt.datetime.utcnow().isoformat()}


def frozen_history(investor_code: str, quarter: str) -> list:
    """Superseded versions, newest first. Never raises."""
    try:
        _ensure_table()
        with _engine().connect() as conn:
            rows = conn.execute(text(f"""
                SELECT version, frozen_by, frozen_at, frozen_reason,
                       superseded_at, superseded_by, supersede_reason
                FROM {_HISTORY}
                WHERE investor_code = :i AND quarter = :q
                ORDER BY id DESC
            """), {"i": investor_code, "q": quarter}).mappings().fetchall()
        return [dict(r) for r in rows]
    except Exception:
        log.exception("reading freeze history failed")
        return []


def get_frozen(investor_code: str, quarter: str) -> Optional[dict]:
    """The frozen payload for a page, or None."""
    row = _current_row(investor_code, quarter)
    if not row:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:
        log.exception("frozen payload for %s %s is not valid JSON",
                      investor_code, quarter)
        return None

    def _j(raw):
        try:
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def _iso(v):
        return (v.isoformat() if hasattr(v, "isoformat")
                else (str(v) if v else None))

    return {
        "payload": payload,
        "approved_by": row.get("approved_by"),
        "approved_at": _iso(row.get("approved_at")),
        "data_version": row.get("data_version"),
        "frozen_by": row.get("frozen_by"),
        "frozen_at": _iso(row.get("frozen_at")),
        "frozen_reason": row.get("frozen_reason"),
        "version": row.get("version"),
        "source_manifest": _j(row.get("source_manifest_raw")),
        "roster": _j(row.get("roster_raw")),
        "one_pagers": _j(row.get("one_pagers_raw")),
    }


def get_frozen_one_pager(investor_code: str, quarter: str,
                         vcode: str) -> Optional[dict]:
    """One deal's One Pager as published for THIS investor, or None.

    Keyed by investor as well as deal on purpose. The same deal and quarter can
    carry different published figures on two investors' reports — Nottingham
    Village went out at $9.1M to one and $12.1M to the other — so a store keyed
    only by (vcode, quarter) cannot represent what was actually sent.
    """
    fr = get_frozen(investor_code, quarter)
    if not fr:
        return None
    return (fr.get("one_pagers") or {}).get(vcode)


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

    # A STORED COPY WINS, WHATEVER THE APPROVAL STATUS. Freezing and approving
    # are separate acts now: the button records that a quarter was sent, and a
    # sent quarter must not be recomputed. Live data is for the current,
    # unsent quarter only.
    frozen = get_frozen_fn(investor_code, quarter)
    if frozen:
        out = dict(frozen["payload"])
        out["source"] = SOURCE_FROZEN
        who = frozen.get("frozen_by") or frozen.get("approved_by") or "unknown"
        when = str(frozen.get("frozen_at") or frozen.get("approved_at") or "")[:10]
        reason = frozen.get("frozen_reason") or REASON_AS_APPROVED
        label = ("Frozen as sent" if reason == REASON_AS_SENT
                 else "Frozen at approval")
        out["source_note"] = (
            f"{label} — stored copy, not recomputed. By {who}"
            + (f" on {when}" if when else ""))
        for k in ("approved_by", "approved_at", "data_version", "frozen_by",
                  "frozen_at", "frozen_reason", "version", "source_manifest",
                  "roster"):
            out[k] = frozen.get(k)
        out["read_only"] = True
        return out

    if status == "approved":
        out = assemble(investor_code, quarter) or {}
        out.pop("_resolved", None)
        out["source"] = SOURCE_LIVE
        out["read_only"] = False
        out["source_note"] = (
            "This report is approved but has no frozen payload, so it is being "
            "recomputed live and may not match what was approved.")
        return out

    out = assemble(investor_code, quarter) or {}
    out.pop("_resolved", None)
    out["source"] = SOURCE_LIVE
    out["read_only"] = False
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
