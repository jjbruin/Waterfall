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

#: The two halves of an investor's report, frozen independently.
#:
#: THEY ARE ALREADY SEPARATE COLUMNS — ``payload`` holds the Snapshot, and
#: ``roster`` / ``one_pagers`` hold the One Pagers. What was missing was
#: separate STATE: one ``frozen_at`` described the whole row, so a Snapshot
#: freeze and a One Pager freeze could not be told apart, and a row frozen for
#: one half claimed both.
PART_SNAPSHOT = "snapshot"
PART_ONE_PAGERS = "one_pagers"
PARTS = (PART_SNAPSHOT, PART_ONE_PAGERS)

#: Which stored columns each part owns: ``payload`` is the Snapshot,
#: ``roster`` + ``one_pagers`` are the One Pagers. ``_write_frozen`` writes
#: only the ones belonging to the parts being frozen and CARRIES THE REST
#: FORWARD, so freezing one half cannot blank the other.
#:
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
    # Per-part state. A row may carry one part, the other, or both, so
    # "is this frozen?" is only answerable per part. The undated columns above
    # stay as the WHOLE-ROW record of the most recent write of either part —
    # existing rows keep meaning what they meant.
    ("snapshot_frozen_at", "TIMESTAMP"),
    ("snapshot_frozen_by", "TEXT"),
    ("one_pagers_frozen_at", "TIMESTAMP"),
    ("one_pagers_frozen_by", "TEXT"),
)

#: Per-part state columns, by part. Written out rather than derived from a
#: naming convention, so a renamed column fails loudly here instead of
#: silently never being written.
_PART_STATE = {
    PART_SNAPSHOT: ("snapshot_frozen_at", "snapshot_frozen_by"),
    PART_ONE_PAGERS: ("one_pagers_frozen_at", "one_pagers_frozen_by"),
}

#: The history table predates per-part freezing and already exists in
#: production, so ``CREATE TABLE IF NOT EXISTS`` would never add this. An
#: archived row without it says which parts it HELD but not which were being
#: rewritten when it was superseded.
_HISTORY_ADDED_COLUMNS = (
    ("frozen_parts", "TEXT"),      # JSON: the parts this write replaced
)


def normalize_parts(part) -> tuple:
    """``part`` as a validated tuple, in a stable order.

    Accepts a single name or an iterable. An unknown part RAISES rather than
    being skipped: a caller asking to freeze ``"onepagers"`` must not be told
    it froze nothing, quietly.
    """
    if part is None:
        return PARTS
    names = (part,) if isinstance(part, str) else tuple(part)
    if not names:
        raise ValueError("no part named; expected any of " + ", ".join(PARTS))
    bad = [n for n in names if n not in PARTS]
    if bad:
        raise ValueError(
            f"unknown freeze part(s) {bad}; expected any of {list(PARTS)}")
    return tuple(p for p in PARTS if p in names)


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
                supersede_reason TEXT NOT NULL,
                frozen_parts TEXT
            )
        """))
    for table, cols in ((_TABLE, _ADDED_COLUMNS),
                        (_HISTORY, _HISTORY_ADDED_COLUMNS)):
        existing = _columns(table)
        for col, typ in cols:
            if col in existing:
                continue
            try:
                with _engine().begin() as conn:
                    conn.execute(
                        text(f"ALTER TABLE {table} ADD COLUMN {col} {typ}"))
            except Exception:
                log.exception("could not add %s.%s", table, col)


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
    """Capture the report as APPROVED — the CEO-approval path.

    A thin call into ``freeze_part``, not a second implementation. It freezes
    the SNAPSHOT part only, which is what this path always wrote: the four
    assembled subtabs plus the approved editable content (comments, footnotes,
    the manual Net ROE / ITD values) as they stood at approval. It has never
    stored One Pagers and does not start now.

    ``assembler`` / ``elements_loader`` exist for the self-test; production uses
    the defaults. Raises on failure — the CALLER wraps, so that an approval is
    never lost to a snapshot write.
    """
    return freeze_part(investor_code, quarter, PART_SNAPSHOT,
                       frozen_by=approved_by,
                       reason=REASON_AS_APPROVED,
                       approved_by=approved_by,
                       assembler=assembler,
                       elements_loader=elements_loader)


def _write_frozen(investor_code: str, quarter: str,
                  frozen_by: str, reason: str, parts,
                  payload: Optional[dict] = None,
                  approved_by: Optional[str] = None,
                  source_manifest: Optional[dict] = None,
                  roster: Optional[list] = None,
                  one_pagers: Optional[dict] = None,
                  supersede_reason: Optional[str] = None) -> dict:
    """THE one writer for a frozen payload. Every freeze path funnels here.

    ``parts`` names which halves this write owns. **Columns belonging to a part
    NOT being written are carried forward from the existing row**, because the
    upsert is DELETE-then-INSERT: without the carry-forward, freezing the
    Snapshot would blank the One Pagers that were already frozen, and the loss
    would be silent — the row would still be there, just empty on one side.

    Any existing row is copied to the history table BEFORE being replaced, so a
    re-freeze never destroys what an investor was actually sent.
    """
    import datetime as _dt
    parts = normalize_parts(parts)
    version_str = _data_version()
    now = _dt.datetime.utcnow()

    _ensure_table()
    prior = _current_row(investor_code, quarter) or {}
    next_version = int(prior.get("version") or 0) + 1

    # Per part: this write's value, or the stored one untouched.
    if PART_SNAPSHOT in parts:
        if payload is None:
            raise ValueError("freezing the snapshot part needs a payload")
        blob = json.dumps(payload)
    else:
        blob = prior.get("payload_raw") or "{}"

    if PART_ONE_PAGERS in parts:
        roster_blob = json.dumps(roster) if roster is not None else None
        op_blob = json.dumps(one_pagers) if one_pagers is not None else None
    else:
        roster_blob = prior.get("roster_raw")
        op_blob = prior.get("one_pagers_raw")

    # Per-part state: stamped for what we froze, carried for what we did not.
    #
    # A LEGACY ROW CARRIES BOTH PARTS WITH NEITHER TIMESTAMP. Reading the raw
    # column would give None for the half we are not writing, so freezing the
    # Snapshot on such a row would leave its One Pager DATA in place while the
    # state said that half was never frozen — and it would silently stop being
    # served. The implied state is resolved through frozen_parts_of, the same
    # reading every consumer uses, and pinned to the row's own frozen_at.
    prior_parts = frozen_parts_of(prior)
    state = {}
    for p in PARTS:
        at_col, by_col = _PART_STATE[p]
        if p in parts:
            state[at_col], state[by_col] = now, frozen_by
        elif prior.get(at_col) is not None:
            state[at_col] = prior.get(at_col)
            state[by_col] = prior.get(by_col)
        elif p in prior_parts:
            state[at_col] = prior.get("frozen_at") or prior.get("approved_at")
            state[by_col] = prior.get("frozen_by") or prior.get("approved_by")
        else:
            state[at_col] = None
            state[by_col] = None

    with _engine().begin() as conn:
        if prior:
            conn.execute(text(f"""
                INSERT INTO {_HISTORY}
                    (investor_code, quarter, payload, data_version, frozen_by,
                     frozen_at, frozen_reason, source_manifest, roster,
                     one_pagers, version, superseded_by, supersede_reason,
                     frozen_parts)
                VALUES (:i, :q, :p, :dv, :fb, :fa, :fr, :sm, :ro, :op, :ver,
                        :sby, :sr, :fp)
            """), {"i": investor_code, "q": quarter,
                   "p": prior.get("payload_raw") or "{}",
                   "dv": prior.get("data_version"), "fb": prior.get("frozen_by"),
                   "fa": prior.get("frozen_at"), "fr": prior.get("frozen_reason"),
                   "sm": prior.get("source_manifest_raw"),
                   "ro": prior.get("roster_raw"), "op": prior.get("one_pagers_raw"),
                   "ver": prior.get("version"), "sby": frozen_by,
                   "sr": supersede_reason or reason,
                   "fp": json.dumps(list(parts))})
        conn.execute(text(f"DELETE FROM {_TABLE} "
                          f"WHERE investor_code = :i AND quarter = :q"),
                     {"i": investor_code, "q": quarter})
        conn.execute(text(f"""
            INSERT INTO {_TABLE}
                (investor_code, quarter, payload, approved_by, data_version,
                 frozen_by, frozen_at, frozen_reason, source_manifest, roster,
                 one_pagers, version,
                 snapshot_frozen_at, snapshot_frozen_by,
                 one_pagers_frozen_at, one_pagers_frozen_by)
            VALUES (:i, :q, :p, :by, :v, :fb, :fa, :fr, :sm, :ro, :op, :ver,
                    :sat, :sby_, :oat, :oby)
        """), {"i": investor_code, "q": quarter, "p": blob,
               # A freeze is not an approval: only the approval path supplies
               # approved_by, and a later part-freeze must not drop it.
               "by": approved_by or prior.get("approved_by"),
               "v": version_str, "fb": frozen_by,
               "fa": now, "fr": reason,
               "sm": (json.dumps(source_manifest) if source_manifest
                      else prior.get("source_manifest_raw")),
               "ro": roster_blob, "op": op_blob, "ver": next_version,
               "sat": state["snapshot_frozen_at"],
               "sby_": state["snapshot_frozen_by"],
               "oat": state["one_pagers_frozen_at"],
               "oby": state["one_pagers_frozen_by"]})

    log.info("Froze %s %s parts=%s (%s, version %s, %s)",
             investor_code, quarter, ",".join(parts), reason,
             next_version, version_str)
    return {"investor_code": investor_code, "quarter": quarter,
            "frozen_by": frozen_by, "frozen_at": now.isoformat(),
            "frozen_reason": reason, "version": next_version,
            "parts": list(parts),
            "frozen_parts": frozen_parts_of(_current_row(investor_code, quarter)),
            "data_version": version_str, "bytes": len(blob),
            "one_pager_count": len(json.loads(op_blob) if op_blob else {}),
            "roster_count": len(json.loads(roster_blob) if roster_blob else [])}


def frozen_parts_of(row: Optional[dict]) -> list:
    """Which parts a stored row actually carries, in a stable order.

    Read from the per-part timestamps, NOT from whether a column has content:
    an investor with an empty roster legitimately freezes zero One Pagers, and
    that is a frozen part, not an absent one.

    A row written BEFORE per-part freezing has no per-part timestamps and is
    reported as carrying BOTH — which is what it is. Treating it as unfrozen
    would silently un-freeze every quarter already sent.
    """
    if not row:
        return []
    have = [p for p in PARTS if row.get(_PART_STATE[p][0]) is not None]
    if have:
        return have
    return list(PARTS) if row.get("frozen_at") or row.get("approved_at") else []


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


def freeze_part(investor_code: str, quarter: str, part, frozen_by: str,
                reason: str = REASON_AS_SENT,
                overlay: Optional[dict] = None,
                roster: Optional[list] = None,
                source_manifest: Optional[dict] = None,
                assembler: Optional[Callable] = None,
                one_pager_getter: Optional[Callable] = None,
                elements_loader: Optional[Callable] = None,
                approved_by: Optional[str] = None,
                supersede_reason: Optional[str] = None) -> dict:
    """THE freeze. One core, parameterized by which half is being frozen.

    ``part`` is ``"snapshot"``, ``"one_pagers"``, or both. Every caller comes
    through here — the per-part batches, the published-overlay freeze, the
    re-freeze and the approval-chain freeze — so there is no second definition
    of what freezing means and the parts cannot drift apart.

    **The overlay takes both parts in ONE call, deliberately.** One overlay
    document carries Snapshot cells and One Pager cells read off the same sent
    PDF; freezing them as two writes would produce two versions and two history
    rows for a single act, and a reader could not tell they belonged together.

    Separate from the approval chain. ``approved_by`` stays NULL unless the
    approval path supplies it — a freeze records who froze it, never a decision
    nobody made.

    **A ONE PAGER FAILURE NO LONGER SINKS THE SNAPSHOT.** The old behaviour
    raised on any failed One Pager so the freeze could not be partial; with the
    halves independent, that would let one unbuildable deal block a Snapshot
    that is perfectly fine. Failures now drop the One Pager part from the write
    and are REPORTED in ``one_pager_errors``. Asking for only the One Pagers and
    having them all fail still raises, because then nothing was frozen and
    returning a receipt would say otherwise.
    """
    from flask_app.serializers import safe_json

    parts = normalize_parts(part)
    assemble = assembler or assemble_full_report
    report = assemble(investor_code, quarter) or {}
    report.pop("_resolved", None)          # not part of the frozen contract

    payload = None
    if PART_SNAPSHOT in parts:
        load_elements = elements_loader
        if load_elements is None:
            from flask_app.services.portfolio_snapshot_persistence import load_page
            load_elements = load_page
        elements = load_elements(investor_code, quarter) or {}
        payload = safe_json({
            "subtabs": report.get("subtabs") or {},
            "errors": report.get("errors") or {},
            "resolution": report.get("resolution") or {},
            # The editable content, frozen alongside the metrics.
            "elements": {
                "comments": elements.get("comments") or [],
                "footnotes": elements.get("footnotes") or [],
                "values": elements.get("values") or [],
            },
        })

    one_pagers = None
    failures = []
    if PART_ONE_PAGERS in parts:
        if roster is None:
            roster = _roster_from_report(report)
        get_op = one_pager_getter or _default_one_pager_getter()
        built = {}
        for vc in roster:
            try:
                built[vc] = get_op(vc, quarter)
            except Exception as exc:                  # noqa: BLE001
                failures.append(f"{vc}: {type(exc).__name__}: {exc}")
        if failures:
            parts = tuple(p for p in parts if p != PART_ONE_PAGERS)
            if not parts:
                raise RuntimeError(
                    f"{len(failures)} One Pager(s) could not be built, so "
                    f"nothing was frozen: " + "; ".join(failures[:5]))
        else:
            one_pagers = safe_json(built)

    applied = 0
    if overlay:
        # Applied AFTER assembly, so every untouched cell stays the computed
        # one and the diff between the two stays recoverable.
        applied = _apply_overlay(payload if payload is not None else {},
                                 one_pagers if one_pagers is not None else {},
                                 overlay)

    manifest = dict(source_manifest or {}) if source_manifest else None
    if manifest is not None:
        manifest["overlay_cells_applied"] = applied

    receipt = _write_frozen(
        investor_code, quarter, frozen_by=frozen_by, reason=reason,
        parts=parts, payload=payload, approved_by=approved_by,
        source_manifest=manifest,
        roster=roster if PART_ONE_PAGERS in parts else None,
        one_pagers=one_pagers, supersede_reason=supersede_reason)
    if failures:
        receipt["one_pager_errors"] = failures
        receipt["one_pagers_frozen"] = False
    return receipt


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
    recorded, unapplied, applied = [], [], 0
    for scope, cells in (overlay or {}).items():
        target = payload if scope == "__subtabs__" else one_pagers.get(scope)
        if target is None:
            continue
        for path, spec in (cells or {}).items():
            if not isinstance(spec, dict):
                spec = {"published": spec}
            before = _read_path(target, path)
            display = spec.get("display")

            if display is not None:
                # A PRINTED-UNITS CELL. The PDF prints something the stored
                # field cannot hold in the same units — the variance columns
                # print a PERCENT OF BUDGET while the field holds a DOLLAR
                # difference. Writing the percent into the dollar field would
                # be a lie about the field, and writing the dollars would not
                # reproduce the page.
                #
                # It also would not even show: the variance the One Pager
                # RENDERS is derived in the browser by `fmtVariance` from
                # ytd_actual and ytd_budget, so the server's stored `variance`
                # reaches no screen. Overlaying the path alone changes nothing.
                #
                # So the printed text is kept verbatim in `published_display`,
                # keyed by the same dotted path, and the view prefers it when
                # the report is frozen. The numeric field is left ALONE.
                target.setdefault("published_display", {})[path] = display
                applied += 1
                recorded.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "display": display,
                    "units": spec.get("units") or "printed",
                    "computed_at_freeze": before,
                    "page": spec.get("page"), "source": spec.get("source"),
                })
                continue

            if _set_path(target, path, spec.get("published")):
                applied += 1
                recorded.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "computed_at_freeze": before,
                    "page": spec.get("page"), "source": spec.get("source"),
                })
            else:
                # NOT SILENTLY DROPPED. `_set_path` refuses a path the live
                # payload does not already carry — which is right, since
                # inventing the key would put a published figure somewhere no
                # reader looks. But a printed cell that never landed means the
                # stored copy does NOT reproduce the page, and that has to be
                # visible rather than showing up only as a lower applied count.
                unapplied.append({
                    "scope": scope, "path": path,
                    "published": spec.get("published"),
                    "display": spec.get("display"),
                    "page": spec.get("page"),
                    "why": "no such field in the assembled report",
                })
    payload["published_overrides"] = recorded
    if unapplied:
        payload["published_unapplied"] = unapplied
    return applied


def _read_path(obj, dotted: str):
    cur = obj
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", dotted):
        try:
            cur = cur[int(part[1:-1])] if part.startswith("[") else cur[part]
        except Exception:
            return None
    return cur


def _norm_title(s) -> str:
    """A deal name reduced to what survives PDF typesetting."""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def resolve_roster(investor_code: str, quarter: str, titles: list) -> tuple:
    """(``{printed title: vcode}``, ``[titles that matched nothing]``).

    The overlay is keyed by the deal TITLE because that is all the sent PDF
    knows; only the app knows vcodes. Resolving here — against the same
    assembled report the freeze itself will store — keeps the preview and the
    freeze on one answer.

    A TITLE MATCHING SEVERAL DEALS IS LEFT UNRESOLVED, not resolved to the
    first. Picking one would attach a whole page of published figures to the
    wrong deal, which is the single worst thing this feature could do, and it
    would be invisible afterwards.
    """
    from flask_app.services import data_service
    from flask_app.services.portfolio_snapshot_service import resolve_investor_deals

    data = data_service.get_data()
    resolved_deals = resolve_investor_deals(investor_code, quarter, data=data)
    fin = build_subtab("financial", investor_code, quarter, data, resolved_deals)

    by_name = {}
    for blk in (fin.get("groups") or {}).values():
        rows = (blk.get("deals") if isinstance(blk, dict) else blk) or []
        for r in rows:
            if r.get("name") and r.get("vcode"):
                by_name.setdefault(_norm_title(r["name"]), set()).add(r["vcode"])
    for r in (fin.get("ownership_flagged") or []):
        if r.get("name") and r.get("vcode"):
            by_name.setdefault(_norm_title(r["name"]), set()).add(r["vcode"])

    out, missing = {}, []
    for title in titles or []:
        key = _norm_title(title)
        hit = by_name.get(key)
        if not hit:
            # A printed title is often a prefix of the stored name, or the
            # other way round. Only an UNAMBIGUOUS partial is accepted.
            cands = {v for k, vs in by_name.items()
                     if key and (k.startswith(key) or key.startswith(k))
                     for v in vs}
            hit = cands
        if hit and len(hit) == 1:
            out[title] = next(iter(hit))
        else:
            missing.append(title)
    return out, missing


def _norm_row(s) -> str:
    """A Snapshot row label reduced for matching.

    Parenthetical suffixes are dropped — the page prints "Camarillo Village
    (Sold)", "Portfolio Totals (38)" and "Excluding development deals (28)",
    and the count in particular moves between quarters.
    """
    s = re.sub(r"\([^)]*\)", " ", str(s or ""))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _index_subtab(sub: str, blk: dict) -> dict:
    """{normalised row label: dotted path prefix} for one assembled subtab.

    Covers every row the sent page prints: deals, the ownership-flagged rows,
    each group's subtotal, Portfolio Totals and the Excluding-development row.
    A label matching two rows is dropped from the index rather than resolved to
    one — publishing a row of figures against the wrong property is the worst
    thing this can do and is invisible afterwards.
    """
    seen: dict = {}

    def add(label, path):
        k = _norm_row(label)
        if not k:
            return
        seen.setdefault(k, []).append(path)

    for gname, g in (blk.get("groups") or {}).items():
        rows = (g.get("deals") if isinstance(g, dict) else g) or []
        for i, r in enumerate(rows):
            if r.get("name"):
                add(r["name"], f"subtabs.{sub}.groups.{gname}.deals[{i}]")
        if isinstance(g, dict) and isinstance(g.get("subtotal"), dict):
            st = g["subtotal"]
            add(st.get("name") or st.get("label") or f"Total {gname}",
                f"subtabs.{sub}.groups.{gname}.subtotal")
    for i, r in enumerate(blk.get("ownership_flagged") or []):
        if r.get("name"):
            add(r["name"], f"subtabs.{sub}.ownership_flagged[{i}]")
    if isinstance(blk.get("total"), dict):
        add("Portfolio Totals", f"subtabs.{sub}.total")
    if isinstance(blk.get("total_excluding_dev"), dict):
        add("Excluding development deals",
            f"subtabs.{sub}.total_excluding_dev")
    return {k: v[0] for k, v in seen.items() if len(v) == 1}


#: Deal-row fields whose RENDERED cell is a `_display` twin, per subtab.
#:
#: THE RAW FIELD DOES NOT REACH THE SCREEN FOR THESE. SnapshotLoan.vue renders
#: `r.ltv_display`, SnapshotOperating.vue renders `r.noi_display.at_close`, and
#: SnapshotFinancial.vue renders `r.debt_display` — so writing only the raw
#: value leaves the frozen page showing the LIVE figure, which is the same
#: defect the One Pager variance had. Subtotal and total rows are the other way
#: round: they render the raw field, so they are NOT translated.
_DISPLAY_TWIN = {
    "financial": {"debt": "debt_display"},
    "operating": {
        "econ_occ": "econ_occ_display",
        "noi.at_close": "noi_display.at_close",
        "noi.uw_ye": "noi_display.uw_ye",
        "noi.projected_ye": "noi_display.projected_ye",
        "expected_growth": "expected_growth_display",
        "actual_growth": "actual_growth_display",
    },
    "loan": {
        "rate": "rate_display", "maturity": "maturity_display",
        "debt": "debt_display", "ytd_dscr": "ytd_dscr_display",
        "ltv": "ltv_display", "debt_yield": "debt_yield_display",
    },
}


def _snapshot_targets(sub: str, field: str, spec: dict, is_deal: bool) -> list:
    """The path suffix(es) one printed cell should be written to.

    A NUMBER GOES TO BOTH the raw field and its display twin: the twin is what
    renders, and the raw is what the subtotals are built from, so writing only
    one leaves the page and its totals disagreeing.

    A SENTINEL GOES TO THE TWIN ALONE. "—" and "Dev" are display strings; the
    twin legitimately holds them, and putting one in the raw numeric field
    would corrupt every sum that reads it.
    """
    twin = _DISPLAY_TWIN.get(sub, {}).get(field) if is_deal else None
    is_sentinel = spec.get("units") == "printed-sentinel"
    if twin and is_sentinel:
        return [twin]
    if twin:
        return [field, twin]
    return [] if is_sentinel and not twin else [field]


def resolve_snapshot_cells(payload: dict, snapshot: dict) -> tuple:
    """(``{dotted path: spec}``, ``[rows that matched nothing]``).

    The overlay is keyed by PRINTED ROW LABEL because that is all the sent page
    knows; the payload keys rows by group and index. Resolving against the
    assembled report — the very payload about to be stored — keeps the preview
    and the freeze on one answer.
    """
    cells, missing = {}, []
    subs = (payload or {}).get("subtabs") or {}
    for sub, rows in (snapshot or {}).items():
        blk = subs.get(sub)
        if not isinstance(blk, dict):
            missing.extend([f"{sub}/{lbl}" for lbl in (rows or {})])
            continue
        index = _index_subtab(sub, blk)
        for label, fields in (rows or {}).items():
            prefix = index.get(_norm_row(label))
            if not prefix:
                missing.append(f"{sub}/{label}")
                continue
            is_deal = ".deals[" in prefix or ".ownership_flagged[" in prefix
            for field, spec in (fields or {}).items():
                for target in _snapshot_targets(sub, field, spec, is_deal):
                    out = dict(spec)
                    if target.endswith("_display") or "_display." in target:
                        # The twin renders, so the printed text goes in as the
                        # VALUE here rather than through `published_display` —
                        # the Snapshot components do not read that map.
                        if spec.get("display") is not None:
                            out = {**spec, "published": spec["display"],
                                   "display": None}
                    cells[f"{prefix}.{target}"] = out
    return cells, missing


def dry_run_unapplied(assembled: dict, overlay: dict) -> list:
    """Which overlay cells would NOT land, without freezing anything.

    Applies the overlay to a DEEP COPY of the assembled report and returns what
    `_apply_overlay` could not place. The preview and the freeze therefore
    answer from the same code — a preview that predicted differently from the
    write it precedes would be worse than none.
    """
    import copy
    payload = copy.deepcopy(assembled or {})
    one_pagers = {k: copy.deepcopy(v) for k, v in
                  (payload.get("one_pagers") or {}).items()}
    # A One Pager scope needs a target to write into; absent ones are reported
    # by the roster resolution, not here.
    for scope in (overlay or {}):
        if scope != "__subtabs__":
            one_pagers.setdefault(scope, {})
    _apply_overlay(payload, one_pagers, overlay or {})
    return payload.get("published_unapplied") or []


#: Roughly how many cells the 26Q2 overlay is expected to change. Shown on the
#: preview so a big deviation is obvious. NOT enforced — the point is that a
#: reader who sees 900 where 114 was expected stops and asks why, which no
#: automatic threshold does as well.
EXPECTED_DIFFERENCES_26Q2 = 114

#: A column is flagged when most of its rows differ, or when the typical
#: overlay/live ratio is nowhere near 1. Both are signatures of a mechanical
#: fault rather than a genuine correction: a units error lands near 1e6 or
#: 1e-6, and a column shift makes almost every row differ at once.
_MOSTLY_DIFFER = 0.60
_RATIO_LO, _RATIO_HI = 0.5, 2.0


def _num_or_none(v):
    if isinstance(v, bool) or v is None or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return None if f != f else f


def get_one_pager_live(vcode: str, quarter: str):
    """One deal's LIVE One Pager, through the same getter the freeze uses.

    The preview compares against exactly what the freeze would store, so it
    must read it the same way — a second route to the One Pager would let the
    preview describe a payload the freeze never sees.
    """
    try:
        return _default_one_pager_getter()(vcode, quarter)
    except Exception:
        log.exception("live One Pager unavailable for %s %s", vcode, quarter)
        return None


def compare_overlay_to_live(targets: dict, overlay: dict) -> dict:
    """What the overlay would CHANGE, per page and per column.

    ``targets`` is ``{"__subtabs__": assembled, vcode: one_pager, ...}`` — the
    live payloads the overlay is about to be written over.

    THREE THINGS A READER CANNOT GET FROM A CELL COUNT. How many cells actually
    move (a freeze that changes nothing means the overlay never landed); which
    columns move TOGETHER (a whole column differing is a column shift, not
    thirty independent corrections); and the typical ratio (a units error sits
    at 1e6 or 1e-6 and every individual cell looks plausible).

    A SENTINEL WHOSE LIVE VALUE IS NOT BLANK is called out separately. The page
    printed "—" while the app holds a figure — East Manchester's loan rate is
    the known case — so the frozen row would show a number that was never sent
    unless the printed dash is stored over it.
    """
    per_page, per_col, differing, sentinels = {}, {}, [], []
    for scope, cells in (overlay or {}).items():
        target = targets.get(scope)
        if target is None:
            continue
        for path, spec in (cells or {}).items():
            page = spec.get("page")
            pg = per_page.setdefault(page, {"cells": 0, "differs": 0})
            pg["cells"] += 1
            col = path.rsplit(".", 1)[-1]
            cc = per_col.setdefault(f"{scope if scope == '__subtabs__' else 'one_pager'}.{col}",
                                    {"n": 0, "differ": 0, "ratios": []})
            cc["n"] += 1

            live = _read_path(target, path)
            pub = spec.get("published")
            disp = spec.get("display")
            is_sentinel = spec.get("units") == "printed-sentinel"

            if is_sentinel:
                if live not in (None, "", "—", "n/a", "N/A", "Dev"):
                    sentinels.append({"scope": scope, "path": path,
                                      "printed": disp, "live": live,
                                      "page": page})
                    pg["differs"] += 1
                    cc["differ"] += 1
                    differing.append({"scope": scope, "path": path,
                                      "published": disp, "live": live,
                                      "page": page, "kind": "sentinel"})
                continue

            want = pub if pub is not None else disp
            same = (want == live)
            ln, wn = _num_or_none(live), _num_or_none(want)
            if ln is not None and wn is not None:
                same = abs(wn - ln) <= max(0.005, abs(wn) * 1e-6)
                if ln:
                    cc["ratios"].append(wn / ln)
            if not same:
                pg["differs"] += 1
                cc["differ"] += 1
                differing.append({"scope": scope, "path": path,
                                  "published": want, "live": live,
                                  "page": page, "kind": "value"})

    warnings = []
    for col, c in per_col.items():
        rs = sorted(c["ratios"])
        med = rs[len(rs) // 2] if rs else None
        c["median_ratio"] = med
        c.pop("ratios", None)
        if c["n"] >= 4 and c["differ"] / c["n"] >= _MOSTLY_DIFFER:
            warnings.append({
                "column": col, "kind": "most-rows-differ",
                "detail": f"{c['differ']} of {c['n']} rows differ — a column "
                          f"shift looks exactly like this"})
        if med is not None and not (_RATIO_LO <= med <= _RATIO_HI):
            warnings.append({
                "column": col, "kind": "ratio-far-from-one",
                "detail": f"typical overlay/live ratio is {med:.4g}"
                          + (" — that is a units error (1e6)"
                             if med > 1e5 or (med and med < 1e-5) else "")})

    return {
        "differs_total": sum(p["differs"] for p in per_page.values()),
        "cells_total": sum(p["cells"] for p in per_page.values()),
        "expected_differences": EXPECTED_DIFFERENCES_26Q2,
        "by_page": {str(k): v for k, v in sorted(
            per_page.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "by_column": per_col,
        "differing_cells": differing,
        "sentinels_live_non_blank": sentinels,
        "warnings": warnings,
    }


def is_frozen(investor_code: str, quarter: str, part=None) -> bool:
    """True when this investor+quarter has the named part frozen.

    ``part=None`` asks "is ANY part frozen", which is what the old whole-row
    question meant and what the write lock wants. Naming a part asks about that
    half alone — the read paths must, or a Snapshot-only freeze would serve
    One Pagers nobody froze.
    """
    have = frozen_parts_of(_current_row(investor_code, quarter))
    if not have:
        return False
    if part is None:
        return True
    return all(p in have for p in normalize_parts(part))


def frozen_parts(investor_code: str, quarter: str) -> list:
    """Which halves are frozen. ``[]`` when the quarter is live."""
    return frozen_parts_of(_current_row(investor_code, quarter))


def quarters_frozen_with_deal(quarter: str, vcode: str) -> list:
    """Investors whose frozen ``quarter`` carries ``vcode`` in its One Pagers.

    EXISTS FOR THE COMMENT LOCK. One Pager comments are keyed (vcode, quarter)
    with no investor, while a freeze is keyed (investor, quarter) — so "is this
    comment frozen?" has no single answer and must be asked of every investor.
    Returns the investor codes, so the refusal can name them instead of saying
    only that something, somewhere, is frozen.
    """
    out = []
    try:
        _ensure_table()
        cols = _columns(_TABLE)
        if "one_pagers" not in cols:
            return out
        with _engine().connect() as conn:
            rows = conn.execute(text(
                f"SELECT investor_code, one_pagers, roster, "
                f"       one_pagers_frozen_at, frozen_at, approved_at "
                f"FROM {_TABLE} WHERE quarter = :q"), {"q": quarter}).mappings()
            for r in rows:
                if PART_ONE_PAGERS not in frozen_parts_of(dict(r)):
                    continue
                for raw in (r.get("one_pagers"), r.get("roster")):
                    try:
                        blob = json.loads(raw) if raw else None
                    except Exception:
                        blob = None
                    if blob and vcode in blob:
                        out.append(r["investor_code"])
                        break
    except Exception:
        log.exception("frozen-deal lookup failed for %s %s", quarter, vcode)
    return out


def refreeze(investor_code: str, quarter: str, frozen_by: str, reason: str,
             part=None, **kw) -> dict:
    """Replace a frozen quarter, keeping the previous version in history.

    Defaults to BOTH parts, which is what a correction to a sent report means.
    A part may be named to restate one half alone.
    """
    if not (reason or "").strip():
        raise ValueError("A reason is required to re-freeze a sent quarter")
    if not is_frozen(investor_code, quarter):
        raise ValueError(f"{investor_code} {quarter} is not frozen")
    out = freeze_part(investor_code, quarter, part, frozen_by, **kw)
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

    have = frozen_parts_of(row)
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
        # Which halves this row actually carries, and who froze each. A
        # consumer must branch on these rather than on the row existing.
        "frozen_parts": have,
        "snapshot_frozen": PART_SNAPSHOT in have,
        "one_pagers_frozen": PART_ONE_PAGERS in have,
        "snapshot_frozen_at": _iso(row.get("snapshot_frozen_at")),
        "snapshot_frozen_by": row.get("snapshot_frozen_by"),
        "one_pagers_frozen_at": _iso(row.get("one_pagers_frozen_at")),
        "one_pagers_frozen_by": row.get("one_pagers_frozen_by"),
    }


def get_frozen_one_pager(investor_code: str, quarter: str,
                         vcode: str) -> Optional[dict]:
    """One deal's One Pager as published for THIS investor, or None.

    Keyed by investor as well as deal on purpose. The same deal and quarter can
    carry different published figures on two investors' reports — Nottingham
    Village went out at $9.1M to one and $12.1M to the other — so a store keyed
    only by (vcode, quarter) cannot represent what was actually sent.

    **Returns None unless the ONE PAGER part is frozen.** A quarter whose
    Snapshot alone was frozen has a row, and before per-part state that row was
    enough to serve a One Pager nobody had frozen — a stale copy presented as
    what was sent.
    """
    fr = get_frozen(investor_code, quarter)
    if not fr or not fr.get("one_pagers_frozen"):
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
    # ...BUT ONLY THE SNAPSHOT PART SPEAKS FOR THIS REPORT. A quarter whose
    # One Pagers alone were frozen has a row, and serving its `payload` would
    # hand back the empty placeholder that row was created with. The Snapshot
    # is live until the Snapshot part is frozen.
    frozen = get_frozen_fn(investor_code, quarter)
    if frozen and frozen.get("snapshot_frozen", True):
        out = dict(frozen["payload"])
        out["source"] = SOURCE_FROZEN
        who = (frozen.get("snapshot_frozen_by") or frozen.get("frozen_by")
               or frozen.get("approved_by") or "unknown")
        when = str(frozen.get("snapshot_frozen_at") or frozen.get("frozen_at")
                   or frozen.get("approved_at") or "")[:10]
        reason = frozen.get("frozen_reason") or REASON_AS_APPROVED
        label = ("Frozen as sent" if reason == REASON_AS_SENT
                 else "Frozen at approval")
        out["source_note"] = (
            f"{label} — stored copy, not recomputed. By {who}"
            + (f" on {when}" if when else ""))
        for k in ("approved_by", "approved_at", "data_version", "frozen_by",
                  "frozen_at", "frozen_reason", "version", "source_manifest",
                  "roster", "frozen_parts", "snapshot_frozen",
                  "one_pagers_frozen", "snapshot_frozen_at",
                  "one_pagers_frozen_at"):
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
    # A LIVE SNAPSHOT MAY STILL HAVE FROZEN ONE PAGERS. Without this the banner
    # would read "Live data" with nothing saying the other half is fixed, and
    # the two halves would appear to disagree for no stated reason.
    if frozen:
        out["frozen_parts"] = frozen.get("frozen_parts") or []
        out["snapshot_frozen"] = bool(frozen.get("snapshot_frozen"))
        out["one_pagers_frozen"] = bool(frozen.get("one_pagers_frozen"))
        out["one_pagers_frozen_at"] = frozen.get("one_pagers_frozen_at")
        out["one_pagers_frozen_by"] = frozen.get("one_pagers_frozen_by")
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
