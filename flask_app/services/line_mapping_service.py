"""One line-mapping flow for both sources that feed the budget comparison.

The comparison `get_budget_review` renders is Estimate | Budget | Valuation, ~27 category
rows from `config.IS_ACCOUNTS`. TWO of those columns are loaded from a spreadsheet:

    Budget      <- the partner's monthly budget workbook
    Valuation   <- the appraiser's Argus download

They are the same job — take somebody else's line names, decide which account each
belongs to, check the result, write it — so they get the same screen AND THE SAME RULES.

Asset management, Sep 28 2026: "We want it to essentially mirror the way the budget
process works." So neither source guesses from wording any more. A line that states our
account number is pre-filled from it ("acct 5050 from the file"); a line mapped before
shows how; anything else is the analyst's to assign. The Argus keyword rules
(`argus_parser.ARGUS_COA_MAP`) used to pre-fill FIRST and so outranked the account the
file itself stated -- the opposite of what anyone wanted.

And the Argus file is read ONCE, here. It used to be uploaded twice: on Assumptions &
Documents, where `argus_parser.parse_monthly_cashflow` read it with its own labels and
keyword accounts, and again on this screen, where a different parser read it and the
mapping was written back BY LABEL onto the first import -- so a line the two parsers named
differently took no mapping at all. `_commit_argus` now writes the Valuation cash flow
from THIS reading of the file, so what the analyst mapped is exactly what lands.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text

from flask_app.services import budget_import_service as budget
from flask_app.services import budget_import_validate as validate_mod
from flask_app.services import valuation_service

logger = logging.getLogger(__name__)

SOURCES = ("budget", "argus")


def save_draft(engine, record_id: int, source: str, filename: str,
               parsed: Dict[str, Any], mapping: Dict[str, Any],
               username: str) -> Dict[str, Any]:
    """Store a mapping in progress, so closing the tab does not throw it away.

    Written on every change, not on a button. Assigning sixty-five partner line names
    to our categories is twenty minutes of judgement, and asking someone to remember
    to save it is asking them to lose it once.

    The PARSED file is stored alongside the mapping. Without it, resuming would mean
    hunting down the original spreadsheet and uploading it again, which is most of the
    friction the draft is meant to remove.
    """
    import json
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Expected one of: {SOURCES}")

    valuation_service.ensure_valuation_tables(engine)
    payload = {
        'rid': record_id, 'src': source, 'fn': filename or '',
        'p': json.dumps(parsed or {}), 'm': json.dumps(mapping or {}),
        'u': username or '',
    }
    with engine.begin() as conn:
        updated = conn.execute(text("""
            UPDATE valuation_mapping_drafts
               SET filename = :fn, parsed_json = :p, mapping_json = :m,
                   status = 'draft', committed_at = NULL,
                   updated_by = :u, updated_at = CURRENT_TIMESTAMP
             WHERE record_id = :rid AND source = :src
        """), payload).rowcount
        if not updated:
            conn.execute(text("""
                INSERT INTO valuation_mapping_drafts
                    (record_id, source, filename, parsed_json, mapping_json,
                     status, updated_by)
                VALUES (:rid, :src, :fn, :p, :m, 'draft', :u)
            """), payload)
    return {'status': 'saved', 'record_id': record_id, 'source': source}


def get_draft(engine, record_id: int, source: str) -> Optional[Dict[str, Any]]:
    """The stored mapping for this record and source, or None.

    Returns a COMMITTED one too, flagged as such. Re-opening the screen after applying
    a mapping used to show an empty page, which reads exactly like the work was lost.
    """
    import json
    valuation_service.ensure_valuation_tables(engine)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT filename, parsed_json, mapping_json, status,
                   committed_at, updated_by, updated_at
              FROM valuation_mapping_drafts
             WHERE record_id = :rid AND source = :src
        """), {'rid': record_id, 'src': source}).fetchone()
    if not row:
        return None
    try:
        parsed = json.loads(row[1] or '{}')
        mapping = json.loads(row[2] or '{}')
    except ValueError:
        # A draft we cannot read is not a draft. Say so rather than half-restoring.
        logger.warning("Unreadable mapping draft for record %s/%s", record_id, source)
        return None
    return {
        'filename': row[0], 'parsed': parsed, 'mapping': mapping,
        'status': row[3] or 'draft', 'committed_at': str(row[4]) if row[4] else None,
        'updated_by': row[5], 'updated_at': str(row[6]) if row[6] else None,
        'line_count': len(parsed.get('lines') or []),
        'mapped_count': sum(1 for v in mapping.values() if (v or {}).get('category')),
    }


def mark_draft_committed(engine, record_id: int, source: str) -> None:
    """Flag the stored mapping as applied, keeping it readable.

    Deleting it on commit would empty the screen again the moment the work succeeded.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE valuation_mapping_drafts
               SET status = 'committed', committed_at = CURRENT_TIMESTAMP
             WHERE record_id = :rid AND source = :src
        """), {'rid': record_id, 'src': source})


def discard_draft(engine, record_id: int, source: str) -> Dict[str, Any]:
    """Throw the stored mapping away, when the analyst says to start over."""
    with engine.begin() as conn:
        n = conn.execute(text("""
            DELETE FROM valuation_mapping_drafts
             WHERE record_id = :rid AND source = :src
        """), {'rid': record_id, 'src': source}).rowcount
    return {'status': 'discarded', 'removed': int(n or 0)}


# Partnership costs are layered in by hand after the appraiser's Argus arrives, because
# an Argus download has no such line. Asset management asked for a $20K/year default to
# GL 5130.
#
# Offered as a PROPOSED LINE the analyst adds, never injected. Automating it would put
# $20,000 into every valuation that nobody typed and nobody can see -- the same class of
# invisible assumption this whole screen exists to remove. It arrives visible, priced,
# editable and refusable.
PARTNERSHIP_DEFAULT = {
    "account": "5130",
    "category": "Partnership Expenses",
    "annual_amount": 20000.0,
    "label": "Partnership costs (house default)",
    "why": "Layered in by hand historically; an Argus download carries no such line.",
}


def proposed_lines(engine, record_id: int, source: str,
                   parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Lines we suggest ADDING, which the spreadsheet does not carry.

    Only ever a suggestion with a price on it. Nothing is added unless the analyst
    says so, and if the file already carries the account the suggestion is withheld
    rather than doubling it.
    """
    out: List[Dict[str, Any]] = []
    if source != "argus":
        return out

    already = any(str(l.get("stated_account") or "") == PARTNERSHIP_DEFAULT["account"]
                  or PARTNERSHIP_DEFAULT["account"] in str(l.get("label") or "")
                  for l in parsed.get("lines", []))
    if already:
        return out

    months = len(parsed.get("periods") or []) or 12
    out.append({
        **PARTNERSHIP_DEFAULT,
        "months": months,
        # Priced over the periods the file actually covers, so a part-year Argus does
        # not silently get a full year of cost.
        "amount": round(PARTNERSHIP_DEFAULT["annual_amount"] * months / 12.0, 2),
    })
    return out


def with_accepted_proposals(parsed: Dict[str, Any], mapping: Dict[str, Any],
                            source: str = "argus") -> tuple:
    """The file's lines plus every proposed line the analyst TICKED, as real lines.

    THE TICK BOX USED TO DO NOTHING. From `v502` until Sep 28 2026 an accepted proposal
    lived only in the browser: it was not sent with the check, not stored in the draft
    and not sent on commit, so ticking "Partnership costs" changed nothing that was
    written. The screen priced it and the import ignored it.

    An accepted proposal now rides on the parsed file (`accepted_proposals`), which is
    what the check, the draft and the commit already carry, and becomes a line here:
    its amount spread evenly over the file's months, mapped to its account. Negative
    row numbers, so it can never collide with a spreadsheet row. Only accounts this
    screen actually offers are accepted -- a crafted payload cannot add an arbitrary
    line under the proposal's name.
    """
    accepted = parsed.get("accepted_proposals") or []
    if not accepted or source != "argus":     # proposals are offered on Argus only
        return parsed, mapping
    periods = parsed.get("periods") or []
    offered = {PARTNERSHIP_DEFAULT["account"]}
    lines = list(parsed.get("lines") or [])
    mapping = dict(mapping or {})
    for i, pl in enumerate(accepted):
        acct = str((pl or {}).get("account") or "")
        try:
            amount = float(pl.get("amount"))
        except (TypeError, ValueError):
            continue
        if acct not in offered or not periods or amount == 0:
            continue
        each = amount / len(periods)
        row = -1 - i
        lines.append({"row": row, "label": pl.get("label") or PARTNERSHIP_DEFAULT["label"],
                      "amounts": {p: each for p in periods}, "total": round(amount, 2),
                      "months": len(periods), "looks_like_total": False,
                      "stated_account": acct, "proposed": True})
        mapping[str(row)] = {"account": acct, "category": budget.category_for_account(acct),
                             "flip": False, "proposed": True}
    return {**parsed, "lines": lines}, mapping


def _norm_label(label: str) -> str:
    """Compare line names on their words alone.

    EXACT text after case and spacing, never fuzzy. A near-match is a guess, and the
    whole point of showing history is that it is a decision somebody actually made.
    """
    return ' '.join(str(label or '').strip().lower().split())


def _mapping_history(engine, labels: List[str], vcode: str) -> Dict[str, Dict[str, Any]]:
    """How each of these line names was mapped before, and where.

    Asset management: "it's hard to select by category and then see which GL codes are
    available, and we end up guessing which category maps to which account code... we
    want to line it up with how they've been mapped in the past."

    That is not a request to guess. A prior mapping is a recorded human decision, so it
    is evidence: shown with the deal it came from and when, and preferred from THIS
    deal's own history before anyone else's.

    Read from `argus_cashflows`, where every mapped Argus line already carries
    (line_item, coa_account, category, vcode), and from mappings committed through this
    screen. A name nobody has mapped simply gets nothing.
    """
    from sqlalchemy import bindparam

    wanted = {_norm_label(l) for l in labels if str(l or '').strip()}
    if not wanted:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT line_item, coa_account, category, vcode, MAX(created_at) AS seen,
                       COUNT(*) AS n
                  FROM argus_cashflows
                 WHERE coa_account IS NOT NULL
                 GROUP BY line_item, coa_account, category, vcode
            """)).fetchall()
    except Exception as e:
        logger.info("No mapping history available: %s", e)
        return {}

    for line_item, acct, cat, rv, seen, n in rows:
        key = _norm_label(line_item)
        if key not in wanted:
            continue
        cand = {
            'account': str(int(acct)) if acct is not None else None,
            'category': cat,
            'vcode': rv,
            'last_seen': str(seen) if seen else None,
            'times': int(n or 0),
            'same_deal': (rv or '').upper() == (vcode or '').upper(),
        }
        prev = out.get(key)
        # This deal's own history wins; otherwise the most recently used mapping.
        if (prev is None
                or (cand['same_deal'] and not prev['same_deal'])
                or (cand['same_deal'] == prev['same_deal']
                    and (cand['last_seen'] or '') > (prev['last_seen'] or ''))):
            out[key] = cand
    return out


def record_vcode(engine, record_id: int) -> str:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT vcode FROM valuation_records WHERE id = :i"),
            {"i": record_id}).fetchone()
    if not row:
        raise ValueError(f"Valuation record {record_id} not found")
    return str(row[0])


def categories(engine, record_id: int, data: dict) -> Dict[str, Any]:
    """The category list for this record's deal, ranked by what it actually uses."""
    vcode = record_vcode(engine, record_id)
    cats = budget.category_choices(vcode, data.get("isbs_raw"))
    return {
        "vcode": vcode,
        "categories": cats,
        "used_count": sum(1 for c in cats if c["used_by_deal"]),
        # A deal with no actuals gets the full list unranked rather than nothing — the
        # caller should say so rather than implying the deal has no accounts.
        "has_history": any(c["used_by_deal"] for c in cats),
    }


def parse(engine, record_id: int, source: str, file_bytes: bytes, filename: str,
          data: dict) -> Dict[str, Any]:
    """Read the file and return its lines with a suggested mapping per line.

    `suggested` is what the screen pre-fills: the account a line STATES, else how the
    same line was mapped before. Identical for both sources -- no keyword guessing.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Expected one of {SOURCES}.")

    vcode = record_vcode(engine, record_id)
    parsed = budget.parse_budget_workbook(file_bytes, filename)
    cats = budget.category_choices(vcode, data.get("isbs_raw"))
    by_cat = {c["category"]: c for c in cats}

    suggested: Dict[str, Dict[str, Any]] = {}
    unknown_accounts: List[Dict[str, Any]] = []
    # A budget line that STATES our account number is pre-filled from it. This is not
    # the guess the flow refuses to make: 4090 in the partner's own "Account Number"
    # column, or on the end of "CAM Reimb - 4090", is our code written down, and
    # reading it is reading, not inferring. The screen marks where each pre-fill came
    # from so the analyst can see the difference.
    for line in parsed["lines"]:
        key = str(line["row"])
        if key in suggested or line["looks_like_total"]:
            continue
        acct = line.get("stated_account")
        if not acct:
            continue
        owning = budget.category_for_account(acct)
        if not owning:
            # The sheet names an account we do not carry. Saying so beats silently
            # dropping it, so it is reported and the line is left for the analyst.
            unknown_accounts.append({"row": line["row"], "label": line["label"],
                                     "account": acct})
            continue
        default_sign = next(
            (a["mri_sign"] for a in by_cat.get(owning, {}).get("accounts", [])
             if a["account"] == acct), 1)
        suggested[key] = {
            "category": owning,
            "account": acct,
            "flip": bool(line["total"]) and ((line["total"] > 0) != (default_sign > 0)),
            "from_file": True,
        }

    # How this exact line name was mapped before. A recorded human decision, not a
    # keyword rule — which is what asset management actually asked for when they said
    # they were struggling to identify the right account numbers.
    history = _mapping_history(engine, [l["label"] for l in parsed["lines"]], vcode)
    for line in parsed["lines"]:
        key = str(line["row"])
        prior = history.get(_norm_label(line["label"]))
        if not prior:
            continue
        if key not in suggested and not line["looks_like_total"]:
            owning = budget.category_for_account(prior["account"])
            if owning:
                default_sign = next(
                    (a["mri_sign"] for a in by_cat.get(owning, {}).get("accounts", [])
                     if a["account"] == prior["account"]), 1)
                suggested[key] = {
                    "category": owning,
                    "account": prior["account"],
                    "flip": bool(line["total"]) and (
                        (line["total"] > 0) != (default_sign > 0)),
                    "from_history": True,
                }
        line["prior_mapping"] = prior

    return {
        "source": source,
        "vcode": vcode,
        **parsed,
        "suggested": suggested,
        "suggested_count": len(suggested),
        "unknown_accounts": unknown_accounts,
        "history_count": sum(1 for l in parsed["lines"] if l.get("prior_mapping")),
        "proposed_lines": proposed_lines(engine, record_id, source, parsed),
        "categories": cats,
    }


def check(engine, record_id: int, source: str, parsed: Dict[str, Any],
          mapping: Dict[str, Any], data: dict) -> Dict[str, Any]:
    """Validation + reconciliation for the mapping as it currently stands."""
    vcode = record_vcode(engine, record_id)
    parsed, mapping = with_accepted_proposals(parsed, mapping, source)
    out = validate_mod.validate(parsed, mapping, vcode, data.get("isbs_raw"))
    out["source"] = source
    out["mapped_count"] = sum(1 for m in (mapping or {}).values() if m.get("account"))
    out["line_count"] = len(parsed.get("lines") or [])
    return out


def commit(engine, record_id: int, source: str, parsed: Dict[str, Any],
           mapping: Dict[str, Any], username: str, data: dict) -> Dict[str, Any]:
    """Write the mapping, refusing anything the checks block.

    The gate is re-run HERE rather than trusted from the client: the screen validates as
    the analyst types, but a commit that skipped the check would let a stale or crafted
    payload through, and this writes to the table that is the only copy of an unapproved
    budget.
    """
    vcode = record_vcode(engine, record_id)
    stored_parsed, stored_mapping = parsed, mapping     # the draft keeps what the screen sent
    parsed, mapping = with_accepted_proposals(parsed, mapping, source)
    gate = validate_mod.validate(parsed, mapping, vcode, data.get("isbs_raw"))
    if not gate["can_import"]:
        raise ValueError("; ".join(b["message"] for b in gate["blocking"]))

    if source == "budget":
        res = validate_mod.commit(engine, vcode, parsed, mapping, username)
        res["target"] = budget.SUPPLEMENT_TABLE
        res["column"] = "Budget"
        # Budgeted occupancy travels with the budget it was read from. A file with no
        # occupancy row leaves what an earlier file stored alone, and says so.
        occ = (parsed.get("occupancy") or {}).get("by_period") or {}
        if occ:
            from flask_app.services import valuation_budget_inputs as inputs
            res["occupancy_months"] = inputs.save_occupancy(
                engine, vcode, occ, parsed.get("filename") or "", username)
        else:
            res["occupancy_months"] = 0
    else:
        # Argus feeds the VALUATION column through the projection the record already
        # links, so the two sources share this screen and these rules while each writes
        # where it belongs. `argus_service.update_coa_mapping` is the existing override
        # mechanism — it has been there all along and was simply never surfaced, which
        # is why asset management could not see or correct the keyword guess.
        res = _commit_argus(engine, record_id, parsed, mapping, username)
    res["source"] = source
    res["warnings"] = gate["warnings"]
    # Keep the mapping readable after it has been applied. Clearing it here would
    # empty the screen at the moment the work succeeded, which is what made a
    # successful commit look like lost work.
    try:
        save_draft(engine, record_id, source, stored_parsed.get("filename") or "",
                   stored_parsed, stored_mapping, username)
        mark_draft_committed(engine, record_id, source)
    except Exception as e:
        # The write that matters already happened; failing to record the draft must
        # not turn a successful import into an error.
        logger.warning("Could not record mapping draft after commit: %s", e)
    return res


def _argus_category(coa: int) -> str:
    """The Argus table's own category word for an account. Mirrors ARGUS_COA_MAP, which
    uses exactly three: capex for 7050, revenue for 4xxx, expense for everything else."""
    import config
    if coa in config.CAPEX_ACCTS or str(coa) in {str(a) for a in config.CAPEX_ACCTS}:
        return "capex"
    return "revenue" if str(coa).startswith("4") else "expense"


def _commit_argus(engine, record_id: int, parsed: Dict[str, Any],
                  mapping: Dict[str, Any], username: str) -> Dict[str, Any]:
    """Write the Valuation cash flow from THIS screen's reading of the file.

    One upload (asset management, Sep 28 2026): choose the file here, map it, apply,
    and the Valuation Yr 1 column shows it. The mapped lines ARE the cash flow -- each
    line's monthly amounts under the account the analyst chose -- so nothing is
    translated back by label onto an import some other parser made.

    Signs come from the ACCOUNT, through `argus_service._normalize_amount`, the same
    function every Argus import has always used; the flip box does not reach this path
    and the screen does not offer it.

    The record's linked import is REPLACED IN PLACE when this record is the only one
    linking it -- re-applying a mapping must not leave a stale projection in Deal
    Analysis's dropdown per revision. When another record links the same import (an
    identical file imported on two cycles shared one), a new import is made instead,
    so one cycle's revision cannot move another cycle's Valuation column.
    """
    import hashlib
    import json
    from flask_app.services import argus_service

    valuation_service._require_not_approved(engine, record_id)
    with engine.connect() as conn:
        rec = conn.execute(text("""
            SELECT r.vcode, r.argus_import_id, c.year FROM valuation_records r
            JOIN valuation_cycles c ON c.id = r.cycle_id WHERE r.id = :i
        """), {"i": record_id}).fetchone()
    if not rec:
        raise ValueError(f"Valuation record {record_id} not found")
    vcode, linked, year = str(rec[0]), rec[1], rec[2]

    by_row = {l["row"]: l for l in parsed["lines"]}
    rows: List[Dict[str, Any]] = []
    accounts = set()
    for row_key, m in (mapping or {}).items():
        if not m.get("account"):
            continue
        line = by_row.get(int(row_key))
        if not line:
            continue
        try:
            coa = int(str(m["account"]).strip())
        except ValueError:
            continue                      # a non-numeric account cannot be an Argus COA
        accounts.add(str(coa))
        for period, value in (line.get("amounts") or {}).items():
            amount = float(value)
            if amount == 0.0:
                continue
            rows.append({"pd": period, "li": line["label"], "coa": coa, "amt": amount,
                         "norm": argus_service._normalize_amount(coa, amount),
                         "cat": _argus_category(coa)})
    if not rows:
        raise ValueError("Nothing to apply — no lines have an account assigned.")

    # What was applied, not the file bytes: the same file re-mapped is a different cash
    # flow, and the hash is what says so.
    digest = hashlib.sha256(json.dumps(
        {"file": parsed.get("filename"), "rows": sorted(
            (r["pd"], r["li"], r["coa"], r["amt"]) for r in rows)},
        sort_keys=True, default=str).encode()).hexdigest()

    with engine.begin() as conn:
        shared = False
        if linked:
            owner = conn.execute(text("SELECT vcode FROM argus_imports WHERE id = :i"),
                                 {"i": int(linked)}).fetchone()
            others = conn.execute(text(
                "SELECT COUNT(*) FROM valuation_records WHERE argus_import_id = :i AND id <> :r"),
                {"i": int(linked), "r": record_id}).scalar() or 0
            shared = others > 0 or not owner or str(owner[0]) != vcode
        if linked and not shared:
            import_id = int(linked)
            replaced = conn.execute(text("DELETE FROM argus_cashflows WHERE import_id = :i"),
                                    {"i": import_id}).rowcount or 0
            conn.execute(text("""
                UPDATE argus_imports SET original_filename = :f, file_hash = :h,
                    imported_by = :u, updated_at = CURRENT_TIMESTAMP WHERE id = :i
            """), {"f": parsed.get("filename"), "h": digest, "u": username, "i": import_id})
        else:
            replaced = 0
            import_id = conn.execute(text("""
                INSERT INTO argus_imports (vcode, import_label, import_type,
                    original_filename, file_hash, is_active, imported_by)
                VALUES (:v, :l, 'valuation', :f, :h, TRUE, :u) RETURNING id
            """), {"v": vcode, "l": f"{year} Valuation", "f": parsed.get("filename"),
                   "h": digest, "u": username}).fetchone()[0]
            conn.execute(text("UPDATE valuation_records SET argus_import_id = :a WHERE id = :r"),
                         {"a": int(import_id), "r": record_id})
        for r in rows:
            conn.execute(text("""
                INSERT INTO argus_cashflows (import_id, vcode, period_date, line_item,
                    coa_account, amount, amount_norm, category)
                VALUES (:iid, :v, :pd, :li, :coa, :amt, :norm, :cat)
            """), {**r, "iid": int(import_id), "v": vcode})

    logger.info("Argus cash flow for record %s -> import %s: %d rows, %d accounts (%s)",
                record_id, import_id, len(rows), len(accounts), username)
    return {"rows_written": len(rows), "rows_replaced": replaced,
            "target": "argus_cashflows", "column": "Valuation",
            "import_id": int(import_id), "periods": sorted({r["pd"] for r in rows}),
            "accounts": sorted(accounts)}

