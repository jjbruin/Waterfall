"""One line-mapping flow for both sources that feed the budget comparison.

The comparison `get_budget_review` renders is Estimate | Budget | Valuation, ~27 category
rows from `config.IS_ACCOUNTS`. TWO of those columns are loaded from a spreadsheet:

    Budget      <- the partner's monthly budget workbook
    Valuation   <- the appraiser's Argus download

They are the same job — take somebody else's line names, decide which of our categories
each one belongs to, check the result, write it — so they get the same screen. The ONLY
material difference is that Argus arrives with a guess already made, by the 56 keyword
rules in `argus_parser.ARGUS_COA_MAP`. Making that guess visible and editable is what
asset management asked for when they said they "can't see or interact with code mapping
for valuation": today it happens silently at import.

A budget line arrives unassigned, because a partner's wording is their own and guessing
would recreate exactly the invisible-mapping problem this flow exists to remove.
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

    `suggested` is what the screen pre-fills. For a budget it is empty by design. For
    Argus it is `argus_parser.map_to_coa`, which returns (account, category) — the same
    pair the mapper works in — surfaced here instead of being applied invisibly.
    """
    if source not in SOURCES:
        raise ValueError(f"Unknown source '{source}'. Expected one of {SOURCES}.")

    vcode = record_vcode(engine, record_id)
    parsed = budget.parse_budget_workbook(file_bytes, filename)
    cats = budget.category_choices(vcode, data.get("isbs_raw"))
    by_cat = {c["category"]: c for c in cats}

    suggested: Dict[str, Dict[str, Any]] = {}
    unknown_accounts: List[Dict[str, Any]] = []
    if source == "argus":
        from argus_parser import map_to_coa
        for line in parsed["lines"]:
            if line["looks_like_total"]:
                continue          # subtotals stay unmapped whatever the keywords say
            acct, cat = map_to_coa(line["label"])
            if not acct:
                continue
            acct = str(acct)
            # The keyword map's category is advisory; the authority is which of OUR
            # categories actually contains the account, since that is what decides the
            # row. Where they disagree, the account wins and the discrepancy is visible
            # because the screen shows both.
            owning = budget.category_for_account(acct) or cat
            if not owning:
                continue
            default_sign = next(
                (a["mri_sign"] for a in by_cat.get(owning, {}).get("accounts", [])
                 if a["account"] == acct), 1)
            suggested[str(line["row"])] = {
                "category": owning,
                "account": acct,
                # A source file that already agrees with MRI's sign needs no flip. The
                # line's own total decides, against how the account behaves for this deal.
                "flip": bool(line["total"]) and (
                    (line["total"] > 0) != (default_sign > 0)),
                "from_keywords": True,
            }

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
        save_draft(engine, record_id, source, parsed.get("filename") or "",
                   parsed, mapping, username)
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
    """Apply the analyst's mapping as COA overrides on the record's Argus import.

    Argus rows are keyed by LINE ITEM TEXT, not by spreadsheet row, so the mapping is
    translated back to labels here. A label appearing twice in the file would otherwise
    have its later mapping silently win for both — that case is already blocked upstream
    as a duplicate account-month, but the translation is done explicitly so the failure
    would be visible rather than implicit.
    """
    from flask_app.services import argus_service

    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT argus_import_id FROM valuation_records WHERE id = :i"),
            {"i": record_id}).fetchone()
    import_id = row[0] if row else None
    if not import_id:
        raise ValueError(
            "This record has no Argus import to map. Upload the appraiser's Argus "
            "download on the record first, then review its mapping here.")

    by_row = {l["row"]: l for l in parsed["lines"]}
    updates: List[Dict[str, Any]] = []
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
        # NOT our category name. `argus_cashflows.category` has its own three-word
        # vocabulary from ARGUS_COA_MAP — revenue / expense / capex — and writing
        # "Rental Income" into it would leave the column holding two different kinds of
        # word depending on who last touched the row. Our category is the screen's
        # concern; the account is what both layers agree on, so derive from it.
        updates.append({"line_item": line["label"], "coa_account": coa,
                        "category": _argus_category(coa)})
    if not updates:
        raise ValueError("Nothing to apply — no lines have an account assigned.")

    argus_service.update_coa_mapping(engine, int(import_id), updates)
    logger.info("Argus mapping for record %s (import %s): %d line item(s) set by %s",
                record_id, import_id, len(updates), username)
    return {"rows_written": len(updates), "rows_replaced": 0,
            "target": "argus_cashflows", "column": "Valuation",
            "import_id": int(import_id),
            "accounts": sorted({str(u["coa_account"]) for u in updates})}
