"""Validation, reconciliation and commit for the budget import.

Split from `budget_import_service` (which reads the spreadsheet and offers the account
list) because this half is where the judgment lives and it is the half that will change
as the team learns what actually goes wrong.

BLOCKING IS RESERVED for things that would make the imported data WRONG. Everything a
competent analyst might legitimately intend is a warning — a check people learn to click
past protects nobody, and the previous draft of these rules blocked on two things
(unmapped lines with a value; a strict sign convention) that are routinely correct.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd
from sqlalchemy import text

from flask_app.services import budget_import_service as budget_service
from flask_app.services.budget_import_service import (
    MAGNITUDE_HIGH, MAGNITUDE_LOW, SUPPLEMENT_TABLE, _BELOW_THE_LINE, account_choices,
)

logger = logging.getLogger(__name__)


#: Warnings that mean a figure is probably wrong, rather than merely worth knowing.
CRITICAL_WARNINGS = {"sign_opposite_prior", "magnitude", "negative_noi"}


def reconcile(parsed: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Spreadsheet totals against what the SELECTED lines actually produce.

    This replaces a rule that blocked on "an unmapped line carrying a value", which was
    wrong: spreadsheets contain subtotal rows and skipping them is correct. Only the
    analyst can tell a legitimately-skipped subtotal from a genuinely missed line, and a
    number they can see is what lets them. A non-zero difference is INFORMATION, not an
    error, and never blocks.

    Lines are classified by the ACCOUNT they were mapped to, never by the partner's
    wording, and reported as positive magnitudes so the three rows read like a statement.

    NOI HERE IS THE BUDGET COLUMN'S NOI, from the same `IS_ACCOUNTS` sections
    `valuation_service.get_budget_review` sums. This used to classify by prefix -- any
    4xxx revenue, any 5xxx expense -- which is a second definition of NOI: it counted
    interest (5190), partnership costs (5120/5130), depreciation, 5195/5210/5220/5400
    and interest income (4050) inside NOI, and missed the tax abatement (7070) that the
    comparison folds into expenses. Asset management, Sep 25 2026: "The NOI per the
    import sheet and what's getting populated in the 2027 budget column are tying out
    but this section is saying the NOI is not tying out." The proposed $20K partnership
    line to 5130 alone put a $20,000 difference on every import that accepted it.
    What is mapped below the line is reported beside the three rows, not dropped.
    """
    import config
    rev_accts = {str(a) for accts in config.IS_ACCOUNTS["REVENUES"].values() for a in accts}
    exp_accts = {str(a) for accts in config.IS_ACCOUNTS["EXPENSES"].values() for a in accts}
    by_row = {l["row"]: l for l in parsed["lines"]}
    rev = exp = 0.0
    outside: Dict[str, float] = {}
    for row_key, m in (mapping or {}).items():
        line = by_row.get(int(row_key))
        if not line or not m.get("account"):
            continue
        amount = line["total"] * (-1 if m.get("flip") else 1)
        acct = str(m["account"]).strip()
        # After flipping, our convention holds: revenue negative, expense positive.
        if acct in rev_accts:
            rev += -amount
        elif acct in exp_accts:
            exp += amount
        else:
            outside[acct] = outside.get(acct, 0.0) + amount

    stated = parsed.get("stated_totals") or {}
    computed = {"revenue": round(rev, 2), "expense": round(exp, 2),
                "noi": round(rev - exp, 2)}
    rows = []
    for k in ("revenue", "expense", "noi"):
        s = stated.get(k)
        rows.append({"line": k, "stated": s, "computed": computed[k],
                     "difference": round(computed[k] - s, 2) if s is not None else None})
    return {"rows": rows,
            "has_stated_totals": any(v is not None for v in stated.values()),
            # Mapped, imported, and outside NOI: debt service, partnership costs,
            # capex and the rest. If the partner's own NOI line includes one of these,
            # this is where the difference comes from.
            "outside_noi": [{"account": a, "amount": round(v, 2)}
                            for a, v in sorted(outside.items()) if round(v, 2) != 0]}


def validate(parsed: Dict[str, Any], mapping: Dict[str, Any], vcode: str,
             isbs_raw: pd.DataFrame) -> Dict[str, Any]:
    """Blocking problems, warnings, and the reconciliation, in one payload."""
    blocking: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    by_row = {l["row"]: l for l in parsed["lines"]}
    mapped = {k: m for k, m in (mapping or {}).items() if m.get("account")}

    if not mapped:
        blocking.append({"code": "no_lines",
                         "message": "No lines have been assigned an account."})

    # SEVERAL LINES ON ONE ACCOUNT ADD UP. They do not collide.
    #
    # Jack, Sep 22 2026: "Our partners budget in far more detail than our chart of
    # accounts carries, so fifteen of their repair lines all belong in one account
    # on our side. The app treats every line after the first as a conflict and
    # won't let us submit." That blocking rule is what forced him to build a
    # rolled-up summary block by hand and delete the detail -- 60 budget lines
    # reduced to 19, with everything that made them checkable thrown away.
    #
    # The old comment claimed the total "would silently double". It would not:
    # ISBS is a JOURNAL, one key legitimately carries many rows and every consumer
    # SUMS them, which is the same property `drop_duplicates` violated at a cost
    # of 358,392 rows (see CLAUDE.md). Two lines on one account-month is the
    # correct representation of two budget lines landing in one account.
    #
    # What the rule was really protecting against is the analyst not SEEING it, so
    # the roll-up is reported instead of refused: every account that takes more
    # than one line is named, with the lines and the combined total.
    rollup: Dict[str, Dict[str, Any]] = {}
    for row_key, m in mapped.items():
        line = by_row.get(int(row_key))
        if not line:
            continue
        acct = str(m["account"]).strip()
        entry = rollup.setdefault(acct, {"labels": [], "total": 0.0})
        entry["labels"].append(line["label"])
        flip = -1 if m.get("flip") else 1
        entry["total"] += sum(line["amounts"].values()) * flip
    combined = {a: e for a, e in rollup.items() if len(e["labels"]) > 1}
    for acct, e in sorted(combined.items()):
        warnings.append({
            "code": "lines_combined",
            "message": (f"{len(e['labels'])} lines add together into account {acct}: "
                        f"{', '.join(e['labels'][:4])}"
                        + (f" and {len(e['labels']) - 4} more" if len(e['labels']) > 4
                           else "")
                        + f" — combined {e['total']:,.2f}.")})

    # A line carries BOTH a category (what the analyst picked, and the row it lands on in
    # the comparison) and an account within it (what the supplement stores, and what NOI,
    # FAD, DSCR and the waterfall actually read). If they disagree, the figure appears on
    # a different row from the one the analyst chose — silently. Blocking, because there
    # is no reading of it that is intended.
    # From the SAME definition the dropdown is built from, so a category the screen
    # offers can always be satisfied. See `_CATEGORY_ACCOUNTS_FOR_BUDGET`.
    # THE ACCOUNT DECIDES THE CATEGORY, so they can no longer disagree. Whatever the
    # screen sends is overwritten with the category that owns the account -- one
    # source of mapping, which is what was asked for. A category that came in
    # disagreeing is not an error to report; it is a field that should not have been
    # an input.
    for m in mapped.values():
        derived = budget_service.category_for_account(m.get("account"))
        if derived:
            m["category"] = derived
    cat_accounts = {cat: set(accts)
                    for cat, accts in budget_service.category_accounts().items()}
    for row_key, m in mapped.items():
        cat = m.get("category")
        if not cat:
            # No category means we do not carry the account at all, which IS worth
            # blocking: the figure would land on no row of the comparison.
            line = by_row.get(int(row_key))
            blocking.append({
                "code": "account_not_in_any_category",
                "message": (f"Account {str(m['account']).strip()} "
                            f"({(line or {}).get('label', 'this line')}) is not on "
                            f"our chart of accounts, so it has no row on the "
                            f"comparison.")})
            continue
        acct = str(m["account"]).strip()
        valid = cat_accounts.get(cat)
        if valid is None:
            blocking.append({
                "code": "unknown_category",
                "message": f"'{cat}' is not a category on the budget comparison."})
        elif acct not in valid:
            line = by_row.get(int(row_key))
            blocking.append({
                "code": "account_not_in_category",
                "message": (f"'{(line or {}).get('label', row_key)}' is mapped to "
                            f"{cat} but account {acct}, which is not in that category "
                            f"({', '.join(sorted(valid))}).")})

    n_periods = len(parsed.get("periods") or [])
    for row_key, m in mapped.items():
        line = by_row.get(int(row_key))
        # A partial year is legitimate for a mid-year acquisition, so this warns.
        if line and 0 < line["months"] < n_periods:
            warnings.append({
                "code": "incomplete_year",
                "message": f"'{line['label']}' has {line['months']} of {n_periods} months."})

    prior = {c["account"]: c for c in account_choices(vcode, isbs_raw)}
    used = {str(m["account"]).strip() for m in mapped.values()}

    # Accounts the deal used in the last 12 months with nothing budgeted against them.
    #
    # ONE warning listing them, not one each. A deal with 17 used accounts and a 3-line
    # file emitted fifteen near-identical lines, which is the same as emitting none. The
    # per-account detail is kept in `accounts` so the screen can expand it on demand.
    missing = [(a, prior[a]) for a in sorted(prior) if a not in used]
    if missing:
        preview = ", ".join(f"{a} {i['description']}" for a, i in missing[:4])
        if len(missing) > 4:
            preview += f", and {len(missing) - 4} more"
        warnings.append({
            "code": "account_not_budgeted",
            "message": (f"{len(missing)} account(s) used in the last 12 months have "
                        f"nothing mapped to them: {preview}."),
            "accounts": [{"account": a, "description": i["description"],
                          "months": i["months"], "prior_total": i["prior_total"]}
                         for a, i in missing]})

    for row_key, m in mapped.items():
        line = by_row.get(int(row_key))
        if not line:
            continue
        acct = str(m["account"]).strip()
        amount = line["total"] * (-1 if m.get("flip") else 1)

        if acct not in prior:
            warnings.append({
                "code": "unknown_account",
                "message": (f"'{line['label']}' is mapped to {acct}, which this deal has "
                            f"not used in the last 12 months.")})
        else:
            base = prior[acct]["prior_total"]
            if base:
                ratio = amount / base
                if ratio <= 0:
                    warnings.append({
                        "code": "sign_opposite_prior",
                        "message": (f"'{line['label']}' to {acct} has the OPPOSITE sign to "
                                    f"the last 12 months ({amount:,.0f} vs {base:,.0f}). "
                                    f"Check the flip.")})
                elif ratio < MAGNITUDE_LOW or ratio > MAGNITUDE_HIGH:
                    # The classic failure is a budget entered in thousands, which lands
                    # around 0.001x and is impossible to miss once stated as a ratio.
                    warnings.append({
                        "code": "magnitude",
                        "message": (f"'{line['label']}' to {acct} is {ratio:.2f}x the last "
                                    f"12 months ({amount:,.0f} vs {base:,.0f}).")})

        # (The per-line "sits below NOI" warning is gone: `reconcile` now reports
        # everything mapped outside NOI in one place, from the comparison's own
        # sections rather than a hand-kept four-account set that missed 5130.)

    recon = reconcile(parsed, mapping)
    noi = next((r for r in recon["rows"] if r["line"] == "noi"), None)
    if noi and noi["computed"] < 0:
        warnings.append({"code": "negative_noi",
                         "message": f"Budgeted NOI is negative ({noi['computed']:,.0f})."})

    # CRITICAL vs everything else. Asset management, Sep 25 2026: "need to have
    # critical checks only and condense this area ... Not sure what a lot of the checks
    # are implying." A 50-line partner file produced a warning per combined account,
    # per partial line and per account new to the deal, and the two that mean the
    # number is probably WRONG -- a sign opposite to the deal's history, a figure off
    # by an order of magnitude -- were lost among them. Critical ones are shown; the
    # rest are kept, counted, and folded away. Nothing is dropped.
    for w in warnings:
        w["critical"] = w["code"] in CRITICAL_WARNINGS

    return {"blocking": blocking, "warnings": warnings, "reconciliation": recon,
            "can_import": not blocking}


def commit(engine, vcode: str, parsed: Dict[str, Any], mapping: Dict[str, Any],
           username: str) -> Dict[str, Any]:
    """Replace this deal's budget rows for the imported months, then insert.

    REPLACE, not append: a budget is re-imported as many times as it takes to reach a
    final version, and appending would stack every revision on the last. Scoped to
    (vcode, the periods in THIS file) so a deal budgeted across two files does not have
    its first import erased by its second.

    Every column name is DOUBLE-QUOTED, AND READ FROM THE TABLE. Quoting alone is not
    enough and this docstring used to claim it was: it asserted the columns "really
    are `vcode`", which was true of the table pandas creates locally and false of the
    one production has. See the note on the DELETE below and
    scripts/sql_mixedcase_identifier_check.py.
    """
    by_row = {l["row"]: l for l in parsed["lines"]}
    rows: List[Dict[str, Any]] = []
    for row_key, m in (mapping or {}).items():
        if not m.get("account"):
            continue
        line = by_row.get(int(row_key))
        if not line:
            continue
        flip = -1 if m.get("flip") else 1
        for period, value in line["amounts"].items():
            rows.append({"vcode": vcode, "dtEntry": period, "vSource": "Budget IS",
                         "vAccount": str(m["account"]).strip(),
                         "mAmount": float(value) * flip,
                         "vInput": f"{line['label']} [{username}]"})
    if not rows:
        raise ValueError("Nothing to import — no lines have an account assigned.")

    periods = sorted({r["dtEntry"] for r in rows})
    _ensure_table(engine)
    # THE COLUMN NAMES ARE READ FROM THE TABLE, NEVER ASSUMED. Production's
    # supplement tables were created by a CSV import and carry `vCode`; the ISBS
    # tables the MRI refresh creates carry `vcode`. A double-quoted identifier is
    # CASE-SENSITIVE on PostgreSQL and case-insensitive on SQLite, so
    # `WHERE "vcode" = ...` worked in every local test and raised
    # UndefinedColumn on production -- the v435 / v496 shape. Every budget import
    # through this path has failed, for every file, since it was written: the
    # DELETE raises, the transaction rolls back, and the analyst sees an empty
    # Budget column after doing the work. Jack rebuilt his spreadsheet eight times
    # against a bug no spreadsheet could have fixed.
    cols = _supplement_columns(engine)
    with engine.begin() as conn:
        deleted = 0
        for period in periods:
            res = conn.execute(
                text(f'DELETE FROM {SUPPLEMENT_TABLE} '
                     f'WHERE "{cols["vcode"]}" = :v AND "{cols["dtEntry"]}" = :d'),
                {"v": vcode, "d": period})
            deleted += res.rowcount or 0
        insert_cols = ', '.join('"%s"' % cols[k] for k in _SUPPLEMENT_FIELDS)
        insert_vals = ', '.join(':%s' % k for k in _SUPPLEMENT_FIELDS)
        for r in rows:
            conn.execute(
                text(f'INSERT INTO {SUPPLEMENT_TABLE} ({insert_cols}) '
                     f'VALUES ({insert_vals})'),
                r)

    # isbs_raw is reassembled from the supplement tables, so the valuation comparison
    # sees this without a restart. refresh_table routes supplements through the ISBS
    # reassembly branch — see scripts/refresh_table_key_check.py.
    try:
        from flask_app.services import data_service
        data_service.refresh_table(SUPPLEMENT_TABLE)
    except Exception as e:                                    # noqa: BLE001
        logger.warning("Budget import: cache refresh failed (%s) — reload to see it", e)

    logger.info("Budget import %s: %d rows over %d periods (replaced %d)",
                vcode, len(rows), len(periods), deleted)
    return {"rows_written": len(rows), "rows_replaced": deleted, "periods": periods,
            "accounts": sorted({r["vAccount"] for r in rows})}


#: The fields this import writes, in our own vocabulary. `_supplement_columns`
#: maps each to whatever the table actually calls it.
_SUPPLEMENT_FIELDS = ("vcode", "dtEntry", "vSource", "vAccount", "mAmount", "vInput")


def _supplement_columns(engine) -> Dict[str, str]:
    """Our field names mapped to the table's ACTUAL column names.

    Matched case-insensitively, because the same logical column is `vCode` on the
    CSV-created supplement tables and `vcode` on the MRI-created ones, and quoting
    the wrong one is a PostgreSQL-only failure that SQLite cannot reproduce.

    Creates the table first, so a fresh database resolves against the DDL below
    rather than against nothing. A field the table does not have at all is left as
    our own spelling and will fail loudly on use -- guessing a near-miss would be
    worse, since it would write to the wrong column.
    """
    from sqlalchemy import inspect
    _ensure_table(engine)
    try:
        actual = [c["name"] for c in inspect(engine).get_columns(SUPPLEMENT_TABLE)]
    except Exception as exc:                                  # noqa: BLE001
        logger.warning("Could not inspect %s (%s) -- using our own spellings",
                       SUPPLEMENT_TABLE, exc)
        actual = []
    lower = {c.lower(): c for c in actual}
    return {f: lower.get(f.lower(), f) for f in _SUPPLEMENT_FIELDS}


def _ensure_table(engine) -> None:
    cols = ('"vcode" TEXT, "dtEntry" TEXT, "vSource" TEXT, "vAccount" TEXT, '
            '"mAmount" DOUBLE PRECISION, "vInput" TEXT')
    if engine.dialect.name != "postgresql":
        cols = cols.replace("DOUBLE PRECISION", "REAL")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {SUPPLEMENT_TABLE} ({cols})"))
