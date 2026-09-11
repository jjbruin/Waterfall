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


def reconcile(parsed: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Spreadsheet totals against what the SELECTED lines actually produce.

    This replaces a rule that blocked on "an unmapped line carrying a value", which was
    wrong: spreadsheets contain subtotal rows and skipping them is correct. Only the
    analyst can tell a legitimately-skipped subtotal from a genuinely missed line, and a
    number they can see is what lets them. A non-zero difference is INFORMATION, not an
    error, and never blocks.

    Lines are classified by the ACCOUNT they were mapped to (4xxx / 5xxx on our chart),
    never by the partner's wording, and reported as positive magnitudes so the three rows
    read like a statement.
    """
    by_row = {l["row"]: l for l in parsed["lines"]}
    rev = exp = 0.0
    for row_key, m in (mapping or {}).items():
        line = by_row.get(int(row_key))
        if not line or not m.get("account"):
            continue
        amount = line["total"] * (-1 if m.get("flip") else 1)
        acct = str(m["account"]).strip()
        # After flipping, our convention holds: revenue negative, expense positive.
        if acct.startswith("4"):
            rev += -amount
        elif acct.startswith("5"):
            exp += amount

    stated = parsed.get("stated_totals") or {}
    computed = {"revenue": round(rev, 2), "expense": round(exp, 2),
                "noi": round(rev - exp, 2)}
    rows = []
    for k in ("revenue", "expense", "noi"):
        s = stated.get(k)
        rows.append({"line": k, "stated": s, "computed": computed[k],
                     "difference": round(computed[k] - s, 2) if s is not None else None})
    return {"rows": rows,
            "has_stated_totals": any(v is not None for v in stated.values())}


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

    # The same account twice in one month cannot both be right — one would overwrite the
    # other's meaning and the total would silently double.
    #
    # REPORTED ONCE PER CLASH, not once per month. A 12-month file with two lines on the
    # same account produced twelve identical blocking messages, which buries the one
    # thing the analyst has to fix and makes a legible panel impossible.
    seen: Dict[tuple, str] = {}
    clashes: Dict[tuple, Dict[str, Any]] = {}
    for row_key, m in mapped.items():
        line = by_row.get(int(row_key))
        if not line:
            continue
        acct = str(m["account"]).strip()
        for period in line["amounts"]:
            k = (acct, period)
            if k in seen:
                pair = (acct, seen[k], line["label"])
                c = clashes.setdefault(pair, {"months": 0})
                c["months"] += 1
            else:
                seen[k] = line["label"]
    for (acct, first, second), info in clashes.items():
        blocking.append({
            "code": "duplicate_account_month",
            "message": (f"'{first}' and '{second}' are both mapped to account {acct} — "
                        f"they collide in {info['months']} month(s). One of them needs a "
                        f"different account, or one should be left unmapped.")})

    # A line carries BOTH a category (what the analyst picked, and the row it lands on in
    # the comparison) and an account within it (what the supplement stores, and what NOI,
    # FAD, DSCR and the waterfall actually read). If they disagree, the figure appears on
    # a different row from the one the analyst chose — silently. Blocking, because there
    # is no reading of it that is intended.
    # From the SAME definition the dropdown is built from, so a category the screen
    # offers can always be satisfied. See `_CATEGORY_ACCOUNTS_FOR_BUDGET`.
    cat_accounts = {cat: set(accts)
                    for cat, accts in budget_service.category_accounts().items()}
    for row_key, m in mapped.items():
        cat = m.get("category")
        if not cat:
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

        if acct in _BELOW_THE_LINE:
            warnings.append({
                "code": "below_the_line",
                "message": (f"'{line['label']}' is mapped to {acct}, which sits below NOI "
                            f"— it will not affect the NOI comparison.")})

    recon = reconcile(parsed, mapping)
    noi = next((r for r in recon["rows"] if r["line"] == "noi"), None)
    if noi and noi["computed"] < 0:
        warnings.append({"code": "negative_noi",
                         "message": f"Budgeted NOI is negative ({noi['computed']:,.0f})."})

    return {"blocking": blocking, "warnings": warnings, "reconciliation": recon,
            "can_import": not blocking}


def commit(engine, vcode: str, parsed: Dict[str, Any], mapping: Dict[str, Any],
           username: str) -> Dict[str, Any]:
    """Replace this deal's budget rows for the imported months, then insert.

    REPLACE, not append: a budget is re-imported as many times as it takes to reach a
    final version, and appending would stack every revision on the last. Scoped to
    (vcode, the periods in THIS file) so a deal budgeted across two files does not have
    its first import erased by its second.

    Every column name is DOUBLE-QUOTED. This table is created by pandas `to_sql`, so on
    PostgreSQL its columns really are `vcode`/`dtEntry`/`vAccount`, and unquoted SQL
    resolves to lower case and raises `column does not exist` — the defect that made the
    valuation publish path fail on Azure while working on every local run. See
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
    with engine.begin() as conn:
        deleted = 0
        for period in periods:
            res = conn.execute(
                text(f'DELETE FROM {SUPPLEMENT_TABLE} '
                     f'WHERE "vcode" = :v AND "dtEntry" = :d'),
                {"v": vcode, "d": period})
            deleted += res.rowcount or 0
        for r in rows:
            conn.execute(
                text(f'INSERT INTO {SUPPLEMENT_TABLE} '
                     f'("vcode", "dtEntry", "vSource", "vAccount", "mAmount", "vInput") '
                     f'VALUES (:vcode, :dtEntry, :vSource, :vAccount, :mAmount, :vInput)'),
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


def _ensure_table(engine) -> None:
    cols = ('"vcode" TEXT, "dtEntry" TEXT, "vSource" TEXT, "vAccount" TEXT, '
            '"mAmount" DOUBLE PRECISION, "vInput" TEXT')
    if engine.dialect.name != "postgresql":
        cols = cols.replace("DOUBLE PRECISION", "REAL")
    with engine.begin() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {SUPPLEMENT_TABLE} ({cols})"))
