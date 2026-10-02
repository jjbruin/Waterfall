"""Intercompany: the Due to/from PSC Manager reconciliation.

The CFO's `BORG_Intercompany Template.xlsx`, tab `DTM Recon Template`, as a screen.
Design and the measurements behind it: `.claude/memory/intercompany.md`.

WHAT IT RECONCILES. Every entity carries what it owes PSC Manager on
`MR15000002` (Due To/From PSC Manager); PSC Manager carries the other side on
`MR15000001` (Due to/from Intercompany), one segment per entity in
`RLTDENTITY`. The two should net to zero:

    A  entity balance       ENTITYID = entity, ACCTNUM = MR15000002
    B  alternate account    the entity's own second account, where it has one
    C  = A + B
    D  manager balance      ENTITYID = PSCMAN, ACCTNUM = MR15000001, RLTDENTITY = entity
    variance = C + D        (entity owes = negative, manager is owed = positive)

THE SOURCE IS `gl_detail`, NOT A NEW QUERY. His three Spreadsheet Server queries
are all "NEW JOURNAL and GHIS Balance Forward", which `queries/MRI_GL_Detail.sql`
already replicates; his DEFINED_CODE is our RLTDENTITY. Measured on production
Sep 29 2026, 202601-202609: entity side -308,316.59 and manager side 313,093.97,
both to the cent, 0 of 61 and 0 of 64 entities differing.

BASIS A AND B ONLY, AND THAT IS LOAD-BEARING. `gl_detail` also holds basis C
(4,622 rows in 2026) and T; with them the same sums read -160,919.94 and
101,804.92. His queries are `A.B`, so this is `A.B`, stated on the screen.

THE YEAR'S OPENING IS A BALANCE-FORWARD ROW AT YYYY01 (BALFOR 'B', basis A), so a
balance at period P is the sum from YYYY01 through P. Where that opening is not
in the GL -- a new year before it is rolled -- the sum would silently be the
year's activity alone, so it is reported as a check instead.

THE ENTITY LIST IS DERIVED. His column B is typed, which is why his first two
checks exist ("entity missing from column B", "segment missing"). Here every
entity with a row on either side is a row, so those two cannot fail; the
blank-segment check stays because it is a real data condition.

CASH IS READ FROM THE GL, not typed. Default `MR1000*`; an entity may name its
own cash accounts instead, or exclude some. 61 of his 63 typed balances equal
the `MR1000*` total; the two that do not are why the settings exist (PSC1 omits
its Liberty Bank money market; Pegasus's cash is `MR99991000`).

NOTHING HERE WRITES TO THE GL. Settings and comments are the app's own.
"""

from __future__ import annotations

import io
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import bindparam, text

from flask_app.db import get_engine

logger = logging.getLogger(__name__)

ENTITY_ACCOUNT = "MR15000002"      # Due To/From PSC Manager, on each entity
MANAGER_ENTITY = "PSCMAN"
MANAGER_ACCOUNT = "MR15000001"     # Due to/from Intercompany, on PSC Manager
BASES = ("A", "B")
DEFAULT_TOLERANCE = 1.0
CASH_PREFIX = "MR1000"
CURRENCIES = ("USD", "CAD")

_ACCOUNT_RE = re.compile(r"^MR\d{8}$")
_PERIOD_RE = re.compile(r"^\d{4}(0[1-9]|1[0-2])$")

#: The per-entity settings the CFO stated, written ONCE into an empty settings
#: table and editable on the screen from then on. Seeded as DATA rather than
#: kept as a constant so a change is an edit on the screen, never a deploy, and
#: each row says where it came from.
_SEED_SETTINGS = [
    dict(entity_id="PEGASU", alt_account="MR99991102", cash_accounts="MR99991000",
         cash_exclude="", currency="USD",
         basis="CFO, Sep 29 2026: alternate account MR99991102 Due To/From "
               "Property Manager; cash is MR99991000 Cash - Bank."),
    dict(entity_id="NOTTNV", alt_account="MR99982002", cash_accounts="",
         cash_exclude="", currency="USD",
         basis="CFO, Sep 29 2026: alternate account MR99982002 Accounts "
               "Payable - Other stays."),
    dict(entity_id="PPI2", alt_account="", cash_accounts="",
         cash_exclude=",".join(("MR10006000", "MR10003000")), currency="USD",
         basis=None),
    dict(entity_id="PSC2", alt_account="", cash_accounts="",
         cash_exclude=",".join(("MR10006000", "MR10003000")), currency="USD",
         basis=None),
    dict(entity_id="PSC1", alt_account="", cash_accounts="",
         cash_exclude="MR10008000", currency="USD", basis=None),
]
#: The CFO's second answers, Sep 30 2026. Also applied to a row already seeded
#: from the first ones -- but ONLY while it is still the seed's, so a setting an
#: accountant has since changed is never put back.
_CAD_TO_USD_BASIS = ("CFO, Sep 30 2026: default to the entity's USD cash accounts "
                     "(MR1000* less the two Canada accounts MR10006000 and "
                     "MR10003000) -- the CAD is converted separately.")
_LIBERTY_BASIS = ("CFO, Sep 30 2026: confirmed -- PSC1's cash excludes the Liberty "
                  "Bank money market MR10008000.")
for _s in _SEED_SETTINGS:
    if _s["basis"] is None:
        _s["basis"] = _LIBERTY_BASIS if _s["entity_id"] == "PSC1" else _CAD_TO_USD_BASIS

#: Fixed wording (CFO, Sep 30 2026), exactly as `JE Template` writes it.
JE_ENTITY_DESCRIPTION = "Intercompany Reimbursement to PSC Manager"
JE_MANAGER_DESCRIPTION = "Intercompany Reimbursement from %s"
#: PSC Manager receives into its PNC operating account on every line of the
#: CFO's template.
MANAGER_CASH_ACCOUNT = "MR10005000"

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ic_entity_settings (
        entity_id      TEXT PRIMARY KEY,
        alt_account    TEXT,
        cash_accounts  TEXT,
        cash_exclude   TEXT,
        currency       TEXT,
        basis          TEXT,
        updated_by     TEXT,
        updated_at     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ic_recon_notes (
        period      TEXT NOT NULL,
        entity_id   TEXT NOT NULL,
        comment     TEXT,
        updated_by  TEXT,
        updated_at  TEXT,
        PRIMARY KEY (period, entity_id)
    )
    """,
    # What the accountant decided for one entity in one period. The CAN AFFORD
    # figure may be adjusted, with a reason, because the GL cash is not all
    # spendable -- AMB6 withholds cash from distributions for tax and audit
    # accruals and reimburses its formation costs $6,750 at a time (CFO, Sep 30
    # 2026). `amount` NULL means "the prompt", i.e. what it can afford.
    """
    CREATE TABLE IF NOT EXISTS ic_pay_rows (
        period          TEXT NOT NULL,
        entity_id       TEXT NOT NULL,
        afford_override DOUBLE PRECISION,
        afford_reason   TEXT,
        amount          DOUBLE PRECISION,
        cash_account    TEXT,
        updated_by      TEXT,
        updated_at      TEXT,
        PRIMARY KEY (period, entity_id)
    )
    """,
    # A generated journal entry. Until the GL shows it, the balance it pays
    # still reads unpaid, so without this the same reimbursement can be paid
    # twice between the download and the next MRI refresh.
    """
    CREATE TABLE IF NOT EXISTS ic_je_batches (
        batch_id    TEXT PRIMARY KEY,
        period      TEXT NOT NULL,
        entrdate    TEXT NOT NULL,
        entities    TEXT NOT NULL,
        csv         TEXT NOT NULL,
        total       DOUBLE PRECISION,
        created_by  TEXT,
        created_at  TEXT,
        voided_by   TEXT,
        voided_at   TEXT
    )
    """,
]

_DDL_DONE = set()


def _apply_sep30_answers(conn) -> None:
    """The CFO's Sep 30 answers onto rows the Sep 29 seed wrote. Idempotent, and
    a row an accountant has edited (updated_by is not 'seed') is left alone."""
    conn.execute(text(
        "UPDATE ic_entity_settings SET cash_accounts = '', cash_exclude = :x, "
        "currency = 'USD', basis = :b WHERE entity_id IN ('PPI2', 'PSC2') "
        "AND updated_by = 'seed' AND currency = 'CAD'"),
        {"x": "MR10006000,MR10003000", "b": _CAD_TO_USD_BASIS})
    conn.execute(text(
        "UPDATE ic_entity_settings SET basis = :b WHERE entity_id = 'PSC1' "
        "AND updated_by = 'seed' AND basis LIKE 'INFERRED%'"), {"b": _LIBERTY_BASIS})


def ensure_tables(engine=None) -> None:
    engine = engine or get_engine()
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DDL_DONE:
        return
    with engine.begin() as conn:
        for ddl in _DDL:
            conn.execute(text(ddl))
        # Seeded only into an EMPTY table. Re-seeding on every start would put
        # back a setting somebody deliberately cleared.
        n = conn.execute(text("SELECT COUNT(*) FROM ic_entity_settings")).scalar()
        if not n:
            now = datetime.utcnow().isoformat(timespec="seconds")
            for s in _SEED_SETTINGS:
                conn.execute(text(
                    "INSERT INTO ic_entity_settings (entity_id, alt_account, "
                    "cash_accounts, cash_exclude, currency, basis, updated_by, "
                    "updated_at) VALUES (:entity_id, :alt_account, :cash_accounts, "
                    ":cash_exclude, :currency, :basis, 'seed', :now)"),
                    {**s, "now": now})
        _apply_sep30_answers(conn)
    _DDL_DONE.add(key)


# ------------------------------------------------------------------ helpers

def _q(engine, name: str) -> str:
    """Quote a gl_detail column. Called only with the fixed names below."""
    return f'"{name}"' if engine.dialect.name == "postgresql" else f"[{name}]"


def _has_table(engine, table: str) -> bool:
    from sqlalchemy import inspect
    try:
        return table in inspect(engine).get_table_names()
    except Exception:
        return False


def _norm(v) -> str:
    return str(v).strip().upper() if v is not None else ""


def _accounts(v) -> List[str]:
    return [a for a in (_norm(x) for x in str(v or "").split(",")) if a]


def _check_period(period) -> str:
    p = str(period or "").strip()
    if not _PERIOD_RE.match(p):
        raise ValueError("Period %r is not YYYYMM." % (period,))
    return p


def _freshness(engine) -> Optional[str]:
    from flask_app.services.gl_ia_query_service import _freshness as f
    return f(engine, "gl_detail")


def get_settings(engine=None) -> Dict[str, dict]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM ic_entity_settings")).mappings().all()
    out = {}
    for r in rows:
        out[_norm(r["entity_id"])] = {
            "entity_id": _norm(r["entity_id"]),
            "alt_account": _norm(r["alt_account"]) or None,
            "cash_accounts": _accounts(r["cash_accounts"]),
            "cash_exclude": _accounts(r["cash_exclude"]),
            "currency": _norm(r["currency"]) or "USD",
            "basis": r["basis"], "updated_by": r["updated_by"],
            "updated_at": r["updated_at"],
        }
    return out


def _notes(engine, period: str) -> Dict[str, dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM ic_recon_notes WHERE period = :p"),
                         {"p": period}).mappings().all()
    return {_norm(r["entity_id"]): dict(r) for r in rows}


def _names(engine) -> Dict[str, str]:
    if not _has_table(engine, "entities"):
        return {}
    try:
        with engine.connect() as c:
            df = pd.read_sql(text("SELECT * FROM entities"), c)
    except Exception:
        logger.info("intercompany: entities unreadable", exc_info=True)
        return {}
    cols = {str(c).lower(): c for c in df.columns}
    idc, nmc = cols.get("entityid"), cols.get("name")
    if not idc or not nmc:
        return {}
    return {_norm(i): str(n or "").strip() for i, n in zip(df[idc], df[nmc])}


# ------------------------------------------------------------------ the rows

def _selection(engine, side: str, entity: Optional[str], settings: Dict[str, dict]):
    """WHICH GL ROWS MAKE A FIGURE -- the one definition the grid and the
    drilldown both use, so the entries behind a figure always add up to it.

    Returns (sql fragment, params, bindparams). `entity=None` means every entity
    (the grid); a named entity narrows it (the drilldown).
    """
    E, A, R = _q(engine, "ENTITYID"), _q(engine, "ACCTNUM"), _q(engine, "RLTDENTITY")
    params, binds = {}, []
    if side == "entity":
        sql = f"{A} = :acct"
        params["acct"] = ENTITY_ACCOUNT
        if entity:
            sql += f" AND UPPER(TRIM({E})) = :ent"
            params["ent"] = entity
    elif side == "manager":
        sql = f"UPPER(TRIM({E})) = :mgr AND {A} = :macct"
        params.update(mgr=MANAGER_ENTITY, macct=MANAGER_ACCOUNT)
        if entity:
            sql += f" AND UPPER(TRIM(COALESCE({R}, ''))) = :ent"
            params["ent"] = entity
    elif side == "alt":
        s = settings.get(entity or "")
        if not entity or not s or not s["alt_account"]:
            return None
        sql = f"UPPER(TRIM({E})) = :ent AND {A} = :alt"
        params.update(ent=entity, alt=s["alt_account"])
    elif side == "cash":
        if not entity:
            return None
        s = settings.get(entity) or {}
        params["ent"] = entity
        if s.get("cash_accounts"):
            sql = f"UPPER(TRIM({E})) = :ent AND {A} IN :cash"
            params["cash"] = s["cash_accounts"]
            binds.append(bindparam("cash", expanding=True))
        else:
            sql = f"UPPER(TRIM({E})) = :ent AND {A} LIKE :cp"
            params["cp"] = CASH_PREFIX + "%"
            if s.get("cash_exclude"):
                sql += f" AND {A} NOT IN :cx"
                params["cx"] = s["cash_exclude"]
                binds.append(bindparam("cx", expanding=True))
    else:
        raise ValueError("Unknown side %r." % side)
    return sql, params, binds


def _read(engine, period: str, where: str, params: dict, binds: list,
          columns: List[str]) -> pd.DataFrame:
    start = period[:4] + "01"
    P, B = _q(engine, "PERIOD"), _q(engine, "BASIS")
    cols = ", ".join(_q(engine, c) for c in columns)
    stmt = text(f"SELECT {cols} FROM gl_detail WHERE {P} >= :start AND {P} <= :end "
                f"AND {B} IN :bases AND ({where})")
    stmt = stmt.bindparams(bindparam("bases", expanding=True), *binds)
    with engine.connect() as c:
        df = pd.read_sql(stmt, c, params={**params, "start": start, "end": period,
                                          "bases": list(BASES)})
    df.columns = [str(c).upper() for c in df.columns]
    if "AMT" in df.columns:
        df["AMT"] = pd.to_numeric(df["AMT"], errors="coerce").fillna(0.0)
    for c in ("ENTITYID", "ACCTNUM", "RLTDENTITY"):
        if c in df.columns:
            df[c] = df[c].map(_norm)
    return df


def periods(engine=None) -> dict:
    engine = engine or get_engine()
    if not _has_table(engine, "gl_detail"):
        return {"available": False, "periods": [],
                "reason": "The GL has not been imported. Run the MRI_GL_Detail "
                          "query from MRI Data."}
    P = _q(engine, "PERIOD")
    with engine.connect() as c:
        ps = [str(r[0]) for r in c.execute(text(
            f"SELECT DISTINCT {P} FROM gl_detail ORDER BY {P} DESC")) if r[0]]
    ps = [p for p in ps if _PERIOD_RE.match(p)]
    return {"available": bool(ps), "periods": ps, "latest": ps[0] if ps else None,
            "reason": None if ps else "The GL table is empty."}


def _opening_present(engine, period: str) -> bool:
    P, BF = _q(engine, "PERIOD"), _q(engine, "BALFOR")
    with engine.connect() as c:
        n = c.execute(text(f"SELECT COUNT(*) FROM gl_detail WHERE {P} = :s "
                           f"AND {BF} = 'B'"), {"s": period[:4] + "01"}).scalar()
    return bool(n)


def reconcile(engine=None, period: Optional[str] = None,
              tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """The reconciliation for one period: rows, totals and checks."""
    engine = engine or get_engine()
    ensure_tables(engine)
    avail = periods(engine)
    if not avail["available"]:
        return {"available": False, "reason": avail["reason"], "rows": []}
    period = _check_period(period or avail["latest"])
    tol = abs(float(tolerance if tolerance is not None else DEFAULT_TOLERANCE))
    settings = get_settings(engine)
    cols = ["ENTITYID", "ACCTNUM", "RLTDENTITY", "RLTDENTITY_NAME", "AMT"]
    pay = _pay_rows(engine, period)
    pending = _pending_entities(engine)

    frames = {}
    for side in ("entity", "manager"):
        w, p, b = _selection(engine, side, None, settings)
        frames[side] = _read(engine, period, w, p, b, cols)
    ent_df, mgr_df = frames["entity"], frames["manager"]

    # Alternate accounts and cash, for every entity the settings name plus every
    # entity on either side -- read in one pass rather than a query per row.
    alt_accts = sorted({s["alt_account"] for s in settings.values() if s["alt_account"]})
    named_cash = sorted({a for s in settings.values() for a in s["cash_accounts"]})
    A = _q(engine, "ACCTNUM")
    extra_where, extra_params, extra_binds = f"{A} LIKE :cp", {"cp": CASH_PREFIX + "%"}, []
    wanted = alt_accts + named_cash
    if wanted:
        extra_where += f" OR {A} IN :wanted"
        extra_params["wanted"] = wanted
        extra_binds.append(bindparam("wanted", expanding=True))
    other = _read(engine, period, extra_where, extra_params, extra_binds,
                  ["ENTITYID", "ACCTNUM", "ACCTNAME", "AMT"])

    names = _names(engine)
    for r in mgr_df.itertuples():
        if r.RLTDENTITY and r.RLTDENTITY not in names and r.RLTDENTITY_NAME:
            names[r.RLTDENTITY] = str(r.RLTDENTITY_NAME).strip()
    acct_names = {}
    for r in other.itertuples():
        acct_names.setdefault(r.ACCTNUM, str(r.ACCTNAME or "").strip())

    ent_bal = ent_df.groupby("ENTITYID")["AMT"].sum()
    mgr_named = mgr_df[mgr_df["RLTDENTITY"] != ""]
    mgr_bal = mgr_named.groupby("RLTDENTITY")["AMT"].sum()
    blank = mgr_df[mgr_df["RLTDENTITY"] == ""]

    entities = sorted(set(ent_df["ENTITYID"]) | set(mgr_named["RLTDENTITY"])
                      | {e for e, s in settings.items() if s["alt_account"]
                         and not other[(other["ENTITYID"] == e)
                                       & (other["ACCTNUM"] == s["alt_account"])].empty})
    entities = [e for e in entities if e and e != MANAGER_ENTITY]
    notes = _notes(engine, period)

    rows = []
    for e in entities:
        s = settings.get(e) or {}
        a = round(float(ent_bal.get(e, 0.0)), 2)
        alt = s.get("alt_account")
        b = 0.0
        if alt:
            b = round(float(other[(other["ENTITYID"] == e)
                                  & (other["ACCTNUM"] == alt)]["AMT"].sum()), 2)
        c = round(a + b, 2)
        d = round(float(mgr_bal.get(e, 0.0)), 2)
        var = round(c + d, 2)
        if abs(c) < 0.005 and abs(d) < 0.005:
            status = "No Balance"
        elif abs(var) <= tol:
            status = "Reconciled"
        else:
            status = "Investigate"

        mine = other[other["ENTITYID"] == e]
        if s.get("cash_accounts"):
            cash_rows = mine[mine["ACCTNUM"].isin(s["cash_accounts"])]
            cash_basis = "named: " + ", ".join(s["cash_accounts"])
        else:
            cash_rows = mine[mine["ACCTNUM"].str.startswith(CASH_PREFIX)]
            cash_basis = CASH_PREFIX + "*"
            if s.get("cash_exclude"):
                cash_rows = cash_rows[~cash_rows["ACCTNUM"].isin(s["cash_exclude"])]
                cash_basis += " less " + ", ".join(s["cash_exclude"])
        by_acct = cash_rows.groupby("ACCTNUM")["AMT"].sum().round(2)
        cash = round(float(by_acct.sum()), 2)
        currency = s.get("currency") or "USD"

        # What it can afford, per the CFO's rule: the lower of what the manager
        # is owed and the cash on hand. Only USD cash can cap a USD amount; the
        # app holds no FX rate, so a CAD row is shown, never computed.
        computed = None
        if d > 0.005 and currency == "USD":
            computed = round(max(0.0, min(d, cash)), 2)
        # THE ACCOUNTANT MAY ADJUST IT, with a reason (CFO, Sep 30 2026): not all
        # the GL cash is spendable. The computed figure is kept beside it.
        pr = pay.get(e) or {}
        override = pr.get("afford_override")
        affordable = round(float(override), 2) if override is not None else computed
        # The pay amount STARTS at what it can afford and may be changed. A CAD
        # row, with nothing to cap it, is prompted with what the manager is owed.
        prompt = None
        if d > 0.005:
            prompt = affordable if affordable is not None else d
        amount = pr.get("amount")
        pay_amount = round(float(amount), 2) if amount is not None else prompt
        default_cash = (MANAGER_CASH_ACCOUNT if MANAGER_CASH_ACCOUNT in by_acct.index
                        else (by_acct.abs().idxmax() if len(by_acct) else
                              (s.get("cash_accounts") or [None])[0]))
        pay_cash = pr.get("cash_account") or default_cash

        n = notes.get(e) or {}
        rows.append({
            "entity_id": e, "name": names.get(e, ""),
            "entity_balance": a, "alt_account": alt, "alt_balance": b,
            "total_entity": c, "manager_balance": d, "variance": var,
            "status": status, "cash_balance": cash, "cash_basis": cash_basis,
            "cash_by_account": [{"account": k, "name": acct_names.get(k, ""),
                                 "amount": float(v)} for k, v in by_acct.items()],
            "currency": currency, "affordable": affordable,
            "affordable_computed": computed,
            "afford_override": affordable if override is not None else None,
            "afford_reason": pr.get("afford_reason") or "",
            "pay_prompt": prompt, "pay_amount": pay_amount,
            "pay_amount_set": amount is not None,
            "pay_cash_account": pay_cash,
            "pay_by": pr.get("updated_by"), "pay_at": pr.get("updated_at"),
            "pending_batch": pending.get(e),
            "comment": n.get("comment") or "", "comment_by": n.get("updated_by"),
            "comment_at": n.get("updated_at"),
            "settings_basis": s.get("basis"),
        })

    def tot(k):
        return round(sum(r[k] for r in rows), 2)

    totals = {k: tot(k) for k in ("entity_balance", "alt_balance", "total_entity",
                                  "manager_balance", "variance", "cash_balance")}
    totals["affordable"] = round(sum(r["affordable"] or 0 for r in rows), 2)
    totals["pay_amount"] = round(sum(r["pay_amount"] or 0 for r in rows), 2)
    opening = _opening_present(engine, period)
    blank_amt = round(float(blank["AMT"].sum()), 2)
    checks = [
        {"key": "opening", "ok": opening,
         "label": "Opening balances for %s are in the GL" % period[:4],
         "detail": None if opening else
         "No balance-forward rows at %s01, so every balance below is the "
         "year's activity alone, not a balance." % period[:4]},
        {"key": "blank_segment", "ok": abs(blank_amt) < 0.005,
         "label": "PSC Manager balance with no entity segment",
         "amount": blank_amt, "rows": int(len(blank)),
         "detail": None if abs(blank_amt) < 0.005 else
         "Part of PSC Manager's %s cannot be attributed to any entity." % MANAGER_ACCOUNT},
        {"key": "investigate", "ok": not any(r["status"] == "Investigate" for r in rows),
         "label": "Entities to investigate",
         "count": sum(1 for r in rows if r["status"] == "Investigate")},
    ]
    return {
        "available": True, "period": period, "year_start": period[:4] + "01",
        "tolerance": tol, "bases": list(BASES),
        "accounts": {"entity": ENTITY_ACCOUNT, "manager_entity": MANAGER_ENTITY,
                     "manager": MANAGER_ACCOUNT,
                     "entity_name": acct_names.get(ENTITY_ACCOUNT)},
        "rows": rows, "totals": totals, "checks": checks,
        "manager_total": round(float(mgr_df["AMT"].sum()), 2),
        "data_as_of": _freshness(engine),
    }


def lines(engine=None, period: Optional[str] = None, entity: str = "",
          side: str = "entity", limit: int = 2000) -> dict:
    """The GL entries behind one figure. Same selection as the grid."""
    engine = engine or get_engine()
    period = _check_period(period)
    entity = _norm(entity)
    if not entity:
        raise ValueError("Name an entity.")
    settings = get_settings(engine)
    sel = _selection(engine, side, entity, settings)
    if sel is None:
        return {"rows": [], "total": 0.0, "count": 0, "side": side,
                "entity_id": entity, "period": period}
    w, p, b = sel
    cols = ["ENTITYID", "PERIOD", "ENTRDATE", "ACCTNUM", "ACCTNAME", "BASIS",
            "BALFOR", "REF", "DESCRPN", "RLTDENTITY", "AMT"]
    df = _read(engine, period, w, p, b, cols)
    df = df.sort_values(["PERIOD", "ENTRDATE"], kind="stable")
    total = round(float(df["AMT"].sum()), 2)
    out = df.head(limit).copy()
    out["ENTRDATE"] = out["ENTRDATE"].astype(str).str.slice(0, 10)
    return {"rows": out.to_dict("records"), "total": total, "count": int(len(df)),
            "truncated": len(df) > limit, "side": side, "entity_id": entity,
            "period": period}


# ------------------------------------------------------------------ writes

def save_note(engine, period: str, entity_id: str, comment: str, user: str) -> dict:
    engine = engine or get_engine()
    ensure_tables(engine)
    period = _check_period(period)
    e = _norm(entity_id)
    if not e:
        raise ValueError("Name an entity.")
    comment = (comment or "").strip()
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text("DELETE FROM ic_recon_notes WHERE period = :p AND entity_id = :e"),
                  {"p": period, "e": e})
        if comment:
            c.execute(text("INSERT INTO ic_recon_notes (period, entity_id, comment, "
                           "updated_by, updated_at) VALUES (:p, :e, :c, :u, :t)"),
                      {"p": period, "e": e, "c": comment, "u": user, "t": now})
    return {"period": period, "entity_id": e, "comment": comment,
            "updated_by": user if comment else None, "updated_at": now if comment else None}


def save_settings(engine, entity_id: str, body: dict, user: str) -> dict:
    """An entity's alternate account, cash accounts and currency.

    Accounts are refused unless they look like an MRI account -- a typo here
    would silently read a zero balance and make a row look reconciled.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    e = _norm(entity_id)
    if not e:
        raise ValueError("Name an entity.")
    alt = _norm(body.get("alt_account"))
    cash = _accounts(body.get("cash_accounts") if not isinstance(body.get("cash_accounts"), list)
                     else ",".join(body.get("cash_accounts")))
    excl = _accounts(body.get("cash_exclude") if not isinstance(body.get("cash_exclude"), list)
                     else ",".join(body.get("cash_exclude")))
    cur = _norm(body.get("currency")) or "USD"
    bad = [a for a in ([alt] if alt else []) + cash + excl if not _ACCOUNT_RE.match(a)]
    if bad:
        raise ValueError("Not an MRI account number: %s." % ", ".join(bad))
    if alt in (ENTITY_ACCOUNT, MANAGER_ACCOUNT):
        raise ValueError("%s is already on the reconciliation; the alternate "
                         "account is a second account." % alt)
    if cash and excl:
        raise ValueError("Name the cash accounts OR exclude some from the default, "
                         "not both.")
    if cur not in CURRENCIES:
        raise ValueError("Currency must be one of %s." % ", ".join(CURRENCIES))
    basis = (body.get("basis") or "").strip() or ("Set by %s." % user)
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text("DELETE FROM ic_entity_settings WHERE entity_id = :e"), {"e": e})
        if alt or cash or excl or cur != "USD":
            c.execute(text(
                "INSERT INTO ic_entity_settings (entity_id, alt_account, cash_accounts, "
                "cash_exclude, currency, basis, updated_by, updated_at) VALUES "
                "(:e, :alt, :cash, :excl, :cur, :basis, :u, :t)"),
                {"e": e, "alt": alt, "cash": ",".join(cash), "excl": ",".join(excl),
                 "cur": cur, "basis": basis, "u": user, "t": now})
    return get_settings(engine).get(e) or {"entity_id": e, "alt_account": None,
                                           "cash_accounts": [], "cash_exclude": [],
                                           "currency": "USD"}


# ------------------------------------------------------------------ pay

def _pay_rows(engine, period: str) -> Dict[str, dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text("SELECT * FROM ic_pay_rows WHERE period = :p"),
                         {"p": period}).mappings().all()
    return {_norm(r["entity_id"]): dict(r) for r in rows}


def _num(v) -> Optional[float]:
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    try:
        return round(float(str(v).replace(",", "").replace("$", "").strip()), 2)
    except (TypeError, ValueError):
        raise ValueError("%r is not an amount." % (v,))


def _batch_rows(engine) -> List[dict]:
    import json
    ensure_tables(engine)
    with engine.connect() as c:
        rows = c.execute(text("SELECT batch_id, period, entrdate, entities, total, "
                              "created_by, created_at, voided_by, voided_at "
                              "FROM ic_je_batches ORDER BY created_at DESC")).mappings().all()
    out = []
    for r in rows:
        d = dict(r)
        d["entities"] = json.loads(d["entities"] or "[]")
        out.append(d)
    return out


def _posted(engine, batches: List[dict]) -> set:
    """(entity, period, amount) for every reimbursement line the GL already
    carries -- how a generated batch is known to have been uploaded."""
    periods_ = sorted({b["period"] for b in batches})
    if not periods_ or not _has_table(engine, "gl_detail"):
        return set()
    E, A, P, D, AMT, B = (_q(engine, x) for x in
                          ("ENTITYID", "ACCTNUM", "PERIOD", "DESCRPN", "AMT", "BASIS"))
    stmt = text(f"SELECT {E}, {P}, {AMT} FROM gl_detail WHERE {A} = :acct AND "
                f"{D} = :d AND {P} IN :ps AND {B} IN :bases").bindparams(
        bindparam("ps", expanding=True), bindparam("bases", expanding=True))
    with engine.connect() as c:
        got = c.execute(stmt, {"acct": ENTITY_ACCOUNT, "d": JE_ENTITY_DESCRIPTION,
                               "ps": periods_, "bases": list(BASES)}).fetchall()
    return {(_norm(e), str(p), round(float(a or 0), 2)) for e, p, a in got}


def batches(engine=None) -> List[dict]:
    """Every generated batch, each saying whether the GL shows it yet."""
    engine = engine or get_engine()
    bs = _batch_rows(engine)
    live = [b for b in bs if not b["voided_at"]]
    posted = _posted(engine, live)
    for b in bs:
        for x in b["entities"]:
            x["posted"] = (x["entity_id"], b["period"], round(x["amount"], 2)) in posted
        n = sum(1 for x in b["entities"] if x["posted"])
        b["status"] = ("voided" if b["voided_at"] else
                       "posted" if n == len(b["entities"]) else
                       "partly posted" if n else "generated, not yet in the GL")
    return bs


def _pending_entities(engine) -> Dict[str, str]:
    """entity -> batch id, for a reimbursement generated and not yet in the GL."""
    out = {}
    for b in batches(engine):
        if b["voided_at"]:
            continue
        for x in b["entities"]:
            if not x["posted"]:
                out.setdefault(x["entity_id"], b["batch_id"])
    return out


def _row_problems(r: dict, amount: Optional[float]) -> List[str]:
    """Why this row cannot be paid this amount. Empty when it can."""
    e = r["entity_id"]
    if amount is None or amount < 0.005:
        return ["%s: there is no amount to pay." % e]
    out = []
    if r["manager_balance"] < 0.005:
        out.append("%s: PSC Manager is owed nothing." % e)
    elif amount > r["manager_balance"] + 0.005:
        out.append("%s: %s is more than PSC Manager is owed (%s)."
                   % (e, _m(amount), _m(r["manager_balance"])))
    if r["affordable"] is not None and amount > r["affordable"] + 0.005:
        out.append("%s: %s is more than it can afford (%s). Adjust the can-afford "
                   "figure, with the reason, to pay more." % (e, _m(amount), _m(r["affordable"])))
    if not r.get("pay_cash_account"):
        out.append("%s: no cash account to pay from." % e)
    if r.get("pending_batch"):
        out.append("%s: already in batch %s, which the GL does not show yet."
                   % (e, r["pending_batch"]))
    return out


def _m(v) -> str:
    return "{:,.2f}".format(float(v))


def save_pay(engine, period: str, entity_id: str, body: dict, user: str) -> dict:
    """An entity's can-afford adjustment, amount to pay and cash account.

    Only the keys present in `body` change. A blank or null clears it back to
    the computed figure. An adjusted can-afford needs a reason, and may not
    exceed what PSC Manager is owed; the amount may not exceed the can-afford.
    """
    engine = engine or get_engine()
    ensure_tables(engine)
    period = _check_period(period)
    e = _norm(entity_id)
    res = reconcile(engine, period)
    row = next((r for r in res.get("rows", []) if r["entity_id"] == e), None)
    if row is None:
        raise ValueError("%s is not on the %s reconciliation." % (e, period))
    cur = _pay_rows(engine, period).get(e) or {}
    rec = {k: cur.get(k) for k in ("afford_override", "afford_reason", "amount",
                                   "cash_account")}
    if "afford_override" in body:
        rec["afford_override"] = _num(body.get("afford_override"))
    if "afford_reason" in body:
        rec["afford_reason"] = (body.get("afford_reason") or "").strip()
    if "amount" in body:
        rec["amount"] = _num(body.get("amount"))
    if "cash_account" in body:
        ca = _norm(body.get("cash_account"))
        if ca and not _ACCOUNT_RE.match(ca):
            raise ValueError("Not an MRI account number: %s." % ca)
        rec["cash_account"] = ca or None

    warnings = []
    ov = rec["afford_override"]
    if ov is not None:
        if ov < 0:
            raise ValueError("Can afford cannot be negative.")
        if ov > row["manager_balance"] + 0.005:
            raise ValueError("Can afford %s is more than PSC Manager is owed (%s)."
                             % (_m(ov), _m(row["manager_balance"])))
        if not rec["afford_reason"]:
            raise ValueError("Say why the can-afford figure is adjusted.")
        if row["currency"] == "USD" and ov > row["cash_balance"] + 0.005:
            warnings.append("%s is more than the %s cash the GL shows."
                            % (_m(ov), _m(row["cash_balance"])))
    else:
        rec["afford_reason"] = None
    cap = ov if ov is not None else row["affordable_computed"]
    if rec["amount"] is not None:
        if rec["amount"] < 0:
            raise ValueError("The amount to pay cannot be negative.")
        if cap is not None and rec["amount"] > cap + 0.005:
            raise ValueError("%s is more than it can afford (%s). Adjust the can-afford "
                             "figure, with the reason, to pay more." % (_m(rec["amount"]), _m(cap)))
        if rec["amount"] > row["manager_balance"] + 0.005:
            raise ValueError("%s is more than PSC Manager is owed (%s)."
                             % (_m(rec["amount"]), _m(row["manager_balance"])))
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text("DELETE FROM ic_pay_rows WHERE period = :p AND entity_id = :e"),
                  {"p": period, "e": e})
        if any(rec[k] is not None for k in ("afford_override", "amount", "cash_account")):
            c.execute(text(
                "INSERT INTO ic_pay_rows (period, entity_id, afford_override, afford_reason,"
                " amount, cash_account, updated_by, updated_at) VALUES (:p, :e, :ov, :why,"
                " :amt, :ca, :u, :t)"),
                {"p": period, "e": e, "ov": ov, "why": rec["afford_reason"],
                 "amt": rec["amount"], "ca": rec["cash_account"], "u": user, "t": now})
    row = next(r for r in reconcile(engine, period)["rows"] if r["entity_id"] == e)
    return {"row": row, "warnings": warnings}


def build_batch(engine, period: str, entity_ids: List[str], je_period: str,
                entrdate: str, user: str, commit: bool = False) -> dict:
    """The MRI GL upload for the ticked rows -- `JE Template`'s shape exactly:
    each entity's two lines, then PSC Manager's two per entity, in the order the
    rows were ticked. `commit=False` is the preview; nothing is stored.

    Refused: an amount over what it can afford or over what is owed, no cash
    account, a row already in a batch the GL does not show, a missing entry
    date or period (both typed by the accountant -- no default), and anything
    `validate_gl` refuses, which now includes an entity left out of balance.
    """
    import json
    import uuid
    from flask_app.services.treasury_upload import _iso, build_gl_csv, validate_gl
    engine = engine or get_engine()
    period = _check_period(period)
    res = reconcile(engine, period)
    by_id = {r["entity_id"]: r for r in res.get("rows", [])}
    errors, chosen = [], []
    ids = []
    for x in entity_ids or []:
        n = _norm(x)
        if n and n not in ids:
            ids.append(n)
    if not ids:
        errors.append("Tick at least one entity to pay.")
    for e in ids:
        r = by_id.get(e)
        if r is None:
            errors.append("%s is not on the %s reconciliation." % (e, period))
            continue
        errors += _row_problems(r, r["pay_amount"])
        chosen.append(r)
    jp = str(je_period or "").strip()
    if not _PERIOD_RE.match(jp):
        errors.append("Type the journal period (YYYYMM).")
    ed = _iso(entrdate)
    if not ed:
        errors.append("Type the entry date.")
    lines = []
    for r in chosen:
        amt = r["pay_amount"] or 0
        for acct, a in ((ENTITY_ACCOUNT, amt), (r["pay_cash_account"], -amt)):
            lines.append({"entityid": r["entity_id"], "acctnum": acct, "amount": a,
                          "descrpn": JE_ENTITY_DESCRIPTION, "rltdentity": "",
                          "period": jp, "basis": "B", "entrdate": ed})
    for r in chosen:
        amt = r["pay_amount"] or 0
        d = JE_MANAGER_DESCRIPTION % r["entity_id"]
        lines.append({"entityid": MANAGER_ENTITY, "acctnum": MANAGER_CASH_ACCOUNT,
                      "amount": amt, "descrpn": d, "rltdentity": "", "period": jp,
                      "basis": "B", "entrdate": ed})
        lines.append({"entityid": MANAGER_ENTITY, "acctnum": MANAGER_ACCOUNT,
                      "amount": -amt, "descrpn": d, "rltdentity": r["entity_id"],
                      "period": jp, "basis": "B", "entrdate": ed})
    v = validate_gl(lines) if lines and not errors else {"errors": [], "warnings": []}
    errors += v["errors"]
    out = {"period": period, "je_period": jp, "entrdate": ed, "lines": lines,
           "errors": errors, "warnings": [w for w in v.get("warnings", [])
                                           if "more than one entity" not in w],
           "total_paid": round(sum(r["pay_amount"] or 0 for r in chosen), 2),
           "entities": [{"entity_id": r["entity_id"], "amount": r["pay_amount"],
                         "cash_account": r["pay_cash_account"]} for r in chosen]}
    if not commit or errors:
        return out
    csv_text = build_gl_csv(lines)
    bid = "IC-%s-%s" % (jp, uuid.uuid4().hex[:6].upper())
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text(
            "INSERT INTO ic_je_batches (batch_id, period, entrdate, entities, csv, total,"
            " created_by, created_at) VALUES (:b, :p, :d, :e, :csv, :t, :u, :now)"),
            {"b": bid, "p": jp, "d": ed, "e": json.dumps(out["entities"]),
             "csv": csv_text, "t": out["total_paid"], "u": user, "now": now})
    out.update(batch_id=bid, csv=csv_text)
    return out


def batch_csv(engine, batch_id: str) -> Optional[dict]:
    engine = engine or get_engine()
    ensure_tables(engine)
    with engine.connect() as c:
        r = c.execute(text("SELECT batch_id, csv FROM ic_je_batches WHERE batch_id = :b"),
                      {"b": batch_id}).fetchone()
    return {"batch_id": r[0], "csv": r[1]} if r else None


def void_batch(engine, batch_id: str, user: str) -> dict:
    """A batch that will not be uploaded. Its rows can then be paid again, so
    voiding one that WAS uploaded would invite paying twice -- refused once the
    GL shows any of it."""
    engine = engine or get_engine()
    b = next((x for x in batches(engine) if x["batch_id"] == batch_id), None)
    if b is None:
        raise ValueError("No batch %s." % batch_id)
    if b["voided_at"]:
        raise ValueError("Batch %s is already void." % batch_id)
    if any(x["posted"] for x in b["entities"]):
        raise ValueError("Batch %s is already in the GL (in part or whole); it "
                         "cannot be voided." % batch_id)
    now = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as c:
        c.execute(text("UPDATE ic_je_batches SET voided_by = :u, voided_at = :t "
                       "WHERE batch_id = :b"), {"u": user, "t": now, "b": batch_id})
    return {"batch_id": batch_id, "voided_by": user, "voided_at": now}


# ------------------------------------------------------------------ export

def to_excel(result: dict) -> bytes:
    cols = [("entity_id", "Entity ID"), ("name", "Entity Name"),
            ("entity_balance", "A  Entity: Due To/From PSC Manager"),
            ("alt_account", "Alt. Entity Acct"), ("alt_balance", "B  Alt. Acct Balance"),
            ("total_entity", "C = A + B  Total Entity"),
            ("manager_balance", "D  PSC Manager: Due to/from Interco"),
            ("variance", "C + D  Variance"), ("status", "Status"),
            ("comment", "Comments"), ("currency", "Cash Currency"),
            ("cash_balance", "Cash Balance"), ("cash_basis", "Cash Accounts"),
            ("affordable_computed", "Can Afford (GL)"), ("affordable", "Can Afford"),
            ("afford_reason", "Adjusted Because"), ("pay_amount", "Amount to Pay"),
            ("pay_cash_account", "Pay From")]
    df = pd.DataFrame([{label: r.get(k) for k, label in cols} for r in result["rows"]])
    head = pd.DataFrame([
        ("Due to / from PSC Manager Reconciliation", ""),
        ("Period", "%s (from %s)" % (result["period"], result["year_start"])),
        ("Basis", ".".join(result["bases"])),
        ("Entity-side account", result["accounts"]["entity"]),
        ("Manager / account", "%s / %s" % (result["accounts"]["manager_entity"],
                                           result["accounts"]["manager"])),
        ("Tolerance", result["tolerance"]),
        ("MRI refresh completed", result.get("data_as_of") or "unknown"),
    ])
    checks = pd.DataFrame([{"Check": c["label"], "OK": "OK" if c["ok"] else "CHECK",
                            "Detail": c.get("detail") or c.get("amount") or c.get("count")}
                           for c in result["checks"]])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        head.to_excel(w, sheet_name="Reconciliation", index=False, header=False)
        df.to_excel(w, sheet_name="Reconciliation", index=False, startrow=len(head) + 1)
        checks.to_excel(w, sheet_name="Checks", index=False)
        ws = w.sheets["Reconciliation"]
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col[:400])
            ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
    return buf.getvalue()
