"""Guardrail: the Due to/from PSC Manager reconciliation.

Two halves.

1. A FIXTURE built to be awkward: basis C and T rows that must be excluded (they
   halve the real figures if they are not), a prior-year and a next-period row,
   the year's opening as a balance-forward row, an alternate account, named and
   excluded cash accounts, a CAD entity, a blank manager segment, a variance just
   inside and just outside the tolerance, and the INVF10 case the CFO's sheet
   gets wrong (a zero balance his formula marks Investigate). Every figure's
   drilldown must add up to the figure.

2. THE CFO'S OWN ROWS, when his workbook is on this machine: his three GEXD tabs
   loaded as `gl_detail`, and his totals reproduced to the cent. Skipped, with the
   reason, where the file is absent (production, CI).
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import create_engine, text

from flask_app.services import intercompany_service as ic

_passed, _failed, _skipped = [], [], []
WORKBOOK = Path(os.path.expanduser(
    r"~/OneDrive - peaceablestreet.com/Documents/BORG_Intercompany Template.xlsx"))


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print(("  PASS  " if cond else "  FAIL  ") + label + ("" if cond else f"   {detail}"))


def skip(label, why):
    _skipped.append(label)
    print(f"  SKIP  {label}   ({why})")


def fresh_engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    ic._DDL_DONE.clear()
    return create_engine(f"sqlite:///{path}"), path


COLS = ["ENTITYID", "PERIOD", "ENTRDATE", "ACCTNAME", "ACCTNUM", "BASIS", "BALFOR",
        "ITEM", "REF", "DESCRPN", "SEGMENTID", "RLTDENTITY", "RLTDENTITY_NAME", "AMT"]


def row(ent, per, acct, amt, basis="B", balfor="N", rlt=None, rlt_name=None, name=""):
    return dict(ENTITYID=ent, PERIOD=per, ENTRDATE=f"{per[:4]}-{per[4:]}-15",
                ACCTNAME=name or acct, ACCTNUM=acct, BASIS=basis, BALFOR=balfor,
                ITEM=1, REF="R", DESCRPN="d", SEGMENTID=None, RLTDENTITY=rlt,
                RLTDENTITY_NAME=rlt_name, AMT=amt)


E, M, MGR = ic.ENTITY_ACCOUNT, ic.MANAGER_ACCOUNT, ic.MANAGER_ENTITY


def fixture():
    r = []
    # AMB6: opening -60 (balance-forward, basis A) + -40 activity = -100; manager +100.
    r += [row("AMB6", "202601", E, -60, basis="A", balfor="B"),
          row("AMB6", "202605", E, -40),
          row(MGR, "202601", M, 60, basis="A", balfor="B", rlt="AMB6", rlt_name="PSC Ambassadors TGA VI"),
          row(MGR, "202607", M, 40, rlt="AMB6")]
    # Rows that must NOT count: basis C and T, last year, next period.
    r += [row("AMB6", "202603", E, -999, basis="C"),
          row("AMB6", "202603", E, -888, basis="T"),
          row("AMB6", "202512", E, -777),
          row("AMB6", "202610", E, -555)]
    # NOTTNV: nothing on MR15000002, -300 on its alternate account; manager +200.
    r += [row("NOTTNV", "202604", "MR99982002", -300),
          row(MGR, "202604", M, 200, rlt="NOTTNV")]
    # PEGASU: alternate -50, manager 0; cash on a named account, and an MR1000*
    # account that must NOT count because the cash accounts are named.
    r += [row("PEGASU", "202602", "MR99991102", -50),
          row("PEGASU", "202602", "MR99991000", 500),
          row("PEGASU", "202602", "MR10005000", 7)]
    # PSC1: cash MR1000* less the excluded Liberty MM.
    r += [row("PSC1", "202602", E, -10), row(MGR, "202602", M, 10, rlt="PSC1"),
          row("PSC1", "202602", "MR10005000", 1000),
          row("PSC1", "202602", "MR10007000", 2000),
          row("PSC1", "202602", "MR10008000", 5000)]
    # PPI2: CAD cash is the two CAD accounts; its USD account must not count.
    r += [row("PPI2", "202602", E, -40), row(MGR, "202602", M, 50, rlt="PPI2"),
          row("PPI2", "202602", "MR10006000", 400),
          row("PPI2", "202602", "MR10003000", 100),
          row("PPI2", "202602", "MR10005000", 999)]
    # INVF10: a balance that nets to zero. The CFO's sheet marks it Investigate.
    r += [row("INVF10", "202602", E, 25), row("INVF10", "202603", E, -25)]
    # AMB23: reconciled, but cash is short of what is owed.
    r += [row("AMB23", "202602", E, -161.34), row(MGR, "202602", M, 161.34, rlt="AMB23"),
          row("AMB23", "202602", "MR10005000", 100)]
    # AMB24: a variance of 0.50 -- inside a $1 tolerance, outside 0.10.
    r += [row("AMB24", "202602", E, -272.16), row(MGR, "202602", M, 271.66, rlt="AMB24")]
    # Blank manager segment netting to zero.
    r += [row(MGR, "202602", M, 5, rlt=None), row(MGR, "202603", M, -5, rlt="")]
    # A PSC1 row on MR15000001 that is NOT the manager's -- must not reach D.
    r += [row("PSC1", "202602", M, 12345, rlt="AMB6")]
    return pd.DataFrame(r, columns=COLS)


def load(engine, df):
    df.to_sql("gl_detail", engine, if_exists="replace", index=False)
    pd.DataFrame([("AMB6", "PSC Ambassadors Fund TGA VI LLC"),
                  ("NOTTNV", "Nottingham Village"), ("PSC1", "Peaceable Street Capital LLC")],
                 columns=["ENTITYID", "NAME"]).to_sql("entities", engine, if_exists="replace",
                                                       index=False)


def section_fixture():
    print("\n== fixture ==")
    engine, path = fresh_engine()
    load(engine, fixture())
    res = ic.reconcile(engine, "202609")
    rows = {r["entity_id"]: r for r in res["rows"]}
    chk("available, period 202609 from 202601", res["available"]
        and res["period"] == "202609" and res["year_start"] == "202601")
    chk("basis is A.B, stated", res["bases"] == ["A", "B"])

    a = rows["AMB6"]
    chk("AMB6 entity = opening + activity only (-100), C/T/prior/next excluded",
        a["entity_balance"] == -100.0, a["entity_balance"])
    chk("AMB6 manager = 100; another entity's MR15000001 does not reach D",
        a["manager_balance"] == 100.0, a["manager_balance"])
    chk("AMB6 reconciled", a["status"] == "Reconciled", a["status"])
    chk("AMB6 named from entities", a["name"] == "PSC Ambassadors Fund TGA VI LLC", a["name"])

    n = rows.get("NOTTNV") or {}
    chk("NOTTNV reaches the grid through its alternate account alone",
        n.get("alt_balance") == -300.0 and n.get("total_entity") == -300.0, n)
    chk("NOTTNV variance -100, Investigate",
        n.get("variance") == -100.0 and n.get("status") == "Investigate", n)

    p = rows.get("PEGASU") or {}
    chk("PEGASU on MR99991102: -50 against 0 -> Investigate",
        p.get("total_entity") == -50.0 and p.get("status") == "Investigate", p)
    chk("PEGASU cash is the NAMED account only (500, not 507)",
        p.get("cash_balance") == 500.0, p.get("cash_balance"))

    s = rows["PSC1"]
    chk("PSC1 cash excludes Liberty MM (3,000)", s["cash_balance"] == 3000.0, s["cash_balance"])
    chk("PSC1 can afford all 10 it owes", s["affordable"] == 10.0, s["affordable"])

    c = rows["PPI2"]
    chk("PPI2 cash is its two CAD accounts (500), USD account ignored",
        c["cash_balance"] == 500.0 and c["currency"] == "CAD", c)
    chk("PPI2 affordability is NOT computed (no FX rate)", c["affordable"] is None)
    chk("PPI2 variance 10 -> Investigate", c["variance"] == 10.0 and c["status"] == "Investigate")

    z = rows.get("INVF10") or {}
    chk("INVF10 nets to zero -> No Balance (the CFO's sheet says Investigate)",
        z.get("status") == "No Balance", z)

    b = rows["AMB23"]
    chk("AMB23 short of cash: can afford 100 of 161.34, not 0 (the sheet's rule)",
        b["affordable"] == 100.0 and b["status"] == "Reconciled", b)

    chk("AMB24 variance -0.50 reconciled within $1",
        rows["AMB24"]["variance"] == -0.5 and rows["AMB24"]["status"] == "Reconciled")
    tight = {r["entity_id"]: r for r in ic.reconcile(engine, "202609", 0.1)["rows"]}
    chk("AMB24 Investigate at a 0.10 tolerance", tight["AMB24"]["status"] == "Investigate")

    chk("PSC Manager is not a row of its own", MGR not in rows)
    ck = {x["key"]: x for x in res["checks"]}
    chk("opening present for 2026", ck["opening"]["ok"])
    chk("blank segment nets to 0 -> OK", ck["blank_segment"]["ok"]
        and ck["blank_segment"]["rows"] == 2, ck["blank_segment"])
    chk("investigate count = 3 (NOTTNV, PEGASU, PPI2)", ck["investigate"]["count"] == 3,
        ck["investigate"])
    chk("totals are the sum of the rows",
        res["totals"]["variance"] == round(sum(r["variance"] for r in res["rows"]), 2))

    # Every figure's drilldown must add up to the figure.
    ok = True
    for r in res["rows"]:
        for side, key in (("entity", "entity_balance"), ("manager", "manager_balance"),
                          ("alt", "alt_balance"), ("cash", "cash_balance")):
            got = ic.lines(engine, "202609", r["entity_id"], side)["total"]
            if round(got, 2) != round(r[key], 2):
                ok = False
                print(f"        {r['entity_id']} {side}: lines {got} vs figure {r[key]}")
    chk("every drilldown adds up to its figure (4 sides x every row)", ok)
    ln = ic.lines(engine, "202609", "AMB6", "entity")
    chk("AMB6 drilldown carries the opening row and not the C/T/out-of-range rows",
        ln["count"] == 2 and any(x["BALFOR"] == "B" for x in ln["rows"]), ln["count"])

    # A blank segment that does NOT net to zero is reported.
    load(engine, pd.concat([fixture(), pd.DataFrame([row(MGR, "202602", M, 7)])]))
    ck2 = {x["key"]: x for x in ic.reconcile(engine, "202609")["checks"]}
    chk("an unattributed manager balance is reported", not ck2["blank_segment"]["ok"]
        and ck2["blank_segment"]["amount"] == 7.0, ck2["blank_segment"])

    # A year with no opening row says so rather than passing activity as a balance.
    load(engine, pd.concat([fixture(), pd.DataFrame([row("AMB6", "202701", E, -1)])]))
    ck3 = {x["key"]: x for x in ic.reconcile(engine, "202701")["checks"]}
    chk("a year without its balance-forward rows is flagged", not ck3["opening"]["ok"])
    chk("periods list newest first", ic.periods(engine)["latest"] == "202701")
    try:
        ic.reconcile(engine, "2026-09")
        chk("a malformed period is refused", False)
    except ValueError:
        chk("a malformed period is refused", True)

    section_writes(engine)
    engine.dispose()
    os.remove(path)


def section_writes(engine):
    print("\n== settings and notes ==")
    st = ic.get_settings(engine)
    chk("five settings seeded, each saying where it came from",
        set(st) == {"PEGASU", "NOTTNV", "PPI2", "PSC2", "PSC1"}
        and all(s["basis"] for s in st.values()), sorted(st))
    chk("PSC1's seed is marked INFERRED", "INFERRED" in st["PSC1"]["basis"])

    ic.save_settings(engine, "PSC1", {}, "tester")
    ic._DDL_DONE.clear()
    ic.ensure_tables(engine)
    chk("a cleared setting is NOT re-seeded", "PSC1" not in ic.get_settings(engine))

    for body, why in (({"alt_account": "MR1500002"}, "short account"),
                      ({"alt_account": "MR15000002"}, "the reconciled account itself"),
                      ({"cash_accounts": "MR10005000", "cash_exclude": "MR10008000"}, "both"),
                      ({"currency": "EUR"}, "currency")):
        try:
            ic.save_settings(engine, "AMB6", body, "tester")
            chk(f"settings refused: {why}", False)
        except ValueError:
            chk(f"settings refused: {why}", True)
    s = ic.save_settings(engine, "amb6", {"cash_accounts": ["MR10005000", "mr10007000"]}, "t")
    chk("settings saved, normalised", s["cash_accounts"] == ["MR10005000", "MR10007000"], s)

    ic.save_note(engine, "202609", "nottnv", "  AP-Other reclass pending  ", "kh")
    r = {x["entity_id"]: x for x in ic.reconcile(engine, "202609")["rows"]}
    chk("a comment is stored per period and shown", r["NOTTNV"]["comment"]
        == "AP-Other reclass pending" and r["NOTTNV"]["comment_by"] == "kh")
    chk("a comment belongs to its period",
        ic.reconcile(engine, "202608")["rows"] and all(
            x["comment"] == "" for x in ic.reconcile(engine, "202608")["rows"]))
    ic.save_note(engine, "202609", "NOTTNV", "", "kh")
    r = {x["entity_id"]: x for x in ic.reconcile(engine, "202609")["rows"]}
    chk("an empty comment clears it", r["NOTTNV"]["comment"] == "")
    chk("the workbook builds", len(ic.to_excel(ic.reconcile(engine, "202609"))) > 2000)


def section_cfo():
    print("\n== the CFO's own rows ==")
    if not WORKBOOK.exists():
        skip("CFO workbook tie-out", f"{WORKBOOK} not on this machine")
        return
    import openpyxl
    from openpyxl.worksheet import worksheet as W
    orig = W.Worksheet.add_table

    def tolerant(self, t):          # the workbook repeats a table name
        try:
            orig(self, t)
        except ValueError:
            pass
    W.Worksheet.add_table = tolerant
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True)
    frames = []
    for sh in ("NEW Due to Manager GEXD", "New Manager Due to Interco GEXD"):
        vals = list(wb[sh].iter_rows(min_row=5, values_only=True))
        df = pd.DataFrame(vals[1:], columns=vals[0]).dropna(how="all")
        frames.append(df)
    W.Worksheet.add_table = orig
    # The manager tab is PSCMAN's rows, which the all-entity tab already holds.
    df = frames[0].rename(columns={"DEFINED_CODE": "RLTDENTITY",
                                   "DESCRIPTION": "RLTDENTITY_NAME"})
    df["PERIOD"] = df["PERIOD"].astype(int).astype(str)
    df["SEGMENTID"] = None
    df = df[COLS]
    engine, path = fresh_engine()
    load(engine, df)
    res = ic.reconcile(engine, "202609")
    rows = {r["entity_id"]: r for r in res["rows"]}
    chk("entity side -308,316.59 (his D75)",
        res["totals"]["entity_balance"] == -308316.59, res["totals"]["entity_balance"])
    chk("manager side 313,093.97 (his H75)",
        res["totals"]["manager_balance"] == 313093.97, res["totals"]["manager_balance"])
    for e, v in (("NOTTNV", -22385.07), ("PSC2", -2467.66), ("PPI2", -274.16)):
        chk(f"{e} variance {v:,.2f}", rows[e]["variance"] == v, rows[e]["variance"])
    chk("INVF10 No Balance (his sheet: Investigate, from a formula error)",
        rows["INVF10"]["status"] == "No Balance", rows["INVF10"]["status"])
    chk("PEGASU on MR99991102: -94,941.70, Investigate",
        rows["PEGASU"]["total_entity"] == -94941.70
        and rows["PEGASU"]["status"] == "Investigate", rows["PEGASU"])
    for e, v in (("AMB23", 10847.16), ("PSC1", 12794339.93), ("PEGASU", 313530.29),
                 ("OWPSC", 51181.0)):
        chk(f"{e} cash {v:,.2f} = his typed figure", rows[e]["cash_balance"] == v,
            rows[e]["cash_balance"])
    engine.dispose()
    os.remove(path)


if __name__ == "__main__":
    section_fixture()
    section_cfo()
    print(f"\n{len(_passed)} passed, {len(_failed)} failed, {len(_skipped)} skipped")
    sys.exit(1 if _failed else 0)
