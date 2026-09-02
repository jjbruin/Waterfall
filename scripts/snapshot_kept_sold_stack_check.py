"""Guardrail: a deal kept on the report after its sale keeps its capital stack.

THE DEFECT. Every equity figure on the Financial subtab is netted from
accounting **through the reported quarter end** — contributions less returns of
capital. For a deal we still hold that is the right question. For one we have
sold it is the wrong one: the sale returns the capital, the netting takes the
row to zero, and a ``KEEP_DESPITE_SOLD`` row — which is on the page precisely
to report that capital and the ROE earned on it — loses the numbers it exists
for. East Manchester (P0000017, sold 6/25/2026) read Total Pref $0, Invested $0
and Total Commitment $0 at 26Q2 against a real $3,600,000 tranche, with only
Ptr. Equity $2.4M left standing.

THE RULE. ``portfolio_snapshot_service.last_held_quarter`` — a kept-despite-sold
row reports its capital stack **as at the last quarter it was held**, which is
always the quarter before the one containing the sale. No vcode appears in it.
It is the same ``get_capitalization_stack`` definition every other row uses,
asked at the last date it was meaningful.

WHY THE LOCAL SNAPSHOT CANNOT SHOW THE DEFECT ON ITS OWN, AND WHAT THIS SCRIPT
DOES ABOUT IT. Local ``waterfall.db`` carries Sale_Date 12/01/2030 for East
Manchester where live has 6/25/2026, and its accounting feed ends 2026-06-02 —
before the sale — so the sale's return-of-capital rows are simply absent. The
check therefore renders three states of the same row: the local one, the live
one (sale date injected, the PE return of capital appended), and the case where
the operating partner is returned too. Only the second reproduces what the
report author sees, and it does so exactly.

    # from a worktree at origin/main, with this file copied in:
    python scripts/snapshot_kept_sold_stack_check.py capture <before.json>
    # from the working tree:
    python scripts/snapshot_kept_sold_stack_check.py capture <after.json>
    python scripts/snapshot_kept_sold_stack_check.py report <before.json> <after.json>
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

INVESTOR = "TGAM"
QUARTERS = ("2026-Q1", "2026-Q2")
EM = "P0000017"
CW = "PCITWES"

#: The sale as live carries it, and the return of capital the local feed stops
#: short of. 3,600,000 to the PE investor is East Manchester's whole tranche —
#: independently corroborated by the `commitments` table, which holds
#: EASTMA/PPI20 3,600,000 dated 2020-12-18.
LIVE_SALE_DATE = "6/25/2026"
ROC_PE = [("PPI20", 3_600_000.0)]
ROC_BOTH = [("PPI20", 3_600_000.0), ("OPVAST", 2_400_000.0)]

#: Everything the page prints for a deal, plus the audit twins. `stack_quarter`
#: is deliberately NOT here: it is new, so it would show as a difference on
#: every row and drown the comparison. It is asserted separately.
ROW_FIELDS = (
    "vcode", "name", "investment_strategy", "is_dev", "kept_despite_sold",
    "sold_label", "pdf_na_cells", "debt", "debt_display", "debt_summable",
    "total_pref", "ptr_equity", "total_cap", "pct_of_pref", "invested",
    "total_commitment", "unfunded", "net_roe_display", "itd_display",
    "total_cap_funded", "total_cap_funded_basis",
    "total_commitment_if_funded", "total_commitment_if_committed",
)


def _flat(fin: dict) -> dict:
    rows = {r["vcode"].upper(): r
            for b in (fin.get("groups") or {}).values() for r in b["deals"]}
    rows.update({r["vcode"].upper(): r
                 for r in (fin.get("ownership_flagged") or [])})
    return {vc: {f: r.get(f) for f in ROW_FIELDS} for vc, r in rows.items()}


def _stack_quarters(fin: dict) -> dict:
    rows = {r["vcode"].upper(): r
            for b in (fin.get("groups") or {}).values() for r in b["deals"]}
    rows.update({r["vcode"].upper(): r
                 for r in (fin.get("ownership_flagged") or [])})
    return {vc: r.get("stack_quarter") for vc, r in rows.items()}


def capture(out_path: str) -> int:
    import pandas as pd
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_freeze import build_subtab
        from flask_app.services.portfolio_snapshot_service import (
            resolve_investor_deals,
        )

        base = dict(data_service.get_data())

        def run(quarter: str, sale_date=None, roc=()):
            d = dict(base)
            if sale_date:
                inv = d["inv"].copy()
                inv.loc[inv["vcode"].astype(str).str.upper() == EM,
                        "Sale_Date"] = sale_date
                d["inv"] = inv
            if roc:
                acct = d["acct"]
                tmpl = acct[acct["InvestmentID"].astype(str).str.strip()
                            == "EASTMA"].iloc[0]
                new = []
                for investor_id, amt in roc:
                    r = tmpl.copy()
                    r["InvestorID"] = investor_id
                    r["MajorType"] = "Distribution"
                    r["Typename"] = "Distribution: Return of Capital"
                    if "TypeName" in r.index:
                        r["TypeName"] = "Distribution: Return of Capital"
                    r["Amt"] = amt
                    r["EffectiveDate"] = pd.Timestamp("2026-06-25")
                    for col, val in (("is_commitment", False),
                                     ("is_distribution", True),
                                     ("is_contribution", False),
                                     ("Capital", "Y")):
                        if col in r.index:
                            r[col] = val
                    new.append(r)
                d["acct"] = pd.concat([acct, pd.DataFrame(new)],
                                      ignore_index=True)
            resolved = resolve_investor_deals(INVESTOR, quarter,
                                              d.get("relationships_raw"),
                                              d["inv"])
            fin = build_subtab("financial", INVESTOR, quarter, d, resolved)
            summ = build_subtab("summary", INVESTOR, quarter, d, resolved)
            return fin, summ

        def _summ(s: dict) -> dict:
            """The page-1 figures a rebase could plausibly touch."""
            return {
                "total_funded": (s.get("asset_allocation") or {})
                                .get("total_funded"),
                "total_committed": (s.get("asset_allocation") or {})
                                   .get("total_committed"),
                "buckets": {b["label"]: [b.get("funded"), b.get("committed"),
                                         b.get("deal_count")]
                            for b in ((s.get("asset_allocation") or {})
                                      .get("buckets") or [])},
                "excluded_from_allocation": {
                    e["vcode"].upper(): e.get("funded_deal_level")
                    for e in (s.get("excluded_from_allocation") or [])},
            }

        out = {"pages": {}, "stack_quarters": {}, "summary": {},
               "scenarios": {}}
        # The report as it really renders on this snapshot — the population the
        # change must not disturb.
        for q in QUARTERS:
            fin, summ = run(q)
            out["pages"][q] = _flat(fin)
            out["stack_quarters"][q] = _stack_quarters(fin)
            out["summary"][q] = _summ(summ)

        # The three renderings of the kept-sold row.
        for label, sd, roc in (
                ("local", LIVE_SALE_DATE, ()),
                ("live_pe_returned", LIVE_SALE_DATE, ROC_PE),
                ("live_both_returned", LIVE_SALE_DATE, ROC_BOTH)):
            fin, summ = run("2026-Q2", sale_date=sd, roc=roc)
            flat = _flat(fin)
            sq = _stack_quarters(fin)
            out["scenarios"][label] = {
                "EM": flat.get(EM), "CW": flat.get(CW),
                "EM_stack_quarter": sq.get(EM),
                "CW_stack_quarter": sq.get(CW),
                "summary": _summ(summ),
            }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(f"wrote {out_path}")
    return 0


CHECKS: list = []


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def _m(v):
    return "blank" if not v else f"{v / 1e6:,.2f}M"


def report(before_path: str, after_path: str) -> int:
    with open(before_path, encoding="utf-8") as fh:
        b = json.load(fh)
    with open(after_path, encoding="utf-8") as fh:
        a = json.load(fh)

    print("\n1. The defect, reproduced — the sale's return of capital blanks "
          "the row")
    bs = b["scenarios"]["live_pe_returned"]["EM"]
    chk("before: Total Pref, Invested and Total Commitment all read zero",
        not bs["total_pref"] and not bs["invested"]
        and not bs["total_commitment"],
        f"Total Pref {_m(bs['total_pref'])} · Invested {_m(bs['invested'])} · "
        f"Total Commitment {_m(bs['total_commitment'])} — a 0 prints as a dash")
    chk("before: Ptr. Equity survives, because only the PE side was returned",
        abs((bs["ptr_equity"] or 0) - 2_400_000) < 1,
        f"{_m(bs['ptr_equity'])} — this is why the author sees ONE figure left")
    chk("before: % of Pref survives too — it is ownership-derived, not netted",
        bs["pct_of_pref"] and 0.7 < bs["pct_of_pref"] < 0.8,
        f"{bs['pct_of_pref']}")
    bl = b["scenarios"]["local"]["EM"]
    chk("and it CANNOT be seen on the unsimulated local snapshot",
        abs((bl["total_pref"] or 0) - 3_600_000) < 1,
        "local accounting ends 2026-06-02, before the 6/25/2026 sale, so the "
        "return of capital is absent and the row still reads 3.60M")

    print("\n2. The rule restores it — from the last quarter it was held")
    for label in ("live_pe_returned", "live_both_returned", "local"):
        s = a["scenarios"][label]["EM"]
        ok = (abs((s["total_pref"] or 0) - 3_600_000) < 1
              and abs((s["ptr_equity"] or 0) - 2_400_000) < 1
              and abs((s["invested"] or 0) - 2_723_400) < 1
              and abs((s["total_commitment"] or 0) - 2_723_400) < 1
              and abs((s["total_cap"] or 0) - 6_000_000) < 1)
        chk(f"after / {label}: the original stack, whatever the sale returned",
            ok,
            f"Total Pref {_m(s['total_pref'])} · Ptr {_m(s['ptr_equity'])} · "
            f"Invested {s['invested']:,.0f} · Commitment "
            f"{s['total_commitment']:,.0f} · Total Cap {_m(s['total_cap'])}")
    chk("read at 2026-Q1 — the quarter before the one holding the sale",
        a["scenarios"]["live_pe_returned"]["EM_stack_quarter"] == "2026-Q1",
        str(a["scenarios"]["live_pe_returned"]["EM_stack_quarter"]))
    s = a["scenarios"]["live_pe_returned"]["EM"]
    chk("Debt is still n/a and still out of the subtotal",
        s["debt_display"] == "n/a" and s["debt_summable"] is None,
        "the rebase does not put a sold asset's loan back on the page")
    chk("Net ROE is still typeable and ITD still manual",
        s["net_roe_display"] == "pending entry"
        and s["itd_display"] == "pending entry",
        "v413 is intact")
    chk("Total Cap still foots to what the row PRINTS",
        abs((s["total_cap"] or 0)
            - ((s["total_pref"] or 0) + (s["ptr_equity"] or 0))) < 1,
        "0 debt (n/a) + 3.60M pref + 2.40M ptr = 6.00M")

    print("\n3. City West — untouched, and provably so")
    for label in ("local", "live_pe_returned", "live_both_returned"):
        chk(f"identical before/after in scenario {label}",
            b["scenarios"][label]["CW"] == a["scenarios"][label]["CW"],
            f"Total Pref {_m(a['scenarios'][label]['CW']['total_pref'])} · "
            f"Ptr {_m(a['scenarios'][label]['CW']['ptr_equity'])} · Net ROE "
            f"{a['scenarios'][label]['CW']['net_roe_display']}")
    chk("its stack IS rebased by the rule — to 2025-Q2, before the "
        "8/30/2025 foreclosure",
        a["scenarios"]["local"]["CW_stack_quarter"] == "2025-Q2",
        str(a["scenarios"]["local"]["CW_stack_quarter"]))
    chk("and that is a NO-OP, because City West has no return of capital at "
        "all — the figures are flat across every quarter",
        b["scenarios"]["local"]["CW"] == a["scenarios"]["local"]["CW"],
        "not asserted — measured: rebasing it changes nothing")

    print("\n4. Every other deal on both pages — byte-identical")
    for q in QUARTERS:
        bp, ap = b["pages"][q], a["pages"][q]
        chk(f"{q}: same population ({len(ap)} deals)",
            set(bp) == set(ap),
            f"only in before {sorted(set(bp) - set(ap))} / "
            f"only in after {sorted(set(ap) - set(bp))}")
        moved = sorted(vc for vc in set(bp) & set(ap) if bp[vc] != ap[vc])
        chk(f"{q}: no deal changed", not moved, f"moved: {moved}")
        rebased = sorted(vc for vc, sq in (a["stack_quarters"][q] or {}).items()
                         if sq != q)
        # City West only. East Manchester's LOCAL Sale_Date is 12/01/2030, so
        # its sold gate never fires on this snapshot and it is reported as a
        # live deal at both quarters — which is exactly why the scenarios in
        # section 2 have to inject the live date to see it at all.
        chk(f"{q}: only the kept-sold rows read another quarter",
            rebased == [CW],
            f"rebased {rebased or 'none'}")
        chk(f"{q}: every deal still held reads the reported quarter",
            all(sq == q for vc, sq in (a["stack_quarters"][q] or {}).items()
                if vc not in (EM, CW)))

    print("\n5. Page 1 — no allocation total moves, and it stops disagreeing "
          "with page 2")
    for q in QUARTERS:
        chk(f"{q}: Summary asset allocation identical before/after",
            b["summary"][q] == a["summary"][q],
            f"total funded {b['summary'][q]['total_funded']} -> "
            f"{a['summary'][q]['total_funded']}")
    bsm = b["scenarios"]["live_pe_returned"]["summary"]
    asm = a["scenarios"]["live_pe_returned"]["summary"]
    chk("allocation totals unmoved in the simulated-live scenario too",
        bsm["total_funded"] == asm["total_funded"]
        and bsm["buckets"] == asm["buckets"],
        "a kept-sold deal is out of every allocation total by construction")
    chk("before: page 1 reported East Manchester's funded pref as 0 while "
        "page 2 was about to print 3.60M",
        not bsm["excluded_from_allocation"].get(EM),
        f"{bsm['excluded_from_allocation'].get(EM)}")
    chk("after: the two pages agree on the same deal",
        abs((asm["excluded_from_allocation"].get(EM) or 0) - 3_600_000) < 1
        and abs((a["scenarios"]["live_pe_returned"]["EM"]["total_pref"] or 0)
                - 3_600_000) < 1,
        f"page 1 funded {asm['excluded_from_allocation'].get(EM)} · page 2 "
        f"Total Pref "
        f"{a['scenarios']['live_pe_returned']['EM']['total_pref']}")

    print("\n6. The rule is a rule — no vcode in it")
    import inspect
    from flask_app.services.portfolio_snapshot_service import (
        last_held_quarter, KEEP_DESPITE_SOLD,
    )
    src = inspect.getsource(last_held_quarter)
    body = "\n".join(l for l in src.splitlines()
                     if not l.strip().startswith(("#", '"', "'")))
    chk("last_held_quarter names no deal", "P0000017" not in body
        and "PCITWES" not in body,
        "it keys on the sale date, and is reached only through the "
        "KEEP_DESPITE_SOLD set that was already there")
    chk("both kept deals are still in KEEP_DESPITE_SOLD",
        {CW, EM} <= set(KEEP_DESPITE_SOLD))

    print("\n7. last_held_quarter — the edge cases")
    import pandas as pd
    for label, meta, want in (
            ("mid-quarter sale -> the previous quarter",
             {"sale_date": pd.Timestamp("2026-06-25")}, "2026-Q1"),
            ("sale ON a quarter end -> still the previous quarter",
             {"sale_date": pd.Timestamp("2026-03-31")}, "2025-Q4"),
            ("first day of a quarter -> the one that just closed",
             {"sale_date": pd.Timestamp("2026-04-01")}, "2026-Q1"),
            ("no sale date -> no rebase",
             {"sale_date": None}, None),
            ("NaT sale date -> no rebase",
             {"sale_date": pd.NaT}, None),
            ("sale before the deal closed -> no rebase, not an empty stack",
             {"sale_date": pd.Timestamp("2026-06-25"),
              "acquisition_date": pd.Timestamp("2026-05-01")}, None),
            ("acquired inside the prior quarter -> rebase is fine",
             {"sale_date": pd.Timestamp("2026-06-25"),
              "acquisition_date": pd.Timestamp("2026-02-01")}, "2026-Q1")):
        got = last_held_quarter(meta)
        chk(label, got == want, f"got {got!r}, want {want!r}")

    passed = sum(CHECKS)
    print(f"\n  {passed}/{len(CHECKS)} checks passed")
    return 0 if passed == len(CHECKS) else 1


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        return capture(sys.argv[2])
    if len(sys.argv) >= 4 and sys.argv[1] == "report":
        return report(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
