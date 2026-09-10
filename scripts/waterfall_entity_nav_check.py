"""Guardrail: the Waterfall Setup entity list.

Two rules, both learned from a deal that could not be worked on:

1. AN ACTIVE DEAL IS SELECTABLE WHETHER OR NOT IT HAS A WATERFALL.  The list
   used to be `rel_vcodes | wf_vcodes`, so a deal with neither a relationships
   row nor a waterfall was absent -- and a deal cannot be given its FIRST
   waterfall without being selectable.  Jefferson Stephens (P0000114) sat in
   that gap while it was the deal Jim was trying to build.  The addition must
   be PURELY ADDITIVE: nothing that used to be listed may disappear, and no
   existing row may change, because `entitiesWithWf` (the "Copy from deal"
   dropdown) is derived from the same array.

2. A PLACEHOLDER `InvestmentID` IS NOT AN ID.  25 deals carry the literal
   string ``NONE``, so keying the id->vcode map on it made one arbitrary deal
   -- whichever iterated last -- the answer for all of them.  No relationships
   row carries ``NONE`` today, which is exactly why this needs a test: the
   failure is invisible on current data and would arrive silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}" + (f"  -> {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"  -> {detail}" if detail else ""))


def main() -> int:
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services import waterfall_service as ws

        data = data_service.load_all(app.config.get("DB_PATH", "waterfall.db"))
        inv, rel, wf = data["inv"], data["relationships_raw"], data["wf"]

        # Report a missing rule instead of dying on an AttributeError, so this
        # runs against an older tree (and against a future one that deletes the
        # helper) and says which rule is gone.
        has_placeholder_rule = hasattr(ws, "_is_placeholder_id")
        if not has_placeholder_rule:
            ws._is_placeholder_id = lambda _iid: False

        ents = ws.get_entity_nav_data(wf, inv, rel)["entities"]
        by_vcode = {e["vcode"]: e for e in ents}

        # ---- the old population, rebuilt here so the test is self-contained
        wf_vcodes = {str(v).strip() for v in wf["vcode"].dropna().unique()}
        id_to_vcode = {}
        for _, r in inv.iterrows():
            vc, iid = str(r.get("vcode", "")), str(r.get("InvestmentID", "")).strip()
            if vc and iid and not ws._is_placeholder_id(iid):
                id_to_vcode[iid] = vc
        rel_vcodes = {
            id_to_vcode.get(e, e)
            for e in rel["InvestmentID"].astype(str).str.strip().unique()
        }
        baseline = rel_vcodes | wf_vcodes

        print("=== rule 1: an active deal is selectable without a waterfall ===")
        live = data_service.exclude_sold(inv)
        active = {v for v in live["vcode"].astype(str).str.strip().tolist() if v}
        unreachable = sorted(active - set(by_vcode))
        check("every active deal is in the entity list", not unreachable,
              f"unreachable: {unreachable[:6]}")
        check("Jefferson Stephens P0000114 is selectable", "P0000114" in by_vcode,
              by_vcode.get("P0000114", {}).get("label", "ABSENT"))

        print()
        print("=== the addition is purely additive ===")
        lost = sorted(baseline - set(by_vcode))
        check("nothing that used to be listed is gone", not lost, f"lost: {lost[:6]}")
        check("every listed entity is a real deal, a relationship or a waterfall",
              not (set(by_vcode) - baseline - active),
              f"stray: {sorted(set(by_vcode) - baseline - active)[:6]}")
        # has_wf drives the "Copy from deal" list -- it must not have moved
        check("has_wf is set from the waterfall table alone",
              {v for v, e in by_vcode.items() if e["has_wf"]} == wf_vcodes,
              f"{len(wf_vcodes)} entities with a waterfall")
        added = sorted(set(by_vcode) - baseline)
        check("every newly listed deal reports has_wf False",
              all(not by_vcode[v]["has_wf"] for v in added), f"{len(added)} added")

        print()
        print("=== a sold deal is not added by this rule ===")
        sold = {str(v).strip() for v in inv["vcode"].dropna().astype(str)} - active
        # Sold deals DO appear when a relationships row or a waterfall reaches
        # them -- that is pre-existing and deliberate (a sold deal's waterfall
        # must stay viewable). The rule under test is only that the new
        # active-deal union adds none of them.
        check("no sold deal is added by the active-deal union",
              not (sold & (set(by_vcode) - baseline)),
              f"{len(sold & set(by_vcode))} sold deals listed, all via "
              f"relationships or a waterfall as before")

        print()
        print("=== the list is ordered by what the dropdown shows ===")
        labels = [e["label"] for e in ents]
        check("sorted by label, case-insensitively",
              labels == sorted(labels, key=str.casefold))
        check("no duplicate vcode", len(by_vcode) == len(ents), f"{len(ents)} entities")

        print()
        print("=== rule 2: a placeholder InvestmentID never resolves to a deal ===")
        check("waterfall_service defines the placeholder rule", has_placeholder_rule,
              "" if has_placeholder_rule else "_is_placeholder_id is absent")
        check("'NONE' is treated as a placeholder", ws._is_placeholder_id("NONE"))
        check("case and padding do not evade it", ws._is_placeholder_id("  none "))
        check("a real id is untouched", not ws._is_placeholder_id("EASTCH"))

        # Inject the row that does not exist today. Under the old rule it was
        # absorbed by whichever NONE-carrying deal iterated last, changing the
        # entity list not at all -- the failure had no symptom.
        injected = pd.concat([rel, pd.DataFrame([{
            "InvestmentID": "NONE", "InvestorID": "PPI22", "OwnershipPct": 0.5,
            "Name": "injected", "StartDate": None, "EndDate": None,
        }])], ignore_index=True)
        after = {e["vcode"] for e in ws.get_entity_nav_data(wf, inv, injected)["entities"]}
        check("an unmapped NONE row stays visible as its own entity",
              "NONE" in after,
              "rather than silently attaching to an arbitrary deal")

        nones = [str(r["vcode"]).strip() for _, r in inv.iterrows()
                 if ws._is_placeholder_id(str(r.get("InvestmentID", "")))]
        # The collision made all of these resolve to ONE deal. Each ACTIVE one
        # must now appear under its own vcode; the sold ones are absent for the
        # ordinary reason (no relationships row, no waterfall), not by collapse.
        active_nones = [v for v in nones if v in active]
        check("every active deal carrying a placeholder id is listed separately",
              len(active_nones) > 1 and all(v in by_vcode for v in active_nones),
              f"{len(active_nones)} active of {len(nones)} placeholder-id deals")
        check("no two of them collapsed onto one entity",
              len({by_vcode[v]["vcode"] for v in active_nones}) == len(active_nones))

    print()
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
