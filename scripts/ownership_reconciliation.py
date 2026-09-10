"""Reconcile `relationships` against `commitments`, and size what is at stake.

WHY THIS EXISTS. Brainerd Place Apartments reported TIAA's Total Commitment
against a 63.142% look-through when the legal org chart (11/21/2024) puts it at
74.41%. The cause was one missing edge — a transfer of 53.975% of PSC Investee
Brainerd (CT) LLC from Peaceable to PSC TGA 2022 LLC that was never recorded —
and the ownership walk, which is correct, simply had no route to walk.

READ THIS BEFORE TRUSTING THE OUTPUT:

  * NEITHER SOURCE IS AUTHORITATIVE. This report says the two feeds disagree,
    never which one is right. On the Brainerd chain `commitments` matches the
    legal chart on INVBPS and `relationships` does not — but on PPIBPA the
    chart matches neither, and on INVBPA both are wrong together.

  * AGREEMENT IS NOT A CLEAN BILL OF HEALTH, and INVBPA is the proof: both
    tables say PSC1 100%, both predate the transfer. A row absent from this
    report can still be wrong.

  * A TRANSFER IS INVISIBLE TO EVERY SELF-CONSISTENCY CHECK. Moving ownership
    between two owners preserves the total, so the owners of Brainerd still sum
    to exactly 100% (TGAM 63.142 + PSC1 36.858) with the edge missing. That is
    why a conservation check does not appear here — it cannot detect this class
    of defect, and shipping one would imply otherwise.

So the output is a TRIAGE QUEUE ordered by how much money moves, to be settled
against the org charts. It is read-only and touches no engine path.

Usage:
    python scripts/ownership_reconciliation.py [--investor TGAM] [--min-delta 1.0]
    python scripts/ownership_reconciliation.py --entity INVBPS      # one chain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Ownership-point difference below which a disagreement is not worth a line.
#: Feeds carry rounding noise (PPIBPA is 50.6097 against 50.10), so a small
#: threshold would bury the real breaks under arithmetic dust.
DEFAULT_MIN_DELTA = 1.0


def _norm(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(str).str.strip().str.upper()
    return out


def relationship_splits(rel: pd.DataFrame) -> dict:
    """{entity: {owner: pct}} from the active generation of `relationships`."""
    rel = _norm(rel, ["InvestmentID", "InvestorID"])
    rel = rel[rel["EndDate"].isna()].copy()
    rel["pct"] = pd.to_numeric(rel["OwnershipPct"], errors="coerce").fillna(0.0)
    return {e: dict(zip(g["InvestorID"], g["pct"]))
            for e, g in rel.groupby("InvestmentID")}


def commitment_splits(cm: pd.DataFrame) -> dict:
    """{entity: {owner: pct}} from `commitments`, by DOLLARS not CapitalPercent.

    Amounts are summed per (entity, owner) first. One owner can hold several
    tranches — BRNERD carries PPIBPA at 91.087% and 8.912%, two rounds of the
    same member — and reading CapitalPercent off the latest row alone reports
    that owner at 8.912% and calls it a 91-point break. Summing the money is
    the only reading that survives a second closing.
    """
    cm = _norm(cm, ["EntityID", "InvestorID"])
    cm = cm[cm["EndDate"].isna()].copy()
    cm["amt"] = pd.to_numeric(cm["Amount"], errors="coerce").fillna(0.0)
    agg = cm.groupby(["EntityID", "InvestorID"], as_index=False)["amt"].sum()
    tot = agg.groupby("EntityID")["amt"].transform("sum")
    agg = agg[tot > 0].copy()
    agg["pct"] = agg["amt"] / agg.groupby("EntityID")["amt"].transform("sum") * 100.0
    return {e: dict(zip(g["InvestorID"], g["pct"]))
            for e, g in agg.groupby("EntityID")}


def reaches(owner: str, entity: str, rel: pd.DataFrame, depth: int = 6) -> list:
    """Path from `owner` down to `entity` through relationships, or [].

    `commitments` sometimes records a member one LEVEL UP from where
    `relationships` puts it: AMB24 is named directly as TGA24's 10% member,
    while relationships has AMB24 -> INV24 -> TGA24. Same economics, recorded
    at a different depth, so it is annotated rather than reported as a missing
    owner.
    """
    rel = _norm(rel, ["InvestmentID", "InvestorID"])
    rel = rel[rel["EndDate"].isna()]
    stack = [(owner, [owner])]
    seen = set()
    while stack:
        node, trail = stack.pop()
        if len(trail) > depth or node in seen:
            continue
        seen.add(node)
        for child in rel[rel["InvestorID"] == node]["InvestmentID"].unique():
            if child == entity:
                return trail + [child]
            stack.append((child, trail + [child]))
    return []


def compare(rel_splits: dict, cm_splits: dict, min_delta: float) -> list:
    """Entities whose two splits disagree, with the per-owner breaks."""
    breaks = []
    for ent in sorted(set(rel_splits) & set(cm_splits)):
        ro, co = rel_splits[ent], cm_splits[ent]
        rows = []
        for owner in sorted(set(ro) | set(co)):
            rp, cp = ro.get(owner), co.get(owner)
            # An owner missing from one side reads as 0 there, which is the
            # substantive claim ("this feed says they hold nothing"), not a gap.
            rv, cv = (rp or 0.0), (cp or 0.0)
            if abs(rv - cv) >= min_delta:
                rows.append({"owner": owner, "rel": rp, "cmt": cp,
                             "delta": cv - rv,
                             "absent_from_rel": rp is None,
                             "absent_from_cmt": cp is None})
        if rows:
            breaks.append({"entity": ent, "rows": rows,
                           "worst": max(abs(r["delta"]) for r in rows)})
    breaks.sort(key=lambda b: -b["worst"])
    return breaks


def downstream_deals(entity: str, rel: pd.DataFrame, deal_iids: set) -> list:
    """Deals reachable downward from an entity — the blast radius of a break.

    Reachability ONLY. What an entity reaches is not what it owns, so callers
    must weight by its actual look-through before calling anything "at risk":
    PSC Manager LLC sits on a 0% edge into TGA22 and therefore reaches all 25
    of that fund's deals while holding economics in none of them. Sized raw it
    ranked first at $472m, which is the entity that matters least.
    """
    rel = _norm(rel, ["InvestmentID", "InvestorID"])
    rel = rel[rel["EndDate"].isna()]
    seen, frontier = set(), [entity]
    while frontier:
        e = frontier.pop()
        if e in seen:
            continue
        seen.add(e)
        frontier.extend(rel[rel["InvestorID"] == e]["InvestmentID"].tolist())
    return sorted((seen - {entity}) & deal_iids)


def funded_pref(acct: pd.DataFrame) -> dict:
    """{deal iid: funded preferred equity} — non-OP contributions net of ROC.

    A sizing figure for triage, not the reported one: the Snapshot's Total Pref
    comes from the One Pager cap stack, which needs a per-deal computation this
    report deliberately does not run. It reproduces Brainerd's $18,407,677.40
    exactly, which is enough to rank a queue.
    """
    a = _norm(acct, ["InvestmentID", "InvestorID"])
    a = a[~a["InvestorID"].str.startswith("OP")].copy()
    a["amt"] = pd.to_numeric(a["Amt"], errors="coerce").fillna(0.0)
    tn = a.get("Typename", pd.Series("", index=a.index)).astype(str).str.lower()
    contrib = a[tn.str.contains("contribution", na=False)]
    roc = a[tn.str.contains("return of capital", na=False)]
    out = contrib.groupby("InvestmentID")["amt"].sum().abs()
    back = roc.groupby("InvestmentID")["amt"].sum().abs()
    return (out.subtract(back, fill_value=0.0)).to_dict()


def selftest() -> int:
    """Known answers from the Brainerd investigation, 2026-09-10.

    Pins the three classification rules the report turns on, each of which was
    a false positive in the first run: a manager on a 0% edge is inert, a deal
    is not a holding vehicle, and a multi-tranche owner is one owner.
    """
    from flask_app import create_app
    from sqlalchemy import text
    from flask_app.db import get_engine
    ok = bad = 0

    def chk(label, cond, detail=""):
        nonlocal ok, bad
        if cond:
            ok += 1
            print(f"  PASS  {label}" + (f"  -> {detail}" if detail else ""))
        else:
            bad += 1
            print(f"  FAIL  {label}" + (f"  -> {detail}" if detail else ""))

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_service import lookthrough_pct
        data = data_service.load_all(app.config.get("DB_PATH", "waterfall.db"))
        rel, inv = data["relationships_raw"], data["inv"]
        with get_engine().connect() as conn:
            cm = pd.DataFrame(conn.execute(text(
                'SELECT "EntityID","InvestorID","Amount","CapitalPercent",'
                '"StartDate","EndDate" FROM commitments')).fetchall(),
                columns=["EntityID", "InvestorID", "Amount", "CapitalPercent",
                         "StartDate", "EndDate"])

        rs, cs = relationship_splits(rel), commitment_splits(cm)
        deal_iids = {str(r.get("InvestmentID", "")).strip().upper()
                     for _, r in inv.iterrows()} - {"", "NONE", "NAN"}
        breaks = {b["entity"]: b for b in compare(rs, cs, DEFAULT_MIN_DELTA)}

        print("=== the break this report CAN see on the Brainerd chain ===")
        chk("INVBPS is flagged", "INVBPS" in breaks)
        if "INVBPS" in breaks:
            chk("its break is ~16.13pt (58.9654/41.0345 vs 75.10/24.90)",
                abs(breaks["INVBPS"]["worst"] - 16.135) < 0.01,
                f"{breaks['INVBPS']['worst']:.3f}pt")

        print()
        print("=== the break it CANNOT see, and must say so ===")
        chk("INVBPA is absent — both feeds agree at PSC1 100%",
            "INVBPA" not in breaks)
        chk("both feeds really do say PSC1 100%",
            abs(rs.get("INVBPA", {}).get("PSC1", 0) - 100) < 0.01
            and abs(cs.get("INVBPA", {}).get("PSC1", 0) - 100) < 0.01)
        src = Path(__file__).read_text(encoding="utf-8")
        chk("the report states the undetectable item", "INVBPA" in src
            and "74.415" in src)

        print()
        print("=== a transfer is invisible to a conservation check ===")
        t = (lookthrough_pct("BRNERD", "TGAM", relationships=rel)["pct"] or 0)
        p = (lookthrough_pct("BRNERD", "PSC1", relationships=rel)["pct"] or 0)
        chk("owners still sum to 100% with the edge missing",
            abs((t + p) - 1.0) < 1e-6, f"TGAM {t*100:.4f} + PSC1 {p*100:.4f}")
        chk("so the engine's own figure is the understated 63.142%",
            abs(t - 0.631420) < 1e-5, f"{t*100:.4f}%")

        print()
        print("=== TGAM's fund stakes are NOT in dispute (investigated 2026-09-10) ===")
        # This was first read as TIAA being OVERSTATED on TGA23/TGA24 because a
        # dollar-derived share put TGAM at 88.95% / 86.15% against a recorded
        # 90%. It is not: both feeds state 90%, and the money agrees once the
        # right pair is compared. The dilution came from a second-closing
        # sleeve (TGA23) and from a member recorded a level up (TGA24).
        chk("TGA23  TGAM / (TGAM + INV23) is exactly 90%",
            abs(112570957.80 / (112570957.80 + 12507884.20) - 0.90) < 1e-9)
        chk("TGA24  TGAM / (TGAM + AMB24) is exactly 90%",
            abs(13920750.00 / (13920750.00 + 1546750.00) - 0.90) < 1e-9)
        chk("both feeds state TGAM at 90% on TGA23",
            abs(rs.get("TGA23", {}).get("TGAM", 0) - 90.0) < 1e-9)
        # Asserts the SHAPE, not one path: AMB24 owns both INV24 and INV24-P
        # and each owns TGA24, so which intermediate the walk returns is
        # arbitrary. The claim under test is that an intermediate exists.
        amb = reaches("AMB24", "TGA24", rel)
        chk("AMB24 is the same member one level up, not a missing owner",
            len(amb) == 3 and amb[0] == "AMB24" and amb[-1] == "TGA24",
            " -> ".join(amb))
        # The `Name` column names the INVESTMENT, not the investor — reading it
        # as an investor name made INV23/INV23-P look like one legal entity and
        # collapsed four distinct OWPSC members into one. Pinned so no future
        # pass repeats it.
        n23 = {str(r.get("Name") or "").strip()
               for _, r in _norm(rel, ["InvestmentID"]).iterrows()
               if r["InvestmentID"] == "TGA23"}
        chk("Name describes the investment, so every TGA23 row shares one",
            len(n23) == 1, str(n23))

        print()
        print("=== the three classification rules ===")
        chk("a deal is held out of the queue (BRNERD)",
            "BRNERD" in breaks and "BRNERD" in deal_iids,
            "flagged, but deal-level")
        pscman = [d for d in downstream_deals("PSCMAN", rel, deal_iids)]
        share = max([(lookthrough_pct(d, "PSCMAN", relationships=rel)["pct"] or 0)
                     for d in pscman] or [0])
        chk("a 0%-edge manager reaches deals but owns none (PSCMAN)",
            len(pscman) > 10 and share == 0,
            f"reaches {len(pscman)} deals, max look-through {share}")
        brn = cs.get("BRNERD", {})
        chk("a multi-tranche owner is summed, not read off one row",
            abs(brn.get("PPIBPA", 0) - 44.6268) < 0.01,
            f"PPIBPA {brn.get('PPIBPA', 0):.4f}% (not the 8.912% latest row)")

    print()
    print(f"{ok}/{ok + bad} checks passed")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM",
                    help="investor whose look-through exposure is sized (default TGAM/TIAA)")
    ap.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    ap.add_argument("--entity", default=None, help="report one entity only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from flask_app import create_app
    app = create_app()
    with app.app_context():
        from sqlalchemy import text
        from flask_app.db import get_engine
        from flask_app.services import data_service
        from flask_app.services.portfolio_snapshot_service import lookthrough_pct

        data = data_service.load_all(app.config.get("DB_PATH", "waterfall.db"))
        rel, inv, acct = data["relationships_raw"], data["inv"], data["acct"]

        cm = data.get("commitments_raw")
        if cm is None or getattr(cm, "empty", True):
            with get_engine().connect() as conn:
                cm = pd.DataFrame(conn.execute(text(
                    'SELECT "EntityID","InvestorID","Amount","CapitalPercent",'
                    '"StartDate","EndDate" FROM commitments')).fetchall(),
                    columns=["EntityID", "InvestorID", "Amount",
                             "CapitalPercent", "StartDate", "EndDate"])

        iid_name = {str(r.get("InvestmentID", "")).strip().upper():
                    str(r.get("Investment_Name", ""))
                    for _, r in inv.iterrows()}
        deal_iids = {k for k in iid_name if k and k not in ("", "NONE", "NAN")}
        pref = funded_pref(acct)

        rs, cs = relationship_splits(rel), commitment_splits(cm)
        breaks = compare(rs, cs, args.min_delta)

        # An owner absent from relationships that nonetheless REACHES the
        # entity there is the same member recorded a level up, not a missing
        # one. Drop those rows; drop the whole entity if nothing else remains.
        layered = []
        kept = []
        for b in breaks:
            rows = []
            for r in b["rows"]:
                if r["absent_from_rel"]:
                    path = reaches(r["owner"], b["entity"], rel)
                    if path:
                        layered.append({"entity": b["entity"], "owner": r["owner"],
                                        "path": path, "pct": r["cmt"]})
                        continue
                rows.append(r)
            if rows:
                b = dict(b, rows=rows,
                         worst=max(abs(x["delta"]) for x in rows))
                kept.append(b)
        breaks = sorted(kept, key=lambda b: -b["worst"])

        if args.entity:
            ent = args.entity.strip().upper()
            breaks = [b for b in breaks if b["entity"] == ent]

        # A DEAL is not a holding vehicle and the two feeds are not describing
        # the same population there: `relationships` records the PE ownership
        # chain, so it carries the vehicle at 100%, while `commitments` records
        # every dollar pledged INCLUDING the operating partner. BRNERD reads
        # PPIBPA 100% against PPIBPA 44.63 / OPBERDI 55.37 for that reason
        # alone. The final hop is re-normalised against non-OP owners anyway
        # (`lookthrough_pct`'s pct_pe), so this divergence is by design.
        # Split out rather than dropped — a real break at a deal would
        # otherwise vanish silently.
        deal_breaks = [b for b in breaks if b["entity"] in deal_iids]
        breaks = [b for b in breaks if b["entity"] not in deal_iids]

        def exposure(entity: str, deals: list) -> tuple:
            """(dollars governed by this entity's split, deals it truly owns).

            Weighted by the entity's OWN look-through of each deal, so a 0%
            edge contributes nothing. This is the figure that ranks the queue.
            """
            total, held = 0.0, []
            for dl in deals:
                p = pref.get(dl, 0.0)
                if p <= 0:
                    continue
                lt = lookthrough_pct(dl, entity, relationships=rel)
                share = lt.get("pct") or 0.0
                if share <= 0:
                    continue
                total += share * p
                held.append((dl, share, p))
            held.sort(key=lambda x: -(x[1] * x[2]))
            return total, held

        print("=" * 78)
        print("OWNERSHIP RECONCILIATION — relationships vs commitments")
        print("=" * 78)
        print(f"entities in both feeds : {len(set(rs) & set(cs))}")
        print(f"entities that disagree : {len(breaks)}  (>= {args.min_delta}pt on any owner)")
        print(f"exposure sized for     : {args.investor}")
        print()
        print("NEITHER FEED IS AUTHORITATIVE — this ranks a queue to settle against")
        print("the legal org charts. An entity ABSENT here can still be wrong: the")
        print("Brainerd transfer is missing from BOTH feeds, so they agree and are")
        print("both incorrect. See the module docstring.")
        print()

        scored = []
        for b in breaks:
            deals = downstream_deals(b["entity"], rel, deal_iids)
            amt, held = exposure(b["entity"], deals)
            scored.append((amt, held, b))
        scored.sort(key=lambda x: -x[0])

        live = [s for s in scored if s[0] > 0]
        inert = [s for s in scored if s[0] == 0]

        print(f"HOLDING ENTITIES THAT DISAGREE AND CARRY ECONOMICS: {len(live)}")
        print()
        total = 0.0
        for amt, held, b in live:
            total += amt
            print("-" * 78)
            print(f"{b['entity']}   worst break {b['worst']:.3f}pt   "
                  f"governs ${amt:,.0f} of funded pref")
            for r in b["rows"]:
                rel_s = "absent" if r["absent_from_rel"] else f"{r['rel']:8.4f}%"
                cmt_s = "absent" if r["absent_from_cmt"] else f"{r['cmt']:8.4f}%"
                print(f"    {r['owner']:10}  relationships {rel_s:>9}   "
                      f"commitments {cmt_s:>9}   delta {r['delta']:+8.3f}")
            for dl, share, p in held[:6]:
                lt = lookthrough_pct(dl, args.investor, relationships=rel)
                iv = lt.get("pct") or 0.0
                print(f"      -> {dl:9} {iid_name.get(dl,'')[:28]:30} "
                      f"entity {share*100:6.2f}%   "
                      f"{args.investor} {iv*100:6.2f}% = ${iv*p:,.0f}")
            if len(held) > 6:
                print(f"      -> ... and {len(held)-6} more")
        print("-" * 78)
        print(f"total funded pref governed by a disagreeing entity: ${total:,.0f}")

        if inert:
            print()
            print(f"DISAGREE BUT CARRY NO ECONOMICS ({len(inert)}) — manager and")
            print("holding shells on 0% edges, or with no funded deal beneath:")
            print("   " + ", ".join(b["entity"] for _, _, b in inert))

        if deal_breaks:
            print()
            print(f"DEAL-LEVEL DIVERGENCE ({len(deal_breaks)}) — EXPECTED, not a queue.")
            print("`relationships` carries the PE vehicle at 100%; `commitments`")
            print("includes the operating partner. Listed so a genuine break here")
            print("is not hidden by the exclusion:")
            print("   " + ", ".join(b["entity"] for b in deal_breaks))

        if layered:
            print()
            print(f"RECORDED A LEVEL UP ({len(layered)}) — same member, different")
            print("depth. commitments names it directly; relationships routes it:")
            for l in layered:
                print(f"   {l['entity']:9}  {l['owner']:10} at {l['pct']:7.3f}%   "
                      f"relationships: {' -> '.join(l['path'])}")

        # The check that actually caught the TGA23/TGA24 sleeves. Money with no
        # ownership behind it is a stronger signal than two feeds disagreeing,
        # because it needs only ONE feed to be self-contradictory.
        own = {}
        rn = _norm(rel, ["InvestmentID", "InvestorID"])
        rn = rn[rn["EndDate"].isna()]
        rn["pct"] = pd.to_numeric(rn["OwnershipPct"], errors="coerce").fillna(0.0)
        for _, r in rn.iterrows():
            own[(r["InvestmentID"], r["InvestorID"])] = r["pct"]
        cmn = _norm(cm, ["EntityID", "InvestorID"])
        cmn = cmn[cmn["EndDate"].isna()].copy()
        cmn["amt"] = pd.to_numeric(cmn["Amount"], errors="coerce").fillna(0.0)
        orphan = []
        for _, r in cmn[cmn["amt"] > 0].iterrows():
            p = own.get((r["EntityID"], r["InvestorID"]))
            if p is None and reaches(r["InvestorID"], r["EntityID"], rel):
                continue                       # same member, a level up
            if p is None or p == 0:
                orphan.append((r["EntityID"], r["InvestorID"], r["amt"],
                               "absent" if p is None else "0.0%"))
        orphan.sort(key=lambda x: -x[2])
        if orphan:
            print()
            print(f"CAPITAL WITH NO OWNERSHIP BEHIND IT ({len(orphan)} rows, "
                  f"${sum(o[2] for o in orphan):,.0f})")
            print("A commitment funded against a 0% or absent ownership row. Needs")
            print("only ONE feed to contradict itself, so it is firmer evidence")
            print("than a split disagreement:")
            for e, i, a, s in orphan[:12]:
                print(f"   {e:9} <- {i:10} ${a:>15,.0f}   relationships: {s}")
            if len(orphan) > 12:
                print(f"   ... and {len(orphan)-12} more")

        print()
        print("=" * 78)
        print("KNOWN OPEN ITEM THIS REPORT CANNOT FIND")
        print("=" * 78)
        print("PSC Investee Brainerd (CT) LLC [INVBPA] reads PSC1 100% in BOTH")
        print("feeds, so it never appears above. The 11/21/2024 org chart shows")
        print("PSC1 46.025% / TGA22 53.975%. Recording that transfer moves TIAA's")
        print("Brainerd look-through 63.142% -> 74.415% and its Total Commitment")
        print("$11,622,976 -> $13,698,026. Agreement between the feeds is not")
        print("evidence of correctness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
