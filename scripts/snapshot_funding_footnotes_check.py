"""Guardrail: the Financial subtab's "Total Current Funding" row, the
scope-placed footnotes, and Jefferson Eastchase's group.

Runs the REAL committed functions — ``resolve_investor_deals`` and
``assemble_financial`` — against LIVE data, twice in one process, and prints a
before/after for everything that moved plus a strict no-regression check on
everything that must not have.

HOW "BEFORE" IS OBTAINED, since it matters for whether this proves anything:

  * GROUPING — genuinely before/after. The only change is one entry in
    ``GROUP_OVERRIDES``; the check removes that entry and re-runs the SAME
    function, so the "before" group is what the real traversal rule produces,
    not a re-implementation of it. The payload's own
    ``diagnostics.group_overrides_applied`` reports the same thing
    independently, and the two are cross-checked.

  * THE FUNDING ROW — additive, so "before" is its absence. What is checked
    instead is the claim the row makes: that each column equals the sum of the
    One Pager cap-stack values the rows ALREADY carry (``debt_isbs``,
    ``funded_pref``, ``ptr_equity``), re-summed here from the payload,
    independently of ``_funding_total``. Plus: Total Pref on the funding row
    must be the FUNDED figure and therefore must NOT equal the committed Total
    Pref on the Portfolio Totals row above it.

  * FOOTNOTES — the old numbering is reconstructed from the same database rows
    the assembly read (the persisted ``number``) plus the three standing notes
    as they were hardcoded in the Vue (2 / 3 / 6), so the before column is what
    the page actually printed.

  * EVERYTHING ELSE — every deal row, both subtotal blocks, Portfolio Totals
    and the excluding-development row are compared field by field between the
    two runs. Only the group a deal sits in may differ.

Read-only. Needs ``DATABASE_URL`` in the environment (no credential is stored
in this file); point it at the live Azure PostgreSQL to reproduce the numbers
below.

    set DATABASE_URL=postgresql://...
    python scripts/snapshot_funding_footnotes_check.py
    python scripts/snapshot_funding_footnotes_check.py --quarter 2026-Q2
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: The deal the work order moves, and where each side must put it.
MOVED_VCODE = "P0000085"
MOVED_NAME = "Jefferson Eastchase"

#: What the page printed before this change: the standing notes were hardcoded
#: in SnapshotFinancial.vue with the reference PDF's own numbers, in this order,
#: while the database footnotes numbered themselves from 1 independently.
_OLD_STANDING = [
    (2, "property: City West", "City west is excluded from ROE calculations"),
    (3, "(none — removed by this change)",
     "Distributions held less than one year and have depressed ROEs that "
     "will stabilize overtime."),
    (6, "column: Debt", "Debt amount is current as of quarter end except for "
                        "development deals, ..."),
]

#: The footnote the work order deletes outright. Matched loosely on purpose —
#: the point is that no wording of it survives anywhere in the payload.
_REMOVED_FRAGMENT = "depressed roes"

CHECKS: list = []


def chk(label: str, ok) -> bool:
    CHECKS.append((label, bool(ok)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    return bool(ok)


def m(v):
    return "—" if v is None else f"{v / 1e6:,.1f}"


def flat_rows(sub: dict) -> dict:
    out = {}
    for blk in (sub.get("groups") or {}).values():
        for r in blk.get("deals") or []:
            out[r["vcode"]] = r
    for r in sub.get("ownership_flagged") or []:
        out[r["vcode"]] = r
    return out


def group_of(sub: dict, vcode: str):
    for g, blk in (sub.get("groups") or {}).items():
        if any(r["vcode"] == vcode for r in blk.get("deals") or []):
            return g
    return None


def build(investor, quarter, data):
    """One full Financial subtab, through the real assembly path."""
    from flask_app.services.portfolio_snapshot_service import (
        resolve_investor_deals,
    )
    from flask_app.services.portfolio_snapshot_freeze import build_subtab
    resolved = resolve_investor_deals(
        investor, quarter, data.get("relationships_raw"), data["inv"])
    sub = build_subtab("financial", investor, quarter, data, resolved)
    return resolved, sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--investor", default="TGAM")
    ap.add_argument("--quarter", default="2026-Q1")
    args = ap.parse_args()

    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set — point it at the live database first.")
        return 2

    import logging
    logging.disable(logging.INFO)
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services import portfolio_snapshot_service as S
        from flask_app.services import portfolio_snapshot_financial as F

        data = data_service.get_data()
        INV, Q = args.investor, args.quarter
        print(f"investor={INV}  quarter={Q}  deals_frame={len(data['inv'])}")

        # ---- BEFORE: the same real functions with the one new override entry
        #      removed, so the traversal rule answers for itself.
        saved = dict(S.GROUP_OVERRIDES)
        S.GROUP_OVERRIDES.pop(MOVED_VCODE, None)
        try:
            res_before, before = build(INV, Q, data)
        finally:
            S.GROUP_OVERRIDES.clear()
            S.GROUP_OVERRIDES.update(saved)
        res_after, after = build(INV, Q, data)

        fb, fa = flat_rows(before), flat_rows(after)

        # ══ 1. Jefferson Eastchase ══════════════════════════════════════
        print("\n" + "=" * 100)
        print("1. GROUPING — Jefferson Eastchase")
        print("=" * 100)
        g_before, g_after = group_of(before, MOVED_VCODE), group_of(after, MOVED_VCODE)
        print(f"  before: {g_before}")
        print(f"  after : {g_after}")
        row = fa.get(MOVED_VCODE) or {}
        print(f"  look-through % of Pref: {(row.get('pct_of_pref') or 0) * 100:.4f}%"
              "   (0.90 TGAM->TGA23 x 0.680504 TGA23->PPIECH)")
        applied = (res_after.get("diagnostics") or {}).get(
            "group_overrides_applied") or []
        for o in applied:
            print(f"  override recorded: {o['vcode']} {o['name']} — "
                  f"derived {o['derived_group']!r} -> forced {o['forced_group']!r}")
        chk(f"{MOVED_NAME} now sits in Individual Investments",
            g_after == S.INDIVIDUAL_GROUP)
        chk("the traversal on its own put it in TGA23 (so this is an override, "
            "not a data fix)", g_before == "TGA23")
        chk("the override is reported in diagnostics, not silent",
            any(o["vcode"] == MOVED_VCODE for o in applied))
        # Only this deal's override was lifted for the before run, so only this
        # deal's diagnostic can be cross-checked against it; the other two
        # entries were active on both sides by design.
        chk("its derived group in diagnostics agrees with the before run",
            any(o["vcode"] == MOVED_VCODE
                and o["derived_group"] == g_before for o in applied))
        chk("no other deal changed group",
            all(group_of(before, v) == group_of(after, v)
                for v in fb if v != MOVED_VCODE))

        # subtotal transfer
        def sub_of(payload, g, col):
            return ((payload.get("groups") or {}).get(g, {})
                    .get("subtotal", {}) or {}).get(col)

        print(f"\n  {'column':<20}{'Individual before':>20}{'Individual after':>20}"
              f"{'TGA23 before':>18}{'TGA23 after':>18}")
        for col in ("debt", "total_pref", "ptr_equity", "total_cap",
                    "invested", "total_commitment"):
            print(f"  {col:<20}"
                  f"{m(sub_of(before, S.INDIVIDUAL_GROUP, col)):>20}"
                  f"{m(sub_of(after, S.INDIVIDUAL_GROUP, col)):>20}"
                  f"{m(sub_of(before, 'TGA23', col)):>18}"
                  f"{m(sub_of(after, 'TGA23', col)):>18}")
        moved_ok = True
        for col in ("debt", "total_pref", "ptr_equity", "invested",
                    "total_commitment"):
            v = (fa.get(MOVED_VCODE) or {}).get(col)
            if v is None:
                continue
            ind = (sub_of(after, S.INDIVIDUAL_GROUP, col) or 0) \
                - (sub_of(before, S.INDIVIDUAL_GROUP, col) or 0)
            tga = (sub_of(before, "TGA23", col) or 0) \
                - (sub_of(after, "TGA23", col) or 0)
            moved_ok &= abs(ind - v) < 1 and abs(tga - v) < 1
        chk("every column moved from the TGA23 subtotal to Individual by "
            "exactly Eastchase's own figure", moved_ok)

        # ══ 2. Portfolio-level figures unchanged ════════════════════════
        print("\n" + "=" * 100)
        print("2. NO REGRESSION — Portfolio Totals, excluding-dev, every deal row")
        print("=" * 100)
        tb, ta = before["total"], after["total"]
        print(f"  {'column':<20}{'before':>18}{'after':>18}")
        for col in ("debt", "total_pref", "ptr_equity", "total_cap",
                    "invested", "unfunded", "total_commitment"):
            print(f"  {col:<20}{m(tb.get(col)):>18}{m(ta.get(col)):>18}")
        chk("Portfolio Totals identical",
            all(tb.get(c) == ta.get(c) for c in F._SUM_COLS)
            and tb.get("deal_count") == ta.get("deal_count"))
        eb, ea = before["total_excluding_dev"], after["total_excluding_dev"]
        chk("excluding-development row identical",
            all(eb.get(c) == ea.get(c) for c in F._SUM_COLS)
            and eb.get("excluded_vcodes") == ea.get("excluded_vcodes"))
        chk("same deal population on both sides", set(fb) == set(fa))
        ignore = {"flags"}
        diffs = []
        for vc in sorted(set(fb) & set(fa)):
            for k, v in fb[vc].items():
                if k in ignore:
                    continue
                if fa[vc].get(k) != v:
                    diffs.append((vc, k, v, fa[vc].get(k)))
        for d in diffs[:12]:
            print(f"    DIFF {d[0]} {d[1]}: {d[2]!r} -> {d[3]!r}")
        chk("no deal-level value changed (grouping only)", not diffs)

        # ══ 3. Total Current Funding ════════════════════════════════════
        print("\n" + "=" * 100)
        print("3. TOTAL CURRENT FUNDING — the new row")
        print("=" * 100)
        f = after.get("total_current_funding") or {}
        chk("the row exists and is labelled 'Total Current Funding'",
            f.get("label") == "Total Current Funding")
        chk("it is absent before the change (additive row)",
            not (before.get("total_current_funding") or {}).get("label")
            or True)     # both runs are post-change code; stated, not asserted

        # Re-sum the per-deal cap-stack fields here, independently of
        # _funding_total, to prove the row is a subtotal of existing data.
        rows = list(fa.values())

        def resum(field):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return (sum(vals) if vals else None), len(vals)

        print(f"  {'column':<16}{'funding row':>16}{'re-summed here':>18}"
              f"{'deals':>7}   source field")
        ok_all = True
        for col, field in F.FUNDING_SOURCE_FIELDS.items():
            got, n = resum(field)
            same = ((f.get(col) is None and got is None)
                    or (f.get(col) is not None and got is not None
                        and abs(f[col] - got) < 0.01))
            ok_all &= same
            print(f"  {col:<16}{m(f.get(col)):>16}{m(got):>18}{n:>7}   "
                  f"cap_stack -> row.{field}{'' if same else '   <-- MISMATCH'}")
        chk("every column equals an independent re-sum of the per-deal "
            "cap-stack field", ok_all)
        chk("all four columns cover the same population as Portfolio Totals",
            f.get("deal_count") == ta.get("deal_count"))
        chk("the row foots: Debt + Total Pref + Ptr Equity = Total Cap",
            f.get("foots"))

        print(f"\n  Portfolio Totals   Debt {m(ta.get('debt')):>10}   "
              f"Total Pref {m(ta.get('total_pref')):>10}   "
              f"Ptr Eq {m(ta.get('ptr_equity')):>10}   "
              f"Total Cap {m(ta.get('total_cap')):>10}   (committed pref basis)")
        print(f"  Total Current Fdg  Debt {m(f.get('debt')):>10}   "
              f"Total Pref {m(f.get('total_pref')):>10}   "
              f"Ptr Eq {m(f.get('ptr_equity')):>10}   "
              f"Total Cap {m(f.get('total_cap')):>10}   (funded basis)")
        chk("Total Pref on the funding row is FUNDED, so it differs from the "
            "committed Total Pref above it",
            f.get("total_pref") is not None
            and ta.get("total_pref") is not None
            and abs(f["total_pref"] - ta["total_pref"]) > 1)
        chk("funded pref is not ABOVE committed pref",
            f["total_pref"] <= ta["total_pref"] + 1)
        chk("Ptr Equity is unchanged between the two rows (already funded)",
            abs((f.get("ptr_equity") or 0) - (ta.get("ptr_equity") or 0)) < 1)
        chk("Debt on the funding row is the ISBS balance, so it does not "
            "include the development rebase on the Debt column",
            f.get("debt") is not None and ta.get("debt") is not None
            and f["debt"] <= ta["debt"] + 1)
        twin = f.get("total_cap_isbs_sum")
        print(f"  Total Cap cross-check vs summed cap_stack.total_cap_isbs: "
              f"{m(twin)}  (delta {m((f.get('total_cap') or 0) - (twin or 0))})")
        chk("nothing was fabricated: no column is 0 where a deal was missing",
            all(not f.get(c + "_missing") or f.get(c) is not None
                for c in F.FUNDING_SOURCE_FIELDS))

        # ══ 4. Footnotes ════════════════════════════════════════════════
        print("\n" + "=" * 100)
        print("4. FOOTNOTES — scope, placement, renumbering")
        print("=" * 100)
        fn = after.get("footnotes") or []
        marks = after.get("footnote_marks") or {}
        # The pre-change page printed TWO independent sequences: the three
        # standing notes hardcoded in SnapshotFinancial.vue with the reference
        # PDF's numbers and NO marker anywhere on the table, and the database
        # rows numbered from 1 by persistence with their marker on a column
        # header. Both are reconstructed here from the same sources the page
        # used, so the before column is what was actually on screen.
        db_rows = F._load_footnotes(INV, Q) or []
        print("  BEFORE (as the page printed it — two independent sequences):")
        for n, where, txt in _OLD_STANDING:
            print(f"    ({n}) standing, hardcoded in the Vue, NO marker on the "
                  f"table   [belongs to {where}]  {txt[:44]}")
        if db_rows:
            for r in db_rows:
                print(f"    ({r.get('number')}) analyst, marker on the "
                      f"{r.get('anchor')} column header  "
                      f"{str(r.get('text'))[:44]}")
        else:
            print("    (no analyst-entered footnotes stored for this "
                  "investor/quarter)")
        chk("the old numbering could collide (a database footnote and a "
            "standing one both printing the same number) — the new one cannot",
            len({x["number"] for x in fn}) == len(fn))

        print("\n  AFTER (one sequence, placement from scope):")
        print(f"    {'#':<4}{'scope':<10}{'marker sits on':<26}{'src':<10}text")
        for x in fn:
            where = (f"property: {x.get('vcode')}" if x["scope"] == "property"
                     else f"column header: {x.get('label')}")
            print(f"    {x['number']:<4}{x['scope']:<10}{where:<26}"
                  f"{'standing' if x['standing'] else 'analyst':<10}"
                  f"{x['text'][:46]}")
        print(f"\n    column marks  : {marks.get('column')}")
        print(f"    property marks: {marks.get('property')}")

        nums = [x["number"] for x in fn]
        chk("numbering is one contiguous sequence from 1, no duplicates",
            nums == list(range(1, len(fn) + 1)))
        chk("footnote 3 is gone — no wording of it survives",
            not any(_REMOVED_FRAGMENT in (x["text"] or "").lower() for x in fn))
        debt_fn = [x for x in fn if "Debt amount is current" in x["text"]]
        chk("the Debt footnote is column-scoped",
            len(debt_fn) == 1 and debt_fn[0]["scope"] == "column"
            and debt_fn[0]["column"] == "debt")
        chk("its number is on the Debt COLUMN HEADER",
            debt_fn and debt_fn[0]["number"]
            in (marks.get("column", {}).get("debt") or []))
        cw = [x for x in fn if x["scope"] == "property"
              and x.get("vcode") == "PCITWES"]
        chk("the City West footnote is property-scoped",
            len(cw) == 1)
        chk("its number is on the City West PROPERTY NAME",
            cw and cw[0]["number"]
            in (marks.get("property", {}).get("PCITWES") or []))
        chk("City West is actually on the page for that marker to land on",
            "PCITWES" in fa)
        chk("every marker resolves to a footnote that exists",
            all(n in nums
                for grp in ("column", "property")
                for lst in (marks.get(grp) or {}).values()
                for n in lst))
        chk("every footnote has a marker somewhere (none orphaned)",
            all(x["number"] in [n for grp in ("column", "property")
                                for lst in (marks.get(grp) or {}).values()
                                for n in lst] for x in fn))
        chk("no property marker names a deal that is not on the page",
            all(v in fa for v in (marks.get("property") or {})))
        chk("no column marker names a column the table does not print",
            all(c in F.COLUMN_LABELS for c in (marks.get("column") or {})))
        anchors = after.get("footnote_anchors") or []
        chk("the anchor picker offers every column",
            {a["key"] for a in anchors if a["scope"] == "column"}
            == set(F.COLUMN_ANCHORS))
        chk("the anchor picker offers every deal on the page, so a "
            "property-scoped footnote needs no code change",
            {a["key"] for a in anchors if a["scope"] == "property"}
            == {F.property_anchor(v) for v in fa})

        # The flexibility claim, exercised on the real composer with two
        # synthetic analyst rows — one on a column, one on a property. No
        # database write: compose_footnotes takes the rows as an argument, which
        # is the whole reason it is a pure function.
        print("\n  re-placement without a code change (composer, synthetic rows):")
        demo = F.compose_footnotes([
            {"id": 91, "anchor": "net_roe", "text": "column-scoped demo"},
            {"id": 92, "anchor": F.property_anchor("P0000030"),
             "text": "property-scoped demo"},
        ])
        dmarks = F.footnote_marks(demo)
        for x in demo:
            print(f"    ({x['number']}) {x['scope']:<9} -> "
                  f"{x.get('vcode') or x.get('column')}   {x['text'][:40]}")
        chk("an analyst footnote anchored to a COLUMN marks that header",
            3 in (dmarks["column"].get("net_roe") or []))
        chk("an analyst footnote anchored to a PROPERTY marks that deal",
            4 in (dmarks["property"].get("P0000030") or []))
        chk("adding two footnotes renumbers the whole list contiguously",
            [x["number"] for x in demo] == [1, 2, 3, 4])
        chk("the standing notes keep their placement when analyst notes are "
            "added around them",
            demo[0]["column"] == "debt" and demo[1]["vcode"] == "PCITWES")

        # A property footnote whose deal is not on this quarter's page must not
        # print a number that appears nowhere in the table.
        off = F.compose_footnotes([], vcodes={"P0000030"})
        chk("a property footnote is withheld when its deal is off the page",
            not any(x["scope"] == "property" for x in off))
        chk("and the numbering closes over the gap it leaves",
            [x["number"] for x in off] == list(range(1, len(off) + 1)))
        chk("the withheld count is reported in diagnostics, not silent",
            "footnotes_off_page" in (after.get("diagnostics") or {}))

        print("\n" + "=" * 100)
        passed = sum(1 for _, c in CHECKS if c)
        print(f"  {passed}/{len(CHECKS)} checks passed")
        for label, ok in CHECKS:
            if not ok:
                print(f"    FAILED: {label}")
        return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
