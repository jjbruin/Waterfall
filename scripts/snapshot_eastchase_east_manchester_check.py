"""Guardrail: the Sep 2 2026 Financial-tab work order, before vs after.

Four changes, verified against LIVE data on both affected quarters:

  1. Jefferson Eastchase (P0000085) goes back to TGA 2023. Its GROUP_OVERRIDES
     entry was added Sep 1 on a work order that meant East MANCHESTER; the
     override is withdrawn, so it groups by the ordinary rule again.
  2. East Manchester (P0000017) is added to KEEP_DESPITE_SOLD, so it stays on
     the report at 26Q2 instead of being dropped by the sold gate (sold
     6/25/2026), in Individual Investments where GROUP_OVERRIDES already put it.
  3. The ROE-exclusion footnote covers BOTH deals, as ONE note with ONE number
     marking BOTH property names.
  4. Both deals print "(Sold)" after the name, on screen and in print.

Plus the thing the work order did NOT ask for and this script exists to police:
East Manchester's 26Q2 Debt is a STALE 9,641,912 — the loan left with the asset
— so blanking the cell without also taking it out of the totals underneath would
just move the misleading figure one row down. The n/a is therefore excluded from
the Debt column, the Total Current Funding Debt leg and the row's own Total Cap.
That exclusion must be a measured NO-OP everywhere else, and these checks are
what measure it.

WHY BEFORE/AFTER AND NOT A FIXTURE: the whole change is about which deals land
in which group and which cells apply, which only the real ownership feed and
the real cap stacks can answer. `capture` runs the committed code on one side,
`report` diffs the two.

    set WF_TOKEN=<jwt>
    git worktree add ../wf-before origin/main
    python scripts/snapshot_eastchase_east_manchester_check.py capture before ../wf-before
    python scripts/snapshot_eastchase_east_manchester_check.py capture after  .
    python scripts/snapshot_eastchase_east_manchester_check.py report

`capture` imports the modules from the tree it is pointed at, so the "before"
column is the code as committed on main, not a re-implementation of it.
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(tempfile.gettempdir(), "snap_em_check")

INV = "TGAM"
QUARTERS = ("2026-Q1", "2026-Q2")

EASTCHASE = "P0000085"
EAST_MANCH = "P0000017"
CITY_WEST = "PCITWES"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

#: Row fields this change ADDS. They are absent on the before side by
#: construction, so a naive dict comparison would report all 35 rows as
#: "changed". Compared separately, and against the old fields they derive from,
#: by ``_new_fields_consistent`` — an additive field still has to be right.
NEW_ROW_FIELDS = ("debt_summable", "debt_isbs_summable", "sold_label")

CHECKS: list = []


def _old(row: dict) -> dict:
    """One row with the newly-added fields removed, for a like-for-like diff."""
    return {k: v for k, v in (row or {}).items() if k not in NEW_ROW_FIELDS}


def chk(label, ok):
    CHECKS.append((label, bool(ok)))
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    return bool(ok)


def m(v):
    return "—" if v is None else f"{v / 1e6:,.2f}M"


# ══════════════════════════════════════════════════════════════════════════
# capture — run the committed code in one tree and dump its payload
# ══════════════════════════════════════════════════════════════════════════

def capture(side: str, tree: str):
    tree = os.path.abspath(tree)
    # The tree under test goes FIRST on the path, so `import
    # flask_app.services...` resolves there and not in whichever tree this
    # script happens to live in.
    for p in (tree, os.path.join(tree, "scripts")):
        sys.path.insert(0, p)
    import pandas as pd
    import live_api as api
    from flask_app.services.portfolio_snapshot_service import (
        resolve_investor_deals, KEEP_DESPITE_SOLD, GROUP_OVERRIDES,
    )
    from flask_app.services import portfolio_snapshot_financial as F

    loaded = os.path.dirname(os.path.abspath(F.__file__))
    print(f"  {side}: assembly loaded from {loaded}")
    if os.path.abspath(tree) not in os.path.abspath(loaded):
        print("  ABORT — the module came from a different tree than requested")
        return 2

    ti = api.token_info()
    print(f"  token={ti['username']} ({ti['hours_left']}h)  "
          f"build={api.get('/api/data/version').get('version')}")

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])

    # Ownership closure, exactly as the module's own self-test walks it.
    seen, frontier, rows = set(), [INV], []

    def fetch(col, v):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": v})
        # The live rows endpoint matches case-insensitively and loosely, so
        # every row is re-checked here — see the note in MEMORY.md.
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == v.upper()]

    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            c = str(r.get("InvestmentID") or "").strip().upper()
            if c:
                rows.extend(fetch("InvestmentID", c))
                if c not in seen:
                    frontier.append(c)
    rel = pd.DataFrame(rows).drop_duplicates()

    cap_cache: dict = {}

    def provider(vcode, quarter):
        key = (vcode, quarter)
        if key not in cap_cache:
            cap_cache[key] = api.get(
                f"/api/financials/{vcode}/one-pager", params={"quarter": quarter})
        return cap_cache[key]

    loans = pd.DataFrame(
        api.get("/api/data/tables/loans/rows",
                params={"page": 1, "page_size": 5000}).get("rows") or [])

    def committed_of(vcode):
        """Committed facility, the same shape build_subtab wires in."""
        if loans.empty:
            return None
        col_v = next((c for c in loans.columns if c.lower() == "vcode"), None)
        col_a = next((c for c in loans.columns
                      if c.lower() == "morigloanamt"), None)
        if not col_v or not col_a:
            return None
        sub = loans[loans[col_v].astype(str).str.strip().str.upper()
                    == str(vcode).strip().upper()]
        if sub.empty:
            return None
        tot = pd.to_numeric(sub[col_a], errors="coerce").fillna(0).sum()
        return float(tot) or None

    snap: dict = {"side": side, "tree": tree,
                  "keep_despite_sold": sorted(KEEP_DESPITE_SOLD),
                  "group_overrides": dict(GROUP_OVERRIDES),
                  "quarters": {}}

    for Q in QUARTERS:
        resolved = resolve_investor_deals(INV, Q, rel, inv)
        out = F.assemble_financial(
            INV, Q, resolved=resolved, one_pager_provider=provider,
            committed_debt_provider=committed_of,
            manual_loader=lambda *_a, **_k: {},
            footnote_loader=lambda *_a, **_k: [])
        rows_by_vc = {}
        for g, blk in (out.get("groups") or {}).items():
            for r in blk["deals"]:
                rows_by_vc[r["vcode"]] = {
                    "group": g, "name": r["name"],
                    "debt": r.get("debt"), "debt_display": r.get("debt_display"),
                    "debt_summable": r.get("debt_summable"),
                    "debt_isbs": r.get("debt_isbs"),
                    "debt_isbs_summable": r.get("debt_isbs_summable"),
                    "total_pref": r.get("total_pref"),
                    "ptr_equity": r.get("ptr_equity"),
                    "total_cap": r.get("total_cap"),
                    "pct_of_pref": r.get("pct_of_pref"),
                    "invested": r.get("invested"),
                    "total_commitment": r.get("total_commitment"),
                    "unfunded": r.get("unfunded"),
                    "net_roe_display": r.get("net_roe_display"),
                    "itd_display": r.get("itd_display"),
                    "pdf_na_cells": r.get("pdf_na_cells"),
                    "kept_despite_sold": r.get("kept_despite_sold"),
                    "sold_label": r.get("sold_label"),
                    "is_dev": r.get("is_dev"),
                }
        snap["quarters"][Q] = {
            "rows": rows_by_vc,
            "members": {g: [r["vcode"] for r in blk["deals"]]
                        for g, blk in (out.get("groups") or {}).items()},
            "subtotals": {g: {k: blk["subtotal"].get(k)
                              for k in ("label", "deal_count", "debt",
                                        "total_pref", "ptr_equity", "total_cap",
                                        "pct_of_pref", "invested",
                                        "total_commitment", "unfunded")}
                          for g, blk in (out.get("groups") or {}).items()},
            "total": {k: out["total"].get(k)
                      for k in ("deal_count", "debt", "total_pref",
                                "ptr_equity", "total_cap", "invested",
                                "total_commitment", "unfunded")},
            "funding": {k: (out.get("total_current_funding") or {}).get(k)
                        for k in ("label", "deal_count", "debt", "total_pref",
                                  "ptr_equity", "total_cap", "foots")},
            "excluding_dev": {
                k: (out.get("total_excluding_dev") or {}).get(k)
                for k in ("label", "deal_count", "excluded_vcodes",
                          "total_commitment", "itd", "net_roe")},
            "footnotes": [
                {"number": f["number"], "text": f["text"],
                 "scope": f.get("scope"), "anchor": f.get("anchor"),
                 "anchors": f.get("anchors"), "label": f.get("label")}
                for f in (out.get("footnotes") or [])],
            "footnote_marks": out.get("footnote_marks"),
            "diagnostics": out.get("diagnostics"),
        }
        r_ec = rows_by_vc.get(EASTCHASE) or {}
        r_em = rows_by_vc.get(EAST_MANCH) or {}
        print(f"    {Q}: {out['total']['deal_count']} deals   "
              f"Eastchase->{r_ec.get('group', 'ABSENT')}   "
              f"East Manchester->{r_em.get('group', 'ABSENT')}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{side}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1, default=str)
    print(f"  wrote {path}")
    return 0


# ══════════════════════════════════════════════════════════════════════════
# report — diff the two captures and assert the four changes
# ══════════════════════════════════════════════════════════════════════════

def _row_line(side, q, vc, snap):
    r = ((snap.get("quarters") or {}).get(q, {}).get("rows") or {}).get(vc)
    if not r:
        return f"    {side:<7} {q}  ABSENT FROM THE REPORT"
    return (f"    {side:<7} {q}  {r['group']:<24}"
            f"debt {str(r['debt_display'])[:12]:>13}  "
            f"pref {m(r['total_pref']):>10}  ptr {m(r['ptr_equity']):>9}  "
            f"cap {m(r['total_cap']):>10}  "
            f"roe {str(r['net_roe_display'])[:13]:>14}  "
            f"sold={r.get('sold_label') or '-'}")


def report():
    with open(os.path.join(OUT, "before.json"), encoding="utf-8") as fh:
        b = json.load(fh)
    with open(os.path.join(OUT, "after.json"), encoding="utf-8") as fh:
        a = json.load(fh)

    print("=" * 108)
    print("CONSTANTS")
    print("=" * 108)
    print(f"  KEEP_DESPITE_SOLD  before {b['keep_despite_sold']}")
    print(f"                     after  {a['keep_despite_sold']}")
    print(f"  GROUP_OVERRIDES    before {b['group_overrides']}")
    print(f"                     after  {a['group_overrides']}")

    for vc, nm in ((EASTCHASE, "JEFFERSON EASTCHASE"),
                   (EAST_MANCH, "EAST MANCHESTER"),
                   (CITY_WEST, "CITY WEST")):
        print("\n" + "=" * 108)
        print(f"{nm}  ({vc})")
        print("=" * 108)
        for q in QUARTERS:
            print(_row_line("before", q, vc, b))
            print(_row_line("after ", q, vc, a))

    print("\n" + "=" * 108)
    print("FOOTNOTES — after")
    print("=" * 108)
    for q in QUARTERS:
        print(f"  {q}")
        for f in a["quarters"][q]["footnotes"]:
            print(f"    ({f['number']}) {f['scope']:<9} {str(f['label'])[:26]:<27}"
                  f"{f['text'][:56]}")
        print(f"    marks: {json.dumps(a['quarters'][q]['footnote_marks'])}")

    print("\n" + "=" * 108)
    print("SUBTOTALS — before vs after")
    print("=" * 108)
    for q in QUARTERS:
        print(f"\n  {q}")
        keys = sorted(set(b["quarters"][q]["subtotals"])
                      | set(a["quarters"][q]["subtotals"]))
        print(f"    {'group':<24}{'n':>4}{'debt':>13}{'pref':>13}"
              f"{'ptr':>12}{'cap':>13}   (before / after)")
        for g in keys:
            sb = b["quarters"][q]["subtotals"].get(g) or {}
            sa = a["quarters"][q]["subtotals"].get(g) or {}
            for tag, s in (("before", sb), ("after ", sa)):
                if not s:
                    print(f"    {g[:23]:<24}{tag:>4}  (absent)")
                    continue
                print(f"    {(g[:23] if tag == 'before' else ''):<24}"
                      f"{s.get('deal_count', 0):>4}{m(s.get('debt')):>13}"
                      f"{m(s.get('total_pref')):>13}{m(s.get('ptr_equity')):>12}"
                      f"{m(s.get('total_cap')):>13}   {tag}")
        for tag, t in (("before", b["quarters"][q]["total"]),
                       ("after ", a["quarters"][q]["total"])):
            print(f"    {'PORTFOLIO TOTALS':<24}{t['deal_count']:>4}"
                  f"{m(t['debt']):>13}{m(t['total_pref']):>13}"
                  f"{m(t['ptr_equity']):>12}{m(t['total_cap']):>13}   {tag}")
        for tag, t in (("before", b["quarters"][q]["funding"]),
                       ("after ", a["quarters"][q]["funding"])):
            print(f"    {'Total Current Funding':<24}{t['deal_count']:>4}"
                  f"{m(t['debt']):>13}{m(t['total_pref']):>13}"
                  f"{m(t['ptr_equity']):>12}{m(t['total_cap']):>13}   {tag}"
                  f"  foots={t['foots']}")

    # ══ 1. Jefferson Eastchase ══════════════════════════════════════════
    print("\n" + "=" * 108)
    print("1. JEFFERSON EASTCHASE IS BACK IN TGA 2023")
    print("=" * 108)
    chk("its GROUP_OVERRIDES entry is gone",
        EASTCHASE in b["group_overrides"]
        and EASTCHASE not in a["group_overrides"])
    for q in QUARTERS:
        rb = b["quarters"][q]["rows"].get(EASTCHASE) or {}
        ra = a["quarters"][q]["rows"].get(EASTCHASE) or {}
        chk(f"{q}: was Individual Investments, is now TGA23",
            rb.get("group") == "Individual Investments"
            and ra.get("group") == "TGA23")
        chk(f"{q}: every figure on the row is unchanged — only the group moved",
            all(rb.get(k) == ra.get(k) for k in
                ("debt", "total_pref", "ptr_equity", "total_cap",
                 "pct_of_pref", "invested", "total_commitment", "unfunded",
                 "is_dev", "net_roe_display", "itd_display")))
        chk(f"{q}: still counted as development (dev tag unchanged)",
            ra.get("is_dev") is True)
        exb = b["quarters"][q]["excluding_dev"]
        exa = a["quarters"][q]["excluding_dev"]
        chk(f"{q}: still removed by the excluding-development row "
            f"(that row is population-scoped, not group-scoped)",
            EASTCHASE in (exb.get("excluded_vcodes") or [])
            and EASTCHASE in (exa.get("excluded_vcodes") or []))
        chk(f"{q}: the excluding-development total did not move",
            exb.get("total_commitment") == exa.get("total_commitment"))

    # ══ 2. East Manchester ══════════════════════════════════════════════
    print("\n" + "=" * 108)
    print("2. EAST MANCHESTER IS BACK ON THE 26Q2 REPORT")
    print("=" * 108)
    chk("it is in KEEP_DESPITE_SOLD, alongside City West",
        EAST_MANCH not in b["keep_despite_sold"]
        and set(a["keep_despite_sold"]) == {CITY_WEST, EAST_MANCH})
    b2 = b["quarters"]["2026-Q2"]["rows"].get(EAST_MANCH)
    a2 = a["quarters"]["2026-Q2"]["rows"].get(EAST_MANCH)
    chk("26Q2: absent before, present after", not b2 and bool(a2))
    chk("26Q2: in Individual Investments, like City West",
        (a2 or {}).get("group") == "Individual Investments")
    chk("26Q2: flagged kept_despite_sold, so it is a KEPT deal and not one "
        "the sold gate merely failed to catch",
        (a2 or {}).get("kept_despite_sold") is True)

    print("\n  the cells, which are NOT a City West clone:")
    print(f"    Debt        raw {m((a2 or {}).get('debt'))}  "
          f"prints {(a2 or {}).get('debt_display')!r}   "
          f"summable {m((a2 or {}).get('debt_summable'))}")
    print(f"    Total Pref  {m((a2 or {}).get('total_pref'))}"
          f"   (capital returned at sale — a real zero)")
    print(f"    Ptr Equity  {m((a2 or {}).get('ptr_equity'))}"
          f"   (the residual this row exists to report)")
    print(f"    Total Cap   {m((a2 or {}).get('total_cap'))}"
          f"   (= pref + ptr; the n/a debt leg is left out)")
    print(f"    Net ROE     {(a2 or {}).get('net_roe_display')!r}")
    print(f"    ITD         {(a2 or {}).get('itd_display')!r}")

    chk("26Q2: Debt reads n/a, not a misleading $9.6M on a sold asset",
        (a2 or {}).get("debt_display") == "n/a")
    chk("26Q2: the raw 9,641,912 is untouched underneath the n/a",
        abs(((a2 or {}).get("debt") or 0) - 9_641_912) < 1)
    chk("26Q2: and it is OUT of the Debt total — an n/a cell is not summed",
        (a2 or {}).get("debt_summable") is None
        and (a2 or {}).get("debt_isbs_summable") is None)
    chk("26Q2: Net ROE reads n/a — it is excluded from ROE per the footnote",
        (a2 or {}).get("net_roe_display") == "n/a")
    chk("26Q2: ITD still prompts — a final distribution figure is real data",
        (a2 or {}).get("itd_display") == "pending entry")
    chk("26Q2: the row foots against what it PRINTS "
        "(Total Cap = Total Pref + Ptr Equity)",
        abs(((a2 or {}).get("total_cap") or 0)
            - (((a2 or {}).get("total_pref") or 0)
               + ((a2 or {}).get("ptr_equity") or 0))) < 1)
    chk("26Q2: Ptr Equity 2.40M is reported as the real residual, not blanked",
        abs(((a2 or {}).get("ptr_equity") or 0) - 2_400_000) < 1)

    print("\n  26Q1 — the quarter it was still HELD, which must not move:")
    b1 = b["quarters"]["2026-Q1"]["rows"].get(EAST_MANCH) or {}
    a1 = a["quarters"]["2026-Q1"]["rows"].get(EAST_MANCH) or {}
    print(_row_line("before", "2026-Q1", EAST_MANCH, b))
    print(_row_line("after ", "2026-Q1", EAST_MANCH, a))
    chk("26Q1: the row is unchanged before and after "
        "(bar the added summable/label fields)",
        _old(b1) == _old(a1) and bool(a1))
    chk("26Q1: its Debt is still IN the totals — the cell applies, so it sums",
        a1.get("debt_summable") == a1.get("debt")
        and a1.get("debt_isbs_summable") == a1.get("debt_isbs"))
    chk("26Q1: its real 9,641,912 debt still PRINTS — the n/a is keyed on the "
        "sale, not on the vcode, so a held quarter keeps its live figure",
        a1.get("debt_display") == a1.get("debt")
        and abs((a1.get("debt") or 0) - 9_641_912) < 1)
    chk("26Q1: not labelled sold, and still prompting for Net ROE",
        a1.get("sold_label") is None
        and a1.get("net_roe_display") == "pending entry")

    # ══ 3. Footnote 2 ═══════════════════════════════════════════════════
    print("\n" + "=" * 108)
    print("3. THE ROE-EXCLUSION FOOTNOTE COVERS BOTH DEALS")
    print("=" * 108)
    for q in QUARTERS:
        fb = next((f for f in b["quarters"][q]["footnotes"]
                   if "excluded from ROE" in f["text"]), None)
        fa = next((f for f in a["quarters"][q]["footnotes"]
                   if "excluded from ROE" in f["text"]), None)
        print(f"  {q}")
        print(f"    before  ({(fb or {}).get('number')}) {(fb or {}).get('text')}")
        print(f"    after   ({(fa or {}).get('number')}) {(fa or {}).get('text')}")
        chk(f"{q}: it names City West AND East Manchester",
            bool(fa) and "City West" in fa["text"]
            and "East Manchester" in fa["text"])
        chk(f"{q}: it is ONE footnote, not two identical ones",
            sum(1 for f in a["quarters"][q]["footnotes"]
                if "excluded from ROE" in f["text"]) == 1)
        marks = (a["quarters"][q]["footnote_marks"] or {}).get("property") or {}
        chk(f"{q}: its number is on BOTH property names",
            marks.get(CITY_WEST) == [fa["number"]]
            and marks.get(EAST_MANCH) == [fa["number"]])
        chk(f"{q}: it is still the SECOND note, after the Debt column note",
            (fa or {}).get("number") == 2)
        chk(f"{q}: no marker is placed on a deal that is not on the page",
            all(vc in a["quarters"][q]["rows"] for vc in marks))

    # ══ 4. "(Sold)" ═════════════════════════════════════════════════════
    print("\n" + "=" * 108)
    print('4. "(Sold)" ON BOTH DEALS')
    print("=" * 108)
    for q in QUARTERS:
        rows = a["quarters"][q]["rows"]
        labelled = sorted(vc for vc, r in rows.items() if r.get("sold_label"))
        print(f"  {q}: labelled (Sold) -> {labelled}")
        expect = ([CITY_WEST] if q == "2026-Q1"
                  else sorted([CITY_WEST, EAST_MANCH]))
        chk(f"{q}: exactly the kept-despite-sold deals carry the label",
            labelled == expect)
        chk(f"{q}: the label text is '(Sold)'",
            all(rows[vc]["sold_label"] == "(Sold)" for vc in labelled))
        chk(f"{q}: no held deal is labelled sold",
            all(r.get("sold_label") is None for vc, r in rows.items()
                if not r.get("kept_despite_sold")))
    # Print survival is a static CSS question, checked by its own guardrail.
    mk = os.path.join(ROOT, "scripts", "snapshot_print_markers_check.py")
    rc = subprocess.run([sys.executable, mk], capture_output=True, text=True)
    chk("the print view does NOT hide it (snapshot_print_markers_check.py)",
        rc.returncode == 0)
    for line in rc.stdout.splitlines():
        if ".sold" in line or "(Sold)" in line:
            print(f"      {line.strip()}")

    # ══ 5. Nothing else moved ═══════════════════════════════════════════
    print("\n" + "=" * 108)
    print("5. NOTHING ELSE MOVED")
    print("=" * 108)
    touched = {EASTCHASE, EAST_MANCH}
    for q in QUARTERS:
        rb, ra = b["quarters"][q]["rows"], a["quarters"][q]["rows"]
        chk(f"{q}: the population changes by East Manchester alone",
            (set(ra) - set(rb)) <= {EAST_MANCH} and not (set(rb) - set(ra)))
        moved = [vc for vc in set(rb) & set(ra)
                 if rb[vc]["group"] != ra[vc]["group"]]
        chk(f"{q}: exactly one deal changed group, and it is Eastchase",
            moved == [EASTCHASE])
        changed = [vc for vc in set(rb) & set(ra) if vc not in touched
                   and _old(rb[vc]) != _old(ra[vc])]
        chk(f"{q}: every other deal's row is unchanged", not changed)
        for vc in changed:
            print(f"        CHANGED {vc} {ra[vc]['name']}")
            for k in _old(rb[vc]):
                if rb[vc][k] != ra[vc].get(k):
                    print(f"          {k}: {rb[vc][k]!r} -> {ra[vc].get(k)!r}")
        # The added fields are additive, not free: on every deal whose Debt
        # applies they must equal the figure they mirror, or the totals would
        # quietly lose a real balance.
        applies = [r for vc, r in ra.items()
                   if "debt" not in (r.get("pdf_na_cells") or [])]
        chk(f"{q}: on all {len(applies)} deals whose Debt applies, the summable "
            f"twins equal the printed Debt and the ISBS balance",
            all(r["debt_summable"] == r["debt"]
                and r["debt_isbs_summable"] == r["debt_isbs"] for r in applies))
        chk(f"{q}: only kept-despite-sold rows gained a sold label",
            {vc for vc, r in ra.items() if r.get("sold_label")}
            == {vc for vc, r in ra.items() if r.get("kept_despite_sold")})

        tb, ta = b["quarters"][q]["total"], a["quarters"][q]["total"]
        # 26Q1 adds nothing, so every portfolio figure must be identical.
        # 26Q2 gains East Manchester: Ptr Equity +2.40M and Total Cap +2.40M
        # are the residual equity the row exists to report; Debt must NOT move,
        # because its stale balance is the figure the n/a keeps off the page.
        d_pref = (ta["total_pref"] or 0) - (tb["total_pref"] or 0)
        d_ptr = (ta["ptr_equity"] or 0) - (tb["ptr_equity"] or 0)
        d_debt = (ta["debt"] or 0) - (tb["debt"] or 0)
        d_cap = (ta["total_cap"] or 0) - (tb["total_cap"] or 0)
        print(f"    {q} portfolio delta: debt {d_debt:+,.0f}  "
              f"pref {d_pref:+,.0f}  ptr {d_ptr:+,.0f}  cap {d_cap:+,.0f}  "
              f"deals {ta['deal_count'] - tb['deal_count']:+d}")
        chk(f"{q}: Portfolio Totals Debt does not move — "
            f"a stale post-sale balance never enters it",
            abs(d_debt) < 1)
        if q == "2026-Q1":
            chk("26Q1: nothing at all moved — same deals, same every total",
                abs(d_pref) < 1 and abs(d_ptr) < 1 and abs(d_cap) < 1
                and ta["deal_count"] == tb["deal_count"])
        else:
            chk("26Q2: +1 deal, +2.40M Ptr Equity, +2.40M Total Cap, "
                "+0 Total Pref, +0 Debt",
                ta["deal_count"] - tb["deal_count"] == 1
                and abs(d_ptr - 2_400_000) < 1 and abs(d_cap - 2_400_000) < 1
                and abs(d_pref) < 1)

        # The n/a exclusion must be inert on every deal it was not aimed at.
        na_debt = {vc: r for vc, r in ra.items()
                   if "debt" in (r.get("pdf_na_cells") or [])}
        print(f"    {q} n/a Debt cells: "
              + ", ".join(f"{vc} raw {m(r['debt'])}"
                          for vc, r in sorted(na_debt.items())))
        chk(f"{q}: the only n/a Debt cell carrying a NON-ZERO balance is "
            f"East Manchester — so excluding n/a debt from the totals is a "
            f"no-op for every other deal",
            {vc for vc, r in na_debt.items() if abs(r["debt"] or 0) > 1}
            <= {EAST_MANCH})
        chk(f"{q}: every n/a Debt row foots against what it prints",
            all(abs((r["total_cap"] or 0)
                    - ((r["total_pref"] or 0) + (r["ptr_equity"] or 0))) < 1
                for r in na_debt.values()))
        fu = a["quarters"][q]["funding"]
        chk(f"{q}: Total Current Funding still foots", fu["foots"] is True)

        # Subtotals: only the two blocks the two deals sit in may move.
        sb, sa_ = b["quarters"][q]["subtotals"], a["quarters"][q]["subtotals"]
        allowed = {"Individual Investments", "TGA23"}
        drifted = [g for g in set(sb) & set(sa_)
                   if g not in allowed and sb[g] != sa_[g]]
        chk(f"{q}: no group subtotal outside "
            f"{{Individual Investments, TGA23}} moved", not drifted)
        for g in drifted:
            print(f"        DRIFTED {g}: {sb[g]} -> {sa_[g]}")

    print("\n" + "=" * 108)
    passed = sum(1 for _, c in CHECKS if c)
    print(f"  {passed}/{len(CHECKS)} checks passed")
    for label, ok in CHECKS:
        if not ok:
            print(f"    FAILED: {label}")
    return 0 if passed == len(CHECKS) else 1


def main(argv):
    if len(argv) >= 2 and argv[0] == "capture":
        tree = argv[2] if len(argv) > 2 else ROOT
        return capture(argv[1], tree)
    if argv and argv[0] == "report":
        return report()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
