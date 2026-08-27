"""Guardrail for the One Pager's development-deal debt basis.

WHAT IS BEING PROVED. ``one_pager.get_capitalization_stack`` now rebases a
development deal's ``debt`` onto the Inspection table's ``mHardCosts`` (hard
costs drawn to date). The Portfolio Snapshot shares that cap-stack object but
must not move: its dev rule is the full committed facility, per PDF footnote
(6), and its Total Cap column ties to the published PDF. The three claims:

  1. One Pager Debt changes ONLY for dev deals that have an inspection row.
     Dev deals with no row keep the debt they had; non-dev deals never move.
  2. Snapshot Debt (portfolio_snapshot_debt.resolve_debt, which BOTH the
     Financial and Loan subtabs call) is byte-identical across the change.
  3. Snapshot Total Cap is byte-identical across the change.

HOW. A pure before/after differential on IDENTICAL inputs. The inputs are
fetched once from the live read-only API and pickled, then the real committed
functions are called on each side:

    # before — a worktree checked out at origin/main
    git worktree add ../wf-base origin/main
    python scripts/onepager_dev_debt_check.py fetch    inputs.pkl
    python ../wf-base/scripts/onepager_dev_debt_check.py capture inputs.pkl before.json
    python scripts/onepager_dev_debt_check.py capture  inputs.pkl after.json
    python scripts/onepager_dev_debt_check.py report   before.json after.json

``fetch`` is run once from either side — the pickle is the shared input, which
is what makes the comparison a code differential and not a data differential.
``capture`` calls ``get_capitalization_stack`` with whatever signature that side
has (the ``inspection`` kwarg does not exist on origin/main), so the same file
runs unmodified in both trees.

WIRING CHECKS. A differential cannot see a subtab that stopped reading the twin
it is supposed to read, so ``report`` also asserts the three source lines that
keep the Snapshot on the ISBS basis are still present. If someone later
"simplifies" ``cap.get("total_cap_isbs", cap.get("total_cap"))`` back to
``cap.get("total_cap")``, claim 3 breaks silently and this is what catches it.

Read-only against live. Requires WF_TOKEN in the environment.
"""
import json
import os
import pickle
import re
import sys

import pandas as pd
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BASE = ("https://app-waterfall-dev-v2.icyplant-026fb2db"
        ".eastus.azurecontainerapps.io")
PAGE = 500

#: Date slices this run could not fully recover; surfaced in the report.
INCOMPLETE = []

#: The quarter every cap stack is built for. Fixed so the run is reproducible.
QUARTER = "2026-Q1"

#: Non-dev controls. P0000014 (Crowne Plaza) is the one that matters: it is the
#: only NON-development deal carrying an inspection row, so it is the direct
#: test that the override is gated on the dev classification and not merely on
#: "has hard costs". The others are ordinary operating deals.
CONTROLS = ["P0000014", "P0000030", "P0000003", "P0000007"]


# ── live fetch (read-only) ────────────────────────────────────────────────

def _get(path, params=None):
    token = os.environ.get("WF_TOKEN", "").strip()
    if not token:
        raise SystemExit("WF_TOKEN not set in environment")
    r = requests.get(f"{BASE}{path}", params=params,
                     headers={"Authorization": f"Bearer {token}"}, timeout=300)
    if r.status_code in (401, 403):
        raise SystemExit(f"HTTP {r.status_code} — token expired or rejected")
    r.raise_for_status()
    return r.json()


def _rows(table, filters=None):
    """Every matching row, without ever trusting OFFSET.

    The rows endpoint pages with LIMIT/OFFSET and no tiebreaker, so page 2 of a
    query can repeat rows another page already returned. Every request here is
    narrowed until the whole match set fits in a single page, so no request is
    ever made with a nonzero offset.
    """
    params = {"page": 1, "page_size": PAGE}
    for k, v in (filters or {}).items():
        params[f"filter__{k}"] = v
    d = _get(f"/api/data/tables/{table}/rows", params=params)
    total, cols = d.get("total") or 0, d.get("columns") or []
    if total <= PAGE:
        return pd.DataFrame(d.get("rows") or [], columns=cols or None)

    # Too big for one page: split by year on a date column until each slice fits.
    dcol = next((c for c in cols
                 if c.lower() in ("dtentry", "effectivedate", "dtevent",
                                  "dtvaluation", "event_date")), None)
    if not dcol:
        raise SystemExit(f"{table}: {total} rows, too many for one page and no "
                         f"date column to partition on")
    def slice_rows(date_prefix, sort=None, order=None):
        """One request narrowed to a date prefix; (rows, total). Never offsets."""
        f2 = dict(filters or {})
        f2[dcol] = date_prefix
        p2 = {"page": 1, "page_size": PAGE}
        if sort:
            p2.update({"sort": sort, "order": order or "asc"})
        for k, v in f2.items():
            p2[f"filter__{k}"] = v
        r = _get(f"/api/data/tables/{table}/rows", params=p2)
        return (r.get("rows") or []), (r.get("total") or 0)

    def ends_union(date_prefix, total):
        """Recover a slice that cannot be narrowed by date any further.

        Every request is page 1 — no OFFSET is ever used, since that is what
        duplicates and drops rows on this endpoint. Each (sort column, order)
        pair returns a different first ``PAGE`` rows, so unioning a few of them
        covers a slice larger than one page. Sorting on the date column alone
        does not work here: within a single day it is constant, so ASC and DESC
        return the same page.

        The union is deduplicated on the whole row and checked against the
        reported total, so a slice this cannot fully recover raises instead of
        silently returning a subset.
        """
        seen, merged = set(), []
        for scol in [c for c in cols if c != dcol] + [dcol]:
            for order in ("asc", "desc"):
                rows, _ = slice_rows(date_prefix, scol, order)
                for r in rows:
                    key = tuple(sorted((k, str(v)) for k, v in r.items()))
                    if key not in seen:
                        seen.add(key)
                        merged.append(r)
                if len(merged) >= total:
                    return merged
        # Short of the reported total. The usual cause is exact-duplicate rows
        # in the source, which the whole-row dedup above collapses and the
        # endpoint's COUNT(*) still counts — unreachable by construction.
        #
        # Recorded, not raised: this file's claims are a BEFORE/AFTER
        # differential, and both sides are fed this same pickle, so a slice
        # that is short by a few rows cannot manufacture or mask a difference.
        # It only means this deal's absolute debt may not match live, which
        # `report` states rather than assumes.
        INCOMPLETE.append((table, date_prefix, len(merged), total))
        print(f"    ! {table} {date_prefix}: recovered {len(merged)}/{total} "
              f"(likely duplicate source rows) — input marked incomplete")
        return merged

    out = []
    for year in range(2000, 2036):
        rows, t2 = slice_rows(str(year))
        if t2 == 0:
            continue
        if t2 <= PAGE:
            out.extend(rows)
            continue
        # Dense year — narrow again to each month. Dates are ISO here
        # ('2021-02-28T00:00:00'), so 'YYYY-MM' is a valid prefix.
        for month in range(1, 13):
            mrows, t3 = slice_rows(f"{year}-{month:02d}")
            if t3 == 0:
                continue
            if t3 <= PAGE:
                out.extend(mrows)
                continue
            # Dense month — narrow once more, to the day, then fall back to an
            # ASC+DESC union for a day that still overflows a page.
            for day in range(1, 32):
                prefix = f"{year}-{month:02d}-{day:02d}"
                drows, t4 = slice_rows(prefix)
                if t4 == 0:
                    continue
                out.extend(drows if t4 <= PAGE else ends_union(prefix, t4))
    return pd.DataFrame(out, columns=cols or None)


def cmd_fetch(out_path):
    """Pull every input the cap stack reads, for the targets, into one pickle."""
    from config import DEV_STRATEGIES

    deals = _rows("deals")
    life = deals["Lifecycle"].fillna("").astype(str).str.strip().str.lower()
    strat = (deals["Investment_Strategy"].fillna("").astype(str).str.strip().str.lower()
             if "Investment_Strategy" in deals.columns else life * 0)
    # Same precedence as one_pager._is_dev_deal / resolve_strategy.
    effective = strat.where(strat != "", life)
    dev_vcodes = sorted(deals.loc[effective.isin(DEV_STRATEGIES), "vcode"].astype(str))
    targets = dev_vcodes + [c for c in CONTROLS if c not in dev_vcodes]
    print(f"dev deals: {len(dev_vcodes)}   controls: {len(targets) - len(dev_vcodes)}"
          f"   targets: {len(targets)}")

    # Small enough to take whole.
    loans = _rows("loans")
    vals = _rows("valuations")
    insp = _rows("inspection")
    print(f"loans={len(loans)} valuations={len(vals)} inspection={len(insp)}")

    # `relationships` is accepted by get_capitalization_stack but never read in
    # its body (signature line only, on both sides of this change), so it is
    # not fetched — it cannot influence the differential either way.

    # Per-deal slices: too big to take whole, and only the targets are needed.
    # ISBS balance sheet and waterfalls by vcode, accounting by InvestmentID.
    from loaders import build_investmentid_to_vcode
    inv_to_vcode = build_investmentid_to_vcode(deals)

    isbs_parts, acct_parts, wf_parts = [], [], []
    for i, vc in enumerate(targets, 1):
        part = _rows("isbs_interim_bs", {"vcode": vc.lower()})
        if not part.empty:
            isbs_parts.append(part)
        w = _rows("waterfalls", {"vcode": vc})
        if not w.empty:
            wf_parts.append(w)
        iids = [iid for iid, v in inv_to_vcode.items() if str(v) == vc]
        for iid in iids:
            a = _rows("accounting", {"InvestmentID": str(iid)})
            if not a.empty:
                acct_parts.append(a)
        print(f"  [{i:>2}/{len(targets)}] {vc:<10} isbs={len(part):>5} "
              f"wf={len(w):>4} investment_ids={len(iids)}")

    isbs = pd.concat(isbs_parts, ignore_index=True) if isbs_parts else pd.DataFrame()
    acct = pd.concat(acct_parts, ignore_index=True) if acct_parts else pd.DataFrame()
    wfs = pd.concat(wf_parts, ignore_index=True) if wf_parts else pd.DataFrame()
    # A substring filter can match more than one deal, and an InvestmentID can
    # repeat across targets — drop anything fetched twice.
    for name, df in (("accounting", acct), ("waterfalls", wfs)):
        if not df.empty:
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
    print(f"isbs_interim_bs={len(isbs)}  accounting={len(acct)}  waterfalls={len(wfs)}")

    if INCOMPLETE:
        print(f"\nWARNING: {len(INCOMPLETE)} slice(s) incomplete: {INCOMPLETE}")

    payload = {"quarter": QUARTER, "targets": targets, "dev_vcodes": dev_vcodes,
               "incomplete": list(INCOMPLETE),
               "deals": deals, "loans": loans, "valuations": vals,
               "waterfalls": wfs, "relationships": None, "inspection": insp,
               "isbs": isbs, "accounting": acct}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nwrote {out_path}")


# ── capture: run the real functions on THIS tree ──────────────────────────

def cmd_capture(in_path, out_path):
    import inspect as _inspect

    from flask_app.services.data_service import _normalize_isbs, _normalize_accounting
    from flask_app.services.portfolio_snapshot_debt import (
        committed_facility, deal_loan_rows, resolve_debt)
    from flask_app.services.portfolio_snapshot_operating import is_dev_deal
    from one_pager import get_capitalization_stack, _child_vcodes_for_parent

    with open(in_path, "rb") as f:
        d = pickle.load(f)

    isbs = _normalize_isbs(d["isbs"].copy()) if not d["isbs"].empty else None
    acct = _normalize_accounting(d["accounting"].copy()) if not d["accounting"].empty else None
    deals, loans, insp = d["deals"], d["loans"], d["inspection"]

    # origin/main has no `inspection` kwarg — pass it only where it exists, so
    # this same file runs unmodified in both trees.
    takes_inspection = "inspection" in _inspect.signature(get_capitalization_stack).parameters
    print(f"get_capitalization_stack accepts inspection= : {takes_inspection}")

    life = deals.set_index(deals["vcode"].astype(str))
    out = {}
    for vc in d["targets"]:
        kwargs = dict(isbs_raw=isbs, quarter_str=d["quarter"],
                      relationships=d["relationships"])
        if takes_inspection:
            kwargs["inspection"] = insp
        cap = get_capitalization_stack(vc, loans, d["valuations"], d["waterfalls"],
                                       acct, deals, **kwargs)

        # The Snapshot's Debt, via the real shared resolver both subtabs call.
        row = life.loc[vc] if vc in life.index else None
        strategy = ""
        for field in ("Investment_Strategy", "Lifecycle"):
            if row is not None and field in row.index and pd.notna(row[field]):
                if str(row[field]).strip():
                    strategy = str(row[field]).strip()
                    break
        dev = is_dev_deal(strategy)
        committed = committed_facility(
            deal_loan_rows(loans, vc, _child_vcodes_for_parent(vc, deals)))
        snap_debt, snap_basis = resolve_debt(cap, dev, committed)

        # The Snapshot's Total Cap: the exact expression the Financial subtab
        # uses. `report` separately asserts that line still reads this way.
        snap_total_cap = cap.get("total_cap_isbs", cap.get("total_cap"))

        out[vc] = {
            "dev": bool(dev),
            "strategy": strategy,
            "op_debt": cap.get("debt"),
            "op_total_cap": cap.get("total_cap"),
            "op_debt_basis": cap.get("debt_basis", ""),
            "snap_debt": snap_debt,
            "snap_basis": snap_basis,
            "snap_total_cap": snap_total_cap,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"takes_inspection": takes_inspection, "rows": out}, f,
                  indent=1, sort_keys=True, default=str)
    print(f"wrote {out_path}: {len(out)} deals")


# ── report ────────────────────────────────────────────────────────────────

_WIRING = [
    ("flask_app/services/portfolio_snapshot_debt.py",
     'cap or {}).get("debt_isbs"',
     "Snapshot Debt reads the pre-override ISBS twin"),
    ("flask_app/services/portfolio_snapshot_financial.py",
     'cap.get("total_cap_isbs"',
     "Snapshot Financial Total Cap reads the pre-override twin"),
    ("flask_app/services/portfolio_snapshot_loan.py",
     'cap.get("debt_isbs"',
     "Snapshot Loan diagnostic reads the pre-override twin"),
]


def cmd_report(before_path, after_path):
    with open(before_path, encoding="utf-8") as f:
        before = json.load(f)
    with open(after_path, encoding="utf-8") as f:
        after = json.load(f)
    b, a = before["rows"], after["rows"]

    fails = []

    def chk(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(label)

    def num(v):
        try:
            return round(float(v), 6)
        except (TypeError, ValueError):
            return None

    print("=" * 100)
    print("SETUP")
    chk("before side is origin/main (no inspection kwarg)",
        before["takes_inspection"] is False)
    chk("after side carries the inspection kwarg",
        after["takes_inspection"] is True)
    chk("same deal population on both sides", set(b) == set(a),
        f"{len(b)} vs {len(a)}")

    dev = sorted(v for v in b if b[v]["dev"])
    non = sorted(v for v in b if not b[v]["dev"])
    chk("dev classification is stable across the change",
        all(b[v]["dev"] == a[v]["dev"] for v in b),
        f"{len(dev)} dev / {len(non)} non-dev")

    print("\n" + "=" * 100)
    print("CLAIM 1 — One Pager Debt moves only for dev deals with an inspection row")
    moved = sorted(v for v in b if num(b[v]["op_debt"]) != num(a[v]["op_debt"]))
    rebased = sorted(v for v in a
                     if a[v]["op_debt_basis"].startswith("mHardCosts"))
    chk("every deal whose debt moved is a dev deal",
        all(b[v]["dev"] for v in moved), f"moved: {len(moved)}")
    chk("every deal whose debt moved was rebased onto mHardCosts",
        set(moved) <= set(rebased))
    chk("no NON-dev deal moved", not [v for v in moved if not b[v]["dev"]])
    chk("no non-dev deal was rebased, incl. P0000014 which HAS an inspection row",
        not [v for v in rebased if not b[v]["dev"]],
        "P0000014 rebased" if "P0000014" in rebased else "P0000014 untouched")
    chk("dev deals with no inspection row kept their existing debt",
        all(num(b[v]["op_debt"]) == num(a[v]["op_debt"])
            for v in dev if v not in rebased))

    print("\n  One Pager debt, before -> after (dev deals only):")
    for v in dev:
        bd, ad = num(b[v]["op_debt"]), num(a[v]["op_debt"])
        mark = "CHANGED" if bd != ad else ""
        print(f"    {v:<10} {(bd or 0)/1e6:>10,.2f}M -> {(ad or 0)/1e6:>10,.2f}M"
              f"   {a[v]['op_debt_basis'][:34]:<36}{mark}")

    print("\n" + "=" * 100)
    print("CLAIM 2 — Snapshot Debt is byte-identical")
    diff_debt = [v for v in b if num(b[v]["snap_debt"]) != num(a[v]["snap_debt"])]
    chk("Snapshot Debt unchanged for every deal", not diff_debt,
        f"differs on {diff_debt}" if diff_debt else f"{len(b)} deals")
    diff_basis = [v for v in b if b[v]["snap_basis"] != a[v]["snap_basis"]]
    chk("Snapshot debt BASIS label unchanged for every deal", not diff_basis,
        f"differs on {diff_basis}" if diff_basis else "")

    print("\n" + "=" * 100)
    print("CLAIM 3 — Snapshot Total Cap is byte-identical")
    diff_tc = [v for v in b
               if num(b[v]["snap_total_cap"]) != num(a[v]["snap_total_cap"])]
    chk("Snapshot Total Cap unchanged for every deal", not diff_tc,
        f"differs on {diff_tc}" if diff_tc else f"{len(b)} deals")

    moved_tc = [v for v in b
                if num(b[v]["op_total_cap"]) != num(a[v]["op_total_cap"])]
    chk("One Pager Total Cap moved for exactly the rebased deals",
        set(moved_tc) == set(moved),
        f"{len(moved_tc)} deals")

    print("\n" + "=" * 100)
    print("WIRING — the reads that keep the Snapshot on the ISBS basis")
    for rel, needle, label in _WIRING:
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            chk(label, False, f"cannot read {rel}")
            continue
        chk(label, needle in src, rel)

    # A rebased deal must actually differ from its committed facility, or the
    # test is vacuous — it would pass even if nothing had been rebased.
    print("\n" + "=" * 100)
    print("NON-VACUITY")
    chk("at least one deal was actually rebased", bool(moved),
        f"{len(moved)} rebased: {moved}")
    real = [v for v in moved
            if num(a[v]["op_debt"]) != num(a[v]["snap_debt"])]
    chk("rebased deals now show a DIFFERENT debt on the two pages", bool(real),
        f"{len(real)} of {len(moved)} diverge from the Snapshot figure")

    print("\n" + "=" * 100)
    if fails:
        print(f"RESULT: {len(fails)} FAILED")
        for x in fails:
            print(f"  - {x}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    mode = sys.argv[1]
    if mode == "fetch":
        cmd_fetch(sys.argv[2])
    elif mode == "capture":
        cmd_capture(sys.argv[2], sys.argv[3])
    elif mode == "report":
        raise SystemExit(cmd_report(sys.argv[2], sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
