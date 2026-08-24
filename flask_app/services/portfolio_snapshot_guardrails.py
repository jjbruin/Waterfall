"""Portfolio Snapshot — Step 8 guardrails.

Read-only auditors over the four assembled subtabs. Nothing here computes a
report figure; every function answers "could a reader mistake this for a real
number?" and returns findings the UI shows and a human resolves.

Design rules, all deliberate:

* **Soft by default.** A guardrail never raises and never blanks a metric. It
  returns findings with a severity. A page with findings still renders — the
  alternative (failing the build) would hide the whole report over one bad deal.
* **`severity` is the only thing callers should branch on.** ``"error"`` means a
  figure is provably wrong or a missing value is rendering as real; ``"warn"``
  means it needs a human; ``"info"`` is a count worth surfacing.
* **The upstream zero problem is detected, not fixed.** ``one_pager.py``
  initialises every cap_stack money field to ``0.0`` rather than ``None``, so a
  deal with no data at all returns ``debt: 0.0`` / ``pref_equity: 0.0``, which
  formats to "0.00" and is indistinguishable from a genuine zero. That contract
  belongs to the One Pager and is shared with other tabs, so this module flags
  the condition instead of changing it. ``detect_empty_cap_stack`` is the
  detector: an *entire* Zone A of zeros means "no data", not "no capital".
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger(__name__)

ERROR, WARN, INFO = "error", "warn", "info"

#: Zone A money fields. All zero together => the One Pager had nothing to report.
_CAP_FIELDS = ("debt", "total_pref", "ptr_equity", "total_cap")

#: The scaled columns that MUST be withheld when ownership is unresolved.
_SCALED_FIELDS = ("pct_of_pref", "invested", "total_commitment", "unfunded")


def _finding(check: str, severity: str, message: str, **extra) -> dict:
    return {"check": check, "severity": severity, "message": message, **extra}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _fin_rows(financial: dict) -> list:
    """Every Financial row, grouped rows plus ownership-flagged ones."""
    out = []
    for blk in (financial.get("groups") or {}).values():
        out.extend(blk.get("deals") or [])
    out.extend(financial.get("ownership_flagged") or [])
    return out


def _plain_rows(payload: dict) -> list:
    """Every row of an Operating/Loan payload (groups are plain lists there)."""
    out = []
    for rows in (payload.get("groups") or {}).values():
        out.extend(rows or [])
    out.extend(payload.get("ownership_flagged") or [])
    return out


# ── 1. missing-data flagging ──────────────────────────────────────────────

def detect_empty_cap_stack(financial: dict) -> list:
    """Deals whose whole Zone A is zero — i.e. the One Pager had no data.

    A single zero can be real (Citizen Storage genuinely had $0 pref at 26Q1,
    before it closed). All four at once cannot: a held deal has *some*
    capitalisation. Reported as an error because those zeros render as "0.00"
    and there is nothing in the payload to tell a reader they are placeholders.
    """
    out = []
    for r in _fin_rows(financial):
        vals = [_num(r.get(f)) for f in _CAP_FIELDS]
        if all(v is not None and v == 0 for v in vals):
            out.append(_finding(
                "empty_cap_stack", ERROR,
                f"{r.get('name')} ({r.get('vcode')}): every cap-stack figure is "
                f"0.00. The One Pager defaults these to 0.0 when it has no data, "
                f"so these are placeholders rendering as real zeros.",
                vcode=r.get("vcode"), fields=list(_CAP_FIELDS)))
    return out


def audit_missing_data(subtabs: dict) -> dict:
    """Classify every deal-metric as value / zero / missing, per subtab.

    The point is the `zero` column: those are the values a reader cannot
    distinguish from data, so they are enumerated rather than counted.
    """
    spec = {
        "financial": (_fin_rows, ("debt", "total_pref", "ptr_equity", "total_cap",
                                  "invested", "total_commitment", "unfunded")),
        "loan": (_plain_rows, ("debt", "valuation", "ltv", "ytd_dscr",
                               "debt_yield")),
        "operating": (_plain_rows, ("expected_growth", "actual_growth")),
    }
    report: dict = {}
    for name, (extract, fields) in spec.items():
        payload = subtabs.get(name) or {}
        if not payload:
            continue
        rows = extract(payload)
        per_field: dict = {}
        for f in fields:
            vals, zeros, missing = 0, [], []
            for r in rows:
                v = _num(r.get(f))
                if v is None:
                    missing.append(r.get("vcode"))
                elif v == 0:
                    zeros.append(r.get("vcode"))
                else:
                    vals += 1
            per_field[f] = {"value": vals, "zero": zeros, "missing": missing}
        report[name] = {"deal_count": len(rows), "fields": per_field}

    # Operating NOI lives in a nested dict, so handle it separately.
    op = subtabs.get("operating") or {}
    if op:
        noi_missing: dict = {"at_close": [], "uw_ye": [], "projected_ye": []}
        for r in _plain_rows(op):
            noi = r.get("noi") or {}
            for k in noi_missing:
                if _num(noi.get(k)) is None:
                    noi_missing[k].append(r.get("vcode"))
        report.setdefault("operating", {})["noi_missing"] = noi_missing
    return report


def check_aggregate_integrity(subtabs: dict) -> list:
    """No aggregate may be a number when every one of its inputs is missing.

    The failure this catches is an accumulator that starts at 0.0 and only ever
    adds: an all-missing group then reports 0.0 and formats as a real zero.
    """
    out = []

    fin = subtabs.get("financial") or {}
    for gname, blk in (fin.get("groups") or {}).items():
        rows = blk.get("deals") or []
        sub = blk.get("subtotal") or {}
        for f in ("debt", "total_pref", "invested", "total_commitment"):
            inputs = [_num(r.get(f)) for r in rows]
            if inputs and all(v is None for v in inputs) and sub.get(f) is not None:
                out.append(_finding(
                    "aggregate_integrity", ERROR,
                    f"Financial subtotal {gname}.{f} is {sub.get(f)} but every "
                    f"deal in the group is missing that figure.",
                    group=gname, field=f))

    summ = subtabs.get("summary") or {}
    for alloc_key in ("asset_allocation", "deal_type_allocation"):
        alloc = summ.get(alloc_key) or {}
        for b in (alloc.get("buckets") or []):
            n = b.get("deal_count") or 0
            for f in ("funded", "committed"):
                miss = b.get(f"{f}_missing") or 0
                if n and miss == n and b.get(f) is not None:
                    out.append(_finding(
                        "aggregate_integrity", ERROR,
                        f"Summary {alloc_key} bucket '{b.get('label')}' reports "
                        f"{f}={b.get(f)} with all {n} deals missing it.",
                        bucket=b.get("label"), field=f))
    return out


# ── 2. pref-equity cross-check ────────────────────────────────────────────

def _codes_in_text(text: str) -> set:
    """Investor-ish tokens in a free-text comment.

    Deliberately loose: the comment is prose written by an analyst, so this
    looks for upper-case alphanumeric runs that resemble entity codes and for
    percentages, and does not attempt to parse a sentence.
    """
    if not text:
        return set()
    return {t for t in re.findall(r"\b[A-Z][A-Z0-9]{2,}\b", str(text))}


def pref_equity_crosscheck(financial: dict, resolved: dict,
                           pe_cap_comments: Optional[dict] = None) -> list:
    """Compare the computed investor mapping against `pe_cap_comment` prose.

    SOFT by design — a disagreement is surfaced for human review and never
    changes a figure. The comment is free text maintained by analysts, so a
    mismatch is as likely to be stale prose as a resolution error; the value is
    in noticing that the two disagree at all.

    ``pe_cap_comments`` maps vcode -> comment text (from `one_pager_comments`).
    With none supplied the check reports that it could not run rather than
    silently passing.
    """
    if not pe_cap_comments:
        return [_finding("pref_equity_crosscheck", INFO,
                         "No pe_cap_comment data supplied — cross-check skipped.")]

    out = []
    investor = str(resolved.get("investor_code") or "").upper()
    checked = 0
    for r in _fin_rows(financial):
        vcode = r.get("vcode")
        comment = pe_cap_comments.get(vcode)
        if not comment:
            continue
        checked += 1
        codes = _codes_in_text(comment)
        if not codes:
            continue
        # The investor this snapshot is for should appear in a comment that
        # names entity codes at all. If it names others but not this one, the
        # computed mapping and the prose disagree about who holds the pref.
        if investor and investor not in codes:
            out.append(_finding(
                "pref_equity_crosscheck", WARN,
                f"{r.get('name')} ({vcode}): pe_cap_comment names "
                f"{sorted(codes)} but not {investor}, while the ownership chain "
                f"resolves {investor} to {r.get('pct_of_pref')}. Prose and "
                f"computed mapping disagree — human review.",
                vcode=vcode, comment_codes=sorted(codes),
                computed_pct=r.get("pct_of_pref")))
    out.append(_finding("pref_equity_crosscheck", INFO,
                        f"Compared {checked} deal(s) carrying a pe_cap_comment.",
                        compared=checked))
    return out


# ── 3. incomplete ownership ───────────────────────────────────────────────

def check_ownership_completeness(resolved: dict, subtabs: dict) -> list:
    """Every ownership-flagged deal must withhold its scaled figures.

    Generalised: driven by whatever `resolved['flagged']` contains, never by a
    hardcoded vcode. 45th & Main is today's only case; any future deal with a
    broken chain is caught by the same assertion.
    """
    out = []
    flagged = {f.get("vcode") for f in (resolved.get("flagged") or [])}

    if not flagged:
        out.append(_finding("ownership_completeness", INFO,
                            "No deal has an unresolved ownership chain."))
        return out

    fin_by_vcode = {r.get("vcode"): r for r in _fin_rows(subtabs.get("financial") or {})}
    for vcode in sorted(flagged):
        row = fin_by_vcode.get(vcode)
        if row is None:
            out.append(_finding(
                "ownership_completeness", WARN,
                f"{vcode} is ownership-flagged but absent from the Financial "
                f"subtab — it should appear with its scaled columns withheld, "
                f"not vanish.", vcode=vcode))
            continue
        leaked = [f for f in _SCALED_FIELDS if row.get(f) is not None]
        if leaked:
            out.append(_finding(
                "ownership_completeness", ERROR,
                f"{row.get('name')} ({vcode}) has no resolvable ownership % but "
                f"reports scaled figure(s) {leaked} — an unscaled number is "
                f"rendering in a scaled column.",
                vcode=vcode, fields=leaked))
        if row.get("pct_of_pref") is not None:
            out.append(_finding(
                "ownership_completeness", ERROR,
                f"{vcode} carries a look-through % despite being flagged.",
                vcode=vcode))

    # Summary must exclude them from the scaled allocation, not zero them in.
    summ = subtabs.get("summary") or {}
    for d in (summ.get("deals") or []):
        if d.get("vcode") in flagged and d.get("lookthrough_pct") is not None:
            out.append(_finding(
                "ownership_completeness", ERROR,
                f"{d.get('vcode')} is flagged but Summary gave it a "
                f"look-through % of {d.get('lookthrough_pct')}.",
                vcode=d.get("vcode")))

    out.append(_finding(
        "ownership_completeness", WARN,
        f"{len(flagged)} deal(s) withhold scaled figures: {sorted(flagged)}. "
        f"Their deal-level (Zone A) figures are ownership-independent and are "
        f"still reported.", vcodes=sorted(flagged)))
    return out


# ── 4. quarter boundary ───────────────────────────────────────────────────

def check_quarter_window(resolved: dict) -> list:
    """Both gates together: acquired <= quarter_end AND not sold <= quarter_end."""
    out = []
    q_end = resolved.get("quarter_end")
    diag = resolved.get("diagnostics") or {}

    sold = {e.get("vcode") for e in (resolved.get("excluded_sold") or [])}
    not_acq = {e.get("vcode") for e in (resolved.get("excluded_not_acquired") or [])}
    held = {e.get("vcode") for items in (resolved.get("groups") or {}).values()
            for e in items}
    held |= {f.get("vcode") for f in (resolved.get("flagged") or [])}

    both = sold & not_acq
    if both:
        out.append(_finding("quarter_window", ERROR,
                            f"Deal(s) excluded under both gates: {sorted(both)}. "
                            f"The two are mutually exclusive.", vcodes=sorted(both)))

    overlap = held & (sold | not_acq)
    if overlap:
        out.append(_finding("quarter_window", ERROR,
                            f"Deal(s) both held and excluded: {sorted(overlap)}.",
                            vcodes=sorted(overlap)))

    missing = diag.get("acquisition_date_missing") or []
    if missing:
        out.append(_finding(
            "quarter_window", WARN,
            f"{len(missing)} held deal(s) have no Acquisition_Date, so the "
            f"acquired gate could not test them and they were kept "
            f"(fail-open): {[m.get('vcode') for m in missing]}.",
            vcodes=[m.get("vcode") for m in missing]))

    out.append(_finding(
        "quarter_window", INFO,
        f"Quarter end {q_end}: {len(held)} held, {len(sold)} excluded as sold, "
        f"{len(not_acq)} excluded as not yet acquired.",
        held=len(held), sold=len(sold), not_acquired=len(not_acq)))
    return out


# ── entry point ───────────────────────────────────────────────────────────

def run_guardrails(resolved: dict, subtabs: dict,
                   pe_cap_comments: Optional[dict] = None) -> dict:
    """All guardrails over one assembled page. Never raises.

    A guardrail that itself fails is reported as a finding rather than taking
    the page down with it — the whole point is to add safety, not a new way to
    break the report.
    """
    findings: list = []
    checks = {
        "empty_cap_stack": lambda: detect_empty_cap_stack(
            subtabs.get("financial") or {}),
        "aggregate_integrity": lambda: check_aggregate_integrity(subtabs),
        "pref_equity_crosscheck": lambda: pref_equity_crosscheck(
            subtabs.get("financial") or {}, resolved, pe_cap_comments),
        "ownership_completeness": lambda: check_ownership_completeness(
            resolved, subtabs),
        "quarter_window": lambda: check_quarter_window(resolved),
    }
    for name, fn in checks.items():
        try:
            findings.extend(fn() or [])
        except Exception as exc:              # a guardrail must never break the page
            log.exception("guardrail %s failed", name)
            findings.append(_finding(name, WARN,
                                     f"Guardrail did not run: {exc}"))

    try:
        audit = audit_missing_data(subtabs)
    except Exception as exc:
        log.exception("missing-data audit failed")
        audit = {"error": str(exc)}

    counts = {ERROR: 0, WARN: 0, INFO: 0}
    for f in findings:
        counts[f.get("severity", INFO)] = counts.get(f.get("severity", INFO), 0) + 1

    return {
        "findings": findings,
        "counts": counts,
        "ok": counts[ERROR] == 0,
        "missing_data_audit": audit,
    }


# ── Self-test ─────────────────────────────────────────────────────────────

def _synthetic():
    """A minimal well-formed page, used to prove each detector fires.

    Synthetic rather than live so the negative tests hold regardless of what the
    portfolio looks like on a given day. An earlier live-dependent version of
    this test silently disarmed itself partway through the build, when the live
    set stopped having any ownership-flagged deal.
    """
    row = {"vcode": "P1", "name": "Alpha", "debt": 10.0, "total_pref": 5.0,
           "ptr_equity": 2.0, "total_cap": 17.0, "pct_of_pref": 0.5,
           "invested": 2.5, "total_commitment": 2.5, "unfunded": 0.0}
    financial = {
        "groups": {"Fund A": {"deals": [dict(row)], "subtotal": {
            "debt": 10.0, "total_pref": 5.0, "invested": 2.5,
            "total_commitment": 2.5}}},
        "ownership_flagged": [],
    }
    summary = {
        "asset_allocation": {"buckets": [
            {"label": "Multifamily", "funded": 2.5, "committed": 2.5,
             "deal_count": 1, "funded_missing": 0, "committed_missing": 0}],
            "total_funded": 2.5, "total_committed": 2.5},
        "deal_type_allocation": {"buckets": [], "total_funded": None,
                                 "total_committed": None},
        "deals": [{"vcode": "P1", "lookthrough_pct": 0.5}],
    }
    resolved = {
        "investor_code": "TGAM", "quarter_end": "2026-03-31",
        "groups": {"Fund A": [{"vcode": "P1", "name": "Alpha"}]},
        "flagged": [], "excluded_sold": [], "excluded_not_acquired": [],
        "diagnostics": {"acquisition_date_missing": []},
    }
    return resolved, {"financial": financial, "summary": summary,
                      "operating": {"groups": {}}, "loan": {"groups": {}}}


def _selftest():                                    # pragma: no cover
    """Prove every detector fires on injected bad data, and stays quiet clean."""
    import copy

    checks = []

    def chk(label, cond):
        checks.append((label, bool(cond)))
        print("    [" + ("PASS" if cond else "FAIL") + "] " + label)

    print("=" * 100)
    print("STEP 8 GUARDRAILS - negative tests "
          "(a guardrail that cannot fail is not a guardrail)")
    print("=" * 100)

    base_res, base_tabs = _synthetic()

    clean = run_guardrails(base_res, base_tabs)
    chk("clean synthetic page yields zero ERROR findings",
        clean["counts"][ERROR] == 0)
    chk("clean page reports ok=True", clean["ok"] is True)

    # 1. empty cap stack
    bad = copy.deepcopy(base_tabs)
    r = bad["financial"]["groups"]["Fund A"]["deals"][0]
    for f in _CAP_FIELDS:
        r[f] = 0.0
    fired = detect_empty_cap_stack(bad["financial"])
    chk("empty_cap_stack fires on an all-zero Zone A", len(fired) == 1)
    chk("empty_cap_stack is severity=error",
        bool(fired) and fired[0]["severity"] == ERROR)
    bad2 = copy.deepcopy(base_tabs)
    bad2["financial"]["groups"]["Fund A"]["deals"][0]["debt"] = 0.0
    chk("a single zero does NOT fire (a real zero is legitimate)",
        detect_empty_cap_stack(bad2["financial"]) == [])

    # 2. aggregate integrity - Financial subtotal
    bad = copy.deepcopy(base_tabs)
    bad["financial"]["groups"]["Fund A"]["deals"][0]["debt"] = None
    fired = check_aggregate_integrity(bad)
    chk("aggregate_integrity fires when a subtotal survives all-missing inputs",
        any(f.get("field") == "debt" for f in fired))

    # 3. aggregate integrity - Summary bucket (the leak Step 8 fixed)
    bad = copy.deepcopy(base_tabs)
    b = bad["summary"]["asset_allocation"]["buckets"][0]
    b["funded_missing"], b["funded"] = b["deal_count"], 0.0
    fired = check_aggregate_integrity(bad)
    chk("aggregate_integrity fires on an all-missing Summary bucket at 0.0",
        any(f.get("bucket") == "Multifamily" for f in fired))

    # 4. ownership completeness
    bad_res = copy.deepcopy(base_res)
    bad_res["flagged"] = [{"vcode": "P1", "name": "Alpha"}]
    fired = check_ownership_completeness(bad_res, copy.deepcopy(base_tabs))
    errs = [f for f in fired if f["severity"] == ERROR]
    chk("ownership_completeness fires when a flagged deal reports scaled figures",
        len(errs) >= 1)
    chk("...and names the leaking fields", any(f.get("fields") for f in errs))

    ok_tabs = copy.deepcopy(base_tabs)
    for f in _SCALED_FIELDS:
        ok_tabs["financial"]["groups"]["Fund A"]["deals"][0][f] = None
    ok_tabs["summary"]["deals"][0]["lookthrough_pct"] = None
    fired = check_ownership_completeness(bad_res, ok_tabs)
    chk("ownership_completeness passes when scaled figures ARE withheld",
        not [f for f in fired if f["severity"] == ERROR])

    gone = copy.deepcopy(base_tabs)
    gone["financial"]["groups"] = {}
    fired = check_ownership_completeness(bad_res, gone)
    chk("ownership_completeness fires when a flagged deal vanishes entirely",
        any("absent from the Financial" in f["message"] for f in fired))

    # 5. quarter window
    bad_res = copy.deepcopy(base_res)
    bad_res["excluded_sold"] = [{"vcode": "PZ"}]
    bad_res["excluded_not_acquired"] = [{"vcode": "PZ"}]
    chk("quarter_window fires when one deal is excluded under both gates",
        any(f["severity"] == ERROR for f in check_quarter_window(bad_res)))

    bad_res = copy.deepcopy(base_res)
    bad_res["excluded_sold"] = [{"vcode": "P1"}]
    chk("quarter_window fires when a deal is both held and excluded",
        any(f["severity"] == ERROR for f in check_quarter_window(bad_res)))

    bad_res = copy.deepcopy(base_res)
    bad_res["diagnostics"]["acquisition_date_missing"] = [{"vcode": "P1"}]
    chk("quarter_window warns on a fail-open missing Acquisition_Date",
        any(f["severity"] == WARN for f in check_quarter_window(bad_res)))

    # 6. pref-equity cross-check
    fired = pref_equity_crosscheck(base_tabs["financial"], base_res, None)
    chk("pref_equity_crosscheck reports SKIPPED rather than passing silently",
        any("skipped" in f["message"].lower() for f in fired))

    fired = pref_equity_crosscheck(base_tabs["financial"], base_res,
                                   {"P1": "Held via KCREIT and DCXVIA at 50%."})
    chk("pref_equity_crosscheck warns when the comment omits the investor",
        any(f["severity"] == WARN for f in fired))

    fired = pref_equity_crosscheck(base_tabs["financial"], base_res,
                                   {"P1": "Held via TGAM at 50%."})
    chk("pref_equity_crosscheck stays quiet when the comment names the investor",
        not [f for f in fired if f["severity"] == WARN])
    chk("pref_equity_crosscheck is never an error (soft by design)",
        not [f for f in fired if f["severity"] == ERROR])

    # 7. a throwing guardrail must not take the page down
    class Boom(dict):
        def __init__(self):
            super().__init__(sentinel=1)     # truthy, so `or {}` cannot mask it

        def get(self, *a, **k):
            raise RuntimeError("injected")

    rep = run_guardrails(base_res, {"financial": Boom()})
    chk("run_guardrails survives a throwing guardrail", isinstance(rep, dict))
    chk("a throwing guardrail is reported as a finding",
        any("did not run" in f["message"] for f in rep["findings"]))
    chk("the surviving report still carries counts", "counts" in rep)

    # 8. missing-data audit classifies all three states
    bad = copy.deepcopy(base_tabs)
    d = bad["financial"]["groups"]["Fund A"]["deals"]
    d.append({"vcode": "P2", "name": "Beta", "debt": 0.0})
    d.append({"vcode": "P3", "name": "Gamma", "debt": None})
    dbg = audit_missing_data(bad)["financial"]["fields"]["debt"]
    chk("audit separates real / zero / missing",
        dbg["value"] == 1 and dbg["zero"] == ["P2"] and dbg["missing"] == ["P3"])

    print("\n" + "=" * 100)
    passed = sum(1 for _, ok in checks if ok)
    print("RESULT: " + str(passed) + "/" + str(len(checks)) + " checks passed")
    for label, ok in checks:
        if not ok:
            print("  FAILED: " + label)
    print("=" * 100)
    return passed == len(checks)


if __name__ == "__main__":                          # pragma: no cover
    import sys
    sys.exit(0 if _selftest() else 1)
