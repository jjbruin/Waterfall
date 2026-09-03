"""Guardrail: the TEMPORARY per-deal Operating suppression.

Hanestowne Waterstone (P0000118) reads n/a across every Operating metric by
name, because the requested outcome cannot be expressed as an ownership age:
Hanestowne at 3.4 months and Plaza Del Mar at 3.5 are THREE DAYS apart and are
to be treated differently. See ``TEMP_OPERATING_SUPPRESS``.

    python scripts/snapshot_temp_suppress_check.py

NONE OF THE THREE DEALS IN THIS CHECK EXIST IN THE LOCAL SNAPSHOT — P0000116,
P0000118 and P0000119 are all newer than ``waterfall.db`` (accounting ends
2026-06-02). So the page is assembled from injected Step 1 entries and injected
One Pager payloads carrying their real closing dates, and the before/after is
taken by emptying ``TEMP_OPERATING_SUPPRESS`` rather than by checking out an
older commit: "before" is therefore exactly this code without this constant.

A real local deal (Burton, P0000109, owned 10.0 months) rides along as the
control that the assembly itself works and that nothing else moves.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKS: list = []

#: vcode -> (name, closing date, at_close, uw_ye, projected_ye, occupancy)
#: Closing dates are back-solved from the ownership ages the report author
#: quoted at 26Q2. The NOI/occupancy figures are plausible stand-ins: this check
#: asserts which cells are WITHHELD, never what they would have said.
DEALS = {
    "P0000118": ("Hanestowne Waterstone", "2026-03-19", 2.10, 2.40, 2.30, 92.0),
    "P0000116": ("Plaza Del Mar", "2026-03-16", 3.05, 3.30, 3.20, 95.0),
    "P0000119": ("Presidential Arms", "2026-05-13", 1.40, 1.60, 1.55, 88.0),
    "P0000109": ("Burton Retail Portfolio", "2025-08-28", 4.00, 4.40, 4.30, 96.0),
}

NOI_KEYS = ("at_close", "uw_ye", "projected_ye")
METRICS = ("econ_occ_display", "expected_growth_display", "actual_growth_display")
Q_END = dt.date(2026, 6, 30)


def chk(label, cond, detail=""):
    CHECKS.append(bool(cond))
    print("  [{}] {}".format("PASS" if cond else "FAIL", label)
          + ("\n           " + detail if detail else ""))


def _provider(vcode, quarter):
    _nm, closed, ac, uw, py, occ = DEALS[str(vcode).strip().upper()]

    def blk(v):
        return {"at_close": v, "uw_ye": v, "actual_ye": v, "ytd_actual": v}

    return {
        "general": {"date_closed": closed},
        "property_performance": {
            "noi": {"at_close": ac, "uw_ye": uw, "actual_ye": py},
            # _payload_unpopulated needs revenue/expenses/noi non-zero
            # somewhere, or the column reads as "no row for this deal".
            "revenue": blk(10.0),
            "expenses": blk(6.0),
            "economic_occ": {"at_close": occ, "uw_ye": occ,
                             "actual_ye": occ, "ytd_actual": occ},
        },
    }


def _assemble(suppress_set):
    """The real ``assemble_operating``, with the constant swapped in."""
    from flask_app.services import portfolio_snapshot_operating as OP
    resolved = {
        "investor_code": "TGAM", "investor_name": "TGAM",
        "groups": {"Individual Investments": [
            {"vcode": v, "name": meta[0], "investment_strategy": "Core Plus"}
            for v, meta in DEALS.items()]},
        "flagged": [],
    }
    saved = OP.TEMP_OPERATING_SUPPRESS
    try:
        OP.TEMP_OPERATING_SUPPRESS = frozenset(suppress_set)
        return OP.assemble_operating(
            "TGAM", "2026-Q2", resolved=resolved,
            one_pager_provider=_provider, comment_loader=lambda *a: {})
    finally:
        OP.TEMP_OPERATING_SUPPRESS = saved


def _cells(row):
    """Every metric cell on the row, as {label: value}."""
    out = {k.replace("_display", ""): row.get(k) for k in METRICS}
    out.update({"noi." + k: (row.get("noi_display") or {}).get(k)
                for k in NOI_KEYS})
    return out


def main() -> int:
    from flask_app import create_app
    from flask_app.services import portfolio_snapshot_operating as OP
    app = create_app()
    with app.app_context():
        before = _assemble(set())            # this code without the constant
        after = _assemble({"P0000118"})      # what ships
        grp = "Individual Investments"
        B = {r["vcode"].upper(): r for r in before["groups"][grp]}
        A = {r["vcode"].upper(): r for r in after["groups"][grp]}
        NA = OP.NA_LABEL

        print("\n1. Before / after, cell by cell")
        for vc, meta in DEALS.items():
            mo = OP.months_owned({"general": {"date_closed": meta[1]}}, Q_END)
            b, a = _cells(B[vc]), _cells(A[vc])
            bn = sum(1 for v in b.values() if v == NA)
            an = sum(1 for v in a.values() if v == NA)
            print("      {:<24} {:5.2f}mo   before {}/{} n/a   ->   "
                  "after {}/{} n/a".format(meta[0], mo, bn, len(b), an, len(a)))

        print("\n2. Hanestowne reads n/a across EVERY Operating metric")
        showing = {k: v for k, v in _cells(A["P0000118"]).items() if v != NA}
        chk("all six cells suppressed", not showing, "still showing: "
            + str(showing))
        chk("and it was showing real figures before",
            all(v != NA for v in _cells(B["P0000118"]).values()),
            "the age rule does not reach it at 3.4 months")
        chk("the row says which rule blanked it",
            A["P0000118"].get("temp_suppressed") is True
            and A["P0000118"].get("insufficient_history") is False,
            "temp_suppressed=True, insufficient_history=False — the age rule "
            "is not credited with a catch it did not make")
        chk("and carries a TEMPORARY flag a reader can see",
            any("TEMPORARY per-deal suppression" in f
                for f in (A["P0000118"].get("flags") or [])),
            str([f[:58] for f in (A["P0000118"].get("flags") or [])]))
        chk("the occupancy basis names the override, not the age rule",
            A["P0000118"].get("econ_occ_basis")
            == "TEMPORARY per-deal suppression",
            str(A["P0000118"].get("econ_occ_basis")))

        print("\n3. The raw values are NOT destroyed")
        chk("raw NOI survives underneath",
            (A["P0000118"].get("noi") or {}).get("uw_ye") == 2.40,
            "display only — a frozen payload keeps its real figures and every "
            "audit still sees them")
        chk("raw occupancy survives too",
            (A["P0000118"].get("econ_occ") or {}).get("at_close") == 92.0)

        print("\n4. Plaza Del Mar keeps its values — 3 days older, opposite call")
        p = _cells(A["P0000116"])
        chk("no cell suppressed", all(v != NA for v in p.values()),
            str({k: v for k, v in p.items() if v == NA}))
        chk("not flagged by the override",
            A["P0000116"].get("temp_suppressed") is False)
        chk("and identical to before", p == _cells(B["P0000116"]))

        print("\n5. Presidential Arms was ALREADY n/a — by the age rule")
        pa = _cells(A["P0000119"])
        chk("all cells n/a", all(v == NA for v in pa.values()))
        chk("credited to the age rule, not the override",
            A["P0000119"].get("insufficient_history") is True
            and A["P0000119"].get("temp_suppressed") is False,
            "owned {} months".format(A["P0000119"].get("months_owned")))
        chk("and unchanged by this commit", pa == _cells(B["P0000119"]))

        print("\n6. The suppressed cells do not feed subtotals")
        bs = before["subtotals"][grp]
        as_ = after["subtotals"][grp]
        for i, k in enumerate(NOI_KEYS):
            drop = (bs["noi"][k] or 0) - (as_["noi"][k] or 0)
            want = DEALS["P0000118"][2 + i]
            chk("subtotal NOI {} falls by exactly Hanestowne's {}".format(k, want),
                abs(drop - want) < 1e-9,
                "{:.2f} -> {:.2f}  (-{:.2f})".format(
                    bs["noi"][k], as_["noi"][k], drop))
        chk("and one fewer row contributes to each NOI column",
            all(as_["noi_contributors"][k] == bs["noi_contributors"][k] - 1
                for k in NOI_KEYS),
            str({k: (bs["noi_contributors"][k], as_["noi_contributors"][k])
                 for k in NOI_KEYS}))
        chk("the subtotal counts it as suppressed",
            as_["suppressed_count"] == bs["suppressed_count"] + 1,
            "{} -> {}".format(bs["suppressed_count"], as_["suppressed_count"]))
        chk("the row is still COUNTED on the page",
            as_["deal_count"] == bs["deal_count"] == len(DEALS),
            "withheld, not removed — the reader can still see the row")
        chk("weighted occupancy drops Hanestowne's reading",
            as_["econ_occ"]["at_close"] != bs["econ_occ"]["at_close"],
            "{:.2f} -> {:.2f}".format(bs["econ_occ"]["at_close"],
                                      as_["econ_occ"]["at_close"]))

        print("\n7. No other deal moves")
        moved = [v for v in DEALS
                 if v != "P0000118" and _cells(A[v]) != _cells(B[v])]
        chk("every other row is cell-for-cell identical", not moved,
            ", ".join(moved))
        chk("Burton, a real local deal, is untouched and showing",
            all(v != NA for v in _cells(A["P0000109"]).values()),
            "owned {} months".format(A["P0000109"].get("months_owned")))

        print("\n8. Scoped, reversible, reported")
        chk("exactly one vcode is hardcoded",
            len(OP.TEMP_OPERATING_SUPPRESS) == 1,
            str(sorted(OP.TEMP_OPERATING_SUPPRESS)))
        chk("the age threshold is untouched",
            OP.INSUFFICIENT_HISTORY_MONTHS == 3.0,
            str(OP.INSUFFICIENT_HISTORY_MONTHS))
        chk("emptying the set restores every figure",
            _cells(_assemble(set())["groups"][grp][0])
            == _cells(B["P0000118"]),
            "one deletion to retire")
        chk("diagnostics count the override",
            after["diagnostics"].get("temp_suppressed") == 1
            and before["diagnostics"].get("temp_suppressed") == 0,
            "{} -> {}".format(before["diagnostics"].get("temp_suppressed"),
                              after["diagnostics"].get("temp_suppressed")))

    passed = sum(CHECKS)
    print("\n  {}/{} checks passed".format(passed, len(CHECKS)))
    return 0 if passed == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
