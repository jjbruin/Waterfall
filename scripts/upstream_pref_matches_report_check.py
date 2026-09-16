"""Guardrail: the upstream analysis uses the VETTED pref balance, not its own.

WHY THIS EXISTS. `seed_states_from_accounting` accrues pref one way (Act/365
Fixed). The Pref Balance Detail report accrues it another (Act/Act, 366 in a
leap year). That report is the calculation the firm has built and vetted.

The upstream screen used the seeding figure, disagreed with the report, and I
explained the difference away in a footnote. Jim, twice: "why are you trying to
recreate a calculation engine that we have already built and vetted? Can't we
rely on what we have built?" Two numbers for one fact is the defect; a note
explaining which to believe is not a fix.

So this asserts EQUALITY, deal by deal and investor by investor: whatever the
report says, the upstream analysis says. It is deliberately independent of what
the right number IS -- the report owns that -- so it holds on any database,
including one whose accounting differs from production's.

Run:  .venv/Scripts/python.exe scripts/upstream_pref_matches_report_check.py
"""
import pathlib
import sys
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask_app import create_app                                   # noqa: E402
from flask_app.services import data_service                        # noqa: E402
from flask_app.services import ownership_service as OS             # noqa: E402
from flask_app.services.reports_service import build_pref_balance_detail  # noqa: E402
from loaders import load_waterfalls                                # noqa: E402

AS_OF = date(2026, 9, 16)
FAIL = []
CHECKED = [0]   # a list, so the module-level loop can bump it


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


app = create_app()
with app.app_context():
    data = data_service.get_data()
    acct, inv, wf = data.get("acct"), data.get("inv"), data.get("wf")
    if acct is None or acct.empty:
        print("SKIP - no accounting data in this database")
        sys.exit(0)

    wf_steps = load_waterfalls(wf)

    # Every deal that has a waterfall AND accounting is a candidate; a handful
    # is enough to catch a rewiring, and the run stays quick.
    with_wf = sorted({str(v).strip() for v in wf_steps["vcode"].unique()
                      if str(v).strip().upper().startswith("P")})
    tested = 0
    for vcode in with_wf:
        if tested >= 6:
            break
        res = OS.run_upstream_analysis(
            entity_id=vcode, distribution_amount=1_000_000.0,
            relationships_raw=data.get("relationships_raw"), wf=wf, inv=inv,
            wf_type="CF_WF", acct=acct, as_of=AS_OF)
        if res.get("error") or not res.get("opening_states"):
            continue
        tested += 1

        # A WATERFALL CANNOT DISTRIBUTE MORE THAN IT HAS. Jim, Sep 16 2026:
        # "How can we allocate $123,179 when the cash flow distribution was only
        # $100,000." It does not reproduce on this database, so the property is
        # asserted directly across every deal and amount this check reaches.
        for amt in (100_000.0, 5_000_000.0, 25_000_000.0):
            rr = OS.run_upstream_analysis(
                entity_id=vcode, distribution_amount=amt,
                relationships_raw=data.get("relationships_raw"), wf=wf, inv=inv,
                wf_type="CF_WF", acct=acct, as_of=AS_OF)
            if rr.get("error"):
                continue
            check(not rr.get("reconciliation_errors"),
                  f"{vcode} @ {amt:,.0f}: " + "; ".join(rr.get("reconciliation_errors") or []))
            dealt = float(rr.get("deal_allocated_total") or 0.0)
            check(abs(dealt - amt) < 0.01,
                  f"{vcode} @ {amt:,.0f}: the steps allocate {dealt:,.2f} — "
                  f"{dealt - amt:+,.2f} against what was distributed")
            bent = float(rr.get("beneficiary_total") or 0.0)
            check(abs(bent - amt) < 0.01,
                  f"{vcode} @ {amt:,.0f}: beneficial owners receive {bent:,.2f} — "
                  f"{bent - amt:+,.2f}; every dollar must arrive exactly once")
            # Each LEVEL of the upstream detail reconciles; the flat list does
            # not, and must not be expected to.
            for lv in rr.get("upstream_total_by_level") or []:
                check(abs(float(lv["allocated"]) - amt) < 0.01,
                      f"{vcode} @ {amt:,.0f}: upstream level {lv['level']} totals "
                      f"{float(lv['allocated']):,.2f}, not the distribution")

        check(res.get("pref_source") == "Pref Balance Detail report",
              f"{vcode}: pref_source is {res.get('pref_source')!r} — the screen "
              f"is not taking its balances from the vetted report")

        deal_steps = wf_steps[wf_steps["vcode"] == vcode]
        for row in res["opening_states"]:
            pc = row["entity_id"]
            try:
                d = build_pref_balance_detail(vcode, pc, AS_OF, acct, inv, deal_steps)
            except Exception as e:
                FAIL.append(f"{vcode}/{pc}: the report itself failed — {str(e)[:90]}")
                continue
            vetted = (d or {}).get("header", {}).get("accrued_pref")
            if vetted is None:
                continue
            CHECKED[0] += 1
            shown = float(row["accrued_pref"])
            # To the cent. A tolerance here would re-admit exactly the kind of
            # "close enough, here is why" this check exists to forbid.
            check(abs(shown - float(vetted)) < 0.01,
                  f"{vcode}/{pc}: the screen shows {shown:,.2f} of accrued pref "
                  f"and the Pref Balance Detail report says {float(vetted):,.2f}. "
                  f"They must be the same number, not two opinions.")

    check(tested > 0, "no deal could be analysed, so nothing was actually compared")

# The rationalisation must be gone, not merely unused.
import inspect  # noqa: E402
src = inspect.getsource(OS.run_upstream_analysis)
check("pref_convention_note" not in src,
      "the day-count footnote is back; it papers over a disagreement that "
      "should not exist now that both take the same figure")
check("build_pref_balance_detail" in src,
      "run_upstream_analysis no longer calls the vetted report")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print(f"OK - {CHECKED[0]} investor balances across the deals tested match the Pref "
      f"Balance Detail report exactly; the screen reports the vetted figure "
      f"rather than a second calculation of it")
