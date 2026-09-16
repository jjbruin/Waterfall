"""A deal must not be reported as having no waterfall while one exists.

THE BUG THIS PINS. `ownership_chain_service` looked a deal's waterfall up by its
PROPERTY code alone. That is right for EASTCH (P0000085, 12 steps) and wrong for
3rd Ave & Indian School, whose six steps are filed under the InvestmentID
`3RDAVE` while `P3RDAVE` has none. The ownership chain therefore reported 3rd Ave
as unconfigured and offered a "Set up P3RDAVE" link — a code the waterfall setup
dropdown does not even carry, so the link landed on an empty page. Jim, Sep 16
2026: it "brought me to the waterfall setup page but did not select the entity".

WHY IT NEEDED A GUARDRAIL RATHER THAN A FIX ALONE. "No waterfall" is a legitimate
state — most upstream entities genuinely have none — so the screen looked correct
while it was wrong, and nothing distinguished the two from the outside. The only
way to catch it is to ask the database the question the screen is not asking:
does this deal have steps under the OTHER code?

WHAT THIS ASSERTS, per deal:
  1. If the chain says a deal has no waterfall, no code for that deal has steps.
  2. The link's target code is the one the steps are actually filed under.
  3. Deals filed under BOTH codes are reported, not silently half-used.

Section 3 is not a failure. Two sets of steps for one deal is a data question
(45th & Main and Burton Retail carry 17 and 18 under each of two codes locally),
and this prints it so it is asked rather than discovered.

Run:  .venv/Scripts/python.exe scripts/ownership_waterfall_key_check.py
Exit 1 if any deal misreports.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> int:
    from flask_app import create_app
    from flask_app.services import ownership_chain_service as ocs

    app = create_app()
    with app.app_context():
        src = ocs._Source()
        deals = ocs.list_pe_investments()
        if not deals:
            print("No PE investments in this database — nothing to check.")
            return 0

        failures, dups = [], []
        for r in deals:
            iid = ocs._norm(r["entity_id"])
            d = src.deal_by_investment.get(iid) or {}
            vc = ocs._norm(d.get("vcode") or "")
            n_vc = int(src.wf_step_counts.get(vc, 0)) if vc else 0
            n_iid = int(src.wf_step_counts.get(iid, 0))
            name = (r.get("name") or iid)[:34]

            # 1 + 2: what the screen reports must match what exists.
            reported = bool(r.get("has_waterfall"))
            exists = bool(n_vc or n_iid)
            if exists and not reported:
                failures.append(
                    "%-34s reported NO waterfall, but %s has %d steps"
                    % (name, vc if n_vc else iid, n_vc or n_iid))
            elif reported:
                code = ocs._norm(r.get("waterfall_code") or "")
                if int(src.wf_step_counts.get(code, 0)) == 0:
                    failures.append(
                        "%-34s links to %s, which has NO steps (vcode=%d, "
                        "invID=%d)" % (name, code, n_vc, n_iid))

            # 3: reported, not a failure.
            if n_vc and n_iid and vc != iid:
                dups.append((name, vc, n_vc, iid, n_iid))

        print("Deals checked: %d" % len(deals))
        print("Misreported  : %d" % len(failures))
        for f in failures:
            print("   FAIL  " + f)

        if dups:
            print()
            print("Filed under BOTH codes (%d) — reported, not merged:" % len(dups))
            for name, vc, a, iid, b in dups:
                print("   %-34s %s=%d  %s=%d" % (name, vc, a, iid, b))
            print("   Only the vcode's steps are used. Whether the two agree is")
            print("   a data question, not something this module can decide.")

        if failures:
            print()
            print("A deal reported as unconfigured while its steps exist sends an")
            print("analyst to set up a waterfall that is already there, under a")
            print("code the setup dropdown does not carry.")
            return 1
        print()
        print("Every deal's reported waterfall status matches the database, and")
        print("every link points at the code the steps are filed under.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
