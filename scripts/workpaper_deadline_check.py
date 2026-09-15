"""A close deadline the CFO cannot have meant must not be stored.

The case: "GL detail reviewed" carried a due date of 2020-01-01 on a cycle
whose period ended 2026-06-30 (found on screen, Sep 14 2026). Nothing
rejected it, and every package in the cycle then rendered that step overdue
in red from the moment it was typed -- which is how a tracker stops meaning
anything.

The rule is REJECT WHAT CANNOT BE TRUE, WARN WHAT IS MERELY ODD:

  refused   not a date; earlier than the period BEGAN
  warned    more than a year after the period end; out of sequence against
            the deadlines already set on other steps
  allowed   clearing a deadline, always

A warning SAVES. The app is not the authority on how long a close takes,
and refusing an unusual-but-deliberate deadline would make the CFO fight
the grid.

THE BOUND IS THE PERIOD START, NOT THE PERIOD END (relaxed Sep 15 2026).
The first version refused anything before period_end, which would have
turned away a legitimate pre-close prep step -- bank statements requested,
confirmations sent -- that is due while the quarter is still running. A
deadline inside the period it closes is now allowed; one from before the
period existed is still refused.

Fails against the code as it stood on Sep 14 2026, which stored anything.
"""
import sys
from datetime import date, timedelta

sys.path.insert(0, ".")

from flask_app import create_app                                  # noqa: E402
from flask_app.services import workpaper_service as ws            # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def refused(cycle_id, key, value):
    try:
        ws.validate_due_date(cycle_id, key, value)
        return None
    except ValueError as e:
        return str(e)


app = create_app()
with app.app_context():
    cycles = ws.list_cycles()
    if not cycles:
        print("No close cycle in this database — nothing to validate against.")
        sys.exit(0)
    cyc = cycles[0]
    cid, period_end = cyc["id"], str(cyc["period_end"])[:10]
    print(f"Cycle {cid}: {cyc['period_label']}, period ended {period_end}")
    key = ws.STEP_TEMPLATE[1]["key"]

    start = ws.period_start(cid, date.fromisoformat(period_end)).isoformat()
    print(f"Period being closed: {start} to {period_end}")

    print("Refused — cannot be true")
    msg = refused(cid, key, "2020-01-01")
    check("a deadline before the period began", msg is not None, msg or "")
    check("  and the message names the period",
          bool(msg) and start in msg and period_end in msg, msg or "")
    check("the day before the period started",
          refused(cid, key,
                  (date.fromisoformat(start) - timedelta(days=1)).isoformat())
          is not None)
    check("not a date at all", refused(cid, key, "not-a-date") is not None)
    check("a real-looking but impossible date",
          refused(cid, key, "2026-02-30") is not None)
    check("an unknown cycle", refused(999999, key, "2026-08-01") is not None)

    print("Allowed — a deadline that could be meant")
    try:
        w = ws.validate_due_date(cid, key, "2026-08-15")
        check("six weeks after period end, no warning", w == [], str(w))
    except ValueError as e:
        check("six weeks after period end, no warning", False, str(e))
    check("clearing a deadline", ws.validate_due_date(cid, key, None) == [])
    check("clearing with an empty string", ws.validate_due_date(cid, key, "") == [])
    check("the period end itself", refused(cid, key, period_end) is None)

    # THE POINT OF THE RELAXATION: a prep step due while the quarter is
    # still running. Refusing this was the reason the rule was loosened.
    check("the first day of the period", refused(cid, key, start) is None)
    mid = date.fromisoformat(start) + (date.fromisoformat(period_end)
                                       - date.fromisoformat(start)) / 2
    check("a pre-close prep step mid-period", refused(cid, key, mid.isoformat()) is None,
          mid.isoformat())

    print("Warned — odd, but saved")
    w = ws.validate_due_date(cid, key, "2028-06-30")
    check("more than a year out warns", any("typo" in x for x in w), str(w))
    check("  and it is a warning, not a refusal", isinstance(w, list))

    print("Out of sequence warns against deadlines already set")
    order = {s_["key"]: i for i, s_ in enumerate(ws.STEP_TEMPLATE)}
    dated = [s_ for s_ in ws.cycle_steps(cid) if s_.get("due_date")]
    case = None
    for anchor in dated:
        try:
            anchor_due = date.fromisoformat(str(anchor["due_date"])[:10])
        except ValueError:
            continue
        day_before = anchor_due - timedelta(days=1)
        if day_before.isoformat() < start:
            continue          # no room between the period start and the anchor
        later = next((s_ for s_ in ws.STEP_TEMPLATE
                      if order[s_["key"]] > order[anchor["key"]]), None)
        if later:
            case = (later["key"], day_before.isoformat(), anchor["label"])
            break
    if case is None:
        print("  SKIP  no usable pair of deadlines in this cycle")
    else:
        step_key, value, anchor_label = case
        w = ws.validate_due_date(cid, step_key, value)
        check(f"a later step due before '{anchor_label}' is noticed",
              any("comes earlier" in x for x in w), str(w))
        check("  and it is saved anyway, not refused",
              refused(cid, step_key, value) is None)

    print("The writer enforces it, not just the checker")
    before = {s_["key"]: s_.get("due_date") for s_ in ws.cycle_steps(cid)}
    try:
        ws.set_step_due_date(cid, key, "2020-01-01", "guardrail")
        check("set_step_due_date refuses an impossible date", False, "it was stored")
    except ValueError:
        check("set_step_due_date refuses an impossible date", True)
    after = {s_["key"]: s_.get("due_date") for s_ in ws.cycle_steps(cid)}
    check("  and the stored value is untouched", after == before,
          f"{before.get(key)} -> {after.get(key)}")

    # VALIDATION IS WRITE-TIME ONLY. A deadline typed before this existed
    # stays where it is -- this check names any it finds rather than
    # pretending the cycle is clean.
    stale = [f"{s_['label']}={s_['due_date']}" for s_ in ws.cycle_steps(cid)
             if s_.get("due_date") and str(s_["due_date"])[:10] < start]
    if stale:
        print(f"  NOTE  deadline(s) predating this rule, still stored: {stale}")

    print("How the period start is inferred")
    check("a quarter end starts that quarter",
          ws.period_start(999999, date(2025, 6, 30)) == date(2025, 4, 1))
    check("a non-quarter month end starts that month",
          ws.period_start(999999, date(2025, 5, 31)) == date(2025, 5, 1))
    check("a mid-month end falls back to a year",
          ws.period_start(999999, date(2025, 5, 14)) == date(2024, 5, 14))

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("All deadline checks passed.")
