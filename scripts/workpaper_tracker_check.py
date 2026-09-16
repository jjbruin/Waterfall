"""The close tracker: shape, validation, ordering, and what carries forward.

WHY EACH SECTION EXISTS. The tracker is the CFO's record of who signed what and
when, so the failures that matter are the quiet ones -- a sign-off that saves
under a date nobody typed, a "cleared" cell that still reads as signed, a
carry-forward that copies last quarter's approvals onto this quarter's work.
None of those look wrong on screen.

Run:  .venv/Scripts/python.exe scripts/workpaper_tracker_check.py
Exit 1 on any failure. Uses a scratch cycle in the local database and removes
it afterwards, so it never touches a real close.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FAILS = []
CHECKS = [0]


def chk(label, cond, detail=""):
    CHECKS[0] += 1
    if cond:
        print("   ok   %s" % label)
    else:
        print("   FAIL %s%s" % (label, ("  -- " + detail) if detail else ""))
        FAILS.append(label)


def main() -> int:
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.db import get_engine
    from flask_app.services import workpaper_service as ws
    from flask_app.services import workpaper_tracker as wt

    app = create_app()
    with app.app_context():
        eng = get_engine()
        ws.ensure_tables(eng)
        wt.ensure_tracker_tables(eng)

        # ---- 1. the shape matches the spreadsheet's column groups ----------
        print("\n1. Deliverables and stages")
        keys = [d["key"] for d in wt.DELIVERABLES]
        chk("four deliverables, in sign-off order",
            keys == ["workpapers", "financial_statements",
                     "capital_accounts", "investment_cafe"], str(keys))
        for k in ("workpapers", "financial_statements", "capital_accounts"):
            chk("%s runs Prepared -> Mgr -> CFO" % k,
                [s["key"] for s in dict(
                    (d["key"], d) for d in wt.DELIVERABLES)[k]["stages"]]
                == ["prepared", "review_1", "review_2"])
        cafe = dict((d["key"], d) for d in wt.DELIVERABLES)["investment_cafe"]
        # FIVE, not three. The FS and the capital account statements are posted
        # and checked separately before the quarter is released; squeezing that
        # into the generic three-column group would discard two columns that
        # the CFO's own sheet carries.
        chk("Investment Cafe keeps all five of its columns",
            [s["key"] for s in cafe["stages"]]
            == ["fs_posted", "fs_reviewed", "cas_posted", "cas_reviewed",
                "released"], str([s["key"] for s in cafe["stages"]]))

        # ---- 2. dates: parsed, or refused. Never guessed. ------------------
        print("\n2. Date handling")
        for raw, want in (("2026-07-03", "2026-07-03"), ("7/3/2026", "2026-07-03"),
                          ("07/03/26", "2026-07-03"), ("", None), (None, None)):
            chk("norm_date(%r) -> %r" % (raw, want), wt.norm_date(raw) == want,
                repr(wt.norm_date(raw)))
        # The spreadsheet's own cells look like this. Parsing "KH 7/7/26" as a
        # date would silently attach a sign-off to a date nobody typed.
        chk("the spreadsheet's free-text cell is REFUSED, not guessed",
            wt.norm_date("KH 7/7/26") is None, repr(wt.norm_date("KH 7/7/26")))

        # ---- 3. a scratch cycle ------------------------------------------
        print("\n3. Writes")
        cyc = ws.create_cycle("ZZ tracker check", "2026-06-30", "check")
        cid = cyc.get("id") or cyc.get("cycle_id")
        ws.sync_packages(cid)
        with eng.connect() as conn:
            pkgs = conn.execute(text(
                "SELECT id FROM wp_packages WHERE cycle_id = :c"),
                {"c": cid}).fetchall()
        if not pkgs:
            print("   (no REP entities in this database -- write checks skipped)")
            _drop(eng, cid)
            return _report()
        pid = int(pkgs[0][0])

        chk("bad deliverable refused",
            "error" in wt.set_target(pid, "nope", "2026-07-03"))
        chk("bad stage refused",
            "error" in wt.set_signoff(pid, "workpapers", "released", "KH",
                                      "2026-07-07"))
        chk("unreadable target date refused",
            "error" in wt.set_target(pid, "workpapers", "KH 7/7/26"))
        chk("non-numeric order refused",
            "error" in wt.set_order(pid, "third"))
        chk("negative order refused", "error" in wt.set_order(pid, -1))

        wt.set_order(pid, 7, "check")
        wt.set_preparer(pid, "KH", "check")
        wt.set_target(pid, "workpapers", "7/3/2026", "check")
        wt.set_signoff(pid, "workpapers", "prepared", "KH", "7/7/2026",
                       None, "check")
        row = _row(wt.grid(cid), pid)
        chk("order saved", row["sort_order"] == 7, repr(row["sort_order"]))
        chk("preparer saved as initials", row["preparer"] == "KH")
        wp = row["deliverables"][0]
        chk("target normalised to ISO", wp["target_date"] == "2026-07-03",
            repr(wp["target_date"]))
        chk("sign-off records WHO and WHEN",
            wp["stages"][0]["signed_by"] == "KH"
            and wp["stages"][0]["signed_on"] == "2026-07-07")

        # ---- 4. clearing must DELETE, not null --------------------------
        print("\n4. Clearing a sign-off")
        wt.set_signoff(pid, "workpapers", "prepared", "", "", None, "check")
        row = _row(wt.grid(cid), pid)
        chk("cleared cell reads as unsigned",
            row["deliverables"][0]["stages"][0]["signed"] is False)
        with eng.connect() as conn:
            left = conn.execute(text(
                "SELECT COUNT(*) FROM wp_tracker_cell WHERE package_id = :p "
                "  AND deliverable = 'workpapers' AND stage = 'prepared'"),
                {"p": pid}).scalar()
        # A surviving row with null columns would make "is this signed?" depend
        # on WHICH column you test, which is how two screens start disagreeing.
        chk("and leaves NO row behind", int(left or 0) == 0, "rows=%s" % left)

        # ---- 5. sequence is reported, never refused ----------------------
        print("\n5. Out of sequence")
        res = wt.set_signoff(pid, "capital_accounts", "prepared", "KH",
                             "2026-08-12", None, "check")
        chk("a sign-off ahead of its turn still SAVES", res.get("ok") is True,
            str(res))
        g = wt.grid(cid)
        row = _row(g, pid)
        chk("and is reported on the row",
            any("Capital Account" in c for c in row["out_of_sequence"]),
            str(row["out_of_sequence"]))
        chk("and in the diagnostics",
            any(r["entityid"] == row["entityid"]
                for r in g["diagnostics"]["out_of_sequence_rows"]))

        # ---- 6. overdue needs a target ----------------------------------
        print("\n6. Overdue")
        wt.set_target(pid, "financial_statements", "", "check")
        row = _row(wt.grid(cid), pid)
        fs = [d for d in row["deliverables"]
              if d["key"] == "financial_statements"][0]
        # No target is not the same as on time. A row with no date set must not
        # render as green, or the CFO cannot see what he has not scheduled.
        chk("no target date is not 'on time'", fs["overdue"] is False
            and fs["target_date"] is None)
        wp = row["deliverables"][0]
        chk("a passed target with work outstanding IS late", wp["overdue"] is True)

        # ---- 7. ordering -------------------------------------------------
        print("\n7. The CFO's order")
        g = wt.grid(cid)
        orders = [r["sort_order"] for r in g["rows"]]
        placed = [o for o in orders if o is not None]
        chk("ordered rows come first and ascending",
            placed == sorted(placed), str(orders))
        chk("unplaced rows sort LAST, not first",
            all(o is not None for o in orders[:len(placed)]), str(orders))
        chk("unordered rows are counted",
            g["diagnostics"]["unordered_count"] == len(orders) - len(placed))
        if len(g["rows"]) > 1:
            other = [r["package_id"] for r in g["rows"]
                     if r["package_id"] != pid][0]
            wt.set_order(other, 7, "check")
            d = wt.grid(cid)["diagnostics"]["duplicate_orders"]
            chk("a duplicate order number is reported",
                any(x["order"] == 7 for x in d), str(d))
            wt.set_order(other, None, "check")

        # ---- 8. carry forward moves the ARRANGEMENT only ------------------
        print("\n8. Carry forward")
        nxt = ws.create_cycle("ZZ tracker check 2", "2026-09-30", "check")
        nid = nxt.get("id") or nxt.get("cycle_id")
        ws.sync_packages(nid)
        cf = wt.carry_forward(cid, nid, "check")
        g2 = wt.grid(nid)
        moved = [r for r in g2["rows"] if r["sort_order"] == 7]
        chk("order and preparer carried", bool(moved) and moved[0]["preparer"] == "KH",
            str(cf))
        # THE POINT OF THE CHECK. Copying a sign-off forward would assert that
        # this quarter's workpapers were reviewed because last quarter's were.
        signed = sum(r["signed_count"] for r in g2["rows"])
        chk("NO sign-off carried into the new quarter", signed == 0,
            "%d signed cells appeared" % signed)
        targets = [d["target_date"] for r in g2["rows"]
                   for d in r["deliverables"] if d["target_date"]]
        chk("NO target date carried either", not targets, str(targets[:4]))

        _drop(eng, cid)
        _drop(eng, nid)

        # ---- 9. the accounting section is the CFO's ---------------------
        #
        # The gate used to be ("admin", "analyst") on the API and `admin` on the
        # screen, and the SCREEN is what actually locked the CFO out: the API
        # would have taken his writes, but the buttons were never rendered. So
        # this checks both sides, because passing one proves nothing.
        print("\n9. Who can run the close")
        import pathlib as _pl
        from flask_app.auth.routes import role_level, ROLE_LEVELS

        def admits(allowed, role):
            known = [role_level(r) for r in allowed if r in ROLE_LEVELS]
            need = min(known) if known else None
            return not (role not in allowed
                        and (need is None or role_level(role) < need))

        root = _pl.Path(__file__).resolve().parents[1]
        api = (root / "flask_app" / "api" / "workpapers.py").read_text(encoding="utf-8")
        chk("no accounting endpoint is still gated without the CFO",
            '@role_required("admin", "analyst")' not in api)
        gates = api.count('@role_required("admin", "cfo", "analyst")')
        chk("every accounting write names the CFO", gates >= 10, "found %d" % gates)
        chk("a CFO is admitted", admits(("admin", "cfo", "analyst"), "cfo"))
        chk("a viewer is not", not admits(("admin", "cfo", "analyst"), "viewer"))

        view = (root / "vue_app" / "src" / "views"
                / "WorkpapersView.vue").read_text(encoding="utf-8")
        # The defect being pinned: a control the API allows but the screen hides
        # is indistinguishable, to the person using it, from having no access.
        chk("the screen gates on running the close, not on being an admin",
            "isAdmin" not in view and "canManageClose" in view)
        chk("and the CFO is in that gate",
            "['admin', 'cfo'].includes" in view)

    return _report()


def _row(g, pid):
    return [r for r in g["rows"] if r["package_id"] == pid][0]


def _drop(eng, cid):
    from sqlalchemy import text
    with eng.begin() as conn:
        ids = [int(r[0]) for r in conn.execute(text(
            "SELECT id FROM wp_packages WHERE cycle_id = :c"),
            {"c": cid}).fetchall()]
        for pid in ids:
            conn.execute(text("DELETE FROM wp_tracker_cell WHERE package_id = :p"),
                         {"p": pid})
            conn.execute(text("DELETE FROM wp_tracker_target WHERE package_id = :p"),
                         {"p": pid})
            conn.execute(text("DELETE FROM wp_package_steps WHERE package_id = :p"),
                         {"p": pid})
            conn.execute(text("DELETE FROM wp_events WHERE package_id = :p"),
                         {"p": pid})
        conn.execute(text("DELETE FROM wp_packages WHERE cycle_id = :c"), {"c": cid})
        conn.execute(text("DELETE FROM wp_cycle_steps WHERE cycle_id = :c"), {"c": cid})
        conn.execute(text("DELETE FROM wp_cycles WHERE id = :c"), {"c": cid})


def _report():
    print("\n%d checks, %d failed." % (CHECKS[0], len(FAILS)))
    if FAILS:
        for f in FAILS:
            print("   FAILED: %s" % f)
        return 1
    print("The tracker records who signed and when, refuses a date it cannot")
    print("read, clears rather than nulls, reports sequence instead of blocking")
    print("it, and carries the arrangement forward without the approvals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
