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
                     "capital_accounts", "investor_delivery"], str(keys))
        # NAMED FOR THE ACT, NOT THE VENDOR. The key is stored on every tracker
        # cell, so a portal's product name in it becomes a migration the day we
        # deliver through anything else.
        chk("the delivery deliverable is not keyed to a vendor",
            "investment_cafe" not in keys and
            dict((d["key"], d) for d in wt.DELIVERABLES)
                ["investor_delivery"].get("channel") == "Investment Cafe")
        for k in ("workpapers", "financial_statements", "capital_accounts"):
            chk("%s runs Prepared -> Mgr -> CFO" % k,
                [s["key"] for s in dict(
                    (d["key"], d) for d in wt.DELIVERABLES)[k]["stages"]]
                == ["prepared", "review_1", "review_2"])
        cafe = dict((d["key"], d) for d in wt.DELIVERABLES)["investor_delivery"]
        # FIVE, not three. The FS and the capital account statements are posted
        # and checked separately before the quarter is released; squeezing that
        # into the generic three-column group would discard two columns that
        # the CFO's own sheet carries.
        chk("investor delivery keeps all five of its columns",
            [s["key"] for s in cafe["stages"]]
            == ["fs_posted", "fs_reviewed", "cas_posted", "cas_reviewed",
                "released"], str([s["key"] for s in cafe["stages"]]))

        # ---- 1b. the population is the REP tag -----------------------------
        #
        # Jim, Sep 16 2026: "the population should be driven by the REP tag, not
        # the spreadsheet I provided." So the check is that ONE place reads the
        # population and it reads the tag -- an entity tagged REP has to appear
        # whether or not anyone remembered to put it on a checklist, which is
        # exactly what the spreadsheet cannot catch.
        print("\n1b. Population")
        import inspect as _i
        src = _i.getsource(ws.sync_packages)
        chk("sync_packages selects on the REP tag", "REP" in src)
        chk("and the tracker derives its rows from packages, not a list",
            "wp_packages" in _i.getsource(wt.grid)
            and "55" not in _i.getsource(wt.grid))

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

        # ---- 10. who can be assigned --------------------------------
        #
        # The dropdown replaced a free-text box, so the failure that matters is
        # it being EMPTY of the people who do the work. Measured Sep 17 2026:
        # every account was `analyst` or `admin` and the close is prepared by
        # KH, NL and RE. Jim is adding accounting roles; until then the list has
        # to keep working from the other two sources.
        print("\n10. Preparers")
        wt.add_preparer("ZZQ", "Check Person", "accountant", engine=eng)
        people = wt.list_preparers(engine=eng)
        inis = [x["initials"] for x in people]
        chk("a maintained preparer appears", "ZZQ" in inis, str(inis[:8]))
        got = [x for x in people if x["initials"] == "ZZQ"][0]
        chk("with the name and role recorded",
            got["name"] == "Check Person" and got["wp_role"] == "accountant")
        chk("an unknown role is refused",
            "error" in wt.add_preparer("ZZR", None, "chief wizard", engine=eng))
        chk("initials are required",
            "error" in wt.add_preparer("  ", engine=eng))
        # AN ASSIGNMENT ALREADY ON A PACKAGE MUST STAY SELECTABLE, or the CFO
        # opens the screen and cannot re-pick his own entry.
        if pkgs:
            wt.set_preparer(pid, "QQX", "check", engine=eng)
            after = [x["initials"] for x in wt.list_preparers(engine=eng)]
            chk("initials already assigned stay in the list", "QQX" in after,
                str(after[:8]))
            src = [x for x in wt.list_preparers(engine=eng)
                   if x["initials"] == "QQX"][0]["source"]
            chk("and are marked as coming from use, not from the list",
                src == "in use", src)
            wt.set_preparer(pid, None, "check", engine=eng)
        chk("a username yields two-letter initials as a SEED only",
            wt._initials_from("jstewart") == "JS"
            and wt._initials_from("j.smith") == "JS")
        # Two people whose usernames give the same letters are two people.
        chk("the role list is the accounting roles, including the CFO",
            set(wt.PREPARER_ROLES) == {"accountant", "accounting_manager", "cfo"})
        wt.remove_preparer("ZZQ", engine=eng)
        chk("removing deactivates rather than deletes, so signed work resolves",
            "ZZQ" not in [x["initials"] for x in wt.list_preparers(engine=eng)]
            and "ZZQ" in [x["initials"] for x in
                          wt.list_preparers(engine=eng, include_inactive=True)])

        # ---- 11. the property column --------------------------------
        #
        # Jim, Sep 17 2026: "import the associated deal name in the Property
        # column." One rule cannot cover 58 entities, so what matters is that it
        # declines rather than guesses.
        print("\n11. Property from the deals")
        derived = wt.derive_properties(engine=eng)
        chk("derivation returns a mapping", isinstance(derived, dict))
        for k, v in derived.items():
            chk_once = v.get("property_name")
            if not chk_once:
                chk("%s has a name or is absent" % k, False)
                break
        else:
            chk("every derived entry carries a name and a basis",
                all(v.get("property_name") and v.get("basis")
                    for v in derived.values()))
        many = [v for v in derived.values() if v["basis"].endswith("deals")]
        chk("an entity holding several deals reads 'Various', the CFO's own word",
            all(v["property_name"] == "Various" for v in many), str(many[:2]))
        # THE ONE THAT WOULD BE SILENT IF WRONG: a typed value must survive.
        #
        # It has to be tested on a row that HAS a derived value to overwrite.
        # Typed onto an unresolved row the check passes vacuously — the row is
        # skipped for having no derivation, not protected for having a value —
        # which is exactly how it passed the first time it was written.
        target = None
        for r in wt.grid(cid, eng)["rows"]:
            if str(r["entityid"]).strip().upper() in derived:
                target = r["package_id"]
                break
        if target is not None:
            wt.set_property(target, "Management Company", "check", engine=eng)
            res = wt.apply_derived_properties(cid, overwrite=False, engine=eng)
            row = _row(wt.grid(cid, eng), target)
            chk("a typed property is not overwritten by a derived one",
                row["property_name"] == "Management Company",
                str(row["property_name"]))
            chk("and the run reports what it left alone",
                res["kept_existing"] >= 1, str(res))
            # overwrite=True is the deliberate act, and it must actually work.
            res2 = wt.apply_derived_properties(cid, overwrite=True, engine=eng)
            row2 = _row(wt.grid(cid, eng), target)
            chk("overwrite=True does replace it when asked",
                row2["property_name"] != "Management Company",
                "%s (%s)" % (row2["property_name"], res2))
        else:
            chk("a package with a derivable property exists to test against",
                False, "no scratch package resolves to a deal")
        res3 = wt.apply_derived_properties(cid, engine=eng)
        chk("entities it could not resolve are named, not silently skipped",
            "unresolved" in res3 and "unresolved_count" in res3)

        # ---- 11b. two hops, and the basis that makes them safe ---------
        #
        # Jim, Sep 17 2026: "build hop 2 with the basis visible." The second hop
        # halves the blanks and can be CONFIDENTLY WRONG — TGA6 is a fund that
        # happens to reach exactly one deal at two levels, so it resolves to
        # that deal where his sheet says "Various". The walk cannot tell a
        # single-purpose chain from a fund with one reachable holding, so what
        # makes it safe is that every value declares how it was reached.
        print("\n11b. Two hops")
        d1 = wt.derive_properties(engine=eng, max_hops=1)
        d2 = wt.derive_properties(engine=eng, max_hops=2)
        chk("two hops never resolves fewer than one",
            set(d1) <= set(d2), "%d vs %d" % (len(d1), len(d2)))
        chk("every derived value carries a basis and a hop count",
            all(v.get("basis") and v.get("hops") for v in d2.values()))
        chk("a one-hop answer does not claim to be deeper",
            all("levels down" not in v["basis"]
                for v in d2.values() if v["hops"] == 1))
        deep = [v for v in d2.values() if v["hops"] > 1]
        chk("a deeper answer says how deep",
            all("levels down" in v["basis"] for v in deep),
            str([v["basis"] for v in deep[:3]]))
        # Shallowest wins: an entity reaching a deal directly must not be
        # described as two levels away just because it also reaches one there.
        chk("the basis reports the SHALLOWEST level a deal was found at",
            all(v["hops"] == d1[k]["hops"] for k, v in d2.items() if k in d1))
        chk("the walk stops at max_hops",
            set(wt.derive_properties(engine=eng, max_hops=1))
            <= set(wt.derive_properties(engine=eng, max_hops=2)))

        # A VALUE AN EARLIER RUN WROTE, before the basis column existed, reads
        # as one somebody typed. Annotated only when the stored name is
        # IDENTICAL to what the walk produces, so an edited name keeps its
        # silence and stays the CFO's.
        if target is not None:
            wt.apply_derived_properties(cid, overwrite=True, engine=eng)
            with eng.begin() as _c:
                _c.execute(text("UPDATE wp_packages SET property_basis = NULL "
                                " WHERE id = :i"), {"i": target})
            _before = _row(wt.grid(cid, eng), target)
            _res = wt.apply_derived_properties(cid, engine=eng)
            _after = _row(wt.grid(cid, eng), target)
            chk("a name an earlier run wrote gets its basis back",
                _before.get("property_basis") is None
                and bool(_after.get("property_basis")), str(_res))
            chk("and the name itself is untouched",
                _after["property_name"] == _before["property_name"])
            chk("the run reports how many it annotated", _res.get("annotated", 0) >= 1)
            # An edited name must NOT be annotated: it is a decision, not a walk.
            wt.set_property(target, "Something The CFO Typed", "check", engine=eng)
            wt.apply_derived_properties(cid, engine=eng)
            _edited = _row(wt.grid(cid, eng), target)
            chk("a name that differs from the derivation stays unannotated",
                _edited["property_name"] == "Something The CFO Typed"
                and _edited.get("property_basis") is None)

        if target is not None:
            wt.apply_derived_properties(cid, overwrite=True, engine=eng)
            row_i = _row(wt.grid(cid, eng), target)
            chk("a filled property carries its basis onto the row",
                bool(row_i.get("property_basis")), str(row_i.get("property_basis")))
            # TYPING OVER IT CLEARS THE BASIS. Leaving it attached would credit
            # the CFO's decision to a walk of the commitments table.
            wt.set_property(target, "Management Company", "check", engine=eng)
            row_t = _row(wt.grid(cid, eng), target)
            chk("typing a property clears the basis",
                row_t["property_name"] == "Management Company"
                and row_t.get("property_basis") is None,
                str(row_t.get("property_basis")))

        _drop(eng, cid)
        _drop(eng, nid)

        # ---- 9. the accounting section is the CFO's ---------------------
        #
        # The gate used to be ("admin", "analyst") on the API and `admin` on the
        # screen, and the SCREEN is what actually locked the CFO out: the API
        # would have taken his writes, but the buttons were never rendered. So
        # this checks both sides, because passing one proves nothing.
        print("\n9. Who can run the close")
        # WHO IS REFUSED IS CHECKED IN scripts/accounting_access_check.py, by
        # enumerating every route in the section from the app and calling each
        # one as each role. These checks used to grep for a decorator's exact
        # text, which proved only that a string was present -- and it missed six
        # writes that had no gate at all, including exhibit deletion, because a
        # string that is absent looks the same as a rule that does not apply.
        # What is left here is the shape of the gate, which is what this file is
        # about.
        import pathlib as _pl
        from flask_app.auth.routes import ACCOUNTING_ROLES, ROLE_LEVELS

        root = _pl.Path(__file__).resolve().parents[1]
        api = (root / "flask_app" / "api" / "workpapers.py").read_text(encoding="utf-8")
        chk("no accounting endpoint is still gated without the CFO",
            '@role_required("admin", "analyst")' not in api)
        chk("the CFO can edit the close", "cfo" in ACCOUNTING_ROLES)
        chk("so can the accountants who prepare it",
            "accountant" in ACCOUNTING_ROLES
            and "accounting_manager" in ACCOUNTING_ROLES)
        # The whole reason the gate is membership rather than level.
        chk("an analyst cannot, although the level model would have let them",
            "analyst" not in ACCOUNTING_ROLES
            and ROLE_LEVELS["analyst"] == ROLE_LEVELS["cfo"])
        chk("a viewer cannot", "viewer" not in ACCOUNTING_ROLES)

        view = (root / "vue_app" / "src" / "views"
                / "WorkpapersView.vue").read_text(encoding="utf-8")
        # The defect being pinned: a control the API allows but the screen hides
        # is indistinguishable, to the person using it, from having no access.
        chk("the screen gates on running the close, not on being an admin",
            "isAdmin" not in view and "canManageClose" in view)
        chk("and it reads the shared gate rather than a list of its own",
            "auth.canEditAccounting" in view
            and "['admin', 'cfo'].includes" not in view)

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
