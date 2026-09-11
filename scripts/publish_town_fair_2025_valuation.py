#!/usr/bin/env python
"""Publish Town Fair Tire Portfolio's 12/31/2025 valuation of 33,910,000.

Every step is scripted; you supply only the credential and run it. It drives the app's
own API — the same calls the /valuations screens make — so the approval is recorded as
YOU, under your name, with the audit trail the workflow is there to produce. Nothing is
written straight into the `valuations` table.

    $env:WATERFALL_API  = 'https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io'
    $env:WATERFALL_TOKEN = '<a JWT from your logged-in browser session>'
    .venv\\Scripts\\python.exe scripts\\publish_town_fair_2025_valuation.py            # dry run
    .venv\\Scripts\\python.exe scripts\\publish_town_fair_2025_valuation.py --commit   # do it

THE VALUE. 33,910,000 is the carrying basis at 12/31/2025, not a market write-up:
purchase 30,750,000 plus capital prefunded for improvements, closing costs, and accrued
pref through the reporting date. Record 71 is already classified `cost` — "Held 10 months
(< 12) at the valuation date" (the deal closed 2025-02-14) — so this
figure is CONSISTENT with that classification, not an override of it. The note written at
sign-off says so, because anyone reconciling 33.91M against the 30.75M purchase price will
otherwise reach the wrong conclusion, as this script's author did.

WHERE IT CAN STOP, and this is not a bug. `committee_approve` requires an active approval
from EVERY role in COMMITTEE_ROLES = (president, ceo, cio). One person completes step 4
only if they hold all three. If you hold fewer, the script tells you exactly which roles
are still outstanding and who must act; steps 1-3 are already saved, so whoever approves
next picks it up from there.

Steps: 1 set the value  2 compute NAV  3 analyst sign-off  4 committee approve
       5 publish -> writes the `valuations` row the One Pager reads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

RECORD_ID = 71
VCODE = "P0000107"
VALUE = 33_910_000.0
APPROVE_NOTE = ("Cost-basis carry for 2025, per the record's classification. Approving to "
                "enter the 12/31/2025 valuation that was missing from the cycle.")
NOTE = ("Carried at cost per the 2025 cycle classification. Cost basis = purchase price "
        "30,750,000 + capital prefunded for improvements + closing costs + accrued "
        "preferred return through 12/31/2025 = 33,910,000. This is the carrying basis, "
        "NOT a market write-up — do not reconcile it to the purchase price alone.")


def api(method, path, token, base, payload=None, allow_fail=False):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return True, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:500]
        try:
            body = json.loads(body).get("error", body)
        except Exception:
            pass
        if allow_fail:
            return False, {"_status": e.code, "_error": body}
        raise SystemExit(f"\nHTTP {e.code} on {method} {path}\n  {body}\n"
                         + ("  (401 = token expired; grab a fresh one.)" if e.code == 401 else
                            "  (403 = your account lacks the role this step needs.)"))
    except urllib.error.URLError as e:
        raise SystemExit(f"\nCould not reach {url}: {e.reason}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", action="store_true", help="actually perform the steps")
    ap.add_argument("--on-behalf", metavar="ROLES",
                    help="admin override: comma-separated committee seats to vote on "
                         "behalf of, e.g. 'president,ceo,cio'. Each is RECORDED as an "
                         "override under your username with the approval note.")
    args = ap.parse_args()

    base = os.environ.get("WATERFALL_API", "").strip()
    token = os.environ.get("WATERFALL_TOKEN", "").strip()

    if not args.commit:
        print("DRY RUN — nothing is written. Add --commit to perform these steps.\n")
        print(f"  record {RECORD_ID}  ({VCODE}, Town Fair Tire Portfolio, 2025 cycle)")
        print(f"  1  PUT    /records/{RECORD_ID}          concluded_value = {VALUE:,.0f}, method = 'cost'")
        print(f"  2  POST   /records/{RECORD_ID}/nav/compute")
        print(f"  3  POST   /records/{RECORD_ID}/action    sign_off")
        print(f"  4  POST   /records/{RECORD_ID}/approve   (needs president + ceo + cio)")
        print(f"  5  POST   /records/{RECORD_ID}/publish   -> writes the valuations row")
        print(f"\n  note recorded at sign-off:\n    {NOTE}")
        print("\nSet WATERFALL_API and WATERFALL_TOKEN, then re-run with --commit.")
        return 0

    if not base or not token:
        raise SystemExit("Set WATERFALL_API and WATERFALL_TOKEN first — see the docstring.")

    print(f"Publishing record {RECORD_ID} ({VCODE}) at {VALUE:,.0f}\n")

    # Who am I, and which committee roles do I hold? Answers step 4 before we get there.
    ok, perms = api("GET", "/api/valuations/permissions", token, base, allow_fail=True)
    roles = perms.get("roles", perms) if ok else {}
    print(f"  permissions: {json.dumps(roles)[:160]}")

    ok, before = api("GET", f"/api/valuations/records/{RECORD_ID}", token, base, allow_fail=True)
    if ok:
        r = before.get("record", before)
        print(f"  record before: status={r.get('status')!r} "
              f"concluded_value={r.get('concluded_value')!r} "
              f"classification={r.get('classification')!r}")

    print("\n  1  setting the concluded value ...")
    api("PUT", f"/api/valuations/records/{RECORD_ID}", token, base,
        {"concluded_value": VALUE, "method": "cost", "override_note": NOTE})
    print("     saved.")

    print("  2  computing NAV ...")
    ok, nav = api("POST", f"/api/valuations/records/{RECORD_ID}/nav/compute", token, base,
                  {}, allow_fail=True)
    print(f"     {'ok' if ok else 'FAILED: ' + str(nav.get('_error'))[:120]}")
    if not ok:
        print("     publish requires a computed NAV — stopping here, steps 1 is saved.")
        return 1

    print("  3  analyst sign-off ...")
    ok, res = api("POST", f"/api/valuations/records/{RECORD_ID}/action", token, base,
                  {"action": "sign_off"}, allow_fail=True)
    print(f"     {'ok' if ok else 'FAILED: ' + str(res.get('_error'))[:120]}")

    print("  4  committee approval ...")
    # --on-behalf casts the committee vote for seats you do not hold. Each one is
    # recorded as an admin override against the approval, under your username, with the
    # note below — the trail then reads "one person voted three seats, and why" rather
    # than "three members agreed". See scripts/valuation_admin_override_check.py.
    body = {"note": APPROVE_NOTE}
    if args.on_behalf:
        body["on_behalf_of"] = [r.strip() for r in args.on_behalf.split(",") if r.strip()]
        print(f"     casting ON BEHALF OF: {body['on_behalf_of']}  (recorded as override)")
    ok, res = api("POST", f"/api/valuations/records/{RECORD_ID}/approve", token, base,
                  body, allow_fail=True)
    if not ok:
        print(f"     NOT APPROVED: {str(res.get('_error'))[:200]}")
        print("     The committee is president + ceo + cio and EVERY required role must")
        print("     approve. Either the remaining holders approve at /valuations, or")
        print("     re-run this with --on-behalf 'president,ceo,cio' as admin.")
        print("     Steps 1-3 are saved either way.")
        return 2
    print(f"     {json.dumps(res)[:260]}")
    if res.get("has_admin_override"):
        print(f"     NOTE: seats {res.get('cast_on_behalf_roles')} were voted by "
              f"{res.get('cast_on_behalf_by')} — recorded as an override.")

    print("  5  publishing ...")
    ok, res = api("POST", f"/api/valuations/records/{RECORD_ID}/publish", token, base,
                  {}, allow_fail=True)
    if not ok:
        print(f"     NOT PUBLISHED: {str(res.get('_error'))[:200]}")
        return 3
    print(f"     {json.dumps(res)[:240]}")

    print("\nDone. The 12/31/2025 row is in `valuations`; the One Pager shows it from "
          "26Q1 onward\nand correctly leaves earlier quarters on the 12/31/2024 figure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
