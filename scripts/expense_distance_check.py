"""Guardrail: the distance wizard -- driving miles between places, measured by Google.

MAKES NO GOOGLE CALLS. The two HTTP functions are replaced by a recorder that
answers like Google does, so what is checked is what is ASKED (an airport code
asked as an airport; the key in a header for Routes, never in the browser) and
what is DONE with the answer (miles from metres, a round trip driven back as its
own leg, the measurement stored server-side and only the owner's usable).

THE PHL CASE IS PINNED. With the real key on Oct 2 2026, Geocoding "PHL" returned
"Philippines". The wizard must ask "PHL airport", and a three-letter code that
Google matches to something that is not an airport must stop and say so.
"""
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_passed, _failed = [], []
ASKED = []


def chk(label, cond, detail=""):
    (_passed if cond else _failed).append(label)
    print("   %s %s%s" % ("ok  " if cond else "FAIL", label,
                          ("   [%s]" % (detail,)) if detail and not cond else ""))


PLACES = {
    "PHL airport": ("Philadelphia International Airport, Philadelphia, PA", "P_PHL", ["airport"]),
    "Philadelphia City Hall": ("1400 John F Kennedy Blvd, Philadelphia, PA", "P_CH", ["premise"]),
    "Market at Poplar, Memphis": ("5050 Poplar Ave, Memphis, TN", "P_MP", ["shopping_mall"]),
    "MSY airport": ("Louis Armstrong New Orleans Intl Airport, Kenner, LA", "P_MSY", ["airport"]),
    "ZZZ airport": ("Zzyzx Rd, Baker, CA", "P_ZZZ", ["route"]),
}
DIST = {("P_PHL", "P_CH"): 19151, ("P_CH", "P_PHL"): 19500,
        ("P_PHL", "P_MP"): 1700000, ("P_MP", "P_MSY"): 640000}


def fake_get(url):
    import urllib.parse
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    addr, key = q["address"][0], q.get("key", [""])[0]
    ASKED.append(("geocode", addr, key))
    if key != "test-key":
        return {"status": "REQUEST_DENIED", "error_message": "The provided API key is invalid."}
    if addr not in PLACES:
        return {"status": "ZERO_RESULTS", "results": []}
    name, pid, types = PLACES[addr]
    return {"status": "OK", "results": [{"formatted_address": name, "place_id": pid, "types": types}]}


def fake_post(url, body, headers):
    ASKED.append(("routes", body, dict(headers)))
    pts = [body["origin"]] + body.get("intermediates", []) + [body["destination"]]
    legs = []
    for a, b in zip(pts, pts[1:]):
        d = DIST.get((a["placeId"], b["placeId"]))
        if d is None:
            return {"routes": []}
        legs.append({"distanceMeters": d})
    return {"routes": [{"distanceMeters": sum(l["distanceMeters"] for l in legs), "legs": legs}]}


def main():
    tmp = tempfile.mkdtemp(prefix="expense_dist_")
    os.environ.pop("DATABASE_URL", None)
    os.environ["DB_PATH"] = os.path.join(tmp, "check.db")
    os.environ.pop("GOOGLE_MAPS_API_KEY", None)

    import jwt
    import pandas as pd
    from sqlalchemy import text
    from flask_app import create_app
    from flask_app.auth.models import create_user, list_users
    from flask_app.db import get_engine
    from flask_app.services import data_service, expense_distance as ed
    from database import PROTECTED_TABLES

    ed._http_get, ed._http_post = fake_get, fake_post
    app = create_app()
    app.config["DATABASE_URL"] = None
    client = app.test_client()
    data_service.get_data = lambda *a, **k: {"inv": pd.DataFrame([
        {"vcode": "P0000001", "Investment_Name": "Apple Self Storage", "Portfolio_Name": "",
         "Sale_Status": "", "Sale_Date": None, "Lifecycle": "Stable"}])}
    people = {"admin": "admin", "emp": "analyst", "other": "analyst"}
    with app.app_context():
        for u, role in people.items():
            create_user(u, "pw-" + u, role=role)
        with get_engine().begin() as c:
            c.execute(text('CREATE TABLE IF NOT EXISTS gl_accounts ("ACCTNUM" TEXT, "ACCTNAME" TEXT, "TYPE" TEXT)'))
            c.execute(text("INSERT INTO gl_accounts VALUES ('MR53000011', 'Other Expense: Travel', 'I')"))
        ids = {u["username"]: u["id"] for u in list_users()}

    def H(name):
        return {"Authorization": "Bearer " + jwt.encode(
            {"sub": str(ids[name]), "username": name, "role": people[name],
             "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
            app.config["JWT_SECRET"], algorithm="HS256")}

    def call(method, path, who, body=None):
        r = client.open("/api/expenses" + path, method=method, json=body, headers=H(who))
        return r.status_code, (r.get_json(silent=True) or {})

    def measure(stops, rt=False, who="emp"):
        return call("POST", "/distance", who, {"stops": stops, "round_trip": rt})

    print("1. Without a key it says so, and miles can still be typed")
    st, b = measure(["PHL", "Philadelphia City Hall"])
    chk("no key: refused, saying the wizard is not set up", st == 400 and "not set up" in b.get("error", ""), b)
    chk("...and nothing was sent to Google", not ASKED)
    os.environ["GOOGLE_MAPS_API_KEY"] = "test-key"

    print("\n2. What is asked of Google")
    st, b = measure(["PHL", "Philadelphia City Hall"])
    chk("PHL is asked as 'PHL airport' -- bare, Google returns the Philippines",
        ("geocode", "PHL airport", "test-key") in ASKED, ASKED[:2])
    chk("a lowercase code is still an airport code", ed.query_for(" msy ") == "MSY airport")
    chk("a longer name is asked as typed", ed.query_for("Market at Poplar, Memphis") == "Market at Poplar, Memphis")
    rq = [a for a in ASKED if a[0] == "routes"][-1]
    chk("Routes gets the key in a header, and resolved places, not text",
        rq[2].get("X-Goog-Api-Key") == "test-key" and rq[1]["origin"] == {"placeId": "P_PHL"})
    chk("Routes is asked to DRIVE", rq[1]["travelMode"] == "DRIVE")
    chk("the key never comes back to the browser", "test-key" not in str(b))

    print("\n3. What comes back")
    chk("each stop says what it resolved to",
        [s["resolved"] for s in b["stops"]] == ["Philadelphia International Airport, Philadelphia, PA",
                                               "1400 John F Kennedy Blvd, Philadelphia, PA"], b.get("stops"))
    chk("miles are metres over 1609.344, to one decimal (19,151 m = 11.9)", b["miles"] == 11.9, b.get("miles"))
    chk("the measurement is stored, with an id", isinstance(b["route_id"], int))
    st, rt = measure(["PHL", "Philadelphia City Hall"], rt=True)
    chk("a round trip drives back as its own leg (19,151 + 19,500 m = 24.0)",
        rt["miles"] == 24.0 and rt["legs"] == [11.9, 12.1], rt)
    st, multi = measure(["PHL", "Market at Poplar, Memphis", "MSY"])
    chk("a stop in between: one route through it", multi["miles"] == round(2340000 / 1609.344, 1)
        and len(multi["legs"]) == 2, multi)

    print("\n4. What it refuses, and says why")
    st, z = measure(["ZZZ", "Philadelphia City Hall"])
    chk("a code Google matches to something that is not an airport stops and says so",
        z["miles"] is None and "not an airport" in (z.get("error") or ""), z)
    st, n = measure(["Nowhere At All", "PHL"])
    chk("a place Google cannot find says so", n["miles"] is None and "found no place" in (n.get("error") or ""))
    st, nr = measure(["MSY", "Philadelphia City Hall"])
    chk("no driving route is reported, not zero", nr["miles"] is None and "could not measure" in (nr.get("error") or ""))
    chk("one stop is not a trip", measure(["PHL"])[0] == 400)
    chk("eleven stops is too many", measure(["PHL"] * 11)[0] == 400)
    os.environ["GOOGLE_MAPS_API_KEY"] = "bad-key"
    st, bad = measure(["PHL", "Philadelphia City Hall"])
    chk("a key Google rejects is reported as refused, not as no place",
        bad["miles"] is None and "refused" in (bad.get("error") or ""), bad.get("error"))
    os.environ["GOOGLE_MAPS_API_KEY"] = "test-key"

    print("\n5. A line carries the measurement")
    call("PUT", "/mileage-rates", "admin", {"effective_date": "2026-01-01", "rate": "0.725"})
    rid = call("POST", "/reports", "emp", {"period_start": "2026-09-01", "period_end": "2026-09-30"})[1]["id"]
    base = {"line_date": "2026-09-10", "category_account": "MR53000011", "purpose": "Property Visit - Existing",
            "deal_code": "P0000001", "comment": "Site visit", "receipt": "N",
            "no_receipt_reason": "Mileage — measured route"}
    st, r = call("POST", "/reports/%d/lines" % rid, "emp", {**base, "miles": "24.0", "route_id": rt["route_id"]})
    ln = r["lines"][-1]
    chk("the line points at its route and shows it", ln["route_id"] == rt["route_id"]
        and ln["route"]["summary"] == "PHL → Philadelphia City Hall → PHL, 24.0 mi measured", ln.get("route"))
    chk("the amount is still miles x the rate", ln["amount"] == round(24.0 * 0.725, 2))
    chk("matching miles raise no warning", not r["check"]["by_line"][str(ln["id"])]["warnings"]
        if str(ln["id"]) in r["check"]["by_line"] else not r["check"]["by_line"][ln["id"]]["warnings"])
    st, r = call("PUT", "/reports/%d/lines/%d" % (rid, ln["id"]), "emp",
                 {**base, "miles": "30", "route_id": rt["route_id"]})
    chk("miles typed over a measurement are kept, and the line says they differ",
        r["lines"][-1]["miles"] == 30 and any("differ from the measured route" in w
                                              for w in r["check"]["warnings"] + sum(
                                                  (v["warnings"] for v in r["check"]["by_line"].values()), [])),
        r["check"]["warnings"])
    st, b = call("POST", "/reports/%d/lines" % rid, "emp", {**base, "amount": "12", "route_id": rt["route_id"]})
    chk("a route on a non-mileage line is refused", st == 400)
    ost, o = measure(["PHL", "Philadelphia City Hall"], who="other")
    st, b = call("POST", "/reports/%d/lines" % rid, "emp", {**base, "miles": "11.9", "route_id": o["route_id"]})
    chk("another employee's measurement cannot be used", st == 400 and "not one of your" in b.get("error", ""))
    chk("er_routes is protected", "er_routes" in PROTECTED_TABLES)

    print("\n%d passed, %d failed" % (len(_passed), len(_failed)))
    return 1 if _failed else 0


if __name__ == "__main__":
    sys.exit(main())
