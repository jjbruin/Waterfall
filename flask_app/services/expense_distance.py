"""Expense reports -- the distance wizard: driving miles between places.

Jim, Oct 2 2026: "a distance wizard to calculate mileage between addresses,
landmarks like airport identifiers, or city names". Google, on Jim's call:
the Routes API measures the drive, the Geocoding API says what each stop was
understood as. The key is the container secret `google-maps-key`, read as
GOOGLE_MAPS_API_KEY, and never reaches the browser.

A BARE AIRPORT CODE IS ASKED AS AN AIRPORT. Measured with the real key on Oct
2 2026: Geocoding "PHL" returns "Philippines". Three letters on their own are
taken to mean the airport and asked as "PHL airport"; the stop then SAYS what
it resolved to, so a wrong match is seen before the miles are used.

THE MEASUREMENT IS THE SERVER'S. Each one is stored (`er_routes`) with its
stops, what they resolved to and the miles, and a line points at it. So
"measured" on a line means Google measured it, not that the browser said so;
miles typed over a measurement are kept, and the line says they differ.

Miles are Google's driving distance, metres over 1609.344, to one decimal.
A round trip drives back to the start as its own leg rather than doubling the
outbound, because the return route can differ.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

from sqlalchemy import text

from flask_app.services import expense_service as ex

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
METRES_PER_MILE = 1609.344
MAX_STOPS = 10
_AIRPORT_CODE = re.compile(r"^[A-Za-z]{3}$")

_DONE: set = set()


def ensure_tables(engine) -> None:
    ex.ensure_tables(engine)


def ensure_route_tables(engine) -> None:
    """Called by `expense_service.ensure_tables`, after er_lines exists."""
    key = str(getattr(engine, "url", "")) or id(engine)
    if key in _DONE:
        return
    pk = ("SERIAL PRIMARY KEY" if engine.dialect.name == "postgresql"
          else "INTEGER PRIMARY KEY AUTOINCREMENT")
    with engine.begin() as c:
        c.execute(text(f"""CREATE TABLE IF NOT EXISTS er_routes (
            id          {pk},
            user_id     INTEGER NOT NULL,
            stops_json  TEXT NOT NULL,
            round_trip  BOOLEAN NOT NULL,
            miles       DOUBLE PRECISION NOT NULL,
            legs_json   TEXT,
            created_at  TEXT)"""))
    from sqlalchemy import inspect
    if "route_id" not in {c["name"] for c in inspect(engine).get_columns("er_lines")}:
        with engine.begin() as c:
            c.execute(text("ALTER TABLE er_lines ADD COLUMN route_id INTEGER"))
    _DONE.add(key)


def _key() -> Optional[str]:
    return (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip() or None


# The two calls, kept tiny so the guardrail can stand in for Google.
def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def _http_post(url: str, body: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read() or b"{}") | {"_status": e.code}
        except ValueError:
            return {"_status": e.code}


def query_for(stop: str) -> str:
    """What is asked of Google for what the employee typed."""
    s = stop.strip()
    return "%s airport" % s.upper() if _AIRPORT_CODE.match(s) else s


def resolve(stop: str, key: str) -> dict:
    """{input, query, resolved, place_id, is_airport, error} for one stop."""
    q = query_for(stop)
    out = {"input": stop.strip(), "query": q, "resolved": None, "place_id": None,
           "is_airport": False, "error": None}
    try:
        g = _http_get(GEOCODE_URL + "?" + urllib.parse.urlencode({"address": q, "key": key}))
    except Exception as e:
        out["error"] = "Google could not be reached (%s)." % e
        return out
    st = g.get("status")
    if st == "ZERO_RESULTS" or (st == "OK" and not g.get("results")):
        out["error"] = "Google found no place called %r. Try an address or a fuller name." % stop
        return out
    if st != "OK":
        out["error"] = "Google refused the lookup (%s%s)." % (
            st, (": " + g["error_message"]) if g.get("error_message") else "")
        return out
    top = g["results"][0]
    out.update(resolved=top.get("formatted_address"), place_id=top.get("place_id"),
               is_airport="airport" in (top.get("types") or []))
    if _AIRPORT_CODE.match(stop.strip()) and not out["is_airport"]:
        out["error"] = ("%s was read as an airport code, but Google matched %s, which is not "
                        "an airport. Type the airport's name instead." % (stop.strip().upper(),
                                                                          out["resolved"]))
    return out


def measure(engine, actor, stops: List[str], round_trip: bool = False) -> dict:
    """Resolve the stops, drive them in order, store the measurement."""
    ensure_tables(engine)
    stops = [s for s in (stops or []) if (s or "").strip()]
    if len(stops) < 2:
        raise ValueError("Give at least a start and a destination.")
    if len(stops) > MAX_STOPS:
        raise ValueError("At most %d stops." % MAX_STOPS)
    key = _key()
    if not key:
        raise ValueError("The distance wizard is not set up on this server (no Google Maps "
                         "key). Type the miles instead.")
    resolved = [resolve(s, key) for s in stops]
    bad = [r for r in resolved if r["error"]]
    if bad:
        return {"stops": resolved, "miles": None, "route_id": None,
                "error": " ".join(r["error"] for r in bad)}
    pts = [{"placeId": r["place_id"]} for r in resolved]
    if round_trip:
        pts.append({"placeId": resolved[0]["place_id"]})
    body = {"origin": pts[0], "destination": pts[-1], "travelMode": "DRIVE"}
    if len(pts) > 2:
        body["intermediates"] = pts[1:-1]
    r = _http_post(ROUTES_URL, body, {"Content-Type": "application/json", "X-Goog-Api-Key": key,
                                      "X-Goog-FieldMask": "routes.distanceMeters,routes.legs.distanceMeters"})
    routes = r.get("routes") or []
    if not routes or not routes[0].get("distanceMeters"):
        msg = ((r.get("error") or {}).get("message") or "no driving route was found between "
               "these places")
        return {"stops": resolved, "miles": None, "route_id": None,
                "error": "Google could not measure the drive: %s." % msg}
    miles = round(routes[0]["distanceMeters"] / METRES_PER_MILE, 1)
    legs = [round((l.get("distanceMeters") or 0) / METRES_PER_MILE, 1)
            for l in routes[0].get("legs") or []]
    with engine.begin() as c:
        rid = c.execute(text(
            "INSERT INTO er_routes (user_id, stops_json, round_trip, miles, legs_json, created_at) "
            "VALUES (:u, :s, :rt, :m, :l, :at) RETURNING id"),
            {"u": int(actor["id"]), "s": json.dumps(resolved), "rt": bool(round_trip), "m": miles,
             "l": json.dumps(legs), "at": ex._now()}).scalar()
    return {"stops": resolved, "round_trip": bool(round_trip), "miles": miles, "legs": legs,
            "route_id": rid, "error": None}


def route_row(engine, route_id) -> Optional[dict]:
    ensure_tables(engine)
    with engine.connect() as c:
        r = c.execute(text("SELECT * FROM er_routes WHERE id = :i"),
                      {"i": int(route_id)}).mappings().first()
    if not r:
        return None
    d = dict(r)
    d["stops"] = json.loads(d.pop("stops_json") or "[]")
    d["legs"] = json.loads(d.pop("legs_json") or "[]")
    d["round_trip"] = bool(d["round_trip"])
    return d


def summary(route: dict) -> str:
    """'PHL -> Market at Poplar -> PHL, 23.8 mi' as the line shows it."""
    names = [s.get("input") or s.get("resolved") for s in route["stops"]]
    if route["round_trip"]:
        names.append(names[0])
    return "%s, %s mi measured" % (" → ".join(names), route["miles"])
