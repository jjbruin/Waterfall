"""Guardrail: every Waterfall Setup payload must be STRICT JSON.

WHY THIS EXISTS TWICE OVER.  `e6858b5` fixed "Waterfall Setup showing 0 steps"
by scrubbing NaN out of `get_waterfall_steps`.  The scrub was written inline,
so the two copy paths kept the unscrubbed line and kept the bug, and it
resurfaced on 2026-09-10 as "the message said the waterfall copied but I do
not see the records" (Jefferson Eastchase -> Jefferson Stephens).

A NaN is emitted by `json.dumps` as a bare ``NaN`` token, which is not valid
JSON.  Axios does NOT reject it: with the default ``silentJSONParsing`` a
response that fails to parse is handed back as the raw STRING, so
``res.data.cf_wf`` is undefined, the store writes ``[]``, and nothing throws.
The UI reports a successful copy over an empty grid -- a silent failure, which
is the failure mode this project keeps paying for.

So the assertion is not "no NaN" but the stronger, end-to-end one: the payload
must survive a STRICT parse, the same way the browser parses it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {label}" + (f"  -> {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"  -> {detail}" if detail else ""))


def strict_json(obj) -> None:
    """Parse the way a browser does: a bare NaN/Infinity token is a hard error."""
    def boom(const):
        raise ValueError(f"non-JSON constant in payload: {const}")
    json.loads(json.dumps(obj), parse_constant=boom)


def main() -> int:
    from flask_app import create_app

    app = create_app()
    with app.app_context():
        from flask_app.services import data_service
        from flask_app.services import waterfall_service as ws

        data = data_service.load_all(app.config.get("DB_PATH", "waterfall.db"))
        wf = data["wf"]

        print("=== the reported case: Jefferson Eastchase -> copy ===")
        got = ws.copy_waterfall_from_entity("P0000085", wf)
        check("copy-from returns CF rows", len(got.get("cf_wf", [])) > 0,
              f"{len(got.get('cf_wf', []))} rows")
        check("copy-from returns Cap rows", len(got.get("cap_wf", [])) > 0,
              f"{len(got.get('cap_wf', []))} rows")
        try:
            strict_json(got)
            check("copy-from payload survives a STRICT parse", True)
        except ValueError as e:
            check("copy-from payload survives a STRICT parse", False, str(e))

        # The scrub must not invent data: a blanked numeric reads 0.0, and the
        # row count is unchanged from what the frame actually holds.
        src = wf[wf["vcode"].astype(str) == "P0000085"]
        check("no row is dropped by the scrub",
              len(got["cf_wf"]) + len(got["cap_wf"]) == len(src),
              f"{len(got['cf_wf'])}+{len(got['cap_wf'])} == {len(src)}")

        print()
        print("=== every entity that has a waterfall, on all three payload paths ===")
        vcodes = sorted({str(v) for v in wf["vcode"].dropna().unique()})
        bad = {"copy_from": [], "get_steps": [], "cf_to_cap": []}
        for vc in vcodes:
            for name, fn in (
                ("copy_from", lambda v: ws.copy_waterfall_from_entity(v, wf)),
                ("get_steps", lambda v: ws.get_waterfall_steps(wf, v)),
                ("cf_to_cap", lambda v: ws.copy_cf_to_cap(wf, v)),
            ):
                try:
                    strict_json(fn(vc))
                except ValueError:
                    bad[name].append(vc)

        check(f"copy_waterfall_from_entity: all {len(vcodes)} entities are strict JSON",
              not bad["copy_from"], f"offenders: {bad['copy_from'][:5]}")
        check(f"get_waterfall_steps: all {len(vcodes)} entities are strict JSON",
              not bad["get_steps"], f"offenders: {bad['get_steps'][:5]}")
        check(f"copy_cf_to_cap: all {len(vcodes)} entities are strict JSON",
              not bad["cf_to_cap"], f"offenders: {bad['cf_to_cap'][:5]}")

        print()
        print("=== the rule has ONE definition ===")
        src_text = Path("flask_app/services/waterfall_service.py").read_text(encoding="utf-8")
        check("steps_to_records exists", "def steps_to_records(" in src_text)
        # Anything building step records must route through the helper rather
        # than re-implementing the sort+dump, which is how the bug survived.
        raw = src_text.count('sort_values("iOrder").to_dict(orient="records")')
        check("no path re-implements sort+dump inline", raw == 0,
              f"{raw} inline occurrence(s)")
        for fn in ("copy_waterfall_from_entity", "copy_cf_to_cap", "get_waterfall_steps"):
            body = src_text.split(f"def {fn}(", 1)[1].split("\ndef ", 1)[0]
            check(f"{fn} routes through steps_to_records",
                  "steps_to_records(" in body)

    print()
    print(f"{PASS}/{PASS + FAIL} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
