"""Guardrail: the traceability tools answer from the dictionary, and never invent.

WHAT THIS PINS. `lookup_field` and `impact_of` are the only two tools whose whole
purpose is to say where a number comes from. An ungrounded answer here is worse
than no answer — it reads as provenance and is not — so the failure mode this
guards is not "the tool errored" but "the tool returned something plausible that
the code does not do".

Runs the SHIPPING tool wrappers (`assistant_service._tool_lookup_field` /
`_tool_impact_of`), not the service beneath them, because the wrappers are where
the pass-through discipline lives: an earlier version of `_tool_impact_of`
recomputed `consumer_count` as `len(consumers)` and would now overwrite the
file's own authoritative 16 with a list length of 4.

ASSERTED IN BOTH DIRECTIONS WHERE IT MATTERS. "economic_occ emits no LaTeX" is
satisfied by emitting no LaTeX anywhere, so the arithmetic fields are checked to
DO emit it, verbatim and on one line. "a miss returns an error" is satisfied by
erroring on everything, so six real fields are checked to resolve.

No database, no network, no API key: both tools read only
flask_app/reference/*.json.

    python scripts/traceability_tools_check.py
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services import assistant_service as A   # noqa: E402

PASSED = FAILED = 0
FAILURES: list = []


def chk(label: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        FAILURES.append(label)
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


def lookup(**inp) -> dict:
    return json.loads(A._tool_lookup_field(inp))


def impact(name: str) -> dict:
    return json.loads(A._tool_impact_of({"name": name}))


def blob(r: dict) -> str:
    return json.dumps(r, ensure_ascii=False)


# ── 1. ROE to Date — time-weighted, not IRR; the 45-day grace is NOT the
#       waterfall's pref-compounding rule of the same length ──────────────
print("\nROE to Date (one_pager.roe_to_date)")
r = lookup(field_id="one_pager.roe_to_date")
b = blob(r)
chk("annualised / time-weighted, explicitly NOT IRR", "NOT IRR" in b)
chk("formula_is_arithmetic -> formula_latex present",
    r.get("formula_is_arithmetic") and bool(r.get("formula_latex")))
chk("formula_latex is a single line (a $$ block broken by \\n never renders)",
    "\n" not in (r.get("formula_latex") or ""))
chk("the 45-day grace is carried as its own input",
    any("45-day" in i["component"] for i in r.get("inputs") or []))
chk("the grace is a LOOK-FORWARD window, asymmetric", "NOT pulled back" in b)
chk("it is distinguished from the waterfall.py pref-compounding grace",
    "waterfall.py" in b)
chk("accounting_feed is named as the distributions source", "accounting_feed" in b)
chk("no textbook net-income/equity formula has crept in",
    "net income" not in b.lower())

# ── 2. UW ROE — 7071 abs, 7073 sign-preserving. The two are NOT symmetric
#       and an answer that says they are is wrong. ───────────────────────
print("\nUW ROE (one_pager.uw_roe_to_date)")
r = lookup(field_id="one_pager.uw_roe_to_date")
b = blob(r)
chk("7071 named", "7071" in b)
chk("7071 is abs() — NOT sign-preserving", "NOT sign-preserving" in b)
chk("7073 named", "7073" in b)
chk("7073 IS sign-preserving", "sign-preserving" in b)
chk("Projected IS named as the vSource", "Projected IS" in b)
chk("formula_latex present", bool(r.get("formula_latex")))

# ── 3. debt — a bare name spans three tabs and they do NOT agree. The tool
#       must SURFACE that, not resolve it. ─────────────────────────────────
print("\ndebt (bare name -> per-tab variants)")
r = lookup(field_id="debt")
b = blob(r)
variants = r.get("variants") or []
chk("multi_tab surfaced rather than one tab guessed", r.get("multi_tab") is True)
chk("the One Pager variant is present",
    any(v["tab"] == "one_pager" for v in variants))
chk("a Snapshot variant is present",
    any(str(v["tab"]).startswith("snapshot") for v in variants))
chk("the dev hard-costs override is named", "hard" in b.lower() and "cost" in b.lower())
chk("hard costs are flagged One Pager ONLY", "One Pager only" in b)
chk("the ISBS debt accounts 2150/2152/2210 are named", "2150" in b and "2210" in b)
chk("no variant claims clean arithmetic (debt is a resolution order, not a sum)",
    all(not v.get("formula_is_arithmetic") for v in variants))
chk("narrowing by tab returns exactly one entry",
    lookup(field_id="debt", tab="one_pager").get("multi_tab") is None)

# ── 4. Total Cap — real arithmetic, three inputs ──────────────────────────
print("\nTotal Cap (one_pager.total_cap)")
r = lookup(field_id="one_pager.total_cap")
b = blob(r)
chk("arithmetic -> LaTeX block available",
    r.get("formula_is_arithmetic") and bool(r.get("formula_latex")))
chk("exactly three inputs", len(r.get("inputs") or []) == 3)
chk("inputs are debt / pref_equity / partner_equity",
    {i["component"] for i in r.get("inputs") or []}
    == {"debt", "pref_equity", "partner_equity"})
chk("the PRINTED (committed-on-dev) debt leg is noted",
    "PRINTED" in b or "committed" in b)

# ── 5. economic_occ — SIX columns on different bases. There is no single
#       expression, and reconstructing one would be fabrication. ──────────
print("\nEconomic Occupancy (one_pager.economic_occ)")
r = lookup(field_id="one_pager.economic_occ")
b = blob(r)
chk("formula_is_arithmetic is False", r.get("formula_is_arithmetic") is False)
chk("formula_latex is null — nothing to typeset", r.get("formula_latex") is None)
chk("the method is still stated in words", bool(r.get("formula")))
chk("the actual leg's 4040/4043 over 4010 is named",
    "4040" in b and "4043" in b and "4010" in b)
chk("the BUDGET leg's extra 4041 is disclosed (the legs are asymmetric)", "4041" in b)
chk("the client-side variance double-compute is disclosed",
    "client-side" in b.lower())

# ── 6. A miss says so and stops ──────────────────────────────────────────
print("\nA field that is not in the dictionary")
r = lookup(field_id="tenant_churn_rate")
chk("returns a clean error", bool(r.get("error")))
chk("invents no formula", "formula_latex" not in r)
chk("invents no source", "source_value" not in r)
chk("offers near matches", bool(r.get("did_you_mean")))
chk("offers the full id list for recovery", bool(r.get("known_field_ids")))

# ── 7. DEBT_BS_ACCTS — the constant, its two duplicate literals, and the
#       count the FILE states (not a list length) ──────────────────────────
print("\nimpact_of: DEBT_BS_ACCTS")
r = impact("DEBT_BS_ACCTS")
chk("matched the CONSTANT, not the source whose prose mentions it",
    r.get("match_type") == "shared_constant")
chk("the value is returned", r.get("value") == ["2150", "2152", "2210"])
chk("the defining location is cited", "config.py:23" in str(r.get("defined_at")))
chk("two duplicate literal locations", len(r.get("duplicates") or []) == 2)
chk("the one_pager.py hardcoded copy is named",
    any("1110" in d for d in r.get("duplicates") or []))
chk("a keep-in-sync warning is present", bool(r.get("duplicate_warning")))
chk("consumer_count is 16", r.get("consumer_count") == 16)
chk("read_by is populated", len(r.get("read_by") or []) > 0)
chk("fields_that_change is populated", len(r.get("fields_that_change") or []) > 0)
chk("blast_radius_note is NOT a second copy of duplicate_warning",
    r.get("blast_radius_note") != r.get("duplicate_warning"))

# ── 8. accounting_feed — a source, with a count that matches its own list ─
print("\nimpact_of: accounting_feed")
r = impact("accounting_feed")
chk("matched a source", r.get("match_type") == "source")
chk("consumer_count is the FILE's 16", r.get("consumer_count") == 16)
chk("feeds is populated", len(r.get("feeds") or []) > 0)
chk("the count agrees with the list the answer prints",
    r.get("consumer_count") == len(r.get("feeds") or []))
chk("a blast radius note is present", bool(r.get("blast_radius_note")))
chk("loads_to is cited", bool(r.get("loads_to")))

# ── 9. Divergences and defects are answerable at all ─────────────────────
print("\nimpact_of: known divergences / defects")
r = impact("IS_ACCOUNTS")
chk("IS_ACCOUNTS resolves", not r.get("error"))
r = impact("variance computed twice")
chk("a known DEFECT is answerable",
    r.get("match_type") == "defect" or bool(r.get("known_defects")))
r = impact("no such source anywhere")
chk("an unknown name is a clean miss", bool(r.get("error")))
chk("the miss lists the known sources for recovery", bool(r.get("known_sources")))

# ── 9b. consumer_count COMES FROM THE FILE — proved on a divergent fixture.
#
# THIS NEEDED A FIXTURE TO MEAN ANYTHING. On the shipped data every entry's
# `consumer_count` happens to equal the length of its own list (all 10 sources,
# all 4 constants), so asserting `== 16` against the real file passes whether
# the wrapper reads the file's number or recomputes `len(consumers)` — the two
# are indistinguishable, and the first version of this check was therefore
# vacuous. Injecting the recompute defect did not fail it.
#
# So the map is swapped for one where the stated count (99) CANNOT be derived
# from the two-item list beneath it. Now the recompute returns 2 and fails.
print("\nconsumer_count provenance (divergent fixture)")
from flask_app.services import data_dictionary_service as D   # noqa: E402

_real = D._DEPS
D._DEPS = {
    "meta": {},
    "sources": [{
        "name": "FIXTURE_SOURCE",
        "loads_to": "fixture",
        "provides": "fixture",
        "feeds": ["a", "b"],                 # len 2 ...
        "blast_radius_note": "fixture",
        "consumer_count": 99,                # ... but the file says 99
    }],
    "shared_constants": [{
        "name": "FIXTURE_CONST",
        "value": ["1"],
        "defined_at": "fixture.py:1",
        "read_by": ["x"],
        "fields_that_change": ["y"],         # 1 + 1 = 2 ...
        "duplicates": ["dup.py:1"],
        "duplicate_warning": "keep in sync",
        "consumer_count": 77,                # ... but the file says 77
    }],
    "known_divergences": [],
    "known_defects": [],
}
try:
    fr = impact("FIXTURE_SOURCE")
    chk("a source's consumer_count is the file's 99, not len(feeds)=2",
        fr.get("consumer_count") == 99, str(fr.get("consumer_count")))
    fc = impact("FIXTURE_CONST")
    chk("a constant's consumer_count is the file's 77, not len(lists)=2",
        fc.get("consumer_count") == 77, str(fc.get("consumer_count")))
    chk("the fixture's duplicate_warning is passed through unchanged",
        fc.get("duplicate_warning") == "keep in sync")
finally:
    D._DEPS = _real

# ── 10. The enum the model is constrained to ─────────────────────────────
print("\nThe field_id enum")
ids = A._FIELD_IDS
chk("the enum carries the 88 full tab.field ids",
    sum(1 for i in ids if "." in i) == 88)
chk("the enum ALSO carries bare names, or 'debt' could never be asked",
    "debt" in ids and "total_cap" in ids)
chk("the enum is built from the file, not written out in code",
    len(ids) > 88)

print("\n" + "=" * 62)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
if FAILURES:
    for f in FAILURES:
        print(f"  - {f}")
sys.exit(1 if FAILED else 0)
