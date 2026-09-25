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
# PHRASING MOVED, THE FACT DID NOT. This read `"NOT pulled back" in b` against
# the pre-2026-09-25 wording. The dictionary was reworded into plain language
# and now says "...contributions and returns of capital in that window are not —
# this asymmetry is intentional". Asserting the retired literal would fail on
# correct content, so the check tests the CLAIM: the window pulls things back,
# and it is explicitly asymmetric. Both halves are required — "pulled back" alone
# is satisfied by a symmetric window, which is the error this exists to catch.
chk("the grace is a LOOK-FORWARD window, asymmetric",
    "pulled back" in b.lower() and "asymmetr" in b.lower())
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
# Case-insensitive for the same reason as the grace check above: the plain-
# language rewrite lowercased the "NOT". The asymmetry between 7071 and 7073 is
# the thing being pinned, and it survives verbatim ("not sign-preserving").
chk("7071 is abs() — NOT sign-preserving", "not sign-preserving" in b.lower())
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
# Was `"One Pager only" in b`; the rewrite says "only on the One Pager". Rather
# than swap one literal for another, this now asserts the scoping sits on the
# HARD-COSTS INPUT ITSELF — stronger than a blob substring, which a stray
# mention of the phrase on any unrelated input would satisfy.
_hard_inputs = [i
                for v in (r.get("variants") or [])
                for i in (v.get("inputs") or [])
                if "hard" in json.dumps(i, ensure_ascii=False).lower()
                and "cost" in json.dumps(i, ensure_ascii=False).lower()]
chk("hard costs are flagged One Pager ONLY",
    bool(_hard_inputs) and any(
        "one pager" in (i.get("note") or "").lower()
        and "only" in (i.get("note") or "").lower()
        for i in _hard_inputs))
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
# DERIVED FROM THE FILE, NOT HARDCODED. This pinned the literal 88 and so failed
# the moment the dictionary was regenerated at 91 fields — a maintenance failure,
# not a real one. What actually matters is that EVERY field in the dictionary
# reaches the enum: a loader that silently dropped entries would leave the model
# unable to ask for them. The `> 0` guard stops it passing vacuously if the file
# ever fails to load and both sides collapse to zero.
_dict_field_ids = [f["field_id"]
                   for f in json.load(open(
                       os.path.join(os.path.dirname(os.path.dirname(
                           os.path.abspath(__file__))),
                           "flask_app", "reference", "data_dictionary.json"),
                       encoding="utf-8"))["fields"]
                   if f.get("field_id")]
chk(f"the enum carries all {len(_dict_field_ids)} full tab.field ids",
    len(_dict_field_ids) > 0
    and sum(1 for i in ids if "." in i) == len(_dict_field_ids),
    f"enum={sum(1 for i in ids if '.' in i)} file={len(_dict_field_ids)}")
chk("the enum ALSO carries bare names, or 'debt' could never be asked",
    "debt" in ids and "total_cap" in ids)
chk("the enum is built from the file, not written out in code",
    len(ids) > 88)

# ── 11. trace_field_value — the live per-deal value trace ────────────────
#
# STRUCTURE ONLY, AND DELIBERATELY SO. Local SQLite is a stub, so the real
# figures cannot be exercised here; they are verified post-deploy against live
# data. What IS checked without a database is the contract the answer format
# depends on — and it is checked in BOTH directions, because "returns a value
# and an input list" is satisfied by returning nonsense. The paired assertions
# are that a breakdown which does NOT add up reports `reconciles: False`, that
# an unchecked one reports None rather than False, and that the two cases the
# tool must refuse (a Snapshot field with no investor, a field with no
# breakdown) refuse instead of inventing an answer.
print("\ntrace_field_value (structure; real figures verified post-deploy)")

from flask_app.services import field_trace_service as TR   # noqa: E402

_spec = next((t for t in A.TOOLS if t["name"] == "trace_field_value"), None)
chk("the tool is registered", _spec is not None)
chk("deal_id and field_id are required",
    bool(_spec) and set(_spec["input_schema"]["required"]) == {"deal_id", "field_id"})
chk("quarter and investor_code are accepted",
    bool(_spec) and {"quarter", "investor_code"} <= set(
        _spec["input_schema"]["properties"]))

# A payload standing in for the One Pager's, so no database is needed. The
# figures are chosen to add up, so a reconciliation FAILURE below is a real one.
_PAYLOAD = {
    "cap_stack": {"debt": 30e6, "pref_equity": 10e6, "partner_equity": 10e6,
                  "total_cap": 50e6, "current_valuation": 80e6,
                  "sold_suppressed": False, "debt_pct": 0.6},
    "property_performance": {"noi": {"actual_ye": 4e6, "ytd_actual": 3e6},
                             "dscr": {"ytd_actual": 1.45}},
    "pe_performance": {"roe_to_date": 0.082},
    "general": {},
}
_real_payload = TR._one_pager_payload
_real_roe = TR._roe_breakdown
try:
    TR._one_pager_payload = lambda v, q: (_PAYLOAD, "26Q2")

    # (a) a calculated field returns a value AND its inputs, without raising.
    TR._roe_breakdown = lambda vc, q, val, uw=False: {
        "available": True, "source_engine": "test",
        "components": [
            {"component": "Distributions (numerator)", "value": 12.3e6},
            {"component": "Weighted-avg capital (denominator)", "value": 45.1e6},
            {"component": "years", "value": 2.3}]}
    r = TR.trace_field_value("P0000109", "one_pager.roe_to_date", quarter="26Q2")
    chk("roe_to_date returns a value", r.get("value") == 0.082)
    chk("roe_to_date returns its input list", len(r.get("inputs") or []) == 3)
    chk("each traced input carries its dictionary source",
        all(i.get("source") for i in (r.get("inputs") or [])))
    chk("the field's formula_latex comes through for rendering",
        bool(r.get("formula_latex")))

    # (b) THE PAGE'S VALUE WINS WHEN THE ENGINES DISAGREE. The breakdown is
    # withheld with a reason rather than shown against a different number.
    TR._roe_breakdown = lambda vc, q, val, uw=False: {
        "available": False, "reason": "engines disagree"}
    r = TR.trace_field_value("P0000109", "one_pager.roe_to_date", quarter="26Q2")
    chk("a withheld ROE breakdown keeps the PAGE's value",
        r.get("value") == 0.082)
    chk("a withheld ROE breakdown shows no inputs and says why",
        r.get("inputs") == [] and bool(r.get("breakdown_unavailable_reason")))
    chk("an unchecked breakdown is reconciles=None, NOT False",
        r.get("reconciles") is None)

    # (c) components that DO add up reconcile...
    r = TR.trace_field_value("P0000109", "one_pager.total_cap", quarter="26Q2")
    chk("total_cap reconciles when its parts add up", r.get("reconciles") is True)
    chk("total_cap names all three parts", len(r.get("inputs") or []) == 3)

    # (d) ...and components that do NOT are reported, not smoothed over. This is
    # the paired direction: without it, always returning True would pass (c).
    _PAYLOAD["cap_stack"]["total_cap"] = 99e6
    r = TR.trace_field_value("P0000109", "one_pager.total_cap", quarter="26Q2")
    chk("a breakdown that does NOT add up reports reconciles=False",
        r.get("reconciles") is False, str(r.get("reconciles")))
    chk("and still reports the PUBLISHED value, not the recomputed one",
        r.get("value") == 99e6)
    _PAYLOAD["cap_stack"]["total_cap"] = 50e6

    # (e) a SOLD-suppressed deal reconciles on a zero debt leg.
    _PAYLOAD["cap_stack"]["sold_suppressed"] = True
    _PAYLOAD["cap_stack"]["total_cap"] = 20e6
    r = TR.trace_field_value("P0000109", "one_pager.total_cap", quarter="26Q2")
    chk("a sold-suppressed deal reports a zero debt leg and still reconciles",
        r.get("reconciles") is True
        and [i for i in r["inputs"] if i["component"] == "debt"][0]["value"] == 0.0)
    _PAYLOAD["cap_stack"]["sold_suppressed"] = False
    _PAYLOAD["cap_stack"]["total_cap"] = 50e6

    # (f) a field with no breakdown says so rather than inventing one.
    r = TR.trace_field_value("P0000109", "one_pager.debt", quarter="26Q2")
    chk("a field with no breakdown returns no invented inputs",
        r.get("inputs") == [] and r.get("breakdown_available") is False)

    # (g) DSCR is five bases, and its denominator is honestly absent.
    r = TR.trace_field_value("P0000109", "one_pager.dscr", quarter="26Q2")
    chk("dscr is reported per basis", r.get("multi_basis") is True
        and len(r.get("bases") or []) == 5)
    chk("dscr's unpublished debt service is marked unavailable, not derived",
        all(b["inputs"][1]["available"] is False for b in r["bases"]))
finally:
    TR._one_pager_payload = _real_payload
    TR._roe_breakdown = _real_roe

# (h) a Snapshot field is investor-scoped and must ASK, never guess.
r = TR.trace_field_value("P0000109", "snapshot_loan.ltv", quarter="26Q2")
chk("a Snapshot field with no investor_code refuses rather than guessing one",
    bool(r.get("error")) and "investor" in r["error"].lower())

# (i) the handler never raises — the model gets JSON whatever happens.
_out = json.loads(A._tool_trace_field_value({"deal_id": "", "field_id": ""}))
chk("the tool wrapper returns JSON on bad input instead of raising",
    bool(_out.get("error")))

# (j) STRINGS ARE NEVER READ AS FIGURES. The NUMERIC string is the assertion
# that carries the weight: "n/a" and "Dev" fail to parse as a float anyway, so a
# _num that coerced strings would still return None for them and this check
# would pass while broken. "1.45" is the case that separates the two — the
# Snapshot writes display strings into the same cells that carry ratios.
chk("a numeric STRING is not read as a number",
    TR._num("1.45") is None, str(TR._num("1.45")))
chk("'n/a' and 'Dev' are not read as numbers",
    TR._num("n/a") is None and TR._num("Dev") is None and TR._num(True) is None)

print("\n" + "=" * 62)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
if FAILURES:
    for f in FAILURES:
        print(f"  - {f}")
sys.exit(1 if FAILED else 0)
