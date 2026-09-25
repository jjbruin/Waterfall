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
import re
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
# Derived, not the old literal 88: the enum must exceed the id count because it
# carries the bare names as well, and pinning a number here just goes stale
# every time the dictionary is regenerated.
chk("the enum is built from the file, not written out in code",
    len(ids) > len(_dict_field_ids))

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

# ── 12. Single-engine ROE components, and the DSCR denominator ───────────
#
# BOTH DIRECTIONS THROUGHOUT. "the breakdown is shown" is satisfied by showing
# one unconditionally, so the withhold path is asserted to still refuse; "the
# denominator is published" is satisfied by inventing one, so a column without
# it is asserted to stay unavailable.
print("\nSingle-engine ROE components + DSCR denominator")

# (k) THE REGRESSION GUARD. `get_pe_performance` swapped calculate_roe for
# calculate_roe_detailed; the displayed ROE must not move. It cannot, because
# detailed DELEGATES to calculate_roe for the scalar — this pins that, including
# the degenerate cases where the early returns could diverge.
from metrics import calculate_roe, calculate_roe_detailed          # noqa: E402
from datetime import date as _d                                    # noqa: E402

_roe_cases = [
    ([], [], _d(2020, 1, 1), _d(2024, 1, 1)),
    ([(_d(2020, 1, 1), -1000.0)], [], _d(2020, 1, 1), _d(2020, 1, 1)),
    ([(_d(2020, 1, 1), 1000.0)], [], _d(2020, 1, 1), _d(2024, 1, 1)),
    ([(_d(2020, 1, 1), -1_000_000.0), (_d(2022, 6, 30), 250_000.0)],
     [(_d(2022, 6, 30), 250_000.0)], _d(2020, 1, 1), _d(2024, 1, 1)),
    ([(_d(2019, 3, 1), -5_000_000.0), (_d(2021, 9, 1), -2_000_000.0),
      (_d(2023, 1, 15), 900_000.0)],
     [(_d(2023, 1, 15), 900_000.0), (_d(2023, 6, 1), -20_000.0)],
     _d(2019, 3, 1), _d(2026, 6, 30)),
]
chk("calculate_roe_detailed returns the IDENTICAL scalar calculate_roe does",
    all(calculate_roe(*c) == calculate_roe_detailed(*c)["roe"] for c in _roe_cases))

# (p) THE TOLERANCE MUST DISCRIMINATE RATIOS. An absolute floor sized for money
# (half a cent) calls a ROE of 0.082 equal to 0.078 — a 5% error — so the
# reconciliation flag would be decorative on every ratio field.
chk("a 5% gap between two ratios is NOT called agreement",
    TR._agree(0.082, 0.078) is False)
chk("float noise between two ratios IS called agreement",
    TR._agree(0.082, 0.082 * (1 + 1e-9)) is True)
chk("two genuine zeros agree", TR._agree(0.0, 0.0) is True)

_P2 = {
    "cap_stack": {}, "general": {},
    "property_performance": {
        "noi": {"ytd_actual": 4.2e6, "ytd_budget": 4.0e6, "actual_ye": 8.5e6,
                "uw_ye": 9.0e6, "at_close": 7.0e6},
        "dscr": {"ytd_actual": 1.45, "ytd_budget": 1.40, "actual_ye": 1.50,
                 "uw_ye": 1.60, "at_close": None},
        # at_close's denominator is deliberately ABSENT — the honest path.
        "debt_service": {"ytd_actual": 4.2e6 / 1.45, "ytd_budget": 4.0e6 / 1.40,
                         "actual_ye": 8.5e6 / 1.50, "uw_ye": 9.0e6 / 1.60,
                         "at_close": None},
    },
    "pe_performance": {
        "roe_to_date": 0.082,
        "roe_components": {
            "total_cf_distributions": 12_300_000.0,
            "weighted_avg_capital": 45_100_000.0,
            "years": 12_300_000.0 / 45_100_000.0 / 0.082,
            "inception": "2021-01-01", "through": "2026-06-30",
            "total_days": 1276},
    },
}
_real_payload2 = TR._one_pager_payload
_real_roe2 = TR._roe_breakdown
try:
    TR._one_pager_payload = lambda v, q: (_P2, "26Q2")

    # (l) the components the PAGE'S OWN engine published are used, and the
    # breakdown is no longer withheld.
    r = TR.trace_field_value("P1", "one_pager.roe_to_date", quarter="26Q2")
    chk("ROE now returns a components breakdown, not a withholding",
        bool(r.get("inputs")) and r.get("breakdown_available") is not False)
    chk("the breakdown names the SCREEN's engine, not the ROE Summary report",
        "calculate_roe_detailed" in (r.get("breakdown_engine") or ""))
    chk("ROE reconciles — dists / wtd-avg capital / years ties to the figure",
        r.get("reconciles") is True)
    chk("the ROE components carry their dictionary sources",
        sum(1 for i in r["inputs"] if i.get("source")) >= 3)

    # (l2) AND IT IS A REAL CHECK, NOT A RUBBER STAMP: components that do not
    # divide out to the published ROE report reconciles=False.
    _bad = dict(_P2["pe_performance"]["roe_components"], years=99.0)
    _P2["pe_performance"] = dict(_P2["pe_performance"], roe_components=_bad)
    r = TR.trace_field_value("P1", "one_pager.roe_to_date", quarter="26Q2")
    chk("ROE components that do NOT tie report reconciles=False",
        r.get("reconciles") is False, str(r.get("reconciles")))
    _P2["pe_performance"] = {"roe_to_date": 0.082, "roe_components": {
        "total_cf_distributions": 12_300_000.0,
        "weighted_avg_capital": 45_100_000.0,
        "years": 12_300_000.0 / 45_100_000.0 / 0.082,
        "inception": "2021-01-01", "through": "2026-06-30", "total_days": 1276}}

    # (n) DSCR now shows numerator, denominator and ratio, per basis.
    r = TR.trace_field_value("P1", "one_pager.dscr", quarter="26Q2")
    _by = {b["basis"]: b for b in r["bases"]}
    chk("DSCR publishes its denominator where the builder resolved one",
        _by["ytd_actual"]["inputs"][1]["available"] is True
        and _by["ytd_actual"]["inputs"][1]["value"] is not None)
    chk("DSCR shows numerator AND denominator AND ratio together",
        _by["ytd_actual"]["inputs"][0]["value"] == 4.2e6
        and _by["ytd_actual"]["dscr"] == 1.45)
    chk("each DSCR basis reconciles against its OWN denominator",
        all(_by[b]["reconciles"] is True
            for b in ("ytd_actual", "ytd_budget", "actual_ye", "uw_ye")))

    # (o) ...and a column whose denominator is absent STAYS honest.
    chk("a DSCR column with no resolved denominator stays unavailable",
        _by["at_close"]["inputs"][1]["available"] is False
        and _by["at_close"]["inputs"][1]["value"] is None)
    chk("and says why, rather than dividing the ratio backwards",
        "backwards" in (_by["at_close"]["inputs"][1].get("reason") or ""))

    # (m) THE WITHHOLD PATH IS STILL LIVE for a payload carrying no components
    # (a frozen snapshot), and still refuses on a real mismatch.
    _P2["pe_performance"] = {"roe_to_date": 0.082}
    TR._roe_breakdown = lambda vc, q, val, uw=False: {
        "available": False, "reason": "engines disagree: 0.071 vs 0.082"}
    r = TR.trace_field_value("P1", "one_pager.roe_to_date", quarter="26Q2")
    chk("with no published components, the withhold path still refuses",
        r.get("inputs") == [] and r.get("breakdown_available") is False)
    chk("the withholding keeps the PAGE's value and names the disagreement",
        r.get("value") == 0.082
        and "disagree" in (r.get("breakdown_unavailable_reason") or ""))
finally:
    TR._one_pager_payload = _real_payload2
    TR._roe_breakdown = _real_roe2

# (q) the Snapshot ratios are FRACTIONS, not percentages — checking against a
# percentage would report every real row as failing to reconcile.
_ROW = {"vcode": "P1", "ltv": 0.65, "debt": 65e6, "valuation": 100e6,
        "debt_yield": 6.4e6 / 65e6, "quarter_noi": 1.6e6,
        "annualised_noi": 6.4e6, "ytd_dscr": 1.45, "ytd_noi": 4.2e6}
_real_row = TR._snapshot_loan_row
try:
    TR._snapshot_loan_row = lambda v, i, q: _ROW
    r = TR.trace_field_value("P1", "snapshot_loan.ltv", quarter="26Q2",
                             investor_code="TIAA")
    chk("snapshot LTV reconciles as a FRACTION (debt / valuation)",
        r.get("reconciles") is True, str(r.get("reconciles")))
    r = TR.trace_field_value("P1", "snapshot_loan.debt_yield", quarter="26Q2",
                             investor_code="TIAA")
    chk("snapshot Debt Yield reconciles off the PUBLISHED annualised NOI",
        r.get("reconciles") is True, str(r.get("reconciles")))
    # The Giant 7 fallback annualises projected year-end NOI, so the numerator
    # is NOT quarter NOI x 4 — checking that product would fail those deals.
    _G7 = dict(_ROW, quarter_noi=None, annualised_noi=7.0e6,
               debt_yield=7.0e6 / 65e6)
    TR._snapshot_loan_row = lambda v, i, q: _G7
    r = TR.trace_field_value("P1", "snapshot_loan.debt_yield", quarter="26Q2",
                             investor_code="TIAA")
    chk("a Debt Yield not built from quarter NOI x 4 still reconciles",
        r.get("reconciles") is True, str(r.get("reconciles")))
finally:
    TR._snapshot_loan_row = _real_row

# (s) THE SHIPPING BUILDERS ACTUALLY PUBLISH THE NEW KEYS. Everything above
# runs against an injected payload, so it would keep passing if one_pager
# stopped emitting them. These call the real builders with empty frames — which
# returns their seed dicts — so the contract is checked against the code that
# ships, not against the fixture. A source grep would not do: it cannot tell a
# key that is emitted from one that is merely mentioned in a comment.
import pandas as _pd                                              # noqa: E402
from one_pager import (get_property_performance as _gpp,           # noqa: E402
                       get_pe_performance as _gpe)
_empty = _pd.DataFrame()
_perf_seed = _gpp("PX", "2026-Q2", _empty, _empty, _empty)
chk("get_property_performance publishes a debt_service block",
    isinstance(_perf_seed.get("debt_service"), dict))
chk("it carries a denominator slot for every DSCR basis",
    set(_perf_seed.get("debt_service") or {}) == set(_perf_seed.get("dscr") or {}))
_pe_seed = _gpe("PX", "2026-Q2", _empty, _empty, _empty)
chk("get_pe_performance publishes roe_components and uw_roe_components",
    "roe_components" in _pe_seed and "uw_roe_components" in _pe_seed)
# `.get()`, not indexing: if the key above is missing this must report a second
# named FAILURE, not raise and take every later check down with it.
chk("they seed to None, so 'no breakdown' stays distinct from a zero breakdown",
    "roe_components" in _pe_seed
    and _pe_seed.get("roe_components") is None
    and _pe_seed.get("uw_roe_components") is None)

# (t) the assistant's One Pager tool now calls the builder the way the ROUTE
# does. This was the pre-existing divergence: without full_data the PE
# enrichment never ran, so the tool could report balances the page does not.
_route_src = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "flask_app", "api", "financials.py"),
    encoding="utf-8").read()


def _op_call_args(src: str) -> set:
    """The argument NAMES of the get_one_pager_data call in `src`."""
    i = src.index("result = get_one_pager_data(")
    k = src.index("(", i)
    depth = 0
    for p in range(k, len(src)):
        if src[p] == "(":
            depth += 1
        elif src[p] == ")":
            depth -= 1
            if depth == 0:
                j = p
                break
    body = re.sub(r"#[^\n]*", "", src[k + 1:j])
    return {a.split("=")[0].strip()
            for a in re.split(r",(?![^()\[\]]*[)\]])", body) if a.strip()}


_assist_src = open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "flask_app", "services",
    "assistant_service.py"), encoding="utf-8").read()
_missing = _op_call_args(_route_src) - _op_call_args(_assist_src)
chk("the assistant's One Pager call passes everything the page's route does",
    not _missing, f"missing: {sorted(_missing)}")
chk("including full_data, without which the PE enrichment never runs",
    "full_data" in _op_call_args(_assist_src))

# ── 13. The Dev tag / (Sold) label fields and the 7083 constant ──────────
#
# THESE EXIST BECAUSE THEY WENT MISSING ONCE. The 791f46b regeneration re-keyed
# the dictionary from bare names to tab.field and, in doing so, dropped
# `dev_tag` and `sold_label` entirely and removed
# one_pager.AT_CLOSE_RESERVE_RELEASE_ACCTS from the dependency map — while all
# three remained live in the app. Nothing failed: the tools simply reported "no
# verified entry", which is the designed SAFE answer and therefore invisible.
# These assertions make a silent re-drop loud.
print("\nDev tag / (Sold) label / 7083 coverage")

_SNAP_TABS = ("snapshot_financial", "snapshot_operating", "snapshot_loan")

# NOTE THE ARGUMENT ORDER: this file's chk is (label, cond) — the opposite of
# verify_traceability_live.py's (cond, label). Getting it backwards makes every
# check pass VACUOUSLY, because a non-empty label string is truthy. It did, and
# printed "PASS True" eight times before it was spotted.
for _tab in _SNAP_TABS:
    for _name in ("dev_tag", "sold_label"):
        _fid = f"{_tab}.{_name}"
        _r = lookup(field_id=_fid)
        chk(f"{_fid} resolves and carries a source",
            not _r.get("error") and bool(_r.get("source_value")),
            str(_r.get("error", ""))[:70])

# The BARE name is what a user asks ("where does the Dev tag come from?"), and
# it must surface all three tabs rather than silently answering for one.
for _name in ("dev_tag", "sold_label"):
    _r = lookup(field_id=_name)
    chk(f"a bare '{_name}' surfaces all three snapshot tabs, not one",
        _r.get("multi_tab") is True
        and set(_r.get("tabs") or []) == set(_SNAP_TABS),
        f"tabs={_r.get('tabs')} error={str(_r.get('error',''))[:50]}")

# The enum is CLOSED, so a field the model cannot name is a field it cannot ask
# about — being in the dictionary is necessary but not sufficient.
chk("the new ids are reachable through the closed field_id enum",
    all(f"{t}.{n}" in A._FIELD_IDS for t in _SNAP_TABS
        for n in ("dev_tag", "sold_label"))
    and "dev_tag" in A._FIELD_IDS and "sold_label" in A._FIELD_IDS,
    "an id absent from the enum cannot be asked for at all")

# Sources must be the real ones, not a placeholder.
_dev = lookup(field_id="snapshot_loan.dev_tag")
chk("the Dev tag names is_dev_deal / Lifecycle as its source",
    "is_dev_deal" in blob(_dev) and "Lifecycle" in blob(_dev))
_sold = lookup(field_id="snapshot_loan.sold_label")
chk("the (Sold) label names is_sold_as_of and the analyst-maintained columns",
    "is_sold_as_of" in blob(_sold) and "Sale_Status" in blob(_sold))

# THE CONSTANT, BY NAME AND BY ACCOUNT NUMBER. "7083" is the form somebody
# actually types, and it already appears inside the Prop_Info_AtClose SOURCE —
# and impact() searches sources BEFORE constants. So this asserts not merely
# that it resolves, but that it resolves to the CONSTANT and not to that source.
_deps_path = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "flask_app", "reference", "dependencies.json")
_consts = json.load(open(_deps_path, encoding="utf-8"))["shared_constants"]
chk("the dependency map carries 5 shared constants",
    len(_consts) == 5, f"got {len(_consts)}")
chk("one of them is AT_CLOSE_RESERVE_RELEASE_ACCTS",
    any("AT_CLOSE_RESERVE_RELEASE_ACCTS" in c["name"] for c in _consts))

for _q in ("AT_CLOSE_RESERVE_RELEASE_ACCTS",
           "one_pager.AT_CLOSE_RESERVE_RELEASE_ACCTS", "7083"):
    _r = impact(_q)
    chk(f"impact_of({_q!r}) resolves to the 7083 constant",
        not _r.get("error") and _r.get("match_type") == "shared_constant"
        and "AT_CLOSE_RESERVE_RELEASE_ACCTS" in str(_r.get("matched")),
        f"match_type={_r.get('match_type')} matched={_r.get('matched')!r} "
        f"error={str(_r.get('error',''))[:50]}")

# And it says what moves — a constant that resolves to an empty blast radius is
# no more useful than a miss.
_r = impact("7083")
chk("the 7083 constant reports what changes if it moves",
    bool(_r.get("fields_that_change")) and "at_close" in blob(_r))

# (r) the meta-cleaned dictionaries are the ones on disk.
_meta = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "flask_app", "reference",
    "data_dictionary.json"), encoding="utf-8"))["meta"]
chk("the dictionary carries the plain-language answer_format order",
    (_meta.get("answer_format") or {}).get("order", [None])[0] == "In plain terms")
_dmeta = json.load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "flask_app", "reference",
    "dependencies.json"), encoding="utf-8"))["meta"]
chk("the dependency map's matcher caveat states the 0.5 floor that is in force",
    "0.5" in (_dmeta.get("matcher_caveat") or "")
    and "floor 0.25" not in (_dmeta.get("matcher_caveat") or ""))

print("\n" + "=" * 62)
print(f"RESULT: {PASSED} passed, {FAILED} failed")
if FAILURES:
    for f in FAILURES:
        print(f"  - {f}")
sys.exit(1 if FAILED else 0)
