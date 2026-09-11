#!/usr/bin/env python
"""Guardrail: an AI appraisal summary says which sections it did not return.

THE CASE THIS EXISTS FOR, both real and both from live data. The same prompt —
unchanged since d6690d6 — produced:

  Asbury Commons  2026-09-08  13 keys   complete
  30 Bearfoot     2026-08-31   6 keys   missing in_place_income, market_overview,
                                        rent_and_leasing, positives, risks,
                                        extraordinary_assumptions, appraiser

Nothing recorded the difference. A model that drops sections on a sparser document is
indistinguishable from a document that has less in it, so asset management compared the
two, concluded the feature was inconsistent, and reported it as broken. It was not
broken. It was unchecked.

What is defended here:
  * every section the prompt asks for is declared in ONE place, `_EXPECTED_SECTIONS`,
    and that list matches the prompt itself — a section added to the prompt without
    being declared would silently stop being checked;
  * an EMPTY section counts as returned. A clean appraisal has no extraordinary
    assumptions, and re-asking for a genuinely empty one would loop;
  * a NULL or absent section counts as missing;
  * completeness is recomputed ON READ, so a summary stored before this check existed
    is judged by the same rule rather than grandfathered in as complete.

Run:  python scripts/valuation_ai_completeness_check.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services import valuation_ai_service as ai  # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


# The two real shapes, reduced to their keys. Values stand in for the real content.
BEARFOOT_KEYS = ["executive_summary", "property", "value_conclusion",
                 "valuation_approach", "key_assumptions"]
ASBURY_KEYS = list(ai._EXPECTED_SECTIONS)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"_EXPECTED_SECTIONS ({len(ai._EXPECTED_SECTIONS)}): "
          f"{', '.join(ai._EXPECTED_SECTIONS)}\n")

    print("1. The declared list matches the prompt — a new section cannot go unchecked")
    src = open(ai.__file__, encoding="utf-8").read()
    # Anchored to the line start: `_PROMPT` is a SUFFIX of `_RETRY_PROMPT`, so an
    # unanchored pattern matches the retry prompt instead and finds no sections at all —
    # which made the "nothing undeclared" check pass against an empty set.
    prompt = re.search(r'^_PROMPT\s*=\s*"""(.*?)"""', src, re.S | re.M)
    chk("the prompt is findable", prompt is not None)
    if prompt:
        body = prompt.group(1)
        # Top-level keys in the prompt's JSON skeleton sit at the start of a line.
        declared_in_prompt = set(re.findall(r'^\s{0,2}"([a-z_]+)":', body, re.M))
        undeclared = sorted(declared_in_prompt - set(ai._EXPECTED_SECTIONS))
        chk("every top-level prompt section is in _EXPECTED_SECTIONS",
            not undeclared, f"in the prompt but never checked: {undeclared}")
        phantom = sorted(set(ai._EXPECTED_SECTIONS) - declared_in_prompt)
        chk("_EXPECTED_SECTIONS asks for nothing the prompt does not",
            not phantom, f"checked but never requested: {phantom}")

    print("\n2. The real 30 Bearfoot shape is reported incomplete, by name")
    bearfoot = {k: "x" for k in BEARFOOT_KEYS}
    miss = ai._missing_sections(bearfoot)
    chk("it is flagged incomplete", len(miss) == 7, f"got {len(miss)}: {miss}")
    for k in ("in_place_income", "market_overview", "rent_and_leasing",
              "positives", "risks", "extraordinary_assumptions", "appraiser"):
        chk(f"  names {k}", k in miss)

    print("\n3. The real Asbury Commons shape is reported complete")
    asbury = {k: "x" for k in ASBURY_KEYS}
    chk("no sections missing", ai._missing_sections(asbury) == [],
        f"got {ai._missing_sections(asbury)}")

    print("\n4. An EMPTY section counts as returned — re-asking for one would loop")
    empty_ok = {k: "x" for k in ASBURY_KEYS}
    empty_ok["extraordinary_assumptions"] = []      # a clean appraisal
    empty_ok["risks"] = []
    empty_ok["appraiser"] = {}
    chk("empty list and empty dict are PRESENT",
        ai._missing_sections(empty_ok) == [], f"got {ai._missing_sections(empty_ok)}")

    print("\n5. A NULL section counts as missing — a returned nothing is still nothing")
    nulled = {k: "x" for k in ASBURY_KEYS}
    nulled["market_overview"] = None
    chk("null is reported missing", ai._missing_sections(nulled) == ["market_overview"],
        f"got {ai._missing_sections(nulled)}")

    print("\n6. The retry prompt asks for exactly the missing keys")
    p = ai._RETRY_PROMPT.format(keys=", ".join(["positives", "risks"]))
    chk("it names them", "positives, risks" in p)
    chk("it tells the model to return [] rather than omit", "empty list" in p)
    chk("it does not leave a stray format placeholder", "{keys}" not in p)

    print(f"\nPASS={PASS} FAIL={FAIL}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
