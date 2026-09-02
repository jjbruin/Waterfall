"""Guardrail: footnotes are free-form — no prefix, every note editable/deletable.

Covers Prompt B of the Sep 2 2026 work order:

  1. scope selection only PLACES the marker; it puts nothing into the text
  2. footnote text is exactly what was typed — no prefix, no label
  3. no stored text ever carried a prefix, so nothing needs stripping
  4. every footnote is editable and deletable, STANDING notes included, and a
     removed standing note can be restored

Runs the real committed ``compose_footnotes`` / ``standing_removed`` with
injected rows: no database, no network, no live token.

    python scripts/snapshot_footnotes_freeform_check.py capture before.json
    python scripts/snapshot_footnotes_freeform_check.py report before.json after.json

The prefix itself was never a backend concern — it was rendered by
SnapshotFinancial.vue in front of the text — so check 2 is asserted on the
component source, and the printed page is the real proof (see
scripts/snapshot_print_check.mjs).
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VUE = os.path.join(ROOT, "vue_app", "src", "components", "snapshot",
                   "SnapshotFinancial.vue")

# Analyst-entered rows exactly as the table holds them. Note row 3: an analyst
# who typed a prefix by hand. It must survive verbatim — the app never wrote
# one, so there is nothing here for the app to "strip", and rewriting a
# person's sentence on their behalf would be the actual data loss.
PLAIN_ROWS = [
    {"id": 11, "anchor": "invested", "text": "Invested is net of the fee "
                                             "allocation described above."},
    {"id": 12, "anchor": "deal:P0000019", "text": "Giant 7 refinanced in March."},
    {"id": 13, "anchor": "total_cap", "text": "Total Cap: column includes the "
                                              "sponsor co-invest."},
]


def _payload(rows):
    from flask_app.services.portfolio_snapshot_financial import (
        compose_footnotes, footnote_marks,
    )
    composed = compose_footnotes(rows)
    out = {"footnotes": [{k: f.get(k) for k in
                          ("number", "text", "standing", "standing_key",
                           "edited", "anchors", "scope", "vcode", "column")}
                         for f in composed],
           "marks": footnote_marks(composed)}
    try:
        from flask_app.services.portfolio_snapshot_financial import standing_removed
        out["standing_removed"] = standing_removed(rows)
    except ImportError:
        out["standing_removed"] = None      # not present before the change
    return out


def capture(outfile: str) -> int:
    from flask_app.services import portfolio_snapshot_financial as F

    scenarios = {"baseline": _payload(PLAIN_ROWS)}

    # The reserved anchors only exist after the change; build them by name so
    # the "before" capture still runs and simply shows them doing nothing.
    edit_pfx = getattr(F, "STANDING_EDIT_PREFIX", "standing-edit:")
    del_pfx = getattr(F, "STANDING_DELETE_PREFIX", "standing-delete:")

    scenarios["edited"] = _payload(PLAIN_ROWS + [
        {"id": 20, "anchor": edit_pfx + "debt_basis",
         "text": "Debt is the quarter-end balance. Development deals show the "
                 "committed facility."}])
    scenarios["deleted"] = _payload(PLAIN_ROWS + [
        {"id": 21, "anchor": del_pfx + "roe_exclusion", "text": ""}])
    scenarios["edited_and_deleted"] = _payload(PLAIN_ROWS + [
        {"id": 20, "anchor": edit_pfx + "debt_basis", "text": "Reworded."},
        {"id": 21, "anchor": del_pfx + "roe_exclusion", "text": ""}])
    scenarios["unknown_key"] = _payload(PLAIN_ROWS + [
        {"id": 22, "anchor": edit_pfx + "a_key_that_was_renamed",
         "text": "Wording an analyst typed against a constant since renamed."}])

    scenarios["blanked"] = _payload(PLAIN_ROWS[:1] + [
        {"id": 30, "anchor": "deal:P0000019", "text": ""},
        {"id": 31, "anchor": "total_cap", "text": "   "}])

    src = io.open(VUE, encoding="utf-8").read()
    scenarios["_vue"] = {
        "clearing_removes": "if (!text) {" in src and "deleteFootnote(f)" in src,
        "remove_is_labelled": ">Remove<" in src,
        "renders_placement_before_text": bool(
            re.search(r'class="fnanchor"', src)),
        "has_editable_footnote_input": 'class="fnedit"' in src,
        "delete_gated_on_db_id_only": "editable && f.id != null" in src,
        "offers_restore": "restoreStandingFootnote" in src,
    }

    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(scenarios, fh, indent=2, default=str)
    print("captured -> " + outfile)
    return 0


def _texts(p):
    return [f["text"] for f in p["footnotes"]]


def report(before_f: str, after_f: str) -> int:
    with open(before_f, encoding="utf-8") as fh:
        b = json.load(fh)
    with open(after_f, encoding="utf-8") as fh:
        a = json.load(fh)

    checks = []

    def chk(label, cond, detail=""):
        checks.append(bool(cond))
        print("  [{}] {}".format("PASS" if cond else "FAIL", label)
              + ("\n           " + detail if detail else ""))

    print("\n1. The prefix was RENDERED, never stored")
    chk("no placement label is printed in front of the text",
        not a["_vue"]["renders_placement_before_text"],
        'baseline ' + ("HAD the <span class=\"fnanchor\">...</span> that made "
                       'footnote 1 read "Debt column: Debt amount is..."'
                       if b["_vue"]["renders_placement_before_text"]
                       else "was already clean"))
    chk("stored text is unchanged by the composer on BOTH sides",
        _texts(b["baseline"])[-3:] == _texts(a["baseline"])[-3:]
        == [r["text"] for r in PLAIN_ROWS],
        "nothing was ever prepended server-side, so there is nothing to strip")
    chk("a prefix an analyst typed BY HAND is left alone",
        "Total Cap: column includes the sponsor co-invest."
        in _texts(a["baseline"]),
        "their sentence, their call — now editable in place")

    print("\n2. Scope selection places the marker and nothing else")
    chk("a property-anchored note marks that property",
        a["baseline"]["marks"]["property"].get("P0000019"),
        str(a["baseline"]["marks"]["property"]))
    chk("a column-anchored note marks that column header",
        a["baseline"]["marks"]["column"].get("invested"),
        str(a["baseline"]["marks"]["column"]))
    chk("and its text contains no scope word",
        all("(property)" not in t and ": column" not in t.replace(
            "Total Cap: column", "")
            for t in _texts(a["baseline"])))

    print("\n3. Every footnote is editable")
    chk("the list renders an editable input per note",
        a["_vue"]["has_editable_footnote_input"],
        "baseline " + ("did not" if not b["_vue"]["has_editable_footnote_input"]
                       else "already did"))
    ed = a["edited"]["footnotes"]
    note = next((f for f in ed if f["standing_key"] == "debt_basis"), None)
    chk("a STANDING note takes an override text",
        note and note["text"].startswith("Debt is the quarter-end balance."),
        str(note and note["text"])[:70])
    chk("and is flagged as edited, so the UI can offer the default back",
        note and note["edited"] is True)
    chk("its number and anchors do not move",
        note and note["number"] == 1 and note["anchors"] == ["debt"])
    _b0 = next((f for f in b["edited"]["footnotes"] if f.get("standing")), {})
    chk("the override, not the default, is what renders",
        note and not note["text"].startswith("Debt amount is current"),
        "baseline rendered "
        + ("the default" if str(_b0.get("text", "")).startswith("Debt amount")
           else "the override too"))
    chk("the override row does NOT also print as a footnote of its own",
        sum(1 for f in ed if "quarter-end balance" in f["text"]) == 1,
        "%d notes total" % len(ed))

    print("\n4. Every footnote is deletable, and a standing note comes back")
    dl = a["deleted"]["footnotes"]
    chk("delete is not gated on a database id",
        not a["_vue"]["delete_gated_on_db_id_only"],
        "baseline gated it" if b["_vue"]["delete_gated_on_db_id_only"]
        else "baseline already ungated")
    chk("AFTER: it is offered for every note",
        not a["_vue"]["delete_gated_on_db_id_only"]
        and a["_vue"]["offers_restore"])
    chk("the deleted standing note is off the page",
        not any(f["standing_key"] == "roe_exclusion" for f in dl),
        str([f["standing_key"] for f in dl]))
    chk("its marker is gone with it",
        not dl and True or not a["deleted"]["marks"]["property"].get("PCITWES"))
    chk("it is published as restorable rather than simply vanishing",
        [r["key"] for r in (a["deleted"]["standing_removed"] or [])]
        == ["roe_exclusion"],
        str(a["deleted"]["standing_removed"]))
    chk("a removed standing note is offered back",
        bool(a["deleted"]["standing_removed"]),
        "baseline offered "
        + str(b["deleted"]["standing_removed"] or "nothing"))
    chk("numbering closes over the gap and stays contiguous",
        [f["number"] for f in dl] == list(range(1, len(dl) + 1)),
        str([f["number"] for f in dl]))
    both = a["edited_and_deleted"]["footnotes"]
    chk("an edit and a deletion coexist on one page",
        any(f["standing_key"] == "debt_basis" and f["text"] == "Reworded."
            for f in both)
        and not any(f["standing_key"] == "roe_exclusion" for f in both))

    print("\n5. A blank footnote is a REMOVED footnote")
    # The reported symptom: "we can add and edit footnotes but there's no way
    # to delete one, especially the number markers (1), (2) can't be cleared."
    # Delete existed; what did not work was the obvious gesture. Clearing the
    # text saved a BLANK footnote that still held its number and still stamped
    # its marker on a column header or a property name.
    bl = a["blanked"]["footnotes"]
    chk("an emptied footnote holds no number", len(bl) == 3,
        f"{len(bl)} notes: {[f['text'][:24] for f in bl]}")
    chk("and no marker on the property it tagged",
        not (a["blanked"]["marks"]["property"] or {}).get("P0000019"),
        str(a["blanked"]["marks"]))
    chk("nor on the column it tagged",
        not (a["blanked"]["marks"]["column"] or {}).get("total_cap"))
    chk("BEFORE: the blanks were numbered and marked",
        len(b["blanked"]["footnotes"]) == 5
        and bool((b["blanked"]["marks"]["property"] or {}).get("P0000019")),
        f"{len(b['blanked']['footnotes'])} notes, marker "
        f"{(b['blanked']['marks']['property'] or {}).get('P0000019')} on "
        f"P0000019 — exactly the marker that could not be cleared")
    chk("the survivors renumber with no gap",
        [f["number"] for f in bl] == [1, 2, 3],
        str([f["number"] for f in bl]))
    chk("clearing the text now REMOVES rather than saving a blank",
        a["_vue"]["clearing_removes"])
    chk("and the remove control is labelled, not a bare glyph",
        a["_vue"]["remove_is_labelled"],
        "baseline had a bare x" if not b["_vue"].get("remove_is_labelled")
        else "already labelled at the baseline")

    print("\n6. A rename does not swallow an analyst's wording")
    uk = a["unknown_key"]["footnotes"]
    chk("an override naming a key that no longer exists still PRINTS",
        any("since renamed" in f["text"] for f in uk),
        "it falls through to an ordinary footnote instead of vanishing")

    passed = sum(checks)
    print("\n  {}/{} checks passed".format(passed, len(checks)))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "capture":
        raise SystemExit(capture(sys.argv[2]))
    if len(sys.argv) >= 4 and sys.argv[1] == "report":
        raise SystemExit(report(sys.argv[2], sys.argv[3]))
    print(__doc__)
    raise SystemExit(2)
