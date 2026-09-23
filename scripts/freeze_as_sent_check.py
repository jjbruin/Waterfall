"""Guardrail: the "Freeze as sent" store, read path, locking and history.

Runs entirely against a scratch SQLite in a temp directory with stub
assemblers — NO application database and NO live call, so it is safe to run
anywhere and cannot freeze a real quarter.

What it pins, and why each one is here rather than assumed:

  1. A freeze stores the SUBTABS, every ONE PAGER, and the ROSTER in printed
     order. A freeze missing any of the three cannot reproduce what was sent.
  2. A freeze is NOT an approval: ``approved_by`` stays NULL and
     ``frozen_reason`` says ``as-sent``. Recording a decision nobody made is
     the failure this separation exists to prevent.
  3. A frozen quarter READS BACK IDENTICALLY after the underlying data moves.
     The check mutates the stub's answers between freeze and read, which is the
     only way to prove the read is served from store rather than recomputed.
  4. One Pagers are keyed by INVESTOR as well as deal. Nottingham Village went
     out at $9.1M on one report and $12.1M on another for the same quarter, so
     a store keyed only by (vcode, quarter) cannot represent what was sent.
  5. A FAILED freeze leaves the quarter unfrozen. Asserted by making one One
     Pager raise: the whole freeze must abort, not store a partial report.
  6. The published overlay overwrites only the cells named, keeps the computed
     value beside each one, and counts what it applied.
  7. Re-freeze and unfreeze keep history and demand a reason.

Usage
    .venv/Scripts/python.exe scripts/freeze_as_sent_check.py
"""
from __future__ import annotations

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import sqlalchemy  # noqa: E402

PASS = FAIL = 0


def chk(label, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK   {label}")
    else:
        FAIL += 1
        print(f"  FAIL {label}" + (f"  -> {detail}" if detail else ""))


# ── a scratch engine, wired in before anything touches a real database ──────
tmpdir = tempfile.mkdtemp(prefix="freeze_check_")
eng = sqlalchemy.create_engine(f"sqlite:///{os.path.join(tmpdir, 't.db')}")

from flask_app.services import portfolio_snapshot_freeze as F  # noqa: E402

F._engine = lambda: eng
F._is_postgres = lambda: False
F._data_version = lambda: "build=test;actuals_through=2026-07-31"


# ── stubs: the report, the One Pagers, the editable elements ────────────────
STATE = {"pref": 9_100_000, "rate": "5.59% | Fixed | 7/1/2031"}


def stub_report(investor, quarter):
    return {
        "subtabs": {"financial": {"groups": {"G1": {"deals": [
            {"vcode": "D1", "name": "Deal One", "total_pref": STATE["pref"]},
            {"vcode": "D2", "name": "Deal Two", "total_pref": 2_000_000},
        ]}}}},
        "errors": {},
        "resolution": {"investor_name": investor, "quarter": quarter},
        "_resolved": {"dropped": "by the freeze"},
    }


def stub_elements(investor, quarter):
    return {"comments": [{"f": "c"}], "footnotes": [{"f": "n"}], "values": [{"f": "v"}]}


def stub_op(vcode, quarter):
    return {"vcode": vcode, "cap_stack": {"loan_terms_str": STATE["rate"],
                                          "pref_equity": STATE["pref"]}}


print("A. a freeze stores subtabs, One Pagers and the roster")
res = F.freeze_as_sent("TGAM", "2026-Q2", "cbui", assembler=stub_report,
                       one_pager_getter=stub_op, elements_loader=stub_elements)
fr = F.get_frozen("TGAM", "2026-Q2")
chk("freeze returns a receipt naming who and when",
    res["frozen_by"] == "cbui" and res["frozen_at"], str(res))
chk("subtabs stored", bool((fr["payload"].get("subtabs") or {}).get("financial")))
chk("every One Pager stored", sorted((fr["one_pagers"] or {})) == ["D1", "D2"],
    str(list(fr["one_pagers"] or {})))
chk("roster stored in printed order", fr["roster"] == ["D1", "D2"], str(fr["roster"]))
chk("editable elements stored alongside",
    (fr["payload"].get("elements") or {}).get("comments") == [{"f": "c"}])
chk("the internal _resolved block is not stored",
    "_resolved" not in fr["payload"])
chk("data version recorded", "build=test" in (fr["data_version"] or ""))

print("\nB. a freeze is not an approval")
chk("approved_by stays NULL", fr["approved_by"] is None, str(fr["approved_by"]))
chk("frozen_reason says as-sent", fr["frozen_reason"] == F.REASON_AS_SENT,
    str(fr["frozen_reason"]))
chk("version starts at 1", fr["version"] == 1, str(fr["version"]))

print("\nC. the stored copy survives the data moving underneath it")
STATE["pref"] = 12_100_000            # the refinancing the app now knows about
STATE["rate"] = "6.86% | Fixed | 6/1/2031"
again = F.get_frozen("TGAM", "2026-Q2")
deal = again["payload"]["subtabs"]["financial"]["groups"]["G1"]["deals"][0]
chk("subtab figure reads back as sent, not as recomputed",
    deal["total_pref"] == 9_100_000, str(deal["total_pref"]))
chk("One Pager loan terms read back as sent",
    again["one_pagers"]["D1"]["cap_stack"]["loan_terms_str"]
    == "5.59% | Fixed | 7/1/2031",
    str(again["one_pagers"]["D1"]["cap_stack"]["loan_terms_str"]))

print("\nD. the read path serves the stored copy and marks it read-only")
rep = F.load_report("TGAM", "2026-Q2", status="draft", assembler=stub_report)
chk("a frozen quarter serves frozen even when NOT approved",
    rep["source"] == F.SOURCE_FROZEN, str(rep.get("source")))
chk("it is marked read-only", rep.get("read_only") is True)
chk("the note says frozen as sent and names who",
    "Frozen as sent" in rep["source_note"] and "cbui" in rep["source_note"],
    rep["source_note"])
live = F.load_report("TGAM", "2026-Q1", status="draft", assembler=stub_report)
chk("an unfrozen quarter still serves live", live["source"] == F.SOURCE_LIVE)
chk("and live is not read-only", live.get("read_only") is False)

print("\nE. One Pagers are keyed by investor as well as deal")
F.freeze_as_sent("KOCINV", "2026-Q2", "cbui", assembler=stub_report,
                 one_pager_getter=stub_op, elements_loader=stub_elements)
a = F.get_frozen_one_pager("TGAM", "2026-Q2", "D1")
b = F.get_frozen_one_pager("KOCINV", "2026-Q2", "D1")
chk("the same deal+quarter can hold two different published copies",
    a["cap_stack"]["pref_equity"] == 9_100_000
    and b["cap_stack"]["pref_equity"] == 12_100_000,
    f"TGAM={a['cap_stack']['pref_equity']} KOCINV={b['cap_stack']['pref_equity']}")
chk("an investor with no freeze returns nothing",
    F.get_frozen_one_pager("WRI", "2026-Q2", "D1") is None)

print("\nF. a failed freeze leaves the quarter unfrozen")
def exploding_op(vcode, quarter):
    if vcode == "D2":
        raise RuntimeError("forecast unavailable")
    return stub_op(vcode, quarter)

before = F.get_frozen("WRI", "2026-Q2")
raised = None
try:
    F.freeze_as_sent("WRI", "2026-Q2", "cbui", assembler=stub_report,
                     one_pager_getter=exploding_op, elements_loader=stub_elements)
except Exception as exc:          # noqa: BLE001
    raised = exc
chk("the freeze raises rather than storing a partial report", raised is not None)
chk("the error names the deal that failed", "D2" in str(raised), str(raised))
chk("nothing was stored", before is None and F.get_frozen("WRI", "2026-Q2") is None)
chk("is_frozen reports it as not frozen", F.is_frozen("WRI", "2026-Q2") is False)

print("\nG. the published overlay")
overlay = {
    "__subtabs__": {
        "subtabs.financial.groups.G1.deals[0].total_pref":
            {"published": 9_100_000, "page": 6},
    },
    "D1": {"cap_stack.loan_terms_str":
           {"published": "3.7% fixed | 7/1/2026", "page": 8}},
}
F.freeze_as_sent("RBS262", "2026-Q2", "cbui", overlay=overlay,
                 assembler=stub_report, one_pager_getter=stub_op,
                 elements_loader=stub_elements,
                 source_manifest={"file": "TIAA.pdf", "sha256": "abc123"})
ov = F.get_frozen("RBS262", "2026-Q2")
d0 = ov["payload"]["subtabs"]["financial"]["groups"]["G1"]["deals"][0]
chk("a published subtab cell is overwritten", d0["total_pref"] == 9_100_000,
    str(d0["total_pref"]))
chk("a published One Pager cell is overwritten",
    ov["one_pagers"]["D1"]["cap_stack"]["loan_terms_str"] == "3.7% fixed | 7/1/2026")
chk("an untouched cell keeps the computed value",
    ov["payload"]["subtabs"]["financial"]["groups"]["G1"]["deals"][1]["total_pref"]
    == 2_000_000)
recs = ov["payload"].get("published_overrides") or []
chk("every override records the computed value beside the published one",
    len(recs) == 2 and all("computed_at_freeze" in r for r in recs), str(recs))
chk("the computed value recorded is the one at freeze time",
    any(r["computed_at_freeze"] == 12_100_000 for r in recs), str(recs))
chk("the page number is kept per cell", all(r.get("page") in (6, 8) for r in recs))
chk("the manifest records the source file and hash",
    (ov["source_manifest"] or {}).get("sha256") == "abc123")
chk("the manifest counts the cells applied",
    (ov["source_manifest"] or {}).get("overlay_cells_applied") == 2,
    str(ov["source_manifest"]))

print("\nH. re-freeze and unfreeze keep history and demand a reason")
try:
    F.refreeze("TGAM", "2026-Q2", "admin", "", assembler=stub_report,
               one_pager_getter=stub_op, elements_loader=stub_elements)
    chk("re-freeze without a reason is refused", False)
except ValueError:
    chk("re-freeze without a reason is refused", True)
F.refreeze("TGAM", "2026-Q2", "admin", "restated after MRI correction",
           assembler=stub_report, one_pager_getter=stub_op,
           elements_loader=stub_elements)
cur = F.get_frozen("TGAM", "2026-Q2")
hist = F.frozen_history("TGAM", "2026-Q2")
chk("the current version advances", cur["version"] == 2, str(cur["version"]))
chk("the previous version is kept, not overwritten", len(hist) == 1, str(hist))
chk("history records the superseding user", hist[0]["superseded_by"] == "admin")
cur_deal = cur["payload"]["subtabs"]["financial"]["groups"]["G1"]["deals"][0]
chk("the re-freeze captured the NEW live value", cur_deal["total_pref"] == 12_100_000,
    str(cur_deal["total_pref"]))
try:
    F.unfreeze("TGAM", "2026-Q2", "admin", "")
    chk("unfreeze without a reason is refused", False)
except ValueError:
    chk("unfreeze without a reason is refused", True)
F.unfreeze("TGAM", "2026-Q2", "admin", "sent in error")
chk("after unfreezing the quarter is live again",
    F.is_frozen("TGAM", "2026-Q2") is False)
chk("and the unfrozen copy is archived, not lost",
    len(F.frozen_history("TGAM", "2026-Q2")) == 2,
    str(len(F.frozen_history("TGAM", "2026-Q2"))))
chk("the archive says why",
    any("sent in error" in str(h["supersede_reason"])
        for h in F.frozen_history("TGAM", "2026-Q2")))

print("\nI. freezing one investor moves nothing for another")
kept = F.get_frozen("KOCINV", "2026-Q2")
chk("KOCINV's stored copy is untouched by everything above",
    kept["one_pagers"]["D1"]["cap_stack"]["pref_equity"] == 12_100_000
    and kept["version"] == 1)

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
