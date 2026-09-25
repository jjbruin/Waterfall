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

# ── G2. a PRINTED-UNITS cell keeps the PDF's text and leaves the number alone
#
# The money-row variances are printed as a percent of budget while the field
# stores a dollar difference (one_pager.py:1913), and the percent ON THE PAGE is
# derived in the BROWSER — so the stored number reaches no screen and
# overwriting it would change nothing. Giant 7's "-100%" is where this showed,
# but all three money rows have it. Asserted in BOTH directions: the text is
# kept AND the numeric field is untouched, because writing the percent into the
# dollar field would satisfy "the text is kept" while corrupting the figure.
print("\nG2. a printed-units cell keeps the PDF's text, and the number is left alone")
F.unfreeze("RBS262", "2026-Q2", "cbui", reason="reset for the display case")
disp_overlay = {
    "D1": {
        "property_performance.noi.variance":
            {"published": None, "display": "-100%",
             "units": "percent_of_budget", "page": 9, "source": "TIAA.pdf"},
    },
}
F.freeze_as_sent("RBS262", "2026-Q2", "cbui", overlay=disp_overlay,
                 assembler=stub_report, one_pager_getter=stub_op,
                 elements_loader=stub_elements,
                 source_manifest={"file": "TIAA.pdf", "sha256": "abc123"})
dv = F.get_frozen("RBS262", "2026-Q2")
op1 = dv["one_pagers"]["D1"]
before_num = (stub_op("D1", "2026-Q2") or {}).get(
    "property_performance", {}).get("noi", {}).get("variance")
chk("the printed text is stored verbatim under published_display",
    (op1.get("published_display") or {}).get(
        "property_performance.noi.variance") == "-100%",
    str(op1.get("published_display")))
chk("the NUMERIC variance field is left exactly as computed",
    (op1.get("property_performance", {}).get("noi", {}).get("variance")
     == before_num),
    f"stored={op1.get('property_performance', {}).get('noi', {}).get('variance')!r} "
    f"computed={before_num!r}")
drecs = [r for r in (dv["payload"].get("published_overrides") or [])
         if r.get("display")]
chk("the override records it as a display cell, with its units",
    len(drecs) == 1 and drecs[0].get("units") == "percent_of_budget",
    str(drecs))
chk("and keeps the page and source file for the printed text",
    bool(drecs) and drecs[0].get("page") == 9
    and drecs[0].get("source") == "TIAA.pdf", str(drecs))
chk("a display cell still counts as a cell applied",
    (dv["source_manifest"] or {}).get("overlay_cells_applied") == 1,
    str(dv["source_manifest"]))

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

# ── J. the 26Q2 overlay reproduces the sent PDFs, cell by cell ─────────────
#
# SKIPS RATHER THAN FAILS when the overlay is absent: it is gitignored, built
# from documents that are not in the repo. Where it IS present this is the
# check that the seeded freeze is faithful — every published cell in the
# frozen copy equals what the PDF printed, with the live value deliberately
# set to something else first so a no-op would be caught.
print("\nJ. the 26Q2 overlay reproduces the sent PDFs, cell by cell")
import json as _json                                                # noqa: E402

_ov_path = os.path.join(ROOT, "overlay_26q2.json")
if not os.path.exists(_ov_path):
    print("  SKIP  overlay_26q2.json not present — build it with "
          "scripts/build_26q2_overlay.py")
else:
    _doc = _json.load(open(_ov_path, encoding="utf-8"))
    _tot_cells = _tot_reports = 0
    _diffs = []
    for _inv, _blk in (_doc.get("investors") or {}).items():
        # One scratch quarter per investor so the two cannot interfere.
        _q = f"OVL-{_inv}"
        _reports = _blk.get("reports") or {}
        _titles = list(_reports)
        _vcodes = {t: f"V{i:03d}" for i, t in enumerate(_titles)}

        def _ov_op(vcode, quarter, _v=_vcodes):
            # Live values deliberately WRONG, so anything the overlay fails to
            # apply shows up as a difference rather than passing by accident.
            return {"vcode": vcode,
                    # Every field a real One Pager carries. An absent key is
                    # REFUSED by _set_path (rightly — inventing it would hide a
                    # published figure where no reader looks), so a thin stub
                    # would make cells vanish and this check fail for the wrong
                    # reason. It did, on pe_coupon.
                    "cap_stack": {"loan_terms_str": "LIVE-NOT-PDF",
                                  "debt": -1.0, "pref_equity": -1.0,
                                  "partner_equity": -1.0, "total_cap": -1.0,
                                  "purchase_price": -1.0,
                                  "pe_coupon": -1.0, "pe_participation": -1.0},
                    "pe_performance": {k: -1.0 for k in
                                       ("committed_pe", "remaining_to_fund",
                                        "funded_to_date", "return_of_capital",
                                        "current_pe_balance", "accrued_balance",
                                        "coupon", "participation")},
                    "property_performance": {
                        k: {c: -1.0 for c in
                            ("at_close", "actual_ye", "uw_ye", "ytd_actual",
                             "ytd_budget", "variance")}
                        for k in ("economic_occ", "revenue", "expenses", "noi")}}

        def _ov_report(investor, quarter, _t=_titles, _v=_vcodes):
            return {"subtabs": {"financial": {"groups": {"G1": {"deals": [
                {"vcode": _v[t], "name": t} for t in _t]}}}},
                "errors": {}, "resolution": {}}

        _overlay = {_vcodes[t]: cells for t, cells in _reports.items()}
        F.freeze_as_sent(_inv, _q, "cbui", overlay=_overlay,
                         roster=[_vcodes[t] for t in _titles],
                         assembler=_ov_report, one_pager_getter=_ov_op,
                         elements_loader=stub_elements,
                         source_manifest=_blk.get("source") or {})
        _fr = F.get_frozen(_inv, _q)
        for _t in _titles:
            _tot_reports += 1
            _stored = (_fr["one_pagers"] or {}).get(_vcodes[_t]) or {}
            for _path, _spec in _reports[_t].items():
                _tot_cells += 1
                if _spec.get("display") is not None:
                    got = (_stored.get("published_display") or {}).get(_path)
                    want = _spec["display"]
                else:
                    got = F._read_path(_stored, _path)
                    want = _spec.get("published")
                if isinstance(want, float) and isinstance(got, (int, float)):
                    # "beyond rounding" — the PDF prints to 0.1M / 1%.
                    ok = abs(float(got) - want) <= max(50_000.0, abs(want) * 1e-9)
                else:
                    ok = got == want
                if not ok:
                    _diffs.append(f"{_inv}/{_t}/{_path}: frozen={got!r} pdf={want!r}")

    chk(f"every published cell matches the PDF ({_tot_cells} cells, "
        f"{_tot_reports} One Pagers)",
        not _diffs, "; ".join(_diffs[:4]))
    chk("the overlay actually carried cells (not a vacuous pass)",
        _tot_cells > 500, str(_tot_cells))
    chk("both sent documents are represented",
        len(_doc.get("investors") or {}) == 2,
        str(list(_doc.get("investors") or {})))

# ── K. batch print serves the FROZEN roster, in printed order ──────────────
#
# Through the real endpoint, because the roster rule lives in the view: the
# service could store a perfect roster and the batch could still rebuild the
# population from today's ownership feed, which is exactly how a deal that was
# in the sent document goes missing from a reprint of it.
print("\nK. batch print serves the frozen roster, in printed order")
from flask import Flask                                             # noqa: E402
from flask_app.config import Config                                 # noqa: E402

_app = Flask(__name__)
_app.config.from_object(Config)
_app.config["TESTING"] = True
import flask_app.api.financials as _finmod                          # noqa: E402
import flask_app.api.portfolio_snapshot as _snapmod                 # noqa: E402
_app.register_blueprint(_finmod.financials_bp, url_prefix="/api/financials")
_app.register_blueprint(_snapmod.portfolio_snapshot_bp,
                        url_prefix="/api/portfolio-snapshot")

# A REAL TOKEN, NOT A BYPASSED DECORATOR. `login_required` is applied at import
# time, so reassigning the module attribute afterwards does nothing — the first
# attempt at this failed with 401 for exactly that reason. Minting a token with
# the app's own secret also makes section L's refusal mean something much
# stronger: the frozen gate turning away an AUTHENTICATED ADMIN, rather than an
# anonymous caller being turned away by auth.
import jwt as _jwt                                                  # noqa: E402
from datetime import datetime as _dt, timedelta as _td              # noqa: E402

_TOK = _jwt.encode({"sub": "1", "username": "cbui", "role": "admin",
                    "exp": _dt.utcnow() + _td(hours=1)},
                   Config.JWT_SECRET, algorithm="HS256")
_HDR = {"Authorization": f"Bearer {_TOK}"}
_ROSTER = [f"R{i:03d}" for i in range(30)]
F.freeze_as_sent("BATCH", "2026-Q2", "cbui",
                 roster=_ROSTER,
                 assembler=lambda i, q: {"subtabs": {}, "errors": {},
                                         "resolution": {}},
                 one_pager_getter=lambda vc, q: {"vcode": vc},
                 elements_loader=stub_elements)
with _app.test_client() as _c:
    # WRAPPED ON PURPOSE. If the frozen-roster branch is ever bypassed the
    # request falls through to the LIVE path, which reaches for the real
    # database and raises — taking the whole run down instead of failing one
    # named check. A crash here IS the failure, so it is reported as one.
    try:
        _r = _c.post("/api/financials/one-pager/batch", headers=_HDR,
                     json={"investor": "BATCH", "quarter": "2026-Q2",
                           "vcodes": ["SOMETHING", "ELSE"]})
        _body = _r.get_json() or {}
        _status = _r.status_code
    except Exception as _exc:                                   # noqa: BLE001
        _body, _status = {}, f"raised {type(_exc).__name__}: {_exc}"
chk("the batch answers from the frozen copy", _body.get("source") == "frozen",
    str(_status) + " " + str(_body)[:120])
chk("it serves the STORED roster, not the vcodes posted",
    [p.get("vcode") for p in (_body.get("pages") or [])] == _ROSTER,
    str([p.get("vcode") for p in (_body.get("pages") or [])])[:120])
chk("in the order it was sent",
    [p.get("vcode") for p in (_body.get("pages") or [])] == _ROSTER)
chk("and says the roster is the frozen one",
    "frozen roster" in (_body.get("roster_source") or ""))
# Rosters the seeded freeze must produce — TIAA 30 without Plaza Del Mar, KOC 15.
if os.path.exists(_ov_path):
    _d2 = _json.load(open(_ov_path, encoding="utf-8"))
    _tg = (_d2["investors"].get("TGAM") or {}).get("roster_titles") or []
    _ko = (_d2["investors"].get("KOCINV") or {}).get("roster_titles") or []
    chk("TIAA's printed roster is 30 One Pagers", len(_tg) == 30, str(len(_tg)))
    chk("Plaza Del Mar is not among them",
        not [t for t in _tg if "plaza" in t.lower() and "mar" in t.lower()],
        str([t for t in _tg if "plaza" in t.lower()]))
    chk("KOC's printed roster is 15 One Pagers", len(_ko) == 15, str(len(_ko)))

# ── L. typed fields on a frozen quarter are REFUSED by the API ─────────────
#
# Not "marked read-only" — REFUSED. A payload flag is a hint to the screen; the
# guarantee has to be that the write does not land, because a frozen quarter is
# the record of what an investor was sent. Driven through the blueprint so the
# before_request gate is the thing being tested.
print("\nL. writes to typed fields on a frozen quarter are refused by the API")
with _app.test_client() as _c:
    for _label, _url, _payload in [
        ("Net ROE", "/api/portfolio-snapshot/value",
         {"investor": "BATCH", "quarter": "2026-Q2", "vcode": "R000",
          "field": "net_roe", "value": 0.12}),
        ("ITD", "/api/portfolio-snapshot/value",
         {"investor": "BATCH", "quarter": "2026-Q2", "vcode": "R000",
          "field": "itd", "value": 1234}),
        ("comment", "/api/portfolio-snapshot/comment",
         {"investor": "BATCH", "quarter": "2026-Q2", "scope": "deal",
          "field": "comment", "text": "edited after sending"}),
        ("footnote", "/api/portfolio-snapshot/footnote",
         {"investor": "BATCH", "quarter": "2026-Q2", "text": "late footnote"}),
    ]:
        _r = _c.put(_url, headers=_HDR, json=_payload)
        if _r.status_code == 405:
            _r = _c.post(_url, headers=_HDR, json=_payload)
        _b = _r.get_json() or {}
        chk(f"a {_label} write on a frozen quarter is refused",
            _r.status_code == 409 and _b.get("frozen") is True,
            f"status={_r.status_code} body={str(_b)[:90]}")
    # The paired direction: an UNFROZEN quarter must still accept writes, or
    # "refused" would be satisfied by refusing everything.
    _r = _c.put("/api/portfolio-snapshot/value", headers=_HDR,
                json={"investor": "BATCH", "quarter": "2099-Q4",
                      "vcode": "R000", "field": "net_roe", "value": 0.12})
    chk("an unfrozen quarter is NOT refused by the frozen gate",
        _r.status_code != 409, f"status={_r.status_code}")

# ── M. the overlay OVERRIDES differing live values, and touches nothing else ─
#
# J proves the frozen copy equals the PDF. That is necessary but not
# sufficient: it would also pass if the live values happened to agree. Here the
# live payload is deliberately perturbed on the cells the overlay covers —
# including a newest-value field (loan rate), a preserved-row cell (Nottingham's
# pref) and a variance cell — and the PDF must win every one. The paired half is
# that cells the overlay does NOT name come through untouched, or "the PDF wins"
# would be satisfied by overwriting the whole payload.
print("\nM. the overlay overrides differing live values, and touches nothing else")

_LIVE = {
    "cap_stack": {
        "loan_terms_str": "9.99% floating | 1/1/2099",   # newest-value field
        "debt": 111.0, "pref_equity": 222.0,
        "untouched_cap": 333.0,                           # not in the overlay
    },
    "pe_performance": {"current_pe_balance": 444.0, "coupon": 0.999,
                       "untouched_pe": 555.0},
    "property_performance": {
        "noi": {"ytd_actual": 666.0, "ytd_budget": 777.0, "variance": 888.0},
    },
    "untouched_block": {"deep": {"value": 999.0}},
}
_PDF = {
    "cap_stack.loan_terms_str": {"published": "5.59% fixed | 7/1/2031", "page": 8},
    "cap_stack.debt": {"published": 95_100_000.0, "page": 8},
    "pe_performance.current_pe_balance": {"published": 9_100_000.0, "page": 9},
    "pe_performance.coupon": {"published": 0.085, "page": 9},
    # The units case: text kept, the number left alone.
    "property_performance.noi.variance": {
        "published": None, "display": "-100%", "units": "percent_of_budget",
        "page": 9},
}
F.unfreeze("KOCINV", "2026-Q2", "cbui", reason="reset for the override case")
F.freeze_as_sent(
    "KOCINV", "2026-Q2", "cbui", overlay={"D1": _PDF}, roster=["D1"],
    assembler=lambda i, q: {"subtabs": {}, "errors": {}, "resolution": {}},
    one_pager_getter=lambda vc, q: _json.loads(_json.dumps(_LIVE)),
    elements_loader=stub_elements,
    source_manifest={"file": "TIAA.pdf", "sha256": "abc123"})
_m = F.get_frozen("KOCINV", "2026-Q2")["one_pagers"]["D1"]

chk("a newest-value field (loan rate) takes the PDF's text",
    _m["cap_stack"]["loan_terms_str"] == "5.59% fixed | 7/1/2031",
    str(_m["cap_stack"]["loan_terms_str"]))
chk("a money cell takes the PDF's figure, not the live one",
    _m["cap_stack"]["debt"] == 95_100_000.0, str(_m["cap_stack"]["debt"]))
chk("a preserved-row cell (pref balance) takes the PDF's figure",
    _m["pe_performance"]["current_pe_balance"] == 9_100_000.0,
    str(_m["pe_performance"]["current_pe_balance"]))
chk("a percent field stored as a fraction takes the PDF's converted value",
    _m["pe_performance"]["coupon"] == 0.085, str(_m["pe_performance"]["coupon"]))
chk("a variance cell keeps the PDF's printed text",
    (_m.get("published_display") or {}).get(
        "property_performance.noi.variance") == "-100%",
    str(_m.get("published_display")))
chk("...and its NUMERIC field keeps the live value, not the percent",
    _m["property_performance"]["noi"]["variance"] == 888.0,
    str(_m["property_performance"]["noi"]["variance"]))
# The paired direction.
chk("a cell the overlay does not name is untouched (cap_stack)",
    _m["cap_stack"]["untouched_cap"] == 333.0)
chk("a cell the overlay does not name is untouched (pe_performance)",
    _m["pe_performance"]["untouched_pe"] == 555.0)
chk("a nested block the overlay does not name is untouched",
    _m["untouched_block"]["deep"]["value"] == 999.0)
chk("ytd_actual and ytd_budget beside the overridden variance are untouched",
    _m["property_performance"]["noi"]["ytd_actual"] == 666.0
    and _m["property_performance"]["noi"]["ytd_budget"] == 777.0)
# Every override keeps the value it displaced, so the drift stays measurable.
_mr = F.get_frozen("KOCINV", "2026-Q2")["payload"].get("published_overrides") or []
chk("each override records the live value it displaced",
    any(r["path"] == "cap_stack.debt" and r["computed_at_freeze"] == 111.0
        for r in _mr), str(_mr)[:140])

# A printed cell that cannot land is REPORTED, not silently dropped.
F.unfreeze("KOCINV", "2026-Q2", "cbui", reason="reset for the unapplied case")
F.freeze_as_sent(
    "KOCINV", "2026-Q2", "cbui",
    overlay={"D1": {"cap_stack.no_such_field": {"published": 1.0, "page": 8}}},
    roster=["D1"],
    assembler=lambda i, q: {"subtabs": {}, "errors": {}, "resolution": {}},
    one_pager_getter=lambda vc, q: _json.loads(_json.dumps(_LIVE)),
    elements_loader=stub_elements, source_manifest={})
_un = (F.get_frozen("KOCINV", "2026-Q2")["payload"].get("published_unapplied")
       or [])
chk("a printed cell with no matching field is reported as unapplied",
    len(_un) == 1 and _un[0]["path"] == "cap_stack.no_such_field", str(_un))

print(f"\n{'=' * 60}\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
