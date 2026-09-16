"""Guardrail: a distribution through a multi-asset fund is marked an estimate.

WHY THIS EXISTS. When the beneficial owner sits above a fund that invests in
more than one asset, the amount traced to it is NOT a payable figure. The fund
distributes on its own performance across every investment crossed inside it,
so this property's contribution can be offset by a result elsewhere before any
cash reaches the owner. (Jim, Sep 15 2026.)

THE LOCAL DATABASE CANNOT EXERCISE THIS. It holds three commitment rows and no
fund holding two assets, so `is_estimate` is False on every row here and the
branch would have shipped having never once executed. The fixtures are built.

Also pinned: the waterfall TYPE is the caller's, not a default. It was
hardcoded to CF_WF at both levels, so a sale — which runs the Capital waterfall
and reduces capital outstanding — came out modelled as an operating
distribution, which does not, with nothing on screen saying which had run.

Run:  .venv/Scripts/python.exe scripts/upstream_crossed_fund_check.py
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask_app.services import ownership_service as OS  # noqa: E402

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)


def by_id(rows):
    return {r["entity_id"]: r for r in rows}


# PPI27 holds one asset. PSC3 holds three — it is the crossed fund.
CROSSED = {"PSC3": ["30BEAR", "EASTCH", "BURTON"]}

# 30BEAR -> PPI27 -> PSC3 -> OWPSC, and 30BEAR -> OPMCCORD directly.
ROWS = [
    {"Path": "30BEAR->PPI27"},
    {"Path": "30BEAR->PPI27->PSC3"},
    {"Path": "30BEAR->PPI27->PSC3->OWPSC"},
    {"Path": "30BEAR->OPMCCORD"},
]
TOTALS = {"OWPSC": 600_000.0, "OPMCCORD": 400_000.0}

rows, note = OS.qualify_beneficiaries(TOTALS, ROWS, CROSSED, "30BEAR", 1_000_000.0)
got = by_id(rows)

# ── 1. Reached THROUGH the crossed fund -> estimate ──────────────────────
check("OWPSC" in got, "OWPSC missing from the beneficiaries")
check(got.get("OWPSC", {}).get("is_estimate") is True,
      "OWPSC is reached through PSC3, which holds three assets, and is NOT "
      "marked an estimate — the whole point of the footnote")
check([f["entity_id"] for f in got.get("OWPSC", {}).get("crossed_funds", [])] == ["PSC3"],
      "OWPSC does not name PSC3 as the fund that makes its figure an estimate")
check(got.get("OWPSC", {}).get("crossed_funds", [{}])[0].get("asset_count") == 3,
      "the fund's asset count is not carried, so the note cannot say how many")

# ── 2. NOT reached through it -> not an estimate ─────────────────────────
check(got.get("OPMCCORD", {}).get("is_estimate") is False,
      "OPMCCORD is paid directly by the deal and must NOT be marked an "
      "estimate; flagging every row makes the mark meaningless")

# ── 3. The note names the fund and says what the number is ───────────────
check(note and "PSC3" in note, "the footnote does not name the crossed fund")
for phrase in ("ESTIMATE", "crossed investments", "offset"):
    check(note and phrase in note,
          f"the footnote does not say {phrase!r}; it must state that the figure "
          f"is a contribution estimate and why it can differ")

# ── 4. No crossed fund anywhere -> no marks and NO note ──────────────────
rows2, note2 = OS.qualify_beneficiaries(TOTALS, ROWS, {}, "30BEAR", 1_000_000.0)
check(not any(r["is_estimate"] for r in rows2),
      "rows are marked as estimates with no crossed fund in the data")
check(note2 is None,
      "a footnote is shown when nothing is an estimate — a caption that is "
      "always there is a caption nobody reads")

# ── 5. The DEAL itself is never the crossed fund ─────────────────────────
# A deal can appear as an investor in its own right. Counting it would flag
# every beneficiary of every chain.
rows3, _ = OS.qualify_beneficiaries(
    TOTALS, ROWS, {"30BEAR": ["A", "B"]}, "30BEAR", 1_000_000.0)
check(not any(r["is_estimate"] for r in rows3),
      "the deal itself was treated as a crossed fund, which flags every row")

# ── 6. Percentages still carried ─────────────────────────────────────────
check(abs(got["OWPSC"]["pct_of_total"] - 0.6) < 1e-9,
      "pct_of_total is wrong: %r" % got["OWPSC"]["pct_of_total"])

# ── 7. The waterfall type is a parameter, not a hardcoded default ────────
sig = inspect.signature(OS.run_upstream_analysis)
check("wf_type" in sig.parameters,
      "run_upstream_analysis no longer takes wf_type; a capital event would be "
      "modelled as an operating distribution again")
src = inspect.getsource(OS.run_upstream_analysis)
live = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
check('wf_name="CF_WF"' not in live and 'wf_type="CF_WF"' not in live,
      "the waterfall type is hardcoded inside run_upstream_analysis again")
check(OS._wf_label("Cap_WF") == "Capital" and OS._wf_label("CF_WF") == "Cash Flow",
      "the waterfall label a user reads is wrong")

# The literal must match `vmisc` in the waterfalls table exactly: run_waterfall
# compares it with == and keys is_cap_wf off "Cap_WF".
check(OS._wf_label("CAP_WF") == "CAP_WF",
      "'CAP_WF' is being treated as a known type; the table's value is 'Cap_WF' "
      "and run_waterfall matches it exactly")

if FAIL:
    print("FAIL")
    for m in FAIL:
        print("  -", m)
    sys.exit(1)
print("OK - beneficiaries above a multi-asset fund are marked estimates and "
      "footnoted, direct ones are not, the deal itself is never the fund, and "
      "the waterfall type is the caller's choice")
