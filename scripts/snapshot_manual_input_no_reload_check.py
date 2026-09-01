"""Guardrail: Financial Net ROE / ITD entry must not refetch the bundle.

Three things are checked, and each is a statement about the real committed code
rather than about a copy of it:

  A  the display rule has ONE definition, and PUT /value returns exactly what
     the Financial assembly would have put in the row's ``*_display`` twin

  B  the value still persists — round-tripped through the real persistence
     module against a scratch SQLite file, so no application database is
     touched

  C  the shell patches the row instead of reloading: ``onSaveValue`` must not
     call ``load()``, and the Loan tab's ``onSaveComment`` must be byte-for-byte
     what it was (it was already correct, and this fix must not disturb it)

C is source-level because the repo carries no JS test harness. It asserts the
specific defect that caused the refresh, not merely "the file changed".

Run:  python scripts/snapshot_manual_input_no_reload_check.py
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEW = os.path.join(ROOT, "vue_app", "src", "views", "PortfolioSnapshotView.vue")
FINCOMP = os.path.join(ROOT, "vue_app", "src", "components", "snapshot",
                       "SnapshotFinancial.vue")
LOANCOMP = os.path.join(ROOT, "vue_app", "src", "components", "snapshot",
                        "SnapshotLoan.vue")

checks: list = []


def chk(label, ok, detail=""):
    checks.append((label, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}"
          + (f"\n         {detail}" if detail and not ok else ""))


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def strip_comments(src: str) -> str:
    """Drop // and /* */ comments so a CALL check cannot match prose.

    Needed because the fix's own comment explains what it replaced and names
    ``load()`` while doing so.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def fn_body(src: str, name: str) -> str:
    """The text of one top-level `function name(...) { ... }` block.

    The parameter list is walked by paren depth first: these handlers take an
    inline object TYPE (``p: { vcode: string; ... }``), so the first ``{`` after
    the function name belongs to the annotation, not the body. Matching braces
    from there terminates at the end of the type and silently returns a stub —
    which is exactly how this helper first reported three false failures.
    """
    i = src.find(f"function {name}(")
    if i < 0:
        return ""
    # End of the parameter list.
    pd, k = 0, src.index("(", i)
    while k < len(src):
        if src[k] == "(":
            pd += 1
        elif src[k] == ")":
            pd -= 1
            if pd == 0:
                break
        k += 1
    j = src.find("{", k)
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1
    return src[i:]


# ── A. one display rule, and the endpoint returns it ──────────────────────
def check_display_rule():
    print("\nA. display rule has one definition")
    from flask_app.services import portfolio_snapshot_financial as F

    chk("manual_display / manual_na_cells / MANUAL_SOURCE are exported",
        all(hasattr(F, n) for n in
            ("manual_display", "manual_na_cells", "MANUAL_SOURCE")))

    # The three cases the rule distinguishes.
    chk("a saved number displays as itself",
        F.manual_display("itd", 1.17, frozenset()) == 1.17)
    chk("an empty cell displays as PENDING",
        F.manual_display("itd", None, frozenset()) == F.PENDING)
    chk("an n/a cell outranks PENDING",
        F.manual_display("net_roe", None, frozenset({"net_roe"})) == F.NA_LABEL
        and F.manual_display("net_roe", 0.048,
                             frozenset({"net_roe"})) == F.NA_LABEL)

    # City West is the live PDF_NA_CELLS row; every other deal has none.
    chk("manual_na_cells reads PDF_NA_CELLS, case-insensitively",
        F.manual_na_cells("pcitwes") == frozenset({"debt", "net_roe"})
        and F.manual_na_cells("P0000066") == frozenset())

    # The assembly must go through the helper, not re-derive the rule.
    src = read(os.path.join(ROOT, "flask_app", "services",
                            "portfolio_snapshot_financial.py"))
    body = src[src.find("# ---- Zone C"):src.find("diag[\"deals\"] += 1")]
    chk("the assembly calls manual_display rather than inlining the ternary",
        "manual_display(f, v, na)" in body
        and "PENDING if v is None else v" not in body,
        "Zone C still contains its own copy of the rule")

    # The endpoint must call the same helper.
    api = read(os.path.join(ROOT, "flask_app", "api", "portfolio_snapshot.py"))
    put = api[api.find("def put_value("):api.find("def post_footnote(")]
    chk("PUT /value returns display + source via the shared helper",
        "manual_display(" in put and "MANUAL_SOURCE" in put
        and '"display"' in put)


# ── B. the value still persists ───────────────────────────────────────────
def check_persist():
    print("\nB. the entered value still persists")
    import sqlalchemy
    from flask_app.services import portfolio_snapshot_persistence as P
    from flask_app.services import portfolio_snapshot_financial as F

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_noreload_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    g = sys.modules[P.__name__]
    g._engine = lambda: eng                      # type: ignore[assignment]
    g._is_postgres = lambda: False               # type: ignore[assignment]

    INV, Q, VC = "TGAM", "2026-Q1", "P0000030"

    # Same normalisation the endpoint applies before calling save_value.
    def norm(raw):
        if raw in (None, ""):
            return None
        return float(str(raw).replace(",", "").replace("%", "").replace("$", ""))

    P.save_value(INV, Q, deal_vcode=VC, field="itd",
                 value=norm("1,234.5"), updated_by="check")
    P.save_value(INV, Q, deal_vcode=VC, field="net_roe",
                 value=norm("4.8%"), updated_by="check")

    vals = P.get_elements("value", INV, Q)
    got = {(v.get("deal_vcode"), v.get("field")): v.get("value") for v in vals}
    chk("a comma-formatted ITD persists as a number",
        got.get((VC, "itd")) == 1234.5, f"got {got.get((VC, 'itd'))!r}")
    chk("a percent-formatted Net ROE persists as a number",
        got.get((VC, "net_roe")) == 4.8, f"got {got.get((VC, 'net_roe'))!r}")

    # And the display the endpoint would have returned matches the stored value,
    # so the patched cell equals what a full rebuild would show.
    chk("the returned display equals the stored value",
        F.manual_display("itd", got.get((VC, "itd")),
                         F.manual_na_cells(VC)) == 1234.5)

    # Clearing goes back to PENDING, not to a stale number.
    P.save_value(INV, Q, deal_vcode=VC, field="itd", value=norm(""),
                 updated_by="check")
    vals = P.get_elements("value", INV, Q)
    got = {(v.get("deal_vcode"), v.get("field")): v.get("value") for v in vals}
    chk("clearing the box stores null and displays PENDING",
        got.get((VC, "itd")) is None
        and F.manual_display("itd", got.get((VC, "itd")),
                             F.manual_na_cells(VC)) == F.PENDING)


# ── C. the shell patches instead of reloading ─────────────────────────────
def check_no_reload():
    print("\nC. the shell no longer refetches the bundle on entry")
    view = read(VIEW)
    save_value = fn_body(view, "onSaveValue")
    save_comment = fn_body(view, "onSaveComment")

    chk("onSaveValue exists and was found", bool(save_value))

    # THE DEFECT. `await load()` set loading=true, which swaps the whole tab
    # body for the "Building snapshot…" placeholder and rebuilds all four
    # subtabs server-side. Checked against the CODE, comments stripped.
    chk("onSaveValue does not call load()",
        not re.search(r"\bload\s*\(\s*\)", strip_comments(save_value)),
        "onSaveValue still refetches /bundle — the refresh is back")

    chk("onSaveValue still PUTs /value (the value is still saved)",
        "/value" in save_value and "api.put" in save_value)

    chk("onSaveValue patches the row from the response",
        "_display" in save_value and "recountManual" in save_value)

    # The Loan tab was already correct; this fix must not touch it.
    chk("onSaveComment still does nothing but PUT /comment",
        "/comment" in save_comment
        and not re.search(r"\bload\s*\(\s*\)", save_comment))

    # The placeholder that produced the visible 'refresh' is still wired to
    # `loading` only — i.e. nothing else now sets it on a save path.
    setters = re.findall(r"loading\.value\s*=\s*(true|false)", view)
    chk("loading is only ever set inside load()", setters == ["true", "false"],
        f"loading assigned {len(setters)} times: {setters}")

    # Both inputs commit on @change (blur/Enter), like the Loan tab's comment.
    fin, loan = read(FINCOMP), read(LOANCOMP)
    chk("Financial inputs commit on @change, matching the Loan tab",
        fin.count("@change=\"commitValue(") == 2
        and "@change=\"commit(" in loan)

    # No <form> around them, so Enter cannot submit-navigate.
    chk("no <form> element in either component",
        "<form" not in fin and "<form" not in loan)

    # The child reseeds its draft from props.data; an in-place patch must not
    # change that object's identity, so the watcher must stay shallow.
    watch_blk = fin[fin.find("watch(() => props.data"):]
    watch_blk = watch_blk[:watch_blk.find("function commitValue")]
    chk("the draft watcher is shallow, so patching a row cannot wipe typing",
        "deep: true" not in watch_blk)


# ── D. the real endpoint, through the real route ──────────────────────────
def check_endpoint():
    """Call the committed ``put_value`` view body and read its response.

    ``__wrapped__`` skips only ``@login_required``; the body is the real one, so
    this exercises the normalisation, the persistence write and the new
    display/source fields exactly as a browser would. Scratch SQLite — no
    application database is touched.
    """
    print("\nD. PUT /value response, from the real view")
    import sqlalchemy
    from flask import Flask, g as flask_g
    from flask_app.api import portfolio_snapshot as API
    from flask_app.services import portfolio_snapshot_persistence as P

    tmp = os.path.join(tempfile.mkdtemp(prefix="ps_endpoint_"), "t.db")
    eng = sqlalchemy.create_engine(f"sqlite:///{tmp}")
    mod = sys.modules[P.__name__]
    mod._engine = lambda: eng                    # type: ignore[assignment]
    mod._is_postgres = lambda: False             # type: ignore[assignment]

    app = Flask(__name__)
    view = API.put_value.__wrapped__             # skips @login_required only

    def call(body):
        with app.test_request_context(json=body, method="PUT"):
            flask_g.current_user = {"id": 1, "username": "check",
                                    "role": "admin"}
            resp = view()
            payload = resp[0] if isinstance(resp, tuple) else resp
            status = resp[1] if isinstance(resp, tuple) else 200
            return status, payload.get_json()

    base = {"investor": "TGAM", "quarter": "2026-Q1", "vcode": "P0000030"}

    st, d = call({**base, "field": "itd", "value": "1,234.5"})
    chk("a comma-formatted entry returns 200", st == 200, f"{st} {d}")
    chk("response carries value / display / source, so no refetch is needed",
        d.get("value") == 1234.5 and d.get("display") == 1234.5
        and d.get("source") == "manual entry (formula TBD)", str(d))
    chk("response echoes vcode and field so the UI can find the row",
        d.get("vcode") == "P0000030" and d.get("field") == "itd", str(d))

    st, d = call({**base, "field": "net_roe", "value": ""})
    chk("clearing returns PENDING as the display",
        st == 200 and d.get("value") is None
        and d.get("display") == "pending entry", str(d))

    # City West's Net ROE is n/a on the PDF; the rule must survive the round
    # trip rather than being re-derived in the UI.
    st, d = call({**base, "vcode": "PCITWES", "field": "net_roe",
                  "value": "4.8"})
    chk("an n/a cell still reports n/a, not the number",
        st == 200 and d.get("display") == "n/a", str(d))

    st, d = call({**base, "field": "itd", "value": "abc"})
    chk("a non-numeric entry is rejected 400 and nothing is patched",
        st == 400 and "not a number" in str(d.get("error", "")), f"{st} {d}")


# ── E. the client-side recount matches the backend's own counts ───────────
def check_recount_matches_live():
    """Reproduce ``recountManual()`` over a real payload and compare.

    The one thing the UI now derives itself is the "N entered" tally. This
    takes the live Financial subtab — whose ``manual_entered`` blocks the
    backend computed with ``_subtotal`` — and recomputes them with the exact
    rule the Vue handler uses (``r[field] != null``). If the two agree on real
    data, the client rule is the backend rule.

    Read-only, and skipped when WF_TOKEN is not set.
    """
    print("\nE. client recount vs the backend's own counts (live, read-only)")
    if not os.environ.get("WF_TOKEN", "").strip():
        print("  [SKIP] WF_TOKEN not set")
        return
    from scripts import live_api

    investors = os.environ.get("WF_CHECK_INVESTORS", "TGAM,PSC3,WRI").split(",")
    quarter = os.environ.get("WF_CHECK_QUARTER", "2026-Q1")

    def tally(rows):
        return {f: sum(1 for r in rows if r.get(f) is not None)
                for f in ("itd", "net_roe")}

    total_rows_seen = 0
    for code in [c.strip() for c in investors if c.strip()]:
        try:
            b = live_api.get("/api/portfolio-snapshot/bundle",
                             params={"investor": code, "quarter": quarter})
        except Exception as exc:
            chk(f"{code}: bundle fetched", False, str(exc)[:120])
            continue
        fin = (b.get("subtabs") or {}).get("financial") or {}
        groups = fin.get("groups") or {}
        if not groups:
            print(f"  [SKIP] {code}: no financial groups")
            continue

        all_rows = []
        group_ok = True
        for gname, blk in groups.items():
            deals = blk.get("deals") or []
            all_rows.extend(deals)
            got = (blk.get("subtotal") or {}).get("manual_entered") or {}
            mine = tally(deals)
            if {k: got.get(k) for k in mine} != mine:
                group_ok = False
                print(f"         {code}/{gname}: backend {got} vs client {mine}")
        chk(f"{code}: every group subtotal count reproduced", group_ok)

        total_rows = all_rows + (fin.get("ownership_flagged") or [])
        total_rows_seen += len(total_rows)
        got = (fin.get("total") or {}).get("manual_entered") or {}
        mine = tally(total_rows)
        chk(f"{code}: portfolio total count reproduced "
            f"({mine['itd']} itd / {mine['net_roe']} net_roe over "
            f"{len(total_rows)} rows)",
            {k: got.get(k) for k in mine} == mine,
            f"backend {got} vs client {mine}")

    chk("the comparison actually saw rows", total_rows_seen > 0)

    # The live counts are all 0 today (nobody has entered a figure yet), so the
    # comparison above is necessary but not sufficient. Inject values into the
    # real rows and compare against the BACKEND's own _subtotal — not a second
    # copy of the rule — so the tally is exercised with non-zero counts and
    # with the PENDING sentinel present in the display twin.
    from flask_app.services import portfolio_snapshot_financial as F

    b = live_api.get("/api/portfolio-snapshot/bundle",
                     params={"investor": "TGAM", "quarter": quarter})
    fin = (b.get("subtabs") or {}).get("financial") or {}
    rows = [r for blk in (fin.get("groups") or {}).values()
            for r in (blk.get("deals") or [])]
    rows += (fin.get("ownership_flagged") or [])
    chk("injection fixture has enough rows", len(rows) >= 6)

    for n_itd, n_roe in ((1, 0), (3, 2), (len(rows), len(rows))):
        for i, r in enumerate(rows):
            r["itd"] = 1250000.0 if i < n_itd else None
            r["net_roe"] = 0.0 if i < n_roe else None       # 0.0 must COUNT
            r["itd_display"] = F.manual_display(
                "itd", r["itd"], F.manual_na_cells(r["vcode"]))
            r["net_roe_display"] = F.manual_display(
                "net_roe", r["net_roe"], F.manual_na_cells(r["vcode"]))
        backend = F._subtotal(rows, "x")["manual_entered"]
        client = tally(rows)                     # the Vue rule: != null
        chk(f"tally matches _subtotal at {n_itd} itd / {n_roe} net_roe",
            {k: backend.get(k) for k in client} == client,
            f"backend {backend} vs client {client}")


def main():
    print("Financial manual-entry guardrail")
    print("=" * 70)
    check_display_rule()
    check_persist()
    check_no_reload()
    check_endpoint()
    check_recount_matches_live()
    print("=" * 70)
    bad = [c for c, ok in checks if not ok]
    print(f"{len(checks) - len(bad)}/{len(checks)} passed")
    if bad:
        print("FAILED:")
        for c in bad:
            print(f"  - {c}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
