"""Assemble one Portfolio Snapshot subtab LOCALLY against live data, as JSON.

Exists so a JS-side check can exercise the local Python backend. The REST
endpoints run the *deployed* build, so a local rollup or assembly change is
invisible through them — scripts/snapshot_summary_charts_check.mjs silently
reported pre-fix allocation numbers until this was threaded in.

Read-only: GET only against live for the inputs (deals, relationships, One
Pager), then the real committed assembler runs in-process.

Usage
    python scripts/snapshot_payload_dump.py summary TGAM 2026-Q1 > payload.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd                                              # noqa: E402
import live_api as api                                           # noqa: E402
from flask_app.serializers import safe_json                       # noqa: E402
from flask_app.services.portfolio_snapshot_service import (       # noqa: E402
    resolve_investor_deals,
)


def _relationships(investor):
    """Narrow per-entity pulls, so OFFSET paging is never used. See
    live-azure-readonly-access: paging this endpoint duplicates rows."""
    def fetch(col, val):
        d = api.get("/api/data/tables/relationships/rows",
                    params={"page": 1, "page_size": 500, f"filter__{col}": val})
        return [r for r in (d.get("rows") or [])
                if str(r.get(col) or "").strip().upper() == val.upper()]

    seen, frontier, rows = set(), [investor], []
    while frontier:
        node = frontier.pop().upper()
        if node in seen:
            continue
        seen.add(node)
        kids = fetch("InvestorID", node)
        rows.extend(kids)
        for r in kids:
            child = str(r.get("InvestmentID") or "").strip().upper()
            if child:
                rows.extend(fetch("InvestmentID", child))
                if child not in seen:
                    frontier.append(child)
    return pd.DataFrame(rows).drop_duplicates()


def main():
    if len(sys.argv) < 4:
        print(__doc__, file=sys.stderr)
        return 2
    subtab, investor, quarter = sys.argv[1], sys.argv[2], sys.argv[3]

    inv = pd.DataFrame(api.get("/api/data/deals/all").get("deals") or [])
    resolved = resolve_investor_deals(investor, quarter,
                                      _relationships(investor), inv)

    cache = {}

    def one_pager(vcode, q):
        if (vcode, q) not in cache:
            cache[(vcode, q)] = api.get(f"/api/financials/{vcode}/one-pager",
                                        params={"quarter": q})
        return cache[(vcode, q)]

    if subtab == "bundle":
        # The whole /bundle payload, assembled LOCALLY. Shaped exactly like the
        # endpoint's response so scripts/snapshot_print_check.mjs can serve it
        # in place of the proxied one — the deployed backend cannot show local
        # assembly work, which is how two rounds of changes went unverified in
        # the printed document.
        from flask_app.services.portfolio_snapshot_summary import assemble_summary
        from flask_app.services.portfolio_snapshot_financial import (
            assemble_financial,
        )
        from flask_app.services.portfolio_snapshot_operating import (
            assemble_operating,
        )
        from flask_app.services.portfolio_snapshot_loan import assemble_loan
        from flask_app.services.portfolio_snapshot_debt import (
            committed_facility, deal_loan_rows,
        )

        def table(name, page_size=500):
            d = api.get(f"/api/data/tables/{name}/rows",
                        params={"page": 1, "page_size": page_size})
            n = d.get("total") or 0
            if n > page_size:
                print(f"WARNING: {name} has {n} rows, only {page_size} fetched "
                      f"— paging this endpoint duplicates rows, so the frame is "
                      f"deliberately left short rather than corrupted",
                      file=sys.stderr)
            return pd.DataFrame(d.get("rows") or [])

        loans = table("loans")
        vals = table("valuations")

        subtabs, errors = {}, {}
        jobs = {
            "summary": lambda: assemble_summary(
                investor, quarter, resolved=resolved,
                one_pager_provider=one_pager,
                comment_loader=lambda i, q: {},
                editable_loader=lambda i, q: True),
            "financial": lambda: assemble_financial(
                investor, quarter, resolved=resolved,
                one_pager_provider=one_pager,
                committed_debt_provider=lambda vc: committed_facility(
                    deal_loan_rows(loans, vc)),
                manual_loader=lambda i, q: {},
                footnote_loader=lambda i, q: []),
            "operating": lambda: assemble_operating(
                investor, quarter, resolved=resolved,
                one_pager_provider=one_pager,
                comment_loader=lambda i, q: {}),
            # NOTE no quarterly_noi_provider. That needs the ISBS frame, which
            # is 800k+ rows and not fetchable over REST, so Debt Yield comes out
            # None here and its column — and its subtotal — reads as a dash. The
            # real app wires it in build_subtab, so this is a limitation of the
            # dump, NOT of the subtab. Do not read the dash as a bug.
            "loan": lambda: assemble_loan(
                investor, quarter, resolved=resolved,
                one_pager_provider=one_pager, loans=loans, valuations=vals,
                inv=inv, comment_loader=lambda i, q: {}),
        }
        for name, fn in jobs.items():
            try:
                subtabs[name] = fn()
            except Exception as exc:                # one bad subtab, not the lot
                errors[name] = f"{type(exc).__name__}: {exc}"
                print(f"WARNING: {name} failed: {exc}", file=sys.stderr)

        out = {
            "subtabs": subtabs,
            "errors": errors,
            "source": "live",
            "resolution": {
                "investor_name": resolved.get("investor_name"),
                "quarter_end": str(resolved.get("quarter_end") or ""),
                "diagnostics": resolved.get("diagnostics"),
                "flagged": resolved.get("flagged"),
            },
        }
    elif subtab == "summary":
        from flask_app.services.portfolio_snapshot_summary import assemble_summary
        out = assemble_summary(investor, quarter, resolved=resolved,
                               one_pager_provider=one_pager,
                               comment_loader=lambda i, q: {},
                               editable_loader=lambda i, q: True)
    elif subtab == "operating":
        from flask_app.services.portfolio_snapshot_operating import (
            assemble_operating,
        )
        out = assemble_operating(investor, quarter, resolved=resolved,
                                 one_pager_provider=one_pager,
                                 comment_loader=lambda i, q: {})
    else:
        print(f"unsupported subtab {subtab!r}", file=sys.stderr)
        return 2

    json.dump(safe_json(out), sys.stdout, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
