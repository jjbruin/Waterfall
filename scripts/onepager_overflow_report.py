"""Which One Pagers overflow one printed page, and by how much.

At the original print layout the sheet is a fixed one page (`height: calc(100vh
- 0.8in)`) with `overflow: hidden`, and the two comment blocks are the only
things that can give. Anything past the bottom is simply not painted — there is
no marker, no ellipsis, and the PDF looks like a complete document. So "does
this deal fit" cannot be answered by reading the page; it has to be answered by
comparing the page against the text that was supposed to be on it.

METHOD. For each deal: fetch the comment fields from the live one-pager API
(the source of truth the asset manager typed), then find the longest PREFIX of
each field that actually appears in the rendered PDF. Whitespace is stripped
from both sides of the comparison, because print re-wraps lines and re-splits
hyphenated words — see scripts/onepager_print_geometry.py for the full reason.
The prefix length is what fitted; the remainder is what was silently dropped.

Prefix rather than containment: the text is clipped at the bottom, so what
survives is always an opening run, and a prefix search tells us WHERE the cut
fell rather than merely that a cut happened.

The report also derives the fitting BUDGET, which is the number an author
needs: the largest narrative that printed whole, against the smallest that did
not. Everything between those two is the uncertain band.

Read-only. Consumes the PDFs from scripts/onepager_print_sweep.mjs.

Usage
  WF_TOKEN=<jwt> .venv/Scripts/python.exe scripts/onepager_overflow_report.py \
      --tag orig
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request

import pdfplumber

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PDF_DIR = os.path.join(ROOT, "vue_app", ".chartcheck")
UPSTREAM = os.environ.get(
    "WF_UPSTREAM",
    "https://app-waterfall-dev-v2.icyplant-026fb2db.eastus.azurecontainerapps.io")

# The two blocks that have to fit. Both are author-controlled free text; every
# other element on the sheet is a fixed-height table.
FIELDS = [("business_plan_comments", "Business Plan"),
          ("econ_comments", "Perf Comments")]


def squash(s: str | None) -> str:
    return re.sub(r"\s+", "", s or "")


def raw_index_for_squashed(raw: str, n_squashed: int) -> int:
    """Length of the RAW prefix whose whitespace-free form is n_squashed long.

    The prefix match has to happen in whitespace-free space, because print
    re-wraps; but the on-screen character counter — the number an author is
    actually looking at while trimming — counts RAW characters, spaces and
    newlines included. Reporting the squashed figure would hand someone a
    budget in units their counter does not use (Burton reads 1,477 on screen
    and 1,217 squashed). This walks the raw string to convert exactly, rather
    than scaling by a ratio.
    """
    seen = 0
    for i, ch in enumerate(raw):
        if not ch.isspace():
            seen += 1
            if seen == n_squashed:
                return i + 1
    return len(raw)


def longest_prefix_present(src: str, page: str) -> int:
    """Length of the longest opening run of `src` that appears in `page`.

    Binary search: `src[:n] in page` is monotonic in n for clipped text — if
    the first n characters printed then so did the first n-1.
    """
    if not src:
        return 0
    if src in page:
        return len(src)
    lo, hi = 0, len(src)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if src[:mid] in page:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--deals",
                    default=os.path.join(HERE, "onepager_print_population.txt"))
    args = ap.parse_args()

    token = os.environ.get("WF_TOKEN")
    if not token:
        print("WF_TOKEN not set", file=sys.stderr)
        return 2

    deals = [l.split() for l in open(args.deals, encoding="utf-8")
             if l.strip() and not l.startswith("#")]

    rows = []
    for vcode, quarter in deals:
        p = os.path.join(PDF_DIR, f"onepager_{vcode}_{quarter}_{args.tag}.pdf")
        if not os.path.exists(p):
            continue
        req = urllib.request.Request(
            f"{UPSTREAM}/api/financials/{vcode}/one-pager?quarter={quarter}",
            headers={"Authorization": f"Bearer {token}"})
        payload = json.load(urllib.request.urlopen(req, timeout=180))
        comments = payload.get("comments") or {}
        name = (payload.get("general") or {}).get("investment_name") or vcode

        with pdfplumber.open(p) as pdf:
            page = squash("".join((pg.extract_text() or "") for pg in pdf.pages))
            # characters per printed line, measured rather than assumed — needed
            # to turn a character overflow into "how many lines to cut".
            pg0 = pdf.pages[0]
            lines: dict[float, int] = {}
            for c in pg0.chars:
                lines[round(c["top"], 1)] = lines.get(round(c["top"], 1), 0) + 1
            body = [n for n in lines.values() if n > 40]
            cpl = (sum(body) / len(body)) if body else 120

        rec = {"vcode": vcode, "name": name, "cpl": cpl}
        for key, label in FIELDS:
            raw = comments.get(key) or ""
            src = squash(raw)
            fit_sq = longest_prefix_present(src, page)
            fit_raw = raw_index_for_squashed(raw, fit_sq) if fit_sq else 0
            # every figure below is in RAW characters, matching the counter
            rec[key] = {"total": len(raw), "fit": fit_raw,
                        "lost": len(raw) - fit_raw if fit_sq < len(src) else 0}
        rec["lost_total"] = sum(rec[k]["lost"] for k, _ in FIELDS)
        rows.append(rec)

    over = sorted([r for r in rows if r["lost_total"] > 0],
                  key=lambda r: -r["lost_total"])
    ok = [r for r in rows if r["lost_total"] == 0]

    print("\nCounts are RAW characters — the same figure the on-screen counter "
          "shows,\nso every target below is directly actionable while editing.")
    print(f"\nDeals rendered: {len(rows)}    "
          f"fit one page: {len(ok)}    OVERFLOW (text silently dropped): {len(over)}\n")

    if over:
        hdr = (f"{'#':>3}  {'deal':<34}{'vcode':<11}"
               f"{'BusPlan':>9}{'fit':>8}{'lost':>7}   "
               f"{'PerfCmt':>9}{'fit':>7}{'lost':>7}   {'~lines':>7}")
        print(hdr)
        print("-" * len(hdr))
        for i, r in enumerate(over, 1):
            bp, ec = r["business_plan_comments"], r["econ_comments"]
            print(f"{i:>3}  {r['name'][:33]:<34}{r['vcode']:<11}"
                  f"{bp['total']:>9}{bp['fit']:>8}{bp['lost']:>7}   "
                  f"{ec['total']:>9}{ec['fit']:>7}{ec['lost']:>7}   "
                  f"{r['lost_total']/max(r['cpl'],1):>7.1f}")

    # ---- the budget an author can actually aim at -----------------------
    print()
    for key, label in FIELDS:
        fitted = [r[key]["total"] for r in rows if r[key]["lost"] == 0 and r[key]["total"]]
        clipped = [r[key]["total"] for r in rows if r[key]["lost"] > 0]
        widest_ok = max(fitted) if fitted else 0
        narrowest_bad = min(clipped) if clipped else None
        print(f"{label}:")
        print(f"   largest that printed WHOLE      : {widest_ok:,} chars")
        if narrowest_bad is not None:
            print(f"   smallest that was CLIPPED       : {narrowest_bad:,} chars")
            print(f"   => safe budget                  : under ~{narrowest_bad:,} chars"
                  f"  (uncertain band {widest_ok:,}-{narrowest_bad:,})")
        else:
            print("   nothing clipped — no budget ceiling observed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
