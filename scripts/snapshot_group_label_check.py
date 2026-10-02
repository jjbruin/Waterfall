"""Guardrail: the Snapshot's fund group labels, header and subtotal.

Pure fixtures — no database, no API, no network — so this runs in the container
and on a laptop with an empty SQLite alike.

WHY THIS FILE EXISTS. A group's printed name reaches the page by two different
routes that look like one:

  * the SUBTOTAL row takes `group_total_label(key)`, which falls back to
    ``"Total <key>"`` for any fund not in the map;
  * the HEADER row on Operating and Loan renders the GROUP KEY itself, with no
    map behind it at all.

So a new fund gets a half-right page without anything failing: TGA6 arrived at
26Q2 and printed ``TGA6`` as its header and ``Total TGA6`` as its subtotal,
while the sent TIAA report calls it ``TGA VI`` in both places. Nothing errored,
nothing was missing, and the only way to notice was to read the page against
the report.

Both directions are asserted throughout. "TGA6 is relabelled" is satisfied by
relabelling every group; "no other fund changed" is satisfied by changing
nothing. Each pair below pins one against the other.

Run:  python scripts/snapshot_group_label_check.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask_app.services.portfolio_snapshot_service import (   # noqa: E402
    GROUP_DISPLAY_LABELS, GROUP_TOTAL_LABELS, INDIVIDUAL_GROUP,
    PORTFOLIO_TOTAL_LABEL, group_display_label, group_total_label,
)

PASS = FAIL = 0
FAILURES = []


def chk(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [ok]   {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  [FAIL] {label}" + (f" - {detail}" if detail else ""))


def section(t):
    print(f"\n{t}")


#: The six groups the TIAA 26Q2 Snapshot actually carries, from live.
LIVE_GROUPS = [INDIVIDUAL_GROUP, "TGA22", "TGA23", "TGA24", "TGA25", "TGA6"]

#: What the SENT TIAA report prints, transcribed. Header, then subtotal.
#: Only TGA6 was wrong; the other five are here so a change to any of them
#: fails rather than passing quietly.
SENT_REPORT = {
    INDIVIDUAL_GROUP: (INDIVIDUAL_GROUP, "Total Individual Investments"),
    "TGA22": ("TGA22", "Total PSC TGA 2022 LLC"),
    "TGA23": ("TGA23", "Total PSC TGA 2023 LLC"),
    "TGA24": ("TGA24", "Total PSC TGA 2024 LLC"),
    "TGA25": ("TGA25", "Total PSC TGA 2025 LLC"),
    "TGA6":  ("TGA VI", "Total TGA VI"),
}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    section("1. TGA6 prints the Roman numeral, in BOTH places")
    chk("the header reads 'TGA VI', not the entity code",
        group_display_label("TGA6") == "TGA VI",
        f"got {group_display_label('TGA6')!r}")
    chk("the subtotal reads 'Total TGA VI', not 'Total TGA6'",
        group_total_label("TGA6") == "Total TGA VI",
        f"got {group_total_label('TGA6')!r}")

    section("2. No other fund's label moved (the other half of the rule)")
    for key, (header, total) in SENT_REPORT.items():
        if key == "TGA6":
            continue
        chk(f"{key} header unchanged", group_display_label(key) == header,
            f"got {group_display_label(key)!r}, want {header!r}")
        chk(f"{key} subtotal unchanged", group_total_label(key) == total,
            f"got {group_total_label(key)!r}, want {total!r}")
    chk("exactly ONE group is given a header override",
        set(GROUP_DISPLAY_LABELS) == {"TGA6"}, str(sorted(GROUP_DISPLAY_LABELS)))

    section("3. The whole sent page, header and subtotal, group by group")
    got = {g: (group_display_label(g), group_total_label(g)) for g in LIVE_GROUPS}
    chk("every group on the TIAA 26Q2 Snapshot matches the sent report",
        got == SENT_REPORT,
        "differs: " + str({k: (v, SENT_REPORT.get(k))
                           for k, v in got.items() if SENT_REPORT.get(k) != v}))

    section("4. The two maps stay distinct")
    # Deriving the header by stripping "Total " would rewrite TGA22's header
    # from "TGA22" to "PSC TGA 2022 LLC". It must not be done that way.
    chk("the header is NOT the total label with 'Total ' removed",
        group_display_label("TGA22") == "TGA22"
        and group_total_label("TGA22") == "Total PSC TGA 2022 LLC")
    chk("a group in neither map keeps its key, and takes the 'Total <key>' form",
        group_display_label("TGA99") == "TGA99"
        and group_total_label("TGA99") == "Total TGA99")
    chk("the portfolio total row is untouched",
        PORTFOLIO_TOTAL_LABEL == "Portfolio Totals")

    section("5. Every subtab reaches the SAME two functions")
    # Financial prints no header and names the fund only on its total row;
    # Operating and Loan print both. All three must label a group identically,
    # which is only guaranteed while they share these functions rather than
    # keeping a copy each.
    import inspect
    from flask_app.services import (portfolio_snapshot_financial as fin,
                                    portfolio_snapshot_loan as loan,
                                    portfolio_snapshot_operating as op)
    for mod, name in ((fin, "financial"), (op, "operating"), (loan, "loan")):
        src = inspect.getsource(mod)
        chk(f"{name} calls group_total_label rather than keeping its own map",
            "group_total_label" in src and "GROUP_TOTAL_LABELS" not in src)
    for mod, name in ((op, "operating"), (loan, "loan")):
        src = inspect.getsource(mod)
        chk(f"{name} publishes group_display_labels for the header",
            '"group_display_labels"' in src and "group_display_label" in src)
    chk("financial does NOT publish a header label — it prints no header row",
        '"group_display_labels"' not in inspect.getsource(fin))

    section("6. The components fall back to the key, so a frozen payload still renders")
    # A Snapshot frozen before this field existed has no `group_display_labels`.
    # The components must render the key rather than an empty header cell.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for comp in ("SnapshotOperating.vue", "SnapshotLoan.vue"):
        path = os.path.join(root, "vue_app", "src", "components", "snapshot", comp)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        chk(f"{comp} renders the mapped header",
            "groupHeader(blk.group)" in src and "{{ blk.group }}" not in src,
            "the raw key is still being rendered")
        chk(f"{comp} falls back to the key when the field is absent",
            "group_display_labels || {})[g] || g" in src)

    print(f"\n{PASS} passed, {FAIL} failed")
    if FAILURES:
        print("failed:")
        for f in FAILURES:
            print(f"  - {f}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
