"""Investment Metrics — the parts of the report that are EDITORIAL, not derived.

Everything in this file is a transcription of the reference document
``PSC Investment Summary 06.30.26``, not a rule the data can produce:

* the footnote text, verbatim (including the places the reference omits a
  full stop — the wording is the reference's, not ours);
* which footnote markers print after which deal's name;
* the row order, which is neither alphabetical nor by date;
* the words that stand in for a CoC figure — ``Dev.``, ``Lease up``,
  ``N/A``, ``Inf.`` — per deal AND per column.

**Why a config and not a rule.** Two reasons, and they are different.

``Dev.`` and ``Lease up`` look derivable from ``deals.Lifecycle``, and they are
not: ``Investment_Strategy`` and ``Investment_Category`` are NULL on all 134
rows of the live table, and ``Lifecycle`` disagrees with the reference — it
carries ``Sold`` for deals the reference shows as ``Dev.``, ``New
Construction`` for two of the three the reference calls ``Lease up``, and
``Value-Add`` for 30 Bearfoot, which the reference labels ``Dev.`` in three
columns and prints a number in the fourth. A per-cell judgement cannot come
from a per-deal column.

``Inf.`` is worse: nobody has said what it means. It appears on four sold
deals whose capital was fully returned, so a return-on-equity denominator of
zero is the obvious reading — but "obvious" is not "confirmed", and inventing
a meaning would put a computed number under a label the reader has been
reading as something else. It is transcribed and flagged, pending Alay.

Keyed by **vcode**, never by name: the reference's display names are hand-
edited and differ from MRI's (``Airport Village`` for ``Airport Plaza``,
``870 Donald Lynch`` for ``Donald Lynch``, ``Pittsburgh Portfolio`` for
``Berger Pittsburgh Portfolio``), so a name key would silently stop matching
the first time either side is retyped.
"""

# ── draft gate ────────────────────────────────────────────────────────────
#: THE REPORT IS NOT SIGNED OFF. While this is True the screen and the printed
#: sheet both carry a DRAFT mark, and the sidebar does NOT link to it — the
#: route stays reachable by direct URL so it can be reviewed, without anybody
#: finding it by accident and mailing a page of figures that are still being
#: argued about.
#:
#: THE MARK PRINTS. A banner that vanishes on the way to the printer is worse
#: than no banner: the screen says "draft" and the PDF that gets forwarded says
#: nothing. Both carry it, and both stop carrying it together.
#:
#: One flag drives all three. It is published on `/api/data/config` as
#: `investment_metrics_draft` so the sidebar reads the same switch the report
#: does, and on the report payload as `draft` so the two views do.
#:
#: TODO: set to False once the mismatch table has been worked through — at
#: that point the sidebar link returns and the print sheet is clean.
INVESTMENT_METRICS_DRAFT = True

DRAFT_BANNER = "DRAFT - figures under review"
DRAFT_MARK = "DRAFT"


# ── currency ──────────────────────────────────────────────────────────────
#: CAD -> USD. The reference's footnote (2) states 0.73; the workbook behind it
#: computed at 0.695 (``Investment Metrics``!W3). They disagree, and the
#: footnote is what the reader is told, so the footnote wins here and the
#: difference is reported rather than absorbed. Both are exposed on the payload
#: (``fx_rate`` / ``fx_rate_workbook``) so the choice is visible.
CAD_TO_USD = 0.73
CAD_TO_USD_WORKBOOK = 0.695

#: Deals reported in a currency other than USD, from ``deals.Currency``.
#: Listed here only so the footnote marker and the conversion cannot drift
#: apart; the conversion itself keys off the column, not this set.
NON_USD_NOTE = "Amounts converted to USD at $1.00 CAD = $0.73 USD."

# ── footnotes ─────────────────────────────────────────────────────────────
# Transcribed from the reference. Note (5), (6) and (7) on the Current page and
# (1), (2) and (4) on the Sold page carry NO closing full stop there; that is
# reproduced rather than tidied, because the brief is to match the document.
FOOTNOTES_CURRENT = [
    (1, "Includes preferred return plus excess cash flow, commitment fee, and "
        "residual payments."),
    (2, "Amounts converted to USD at $1.00 CAD = $0.73 USD."),
    (3, "Proceeds to date include a full or partial realization."),
    (4, "PSC receives 8.75% IRR lookback plus 12.0% IRR claw-back against "
        "partner’s cash flow split."),
    (5, "Investments with less than 1 year of operating history reflect "
        "projected year 1 CoC returns in the Act. Yr-1 CoC Returns column"),
    (6, "Investments with less than 1 quarter of operating history reflect "
        "projected year 1 CoC returns in all CoC columns"),
    (7, "Past U/W projected hold period so an ROE to date is not possible"),
    (8, "The transaction size is representative prior to the recapitalization "
        "by TIAA paying off the debt."),
]

FOOTNOTES_SOLD = [
    (1, "Includes preferred return plus excess cash flow, commitment fee, and "
        "residual payments"),
    (2, "City West lost to foreclosure on 08/05/2025, included in IRR "
        "calculations and excluded from CoC calculations"),
    (3, "Apple portfolio is sold on January, 2026; However, the portfolio is "
        "not moved to the sold section until final distributions are "
        "received, which is expected by the end of 2Q27."),
    (4, "Deals sold after June 2026"),
]

#: Printed under both tables, in italic, wrapping to two lines.
DISCLAIMER = (
    "Projected and realized returns are (i) gross returns before any carried "
    "interest or other expenses and (ii) based on estimates believed to be "
    "reasonable in light of information presently available. The material "
    "factors or assumptions that were applied in making the projections and "
    "material factors that could cause actual results to differ materially "
    "from the projections will be provided upon request. There is no "
    "guarantee that the venture will be able to successfully execute on its "
    "strategy or achieve the projected returns."
)

# ── row markers ───────────────────────────────────────────────────────────
#: vcode -> the footnote numbers printed after the deal's name, in order.
ROW_MARKERS_CURRENT = {
    "P0000115": [2, 3, 5],   # Apple - Bales Drive
    "P0000036": [3],         # PMAT Midwest
    "P0000031": [3],         # Old Kinderhook
    "P0000044": [4],         # Woodlands Square
    "P0000037": [3],         # Pontchartrain Landing
    "P0000041": [3],         # The Gathering
    "P0000066": [8],         # Life Storage Staten Island
    "P0000109": [5],         # Burton Retail Portfolio
    "P0000110": [5],         # Trolley Square
    "P0000114": [5],         # Jefferson Stephens
    "P0000116": [5, 6],      # Plaza Del Mar
    "P0000118": [5, 6],      # Hanestowne Village
    "P0000119": [5, 6],      # Presidential Arms
    "P0000120": [5, 6],      # Swartrz Creek Mini Storage
    "P0000117": [5, 6],      # Fairview Center
}

ROW_MARKERS_SOLD = {
    "P0000011": [2],         # City West
    "P0000012": [4],         # Clima Secur
    "P0000001": [4],         # 30 Bearfoot
    "P0000049": [4],         # 870 Donald Lynch
}

#: The Sold table's Total row carries (3) against its realized IRR — the Apple
#: realization is inside that average while the deal itself is in neither
#: table.
SOLD_TOTAL_MARKERS = {"realized_irr": [3]}

# ── the words that stand in for a figure ──────────────────────────────────
#: vcode -> {column: label}. Per CELL, because the reference is: 30 Bearfoot
#: prints ``Dev.`` in three CoC columns and 20.1% in the fourth.
#:
#: The four CoC columns are the only ones ever labelled.
COC_COLUMNS = (
    "proj_yr1_coc", "act_yr1_coc", "proj_coc_since_close", "act_coc_since_close",
)


def _all_four(label):
    return {c: label for c in COC_COLUMNS}


CELL_LABELS_CURRENT = {
    "P0000021": _all_four("Dev."),        # JB Fair Park
    "P0000014": _all_four("Dev."),        # Crowne Plaza
    "P0000067": _all_four("Dev."),        # Brainerd Place
    "P0000078": _all_four("Dev."),        # Jefferson Waters Creek
    "P0000077": _all_four("Dev."),        # Jefferson Addison Heights
    "P0000085": _all_four("Dev."),        # Jefferson Eastchase
    "P0000089": _all_four("Dev."),        # 45th & Main
    "P0000100": _all_four("Dev."),        # Outlook Green Valley Ranch
    "P0000110": _all_four("Dev."),        # Trolley Square
    "P0000114": _all_four("Dev."),        # Jefferson Stephens
    # Lease up — note the reference capitalises Staten Island's differently.
    "P0000006": _all_four("Lease up"),    # 450 Belleville
    "P0000029": _all_four("Lease up"),    # Middle Island
    "P0000066": _all_four("Lease Up"),    # Life Storage Staten Island
    # Past the U/W hold period, so no ROE to date — footnote (7)'s column.
    "P0000028": {"proj_coc_since_close": "N/A"},   # Merle Hay Mall
    "P0000036": {"proj_coc_since_close": "N/A"},   # PMAT Midwest
    "P0000031": {"proj_coc_since_close": "N/A"},   # Old Kinderhook
}

CELL_LABELS_SOLD = {
    "P0000002": _all_four("Dev."),        # 3rd Avenue & Indian
    "P0000024": _all_four("Dev."),        # Jefferson West
    "P0000026": _all_four("Dev."),        # Leander
    "P0000022": _all_four("Dev."),        # Jefferson Oakhurst
    "P0000023": _all_four("Dev."),        # Jefferson Centura
    "P0000034": _all_four("Dev."),        # Outlook Nine Mile
    "P0000011": _all_four("N/A"),         # City West — foreclosure, footnote (2)
    # 30 Bearfoot: three labelled, the fourth a real figure. This row is the
    # reason CELL_LABELS is keyed per column.
    "P0000001": {"proj_yr1_coc": "Dev.", "act_yr1_coc": "Dev.",
                 "proj_coc_since_close": "Dev."},
    # "Inf." — transcribed, NOT interpreted. See the module docstring.
    "P0000039": {"proj_coc_since_close": "Inf.",
                 "act_coc_since_close": "Inf."},   # Shoppers World
    "P0000043": {"proj_coc_since_close": "Inf.",
                 "act_coc_since_close": "Inf."},   # Willowdale
    "P0000042": {"proj_coc_since_close": "Inf.",
                 "act_coc_since_close": "Inf."},   # Village Square
    "P0000007": {"proj_coc_since_close": "Inf.",
                 "act_coc_since_close": "Inf."},   # Pittsburgh Portfolio
    "P0000032": {"proj_coc_since_close": "Inf."},  # Orange Grove
}

#: Realized IRR cells the reference leaves as a dash rather than 0.0%. A
#: foreclosure and a deal whose flows will not solve are not a zero return, and
#: printing 0.0% says they are.
REALIZED_IRR_SUPPRESSED = {
    "P0000011": "lost to foreclosure 08/05/2025 — see footnote (2)",
    "P0000064": "flows do not produce a solvable IRR",   # Adirondack RV Park
}

# ── row order ─────────────────────────────────────────────────────────────
#: The reference's order: Current is broadly by investment date with Apple -
#: Bales Drive lifted to the top; Sold has no derivable order at all. Deals not
#: listed are appended in investment-date order and REPORTED, so a new deal
#: appears rather than vanishing.
ROW_ORDER_CURRENT = [
    "P0000115", "P0000028", "P0000010", "P0000036", "P0000031", "P0000035",
    "P0000044", "P0000004", "P0000019", "P0000040", "P0000033", "P0000037",
    "P0000021", "P0000006", "P0000030", "P0000008", "P0000018", "P0000029",
    "P0000041", "P0000014", "P0000065", "P0000066", "P0000067", "P0000068",
    "P0000069", "P0000079", "P0000078", "P0000076", "P0000077", "P0000080",
    "P0000075", "P0000081", "P0000084", "P0000082", "P0000087", "P0000085",
    "P0000086", "P0000088", "P0000089", "P0000107", "P0000099", "P0000100",
    "P0000109", "P0000110", "P0000114", "P0000116", "P0000118", "P0000119",
    "P0000120", "P0000117",
]

ROW_ORDER_SOLD = [
    "P0000020", "P0000013", "P0000002", "P0000024", "P0000016", "P0000026",
    "P0000005", "P0000022", "P0000025", "P0000023", "P0000039", "P0000009",
    "P0000034", "P0000043", "P0000042", "P0000011", "P0000015", "P0000032",
    "P0000064", "P0000038", "P0000083", "P0000007", "P0000017", "P0000012",
    "P0000001", "P0000049",
]

# ── first lien basis ──────────────────────────────────────────────────────
#: ONE GLOBAL RULE, chosen by measurement rather than by argument. Every
#: candidate was tested against the reference across all 76 deals:
#:
#:     basis                          dev     non-dev      all   no value
#:     A   earliest loan record      7/10      34/66     41/76       19
#:     A+k earliest loan, +children  7/10      34/66     41/76       16
#:     B   summed facility           7/10      34/66     41/76       19
#:     B+k summed facility, +children 7/10     35/66     42/76       16
#:     C   earliest ISBS             0/10      27/66     27/76       18
#:     C+k earliest ISBS, +children  0/10      25/66     25/76       17
#:
#: and the dev / non-dev SPLIT the report used to apply (dev = B+k, everything
#: else = C) scores **34/76** — eight worse than simply using B+k on every
#: deal. The split was a reasonable-sounding rule that the data does not
#: support: the earliest balance-sheet row wins on NO development deal and on
#: only 27 of the 66 others.
#:
#: So the default is B+k for every deal, and `FIRST_LIEN_FALLBACKS` names what
#: to try when it yields nothing.
FIRST_LIEN_BASIS = "summed_facility"
FIRST_LIEN_FALLBACKS = ("earliest_loan", "earliest_isbs")

#: Development deals. NO LONGER SELECTS THE FIRST-LIEN BASIS — kept because it
#: is the population footnote (5)-era labelling refers to, and the guardrail
#: asserts it agrees with the deals labelled ``Dev.`` in CELL_LABELS.
#:
#: Named here rather than read from ``Lifecycle`` for the reason in the module
#: docstring — ``Lifecycle`` says ``Sold`` for six of these.
DEV_DEALS = {
    "P0000021",   # JB Fair Park
    "P0000014",   # Crowne Plaza
    "P0000067",   # Brainerd Place
    "P0000078",   # Jefferson Waters Creek
    "P0000077",   # Jefferson Addison Heights
    "P0000085",   # Jefferson Eastchase
    "P0000089",   # 45th & Main
    "P0000100",   # Outlook Green Valley Ranch
    "P0000110",   # Trolley Square
    "P0000114",   # Jefferson Stephens
}

#: Deals the reference carries in NEITHER table, excluded by name so the
#: omission is deliberate and visible rather than a rule nobody can find.
#:
#: Apple Self Storage sold in January 2026 but its ``Sale_Status`` is still
#: NULL, so every population rule puts it in Current. The reference does not
#: show it in Current and does not show it in Sold either — footnote (3) on the
#: Sold total explains that the portfolio is not moved across until the final
#: distributions land, expected end of 2Q27. Until then it belongs nowhere, and
#: the report says so out loud in `diagnostics.excluded_deals`.
EXCLUDED_DEALS = {
    "P0000003": "Apple Self Storage — realized Jan 2026, Sale_Status still "
                "NULL; the reference carries it in neither table until final "
                "distributions are received (Sold footnote 3, expected 2Q27)",
}

LEASE_UP_DEALS = {
    "P0000006",   # 450 Belleville
    "P0000029",   # Middle Island
    "P0000066",   # Life Storage Staten Island
}

# ── figures that do not exist in the app yet ──────────────────────────────
# THREE COLUMNS HAVE NO SOURCE THE APP CAN READ, and each is here rather than
# hard-coded to None in the engine so that switching one on is a one-line
# change in one file.
#
#   `mode`     what the report PRINTS:
#                "mri"       read `field` off the deal's row; em dash while
#                            that column does not exist or is NULL
#                "computed"  print the figure the engine derives
#                "none"      always an em dash; the derived figure is kept in
#                            the row's `alternates` and in diagnostics, never
#                            rendered
#   `table`    where `field` is expected to land
#   `field`    the column name. **None until Alay loads it.**
#   `variant`  for a computed figure with more than one defined form
#
# TODO(alay): set `field` for each of the three below once the values are
# loaded into MRI, and move `mode` to "mri". Nothing else has to change — the
# engine reads this dict, the footnote (5)/(6) substitution reads the resulting
# `proj_yr1_coc`, and the pref-weighted averages skip whatever is still None.
#
# WHY act_yr1_coc PRINTS NOTHING TODAY even though the engine can derive it:
# the reference's figures in that column are not reproducible from the
# accounting feed under any window (see `act_year_one_coc`), so publishing a
# derived number under the same heading would invite it to be read as the same
# quantity. It is computed, kept, and not shown.
UNLOADED_FIGURES = {
    "uw_irr": {
        "mode": "none",
        "table": "deal_terms",
        "field": None,                       # TODO(alay): e.g. "uw_projected_irr"
        "label": "UW Proj. IRR",
        "note": "underwritten projected IRR is not held anywhere in MRI",
    },
    "proj_yr1_coc": {
        "mode": "none",
        "table": "deal_terms",
        "field": None,                       # TODO(alay): e.g. "proj_yr1_coc"
        "label": "Proj Yr-1 CoC Returns",
        "note": "projected year-1 cash-on-cash is not held anywhere in MRI; "
                "footnotes (5) and (6) cannot substitute without it",
    },
    "act_yr1_coc": {
        "mode": "none",
        "table": "deal_terms",
        "field": None,                       # TODO(alay): if actuals are loaded
        "variant": "funded",                 # "funded" | "commitment"
        "label": "Act. Yr-1 CoC Returns",
        "note": "the derived figure is a different quantity from the "
                "reference's — kept in alternates, not rendered",
    },
}


# ── proceeds ──────────────────────────────────────────────────────────────
#: ``SubtypeUID`` values that make up Proceeds to Date on a CURRENT deal, taken
#: from the reference workbook's own formula (``Investment Metrics``!V):
#: 1021 Acquisition Fee, 1016 Return of Capital, 1019 Preferred Return,
#: 1020 Excess Cash Flow. Footnote (1) describes exactly this set — "preferred
#: return plus excess cash flow, commitment fee, and residual payments".
#:
#: NOTE the acquisition fee is INCLUDED here and EXCLUDED from ROE by
#: ``one_pager.get_pe_performance``. That is not an inconsistency: a fee is
#: cash the investor received (so it is a proceed) and is not a return on
#: equity (so it is not in ROE).
PROCEEDS_SUBTYPES_CURRENT = (1016, 1019, 1020, 1021)

#: On a SOLD deal every distribution counts, because the deal is finished and
#: the question is what came back in total.
PROCEEDS_MAJORTYPE_SOLD = "distri"

#: Preferred Return — the subtype whose first 365 days make Year-1 CoC.
SUBTYPE_PREFERRED_RETURN = 1019

# ── column headings and geometry, measured from the reference ─────────────
# The heading text is transcribed exactly, including the ``(1)``/``(5)``/``(7)``
# markers the reference prints inside a heading and the trailing space after
# "Since Close" on the Current page's last CoC column.
#
# ``w`` is the column's width in POINTS, taken from the per-column underline
# rules drawn under the third header row: the cell boundaries are the underline
# starts, and each underline stops 1.08pt short of the next so the rules read
# as separate columns. ``align`` was measured too, not assumed — the five text
# columns are left-aligned on the boundary and EVERY numeric column is
# CENTRED, which is not what a financial table usually does and is easy to
# "correct" by mistake.
#
# key / row1 / row2 / row3 / w / align
COLUMNS_CURRENT = [
    ("name", "", "", "", 68.16, "left"),
    ("asset_class", "", "", "Asset Class", 51.12, "left"),
    ("dma", "", "", "DMA/Location", 40.92, "left"),
    ("invest_date", "PSC", "Invest.", "Date", 40.92, "left"),
    ("partner", "", "Oper.", "Partner", 44.28, "left"),
    ("total_size", "Total", "Trans.", "Size", 22.68, "center"),
    ("first_lien", "", "", "Amount", 37.80, "center"),
    ("first_lien_pct", "", "", "% of Cap.", 22.68, "center"),
    ("pref", "", "", "Amount", 35.52, "center"),
    ("pref_pct", "", "", "% of Cap.", 22.68, "center"),
    ("first_loss", "", "", "Amount", 37.80, "center"),
    ("first_loss_pct", "", "", "% of Cap.", 22.68, "center"),
    ("uw_irr", "UW", "Proj.", "IRR(1)", 22.68, "center"),
    ("proceeds", "", "Proceeds", "To-Date", 22.68, "center"),
    ("proj_yr1_coc", "Proj Yr-1", "CoC", "Returns (1)", 22.68, "center"),
    ("act_yr1_coc", "Act.Yr-1", "CoC", "Returns (1) (5)", 29.52, "center"),
    ("proj_coc_since_close", "CoC", "Proj. Returns", "Since Close (7)", 41.40, "center"),
    ("act_coc_since_close", "CoC", "Act. Returns", "Since Close ", 30.12, "center"),
    ("pref_coupon", "PSC", "Pref", "Coupon", 28.56, "center"),
    ("residual_cf_split", "", "Residual", "CF Split", 22.32, "center"),
    ("irr_lookback", "", "IRR", "Lookback", 23.16, "center"),
    ("_spacer", "", "", "", 23.16, "center"),
]

COLUMNS_SOLD = [
    ("name", "", "", "", 68.16, "left"),
    ("asset_class", "", "", "Asset Class", 51.12, "left"),
    ("dma", "", "", "DMA/Location", 40.92, "left"),
    ("invest_date", "PSC", "Invest.", "Date", 40.92, "left"),
    ("partner", "", "Oper.", "Partner", 44.28, "left"),
    ("total_size", "Total", "Trans.", "Size", 22.68, "center"),
    ("first_lien", "", "", "Amount", 37.80, "center"),
    ("first_lien_pct", "", "", "% of Cap.", 22.68, "center"),
    ("pref", "", "", "Amount", 35.52, "center"),
    ("pref_pct", "", "", "% of Cap.", 22.68, "center"),
    ("first_loss", "", "", "Amount", 37.80, "center"),
    ("first_loss_pct", "", "", "% of Cap.", 22.68, "center"),
    ("uw_irr", "UW", "Proj.", "IRR(1)", 22.68, "center"),
    ("realized_irr", "Realized", "Final", "IRR(1)", 22.68, "center"),
    ("proceeds", "", "Proceeds", "To-Date", 22.68, "center"),
    ("proj_yr1_coc", "Proj Yr-1", "CoC", "Returns (1)", 29.52, "center"),
    ("act_yr1_coc", "Act.Yr-1", "CoC", "Returns (1)", 41.40, "center"),
    ("proj_coc_since_close", "CoC", "Proj. Returns", "Since Close", 30.12, "center"),
    ("act_coc_since_close", "CoC", "Act. Returns", "Since Close", 28.56, "center"),
    ("pref_coupon", "PSC", "Pref", "Coupon", 22.32, "center"),
    ("residual_cf_split", "", "Residual", "CF Split", 23.16, "center"),
    ("irr_lookback", "", "IRR", "Lookback", 23.16, "center"),
]

#: The grouped heading that spans the three capitalization pairs, and the three
#: pair headings under it. ``start``/``span`` are column INDEXES into the lists
#: above — identical on both tables, because the capitalization block sits in
#: the same six slots on each. Each carries a rule under it in the reference.
CAP_GROUP_HEADING = "Underwritten Capitalization at Stabilization"
CAP_GROUP_SPAN = {"start": 6, "span": 6}
CAP_PAIR_HEADINGS = [
    {"label": "First Lien Mortgage", "start": 6, "span": 2},
    {"label": "PSC Pref. Equity", "start": 8, "span": 2},
    {"label": "First-Loss Equity", "start": 10, "span": 2},
]

#: Vertical rules, as column indexes: the rule is drawn on the LEFT edge of the
#: named column and runs from the group-heading row to the foot of the total
#: row. The reference's own choices; they are not symmetrical and not derivable.
VERTICAL_RULES_CURRENT = [8, 10, 13, 18]
VERTICAL_RULES_SOLD = [8, 10, 13, 15]

#: Page geometry, in points (1pt = 1/72in). Letter landscape is 792 x 612.
PAGE = {
    "width": 792.0, "height": 612.0,
    "left": 18.0, "top": 53.64, "right": 734.16, "bottom": 404.16,
    "row_height": 4.68,
    "font_size": 3.6,
    "note_font_size": 4.68,
    "rule_gap": 1.08,
    "band": "#D9D9D9",
}

TITLE_CURRENT = "PSC Investment Summary - Current Portfolio"
TITLE_SOLD = "PSC Investment Summary - Sold Portfolio"
UNITS_NOTE = "($ in USD millions)"
TOTAL_LABEL = "Total / Average"
GRAND_TOTAL_LABEL = "Grand Total"
