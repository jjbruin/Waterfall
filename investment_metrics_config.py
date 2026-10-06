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
#: SWITCHED OFF Sep 30 2026 on Jim's instruction: the banner, the printed
#: DRAFT line and the watermark are gone, and the sidebar links to the report
#: under Asset Management.
#:
#: WHAT THIS DOES NOT MEAN. It is a presentation gate and nothing else — no
#: figure, label, footnote or layout changes with it, which the guardrail
#: asserts in both directions. The open items it was raised over are still
#: open: `UNLOADED_FIGURES` still holds all three Yr-1 / UW IRR columns in
#: "none" mode against an Alay TODO, and the first-lien column still
#: reproduces the reference on only 42 of 76 deals.
INVESTMENT_METRICS_DRAFT = False

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

#: Footnote (2) on the CURRENT page marks the deal whose figures were converted.
#: DERIVED from ``deals.Currency != 'USD'``, not listed — the conversion already
#: keys off that column, so a hand-maintained marker list is a second answer to
#: a question the data settles. Measured on the live table: exactly one deal is
#: non-USD (P0000115 Apple - Bales Drive, ``CAD``) and no row has a blank
#: currency, so the derivation reproduces the reference exactly.
#:
#: The SOLD page's (2) is a different footnote — City West's foreclosure — and
#: stays hardcoded in ``ROW_MARKERS_SOLD``.
NON_USD_MARKER = 2

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
#: vcode -> the footnote numbers printed after the deal's name.
#:
#: THREE OF THESE MARKERS ARE NO LONGER LISTED HERE, because the data settles
#: them and a transcribed copy would be a second answer that goes stale the
#: first time a deal ages past a boundary:
#:
#:   (2) on the Current page   -> ``deals.Currency != 'USD'``  (NON_USD_MARKER)
#:   (5) on the Current page   -> invest date + 12 months > as-of (YOUNG_DEAL_*)
#:   (4) on the Sold page      -> ``sale_date > as_of``   (SOLD_AFTER_AS_OF_MARKER)
#:
#: Everything else stays transcribed: (3) proceeds include a realization, (4) on
#: the CURRENT page (Woodlands Square's lookback/claw-back terms — a different
#: footnote from the Sold page's (4)), (6) under a quarter of operating history,
#: (7) past the U/W hold, (8) the TIAA recapitalization, and the Sold page's (2)
#: City West foreclosure. None of those is a rule the app's tables can express.
#:
#: Derived markers are MERGED with these and sorted, so Apple prints (2)(3)(5)
#: from one hardcoded entry and two derivations.
ROW_MARKERS_CURRENT = {
    "P0000115": [3],         # Apple - Bales Drive   — (2) and (5) derived
    "P0000036": [3],         # PMAT Midwest
    "P0000031": [3],         # Old Kinderhook
    "P0000044": [4],         # Woodlands Square      — the CURRENT page's (4)
    "P0000037": [3],         # Pontchartrain Landing
    "P0000041": [3],         # The Gathering
    "P0000066": [8],         # Life Storage Staten Island
    "P0000109": [],          # Burton Retail Portfolio    — (5) derived
    "P0000110": [],          # Trolley Square             — (5) derived
    "P0000114": [],          # Jefferson Stephens         — (5) derived
    "P0000116": [6],         # Plaza Del Mar              — (5) derived
    "P0000118": [6],         # Hanestowne Village         — (5) derived
    "P0000119": [6],         # Presidential Arms          — (5) derived
    "P0000120": [6],         # Swartrz Creek Mini Storage — (5) derived
    "P0000117": [6],         # Fairview Center            — (5) derived
}

ROW_MARKERS_SOLD = {
    "P0000011": [2],         # City West — foreclosure; NOT the currency note
    # Clima Secur, 30 Bearfoot and 870 Donald Lynch carried (4) here and now
    # derive it from their sale dates. See SOLD_AFTER_AS_OF_MARKER.
}

#: Footnote (4) on the SOLD page — "Deals sold after June 2026". DERIVED from
#: ``sale_date > as_of``, which is the same test ``classify`` already relies on
#: to keep those deals in the Sold table at all.
#:
#: Verified at as-of 2026-06-30 against the reference: the rule selects exactly
#: Clima Secur (2026-07-01), 30 Bearfoot (2026-09-04) and 870 Donald Lynch
#: (2026-09-04), and nothing else. The margin is five days — East Manchester
#: sold 2026-06-25 and is correctly NOT marked — so the guardrail asserts the
#: set rather than the count.
SOLD_AFTER_AS_OF_MARKER = 4

#: Footnote (5) — "less than 1 year of operating history". DERIVED: the deal's
#: PSC Invest. Date plus ``YOUNG_DEAL_MONTHS`` calendar months falls after the
#: as-of date.
#:
#: Verified at as-of 2026-06-30 against the reference: exactly the nine deals
#: the reference marks (Apple, Burton, Trolley Square, Jefferson Stephens,
#: Plaza Del Mar, Hanestowne, Presidential Arms, Swartz Creek, Fairview) and no
#: others. The boundary is not tight — the newest UNMARKED deal is Green Valley
#: Ranch at 16 months and the oldest marked one is Apple at 11.5 — so the rule
#: is not balanced on a single day.
#:
#: (6) — under a QUARTER of operating history — stays hardcoded. Every (6) deal
#: is also a (5) deal, so it selects no cell (5) has not already selected, and
#: the reference's choice of which deals carry it is editorial.
YOUNG_DEAL_MARKER = 5
YOUNG_DEAL_MONTHS = 12

#: What a (5) deal shows instead of its own figure: the PROJECTED year-1 CoC.
#:
#: NOTE WHAT THE REFERENCE ACTUALLY DOES. The printed note names only the
#: "Act. Yr-1 CoC Returns column", but the table applies the projected value in
#: THREE CoC columns — Apple prints 4.0% in Act. Yr-1, CoC Proj. Since Close and
#: CoC Act. Since Close alike, all equal to its Proj Yr-1 CoC. The table is
#: reproduced, and the wording of the note is reproduced verbatim beside it;
#: neither is edited to agree with the other.
#:
#: ``proj_yr1_coc`` is the SOURCE and is never itself substituted.
YOUNG_DEAL_SUBSTITUTED_COLUMNS = (
    "act_yr1_coc", "proj_coc_since_close", "act_coc_since_close",
)

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

#: MAY THE EARLIEST BALANCE-SHEET ROW STAND IN FOR A MISSING LOAN RECORD?
#:
#: NO. Measured at as-of 2026-06-30 against the reference: of the seven deals
#: that reach this fallback (Life Storage / Pegasus, Jefferson Oakhurst,
#: Jefferson Centura, Shoppers World, Camarillo, Willowdale, Village Square)
#: NONE ties the reference at +/-$0.05M. The differences run from -$35.1M to
#: +$22.2M, and four of the seven are sold deals whose balance sheet stops years
#: before the sale -- the first row on file is whatever was outstanding when the
#: books begin, not what the deal was capitalised with. A number that is wrong on
#: every deal it touches, and looks like any other, is worse than a dash.
#:
#: With this False a deal with no usable loan record prints an em dash for First
#: Lien; Total Size and the three "% of Cap." cells cascade to dashes exactly as
#: they do for any other unknown first lien, and the basis reads "none". The ISBS
#: figure is still computed and published under ``alternates.first_lien``, and
#: every deal it was withheld from is listed in
#: ``diagnostics["first_lien_isbs_withheld"]`` -- the engine flags, it never
#: drops. Set True to restore the fallback (``earliest_isbs`` then follows the
#: loan bases in ``FIRST_LIEN_FALLBACKS``).
FIRST_LIEN_ISBS_FALLBACK = False

#: THE FIRST LIEN IS THE LOAN THAT WAS ORIGINATED FIRST, when the data can say so.
#:
#: "First lien" means the senior mortgage, and seniority is settled at
#: origination — so where MRI records an origination date on every one of a
#: deal's loans, the first lien is the loan (or loans) sharing the EARLIEST one,
#: and a later supplemental or mezzanine facility is not part of it.
#:
#: MATURITY IS NEVER A PROXY. ``dtEvent`` carries a maturity date on 83 of the 91
#: live loan rows, and the existing ``earliest_loan`` basis sorts on it — so what
#: it calls "the earliest loan" is the earliest-MATURING facility, which has no
#: relationship to seniority. A five-year supplemental taken out in year three
#: matures before a ten-year first mortgage taken out at closing.
#:
#: THE RULE, per deal, against the RAW loans frame:
#:
#:   * exactly one loan                      -> that loan;
#:   * several loans AND an Origination row
#:     on every one of them                  -> the sum of the loans sharing the
#:                                              earliest origination date;
#:   * anything else                         -> the basis above, unchanged, and
#:                                              the LoanIDs with no origination
#:                                              date are named in
#:                                              ``first_lien_origination_missing``.
#:
#: PAID-OFF LOANS COUNT. The column is the capitalization AT STABILIZATION, which
#: is a fact about how the deal was financed, not about what is outstanding now —
#: so a facility since repaid is still part of it. This is why the report reads
#: ``mri_loans_all`` rather than the filtered ``mri_loans_raw``.
#:
#: DEVELOPMENT DEALS ARE UNCHANGED: they take the committed facility, because a
#: construction loan's origination tells you when the draw began, not what was
#: committed.
#:
#: MEASURED BEFORE SHIPPING, on the live table at as-of 2026-06-30: 49 deals have
#: exactly one loan, 16 have none, and 11 have several — and NOT ONE of the 11
#: carries an origination date on every loan, because the whole live table holds
#: only four Origination rows and all four sit on single-loan deals. So the rule
#: changes no printed figure today. It is here so that the first loan loaded with
#: an origination date is read as a first lien rather than as a maturity.
FIRST_LIEN_FROM_ORIGINATION = True

#: Which vcodes the origination search covers on a deal that has child
#: properties. ``"deal"`` takes the earliest origination across the parent and
#: every child together; ``"property"`` takes each property's own earliest and
#: sums those.
#:
#: They differ only for a portfolio whose properties closed on different days.
#: Both are computed on every row and published under
#: ``alternates.first_lien_origination`` so the choice stays measurable; at
#: as-of 2026-06-30 no deal reaches the origination path at all, so the two are
#: identical on all 76 rows and neither ties the reference better than the other.
FIRST_LIEN_CHILD_BASIS = "deal"

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
# `uw_irr` and `proj_yr1_coc` are now "mri": `Prop_Info_DealTerms.sql` pivots
# them into `deal_terms.uw_irr` / `deal_terms.proj_yr1_coc`. Until a refresh of
# that table lands them the column is ABSENT and every deal prints an em dash;
# after it, a deal MRI has no row for is NULL and prints an em dash too. The two
# are reported separately (`unloaded_figure_field_absent` /
# `unloaded_figure_value_null`). The footnote (5)/(6) substitution reads the
# resulting `proj_yr1_coc`, and the pref-weighted averages skip None.
#
# WHY act_yr1_coc PRINTS NOTHING TODAY even though the engine can derive it:
# the reference's figures in that column are not reproducible from the
# accounting feed under any window (see `act_year_one_coc`), so publishing a
# derived number under the same heading would invite it to be read as the same
# quantity. It is computed, kept, and not shown.
UNLOADED_FIGURES = {
    "uw_irr": {
        "mode": "mri",
        "table": "deal_terms",
        "field": "uw_irr",                   # MRI vtranstype 'U/W IRR'
        "label": "UW Proj. IRR",
        "note": "MRI Financial Transaction Summary 'U/W IRR', an undated "
                "underwriting figure; latest dtEffective per deal",
    },
    "proj_yr1_coc": {
        "mode": "mri",
        "table": "deal_terms",
        "field": "proj_yr1_coc",             # MRI vtranstype 'Projected Yr 1 CoC Returns'
        "label": "Proj Yr-1 CoC Returns",
        "note": "MRI Financial Transaction Summary 'Projected Yr 1 CoC "
                "Returns', an undated underwriting figure; feeds the "
                "footnote (5)/(6) substitution",
    },
    # SWITCHED ON. This column now PRINTS a derived figure — the one defined by
    # `investment_metrics.act_year_one_coc_roe`: the One Pager's ROE engine,
    # windowed to the deal's first twelve months.
    #
    # WHAT CHANGED, AND WHY IT IS NOT THE QUANTITY THE OLD NOTE REFUSED TO PRINT.
    # The figure previously computed here was preferred return received in the
    # first 365 days over funded-to-date. That is a coupon-collection ratio, not
    # a return on equity: it ignores excess cash flow, ignores the capital
    # actually at risk over the window, and returns 0.0% on ten deals that paid
    # no pref in year one. The ROE-windowed figure is the same quantity the
    # reference's column is headed with, measured the way the app measures ROE
    # everywhere else.
    #
    # MEASURED AGAINST THE REFERENCE at as-of 2026-06-30, at display rounding:
    #   ROE-windowed   24/76      <- shipped
    #   pref / funded  23/76      <- the old alternate
    #   pref / commit  23/76
    # and the column as shipped BEFORE this change (always an em dash, labels
    # only) scored 21/76. So it is three cells better than printing nothing.
    #
    # IT IS BIASED LOW: of the 51 rows where both sides carry a number, 40 come
    # in below the reference and 11 above, mean -1.21pp. 21 are within 1pp and
    # 30 within 2pp. The reference's own cells are typed-in constants on 75 of
    # its 76 rows, so the gap is not a window that needs tuning — it is the
    # difference between a derived figure and an editorial one, and it is
    # reported rather than closed by fitting.
    #
    # Both older denominators are still carried in `alternates.act_yr1_coc`.
    "act_yr1_coc": {
        "mode": "computed",
        "table": "deal_terms",
        "field": None,                       # TODO(alay): if actuals are loaded
        "variant": "roe_window",             # "roe_window"|"funded"|"commitment"
        "label": "Act. Yr-1 CoC Returns",
        "note": "ROE over the deal's first twelve months, from the accounting "
                "feed; the reference's cells are typed in and run higher",
    },
}


# ── as-of ─────────────────────────────────────────────────────────────────
#: The quarter the report OPENS ON, pinned.
#:
#: WHY A PIN AND NOT A RULE. ``latest_quarter_end`` returns the most recent
#: quarter end strictly before today, which on 2026-10-02 is 2026-09-30 — a
#: quarter that closed two days ago and behind which there is no closed
#: accounting. The report opened on it and every figure read as a quarter's
#: worth of nothing.
#:
#: ``DEFAULT_QUARTER_LAG_DAYS`` is the rule that replaces this: open on the most
#: recent quarter end that is at least this many days in the past. It is
#: DELIBERATELY NOT WIRED UP — it is defined, asserted inert by the guardrail,
#: and switched on by a separate decision. On 2026-10-02 it would select
#: 2026-06-30, the same answer the pin gives.
#:
#: Set to None to go back to "the most recent quarter end that has finished".
#: Every other quarter stays selectable; this names only which one opens.
DEFAULT_QUARTER = "2026-06-30"

#: Staged, UNUSED. See DEFAULT_QUARTER.
DEFAULT_QUARTER_LAG_DAYS = 45


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

#: Preferred Return — the subtype behind the legacy Year-1 CoC alternate.
SUBTYPE_PREFERRED_RETURN = 1019

#: How long after the quarter end a distribution may land and still count as
#: proceeds for that quarter.
#:
#: ``None`` IS TODAY'S BEHAVIOUR AND THE DEFAULT. Proceeds to Date carries no
#: date bound at all — the column is headed *To-Date* and the reference
#: workbook's own formula has no cutoff either — so nothing is excluded and
#: nothing is admitted late. The setting exists because the ROE engine's 45-day
#: pref grace raises the obvious question for this column too, and the answer
#: should be a named switch rather than a number buried in a filter.
#:
#: Setting it to an integer would mean: count distributions through
#: ``as_of + N days``. It is inert while None, which the guardrail asserts.
PROCEEDS_CUTOFF_DAYS_AFTER_QUARTER = None

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
