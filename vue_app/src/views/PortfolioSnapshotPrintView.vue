<script setup lang="ts">
/**
 * Portfolio Snapshot — the consolidated 4-page print document.
 *
 * WHY A SEPARATE ROUTE. The interactive shell mounts ONE subtab at a time
 * (`v-if="activeTab === ..."`), so printing it produces whichever tab happens
 * to be open — one page, never the document. Rather than teach the shell to
 * mount all four and then hide three, this route mounts all four with no v-if
 * and lays them out as pages. The shell keeps its tab behaviour untouched.
 *
 * PAGE ORDER follows the reference PDF:
 *   1  page header + PORTFOLIO SNAPSHOT title + the two narratives and charts
 *   2  Financial   — fund groups, subtotals, excluding-development, footnotes
 *   3  Operating
 *   4  Loan
 * The big centred title appears on pages 1-2 only; the TIAA client line and the
 * "Balances as of" subtitle repeat on every page, exactly as published.
 *
 * READ-ONLY BY CONSTRUCTION. Every subtab gets `:editable="false"`, so the
 * textareas render as plain text and no save event can fire from a document
 * whose only purpose is to be printed. No mutation endpoint is reachable here.
 */
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api/client'
import SnapshotFinancial from '../components/snapshot/SnapshotFinancial.vue'
import SnapshotOperating from '../components/snapshot/SnapshotOperating.vue'
import SnapshotLoan from '../components/snapshot/SnapshotLoan.vue'
import AllocationStackedBar from '../components/snapshot/AllocationStackedBar.vue'
import DealTypePie from '../components/snapshot/DealTypePie.vue'

// The axios client's baseURL is '/', so this carries the /api prefix — same
// constant as PortfolioSnapshotView. Without it the request 404s to the SPA
// fallback, which returns index.html and leaves `bundle` an object with no
// subtabs: the page then renders its headers and nothing else.
const BASE = '/api/portfolio-snapshot'
const route = useRoute()

const investor = String(route.query.investor || '')
const quarter = String(route.query.quarter || '')
const autoPrint = String(route.query.autoprint || '') === '1'

const bundle = ref<any>(null)
const loading = ref(true)
const loadError = ref('')

const subtabs = computed(() => bundle.value?.subtabs || {})
const errors = computed(() => bundle.value?.errors || {})
const resolution = computed(() => bundle.value?.resolution || null)
const summary = computed(() => subtabs.value.summary || null)

const investorName = computed(
  () => resolution.value?.investor_name || investor)

/** ISO -> M/D/YYYY by regex; `new Date('2026-03-31')` is midnight UTC and
 *  renders as the previous day in US timezones. Same fix as format.fmtDate. */
function fmtDate(v?: string | null): string {
  if (!v) return ''
  const m = String(v).match(/^(\d{4})-(\d{2})-(\d{2})/)
  return m ? `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}` : String(v)
}

const asOf = computed(() => fmtDate(resolution.value?.quarter_end))

/** Narrative text, read-only — the print document never writes. */
const narratives = computed<any[]>(() => summary.value?.narratives || [])
function narrativeAt(i: number): string {
  return narratives.value[i]?.text || ''
}

// The document title is blanked while printing so the browser's own header
// cannot put "Waterfall XIRR" across the top of every page. Chrome is also
// launched with --no-pdf-header-footer in the harness, and @page margin is 0
// with the sheet's own padding standing in — three belts for the same braces.
//
// NO RENDERED TIMESTAMP. The One Pager prints one in its corner, and this view
// copied that; the reference TIAA document has none, and it was the visible
// "8/25/2026, 3:31 PM" on every page. It was never the browser's header — it
// was ours.
function doPrint() {
  const orig = document.title
  document.title = ' '
  nextTick(() => {
    window.print()
    document.title = orig
  })
}

onMounted(async () => {
  if (!investor || !quarter) {
    loadError.value = 'investor and quarter are required'
    loading.value = false
    return
  }
  try {
    const res = await api.get(`${BASE}/bundle`, {
      params: { investor, quarter },
    })
    bundle.value = res.data
  } catch (e: any) {
    loadError.value = e?.response?.data?.error || 'Failed to load snapshot'
  } finally {
    loading.value = false
  }
  // Charts need a frame to size themselves before the print dialog captures
  // them; without the wait an auto-print can fire on an empty canvas.
  if (autoPrint && !loadError.value) {
    await nextTick()
    setTimeout(doPrint, 600)
  }
})
</script>

<template>
  <div class="print-doc">
    <div class="toolbar no-print">
      <button class="btn" :disabled="loading || !!loadError" @click="doPrint">
        Print
      </button>
      <span class="hint">
        4-page document · {{ investorName }} · {{ quarter }}
        <template v-if="bundle?.source === 'frozen'"> · approved snapshot</template>
      </span>
    </div>

    <p v-if="loading" class="placeholder">Building the document…</p>
    <p v-else-if="loadError" class="placeholder err">{{ loadError }}</p>

    <template v-else-if="bundle">
      <!-- ══ PAGE 1 — summary: narratives + the two charts ══ -->
      <section class="print-page">
        <h1 class="pdf-title">PORTFOLIO SNAPSHOT</h1>
        <div class="pdf-client">{{ investorName }}</div>
        <div class="pdf-sub">
          Current Portfolio Update (Balances as of {{ asOf }}, $ millions)
        </div>

        <p v-if="errors.summary" class="err">Summary failed: {{ errors.summary }}</p>
        <template v-else-if="summary">
          <h2 class="sect">Portfolio Exposure &amp; Performance</h2>

          <p class="narr">{{ narrativeAt(0) }}</p>
          <div class="chartwrap">
            <div class="chart-title">Asset Allocation: Funded vs. Total Commitment</div>
            <AllocationStackedBar
              :alloc="summary.asset_allocation" renderer="svg" height="300px" />
          </div>

          <p class="narr">{{ narrativeAt(1) }}</p>
          <div class="chartwrap">
            <div class="chart-title">Deal Type Allocation</div>
            <DealTypePie
              :alloc="summary.deal_type_allocation" renderer="svg" height="280px" />
          </div>
        </template>
      </section>

      <!-- ══ PAGE 2 — Financial ══ -->
      <section class="print-page">
        <h1 class="pdf-title repeat">PORTFOLIO SNAPSHOT</h1>
        <div class="pdf-client">{{ investorName }}</div>
        <div class="pdf-sub">
          Current Portfolio Update (Balances as of {{ asOf }}, $ millions)
        </div>
        <p v-if="errors.financial" class="err">
          Financial failed: {{ errors.financial }}
        </p>
        <SnapshotFinancial
          v-else-if="subtabs.financial"
          :data="subtabs.financial" :editable="false" />
      </section>

      <!-- ══ PAGE 3 — Operating ══ -->
      <section class="print-page">
        <div class="pdf-client">{{ investorName }}</div>
        <div class="pdf-sub">
          Current Portfolio Update (Balances as of {{ asOf }}, $ millions)
        </div>
        <p v-if="errors.operating" class="err">
          Operating failed: {{ errors.operating }}
        </p>
        <SnapshotOperating
          v-else-if="subtabs.operating"
          :data="subtabs.operating" :editable="false" />
      </section>

      <!-- ══ PAGE 4 — Loan ══ -->
      <section class="print-page last">
        <div class="pdf-client">{{ investorName }}</div>
        <div class="pdf-sub">
          Current Portfolio Update (Balances as of {{ asOf }}, $ millions)
        </div>
        <p v-if="errors.loan" class="err">Loan failed: {{ errors.loan }}</p>
        <SnapshotLoan
          v-else-if="subtabs.loan"
          :data="subtabs.loan" :editable="false" />
      </section>
    </template>
  </div>
</template>

<style scoped>
.print-doc { background: #f4f5f7; padding: 12px 0 40px 0; }

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  max-width: 11in;
  margin: 0 auto 12px auto;
}
.btn {
  padding: 7px 16px;
  border: 1px solid var(--color-accent);
  background: var(--color-accent);
  color: #fff;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
}
.btn:disabled { opacity: 0.5; cursor: default; }
.hint { font-size: 12px; color: var(--color-text-secondary); }

/* On screen each page is a sheet, so the pagination is visible before printing
   rather than being a surprise at the dialog. */
.print-page {
  /* LANDSCAPE. The tables, not the charts, decide this: every one of pages
     2-4 laid out WIDER than a portrait column (7.68in of content in 7.50in)
     while leaving 2-3in of blank paper at the foot, and the only way they fit
     at all was 4.5pt type — below legal fine print. Landscape trades 2.5in of
     height for 2.5in of width, which is the exchange this document wants; the
     vertical slack absorbs most of the loss. Page 1 is landscape too rather
     than mixing orientation mid-report, and gains by it: it was using 5.7in of
     a 7.5in column with nearly 5in of dead space beneath two stacked charts. */
  width: 11in;
  min-height: 8.5in;
  box-sizing: border-box;
  /* Identical to the print padding below, so the sheet on screen is the sheet
     that comes out of the printer. */
  padding: 0.36in 0.5in 0.30in 0.5in;
  margin: 0 auto 16px auto;
  background: #fff;
  border: 1px solid var(--color-border);
  position: relative;
}

.pdf-title {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-align: center;
  margin: 0 0 8px 0;
  padding-bottom: 7px;
  border-bottom: 3px solid var(--color-text);
}
.pdf-client { font-size: 12px; font-weight: 700; }
.pdf-sub {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
}

.sect {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin: 0 0 6px 0;
}
.narr {
  font-size: 11.5px;
  line-height: 1.5;
  margin: 0 0 6px 0;
  /* No min-height: an empty narrative reserved a blank line on every page. */
  white-space: pre-wrap;
}
/* Centred by filling the column with balanced margins, which is how the
   reference page centres its charts — not a narrow figure floating in gutters. */
.chartwrap {
  margin: 0 auto 10px auto;
  max-width: 6.6in;
}
.chart-title {
  font-size: 11px;
  font-weight: 700;
  text-align: center;
  margin-bottom: 2px;
}

.placeholder {
  text-align: center;
  font-style: italic;
  color: var(--color-text-secondary);
  padding: 60px 0;
}
.err { color: #a12622; font-size: 12px; }

/* ── table fitting: ON SCREEN AS WELL AS ON PAPER ─────────────────────────
   These used to live inside @media print, which meant this page — whose only
   purpose is to show what will print — did not itself show what would print.
   On screen the tables kept overflow-x:auto, their full interactive font and
   the sticky column's min-width, so they overflowed the sheet with a scrollbar
   and clipped columns. Reviewers reasonably read that as broken formatting.
   A preview that does not preview is worse than no preview, so they apply
   always and the two renderings cannot diverge again. */

/* The interactive scroll container clips the table at its own width, which on
   paper means silently losing right-hand columns. */
:deep(.scroll) {
  overflow: visible !important;
  border: 1px solid #ccc;
}
:deep(.sticky-l) {
  position: static !important;
  min-width: 0 !important;
}
/* Full width between the balanced page margins — a table that does not fill
   the measure reads as drifting to one side. */
:deep(.scroll), :deep(table.grid) { width: 100% !important; }
/* SIZED TO THE LIVE ROW COUNT, NOT THE ONE ON SCREEN HERE.
   It was 7.5px, printing at 4.5pt — below legal fine print, and the least
   professional thing about the document. 8px prints at ~6pt.

   Why not larger, when landscape looks like it affords it: rotating a sheet
   does not add area, it reshapes it. Letter is 93.5 sq in either way. Landscape
   buys 2.5in of measure — which is what stops the tables overflowing sideways
   and lets the comment column breathe — and pays for it with 2.5in of height,
   which is the dimension a 36-deal table is already short of.

   The budget, measured from a real printed sheet rather than estimated: 7.74in
   of usable height, ~1.9in of it fixed (titles, column heads, the footnote
   block), leaving 5.8in for ~44 rows of deals, groups and subtotals. That caps
   the row pitch at 0.132in, which is 8px. At 9.5px the pitch is 0.157in and the
   Financial table runs 0.5in past the foot of the page ON LIVE — invisible in
   this local snapshot, which carries 30 deals where live carries 36. */
:deep(table.grid) { font-size: 8px !important; table-layout: auto; }
/* LINE HEIGHT, not font size, is what sets the row pitch at these sizes — at
   8px the default leading still held the pitch at 0.146in, barely below the
   0.157in it was at 9.5px. Pinning it to 1.05 drops the pitch to ~0.12in and
   BUYS BACK the type size: 9px here prints larger than 8px did with default
   leading, on a shorter table. Vertical padding is minimal for the same
   reason; horizontal padding is generous because width is what landscape
   bought. */
:deep(table.grid th), :deep(table.grid td) {
  padding: 0.5px 5px !important;
  line-height: 1.05 !important;
}
/* Uncapped from 1.5in. The cap existed only because a portrait measure could
   not afford the width; on a landscape sheet a wider comment column also BUYS
   height back, since most comments then set on one line instead of two. */
:deep(table.grid .cmt) { max-width: 2.9in; }

/* ── the printed document ─────────────────────────────────────────────── */
@media print {
  /* The page box (letter portrait, zero margin — which is what suppresses the
     browser's own header/footer) is set ONCE globally in App.vue. The
     .print-page padding below stands in for the margin. */

  .print-doc { background: #fff; padding: 0; }
  .no-print { display: none !important; }

  .print-page {
    width: auto;
    min-height: 0;
    margin: 0;
    /* Balanced side padding IS the centring: the content block fills the page
       between equal margins rather than sitting in one. Slightly tighter top
       and bottom than the sides, matching the reference page's density — it is
       a fairly full page, not an airy one. */
    padding: 0.36in 0.5in 0.30in 0.5in;
    border: none;
    /* Opt this document into the LANDSCAPE page box. Named page boxes are the
       only way orientation can differ per document: `@page` itself cannot be
       scoped, so a view that redefines the default breaks every other view
       that prints (v421). Both boxes are declared in App.vue; a named one
       affects nothing until an element asks for it, as here. */
    page: landscape-sheet;
    /* Each subtab starts a new sheet; the last must not emit a trailing blank. */
    page-break-after: always;
    break-after: page;
  }
  .print-page.last { page-break-after: auto; break-after: auto; }

  /* Charts: centred, and wide enough to hold a landscape measure without
     stretching flat. */
  .chartwrap { max-width: 7.6in; margin: 0 auto 10px auto; }
  .chart-title { margin-bottom: 0; }
  .sect { margin-bottom: 4px; }
  .narr { margin-bottom: 4px; }
  .pdf-sub { margin-bottom: 3px; }
  .pdf-title { font-size: 20px; padding-bottom: 6px; margin-bottom: 6px; }
  /* The cover title repeated on the Financial page costs it 0.53in that
     Operating and Loan do not spend — measured: its table starts at 1.30in
     where theirs start at 0.77in — and Financial is also the only page
     carrying the footnote block. That combination is what put a three-line
     note onto a sheet of its own. The client line and "Balances as of" stay,
     so the page still identifies itself; only the second printing of the
     document's cover title goes. */
  .pdf-title.repeat { display: none; }

  /* ---- placeholders are screen chrome, not document content ----
     An un-entered manual figure reads "pending entry" on screen so whoever is
     filling the report can see what is missing. On paper the reference document
     shows a clean cell, and 67 italic "pending entry" strings down two columns
     is worse than a blank. visibility, not display: the cell keeps its width so
     the column does not collapse. */
  :deep(.numinput.pending) { visibility: hidden !important; }
  :deep(.manual.small) { visibility: hidden !important; }

  /* ---- fund-group separator rules (Financial, Operating, Loan) ----
     The 26Q1 report breaks the table up with horizontal lines so the eye can
     find where one fund ends and its subtotal begins, instead of reading 30
     unbroken rows. Declared HERE, once, with :deep() rather than three times
     in three scoped components, so the three tables cannot drift apart — the
     same reason group_total_label lives in the service layer.

     Placement: a rule above every fund subtotal, a heavier one above the
     portfolio total, and the inter-group spacer keeps the groups apart. NOT on
     the Summary page, which is narrative and charts and has no such table —
     the selectors are anchored on `table.grid`, which only the three data
     subtabs render.

     The reference PDF is not in this repository (only its values are, in
     scripts/snapshot_pdf_variance_pdfdata.py), so the weights match the
     convention the Operating and Loan subtabs already use on screen rather
     than a measurement of the original. One place to change if it needs to be
     heavier or lighter. */
  :deep(table.grid tr.subtotal td) {
    border-top: 1px solid var(--color-text-secondary) !important;
  }
  :deep(table.grid tfoot tr:first-child td) {
    border-top: 2px solid var(--color-text) !important;
  }
  /* The blank row between one group and the next. It carries the separation on
     paper, so it must not be squeezed out by the fit rules below. */
  :deep(table.grid tr.spacer td) {
    height: 6px !important;
    padding: 0 !important;
    border: none !important;
  }

  /* On-screen chrome that means nothing on paper. */
  :deep(.legend) { display: none !important; }
  :deep(.diag), :deep(.fnadd), :deep(.hint) { display: none !important; }
  /* Page 1 is narrative + charts, as published — the per-chart data tables
     belong to the interactive view. */
  :deep(.showdata) { display: none !important; }
  /* The Summary card chrome would draw a box inside each printed page. */
  :deep(.summary .card) { border: none !important; padding: 0 !important; }

  /* ---- form controls print as TEXT, in the table's own font ----
     `font: inherit` is the load-bearing line. A textarea or input does NOT
     inherit font-family or font-size from its container — the browser gives it
     a UA default of about 13.3px in a system font, while the printed table is
     7.5px. That is why the Operating and Loan comments and the ITD / Net ROE
     cells came out visibly bigger and in a different typeface from every other
     cell on the page. Setting the border, padding and background alone (which
     is all this used to do) leaves the type wrong.

     Shorthand `font` rather than font-size + font-family, so nothing else the
     UA sets on a form control — weight, style, line-height, variant — survives
     either. */
  /* `.cmt-text` is the READ-ONLY rendering of a comment — the print view sets
     `editable: false`, so the Operating and Loan comments come through as text
     rather than as a textarea, and the rule below on form controls never
     reaches them. It carries a hardcoded `font-size: 12px`, which is right on
     screen (where the table is 12px) and 60% too large on paper (where the
     table is 7.5px). Measured: the comment printed at 7.4pt against 4.6pt for
     every other cell. */
  :deep(.cmt-text) {
    font: inherit !important;
    line-height: 1.25 !important;
  }

  :deep(textarea), :deep(input) {
    font: inherit !important;
    letter-spacing: inherit !important;
    color: var(--color-text) !important;
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
  }
  :deep(textarea) {
    resize: none;
    overflow: visible;
    height: auto !important;
  }
  /* A right-aligned numeric input has to keep its alignment; the reset above
     would otherwise leave it reading left against the column beside it. */
  :deep(.numinput) { text-align: right !important; width: 100% !important; }

  /* ---- column separators, per the 26Q1 reference ----
     Light VERTICAL rules structuring the table, plus the horizontal rules
     above. Hairline weight and a pale grey deliberately: the reference is not
     a full boxed grid, it is a set of faint guides that stop the eye sliding
     between columns on a 30-row page.

     Placement follows the reference: a rule at every numeric column boundary,
     and a slightly stronger one at the ZONE boundaries — where the deal-level
     cap stack gives way to the "TIAA Investment" block, and where that gives
     way to the manual ITD / Net ROE columns. The zone classes already exist on
     Financial; Operating and Loan have no zones and simply take the light
     rule throughout. */
  :deep(table.grid th), :deep(table.grid td) {
    border-right: 0.5px solid #e3e6ea !important;
  }
  :deep(table.grid th:last-child), :deep(table.grid td:last-child) {
    border-right: none !important;
  }
  /* The first zone-b / manual cell in each row opens its block. */
  :deep(table.grid .zone-b + .zone-b) { border-left: none !important; }
  :deep(table.grid td.zone-b:first-of-type),
  :deep(table.grid th.zone-b),
  :deep(table.grid td.manual),
  :deep(table.grid th.manual) {
    border-left: 0.5px solid #c9ced6 !important;
  }
  :deep(table.grid .manual + .manual) { border-left: none !important; }

  /* Numeric columns: enough room that a figure never crowds its neighbour,
     not so much that the table stops fitting. The deal-name column is the one
     that flexes, since it is the only text of variable length. */
  :deep(table.grid td.num), :deep(table.grid th.r) {
    min-width: 0.42in;
    white-space: nowrap;
  }
  :deep(table.grid .sticky-l) { min-width: 1.35in; }

  /* ---- table pagination ----
     `display: table-header-group` is what makes a thead REPEAT at the top of
     each sheet a table spills onto. Without it the header prints once and the
     continuation columns are unlabelled.

     The previous `break-inside: avoid` on tbody is REMOVED, and it was the
     cause of the detached headers: when a fund block did not fit, the browser
     pushed the whole tbody to the next sheet but had already laid the repeated
     thead down at the bottom of the current one, leaving a header row with no
     data under it. Avoiding breaks inside a ROW is safe and enough — a row is
     one line, a fund block is twenty. */
  :deep(thead) { display: table-header-group; }
  :deep(tfoot) { display: table-footer-group; }
  :deep(tr) { break-inside: avoid; page-break-inside: avoid; }

  /* ---- internal markers are not part of the document ----
     "DEV", "CHILD", "NEW", the "!" flag dots and the "*" exception star are
     working annotations for an analyst reading the screen. The reference
     document shows none of them: a development deal simply reads n/a. Hidden
     rather than removed, because on screen they are how a reader knows WHY a
     cell is n/a.

     `.tag` covers every variant — the plain "Dev", `.tag.new` on Operating and
     `.tag.alt` ("child") on Loan — because they all carry the base class. No
     "?" marker exists on any subtab; if one is ever added it must be added to
     this list, and `scripts/snapshot_print_markers_check.py` fails if a new
     marker class appears that is not suppressed here. */
  :deep(.tag), :deep(.warn-dot), :deep(.star) { display: none !important; }

  /* NOT hidden: `.fnmark`, the footnote marker on a property name. It sits in
     the same cell as the star and the flag dots and looks like more of the
     same, but it is part of the published document — it is what ties City West
     and East Manchester to the ROE-exclusion footnote. Stated here so a later
     tidy-up does not sweep it in with the annotations above.

     NOT hidden either: `.sold`, the "(Sold)" after a property name. Same
     reasoning — a deal reported after its disposition must say so on paper as
     well as on screen, which is why it is plain italic text rather than a
     `.tag` pill (those are suppressed above). Do not add it to the hide list. */

  /* ---- small auto-written notes beside the figures ----
     Text the app composes to explain a row, as opposed to data or an authored
     footnote: the deal counts appended to the Financial excluding-development
     label, and the Loan tab's "summary ratios already exclude..." aside. Both
     read as clutter next to the numbers on paper and neither is on the
     reference document. The labels and every figure stay. */
  :deep(.exdev-n) { display: none !important; }
  :deep(tfoot .note) { display: none !important; }
}
</style>
