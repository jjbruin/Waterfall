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

// The One Pager's print pattern: a timestamp we render ourselves, and the
// document title blanked so the browser's own header does not print
// "Waterfall XIRR" across the top of every page.
const printTimestamp = ref('')

function stamp() {
  const now = new Date()
  printTimestamp.value = `${now.getMonth() + 1}/${now.getDate()}/`
    + `${now.getFullYear()}, `
    + now.toLocaleTimeString('en-US',
        { hour: 'numeric', minute: '2-digit', hour12: true })
}

function doPrint() {
  stamp()
  const orig = document.title
  document.title = ' '
  nextTick(() => {
    window.print()
    document.title = orig
  })
}

onMounted(async () => {
  stamp()
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
        <div class="stamp">{{ printTimestamp }}</div>
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
        <div class="stamp">{{ printTimestamp }}</div>
        <h1 class="pdf-title">PORTFOLIO SNAPSHOT</h1>
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
        <div class="stamp">{{ printTimestamp }}</div>
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
        <div class="stamp">{{ printTimestamp }}</div>
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
  max-width: 8.5in;
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
  width: 8.5in;
  min-height: 11in;
  box-sizing: border-box;
  padding: 0.5in;
  margin: 0 auto 16px auto;
  background: #fff;
  border: 1px solid var(--color-border);
  position: relative;
}

.stamp {
  position: absolute;
  top: 0.18in;
  left: 0.5in;
  font-size: 8px;
  color: var(--color-text-secondary);
}

.pdf-title {
  font-family: Georgia, 'Times New Roman', serif;
  font-size: 24px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-align: center;
  margin: 6px 0 10px 0;
  padding-bottom: 9px;
  border-bottom: 3px solid var(--color-text);
}
.pdf-client { font-size: 12px; font-weight: 700; }
.pdf-sub {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

.sect {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  margin: 0 0 8px 0;
}
.narr {
  font-size: 11.5px;
  line-height: 1.5;
  margin: 0 0 8px 0;
  min-height: 1em;
  white-space: pre-wrap;
}
.chartwrap { margin-bottom: 14px; }
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

/* ── the printed document ─────────────────────────────────────────────── */
@media print {
  /* Suppress the browser's own header/footer (URL, title, page number) the way
     the One Pager does: zero @page margin, and the sheet's own padding stands
     in for it. */
  @page { size: letter portrait; margin: 0; }

  .print-doc { background: #fff; padding: 0; }
  .no-print { display: none !important; }

  .print-page {
    width: auto;
    min-height: 0;
    margin: 0;
    padding: 0.5in;
    border: none;
    /* Each subtab starts a new sheet; the last must not emit a trailing blank. */
    page-break-after: always;
    break-after: page;
  }
  .print-page.last { page-break-after: auto; break-after: auto; }

  /* Wide tables: the interactive scroll container clips them at the viewport,
     which on paper means silently losing right-hand columns. Let them lay out,
     and drop the sticky-column min-width that was reserving screen space. */
  :deep(.scroll) {
    overflow: visible !important;
    border: 1px solid #ccc;
  }
  :deep(.sticky-l) {
    position: static !important;
    min-width: 0 !important;
  }
  :deep(table.grid) { font-size: 7.5px !important; width: 100% !important; }
  :deep(table.grid th), :deep(table.grid td) { padding: 2px 3px !important; }

  /* On-screen chrome that means nothing on paper. */
  :deep(.legend) { display: none !important; }
  :deep(.diag), :deep(.fnadd), :deep(.hint) { display: none !important; }
  /* Page 1 is narrative + charts, as published — the per-chart data tables
     belong to the interactive view. */
  :deep(.showdata) { display: none !important; }
  /* The Summary card chrome would draw a box inside each printed page. */
  :deep(.summary .card) { border: none !important; padding: 0 !important; }

  /* Textareas print as text, not as form controls. */
  :deep(textarea) {
    border: none !important;
    padding: 0 !important;
    resize: none;
    overflow: visible;
    height: auto !important;
    background: transparent !important;
    color: var(--color-text) !important;
  }
  :deep(.numinput) {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
  }

  /* Keep a table from splitting a fund block across sheets where it can. */
  :deep(tbody) { break-inside: avoid; }
}
</style>
