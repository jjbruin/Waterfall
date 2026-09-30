<script setup lang="ts">
/**
 * Investment Metrics — the Current and Sold portfolio summary, on screen.
 *
 * The printed document is a separate route (`/investment-metrics/print`), on
 * the same pattern as the One Pager and the Portfolio Snapshot: this page is
 * readable at a normal size, that one is the reference document at its own
 * 3.6pt. Both draw the same table component off the same payload, so a column
 * cannot appear on one and not the other.
 */
import { computed, onMounted, ref, watch } from 'vue'
import api from '@/api/client'
import InvestmentMetricsTable from '@/components/reports/InvestmentMetricsTable.vue'

const data = ref<any>(null)
const quarters = ref<string[]>([])
const asOf = ref('')
const loading = ref(true)
const loadError = ref('')
const showDiagnostics = ref(false)

const diag = computed(() => data.value?.diagnostics || {})
const diagCounts = computed(() => {
  const d = diag.value
  return [
    ['Twins merged', (d.twins_merged || []).length],
    ['Child properties excluded', (d.children_excluded || []).length],
    ['Orphan rows dropped', (d.orphans || []).length],
    ['Deals with no deal_terms row', (d.missing_deal_terms || []).length],
    ['Not in the reference order', (d.not_in_reference_order || []).length],
    ['Reference rows with no deal', (d.reference_rows_absent || []).length],
  ].filter(([, n]) => (n as number) > 0)
})

async function load() {
  loading.value = true
  loadError.value = ''
  try {
    const res = await api.get('/api/investment-metrics',
      { params: asOf.value ? { as_of: asOf.value } : {} })
    data.value = res.data
  } catch (e: any) {
    loadError.value = e?.response?.data?.error || e?.message || 'failed to load'
    data.value = null
  } finally {
    loading.value = false
  }
}

/**
 * THE SCREEN DOES NOT PIN A QUARTER. Five hard-coded `2026-Q2` spellings
 * across two views and two scripts is what v530 had to unpick; the server
 * answers which quarters exist and which one opens.
 */
async function loadQuarters() {
  try {
    const res = await api.get('/api/investment-metrics/quarters')
    quarters.value = res.data.quarters || []
    if (!asOf.value) asOf.value = res.data.default || ''
  } catch {
    /* the report still loads on the server's own default */
  }
}

function openPrint() {
  const q = asOf.value ? `?as_of=${encodeURIComponent(asOf.value)}` : ''
  window.open(`/investment-metrics/print${q}`, '_blank')
}

onMounted(async () => {
  await loadQuarters()
  await load()
})
watch(asOf, (v, old) => { if (old !== '' && v !== old) load() })
</script>

<template>
  <div class="im-page">
    <div class="controls-bar no-print">
      <h2 class="page-title">Investment Metrics</h2>
      <div class="spacer"></div>
      <label class="fld">
        As of
        <select v-model="asOf">
          <option v-for="q in quarters" :key="q" :value="q">{{ q }}</option>
        </select>
      </label>
      <button class="btn btn-sm" @click="showDiagnostics = !showDiagnostics">
        {{ showDiagnostics ? 'Hide' : 'Show' }} how it was built
      </button>
      <button class="btn btn-sm btn-primary" :disabled="!data" @click="openPrint">
        Print
      </button>
    </div>

    <div v-if="loading" class="msg">Loading…</div>
    <div v-else-if="loadError" class="msg err">
      This report could not be loaded: {{ loadError }}
    </div>

    <template v-else-if="data">
      <div v-if="showDiagnostics" class="diagnostics no-print">
        <p class="dhead">
          As of <strong>{{ data.as_of_display }}</strong>.
          CAD converted at {{ data.fx_rate }}
          <span class="muted">(the source workbook used
            {{ data.fx_rate_workbook }})</span>.
        </p>
        <ul>
          <li v-for="[label, n] in diagCounts" :key="label">{{ label }}: {{ n }}</li>
        </ul>
        <p v-if="(diag.orphans || []).length" class="muted">
          Dropped, because nothing keys them:
          {{ (diag.orphans || []).map((o: any) => `${o.vcode} ${o.name}`).join('; ') }}
        </p>
        <p v-if="(diag.missing_deal_terms || []).length" class="muted">
          No deal_terms row, so coupon / split / lookback print as a dash:
          {{ (diag.missing_deal_terms || []).map((o: any) => o.vcode).join(', ') }}
        </p>
      </div>

      <section class="tbl-block">
        <div class="tbl-head">
          <h3>{{ data.current.title }}</h3>
          <span class="units">{{ data.as_of_display }} · {{ data.units_note }}</span>
        </div>
        <div class="scroller">
          <InvestmentMetricsTable :table="data.current" />
        </div>
        <div class="notes">
          <div v-for="f in data.current.footnotes" :key="f.n">
            ({{ f.n }}) {{ f.text }}
          </div>
          <p class="disclaimer">{{ data.disclaimer }}</p>
        </div>
      </section>

      <section class="tbl-block">
        <div class="tbl-head">
          <h3>{{ data.sold.title }}</h3>
          <span class="units">{{ data.units_note }}</span>
        </div>
        <div class="scroller">
          <InvestmentMetricsTable
            :table="data.sold"
            :total-markers="data.sold.total_markers"
            :grand-total="data.grand_total"
          />
        </div>
        <div class="notes">
          <div v-for="f in data.sold.footnotes" :key="f.n">
            ({{ f.n }}) {{ f.text }}
          </div>
          <p class="disclaimer">{{ data.disclaimer }}</p>
        </div>
      </section>
    </template>
  </div>
</template>

<style scoped>
.im-page { padding: 12px 16px 40px 16px; }
.controls-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 10px;
  border-bottom: 1px solid #e5e7eb;
  margin-bottom: 12px;
}
.page-title { margin: 0; font-size: 18px; }
.spacer { flex: 1; }
.fld { font-size: 13px; display: flex; align-items: center; gap: 6px; }
.btn {
  border: 1px solid #cbd5e1;
  background: #fff;
  border-radius: 4px;
  padding: 4px 10px;
  font-size: 13px;
  cursor: pointer;
}
.btn-primary { background: #1f5f3f; border-color: #1f5f3f; color: #fff; }
.btn:disabled { opacity: .5; cursor: default; }
.msg { padding: 20px 4px; font-size: 14px; }
.msg.err { color: #b91c1c; }

.diagnostics {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 4px;
  padding: 10px 14px;
  margin-bottom: 14px;
  font-size: 12.5px;
}
.diagnostics ul { margin: 6px 0; padding-left: 18px; }
.dhead { margin: 0; }
.muted { color: #64748b; }

.tbl-block { margin-bottom: 26px; }
.tbl-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 4px;
}
.tbl-head h3 { margin: 0; font-size: 14px; }
.units { font-size: 12px; color: #475569; }

/* Wide content scrolls INSIDE its own box; the page itself must never scroll
   sideways. Twenty-one columns will not fit a laptop screen at a readable
   size, and shrinking them until they do is how the printed document ends up
   being the only legible copy. */
.scroller { overflow-x: auto; border: 1px solid #d7dde5; background: #fff; }

:deep(.im-grid) {
  font-family: Garamond, Georgia, "Times New Roman", serif;
  font-size: 11px;
  width: max-content;
  min-width: 100%;
}
:deep(.im-grid th),
:deep(.im-grid td) {
  padding: 2px 5px;
  height: 17px;
}
:deep(.im-grid thead th) {
  font-weight: 600;
  font-size: 10.5px;
  background: #fff;
  position: sticky;
  top: 0;
}
:deep(.im-grid .h3 th.ruled),
:deep(.im-grid .h1 th.grouped),
:deep(.im-grid .h2 th.grouped) { border-bottom: 1px solid #111; }
:deep(.im-grid .vrule) { border-left: 1px solid #111; }
:deep(.im-grid tbody tr.band td:not(.spacer)) { background: #f1f2f4; }
:deep(.im-grid tbody tr.total td) { font-weight: 700; border-top: 1px solid #111; }
:deep(.im-grid tbody tr.gap td) { height: 6px; }
/* Column widths are measured in points for the printed sheet; on screen they
   would clip the names, so the table sizes to its content instead. */
:deep(.im-grid colgroup col) { width: auto !important; }

.notes { font-size: 11px; font-style: italic; color: #334155; margin-top: 6px; }
.disclaimer { margin-top: 8px; max-width: 62em; }
</style>
