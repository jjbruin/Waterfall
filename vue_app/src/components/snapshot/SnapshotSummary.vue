<script setup lang="ts">
/**
 * Summary subtab — PDF page 1: narrative, chart, narrative, chart.
 *
 * LAYOUT follows the published page, which interleaves rather than stacking:
 * the first paragraph sets up the asset-allocation chart and the second sets up
 * the deal-type pie. The two narratives used to sit together in a "Commentary"
 * card below both tables; each now sits directly above the chart it describes.
 *
 * The CSS-width bar lists they replace are gone from the headline view but the
 * numbers are not — each chart keeps a "Show data" table underneath carrying
 * every figure the bars showed plus the deal counts, which is also the relief
 * the palette's contrast WARN obliges (see ./palette.ts).
 *
 * Presentational: props in, save events out. No computation of allocation
 * figures happens here — the charts read the backend's existing
 * asset_allocation / deal_type_allocation payloads untouched.
 */
import { computed, ref, watch } from 'vue'
import { fmtM, fmtM$, fmtPct } from './format'
import AllocationStackedBar from './AllocationStackedBar.vue'
import DealTypePie from './DealTypePie.vue'

const props = defineProps<{ data: any; editable: boolean }>()
const emit = defineEmits<{
  saveComment: [p: { scope: string; field: string; scope_key?: string; text: string }]
}>()

import { hueFor } from './palette'

const asset = computed(() => props.data?.asset_allocation || null)
const dealType = computed(() => props.data?.deal_type_allocation || null)
const flags = computed<string[]>(() => props.data?.flags || [])
const flagged = computed<any[]>(() => props.data?.ownership_flagged || [])

/** Local narrative text, seeded from props and re-seeded when the page changes. */
const text = ref<Record<string, string>>({})

watch(() => props.data, (d) => {
  const next: Record<string, string> = {}
  for (const n of (d?.narratives || [])) next[n.field] = n.text || ''
  text.value = next
}, { immediate: true, deep: false })

const narratives = computed<any[]>(() => props.data?.narratives || [])

function commit(field: string) {
  emit('saveComment', { scope: 'report', field, text: text.value[field] ?? '' })
}

function label(field: string) {
  const n = narratives.value.findIndex((x) => x.field === field)
  return n === 0 ? 'Narrative — first paragraph' : 'Narrative — second paragraph'
}

/** The narrative that belongs above a given chart, by position. */
function narrativeAt(i: number) {
  return narratives.value[i] || null
}
</script>

<template>
  <div v-if="!data" class="placeholder">No summary data.</div>
  <div v-else class="summary">
    <p v-for="(f, i) in flags" :key="i" class="flagline">{{ f }}</p>

    <!-- ── Asset Allocation: narrative, then chart (PDF page 1 order) ── -->
    <section class="card">
      <div v-if="narrativeAt(0)" class="narrative lead">
        <label>{{ label(narrativeAt(0).field) }}</label>
        <textarea
          v-model="text[narrativeAt(0).field]"
          rows="4"
          spellcheck="true"
          lang="en"
          :readonly="!editable"
          :placeholder="editable ? 'Type commentary…' : ''"
          @change="commit(narrativeAt(0).field)"
        ></textarea>
        <span class="hint">
          <template v-if="!editable">Locked — this snapshot is approved.</template>
          <template v-else-if="narrativeAt(0).is_blank">Blank. Nothing is auto-generated.</template>
          <template v-else>{{ narrativeAt(0).char_count }} characters saved.</template>
        </span>
      </div>

      <header>
        <h3>Asset Allocation: Funded vs. Total Commitment</h3>
        <span class="sub">{{ data.basis_note }}</span>
      </header>
      <AllocationStackedBar :alloc="asset" />

      <details class="showdata">
        <summary>Show data</summary>
        <table class="alloc">
          <thead>
            <tr>
              <th>Asset Type</th>
              <th class="r">Funded ($M)</th>
              <th class="c">%</th>
              <th class="r">Committed ($M)</th>
              <th class="c">%</th>
              <th class="c">Deals</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(b, idx) in (asset?.buckets || [])" :key="b.label">
              <td><span class="swatch" :style="{ background: hueFor(idx) }"></span>{{ b.label }}</td>
              <td class="r num">{{ idx === 0 ? fmtM$(b.funded) : fmtM(b.funded) }}</td>
              <td class="c num">{{ fmtPct(b.funded_pct) }}</td>
              <td class="r num">{{ idx === 0 ? fmtM$(b.committed) : fmtM(b.committed) }}</td>
              <td class="c num">{{ fmtPct(b.committed_pct) }}</td>
              <td class="c num">{{ b.deal_count }}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td>Total</td>
              <td class="r num">{{ fmtM$(asset?.total_funded) }}</td>
              <td class="c num">100.0%</td>
              <td class="r num">{{ fmtM$(asset?.total_committed) }}</td>
              <td class="c num">100.0%</td>
              <td class="c num">{{ asset?.deal_count }}</td>
            </tr>
          </tfoot>
        </table>
      </details>
    </section>

    <!-- ── Deal Type: narrative, then pie ── -->
    <section class="card">
      <div v-if="narrativeAt(1)" class="narrative lead">
        <label>{{ label(narrativeAt(1).field) }}</label>
        <textarea
          v-model="text[narrativeAt(1).field]"
          rows="4"
          spellcheck="true"
          lang="en"
          :readonly="!editable"
          :placeholder="editable ? 'Type commentary…' : ''"
          @change="commit(narrativeAt(1).field)"
        ></textarea>
        <span class="hint">
          <template v-if="!editable">Locked — this snapshot is approved.</template>
          <template v-else-if="narrativeAt(1).is_blank">Blank. Nothing is auto-generated.</template>
          <template v-else>{{ narrativeAt(1).char_count }} characters saved.</template>
        </span>
      </div>

      <header>
        <h3>Deal Type Allocation</h3>
        <span class="sub">funded dollars by investment strategy</span>
      </header>
      <DealTypePie :alloc="dealType" />

      <details class="showdata">
        <summary>Show data</summary>
        <table class="alloc">
          <thead>
            <tr>
              <th>Deal Type</th>
              <th class="r">Funded ($M)</th>
              <th class="c">%</th>
              <th class="c">Deals</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(b, idx) in (dealType?.buckets || [])" :key="b.label">
              <td><span class="swatch" :style="{ background: hueFor(idx) }"></span>{{ b.label }}</td>
              <td class="r num">{{ idx === 0 ? fmtM$(b.funded) : fmtM(b.funded) }}</td>
              <td class="c num">{{ fmtPct(b.funded_pct) }}</td>
              <td class="c num">{{ b.deal_count }}</td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td>Total</td>
              <td class="r num">{{ fmtM$(dealType?.total_funded) }}</td>
              <td class="c num">100.0%</td>
              <td class="c num">{{ dealType?.deal_count }}</td>
            </tr>
          </tfoot>
        </table>
      </details>
    </section>

    <!-- Ownership-flagged deals: excluded from the scaled allocation -->
    <section v-if="flagged.length" class="card warn">
      <header><h3>Excluded from the allocation</h3></header>
      <p v-for="f in flagged" :key="f.vcode" class="flagline">
        {{ f.vcode }} {{ f.name }} — {{ f.reason }}
        (deal-level funded {{ fmtM(f.funded_deal_level) }}M withheld rather than shown unscaled)
      </p>
    </section>

    <!--
      Any narrative beyond the two the page lays out. Nothing produces a third
      today, but a silently dropped comment would be worse than an extra box.
    -->
    <section v-if="narratives.length > 2" class="card">
      <header><h3>Further commentary</h3></header>
      <div class="narratives">
        <div v-for="n in narratives.slice(2)" :key="n.field" class="narrative">
          <label>{{ n.field }}</label>
          <textarea
            v-model="text[n.field]"
            rows="5"
            spellcheck="true"
            lang="en"
            :readonly="!editable"
            :placeholder="editable ? 'Type commentary…' : ''"
            @change="commit(n.field)"
          ></textarea>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.summary { display: flex; flex-direction: column; gap: 16px; }

/* The lead narrative sits above its chart, as on the published page. */
.narrative.lead { margin-bottom: 14px; }

/* Every figure the old bar list carried, one click away — and the relief the
   palette's contrast WARN requires. */
.showdata { margin-top: 10px; }
.showdata summary {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  cursor: pointer;
  padding: 4px 0;
}
.showdata[open] summary { margin-bottom: 6px; }

/* Legend swatch in the table, so the table and the chart share one identity
   mapping rather than the reader inferring it. */
.swatch {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: baseline;
}

.card {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 14px 16px;
  background: var(--color-surface);
}
.card.warn { background: #fff8e1; border-color: #ffe082; }

.card header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.card h3 { font-size: 14px; margin: 0; }
.sub { font-size: 11px; color: var(--color-text-secondary); font-style: italic; }

table.alloc { width: 100%; border-collapse: collapse; font-size: 12px; }
table.alloc th {
  text-align: left;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
  padding: 4px 6px;
  font-weight: 700;
  white-space: nowrap;
}
table.alloc td { padding: 6px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }
table.alloc th.r, table.alloc td.r { text-align: right; min-width: 80px; }
table.alloc th.c, table.alloc td.c { text-align: center; min-width: 50px; }
table.alloc tfoot td {
  border-top: 2px solid var(--color-border);
  border-bottom: none;
  font-weight: 700;
}
.r { text-align: right; }
.c { text-align: center; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }


.flagline { font-size: 12px; color: #856404; margin: 2px 0; }

.narratives { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.narrative label {
  display: block;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}
.narrative textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 8px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
  line-height: 1.45;
  resize: vertical;
}
.narrative textarea[readonly] { background: #fafafa; color: var(--color-text-secondary); }
.hint { font-size: 11px; color: var(--color-text-secondary); font-style: italic; }

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}

@media (max-width: 1100px) {
  .narratives { grid-template-columns: 1fr; }
}

@media print {
  .flagline { display: none; }
  .card { border: 1px solid #ccc; break-inside: avoid; }
  .card.warn { display: none; }
  /* Canvas prints as-is; the expander is opened so the figures land on paper
     even though its disclosure arrow means nothing there. */
  .showdata { display: block; }
  .showdata summary { display: none; }
  .showdata > table { display: table !important; }
  .narrative textarea {
    border: none;
    padding: 0;
    resize: none;
    overflow: visible;
    height: auto !important;
  }
  .hint { display: none; }
}
</style>
