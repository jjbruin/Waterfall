<script setup lang="ts">
/**
 * Summary subtab — allocation rollups + two narrative boxes.
 *
 * Presentational: props in, save events out. The allocation bars are plain
 * CSS-width divs rather than a chart library — four to six buckets with a
 * dollar and a percentage each read better as a labelled bar list, and it adds
 * no dependency to a build we cannot preview.
 */
import { computed, ref, watch } from 'vue'
import { fmtM, fmtM$, fmtPct } from './format'

const props = defineProps<{ data: any; editable: boolean }>()
const emit = defineEmits<{
  saveComment: [p: { scope: string; field: string; scope_key?: string; text: string }]
}>()

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

/** Bar width as a percentage of the largest bucket, so bars stay comparable. */
function barWidth(bucket: any, buckets: any[]): string {
  const max = Math.max(...buckets.map((b) => b.funded || 0), 0)
  if (!max) return '0%'
  return ((bucket.funded || 0) / max * 100).toFixed(1) + '%'
}
</script>

<template>
  <div v-if="!data" class="placeholder">No summary data.</div>
  <div v-else class="summary">
    <p v-for="(f, i) in flags" :key="i" class="flagline">{{ f }}</p>

    <div class="alloc-grid">
      <!-- Asset Allocation -->
      <section class="card">
        <header>
          <h3>Asset Allocation</h3>
          <span class="sub">{{ data.basis_note }}</span>
        </header>
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
              <td>
                <div class="bar-label">{{ b.label }}</div>
                <div class="bar-track">
                  <div class="bar-fill" :style="{ width: barWidth(b, asset.buckets) }"></div>
                </div>
              </td>
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
      </section>

      <!-- Deal Type Allocation -->
      <section class="card">
        <header>
          <h3>Deal Type Allocation</h3>
          <span class="sub">funded dollars by investment strategy</span>
        </header>
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
              <td>
                <div class="bar-label">{{ b.label }}</div>
                <div class="bar-track">
                  <div class="bar-fill alt" :style="{ width: barWidth(b, dealType.buckets) }"></div>
                </div>
              </td>
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
      </section>
    </div>

    <!-- Ownership-flagged deals: excluded from the scaled allocation -->
    <section v-if="flagged.length" class="card warn">
      <header><h3>Excluded from the allocation</h3></header>
      <p v-for="f in flagged" :key="f.vcode" class="flagline">
        {{ f.vcode }} {{ f.name }} — {{ f.reason }}
        (deal-level funded {{ fmtM(f.funded_deal_level) }}M withheld rather than shown unscaled)
      </p>
    </section>

    <!-- Narrative boxes -->
    <section class="card">
      <header><h3>Commentary</h3></header>
      <div class="narratives">
        <div v-for="n in narratives" :key="n.field" class="narrative">
          <label>{{ label(n.field) }}</label>
          <textarea
            v-model="text[n.field]"
            rows="7"
            spellcheck="true"
            lang="en"
            :readonly="!editable"
            :placeholder="editable ? 'Type commentary…' : ''"
            @change="commit(n.field)"
          ></textarea>
          <span class="hint">
            <template v-if="!editable">Locked — this snapshot is approved.</template>
            <template v-else-if="n.is_blank">Blank. Nothing is auto-generated.</template>
            <template v-else>{{ n.char_count }} characters saved.</template>
          </span>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.summary { display: flex; flex-direction: column; gap: 16px; }

.alloc-grid {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: 16px;
  align-items: start;
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

.bar-label { font-weight: 600; margin-bottom: 3px; }
.bar-track { height: 6px; background: #eceff1; border-radius: 3px; overflow: hidden; min-width: 90px; }
.bar-fill { height: 100%; background: var(--color-accent); border-radius: 3px; }
.bar-fill.alt { background: var(--color-pref, #4caf50); }

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
  .alloc-grid, .narratives { grid-template-columns: 1fr; }
}

@media print {
  .flagline { display: none; }
  .card { border: 1px solid #ccc; break-inside: avoid; }
  .card.warn { display: none; }
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
