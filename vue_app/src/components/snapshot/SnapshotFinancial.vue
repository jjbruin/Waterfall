<script setup lang="ts">
/**
 * Financial subtab — deal-level cap stack, the four scaled investor columns,
 * the two manual figures, and anchored auto-numbered footnotes.
 *
 * Zones, per the build spec:
 *   A  Debt / Total Pref / Ptr Equity / Total Cap — deal-level, unscaled
 *   B  % of Pref / Invested / Total Commitment / Un-funded — scaled by
 *      look-through %, marked in the header so the two are never confused
 *   C  Net ROE / ITD — manual entry, read only through the backend accessors
 *
 * Presentational: props in, save events out.
 */
import { computed, ref, watch } from 'vue'
import { fmtM, fmtPct, disp } from './format'

const props = defineProps<{ data: any; editable: boolean }>()
const emit = defineEmits<{
  saveValue: [p: { vcode: string; field: string; value: string | number | null }]
  addFootnote: [p: { anchor: string; text: string }]
  removeFootnote: [id: number]
}>()

const groups = computed<Record<string, any>>(() => props.data?.groups || {})
const total = computed(() => props.data?.total || null)
const footnotes = computed<any[]>(() => props.data?.footnotes || [])
const flaggedRows = computed<any[]>(() => props.data?.ownership_flagged || [])

/** Anchors an analyst can attach a footnote to. */
const ANCHORS = [
  { key: 'invested', label: 'Invested' },
  { key: 'total_commitment', label: 'Total Commitment' },
  { key: 'unfunded', label: 'Un-funded' },
  { key: 'net_roe', label: 'Net ROE' },
  { key: 'itd_distributions', label: 'ITD Distributions' },
]

/** anchor -> footnote numbers, so the header can render its "(n)" markers. */
const marks = computed<Record<string, number[]>>(() => {
  const m: Record<string, number[]> = {}
  for (const f of footnotes.value) {
    const a = f.anchor || ''
    if (!m[a]) m[a] = []
    if (f.number != null) m[a].push(f.number)
  }
  return m
})

function markerFor(anchor: string): string {
  const nums = marks.value[anchor]
  return nums && nums.length ? `(${nums.join(',')})` : ''
}

/** Local draft of the two manual figures, keyed `vcode:field`. */
const draft = ref<Record<string, string>>({})

watch(() => props.data, (d) => {
  const next: Record<string, string> = {}
  for (const blk of Object.values<any>(d?.groups || {})) {
    for (const r of (blk.deals || [])) {
      next[`${r.vcode}:net_roe`] = r.net_roe == null ? '' : String(r.net_roe)
      next[`${r.vcode}:itd`] = r.itd == null ? '' : String(r.itd)
    }
  }
  draft.value = next
}, { immediate: true })

function commitValue(vcode: string, field: 'net_roe' | 'itd') {
  const raw = (draft.value[`${vcode}:${field}`] ?? '').trim()
  emit('saveValue', { vcode, field, value: raw === '' ? null : raw })
}

const newFootnote = ref<{ anchor: string; text: string }>({ anchor: ANCHORS[0].key, text: '' })

function addFootnote() {
  if (!newFootnote.value.text.trim()) return
  emit('addFootnote', { anchor: newFootnote.value.anchor, text: newFootnote.value.text })
  newFootnote.value = { anchor: newFootnote.value.anchor, text: '' }
}

function anchorLabel(a: string): string {
  return ANCHORS.find((x) => x.key === a)?.label || a
}
</script>

<template>
  <div v-if="!data" class="placeholder">No financial data.</div>
  <div v-else class="fin">
    <div class="legend">
      <span class="chip zone-b">shaded = investor share (scaled by % of Pref)</span>
      <span class="chip">commitment basis: {{ data.commitment_basis }}</span>
    </div>

    <div class="scroll">
      <table class="grid">
        <thead>
          <tr>
            <th class="sticky-l">Deal</th>
            <th class="r">Debt</th>
            <th class="r">Total Pref</th>
            <th class="r">Ptr Equity</th>
            <th class="r">Total Cap</th>
            <th class="r zone-b">% of Pref</th>
            <th class="r zone-b">Invested{{ markerFor('invested') }}</th>
            <th class="r zone-b">Total Commitment{{ markerFor('total_commitment') }}</th>
            <th class="r zone-b">Un-funded{{ markerFor('unfunded') }}</th>
            <th class="r manual">Net ROE{{ markerFor('net_roe') }}</th>
            <th class="r manual">ITD Distributions{{ markerFor('itd_distributions') }}</th>
          </tr>
          <tr class="unitrow">
            <th class="sticky-l"></th>
            <th class="r">$M</th><th class="r">$M</th><th class="r">$M</th><th class="r">$M</th>
            <th class="r zone-b"></th><th class="r zone-b">$M</th>
            <th class="r zone-b">$M</th><th class="r zone-b">$M</th>
            <th class="r manual"></th><th class="r manual">$</th>
          </tr>
        </thead>

        <template v-for="(blk, gname) in groups" :key="gname">
          <tbody>
            <tr class="grouprow">
              <td class="sticky-l" colspan="11">{{ gname }}</td>
            </tr>
            <tr v-for="r in blk.deals" :key="r.vcode">
              <td class="sticky-l">
                {{ r.name }}
                <span v-if="r.is_dev" class="tag">Dev</span>
                <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
              </td>
              <td class="r num">{{ fmtM(r.debt) }}</td>
              <td class="r num">{{ fmtM(r.total_pref) }}</td>
              <td class="r num">{{ fmtM(r.ptr_equity) }}</td>
              <td class="r num">{{ fmtM(r.total_cap) }}</td>
              <td class="r num zone-b">{{ fmtPct(r.pct_of_pref) }}</td>
              <td class="r num zone-b">{{ fmtM(r.invested) }}</td>
              <td class="r num zone-b">{{ fmtM(r.total_commitment) }}</td>
              <td class="r num zone-b">{{ fmtM(r.unfunded) }}</td>
              <td class="r manual">
                <input
                  v-model="draft[`${r.vcode}:net_roe`]"
                  class="numinput"
                  :class="{ pending: r.net_roe == null }"
                  :readonly="!editable"
                  :placeholder="String(r.net_roe_display ?? '')"
                  :title="r.net_roe_source"
                  @change="commitValue(r.vcode, 'net_roe')"
                />
              </td>
              <td class="r manual">
                <input
                  v-model="draft[`${r.vcode}:itd`]"
                  class="numinput"
                  :class="{ pending: r.itd == null }"
                  :readonly="!editable"
                  :placeholder="String(r.itd_display ?? '')"
                  :title="r.itd_source"
                  @change="commitValue(r.vcode, 'itd')"
                />
              </td>
            </tr>
            <tr class="subtotal">
              <td class="sticky-l">Subtotal — {{ gname }} ({{ blk.subtotal?.deal_count }})</td>
              <td class="r num">{{ fmtM(blk.subtotal?.debt) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal?.total_pref) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal?.ptr_equity) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal?.total_cap) }}</td>
              <td class="r num zone-b">{{ fmtPct(blk.subtotal?.pct_of_pref) }}</td>
              <td class="r num zone-b">{{ fmtM(blk.subtotal?.invested) }}</td>
              <td class="r num zone-b">{{ fmtM(blk.subtotal?.total_commitment) }}</td>
              <td class="r num zone-b">{{ fmtM(blk.subtotal?.unfunded) }}</td>
              <td class="r manual small">{{ blk.subtotal?.manual_entered?.net_roe ?? 0 }} entered</td>
              <td class="r manual small">{{ blk.subtotal?.manual_entered?.itd ?? 0 }} entered</td>
            </tr>
          </tbody>
        </template>

        <tbody v-if="flaggedRows.length">
          <tr class="grouprow"><td class="sticky-l" colspan="11">Ownership % unavailable</td></tr>
          <tr v-for="r in flaggedRows" :key="r.vcode">
            <td class="sticky-l">
              {{ r.name }}
              <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
            </td>
            <td class="r num">{{ fmtM(r.debt) }}</td>
            <td class="r num">{{ fmtM(r.total_pref) }}</td>
            <td class="r num">{{ fmtM(r.ptr_equity) }}</td>
            <td class="r num">{{ fmtM(r.total_cap) }}</td>
            <td class="r num zone-b" colspan="4">withheld — ownership chain unresolved</td>
            <td class="r manual">{{ disp(r.net_roe_display) }}</td>
            <td class="r manual">{{ disp(r.itd_display) }}</td>
          </tr>
        </tbody>

        <tfoot v-if="total">
          <tr>
            <td class="sticky-l">{{ total.label }} ({{ total.deal_count }})</td>
            <td class="r num">{{ fmtM(total.debt) }}</td>
            <td class="r num">{{ fmtM(total.total_pref) }}</td>
            <td class="r num">{{ fmtM(total.ptr_equity) }}</td>
            <td class="r num">{{ fmtM(total.total_cap) }}</td>
            <td class="r num zone-b">{{ fmtPct(total.pct_of_pref) }}</td>
            <td class="r num zone-b">{{ fmtM(total.invested) }}</td>
            <td class="r num zone-b">{{ fmtM(total.total_commitment) }}</td>
            <td class="r num zone-b">{{ fmtM(total.unfunded) }}</td>
            <td class="r manual small">{{ total.manual_entered?.net_roe ?? 0 }} entered</td>
            <td class="r manual small">{{ total.manual_entered?.itd ?? 0 }} entered</td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!-- Footnotes -->
    <section class="footnotes">
      <h4>Footnotes</h4>
      <ol v-if="footnotes.length" class="fnlist">
        <li v-for="f in footnotes" :key="f.id">
          <span class="fnnum">({{ f.number }})</span>
          <span class="fnanchor">{{ anchorLabel(f.anchor) }}:</span>
          <span class="fntext">{{ f.text }}</span>
          <button v-if="editable" class="btn-x" title="Remove; the rest re-number"
                  @click="emit('removeFootnote', f.id)">&times;</button>
        </li>
      </ol>
      <p v-else class="hint">No footnotes. Numbering is assigned automatically and re-sequences on removal.</p>

      <div v-if="editable" class="fnadd">
        <select v-model="newFootnote.anchor">
          <option v-for="a in ANCHORS" :key="a.key" :value="a.key">{{ a.label }}</option>
        </select>
        <input v-model="newFootnote.text" placeholder="Footnote text…" @keyup.enter="addFootnote" />
        <button class="btn-sm" :disabled="!newFootnote.text.trim()" @click="addFootnote">Add footnote</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.fin { display: flex; flex-direction: column; gap: 14px; }

.legend { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 10px;
  background: #f0f0f0;
  color: var(--color-text-secondary);
}
.chip.zone-b { background: #e8eef8; color: #2c4f8c; font-weight: 600; }

.scroll { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 8px; }

table.grid { width: 100%; border-collapse: collapse; font-size: 12px; white-space: nowrap; }

table.grid th {
  text-align: left;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--color-text-secondary);
  padding: 6px 8px;
  background: #fafafa;
  border-bottom: 1px solid var(--color-border);
  font-weight: 700;
  position: sticky;
  top: 0;
  z-index: 2;
}
.unitrow th { top: 26px; font-size: 9px; text-transform: none; letter-spacing: 0; }

table.grid td { padding: 5px 8px; border-bottom: 1px solid #f2f2f2; }

.sticky-l {
  position: sticky;
  left: 0;
  background: var(--color-surface);
  z-index: 1;
  white-space: nowrap;
  min-width: 210px;
}
th.sticky-l { background: #fafafa; z-index: 3; }

.r { text-align: right; }
.num { font-variant-numeric: tabular-nums; }

.zone-b { background: #f4f7fc; }
.manual { background: #fffdf5; }

.grouprow td {
  background: #eceff1;
  font-weight: 700;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  padding: 5px 8px;
}

.subtotal td {
  font-weight: 700;
  border-top: 1px solid var(--color-border);
  background: #f8f9fa;
}
.subtotal .small, tfoot .small { font-weight: 400; font-size: 10px; color: var(--color-text-secondary); }

tfoot td {
  font-weight: 800;
  border-top: 2px solid var(--color-border);
  background: #f2f4f7;
}

.tag {
  font-size: 9px;
  font-weight: 700;
  background: #eceff1;
  color: #455a64;
  padding: 1px 5px;
  border-radius: 8px;
  margin-left: 5px;
  text-transform: uppercase;
}
.warn-dot {
  display: inline-block;
  margin-left: 4px;
  color: #b26a00;
  font-weight: 800;
  cursor: help;
}

.numinput {
  width: 92px;
  padding: 3px 6px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 12px;
  text-align: right;
  font-variant-numeric: tabular-nums;
  box-sizing: border-box;
}
.numinput[readonly] { background: #fafafa; color: var(--color-text-secondary); }
.numinput.pending::placeholder { color: #b0b0b0; font-style: italic; }

/* --- footnotes --- */
.footnotes {
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 16px;
  background: var(--color-surface);
}
.footnotes h4 { font-size: 13px; margin: 0 0 8px 0; }
.fnlist { margin: 0; padding-left: 0; list-style: none; font-size: 12px; }
.fnlist li { display: flex; gap: 6px; align-items: baseline; padding: 3px 0; }
.fnnum { font-weight: 700; color: var(--color-accent); }
.fnanchor { font-weight: 600; color: var(--color-text-secondary); }
.fntext { flex: 1; }
.btn-x {
  border: none;
  background: transparent;
  cursor: pointer;
  font-size: 15px;
  line-height: 1;
  color: #a12622;
  padding: 0 4px;
}
.fnadd { display: flex; gap: 8px; margin-top: 10px; }
.fnadd select, .fnadd input {
  padding: 5px 8px;
  border: 1px solid var(--color-border);
  border-radius: 5px;
  font-size: 12px;
}
.fnadd input { flex: 1; }
.btn-sm {
  padding: 4px 12px;
  border: 1px solid var(--color-accent);
  background: var(--color-accent);
  color: white;
  border-radius: 5px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
}
.btn-sm:disabled { opacity: 0.5; cursor: not-allowed; }

.hint { font-size: 11px; color: var(--color-text-secondary); font-style: italic; margin: 0; }
.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}
</style>
