<script setup lang="ts">
/**
 * Operating subtab — Economic Occupancy, the three NOI points, Expected and
 * Actual Growth, plus a per-deal operating comment.
 *
 * Operating metrics are property-level and never scaled by ownership — only the
 * four Financial columns are. Grouped by fund with no subtotals: averaging
 * occupancy or summing NOI across a fund would invent a figure the backend
 * never computed.
 *
 * Presentational: props in, save events out.
 */
import { computed, ref, watch } from 'vue'
import { fmtPct, disp, isLiteral } from './format'

const props = defineProps<{ data: any; editable: boolean }>()
const emit = defineEmits<{
  saveComment: [p: { scope: string; field: string; scope_key?: string; text: string }]
}>()

const groups = computed<Record<string, any[]>>(() => props.data?.groups || {})
const flaggedRows = computed<any[]>(() => props.data?.ownership_flagged || [])
const diag = computed(() => props.data?.diagnostics || {})

const comments = ref<Record<string, string>>({})

watch(() => props.data, (d) => {
  const next: Record<string, string> = {}
  for (const rows of Object.values<any>(d?.groups || {})) {
    for (const r of (rows || [])) next[r.vcode] = r.operating_comment || ''
  }
  for (const r of (d?.ownership_flagged || [])) next[r.vcode] = r.operating_comment || ''
  comments.value = next
}, { immediate: true })

function commit(vcode: string) {
  emit('saveComment', {
    scope: 'deal', field: 'operating', scope_key: vcode,
    text: comments.value[vcode] ?? '',
  })
}

/** Growth is a ratio; colour it by sign so a fund reads at a glance.
 *
 * Takes the *_display value, which is polymorphic — a suppressed dev row hands
 * us the literal "n/a", and colouring that green or red would imply a reading
 * the report is deliberately withholding.
 */
function growthClass(v: unknown): string {
  if (typeof v !== 'number') return ''
  return v < 0 ? 'neg' : v > 0 ? 'pos' : ''
}

const allRows = computed(() => {
  const out: { group: string; rows: any[] }[] = []
  for (const [g, rows] of Object.entries(groups.value)) out.push({ group: g, rows: rows || [] })
  if (flaggedRows.value.length) {
    out.push({ group: 'Ownership % unavailable', rows: flaggedRows.value })
  }
  return out
})
</script>

<template>
  <div v-if="!data" class="placeholder">No operating data.</div>
  <div v-else class="op">
    <div class="legend">
      <span class="chip">property-level — not scaled by ownership</span>
      <span v-if="diag.dev" class="chip">
        {{ diag.dev }} development deal(s) — every metric shown as “n/a”
      </span>
      <span v-if="diag.dev_exceptions" class="chip warn">
        {{ diag.dev_exceptions }} temporary exception(s) marked * show real values
      </span>
      <span v-if="diag.missing_at_close" class="chip warn">
        {{ diag.missing_at_close }} deal(s) have no At Close NOI, so growth is unavailable
        where it is not already suppressed
      </span>
    </div>

    <div class="scroll">
      <table class="grid">
        <thead>
          <tr>
            <th class="sticky-l">Deal</th>
            <th class="r">Econ Occ</th>
            <th class="r">NOI At Close</th>
            <th class="r">NOI U/W YE</th>
            <th class="r">NOI Projected YE</th>
            <th class="r">Expected Growth</th>
            <th class="r">Actual Growth</th>
            <th class="cmt">Operating comment</th>
          </tr>
          <tr class="unitrow">
            <th class="sticky-l"></th>
            <th class="r"></th>
            <th class="r">$M</th><th class="r">$M</th><th class="r">$M</th>
            <th class="r"></th><th class="r"></th><th class="cmt"></th>
          </tr>
        </thead>

        <template v-for="blk in allRows" :key="blk.group">
          <tbody>
            <tr class="grouprow"><td class="sticky-l" colspan="8">{{ blk.group }}</td></tr>
            <tr v-for="r in blk.rows" :key="r.vcode">
              <td class="sticky-l">
                {{ r.name }}
                <span v-if="r.is_dev" class="tag">Dev</span>
                <span v-if="r.dev_display_exception?.length" class="star"
                      :title="`Temporary exception — real ${r.dev_display_exception.join(', ')} shown despite dev classification`">*</span>
                <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
              </td>
              <!--
                Every metric renders its backend *_display twin, never the raw
                field: a development row is suppressed to the literal "n/a"
                there, and formatting `r.noi` or `r.expected_growth` directly
                would opt that cell out of the rule. Green Valley Ranch showed
                Expected Growth of -2761.9% that way.

                'pctpts', not 'pct', on Econ Occ: it arrives from the One Pager
                already in percentage points (92.23), so 'pct' multiplied it by
                100 again and rendered "9223.0%". Growth IS a ratio, so 'pct'.
              -->
              <td class="r num" :class="{ lit: isLiteral(r.econ_occ_display) }">
                {{ disp(r.econ_occ_display, 'pctpts') }}
              </td>
              <td class="r num" :class="{ lit: isLiteral(r.noi_display?.at_close) }">
                {{ disp(r.noi_display?.at_close, 'm') }}
              </td>
              <td class="r num" :class="{ lit: isLiteral(r.noi_display?.uw_ye) }">
                {{ disp(r.noi_display?.uw_ye, 'm') }}
              </td>
              <td class="r num" :class="{ lit: isLiteral(r.noi_display?.projected_ye) }">
                {{ disp(r.noi_display?.projected_ye, 'm') }}
              </td>
              <td class="r num" :class="[growthClass(r.expected_growth_display),
                                         { lit: isLiteral(r.expected_growth_display) }]">
                {{ disp(r.expected_growth_display, 'pct') }}
              </td>
              <td class="r num" :class="[growthClass(r.actual_growth_display),
                                         { lit: isLiteral(r.actual_growth_display) }]">
                {{ disp(r.actual_growth_display, 'pct') }}
              </td>
              <td class="cmt">
                <textarea
                  v-model="comments[r.vcode]"
                  rows="2"
                  spellcheck="true"
                  lang="en"
                  :readonly="!editable"
                  :placeholder="editable ? 'Comment…' : ''"
                  @change="commit(r.vcode)"
                ></textarea>
              </td>
            </tr>
          </tbody>
        </template>
      </table>
    </div>

    <p class="hint">
      Actual Growth uses Projected YE (YTD actual + remainder-of-year budget), so it moves
      for a past quarter as actuals land — a quarter recomputed later will not match a
      PDF produced earlier.
    </p>
  </div>
</template>

<style scoped>
.op { display: flex; flex-direction: column; gap: 12px; }

.legend { display: flex; gap: 8px; flex-wrap: wrap; }
.chip {
  font-size: 11px;
  padding: 3px 9px;
  border-radius: 10px;
  background: #f0f0f0;
  color: var(--color-text-secondary);
}
.chip.warn { background: #fff8e1; color: #856404; }

.scroll { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 8px; }
table.grid { width: 100%; border-collapse: collapse; font-size: 12px; }

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
  white-space: nowrap;
  position: sticky;
  top: 0;
  z-index: 2;
}
.unitrow th { top: 26px; font-size: 9px; text-transform: none; }

table.grid td { padding: 5px 8px; border-bottom: 1px solid #f2f2f2; vertical-align: top; }

.sticky-l {
  position: sticky;
  left: 0;
  background: var(--color-surface);
  z-index: 1;
  min-width: 210px;
  white-space: nowrap;
}
th.sticky-l { background: #fafafa; z-index: 3; }

.r { text-align: right; white-space: nowrap; }
table.grid th.r { text-align: right; }
.num { font-variant-numeric: tabular-nums; }
.pos { color: #2e7d32; }
.neg { color: #a12622; }
/* Backend literals ("n/a") read as muted italics, not as data. Same treatment
   as the Loan subtab gives "Dev". */
.lit { color: var(--color-text-secondary); font-style: italic; }
.star { color: #b26a00; font-weight: 800; cursor: help; margin-left: 3px; }

.grouprow td {
  background: #eceff1;
  font-weight: 700;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.cmt { min-width: 260px; }
.cmt textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 4px 7px;
  border: 1px solid var(--color-border);
  border-radius: 5px;
  font-size: 12px;
  font-family: inherit;
  line-height: 1.35;
  resize: vertical;
}
.cmt textarea[readonly] { background: #fafafa; color: var(--color-text-secondary); }

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

.hint { font-size: 11px; color: var(--color-text-secondary); font-style: italic; margin: 0; }
.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  text-align: center;
  padding: 40px 0;
}

@media print {
  .legend { display: none; }
  .hint { display: none; }
  .scroll { overflow: visible; border: 1px solid #ccc; }
  .cmt textarea {
    border: none;
    padding: 0;
    resize: none;
    overflow: visible;
    height: auto !important;
  }
  table.grid { font-size: 10px; }
}
</style>
