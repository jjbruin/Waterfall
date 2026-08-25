<script setup lang="ts">
/**
 * Operating subtab — Economic Occupancy, the three NOI points, Expected and
 * Actual Growth, plus a per-deal operating comment.
 *
 * Operating metrics are property-level and never scaled by ownership — only the
 * four Financial columns are.
 *
 * Grouped by fund WITH subtotals and a portfolio total, labelled as the
 * reference PDF labels them. This reverses the module's original "no subtotals"
 * position: the concern was inventing a figure, and the answer is that every
 * total here is computed by `operating_subtotal` in the backend — NOI summed,
 * occupancy NOI-weighted, growth recomputed from the sums — all three derived
 * from the published page's own arithmetic, so a freeze captures them and the
 * component never invents anything.
 *
 * Those totals sum only the cells this table actually shows: a row reading
 * "n/a" — a development deal, or an acquisition too recent to report —
 * contributes nothing to the total beneath it. Worth knowing while reading this
 * file, because it means a column here really does add up, which was not true
 * before 2026-08-25.
 *
 * Presentational: props in, save events out.
 */
import { computed, ref, watch } from 'vue'
import { fmtM, fmtPct, fmtPctPts, disp, isLiteral } from './format'

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

/** Backend-computed fund subtotals and the portfolio total (see
 *  operating_subtotal — NOI summed, occupancy NOI-weighted, growth from the
 *  sums). Never recomputed here: a total the component invented could disagree
 *  with the frozen payload. */
const subtotals = computed<Record<string, any>>(() => props.data?.subtotals || {})
const total = computed(() => props.data?.total || null)

const allRows = computed(() => {
  const out: { group: string; rows: any[]; subtotal: any }[] = []
  for (const [g, rows] of Object.entries(groups.value)) {
    out.push({ group: g, rows: rows || [], subtotal: subtotals.value[g] || null })
  }
  if (flaggedRows.value.length) {
    out.push({
      group: 'Ownership % unavailable', rows: flaggedRows.value, subtotal: null,
    })
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
      <span v-if="diag.insufficient_history" class="chip">
        {{ diag.insufficient_history }} recent acquisition(s) — owned under a quarter,
        every metric shown as “n/a”
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
                <!--
                  A non-dev row reading n/a in every column needs to say why,
                  for the same reason the Dev tag exists: without it the reader
                  cannot tell a withheld figure from a missing one.
                -->
                <span v-if="r.insufficient_history" class="tag new"
                      :title="`Owned ${r.months_owned} month(s) at quarter end — not enough operating history to report`">New</span>
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
                <!--
                  Read-only renders TEXT, not a disabled control. A textarea
                  keeps its rows="2" height even when empty, which on paper made
                  every deal row ~48px tall and pushed a 33-deal table onto two
                  sheets. It also just reads better: an approved snapshot should
                  not look like a form.
                -->
                <textarea
                  v-if="editable"
                  v-model="comments[r.vcode]"
                  rows="2"
                  spellcheck="true"
                  lang="en"
                  placeholder="Comment…"
                  @change="commit(r.vcode)"
                ></textarea>
                <span v-else class="cmt-text">{{ r.operating_comment || '' }}</span>
              </td>
            </tr>
            <!--
              Fund total, labelled as the PDF labels it ("Total PSC TGA 2022
              LLC"). Occupancy is NOI-weighted and growth is recomputed from the
              summed NOI, both derived from the published page — see
              operating_subtotal.
            -->
            <tr v-if="blk.subtotal" class="subtotal">
              <td class="sticky-l">{{ blk.subtotal.label }}</td>
              <td class="r num">{{ fmtPctPts(blk.subtotal.econ_occ?.projected_ye) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal.noi?.at_close) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal.noi?.uw_ye) }}</td>
              <td class="r num">{{ fmtM(blk.subtotal.noi?.projected_ye) }}</td>
              <td class="r num" :class="growthClass(blk.subtotal.expected_growth)">
                {{ fmtPct(blk.subtotal.expected_growth) }}
              </td>
              <td class="r num" :class="growthClass(blk.subtotal.actual_growth)">
                {{ fmtPct(blk.subtotal.actual_growth) }}
              </td>
              <td class="cmt"></td>
            </tr>
            <tr class="spacer"><td colspan="8"></td></tr>
          </tbody>
        </template>

        <tfoot v-if="total">
          <tr>
            <td class="sticky-l">{{ total.label }} ({{ total.deal_count }})</td>
            <td class="r num">{{ fmtPctPts(total.econ_occ?.projected_ye) }}</td>
            <td class="r num">{{ fmtM(total.noi?.at_close) }}</td>
            <td class="r num">{{ fmtM(total.noi?.uw_ye) }}</td>
            <td class="r num">{{ fmtM(total.noi?.projected_ye) }}</td>
            <td class="r num" :class="growthClass(total.expected_growth)">
              {{ fmtPct(total.expected_growth) }}
            </td>
            <td class="r num" :class="growthClass(total.actual_growth)">
              {{ fmtPct(total.actual_growth) }}
            </td>
            <td class="cmt"></td>
          </tr>
        </tfoot>
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

/* Fund total rows and the portfolio total, as on PDF page 3. */
tr.subtotal td {
  font-weight: 700;
  border-top: 1px solid var(--color-text-secondary);
  background: #f7f9fb;
}
tr.spacer td { height: 8px; padding: 0; border-bottom: none; background: transparent; }
tfoot td {
  font-weight: 700;
  border-top: 2px solid var(--color-text);
  background: #eef2f7;
  padding: 6px 8px;
}

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
.cmt-text { font-size: 12px; line-height: 1.35; white-space: pre-wrap; }

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
/* Distinct from Dev: a different reason for the same n/a. */
.tag.new { background: #e8f0fe; color: #1a4f8a; }
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
