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
import { fmtM, fmtM$, fmtPct, disp, isLiteral } from './format'

const props = defineProps<{ data: any; editable: boolean }>()
const emit = defineEmits<{
  saveValue: [p: { vcode: string; field: string; value: string | number | null }]
  addFootnote: [p: { anchor: string; text: string }]
  removeFootnote: [id: number]
}>()

const groups = computed<Record<string, any>>(() => props.data?.groups || {})
const total = computed(() => props.data?.total || null)
/** The funded-to-date twin of Portfolio Totals, directly beneath it.
 *  Backend-computed by `_funding_total`: each column is the sum of the One
 *  Pager cap-stack value every row already carries, on the FUNDED basis. */
const funding = computed(() => props.data?.total_current_funding || null)
/** The PDF's "Excluding Development Deals" row. Backend-computed; see
 *  EXCLUDING_DEV_VCODES for why its population is not simply `is_dev`. */
const exDev = computed(() => props.data?.total_excluding_dev || null)
const flaggedRows = computed<any[]>(() => props.data?.ownership_flagged || [])

/**
 * Footnotes — ONE list, already numbered and scope-resolved by the backend.
 *
 * There is no second hardcoded list here any more. The page's standing notes
 * live in `STANDING_FOOTNOTES` in portfolio_snapshot_financial.py alongside the
 * analyst-entered rows, `compose_footnotes` numbers both in one sequence, and
 * `footnote_marks` says where each number goes. The component renders that
 * index and decides nothing: a footnote whose scope is a COLUMN marks its
 * column header, one whose scope is a PROPERTY marks that deal's name.
 */
const footnotes = computed<any[]>(() => props.data?.footnotes || [])

/** {column: {colKey: [n]}, property: {vcode: [n]}} */
const marks = computed<any>(
  () => props.data?.footnote_marks || { column: {}, property: {} })

/** Anchors the "Add footnote" picker offers — every column plus every deal on
 *  this page, built server-side so a property-scoped note needs no code change. */
const ANCHORS = computed<any[]>(() => props.data?.footnote_anchors || [])

/** Marker for a COLUMN header, by the column's field key. */
function colMark(col: string): string {
  const nums = marks.value?.column?.[col]
  return nums && nums.length ? `(${nums.join(',')})` : ''
}

/** Marker for a PROPERTY name, by the deal's vcode. */
function dealMark(vcode: string): string {
  const nums = marks.value?.property?.[String(vcode || '').toUpperCase()]
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

const newFootnote = ref<{ anchor: string; text: string }>({ anchor: '', text: '' })

watch(ANCHORS, (list) => {
  if (!newFootnote.value.anchor && list?.length) {
    newFootnote.value.anchor = list[0].key
  }
}, { immediate: true })

function addFootnote() {
  if (!newFootnote.value.text.trim() || !newFootnote.value.anchor) return
  emit('addFootnote', { anchor: newFootnote.value.anchor, text: newFootnote.value.text })
  newFootnote.value = { anchor: newFootnote.value.anchor, text: '' }
}

/** Where a footnote's marker sits, for the list entry. The backend resolved the
 *  scope; this only words it. */
function placementLabel(f: any): string {
  if (f?.scope === 'property') {
    return `${f.label || f.vcode} (property)`
  }
  return f?.column ? `${f.label} column` : (f?.label || '')
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
          <!--
            Spanning band, per PDF page 2: "TIAA Investment" sits over the four
            scaled columns, separating them from the deal-level cap stack on the
            left. The zone-b tint stays — it carries the same meaning on screen
            and is what the legend chip explains.
          -->
          <tr class="spanrow">
            <th class="sticky-l"></th>
            <th colspan="4"></th>
            <th class="span-tiaa" colspan="4">TIAA Investment</th>
            <th colspan="2"></th>
          </tr>
          <tr>
            <!--
              Every header carries its own marker slot. A footnote that
              describes how a COLUMN is calculated belongs here, and `colMark`
              is the only thing that decides whether one shows — no header is
              special-cased, so re-anchoring a footnote moves its number with
              no change to this markup.
            -->
            <th class="sticky-l">Property</th>
            <th class="r">Debt{{ colMark('debt') }}</th>
            <th class="r">Total Pref{{ colMark('total_pref') }}</th>
            <th class="r">Ptr. Equity{{ colMark('ptr_equity') }}</th>
            <th class="r">Total Cap{{ colMark('total_cap') }}</th>
            <th class="r zone-b">% of Pref{{ colMark('pct_of_pref') }}</th>
            <th class="r zone-b">Invested{{ colMark('invested') }}</th>
            <th class="r zone-b">Un-funded{{ colMark('unfunded') }}</th>
            <th class="r zone-b">Total Commitment{{ colMark('total_commitment') }}</th>
            <th class="r manual">ITD Distributions{{ colMark('itd') }}</th>
            <th class="r manual">Net ROE{{ colMark('net_roe') }}</th>
          </tr>
          <tr class="unitrow">
            <th class="sticky-l"></th>
            <th class="r">$M</th><th class="r">$M</th><th class="r">$M</th><th class="r">$M</th>
            <th class="r zone-b"></th><th class="r zone-b">$M</th>
            <th class="r zone-b">$M</th><th class="r zone-b">$M</th>
            <th class="r manual">$</th><th class="r manual"></th>
          </tr>
        </thead>

        <!--
          No group HEADER row: PDF page 2 lists a fund's deals and then names the
          fund on its total row ("Total PSC TGA 2022 LLC"), with a blank spacer
          between blocks. A header would repeat the label two rows above itself.
        -->
        <template v-for="(blk, gname) in groups" :key="gname">
          <tbody>
            <tr v-for="r in blk.deals" :key="r.vcode">
              <td class="sticky-l">
                {{ r.name }}<span v-if="dealMark(r.vcode)" class="fnmark">{{ dealMark(r.vcode) }}</span>
                <span v-if="r.is_dev" class="tag">Dev</span>
                <span v-if="r.pdf_na_cells?.length || r.kept_despite_sold" class="star"
                      :title="(r.flags || []).join(' · ')">*</span>
                <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
              </td>
              <!-- debt_display, not debt: the PDF blanks it for City West -->
              <td class="r num" :class="{ lit: isLiteral(r.debt_display) }">
                {{ disp(r.debt_display, 'm') }}
              </td>
              <td class="r num">{{ fmtM(r.total_pref) }}</td>
              <td class="r num">{{ fmtM(r.ptr_equity) }}</td>
              <td class="r num">{{ fmtM(r.total_cap) }}</td>
              <td class="r num zone-b">{{ fmtPct(r.pct_of_pref) }}</td>
              <td class="r num zone-b">{{ fmtM(r.invested) }}</td>
              <td class="r num zone-b">{{ fmtM(r.unfunded) }}</td>
              <td class="r num zone-b">{{ fmtM(r.total_commitment) }}</td>
              <td class="r manual">
                <input
                  v-model="draft[`${r.vcode}:itd`]"
                  class="numinput"
                  :class="{ pending: r.itd == null }"
                  :readonly="!editable || r.pdf_na_cells?.includes('itd')"
                  :placeholder="String(r.itd_display ?? '')"
                  :title="r.itd_source"
                  @change="commitValue(r.vcode, 'itd')"
                />
              </td>
              <td class="r manual">
                <input
                  v-model="draft[`${r.vcode}:net_roe`]"
                  class="numinput"
                  :class="{ pending: r.net_roe == null }"
                  :readonly="!editable || r.pdf_na_cells?.includes('net_roe')"
                  :placeholder="String(r.net_roe_display ?? '')"
                  :title="r.net_roe_source"
                  @change="commitValue(r.vcode, 'net_roe')"
                />
              </td>
            </tr>
            <tr class="subtotal">
              <td class="sticky-l">{{ blk.subtotal?.label || gname }}</td>
              <td class="r num">{{ fmtM$(blk.subtotal?.debt) }}</td>
              <td class="r num">{{ fmtM$(blk.subtotal?.total_pref) }}</td>
              <td class="r num">{{ fmtM$(blk.subtotal?.ptr_equity) }}</td>
              <td class="r num">{{ fmtM$(blk.subtotal?.total_cap) }}</td>
              <td class="r num zone-b">{{ fmtPct(blk.subtotal?.pct_of_pref) }}</td>
              <td class="r num zone-b">{{ fmtM$(blk.subtotal?.invested) }}</td>
              <td class="r num zone-b">{{ fmtM$(blk.subtotal?.unfunded) }}</td>
              <td class="r num zone-b">{{ fmtM$(blk.subtotal?.total_commitment) }}</td>
              <td class="r manual small">{{ blk.subtotal?.manual_entered?.itd ?? 0 }} entered</td>
              <td class="r manual small">{{ blk.subtotal?.manual_entered?.net_roe ?? 0 }} entered</td>
            </tr>
            <tr class="spacer"><td colspan="11"></td></tr>
          </tbody>
        </template>

        <tbody v-if="flaggedRows.length">
          <tr class="grouprow"><td class="sticky-l" colspan="11">Ownership % unavailable</td></tr>
          <tr v-for="r in flaggedRows" :key="r.vcode">
            <td class="sticky-l">
              {{ r.name }}<span v-if="dealMark(r.vcode)" class="fnmark">{{ dealMark(r.vcode) }}</span>
              <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
            </td>
            <td class="r num">{{ fmtM(r.debt) }}</td>
            <td class="r num">{{ fmtM(r.total_pref) }}</td>
            <td class="r num">{{ fmtM(r.ptr_equity) }}</td>
            <td class="r num">{{ fmtM(r.total_cap) }}</td>
            <td class="r num zone-b" colspan="4">withheld — ownership chain unresolved</td>
            <td class="r manual">{{ disp(r.itd_display) }}</td>
            <td class="r manual">{{ disp(r.net_roe_display) }}</td>
          </tr>
        </tbody>

        <tfoot v-if="total">
          <tr>
            <td class="sticky-l">{{ total.label }} ({{ total.deal_count }})</td>
            <td class="r num">{{ fmtM$(total.debt) }}</td>
            <td class="r num">{{ fmtM$(total.total_pref) }}</td>
            <td class="r num">{{ fmtM$(total.ptr_equity) }}</td>
            <td class="r num">{{ fmtM$(total.total_cap) }}</td>
            <td class="r num zone-b">{{ fmtPct(total.pct_of_pref) }}</td>
            <td class="r num zone-b">{{ fmtM$(total.invested) }}</td>
            <td class="r num zone-b">{{ fmtM$(total.unfunded) }}</td>
            <td class="r num zone-b">{{ fmtM$(total.total_commitment) }}</td>
            <td class="r manual small">{{ total.manual_entered?.itd ?? 0 }} entered</td>
            <td class="r manual small">{{ total.manual_entered?.net_roe ?? 0 }} entered</td>
          </tr>
          <!--
            "Total Current Funding" — directly under Portfolio Totals, same
            deal population, funded basis. Each of the four cells is its OWN
            column total (not one summed figure), and each is a sum of the One
            Pager cap-stack value the rows above already carry: Debt is the
            quarter-end balance-sheet balance rather than the development
            rebase the Debt column itself uses, and Total Pref is FUNDED pref
            rather than the committed tranche. The scaled TIAA columns are left
            blank — they are already a funded/committed pair of their own
            (Invested / Total Commitment) and a third basis under them would
            invite the wrong subtraction.
          -->
          <tr v-if="funding" class="funding">
            <td class="sticky-l" :title="funding.basis">{{ funding.label }}</td>
            <td class="r num" :title="funding.debt_source">{{ fmtM$(funding.debt) }}</td>
            <td class="r num" :title="funding.total_pref_source">{{ fmtM$(funding.total_pref) }}</td>
            <td class="r num" :title="funding.ptr_equity_source">{{ fmtM$(funding.ptr_equity) }}</td>
            <td class="r num" :title="funding.total_cap_source">{{ fmtM$(funding.total_cap) }}</td>
            <td class="r num zone-b"></td>
            <td class="r num zone-b"></td>
            <td class="r num zone-b"></td>
            <td class="r num zone-b"></td>
            <td class="r manual"></td>
            <td class="r manual"></td>
          </tr>
          <!--
            "Excluding Development Deals", per PDF page 2: a right-aligned label
            running up to the Un-funded column, then values in Total Commitment,
            ITD Distributions and Net ROE only. Every other cell is blank on the
            published page, and the backend sends null for them rather than a
            figure nobody published.
          -->
          <tr v-if="exDev" class="exdev">
            <td class="sticky-l"></td>
            <td class="r label" colspan="7" :title="exDev.basis">
              Excluding Development Deals =
              <span class="exdev-n">({{ exDev.excluded_count }} removed,
                {{ exDev.deal_count }} remaining)</span>
            </td>
            <td class="r num zone-b">{{ fmtM$(exDev.total_commitment) }}</td>
            <td class="r manual small">{{ disp(exDev.itd_display, 'currency') }}</td>
            <td class="r manual small">{{ disp(exDev.net_roe_display, 'pct') }}</td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!--
      Footnotes — one list, one sequence.

      The page's standing notes and the analyst-entered ones are numbered
      together by the backend, so no two footnotes can share a number and every
      marker on a header or a property name resolves here. The "where" line
      states the placement the scope produced, which is how a misplaced marker
      shows up as a misplaced marker rather than as a mystery.
    -->
    <section class="footnotes">
      <h4>Footnotes</h4>
      <ol v-if="footnotes.length" class="fnlist">
        <li v-for="f in footnotes" :key="f.number">
          <span class="fnnum">({{ f.number }})</span>
          <span class="fnanchor" :class="{ prop: f.scope === 'property' }">{{ placementLabel(f) }}:</span>
          <span class="fntext">{{ f.text }}</span>
          <!-- Only an analyst-entered footnote is removable here; a standing
               note has no database row to delete. -->
          <button v-if="editable && f.id != null" class="btn-x"
                  title="Remove; the rest re-number"
                  @click="emit('removeFootnote', f.id)">&times;</button>
        </li>
      </ol>
      <p v-else class="hint">No footnotes. Numbering is assigned automatically and re-sequences on removal.</p>

      <div v-if="editable" class="fnadd">
        <select v-model="newFootnote.anchor">
          <optgroup label="Column (marker on the column header)">
            <option v-for="a in ANCHORS.filter((x) => x.scope === 'column')"
                    :key="a.key" :value="a.key">{{ a.label }}</option>
          </optgroup>
          <optgroup label="Property (marker on the property name)">
            <option v-for="a in ANCHORS.filter((x) => x.scope === 'property')"
                    :key="a.key" :value="a.key">{{ a.label }}</option>
          </optgroup>
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
table.grid th.r { text-align: right; }
.num { font-variant-numeric: tabular-nums; }

.zone-b { background: #f4f7fc; }
.manual { background: #fffdf5; }

/* "TIAA Investment" band — centred over its four columns with the PDF's rule
   under the label only, not across the whole row. */
.spanrow th { padding: 3px 8px 1px 8px; border-bottom: none; background: #fafafa; }
/* `table.grid th` sets text-align: left and is element+class, so it outranked a
   bare `.span-tiaa`; the band label rendered left-aligned over its four columns
   instead of centred above them. Matching the selector's specificity fixes it. */
table.grid th.span-tiaa {
  text-align: center;
  font-size: 10px;
  text-transform: none;
  letter-spacing: 0.2px;
  border-bottom: 1px solid var(--color-text-secondary) !important;
  background: #f4f7fc !important;
}
.unitrow th { top: 46px; }

/* Blank line between fund blocks, as on the published page. */
tr.spacer td { height: 9px; padding: 0; border-bottom: none; background: transparent; }

/* Excluding-development row: label right-aligned into the Un-funded column,
   values in the three columns the PDF populates. */
tr.exdev td { border-top: none; font-weight: 600; }
tr.exdev .label {
  text-align: right;
  font-weight: 600;
  white-space: nowrap;
  background: transparent;
}
.exdev-n { font-weight: 400; font-size: 10px; color: var(--color-text-secondary); }
.lit { color: var(--color-text-secondary); font-style: italic; }
.star { color: #b26a00; font-weight: 800; cursor: help; margin-left: 3px; }

/* Footnote marker on a property name. Superscript so it reads as a reference
   and cannot be mistaken for part of the deal name. Column-header markers are
   inline text in the <th> and need no styling of their own. */
.fnmark {
  font-size: 9px;
  vertical-align: super;
  line-height: 0;
  color: var(--color-text-secondary);
  margin-left: 1px;
}

/* "Total Current Funding" — a subtotal of the same population on the funded
   basis, so it reads as part of the total block but lighter than it. */
tfoot tr.funding td {
  font-weight: 600;
  border-top: none;
  background: #f7f9fb;
}

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
.fnanchor.prop { color: #2c4f8c; }
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

@media print {
  .legend { display: none; }
  .scroll { overflow: visible; border: 1px solid #ccc; }
  .fnadd { display: none; }
  .footnotes { break-inside: avoid; border: 1px solid #ccc; }
  .numinput { border: none; background: transparent !important; padding: 0; }
  table.grid { font-size: 10px; }
}
</style>
