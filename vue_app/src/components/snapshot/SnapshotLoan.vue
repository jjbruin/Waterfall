<script setup lang="ts">
/**
 * Loan subtab — Rate, Maturity, Debt, YTD DSCR, LTV, Debt Yield, plus a
 * per-deal loan comment (independent of the Operating comment on the same deal).
 *
 * The three ratio columns render whatever the backend put in `*_display`:
 * a number, the literal "Dev" for a development deal, or n/a when the loan
 * block is empty. `disp()` passes strings through, so nothing here recomputes
 * or overrides a backend decision — that is what keeps "Dev" and n/a honest.
 *
 * Presentational: props in, save events out.
 */
import { computed, ref, watch } from 'vue'
import { fmtM, fmtPct, fmtX, disp, isLiteral, DASH } from './format'

/** The per-loan breakdown, for the cell's tooltip: which facility is which. */
function termsTitle(r: any): string {
  const list = r?.terms_list
  if (!Array.isArray(list) || list.length < 2) return ''
  return list.map((t: any, i: number) =>
    `${i + 1}. ${t.rate_display || '—'} · ${t.maturity_display || '—'}`
    + (t.amount ? ` · $${(t.amount / 1e6).toFixed(1)}M` : '')).join('\n')
}

const props = defineProps<{
  data: any
  editable: boolean
  /**
   * Screen-only annotations. Defaults to off so a render path that does not
   * ask for them never gets them — the consolidated print view omits it.
   * Deliberately NOT derived from `editable`: the app view sets that false on
   * a locked/approved quarter, which is exactly when a reader is comparing
   * against a published PDF and most needs the caveat.
   */
  screenNote?: boolean
}>()
const emit = defineEmits<{
  saveComment: [p: { scope: string; field: string; scope_key?: string; text: string }]
  saveValue: [p: { vcode: string; field: string; value: string | number | null }]
}>()

/** The three ratio columns, where a row says they are typed rather than
 *  computed (`*_is_manual`, from MANUAL_RATIO_SEEDS). */
type RatioField = 'ltv' | 'ytd_dscr' | 'debt_yield'

const groups = computed<Record<string, any[]>>(() => props.data?.groups || {})
const flaggedRows = computed<any[]>(() => props.data?.ownership_flagged || [])
const diag = computed(() => props.data?.diagnostics || {})
const ceiling = computed(() => props.data?.ltv_review_ceiling ?? 1.5)

const comments = ref<Record<string, string>>({})

watch(() => props.data, (d) => {
  const next: Record<string, string> = {}
  for (const rows of Object.values<any>(d?.groups || {})) {
    for (const r of (rows || [])) next[r.vcode] = r.loan_comment || ''
  }
  for (const r of (d?.ownership_flagged || [])) next[r.vcode] = r.loan_comment || ''
  comments.value = next
}, { immediate: true })

function commit(vcode: string) {
  emit('saveComment', {
    scope: 'deal', field: 'loan', scope_key: vcode,
    text: comments.value[vcode] ?? '',
  })
}

/* ── Typed ratio cells ────────────────────────────────────────────────────
 *
 * Same three-part pattern as the Financial subtab's Net ROE / ITD, and
 * deliberately the same code shape so the two cannot drift: a `draft` map of
 * what is in each box, a `focusKey` so a cell shows its FORMATTED value
 * ("69.0%", "1.9x") at rest and the bare number while it is being typed, and
 * a commit that emits the raw text for the server to parse.
 *
 * The value stored is in the unit the column displays — percentage points for
 * LTV and Debt Yield, a multiple for DSCR — so nothing is scaled here. See
 * format_manual_ratio on the backend, which owns the rendering rule; this
 * component never formats a manual figure itself.
 */
const draft = ref<Record<string, string>>({})
const focusKey = ref<string | null>(null)

const RATIO_FIELDS: RatioField[] = ['ltv', 'ytd_dscr', 'debt_yield']

watch(() => props.data, (d) => {
  const next: Record<string, string> = {}
  const seed = (r: any) => {
    for (const f of RATIO_FIELDS) {
      if (!r?.[`${f}_is_manual`]) continue
      const v = r[`${f}_manual`]
      next[`${r.vcode}:${f}`] = v == null ? '' : String(v)
    }
  }
  for (const rows of Object.values<any>(d?.groups || {})) for (const r of (rows || [])) seed(r)
  for (const r of (d?.ownership_flagged || [])) seed(r)
  draft.value = next
}, { immediate: true })

function commitValue(vcode: string, field: RatioField) {
  focusKey.value = null
  const raw = (draft.value[`${vcode}:${field}`] ?? '').trim()
  emit('saveValue', { vcode, field, value: raw === '' ? null : raw })
}

/** What a typed input shows right now: the bare number while it has focus,
 *  the backend's formatted string otherwise, and '' when the cell has been
 *  cleared so the placeholder comes through. */
function cellText(vcode: string, field: RatioField, display: unknown): string {
  const k = `${vcode}:${field}`
  if (focusKey.value === k) return draft.value[k] ?? ''
  if ((draft.value[k] ?? '') === '') return ''
  return typeof display === 'string' ? display : String(draft.value[k])
}

/** The tooltip on a typed cell: whether it is an entry or still the
 *  pre-filled figure, and what the engine computes for the same cell — so the
 *  typed number can always be read against its own arithmetic instead of
 *  hiding it. */
function manualTitle(r: any, field: RatioField, kind: 'pct' | 'x'): string {
  const src = r?.[`${field}_source`] || 'manual entry'
  const c = r?.[`${field}_computed`]
  const computed = c == null ? 'not computable from the data on record'
    : (kind === 'x' ? fmtX(c) : fmtPct(c))
  return `${src} — computed: ${computed}`
}

/** Backend-computed fund subtotals and portfolio total (see loan_subtotal —
 *  Debt summed over every deal, ratios debt-weighted over the deals carrying a
 *  value). Never recomputed here. */
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

/** Excluding-development subtotal — this subtab is the only place it lives. */
const exDevTotal = computed(() => {
  let debt = 0
  let n = 0
  let any = false
  for (const rows of Object.values(groups.value)) {
    for (const r of (rows || [])) {
      if (r.is_dev) continue
      n++
      if (r.debt != null) { debt += r.debt; any = true }
    }
  }
  return { debt: any ? debt : null, deal_count: n }
})

const devCount = computed(() => {
  let n = 0
  for (const rows of Object.values(groups.value)) {
    for (const r of (rows || [])) if (r.is_dev) n++
  }
  return n
})

/**
 * The Debt cell's value.
 *
 * `debt_display` arrived 2026-09-01 so a debt-free deal could print a dash
 * instead of its real 0.0 balance. Snapshots frozen before that date have no
 * such key, and reading it straight would render every one of their Debt cells
 * as a dash — so the fallback tests for the KEY, not for a nullish value: a
 * present null is the deliberate dash, an absent key is an old payload.
 */
function debtCell(r: any): unknown {
  return r && 'debt_display' in r ? r.debt_display : r?.debt
}
</script>

<template>
  <div v-if="!data" class="placeholder">No loan data.</div>
  <div v-else class="loan">
    <div class="legend">
      <span class="chip">property-level — not scaled by ownership</span>
      <span class="chip">Debt Yield = single-quarter NOI &times; 4 &divide; debt</span>
      <span v-if="devCount" class="chip">
        {{ devCount }} development deal(s) show “Dev” for LTV, DSCR and Debt Yield
      </span>
      <span v-if="diag.debt_free" class="chip">
        {{ diag.debt_free }} debt-free deal(s) show “N/A”
      </span>
      <span v-if="diag.manual_deals" class="chip">
        {{ diag.manual_deals }} recent acquisition(s) carry typed LTV / DSCR /
        Debt Yield — {{ diag.manual_entered || 0 }} of {{ diag.manual_cells }}
        cells entered, the rest pre-filled
      </span>
      <span v-if="diag.dev_no_data" class="chip warn">
        {{ diag.dev_no_data }} development deal(s) have an empty loan block
      </span>
      <span v-if="diag.ltv_flagged_review" class="chip warn">
        {{ diag.ltv_flagged_review }} LTV withheld above {{ fmtPct(ceiling, 0) }}
      </span>
    </div>

    <div class="scroll">
      <table class="grid">
        <thead>
          <tr>
            <th class="sticky-l">Deal</th>
            <th>Rate</th>
            <th>Maturity</th>
            <th class="r">Debt</th>
            <th class="r">YTD DSCR</th>
            <th class="r">LTV</th>
            <th class="r">Debt Yield</th>
            <th class="cmt">Loan comment</th>
          </tr>
          <tr class="unitrow">
            <th class="sticky-l"></th>
            <th></th><th></th>
            <th class="r">$M</th>
            <th class="r"></th><th class="r"></th><th class="r"></th>
            <th class="cmt"></th>
          </tr>
        </thead>

        <template v-for="blk in allRows" :key="blk.group">
          <tbody>
            <tr class="grouprow"><td class="sticky-l" colspan="8">{{ blk.group }}</td></tr>
            <tr v-for="r in blk.rows" :key="r.vcode">
              <td class="sticky-l">
                {{ r.name }}
                <span v-if="r.is_dev" class="tag">Dev</span>
                <span v-if="r.loans_inherited_from_children" class="tag alt"
                      title="No loans on this deal; terms inherited from its child properties">child</span>
                <span v-for="(f, i) in (r.flags || [])" :key="i" class="warn-dot" :title="f">!</span>
              </td>
              <!-- A multi-loan deal lists each facility's real terms, largest
                   first, joined by " | " and in the same order across both
                   columns, so the first rate belongs to the first maturity.
                   `multi` only lets the cell wrap between loans; it is real
                   data now and is not dimmed. It used to read "Various", which
                   is why these were styled as a soft literal. -->
              <td :class="{ multi: r.terms_various }" :title="termsTitle(r)">
                {{ r.rate_display ?? DASH }}</td>
              <td :class="{ multi: r.terms_various }" :title="termsTitle(r)">
                {{ r.maturity_display ?? DASH }}</td>
              <!-- debt_display, not debt: a debt-free deal prints a dash where
                   its real 0.0 balance would otherwise read as "$0.0". Same
                   pattern as SnapshotFinancial's debt_display. -->
              <td class="r num" :class="{ lit: isLiteral(debtCell(r)) }"
                  :title="r.debt_free ? 'Held with no debt' : r.debt_basis">
                {{ disp(debtCell(r), 'm') }}
              </td>
              <!-- The three ratio columns. A row the backend marks
                   `*_is_manual` renders a typeable box holding the entry (or
                   the figure it was pre-filled with); every other row is
                   exactly the read-only cell it always was. The `_computed`
                   figure rides along in the tooltip so a typed cell can be
                   read against its own arithmetic. -->
              <td class="r num" :class="{ manual: r.ytd_dscr_is_manual, lit: !r.ytd_dscr_is_manual && isLiteral(r.ytd_dscr_display) }">
                <input
                  v-if="r.ytd_dscr_is_manual"
                  :value="cellText(r.vcode, 'ytd_dscr', r.ytd_dscr_display)"
                  @input="draft[`${r.vcode}:ytd_dscr`] = ($event.target as HTMLInputElement).value"
                  @focus="focusKey = `${r.vcode}:ytd_dscr`"
                  @blur="focusKey = null"
                  class="numinput"
                  :class="{ pending: r.ytd_dscr_manual == null }"
                  :readonly="!editable"
                  :placeholder="String(r.ytd_dscr_display ?? '')"
                  :title="manualTitle(r, 'ytd_dscr', 'x')"
                  @change="commitValue(r.vcode, 'ytd_dscr')"
                />
                <template v-else>{{ disp(r.ytd_dscr_display, 'x') }}</template>
              </td>
              <td class="r num" :class="{ manual: r.ltv_is_manual, lit: !r.ltv_is_manual && isLiteral(r.ltv_display) }"
                  :title="r.ltv_review_flag || (r.ltv_dev_exception ? 'Temporary exception — real LTV shown despite dev classification' : '')">
                <input
                  v-if="r.ltv_is_manual"
                  :value="cellText(r.vcode, 'ltv', r.ltv_display)"
                  @input="draft[`${r.vcode}:ltv`] = ($event.target as HTMLInputElement).value"
                  @focus="focusKey = `${r.vcode}:ltv`"
                  @blur="focusKey = null"
                  class="numinput"
                  :class="{ pending: r.ltv_manual == null }"
                  :readonly="!editable"
                  :placeholder="String(r.ltv_display ?? '')"
                  :title="manualTitle(r, 'ltv', 'pct')"
                  @change="commitValue(r.vcode, 'ltv')"
                />
                <template v-else>
                  {{ disp(r.ltv_display, 'pct') }}
                  <span v-if="r.ltv_dev_exception" class="star" title="Temporary LTV exception">*</span>
                </template>
              </td>
              <td class="r num" :class="{ manual: r.debt_yield_is_manual, lit: !r.debt_yield_is_manual && isLiteral(r.debt_yield_display) }">
                <input
                  v-if="r.debt_yield_is_manual"
                  :value="cellText(r.vcode, 'debt_yield', r.debt_yield_display)"
                  @input="draft[`${r.vcode}:debt_yield`] = ($event.target as HTMLInputElement).value"
                  @focus="focusKey = `${r.vcode}:debt_yield`"
                  @blur="focusKey = null"
                  class="numinput"
                  :class="{ pending: r.debt_yield_manual == null }"
                  :readonly="!editable"
                  :placeholder="String(r.debt_yield_display ?? '')"
                  :title="manualTitle(r, 'debt_yield', 'pct')"
                  @change="commitValue(r.vcode, 'debt_yield')"
                />
                <template v-else>{{ disp(r.debt_yield_display, 'pct') }}</template>
              </td>
              <td class="cmt">
                <!-- Read-only renders TEXT — see the note in SnapshotOperating:
                     an empty textarea keeps its rows="2" height and doubled the
                     printed table's length. -->
                <textarea
                  v-if="editable"
                  v-model="comments[r.vcode]"
                  rows="2"
                  spellcheck="true"
                  lang="en"
                  placeholder="Comment…"
                  @change="commit(r.vcode)"
                ></textarea>
                <span v-else class="cmt-text">{{ r.loan_comment || '' }}</span>
              </td>
            </tr>
            <!--
              Fund total, labelled as the PDF labels it. Debt is summed over
              every deal including development; the three ratios are
              debt-weighted over the deals carrying a value, which is what
              reproduces the published LTVs exactly. See loan_subtotal.
            -->
            <tr v-if="blk.subtotal" class="subtotal">
              <td class="sticky-l">{{ blk.subtotal.label }}</td>
              <td></td><td></td>
              <td class="r num">{{ fmtM(blk.subtotal.debt) }}</td>
              <td class="r num">{{ fmtX(blk.subtotal.ytd_dscr) }}</td>
              <td class="r num">{{ fmtPct(blk.subtotal.ltv) }}</td>
              <td class="r num">{{ fmtPct(blk.subtotal.debt_yield) }}</td>
              <td class="cmt"></td>
            </tr>
            <tr class="spacer"><td colspan="8"></td></tr>
          </tbody>
        </template>

        <tfoot>
          <tr v-if="total">
            <td class="sticky-l">{{ total.label }} ({{ total.deal_count }})</td>
            <td></td><td></td>
            <td class="r num">{{ fmtM(total.debt) }}</td>
            <td class="r num">{{ fmtX(total.ytd_dscr) }}</td>
            <td class="r num">{{ fmtPct(total.ltv) }}</td>
            <td class="r num">{{ fmtPct(total.debt_yield) }}</td>
            <td class="cmt"></td>
          </tr>
          <tr class="exdev-note">
            <td class="sticky-l">
              Excluding development deals ({{ exDevTotal.deal_count }})
            </td>
            <td></td><td></td>
            <td class="r num">{{ fmtM(exDevTotal.debt) }}</td>
            <td colspan="4" class="note">
              summary ratios already exclude the development deals — they carry no
              value to weight
            </td>
          </tr>
        </tfoot>
      </table>
    </div>

    <p class="hint">
      Development deals use the committed facility (<code>mOrigLoanAmt</code>) for Debt;
      operating deals use the ISBS balance-sheet balance as of quarter end. Hover a Debt
      figure for its basis.
    </p>

    <!--
      Screen only. Suppressed twice over: this v-if (the print view omits the
      prop) and `:deep(.hint) { display: none }` under @media print in
      PortfolioSnapshotPrintView. Lives in the template, never in `data`, so it
      cannot reach the frozen snapshot payload or the footnotes table.
    -->
    <p v-if="screenNote" class="hint">
      Loan details reflect the current facility on record. Loans refinanced or paid
      off after a given quarter's report date may not match that quarter's published
      figures.
    </p>
  </div>
</template>

<style scoped>
.loan { display: flex; flex-direction: column; gap: 12px; }

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

tr.subtotal td {
  font-weight: 700;
  border-top: 1px solid var(--color-text-secondary);
  background: #f7f9fb;
}
tr.spacer td { height: 8px; padding: 0; border-bottom: none; background: transparent; }
tfoot td { font-weight: 700; border-top: 2px solid var(--color-text); background: #eef2f7; }
tfoot tr.exdev-note td { font-weight: 400; border-top: 1px solid var(--color-border); background: transparent; }

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

table.grid td { padding: 5px 8px; border-bottom: 1px solid #f2f2f2; vertical-align: top; white-space: nowrap; }

.sticky-l {
  position: sticky;
  left: 0;
  background: var(--color-surface);
  z-index: 1;
  min-width: 210px;
}
th.sticky-l { background: #fafafa; z-index: 3; }

.r { text-align: right; }
table.grid th.r { text-align: right; }
.num { font-variant-numeric: tabular-nums; }

/* A backend literal ("Dev", n/a) is not a measurement — de-emphasise it. */
.lit { color: var(--color-text-secondary); font-style: italic; }

/* A typed cell, tinted the same as the Financial subtab's manual columns so
   "this figure was entered, not computed" reads the same on both pages. */
td.manual { background: #fffdf5; }
.numinput {
  width: 72px;
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
/* A cell listing more than one loan. Normal weight and colour — it is data,
   not a placeholder — and allowed to wrap at the separator so two facilities
   never force the table wider than one printed page. */
.multi { white-space: normal; }
.star { color: #b26a00; font-weight: 800; cursor: help; }

.grouprow td {
  background: #eceff1;
  font-weight: 700;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

tfoot td {
  font-weight: 800;
  border-top: 2px solid var(--color-border);
  background: #f2f4f7;
}
tfoot .note {
  font-weight: 400;
  font-size: 10px;
  color: var(--color-text-secondary);
  font-style: italic;
  white-space: normal;
}

.cmt { min-width: 260px; white-space: normal; }
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
.tag.alt { background: #e8eef8; color: #2c4f8c; }
.warn-dot {
  display: inline-block;
  margin-left: 4px;
  color: #b26a00;
  font-weight: 800;
  cursor: help;
}

.hint { font-size: 11px; color: var(--color-text-secondary); font-style: italic; margin: 0; }
.hint code { font-size: 11px; background: #f5f5f5; padding: 1px 4px; border-radius: 3px; }
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
  /* A typed cell prints as the figure, not as a form control — same rule the
     Financial subtab uses for Net ROE and ITD. The value is the input's own
     text (the backend's formatted string at rest), so it carries its unit
     into print. Width auto so the number is not clipped at 72px. */
  .numinput {
    border: none;
    background: transparent !important;
    padding: 0;
    width: auto;
    color: inherit;
  }
  td.manual { background: transparent; }
  table.grid { font-size: 10px; }
}
</style>
