<template>
  <div class="line-mapping">
    <!-- ── Step 1: the file ─────────────────────────────────────────────── -->
    <div class="lm-head">
      <div>
        <h4>{{ title }}</h4>
        <p class="lm-note">{{ blurb }}</p>
      </div>
      <div class="lm-file no-print">
        <input type="file" ref="fileInput" accept=".xlsx,.xls,.csv" style="display:none"
               @change="onFile" />
        <button class="btn-secondary" :disabled="parsing || !editable"
                @click="$refs.fileInput.click()">
          {{ parsed ? 'Choose a different file' : 'Choose file' }}
        </button>
        <span v-if="parsed" class="lm-filename">{{ parsed.filename }}</span>
      </div>
    </div>

    <div v-if="parsing" class="loading-text">Reading the spreadsheet...</div>
    <div v-if="error" class="lm-error">{{ error }}</div>

    <template v-if="parsed">
      <p class="lm-note">
        {{ parsed.lines.length }} line(s), {{ parsed.periods.length }} month(s)
        — {{ fmtPeriod(parsed.periods[0]) }} to {{ fmtPeriod(parsed.periods[parsed.periods.length - 1]) }}.
        <span v-if="source === 'argus' && parsed.suggested_count">
          {{ parsed.suggested_count }} line(s) pre-filled from the Argus keyword rules —
          <strong>check them</strong>, they are a guess.
        </span>
        <span v-else-if="source === 'budget'">
          Nothing is pre-filled: a partner's wording is their own, so every line is yours to assign.
        </span>
        <span v-if="!cats.has_history" class="warn-note">
          This deal has no recent actuals, so the list is not ranked and there are no defaults from history.
        </span>
      </p>

      <!-- ── Step 2: the mapping ───────────────────────────────────────── -->
      <div class="table-scroll">
        <table class="data-table lm-table">
          <thead>
            <tr>
              <th>Line on the spreadsheet</th>
              <th class="num">Total</th>
              <th>Category</th>
              <th>Account</th>
              <th class="ctr">Flip sign</th>
              <th class="num">As imported</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="line in parsed.lines" :key="line.row"
                :class="{ 'lm-subtotal': line.looks_like_total, 'lm-mapped': !!m(line.row).account }">
              <td>
                {{ line.label }}
                <span v-if="line.looks_like_total" class="lm-tag">subtotal</span>
                <span v-if="m(line.row).from_keywords" class="lm-tag lm-guess">keyword guess</span>
                <span v-if="line.months < parsed.periods.length" class="lm-tag lm-partial">
                  {{ line.months }} of {{ parsed.periods.length }} mo
                </span>
              </td>
              <td class="num">{{ fmtCurrency(line.total) }}</td>
              <td>
                <select :value="m(line.row).category || ''" :disabled="!editable"
                        @change="setCategory(line.row, $event.target.value)">
                  <option value="">— not imported —</option>
                  <optgroup label="Used by this deal">
                    <option v-for="c in usedCats" :key="c.category" :value="c.category">
                      {{ c.category }}{{ c.prior_total ? ` — ${fmtCurrency(c.prior_total)}` : '' }}
                    </option>
                  </optgroup>
                  <optgroup label="Other categories">
                    <option v-for="c in unusedCats" :key="c.category" :value="c.category">
                      {{ c.category }}
                    </option>
                  </optgroup>
                </select>
              </td>
              <td>
                <select :value="m(line.row).account || ''" :disabled="!editable || !m(line.row).category"
                        @change="setAccount(line.row, $event.target.value)">
                  <option v-for="a in accountsFor(m(line.row).category)" :key="a.account" :value="a.account">
                    {{ a.account }} {{ a.description || '' }}{{ a.used ? '' : '  (not used recently)' }}
                  </option>
                </select>
              </td>
              <td class="ctr">
                <input type="checkbox" :checked="!!m(line.row).flip" :disabled="!editable || !m(line.row).account"
                       @change="setFlip(line.row, $event.target.checked)" />
              </td>
              <td class="num" :class="{ neg: imported(line) < 0 }">
                {{ m(line.row).account ? fmtCurrency(imported(line)) : '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ── Step 3: does it tie? ──────────────────────────────────────── -->
      <div class="lm-split">
        <div class="lm-recon">
          <h5>Does it tie to the spreadsheet?</h5>
          <table class="data-table">
            <thead>
              <tr>
                <th></th>
                <th class="num">On the spreadsheet</th>
                <th class="num">As we will import it</th>
                <th class="num">Difference</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in reconRows" :key="r.line" :class="{ 'row-total': r.line === 'noi' }">
                <td>{{ reconLabel(r.line) }}</td>
                <td class="num">{{ r.stated === null ? '—' : fmtCurrency(r.stated) }}</td>
                <td class="num">{{ fmtCurrency(r.computed) }}</td>
                <td class="num" :class="{ neg: (r.difference ?? 0) !== 0 }">
                  {{ r.difference === null ? '—' : fmtCurrency(r.difference) }}
                </td>
              </tr>
            </tbody>
          </table>
          <p class="lm-note">
            A difference is <strong>not an error</strong> — spreadsheets carry subtotal rows and
            skipping them is correct. It is here so you can tell a skipped subtotal from a
            line you missed.
            <span v-if="!check?.reconciliation?.has_stated_totals">
              This sheet states no totals of its own, so only our side is shown.
            </span>
          </p>
        </div>

        <!-- ── Step 4: what to look at ─────────────────────────────────── -->
        <div class="lm-checks">
          <h5>
            Checks
            <span class="lm-count">{{ check ? check.mapped_count : 0 }} of
              {{ parsed.lines.length }} lines mapped</span>
          </h5>
          <div v-if="checking" class="loading-text">Checking...</div>
          <template v-else-if="check">
            <div v-for="(b, i) in check.blocking" :key="'b' + i" class="lm-block">
              {{ b.message }}
            </div>
            <div v-for="(w, i) in check.warnings" :key="'w' + i" class="lm-warn">
              {{ w.message }}
              <button v-if="w.accounts?.length" class="lm-more" @click="toggle(i)">
                {{ open[i] ? 'hide' : 'show all' }}
              </button>
              <ul v-if="open[i] && w.accounts" class="lm-acct-list">
                <li v-for="a in w.accounts" :key="a.account">
                  {{ a.account }} {{ a.description }} — {{ a.months }} mo,
                  {{ fmtCurrency(a.prior_total) }}
                </li>
              </ul>
            </div>
            <p v-if="!check.blocking.length && !check.warnings.length" class="lm-clean">
              Nothing flagged.
            </p>
          </template>
        </div>
      </div>

      <!-- ── Step 5: commit ────────────────────────────────────────────── -->
      <div class="form-actions no-print">
        <button class="btn-primary" :disabled="!canCommit" @click="doCommit">
          {{ committing ? 'Importing...' : commitLabel }}
        </button>
        <span v-if="check && !check.can_import" class="lm-blocked-note">
          Resolve the {{ check.blocking.length }} blocking item(s) above first.
        </span>
        <span v-else-if="check?.warnings.length" class="lm-warn-note">
          {{ check.warnings.length }} warning(s) — you can import anyway.
        </span>
      </div>

      <div v-if="result" class="lm-result">
        Imported {{ result.rows_written }} row(s)<template v-if="result.rows_replaced">,
        replacing {{ result.rows_replaced }} from a previous version</template>
        into the <strong>{{ result.column }}</strong> column.
        <template v-if="result.periods?.length">
          {{ fmtPeriod(result.periods[0]) }} to {{ fmtPeriod(result.periods[result.periods.length - 1]) }}.
        </template>
        <div class="lm-note">
          Re-import as many times as you need — the same months are replaced, not stacked,
          so the last version you load is the one the comparison uses.
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
/**
 * ONE screen for the two spreadsheets that feed the budget comparison.
 *
 * The comparison is Estimate | Budget | Valuation. Budget comes from the partner's
 * workbook, Valuation from the appraiser's Argus download, and they are the same job:
 * take somebody else's line names and decide which of our categories each belongs to.
 * So they get the same component, and `source` is the only thing that differs.
 *
 * Category FIRST, then an account within it. The categories are the ~27 rows the
 * comparison actually renders, which is the vocabulary the analyst is already reading on
 * screen; a bare account number asks them to translate in their head from a 169-item
 * list into a row they cannot see. The account is still required, because the supplement
 * stores vAccount and NOI, FAD, DSCR and the waterfall all read individual accounts —
 * it defaults to the one this deal used most, so the common case is one click.
 */
import { ref, computed, watch } from 'vue'
import api from '@/api/client'

const props = defineProps({
  recordId: { type: [Number, String], required: true },
  source: { type: String, required: true },      // 'budget' | 'argus'
  editable: { type: Boolean, default: true },
})
const emit = defineEmits(['committed'])

const parsed = ref(null)
const cats = ref({ categories: [], has_history: true })
const mapping = ref({})
const check = ref(null)
const result = ref(null)
const error = ref('')
const parsing = ref(false)
const checking = ref(false)
const committing = ref(false)
const open = ref({})

const title = computed(() => props.source === 'argus'
  ? "Appraiser's Argus cash flow"
  : "Partner's monthly budget")

const blurb = computed(() => props.source === 'argus'
  ? 'Review how each Argus line is coded before it feeds the Valuation column. The '
    + 'keyword rules pre-fill a guess — this is where you correct it.'
  : 'Load the budget the partner sent. It feeds the Budget column, and re-importing '
    + 'replaces it, so you can keep loading revisions until the version is final.')

const commitLabel = computed(() => props.source === 'argus'
  ? 'Apply mapping to the Valuation column'
  : 'Import into the Budget column')

const usedCats = computed(() => (cats.value.categories || []).filter(c => c.used_by_deal))
const unusedCats = computed(() => (cats.value.categories || []).filter(c => !c.used_by_deal))
const reconRows = computed(() => check.value?.reconciliation?.rows || [])
const canCommit = computed(() =>
  props.editable && !committing.value && !!check.value?.can_import)

function m(row) { return mapping.value[String(row)] || {} }
function catByName(name) {
  return (cats.value.categories || []).find(c => c.category === name) || null
}
function accountsFor(name) { return catByName(name)?.accounts || [] }

/**
 * Picking a category selects the deal's most-used account within it, and sets the flip
 * from how that account actually behaved for this deal. NOT from the 4xxx/5xxx prefix:
 * 4030 Residential Vacancy and 4042 Loss to Lease are 4xxx accounts stored POSITIVE, and
 * 5220 Other (Income) Expense is 5xxx stored NEGATIVE, so a prefix rule gets all three
 * backwards and silently inverts NOI.
 */
function setCategory(row, name) {
  const key = String(row)
  if (!name) { delete mapping.value[key]; mapping.value = { ...mapping.value }; return void runCheck() }
  const c = catByName(name)
  const acct = c?.default_account || null
  mapping.value = {
    ...mapping.value,
    [key]: { category: name, account: acct, flip: defaultFlip(row, name, acct) },
  }
  runCheck()
}

function setAccount(row, acct) {
  const key = String(row)
  const cur = m(row)
  mapping.value = {
    ...mapping.value,
    [key]: { ...cur, account: acct, flip: defaultFlip(row, cur.category, acct) },
  }
  runCheck()
}

function setFlip(row, on) {
  const key = String(row)
  mapping.value = { ...mapping.value, [key]: { ...m(row), flip: on } }
  runCheck()
}

function defaultFlip(row, category, account) {
  const line = (parsed.value?.lines || []).find(l => String(l.row) === String(row))
  const a = (catByName(category)?.accounts || []).find(x => x.account === account)
  if (!line || !a || !line.total) return false
  return (line.total > 0) !== (a.mri_sign > 0)
}

function imported(line) {
  return (line.total || 0) * (m(line.row).flip ? -1 : 1)
}

function toggle(i) { open.value = { ...open.value, [i]: !open.value[i] } }

async function onFile(e) {
  const file = e.target.files?.[0]
  if (!file) return
  parsing.value = true
  error.value = ''
  result.value = null
  check.value = null
  try {
    const form = new FormData()
    form.append('file', file)
    form.append('source', props.source)
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/parse`, form)
    parsed.value = res.data
    cats.value = { categories: res.data.categories, has_history: true }
    // Argus pre-fills; a budget starts empty. Either way the analyst sees it before
    // anything is written — which is the whole point of this screen existing.
    mapping.value = { ...(res.data.suggested || {}) }
    await loadCategories()
    await runCheck()
  } catch (err) {
    error.value = err.response?.data?.error || 'Could not read that file.'
    parsed.value = null
  } finally {
    parsing.value = false
    e.target.value = ''
  }
}

async function loadCategories() {
  try {
    const res = await api.get(
      `/api/valuations/records/${props.recordId}/mapping/categories`)
    cats.value = res.data
  } catch { /* parse already returned the list; the ranking is the only thing lost */ }
}

let checkSeq = 0
async function runCheck() {
  if (!parsed.value) return
  const seq = ++checkSeq
  checking.value = true
  try {
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/check`,
      { source: props.source, parsed: parsed.value, mapping: mapping.value })
    // Only the newest reply wins — the analyst edits faster than the round trip and a
    // late response would otherwise overwrite the current state with a stale verdict.
    if (seq === checkSeq) { check.value = res.data; open.value = {} }
  } catch (err) {
    if (seq === checkSeq) error.value = err.response?.data?.error || 'Check failed.'
  } finally {
    if (seq === checkSeq) checking.value = false
  }
}

async function doCommit() {
  committing.value = true
  error.value = ''
  try {
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/commit`,
      { source: props.source, parsed: parsed.value, mapping: mapping.value })
    result.value = res.data
    emit('committed', res.data)
  } catch (err) {
    error.value = err.response?.data?.error || 'Import failed.'
  } finally {
    committing.value = false
  }
}

function reconLabel(k) {
  return { revenue: 'Total revenue', expense: 'Total expenses', noi: 'NOI' }[k] || k
}

function fmtCurrency(v) {
  if (v === null || v === undefined || isNaN(v)) return '—'
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })
}

function fmtPeriod(p) {
  if (!p) return ''
  const d = new Date(p + 'T00:00:00')
  return isNaN(d) ? p : d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

// Switching records must not leave the previous deal's file on screen.
watch(() => props.recordId, () => {
  parsed.value = null; mapping.value = {}; check.value = null
  result.value = null; error.value = ''
})
</script>

<style scoped>
.line-mapping { display: flex; flex-direction: column; gap: 14px; }
.lm-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.lm-head h4 { margin: 0 0 4px; }
.lm-note { font-size: 12px; color: #666; margin: 4px 0; line-height: 1.5; }
.lm-file { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.lm-filename { font-size: 12px; color: #444; }
.lm-error { background: #fdecea; border-left: 3px solid #c0392b; padding: 8px 10px;
  font-size: 13px; color: #922; }

.lm-table select { width: 100%; max-width: 260px; font-size: 12px; padding: 3px 4px; }
.lm-table td { vertical-align: middle; }
.lm-subtotal { background: #fafafa; color: #888; font-style: italic; }
.lm-mapped { background: #f6fbf7; font-style: normal; color: inherit; }
.lm-tag { font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
  background: #eee; color: #666; padding: 1px 5px; border-radius: 3px; margin-left: 6px; }
.lm-guess { background: #fff4d6; color: #8a6100; }
.lm-partial { background: #e8f0fe; color: #2c5aa0; }
.ctr { text-align: center; }

.lm-split { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 1100px) { .lm-split { grid-template-columns: 1fr; } }
.lm-recon h5, .lm-checks h5 { margin: 0 0 8px; display: flex; justify-content: space-between;
  align-items: baseline; }
.lm-count { font-size: 11px; font-weight: 400; color: #777; }

.lm-block { background: #fdecea; border-left: 3px solid #c0392b; padding: 7px 10px;
  font-size: 12px; margin-bottom: 6px; line-height: 1.45; }
.lm-warn { background: #fff8e6; border-left: 3px solid #e0a800; padding: 7px 10px;
  font-size: 12px; margin-bottom: 6px; line-height: 1.45; }
.lm-clean { font-size: 12px; color: #2e7d32; }
.lm-more { background: none; border: none; color: #2c5aa0; font-size: 11px;
  cursor: pointer; padding: 0 0 0 4px; text-decoration: underline; }
.lm-acct-list { margin: 6px 0 0; padding-left: 18px; font-size: 11px; color: #555; }
.lm-acct-list li { margin: 2px 0; }

.lm-blocked-note { font-size: 12px; color: #c0392b; margin-left: 10px; }
.lm-warn-note { font-size: 12px; color: #8a6100; margin-left: 10px; }
.lm-result { background: #f0f7f1; border-left: 3px solid #2e7d32; padding: 9px 12px;
  font-size: 13px; }
.neg { color: #c0392b; }
</style>
