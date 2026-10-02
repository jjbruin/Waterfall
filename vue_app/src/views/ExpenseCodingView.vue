<script setup lang="ts">
/**
 * Expense Coding — accounting's side of expense reports (phase 3).
 *
 * Every approved line arrives PRE-CODED: the booking follows the deal
 * (operations → the expense category, pipeline → Deal Cost Receivable, an owned
 * deal → intercompany to the entity that owns it, read from the ownership
 * chain), with accounting's ER - employee - deal - comment description.
 * Accounting changes what needs it; what the app proposed stays beside it.
 * One batch per payroll date produces the MRI GL upload.
 * Design: .claude/memory/expense_reporting.md.
 */
import { ref, computed, onMounted } from 'vue'
import api from '@/api/client'
import { useDataStore } from '@/stores/data'
import ReceiptViewer from '@/components/expenses/ReceiptViewer.vue'

const dataStore = useDataStore()
type Tab = 'code' | 'batch' | 'batches' | 'settings'
const tab = ref<Tab>('code')

const rows = ref<any[]>([])
const categories = ref<any[]>([])
const currencies = ref<Record<string, string>>({})
const recurringItems = ref<any[]>([])
const employees = ref<any[]>([])
const loading = ref(false)

const fmt = (v: any) => v == null ? '' :
  Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const BOOK: Record<string, string> = {
  expense: 'Expense (PSC Manager)', deal_cost: 'Deal Cost Receivable', interco: 'Intercompany',
}
function fail(e: any, what: string) {
  dataStore.addToast(`${what}: ${e?.response?.data?.error || e?.message || e}`, 'error')
}

async function load() {
  loading.value = true
  try {
    const [l, s, o] = await Promise.all([
      api.get('/api/expense-coding/lines'), api.get('/api/expense-coding/settings'),
      api.get('/api/expenses/options')])
    rows.value = l.data.rows
    currencies.value = s.data.currencies
    recurringItems.value = s.data.recurring
    categories.value = o.data.categories
    try { employees.value = (await api.get('/api/expenses/employees')).data.employees } catch { /* list optional */ }
  } catch (e) { fail(e, 'Could not load the coding grid') }
  finally { loading.value = false }
}

const reportsOnGrid = computed(() => {
  const m = new Map<number, any>()
  for (const r of rows.value) {
    const x = m.get(r.report_id) || { id: r.report_id, employee: r.employee, total: 0, lines: 0, problems: 0 }
    x.total += Number(r.amount || 0); x.lines += 1; x.problems += r.problems.length
    m.set(r.report_id, x)
  }
  return [...m.values()]
})

// ---- coding a row ----
const open = ref<any>(null)       // the row being coded
const draft = ref<any>(null)
function startEdit(r: any) {
  open.value = r
  draft.value = {
    booking: r.booking, expense_account: r.expense_account || '',
    description: r.description,
    interco: (r.interco || []).map((a: any) => ({ entity: a.entity, pct: a.pct, rltd: a.rltd || '' })),
  }
}
const pctTotal = computed(() =>
  Math.round((draft.value?.interco || []).reduce((t: number, a: any) => t + (parseFloat(a.pct) || 0), 0) * 10000) / 10000)
async function saveRow() {
  const r = open.value, d = draft.value
  const body: any = {
    // Only what DIFFERS from the proposal is sent, so a later correction to the
    // chart or the ownership record still reaches a row nobody decided by hand.
    booking: d.booking !== r.booking_proposed ? d.booking : null,
    expense_account: d.expense_account !== r.employee_account ? d.expense_account : null,
    description: d.description !== r.description_proposed ? d.description : null,
    interco: null,
  }
  if (d.booking === 'interco') {
    const prop = JSON.stringify((r.interco_proposed?.allocations || []).map((a: any) => [a.entity, a.pct, a.rltd || '']))
    const mine = JSON.stringify(d.interco.map((a: any) => [String(a.entity).toUpperCase(), Number(a.pct), String(a.rltd || '').toUpperCase()]))
    if (mine !== prop) body.interco = d.interco
  }
  try {
    await api.put(`/api/expense-coding/lines/${r.line_id}/${r.split_id}`, body)
    open.value = null
    await load()
  } catch (e) { fail(e, 'Could not save the coding') }
}
async function resetRow(r: any) {
  try {
    await api.put(`/api/expense-coding/lines/${r.line_id}/${r.split_id}`, {})
    open.value = null
    await load()
  } catch (e) { fail(e, 'Could not reset') }
}
const viewing = ref<any>(null)

// ---- the batch ----
const pick = ref<Record<number, boolean>>({})
const payrollDate = ref('')
const suffix = ref('End of Month')
const fx = ref<Record<string, string>>({})
const recPick = ref<Record<number, boolean>>({})
const preview = ref<any>(null)
const busy = ref(false)
const pickedIds = computed(() => Object.entries(pick.value).filter(([, v]) => v).map(([k]) => Number(k)))
const nonUsd = computed(() => Object.keys(currencies.value))

function batchBody(commit: boolean) {
  return {
    report_ids: pickedIds.value, payroll_date: payrollDate.value, credit_suffix: suffix.value,
    fx: Object.fromEntries(Object.entries(fx.value).filter(([, v]) => v !== '')),
    recurring_ids: Object.entries(recPick.value).filter(([, v]) => v).map(([k]) => Number(k)),
    commit,
  }
}
async function runPreview() {
  busy.value = true
  try { preview.value = (await api.post('/api/expense-coding/batches', batchBody(false))).data }
  catch (e: any) { preview.value = e?.response?.data || null; fail(e, 'Could not preview') }
  finally { busy.value = false }
}
async function generate() {
  busy.value = true
  try {
    const r = await api.post('/api/expense-coding/batches', batchBody(true))
    downloadText(r.data.csv, `${r.data.batch_id}.csv`)
    dataStore.addToast(`Batch ${r.data.batch_id} generated — upload the file to MRI.`, 'success')
    preview.value = null; pick.value = {}
    await load(); await loadBatches(); tab.value = 'batches'
  } catch (e: any) { preview.value = e?.response?.data || preview.value; fail(e, 'Could not generate') }
  finally { busy.value = false }
}
function downloadText(textBody: string, name: string) {
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([textBody], { type: 'text/csv' }))
  a.download = name; a.click(); URL.revokeObjectURL(a.href)
}

// Accounting sends an approved report back, with a reason. It returns to the
// employee and has to be approved again before it can be paid.
const returning = ref<number | null>(null)
const returnNote = ref('')
async function returnReport(r: any) {
  try {
    await api.post(`/api/expenses/reports/${r.id}/accounting-return`, { note: returnNote.value })
    dataStore.addToast(`Returned to ${r.employee}.`, 'success')
    returning.value = null
    delete pick.value[r.id]
    await load()
  } catch (e) { fail(e, 'Could not return the report') }
}

// ---- batches ----
const batchList = ref<any[]>([])
const confirmVoid = ref<string | null>(null)
async function loadBatches() {
  try { batchList.value = (await api.get('/api/expense-coding/batches')).data.batches }
  catch (e) { fail(e, 'Could not load batches') }
}
async function downloadBatch(b: any) {
  try {
    const r = await api.get(`/api/expense-coding/batches/${b.batch_id}/csv`, { responseType: 'text' })
    downloadText(r.data, `${b.batch_id}.csv`)
  } catch (e) { fail(e, 'Could not download') }
}
async function voidBatch(b: any) {
  try {
    await api.post(`/api/expense-coding/batches/${b.batch_id}/void`)
    confirmVoid.value = null
    await loadBatches(); await load()
  } catch (e) { fail(e, 'Could not void') }
}

// ---- settings ----
const curEntity = ref('')
const curCode = ref('CAD')
async function saveCurrency(entity: string, code: string) {
  try {
    currencies.value = (await api.put('/api/expense-coding/currency', { entity_id: entity, currency: code })).data
    curEntity.value = ''
  } catch (e) { fail(e, 'Could not save the currency') }
}
const rec = ref<any>({ user_id: '', description: '', account: '', amount: '' })
async function addRecurring() {
  try {
    recurringItems.value = (await api.post('/api/expense-coding/recurring', rec.value)).data
    rec.value = { user_id: '', description: '', account: '', amount: '' }
  } catch (e) { fail(e, 'Could not add') }
}
async function toggleRecurring(r: any) {
  try {
    recurringItems.value = (await api.put(`/api/expense-coding/recurring/${r.id}`, { ...r, active: !r.active })).data
  } catch (e) { fail(e, 'Could not change') }
}

onMounted(async () => { await load(); loadBatches() })
</script>

<template>
  <div class="ec">
    <h2>Expense Coding</h2>
    <div class="subtitle">Approved expense reports, pre-coded for the journal entry. Correct what needs it;
      the app's proposal stays beside your change. One batch per payroll date makes the MRI upload.</div>

    <div class="tabs">
      <button :class="{ active: tab === 'code' }" @click="tab = 'code'">Code lines ({{ rows.length }})</button>
      <button :class="{ active: tab === 'batch' }" @click="tab = 'batch'">Payroll batch</button>
      <button :class="{ active: tab === 'batches' }" @click="tab = 'batches'; loadBatches()">Batches</button>
      <button :class="{ active: tab === 'settings' }" @click="tab = 'settings'">Currencies &amp; recurring</button>
    </div>

    <!-- ============ code ============ -->
    <template v-if="tab === 'code'">
      <div v-if="loading" class="muted">Loading…</div>
      <div v-else-if="!rows.length" class="muted">No approved reports are waiting.</div>
      <div v-else :class="{ split: viewing }">
        <table class="data-table">
          <thead><tr>
            <th>Employee</th><th>Date</th><th>Deal</th><th>Comment</th><th class="num">Amount</th>
            <th>Booking</th><th>Account</th><th>JE description</th><th></th>
          </tr></thead>
          <tbody>
            <template v-for="r in rows" :key="`${r.line_id}-${r.split_id}`">
              <tr :class="{ bad: r.problems.length, editing: open === r }">
                <td>{{ r.employee }}</td>
                <td>{{ r.line_date }}<template v-if="r.line_date_end"> – {{ r.line_date_end }}</template></td>
                <td>{{ r.deal_kind === 'operations' ? 'Operations' : r.deal_name }}
                  <span v-if="r.deal_kind === 'pipeline'" class="muted">(pipeline)</span></td>
                <td class="clip" :title="r.comment">{{ r.comment }}</td>
                <td class="num">{{ fmt(r.amount) }}</td>
                <td>{{ BOOK[r.booking] }}<span v-if="r.booking !== r.booking_proposed" class="chg" title="Changed by accounting">*</span>
                  <div v-if="r.booking === 'interco'" class="muted">
                    <span v-for="a in r.interco || []" :key="a.entity">{{ a.entity }} {{ Number(a.pct).toFixed(2) }}% </span>
                    <span v-if="r.interco_changed" class="chg">*</span></div></td>
                <td :title="r.employee_category">{{ r.booking === 'deal_cost' ? 'MR11000012' : r.expense_account }}
                  <span v-if="r.expense_account_changed" class="chg" :title="`Employee chose ${r.employee_account}`">*</span></td>
                <td class="clip wide" :title="r.description">{{ r.description }}
                  <span v-if="r.description !== r.description_proposed" class="chg">*</span></td>
                <td class="row-actions">
                  <button class="link" @click="startEdit(r)">Code</button>
                  <button v-if="r.receipt_id" class="link" @click="viewing = r">📎</button>
                </td>
              </tr>
              <tr v-if="r.problems.length"><td colspan="9" class="err-text">{{ r.problems.join('; ') }}</td></tr>
              <tr v-if="r.warnings?.length"><td colspan="9" class="warn-text">{{ r.warnings.join('; ') }}</td></tr>
              <tr v-if="open === r"><td colspan="9">
                <div class="code-form">
                  <label>Booking
                    <select v-model="draft.booking">
                      <option v-for="(v, k) in BOOK" :key="k" :value="k">{{ v }}</option>
                    </select></label>
                  <label v-if="draft.booking !== 'deal_cost'">Expense account
                    <input v-model="draft.expense_account" list="ec-accounts" class="acct" /></label>
                  <span v-if="draft.booking !== 'deal_cost'" class="muted">employee chose {{ r.employee_category || r.employee_account }}</span>
                  <label class="grow">JE description <input v-model="draft.description" /></label>
                  <div v-if="draft.booking === 'interco'" class="interco">
                    <div class="muted">{{ r.interco_proposed?.basis || r.interco_proposed?.error || 'Set the entities that own this expense.' }}</div>
                    <div v-for="(a, i) in draft.interco" :key="i" class="row">
                      <input v-model="a.entity" placeholder="entity" class="acct" />
                      <input v-model="a.pct" class="num-in" placeholder="%" />
                      <input v-model="a.rltd" placeholder="related entity" class="acct" />
                      <span class="muted" v-if="r.interco_proposed?.allocations?.[i]?.path">{{ r.interco_proposed.allocations[i].path }}</span>
                      <button class="link" @click="draft.interco.splice(i, 1)">remove</button>
                    </div>
                    <div class="row">
                      <button class="link" @click="draft.interco.push({ entity: '', pct: '', rltd: '' })">+ entity</button>
                      <span :class="Math.abs(pctTotal - 100) < 0.0001 ? 'muted' : 'warn-text'">total {{ pctTotal }}%</span>
                    </div>
                  </div>
                  <div class="row">
                    <button class="btn-primary" @click="saveRow">Save</button>
                    <button class="btn-secondary" @click="open = null">Cancel</button>
                    <button class="link" @click="resetRow(r)">Back to the proposal</button>
                  </div>
                </div>
              </td></tr>
            </template>
          </tbody>
        </table>
        <div v-if="viewing" class="viewer-col">
          <div class="row"><button class="link" @click="viewing = null">Close receipt</button></div>
          <ReceiptViewer :report-id="viewing.report_id" :receipt-id="viewing.receipt_id"
                         :page="viewing.receipt_page" :filename="`${viewing.employee} — ${viewing.comment}`"
                         :amount="viewing.amount" />
        </div>
        <datalist id="ec-accounts">
          <option v-for="c in categories" :key="c.account" :value="c.account">{{ c.name }}</option>
        </datalist>
      </div>
    </template>

    <!-- ============ batch ============ -->
    <template v-else-if="tab === 'batch'">
      <p class="muted">One batch per payroll date. The credit is one line to MR20000001, reimbursed
        through TriNet payroll. A batched report is locked; voiding the batch releases it.</p>
      <table class="data-table narrow">
        <thead><tr><th></th><th>Employee</th><th class="num">Lines</th><th class="num">Total</th><th>Problems</th><th></th></tr></thead>
        <tbody>
          <tr v-for="r in reportsOnGrid" :key="r.id">
            <td><input type="checkbox" v-model="pick[r.id]" /></td>
            <td>{{ r.employee }} <span class="muted">#{{ r.id }}</span></td>
            <td class="num">{{ r.lines }}</td><td class="num">{{ fmt(r.total) }}</td>
            <td :class="{ 'err-text': r.problems }">{{ r.problems || '' }}</td>
            <td>
              <button v-if="returning !== r.id" class="link" @click="returning = r.id; returnNote = ''">Return…</button>
              <span v-else class="row">
                <input v-model="returnNote" placeholder="why — required" class="note-in" />
                <button class="link" :disabled="!returnNote.trim()" @click="returnReport(r)">Return to {{ r.employee }}</button>
                <button class="link" @click="returning = null">cancel</button>
              </span>
            </td>
          </tr>
          <tr v-if="!reportsOnGrid.length"><td colspan="6" class="muted">No approved reports.</td></tr>
        </tbody>
      </table>
      <div class="row">
        <label>Payroll date <input type="date" v-model="payrollDate" /></label>
        <label class="grow">Credit description ends <input v-model="suffix" /></label>
        <label v-for="e in nonUsd" :key="e">{{ e }} USD→{{ currencies[e] }} rate
          <input v-model="fx[e]" class="num-in" placeholder="1.4134" /></label>
      </div>
      <div v-if="recurringItems.filter(r => r.active).length" class="row">
        <strong>Recurring:</strong>
        <label v-for="r in recurringItems.filter(r => r.active)" :key="r.id" class="check">
          <input type="checkbox" v-model="recPick[r.id]" /> {{ r.employee }} — {{ r.description }} {{ fmt(r.amount) }}</label>
      </div>
      <div class="row">
        <button class="btn-secondary" :disabled="busy || !payrollDate || (!pickedIds.length && !Object.values(recPick).some(Boolean))"
                @click="runPreview">Preview</button>
        <button class="btn-primary" :disabled="busy || !preview || preview.errors?.length" @click="generate">
          Generate &amp; download</button>
      </div>
      <template v-if="preview">
        <div v-if="preview.errors?.length" class="notice"><strong>Cannot generate:</strong>
          <ul><li v-for="m in preview.errors" :key="m">{{ m }}</li></ul></div>
        <div v-else class="muted">{{ preview.lines.length }} lines, {{ fmt(preview.total) }} reimbursed, period {{ preview.period }}.</div>
        <table v-if="preview.lines?.length" class="data-table">
          <thead><tr><th>Entity</th><th>Account</th><th class="num">Amount</th><th>Description</th><th>Related</th></tr></thead>
          <tbody><tr v-for="(l, i) in preview.lines" :key="i">
            <td>{{ l.entityid }}</td><td>{{ l.acctnum }}</td><td class="num">{{ fmt(l.amount) }}</td>
            <td class="clip wide" :title="l.descrpn">{{ l.descrpn }}</td><td>{{ l.rltdentity }}</td></tr></tbody>
        </table>
      </template>
    </template>

    <!-- ============ batches ============ -->
    <template v-else-if="tab === 'batches'">
      <table class="data-table">
        <thead><tr><th>Batch</th><th>Payroll date</th><th>Period</th><th class="num">Total</th><th>Status</th><th>By</th><th></th></tr></thead>
        <tbody>
          <tr v-for="b in batchList" :key="b.batch_id">
            <td>{{ b.batch_id }}</td><td>{{ b.payroll_date }}</td><td>{{ b.period }}</td>
            <td class="num">{{ fmt(b.total) }}</td>
            <td :class="{ 'warn-text': b.status.startsWith('generated') }">{{ b.status }}</td>
            <td>{{ b.created_by }}</td>
            <td class="row-actions">
              <button class="link" @click="downloadBatch(b)">Download</button>
              <template v-if="!b.voided_at">
                <button v-if="confirmVoid !== b.batch_id" class="link" @click="confirmVoid = b.batch_id">Void</button>
                <template v-else><span class="warn-text">Void it and release its reports?</span>
                  <button class="link" @click="voidBatch(b)">Yes</button>
                  <button class="link" @click="confirmVoid = null">No</button></template>
              </template>
            </td>
          </tr>
          <tr v-if="!batchList.length"><td colspan="7" class="muted">No batches yet.</td></tr>
        </tbody>
      </table>
      <p class="muted">"Posted" means MRI's GL now carries the batch's payroll credit. Until then, its reports
        stay batched, so they cannot be paid twice.</p>
    </template>

    <!-- ============ settings ============ -->
    <template v-else>
      <h4>Entities that book in another currency</h4>
      <p class="muted">USD unless listed. A batch asks for a rate for each listed entity it touches.</p>
      <table class="data-table narrow"><tbody>
        <tr v-for="(c, e) in currencies" :key="e"><td>{{ e }}</td><td>{{ c }}</td>
          <td><button class="link" @click="saveCurrency(String(e), 'USD')">back to USD</button></td></tr>
        <tr v-if="!Object.keys(currencies).length"><td colspan="3" class="muted">None.</td></tr>
      </tbody></table>
      <div class="row">
        <label>Entity <input v-model="curEntity" class="acct" placeholder="PPI2" /></label>
        <label>Currency <input v-model="curCode" class="acct" /></label>
        <button class="btn-secondary" :disabled="!curEntity" @click="saveCurrency(curEntity, curCode)">Set</button>
      </div>

      <h4>Recurring reimbursements</h4>
      <p class="muted">Standing items added to a batch when ticked (e.g. a monthly benefits reimbursement).</p>
      <table class="data-table narrow"><tbody>
        <tr v-for="r in recurringItems" :key="r.id" :class="{ muted: !r.active }">
          <td>{{ r.employee }}</td><td>{{ r.description }}</td><td>{{ r.account }}</td>
          <td class="num">{{ fmt(r.amount) }}</td>
          <td><button class="link" @click="toggleRecurring(r)">{{ r.active ? 'Retire' : 'Reinstate' }}</button></td></tr>
        <tr v-if="!recurringItems.length"><td colspan="5" class="muted">None.</td></tr>
      </tbody></table>
      <div class="row">
        <label>Employee <select v-model="rec.user_id">
          <option value="" disabled>choose…</option>
          <option v-for="e in employees" :key="e.user_id" :value="e.user_id">{{ e.full_name || e.username }}</option>
        </select></label>
        <label class="grow">Description <input v-model="rec.description" placeholder="Benefits Reimbursement" /></label>
        <label>Account <input v-model="rec.account" class="acct" placeholder="MR51000005" /></label>
        <label>Amount <input v-model="rec.amount" class="num-in" /></label>
        <button class="btn-secondary" @click="addRecurring">Add</button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.ec { padding: 20px; }
.ec h2 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; }
.tabs { display: flex; gap: 6px; margin: 14px 0 12px; }
.tabs button { padding: 6px 14px; border: 1px solid var(--color-border); background: none; border-radius: 6px; cursor: pointer; font-size: 13px; }
.tabs button.active { background: var(--color-primary, #2f6f4f); color: #fff; border-color: var(--color-primary, #2f6f4f); }
.muted { color: var(--color-text-secondary); font-size: 12px; }
.warn-text { color: #a5560b; font-size: 12.5px; }
.err-text { color: #8a1f1f; font-size: 11.5px; }
.chg { color: #a5560b; font-weight: 700; margin-left: 2px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 6px 0 10px; }
.data-table.narrow { width: auto; min-width: 520px; }
.data-table th { text-align: left; padding: 6px 8px; background: var(--color-surface); border-bottom: 2px solid var(--color-border); font-size: 11px; text-transform: uppercase; color: var(--color-text-secondary); white-space: nowrap; }
.data-table td { padding: 5px 8px; border-bottom: 1px solid var(--color-border); vertical-align: top; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tr.bad td { background: #fff7f2; }
.data-table tr.editing td { background: var(--color-surface); }
.clip { max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.clip.wide { max-width: 340px; }
.row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; margin: 8px 0; }
label { display: flex; flex-direction: column; font-size: 11px; color: var(--color-text-secondary); gap: 3px; }
label.check { flex-direction: row; align-items: center; gap: 5px; font-size: 12.5px; }
label.grow { flex: 1; min-width: 260px; }
input, select { border: 1px solid var(--color-border); border-radius: 4px; padding: 4px 6px; font-size: 12.5px; background: var(--color-surface); color: var(--color-text); }
.num-in { width: 90px; text-align: right; }
.acct { width: 120px; }
.note-in { width: 240px; }
.code-form { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; padding: 6px 0; }
.interco { width: 100%; border-top: 1px dashed var(--color-border); padding-top: 6px; }
.link { background: none; border: none; color: var(--color-primary, #2f6f4f); cursor: pointer; padding: 0 4px; font-size: 12px; }
.row-actions { white-space: nowrap; }
.split { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(320px, 1fr); gap: 16px; }
.viewer-col { display: flex; flex-direction: column; height: 700px; position: sticky; top: 10px; }
.viewer-col > :last-child { flex: 1; }
.notice { margin: 8px 0; padding: 8px 12px; font-size: 12.5px; border: 1px solid #e6c9a8; background: #fdf6ee; border-radius: 6px; }
.notice ul { margin: 4px 0 0 18px; padding: 0; }
</style>
