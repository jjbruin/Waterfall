<script setup lang="ts">
/**
 * Expenses — employee expense reports, phase 1.
 *
 * Open a report for a period, enter its lines, submit it to your approver; the
 * approver approves it or returns it with a note. Accounting's coding and the
 * MRI upload are phase 3, and receipts are read from their images in phase 2.
 * Design: .claude/memory/expense_reporting.md.
 *
 * WHO SEES WHICH REPORT IS DECIDED BY THE SERVER, per record. This screen only
 * shows what the API returned and the buttons its `permissions` allow; it
 * never infers a permission of its own.
 */
import { ref, computed, onMounted, watch } from 'vue'
import api from '@/api/client'
import { useAuthStore } from '@/stores/auth'
import { useDataStore } from '@/stores/data'

const auth = useAuthStore()
const dataStore = useDataStore()

type Tab = 'mine' | 'to_approve' | 'all' | 'setup'
const tab = ref<Tab>('mine')
const canSeeSetup = computed(() => auth.canEditAccounting)   // admin + accounting roles
const isAdminRole = computed(() => auth.isAdmin)

const options = ref<any>({ categories: [], purposes: [], deals: [], mileage_rates: [] })
const me = ref<any>(null)
const reports = ref<any[]>([])
const toApproveCount = ref(0)
const report = ref<any>(null)
const loading = ref(false)
const error = ref<string | null>(null)

const fmt = (v: any) => v == null ? '' :
  Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const STATUS: Record<string, string> = {
  draft: 'Draft', submitted: 'Submitted', returned: 'Returned', approved: 'Approved',
}
const dealName = (code: string) =>
  (options.value.deals.find((d: any) => d.code === code) || {}).name || code

function fail(e: any, what: string) {
  const msg = e?.response?.data?.error || e?.message || String(e)
  dataStore.addToast(`${what}: ${msg}`, 'error')
}

async function loadOptions() {
  const [o, m] = await Promise.all([api.get('/api/expenses/options'), api.get('/api/expenses/me')])
  options.value = o.data
  me.value = m.data
}

async function loadList() {
  if (tab.value === 'setup') return
  loading.value = true
  error.value = null
  try {
    const r = await api.get('/api/expenses/reports', { params: { scope: tab.value } })
    reports.value = r.data.reports
    const t = tab.value === 'to_approve' ? r :
      await api.get('/api/expenses/reports', { params: { scope: 'to_approve' } })
    toApproveCount.value = t.data.reports.length
  } catch (e: any) {
    error.value = e?.response?.data?.error || String(e)
  } finally {
    loading.value = false
  }
}

async function openReport(id: number) {
  try {
    report.value = (await api.get(`/api/expenses/reports/${id}`)).data
    editing.value = null
  } catch (e) { fail(e, 'Could not open the report') }
}

function closeReport() { report.value = null; editing.value = null; loadList() }

// ---- a new report ----
const newStart = ref('')
const newEnd = ref('')
const newTitle = ref('')
async function createReport() {
  try {
    const r = await api.post('/api/expenses/reports',
      { period_start: newStart.value, period_end: newEnd.value, title: newTitle.value })
    newStart.value = newEnd.value = newTitle.value = ''
    report.value = r.data
    loadList()
  } catch (e) { fail(e, 'Could not open a report') }
}

async function saveHeader() {
  try {
    report.value = (await api.put(`/api/expenses/reports/${report.value.id}`, {
      period_start: report.value.period_start, period_end: report.value.period_end,
      title: report.value.title,
    })).data
  } catch (e) { fail(e, 'Could not save the period') }
}

const confirmDelete = ref(false)
async function deleteReport() {
  try {
    await api.delete(`/api/expenses/reports/${report.value.id}`)
    confirmDelete.value = false
    closeReport()
  } catch (e) { fail(e, 'Could not delete the report') }
}

// ---- a line ----
const blankLine = () => ({
  id: null as number | null, line_date: '', line_date_end: '', category_account: '',
  purpose: '', deal_code: 'OPERATIONS', vendor: '', comment: '', amount: '',
  miles: '', receipt: 'Y', no_receipt_reason: '',
  isPeriod: false, isMileage: false, isSplit: false,
  splits: [] as { deal_code: string; amount: string; pct: string }[],
})
const editing = ref<any>(null)

function newLine() { editing.value = blankLine() }
function editLine(ln: any) {
  editing.value = {
    ...blankLine(), ...ln,
    line_date_end: ln.line_date_end || '', vendor: ln.vendor || '',
    comment: ln.comment || '', no_receipt_reason: ln.no_receipt_reason || '',
    amount: ln.amount ?? '', miles: ln.miles ?? '', deal_code: ln.deal_code || '',
    isPeriod: !!ln.line_date_end, isMileage: ln.miles != null,
    isSplit: (ln.splits || []).length > 0,
    splits: (ln.splits || []).map((s: any) => ({ deal_code: s.deal_code, amount: String(s.amount), pct: '' })),
  }
}

// The rate in force on the line's date, as the server will apply it.
const rateForLine = computed(() => {
  const d = editing.value?.line_date
  if (!d) return null
  return (options.value.mileage_rates || []).find((r: any) => r.effective_date <= d) || null
})
const mileageAmount = computed(() => {
  const r = rateForLine.value, m = parseFloat(editing.value?.miles)
  return r && m > 0 ? Math.round(m * r.rate * 100) / 100 : null
})
const lineAmount = computed(() =>
  editing.value?.isMileage ? mileageAmount.value : parseFloat(editing.value?.amount))
const splitTotal = computed(() => Math.round(
  (editing.value?.splits || []).reduce((t: number, s: any) => t + (parseFloat(s.amount) || 0), 0) * 100) / 100)

function startSplit() {
  const e = editing.value
  e.isSplit = true
  if (!e.splits.length) {
    e.splits = [{ deal_code: e.deal_code === 'OPERATIONS' ? '' : e.deal_code, amount: '', pct: '' },
                { deal_code: '', amount: '', pct: '' }]
  }
}
// Percentages are a typing aid: they become amounts here, rounded ONCE, with
// the last row taking the remainder so the split foots to the cent.
function applyPercents() {
  const total = lineAmount.value
  const rows = editing.value.splits
  if (!(total > 0)) return
  let used = 0
  rows.forEach((s: any, i: number) => {
    if (i === rows.length - 1) { s.amount = (Math.round((total - used) * 100) / 100).toFixed(2); return }
    const a = Math.round(total * (parseFloat(s.pct) || 0)) / 100
    s.amount = a.toFixed(2); used += a
  })
}

async function saveLine() {
  const e = editing.value
  const body: any = {
    line_date: e.line_date, line_date_end: e.isPeriod ? e.line_date_end : '',
    category_account: e.category_account, purpose: e.purpose,
    deal_code: e.isSplit ? '' : e.deal_code, vendor: e.vendor, comment: e.comment,
    amount: e.isMileage ? '' : e.amount, miles: e.isMileage ? e.miles : '',
    receipt: e.receipt, no_receipt_reason: e.receipt === 'N' ? e.no_receipt_reason : '',
    splits: e.isSplit ? e.splits.map((s: any) => ({ deal_code: s.deal_code, amount: s.amount })) : [],
  }
  try {
    const url = `/api/expenses/reports/${report.value.id}/lines`
    report.value = (e.id ? await api.put(`${url}/${e.id}`, body) : await api.post(url, body)).data
    editing.value = null
  } catch (err) { fail(err, 'Could not save the line') }
}

async function deleteLine(ln: any) {
  try {
    report.value = (await api.delete(`/api/expenses/reports/${report.value.id}/lines/${ln.id}`)).data
  } catch (e) { fail(e, 'Could not remove the line') }
}

// ---- workflow ----
async function act(path: string, body: any = {}, what = 'Could not do that') {
  try {
    report.value = (await api.post(`/api/expenses/reports/${report.value.id}/${path}`, body)).data
    loadList()
  } catch (e) { fail(e, what) }
}
const decisionNote = ref('')
async function decide(action: 'approve' | 'return') {
  await act('decide', { action, note: decisionNote.value },
    action === 'approve' ? 'Could not approve' : 'Could not return')
  decisionNote.value = ''
}

// ---- setup ----
const employees = ref<any[]>([])
async function loadEmployees() {
  try { employees.value = (await api.get('/api/expenses/employees')).data.employees }
  catch (e) { fail(e, 'Could not load employees') }
}
async function saveEmployee(e: any) {
  try {
    const r = await api.put(`/api/expenses/employees/${e.user_id}`,
      { full_name: e.full_name, approver_user_id: e.approver_user_id || null })
    Object.assign(e, r.data)
    await loadEmployees()          // one save can change another employee's route
  } catch (err) { fail(err, 'Could not save'); loadEmployees() }
}
const rateDate = ref('')
const rateValue = ref('')
const rateBasis = ref('')
async function saveRate() {
  try {
    options.value.mileage_rates = (await api.put('/api/expenses/mileage-rates', {
      effective_date: rateDate.value, rate: rateValue.value, basis: rateBasis.value,
    })).data
    rateDate.value = rateValue.value = rateBasis.value = ''
  } catch (e) { fail(e, 'Could not save the rate') }
}

watch(tab, t => { report.value = null; if (t === 'setup') loadEmployees(); else loadList() })
onMounted(async () => {
  try { await loadOptions() } catch (e) { fail(e, 'Could not load the expense form') }
  loadList()
})
</script>

<template>
  <div class="exp">
    <div class="header-row">
      <div>
        <h2>Expenses</h2>
        <div class="subtitle">
          Open a report for a period, enter each expense, and submit it to your approver.
          <template v-if="me?.route">
            <span v-if="me.route.error" class="warn-text"> {{ me.route.error }}</span>
            <span v-else> Your reports go to <strong>{{ me.route.label }}</strong>.</span>
          </template>
        </div>
      </div>
    </div>

    <div class="tabs">
      <button :class="{ active: tab === 'mine' }" @click="tab = 'mine'">My reports</button>
      <button :class="{ active: tab === 'to_approve' }" @click="tab = 'to_approve'">
        To approve<span v-if="toApproveCount" class="badge">{{ toApproveCount }}</span></button>
      <button :class="{ active: tab === 'all' }" @click="tab = 'all'">Tracking</button>
      <button v-if="canSeeSetup" :class="{ active: tab === 'setup' }" @click="tab = 'setup'">
        Employees &amp; approvers</button>
    </div>

    <div v-if="error" class="error-banner">{{ error }}</div>

    <!-- ================= one report ================= -->
    <div v-if="report" class="report">
      <div class="report-head">
        <button class="btn-secondary" @click="closeReport">&larr; Back</button>
        <h3>{{ report.employee }} — {{ report.title || 'Expense report' }}</h3>
        <span class="status" :class="report.status">{{ STATUS[report.status] }}</span>
        <span v-if="report.waiting_on" class="muted">waiting on {{ report.waiting_on }}</span>
        <span v-if="report.decided_basis" class="muted">
          {{ report.status === 'approved' ? 'approved' : 'returned' }} by {{ report.decided_by }}
          ({{ report.decided_basis }})</span>
      </div>

      <div class="period">
        <label>Period
          <input type="date" v-model="report.period_start" :disabled="!report.permissions.edit" />
          to
          <input type="date" v-model="report.period_end" :disabled="!report.permissions.edit" />
        </label>
        <label>Title
          <input v-model="report.title" placeholder="e.g. September 2026"
                 :disabled="!report.permissions.edit" />
        </label>
        <button v-if="report.permissions.edit" class="btn-secondary" @click="saveHeader">Save</button>
      </div>

      <table class="data-table">
        <thead><tr>
          <th>Date / period</th><th>Category</th><th>Purpose</th><th>Deal</th>
          <th>Vendor</th><th>Comment</th><th class="num">Amount</th><th>Receipt</th><th></th>
        </tr></thead>
        <tbody>
          <tr v-for="ln in report.lines" :key="ln.id"
              :class="{ bad: report.check.by_line[ln.id]?.errors.length }">
            <td>{{ ln.line_date }}<template v-if="ln.line_date_end"> – {{ ln.line_date_end }}</template></td>
            <td :title="ln.category_account">{{ ln.category_name || ln.category_account }}</td>
            <td>{{ ln.purpose }}</td>
            <td>
              <template v-if="ln.splits.length">
                <div v-for="s in ln.splits" :key="s.id" class="split-cell">
                  {{ s.deal_name }} <span class="muted">{{ fmt(s.amount) }}</span></div>
              </template>
              <template v-else>{{ ln.deal_name }}</template>
            </td>
            <td>{{ ln.vendor }}</td>
            <td class="comment" :title="ln.comment">{{ ln.comment }}
              <div v-if="ln.miles != null" class="muted">{{ ln.miles }} mi × ${{ ln.mileage_rate }}</div>
              <div v-for="m in report.check.by_line[ln.id]?.errors" :key="m" class="err-text">Line {{ m }}</div>
            </td>
            <td class="num">{{ fmt(ln.amount) }}</td>
            <td>{{ ln.receipt === 'N' ? 'No — ' + (ln.no_receipt_reason || '') : ln.receipt === 'Y' ? 'Yes' : '' }}</td>
            <td class="row-actions">
              <template v-if="report.permissions.edit">
                <button class="link" @click="editLine(ln)">Edit</button>
                <button class="link" @click="deleteLine(ln)">Remove</button>
              </template>
            </td>
          </tr>
          <tr v-if="!report.lines.length"><td colspan="9" class="muted">No lines yet.</td></tr>
        </tbody>
        <tfoot><tr>
          <td colspan="6" class="num"><strong>Total</strong></td>
          <td class="num"><strong>{{ fmt(report.total) }}</strong></td><td colspan="2"></td>
        </tr></tfoot>
      </table>

      <button v-if="report.permissions.edit && !editing" class="btn-secondary add" @click="newLine">
        + Add an expense</button>

      <!-- ---------- line form ---------- -->
      <div v-if="editing" class="line-form">
        <div class="row">
          <label>Date <input type="date" v-model="editing.line_date" /></label>
          <label class="check"><input type="checkbox" v-model="editing.isPeriod" /> a period</label>
          <label v-if="editing.isPeriod">to <input type="date" v-model="editing.line_date_end" /></label>
          <label>Category
            <select v-model="editing.category_account">
              <option value="" disabled>choose…</option>
              <option v-for="c in options.categories" :key="c.account" :value="c.account">{{ c.name }}</option>
            </select>
          </label>
          <label>Purpose
            <select v-model="editing.purpose">
              <option value="" disabled>choose…</option>
              <option v-for="p in options.purposes" :key="p" :value="p">{{ p }}</option>
            </select>
          </label>
        </div>
        <div class="row">
          <label v-if="!editing.isSplit">Deal
            <select v-model="editing.deal_code">
              <option v-for="d in options.deals" :key="d.code" :value="d.code">{{ d.name }}</option>
            </select>
          </label>
          <button v-if="!editing.isSplit" class="link" @click="startSplit">Split across deals…</button>
          <label>Vendor <input v-model="editing.vendor" placeholder="if applicable" /></label>
          <label class="grow">Comment <input v-model="editing.comment"
                 placeholder="e.g. Market Poplar Site Visit - Airport Parking" /></label>
        </div>
        <div class="row">
          <label class="check"><input type="checkbox" v-model="editing.isMileage" /> mileage</label>
          <template v-if="editing.isMileage">
            <label>Miles <input type="number" step="0.1" v-model="editing.miles" class="num-in" /></label>
            <span v-if="rateForLine" class="muted">× ${{ rateForLine.rate }} (from {{ rateForLine.effective_date }})
              = <strong>{{ fmt(mileageAmount) }}</strong>. Tolls go on their own line.</span>
            <span v-else class="warn-text">No mileage rate is in force on that date — accounting sets it.</span>
          </template>
          <label v-else>Amount <input v-model="editing.amount" class="num-in" placeholder="0.00" /></label>
          <label>Receipt submitted?
            <select v-model="editing.receipt"><option value="Y">Yes</option><option value="N">No</option></select>
          </label>
          <label v-if="editing.receipt === 'N'" class="grow">If no receipt, why
            <input v-model="editing.no_receipt_reason" /></label>
        </div>
        <div v-if="editing.isSplit" class="splits">
          <div class="muted">Split this expense across deals. Enter amounts, or percentages and
            <button class="link" @click="applyPercents">convert to amounts</button>.</div>
          <div v-for="(s, i) in editing.splits" :key="i" class="row">
            <select v-model="s.deal_code">
              <option value="" disabled>deal…</option>
              <option v-for="d in options.deals" :key="d.code" :value="d.code">{{ d.name }}</option>
            </select>
            <input v-model="s.pct" class="num-in" placeholder="%" />
            <input v-model="s.amount" class="num-in" placeholder="amount" />
            <button class="link" @click="editing.splits.splice(i, 1)">remove</button>
          </div>
          <div class="row">
            <button class="link" @click="editing.splits.push({ deal_code: '', amount: '', pct: '' })">+ another deal</button>
            <span :class="Math.abs(splitTotal - (lineAmount || 0)) < 0.005 ? 'muted' : 'warn-text'">
              split {{ fmt(splitTotal) }} of {{ fmt(lineAmount) }}</span>
            <button class="link" @click="editing.isSplit = false; editing.splits = []">no split</button>
          </div>
        </div>
        <div class="row">
          <button class="btn-primary" @click="saveLine">Save line</button>
          <button class="btn-secondary" @click="editing = null">Cancel</button>
        </div>
      </div>

      <!-- ---------- what stops a submit ---------- -->
      <div v-if="report.permissions.submit && report.check.errors.length" class="notice">
        <strong>Before it can be submitted:</strong>
        <ul><li v-for="m in report.check.errors" :key="m">{{ m }}</li></ul>
      </div>
      <div v-if="report.check.warnings.length" class="notice soft">
        <ul><li v-for="m in report.check.warnings" :key="m">{{ m }}</li></ul>
      </div>

      <div class="actions">
        <template v-if="report.permissions.submit">
          <span v-if="report.route?.error" class="warn-text">{{ report.route.error }}</span>
          <button class="btn-primary"
                  :disabled="!!report.check.errors.length || !!report.route?.error"
                  @click="act('submit', {}, 'Could not submit')">
            Submit to {{ report.route?.label || 'approver' }}</button>
        </template>
        <button v-if="report.permissions.recall" class="btn-secondary"
                @click="act('recall', {}, 'Could not recall')">Recall</button>
        <template v-if="report.permissions.delete">
          <button v-if="!confirmDelete" class="btn-secondary" @click="confirmDelete = true">Delete draft</button>
          <template v-else>
            <span class="warn-text">Delete this draft and its {{ report.lines.length }} lines?</span>
            <button class="btn-danger" @click="deleteReport">Yes, delete</button>
            <button class="btn-secondary" @click="confirmDelete = false">No</button>
          </template>
        </template>
      </div>

      <div v-if="report.permissions.decide" class="decision">
        <h4>Your decision <span class="muted">(as {{ report.permissions.decide_as }})</span></h4>
        <textarea v-model="decisionNote" rows="2"
                  placeholder="A note — required to return the report"></textarea>
        <div class="row">
          <button class="btn-primary" @click="decide('approve')">Approve</button>
          <button class="btn-secondary" :disabled="!decisionNote.trim()" @click="decide('return')">
            Return to {{ report.employee }}</button>
        </div>
      </div>

      <h4>History</h4>
      <ul class="history">
        <li v-for="e in report.events" :key="e.id">
          <span class="muted">{{ e.at?.replace('T', ' ').slice(0, 16) }}</span>
          {{ e.action }} by {{ e.actor_name }}<template v-if="e.basis"> ({{ e.basis }})</template>
          <div v-if="e.note" class="note">“{{ e.note }}”</div>
        </li>
      </ul>
    </div>

    <!-- ================= lists ================= -->
    <template v-else-if="tab !== 'setup'">
      <div v-if="tab === 'mine'" class="new-report">
        <label>New report for <input type="date" v-model="newStart" /></label>
        <label>to <input type="date" v-model="newEnd" /></label>
        <label><input v-model="newTitle" placeholder="title, e.g. September 2026" /></label>
        <button class="btn-primary" :disabled="!newStart || !newEnd" @click="createReport">Open report</button>
      </div>
      <div v-if="loading" class="muted">Loading…</div>
      <table v-else class="data-table">
        <thead><tr>
          <th v-if="tab !== 'mine'">Employee</th><th>Period</th><th>Title</th><th>Status</th>
          <th>Waiting on / decided</th><th class="num">Lines</th><th class="num">Total</th><th>Submitted</th>
        </tr></thead>
        <tbody>
          <tr v-for="r in reports" :key="r.id" class="clickable" @click="openReport(r.id)">
            <td v-if="tab !== 'mine'">{{ r.employee }}</td>
            <td>{{ r.period_start }} – {{ r.period_end }}</td>
            <td>{{ r.title }}</td>
            <td><span class="status" :class="r.status">{{ STATUS[r.status] }}</span></td>
            <td>{{ r.waiting_on || (r.decided_by ? `${r.decided_by} (${r.decided_basis})` : '') }}</td>
            <td class="num">{{ r.line_count }}</td>
            <td class="num">{{ fmt(r.total) }}</td>
            <td>{{ r.submitted_at?.slice(0, 10) }}</td>
          </tr>
          <tr v-if="!reports.length"><td colspan="8" class="muted">
            {{ tab === 'to_approve' ? 'Nothing is waiting for you.' : 'No reports.' }}</td></tr>
        </tbody>
      </table>
    </template>

    <!-- ================= setup ================= -->
    <template v-else>
      <p class="muted">
        The name an employee carries on their reports and the journal entry, and who approves
        them. An approver's own report goes to the CFO, and the CFO's to the CEO or President;
        the CEO or President may approve any report when its approver is out.
        <template v-if="!isAdminRole"> Only the admin changes these.</template>
      </p>
      <table class="data-table">
        <thead><tr><th>User</th><th>Role</th><th>Name on reports</th><th>Approver</th><th>Reports go to</th></tr></thead>
        <tbody>
          <tr v-for="e in employees" :key="e.user_id">
            <td>{{ e.username }}</td>
            <td>{{ e.role }}</td>
            <td><input v-model="e.full_name" :disabled="!isAdminRole" @change="saveEmployee(e)" /></td>
            <td>
              <select v-model="e.approver_user_id" :disabled="!isAdminRole" @change="saveEmployee(e)">
                <option :value="null">—</option>
                <option v-for="a in employees.filter(x => x.user_id !== e.user_id)" :key="a.user_id"
                        :value="a.user_id">{{ a.full_name || a.username }}</option>
              </select>
            </td>
            <td :class="{ 'warn-text': e.route.error }">{{ e.route.error || e.route.label }}</td>
          </tr>
        </tbody>
      </table>

      <h4>Mileage rate</h4>
      <p class="muted">Mileage is miles × the rate in force on the expense's date, computed by the
        app and never typed. Accounting sets the rate.</p>
      <table class="data-table narrow">
        <thead><tr><th>Effective</th><th class="num">Rate / mile</th><th>Basis</th><th>Set by</th></tr></thead>
        <tbody>
          <tr v-for="r in options.mileage_rates" :key="r.effective_date">
            <td>{{ r.effective_date }}</td><td class="num">{{ r.rate }}</td>
            <td>{{ r.basis }}</td><td>{{ r.set_by }}</td></tr>
          <tr v-if="!options.mileage_rates.length"><td colspan="4" class="warn-text">
            No rate set — mileage lines cannot be entered until one is.</td></tr>
        </tbody>
      </table>
      <div v-if="auth.canEditAccounting" class="row">
        <label>Effective <input type="date" v-model="rateDate" /></label>
        <label>Rate <input v-model="rateValue" class="num-in" placeholder="0.725" /></label>
        <label class="grow">Basis <input v-model="rateBasis" placeholder="e.g. IRS standard rate 2026" /></label>
        <button class="btn-primary" :disabled="!rateDate || !rateValue" @click="saveRate">Set rate</button>
      </div>
    </template>
  </div>
</template>

<style scoped>
.exp { padding: 20px; }
.header-row h2 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; }
.tabs { display: flex; gap: 6px; margin: 14px 0 12px; }
.tabs button {
  padding: 6px 14px; border: 1px solid var(--color-border); background: none;
  border-radius: 6px; cursor: pointer; font-size: 13px;
}
.tabs button.active {
  background: var(--color-primary, #2f6f4f); color: #fff; border-color: var(--color-primary, #2f6f4f);
}
.badge { margin-left: 6px; background: #c0392b; color: #fff; border-radius: 9px; padding: 0 6px; font-size: 11px; }
.error-banner { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: #fdeaea; color: #8a1f1f; font-size: 13px; }
.muted { color: var(--color-text-secondary); font-size: 12px; }
.warn-text { color: #a5560b; font-size: 12.5px; }
.err-text { color: #8a1f1f; font-size: 11.5px; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 6px 0 10px; }
.data-table.narrow { width: auto; min-width: 520px; }
.data-table th {
  text-align: left; padding: 6px 8px; background: var(--color-surface);
  border-bottom: 2px solid var(--color-border); font-size: 11px; text-transform: uppercase;
  color: var(--color-text-secondary); white-space: nowrap;
}
.data-table td { padding: 5px 8px; border-bottom: 1px solid var(--color-border); vertical-align: top; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tr.bad td { background: #fff7f2; }
.data-table td.comment { max-width: 320px; }
.clickable { cursor: pointer; }
.clickable:hover td { background: var(--color-surface); }
.status { padding: 1px 8px; border-radius: 9px; font-size: 11.5px; background: #eef0f3; }
.status.submitted { background: #e7f0fb; color: #1f4f8a; }
.status.returned { background: #fdf0e3; color: #8a4f1f; }
.status.approved { background: #e5f4ea; color: #1f6b3a; }
.report-head { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.report-head h3 { margin: 0; }
.period, .new-report, .row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; margin: 10px 0; }
label { display: flex; flex-direction: column; font-size: 11px; color: var(--color-text-secondary); gap: 3px; }
label.check { flex-direction: row; align-items: center; gap: 5px; font-size: 12.5px; }
label.grow { flex: 1; min-width: 240px; }
input, select, textarea {
  border: 1px solid var(--color-border); border-radius: 4px; padding: 4px 6px;
  font-size: 12.5px; background: var(--color-surface); color: var(--color-text);
}
.num-in { width: 90px; text-align: right; }
.line-form { border: 1px solid var(--color-border); border-radius: 6px; padding: 8px 12px; margin: 10px 0; }
.splits { border-top: 1px dashed var(--color-border); padding-top: 6px; }
.split-cell { white-space: nowrap; }
.link { background: none; border: none; color: var(--color-primary, #2f6f4f); cursor: pointer; padding: 0 4px; font-size: 12px; }
.row-actions { white-space: nowrap; }
.add { margin: 4px 0 10px; }
.notice { margin: 8px 0; padding: 8px 12px; font-size: 12.5px; border: 1px solid #e6c9a8; background: #fdf6ee; border-radius: 6px; }
.notice.soft { border-color: var(--color-border); background: var(--color-surface); }
.notice ul { margin: 4px 0 0 18px; padding: 0; }
.actions { display: flex; gap: 10px; align-items: center; margin: 12px 0; flex-wrap: wrap; }
.decision { border: 1px solid var(--color-border); border-radius: 6px; padding: 8px 12px; margin: 12px 0; }
.decision textarea { width: 100%; }
.history { list-style: none; padding: 0; font-size: 12.5px; }
.history li { padding: 3px 0; }
.note { margin-left: 18px; font-style: italic; }
.btn-danger { background: #b03a2e; color: #fff; border: none; border-radius: 4px; padding: 5px 12px; cursor: pointer; }
</style>
