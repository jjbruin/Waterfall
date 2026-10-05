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
import ReceiptViewer from '@/components/expenses/ReceiptViewer.vue'

const auth = useAuthStore()
const dataStore = useDataStore()

type Tab = 'mine' | 'to_approve' | 'all' | 'setup'
const tab = ref<Tab>('mine')
const canSeeSetup = computed(() => auth.canEditAccounting)   // admin + accounting roles
// The `admin` USERNAME, not the role: the server says so (Jim, Oct 2 2026).
const isAdminRole = computed(() => auth.canAssignSections)

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
  batched: 'Batched for payroll',
}
// A pipeline deal is not on any list: the employee types its name (Jim, Oct 2
// 2026). PIPELINE is what the form SENDS; the server stores the typed name.
const PIPELINE = 'PIPELINE'
const dealLabel = (d: any) => d.label || d.name
const dealOf = (x: any) => x.deal_kind === 'pipeline'
  ? `${x.deal_name} (pipeline)` : x.deal_name

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
  purpose: '', deal_code: 'OPERATIONS', deal_name: '', vendor: '', comment: '', amount: '',
  miles: '', receipt: '', no_receipt_reason: '',
  receiptChoice: '' as string | number, receipt_page: 1 as number | null, extracted: null as any,
  isPeriod: false, isMileage: false, isSplit: false, recurring: false,
  splits: [] as { deal_code: string; deal_name: string; amount: string; pct: string }[],
})
const editing = ref<any>(null)

function newLine() { wiz.value = null; editing.value = blankLine() }
function editLine(ln: any) {
  wiz.value = null          // a route measured for another line must not follow this one
  editing.value = {
    ...blankLine(), ...ln,
    line_date_end: ln.line_date_end || '', vendor: ln.vendor || '',
    comment: ln.comment || '', no_receipt_reason: ln.no_receipt_reason || '',
    amount: ln.amount ?? '', miles: ln.miles ?? '',
    deal_code: ln.deal_kind === 'pipeline' ? PIPELINE : (ln.deal_code || ''),
    deal_name: ln.deal_kind === 'pipeline' ? (ln.deal_name || '') : '',
    receiptChoice: ln.receipt_id ? ln.receipt_id : (ln.receipt === 'N' ? 'N' : ''),
    receipt_page: ln.receipt_page || 1, extracted: ln.extracted || null,
    // the miles a stored measurement gave, so editing them away drops the route
    route_miles: ln.route_id ? String(ln.miles) : undefined,
    isPeriod: !!ln.line_date_end, isMileage: ln.miles != null, recurring: !!ln.recurring,
    isSplit: (ln.splits || []).length > 0,
    splits: (ln.splits || []).map((s: any) => ({
      deal_code: s.deal_kind === 'pipeline' ? PIPELINE : s.deal_code,
      deal_name: s.deal_kind === 'pipeline' ? (s.deal_name || '') : '',
      amount: String(s.amount), pct: '' })),
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
    e.splits = [{ deal_code: e.deal_code === 'OPERATIONS' ? '' : e.deal_code,
                  deal_name: e.deal_name, amount: '', pct: '' },
                { deal_code: '', deal_name: '', amount: '', pct: '' }]
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
  // SAVE DOES THE WORK (Jim, Oct 5 2026: an employee pressed Save without
  // "Use miles" and had to reopen the line). A route typed into the wizard but
  // never measured is measured now; a failed measurement stops the save and
  // says why rather than saving the line without miles.
  if (e.isMileage && wiz.value && !wiz.value.result
      && wiz.value.stops.filter((x: string) => x.trim()).length >= 2) {
    await measureRoute()
    if (!wiz.value?.result?.miles) return
  }
  // Miles typed over a measurement are the employee's figure, not Google's:
  // the line keeps the number and drops the measured route it no longer matches.
  if (e.route_id && e.route_miles !== undefined && String(e.route_miles) !== String(e.miles)) {
    e.route_id = ''
    e.route = null
  }
  const body: any = {
    line_date: e.line_date, line_date_end: e.isPeriod ? e.line_date_end : '',
    category_account: e.category_account, purpose: e.purpose,
    deal_code: e.isSplit ? '' : e.deal_code,
    deal_name: !e.isSplit && e.deal_code === PIPELINE ? e.deal_name : '',
    vendor: e.vendor, comment: e.comment, recurring: e.recurring,
    amount: e.isMileage ? '' : e.amount, miles: e.isMileage ? e.miles : '',
    receipt: e.receiptChoice === 'N' ? 'N' : '',
    no_receipt_reason: e.receiptChoice === 'N' ? e.no_receipt_reason : '',
    receipt_id: typeof e.receiptChoice === 'number' ? e.receiptChoice : '',
    receipt_page: typeof e.receiptChoice === 'number' ? e.receipt_page : '',
    route_id: e.isMileage ? (e.route_id || '') : '',
    splits: e.isSplit ? e.splits.map((s: any) => ({
      deal_code: s.deal_code, amount: s.amount,
      deal_name: s.deal_code === PIPELINE ? s.deal_name : '' })) : [],
  }
  try {
    const url = `/api/expenses/reports/${report.value.id}/lines`
    report.value = (e.id ? await api.put(`${url}/${e.id}`, body) : await api.post(url, body)).data
    editing.value = null
  } catch (err) { fail(err, 'Could not save the line') }
}

// ---- the distance wizard (Jim, Oct 2 2026) ----
// Google measures the drive on the SERVER, which keeps the measurement; the
// line points at it, so "measured" means Google measured it. A three-letter
// code is asked as an airport ("PHL" alone geocodes to the Philippines).
const wiz = ref<any>(null)
function openWizard() {
  wiz.value = { stops: ['', ''], round_trip: false, busy: false, result: null, error: '' }
}
async function measureRoute() {
  const w = wiz.value
  w.busy = true; w.error = ''; w.result = null
  try {
    const r = (await api.post('/api/expenses/distance',
      { stops: w.stops, round_trip: w.round_trip })).data
    w.result = r
    if (r.error) w.error = r.error
    // Measuring FILLS the miles: no second button to forget.
    else if (r.miles) applyRoute()
  } catch (e: any) { w.error = e?.response?.data?.error || String(e) }
  finally { w.busy = false }
}
function applyRoute() {
  const e = editing.value, r = wiz.value.result
  e.miles = String(r.miles)
  e.route_miles = String(r.miles)
  e.route_id = r.route_id
  e.route = { summary: wiz.value.stops.filter((x: string) => x.trim()).join(' → ') +
    (wiz.value.round_trip ? ' → ' + wiz.value.stops[0] : '') + `, ${r.miles} mi measured` }
  // Accounting's template records mileage as "no receipt"; say so for them.
  if (e.receiptChoice === '') { e.receiptChoice = 'N' }
  if (e.receiptChoice === 'N' && !e.no_receipt_reason) e.no_receipt_reason = 'Mileage — measured route'
}

// ---- inline Purpose / Deal on the table (Jim, Oct 2 2026) ----
// The server REPLACES a line from the body it is sent, so a quick change
// sends the whole line as stored with only the one field changed -- sending
// just the field would blank everything else.
function lineBody(ln: any, over: Record<string, any> = {}) {
  const pipe = ln.deal_kind === 'pipeline'
  return {
    line_date: ln.line_date || '', line_date_end: ln.line_date_end || '',
    category_account: ln.category_account || '', purpose: ln.purpose || '',
    deal_code: ln.splits?.length ? '' : (pipe ? PIPELINE : (ln.deal_code || '')),
    deal_name: pipe ? (ln.deal_name || '') : '',
    vendor: ln.vendor || '', comment: ln.comment || '', recurring: !!ln.recurring,
    amount: ln.miles != null ? '' : (ln.amount ?? ''), miles: ln.miles ?? '',
    receipt: ln.receipt_id ? '' : (ln.receipt === 'N' ? 'N' : ''),
    no_receipt_reason: ln.receipt_id ? '' : (ln.no_receipt_reason || ''),
    receipt_id: ln.receipt_id || '', receipt_page: ln.receipt_id ? (ln.receipt_page || 1) : '',
    route_id: ln.route_id || '',
    splits: (ln.splits || []).map((x: any) => ({
      deal_code: x.deal_kind === 'pipeline' ? PIPELINE : x.deal_code,
      deal_name: x.deal_kind === 'pipeline' ? x.deal_name : '', amount: x.amount })),
    ...over,
  }
}
const inlineSaving = ref<number | null>(null)
async function quickSave(ln: any, over: Record<string, any>) {
  inlineSaving.value = ln.id
  try {
    report.value = (await api.put(`/api/expenses/reports/${report.value.id}/lines/${ln.id}`,
      lineBody(ln, over))).data
  } catch (e) { fail(e, 'Could not save the change') }
  finally { inlineSaving.value = null }
}
// A pipeline deal is a NAME, so choosing it inline opens a box for the name and
// nothing is saved until one is typed.
const pipelineName = ref<Record<number, string>>({})
function dealChoiceOf(ln: any) {
  if (pipelineName.value[ln.id] !== undefined) return PIPELINE
  return ln.deal_kind === 'pipeline' ? PIPELINE : (ln.deal_code || '')
}
function chooseDeal(ln: any, code: string) {
  if (code === PIPELINE) {
    pipelineName.value = { ...pipelineName.value, [ln.id]: ln.deal_kind === 'pipeline' ? ln.deal_name : '' }
    return
  }
  const { [ln.id]: _, ...rest } = pipelineName.value
  pipelineName.value = rest
  quickSave(ln, { deal_code: code, deal_name: '' })
}
function savePipelineName(ln: any) {
  const name = (pipelineName.value[ln.id] || '').trim()
  if (!name) return
  const { [ln.id]: _, ...rest } = pipelineName.value
  pipelineName.value = rest
  quickSave(ln, { deal_code: PIPELINE, deal_name: name })
}

// ---- receipts ----
const receiptOf = (id: any) => (report.value?.receipts || []).find((r: any) => r.id === id)
const editingReceipt = computed(() =>
  typeof editing.value?.receiptChoice === 'number' ? receiptOf(editing.value.receiptChoice) : null)
const RSTATUS: Record<string, string> = {
  pending: 'not read yet', read: 'read', no_receipt: 'no receipt found', error: 'could not be read',
}
const progress = ref('')
const RECEIPT_ACCEPT = '.pdf,.jpg,.jpeg,.png,.gif,.webp,.heic,.heif,.tif,.tiff,.bmp'

// ONE FILE PER REQUEST: a folder of phone photos runs past the server's
// request-size limit in one go, and counting through them is the progress.
async function uploadFiles(ev: Event) {
  const input = ev.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length || !report.value) return
  const problems: string[] = []
  const stored: number[] = []
  for (let i = 0; i < files.length; i++) {
    progress.value = `Uploading ${i + 1} of ${files.length}…`
    const fd = new FormData()
    fd.append('files', files[i], files[i].name)
    try {
      const r = await api.post(`/api/expenses/reports/${report.value.id}/receipts`, fd)
      for (const x of r.data.results) {
        if (x.result === 'stored') stored.push(x.receipt_id)
        if (x.why && x.result !== 'skipped') problems.push(`${x.file}: ${x.why}`)
      }
    } catch (e: any) {
      problems.push(`${files[i].name}: ${e?.response?.data?.error || 'upload failed'}`)
    }
  }
  await readReceipts(stored)
  if (problems.length) dataStore.addToast(problems.join(' · '), 'info')
}

// Reads one file per request, for the same reason, and so the employee sees
// "Reading 3 of 12" rather than a spinner that might be stuck.
async function readReceipts(ids?: number[]) {
  if (!report.value) return
  const todo = ids ?? (report.value.receipts || [])
    .filter((r: any) => r.status === 'pending').map((r: any) => r.id)
  for (let i = 0; i < todo.length; i++) {
    progress.value = `Reading receipt ${i + 1} of ${todo.length}…`
    try {
      report.value = (await api.post(
        `/api/expenses/reports/${report.value.id}/receipts/${todo[i]}/extract`)).data
    } catch (e) { fail(e, 'Could not read a receipt') }
  }
  progress.value = ''
  report.value = (await api.get(`/api/expenses/reports/${report.value.id}`)).data
}

async function rereadReceipt(rc: any) { await readReceipts([rc.id]) }
async function removeReceipt(rc: any) {
  try {
    report.value = (await api.delete(`/api/expenses/reports/${report.value.id}/receipts/${rc.id}`)).data
  } catch (e) { fail(e, 'Could not remove the receipt') }
}
function lineForReceipt(rc: any) {
  editing.value = { ...blankLine(), receiptChoice: rc.id, receipt_page: 1 }
}
const pendingCount = computed(() =>
  (report.value?.receipts || []).filter((r: any) => r.status === 'pending').length)
const linesFrom = (rcId: number) =>
  (report.value?.lines || []).filter((l: any) => l.receipt_id === rcId).length

// ---- phase 4 ----
async function copyRecurring() {
  try {
    const r = await api.post(`/api/expenses/reports/${report.value.id}/copy-recurring`)
    report.value = r.data
    const c = r.data.copied
    dataStore.addToast(c.added ? `${c.added} recurring line(s) copied from report #${c.from_report}` +
      ' -- set this month\'s date and attach the receipt.' :
      'The recurring lines are already on this report.', 'info')
  } catch (e) { fail(e, 'Could not copy recurring lines') }
}
const returnNote = ref('')
async function accountingReturn() {
  try {
    await api.post(`/api/expenses/reports/${report.value.id}/accounting-return`, { note: returnNote.value })
    dataStore.addToast(`Returned to ${report.value.employee}.`, 'success')
    returnNote.value = ''
    closeReport()
  } catch (e) { fail(e, 'Could not return the report') }
}

// A read-only look at a line's receipt -- what an approver uses.
const viewing = ref<any>(null)

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
          <th></th><th>Date / period</th><th>Category</th><th>Purpose</th><th>Deal</th>
          <th>Vendor</th><th>Comment</th><th class="num">Amount</th><th>Receipt</th>
        </tr></thead>
        <tbody>
          <tr v-for="ln in report.lines" :key="ln.id"
              :class="{ bad: report.check.by_line[ln.id]?.errors.length }">
            <!-- First, not last: the dropdowns widen the table, and an Edit at the
                 far right went off the screen (Jim, Oct 2 2026). -->
            <td class="row-actions first">
              <template v-if="report.permissions.edit">
                <button class="link" @click="editLine(ln)">Edit</button>
                <button class="link danger" @click="deleteLine(ln)">Remove</button>
              </template>
              <button v-else class="link" @click="viewing = ln">View</button>
            </td>
            <td>{{ ln.line_date }}<template v-if="ln.line_date_end"> – {{ ln.line_date_end }}</template></td>
            <td :title="ln.category_account">{{ ln.category_name || ln.category_account }}</td>
            <td>
              <select v-if="report.permissions.edit" class="cell-select" :value="ln.purpose || ''"
                      :disabled="inlineSaving === ln.id"
                      @change="(e: any) => quickSave(ln, { purpose: e.target.value })">
                <option value="" disabled>choose…</option>
                <option v-for="p in options.purposes" :key="p" :value="p">{{ p }}</option>
              </select>
              <template v-else>{{ ln.purpose }}</template>
            </td>
            <td>
              <template v-if="ln.splits.length">
                <div v-for="s in ln.splits" :key="s.id" class="split-cell">
                  {{ dealOf(s) }} <span class="muted">{{ fmt(s.amount) }}</span></div>
                <div v-if="report.permissions.edit" class="muted">split — Edit to change</div>
              </template>
              <template v-else-if="report.permissions.edit">
                <select class="cell-select" :value="dealChoiceOf(ln)" :disabled="inlineSaving === ln.id"
                        @change="(e: any) => chooseDeal(ln, e.target.value)">
                  <option value="" disabled>choose…</option>
                  <option v-for="d in options.deals" :key="d.code" :value="d.code">{{ dealLabel(d) }}</option>
                  <option :value="PIPELINE">Pipeline deal — type its name…</option>
                </select>
                <input v-if="pipelineName[ln.id] !== undefined" v-model="pipelineName[ln.id]"
                       class="cell-select" placeholder="pipeline deal name, then Enter"
                       @keyup.enter="savePipelineName(ln)" @blur="savePipelineName(ln)" />
                <!-- The choice reads "Pipeline deal", so the NAME is shown under it. -->
                <button v-else-if="ln.deal_kind === 'pipeline'" class="link pipe-name"
                        title="Rename the pipeline deal"
                        @click="pipelineName = { ...pipelineName, [ln.id]: ln.deal_name || '' }">
                  {{ ln.deal_name }}</button>
              </template>
              <template v-else>{{ dealOf(ln) }}</template>
            </td>
            <td>{{ ln.vendor }}</td>
            <td class="comment" :title="ln.comment">{{ ln.comment }}
              <div v-if="ln.miles != null" class="muted">{{ ln.miles }} mi × ${{ ln.mileage_rate }}</div>
              <div v-if="ln.route" class="muted" :title="ln.route.stops.map((x: any) => `${x.input} → ${x.resolved}`).join('\n')">
                ↦ {{ ln.route.summary }}</div>
              <div v-for="m in report.check.by_line[ln.id]?.errors" :key="m" class="err-text">Line {{ m }}</div>
              <div v-for="m in report.check.by_line[ln.id]?.warnings" :key="m" class="warn-text">{{ m }}</div>
              <div v-if="ln.recurring" class="muted">↻ recurring</div>
            </td>
            <td class="num">{{ fmt(ln.amount) }}</td>
            <td>
              <button v-if="ln.receipt_id" class="link" @click="viewing = ln">
                📎 {{ receiptOf(ln.receipt_id)?.filename }}<template v-if="ln.receipt_page > 1"> p.{{ ln.receipt_page }}</template></button>
              <template v-else-if="ln.receipt === 'N'">No — {{ ln.no_receipt_reason }}</template>
              <div v-if="ln.extracted?.handwritten_amount" class="warn-text">handwritten amount</div>
            </td>
          </tr>
          <tr v-if="!report.lines.length"><td colspan="9" class="muted">No lines yet.</td></tr>
        </tbody>
        <tfoot><tr>
          <td colspan="7" class="num"><strong>Total</strong></td>
          <td class="num"><strong>{{ fmt(report.total) }}</strong></td><td></td>
        </tr></tfoot>
      </table>

      <div class="row">
        <button v-if="report.permissions.edit && !editing" class="btn-secondary add" @click="newLine">
          + Add an expense</button>
        <button v-if="report.permissions.copy_recurring && !editing" class="btn-secondary add"
                @click="copyRecurring">↻ Copy recurring lines from my last report</button>
      </div>

      <!-- ---------- line form: a pop-up, the receipt beside the entry ---------- -->
      <div v-if="editing" class="modal-backdrop" @click.self="editing = null">
       <div class="modal" role="dialog" aria-modal="true">
        <div class="modal-head">
          <strong>{{ editing.id ? 'Edit expense' : 'New expense' }}</strong>
          <button class="link" @click="editing = null">✕ Close</button>
        </div>
      <div class="line-form" :class="{ 'with-receipt': editingReceipt }">
       <div class="form-col">
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
              <option v-for="d in options.deals" :key="d.code" :value="d.code">{{ dealLabel(d) }}</option>
              <option :value="PIPELINE">Pipeline deal — type its name…</option>
            </select>
          </label>
          <label v-if="!editing.isSplit && editing.deal_code === PIPELINE">Pipeline deal
            <input v-model="editing.deal_name" placeholder="e.g. Market at Poplar" /></label>
          <button v-if="!editing.isSplit" class="link" @click="startSplit">Split across deals…</button>
          <label>Vendor <input v-model="editing.vendor" placeholder="if applicable" /></label>
          <label class="grow">Comment <input v-model="editing.comment"
                 placeholder="e.g. Market Poplar Site Visit - Airport Parking" /></label>
        </div>
        <div class="row">
          <label class="check" title="Copied forward to your next report by the recurring button">
            <input type="checkbox" v-model="editing.recurring" /> recurring every month</label>
          <label class="check"><input type="checkbox" v-model="editing.isMileage" /> mileage</label>
          <template v-if="editing.isMileage">
            <label>Miles <input type="number" step="0.1" v-model="editing.miles" class="num-in" /></label>
            <span v-if="rateForLine" class="muted">× ${{ rateForLine.rate }} (from {{ rateForLine.effective_date }})
              = <strong>{{ fmt(mileageAmount) }}</strong>. Tolls go on their own line.</span>
            <span v-else class="warn-text">No mileage rate is in force on that date — accounting sets it.</span>
            <button v-if="!wiz" class="link" @click="openWizard">Measure route…</button>
            <span v-if="editing.route?.summary" class="muted">
              {{ editing.route.summary }}</span>
          </template>
          <label v-else>Amount <input v-model="editing.amount" class="num-in" placeholder="0.00" /></label>
          <label>Receipt
            <select v-model="editing.receiptChoice">
              <option value="" disabled>attach one…</option>
              <option v-for="rc in report.receipts" :key="rc.id" :value="rc.id">{{ rc.filename }}</option>
              <option value="N">No receipt</option>
            </select>
          </label>
          <label v-if="editingReceipt && editingReceipt.page_count > 1">Page
            <input type="number" min="1" :max="editingReceipt.page_count" v-model.number="editing.receipt_page" class="num-in" /></label>
          <label v-if="editing.receiptChoice === 'N'" class="grow">Why is there no receipt?
            <input v-model="editing.no_receipt_reason" /></label>
        </div>
        <div v-if="wiz && editing.isMileage" class="wizard">
          <div class="muted">Addresses, landmarks, city names or airport codes (PHL, MSY). Google measures the drive.</div>
          <div v-for="(st, i) in wiz.stops" :key="i" class="row">
            <label class="grow">{{ i === 0 ? 'From' : i === wiz.stops.length - 1 ? 'To' : 'Stop ' + i }}
              <input v-model="wiz.stops[i]" :placeholder="i === 0 ? 'e.g. 1 Main St, Philadelphia, or PHL' : 'e.g. Market at Poplar, Memphis'"
                     @keyup.enter="measureRoute" /></label>
            <button v-if="wiz.stops.length > 2" class="link" @click="wiz.stops.splice(i, 1)">remove</button>
          </div>
          <div class="row">
            <button class="link" @click="wiz.stops.splice(wiz.stops.length - 1, 0, '')">+ a stop in between</button>
            <label class="check"><input type="checkbox" v-model="wiz.round_trip" /> round trip (drive back to the start)</label>
            <button class="btn-secondary" :disabled="wiz.busy || wiz.stops.filter((x: string) => x.trim()).length < 2"
                    @click="measureRoute">{{ wiz.busy ? 'Measuring…' : 'Measure' }}</button>
            <button class="link" @click="wiz = null">cancel</button>
          </div>
          <div v-if="wiz.result?.stops" class="resolved">
            <div v-for="(st, i) in wiz.result.stops" :key="i" :class="{ 'err-text': st.error }">
              <strong>{{ st.input }}</strong> → {{ st.resolved || '—' }}
              <span v-if="st.error"> — {{ st.error }}</span>
            </div>
          </div>
          <div v-if="wiz.error" class="err-text">{{ wiz.error }}</div>
          <div v-if="wiz.result?.miles" class="row">
            <strong>{{ wiz.result.miles }} miles</strong>
            <span v-if="wiz.result.legs?.length > 1" class="muted">({{ wiz.result.legs.join(' + ') }})</span>
            <span class="ok-text">filled in above — Save line keeps it</span>
          </div>
        </div>
        <div v-if="editing.isSplit" class="splits">
          <div class="muted">Split this expense across deals. Enter amounts, or percentages and
            <button class="link" @click="applyPercents">convert to amounts</button>.</div>
          <div v-for="(s, i) in editing.splits" :key="i" class="row">
            <select v-model="s.deal_code">
              <option value="" disabled>deal…</option>
              <option v-for="d in options.deals" :key="d.code" :value="d.code">{{ dealLabel(d) }}</option>
              <option :value="PIPELINE">Pipeline deal — type its name…</option>
            </select>
            <input v-if="s.deal_code === PIPELINE" v-model="s.deal_name" placeholder="pipeline deal name" />
            <input v-model="s.pct" class="num-in" placeholder="%" />
            <input v-model="s.amount" class="num-in" placeholder="amount" />
            <button class="link" @click="editing.splits.splice(i, 1)">remove</button>
          </div>
          <div class="row">
            <button class="link" @click="editing.splits.push({ deal_code: '', deal_name: '', amount: '', pct: '' })">+ another deal</button>
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
       <!-- The receipt beside the line it supports, so a handwritten amount the
            reader missed can be read off the image and corrected here. -->
       <div v-if="editingReceipt" class="receipt-col">
         <ReceiptViewer :report-id="report.id" :receipt-id="editingReceipt.id" :page="editing.receipt_page"
                        :content-type="editingReceipt.content_type" :view-type="editingReceipt.view_type"
                        :filename="editingReceipt.filename" :extracted="editing.extracted"
                        :amount="editing.isMileage ? mileageAmount : parseFloat(editing.amount)" />
       </div>
      </div>
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

      <!-- ---------- receipts ---------- -->
      <div class="receipts">
        <div class="row">
          <strong>Receipts</strong>
          <template v-if="report.permissions.edit">
            <label class="btn-primary file-btn">Upload files
              <input type="file" multiple :accept="RECEIPT_ACCEPT" hidden @change="uploadFiles" /></label>
            <label class="btn-primary file-btn">Upload a folder
              <input type="file" webkitdirectory hidden @change="uploadFiles" /></label>
            <button v-if="pendingCount && !progress" class="btn-secondary" @click="readReceipts()">
              Read {{ pendingCount }} waiting</button>
          </template>
          <span v-if="progress" class="muted">{{ progress }}</span>
          <span v-else class="muted">PDF, JPG, PNG, iPhone HEIC and other images. Each receipt read becomes a line for you to complete.</span>
        </div>
        <table v-if="report.receipts?.length" class="data-table compact">
          <tbody>
            <tr v-for="rc in report.receipts" :key="rc.id">
              <td><button class="link" @click="viewing = { receipt_id: rc.id, receipt_page: 1 }">📎 {{ rc.filename }}</button>
                <span v-if="rc.page_count > 1" class="muted"> ({{ rc.page_count }} pages)</span></td>
              <td :class="{ 'warn-text': rc.status === 'error' || rc.status === 'no_receipt' }">
                {{ RSTATUS[rc.status] || rc.status }}<template v-if="rc.error"> — {{ rc.error }}</template></td>
              <td class="muted">{{ linesFrom(rc.id) }} line(s)</td>
              <td><span v-if="rc.duplicate_of" class="warn-text">also on another report</span></td>
              <td class="row-actions">
                <template v-if="report.permissions.edit">
                  <button v-if="!linesFrom(rc.id) && rc.status !== 'pending'" class="link" @click="rereadReceipt(rc)">Read again</button>
                  <button class="link" @click="lineForReceipt(rc)">Add a line for it</button>
                  <button class="link" @click="removeReceipt(rc)">Remove</button>
                </template>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ---------- a line, read-only: what the approver sees ---------- -->
      <div v-if="viewing && !editing" class="modal-backdrop" @click.self="viewing = null">
       <div class="modal" role="dialog" aria-modal="true">
        <div class="modal-head">
          <strong>{{ viewing.id ? 'Expense' : receiptOf(viewing.receipt_id)?.filename }}</strong>
          <span>
            <button v-if="viewing.id && report.permissions.edit" class="link"
                    @click="editLine(viewing); viewing = null">Edit</button>
            <button class="link" @click="viewing = null">✕ Close</button>
          </span>
        </div>
        <div class="line-form" :class="{ 'with-receipt': viewing.id && viewing.receipt_id }">
          <div v-if="viewing.id" class="form-col">
            <dl class="details">
              <dt>Date</dt><dd>{{ viewing.line_date }}<template v-if="viewing.line_date_end"> – {{ viewing.line_date_end }}</template></dd>
              <dt>Category</dt><dd>{{ viewing.category_name || viewing.category_account || '—' }}</dd>
              <dt>Purpose</dt><dd>{{ viewing.purpose || '—' }}</dd>
              <dt>Deal</dt>
              <dd>
                <template v-if="viewing.splits?.length">
                  <div v-for="x in viewing.splits" :key="x.id">{{ dealOf(x) }} — {{ fmt(x.amount) }}</div>
                </template>
                <template v-else>{{ dealOf(viewing) || '—' }}</template>
              </dd>
              <dt>Vendor</dt><dd>{{ viewing.vendor || '—' }}</dd>
              <dt>Comment</dt><dd>{{ viewing.comment || '—' }}</dd>
              <dt>Amount</dt><dd><strong>{{ fmt(viewing.amount) }}</strong></dd>
              <template v-if="viewing.miles != null">
                <dt>Mileage</dt><dd>{{ viewing.miles }} mi × ${{ viewing.mileage_rate }}</dd>
              </template>
              <template v-if="viewing.route">
                <dt>Route</dt>
                <dd>{{ viewing.route.summary }}
                  <div v-for="(x, i) in viewing.route.stops" :key="i" class="muted">{{ x.input }} → {{ x.resolved }}</div></dd>
              </template>
              <dt>Receipt</dt>
              <dd>{{ viewing.receipt_id ? receiptOf(viewing.receipt_id)?.filename
                     : viewing.receipt === 'N' ? 'None — ' + (viewing.no_receipt_reason || '') : '—' }}</dd>
              <template v-if="viewing.recurring"><dt>Recurring</dt><dd>every month</dd></template>
            </dl>
            <div v-for="m in report.check.by_line[viewing.id]?.errors" :key="m" class="err-text">Line {{ m }}</div>
            <div v-for="m in report.check.by_line[viewing.id]?.warnings" :key="m" class="warn-text">{{ m }}</div>
          </div>
          <div v-if="viewing.receipt_id" class="receipt-col">
            <ReceiptViewer :report-id="report.id" :receipt-id="viewing.receipt_id" :page="viewing.receipt_page"
                           :content-type="receiptOf(viewing.receipt_id)?.content_type"
                           :view-type="receiptOf(viewing.receipt_id)?.view_type"
                           :filename="receiptOf(viewing.receipt_id)?.filename"
                           :extracted="viewing.extracted" :amount="viewing.amount" />
          </div>
        </div>
       </div>
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

      <div v-if="report.permissions.accounting_return" class="decision">
        <h4>Accounting</h4>
        <textarea v-model="returnNote" rows="2"
                  placeholder="Why it is going back — required. It returns to the employee and must be approved again."></textarea>
        <div class="row">
          <button class="btn-secondary" :disabled="!returnNote.trim()" @click="accountingReturn">
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
        the CFO, CEO or President may approve any report when its approver is out.
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

      <div v-if="options.missing_categories?.length" class="notice">
        These categories from accounting's template are not in the chart of accounts, so
        employees cannot choose them: {{ options.missing_categories.join(', ') }}.
      </div>

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
.modal-backdrop { position: fixed; inset: 0; z-index: 1000; background: rgba(15, 20, 30, .45);
  display: flex; align-items: flex-start; justify-content: center; padding: 3vh 2vw; overflow: auto; }
.modal { background: var(--color-bg, #fff); color: var(--color-text); border-radius: 8px;
  width: min(1320px, 96vw); max-height: 94vh; overflow: auto; padding: 10px 16px 14px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, .25); }
.modal-head { display: flex; justify-content: space-between; align-items: center; gap: 12px;
  padding-bottom: 6px; border-bottom: 1px solid var(--color-border); margin-bottom: 8px; }
.modal .line-form { border: none; margin: 0; padding: 0; }
.modal .receipt-col { min-height: 72vh; }
.modal:not(:has(.receipt-col)) { width: min(760px, 96vw); }
.details { display: grid; grid-template-columns: 110px 1fr; gap: 4px 12px; margin: 0 0 8px; font-size: 13px; }
.details dt { color: var(--color-text-secondary); font-size: 11.5px; text-transform: uppercase; padding-top: 2px; }
.details dd { margin: 0; }
td.row-actions.first { white-space: nowrap; width: 1%; }
.link.danger { color: #a33; }
/* On a narrow window the receipt goes UNDER the entry, not off to the right. */
@media (max-width: 900px) {
  .line-form.with-receipt { grid-template-columns: 1fr; }
  .modal .receipt-col { min-height: 60vh; }
}
.line-form.with-receipt { display: grid; grid-template-columns: minmax(420px, 1fr) minmax(360px, 1fr); gap: 16px; }
.form-col { min-width: 0; }
.receipt-col { min-width: 0; display: flex; }
.receipt-col > * { flex: 1; }
.receipts { margin: 10px 0; }
.data-table.compact td { padding: 3px 8px; }
.file-btn { display: inline-flex; flex-direction: row; align-items: center; }
/* The page used these two classes without ever styling them, so the key actions
   rendered as plain text (Jim, Oct 5 2026: "the upload files and upload folder
   are key buttons but only look like links"). */
.btn-primary, .btn-secondary {
  padding: 5px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-primary, #2f6f4f); line-height: 1.3; }
.btn-primary { background: var(--color-primary, #2f6f4f); color: #fff; font-weight: 600; }
.btn-primary:hover:not(:disabled) { filter: brightness(1.08); }
.btn-secondary { background: var(--color-surface, #fff); color: var(--color-primary, #2f6f4f); }
.btn-secondary:hover:not(:disabled) { background: rgba(47, 111, 79, .08); }
.btn-primary:disabled, .btn-secondary:disabled { opacity: .55; cursor: default; }
.ok-text { color: var(--color-primary, #2f6f4f); font-size: 12.5px; }

.splits { border-top: 1px dashed var(--color-border); padding-top: 6px; }
.wizard { border: 1px solid var(--color-border); border-radius: 6px; padding: 6px 10px; margin: 6px 0; background: var(--color-surface); }
.wizard .resolved { font-size: 12.5px; margin: 4px 0; }
.split-cell { white-space: nowrap; }
.cell-select { width: 100%; min-width: 150px; max-width: 230px; font-size: 12px; padding: 3px 4px; }
.pipe-name { display: block; padding: 2px 0 0; text-align: left; }
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
