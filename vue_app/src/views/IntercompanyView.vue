<script setup lang="ts">
/**
 * Intercompany — the Due to/from PSC Manager reconciliation.
 *
 * The CFO's `DTM Recon Template` as a screen, read from the app's copy of the GL
 * (`gl_detail`) rather than three Spreadsheet Server tabs. Every figure opens the
 * GL lines behind it, and those lines are selected by the same rule as the figure,
 * so they always add up to it. Design: .claude/memory/intercompany.md.
 *
 * Pay (CFO, Sep 30 2026): the amount to pay STARTS at what the entity can afford
 * and may be changed; the can-afford figure itself may be adjusted, with a reason,
 * because not all GL cash is spendable (AMB6 withholds cash for tax and audit
 * accruals). Ticked rows become the MRI GL upload -- the JE Template's shape.
 */
import { ref, computed, onMounted, watch } from 'vue'
import api from '@/api/client'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const canEdit = computed(() => auth.canEditAccounting)

const periods = ref<string[]>([])
const period = ref('')
const tolerance = ref(1)
const result = ref<any>(null)
const loading = ref(false)
const exporting = ref(false)
const error = ref<string | null>(null)
const unavailable = ref<string | null>(null)

async function loadPeriods() {
  try {
    const res = await api.get('/api/intercompany/periods')
    if (!res.data.available) { unavailable.value = res.data.reason; return }
    periods.value = res.data.periods
    period.value = res.data.latest
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

async function load() {
  if (!period.value) return
  loading.value = true
  error.value = null
  try {
    const res = await api.get('/api/intercompany/reconciliation', {
      params: { period: period.value, tolerance: tolerance.value },
    })
    result.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function exportExcel() {
  exporting.value = true
  try {
    const res = await api.get('/api/intercompany/reconciliation/excel', {
      params: { period: period.value, tolerance: tolerance.value },
      responseType: 'blob',
    })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([res.data]))
    a.download = `Due_to_Manager_Recon_${period.value}.xlsx`
    a.click()
    URL.revokeObjectURL(a.href)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    exporting.value = false
  }
}

// A result, and anything opened from it, belongs to the period that produced it.
watch(period, () => { closeDetail(); load() })

// ── formatting ──────────────────────────────────────────────────────────
function money(v: any): string {
  if (v === null || v === undefined || v === '') return ''
  const n = Number(v)
  if (Number.isNaN(n)) return String(v)
  if (Math.abs(n) < 0.005) return '—'
  const s = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2,
                                                 maximumFractionDigits: 2 })
  return n < 0 ? `(${s})` : s
}

// ── the grid: status filter, sort and per-column filter ────────────────
const showNoBalance = ref(false)
const statusFilter = ref<string>('')
const sortKey = ref('')
const sortDir = ref<'asc' | 'desc'>('asc')
const colFilters = ref<Record<string, string>>({})
const filtersOpen = ref(false)

const COLS = [
  { key: 'entity_id', label: 'Entity' },
  { key: 'name', label: 'Name', clip: true },
  { key: 'entity_balance', label: 'A', num: true, side: 'entity',
    tip: "A — the entity's Due To/From PSC Manager (MR15000002)" },
  { key: 'alt_account', label: 'Alt acct', tip: "The entity's alternate Due To/From account, if it has one" },
  { key: 'alt_balance', label: 'B', num: true, side: 'alt', tip: 'B — balance on the alternate account' },
  { key: 'total_entity', label: 'C = A+B', num: true, tip: 'C — total entity balance' },
  { key: 'manager_balance', label: 'D Manager', num: true, side: 'manager',
    tip: "D — PSC Manager's Due to/from Intercompany (MR15000001) for this entity's segment" },
  { key: 'variance', label: 'C + D', num: true, tip: 'Variance — should be zero' },
  { key: 'status', label: 'Status' },
  { key: 'comment', label: 'Comments' },
  { key: 'cash_balance', label: 'Cash', num: true, side: 'cash', tip: 'Cash per the GL at period end' },
  { key: 'affordable', label: 'Can afford', num: true,
    tip: "The lower of what PSC Manager is owed (D) and the entity's cash — adjustable, with a reason" },
  { key: 'pay_amount', label: 'To pay', num: true,
    tip: 'Starts at what it can afford; type a different amount to pay less' },
  { key: 'pay_cash_account', label: 'Pay from', tip: 'The cash account the entry credits' },
]

function toggleSort(key: string) {
  if (sortKey.value === key) {
    if (sortDir.value === 'asc') sortDir.value = 'desc'
    else { sortKey.value = ''; sortDir.value = 'asc' }
  } else { sortKey.value = key; sortDir.value = 'asc' }
}

const allRows = computed<any[]>(() => result.value?.rows || [])
const noBalanceCount = computed(() => allRows.value.filter(r => r.status === 'No Balance').length)
const statusCounts = computed(() => {
  const c: Record<string, number> = {}
  for (const r of allRows.value) c[r.status] = (c[r.status] || 0) + 1
  return c
})

const viewRows = computed<any[]>(() => {
  let rows = allRows.value
  if (!showNoBalance.value && statusFilter.value !== 'No Balance')
    rows = rows.filter(r => r.status !== 'No Balance')
  if (statusFilter.value) rows = rows.filter(r => r.status === statusFilter.value)
  for (const [k, v] of Object.entries(colFilters.value)) {
    const needle = (v || '').trim().toLowerCase()
    if (needle) rows = rows.filter(r => String(r[k] ?? '').toLowerCase().includes(needle))
  }
  if (sortKey.value) {
    const k = sortKey.value, dir = sortDir.value === 'asc' ? 1 : -1
    // Copied first: sort mutates, and clearing the sort must restore the order.
    rows = rows.slice().sort((a, b) => {
      const x = a[k], y = b[k]
      if (x === null || x === undefined || x === '') return 1
      if (y === null || y === undefined || y === '') return -1
      if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir
      return String(x).localeCompare(String(y), undefined, { numeric: true }) * dir
    })
  }
  return rows
})

// The totals row covers the rows SHOWN. Hidden rows are no-balance by definition
// unless a filter is on, and then the grand totals are shown beside it.
const shownTotals = computed(() => {
  const t: Record<string, number> = {}
  for (const c of COLS) if (c.num) t[c.key] = 0
  for (const r of viewRows.value)
    for (const k of Object.keys(t)) t[k] += Number(r[k] || 0)
  return t
})
const filtered = computed(() =>
  !!statusFilter.value || Object.values(colFilters.value).some(v => (v || '').trim()))

// ── drilldown: the GL lines behind a figure ────────────────────────────
const detail = ref<any>(null)
const detailRow = ref<any>(null)
const detailLoading = ref(false)
const SIDE_LABEL: Record<string, string> = {
  entity: 'Due To/From PSC Manager', alt: 'Alternate account',
  manager: 'PSC Manager — Due to/from Intercompany', cash: 'Cash',
}
const SIDE_KEY: Record<string, string> = {
  entity: 'entity_balance', alt: 'alt_balance', manager: 'manager_balance',
  cash: 'cash_balance',
}

async function openLines(r: any, side: string) {
  settingsFor.value = null
  detailRow.value = r
  detail.value = { side, loading: true }
  detailLoading.value = true
  try {
    const res = await api.get('/api/intercompany/lines', {
      params: { period: period.value, entity: r.entity_id, side },
    })
    detail.value = res.data
  } catch (e: any) {
    detail.value = { side, error: e.response?.data?.error || e.message }
  } finally {
    detailLoading.value = false
  }
}
function closeDetail() { detail.value = null; detailRow.value = null }
// Whether the lines add up to the figure they were opened from is the first thing
// said: a drilldown that does not reconcile makes a correct figure look wrong.
const detailTies = computed(() => {
  if (!detail.value || detail.value.total === undefined || !detailRow.value) return null
  const fig = Number(detailRow.value[SIDE_KEY[detail.value.side]] || 0)
  return Math.abs(fig - Number(detail.value.total)) < 0.005
})

// ── comments ───────────────────────────────────────────────────────────
const editing = ref<string | null>(null)
const draft = ref('')
const savingNote = ref(false)
function startNote(r: any) {
  if (!canEdit.value) return
  editing.value = r.entity_id
  draft.value = r.comment || ''
}
async function saveNote(r: any) {
  if (editing.value !== r.entity_id) return
  if ((draft.value || '').trim() === (r.comment || '')) { editing.value = null; return }
  savingNote.value = true
  try {
    const res = await api.put('/api/intercompany/notes', {
      period: period.value, entity_id: r.entity_id, comment: draft.value,
    })
    r.comment = res.data.comment
    r.comment_by = res.data.updated_by
    r.comment_at = res.data.updated_at
    editing.value = null
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    savingNote.value = false
  }
}

// ── per-entity settings ────────────────────────────────────────────────
const settingsFor = ref<any>(null)
const sForm = ref<any>({})
const savingSettings = ref(false)
const settingsError = ref<string | null>(null)
async function openSettings(r: any) {
  closeDetail()
  settingsError.value = null
  settingsFor.value = r
  try {
    const res = await api.get('/api/intercompany/settings')
    const s = (res.data.settings || []).find((x: any) => x.entity_id === r.entity_id) || {}
    sForm.value = {
      alt_account: s.alt_account || '',
      cash_mode: (s.cash_accounts || []).length ? 'named'
        : (s.cash_exclude || []).length ? 'exclude' : 'default',
      cash_accounts: (s.cash_accounts || []).join(', '),
      cash_exclude: (s.cash_exclude || []).join(', '),
      currency: s.currency || 'USD',
      basis: s.basis || '',
      updated_by: s.updated_by, updated_at: s.updated_at,
    }
  } catch (e: any) {
    settingsError.value = e.response?.data?.error || e.message
  }
}
async function saveSettings() {
  savingSettings.value = true
  settingsError.value = null
  const f = sForm.value
  try {
    await api.put(`/api/intercompany/settings/${settingsFor.value.entity_id}`, {
      alt_account: f.alt_account,
      cash_accounts: f.cash_mode === 'named' ? f.cash_accounts : '',
      cash_exclude: f.cash_mode === 'exclude' ? f.cash_exclude : '',
      currency: f.currency,
    })
    settingsFor.value = null
    await load()
  } catch (e: any) {
    settingsError.value = e.response?.data?.error || e.message
  } finally {
    savingSettings.value = false
  }
}

// ── pay ────────────────────────────────────────────────────────────────
// A row can be ticked when PSC Manager is owed something and it is not already
// in a batch the GL does not show. The ORDER ticked is the order in the file.
const selected = ref<string[]>([])
const payable = (r: any) => r.manager_balance > 0.005 && !r.pending_batch
function toggleSelect(r: any) {
  const i = selected.value.indexOf(r.entity_id)
  if (i >= 0) selected.value.splice(i, 1)
  else selected.value.push(r.entity_id)
}
const payEdit = ref<{ entity: string, field: string } | null>(null)
const payDraft = ref<any>({})
const payError = ref<string | null>(null)
const payWarn = ref<string | null>(null)
const savingPay = ref(false)
function startPay(r: any, field: string) {
  if (!canEdit.value || !payable(r)) return
  payError.value = null
  payEdit.value = { entity: r.entity_id, field }
  payDraft.value = field === 'afford'
    ? { value: r.afford_override ?? r.affordable_computed ?? '', reason: r.afford_reason || '' }
    : field === 'amount' ? { value: r.pay_amount ?? '' }
      : { value: r.pay_cash_account || '' }
}
async function savePay(r: any, patch: Record<string, any>) {
  savingPay.value = true
  payError.value = null
  payWarn.value = null
  try {
    const res = await api.put('/api/intercompany/pay', {
      period: period.value, entity_id: r.entity_id, ...patch,
    })
    Object.assign(r, res.data.row)
    payWarn.value = (res.data.warnings || []).length
      ? `${r.entity_id}: ${res.data.warnings.join(' ')}` : null
    payEdit.value = null
  } catch (e: any) {
    payError.value = e.response?.data?.error || e.message
  } finally {
    savingPay.value = false
  }
}
function commitPay(r: any) {
  const f = payEdit.value?.field, d = payDraft.value
  if (f === 'afford') savePay(r, { afford_override: d.value, afford_reason: d.reason })
  else if (f === 'amount') savePay(r, { amount: d.value })
  else savePay(r, { cash_account: d.value })
}

const jePeriod = ref('')
const entrDate = ref('')
const preview = ref<any>(null)
const generating = ref(false)
const selectedRows = computed(() =>
  selected.value.map(e => allRows.value.find(r => r.entity_id === e)).filter(Boolean))
const selectedTotal = computed(() =>
  selectedRows.value.reduce((t: number, r: any) => t + Number(r.pay_amount || 0), 0))
async function runBatch(commit: boolean) {
  generating.value = true
  try {
    const res = await api.post('/api/intercompany/batches', {
      period: period.value, entities: selected.value, je_period: jePeriod.value,
      entrdate: entrDate.value, commit,
    }, { validateStatus: s => s < 500 })
    preview.value = res.data
    if (commit && res.data.batch_id) {
      downloadBatch(res.data.batch_id)
      selected.value = []
      await load()
      await loadBatches()
    }
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    generating.value = false
  }
}
async function downloadBatch(id: string) {
  const res = await api.get(`/api/intercompany/batches/${id}/csv`, { responseType: 'blob' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(new Blob([res.data], { type: 'text/csv' }))
  a.download = `${id}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}
const batchList = ref<any[]>([])
async function loadBatches() {
  try { batchList.value = (await api.get('/api/intercompany/batches')).data.batches || [] }
  catch { batchList.value = [] }
}
async function voidBatch(b: any) {
  if (!confirm(`Void ${b.batch_id}? Only if it will NOT be uploaded to MRI.`)) return
  try {
    await api.post(`/api/intercompany/batches/${b.batch_id}/void`)
    await load()
    await loadBatches()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}
watch(period, () => { selected.value = []; preview.value = null })

onMounted(async () => { await loadPeriods(); await load(); await loadBatches() })
</script>

<template>
  <div class="ic">
    <div class="header-row">
      <div>
        <h2>Intercompany</h2>
        <div class="subtitle">
          Due to / from PSC Manager reconciliation, read from the app's copy of the GL.
        </div>
      </div>
      <div class="header-controls" v-if="periods.length">
        <label class="ctl">Period
          <select v-model="period">
            <option v-for="p in periods" :key="p" :value="p">{{ p }}</option>
          </select>
        </label>
        <label class="ctl">Tolerance $
          <input type="number" min="0" step="0.5" v-model.number="tolerance"
                 class="tol" @change="load" />
        </label>
        <button class="btn-secondary" :disabled="exporting || !result?.available"
                @click="exportExcel">{{ exporting ? 'Exporting…' : 'Export to Excel' }}</button>
        <button class="btn-primary" :disabled="loading" @click="load">
          {{ loading ? 'Loading…' : 'Refresh' }}</button>
      </div>
    </div>

    <div v-if="error" class="error-banner">{{ error }}</div>
    <div v-if="unavailable" class="notice">{{ unavailable }}</div>

    <template v-if="result?.available">
      <div class="meta">
        Balances from {{ result.year_start }} through {{ result.period }},
        basis {{ result.bases.join('.') }} ·
        entity side <b>{{ result.accounts.entity }}</b>,
        manager side <b>{{ result.accounts.manager_entity }} {{ result.accounts.manager }}</b>
        by entity segment ·
        <span v-if="result.data_as_of">MRI refresh completed
          {{ String(result.data_as_of).slice(0, 16).replace('T', ' ') }}</span>
        <span v-else>No completed MRI refresh recorded — freshness unknown.</span>
      </div>

      <!-- Checks first: an unattributed manager balance or a missing opening
           changes how every row below should be read. -->
      <div class="checks">
        <div v-for="c in result.checks" :key="c.key" class="check"
             :class="c.key === 'investigate' ? (c.count ? 'warn' : 'ok') : (c.ok ? 'ok' : 'bad')">
          <span class="mark">{{ c.key === 'investigate' ? (c.count ? '!' : '✓') : (c.ok ? '✓' : '✕') }}</span>
          <span>{{ c.label }}</span>
          <b v-if="c.key === 'investigate'">{{ c.count }}</b>
          <b v-else-if="c.amount !== undefined && !c.ok">{{ money(c.amount) }}
            ({{ c.rows }} rows)</b>
          <div v-if="c.detail" class="check-detail">{{ c.detail }}</div>
        </div>
        <div class="check neutral">
          <span>Net variance</span><b>{{ money(result.totals.variance) }}</b>
        </div>
      </div>

      <div class="toolbar">
        <div class="chips">
          <button :class="{ on: !statusFilter }" @click="statusFilter = ''">All</button>
          <button v-for="s in ['Investigate', 'Reconciled', 'No Balance']" :key="s"
                  :class="{ on: statusFilter === s, [s.replace(' ', '')]: true }"
                  @click="statusFilter = statusFilter === s ? '' : s">
            {{ s }} <span class="n">{{ statusCounts[s] || 0 }}</span>
          </button>
        </div>
        <label class="cb" v-if="!statusFilter">
          <input type="checkbox" v-model="showNoBalance" />
          Show {{ noBalanceCount }} with no balance
        </label>
        <button class="linkish" @click="filtersOpen = !filtersOpen">
          {{ filtersOpen ? 'Hide' : 'Filter' }} by column</button>
        <button v-if="sortKey || filtered" class="linkish"
                @click="sortKey = ''; colFilters = {}; statusFilter = ''">Clear</button>
        <span class="hint-inline">Click a figure for the GL lines behind it<template
          v-if="canEdit">; an entity ID for its accounts; a comment to edit it</template>.</span>
      </div>

      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th v-if="canEdit" class="tick" title="Tick to pay; the file follows the order ticked">Pay</th>
              <th v-for="c in COLS" :key="c.key"
                  :class="{ num: c.num, sorted: sortKey === c.key }"
                  :title="(c as any).tip || `Sort by ${c.label}`"
                  @click="toggleSort(c.key)">
                {{ c.label }}<span class="arrow">{{
                  sortKey === c.key ? (sortDir === 'asc' ? '▲' : '▼') : '' }}</span>
              </th>
            </tr>
            <tr v-if="filtersOpen" class="filter-row">
              <th v-if="canEdit"></th>
              <th v-for="c in COLS" :key="c.key">
                <input v-model="colFilters[c.key]" class="colf" :placeholder="c.label" />
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in viewRows" :key="r.entity_id"
                :class="{ active: detailRow?.entity_id === r.entity_id
                          || settingsFor?.entity_id === r.entity_id }">
              <td v-if="canEdit" class="tick">
                <input v-if="payable(r)" type="checkbox" :checked="selected.includes(r.entity_id)"
                       @change="toggleSelect(r)" />
                <span v-else-if="r.pending_batch" class="ph" :title="`In ${r.pending_batch}, not yet in the GL`">in batch</span>
              </td>
              <td>
                <button class="ent" @click="openSettings(r)"
                        :title="r.settings_basis || 'Default accounts'">{{ r.entity_id }}</button>
                <span v-if="r.currency !== 'USD'" class="tag">{{ r.currency }}</span>
              </td>
              <td class="clip" :title="r.name">{{ r.name }}</td>
              <td class="num"><button class="fig" @click="openLines(r, 'entity')">{{
                money(r.entity_balance) }}</button></td>
              <td class="mono">{{ r.alt_account || '' }}</td>
              <td class="num"><button v-if="r.alt_account" class="fig"
                      @click="openLines(r, 'alt')">{{ money(r.alt_balance) }}</button></td>
              <td class="num">{{ money(r.total_entity) }}</td>
              <td class="num"><button class="fig" @click="openLines(r, 'manager')">{{
                money(r.manager_balance) }}</button></td>
              <td class="num" :class="{ vbad: r.status === 'Investigate' }">{{ money(r.variance) }}</td>
              <td><span class="status" :class="r.status.replace(' ', '')">{{ r.status }}</span></td>
              <td class="comment" @click="startNote(r)"
                  :title="r.comment_by ? `${r.comment_by}, ${String(r.comment_at).slice(0, 10)}` : ''">
                <input v-if="editing === r.entity_id" v-model="draft" class="note-input"
                       :disabled="savingNote" @keyup.enter="saveNote(r)"
                       @keyup.esc="editing = null" @blur="saveNote(r)" autofocus />
                <span v-else-if="r.comment">{{ r.comment }}</span>
                <span v-else-if="canEdit && r.status === 'Investigate'" class="ph">add…</span>
              </td>
              <td class="num"><button class="fig" @click="openLines(r, 'cash')"
                      :title="r.cash_basis">{{ money(r.cash_balance) }}</button></td>
              <td class="num pay" @click="startPay(r, 'afford')"
                  :title="r.afford_override !== null
                    ? `Adjusted from ${money(r.affordable_computed)}: ${r.afford_reason}` : ''">
                <span v-if="payEdit?.entity === r.entity_id && payEdit?.field === 'afford'" class="pay-edit"
                      @click.stop>
                  <input v-model="payDraft.value" class="amt" placeholder="amount" />
                  <input v-model="payDraft.reason" class="why" placeholder="why (required)"
                         @keyup.enter="commitPay(r)" />
                  <button class="mini" :disabled="savingPay" @click="commitPay(r)">Save</button>
                  <button class="mini" v-if="r.afford_override !== null" :disabled="savingPay"
                          @click="savePay(r, { afford_override: null })">Reset</button>
                  <button class="mini" @click="payEdit = null">✕</button>
                </span>
                <template v-else>
                  <span v-if="r.affordable !== null" :class="{ adj: r.afford_override !== null }">
                    {{ money(r.affordable) }}</span>
                  <span v-else-if="r.currency !== 'USD' && r.manager_balance > 0" class="ph"
                        title="CAD cash cannot cap a USD amount — the app holds no exchange rate">
                    CAD</span>
                </template>
              </td>
              <td class="num pay" @click="startPay(r, 'amount')">
                <span v-if="payEdit?.entity === r.entity_id && payEdit?.field === 'amount'" class="pay-edit"
                      @click.stop>
                  <input v-model="payDraft.value" class="amt" @keyup.enter="commitPay(r)"
                         @keyup.esc="payEdit = null" />
                  <button class="mini" :disabled="savingPay" @click="commitPay(r)">Save</button>
                  <button class="mini" v-if="r.pay_amount_set" :disabled="savingPay"
                          @click="savePay(r, { amount: null })">Reset</button>
                </span>
                <span v-else-if="r.pay_amount !== null" :class="{ adj: r.pay_amount_set }"
                      :title="r.pay_amount_set ? `Typed by ${r.pay_by}; can afford ${money(r.affordable)}` : 'What it can afford'">
                  {{ money(r.pay_amount) }}</span>
              </td>
              <td class="mono pay" @click="startPay(r, 'cash')">
                <span v-if="payEdit?.entity === r.entity_id && payEdit?.field === 'cash'" class="pay-edit"
                      @click.stop>
                  <select v-model="payDraft.value">
                    <option v-for="a in r.cash_by_account" :key="a.account" :value="a.account">
                      {{ a.account }} {{ money(a.amount) }}</option>
                    <option v-if="r.pay_cash_account && !r.cash_by_account.some((a: any) => a.account === r.pay_cash_account)"
                            :value="r.pay_cash_account">{{ r.pay_cash_account }}</option>
                  </select>
                  <button class="mini" :disabled="savingPay" @click="commitPay(r)">Save</button>
                </span>
                <span v-else-if="r.manager_balance > 0.005">{{ r.pay_cash_account || '—' }}</span>
              </td>
            </tr>
            <tr v-if="!viewRows.length">
              <td :colspan="COLS.length + (canEdit ? 1 : 0)" class="empty">No row matches.</td>
            </tr>
          </tbody>
          <tfoot v-if="viewRows.length">
            <tr>
              <td v-if="canEdit"></td>
              <td colspan="2">{{ filtered ? 'Shown' : 'Totals' }} ({{ viewRows.length }})</td>
              <td class="num">{{ money(shownTotals.entity_balance) }}</td>
              <td></td>
              <td class="num">{{ money(shownTotals.alt_balance) }}</td>
              <td class="num">{{ money(shownTotals.total_entity) }}</td>
              <td class="num">{{ money(shownTotals.manager_balance) }}</td>
              <td class="num">{{ money(shownTotals.variance) }}</td>
              <td colspan="2"></td>
              <td class="num">{{ money(shownTotals.cash_balance) }}</td>
              <td class="num">{{ money(shownTotals.affordable) }}</td>
              <td class="num">{{ money(shownTotals.pay_amount) }}</td>
              <td></td>
            </tr>
            <tr v-if="filtered" class="grand">
              <td v-if="canEdit"></td>
              <td colspan="2">All {{ allRows.length }}</td>
              <td class="num">{{ money(result.totals.entity_balance) }}</td>
              <td></td>
              <td class="num">{{ money(result.totals.alt_balance) }}</td>
              <td class="num">{{ money(result.totals.total_entity) }}</td>
              <td class="num">{{ money(result.totals.manager_balance) }}</td>
              <td class="num">{{ money(result.totals.variance) }}</td>
              <td colspan="6"></td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div v-if="payError" class="error-banner">{{ payError }}</div>
      <div v-if="payWarn" class="notice">{{ payWarn }}</div>

      <!-- ===== pay: the journal entry ===== -->
      <div v-if="canEdit && selected.length" class="drawer">
        <div class="drawer-head">
          <div>
            <h3>Pay {{ selected.length }} entit{{ selected.length === 1 ? 'y' : 'ies' }} —
              {{ money(selectedTotal) }}</h3>
            <div class="sub">In the order ticked: {{ selected.join(', ') }}. Each entity's two
              lines, then PSC Manager's, as the JE Template.</div>
          </div>
          <button class="btn-secondary" @click="selected = []; preview = null">Clear</button>
        </div>
        <div class="pay-form">
          <label class="ctl">Journal period
            <input v-model="jePeriod" placeholder="YYYYMM" class="mono" /></label>
          <label class="ctl">Entry date
            <input v-model="entrDate" type="date" /></label>
          <button class="btn-secondary" :disabled="generating" @click="runBatch(false)">Preview</button>
          <button class="btn-primary" :disabled="generating || !jePeriod || !entrDate"
                  @click="runBatch(true)">{{ generating ? 'Working…' : 'Generate & download' }}</button>
        </div>
        <div v-if="preview?.errors?.length" class="error-banner">
          <div v-for="(e, i) in preview.errors" :key="i">{{ e }}</div>
        </div>
        <div v-else-if="preview?.batch_id" class="notice">
          {{ preview.batch_id }} generated and downloaded. It shows as pending until the next
          MRI refresh brings it into the GL.</div>
        <div v-if="preview?.lines?.length && !preview?.batch_id" class="table-scroll short">
          <table class="data-table">
            <thead><tr><th>Entity</th><th>Account</th><th class="num">Amount</th>
              <th>Description</th><th>Related</th><th>Period</th><th>Basis</th><th>Date</th></tr></thead>
            <tbody>
              <tr v-for="(l, i) in preview.lines" :key="i">
                <td>{{ l.entityid }}</td><td class="mono">{{ l.acctnum }}</td>
                <td class="num">{{ money(l.amount) }}</td><td>{{ l.descrpn }}</td>
                <td>{{ l.rltdentity }}</td><td>{{ l.period }}</td><td>{{ l.basis }}</td>
                <td>{{ l.entrdate }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ===== generated batches ===== -->
      <div v-if="batchList.length" class="drawer">
        <div class="drawer-head"><h3>Journal entries generated</h3></div>
        <table class="data-table">
          <thead><tr><th>Batch</th><th>Period</th><th>Entry date</th><th>Entities</th>
            <th class="num">Total</th><th>Status</th><th>By</th><th></th></tr></thead>
          <tbody>
            <tr v-for="b in batchList" :key="b.batch_id">
              <td class="mono">{{ b.batch_id }}</td><td>{{ b.period }}</td><td>{{ b.entrdate }}</td>
              <td class="clip" :title="b.entities.map((x: any) => x.entity_id + (x.posted ? ' ✓' : '')).join(', ')">
                {{ b.entities.length }}</td>
              <td class="num">{{ money(b.total) }}</td>
              <td>{{ b.status }}</td>
              <td>{{ b.created_by }}, {{ String(b.created_at).slice(0, 10) }}</td>
              <td>
                <button class="linkish" @click="downloadBatch(b.batch_id)">CSV</button>
                <button v-if="canEdit && b.status.startsWith('generated')" class="linkish"
                        @click="voidBatch(b)">Void</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ===== drilldown ===== -->
      <div v-if="detail" class="drawer">
        <div class="drawer-head">
          <div>
            <h3>{{ detailRow.entity_id }} — {{ SIDE_LABEL[detail.side] }}</h3>
            <div class="sub" v-if="detail.total !== undefined">
              {{ detail.count }} line{{ detail.count === 1 ? '' : 's' }},
              {{ result.year_start }}–{{ result.period }}, basis {{ result.bases.join('.') }},
              totalling <b>{{ money(detail.total) }}</b>
              <span v-if="detailTies === true" class="ties">✓ ties to the figure</span>
              <span v-else-if="detailTies === false" class="noties">✕ does not tie to
                {{ money(detailRow[SIDE_KEY[detail.side]]) }}</span>
            </div>
          </div>
          <button class="btn-secondary" @click="closeDetail">Close</button>
        </div>
        <div v-if="detailLoading" class="loading-text">Loading…</div>
        <div v-else-if="detail.error" class="error-banner">{{ detail.error }}</div>
        <div v-else-if="!detail.rows?.length" class="notice">No GL lines.</div>
        <div v-else class="table-scroll short">
          <table class="data-table">
            <thead><tr>
              <th>Entity</th><th>Period</th><th>Date</th><th>Account</th><th>Name</th>
              <th>Basis</th><th>Ref</th><th>Description</th><th>Related</th>
              <th class="num">Amount</th>
            </tr></thead>
            <tbody>
              <tr v-for="(l, i) in detail.rows" :key="i">
                <td>{{ l.ENTITYID }}</td><td>{{ l.PERIOD }}</td>
                <td>{{ l.BALFOR === 'B' ? 'B/fwd' : l.ENTRDATE }}</td>
                <td class="mono">{{ l.ACCTNUM }}</td><td class="clip">{{ l.ACCTNAME }}</td>
                <td>{{ l.BASIS }}</td><td>{{ l.REF }}</td>
                <td class="clip" :title="l.DESCRPN">{{ l.DESCRPN }}</td>
                <td>{{ l.RLTDENTITY }}</td>
                <td class="num">{{ money(l.AMT) }}</td>
              </tr>
            </tbody>
          </table>
          <div v-if="detail.truncated" class="notice">
            Showing the first {{ detail.rows.length }} of {{ detail.count }}; the total covers all.
          </div>
        </div>
      </div>

      <!-- ===== settings ===== -->
      <div v-if="settingsFor" class="drawer">
        <div class="drawer-head">
          <div>
            <h3>{{ settingsFor.entity_id }} — accounts</h3>
            <div class="sub">{{ settingsFor.name }}. These carry to every period.</div>
          </div>
          <button class="btn-secondary" @click="settingsFor = null">Close</button>
        </div>
        <div v-if="settingsError" class="error-banner">{{ settingsError }}</div>
        <div class="sform">
          <label>Alternate Due To/From account (B)
            <input v-model="sForm.alt_account" :disabled="!canEdit" placeholder="none"
                   class="mono" />
          </label>
          <fieldset :disabled="!canEdit">
            <legend>Cash</legend>
            <label class="radio"><input type="radio" value="default" v-model="sForm.cash_mode" />
              Every MR1000* account</label>
            <label class="radio"><input type="radio" value="exclude" v-model="sForm.cash_mode" />
              MR1000* except
              <input v-model="sForm.cash_exclude" :disabled="sForm.cash_mode !== 'exclude'"
                     class="mono wide" placeholder="MR10008000" /></label>
            <label class="radio"><input type="radio" value="named" v-model="sForm.cash_mode" />
              Only
              <input v-model="sForm.cash_accounts" :disabled="sForm.cash_mode !== 'named'"
                     class="mono wide" placeholder="MR99991000" /></label>
          </fieldset>
          <label>Cash currency
            <select v-model="sForm.currency" :disabled="!canEdit">
              <option>USD</option><option>CAD</option>
            </select>
          </label>
          <div class="basis" v-if="sForm.basis">
            <b>Basis:</b> {{ sForm.basis }}
            <span v-if="sForm.updated_by"> — {{ sForm.updated_by }},
              {{ String(sForm.updated_at || '').slice(0, 10) }}</span>
          </div>
          <div v-if="canEdit">
            <button class="btn-primary" :disabled="savingSettings" @click="saveSettings">
              {{ savingSettings ? 'Saving…' : 'Save' }}</button>
          </div>
          <div v-else class="hint-inline">Read only — editing is for the accounting team.</div>
        </div>
      </div>
    </template>
    <div v-else-if="loading" class="loading-text">Loading…</div>
  </div>
</template>

<style scoped>
.ic { padding: 20px; }
.header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.header-row h2 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; }
.header-controls { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
.ctl { display: flex; flex-direction: column; font-size: 11px; text-transform: uppercase;
  letter-spacing: .03em; color: var(--color-text-secondary); gap: 3px; }
.ctl select, .ctl input { font-size: 12.5px; padding: 4px 6px; border: 1px solid var(--color-border);
  border-radius: 4px; background: var(--color-surface); color: var(--color-text); }
.tol { width: 70px; }
.error-banner { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: #fdeaea;
  color: #8a1f1f; font-size: 13px; }
.notice { margin: 8px 0; padding: 8px 12px; font-size: 12.5px; background: var(--color-surface);
  border: 1px solid var(--color-border); border-radius: 6px; }
.meta { font-size: 12px; color: var(--color-text-secondary); margin: 12px 0 10px; line-height: 1.5; }
.checks { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
.check { border: 1px solid var(--color-border); border-radius: 6px; padding: 7px 11px;
  font-size: 12.5px; display: flex; gap: 7px; align-items: baseline; flex-wrap: wrap; max-width: 460px; }
.check .mark { font-weight: 700; }
.check.ok .mark { color: #2f7a3d; }
.check.bad { border-color: #d9534f; background: #fdf1f0; }
.check.bad .mark { color: #b52b27; }
.check.warn { border-color: #d9a441; background: #fdf7ea; }
.check.warn .mark { color: #9a6700; }
.check-detail { flex-basis: 100%; font-size: 11.5px; color: var(--color-text-secondary); }
.toolbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 4px 0 8px; font-size: 12px; }
.chips { display: flex; gap: 5px; }
.chips button { border: 1px solid var(--color-border); background: none; border-radius: 12px;
  padding: 3px 10px; font-size: 12px; cursor: pointer; color: var(--color-text); }
.chips button.on { background: var(--color-primary, #2f6f4f); color: #fff; border-color: var(--color-primary, #2f6f4f); }
.chips .n { opacity: .75; margin-left: 3px; }
.cb { display: flex; gap: 5px; align-items: center; }
.hint-inline { color: var(--color-text-secondary); font-size: 11.5px; }
.linkish { background: none; border: none; padding: 0; cursor: pointer; font-size: 12px;
  color: var(--color-primary, #2b6cb0); text-decoration: underline; }
.table-scroll { overflow: auto; max-height: 60vh; }
.table-scroll.short { max-height: 40vh; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.data-table th { position: sticky; top: 0; z-index: 1; text-align: left; padding: 6px 6px;
  background: var(--color-surface); border-bottom: 2px solid var(--color-border); font-size: 11px;
  text-transform: uppercase; white-space: nowrap; color: var(--color-text-secondary);
  cursor: pointer; user-select: none; }
.data-table th.sorted { color: var(--color-text); }
.arrow { font-size: 9px; margin-left: 3px; }
.filter-row th { padding: 2px 4px; cursor: default; }
.colf { width: 100%; min-width: 50px; box-sizing: border-box; font-size: 11px; padding: 2px 4px;
  border: 1px solid var(--color-border); border-radius: 3px; }
.data-table td { padding: 4px 6px; border-bottom: 1px solid var(--color-border); white-space: nowrap; }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tr.active td { background: rgba(47, 111, 79, 0.08); }
.data-table tfoot td { font-weight: 600; border-top: 2px solid var(--color-border); }
.data-table tfoot tr.grand td { font-weight: 400; color: var(--color-text-secondary); }
.clip { max-width: 160px; overflow: hidden; text-overflow: ellipsis; }
.mono { font-family: ui-monospace, Consolas, monospace; font-size: 11.5px; }
.fig, .ent { background: none; border: none; padding: 0; cursor: pointer; font: inherit;
  color: inherit; font-variant-numeric: tabular-nums; }
.fig:hover, .ent:hover { color: var(--color-primary, #2b6cb0); text-decoration: underline; }
.ent { font-weight: 600; }
.tag { font-size: 10px; margin-left: 5px; padding: 1px 5px; border-radius: 3px;
  background: #e7eef9; color: #2b4c7e; }
.status { font-size: 11px; padding: 2px 7px; border-radius: 10px; }
.status.Reconciled { background: #e6f3e8; color: #2f7a3d; }
.status.Investigate { background: #fdecd6; color: #9a5200; }
.status.NoBalance { background: #eef0f2; color: #6b7280; }
.vbad { color: #b52b27; }
.comment { min-width: 100px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; cursor: text; }
.note-input { width: 100%; box-sizing: border-box; font-size: 12px; padding: 2px 4px; }
.ph { color: var(--color-text-secondary); font-style: italic; font-size: 11.5px; }
.empty { text-align: center; color: var(--color-text-secondary); padding: 14px; }
.drawer { margin-top: 16px; border: 1px solid var(--color-border); border-radius: 6px; padding: 12px 14px; }
.drawer-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 8px; }
.drawer-head h3 { margin: 0; font-size: 15px; }
.sub { font-size: 12px; color: var(--color-text-secondary); margin-top: 3px; }
.ties { color: #2f7a3d; margin-left: 8px; font-weight: 600; }
.noties { color: #b52b27; margin-left: 8px; font-weight: 600; }
.loading-text { padding: 14px 0; color: var(--color-text-secondary); }
.sform { display: flex; flex-direction: column; gap: 12px; max-width: 520px; font-size: 12.5px; }
.sform label { display: flex; flex-direction: column; gap: 4px; }
.sform input, .sform select { padding: 4px 6px; border: 1px solid var(--color-border); border-radius: 4px;
  font-size: 12.5px; background: var(--color-surface); color: var(--color-text); }
.sform fieldset { border: 1px solid var(--color-border); border-radius: 6px; padding: 8px 10px;
  display: flex; flex-direction: column; gap: 6px; }
.sform label.radio { flex-direction: row; align-items: center; gap: 6px; }
.wide { width: 220px; }
.basis { font-size: 11.5px; color: var(--color-text-secondary); line-height: 1.45; }
.tick { width: 34px; text-align: center; }
td.pay { cursor: pointer; }
.pay-edit { display: inline-flex; gap: 4px; align-items: center; }
.pay-edit input, .pay-edit select { font-size: 12px; padding: 1px 4px; border: 1px solid var(--color-border); border-radius: 3px; }
.pay-edit .amt { width: 80px; text-align: right; }
.pay-edit .why { width: 160px; }
.mini { font-size: 11px; padding: 1px 6px; cursor: pointer; }
.adj { font-style: italic; color: #2b4c7e; }
.pay-form { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 8px; }
.pay-form input { font-size: 12.5px; padding: 4px 6px; border: 1px solid var(--color-border); border-radius: 4px; }
</style>
