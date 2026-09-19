<script setup lang="ts">
/**
 * GL / IA Query — the CFO's Spreadsheet Server filters, in the app.
 *
 * His workbook lists what he needs to vary: for GL, multiple entities, the period,
 * account(s) and export; for IA, multiple investment IDs, investor IDs, a date,
 * MajorType(s), SubType(s) and export. This is that, reading the copies of those
 * two MRI tables the app already imports rather than re-running his SQL.
 */
import { ref, computed, onMounted, watch } from 'vue'
import api from '@/api/client'
import MultiPicker from '@/components/common/MultiPicker.vue'

type Tab = 'gl' | 'ia'
const tab = ref<Tab>('gl')

const glOptions = ref<any>(null)
const iaOptions = ref<any>(null)
const result = ref<any>(null)
const loading = ref(false)
const exporting = ref(false)
const error = ref<string | null>(null)

// ---- GL filters ----
const glEntities = ref<string[]>([])
const glAccounts = ref<string[]>([])
const glBases = ref<string[]>([])
const glPeriodFrom = ref('')
const glPeriodTo = ref('')

// ---- IA filters ----
const iaInvestments = ref<string[]>([])
const iaInvestors = ref<string[]>([])
const iaMajorTypes = ref<string[]>([])
const iaSubTypes = ref<string[]>([])
const iaDateField = ref('TransactionDate')
const iaDateFrom = ref('')
const iaDateTo = ref('')

const options = computed(() => (tab.value === 'gl' ? glOptions.value : iaOptions.value))

// Every picker takes {id, label}. The label carries the name as well as the code so
// the search finds either — an accountant looks for "Eastchase", not "PPIECH".
const withName = (rows: any[]) => (rows || []).map((r: any) => ({
  id: r.id, label: r.name ? `${r.id} — ${r.name}` : r.id,
}))
const entityChoices = computed(() => withName(glOptions.value?.entities))
const accountChoices = computed(() => (glOptions.value?.accounts || []).map((a: any) => ({
  id: a.account, label: a.name ? `${a.account} — ${a.name}` : a.account,
})))
const basisChoices = computed(() =>
  (glOptions.value?.bases || []).map((b: string) => ({ id: b, label: b })))
const investmentChoices = computed(() => withName(iaOptions.value?.investments))
const investorChoices = computed(() => withName(iaOptions.value?.investors))
const majorChoices = computed(() =>
  (iaOptions.value?.major_types || []).map((m: string) => ({ id: m, label: m })))
const subChoices = computed(() => subTypeChoices.value.map((s: any) => ({
  id: s.sub_type, label: s.sub_type,
})))

// Sub types are offered per major type, because "Return of Capital" under
// Distribution is a different line from one under Contribution and a flat list
// would let the two be picked as though they were the same thing.
const subTypeChoices = computed(() => {
  const all = iaOptions.value?.sub_types || []
  if (!iaMajorTypes.value.length) return all
  return all.filter((s: any) => iaMajorTypes.value.includes(s.major_type))
})

async function loadOptions() {
  error.value = null
  try {
    const [g, i] = await Promise.all([
      api.get('/api/gl-ia-query/gl/options'),
      api.get('/api/gl-ia-query/ia/options'),
    ])
    glOptions.value = g.data
    iaOptions.value = i.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function glBody() {
  return {
    entities: glEntities.value,
    accounts: glAccounts.value,
    bases: glBases.value,
    period_from: glPeriodFrom.value || null,
    period_to: glPeriodTo.value || null,
  }
}
function iaBody() {
  return {
    investments: iaInvestments.value,
    investors: iaInvestors.value,
    major_types: iaMajorTypes.value,
    sub_types: iaSubTypes.value,
    date_field: iaDateField.value,
    date_from: iaDateFrom.value || null,
    date_to: iaDateTo.value || null,
  }
}

async function run() {
  loading.value = true
  error.value = null
  result.value = null
  try {
    const url = tab.value === 'gl' ? '/api/gl-ia-query/gl' : '/api/gl-ia-query/ia'
    const res = await api.post(url, tab.value === 'gl' ? glBody() : iaBody())
    result.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function exportExcel() {
  exporting.value = true
  error.value = null
  try {
    const url = tab.value === 'gl'
      ? '/api/gl-ia-query/gl/excel' : '/api/gl-ia-query/ia/excel'
    const res = await api.post(url, tab.value === 'gl' ? glBody() : iaBody(),
                               { responseType: 'blob' })
    const blob = new Blob([res.data])
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = tab.value === 'gl' ? 'GL_Detail.xlsx' : 'IA_Detail.xlsx'
    a.click()
    URL.revokeObjectURL(a.href)
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    exporting.value = false
  }
}

function clearFilters() {
  if (tab.value === 'gl') {
    glEntities.value = []; glAccounts.value = []; glBases.value = []
    glPeriodFrom.value = ''; glPeriodTo.value = ''
  } else {
    iaInvestments.value = []; iaInvestors.value = []
    iaMajorTypes.value = []; iaSubTypes.value = []
    iaDateFrom.value = ''; iaDateTo.value = ''
    iaDateField.value = 'TransactionDate'
  }
  result.value = null
}

// A result belongs to the filters that produced it. Leaving it on screen while the
// tab changes would show GL rows under IA headings.
watch(tab, () => { result.value = null; error.value = null })

// Dropping a major type must drop the sub types that only existed under it,
// otherwise a filter stays applied that the screen no longer offers.
watch(iaMajorTypes, () => {
  const allowed = new Set(subTypeChoices.value.map((s: any) => s.sub_type))
  iaSubTypes.value = iaSubTypes.value.filter(s => allowed.has(s))
})

function fmt(v: any, key: string): string {
  if (v === null || v === undefined || v === '') return ''
  if (key === 'AMT' || key === 'Amount') {
    const n = Number(v)
    if (isNaN(n)) return String(v)
    const s = Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2,
                                                   maximumFractionDigits: 2 })
    return n < 0 ? `(${s})` : s
  }
  if (key.toLowerCase().includes('date')) return String(v).slice(0, 10)
  return String(v)
}
function isNum(key: string) { return key === 'AMT' || key === 'Amount' }

const totalKey = computed(() => (tab.value === 'gl' ? 'AMT' : 'Amount'))
const totalValue = computed(() => result.value?.totals?.[totalKey.value])

onMounted(loadOptions)
</script>

<template>
  <div class="glia">
    <div class="header-row">
      <div>
        <h2>GL / IA Query</h2>
        <div class="subtitle">
          The two Spreadsheet Server queries with filters, run against the app's
          copy of the MRI tables.
        </div>
      </div>
      <div class="header-controls">
        <button class="btn-secondary" @click="clearFilters">Clear filters</button>
        <button class="btn-secondary" :disabled="exporting || !options?.available"
                @click="exportExcel">
          {{ exporting ? 'Exporting…' : 'Export to Excel' }}
        </button>
        <button class="btn-primary" :disabled="loading || !options?.available"
                @click="run">
          {{ loading ? 'Running…' : 'Run query' }}
        </button>
      </div>
    </div>

    <div v-if="error" class="error-banner">{{ error }}</div>

    <div class="tabs">
      <button :class="{ active: tab === 'gl' }" @click="tab = 'gl'">GL Detail</button>
      <button :class="{ active: tab === 'ia' }" @click="tab = 'ia'">IA Detail</button>
    </div>

    <!-- A table that was never imported is said plainly, with what to do about it,
         rather than rendering an empty picker that reads as "no data exists". -->
    <div v-if="options && !options.available" class="notice">
      {{ options.reason }}
    </div>

    <template v-else-if="options">
      <!-- ============ GL filters ============ -->
      <div v-if="tab === 'gl'" class="filters">
        <!-- Several entities AND several accounts in one query: tick as many as
             you like in each, and they narrow together. -->
        <MultiPicker v-model="glEntities" :options="entityChoices" label="Entities" />
        <MultiPicker v-model="glAccounts" :options="accountChoices" label="Accounts" />
        <div class="filter narrow">
          <label>Period from</label>
          <select v-model="glPeriodFrom">
            <option value="">(earliest)</option>
            <option v-for="p in options.periods" :key="'f' + p" :value="p">{{ p }}</option>
          </select>
          <label style="margin-top:10px">Period to</label>
          <select v-model="glPeriodTo">
            <option value="">(latest)</option>
            <option v-for="p in options.periods" :key="'t' + p" :value="p">{{ p }}</option>
          </select>
          <div class="hint">
            Our copy of the GL begins at period {{ options.period_floor }}.
          </div>
        </div>
        <MultiPicker v-model="glBases" :options="basisChoices" label="Basis"
                     :searchable="false" max-height="110px" />
      </div>

      <!-- ============ IA filters ============ -->
      <div v-else class="filters">
        <MultiPicker v-model="iaInvestments" :options="investmentChoices"
                     label="Investments" />
        <MultiPicker v-model="iaInvestors" :options="investorChoices"
                     label="Investors" />
        <div class="stack">
          <MultiPicker v-model="iaMajorTypes" :options="majorChoices"
                       label="Major types" :searchable="false" max-height="90px" />
          <MultiPicker v-model="iaSubTypes" :options="subChoices" label="Sub types"
                       :searchable="false" max-height="110px"
                       hint="Sub types are those belonging to the major types picked." />
        </div>
        <div class="filter narrow">
          <label>Date field</label>
          <select v-model="iaDateField">
            <option value="TransactionDate">Transaction Date</option>
            <option value="EffectiveDate">Effective Date</option>
          </select>
          <label style="margin-top:10px">From</label>
          <input type="date" v-model="iaDateFrom" />
          <label style="margin-top:10px">To</label>
          <input type="date" v-model="iaDateTo" />
          <!-- Said here rather than discovered later: his sheet uses "before", so a
               transaction dated exactly on the To date ties differently. -->
          <div class="hint">
            To is inclusive. The CFO's query uses <em>before</em> the date, so set
            To one day earlier to reproduce that figure exactly.
          </div>
        </div>
      </div>
    </template>

    <!-- ============ result ============ -->
    <div v-if="loading" class="loading-text">Running…</div>

    <template v-else-if="result && result.available">
      <div class="result-head">
        <div>
          <strong>{{ result.row_count.toLocaleString() }}</strong>
          row{{ result.row_count === 1 ? '' : 's' }}
          <template v-if="result.truncated">
            — showing the first {{ result.shown.toLocaleString() }}
          </template>
          <span v-if="totalValue !== undefined && totalValue !== null" class="total">
            Total {{ tab === 'gl' ? 'amount' : 'amount' }}:
            {{ fmt(totalValue, totalKey) }}
          </span>
        </div>
        <div class="asof" v-if="result.data_as_of">
          MRI refresh completed {{ String(result.data_as_of).slice(0, 19) }}
        </div>
        <div class="asof" v-else>
          No completed MRI refresh recorded — freshness unknown.
        </div>
      </div>

      <div v-for="(n, i) in result.notes" :key="i" class="notice">{{ n }}</div>

      <div v-if="!result.rows.length" class="notice">
        Nothing matched these filters.
      </div>
      <div v-else class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th v-for="c in result.columns" :key="c.key"
                  :class="{ num: isNum(c.key) }">{{ c.label }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(r, i) in result.rows" :key="i">
              <td v-for="c in result.columns" :key="c.key"
                  :class="{ num: isNum(c.key) }">{{ fmt(r[c.key], c.key) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<style scoped>
.glia { padding: 20px; }
.header-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.header-row h2 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; }
.header-controls { display: flex; gap: 8px; flex-wrap: wrap; }
.error-banner {
  margin: 12px 0; padding: 8px 12px; border-radius: 6px;
  background: #fdeaea; color: #8a1f1f; font-size: 13px;
}
.tabs { display: flex; gap: 6px; margin: 14px 0 12px; }
.tabs button {
  padding: 6px 14px; border: 1px solid var(--color-border); background: none;
  border-radius: 6px; cursor: pointer; font-size: 13px;
}
.tabs button.active {
  background: var(--color-primary, #2f6f4f); color: #fff;
  border-color: var(--color-primary, #2f6f4f);
}
.filters { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px; }
.filter { display: flex; flex-direction: column; min-width: 240px; flex: 1; }
.filter.narrow { min-width: 190px; flex: 0 0 auto; }
.stack { display: flex; flex-direction: column; gap: 12px; min-width: 210px; }
.filter label { font-size: 11px; text-transform: uppercase; letter-spacing: .03em;
  color: var(--color-text-secondary); margin-bottom: 4px; }
.cnt { text-transform: none; letter-spacing: 0; font-style: italic; }
.filter select, .filter input {
  border: 1px solid var(--color-border); border-radius: 4px;
  padding: 4px 6px; font-size: 12.5px; background: var(--color-surface);
  color: var(--color-text);
}
.hint { font-size: 11px; color: var(--color-text-secondary); margin-top: 6px; line-height: 1.35; }
.notice {
  margin: 8px 0; padding: 8px 12px; font-size: 12.5px; line-height: 1.45;
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: 6px;
}
.loading-text { padding: 16px 0; color: var(--color-text-secondary); }
.result-head {
  display: flex; justify-content: space-between; align-items: baseline;
  gap: 16px; flex-wrap: wrap; margin: 14px 0 6px; font-size: 13px;
}
.total { margin-left: 16px; font-variant-numeric: tabular-nums; }
.asof { font-size: 11.5px; color: var(--color-text-secondary); }
.table-scroll { overflow: auto; max-height: 62vh; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.data-table th {
  position: sticky; top: 0; z-index: 1; text-align: left;
  padding: 6px 9px; background: var(--color-surface);
  border-bottom: 2px solid var(--color-border);
  font-size: 11px; text-transform: uppercase; white-space: nowrap;
  color: var(--color-text-secondary);
}
.data-table td { padding: 4px 9px; border-bottom: 1px solid var(--color-border); white-space: nowrap; }
.data-table th.num, .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
