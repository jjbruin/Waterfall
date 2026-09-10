<script setup lang="ts">
import { onMounted, ref, computed, watch, nextTick } from 'vue'
import { useDataStore } from '../stores/data'
import { useDealsStore } from '../stores/deals'
import { useRoute } from 'vue-router'
import api from '../api/client'
import ReviewPanel from '../components/common/ReviewPanel.vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, TitleComponent } from 'echarts/components'

use([CanvasRenderer, BarChart, LineChart, GridComponent, TooltipComponent, LegendComponent, TitleComponent])

const data = useDataStore()
const deals = useDealsStore()
const route = useRoute()

// Review workflow state
const reviewStatus = ref<any>(null)
const reviewLoading = ref(false)

onMounted(async () => {
  if (data.deals.length === 0) await data.loadDeals()
  // Handle query params (from Review Tracking navigation)
  const qVcode = route.query.vcode as string
  const qQuarter = route.query.quarter as string
  if (qVcode) {
    deals.selectDeal(qVcode)
    if (qQuarter) selectedQuarter.value = qQuarter
    await loadOnePager(qVcode)
  }
})

// ============================================================
// Mode: 'single' (one deal) or 'batch' (investor batch)
// ============================================================
const mode = ref<'single' | 'batch'>('single')

// ============================================================
// Deal selection (single mode)
// ============================================================
const error = ref<string | null>(null)

async function onDealSelect(event: Event) {
  const vcode = (event.target as HTMLSelectElement).value
  if (!vcode) return
  deals.selectDeal(vcode)
  opData.value = null
  chartResult.value = null
  selectedQuarter.value = ''
  batchPages.value = []
  await loadOnePager(vcode)
}

// ============================================================
// Data (single mode)
// ============================================================
const opData = ref<Record<string, any> | null>(null)
const chartResult = ref<Record<string, any> | null>(null)
const selectedQuarter = ref('')
const loading = ref(false)

// Filter deals for the One Pager dropdown: when a quarter is selected,
// only show deals that were owned during that quarter (sale date >= quarter start).
const filteredDeals = computed(() => {
  if (!selectedQuarter.value) return data.deals
  // Parse quarter start from "YYYY-QN" format
  const parts = selectedQuarter.value.split('-Q')
  if (parts.length !== 2) return data.deals
  const qYear = parseInt(parts[0])
  const qNum = parseInt(parts[1])
  if (isNaN(qYear) || isNaN(qNum)) return data.deals
  const qStartMonth = (qNum - 1) * 3  // 0-based: Q1=0, Q2=3, Q3=6, Q4=9
  const quarterStart = new Date(qYear, qStartMonth, 1)

  return data.deals.filter((d: any) => {
    const isSold = d.Sale_Status?.toUpperCase() === 'SOLD' ||
                   d.Lifecycle?.trim().toUpperCase() === 'SOLD'
    if (!isSold) return true
    if (!d.Sale_Date) return false
    // Parse sale date — handle both ISO and US formats
    const sd = new Date(d.Sale_Date)
    if (isNaN(sd.getTime())) return false
    // Keep if sale date >= quarter start (owned during some part of quarter)
    return sd >= quarterStart
  })
})
const saving = ref(false)

// Editable comments
const econComments = ref('')
const businessPlanComments = ref('')
const accruedPrefComment = ref('')
const peCapComment = ref('')

function parseQuarter(q: string): [number, number] {
  const [yStr, qStr] = q.split('-')
  return [parseInt(yStr), parseInt(qStr.replace('Q', ''))]
}

function getMostRecentCompletedQuarter(quarters: string[]): string {
  const now = new Date()
  const curYear = now.getFullYear()
  const curQuarter = Math.ceil((now.getMonth() + 1) / 3)
  const completed = quarters.filter(q => {
    const [y, qn] = parseQuarter(q)
    return y < curYear || (y === curYear && qn < curQuarter)
  })
  if (!completed.length) return quarters[quarters.length - 1]
  return completed.reduce((a, b) => {
    const [ay, aq] = parseQuarter(a)
    const [by, bq] = parseQuarter(b)
    return (by > ay || (by === ay && bq > aq)) ? b : a
  })
}

async function loadOnePager(vcode: string) {
  loading.value = true
  try {
    const params: any = {}
    if (selectedQuarter.value) params.quarter = selectedQuarter.value
    // The chart window now ends at the selected quarter, so it can only be
    // requested once the quarter is known. On first load the quarter comes back
    // with the one-pager; after that it is already set and both go out together.
    const chartReq = selectedQuarter.value
      ? api.get(`/api/financials/${vcode}/one-pager/chart`, { params })
      : null
    // Awaited below — this only keeps a chart failure from surfacing as an
    // unhandled rejection if the one-pager request rejects first.
    chartReq?.catch(() => {})
    const opRes = await api.get(`/api/financials/${vcode}/one-pager`, { params })
    opData.value = opRes.data
    if (!selectedQuarter.value && opRes.data.available_quarters?.length) {
      selectedQuarter.value = getMostRecentCompletedQuarter(opRes.data.available_quarters)
    }
    const chartRes = await (chartReq ?? api.get(
      `/api/financials/${vcode}/one-pager/chart`,
      { params: selectedQuarter.value ? { quarter: selectedQuarter.value } : {} }))
    chartResult.value = chartRes.data
    const c = opRes.data.comments || {}
    econComments.value = c.econ_comments || ''
    businessPlanComments.value = c.business_plan_comments || ''
    accruedPrefComment.value = c.accrued_pref_comment || ''
    peCapComment.value = c.pe_cap_comment || ''
    // Load review status after quarter is known
    await loadReviewStatus()
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

async function refreshQuarter() {
  if (deals.currentVcode) {
    await loadOnePager(deals.currentVcode)
  }
}

async function saveComments() {
  if (!deals.currentVcode || !selectedQuarter.value) return
  saving.value = true
  try {
    await api.put(`/api/financials/${deals.currentVcode}/one-pager/comments`, {
      quarter: selectedQuarter.value,
      econ_comments: econComments.value,
      business_plan_comments: businessPlanComments.value,
      accrued_pref_comment: accruedPrefComment.value,
      pe_cap_comment: peCapComment.value,
    })
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    saving.value = false
  }
}

// ============================================================
// Review workflow
// ============================================================
async function loadReviewStatus() {
  if (!deals.currentVcode || !selectedQuarter.value) return
  reviewLoading.value = true
  try {
    const res = await api.get(`/api/reviews/${deals.currentVcode}/${selectedQuarter.value}`)
    reviewStatus.value = res.data
  } catch {
    reviewStatus.value = null
  } finally {
    reviewLoading.value = false
  }
}

const commentsLocked = computed(() => {
  if (viewingSnapshot.value) return true
  if (!reviewStatus.value) return false
  return !reviewStatus.value.is_editable
})

async function handleReviewSubmit(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/submit`,
      { note: note || undefined }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewApprove(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/approve`,
      { note: note || undefined }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewReturn(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/return`,
      { note }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleReviewNote(note: string) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/note`,
      { note }
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

async function handleAcknowledgeNote(noteId: number) {
  reviewLoading.value = true
  try {
    const res = await api.post(
      `/api/reviews/${deals.currentVcode}/${selectedQuarter.value}/notes/${noteId}/acknowledge`
    )
    reviewStatus.value = res.data
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    reviewLoading.value = false
  }
}

// ============================================================
// Batch mode
// ============================================================
interface Investor {
  investor_id: string
  name: string
  display: string
  deal_count: number
  vcodes: string[]
}

interface BatchPage {
  vcode: string
  data: Record<string, any> | null
  chart: Record<string, any> | null
  error?: string
}

const investors = ref<Investor[]>([])
const selectedInvestor = ref('')
const batchQuarter = ref('2026-Q2')
const batchPages = ref<BatchPage[]>([])
const batchLoading = ref(false)
const batchProgress = ref('')

async function loadInvestors() {
  try {
    const res = await api.get('/api/financials/one-pager/investors')
    investors.value = res.data.investors || []
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  }
}

function onModeChange() {
  if (mode.value === 'batch' && investors.value.length === 0) {
    loadInvestors()
  }
}

const selectedInvestorInfo = computed(() =>
  investors.value.find(i => i.investor_id === selectedInvestor.value)
)

async function loadBatch() {
  const inv = selectedInvestorInfo.value
  if (!inv) return
  batchLoading.value = true
  batchPages.value = []
  batchProgress.value = `Loading ${inv.vcodes.length} deals...`
  error.value = null
  try {
    const res = await api.post('/api/financials/one-pager/batch', {
      vcodes: inv.vcodes,
      quarter: batchQuarter.value || undefined,
    })
    batchPages.value = res.data.pages || []
    // Auto-set quarter from first page with available_quarters
    if (!batchQuarter.value) {
      for (const pg of batchPages.value) {
        const aq = pg.data?.available_quarters
        if (aq?.length) {
          batchQuarter.value = getMostRecentCompletedQuarter(aq)
          break
        }
      }
    }
    batchProgress.value = ''
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
    batchProgress.value = ''
  } finally {
    batchLoading.value = false
  }
}

async function refreshBatch() {
  if (selectedInvestorInfo.value) await loadBatch()
}

// ============================================================
// Shortcut accessors (single mode)
// ============================================================
const gen = computed(() => activeOpData.value?.general || {})
const cap = computed(() => activeOpData.value?.cap_stack || {})
const perf = computed(() => activeOpData.value?.property_performance || {})
const pe = computed(() => activeOpData.value?.pe_performance || {})

// ============================================================
// Formatting helpers
// ============================================================
function fmtMil(val: number | null | undefined): string {
  if (val == null || isNaN(val) || val === 0) return '—'
  return '$' + (val / 1_000_000).toFixed(1) + 'M'
}
function fmtMil0(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return '$' + (val / 1_000_000).toFixed(1) + 'M'
}
function fmtPct(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  const pct = val > 1 ? val : val * 100
  return pct.toFixed(1) + '%'
}
/**
 * Economic occupancy, whose unit is fixed BY CONTRACT: the backend always
 * sends percentage points (72.3 means 72.3%), never a decimal fraction.
 *
 * Deliberately not `fmtPct`. That one has to guess the unit from the magnitude
 * (`val > 1 ? val : val * 100`) because it also formats genuinely-decimal
 * fields — pe_coupon arrives as 0.08 and must print "8%". `fmtPct` is correct
 * for those and is left exactly as it is.
 *
 * The guess is wrong for occupancy in BOTH directions, and it is a guess about
 * a unit, which is the thing a formatter should never infer from the size of a
 * number:
 *
 *   -12.8468  ->  "-1284.7%"   a negative never satisfies `> 1`, so it was
 *                              scaled by 100. This is Jefferson Waters Creek's
 *                              YTD Budget cell.
 *     0.5     ->  "50%"        a genuinely low occupancy, silently centupled.
 *
 * Fixing the unit here rather than bounding the value keeps the page honest
 * about what the data says: -12.85% is what the calculation produced, and it
 * now prints as -12.85% instead of as a number nobody can act on. Whether that
 * figure OUGHT to be negative is a data question (Waters Creek's budget books
 * 51% of rental income to account 4043 during lease-up), not a display one.
 *
 * ONE decimal, matching every other percentage cell on the page, so that no
 * correctly-formatted value moves: 95.4 still prints "95.4%", not "95.40%".
 * The consequence is that Waters Creek's -12.8468 reads "-12.8%" rather than
 * "-12.85%" — the same figure at the page's standard precision. Switching to
 * two decimals would show "-12.85%" but would restate every occupancy cell on
 * the report, which is a wider change than this one is.
 */
/**
 * The Debt cell of the capitalisation stack.
 *
 * `debt_display` is null on a deal sold by the reported quarter: MRI stops the
 * balance sheet at disposal rather than writing the payoff down, so the last
 * period on file is a pre-sale balance that would otherwise print forever
 * (East Manchester, sold 2026-06-25, showed a Nov-2025 $9,641,912). The
 * Portfolio Snapshot has suppressed this for months; this keeps the two views
 * telling the same story.
 *
 * Tests for the KEY, not for a nullish value — a present null is the deliberate
 * suppression, an absent key is a snapshot frozen before the field existed and
 * must keep rendering its raw `debt` exactly as it always did. Same rule as
 * `debtCell` in SnapshotLoan.vue.
 */
function capDebtCell(c: any): number | null | undefined {
  return c && 'debt_display' in c ? c.debt_display : c?.debt
}
function fmtOcc(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  return val.toFixed(1) + '%'
}
function fmtPctInt(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return '—'
  const pct = val > 1 ? val : val * 100
  return Math.round(pct) + '%'
}
function fmtDscr(val: number | null | undefined): string {
  if (val == null || isNaN(val)) return ''
  return val.toFixed(2) + 'X'
}
function fmtDate(val: string | null | undefined): string {
  if (!val) return 'N/A'
  // Parse as local date parts to avoid UTC→local timezone shift (e.g. 2026-05-01 midnight UTC → 4/30 in US timezones)
  const m = String(val).match(/^(\d{4})-(\d{2})-(\d{2})/)
  if (m) return `${parseInt(m[2])}/${parseInt(m[3])}/${m[1]}`
  const d = new Date(val)
  if (isNaN(d.getTime())) return String(val)
  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`
}
function fmtVariance(actual: number | null | undefined, budget: number | null | undefined): string {
  if (actual == null || budget == null || budget === 0) return ''
  const pct = ((actual - budget) / Math.abs(budget)) * 100
  return Math.round(pct) + '%'
}
/**
 * Economic occupancy variance — actual less budget, in percentage POINTS.
 *
 * Not `fmtVariance`: that one is a percent-of-budget change and rounds to a
 * whole number, which is right for Revenue / Expenses / NOI. Occupancy is
 * already a percentage, so its variance is the plain difference at the same one
 * decimal as the two cells beside it.
 *
 * The `-0.0` guard is the whole reason this is a function. `toFixed` keeps the
 * sign of a value that rounds away to zero, so a variance too small to show at
 * one decimal still printed its minus sign. Mount Prospect Plaza at 26Q2 is the
 * live case: actual 95.6000, budget 95.6439, a variance of -0.0439 that reads
 * "-0.0%". A minus sign in front of a zero reads as a shortfall against budget
 * and the report is not claiming one — at the precision it prints, the two
 * figures are the same. Only the exact string is rewritten, so nothing that
 * rounds to a genuine figure moves: -0.06 still prints "-0.1%".
 *
 * "0.0%", not an em dash: both readings exist and their difference really is
 * zero. A dash would claim the variance is unknown.
 */
function fmtOccVariance(actual: number | null | undefined, budget: number | null | undefined): string {
  if (actual == null || budget == null) return ''
  const s = (actual - budget).toFixed(1)
  return (s === '-0.0' ? '0.0' : s) + '%'
}

// Property Performance table rows (single mode)
const perfRows = computed(() => {
  if (!perf.value || !perf.value.revenue) return []
  const p = perf.value
  return buildPerfRows(p)
})

function buildPerfRows(p: any) {
  return [
    { label: 'Economic Occ.', ytdA: fmtOcc(p.economic_occ?.ytd_actual), ytdB: fmtOcc(p.economic_occ?.ytd_budget),
      variance: fmtOccVariance(p.economic_occ?.ytd_actual, p.economic_occ?.ytd_budget),
      atClose: fmtOcc(p.economic_occ?.at_close), actualYE: fmtOcc(p.economic_occ?.actual_ye), uwYE: fmtOcc(p.economic_occ?.uw_ye) },
    { label: 'Revenue', ytdA: fmtMil(p.revenue?.ytd_actual), ytdB: fmtMil(p.revenue?.ytd_budget),
      variance: fmtVariance(p.revenue?.ytd_actual, p.revenue?.ytd_budget),
      atClose: fmtMil(p.revenue?.at_close), actualYE: fmtMil(p.revenue?.actual_ye), uwYE: fmtMil(p.revenue?.uw_ye) },
    { label: 'Expenses', ytdA: fmtMil(p.expenses?.ytd_actual), ytdB: fmtMil(p.expenses?.ytd_budget),
      variance: fmtVariance(p.expenses?.ytd_actual, p.expenses?.ytd_budget),
      atClose: fmtMil(p.expenses?.at_close), actualYE: fmtMil(p.expenses?.actual_ye), uwYE: fmtMil(p.expenses?.uw_ye), underline: true },
    { label: 'NOI', ytdA: fmtMil(p.noi?.ytd_actual), ytdB: fmtMil(p.noi?.ytd_budget),
      variance: fmtVariance(p.noi?.ytd_actual, p.noi?.ytd_budget),
      atClose: fmtMil(p.noi?.at_close), actualYE: fmtMil(p.noi?.actual_ye), uwYE: fmtMil(p.noi?.uw_ye) },
    { label: 'DSCR', ytdA: fmtDscr(p.dscr?.ytd_actual), ytdB: fmtDscr(p.dscr?.ytd_budget),
      variance: '',
      atClose: fmtDscr(p.dscr?.at_close), actualYE: fmtDscr(p.dscr?.actual_ye), uwYE: fmtDscr(p.dscr?.uw_ye) },
  ]
}

// As-of date from quarter string
function getAsOfDate(q: string): string {
  if (!q) return ''
  const [yearStr, qStr] = q.split('-')
  const qNum = parseInt(qStr.replace('Q', ''))
  const month = qNum * 3
  const lastDay = new Date(parseInt(yearStr), month, 0).getDate()
  return `${month}/${lastDay}/${yearStr}`
}

const asOfDate = computed(() => getAsOfDate(selectedQuarter.value))

// ============================================================
// Chart option builder
// ============================================================

/** Smallest "round" number >= v that divides into 5 clean axis ticks. */
function niceCeil(v: number) {
  if (!(v > 0)) return 0
  const mag = Math.pow(10, Math.floor(Math.log10(v / 5)))
  for (const s of [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) {
    if (s * mag * 5 >= v) return s * mag * 5
  }
  return Math.ceil(v)
}

/**
 * Right-axis (NOI, $M) bounds for the Occupancy vs. NOI chart.
 *
 * Occupancy is pinned 0-100 on the left, so a bar's top sits at occ/100 of the
 * plot height. Left to itself, ECharts auto-fits the right axis to the NOI data
 * — the peak NOI point lands near the top of the plot area and the auto min sits
 * above zero — so the lines render above the bar tops and read as "NOI exceeds
 * occupancy" even though the units are unrelated.
 *
 * So scale the axis from the window's own data (never hardcoded): the tallest
 * NOI point may reach at most `headroom` of the plot height, taken from the
 * shortest real occupancy bar in the window. Zero-occupancy slots are ignored —
 * a pre-close quarter is 0 on all three series by definition, so there is no bar
 * to sit under. The 0.45 floor keeps a lease-up window from crushing the lines
 * into the axis; scripts/onepager_chart_axis_check.py verifies per quarter that
 * no NOI point renders above its own bar.
 */
function noiAxisBounds(uw: (number | null)[], act: (number | null)[], occ: (number | null)[]) {
  const noi = [...uw, ...act].filter((v): v is number => v != null)
  const bars = occ.filter((v): v is number => v != null && v > 0)
  // No NOI anywhere — an empty frame. Fall through and the 0.05 floor below
  // would scale the axis 0.00-0.05, which is not wrong but reads as an
  // arbitrary magnification of nothing. A plain 0.00-1.00 gives five clean
  // 0.20 steps against the left axis's 0-100, so the two sets of gridlines
  // still line up and the empty chart looks deliberate rather than broken.
  if (!noi.length) return { min: 0, max: 1 }
  const maxNoi = noi.length ? Math.max(...noi) : 0
  const minNoi = noi.length ? Math.min(...noi) : 0
  const headroom = Math.min(0.95, Math.max(0.45, (bars.length ? Math.min(...bars) : 95) / 100))
  // 0 is always a meaningful reference, so it anchors whichever end the data leaves open
  const min = minNoi < 0 ? -niceCeil(-minNoi) : 0
  const top = Math.max(maxNoi, 0)
  // solve (top - min) / (max - min) <= headroom
  const max = niceCeil(Math.max(min + (top - min) / headroom, 0.05))
  return { min, max }
}

/**
 * The chart option. ALWAYS returns a frame — never null.
 *
 * It used to return null when there was no data, and the template rendered
 * "No chart data available." in place of the chart. A deal with nothing to plot
 * yet is not a deal with a broken chart: the axes, gridlines, labels and legend
 * are the report's structure and belong on the page whether or not there is a
 * line to draw on them. Four deals at 26Q2 print that message — Donald Lynch,
 * Jefferson Stephens, Fairview Heights and Citizen Storage — all new or
 * development deals with no ISBS rows and no occupancy readings yet.
 *
 * Missing series arrive as nulls, which ECharts simply does not plot, so the
 * empty case needs no special drawing path: same axes, same legend, no marks.
 *
 * `periods` is normally supplied even with no data (the backend builds the
 * calendar window), so the x-axis carries its ten quarter labels. The `?? []`
 * fallbacks cover the one case where it cannot — a chart request that failed
 * outright, which surfaces its own error banner separately.
 */
function buildChartOption(cr: Record<string, any> | null) {
  const labels = (cr?.periods ?? []).map((p: string) => {
    const m = p.match(/^Q(\d)\s+(\d{4})$/)
    return m ? `${m[2]}-Q${m[1]}` : p
  })
  const actualNoi = (cr?.actual_noi ?? []).map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const uwNoi = (cr?.uw_noi ?? []).map((v: number | null) => v != null ? +(v / 1_000_000).toFixed(2) : null)
  const occ = (cr?.occupancy ?? []).map((v: number | null) => v != null ? +v.toFixed(1) : null)
  const noiAxis = noiAxisBounds(uwNoi, actualNoi, occ)
  return {
    title: { text: 'Physical Occupancy vs. NOI', subtext: '($ Millions)', left: 'center', top: 0,
      textStyle: { fontSize: 13, fontWeight: 'bold' }, subtextStyle: { fontSize: 11 } },
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, textStyle: { fontSize: 10 } },
    grid: { left: 55, right: 55, top: 55, bottom: 45 },
    xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 10, rotate: 0 } },
    yAxis: [
      { type: 'value', name: '', position: 'left', min: 0, max: 100, interval: 20,
        axisLabel: { formatter: '{value}.0%', fontSize: 10 } },
      // 5 intervals on both axes keeps the two sets of gridlines on top of each other
      { type: 'value', name: '', position: 'right',
        min: noiAxis.min, max: noiAxis.max, interval: (noiAxis.max - noiAxis.min) / 5,
        axisLabel: { formatter: (v: number) => v.toFixed(2), fontSize: 10 } },
    ],
    series: [
      { name: 'Physical Occupancy', type: 'bar', yAxisIndex: 0, data: occ,
        itemStyle: { color: '#5B9BD5' }, barMaxWidth: 45,
        label: { show: true, position: 'top', formatter: (p: any) => p.value != null ? p.value.toFixed(1) + '%' : '', fontSize: 9 } },
      { name: 'NOI U/W', type: 'line', yAxisIndex: 1, data: uwNoi,
        lineStyle: { color: '#ED7D31', width: 2 }, itemStyle: { color: '#ED7D31' }, symbol: 'circle', symbolSize: 5 },
      { name: 'NOI ACT', type: 'line', yAxisIndex: 1, data: actualNoi,
        lineStyle: { color: '#A5A5A5', width: 2 }, itemStyle: { color: '#A5A5A5' }, symbol: 'circle', symbolSize: 5 },
    ],
  }
}

const chartOption = computed(() => buildChartOption(activeChartResult.value))

// ============================================================
// Approved Snapshot
// ============================================================
const viewingSnapshot = ref(false)
const snapshotMeta = ref<{ approved_by: string; approved_at: string } | null>(null)
const snapshotData = ref<Record<string, any> | null>(null)
const snapshotChart = ref<Record<string, any> | null>(null)

const hasApprovedSnapshot = computed(() => {
  return reviewStatus.value?.has_snapshot === true
})

// Saved live comment values for restoring after snapshot view
const savedComments = ref({ econ: '', bp: '', pref: '', peCap: '' })

async function toggleSnapshot() {
  if (viewingSnapshot.value) {
    // Switch back to live data — restore saved comments
    viewingSnapshot.value = false
    econComments.value = savedComments.value.econ
    businessPlanComments.value = savedComments.value.bp
    accruedPrefComment.value = savedComments.value.pref
    peCapComment.value = savedComments.value.peCap
    snapshotData.value = null
    snapshotChart.value = null
    snapshotMeta.value = null
    return
  }
  // Load snapshot
  if (!deals.currentVcode || !selectedQuarter.value) return
  // Save current live comments before switching
  savedComments.value = {
    econ: econComments.value,
    bp: businessPlanComments.value,
    pref: accruedPrefComment.value,
    peCap: peCapComment.value,
  }
  loading.value = true
  try {
    const res = await api.get(
      `/api/financials/${deals.currentVcode}/one-pager/snapshot`,
      { params: { quarter: selectedQuarter.value } }
    )
    snapshotData.value = res.data.snapshot.data
    snapshotChart.value = res.data.snapshot.chart
    snapshotMeta.value = { approved_by: res.data.approved_by, approved_at: res.data.approved_at }
    // Load snapshot comments into the refs
    const sc = res.data.snapshot.data?.comments || {}
    econComments.value = sc.econ_comments || ''
    businessPlanComments.value = sc.business_plan_comments || ''
    accruedPrefComment.value = sc.accrued_pref_comment || ''
    peCapComment.value = sc.pe_cap_comment || ''
    viewingSnapshot.value = true
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

// Active data source — snapshot or live
const activeOpData = computed(() => viewingSnapshot.value ? snapshotData.value : opData.value)
const activeChartResult = computed(() => viewingSnapshot.value ? snapshotChart.value : chartResult.value)

// ============================================================
// Print
// ============================================================
/**
 * NO RENDERED TIMESTAMP.
 *
 * This view used to print `M/D/YYYY, h:mm AM/PM` into the top-left corner of
 * every sheet, from a `.print-date` div that is `display:none` on screen and
 * `display:block !important` under `@media print`. Because it appeared only in
 * the printed file, and in exactly the format Chrome uses, it read as the
 * browser's own print header bleeding through — it was reported as such
 * ("9/3/2026, 2:39 PM" on Flats at Dorsett Ridge). It was never the browser's.
 * It was ours, and the browser's header has been suppressed since v421.
 *
 * The Portfolio Snapshot print view hit the identical confusion and removed its
 * copy for the identical reason (see PortfolioSnapshotPrintView.vue) — the
 * reference investor document carries no timestamp. This is that same removal.
 *
 * The browser's header is still suppressed three ways and none of them is this:
 * `@page { margin: 0 }` in App.vue leaves Chrome no room to draw one, the page
 * padding stands in for the margin, and the title is blanked below so a header
 * drawn anyway could not say "Waterfall XIRR".
 */
function printOnePager() {
  // Blank the page title so browser doesn't print "Waterfall XIRR" in the header
  const origTitle = document.title
  document.title = ' '
  nextTick(() => {
    window.print()
    document.title = origTitle
  })
}
</script>

<template>
  <div class="one-pager-page">
    <!-- Controls bar (hidden in print) -->
    <div class="controls-bar no-print">
      <!-- Mode toggle -->
      <div class="mode-toggle">
        <label>
          <input type="radio" value="single" v-model="mode" @change="onModeChange" /> Single Deal
        </label>
        <label>
          <input type="radio" value="batch" v-model="mode" @change="onModeChange" /> Batch by Investor
        </label>
      </div>

      <!-- Single deal controls -->
      <template v-if="mode === 'single'">
        <div class="deal-selector">
          <label>Deal:</label>
          <select :value="deals.currentVcode" @change="onDealSelect">
            <option value="">-- Choose a deal --</option>
            <option v-for="d in filteredDeals" :key="d.vcode" :value="d.vcode">
              {{ d.Investment_Name || d.vcode }}{{ (d.Sale_Status?.toUpperCase() === 'SOLD' || d.Lifecycle?.trim().toUpperCase() === 'SOLD') ? ' (Sold)' : '' }}
            </option>
          </select>
        </div>
        <div v-if="opData" class="quarter-selector">
          <label>Quarter:</label>
          <select v-model="selectedQuarter" @change="refreshQuarter">
            <option v-for="q in (opData.available_quarters || [])" :key="q" :value="q">{{ q }}</option>
          </select>
        </div>
        <button v-if="opData" class="btn btn-sm" @click="printOnePager">Print</button>
        <button v-if="opData && !commentsLocked && !viewingSnapshot" class="btn btn-sm btn-save" @click="saveComments" :disabled="saving">
          {{ saving ? 'Saving...' : 'Save Comments' }}
        </button>
        <button v-if="hasApprovedSnapshot" class="btn btn-sm" :class="{ 'btn-active': viewingSnapshot }" @click="toggleSnapshot">
          {{ viewingSnapshot ? 'View Live Data' : 'View Approved Version' }}
        </button>
      </template>

      <!-- Batch controls -->
      <template v-if="mode === 'batch'">
        <div class="deal-selector">
          <label>Investor:</label>
          <select v-model="selectedInvestor">
            <option value="">-- Choose an investor --</option>
            <option v-for="inv in investors" :key="inv.investor_id" :value="inv.investor_id">
              {{ inv.display }} ({{ inv.deal_count }} deals)
            </option>
          </select>
        </div>
        <div v-if="selectedInvestor" class="quarter-selector">
          <label>Quarter:</label>
          <input type="text" v-model="batchQuarter" placeholder="e.g. 2025-Q4" class="quarter-input" />
        </div>
        <button v-if="selectedInvestor" class="btn btn-sm" @click="loadBatch" :disabled="batchLoading">
          {{ batchLoading ? 'Loading...' : 'Load All' }}
        </button>
        <button v-if="batchPages.length" class="btn btn-sm" @click="printOnePager">Print All</button>
      </template>
    </div>

    <div v-if="error" class="error-banner no-print">
      {{ error }}
      <button @click="error = null">Dismiss</button>
    </div>

    <!-- Review Panel (single mode only) -->
    <ReviewPanel
      v-if="mode === 'single' && opData && selectedQuarter && !viewingSnapshot"
      :review="reviewStatus"
      :loading="reviewLoading"
      @submit="handleReviewSubmit"
      @approve="handleReviewApprove"
      @return="handleReviewReturn"
      @add-note="handleReviewNote"
      @acknowledge-note="handleAcknowledgeNote"
    />

    <!-- ==================== SINGLE DEAL MODE ==================== -->
    <template v-if="mode === 'single'">
      <div v-if="loading" class="loading">Loading one pager...</div>

      <template v-else-if="activeOpData">
        <!-- Snapshot banner -->
        <div v-if="viewingSnapshot && snapshotMeta" class="snapshot-banner no-print">
          Viewing approved snapshot — approved by {{ snapshotMeta.approved_by }} on {{ new Date(snapshotMeta.approved_at).toLocaleDateString() }}
        </div>

        <div class="op-sheet">
        <h1 class="op-title">{{ gen.investment_name || deals.currentVcode }}</h1>

        <!-- GENERAL INFORMATION -->
        <div class="section-header">GENERAL INFORMATION</div>
        <table class="info-table">
          <tbody>
            <tr>
              <td class="lbl">Partner:</td><td class="val">{{ gen.partner || '—' }}</td>
              <td class="lbl">Investment Strategy:</td><td class="val">{{ gen.investment_strategy || '—' }}</td>
            </tr>
            <tr>
              <td class="lbl">Location:</td><td class="val">{{ gen.location || '—' }}</td>
              <td class="lbl">Date Closed:</td><td class="val">{{ fmtDate(gen.date_closed) }}</td>
            </tr>
            <tr>
              <td class="lbl">Asset Type / Year Built:</td>
              <td class="val">{{ gen.asset_type || '—' }} | {{ gen.year_built || '—' }}</td>
              <td class="lbl">Underwritten Exit:</td><td class="val">{{ fmtDate(gen.anticipated_exit) }}</td>
            </tr>
            <tr>
              <td class="lbl"># Units / SF:</td>
              <td class="val">{{ gen.units ? gen.units.toLocaleString() : '—' }}{{ gen.sqft ? ' | ' + (gen.sqft >= 1000 ? Math.round(gen.sqft / 1000).toLocaleString() + 'K' : gen.sqft.toLocaleString()) : '' }}</td>
              <td class="lbl">Current Anticipated Exit:</td><td class="val">{{ fmtDate(gen.current_anticipated_exit) }}</td>
            </tr>
          </tbody>
        </table>

        <!-- CAPITALIZATION -->
        <div class="section-header">CAPITALIZATION / EXPOSURE / DEAL TERMS</div>
        <table class="info-table cap-table">
          <tbody>
            <tr>
              <td class="lbl">Purchase Price:</td><td class="val">{{ fmtMil(cap.purchase_price) }}</td>
              <td class="lbl cap-hdr" colspan="2">Capitalization</td><td></td>
            </tr>
            <tr>
              <td class="lbl">P.E. Coupon:</td><td class="val">{{ cap.pe_coupon ? fmtPct(cap.pe_coupon) : 'N/A' }}</td>
              <td class="lbl">Debt:</td><td class="val right">{{ fmtMil(capDebtCell(cap)) }}</td>
              <td class="val right">{{ fmtPctInt(cap.debt_pct) }}</td>
            </tr>
            <tr>
              <!-- != null, not truthiness: a nil participation (0) is a real
                   term and prints 0.0%; only an absent one prints N/A. -->
              <td class="lbl">P.E. Participation:</td><td class="val">{{ cap.pe_participation != null ? fmtPct(cap.pe_participation) : 'N/A' }}</td>
              <td class="lbl">Pref. Equity:</td><td class="val right">{{ fmtMil(cap.pref_equity) }}</td>
              <td class="val right">{{ fmtPctInt(cap.pref_equity_pct) }}</td>
            </tr>
            <tr>
              <td class="lbl">Loan Terms:</td><td class="val">{{ cap.loan_terms_str || 'N/A' }}</td>
              <td class="lbl">Partner Equity:</td><td class="val right">{{ fmtMil(cap.partner_equity) }}</td>
              <td class="val right">{{ fmtPctInt(cap.partner_equity_pct) }}</td>
            </tr>
            <tr>
              <td class="lbl">2nd Loan Terms:</td><td class="val">{{ cap.second_loan_terms_str || 'N/A' }}</td>
              <td class="lbl">Total Cap:</td><td class="val right">{{ fmtMil(cap.total_cap) }}</td><td></td>
            </tr>
            <tr>
              <td class="lbl">Rate Cap:</td><td class="val">{{ cap.rate_cap || 'N/A' }}</td>
              <td class="lbl">{{ cap.valuation_year || '' }} Valuation:</td>
              <td class="val right">{{ fmtMil(cap.current_valuation) }}</td><td></td>
            </tr>
            <tr>
              <td class="lbl">P.E. Yield on Exposure:</td><td class="val">{{ cap.pe_yield_on_exposure ? fmtPct(cap.pe_yield_on_exposure) : 'N/A' }}</td>
              <td class="lbl">P.E. Expos. on Total Cap:</td><td></td>
              <td class="val right">{{ fmtPct(cap.pe_exposure_on_cap) }}</td>
            </tr>
            <tr>
              <td class="lbl">Pref Equity capitalization:</td>
              <!-- `rows="1"` with `overflow: hidden` shows exactly one line, so
                   anything that wraps was not printed at all — 7 deals lose the
                   tail of their ownership split, e.g. Nottingham Village's
                   "KOC 44%, TIAA 41%, PSC 13%, Declaration 2%". Same print-only
                   twin the Business Plan and the two comment blocks use. -->
              <td class="val">
                <textarea v-model="peCapComment" class="inline-comment print-hide" rows="1" placeholder="" spellcheck="true" lang="en" :readonly="commentsLocked"></textarea>
                <div class="pecap-print-text print-only">{{ peCapComment }}</div>
              </td>
              <td class="lbl">P.E. Expos. on {{ cap.valuation_year ? cap.valuation_year.slice(-2) : '' }} Value:</td>
              <td></td><td class="val right">{{ fmtPct(cap.pe_exposure_on_value) }}</td>
            </tr>
          </tbody>
        </table>

        <!-- PROPERTY PERFORMANCE -->
        <div class="section-header">PROPERTY PERFORMANCE</div>
        <table class="perf-table">
          <thead>
            <tr>
              <th></th>
              <th class="sub-header" colspan="3"><span class="sub-label">Annual Financial Comparison</span></th>
              <th class="spacer-col"></th>
              <th class="sub-header" colspan="3"><span class="sub-label">As of: {{ asOfDate }}</span></th>
            </tr>
            <tr>
              <th></th><th class="col-hdr">At Close</th><th class="col-hdr">Projected YE</th><th class="col-hdr">U/W YE</th>
              <th class="spacer-col"></th><th class="col-hdr">YTD (Actual)</th><th class="col-hdr">YTD (Budget)</th><th class="col-hdr">Variance</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in perfRows" :key="row.label" :class="{ 'underline-row': row.underline }">
              <td class="row-label">{{ row.label }}</td>
              <td class="val right">{{ row.atClose }}</td><td class="val right">{{ row.actualYE }}</td>
              <td class="val right">{{ row.uwYE }}</td><td class="spacer-col"></td>
              <td class="val right">{{ row.ytdA }}</td><td class="val right">{{ row.ytdB }}</td>
              <td class="val right">{{ row.variance }}</td>
            </tr>
          </tbody>
        </table>

        <!-- Comments -->
        <table class="comments-row-table">
          <tbody><tr>
            <td class="lbl" style="vertical-align: top; width: 80px;">Comments:</td>
            <!-- A textarea does not grow to its content, so in print it shows
                 only its `rows` and silently drops the rest — Dorsett Ridge lost
                 the second half of a 458-character comment, and 33 of 45 deals
                 carry more than three rows' worth. Same print-only twin the
                 Business Plan has had all along. -->
            <td>
              <textarea v-model="econComments" class="comment-input print-hide" rows="3" placeholder="Property performance comments..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea>
              <div class="econ-print-text print-only">{{ econComments }}</div>
            </td>
          </tr></tbody>
        </table>

        <!-- PE PERFORMANCE -->
        <div class="section-header">PREFERRED EQUITY PERFORMANCE</div>
        <table class="pe-table">
          <tbody>
            <tr>
              <td class="lbl">Committed Pref Equity:</td><td class="val">{{ fmtMil0(pe.committed_pe) }}</td>
              <td class="lbl">Coupon:</td><td class="val">{{ pe.coupon ? fmtPct(pe.coupon) : 'N/A' }}</td><td></td><td></td>
            </tr>
            <tr>
              <td class="lbl">Remaining to Fund:</td><td class="val">{{ fmtMil0(pe.remaining_to_fund) }}</td>
              <td class="lbl">Participation:</td><td class="val">{{ pe.participation != null ? fmtPct(pe.participation) : 'N/A' }}</td><td></td><td></td>
            </tr>
            <tr><td colspan="6" style="height: 6px;"></td></tr>
            <tr>
              <td class="lbl">Funded to Date:</td><td class="val">{{ fmtMil0(pe.funded_to_date) }}</td>
              <td></td><td></td><td></td><td></td>
            </tr>
            <tr>
              <td class="lbl">Return of Capital:</td><td class="val">{{ fmtMil0(pe.return_of_capital) }}</td>
              <td class="lbl">ROE to Date:</td><td class="val">{{ pe.roe_to_date ? fmtPct(pe.roe_to_date) : '—' }}</td>
              <td class="lbl">U/W ROE to Date:</td><td class="val">{{ pe.uw_roe_to_date ? fmtPct(pe.uw_roe_to_date) : '—' }}</td>
            </tr>
            <tr>
              <td class="lbl">Current Pref Equity Balance:</td><td class="val">{{ fmtMil0(pe.current_pe_balance) }}</td>
              <td class="lbl">Accrued Balance:</td><td class="val">{{ fmtMil0(pe.accrued_balance) }}</td>
              <td colspan="2">
                <textarea v-model="accruedPrefComment" class="comment-input small print-hide" rows="2" placeholder="Accrued pref comment..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea>
                <div class="pref-print-text print-only">{{ accruedPrefComment }}</div>
              </td>
            </tr>
          </tbody>
        </table>

        <!-- BUSINESS PLAN -->
        <div class="section-header">BUSINESS PLAN &amp; UPDATES</div>
        <div class="bp-section">
          <textarea v-model="businessPlanComments" class="comment-input bp-input print-hide" rows="6" placeholder="Business plan and updates..." spellcheck="true" lang="en" :readonly="commentsLocked"></textarea>
          <div class="bp-print-text print-only">{{ businessPlanComments }}</div>
        </div>

        <!-- CHART -->
        <div class="chart-section">
          <!-- No v-if / no "no data" fallback: buildChartOption always returns
               a frame, and a deal with nothing to plot shows empty axes rather
               than a message where the chart should be. -->
          <v-chart :option="chartOption" style="width: 100%; height: 170px;" autoresize />
        </div>
      </div>
      </template>

      <p v-else-if="!loading" class="empty no-print">Select a deal to view the one pager.</p>
    </template>

    <!-- ==================== BATCH MODE ==================== -->
    <template v-if="mode === 'batch'">
      <div v-if="batchLoading" class="loading">{{ batchProgress || 'Loading...' }}</div>

      <p v-else-if="!batchPages.length && selectedInvestor" class="empty no-print">
        Select an investor and click "Load All" to generate one pagers for all {{ selectedInvestorInfo?.deal_count }} deals.
      </p>
      <p v-else-if="!selectedInvestor" class="empty no-print">
        Select an investor to generate batch one pagers.
      </p>

      <!-- Batch pages — one op-sheet per deal, page-break between -->
      <template v-for="(pg, idx) in batchPages" :key="pg.vcode">
        <div v-if="pg.data" class="op-sheet" :class="{ 'page-break': idx < batchPages.length - 1 }">
          <h1 class="op-title">{{ pg.data.general?.investment_name || pg.vcode }}</h1>

          <!-- GENERAL INFORMATION -->
          <div class="section-header">GENERAL INFORMATION</div>
          <table class="info-table">
            <tbody>
              <tr>
                <td class="lbl">Partner:</td><td class="val">{{ pg.data.general?.partner || '—' }}</td>
                <td class="lbl">Investment Strategy:</td><td class="val">{{ pg.data.general?.investment_strategy || '—' }}</td>
              </tr>
              <tr>
                <td class="lbl">Location:</td><td class="val">{{ pg.data.general?.location || '—' }}</td>
                <td class="lbl">Date Closed:</td><td class="val">{{ fmtDate(pg.data.general?.date_closed) }}</td>
              </tr>
              <tr>
                <td class="lbl">Asset Type / Year Built:</td>
                <td class="val">{{ pg.data.general?.asset_type || '—' }} | {{ pg.data.general?.year_built || '—' }}</td>
                <td class="lbl">Underwritten Exit:</td><td class="val">{{ fmtDate(pg.data.general?.anticipated_exit) }}</td>
              </tr>
              <tr>
                <td class="lbl"># Units / SF:</td>
                <td class="val">{{ pg.data.general?.units ? pg.data.general.units.toLocaleString() : '—' }}{{ pg.data.general?.sqft ? ' | ' + (pg.data.general.sqft >= 1000 ? Math.round(pg.data.general.sqft / 1000).toLocaleString() + 'K' : pg.data.general.sqft.toLocaleString()) : '' }}</td>
                <td class="lbl">Current Anticipated Exit:</td><td class="val">{{ fmtDate(pg.data.general?.current_anticipated_exit) }}</td>
              </tr>
            </tbody>
          </table>

          <!-- CAPITALIZATION -->
          <div class="section-header">CAPITALIZATION / EXPOSURE / DEAL TERMS</div>
          <table class="info-table cap-table">
            <tbody>
              <tr>
                <td class="lbl">Purchase Price:</td><td class="val">{{ fmtMil(pg.data.cap_stack?.purchase_price) }}</td>
                <td class="lbl cap-hdr" colspan="2">Capitalization</td><td></td>
              </tr>
              <tr>
                <td class="lbl">P.E. Coupon:</td><td class="val">{{ pg.data.cap_stack?.pe_coupon ? fmtPct(pg.data.cap_stack.pe_coupon) : 'N/A' }}</td>
                <td class="lbl">Debt:</td><td class="val right">{{ fmtMil(capDebtCell(pg.data.cap_stack)) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.debt_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">P.E. Participation:</td><td class="val">{{ pg.data.cap_stack?.pe_participation != null ? fmtPct(pg.data.cap_stack.pe_participation) : 'N/A' }}</td>
                <td class="lbl">Pref. Equity:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.pref_equity) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.pref_equity_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">Loan Terms:</td><td class="val">{{ pg.data.cap_stack?.loan_terms_str || 'N/A' }}</td>
                <td class="lbl">Partner Equity:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.partner_equity) }}</td>
                <td class="val right">{{ fmtPctInt(pg.data.cap_stack?.partner_equity_pct) }}</td>
              </tr>
              <tr>
                <td class="lbl">2nd Loan Terms:</td><td class="val">{{ pg.data.cap_stack?.second_loan_terms_str || 'N/A' }}</td>
                <td class="lbl">Total Cap:</td><td class="val right">{{ fmtMil(pg.data.cap_stack?.total_cap) }}</td><td></td>
              </tr>
              <tr>
                <td class="lbl">Rate Cap:</td><td class="val">{{ pg.data.cap_stack?.rate_cap || 'N/A' }}</td>
                <td class="lbl">{{ pg.data.cap_stack?.valuation_year || '' }} Valuation:</td>
                <td class="val right">{{ fmtMil(pg.data.cap_stack?.current_valuation) }}</td><td></td>
              </tr>
              <tr>
                <td class="lbl">P.E. Yield on Exposure:</td><td class="val">{{ pg.data.cap_stack?.pe_yield_on_exposure ? fmtPct(pg.data.cap_stack.pe_yield_on_exposure) : 'N/A' }}</td>
                <td class="lbl">P.E. Expos. on Total Cap:</td><td></td>
                <td class="val right">{{ fmtPct(pg.data.cap_stack?.pe_exposure_on_cap) }}</td>
              </tr>
              <tr>
                <td class="lbl">Pref Equity capitalization:</td><td class="val">{{ pg.data.comments?.pe_cap_comment || '' }}</td>
                <td class="lbl">P.E. Expos. on {{ pg.data.cap_stack?.valuation_year ? pg.data.cap_stack.valuation_year.slice(-2) : '' }} Value:</td>
                <td></td><td class="val right">{{ fmtPct(pg.data.cap_stack?.pe_exposure_on_value) }}</td>
              </tr>
            </tbody>
          </table>

          <!-- PROPERTY PERFORMANCE -->
          <div class="section-header">PROPERTY PERFORMANCE</div>
          <table class="perf-table">
            <thead>
              <tr>
                <th></th>
                <th class="sub-header" colspan="3"><span class="sub-label">Annual Financial Comparison</span></th>
                <th class="spacer-col"></th>
                <th class="sub-header" colspan="3"><span class="sub-label">As of: {{ getAsOfDate(batchQuarter) }}</span></th>
              </tr>
              <tr>
                <th></th><th class="col-hdr">At Close</th><th class="col-hdr">Projected YE</th><th class="col-hdr">U/W YE</th>
                <th class="spacer-col"></th><th class="col-hdr">YTD (Actual)</th><th class="col-hdr">YTD (Budget)</th><th class="col-hdr">Variance</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in buildPerfRows(pg.data.property_performance || {})" :key="row.label" :class="{ 'underline-row': row.underline }">
                <td class="row-label">{{ row.label }}</td>
                <td class="val right">{{ row.atClose }}</td><td class="val right">{{ row.actualYE }}</td>
                <td class="val right">{{ row.uwYE }}</td><td class="spacer-col"></td>
                <td class="val right">{{ row.ytdA }}</td><td class="val right">{{ row.ytdB }}</td>
                <td class="val right">{{ row.variance }}</td>
              </tr>
            </tbody>
          </table>

          <!-- Comments (read-only in batch) -->
          <table class="comments-row-table">
            <tbody><tr>
              <td class="lbl" style="vertical-align: top; width: 80px;">Comments:</td>
              <td class="comment-text">{{ pg.data.comments?.econ_comments || '' }}</td>
            </tr></tbody>
          </table>

          <!-- PE PERFORMANCE -->
          <div class="section-header">PREFERRED EQUITY PERFORMANCE</div>
          <table class="pe-table">
            <tbody>
              <tr>
                <td class="lbl">Committed Pref Equity:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.committed_pe) }}</td>
                <td class="lbl">Coupon:</td><td class="val">{{ pg.data.pe_performance?.coupon ? fmtPct(pg.data.pe_performance.coupon) : 'N/A' }}</td><td></td><td></td>
              </tr>
              <tr>
                <td class="lbl">Remaining to Fund:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.remaining_to_fund) }}</td>
                <td class="lbl">Participation:</td><td class="val">{{ pg.data.pe_performance?.participation != null ? fmtPct(pg.data.pe_performance.participation) : 'N/A' }}</td><td></td><td></td>
              </tr>
              <tr><td colspan="6" style="height: 6px;"></td></tr>
              <tr>
                <td class="lbl">Funded to Date:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.funded_to_date) }}</td>
                <td></td><td></td><td></td><td></td>
              </tr>
              <tr>
                <td class="lbl">Return of Capital:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.return_of_capital) }}</td>
                <td class="lbl">ROE to Date:</td><td class="val">{{ pg.data.pe_performance?.roe_to_date ? fmtPct(pg.data.pe_performance.roe_to_date) : '—' }}</td>
                <td class="lbl">U/W ROE to Date:</td><td class="val">{{ pg.data.pe_performance?.uw_roe_to_date ? fmtPct(pg.data.pe_performance.uw_roe_to_date) : '—' }}</td>
              </tr>
              <tr>
                <td class="lbl">Current Pref Equity Balance:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.current_pe_balance) }}</td>
                <td class="lbl">Accrued Balance:</td><td class="val">{{ fmtMil0(pg.data.pe_performance?.accrued_balance) }}</td>
                <td colspan="2" class="comment-text" style="font-size: 9px;">{{ pg.data.comments?.accrued_pref_comment || '' }}</td>
              </tr>
            </tbody>
          </table>

          <!-- BUSINESS PLAN -->
          <div class="section-header">BUSINESS PLAN &amp; UPDATES</div>
          <div class="bp-section">
            <div class="comment-text bp-text print-hide">{{ pg.data.comments?.business_plan_comments || '' }}</div>
            <div class="bp-print-text print-only">{{ pg.data.comments?.business_plan_comments || '' }}</div>
          </div>

          <!-- CHART -->
          <div class="chart-section">
            <!-- Same as single mode: always a frame, never a message. -->
            <v-chart :option="buildChartOption(pg.chart)" style="width: 100%; height: 170px;" autoresize />
          </div>
        </div>

        <!-- Error page -->
        <div v-else class="op-sheet page-break">
          <h1 class="op-title">{{ pg.vcode }}</h1>
          <p class="empty">{{ pg.error || 'Failed to load data' }}</p>
        </div>
      </template>
    </template>
  </div>
</template>

<style scoped>
/* ============================================================
   SCREEN STYLES
   ============================================================ */
.one-pager-page {
  max-width: 960px;
  margin: 0 auto;
  font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
  font-size: 11px;
  color: #000;
  line-height: 1.35;
}

/* Controls bar */
.controls-bar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.mode-toggle {
  display: flex;
  gap: 12px;
  font-size: 13px;
  padding-right: 8px;
  border-right: 1px solid #ccc;
}
.mode-toggle label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  font-weight: 500;
}
.deal-selector, .quarter-selector {
  display: flex;
  align-items: center;
  gap: 8px;
}
.deal-selector label, .quarter-selector label {
  font-size: 13px;
  font-weight: 500;
}
.deal-selector select, .quarter-selector select {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 13px;
  min-width: 280px;
}
.quarter-selector select { min-width: 120px; }
.quarter-input {
  padding: 6px 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 13px;
  width: 120px;
}
.btn { padding: 6px 16px; border: none; border-radius: 4px; background: #1F4E79; color: white; cursor: pointer; font-size: 12px; }
.btn:hover { opacity: 0.9; }
.btn-sm { padding: 5px 12px; }
.btn-save { background: #548235; }
.btn-save:disabled { opacity: 0.5; cursor: default; }
.snapshot-banner { background: #eff6ff; border: 1px solid #93c5fd; color: #1e40af; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; font-size: 13px; text-align: center; font-weight: 500; }
.btn-active { background: #1e40af !important; color: #fff !important; }
.error-banner { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 8px 14px; border-radius: 6px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; }
.error-banner button { background: none; border: 1px solid #fca5a5; color: #991b1b; padding: 3px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
.loading { text-align: center; padding: 40px; color: #666; font-style: italic; font-size: 14px; }
.empty { text-align: center; padding: 40px; color: #999; font-style: italic; font-size: 14px; }

/* ============================================================
   ONE PAGER SHEET (printable area)
   ============================================================ */
.op-sheet {
  background: #fff;
  border: 1px solid #ddd;
  padding: 20px 28px;
  margin-bottom: 16px;
}

/* Print date — hidden on screen */
/* Title */
.op-title {
  text-align: center;
  font-size: 20px;
  font-weight: 700;
  margin: 0 0 2px 0;
  border-bottom: 2px solid #000;
  padding-bottom: 4px;
}

/* Section headers */
.section-header {
  text-align: center;
  font-weight: 700;
  font-size: 11px;
  border-bottom: 1px solid #000;
  padding: 5px 0 2px 0;
  margin: 2px 0 2px 0;
}

/* Generic info table */
.info-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.info-table td {
  padding: 1px 6px 1px 0;
  vertical-align: top;
}
.info-table .lbl {
  font-weight: 700;
  font-style: italic;
  white-space: nowrap;
  width: 22%;
}
.info-table .val {
  width: 28%;
}

/* Cap table overrides */
.cap-table .lbl { width: 18%; }
.cap-table .val { width: 18%; }
.cap-table .cap-hdr {
  font-weight: 700;
  font-style: italic;
  text-decoration: underline;
}
.cap-table .right { text-align: right; }
.cap-table td:nth-child(3) { padding-left: 20px; }

/* Performance table */
.perf-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.perf-table th, .perf-table td {
  padding: 1px 6px;
  font-size: 11px;
}
.perf-table .sub-header {
  text-align: center;
  border-bottom: none;
  padding-bottom: 0;
}
.perf-table .sub-label {
  font-weight: 400;
  font-size: 10px;
  font-style: italic;
}
.perf-table .col-hdr {
  text-align: right;
  font-weight: 700;
  font-size: 10px;
  text-decoration: underline;
  padding-bottom: 2px;
}
.perf-table .row-label {
  font-weight: 700;
  white-space: nowrap;
  padding-right: 12px;
}
.perf-table .val { font-size: 11px; }
.perf-table .right { text-align: right; }
.perf-table .spacer-col { width: 20px; }
.perf-table tr.underline-row td { border-bottom: 1px solid #000; }

/* Comments row */
.comments-row-table {
  width: 100%;
  border-collapse: collapse;
  margin: 2px 0;
}
.comments-row-table td { padding: 2px 4px; }
.comment-input {
  width: 100%;
  border: 1px solid #ccc;
  border-radius: 3px;
  padding: 4px 6px;
  font-family: inherit;
  font-size: 10px;
  line-height: 1.35;
  resize: vertical;
  color: #000;
}
.comment-input.small { font-size: 9px; }
.inline-comment {
  width: 100%;
  border: none;
  padding: 0;
  font-family: inherit;
  font-size: 10px;
  line-height: 1.35;
  resize: none;
  background: transparent;
  outline: none;
  overflow: hidden;
}
.comment-text {
  font-size: 10px;
  line-height: 1.35;
  white-space: pre-wrap;
  min-height: 2.8em;
  border-bottom: 1px solid #eee;
}
.comment-text.bp-text {
  min-height: 5.5em;
}

/* PE table */
.pe-table {
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 0;
}
.pe-table td {
  padding: 1px 6px 1px 0;
  vertical-align: top;
  font-size: 11px;
}
.pe-table .lbl {
  font-weight: 700;
  font-style: italic;
  white-space: nowrap;
}
.pe-table .val { }

/* Business plan section */
.bp-section {
  margin: 2px 0;
}
.bp-input {
  min-height: 80px;
}
.bp-print-text,
.econ-print-text,
.pref-print-text,
.pecap-print-text {
  display: none;
}
.print-only {
  display: none;
}

/* Chart */
.chart-section {
  margin-top: 6px;
  border-top: 1px solid #ccc;
  padding-top: 4px;
}

/* ============================================================
   PRINT STYLES
   ============================================================ */
@media print {
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

  .no-print, .print-hide { display: none !important; }
  .print-only { display: block !important; }

  /* The page box (letter portrait, zero margin) is set ONCE globally in
     App.vue — see the note there. The 0.4in/0.5in padding below is this view's
     real margin. */

  body, html {
    margin: 0 !important;
    padding: 0 !important;
    font-size: 10px !important;
  }

  /* Content padding replaces @page margin (keeps headers/footers off the page) */
  .one-pager-page {
    max-width: none;
    margin: 0;
    padding: 0.4in 0.5in !important;
  }

  .op-sheet {
    border: none;
    padding: 0;
    box-shadow: none;
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
    /* EXACTLY one page, and nothing is clipped to achieve it.
       ------------------------------------------------------------------
       Three requirements that look contradictory: one page per deal, every
       character rendered, and no second page. They are reconcilable because
       the CHART is the only thing that does not fit — measured across all 45
       narrative deals, the tallest TEXT on any of them ends at 664pt of the
       734.4pt box (Poplar Prairie), so the words always fit on their own.
       So the chart comes OUT of the flow (see .chart-section) and is pinned to
       the foot of the sheet. Flowed content is then text only, which fits, so
       the fixed height below can never spill onto a second page and never has
       to clip anything to avoid one.
       This replaces `min-height`, which was correct for "never lose a word" but
       let the sheet grow to two pages when the chart could not fit. The earlier
       `height` + `overflow: hidden` was the opposite error: one page, bought by
       silently deleting the narrative. This is one page AND every word. */
    position: relative;
    height: calc(100vh - 0.8in);
    overflow: visible;
  }

  .op-sheet.page-break {
    page-break-after: always;
  }

  /* Print-only date/time in upper left */
  .op-title { font-size: 16px; margin-bottom: 1px !important; padding-bottom: 2px !important; }

  /* Uniform tight spacing between all sections */
  .section-header {
    font-size: 11px;
    padding: 2px 0 1px 0 !important;
    margin: 1px 0 1px 0 !important;
  }

  /* Tighten comments row between Property Performance and PE section */
  .comments-row-table { margin: 0 !important; }
  .comments-row-table td { padding: 1px 4px !important; }

  /* Remove internal spacer row in PE table */
  .pe-table tr td[style*="height"] { height: 0px !important; padding: 0 !important; }

  .info-table td, .cap-table td, .perf-table th, .perf-table td, .pe-table td {
    font-size: 10.7px;
    padding: 0.5px 3px 0.5px 0;
  }
  .info-table, .pe-table { margin-bottom: 0 !important; }
  .perf-table { margin-bottom: 0 !important; }

  /* Tighten label widths to prevent wrapping at larger font */
  .info-table .lbl { width: 20% !important; }
  .info-table .val { width: 30% !important; }
  .cap-table .lbl { width: 16% !important; }
  .cap-table .val { width: 16% !important; }
  .cap-table td:nth-child(3) { padding-left: 12px !important; }
  .pe-table td { padding: 1px 4px 1px 0 !important; }
  .perf-table .spacer-col { width: 12px !important; }
  .perf-table .row-label { padding-right: 6px !important; }

  /* Make textareas and comment text look like plain text in print */
  .comment-input,
  .comment-text {
    border: none !important;
    border-bottom: none !important;
    padding: 0 !important;
    resize: none !important;
    background: transparent !important;
    font-size: 10.7px !important;
    overflow: visible !important;
    height: auto !important;
    min-height: 0 !important;
  }

  /* Business plan: print all of it. It is narrative an asset manager wrote for
     an investor, and dropping the end of it silently is worse than a second
     page.
     ------------------------------------------------------------------------
     This block used to read "fill remaining space, clip if too long", and it
     did the clipping with `flex: 1 1 auto; min-height: 0; overflow: hidden`
     inside a sheet fixed at exactly one page. Two things made that far more
     destructive than "if too long" suggests:
       * the chart below is `flex-shrink: 0` at a fixed 300px, and everything
         above is fixed too, so .bp-section was the ONLY flexible item and
         absorbed the whole shortfall;
       * `min-height: 0` let it shrink below its content, and `overflow:
         hidden` then painted none of it.
     The result was not a trimmed tail. Measured on the real print path, the
     block collapsed to between zero and half a line on every deal that has a
     narrative — Belleville's 292 characters printed nothing at all, and Flats
     at Dorsett Ridge showed one horizontally-sliced line. 45 of 56 active
     deals carry a narrative and all of it was missing from the paper.
     `flex: 0 0 auto` stops it being shrunk below its content, and with the
     sheet on min-height the page grows instead. */
  .bp-section {
    flex: 0 0 auto;
    overflow: visible !important;
    /* Above the out-of-flow chart, so an overlapping narrative stays legible. */
    position: relative;
    z-index: 1;
  }
  /* The print-only twins of the two textareas. 10.7px is 8pt exactly
     (8 * 96/72), the legibility floor for body text on this page — nothing
     here goes below it. */
  .econ-print-text,
  .pref-print-text,
  .pecap-print-text {
    display: block !important;
    font-size: 10.7px !important;
    font-family: inherit;
    white-space: pre-wrap;
    overflow: visible !important;
    height: auto;
  }
  .bp-print-text {
    display: block !important;
    font-size: 10.7px !important;
    font-family: inherit;
    white-space: pre-wrap;
    overflow: visible !important;
    height: auto;
  }
  .comment-text.bp-text {
    overflow: visible !important;
    max-height: none !important;
    min-height: 0 !important;
  }

  /* Chart anchored to bottom of page.
     `margin-top: auto` still pins it to the foot of the sheet whenever the
     content leaves free space; with a narrative now printing above it there
     usually is none, and the chart moves to the second page instead.
     NO EXTRA TOP PADDING HERE, and the reason is measured rather than assumed.
     When the chart is pushed to a second page it lands ~3.7pt from the physical
     paper edge, because the page box is `margin: 0` app-wide (App.vue) — a
     non-zero page margin is what gives Chrome room to draw its own header and
     footer, so continuation pages get no top margin of their own. A
     `padding-top: 0.4in` here does fix that (measured: chart top moves 3.7pt ->
     29.2pt, matching page 1) but it COSTS 28.8pt on page one, and page one has
     only about 3pt of slack. It therefore pushed the chart onto a second page
     for the 11 deals that have no narrative at all and were printing correctly
     on one page. Protecting 11 working documents outweighs a 3.7pt offset that
     is only a risk on a physical printer's unprintable strip, not in the PDF.
     The real fix is a page-box margin for continuation pages, which conflicts
     with the header suppression print_page_rule_check enforces — flagged, not
     taken unilaterally. */
  .chart-section {
    /* OUT OF FLOW, pinned to the foot of the sheet.
       This is what makes "one page, nothing clipped" possible: the chart is
       the only element that does not fit, so it stops competing for flow space
       and the text — which always fits on its own — decides the page height.
       A long narrative now OVERLAPS the chart instead of pushing it to a second
       page or being cut. Poplar Prairie is the only deal where that happens
       today: its text ends 30pt into the chart's band. Both are fully present
       in the PDF and the chart can be dragged clear in Acrobat.
       `margin-top: auto` is gone with the flow position; `break-inside` no
       longer applies to an out-of-flow box and is dropped with it. */
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    /* Below the narrative in paint order, so where they overlap the WORDS stay
       readable and the chart shows through behind them. Positioned elements
       otherwise paint over in-flow content, which would hide exactly the three
       lines this change exists to keep. */
    z-index: 0;
  }

  /* Force chart to print */
  .chart-section canvas {
    max-width: 100% !important;
  }
}
</style>
