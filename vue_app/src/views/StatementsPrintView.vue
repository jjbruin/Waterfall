<script setup lang="ts">
/**
 * Investor-ready financial statements, one entity or a batch.
 *
 * THE SAME SHAPE AS THE ONE PAGER'S BATCH PRINT, which Jim named as the thing
 * to borrow: the SERVER assembles every entity in one request, the CLIENT
 * stacks them with a page break between, and a single `window.print()` produces
 * one PDF whether it is one entity or fifty-eight.
 *
 * THIS IS NOT THE WORKPAPER. The downloaded .xlsx is a working document — it
 * carries the provenance notes, the GL figure beside each line, the dormant
 * count. None of that belongs in front of an investor, so none of it is here.
 * What is here is the five statements, in the order the reference package binds
 * them, under the header that package uses.
 *
 * THE HEADER IS THE DELIVERED PACKAGE'S, five lines centred:
 *     entity name / (A Limited Liability Company) / blank /
 *     Statement of ... (Unaudited) / blank / period phrase
 * and "as of" versus "for the period" follows the STATEMENT, not a preference:
 * a balance sheet and a schedule of investments are as of an instant, while
 * operations, members' capital and cash flows cover a span. Reversed, that is a
 * false statement at the top of an investor document.
 */
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api/client'

const route = useRoute()
const loading = ref(true)
const error = ref('')
const pages = ref<any[]>([])
const periodEnd = ref('')
const failedCount = ref(0)

/** Statements in the order the reference package binds them, with the formal
 *  name each carries and whether it is dated at an instant or over a period. */
const STATEMENTS = [
  { key: 'balance_sheet', title: "Statement of Assets, Liabilities and Members' Capital", dated: 'as_of' },
  { key: 'soi', title: 'Schedule of Investment', dated: 'as_of' },
  { key: 'income_statement', title: 'Statement of Operations', dated: 'period' },
  { key: 'members_capital', title: "Statement of Changes in Members' Capital", dated: 'period' },
  { key: 'cash_flow', title: 'Statement of Cash Flows', dated: 'period' },
]

const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
  'August', 'September', 'October', 'November', 'December']
const MONTH_WORD = ['', 'one month', 'two months', 'three months', 'four months',
  'five months', 'six months', 'seven months', 'eight months', 'nine months',
  'ten months', 'eleven months', 'year']

function longDate(iso: string) {
  if (!iso) return ''
  const [y, m, d] = String(iso).slice(0, 10).split('-').map(Number)
  if (!y || !m || !d) return iso
  return `${MONTHS[m - 1]} ${d}, ${y}`
}

function periodPhrase(iso: string, dated: string) {
  if (dated === 'as_of') return `As of ${longDate(iso)}`
  const m = Number(String(iso).slice(5, 7))
  if (m === 12) return `For the year ended ${longDate(iso)}`
  const w = MONTH_WORD[m]
  // An unknown period length says the date and NOT a span we cannot support.
  return w ? `For the ${w} ended ${longDate(iso)}` : `For the period ended ${longDate(iso)}`
}

/** "(A Limited Liability Company)" only when the name says so. Printing a legal
 *  form is an assertion about the entity; inventing one would be wrong in the
 *  single place nobody would question it. */
function legalForm(name: string) {
  const n = String(name || '').trim().replace(/\.$/, '')
  if (/\bL\.?L\.?C\.?$/i.test(n)) return '(A Limited Liability Company)'
  if (/\bL\.?P\.?$/i.test(n)) return '(A Limited Partnership)'
  return ''
}

const money = (v: any) => {
  if (v == null || v === '') return ''
  const n = Number(v)
  if (!isFinite(n)) return ''
  // Parentheses for negatives and a dash for zero, as the reference's own
  // number format does — never a minus sign and never "0".
  if (Math.abs(n) < 0.005) return '—'
  const s = Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 0 })
  return n < 0 ? `(${s})` : s
}

/** Only the lines that belong on a printed statement: a line with no balance
 *  and no movement is a trial-balance artefact, not a statement line. */
function shownLines(block: any) {
  if (!block || !block.sections) return []
  return block.sections.map((sec: any) => ({
    section: sec.section,
    total: sec.total,
    lines: (sec.lines || []).filter((l: any) => !l.dormant),
  })).filter((sec: any) => sec.lines.length)
}

function hasContent(page: any, key: string) {
  const b = page?.[key]
  if (!b) return false
  if (b.sections) return shownLines(b).length > 0
  if (Array.isArray(b.lines)) return b.lines.length > 0
  if (Array.isArray(b.rows)) return b.rows.length > 0
  return false
}

/** Flattened page list: every (entity, statement) pair that has something to
 *  say, so page breaks and numbering follow what actually prints. */
const sheets = computed(() => {
  const out: any[] = []
  for (const p of pages.value) {
    if (p.error) { out.push({ page: p, error: true }); continue }
    for (const st of STATEMENTS) {
      if (hasContent(p, st.key)) out.push({ page: p, st })
    }
  }
  return out
})

async function load() {
  loading.value = true
  error.value = ''
  try {
    const body: any = {}
    const ids = String(route.query.entities || '').split(',').map(s => s.trim()).filter(Boolean)
    if (ids.length) body.entity_ids = ids
    if (route.query.cycle_id) body.cycle_id = Number(route.query.cycle_id)
    if (route.query.period_end) body.period_end = String(route.query.period_end)
    const res = await api.post('/api/workpapers/statements/batch', body)
    pages.value = res.data.pages || []
    periodEnd.value = res.data.period_end || ''
    failedCount.value = res.data.failed || 0
  } catch (e: any) {
    error.value = e.response?.data?.error || e.message
  } finally {
    loading.value = false
  }
}

/** `window` is not exposed to the template, so the print call needs a method. */
function doPrint() { window.print() }

onMounted(load)
</script>

<template>
  <div class="sp">
    <!-- Screen-only controls; `@media print` removes them. -->
    <div class="sp-bar no-print">
      <div>
        <b>{{ pages.length }}</b> entit{{ pages.length === 1 ? 'y' : 'ies' }} ·
        <b>{{ sheets.filter(s => !s.error).length }}</b> statement pages
        <span v-if="failedCount" class="warn">
          · {{ failedCount }} could not be built and print as a notice
        </span>
      </div>
      <button class="btn" :disabled="loading || !pages.length" @click="doPrint">
        Print / Save as PDF
      </button>
    </div>

    <div v-if="loading" class="sp-msg">Building statements…</div>
    <div v-else-if="error" class="sp-msg err">{{ error }}</div>
    <div v-else-if="!pages.length" class="sp-msg">
      Nothing to print. Pass <code>?entities=PPI35,PPI36&amp;period_end=2026-06-30</code>
      or <code>?cycle_id=1</code>.
    </div>

    <template v-else>
      <section v-for="(s, i) in sheets" :key="i" class="sheet"
               :class="{ 'page-break': i < sheets.length - 1 }">
        <!-- AN ENTITY THAT FAILED STILL GETS A PAGE, saying so. A batch that
             silently omits an entity is worse than one that prints a notice:
             the reader cannot tell a missing statement from a missing entity. -->
        <template v-if="s.error">
          <header class="fs-head">
            <div class="fs-entity">{{ s.page.name }}</div>
            <div class="fs-sub">{{ legalForm(s.page.name) }}</div>
          </header>
          <p class="fs-fail">
            These statements could not be built.<br />
            <span class="fs-fail-why">{{ s.page.error }}</span>
          </p>
        </template>

        <template v-else>
          <header class="fs-head">
            <div class="fs-entity">{{ s.page.name }}</div>
            <div class="fs-sub">{{ legalForm(s.page.name) }}</div>
            <div class="fs-title">{{ s.st.title }} (Unaudited)</div>
            <div class="fs-period">{{ periodPhrase(periodEnd, s.st.dated) }}</div>
          </header>

          <table class="fs">
            <tbody>
              <template v-for="sec in shownLines(s.page[s.st.key])" :key="sec.section">
                <tr class="fs-sec"><td colspan="2">{{ sec.section }}</td></tr>
                <tr v-for="l in sec.lines" :key="l.fs_line">
                  <td class="fs-line">{{ l.fs_line }}</td>
                  <td class="fs-amt">{{ money(l.amount) }}</td>
                </tr>
                <tr class="fs-total">
                  <td class="fs-line">Total {{ sec.section }}</td>
                  <td class="fs-amt">{{ money(sec.total) }}</td>
                </tr>
              </template>
              <!-- The closing line, whichever statement this is. The
                   reconciliation note is NOT printed: an investor document
                   states the figure, and a statement that does not tie is a
                   matter for the close, not for the reader. It is on the
                   workbench and in the workbook, where it can be acted on. -->
              <tr v-if="s.page[s.st.key]?.footing" class="fs-total fs-grand">
                <td class="fs-line">{{ s.page[s.st.key].footing.label }}</td>
                <td class="fs-amt">{{ money(s.page[s.st.key].footing.amount) }}</td>
              </tr>
            </tbody>
          </table>
        </template>

        <footer class="fs-foot">{{ i + 1 }}</footer>
      </section>
    </template>
  </div>
</template>

<style scoped>
.sp { background: #f4f6fa; min-height: 100vh; padding: 0 0 40px; }
.sp-bar {
  position: sticky; top: 0; z-index: 5;
  display: flex; justify-content: space-between; align-items: center;
  background: #fff; border-bottom: 1px solid #dde3ec;
  padding: 10px 18px; font-size: 13px;
}
.sp-bar .warn { color: #b4232a; }
.btn {
  background: #1d4e7e; color: #fff; border: none; border-radius: 4px;
  padding: 6px 16px; font-size: 13px; font-weight: 600; cursor: pointer;
}
.btn:disabled { opacity: .5; cursor: default; }
.sp-msg { padding: 40px; text-align: center; color: #5a6475; }
.sp-msg.err { color: #b4232a; }

/* One sheet is one printed page. On screen it is shown as a page too, so what
   is on the paper is what was reviewed. */
.sheet {
  background: #fff; width: 8.5in; min-height: 10.4in;
  margin: 18px auto; padding: 0.75in 1in;
  box-shadow: 0 1px 4px rgba(0,0,0,.14);
  font-family: Arial, Helvetica, sans-serif; font-size: 11pt;
  color: #000; position: relative; box-sizing: border-box;
}
.fs-head { text-align: center; margin-bottom: 26px; }
.fs-entity { font-weight: 700; }
.fs-sub { }
.fs-title { margin-top: 14px; }
.fs-period { margin-top: 14px; }

table.fs { width: 100%; border-collapse: collapse; }
table.fs td { padding: 2px 0; vertical-align: bottom; }
.fs-sec td { font-weight: 700; padding-top: 12px; }
.fs-line { padding-left: 14px; }
.fs-sec .fs-line, .fs-total .fs-line { padding-left: 0; }
.fs-amt { text-align: right; width: 1.6in; font-variant-numeric: tabular-nums; }
.fs-total td {
  font-weight: 700; border-top: 1px solid #000;
  border-bottom: 3px double #000; padding-top: 3px;
}
.fs-total.fs-grand td { padding-top: 10px; }
.fs-fail { margin-top: 40px; text-align: center; }
.fs-fail-why { color: #666; font-size: 9pt; }
.fs-foot {
  position: absolute; bottom: 0.4in; left: 0; right: 0;
  text-align: center; font-size: 10pt;
}

@media print {
  .no-print { display: none !important; }
  .sp { background: none; padding: 0; }
  .sheet {
    width: auto; min-height: 0; margin: 0; box-shadow: none;
    /* The reference's own margins for a statement page. `@page` supplies the
       physical margin; this keeps the content box matching it. */
    padding: 0 0 0 0;
  }
  .sheet.page-break { page-break-after: always; }
  .fs-foot { position: fixed; bottom: 0.25in; }
}
@page { size: letter portrait; margin: 1in 1in 0.75in 1in; }
</style>
