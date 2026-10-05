<script setup lang="ts">
/**
 * PSC Preferred Equity Exposure — accounting's tracker, from our MRI copy.
 *
 * Replaces "PSC Preferred Equity Tracker - <date>.xlsx" (Jim, Oct 5 2026). Every
 * figure is the one another screen already shows: Cost is the Pref Balance
 * Detail balance less realized losses, FMV adds the unrealized marks from
 * accounting's IA query, the investor split walks the commitments in force at
 * the quarter end, and future funding is the One Pager's remaining to fund.
 * Click a row to see the ownership chains its percentages came from.
 */
import { ref, computed, onMounted } from 'vue'
import api from '@/api/client'

defineProps<{ embedded?: boolean }>()

type Tab = 'cost' | 'fmv'
const tab = ref<Tab>('cost')
const showPct = ref(false)
const quarters = ref<string[]>([])
const liveDate = ref('')
const asOf = ref('')
const report = ref<any>(null)
const loading = ref(false)
const downloading = ref(false)
const error = ref<string | null>(null)
const open = ref<string | null>(null)

const groups = computed<string[]>(() => report.value?.groups || [])

const lead = computed(() => tab.value === 'cost'
  ? [{ key: 'balance', label: 'Capital balance' }, { key: 'realized_loss', label: 'Realized loss' },
     { key: 'cost', label: 'Cost' }]
  : [{ key: 'cost', label: 'Cost' }, { key: 'unrealized', label: 'Unrealized' },
     { key: 'fmv', label: 'FMV' }])
const groupKey = computed(() => tab.value === 'cost' ? 'cost_by_group' : 'fmv_by_group')
const amountKey = computed(() => tab.value === 'cost' ? 'cost' : 'fmv')

function money(v: number | null | undefined) {
  if (v === null || v === undefined) return '—'
  if (Math.abs(v) < 0.5) return '–'
  const s = Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })
  return v < 0 ? `(${s})` : s
}
function pct(v: number | null | undefined) {
  if (v === null || v === undefined) return '—'
  if (Math.abs(v) < 0.00005) return '–'
  return (v * 100).toFixed(2) + '%'
}
function rowKey(r: any) { return `${r.investment_id}|${r.holder}` }

const totals = computed(() => report.value?.totals || {})
const grandKey = computed(() => tab.value === 'cost' ? 'grand_cost' : 'grand_fmv')
const grandGroupKey = computed(() => tab.value === 'cost' ? 'grand_cost_by_group' : 'grand_fmv_by_group')
const committedTotal = computed(() => totals.value.grand_cost)
// A group's share of a line: its amount over the line's amount.
function share(byGroup: Record<string, number> | null | undefined, amount: number | null | undefined, g: string) {
  if (!byGroup || !amount) return null
  return (byGroup[g] || 0) / amount
}

async function load() {
  loading.value = true
  error.value = null
  try {
    const { data } = await api.get('/api/reports/pe-exposure', { params: { as_of: asOf.value } })
    report.value = data
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Could not build the report'
  } finally {
    loading.value = false
  }
}

async function download() {
  downloading.value = true
  try {
    const res = await api.get('/api/reports/pe-exposure/excel', { params: { as_of: asOf.value }, responseType: 'blob' })
    const url = URL.createObjectURL(new Blob([res.data]))
    const a = document.createElement('a')
    a.href = url
    a.download = `PSC_PE_Exposure_${report.value?.as_of || asOf.value}.xlsx`
    a.click()
    URL.revokeObjectURL(url)
  } catch (e: any) {
    error.value = 'Could not download the workbook'
  } finally {
    downloading.value = false
  }
}

onMounted(async () => {
  try {
    const { data } = await api.get('/api/reports/pe-exposure/quarters')
    quarters.value = data.quarters || []
    liveDate.value = data.live
    asOf.value = data.default
    await load()
  } catch (e: any) {
    error.value = e?.response?.data?.error || 'Could not load the quarters'
  }
})
</script>

<template>
  <div class="pe">
    <div class="head">
      <div>
        <h2 v-if="!embedded">PSC Preferred Equity Exposure</h2>
        <h3 v-else>PSC Preferred Equity Exposure</h3>
        <div class="subtitle">
          Cost and fair market value by holding, split to the investors through the ownership chain.
          Click a row to see where its percentages come from.
        </div>
      </div>
      <div class="controls">
        <label>As of
          <select v-model="asOf" @change="load">
            <option value="live">Live (today, {{ liveDate }})</option>
            <option v-for="q in quarters" :key="q" :value="q">Quarter end {{ q }}</option>
          </select>
        </label>
        <label class="chk"><input type="checkbox" v-model="showPct" /> Show %</label>
        <button class="btn" :disabled="!report || downloading" @click="download">
          {{ downloading ? 'Preparing…' : 'Download Excel' }}
        </button>
      </div>
    </div>

    <div v-if="error" class="error-banner">{{ error }}</div>
    <div v-if="loading" class="muted pad">Building the report…</div>

    <template v-if="report && !loading">
      <div class="cards">
        <div class="card"><div class="lbl">Cost</div><div class="val">{{ money(totals.cost) }}</div></div>
        <div class="card"><div class="lbl">Fair market value</div><div class="val">{{ money(totals.fmv) }}</div></div>
        <div class="card"><div class="lbl">Future funding</div><div class="val">{{ money(totals.future_funding) }}</div></div>
        <div class="card"><div class="lbl">Invested + committed</div><div class="val">{{ money(committedTotal) }}</div></div>
      </div>
      <div v-if="!report.is_quarter_end" class="live-note">
        Live view as of {{ report.as_of }}: accounting, commitments and rates through today.
      </div>
      <div class="fx muted">
        <template v-if="report.fx">
          CAD converted at {{ report.fx.value.toFixed(4) }} CAD per USD ({{ report.fx.source }}, {{ report.fx.date }}).
        </template>
        <template v-else>No USD/CAD rate is stored for this quarter end — CAD holdings are shown in CAD and left out of the totals.</template>
      </div>

      <div class="tabs">
        <button :class="{ active: tab === 'cost' }" @click="tab = 'cost'">Cost</button>
        <button :class="{ active: tab === 'fmv' }" @click="tab = 'fmv'">Fair market value</button>
      </div>

      <div class="tbl-wrap">
        <table class="grid">
          <thead>
            <tr>
              <th class="sticky-l">Property</th><th>Holding</th>
              <th v-for="c in lead" :key="c.key" class="num">{{ c.label }}</th>
              <th v-for="g in groups" :key="g" class="num grp">{{ g }}</th>
            </tr>
          </thead>
          <tbody>
            <tr class="section"><td class="sticky-l" :colspan="2 + lead.length + groups.length">Current exposure</td></tr>
            <template v-for="r in report.rows" :key="rowKey(r)">
              <tr class="row" :class="{ open: open === rowKey(r) }" @click="open = open === rowKey(r) ? null : rowKey(r)">
                <td class="sticky-l">
                  {{ r.deal_name }}
                  <span v-if="r.currency !== 'USD'" class="tag"
                        :title="r.fx ? `Converted from ${r.currency} at ${r.fx.rate.toFixed(4)} (${r.fx.date})` : 'No rate stored'">{{ r.currency }}</span>
                  <span v-if="r.problems.length" class="tag warn" :title="r.problems.join('; ')">!</span>
                </td>
                <td class="muted">{{ r.holder }}</td>
                <td v-for="c in lead" :key="c.key" class="num" :class="{ strong: c.key === amountKey }">
                  {{ r.fx_missing ? money(r[c.key + '_local']) + ' CAD' : money(r[c.key]) }}
                </td>
                <td v-for="g in groups" :key="g" class="num grp">
                  {{ showPct ? pct(r.shares[g]) : money((r[groupKey] || {})[g]) }}
                </td>
              </tr>
              <tr v-if="open === rowKey(r)" class="routes">
                <td :colspan="2 + lead.length + groups.length">
                  <div class="routes-title">How {{ r.holder }} is held on {{ report.as_of }} — commitments in force, multiplied down the chain</div>
                  <table class="mini">
                    <tr v-for="(rt, i) in r.routes" :key="i">
                      <td>{{ rt.path.join(' › ') }}</td><td class="num">{{ pct(rt.share) }}</td><td>{{ rt.group }}</td>
                    </tr>
                  </table>
                  <div v-if="r.problems.length" class="warn-text">{{ r.problems.join('; ') }}</div>
                </td>
              </tr>
            </template>
            <tr class="subtotal">
              <td class="sticky-l">Net invested equity</td><td></td>
              <td v-for="c in lead" :key="c.key" class="num">{{ c.key === amountKey ? money(totals[amountKey]) : '' }}</td>
              <td v-for="g in groups" :key="g" class="num grp">
                {{ showPct ? pct(share(totals[groupKey], totals[amountKey], g)) : money((totals[groupKey] || {})[g]) }}
              </td>
            </tr>

            <tr class="section"><td class="sticky-l" :colspan="2 + lead.length + groups.length">
              Future funding <span class="muted normal">— committed, not yet funded (the One Pager's remaining to fund)</span>
            </td></tr>
            <tr v-for="f in report.future_funding" :key="'f' + f.vcode" class="frow">
              <td class="sticky-l" :title="`Committed ${money(f.committed)}, funded ${money(f.funded)}. ${f.basis || ''}`">
                {{ f.deal_name }}
                <span v-if="f.below_funded" class="tag warn" title="The commitment on file is below what has been funded">below funded</span>
              </td>
              <td class="muted">{{ f.holders.join(', ') }}</td>
              <td v-for="c in lead" :key="c.key" class="num" :class="{ strong: c.key === amountKey }">
                {{ c.key === amountKey ? money(f.remaining_to_fund_usd) : '' }}
              </td>
              <td v-for="g in groups" :key="g" class="num grp">
                <template v-if="f.by_group">{{ showPct ? pct(share(f.by_group, f.remaining_to_fund_usd, g)) : money(f.by_group[g]) }}</template>
                <template v-else-if="g === groups[0]"><span class="muted">several holders</span></template>
              </td>
            </tr>
            <tr v-if="!report.future_funding.length"><td class="sticky-l muted" :colspan="2 + lead.length + groups.length">None.</td></tr>
            <tr class="subtotal">
              <td class="sticky-l">Total future funding</td><td></td>
              <td v-for="c in lead" :key="c.key" class="num">{{ c.key === amountKey ? money(totals.future_funding) : '' }}</td>
              <td v-for="g in groups" :key="g" class="num grp">
                {{ showPct ? pct(share(totals.future_by_group, totals.future_funding, g)) : money((totals.future_by_group || {})[g]) }}
              </td>
            </tr>
          </tbody>
          <tfoot>
            <tr>
              <td class="sticky-l">Total equity invested / committed</td><td></td>
              <td v-for="c in lead" :key="c.key" class="num">{{ c.key === amountKey ? money(totals[grandKey]) : '' }}</td>
              <td v-for="g in groups" :key="g" class="num grp">
                {{ showPct ? pct(share(totals[grandGroupKey], totals[grandKey], g)) : money((totals[grandGroupKey] || {})[g]) }}
              </td>
            </tr>
          </tfoot>
        </table>
      </div>
      <div v-if="report.future_funding_basis" class="live-note">{{ report.future_funding_basis }}</div>
      <div v-if="totals.future_unsplit" class="muted pad">
        {{ money(totals.future_unsplit) }} of future funding sits with deals that have several holders and is
        not split between investors, so the investor columns of the totals leave it out.
      </div>
      <div class="muted pad">Future funding is not floored: a negative figure means the commitment on file is
        below what has been funded.</div>

      <div v-if="report.notes.length" class="notes">
        <div class="notes-title">Notes</div>
        <ul><li v-for="(n, i) in report.notes" :key="i">{{ n }}</li></ul>
      </div>
    </template>
  </div>
</template>

<style scoped>
.pe { padding: 4px 0; }
.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
.head h2, .head h3 { margin: 0; }
.subtitle { font-size: 12.5px; color: var(--color-text-secondary); margin-top: 3px; max-width: 640px; }
.controls { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
.controls label { display: flex; flex-direction: column; font-size: 11px; text-transform: uppercase;
  letter-spacing: .03em; color: var(--color-text-secondary); }
.controls label.chk { flex-direction: row; align-items: center; gap: 5px; text-transform: none; font-size: 12.5px; }
.controls select { margin-top: 3px; padding: 4px 6px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); }
.btn { padding: 6px 14px; border: 1px solid var(--color-primary, #2f6f4f); border-radius: 6px;
  background: var(--color-primary, #2f6f4f); color: #fff; cursor: pointer; font-size: 13px; }
.btn:disabled { opacity: .6; cursor: default; }
.error-banner { margin: 12px 0; padding: 8px 12px; border-radius: 6px; background: #fdeaea; color: #8a1f1f; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 10px; margin: 14px 0 6px; }
.card { border: 1px solid var(--color-border); border-radius: 8px; padding: 10px 12px; }
.card .lbl { font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: var(--color-text-secondary); }
.card .val { font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums; margin-top: 2px; }
.fx { font-size: 12px; margin-bottom: 8px; }
.live-note { font-size: 12.5px; margin: 6px 0; padding: 6px 10px; border-radius: 6px; background: #eef4fb; color: #24456e; }
.tabs { display: flex; gap: 6px; margin: 10px 0; }
.tabs button { padding: 6px 14px; border: 1px solid var(--color-border); background: none; border-radius: 6px;
  cursor: pointer; font-size: 13px; color: var(--color-text); }
.tabs button.active { background: var(--color-primary, #2f6f4f); color: #fff; border-color: var(--color-primary, #2f6f4f); }
.tbl-wrap { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 6px; }
.grid { border-collapse: collapse; font-size: 12px; width: 100%; }
.grid th, .grid td { padding: 5px 8px; border-bottom: 1px solid var(--color-border); white-space: nowrap; text-align: left; }
.grid th { font-size: 11px; text-transform: uppercase; letter-spacing: .02em; color: var(--color-text-secondary);
  background: var(--color-surface); position: sticky; top: 0; }
.grid .num { text-align: right; font-variant-numeric: tabular-nums; }
.grid .grp { min-width: 92px; }
.grid .strong { font-weight: 600; }
.grid tfoot td { font-weight: 700; border-top: 2px solid var(--color-text-secondary); background: var(--color-surface); }
.section td { font-weight: 600; background: rgba(47,111,79,.06); font-size: 12px; }
.section .normal { font-weight: 400; }
.subtotal td { font-weight: 600; border-top: 1px solid var(--color-text-secondary); }
.frow td { background: var(--color-surface); }
.sticky-l { position: sticky; left: 0; background: var(--color-surface); z-index: 1; min-width: 210px; }
.row { cursor: pointer; }
.row:hover td { background: var(--color-surface-hover, rgba(0,0,0,.03)); }
.row.open td { background: rgba(47,111,79,.08); }
.routes td { background: var(--color-surface); white-space: normal; }
.routes-title { font-size: 11.5px; color: var(--color-text-secondary); margin: 4px 0 6px; }
.mini { border-collapse: collapse; font-size: 11.5px; }
.mini td { padding: 2px 10px 2px 0; border: none; }
.tag { margin-left: 6px; font-size: 10px; padding: 1px 5px; border-radius: 8px; background: #e7eef9; color: #2c4a7a; }
.tag.warn { background: #fff1db; color: #8a5a00; }
.warn-text { color: #8a5a00; font-size: 12px; margin-top: 6px; }
.muted { color: var(--color-text-secondary); }
.pad { padding: 8px 4px; font-size: 12px; }
.notes { margin-top: 14px; font-size: 12.5px; }
.notes-title { font-weight: 600; margin-bottom: 4px; }
@media (max-width: 900px) { .cards { grid-template-columns: repeat(2, 1fr); } }
</style>
