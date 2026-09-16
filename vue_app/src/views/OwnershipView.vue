<script setup lang="ts">
/**
 * Ownership — the chain above each preferred equity investment.
 *
 * Reads from /api/ownership/chain/*, which derives every share from committed
 * dollars rather than a stored percentage. See ownership_chain_service.py for
 * why: on the one investment where the two could be compared, the amounts were
 * right and both percentage fields were wrong.
 *
 * The tree runs LEFT TO RIGHT: the investment sits on the left, its owners to
 * its right, their owners further right, up to OWPSC. That orientation is the
 * point — a level is a column, so an analyst scanning for "which levels still
 * need a waterfall" reads down a column rather than chasing indentation.
 */
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import DataTable from '../components/common/DataTable.vue'
import api from '../api/client'
import { useDataStore } from '../stores/data'

const dataStore = useDataStore()

// Two tabs, two different sources, deliberately.
//
// "chain" derives ownership from committed dollars (ownership_chain_service).
// "upstream" runs a test distribution through the waterfalls and traces the
// cash, and reads `relationships` via the older /tree endpoints. They are not
// redundant: the first answers "who owns this and is the waterfall built", the
// second answers "if we distribute, where does the money actually land". Where
// the two sources disagree the chain tab flags it, which is the honest
// treatment while accounting is mid-update on the commitments table.
const tab = ref<'chain' | 'upstream'>('chain')

interface Node {
  entity_id: string
  name: string
  level: number
  committed: number
  balance: number | null
  effective_pct: number | null
  look_through: number | null
  look_through_balance: number | null
  balance_detail?: Array<{ typename: string; rows: number; effect: number }>
  balance_by_flag?: number | null
  balance_disputed?: boolean
  balance_direct?: number | null
  balance_from_parent?: number | null
  into_pct?: number | null
  into_entity_id?: string
  into_name?: string
  ultimate_owner?: boolean
  pct: number | null
  pct_stated: number | null
  pct_disagrees: boolean
  commitment_count?: number
  since?: string | null
  has_waterfall: boolean
  step_count: number
  waterfall_code: string
  waterfall_url: string | null
  terminal: boolean
  truncated_reason?: string
  owners: Node[]
  vcode?: string
  is_pe_investment?: boolean
}

interface Investment {
  entity_id: string
  name: string
  vcode: string
  portfolio: string
  owner_count: number
  total_committed: number
  has_waterfall: boolean
  step_count: number
  waterfall_url: string | null
}

const investments = ref<Investment[]>([])
const listLoading = ref(false)
const chainLoading = ref(false)
const selected = ref<string | null>(null)
const root = ref<Node | null>(null)
const summary = ref<any>(null)
const health = ref<any>(null)
const filter = ref('')
const onlyMissing = ref(false)

// Collapse state is per entity id. Everything starts open: this screen exists
// to show what is missing, and a collapsed tree hides exactly that.
const collapsed = ref<Record<string, boolean>>({})
// Which node's balance breakdown is open. One at a time: these are dense and
// several open at once makes the columns jump.
const showBal = ref<string | null>(null)
function toggle(id: string) {
  collapsed.value[id] = !collapsed.value[id]
}

const shownInvestments = computed(() => {
  const q = filter.value.trim().toLowerCase()
  return investments.value.filter(i => {
    if (onlyMissing.value && i.has_waterfall) return false
    if (!q) return true
    return (i.name || '').toLowerCase().includes(q)
      || (i.entity_id || '').toLowerCase().includes(q)
      || (i.vcode || '').toLowerCase().includes(q)
  })
})

async function loadInvestments() {
  listLoading.value = true
  try {
    const res = await api.get('/api/ownership/chain/investments')
    investments.value = res.data.investments || []
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to load investments', 'error')
  } finally {
    listLoading.value = false
  }
}

async function loadChain(id: string) {
  selected.value = id
  chainLoading.value = true
  root.value = null
  try {
    const res = await api.get(`/api/ownership/chain/${encodeURIComponent(id)}`)
    root.value = res.data.root
    summary.value = res.data.summary
    health.value = res.data.data_health
    collapsed.value = {}
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to load ownership chain', 'error')
  } finally {
    chainLoading.value = false
  }
}

/**
 * Flatten the tree into columns, one per ownership level.
 *
 * Each node instance gets a UID rather than being keyed on entity_id: the same
 * entity can own into more than one place and would otherwise collide, both in
 * the v-for key and in the connector lines.
 */
const edges = ref<Array<{ from: string; to: string }>>([])

const columns = computed<Node[][]>(() => {
  if (!root.value) return []
  const cols: Node[][] = []
  const es: Array<{ from: string; to: string }> = []
  const walk = (n: Node, depth: number, uid: string) => {
    ;(n as any).__uid = uid
    if (!cols[depth]) cols[depth] = []
    cols[depth].push(n)
    if (collapsed.value[n.entity_id]) return
    ;(n.owners || []).forEach((o, i) => {
      const childUid = `${uid}/${o.entity_id}-${i}`
      es.push({ from: uid, to: childUid })
      walk(o, depth + 1, childUid)
    })
  }
  walk(root.value, 0, root.value.entity_id)
  edges.value = es
  return cols
})

function uidOf(n: Node): string {
  return (n as any).__uid || n.entity_id
}

/**
 * Connector lines.
 *
 * The columns pack each level top-to-bottom, so an owner is rarely level with
 * the investment it owns into — by the third level the two can be hundreds of
 * pixels apart and the reader has to guess. These are drawn from measured DOM
 * positions rather than computed from the data, because the data does not know
 * how the browser wrapped the cards.
 */
const linkPaths = ref<Array<{ d: string; unresolved: boolean }>>([])
const svgBox = ref({ w: 0, h: 0 })
const colsEl = ref<HTMLElement | null>(null)

function drawLinks() {
  const host = colsEl.value
  if (!host || !root.value) { linkPaths.value = []; return }
  const base = host.getBoundingClientRect()
  svgBox.value = { w: host.scrollWidth, h: host.scrollHeight }

  const box = (uid: string) => {
    const el = host.querySelector(`[data-uid="${CSS.escape(uid)}"]`) as HTMLElement | null
    if (!el) return null
    const r = el.getBoundingClientRect()
    return {
      left: r.left - base.left, right: r.right - base.left,
      mid: r.top - base.top + r.height / 2,
    }
  }

  const out: Array<{ d: string; unresolved: boolean }> = []
  for (const e of edges.value) {
    const a = box(e.from), b = box(e.to)
    if (!a || !b) continue
    // Out of the owner's right edge, into the investment's left edge. The
    // control points sit half the gap away so the curve leaves and arrives
    // horizontally, which keeps the arrowhead readable when the two cards are
    // far apart vertically.
    const gap = Math.max(18, (b.left - a.right) / 2)
    out.push({
      d: `M ${a.right} ${a.mid} C ${a.right + gap} ${a.mid}, ${b.left - gap} ${b.mid}, ${b.left - 7} ${b.mid}`,
      unresolved: false,
    })
  }
  linkPaths.value = out
}

let redrawTimer: number | undefined
function scheduleRedraw() {
  window.clearTimeout(redrawTimer)
  redrawTimer = window.setTimeout(drawLinks, 30)
}

function levelLabel(i: number): string {
  if (i === 0) return 'Preferred equity investment'
  return `Ownership level ${i}`
}

/** Total committed at a level — the figure that reveals a missing row. */
function levelTotal(nodes: Node[]): number {
  return nodes.reduce((s, n) => s + (n.committed || 0), 0)
}

/* ── Upstream analysis (second tab) ──────────────────────────────────
 *
 * Restored Sep 15 2026 after the chain rebuild replaced the whole view and
 * dropped it. The endpoints had never gone away; only the way in had.
 *
 * Its entity list and tree come from /api/ownership/tree, loaded the first
 * time this tab is opened rather than on mount — it walks the full
 * relationship graph, and the chain tab should not wait for it.
 */
const upstreamReady = ref(false)
const treeData = ref<any>(null)
const treeLoading = ref(false)
const selectedEntity = ref('')
const distributionAmount = ref(100000)
// Operating cash or a capital event. The two run DIFFERENT waterfalls and only
// the second reduces capital outstanding, so this was never a detail the screen
// could pick on the user's behalf — it was hardcoded to CF_WF and a sale came
// out modelled as an operating distribution.
const wfType = ref<'CF_WF' | 'Cap_WF'>('CF_WF')
const upstreamResult = ref<any>(null)
const upstreamLoading = ref(false)
const upstreamError = ref('')

/**
 * THE SAME LIST DEAL ANALYSIS USES — `data.deals`, keyed by vcode.
 *
 * It used to be built from the relationship tree's own nodes, which is a
 * different population in a different order and included upstream entities
 * nobody runs a distribution against. An analyst moving between the two screens
 * had to translate. (Jim, Sep 15 2026.)
 */
const entities = computed(() => {
  return (dataStore.deals || [])
    .map((d: any) => ({ id: d.vcode, name: d.Investment_Name || d.vcode }))
    .sort((a: any, b: any) => String(a.name).localeCompare(String(b.name)))
})

async function openUpstream() {
  tab.value = 'upstream'
  if (upstreamReady.value || treeLoading.value) return
  treeLoading.value = true
  try {
    if (!(dataStore.deals || []).length) await dataStore.loadDeals()
    upstreamReady.value = true
  } catch (e: any) {
    dataStore.addToast(
      'Failed to load the deal list: ' + (e.response?.data?.error || e.message), 'error')
  } finally {
    treeLoading.value = false
  }
}

async function runUpstreamAnalysis() {
  if (!selectedEntity.value) return
  upstreamLoading.value = true
  upstreamResult.value = null
  upstreamError.value = ''
  try {
    const res = await api.post('/api/ownership/upstream-analysis', {
      entity_id: selectedEntity.value,
      distribution_amount: distributionAmount.value,
      wf_type: wfType.value,
    })
    if (res.data.error) upstreamError.value = res.data.error
    else upstreamResult.value = res.data
  } catch (e: any) {
    upstreamError.value = e.response?.data?.error || 'Upstream analysis failed'
  } finally {
    upstreamLoading.value = false
  }
}

const dealAllocColumns = [
  { key: 'iOrder', label: 'Step', align: 'right' },
  { key: 'PropCode', label: 'PropCode' },
  { key: 'vState', label: 'vState' },
  { key: 'step_description', label: 'What the step does' },
  { key: 'Allocated', label: 'Allocated', format: 'currency2', align: 'right' },
]
const beneficiaryColumns = [
  { key: 'entity_id', label: 'Entity' },
  { key: 'amount', label: 'Amount', format: 'currency2', align: 'right' },
  { key: 'pct_of_total', label: '% of Total', format: 'percent', align: 'right' },
]
const upstreamColumns = [
  { key: 'Entity', label: 'Entity' },
  { key: 'PropCode', label: 'PropCode' },
  { key: 'vState', label: 'vState' },
  { key: 'Allocated', label: 'Allocated', format: 'currency2', align: 'right' },
  { key: 'Level', label: 'Level', align: 'right' },
  { key: 'Path', label: 'Path' },
]

function fmtMoney(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  if (v === 0) return '$0'
  return '$' + Math.round(v).toLocaleString()
}
function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—'
  return v.toFixed(2) + '%'
}

let ro: ResizeObserver | undefined
onMounted(() => {
  loadInvestments()
  // The cards change height with their own content (a disagreement note adds a
  // line), so a window listener alone is not enough — observe the container.
  if (typeof ResizeObserver !== 'undefined' && colsEl.value) {
    ro = new ResizeObserver(scheduleRedraw)
    ro.observe(colsEl.value)
  }
  window.addEventListener('resize', scheduleRedraw)
})
onBeforeUnmount(() => {
  ro?.disconnect()
  window.removeEventListener('resize', scheduleRedraw)
  window.clearTimeout(redrawTimer)
})

// Redraw after the DOM has the new cards, not before.
watch([root, collapsed], () => nextTick(() => {
  if (colsEl.value && ro) { ro.disconnect(); ro.observe(colsEl.value) }
  drawLinks()
}), { deep: true })
</script>

<template>
  <div class="ownership">
    <header class="head">
      <div>
        <h2>Ownership</h2>
        <p class="sub">
          Who owns each preferred equity investment, level by level, up to OWPSC.
          Shares are derived from committed dollars.
        </p>
      </div>
      <div class="head-actions">
        <button v-if="tab === 'chain'" class="btn" @click="loadInvestments" :disabled="listLoading">
          {{ listLoading ? 'Loading…' : 'Refresh' }}
        </button>
      </div>
    </header>

    <nav class="tabs" role="tablist">
      <button
        class="tab" :class="{ active: tab === 'chain' }"
        role="tab" :aria-selected="tab === 'chain'"
        @click="tab = 'chain'"
      >Ownership chain</button>
      <button
        class="tab" :class="{ active: tab === 'upstream' }"
        role="tab" :aria-selected="tab === 'upstream'"
        @click="openUpstream"
      >Upstream analysis</button>
    </nav>

    <div v-show="tab === 'chain'" class="layout">
      <!-- Left: the PE investment level -->
      <aside class="picker">
        <div class="picker-controls">
          <input v-model="filter" class="search" type="search" placeholder="Filter investments…" />
          <label class="chk">
            <input type="checkbox" v-model="onlyMissing" />
            Missing waterfall only
          </label>
          <p class="count">
            {{ shownInvestments.length }} of {{ investments.length }} investments
          </p>
        </div>

        <div v-if="listLoading" class="muted pad">Loading investments…</div>
        <ul v-else class="inv-list">
          <li v-for="inv in shownInvestments" :key="inv.entity_id">
            <button
              class="inv"
              :class="{ active: selected === inv.entity_id }"
              @click="loadChain(inv.entity_id)"
            >
              <span class="inv-name">{{ inv.name }}</span>
              <span class="inv-meta">
                <code>{{ inv.entity_id }}</code>
                <span class="dot" :class="inv.has_waterfall ? 'ok' : 'gap'"
                      :title="inv.has_waterfall ? `${inv.step_count} waterfall steps` : 'No waterfall set up'"></span>
              </span>
              <span class="inv-sub">
                {{ inv.owner_count }} owner{{ inv.owner_count === 1 ? '' : 's' }}
                · {{ fmtMoney(inv.total_committed) }}
              </span>
            </button>
          </li>
          <li v-if="!shownInvestments.length" class="muted pad">Nothing matches.</li>
        </ul>
      </aside>

      <!-- Right: the horizontal chain -->
      <section class="chain">
        <div v-if="!selected" class="empty">
          <p>Select an investment to see who owns it.</p>
        </div>
        <div v-else-if="chainLoading" class="empty"><p>Building ownership chain…</p></div>

        <template v-else-if="root">
          <div class="chain-summary">
            <div class="stat">
              <span class="n">{{ summary?.entities_above ?? 0 }}</span>
              <span class="l">entities above</span>
            </div>
            <div class="stat" :class="{ warn: (summary?.levels_missing_waterfall ?? 0) > 0 }">
              <span class="n">{{ summary?.levels_missing_waterfall ?? 0 }}</span>
              <span class="l">without a waterfall</span>
            </div>
            <div class="stat">
              <span class="n">{{ summary?.max_depth_reached ?? 0 }}</span>
              <span class="l">levels deep</span>
            </div>
          </div>

          <div v-if="health?.notes?.length" class="notes">
            <p class="notes-title">Check before building a waterfall on this split</p>
            <ul><li v-for="(n, i) in health.notes" :key="i">{{ n }}</li></ul>
          </div>

          <div class="cols-scroll" @scroll="scheduleRedraw">
            <div class="cols" ref="colsEl">
              <!-- Connectors. Behind the cards, never intercepting a click. -->
              <svg class="links" :width="svgBox.w" :height="svgBox.h"
                   :viewBox="`0 0 ${svgBox.w} ${svgBox.h}`" aria-hidden="true">
                <defs>
                  <marker id="own-arrow" viewBox="0 0 8 8" refX="7" refY="4"
                          markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 8 4 L 0 8 z" fill="#9fb0c4" />
                  </marker>
                </defs>
                <path v-for="(p, i) in linkPaths" :key="i" :d="p.d"
                      fill="none" stroke="#9fb0c4" stroke-width="1.5"
                      marker-end="url(#own-arrow)" />
              </svg>
              <div v-for="(col, i) in columns" :key="i" class="col">
                <div class="col-head">
                  <span class="col-label">{{ levelLabel(i) }}</span>
                  <span class="col-total">{{ fmtMoney(levelTotal(col)) }}</span>
                </div>

                <div v-for="n in col" :key="uidOf(n)" class="node"
                     :data-uid="uidOf(n)"
                     :class="{ pe: i === 0, terminal: n.terminal, ubo: n.ultimate_owner }">
                  <div class="node-top">
                    <button
                      v-if="(n.owners || []).length"
                      class="twist"
                      @click="toggle(n.entity_id)"
                      :aria-label="collapsed[n.entity_id] ? 'Expand owners' : 'Collapse owners'"
                    >{{ collapsed[n.entity_id] ? '▸' : '▾' }}</button>
                    <span v-else class="twist spacer"></span>
                    <div class="node-id">
                      <code>{{ n.entity_id }}</code>
                      <span class="node-name">{{ n.name }}</span>
                    </div>
                  </div>

                  <div class="node-figs">
                    <span v-if="n.effective_pct !== null && i > 0" class="pct"
                          :title="`Share of this deal, the ownership percentages multiplied down the chain`">
                      {{ fmtPct(n.effective_pct) }}
                      <em>of deal</em>
                    </span>
                    <span v-if="i > 1 && n.pct !== null" class="pct-par">
                      {{ fmtPct(n.pct) }} of {{ n.into_entity_id }}
                    </span>
                  </div>
                  <dl class="bal">
                    <dt>
                      {{ i === 0 ? 'committed in' : 'in this deal' }}
                      <!-- The date qualifies the COMMITMENT, not the balance:
                           it is the StartDate of the commitment row in force,
                           used to pick which row is current when an entity has
                           several. It sat on its own line directly above the
                           balance and was read as the balance's as-of date
                           (Jim, Sep 15 2026), which it has never been. -->
                      <em v-if="n.since" :title="`The commitment in force from ${n.since}. Not a balance date.`">
                        as of {{ n.since }}
                      </em>
                    </dt>
                    <dd>{{ fmtMoney(i === 0 ? n.committed : n.look_through) }}</dd>
                    <template v-if="i > 0">
                      <dt>
                        balance
                        <em title="Every capital contribution and return on record for this pair, with no date cutoff — a cumulative position, not a point in time.">today</em>
                      </dt>
                      <dd :class="{ none: n.look_through_balance === null
                                          || n.look_through_balance === undefined,
                                    neg: (n.look_through_balance ?? 0) < 0,
                                    disputed: n.balance_disputed }">
                        <button v-if="(n.balance_detail || []).length || i > 1"
                                class="bal-btn" type="button"
                                @click="showBal = showBal === uidOf(n) ? null : uidOf(n)"
                                :title="'What this balance is made of'">
                          {{ fmtMoney(n.look_through_balance) }}
                          <span class="caret">{{ showBal === uidOf(n) ? '▾' : '▸' }}</span>
                        </button>
                        <template v-else>
                          {{ n.look_through_balance === null || n.look_through_balance === undefined
                             ? 'no data' : fmtMoney(n.look_through_balance) }}
                        </template>
                      </dd>
                    </template>
                  </dl>

                  <!-- WHAT THE BALANCE IS MADE OF. Opened on demand, because a
                       figure nobody can take apart is a figure nobody can argue
                       with — and this one was wrong once already. -->
                  <!-- Above level 1 the raw accounting is about a different
                       relationship, so the panel shows the DERIVATION into this
                       deal instead of BRECO's whole history with PSC3. -->
                  <div v-if="showBal === uidOf(n) && i > 1" class="bal-detail">
                    <div class="bal-row">
                      <span class="bal-t">{{ n.into_entity_id }} in this deal</span>
                      <span class="bal-n">{{ fmtMoney(n.balance_from_parent) }}</span>
                      <span class="bal-c"></span>
                    </div>
                    <div class="bal-row">
                      <span class="bal-t">× {{ n.entity_id }} share of {{ n.into_entity_id }}</span>
                      <span class="bal-n">{{ fmtPct(n.pct) }}</span>
                      <span class="bal-c"></span>
                    </div>
                    <div class="bal-row total">
                      <span class="bal-t">{{ n.entity_id }} in this deal</span>
                      <span class="bal-n">{{ fmtMoney(n.look_through_balance) }}</span>
                      <span class="bal-c"></span>
                    </div>
                    <p class="bal-note">
                      {{ n.entity_id }}'s whole position in {{ n.into_entity_id }} is
                      {{ fmtMoney(n.balance_direct) }}, spread across everything
                      {{ n.into_entity_id }} holds. Only the share above belongs to this deal.
                    </p>
                  </div>

                  <div v-if="showBal === uidOf(n) && i === 1 && (n.balance_detail || []).length"
                       class="bal-detail">
                    <div v-for="(d, k) in n.balance_detail" :key="k" class="bal-row">
                      <span class="bal-t">{{ d.typename }}</span>
                      <span class="bal-n" :class="{ neg: d.effect < 0 }">
                        {{ fmtMoney(d.effect) }}
                      </span>
                      <span class="bal-c">{{ d.rows }}</span>
                    </div>
                    <div class="bal-row total">
                      <span class="bal-t">direct balance</span>
                      <span class="bal-n">{{ fmtMoney(n.balance) }}</span>
                      <span class="bal-c"></span>
                    </div>
                    <p v-if="n.balance_disputed" class="bal-warn">
                      The <code>Capital</code> flag gives {{ fmtMoney(n.balance_by_flag) }}
                      for the same rows. The two classifiers disagree — see
                      open_items §2.3.
                    </p>
                  </div>

                  <!-- The RAW commitment, shown only where it differs from the
                       look-through — i.e. above the first level, where it is a
                       commitment to a fund and not to this property. Jim's
                       example: OWPSC's $64M into PSC3 is spread across
                       everything PSC3 holds. -->
                  <p v-if="i > 1 && n.committed !== n.look_through" class="direct">
                    direct: {{ fmtMoney(n.committed) }} into {{ n.into_entity_id }}
                    <span class="direct-note">— across all its holdings</span>
                  </p>



                  <p v-if="n.pct_disagrees" class="disagree">
                    Stored {{ fmtPct(n.pct_stated) }} — likely a missing commitment row
                  </p>

                  <div class="node-wf">
                    <span class="badge"
                          :class="n.has_waterfall ? 'ok' : (n.ultimate_owner ? 'ubo' : 'gap')">
                      {{ n.has_waterfall
                        ? `Waterfall · ${n.step_count} steps`
                        : (n.ultimate_owner ? 'Beneficial owner' : 'No waterfall') }}
                    </span>
                    <a v-if="n.waterfall_url && !(n.ultimate_owner && !n.has_waterfall)"
                       :href="n.waterfall_url" class="wf-link">
                      {{ n.has_waterfall ? 'Open' : 'Set up' }} {{ n.waterfall_code }} →
                    </a>
                  </div>

                  <p v-if="n.truncated_reason" class="trunc">{{ n.truncated_reason }}</p>
                </div>
              </div>
            </div>
          </div>
        </template>
      </section>
    </div>

    <!-- Upstream analysis -->
    <div v-show="tab === 'upstream'" class="upstream">
      <p class="sub">
        Run a test distribution through an entity's waterfall and trace where the cash
        actually lands. Ownership here comes from the relationships feed, not from
        committed dollars — see the chain tab for where the two disagree.
      </p>

      <div v-if="treeLoading" class="empty"><p>Loading entities…</p></div>

      <template v-else>
        <div class="up-controls">
          <label class="fld">
            <span>Entity</span>
            <select v-model="selectedEntity">
              <option value="">— Select an entity —</option>
              <option v-for="e in entities" :key="e.id" :value="e.id">
                {{ e.name }} ({{ e.id }})
              </option>
            </select>
          </label>
          <label class="fld">
            <span>Distribution amount ($)</span>
            <input type="number" v-model.number="distributionAmount" min="0" step="10000" />
          </label>
          <fieldset class="fld kind">
            <legend>Distribution type</legend>
            <label><input type="radio" value="CF_WF" v-model="wfType" /> Cash flow</label>
            <label><input type="radio" value="Cap_WF" v-model="wfType" /> Capital</label>
          </fieldset>
          <button
            class="btn primary"
            @click="runUpstreamAnalysis"
            :disabled="!selectedEntity || upstreamLoading"
          >{{ upstreamLoading ? 'Running…' : 'Run analysis' }}</button>
        </div>

        <div v-if="upstreamError" class="err">{{ upstreamError }}</div>

        <div v-else-if="!upstreamResult" class="empty">
          <p>Pick an entity and an amount, then run the analysis.</p>
        </div>

        <template v-else>
          <div class="chain-summary">
            <div class="stat">
              <span class="n">{{ fmtMoney(upstreamResult.distribution_amount) }}</span>
              <span class="l">{{ upstreamResult.wf_label }} distribution</span>
            </div>
            <div class="stat"
                 :class="{ warn: (upstreamResult.reconciliation_errors || []).length }">
              <span class="n">{{ fmtMoney(upstreamResult.deal_allocated_total) }}</span>
              <span class="l">allocated by the steps</span>
            </div>
            <div class="stat">
              <span class="n">{{ fmtMoney(upstreamResult.total_allocated) }}</span>
              <span class="l">total allocated</span>
            </div>
          </div>

          <!-- WHAT THE WATERFALL STARTED FROM. Without this the deal-level
               split looks arbitrary: the first dollars go to accrued pref, and
               a reader who cannot see the balances cannot tell why one partner
               takes everything before another takes anything. -->
          <div v-if="(upstreamResult.opening_states || []).length" class="opening">
            <h3 class="up-h">Opening balances the waterfall starts from</h3>
            <div class="ben-table">
              <div class="ben-head">
                <span>Entity</span><span>Capital outstanding</span><span>Accrued pref</span>
              </div>
              <div v-for="o in upstreamResult.opening_states" :key="o.entity_id" class="ben-row">
                <span class="ben-e">{{ o.entity_id }}</span>
                <span class="ben-a">{{ fmtMoney(o.capital_outstanding) }}</span>
                <span class="ben-a">{{ fmtMoney(o.accrued_pref) }}</span>
              </div>
            </div>
            <p class="opening-note">
              Seeded from accounting by the same engine Deal Analysis uses, through
              <code>seed_states_from_accounting</code>.
            </p>
          </div>

          <div v-if="upstreamResult.seeding_warning" class="err">
            {{ upstreamResult.seeding_warning }}
          </div>

          <!-- A waterfall cannot distribute more than it has. If the arithmetic
               does not close, say so loudly rather than let a reader discover
               it by adding up a column. -->
          <div v-for="(m, i) in upstreamResult.reconciliation_errors || []"
               :key="'recon' + i" class="err recon">
            <strong>Does not reconcile.</strong> {{ m }}
          </div>

          <h3 class="up-h">Deal-level allocations</h3>
          <DataTable :columns="dealAllocColumns" :rows="upstreamResult.deal_allocations || []" />

          <h3 class="up-h">Beneficial owners</h3>
          <div class="ben-table">
            <div class="ben-head">
              <span>Entity</span><span>Amount</span><span>% of total</span>
            </div>
            <div v-for="b in upstreamResult.beneficiaries || []" :key="b.entity_id"
                 class="ben-row" :class="{ est: b.is_estimate }">
              <span class="ben-e">
                {{ b.entity_id }}<sup v-if="b.is_estimate" class="est-mark">est</sup>
              </span>
              <span class="ben-a">{{ fmtMoney(b.amount) }}</span>
              <span class="ben-p">{{ fmtPct((b.pct_of_total || 0) * 100) }}</span>
            </div>
          </div>
          <p v-if="upstreamResult.estimate_footnote" class="est-note">
            <sup class="est-mark">est</sup> {{ upstreamResult.estimate_footnote }}
          </p>

          <template v-if="upstreamResult.upstream_allocations?.length">
            <h3 class="up-h">Upstream allocation detail</h3>
            <p class="not-summable">
              {{ upstreamResult.upstream_not_summable }}
            </p>
            <div v-if="(upstreamResult.upstream_total_by_level || []).length"
                 class="level-totals">
              <span v-for="l in upstreamResult.upstream_total_by_level" :key="l.level">
                Level {{ l.level }}: <b>{{ fmtMoney(l.allocated) }}</b>
              </span>
            </div>
            <DataTable :columns="upstreamColumns" :rows="upstreamResult.upstream_allocations" />
          </template>
        </template>
      </template>
    </div>
  </div>
</template>

<style scoped>
.ownership { padding: 20px; }

.head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap; }
h2 { margin: 0 0 4px; font-size: 20px; }
.sub { margin: 0; color: #667; font-size: 13px; max-width: 70ch; }
.btn {
  padding: 7px 14px; border: 1px solid #ccd; background: #fff; border-radius: 6px;
  cursor: pointer; font-size: 13px;
}
.btn:hover:not(:disabled) { background: #f5f7fa; }
.btn:disabled { opacity: .6; cursor: default; }

.layout { display: grid; grid-template-columns: 290px 1fr; gap: 18px; margin-top: 18px; align-items: start; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

/* ── Investment picker ─────────────────────────────── */
.picker { border: 1px solid #e2e6ee; border-radius: 8px; background: #fff; overflow: hidden; }
.picker-controls { padding: 12px; border-bottom: 1px solid #eef1f5; display: flex; flex-direction: column; gap: 8px; }
.search { padding: 7px 10px; border: 1px solid #ccd; border-radius: 6px; font-size: 13px; width: 100%; }
.chk { font-size: 12.5px; color: #556; display: flex; align-items: center; gap: 6px; cursor: pointer; }
.count { margin: 0; font-size: 11.5px; color: #889; }

.inv-list { list-style: none; margin: 0; padding: 0; max-height: 640px; overflow-y: auto; }
.inv {
  width: 100%; text-align: left; background: none; border: none;
  border-bottom: 1px solid #f2f4f8; padding: 10px 12px; cursor: pointer;
  display: flex; flex-direction: column; gap: 3px;
}
.inv:hover { background: #f7f9fc; }
.inv.active { background: #eef4ff; box-shadow: inset 3px 0 0 #1d4e7e; }
.inv-name { font-size: 13px; font-weight: 600; color: #223; }
.inv-meta { display: flex; align-items: center; gap: 7px; }
.inv-meta code { font-size: 11px; color: #667; }
.inv-sub { font-size: 11.5px; color: #889; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
.dot.ok { background: #2e7d32; }
.dot.gap { background: #c77700; }

/* ── Chain ─────────────────────────────────────────── */
.chain { min-width: 0; }
.empty { border: 1px dashed #d5dae5; border-radius: 8px; padding: 46px; text-align: center; color: #889; }
.empty p { margin: 0; }

.chain-summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.stat {
  border: 1px solid #e2e6ee; border-radius: 8px; background: #fff;
  padding: 10px 16px; min-width: 120px;
}
.stat.warn { border-color: #e8c07a; background: #fffaf0; }
.stat .n { display: block; font-size: 22px; font-weight: 700; color: #223; font-variant-numeric: tabular-nums; }
.stat .l { display: block; font-size: 11.5px; color: #778; margin-top: 2px; }

.notes {
  border-left: 3px solid #c77700; background: #fff8ec;
  padding: 11px 14px; border-radius: 0 6px 6px 0; margin-bottom: 14px;
}
.notes-title { margin: 0 0 6px; font-size: 12px; font-weight: 700; color: #a35f00; text-transform: uppercase; letter-spacing: .04em; }
.notes ul { margin: 0; padding-left: 18px; }
.notes li { font-size: 12.5px; color: #664; margin-bottom: 4px; }

/* BOTH AXES, BOUNDED, WITH THE SCROLLBARS VISIBLE.
   A deep chain runs off the right and a wide level runs off the bottom, and an
   overlay scrollbar that only appears mid-gesture gives no hint either exists.
   Capping the height also keeps the level headings on screen (they are sticky
   below) instead of scrolling away with the page. */
.cols-scroll {
  overflow: auto;
  max-height: min(72vh, 780px);
  padding-bottom: 8px;
  border: 1px solid #e8ecf3;
  border-radius: 8px;
  background:
    linear-gradient(to right, #fff 30%, rgba(255,255,255,0)) left center,
    linear-gradient(to left, #fff 30%, rgba(255,255,255,0)) right center,
    radial-gradient(farthest-side at 0% 50%, rgba(31,45,61,.14), transparent) left center,
    radial-gradient(farthest-side at 100% 50%, rgba(31,45,61,.14), transparent) right center;
  background-repeat: no-repeat;
  background-size: 34px 100%, 34px 100%, 12px 100%, 12px 100%;
  background-attachment: local, local, scroll, scroll;
  scrollbar-width: thin;
  scrollbar-color: #b6c0ce #eef1f6;
}
.cols-scroll::-webkit-scrollbar { width: 12px; height: 12px; }
.cols-scroll::-webkit-scrollbar-track { background: #eef1f6; border-radius: 6px; }
.cols-scroll::-webkit-scrollbar-thumb {
  background: #b6c0ce; border-radius: 6px; border: 3px solid #eef1f6;
}
.cols-scroll::-webkit-scrollbar-thumb:hover { background: #93a1b3; }
.cols-scroll::-webkit-scrollbar-corner { background: #eef1f6; }
.cols {
  display: flex; gap: 40px; align-items: flex-start; min-width: min-content;
  position: relative;   /* the connector svg is positioned against this */
  padding: 12px 14px;
}

/* Connectors sit behind the cards and never take a click. */
.links { position: absolute; inset: 0; pointer-events: none; z-index: 0; overflow: visible; }
.node { position: relative; z-index: 1; }
.col { min-width: 244px; max-width: 244px; display: flex; flex-direction: column; gap: 10px; }
.col-head {
  display: flex; justify-content: space-between; align-items: baseline; gap: 8px;
  padding: 2px 0 6px; border-bottom: 2px solid #dde3ec;
  position: sticky; top: 0; z-index: 2; background: #fff;
}
.col-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: #7a8394; }
.col-total { font-size: 11px; color: #889; font-variant-numeric: tabular-nums; }

.node {
  border: 1px solid #e2e6ee; border-radius: 8px; background: #fff;
  padding: 10px 12px; display: flex; flex-direction: column; gap: 6px;
}
.node.pe { border-left: 3px solid #1d4e7e; }
.node.terminal { background: #fafbfc; }

.node-top { display: flex; align-items: flex-start; gap: 6px; }
.twist {
  background: none; border: none; cursor: pointer; padding: 0; width: 14px;
  color: #7a8394; font-size: 11px; line-height: 1.5; flex-shrink: 0;
}
.twist.spacer { cursor: default; }
.node-id { min-width: 0; }
.node-id code { font-size: 11.5px; font-weight: 700; color: #1d4e7e; display: block; }
.node-name { font-size: 12px; color: #556; display: block; word-break: break-word; }

.node-figs { display: flex; justify-content: flex-end; align-items: baseline; gap: 8px; padding-left: 20px; }
.amt { display: none; }  /* the dl below carries it, labelled */
.pct {
  font-size: 15px; font-weight: 700; color: #1d4e7e;
  font-variant-numeric: tabular-nums;
}
.pct em {
  font-style: normal; font-size: 9.5px; font-weight: 600; color: #7a8394;
  letter-spacing: .05em; text-transform: uppercase; margin-left: 3px;
}
.pct-par {
  font-size: 10.5px; color: #8a93a3; font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.node-figs { flex-wrap: wrap; row-gap: 2px; }

/* The raw commitment, kept but visibly subordinate: it is the right answer to
   a different question and must not be mistaken for this deal's number. */
.direct {
  margin: 4px 0 0 20px; font-size: 10.5px; color: #98a1ae;
  font-variant-numeric: tabular-nums;
}
.direct-note { font-style: italic; }

/* Balance breakdown */
.bal-btn {
  background: none; border: none; padding: 0; cursor: pointer;
  font: inherit; color: inherit; font-variant-numeric: tabular-nums;
  text-decoration: underline dotted #b9c2cf; text-underline-offset: 2px;
}
.bal-btn:hover { color: #1d4e7e; }
.caret { font-size: 8px; color: #8a93a3; margin-left: 2px; }
.bal dd.disputed { color: #a35f00; }

.bal-detail {
  margin: 5px 0 0 20px; padding: 7px 9px;
  background: #fff; border: 1px solid #e2e6ee; border-radius: 4px;
}
.bal-row {
  display: grid; grid-template-columns: 1fr auto 20px; gap: 8px;
  font-size: 10.5px; line-height: 1.5;
}
.bal-row.total {
  border-top: 1px solid #e2e6ee; margin-top: 3px; padding-top: 3px; font-weight: 700;
}
.bal-t { color: #556; }
.bal-n { color: #223; font-variant-numeric: tabular-nums; text-align: right; }
.bal-n.neg { color: #b3261e; }
.bal-c { color: #a8b0bc; text-align: right; }
.bal-warn {
  margin: 6px 0 0; font-size: 10px; color: #a35f00; line-height: 1.45;
}
.bal-warn code { background: #fff4e0; padding: 0 3px; border-radius: 2px; }
.bal-note {
  margin: 6px 0 0; font-size: 10px; color: #8a93a3; line-height: 1.5;
}

/* Committed vs balance. Two figures that are easy to confuse, so each is
   labelled and they line up on the decimal. */
.bal {
  display: grid; grid-template-columns: auto 1fr; gap: 1px 10px;
  margin: 0 0 0 20px; padding: 5px 8px;
  background: #f7f9fc; border-radius: 4px;
}
.bal dt {
  font-size: 9.5px; letter-spacing: .06em; text-transform: uppercase;
  color: #8a93a3; align-self: center;
}
.bal dd {
  margin: 0; text-align: right; font-size: 12.5px; font-weight: 600;
  color: #223; font-variant-numeric: tabular-nums;
}
.bal dd.none { color: #a0a7b4; font-weight: 400; font-style: italic; }
.bal dd.neg { color: #b3261e; }

.node.ubo { background: #fcfdfe; border-style: dashed; }
.badge.ubo { background: #eef2f7; color: #4a5768; }

.bal dt em {
  font-style: normal; text-transform: none; letter-spacing: 0;
  color: #aab2bf; margin-left: 4px; font-size: 9px;
  border-bottom: 1px dotted #ccd3dd; cursor: help;
}

.disagree {
  margin: 0 0 0 20px; font-size: 11px; color: #a35f00;
  background: #fff8ec; border-radius: 4px; padding: 3px 6px;
}

.node-wf { display: flex; flex-direction: column; gap: 4px; padding-left: 20px; }
.badge {
  font-size: 10.5px; font-weight: 600; padding: 2px 7px; border-radius: 10px;
  align-self: flex-start;
}
.badge.ok { background: #e8f5e9; color: #2e7d32; }
.badge.gap { background: #fff3e0; color: #b25f00; }
.wf-link { font-size: 11px; color: #1d4e7e; text-decoration: none; }
.wf-link:hover { text-decoration: underline; }

.trunc { margin: 0 0 0 20px; font-size: 11px; color: #99a; font-style: italic; }

.muted { color: #889; font-size: 13px; }
.pad { padding: 14px; }

/* ── Tabs ──────────────────────────────────────────── */
.tabs { display: flex; gap: 2px; margin-top: 16px; border-bottom: 1px solid #dde3ec; }
.tab {
  background: none; border: none; cursor: pointer;
  padding: 9px 16px; font-size: 13px; font-weight: 600; color: #7a8394;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tab:hover { color: #445; }
.tab.active { color: #1d4e7e; border-bottom-color: #1d4e7e; }

/* ── Upstream analysis ─────────────────────────────── */
.upstream { margin-top: 18px; }
.upstream > .sub { margin-bottom: 16px; }
.up-controls {
  display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap;
  border: 1px solid #e2e6ee; border-radius: 8px; background: #fff;
  padding: 14px; margin-bottom: 16px;
}
.fld { display: flex; flex-direction: column; gap: 4px; }
.fld span { font-size: 11.5px; font-weight: 600; color: #7a8394; text-transform: uppercase; letter-spacing: .04em; }
.fld select, .fld input {
  padding: 7px 10px; border: 1px solid #ccd; border-radius: 6px; font-size: 13px;
  min-width: 190px;
}
.btn.primary { background: #1d4e7e; color: #fff; border-color: #1d4e7e; }
.btn.primary:hover:not(:disabled) { background: #17405f; }
.err {
  border-left: 3px solid #c62828; background: #fdecea; color: #8d2019;
  padding: 11px 14px; border-radius: 0 6px 6px 0; font-size: 13px; margin-bottom: 14px;
}
.up-h { font-size: 13px; font-weight: 700; color: #445; margin: 20px 0 8px; }

fieldset.kind { border: none; margin: 0; padding: 0; }
fieldset.kind legend {
  font-size: 11.5px; font-weight: 600; color: #7a8394;
  text-transform: uppercase; letter-spacing: .04em; padding: 0 0 4px;
}
fieldset.kind label {
  font-size: 13px; color: #334; margin-right: 12px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 4px;
}

.ben-table { border: 1px solid #e2e6ee; border-radius: 6px; overflow: hidden; background: #fff; }
.ben-head, .ben-row {
  display: grid; grid-template-columns: 1fr 140px 90px; gap: 10px;
  padding: 7px 12px; align-items: baseline;
}
.ben-head {
  background: #f3f5f9; font-size: 10.5px; font-weight: 700; color: #7a8394;
  text-transform: uppercase; letter-spacing: .05em;
}
.ben-head span:not(:first-child), .ben-a, .ben-p { text-align: right; }
.ben-row { border-top: 1px solid #eef1f5; font-size: 13px; }
.ben-row.est { background: #fffaf0; }
.ben-a, .ben-p { font-variant-numeric: tabular-nums; }
.ben-e { font-weight: 600; color: #223; }
.est-mark {
  font-size: 8.5px; font-weight: 700; color: #b25f00; letter-spacing: .04em;
  margin-left: 2px; background: #fff3e0; padding: 1px 3px; border-radius: 2px;
  vertical-align: super;
}
.err.recon { border-left-color: #b3261e; background: #fdecea; }
.not-summable {
  margin: 0 0 6px; font-size: 11.5px; color: #7a6a52; line-height: 1.5;
  background: #fffaf0; border-left: 3px solid #e8c07a; padding: 7px 10px;
  border-radius: 0 4px 4px 0; max-width: 82ch;
}
.level-totals {
  display: flex; gap: 16px; flex-wrap: wrap; margin: 0 0 8px;
  font-size: 11.5px; color: #556;
}
.level-totals b { font-variant-numeric: tabular-nums; color: #223; }

.opening { margin-top: 4px; }
.opening .ben-head, .opening .ben-row { grid-template-columns: 1fr 170px 170px; }
.opening-note { margin: 6px 0 0; font-size: 11px; color: #8a93a3; }
.opening-note code {
  background: #eef1f6; padding: 1px 4px; border-radius: 2px; font-size: 10.5px;
}

.est-note {
  margin: 8px 0 0; font-size: 11.5px; color: #7a6a52; line-height: 1.55;
  max-width: 78ch; background: #fffaf0; border-left: 3px solid #e8c07a;
  padding: 8px 11px; border-radius: 0 4px 4px 0;
}
</style>
