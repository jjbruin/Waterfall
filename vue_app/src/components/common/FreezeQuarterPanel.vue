<script setup lang="ts">
/**
 * Freeze one HALF of a quarter, for every investor, from the tab that owns it.
 *
 * ONE COMPONENT, TWO TABS. The Snapshot tab and the One Pager tab ask the same
 * question about different halves, so this is written once and given a `part`.
 * Two copies would drift the moment one of them grew a state the other lacked —
 * and the halves are already the thing readers confuse, so the screens must not
 * describe them differently.
 *
 * IT DOES NOT DECIDE WHAT FREEZING MEANS. Every run goes to the same
 * `freeze-all/*` endpoint, which calls the one `freeze_part` core. This file
 * chooses WHO to ask about and HOW to report it, nothing else.
 *
 * WHY IT POSTS IN CHUNKS. The batch covers ~127 investors. Sent as one request
 * the screen can only show a spinner and hope, and a single long request is the
 * one most likely to hit a proxy timeout with no record of how far it got. The
 * endpoint already accepts an `investors` subset — that parameter exists and is
 * used by nothing else — so the client walks the list in slices and reports
 * real progress. Per-investor isolation is unchanged: each slice returns its
 * own rows, and a slice that fails outright is recorded against the investors
 * it covered rather than silently vanishing.
 */
import { ref, computed, watch, onMounted } from 'vue'
import api from '../../api/client'
import { useAuthStore } from '../../stores/auth'

const props = defineProps<{
  part: 'snapshot' | 'one_pagers'
  quarter: string
  /** Set while the host page is loading, so the button cannot run on a stale quarter. */
  disabled?: boolean
}>()
const emit = defineEmits<{ (e: 'frozen'): void }>()

const auth = useAuthStore()

const PART_LABEL: Record<string, string> = {
  snapshot: 'Snapshots', one_pagers: 'One Pagers',
}
const OTHER_LABEL: Record<string, string> = {
  snapshot: 'One Pagers', one_pagers: 'Snapshots',
}
const label = computed(() => PART_LABEL[props.part])
const otherLabel = computed(() => OTHER_LABEL[props.part])

// --- the quarter's state, per part -----------------------------------------
const status = ref<any>(null)
const statusError = ref<string | null>(null)
const statusLoading = ref(false)

const mine = computed(() => status.value?.parts?.[props.part] || null)

/** not frozen / partly frozen (N of M investors) / frozen. */
const stateLabel = computed(() => {
  const m = mine.value
  if (!m) return ''
  if (m.state === 'all') return `${label.value} frozen — all ${m.total} investors`
  if (m.state === 'partly') return `Partly frozen — ${m.frozen} of ${m.total} investors`
  return `Not frozen — 0 of ${m.total} investors`
})
const stateClass = computed(() => mine.value?.state || 'none')

const overlayPending = computed(() => !!status.value?.overlay?.pending)

async function loadStatus() {
  if (!props.quarter) { status.value = null; return }
  statusLoading.value = true
  statusError.value = null
  try {
    const res = await api.get('/api/portfolio-snapshot/quarter-status',
                              { params: { quarter: props.quarter } })
    status.value = res.data
    // A read that half-worked must say so: the counts would otherwise read as
    // "nothing is frozen", which is the one wrong answer that invites a
    // needless re-freeze.
    if (res.data?.read_error) statusError.value = res.data.read_error
  } catch (e: any) {
    status.value = null
    statusError.value = e?.response?.data?.error || e?.message || 'Could not read the quarter state'
  } finally {
    statusLoading.value = false
  }
}
onMounted(loadStatus)
watch(() => [props.quarter, props.part], loadStatus)

// --- the confirmation ------------------------------------------------------
const confirming = ref(false)
const onePagerCount = ref<any>(null)
const countLoading = ref(false)

async function openConfirm() {
  confirming.value = true
  result.value = null
  runError.value = null
  await loadStatus()
  // Only the One Pager button promises a page count, and only it pays for one.
  if (props.part === 'one_pagers') {
    countLoading.value = true
    onePagerCount.value = null
    try {
      const res = await api.get('/api/portfolio-snapshot/quarter-status', {
        params: { quarter: props.quarter, count_one_pagers: 1 },
      })
      onePagerCount.value = res.data?.one_pagers ?? null
    } catch {
      onePagerCount.value = null        // null, never 0 — see the endpoint
    } finally {
      countLoading.value = false
    }
  }
}

function closeConfirm() {
  confirming.value = false
  result.value = null
  runError.value = null
  progress.value = { done: 0, total: 0 }
}

// --- the run ---------------------------------------------------------------
const CHUNK = 10
const running = ref(false)
const progress = ref<{ done: number; total: number }>({ done: 0, total: 0 })
const result = ref<any>(null)
const runError = ref<string | null>(null)

const pct = computed(() => {
  const p = progress.value
  return p.total ? Math.round((p.done / p.total) * 100) : 0
})

const url = computed(() => props.part === 'snapshot'
  ? '/api/portfolio-snapshot/freeze-all/snapshots'
  : '/api/portfolio-snapshot/freeze-all/one-pagers')

async function run() {
  running.value = true
  runError.value = null
  result.value = null
  const rows: any[] = []
  try {
    const inv = await api.get('/api/portfolio-snapshot/investors')
    const codes: string[] = (inv.data?.investors || []).map((r: any) => r.code)
    if (!codes.length) throw new Error('No investors found to freeze.')
    progress.value = { done: 0, total: codes.length }

    for (let i = 0; i < codes.length; i += CHUNK) {
      const slice = codes.slice(i, i + CHUNK)
      try {
        // 207 is PARTIAL success, not failure — axios treats it as success and
        // the per-investor rows are the answer either way.
        const res = await api.post(url.value,
                                   { quarter: props.quarter, investors: slice })
        rows.push(...(res.data?.results || []))
      } catch (e: any) {
        // A whole slice failing is still per-investor news: record it against
        // the investors it covered rather than losing them from the report.
        const msg = e?.response?.data?.error || e?.message || 'request failed'
        for (const c of slice) rows.push({ investor: c, frozen: false, error: msg })
      }
      progress.value = { done: Math.min(i + CHUNK, codes.length), total: codes.length }
    }

    result.value = {
      quarter: props.quarter,
      part: props.part,
      investors: rows.length,
      frozen: rows.filter(r => r.frozen).length,
      skipped: rows.filter(r => r.skipped).length,
      failed: rows.filter(r => r.error).length,
      results: rows,
    }
    await loadStatus()
    emit('frozen')
  } catch (e: any) {
    runError.value = e?.response?.data?.error || e?.message || 'Freeze failed'
  } finally {
    running.value = false
  }
}
</script>

<template>
  <!-- ADMIN ONLY, and the whole block goes: a state line for an action the
       reader cannot take is just noise on the page. -->
  <div v-if="auth.isAdmin" class="fqp">
    <div class="fqp-head">
      <span :class="['fqp-state', stateClass]">
        <template v-if="statusLoading">Reading the quarter state…</template>
        <template v-else-if="stateLabel">{{ stateLabel }}</template>
        <template v-else>—</template>
      </span>
      <button class="btn-sm primary fqp-btn"
              :disabled="disabled || !quarter || running || mine?.state === 'all'"
              @click="openConfirm">
        Freeze {{ quarter }} {{ label }} — all investors
      </button>
    </div>

    <p v-if="statusError" class="fqp-err">
      The quarter state could not be read in full ({{ statusError }}) — the
      counts above may be short.
    </p>

    <!-- Requirement of ORDER, not of permission: freezing from live data first
         would leave the published figures unable to land, because this button
         skips whatever is already frozen. -->
    <p v-if="overlayPending" class="fqp-warn">
      This quarter has a published overlay built but not yet applied
      ({{ status?.overlay?.file }}). <strong>Apply the published PDFs first</strong> —
      freezing from live data now would skip those investors later, and what was
      sent would never be stored.
    </p>

    <div v-if="confirming" class="fqp-confirm">
      <strong>Freeze {{ quarter }} {{ label }} for every investor?</strong>
      <p>
        This freezes the <em>{{ label }}</em> for
        <strong>{{ mine?.total ?? '—' }}</strong> investor(s) in
        <strong>{{ quarter }}</strong>.
        <template v-if="part === 'one_pagers'">
          <template v-if="countLoading"> Counting One Pagers…</template>
          <template v-else-if="onePagerCount">
            About <strong>{{ onePagerCount.pages }}</strong> One Pager(s) would be
            built<template v-if="!onePagerCount.complete">
              (counted across {{ onePagerCount.investors_counted }} of
              {{ onePagerCount.investors }} investors, so treat it as a floor)</template>.
          </template>
          <template v-else> The One Pager count could not be worked out.</template>
        </template>
      </p>
      <p class="fqp-muted">
        <template v-if="mine?.frozen">
          {{ mine.frozen }} investor(s) are already frozen for this half and will be
          <strong>skipped, not re-frozen</strong> — including anything frozen from the
          published PDFs.
        </template>
        <template v-else>
          Nothing in this quarter is frozen for this half yet.
        </template>
        The {{ otherLabel }} are left alone. Later data changes cannot move what is
        stored; an admin can Re-freeze or Unfreeze afterwards, with a reason.
      </p>

      <div v-if="running || progress.total" class="fqp-progress">
        <div class="fqp-bar"><div class="fqp-fill" :style="{ width: pct + '%' }"></div></div>
        <span>{{ progress.done }} of {{ progress.total }} investor(s)</span>
      </div>

      <p v-if="runError" class="fqp-err">{{ runError }}</p>

      <!-- PER INVESTOR. A single count cannot say WHICH investor failed, and a
           batch that half-ran is exactly when that matters. -->
      <div v-if="result" class="fqp-results">
        <p>
          <strong>{{ result.frozen }}</strong> frozen,
          <strong>{{ result.skipped }}</strong> already frozen,
          <strong>{{ result.failed }}</strong> failed,
          of {{ result.investors }} investor(s).
        </p>
        <ul>
          <li v-for="r in result.results" :key="r.investor"
              :class="{ bad: r.error, skip: r.skipped }">
            <span class="who">{{ r.investor }}</span>
            <span v-if="r.error">{{ r.error }}</span>
            <span v-else-if="r.skipped">{{ r.reason }}</span>
            <span v-else>
              frozen — version {{ r.receipt?.version }},
              {{ r.receipt?.one_pager_count ?? 0 }} One Pager(s)
            </span>
          </li>
        </ul>
      </div>

      <div class="fqp-actions">
        <button v-if="!result" class="btn-sm primary" :disabled="running"
                @click="run">
          {{ running ? 'Freezing…' : `Freeze ${quarter} ${label}` }}
        </button>
        <button class="btn-sm" :disabled="running" @click="closeConfirm">
          {{ result ? 'Close' : 'Cancel' }}
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.fqp { margin: 8px 0; font-size: 12px; }
.fqp-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.fqp-btn { margin-left: auto; }
.fqp-state { font-weight: 600; }
.fqp-state.none { color: #6b7684; }
.fqp-state.partly { color: #9a6700; }
.fqp-state.all { color: #1a7f37; }
.fqp-warn {
  margin: 6px 0 0; padding: 6px 8px; border-radius: 4px;
  background: #fff8e5; border: 1px solid #f0d8a8; color: #7a5b00; line-height: 1.45;
}
.fqp-err {
  margin: 6px 0 0; padding: 6px 8px; border-radius: 4px;
  background: #fdf1f1; border: 1px solid #f0c9c9; color: #b4232c; line-height: 1.45;
}
.fqp-confirm {
  margin-top: 8px; padding: 10px; border: 1px solid #d7dde5;
  border-radius: 4px; background: #fbfcfe;
}
.fqp-confirm strong { font-size: 13px; }
.fqp-confirm p { margin: 0 0 8px; line-height: 1.45; }
.fqp-muted { color: #6b7684; }
.fqp-progress { display: flex; align-items: center; gap: 8px; margin: 8px 0; }
.fqp-bar {
  flex: 1; height: 6px; background: #e6eaf0; border-radius: 3px; overflow: hidden;
}
.fqp-fill { height: 100%; background: #2f6feb; transition: width .2s ease; }
.fqp-results { margin: 8px 0; }
.fqp-results ul { margin: 4px 0 0; padding-left: 0; list-style: none; max-height: 240px; overflow-y: auto; }
.fqp-results li { display: flex; gap: 8px; padding: 1px 0; }
.fqp-results li .who { min-width: 92px; font-weight: 600; }
.fqp-results li.bad { color: #b4232c; }
.fqp-results li.skip { color: #6b7684; }
.fqp-actions { display: flex; gap: 8px; }
</style>
