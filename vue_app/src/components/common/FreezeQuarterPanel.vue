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
 * IT STARTS A JOB AND WATCHES IT. The batch covers ~130 investors and runs on
 * a background thread server-side; this POSTs once, gets a job id back, and
 * polls. Run inside the request it took the app down for ~35 minutes on Sep 29
 * 2026 — one gunicorn worker, one request, everything else queued behind it.
 *
 * A FAILED POLL IS NOT A FAILED JOB. The work is server-side and survives a
 * dropped poll, so polling continues and only says so after it keeps failing.
 * A page opened mid-run picks the existing job up rather than showing nothing.
 */
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
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


// FREEZING SWITCHED OFF APP-WIDE. The server refuses regardless of what this
// says; the flag only lets the screen explain a disabled button instead of
// rendering a dead control that fails on click. Treated as DISABLED until the
// status read says otherwise, so a failed read cannot present the button as
// live — the same fail-closed direction as the server gate.
const freezeEnabled = computed(() => status.value?.freeze_enabled === true)
const freezeOffReason = computed(
  () => status.value?.freeze_disabled_reason
        || 'Freezing is temporarily disabled.')

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
  job.value = null
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
  // Closing the panel does NOT stop the job — it is server-side. It only stops
  // watching, and reopening picks it up again.
  stopPolling()
  confirming.value = false
  job.value = null
  runError.value = null
}

// --- the run ---------------------------------------------------------------
const running = ref(false)
const job = ref<any>(null)
const pollErrors = ref(0)
const runError = ref<string | null>(null)

const pct = computed(() => {
  const j = job.value
  return j && j.total ? Math.round(((j.done || 0) / j.total) * 100) : 0
})
const finished = computed(() => !!job.value && job.value.status !== 'running')

const url = computed(() => props.part === 'snapshot'
  ? '/api/portfolio-snapshot/freeze-all/snapshots'
  : '/api/portfolio-snapshot/freeze-all/one-pagers')

const POLL_MS = 1500
let poller: any = null

async function pollOnce(id: number) {
  try {
    const r = await api.get(`/api/portfolio-snapshot/freeze-job/${id}`)
    job.value = r.data
    if (r.data?.status && r.data.status !== 'running') {
      stopPolling()
      running.value = false
      await loadStatus()
      emit('frozen')
    }
  } catch (e: any) {
    // A failed poll is not a failed job — the job runs server-side. Keep
    // polling; say so only if it keeps failing.
    pollErrors.value += 1
    if (pollErrors.value > 8) {
      stopPolling()
      running.value = false
      runError.value = 'Lost contact with the job. It may still be running — '
        + 'reopen this page to pick it up.'
    }
  }
}

function stopPolling() {
  if (poller) { clearInterval(poller); poller = null }
}
onUnmounted(stopPolling)

function watchJob(id: number) {
  pollErrors.value = 0
  stopPolling()
  poller = setInterval(() => pollOnce(id), POLL_MS)
  pollOnce(id)
}

async function run() {
  running.value = true
  runError.value = null
  job.value = null
  try {
    const res = await api.post(url.value, { quarter: props.quarter })
    job.value = res.data
    watchJob(res.data.id)
  } catch (e: any) {
    running.value = false
    const d = e?.response?.data
    // 409 = another job is already running. That is an answer, not a fault.
    runError.value = d?.error || e?.message || 'Freeze failed'
  }
}

// --- unfreeze this quarter's half, for everyone --------------------------
//
// NOT A JOB. There is no report to assemble and no deal to build: the server
// archives and clears in one transaction, so this returns when it is done.
// Deliberately NOT gated on FREEZE_ENABLED — with freezing off, undoing a
// mistake made before the switch has to stay possible.
const unfreezing = ref(false)
const unfreezeOpen = ref(false)
const unfreezeReason = ref('')
const unfreezeCount = ref<number | null>(null)
const unfreezeResult = ref<any>(null)
const unfreezeError = ref<string | null>(null)

async function openUnfreeze() {
  unfreezeOpen.value = true
  unfreezeResult.value = null
  unfreezeError.value = null
  unfreezeCount.value = null
  try {
    const r = await api.get('/api/portfolio-snapshot/unfreeze-quarter/preview',
                            { params: { quarter: props.quarter, part: props.part } })
    unfreezeCount.value = r.data?.investors ?? null
  } catch {
    unfreezeCount.value = null      // null, never 0 — 0 would read as "none"
  }
}

async function doUnfreeze() {
  if (!unfreezeReason.value.trim()) return
  unfreezing.value = true
  unfreezeError.value = null
  try {
    const r = await api.post('/api/portfolio-snapshot/unfreeze-quarter', {
      quarter: props.quarter, part: props.part,
      reason: unfreezeReason.value.trim(),
    })
    unfreezeResult.value = r.data
    unfreezeReason.value = ''
    await loadStatus()
    emit('frozen')                  // the host reloads either way
  } catch (e: any) {
    unfreezeError.value = e?.response?.data?.error || e?.message || 'Unfreeze failed'
  } finally {
    unfreezing.value = false
  }
}

// A page opened mid-run picks the bar up rather than showing nothing.
onMounted(async () => {
  try {
    const r = await api.get('/api/portfolio-snapshot/freeze-job/active')
    const j = r.data?.job
    if (j && j.part === props.part && j.quarter === props.quarter) {
      job.value = j
      running.value = true
      confirming.value = true
      watchJob(j.id)
    }
  } catch { /* nothing running, or not readable — the button still works */ }
})

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
              :disabled="!freezeEnabled || disabled || !quarter || running
                         || mine?.state === 'all'"
              :title="freezeEnabled ? '' : freezeOffReason"
              @click="openConfirm">
        Freeze {{ quarter }} {{ label }} — all investors
      </button>
    </div>

    <!-- SAID, NOT JUST GREYED OUT. A disabled button with no reason reads as a
         bug or a missing permission; this is neither, and it is temporary. -->
    <p v-if="!freezeEnabled" class="fqp-off">{{ freezeOffReason }}</p>

    <p v-if="statusError" class="fqp-err">
      The quarter state could not be read in full ({{ statusError }}) — the
      counts above may be short.
    </p>


    <!-- UNFREEZING IS THE ADMIN'S WAY BACK, and stays available even with
         freezing switched off. One action per half; no investor to choose. -->
    <div v-if="mine && mine.frozen > 0" class="fqp-unfreeze-row">
      <button class="btn-sm" :disabled="unfreezing" @click="openUnfreeze">
        Unfreeze {{ quarter }} {{ label }}…
      </button>
    </div>

    <div v-if="unfreezeOpen" class="fqp-confirm">
      <strong>Unfreeze {{ quarter }} {{ label }} for every investor?</strong>
      <p>
        This returns the <em>{{ label }}</em> to live for
        <strong>{{ unfreezeCount ?? '—' }}</strong> investor(s) in
        <strong>{{ quarter }}</strong>.
        Every affected investor's stored copy is archived to history first — it
        is not lost — and the {{ otherLabel }} are left exactly as they are.
      </p>
      <p class="fqp-muted">
        A reason is required, and it is stored with the archived copy.
      </p>
      <input v-model="unfreezeReason" class="fqp-reason" type="text"
             placeholder="Why is this being unfrozen?" />
      <p v-if="unfreezeError" class="fqp-err">{{ unfreezeError }}</p>
      <p v-if="unfreezeResult" class="fqp-muted">
        Unfroze <strong>{{ unfreezeResult.investors }}</strong> investor(s);
        {{ unfreezeResult.archived }} archived to history,
        {{ unfreezeResult.rows_kept }} kept because the {{ otherLabel }} are
        still frozen.
      </p>
      <div class="fqp-actions">
        <button v-if="!unfreezeResult" class="btn-sm primary"
                :disabled="unfreezing || !unfreezeReason.trim()"
                @click="doUnfreeze">
          {{ unfreezing ? 'Unfreezing…' : `Unfreeze ${quarter} ${label}` }}
        </button>
        <button class="btn-sm" :disabled="unfreezing"
                @click="unfreezeOpen = false; unfreezeResult = null">
          {{ unfreezeResult ? 'Close' : 'Cancel' }}
        </button>
      </div>
    </div>

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

      <!-- LIVE, from the job row the server updates after EVERY investor. A
           bar that only moves when the work is done is not a progress bar. -->
      <div v-if="job" class="fqp-progress">
        <div class="fqp-bar"><div class="fqp-fill" :style="{ width: pct + '%' }"></div></div>
        <span>{{ job.done || 0 }} of {{ job.total || 0 }} investor(s)</span>
      </div>
      <p v-if="job" class="fqp-muted">
        <strong>{{ job.frozen || 0 }}</strong> frozen,
        <strong>{{ job.skipped || 0 }}</strong> already frozen,
        <strong>{{ job.failed || 0 }}</strong> failed.
        <template v-if="job.status === 'running'">Running in the background —
          you can leave this page.</template>
        <template v-else-if="job.status === 'interrupted'">
          <strong>Interrupted</strong> — the worker restarted. Investors frozen
          before that are still frozen; run it again to finish the rest.
        </template>
        <template v-else-if="job.status === 'failed'">
          <strong>Failed</strong>{{ job.message ? ` — ${job.message}` : '' }}
        </template>
        <template v-else>Finished.</template>
        <template v-if="job.stats?.deal_builds != null">
          Built {{ job.stats.deal_builds }} deal(s) once and reused them
          {{ job.stats.deal_reuses }} time(s).
        </template>
      </p>

      <p v-if="runError" class="fqp-err">{{ runError }}</p>

      <!-- PER INVESTOR. A single count cannot say WHICH investor failed, and a
           batch that half-ran is exactly when that matters. Only present once
           the job has finished; while it runs the counts above are the answer. -->
      <div v-if="job?.results" class="fqp-results">
        <ul>
          <li v-for="r in job.results" :key="r.investor"
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
        <button v-if="!job" class="btn-sm primary" :disabled="running"
                @click="run">
          {{ running ? 'Starting…' : `Freeze ${quarter} ${label}` }}
        </button>
        <button class="btn-sm" @click="closeConfirm">
          {{ finished ? 'Close' : 'Hide' }}
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
.fqp-off {
  margin: 6px 0 0; padding: 6px 8px; border-radius: 4px;
  background: #eef1f5; border: 1px solid #d7dde5; color: #48505c;
  line-height: 1.45;
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
.fqp-unfreeze-row { margin-top: 6px; }
.fqp-reason { width: 100%; padding: 4px 6px; margin-bottom: 8px;
  border: 1px solid #d7dde5; border-radius: 4px; font-size: 12px; }
</style>
