<template>
  <div class="line-mapping">
    <!-- ── Step 1: the file ─────────────────────────────────────────────── -->
    <div class="lm-head">
      <div>
        <h4>{{ title }}</h4>
        <p class="lm-note">{{ blurb }}</p>
      </div>
      <div class="lm-file no-print">
        <input type="file" ref="fileInput" accept=".xlsx,.xls,.csv" style="display:none"
               @change="onFile" />
        <button class="btn-secondary" :disabled="parsing || !editable"
                @click="$refs.fileInput.click()">
          {{ parsed ? 'Choose a different file' : 'Choose file' }}
        </button>
        <span v-if="parsed" class="lm-filename">{{ parsed.filename }}</span>
      </div>
    </div>

    <div v-if="parsing" class="loading-text">Reading the spreadsheet...</div>
    <div v-if="loadingDraft" class="loading-text">Looking for saved mapping work...</div>
    <div v-if="error" class="lm-error">{{ error }}</div>

    <!-- Your work is here. It was not, and that was the complaint. -->
    <div v-if="draftLoaded" class="lm-resumed">
      <template v-if="draftLoaded.status === 'committed'">
        <strong>Applied mapping.</strong>
        {{ draftLoaded.mapped_count }} of {{ draftLoaded.line_count }} line(s) mapped
        from <em>{{ draftLoaded.filename }}</em><template v-if="draftLoaded.updated_by">,
        by {{ draftLoaded.updated_by }}</template>. Change anything and apply again to
        replace it.
      </template>
      <template v-else>
        <strong>Picked up where you left off.</strong>
        {{ draftLoaded.mapped_count }} of {{ draftLoaded.line_count }} line(s) mapped
        from <em>{{ draftLoaded.filename }}</em><template v-if="draftLoaded.updated_at">,
        saved {{ draftLoaded.updated_at }}</template>. Nothing has been imported yet.
      </template>
      <button class="lm-discard no-print" :disabled="!editable" @click="discardDraft">
        Start over
      </button>
    </div>

    <template v-if="parsed">
      <!-- Budgeted occupancy is read off its own row and kept out of the lines below,
           so it cannot be mapped to an account. Said out loud so the analyst can see
           what was read -- and catch a wrong reading before it is imported. -->
      <p v-if="source === 'budget' && parsed.occupancy?.by_period" class="lm-note lm-occ">
        <strong>Budgeted occupancy</strong> read from "{{ parsed.occupancy.label }}":
        <template v-for="(q, i) in occQuarters" :key="q.q">{{ i ? ', ' : '' }}{{ q.q }}
          {{ q.v.toFixed(1) }}%</template>.
        <template v-if="parsed.occupancy.basis === 'annual'">
          One figure on the row, applied to every month.
        </template>
        <span v-if="parsed.occupancy.rejected_periods?.length" class="warn-note">
          {{ parsed.occupancy.rejected_periods.length }} month(s) outside 0–100% were not read.
        </span>
        It is saved with the budget on import.
      </p>
      <p v-else-if="source === 'budget'" class="lm-note">
        No occupancy row found. To load budgeted occupancy, add a row labelled
        "Budgeted Occupancy" with a percentage for each month (or one for the year).
      </p>
      <p class="lm-note">
        {{ parsed.lines.length }} line(s), {{ parsed.periods.length }} month(s)
        — {{ fmtPeriod(parsed.periods[0]) }} to {{ fmtPeriod(parsed.periods[parsed.periods.length - 1]) }}.
        <span v-if="source === 'argus' && parsed.suggested_count">
          {{ parsed.suggested_count }} line(s) pre-filled from the Argus keyword rules —
          <strong>check them</strong>, they are a guess.
        </span>
        <span v-else-if="source === 'budget'">
          <template v-if="parsed.stated_account_count">
            {{ parsed.stated_account_count }} line(s) carry an account number in the
            spreadsheet and are filled in from it — that is read, not guessed.
          </template>
          <template v-if="parsed.history_count">
            {{ parsed.history_count }} line(s) have been mapped before and show what was
            chosen last time.
          </template>
          <template v-if="!parsed.stated_account_count && !parsed.history_count">
            Nothing is pre-filled: this sheet states no account numbers and none of these
            lines has been mapped before, so every line is yours to assign.
          </template>
        </span>
        <span v-if="parsed.unknown_accounts && parsed.unknown_accounts.length" class="warn-note">
          {{ parsed.unknown_accounts.length }} line(s) name an account we do not carry
          ({{ parsed.unknown_accounts.map(u => u.account).join(', ') }}) — left for you
          rather than dropped.
        </span>
        <span v-if="!cats.has_history" class="warn-note">
          This deal has no recent actuals, so the list is not ranked and there are no defaults from history.
        </span>
      </p>

      <!-- ── Step 2: the mapping ───────────────────────────────────────── -->
      <div class="lm-coa-toggle no-print">
        <button class="btn-secondary lm-coa-btn" @click="toggleCoa">
          {{ coaOpen ? 'Hide the chart of accounts' : 'Show the chart of accounts' }}
        </button>
        <!-- Asked for as "only pick up rows with a 3+ digit acct number, so we're not
             pulling in blank rows". Offered rather than imposed: a sheet with no
             account numbers at all would show nothing, and the count says what is
             being held back so it is never a silent drop. -->
        <label v-if="parsed.stated_account_count">
          <input type="checkbox" v-model="onlyNumbered" />
          Only show lines with an account number
          <span class="lm-note-inline">({{ unnumberedCount }} hidden)</span>
        </label>
        <span class="lm-note-inline">
          The account number is the mapping. The category is whatever our chart of
          accounts says that account is.
        </span>
      </div>
      <!-- Suggested, priced, and off until ticked. Automating it would put $20,000
           into every valuation that nobody typed and nobody can see. -->
      <div v-if="parsed.proposed_lines && parsed.proposed_lines.length" class="lm-proposed">
        <strong>Lines we suggest adding</strong>
        <div v-for="pl in parsed.proposed_lines" :key="pl.account" class="lm-proposed-row">
          <label>
            <input type="checkbox" :disabled="!editable"
                   :checked="!!acceptedProposals[pl.account]"
                   @change="toggleProposal(pl, $event.target.checked)" />
            {{ pl.label }} — {{ pl.account }} {{ pl.category }},
            {{ fmtCurrency(pl.amount) }} over {{ pl.months }} month(s)
          </label>
          <span class="lm-note-inline">{{ pl.why }}</span>
          <input v-if="acceptedProposals[pl.account]" type="number" class="lm-proposed-amt"
                 :value="acceptedProposals[pl.account].amount" :disabled="!editable"
                 @change="setProposalAmount(pl, $event.target.value)" />
        </div>
      </div>

      <!-- The chart in statement order, beside the work rather than in another tab:
           the question "is this income, opex, below the line or capex" comes up while
           mapping, not before it. -->
      <div v-if="coaOpen" class="lm-coa">
        <div v-if="coaLoading" class="loading-text">Loading the chart of accounts...</div>
        <template v-else>
          <p class="lm-note">
            {{ coa.account_count }} accounts in statement order.
            <template v-if="coa.ranked_for_deal">
              Accounts this deal has used recently are marked.
            </template>
          </p>
          <div v-for="sec in coa.sections" :key="sec.key" class="lm-coa-sec">
            <h5>{{ sec.title }} <span class="lm-note-inline">{{ sec.note }}</span></h5>
            <div v-for="c in sec.categories" :key="c.category" class="lm-coa-cat">
              <span class="lm-coa-catname">{{ c.category }}</span>
              <span v-for="a in c.accounts" :key="a.account" class="lm-coa-acct"
                    :class="{ 'lm-coa-used': a.used_by_deal }"
                    :title="a.description || ''">{{ a.account }}</span>
            </div>
            <div v-if="sec.subtotal_after" class="lm-coa-sub">= {{ sec.subtotal_after }}</div>
          </div>
        </template>
      </div>

      <div class="table-scroll">
        <table class="data-table lm-table">
          <thead>
            <tr>
              <th>Line on the spreadsheet</th>
              <th class="num">Total</th>
              <th>Category</th>
              <th>Account</th>
              <th class="ctr">Flip sign</th>
              <th class="num">As imported</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="line in visibleLines" :key="line.row"
                :class="{ 'lm-subtotal': line.looks_like_total, 'lm-mapped': !!m(line.row).account }">
              <td>
                {{ line.label }}
                <span v-if="line.looks_like_total" class="lm-tag">subtotal</span>
                <span v-if="m(line.row).from_keywords" class="lm-tag lm-guess">keyword guess</span>
                <!-- Where a pre-fill came from decides how much to trust it. An
                     account number the sheet states is a fact; a prior mapping is a
                     decision somebody made; a keyword match is a guess. -->
                <span v-if="m(line.row).from_file" class="lm-tag lm-stated"
                      title="The account number is in the spreadsheet — read, not inferred">
                  acct {{ line.stated_account }} from the file
                </span>
                <span v-if="m(line.row).from_history" class="lm-tag lm-prior"
                      title="Mapped this way before">as mapped before</span>
                <span v-if="line.prior_mapping && !m(line.row).from_history"
                      class="lm-prior-note"
                      :title="'Last mapped ' + (line.prior_mapping.last_seen || '')">
                  last time: {{ line.prior_mapping.account }}
                  <template v-if="line.prior_mapping.category">({{ line.prior_mapping.category }})</template>
                  <template v-if="!line.prior_mapping.same_deal">on {{ line.prior_mapping.vcode }}</template>
                </span>
                <span v-if="line.months < parsed.periods.length" class="lm-tag lm-partial">
                  {{ line.months }} of {{ parsed.periods.length }} mo
                </span>
              </td>
              <td class="num">{{ fmtCurrency(line.total) }}</td>
              <!-- READ ONLY. Jack, Sep 22 2026: "The category dropdown should come
                   out entirely and just display whatever the account dictates." It was
                   two controls for one decision, and they could disagree — the analyst
                   set the account, the category stayed where it was, and submit
                   refused. The account is the mapping; this says where it lands. -->
              <td class="lm-derived">
                <template v-if="m(line.row).account">{{ categoryOf(line.row) || '—' }}</template>
                <span v-else class="lm-note-inline">not imported</span>
              </td>
              <td>
                <select :value="m(line.row).account || ''" :disabled="!editable"
                        @change="setAccount(line.row, $event.target.value)">
                  <option value="">— not imported —</option>
                  <optgroup v-if="dealAccounts.length" label="Used by this deal">
                    <option v-for="a in dealAccounts" :key="'d-' + a.account" :value="a.account">
                      {{ a.account }} {{ a.description || '' }} — {{ a.category }}
                    </option>
                  </optgroup>
                  <optgroup label="Whole chart of accounts">
                    <option v-for="a in allAccounts" :key="'all-' + a.account" :value="a.account">
                      {{ a.account }} {{ a.description || '' }} — {{ a.category }}
                    </option>
                  </optgroup>
                </select>
              </td>
              <td class="ctr">
                <input type="checkbox" :checked="!!m(line.row).flip" :disabled="!editable || !m(line.row).account"
                       @change="setFlip(line.row, $event.target.checked)" />
              </td>
              <td class="num" :class="{ neg: imported(line) < 0 }">
                {{ m(line.row).account ? fmtCurrency(imported(line)) : '—' }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- ── Step 3: does it tie? ──────────────────────────────────────── -->
      <div class="lm-split">
        <div class="lm-recon">
          <h5>Does it tie to the spreadsheet?</h5>
          <table class="data-table">
            <thead>
              <tr>
                <th></th>
                <th class="num">On the spreadsheet</th>
                <th class="num">As we will import it</th>
                <th class="num">Difference</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in reconRows" :key="r.line" :class="{ 'row-total': r.line === 'noi' }">
                <td>{{ reconLabel(r.line) }}</td>
                <td class="num">{{ r.stated === null ? '—' : fmtCurrency(r.stated) }}</td>
                <td class="num">{{ fmtCurrency(r.computed) }}</td>
                <td class="num" :class="{ neg: (r.difference ?? 0) !== 0 }">
                  {{ r.difference === null ? '—' : fmtCurrency(r.difference) }}
                </td>
              </tr>
            </tbody>
          </table>
          <p class="lm-note">
            A difference is <strong>not an error</strong> — spreadsheets carry subtotal rows and
            skipping them is correct. It is here so you can tell a skipped subtotal from a
            line you missed.
            <span v-if="!check?.reconciliation?.has_stated_totals">
              This sheet states no totals of its own, so only our side is shown.
            </span>
          </p>
          <!-- NOI here is the Budget column's NOI. Anything mapped below the line is
               imported but sits outside it — if the partner's own NOI includes one of
               these (replacement reserves, partnership costs), that is the difference. -->
          <p v-if="outsideNoi.length" class="lm-note">
            Imported, but below NOI (as on the Budget column):
            <span v-for="(o, i) in outsideNoi" :key="o.account">{{ i ? ', ' : '' }}{{ o.account }}
              {{ fmtCurrency(o.amount) }}</span>.
          </p>
        </div>

        <!-- ── Step 4: what to look at ─────────────────────────────────── -->
        <div class="lm-checks">
          <h5>
            Checks
            <span class="lm-count">{{ check ? check.mapped_count : 0 }} of
              {{ parsed.lines.length }} lines mapped</span>
          </h5>
          <div v-if="checking" class="loading-text">Checking...</div>
          <template v-else-if="check">
            <div v-for="(b, i) in check.blocking" :key="'b' + i" class="lm-block">
              {{ b.message }}
            </div>
            <!-- CRITICAL ONLY up front: a sign opposite to the deal's history, a figure
                 off by an order of magnitude, a negative NOI. Asset management, Sep 25
                 2026, asked for critical checks only; the rest are kept and folded. -->
            <div v-for="(w, i) in criticalWarnings" :key="'w' + i" class="lm-warn">
              {{ w.message }}
            </div>
            <p v-if="!check.blocking.length && !criticalWarnings.length" class="lm-clean">
              No critical issues.
            </p>
            <div v-if="infoWarnings.length" class="lm-info">
              <button class="lm-more" @click="showInfo = !showInfo">
                {{ showInfo ? 'Hide' : 'Show' }} {{ infoWarnings.length }} note(s) for reference
              </button>
              <template v-if="showInfo">
                <div v-for="(w, i) in infoWarnings" :key="'n' + i" class="lm-note-item">
                  {{ w.message }}
                  <button v-if="w.accounts?.length" class="lm-more" @click="toggle(i)">
                    {{ open[i] ? 'hide' : 'show all' }}
                  </button>
                  <ul v-if="open[i] && w.accounts" class="lm-acct-list">
                    <li v-for="a in w.accounts" :key="a.account">
                      {{ a.account }} {{ a.description }} — {{ a.months }} mo,
                      {{ fmtCurrency(a.prior_total) }}
                    </li>
                  </ul>
                </div>
              </template>
            </div>
          </template>
        </div>
      </div>

      <!-- ── Step 5: commit ────────────────────────────────────────────── -->
      <div class="form-actions no-print">
        <button class="btn-primary" :disabled="!canCommit" @click="doCommit">
          {{ committing ? 'Importing...' : commitLabel }}
        </button>
        <span class="lm-draft-state" :class="'lm-draft-' + (draftState || 'idle')">
          <template v-if="draftState === 'saving'">Saving your mapping...</template>
          <template v-else-if="draftState === 'saved'">
            Mapping saved{{ draftSavedAt ? ' at ' + draftSavedAt : '' }} — it will be
            here if you come back.
          </template>
          <template v-else-if="draftState === 'error'">
            Could not save your mapping. Your work is still on screen — do not refresh.
          </template>
        </span>
        <span v-if="check && !check.can_import" class="lm-blocked-note">
          Resolve the {{ check.blocking.length }} blocking item(s) above first.
        </span>
        <span v-else-if="criticalWarnings.length" class="lm-warn-note">
          {{ criticalWarnings.length }} warning(s) — you can import anyway.
        </span>
      </div>

      <div v-if="result" class="lm-result">
        Imported {{ result.rows_written }} row(s)<template v-if="result.rows_replaced">,
        replacing {{ result.rows_replaced }} from a previous version</template>
        into the <strong>{{ result.column }}</strong> column.
        <template v-if="result.periods?.length">
          {{ fmtPeriod(result.periods[0]) }} to {{ fmtPeriod(result.periods[result.periods.length - 1]) }}.
        </template>
        <template v-if="result.occupancy_months">
          Budgeted occupancy saved for {{ result.occupancy_months }} month(s).
        </template>
        <div class="lm-note">
          Re-import as many times as you need — the same months are replaced, not stacked,
          so the last version you load is the one the comparison uses.
        </div>
      </div>
    </template>
  </div>
</template>

<script setup>
/**
 * ONE screen for the two spreadsheets that feed the budget comparison.
 *
 * The comparison is Estimate | Budget | Valuation. Budget comes from the partner's
 * workbook, Valuation from the appraiser's Argus download, and they are the same job:
 * take somebody else's line names and decide which of our categories each belongs to.
 * So they get the same component, and `source` is the only thing that differs.
 *
 * Category FIRST, then an account within it. The categories are the ~27 rows the
 * comparison actually renders, which is the vocabulary the analyst is already reading on
 * screen; a bare account number asks them to translate in their head from a 169-item
 * list into a row they cannot see. The account is still required, because the supplement
 * stores vAccount and NOI, FAD, DSCR and the waterfall all read individual accounts —
 * it defaults to the one this deal used most, so the common case is one click.
 */
import { ref, computed, watch, onMounted } from 'vue'
import api from '@/api/client'

const props = defineProps({
  recordId: { type: [Number, String], required: true },
  source: { type: String, required: true },      // 'budget' | 'argus'
  editable: { type: Boolean, default: true },
})
const emit = defineEmits(['committed'])

const parsed = ref(null)
const cats = ref({ categories: [], has_history: true })
const mapping = ref({})
const check = ref(null)
const result = ref(null)
const error = ref('')
const parsing = ref(false)
const checking = ref(false)
const committing = ref(false)
const open = ref({})

// Draft state. The mapping used to live only here, so a refresh threw away twenty
// minutes of judgement and re-opening after a successful import showed a blank page.
const draftState = ref('')          // '' | 'saving' | 'saved' | 'error'
const draftSavedAt = ref('')
const draftLoaded = ref(null)       // the stored mapping this screen resumed
const loadingDraft = ref(false)
const uploadedName = ref('')

const title = computed(() => props.source === 'argus'
  ? "Appraiser's Argus cash flow"
  : "Partner's monthly budget")

const blurb = computed(() => props.source === 'argus'
  ? 'Review how each Argus line is coded before it feeds the Valuation column. The '
    + 'keyword rules pre-fill a guess — this is where you correct it.'
  : 'Load the budget the partner sent. It feeds the Budget column, and re-importing '
    + 'replaces it, so you can keep loading revisions until the version is final.')

// It IS the save, and it did not say so: asset management reported not finding a save
// button on a screen whose only button was this one.
const commitLabel = computed(() => props.source === 'argus'
  ? 'Save and apply to the Valuation column'
  : 'Save and import into the Budget column')

const onlyNumbered = ref(false)
const acceptedProposals = ref({})

function toggleProposal(pl, on) {
  const next = { ...acceptedProposals.value }
  if (on) next[pl.account] = { ...pl }
  else delete next[pl.account]
  acceptedProposals.value = next
  runCheck()
}
function setProposalAmount(pl, v) {
  const amt = Number(v)
  if (!isFinite(amt)) return
  acceptedProposals.value = {
    ...acceptedProposals.value,
    [pl.account]: { ...acceptedProposals.value[pl.account], amount: amt },
  }
  runCheck()
}

const coaOpen = ref(false)
const coaLoading = ref(false)
const coa = ref({ sections: [], account_count: 0 })

async function toggleCoa() {
  coaOpen.value = !coaOpen.value
  if (!coaOpen.value || coa.value.sections.length) return
  coaLoading.value = true
  try {
    const res = await api.get('/api/valuations/chart-of-accounts',
      { params: parsed.value?.vcode ? { vcode: parsed.value.vcode } : {} })
    coa.value = res.data
  } catch { coaOpen.value = false }
  finally { coaLoading.value = false }
}

const visibleLines = computed(() => {
  const all = parsed.value?.lines || []
  return onlyNumbered.value ? all.filter(l => l.stated_account) : all
})
const unnumberedCount = computed(() =>
  (parsed.value?.lines || []).filter(l => !l.stated_account).length)

// Every account in the chart, each carrying the category that owns it, sorted by
// number — which is the order the accountants think in and the order Jack reads them
// off his spreadsheet.
const allAccounts = computed(() => {
  const out = []
  for (const c of cats.value.categories || []) {
    for (const a of c.accounts || []) {
      out.push({ ...a, category: c.category })
    }
  }
  out.sort((x, y) => String(x.account).localeCompare(String(y.account)))
  return out
})

// The accounts this deal has actually used, offered first. Same list, ranked — not a
// different set, so nothing is reachable in one group and missing from the other.
const dealAccounts = computed(() =>
  allAccounts.value.filter(a => a.used || a.used_by_deal))
const reconRows = computed(() => check.value?.reconciliation?.rows || [])
// The occupancy read off the sheet, averaged per quarter as the chart will show it.
const occQuarters = computed(() => {
  const by = parsed.value?.occupancy?.by_period || {}
  const b = {}
  for (const [p, v] of Object.entries(by)) {
    const q = `${p.slice(0, 4)} Q${Math.floor((Number(p.slice(5, 7)) - 1) / 3) + 1}`
    ;(b[q] = b[q] || []).push(Number(v))
  }
  return Object.keys(b).sort().map(q => ({ q, v: b[q].reduce((a, x) => a + x, 0) / b[q].length }))
})
const outsideNoi = computed(() => check.value?.reconciliation?.outside_noi || [])
const criticalWarnings = computed(() => (check.value?.warnings || []).filter(w => w.critical))
const infoWarnings = computed(() => (check.value?.warnings || []).filter(w => !w.critical))
const showInfo = ref(false)
const canCommit = computed(() =>
  props.editable && !committing.value && !!check.value?.can_import)

function m(row) { return mapping.value[String(row)] || {} }

/** Where this line lands, read off the account. The server derives it the same way
 *  from the same map and ignores whatever the screen sends, so the two cannot drift. */
function categoryOf(row) {
  const acct = m(row).account
  if (!acct) return null
  return allAccounts.value.find(a => String(a.account) === String(acct))?.category || null
}

function setAccount(row, acct) {
  const key = String(row)
  const cur = m(row)
  if (!acct) {
    delete mapping.value[key]
    mapping.value = { ...mapping.value }
    return void runCheck()
  }
  // The account brings its own category, ALWAYS — never `cur.category || ...`, which
  // would let a category set earlier survive a change of account and put the line on a
  // row its account does not belong to.
  const owning = allAccounts.value.find(a => String(a.account) === String(acct))
  const category = owning?.category || null
  mapping.value = {
    ...mapping.value,
    [key]: { ...cur, category, account: acct, flip: defaultFlip(row, category, acct) },
  }
  runCheck()
}

function setFlip(row, on) {
  const key = String(row)
  mapping.value = { ...mapping.value, [key]: { ...m(row), flip: on } }
  runCheck()
}

/**
 * The flip comes from how the account ACTUALLY behaved for this deal, never from the
 * 4xxx/5xxx prefix: 4030 Residential Vacancy and 4042 Loss to Lease are 4xxx accounts
 * stored POSITIVE, and 5220 Other (Income) Expense is 5xxx stored NEGATIVE, so a prefix
 * rule gets all three backwards and silently inverts NOI.
 */
function defaultFlip(row, category, account) {
  const line = (parsed.value?.lines || []).find(l => String(l.row) === String(row))
  const a = allAccounts.value.find(x => String(x.account) === String(account))
  if (!line || !a || !line.total) return false
  return (line.total > 0) !== (a.mri_sign > 0)
}

function imported(line) {
  return (line.total || 0) * (m(line.row).flip ? -1 : 1)
}

function toggle(i) { open.value = { ...open.value, [i]: !open.value[i] } }

async function onFile(e) {
  const file = e.target.files?.[0]
  if (!file) return
  parsing.value = true
  error.value = ''
  result.value = null
  check.value = null
  try {
    const form = new FormData()
    form.append('file', file)
    form.append('source', props.source)
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/parse`, form)
    parsed.value = res.data
    cats.value = { categories: res.data.categories, has_history: true }
    // Argus pre-fills; a budget starts empty. Either way the analyst sees it before
    // anything is written — which is the whole point of this screen existing.
    mapping.value = { ...(res.data.suggested || {}) }
    uploadedName.value = file.name || ''
    draftLoaded.value = null
    await loadCategories()
    await runCheck()
    // Store it immediately: an upload followed by a refresh should not mean
    // hunting down the spreadsheet again.
    await saveDraft()
  } catch (err) {
    error.value = err.response?.data?.error || 'Could not read that file.'
    parsed.value = null
  } finally {
    parsing.value = false
    e.target.value = ''
  }
}

async function loadCategories() {
  try {
    const res = await api.get(
      `/api/valuations/records/${props.recordId}/mapping/categories`)
    cats.value = res.data
  } catch { /* parse already returned the list; the ranking is the only thing lost */ }
}

// Saved as the analyst works, not on a button: asking someone to remember to save
// twenty minutes of judgement is asking them to lose it once. Debounced so a burst of
// edits is one write, and sequenced so a slow reply cannot report a stale state.
let draftTimer = null
let draftSeq = 0
function scheduleDraftSave() {
  if (!props.editable || !parsed.value) return
  clearTimeout(draftTimer)
  draftState.value = 'saving'
  draftTimer = setTimeout(saveDraft, 700)
}

async function saveDraft() {
  if (!parsed.value) return
  const seq = ++draftSeq
  try {
    await api.put(`/api/valuations/records/${props.recordId}/mapping/draft`, {
      source: props.source,
      filename: parsed.value.filename || uploadedName.value || '',
      parsed: parsed.value,
      mapping: mapping.value,
    })
    if (seq === draftSeq) {
      draftState.value = 'saved'
      draftSavedAt.value = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    }
  } catch {
    // Never block the analyst on this; say it plainly and keep their work on screen.
    if (seq === draftSeq) draftState.value = 'error'
  }
}

async function loadDraft() {
  loadingDraft.value = true
  draftLoaded.value = null
  try {
    const res = await api.get(
      `/api/valuations/records/${props.recordId}/mapping/draft`,
      { params: { source: props.source } })
    const d = res.data
    if (d && d.parsed && (d.parsed.lines || []).length) {
      parsed.value = d.parsed
      mapping.value = { ...(d.mapping || {}) }
      draftLoaded.value = d
      uploadedName.value = d.filename || ''
      await loadCategories()
      await runCheck()
    }
  } catch { /* nothing stored, or unreadable — the screen simply starts empty */ }
  finally { loadingDraft.value = false }
}

async function discardDraft() {
  try {
    await api.delete(`/api/valuations/records/${props.recordId}/mapping/draft`,
      { params: { source: props.source } })
  } catch { /* a draft we cannot delete is not worth blocking on */ }
  parsed.value = null; mapping.value = {}; check.value = null
  result.value = null; draftLoaded.value = null; uploadedName.value = ''
  draftState.value = ''; draftSavedAt.value = ''
}

let checkSeq = 0
async function runCheck() {
  if (!parsed.value) return
  scheduleDraftSave()
  const seq = ++checkSeq
  checking.value = true
  try {
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/check`,
      { source: props.source, parsed: parsed.value, mapping: mapping.value })
    // Only the newest reply wins — the analyst edits faster than the round trip and a
    // late response would otherwise overwrite the current state with a stale verdict.
    if (seq === checkSeq) { check.value = res.data; open.value = {} }
  } catch (err) {
    if (seq === checkSeq) error.value = err.response?.data?.error || 'Check failed.'
  } finally {
    if (seq === checkSeq) checking.value = false
  }
}

async function doCommit() {
  committing.value = true
  error.value = ''
  try {
    const res = await api.post(
      `/api/valuations/records/${props.recordId}/mapping/commit`,
      { source: props.source, parsed: parsed.value, mapping: mapping.value })
    result.value = res.data
    draftState.value = 'saved'
    if (draftLoaded.value) draftLoaded.value.status = 'committed'
    emit('committed', res.data)
  } catch (err) {
    error.value = err.response?.data?.error || 'Import failed.'
  } finally {
    committing.value = false
  }
}

function reconLabel(k) {
  return { revenue: 'Total revenue', expense: 'Total expenses', noi: 'NOI' }[k] || k
}

function fmtCurrency(v) {
  if (v === null || v === undefined || isNaN(v)) return '—'
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: 0 })
}

function fmtPeriod(p) {
  if (!p) return ''
  const d = new Date(p + 'T00:00:00')
  return isNaN(d) ? p : d.toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

// Switching records must not leave the previous deal's file on screen -- and the new
// record's stored mapping should come up in its place, not a blank page.
watch(() => [props.recordId, props.source], () => {
  parsed.value = null; mapping.value = {}; check.value = null
  result.value = null; error.value = ''
  draftLoaded.value = null; draftState.value = ''; draftSavedAt.value = ''
  uploadedName.value = ''
  if (props.recordId) loadDraft()
})

onMounted(() => { if (props.recordId) loadDraft() })
</script>

<style scoped>
.line-mapping { display: flex; flex-direction: column; gap: 14px; }
.lm-coa-toggle {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  font-size: 0.8rem; padding: 6px 0;
}
.lm-coa-toggle label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.lm-note-inline { color: #777; font-size: 0.76rem; }
.lm-derived { color: #33475b; }
.lm-coa-btn { font-size: 0.78rem; padding: 3px 10px; }
.lm-proposed {
  border: 1px solid #e0a800; background: #fff8e5; border-radius: 4px;
  padding: 8px 12px; margin-bottom: 10px; font-size: 0.82rem;
}
.lm-proposed-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 4px; }
.lm-proposed-row label { display: flex; align-items: center; gap: 6px; cursor: pointer; }
.lm-proposed-amt { width: 110px; padding: 2px 6px; font-size: 0.8rem; }
.lm-coa {
  border: 1px solid #d7e3ee; background: #f7fbff; border-radius: 4px;
  padding: 10px 14px; margin-bottom: 10px;
}
.lm-coa-sec { margin-bottom: 10px; }
.lm-coa-sec h5 {
  margin: 0 0 4px; font-size: 0.82rem; color: #1F4E79;
  border-bottom: 1px solid #d7e3ee; padding-bottom: 2px;
}
.lm-coa-cat { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px; padding: 2px 0; }
.lm-coa-catname { font-size: 0.78rem; min-width: 210px; color: #333; }
.lm-coa-acct {
  font-size: 0.72rem; font-family: ui-monospace, Menlo, Consolas, monospace;
  background: #fff; border: 1px solid #dde5ee; border-radius: 3px; padding: 0 5px;
  color: #666;
}
.lm-coa-used { background: #e6f4ea; border-color: #9ed3b0; color: #1e7a3c; font-weight: 600; }
.lm-coa-sub {
  font-size: 0.8rem; font-weight: 600; color: #1F4E79;
  border-top: 2px solid #1F4E79; padding-top: 3px; margin-top: 4px;
}
.lm-stated { background: #e6f4ea; color: #1e7a3c; }
.lm-prior { background: #eef5fc; color: #14507a; }
.lm-prior-note { font-size: 0.72rem; color: #777; margin-left: 6px; font-style: italic; }
.lm-resumed {
  display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
  font-size: 0.82rem; color: #14507a; background: #eef5fc;
  border-left: 3px solid #1a73e8; padding: 8px 12px; border-radius: 3px;
}
.lm-discard {
  margin-left: auto; border: 1px solid #c3ccd9; background: #fff; color: #444;
  padding: 2px 10px; border-radius: 3px; font-size: 0.76rem; cursor: pointer;
}
.lm-discard:hover:not(:disabled) { background: #fff1f0; border-color: #c0392b; color: #8a1c14; }
.lm-draft-state { font-size: 0.78rem; margin-left: 10px; }
.lm-draft-saving { color: #888; }
.lm-draft-saved { color: #1e7a3c; }
.lm-draft-error { color: #c0392b; font-weight: 600; }
.lm-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
.lm-head h4 { margin: 0 0 4px; }
.lm-note { font-size: 12px; color: #666; margin: 4px 0; line-height: 1.5; }
.lm-file { display: flex; align-items: center; gap: 8px; white-space: nowrap; }
.lm-filename { font-size: 12px; color: #444; }
.lm-error { background: #fdecea; border-left: 3px solid #c0392b; padding: 8px 10px;
  font-size: 13px; color: #922; }

.lm-table select { width: 100%; max-width: 260px; font-size: 12px; padding: 3px 4px; }
.lm-table td { vertical-align: middle; }
.lm-subtotal { background: #fafafa; color: #888; font-style: italic; }
.lm-mapped { background: #f6fbf7; font-style: normal; color: inherit; }
.lm-tag { font-size: 10px; text-transform: uppercase; letter-spacing: .04em;
  background: #eee; color: #666; padding: 1px 5px; border-radius: 3px; margin-left: 6px; }
.lm-guess { background: #fff4d6; color: #8a6100; }
.lm-partial { background: #e8f0fe; color: #2c5aa0; }
.ctr { text-align: center; }

.lm-split { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 1100px) { .lm-split { grid-template-columns: 1fr; } }
.lm-recon h5, .lm-checks h5 { margin: 0 0 8px; display: flex; justify-content: space-between;
  align-items: baseline; }
.lm-count { font-size: 11px; font-weight: 400; color: #777; }

.lm-block { background: #fdecea; border-left: 3px solid #c0392b; padding: 7px 10px;
  font-size: 12px; margin-bottom: 6px; line-height: 1.45; }
.lm-warn { background: #fff8e6; border-left: 3px solid #e0a800; padding: 7px 10px;
  font-size: 12px; margin-bottom: 6px; line-height: 1.45; }
.lm-clean { font-size: 12px; color: #2e7d32; }
.lm-info { margin-top: 6px; }
.lm-note-item { font-size: 11px; color: #555; padding: 4px 10px; border-left: 2px solid #ddd;
  margin-bottom: 4px; line-height: 1.4; }
.lm-more { background: none; border: none; color: #2c5aa0; font-size: 11px;
  cursor: pointer; padding: 0 0 0 4px; text-decoration: underline; }
.lm-acct-list { margin: 6px 0 0; padding-left: 18px; font-size: 11px; color: #555; }
.lm-acct-list li { margin: 2px 0; }

.lm-blocked-note { font-size: 12px; color: #c0392b; margin-left: 10px; }
.lm-warn-note { font-size: 12px; color: #8a6100; margin-left: 10px; }
.lm-result { background: #f0f7f1; border-left: 3px solid #2e7d32; padding: 9px 12px;
  font-size: 13px; }
.neg { color: #c0392b; }
</style>
