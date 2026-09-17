<script setup lang="ts">
/**
 * Treasury management — the bank side of the close.
 *
 * THREE TABS, BECAUSE IT IS THREE JOBS. Where the cash is (accounts), getting
 * the bank's own record into the app (import), and proving it against the
 * ledger (reconciliation). Jim, Sep 17 2026: "On the initial tab of the
 * section, I would like to see the list of accounts, current ledger and
 * current available."
 *
 * CURRENT AVAILABLE IS BLANK AND SAYS WHY. Available is ledger less holds,
 * float and pending debits — facts that live at the bank and appear nowhere in
 * an activity export. The column is here because it is the number a treasurer
 * acts on, and it stays empty until the PNC connection can fill it. Printing
 * the ledger figure under an "available" heading would be inventing it.
 *
 * NOTHING HERE POSTS TO MRI. The reconciliation produces the accountant's
 * working papers — what matched, what is in transit, what is outstanding. The
 * GL and IA upload templates are the next phase and are built from this, not
 * instead of it.
 */
import { ref, computed, onMounted } from 'vue'
import api from '../api/client'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
// The accounting section is the CFO's (Sep 16 2026). Mapping an account to an
// entity and closing a period change the arrangement; importing does not.
const canManage = computed(
  () => ['admin', 'cfo'].includes(auth.user?.role || ''))

const tab = ref<'accounts' | 'import' | 'reconcile'>('accounts')
const msg = ref('')
const error = ref('')
const busy = ref(false)

const accounts = ref<any[]>([])
const cashAccounts = ref<string[]>([])
const defaultCash = ref('')
const loadingAccounts = ref(false)

const month = ref(new Date().toISOString().slice(0, 7))
const period = computed(() => month.value.replace('-', ''))
const selected = ref('')

const tie = ref<any>(null)
const matchRes = ref<any>(null)
const glNet = ref('')
const pickBank = ref<number | null>(null)
const pickGl = ref<string>('')

const activityFile = ref<File | null>(null)
const statementFile = ref<File | null>(null)
const lastImport = ref<any>(null)
const lastStatement = ref<any>(null)

function flash(m: string) {
  msg.value = m
  setTimeout(() => (msg.value = ''), 4500)
}

function fail(e: any, what: string) {
  error.value = e?.response?.data?.error || e?.message || `${what} failed.`
  setTimeout(() => (error.value = ''), 9000)
}

// A missing figure prints as an em dash, never as 0.00 — a real zero balance
// and an unknown one are not the same fact.
function money(v: any): string {
  if (v === null || v === undefined || v === '') return '—'
  const n = Number(v)
  if (!isFinite(n)) return '—'
  return n.toLocaleString('en-US', { minimumFractionDigits: 2,
                                     maximumFractionDigits: 2 })
}

async function loadAccounts() {
  loadingAccounts.value = true
  try {
    const { data } = await api.get('/api/treasury/accounts')
    accounts.value = data.accounts || []
    cashAccounts.value = data.cash_accounts || []
    defaultCash.value = data.default_cash_account || ''
    if (!selected.value && accounts.value.length)
      selected.value = accounts.value[0].account_number
  } catch (e) { fail(e, 'Loading accounts') } finally {
    loadingAccounts.value = false
  }
}

async function saveAccount(a: any) {
  try {
    await api.put(`/api/treasury/accounts/${encodeURIComponent(a.account_number)}`,
      { entityid: a.entityid, gl_cash_account: a.gl_cash_account })
    flash(`Saved ${a.account_number}.`)
    await loadAccounts()
  } catch (e) { fail(e, 'Saving the account') }
}

async function uploadActivity() {
  if (!activityFile.value) return
  busy.value = true
  try {
    const fd = new FormData()
    fd.append('file', activityFile.value)
    const { data } = await api.post('/api/treasury/import/activity', fd)
    lastImport.value = data
    flash(`${data.inserted} transactions imported, ${data.already_held} already held.`)
    await loadAccounts()
  } catch (e) { fail(e, 'Importing the activity export') } finally {
    busy.value = false
  }
}

async function uploadStatement() {
  if (!statementFile.value || !selected.value) return
  busy.value = true
  try {
    const fd = new FormData()
    fd.append('file', statementFile.value)
    fd.append('account_number', selected.value)
    const { data } = await api.post('/api/treasury/import/statement', fd)
    lastStatement.value = data
    flash('Statement balances filed.')
  } catch (e: any) {
    // A refused statement still shows what WAS read: a scanned PDF is a
    // different problem from one whose figures disagree.
    lastStatement.value = e?.response?.data || null
    fail(e, 'Importing the statement')
  } finally { busy.value = false }
}

async function runReconcile() {
  if (!selected.value) return
  busy.value = true
  tie.value = null
  matchRes.value = null
  try {
    const q: any = { account_number: selected.value, period: period.value }
    if (glNet.value !== '') q.gl_net = glNet.value
    const { data } = await api.get('/api/treasury/reconcile', { params: q })
    tie.value = data
    const m = await api.get('/api/treasury/match', {
      params: { account_number: selected.value, period: period.value } })
    matchRes.value = m.data
  } catch (e) { fail(e, 'Reconciling') } finally { busy.value = false }
}

async function pair() {
  if (pickBank.value === null || !pickGl.value) return
  try {
    await api.put('/api/treasury/match', {
      account_number: selected.value, period: period.value,
      bank_id: pickBank.value, gl_item: pickGl.value })
    pickBank.value = null
    pickGl.value = ''
    await runReconcile()
  } catch (e) { fail(e, 'Pairing') }
}

async function unpair(bankId: number) {
  try {
    await api.put('/api/treasury/match', {
      account_number: selected.value, period: period.value,
      bank_id: bankId, gl_item: null })
    await runReconcile()
  } catch (e) { fail(e, 'Removing the pairing') }
}

async function closePeriod() {
  try {
    const { data } = await api.post('/api/treasury/close', {
      account_number: selected.value, period: period.value })
    if (data.error) { error.value = data.error; return }
    flash(`${period.value} closed at ${money(data.computed_ending)}.`)
    await loadAccounts()
    await runReconcile()
  } catch (e) { fail(e, 'Closing the period') }
}

const seedAmount = ref('')
async function seed() {
  try {
    const { data } = await api.post('/api/treasury/seed-opening', {
      account_number: selected.value, period: period.value,
      amount: Number(seedAmount.value) })
    if (data.error) { error.value = data.error; return }
    flash(`${period.value} opens at ${money(seedAmount.value)}.`)
    seedAmount.value = ''
    await runReconcile()
  } catch (e) { fail(e, 'Seeding the opening balance') }
}

onMounted(loadAccounts)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>Treasury</h1>
        <p class="subtitle">
          Bank accounts, imported activity, and the monthly reconciliation
        </p>
      </div>
      <div class="head-actions">
        <select v-model="selected" class="sel">
          <option v-for="a in accounts" :key="a.account_number"
                  :value="a.account_number">
            {{ a.account_number }}<span v-if="a.entityid"> — {{ a.entityid }}</span>
          </option>
        </select>
        <input type="month" v-model="month" class="sel" />
      </div>
    </div>

    <nav class="tabs" role="tablist">
      <button class="tab" :class="{ active: tab === 'accounts' }" role="tab"
              :aria-selected="tab === 'accounts'" @click="tab = 'accounts'">
        Accounts
      </button>
      <button class="tab" :class="{ active: tab === 'import' }" role="tab"
              :aria-selected="tab === 'import'" @click="tab = 'import'">
        Import
      </button>
      <button class="tab" :class="{ active: tab === 'reconcile' }" role="tab"
              :aria-selected="tab === 'reconcile'" @click="tab = 'reconcile'">
        Reconciliation
        <span class="tab-sub">{{ period }}</span>
      </button>
    </nav>

    <div v-if="msg" class="banner ok">{{ msg }}</div>
    <div v-if="error" class="banner err">{{ error }}</div>

    <!-- ── Tab 1: accounts ─────────────────────────────── -->
    <div v-show="tab === 'accounts'" class="tab-body">
      <div v-if="loadingAccounts" class="placeholder">Loading accounts…</div>
      <div v-else-if="!accounts.length" class="placeholder">
        No bank accounts yet. An account registers itself the first time its
        activity is imported — start on the Import tab.
      </div>

      <template v-else>
        <div class="grid-wrap">
          <table class="grid">
            <thead>
              <tr>
                <th class="l">Account</th>
                <th class="l">Name</th>
                <th class="l">Entity</th>
                <th class="l">GL cash account</th>
                <th class="r">Current ledger</th>
                <th class="r">Current available</th>
                <th class="l">Last closed</th>
                <th v-if="canManage"></th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="a in accounts" :key="a.account_number">
                <td class="l mono">{{ a.account_number }}</td>
                <td class="l">{{ a.account_name || '—' }}</td>
                <td class="l">
                  <input v-if="canManage" v-model="a.entityid" class="mini"
                         placeholder="ENTITYID" />
                  <span v-else>{{ a.entityid || '—' }}</span>
                </td>
                <td class="l">
                  <select v-if="canManage" v-model="a.gl_cash_account" class="mini">
                    <option v-for="c in cashAccounts" :key="c" :value="c">
                      {{ c }}<span v-if="c === defaultCash"> (default)</span>
                    </option>
                  </select>
                  <span v-else class="mono">{{ a.gl_cash_account }}</span>
                </td>
                <td class="r num" :class="{ unknown: a.current_ledger === null }">
                  <span :title="a.ledger_reason">{{ money(a.current_ledger) }}</span>
                  <div class="why">{{ a.ledger_reason }}</div>
                </td>
                <td class="r num unknown">
                  <span :title="a.available_reason">—</span>
                </td>
                <td class="l">
                  <template v-if="a.last_period">
                    {{ a.last_period }} at {{ money(a.last_balance) }}
                  </template>
                  <span v-else class="why">never closed</span>
                  <div v-if="a.unclosed_months && a.unclosed_months.length"
                       class="why">
                    activity in {{ a.unclosed_months.join(', ') }} not yet closed
                  </div>
                </td>
                <td v-if="canManage">
                  <button class="btn xs" @click="saveAccount(a)">Save</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <p class="footnote">
          <b>Current ledger</b> is carried forward from the last closed period
          plus every transaction imported since — a position computed from what
          the app holds, not a reading of PNC's ledger balance.
          <b>Current available</b> is blank on purpose: it is the ledger less
          holds, float and pending debits, which exist only at the bank and
          appear nowhere in an activity export. It fills in when the PNC
          connection does.
        </p>
      </template>
    </div>

    <!-- ── Tab 2: import ───────────────────────────────── -->
    <div v-show="tab === 'import'" class="tab-body">
      <div class="cards">
        <section class="card">
          <h3>Activity export (CSV)</h3>
          <p class="hint">
            The transaction export from PINACLE. Re-importing the same file adds
            nothing — each row is identified by its own contents, so a re-run
            after a correction is safe.
          </p>
          <input type="file" accept=".csv"
                 @change="activityFile = ($event.target as HTMLInputElement).files?.[0] || null" />
          <button class="btn primary" :disabled="!activityFile || busy"
                  @click="uploadActivity">Import activity</button>

          <div v-if="lastImport" class="result">
            <div>
              <b>{{ lastImport.inserted }}</b> imported,
              <b>{{ lastImport.already_held }}</b> already held<span
                v-if="lastImport.skipped_count">,
              <b>{{ lastImport.skipped_count }}</b> could not be read</span>.
            </div>
            <div v-if="lastImport.accounts && lastImport.accounts.length" class="why">
              Accounts touched: {{ lastImport.accounts.join(', ') }}
            </div>
            <!-- Shown, not logged: a transaction dropped in silence makes a
                 period tie for the wrong reason. -->
            <ul v-if="lastImport.skipped_count" class="skipped">
              <li v-for="(s, i) in lastImport.skipped" :key="i">{{ s }}</li>
            </ul>
          </div>
        </section>

        <section class="card">
          <h3>Statement (PDF)</h3>
          <p class="hint">
            The official monthly statement, for its beginning and ending
            balances. Filed against <b class="mono">{{ selected || '—' }}</b> —
            change the account in the header above.
          </p>
          <input type="file" accept=".pdf"
                 @change="statementFile = ($event.target as HTMLInputElement).files?.[0] || null" />
          <button class="btn primary" :disabled="!statementFile || !selected || busy"
                  @click="uploadStatement">Import statement</button>

          <div v-if="lastStatement" class="result">
            <div v-if="lastStatement.error" class="why err-text">
              {{ lastStatement.error }}
            </div>
            <table v-if="lastStatement.parsed" class="mini-table">
              <tr><td>Period</td>
                  <td class="r">{{ lastStatement.parsed.period_start || '—' }}
                      → {{ lastStatement.parsed.period_end || '—' }}</td></tr>
              <tr><td>Beginning balance</td>
                  <td class="r num">{{ money(lastStatement.parsed.beginning_balance) }}</td></tr>
              <tr><td>Ending balance</td>
                  <td class="r num">{{ money(lastStatement.parsed.ending_balance) }}</td></tr>
            </table>
          </div>
        </section>
      </div>
    </div>

    <!-- ── Tab 3: reconciliation ───────────────────────── -->
    <div v-show="tab === 'reconcile'" class="tab-body">
      <div class="recon-bar">
        <span class="mono">{{ selected || 'no account selected' }}</span>
        <span class="sep">·</span>
        <span>{{ period }}</span>
        <label class="inline">
          GL net movement
          <input v-model="glNet" class="mini" placeholder="optional" />
        </label>
        <button class="btn primary" :disabled="!selected || busy"
                @click="runReconcile">Reconcile</button>
        <span class="spacer"></span>
        <button v-if="canManage && tie && tie.computed_ending !== null"
                class="btn" @click="closePeriod">Close {{ period }}</button>
      </div>

      <div v-if="!tie" class="placeholder">
        Pick an account and a month, then reconcile.
      </div>

      <template v-else>
        <div class="headline" :class="{ ok: tie.ties_to_statement === true,
                                        warn: tie.ties_to_statement === false }">
          {{ tie.headline }}
        </div>

        <!-- The three-way tie, each leg on its own line. Which leg disagrees
             is the only thing the difference is for. -->
        <table class="mini-table tie">
          <tr>
            <td>Opening balance</td>
            <td class="r num">{{ money(tie.opening_balance) }}</td>
            <td class="why">{{ tie.opening_source }}</td>
          </tr>
          <tr>
            <td>Net movement, imported activity</td>
            <td class="r num">{{ money(tie.bank_movement) }}</td>
            <td class="why">{{ tie.transaction_count }} transactions</td>
          </tr>
          <tr class="total">
            <td>Computed ending</td>
            <td class="r num">{{ money(tie.computed_ending) }}</td>
            <td></td>
          </tr>
          <tr>
            <td>Statement ending</td>
            <td class="r num">{{ money(tie.statement_ending) }}</td>
            <td class="why">
              <span v-if="tie.ties_to_statement === null">no statement filed</span>
              <span v-else-if="tie.ties_to_statement">ties</span>
              <span v-else class="err-text">differs by
                {{ money(tie.statement_difference) }}</span>
            </td>
          </tr>
          <tr v-if="tie.gl_net !== null && tie.gl_net !== undefined">
            <td>Ledger net movement</td>
            <td class="r num">{{ money(tie.gl_net) }}</td>
            <td class="why">
              <span v-if="tie.ties_to_gl">ties</span>
              <span v-else class="err-text">differs by
                {{ money(tie.gl_difference) }}</span>
            </td>
          </tr>
        </table>

        <div v-if="tie.opening_balance === null && canManage"
             class="seed">
          <span class="hint">
            No prior period has been closed, so there is nothing to carry
            forward. Enter the opening balance for this month to start the
            chain.
          </span>
          <input v-model="seedAmount" class="mini" placeholder="0.00" />
          <button class="btn" @click="seed">Seed opening</button>
        </div>

        <!-- ── the matcher ── -->
        <template v-if="matchRes">
          <!-- A refusal does not hide the bank side. Knowing what came through
               the account is what makes "map this account first" actionable. -->
          <div v-if="matchRes.error" class="banner err">{{ matchRes.error }}</div>

          <template v-if="!matchRes.error || matchRes.bank_only.length">
            <p v-if="!matchRes.error" class="headline sub">{{ matchRes.headline }}</p>

            <div class="two-col">
              <section class="card">
                <h3>Deposits in transit &amp; outstanding payments
                  <span class="count">{{ matchRes.gl_only.length }}</span></h3>
                <p class="hint">
                  On the ledger, not yet on the bank. These are the reconciling
                  items.
                </p>
                <table class="mini-table pick">
                  <tr v-for="g in matchRes.gl_only" :key="g.gl_item"
                      :class="{ picked: pickGl === g.gl_item }"
                      @click="pickGl = (pickGl === g.gl_item ? '' : g.gl_item)">
                    <td class="l">{{ g.date }}</td>
                    <td class="l ellip">{{ g.description }}</td>
                    <td class="l kind">{{ g.kind }}</td>
                    <td class="r num">{{ money(g.amount) }}</td>
                  </tr>
                  <tr v-if="!matchRes.gl_only.length">
                    <td colspan="4" class="why">Nothing outstanding.</td>
                  </tr>
                </table>
              </section>

              <section class="card">
                <h3>Not recorded on the ledger
                  <span class="count">{{ matchRes.bank_only.length }}</span></h3>
                <p class="hint">
                  On the bank, not yet on the ledger — each needs an entry or a
                  pairing.
                </p>
                <table class="mini-table pick">
                  <tr v-for="b in matchRes.bank_only" :key="b.bank_id"
                      :class="{ picked: pickBank === b.bank_id }"
                      @click="pickBank = (pickBank === b.bank_id ? null : b.bank_id)">
                    <td class="l">{{ b.date }}</td>
                    <td class="l ellip">{{ b.description }}</td>
                    <td class="l kind">{{ b.transaction_type }}</td>
                    <td class="r num">{{ money(b.signed_amount) }}</td>
                  </tr>
                  <tr v-if="!matchRes.bank_only.length">
                    <td colspan="4" class="why">Everything is on the ledger.</td>
                  </tr>
                </table>
              </section>
            </div>

            <div v-if="!matchRes.error" class="pair-bar">
              <span class="hint">
                Pick one from each side to pair them by hand. A pairing you make
                is honoured before the matcher's own and is never re-decided.
              </span>
              <button class="btn" :disabled="pickBank === null || !pickGl"
                      @click="pair">Pair selected</button>
            </div>

            <details v-if="!matchRes.error" class="matched">
              <summary>
                {{ matchRes.matched.length }} matched
                <span v-if="matchRes.ambiguous" class="why">
                  — {{ matchRes.ambiguous }} had more than one candidate at the
                  same amount and were paired on the nearest date</span>
              </summary>
              <table class="mini-table">
                <thead>
                  <tr><th class="l">Bank</th><th class="l">Ledger</th>
                      <th class="r">Amount</th><th></th></tr>
                </thead>
                <tbody>
                  <tr v-for="m in matchRes.matched" :key="m.bank_id"
                      :class="{ far: m.far_apart }">
                    <td class="l">{{ m.bank_date }} · <span class="ellip">{{ m.bank_description }}</span></td>
                    <td class="l">{{ m.gl_date }} · <span class="ellip">{{ m.gl_description }}</span></td>
                    <td class="r num">{{ money(m.bank_amount) }}</td>
                    <td class="r">
                      <span v-if="m.manual" class="chip">by hand</span>
                      <span v-if="m.far_apart" class="chip warn"
                            :title="`${m.days_apart} days apart`">{{ m.days_apart }}d</span>
                      <button v-if="m.manual" class="btn xs"
                              @click="unpair(m.bank_id)">Unpair</button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </details>
          </template>
        </template>
      </template>
    </div>
  </div>
</template>

<style scoped>
.page { padding: 18px 22px; min-width: 0; max-width: 100%; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
h1 { margin: 0; font-size: 22px; }
h3 { margin: 0 0 6px; font-size: 14px; }
.subtitle { margin: 2px 0 0; color: var(--color-text-secondary); font-size: 13px; }
.head-actions { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.sel, input[type=month], input[type=file] {
  padding: 5px 8px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); font-size: 13px; }
.btn { padding: 5px 12px; border: 1px solid var(--color-border); border-radius: 4px;
  background: var(--color-surface); color: var(--color-text); cursor: pointer; font-size: 13px; }
.btn.primary { background: #1f3864; color: #fff; border-color: #1f3864; }
.btn.xs { padding: 2px 7px; font-size: 11px; }
.btn:disabled { opacity: .5; cursor: default; }
.hint { color: var(--color-text-secondary); font-size: 12px; margin: 0 0 8px; }
.banner { margin: 10px 0; padding: 8px 12px; border-radius: 5px; font-size: 13px; }
.banner.ok { background: #eaf6ec; border: 1px solid #9ccfa6; color: #205c2c; }
.banner.err { background: #fdecea; border: 1px solid #e0a09a; color: #7a231b; }
.placeholder { margin: 24px 0; color: var(--color-text-secondary); }

.tabs { display: flex; gap: 2px; margin-top: 16px; border-bottom: 1px solid #dde3ec; }
.tab { background: none; border: none; cursor: pointer; padding: 9px 16px;
  font-size: 13px; font-weight: 600; color: #7a8394;
  border-bottom: 2px solid transparent; margin-bottom: -1px; }
.tab.active { color: #1d4e7e; border-bottom-color: #1d4e7e; }
.tab-sub { font-weight: 500; color: #9aa3b2; margin-left: 6px; font-size: 11px; }
.tab-body { margin-top: 14px; }

.grid-wrap { overflow-x: auto; border: 1px solid var(--color-border); border-radius: 6px; }
.grid { border-collapse: collapse; font-size: 12px; width: 100%; }
.grid th { background: #f4f6fa; text-align: right; padding: 7px 9px;
  border-bottom: 1px solid #dde3ec; font-weight: 600; white-space: nowrap; }
.grid td { padding: 6px 9px; border-bottom: 1px solid #eef1f6; vertical-align: top; }
.grid tbody tr:hover { background: #fafbfd; }
.l { text-align: left; }
.r { text-align: right; }
.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
.mono { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px; }
.unknown span { color: #9aa3b2; }
.why { color: #9aa3b2; font-size: 10.5px; font-weight: 400; margin-top: 2px;
  white-space: normal; max-width: 260px; }
.err-text { color: #a8362b; }
.mini { padding: 2px 6px; border: 1px solid var(--color-border); border-radius: 3px;
  font-size: 11.5px; background: var(--color-surface); color: var(--color-text);
  max-width: 150px; }
.footnote { margin-top: 12px; font-size: 11.5px; color: var(--color-text-secondary);
  max-width: 900px; line-height: 1.55; }

.cards { display: flex; gap: 16px; flex-wrap: wrap; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }
@media (max-width: 1000px) { .two-col { grid-template-columns: 1fr; } }
.card { flex: 1 1 380px; border: 1px solid #e2e6ee; border-radius: 8px;
  background: var(--color-surface); padding: 14px 16px; min-width: 0; }
.card input[type=file] { display: block; margin-bottom: 8px; max-width: 100%; }
.count { color: #9aa3b2; font-weight: 500; font-size: 12px; margin-left: 4px; }
.result { margin-top: 10px; font-size: 12px; }
.skipped { margin: 6px 0 0; padding-left: 18px; font-size: 11px; color: #8a5a00; }

.recon-bar { display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  border: 1px solid #e2e6ee; border-radius: 8px; background: var(--color-surface);
  padding: 10px 14px; font-size: 12px; }
.recon-bar .spacer { flex: 1 1 auto; }
.recon-bar .sep { color: #c3c9d4; }
.inline { display: flex; align-items: center; gap: 6px; color: var(--color-text-secondary); }

.headline { margin: 14px 0 8px; padding: 9px 13px; border-radius: 6px;
  background: #f4f6fa; border: 1px solid #e2e6ee; font-size: 12.5px; line-height: 1.6; }
.headline.ok { background: #eaf6ec; border-color: #9ccfa6; }
.headline.warn { background: #fdf4e6; border-color: #e6c68a; }
.headline.sub { background: none; border: none; padding: 4px 0; color: var(--color-text-secondary); }

.mini-table { border-collapse: collapse; font-size: 12px; width: 100%; }
.mini-table td, .mini-table th { padding: 4px 8px; border-bottom: 1px solid #eef1f6; }
.mini-table th { text-align: right; color: #7a8394; font-weight: 600; }
.mini-table.tie { max-width: 640px; margin-bottom: 12px; }
.mini-table.tie .total td { border-top: 1px solid #c3c9d4; font-weight: 600; }
.mini-table.pick tr { cursor: pointer; }
.mini-table.pick tr:hover { background: #fafbfd; }
.mini-table.pick tr.picked { background: #e8f0f9; }
.kind { color: #7a8394; font-size: 11px; }
.ellip { display: inline-block; max-width: 260px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; vertical-align: bottom; }

.seed { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  margin: 10px 0; padding: 10px 13px; border: 1px dashed #d6c08a;
  border-radius: 6px; background: #fdfaf2; }
.pair-bar { display: flex; align-items: center; gap: 12px; margin-top: 10px; flex-wrap: wrap; }
.matched { margin-top: 14px; font-size: 12px; }
.matched summary { cursor: pointer; color: #1d4e7e; font-weight: 600; }
.matched tr.far { background: #fdfaf2; }
.chip { display: inline-block; padding: 1px 6px; border-radius: 9px; font-size: 10px;
  background: #e8f0f9; color: #1d4e7e; margin-right: 5px; }
.chip.warn { background: #fdf4e6; color: #8a5a00; }
</style>
