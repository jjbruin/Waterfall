<script setup lang="ts">
/**
 * Accounting's allocation overrides (Oct 5 2026): one entity's split, for one
 * investment, from a date -- entered as the AMOUNTS each investor funded, and the
 * share is each amount over their total. MRI's commitments are entity-level, so
 * "PSC3, for JBFAIR only" cannot be said there. Everyone sees the overrides; only
 * accounting enters or removes them, and the server is what enforces that.
 */
import { ref, computed, onMounted } from 'vue'
import api from '@/api/client'

const props = defineProps<{ asOf?: string }>()
const emit = defineEmits<{ (e: 'changed'): void }>()

const items = ref<any[]>([])
const canEdit = ref(false)
const showRemoved = ref(false)
const error = ref('')
const notice = ref('')
const busy = ref(false)

const blank = () => ({ entity_id: '', investment_id: '', effective_date: '', reason: '',
  lines: [] as { investor_id: string; amount: string }[] })
const form = ref(blank())
const adding = ref(false)

const shown = computed(() => items.value.filter(o => showRemoved.value || !o.deleted_at))
const formTotal = computed(() => form.value.lines.reduce((t, l) => t + (Number(String(l.amount).replace(/,/g, '')) || 0), 0))
function linePct(l: { amount: string }) {
  const a = Number(String(l.amount).replace(/,/g, '')) || 0
  return formTotal.value ? (a / formTotal.value * 100).toFixed(4) + '%' : '—'
}
const money = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 2 })

function fail(e: any) { error.value = e?.response?.data?.error || e?.response?.data?.message || String(e) }

async function load() {
  const { data } = await api.get('/api/reports/pe-exposure/overrides')
  items.value = data.overrides; canEdit.value = data.can_edit
}

async function loadInvestors() {
  error.value = ''
  if (!form.value.entity_id.trim()) { error.value = 'Name the entity first'; return }
  try {
    const { data } = await api.get('/api/reports/pe-exposure/overrides/owners', {
      params: { entity: form.value.entity_id, as_of: form.value.effective_date || props.asOf } })
    form.value.lines = data.owners.map((o: any) => ({ investor_id: o.investor_id,
      amount: o.committed != null ? String(o.committed) : '' }))
    notice.value = `Started from ${data.entity}'s commitments in MRI on ${data.as_of} -- change the amounts to what each investor funded.`
  } catch (e) { fail(e) }
}

async function save() {
  error.value = ''; notice.value = ''; busy.value = true
  try {
    const { data } = await api.post('/api/reports/pe-exposure/overrides', form.value)
    notice.value = data.warnings?.length ? 'Saved. ' + data.warnings.join('; ') : 'Saved.'
    form.value = blank(); adding.value = false
    await load(); emit('changed')
  } catch (e) { fail(e) } finally { busy.value = false }
}

async function removeOne(o: any) {
  if (!confirm(`Remove the ${o.entity_id} override for ${o.investment_id} effective ${o.effective_date}? ` +
               'The record is kept; the report goes back to commitments for those dates.')) return
  error.value = ''
  try { await api.delete(`/api/reports/pe-exposure/overrides/${o.id}`); await load(); emit('changed') }
  catch (e) { fail(e) }
}

onMounted(() => load().catch(fail))
</script>

<template>
  <div class="ovr">
    <div class="ovr-head">
      <div>
        <div class="ovr-title">Allocation overrides</div>
        <div class="muted">Where accounting records how an entity's investors funded ONE investment, that split
          replaces the entity's commitment ratios for that investment, from the effective date on.</div>
      </div>
      <label class="muted"><input v-model="showRemoved" type="checkbox" /> show removed</label>
    </div>
    <div v-if="error" class="err">{{ error }}</div>
    <div v-if="notice" class="muted">{{ notice }}</div>

    <table v-if="shown.length" class="mini">
      <thead><tr><th>Entity</th><th>Investment</th><th>From</th><th>Investors (amount funded → share)</th>
        <th>Reason</th><th>Entered</th><th></th></tr></thead>
      <tbody>
        <tr v-for="o in shown" :key="o.id" :class="{ removed: o.deleted_at }">
          <td>{{ o.entity_id }}</td><td>{{ o.investment_id }}</td><td>{{ o.effective_date }}</td>
          <td><div v-for="l in o.lines" :key="l.investor_id">{{ l.investor_id }}: {{ money(l.amount) }} →
            {{ l.pct == null ? '—' : l.pct.toFixed(4) + '%' }}</div></td>
          <td class="wrap">{{ o.reason }}</td>
          <td class="muted">{{ o.entered_by }}<template v-if="o.deleted_at"><br />removed by {{ o.deleted_by }}</template></td>
          <td><button v-if="canEdit && !o.deleted_at" class="btn-link" @click="removeOne(o)">remove</button></td>
        </tr>
      </tbody>
    </table>
    <div v-else class="muted">None recorded.</div>

    <template v-if="canEdit">
      <button v-if="!adding" class="btn-secondary" @click="adding = true">+ Add an override</button>
      <div v-else class="form">
        <div class="row">
          <label>Entity <input v-model="form.entity_id" placeholder="e.g. PSC3" /></label>
          <label>Investment <input v-model="form.investment_id" placeholder="e.g. JBFAIR" /></label>
          <label>Effective from <input v-model="form.effective_date" type="date" /></label>
          <button class="btn-secondary" type="button" @click="loadInvestors">Start from MRI's investors</button>
        </div>
        <table class="mini">
          <thead><tr><th>Investor</th><th>Amount funded</th><th>Share</th><th></th></tr></thead>
          <tbody>
            <tr v-for="(l, i) in form.lines" :key="i">
              <td><input v-model="l.investor_id" /></td>
              <td><input v-model="l.amount" class="num-in" /></td>
              <td class="num">{{ linePct(l) }}</td>
              <td><button class="btn-link" type="button" @click="form.lines.splice(i, 1)">remove</button></td>
            </tr>
            <tr><td><button class="btn-link" type="button"
                            @click="form.lines.push({ investor_id: '', amount: '' })">+ investor</button></td>
              <td class="num"><strong>{{ money(formTotal) }}</strong></td><td class="num">100%</td><td></td></tr>
          </tbody>
        </table>
        <label class="wide">Reason <textarea v-model="form.reason" rows="2"
          placeholder="e.g. DCXVIA and DCXVIB opted out of JBFAIR; their initial commitment was returned" /></label>
        <div class="row">
          <button class="btn-primary" :disabled="busy" @click="save">Save override</button>
          <button class="btn-secondary" type="button" @click="adding = false; form = blank()">Cancel</button>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.ovr { margin-top: 18px; padding-top: 10px; border-top: 1px solid var(--color-border, #e5e7eb); font-size: 12.5px; }
.ovr-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.ovr-title { font-weight: 600; font-size: 13.5px; }
.muted { color: var(--color-text-secondary, #6b7280); }
.err { color: #b42318; margin: 6px 0; }
.mini { border-collapse: collapse; margin: 8px 0; width: 100%; }
.mini th, .mini td { border-bottom: 1px solid var(--color-border, #e5e7eb); padding: 4px 6px; text-align: left; vertical-align: top; }
.mini .num { text-align: right; }
.wrap { white-space: normal; max-width: 360px; }
tr.removed td { opacity: .5; text-decoration: line-through; }
.form { display: flex; flex-direction: column; gap: 8px; margin-top: 8px; padding: 10px;
  border: 1px dashed var(--color-border, #ccc); border-radius: 6px; }
.row { display: flex; gap: 10px; align-items: flex-end; flex-wrap: wrap; }
.row label, .wide { display: flex; flex-direction: column; gap: 2px; }
.wide textarea { width: 100%; font: inherit; }
.num-in { width: 140px; text-align: right; }
.btn-link { background: none; border: none; color: var(--color-primary, #1f4e79); cursor: pointer; padding: 0; }
.btn-primary, .btn-secondary { padding: 5px 12px; border-radius: 6px; font-size: 13px; cursor: pointer;
  border: 1px solid var(--color-primary, #1f4e79); align-self: flex-start; }
.btn-primary { background: var(--color-primary, #1f4e79); color: #fff; font-weight: 600; }
.btn-secondary { background: var(--color-surface, #fff); color: var(--color-primary, #1f4e79); }
</style>
