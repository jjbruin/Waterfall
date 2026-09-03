<script setup lang="ts">
import { ref, computed, onMounted, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '../api/client'

const route = useRoute()
const router = useRouter()

interface AbstractSection {
  section_key: string
  section_title: string
  content: string
  lease_ref: string
  sort_order: number
}

interface AbstractData {
  tenant_name: string
  suite: string
  property_name: string
  tenant_id: number
  review_id: number
  sections: AbstractSection[]
}

interface TenantEntry {
  tenant_id: number
  tenant_name: string
  suite: string
  is_vacant: boolean
  has_abstract: boolean
  section_count: number
}

const loading = ref(false)
const saving = ref(false)
const editing = ref(false)
const abstractData = ref<AbstractData | null>(null)
const tenantList = ref<TenantEntry[]>([])

// From query params
const reviewId = ref<number | null>(null)
const tenantId = ref<number | null>(null)

// Print timestamp
const printTimestamp = ref('')

function updatePrintTimestamp() {
  const d = new Date()
  printTimestamp.value = `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()} ${d.toLocaleTimeString()}`
}

async function loadTenantList() {
  if (!reviewId.value) return
  try {
    const res = await api.get(`/api/lease-review/reviews/${reviewId.value}/abstracts`)
    tenantList.value = res.data || []
  } catch (e: any) {
    console.error('Failed to load tenant list:', e)
  }
}

async function loadAbstract() {
  if (!reviewId.value || !tenantId.value) return
  loading.value = true
  editing.value = false
  try {
    const res = await api.get(
      `/api/lease-review/reviews/${reviewId.value}/tenants/${tenantId.value}/abstract`
    )
    abstractData.value = res.data
  } catch (e: any) {
    console.error('Failed to load abstract:', e)
    abstractData.value = null
  } finally {
    loading.value = false
  }
}

async function saveAbstract() {
  if (!abstractData.value || !tenantId.value || !reviewId.value) return
  saving.value = true
  try {
    await api.put(
      `/api/lease-review/reviews/${reviewId.value}/tenants/${tenantId.value}/abstract`,
      { sections: abstractData.value.sections }
    )
    editing.value = false
    // Refresh tenant list to update has_abstract status
    await loadTenantList()
  } catch (e: any) {
    alert('Failed to save: ' + (e.response?.data?.error || e.message))
  } finally {
    saving.value = false
  }
}

function selectTenant(tid: number) {
  tenantId.value = tid
  router.replace({
    query: { ...route.query, tenant: String(tid) },
  })
}

function printAbstract() {
  updatePrintTimestamp()
  nextTick(() => {
    const origTitle = document.title
    document.title = ''
    window.print()
    document.title = origTitle
  })
}

// Navigate to prev/next tenant
const currentIndex = computed(() =>
  tenantList.value.findIndex(t => t.tenant_id === tenantId.value)
)
function goTenant(delta: number) {
  const idx = currentIndex.value + delta
  if (idx >= 0 && idx < tenantList.value.length) {
    selectTenant(tenantList.value[idx].tenant_id)
  }
}

// Initialize from query params
onMounted(() => {
  reviewId.value = Number(route.query.review) || null
  tenantId.value = Number(route.query.tenant) || null
  if (reviewId.value) {
    loadTenantList()
    if (tenantId.value) loadAbstract()
  }
})

watch(tenantId, () => { loadAbstract() })
</script>

<template>
  <div class="lease-abstract-page">
    <!-- Header (hidden in print) -->
    <div class="abstract-header no-print">
      <div class="header-left">
        <button class="btn-back" @click="router.push({
          path: '/lease-risk-analysis',
          query: { review: reviewId ? String(reviewId) : undefined }
        })">
          &larr; Back to Risk Analysis
        </button>
        <h1>Lease Abstract</h1>
      </div>
      <div class="header-controls">
        <select
          v-if="tenantList.length"
          :value="tenantId"
          @change="selectTenant(Number(($event.target as HTMLSelectElement).value))"
          class="tenant-select"
        >
          <option v-for="t in tenantList" :key="t.tenant_id" :value="t.tenant_id">
            {{ t.tenant_name }} ({{ t.suite || 'N/A' }})
            {{ t.has_abstract ? '' : ' [new]' }}
          </option>
        </select>
        <button class="btn-nav" :disabled="currentIndex <= 0" @click="goTenant(-1)">&lsaquo; Prev</button>
        <button class="btn-nav" :disabled="currentIndex >= tenantList.length - 1" @click="goTenant(1)">Next &rsaquo;</button>
        <template v-if="abstractData">
          <button v-if="!editing" class="btn-edit" @click="editing = true">Edit</button>
          <template v-else>
            <button class="btn-save" @click="saveAbstract" :disabled="saving">
              {{ saving ? 'Saving...' : 'Save' }}
            </button>
            <button class="btn-cancel" @click="editing = false; loadAbstract()">Cancel</button>
          </template>
          <button class="btn-print" @click="printAbstract">Print</button>
        </template>
      </div>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="loading-bar">Loading abstract...</div>

    <!-- No selection -->
    <div v-else-if="!tenantId" class="empty-state">
      Select a tenant to view their lease abstract.
    </div>

    <!-- Abstract Content -->
    <div v-else-if="abstractData" class="abstract-page" id="abstract-printable">
      <!-- Print timestamp -->
      <div class="print-timestamp print-only">{{ printTimestamp }}</div>

      <!-- Title block -->
      <div class="abstract-title-block">
        <h2 class="abstract-property">{{ abstractData.property_name }}</h2>
        <h3 class="abstract-tenant">{{ abstractData.tenant_name }}</h3>
        <div class="abstract-suite" v-if="abstractData.suite">Suite {{ abstractData.suite }}</div>
      </div>

      <!-- Sections as a single-column table matching Word format -->
      <table class="abstract-table">
        <tbody>
          <template v-for="s in abstractData.sections" :key="s.section_key">
            <tr class="section-row">
              <td class="section-label">{{ s.section_title }}:</td>
              <td class="section-content">
                <template v-if="editing">
                  <textarea
                    v-model="s.content"
                    class="section-textarea"
                    :rows="Math.max(2, (s.content || '').split('\n').length + 1)"
                  />
                  <div class="ref-edit">
                    <label>Lease Ref:</label>
                    <input v-model="s.lease_ref" class="ref-input" placeholder="e.g. Lease Sec: 5.1" />
                  </div>
                </template>
                <template v-else>
                  <span class="content-text" :class="{ empty: !s.content }">
                    {{ s.content || '—' }}
                  </span>
                  <span v-if="s.lease_ref" class="lease-ref">({{ s.lease_ref }})</span>
                </template>
              </td>
            </tr>
          </template>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
/* ─── Page layout ─── */
.lease-abstract-page {
  padding: 24px 32px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #1a1a1a;
}

/* ─── Header ─── */
.abstract-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  flex-wrap: wrap;
  gap: 12px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}
.header-left h1 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1F4E79;
  margin: 0;
}
.header-controls {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.btn-back {
  padding: 6px 14px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 0.85rem;
  color: #1F4E79;
}
.btn-back:hover { background: #f0f4f8; }
.tenant-select {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 0.85rem;
  min-width: 280px;
}
.btn-nav {
  padding: 6px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  background: #fff;
  cursor: pointer;
  font-size: 0.9rem;
}
.btn-nav:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-nav:not(:disabled):hover { background: #f0f4f8; }
.btn-edit {
  padding: 6px 16px;
  border: 1px solid #1F4E79;
  border-radius: 4px;
  background: #fff;
  color: #1F4E79;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.85rem;
}
.btn-edit:hover { background: #e3eef8; }
.btn-save {
  padding: 6px 16px;
  border: none;
  border-radius: 4px;
  background: #28a745;
  color: #fff;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.85rem;
}
.btn-save:hover { background: #218838; }
.btn-save:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-cancel {
  padding: 6px 16px;
  border: 1px solid #999;
  border-radius: 4px;
  background: #fff;
  color: #666;
  cursor: pointer;
  font-size: 0.85rem;
}
.btn-cancel:hover { background: #f0f0f0; }
.btn-print {
  padding: 6px 16px;
  border: 1px solid #1F4E79;
  border-radius: 4px;
  background: #1F4E79;
  color: #fff;
  cursor: pointer;
  font-weight: 600;
  font-size: 0.85rem;
}
.btn-print:hover { background: #16395a; }

/* ─── States ─── */
.loading-bar {
  padding: 40px;
  text-align: center;
  color: #666;
  font-style: italic;
}
.empty-state {
  padding: 40px;
  text-align: center;
  color: #999;
}

/* ─── Abstract page content ─── */
.abstract-page {
  max-width: 900px;
  margin: 0 auto;
  background: #fff;
  border: 1px solid #ddd;
  padding: 32px 40px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.abstract-title-block {
  text-align: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 2px solid #1F4E79;
}
.abstract-property {
  font-size: 1.3rem;
  font-weight: 700;
  color: #1F4E79;
  margin: 0 0 4px;
}
.abstract-tenant {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
  margin: 0 0 2px;
}
.abstract-suite {
  font-size: 0.9rem;
  color: #666;
}

/* ─── Abstract table ─── */
.abstract-table {
  width: 100%;
  border-collapse: collapse;
}
.section-row td {
  padding: 6px 8px;
  vertical-align: top;
  border-bottom: 1px solid #eee;
}
.section-label {
  width: 200px;
  font-weight: 600;
  color: #1F4E79;
  font-size: 0.85rem;
  white-space: nowrap;
}
.section-content {
  font-size: 0.85rem;
  line-height: 1.5;
}
.content-text {
  white-space: pre-wrap;
  word-wrap: break-word;
}
.content-text.empty {
  color: #bbb;
  font-style: italic;
}
.lease-ref {
  display: inline-block;
  margin-left: 8px;
  color: #888;
  font-size: 0.78rem;
  font-style: italic;
}

/* ─── Edit mode ─── */
.section-textarea {
  width: 100%;
  padding: 6px 8px;
  border: 1px solid #ccc;
  border-radius: 3px;
  font-family: inherit;
  font-size: 0.85rem;
  line-height: 1.5;
  resize: vertical;
  min-height: 36px;
}
.section-textarea:focus {
  border-color: #1F4E79;
  outline: none;
  box-shadow: 0 0 0 2px rgba(31,78,121,0.15);
}
.ref-edit {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
}
.ref-edit label {
  font-size: 0.75rem;
  color: #888;
  white-space: nowrap;
}
.ref-input {
  flex: 1;
  padding: 3px 6px;
  border: 1px solid #ddd;
  border-radius: 3px;
  font-size: 0.78rem;
  color: #666;
}

/* ─── Print timestamp ─── */
.print-timestamp {
  display: none;
}

/* ─── Print styles ─── */
@media print {
  .no-print { display: none !important; }
  .print-only { display: block !important; }
  .print-timestamp {
    display: block;
    font-size: 8pt;
    color: #999;
    margin-bottom: 8px;
  }

  /* This view's real margin. It used to be `@page { margin: 0.5in }`, which
     could not be scoped and so reset the page box for every other print view
     in the app. Same 0.5in on paper, contained to this view. See App.vue.
     The page box itself (letter portrait) is now the global one — this rule
     said `size: letter` without an orientation, which left the choice to the
     print dialog; portrait is what it has always rendered as. */
  .lease-abstract-page {
    padding: 0.5in;
  }
  .abstract-page {
    border: none;
    box-shadow: none;
    padding: 0;
    max-width: none;
  }
  .abstract-title-block {
    border-bottom-color: #000;
  }
  .abstract-property {
    color: #000;
  }
  .section-label {
    color: #000;
  }
  .section-row td {
    border-bottom-color: #ccc;
    padding: 4px 6px;
    font-size: 9pt;
  }
  .section-label {
    font-size: 9pt;
  }
  .lease-ref {
    color: #666;
  }
  .content-text.empty {
    display: none;
  }
}
</style>
