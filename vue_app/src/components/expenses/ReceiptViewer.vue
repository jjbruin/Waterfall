<script setup lang="ts">
/**
 * The receipt beside the line it supports (Jim, Oct 2 2026: "see the image of
 * the uploaded invoice related to the line they are completing ... to correct
 * amounts that may have been hand written").
 *
 * The file is fetched WITH the signed-in token and shown from a blob URL: an
 * <img src="/api/..."> cannot send the Authorization header, and the endpoint
 * is per-record (owner, approver, accounting), so it must not be reachable
 * without it. A PDF opens at the receipt's page.
 */
import { ref, watch, onBeforeUnmount, computed } from 'vue'
import api from '@/api/client'

const props = defineProps<{
  reportId: number
  receiptId: number | null
  page?: number | null
  contentType?: string
  viewType?: string | null
  filename?: string
  extracted?: any            // what the reader read for THIS line
  amount?: number | null     // the line's amount as it stands
}>()

const url = ref<string | null>(null)
const error = ref<string | null>(null)
const loading = ref(false)
const rotate = ref(0)
// What the server actually sent. A caller that does not pass the content type
// (Expense Coding's rows carry none) must still show a PDF as a PDF -- before
// this, the PDF went into an <img> and rendered as a broken image.
const blobType = ref<string | null>(null)
const zoom = ref(1)
let current: string | null = null

const isPdf = computed(() =>
  (props.viewType || props.contentType || blobType.value) === 'application/pdf')
const src = computed(() =>
  url.value && isPdf.value ? `${url.value}#page=${props.page || 1}&view=FitH` : url.value)

async function load() {
  if (current) { URL.revokeObjectURL(current); current = null }
  url.value = null
  error.value = null
  rotate.value = 0
  zoom.value = 1
  if (!props.receiptId) return
  loading.value = true
  try {
    const r = await api.get(
      `/api/expenses/reports/${props.reportId}/receipts/${props.receiptId}/file`,
      { responseType: 'blob' })
    blobType.value = r.data?.type || null
    current = URL.createObjectURL(r.data)
    url.value = current
  } catch (e: any) {
    error.value = e?.response?.status === 404 ? 'This receipt is not available.' :
      'The receipt could not be loaded.'
  } finally {
    loading.value = false
  }
}
watch(() => [props.reportId, props.receiptId], load, { immediate: true })
onBeforeUnmount(() => { if (current) URL.revokeObjectURL(current) })

const fmt = (v: any) => v == null ? '—' :
  Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const corrected = computed(() => {
  const x = props.extracted
  return x && x.total != null && props.amount != null &&
    Math.abs(Number(x.total) - Number(props.amount)) >= 0.005
})
</script>

<template>
  <div class="viewer">
    <div class="bar">
      <span class="name" :title="filename">{{ filename }}<template v-if="page && isPdf"> · page {{ page }}</template></span>
      <template v-if="url && !isPdf">
        <button class="link" title="Rotate" @click="rotate = (rotate + 90) % 360">⟳ rotate</button>
        <button class="link" @click="zoom = zoom >= 3 ? 1 : zoom + 0.5">
          {{ zoom >= 3 ? 'fit' : 'zoom +' }}</button>
      </template>
      <a v-if="url" class="link" :href="url" target="_blank" rel="noopener">open</a>
    </div>

    <!-- What the reader saw, so a corrected figure can be checked against it.
         A handwritten amount is POINTED AT: that is the case Jim named. -->
    <div v-if="extracted" class="read" :class="{ warn: extracted.handwritten_amount || extracted.amount_note }">
      <div>
        Read: <strong>{{ fmt(extracted.total) }}</strong>
        <span v-if="extracted.printed_total != null && extracted.printed_total !== extracted.total">
          (printed {{ fmt(extracted.printed_total) }})</span>
        <span v-if="extracted.tip != null"> · tip {{ fmt(extracted.tip) }}</span>
        <span v-if="extracted.date"> · {{ extracted.date }}</span>
        <span v-if="extracted.vendor"> · {{ extracted.vendor }}</span>
      </div>
      <div v-if="extracted.handwritten_amount" class="note">
        An amount on this receipt is handwritten — check the total against the image.</div>
      <div v-if="extracted.amount_note" class="note">{{ extracted.amount_note }}</div>
      <div v-if="corrected" class="note ok">
        You changed the amount from {{ fmt(extracted.total) }} to {{ fmt(amount) }}.</div>
    </div>

    <div class="frame">
      <div v-if="loading" class="muted">Loading the receipt…</div>
      <div v-else-if="error" class="muted">{{ error }}</div>
      <iframe v-else-if="src && isPdf" :src="src" title="Receipt"></iframe>
      <div v-else-if="src" class="img-wrap">
        <img :src="src" alt="Receipt"
             :style="{ transform: `rotate(${rotate}deg)`, width: `${zoom * 100}%` }" />
      </div>
      <div v-else class="muted">No receipt is attached to this line.</div>
    </div>
  </div>
</template>

<style scoped>
.viewer { display: flex; flex-direction: column; gap: 6px; min-width: 0; height: 100%; }
.bar { display: flex; gap: 10px; align-items: center; font-size: 12px; }
.name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text-secondary); }
.link { background: none; border: none; color: var(--color-primary, #2f6f4f); cursor: pointer; padding: 0; font-size: 12px; }
.read { font-size: 12.5px; padding: 6px 8px; border: 1px solid var(--color-border); border-radius: 5px; background: var(--color-surface); }
.read.warn { border-color: #e6c9a8; background: #fdf6ee; }
.note { margin-top: 3px; color: #8a4f1f; }
.note.ok { color: var(--color-text-secondary); }
.frame { flex: 1; min-height: 420px; border: 1px solid var(--color-border); border-radius: 5px; overflow: auto; background: #f6f7f9; }
.frame iframe { width: 100%; height: 100%; min-height: 560px; border: 0; }
.img-wrap { padding: 6px; text-align: center; }
.img-wrap img { max-width: none; transition: transform .15s; transform-origin: center center; }
.muted { color: var(--color-text-secondary); font-size: 12px; padding: 12px; }
</style>
