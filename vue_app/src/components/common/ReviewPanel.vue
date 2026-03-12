<script setup lang="ts">
import { ref, computed } from 'vue'

interface ReviewNote {
  id: number
  user_id: number
  username: string
  review_role: string | null
  action: string
  note_text: string | null
  created_at: string
}

interface ReviewStatus {
  id: number | null
  vcode: string
  quarter: string
  status: string
  current_step: number
  current_step_label: string
  current_step_role: string | null
  can_submit: boolean
  can_approve: boolean
  can_return: boolean
  is_editable: boolean
  user_review_roles: string[]
  notes: ReviewNote[]
  submitted_by: number | null
  returned_to_step: number | null
  updated_at: string | null
}

const props = defineProps<{
  review: ReviewStatus | null
  loading: boolean
}>()

const emit = defineEmits<{
  submit: [note: string]
  approve: [note: string]
  return: [note: string]
  addNote: [note: string]
}>()

const noteText = ref('')
const returnNote = ref('')
const approveNote = ref('')
const showNotes = ref(true)
const showReturnForm = ref(false)

const statusColor = computed(() => {
  if (!props.review) return '#999'
  switch (props.review.status) {
    case 'draft': return '#666'
    case 'returned': return '#e65100'
    case 'approved': return '#2e7d32'
    default: return '#1565c0'  // pending states
  }
})

const statusLabel = computed(() => {
  if (!props.review) return 'Loading...'
  const r = props.review
  if (r.status === 'draft') return 'Draft'
  if (r.status === 'returned') return 'Returned'
  if (r.status === 'approved') return 'Approved'
  return `Pending: ${r.current_step_label}`
})

function formatDate(dt: string | null): string {
  if (!dt) return ''
  const d = new Date(dt)
  if (isNaN(d.getTime())) return dt
  return d.toLocaleDateString('en-US', { month: 'numeric', day: 'numeric' }) +
    ' ' + d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
}

function formatRole(role: string | null): string {
  if (!role) return ''
  return role.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatAction(action: string): string {
  switch (action) {
    case 'submit': return 'Submitted'
    case 'approve': return 'Approved'
    case 'return': return 'Returned'
    case 'note': return 'Note'
    default: return action
  }
}

function actionClass(action: string): string {
  switch (action) {
    case 'approve': return 'action-approve'
    case 'return': return 'action-return'
    case 'submit': return 'action-submit'
    default: return ''
  }
}

function handleSubmit() {
  emit('submit', noteText.value)
  noteText.value = ''
}

function handleApprove() {
  emit('approve', approveNote.value)
  approveNote.value = ''
}

function handleReturn() {
  if (!returnNote.value.trim()) return
  emit('return', returnNote.value)
  returnNote.value = ''
  showReturnForm.value = false
}

function handleAddNote() {
  if (!noteText.value.trim()) return
  emit('addNote', noteText.value)
  noteText.value = ''
}
</script>

<template>
  <div class="review-panel no-print" v-if="review">
    <!-- Status bar -->
    <div class="review-status-bar">
      <div class="status-indicator">
        <span class="status-dot" :style="{ background: statusColor }"></span>
        <span class="status-text">Status: <strong>{{ statusLabel }}</strong></span>
      </div>

      <div class="review-actions">
        <!-- Submit button (for asset managers when in draft/returned) -->
        <button
          v-if="review.can_submit"
          class="btn-review btn-submit"
          @click="handleSubmit"
          :disabled="loading"
        >
          Submit for Review
        </button>

        <!-- Approve button -->
        <template v-if="review.can_approve">
          <div class="approve-group">
            <input
              type="text"
              v-model="approveNote"
              placeholder="Optional note..."
              class="approve-input"
              @keyup.enter="handleApprove"
            />
            <button class="btn-review btn-approve" @click="handleApprove" :disabled="loading">
              Approve
            </button>
          </div>
        </template>

        <!-- Return button -->
        <template v-if="review.can_return">
          <button
            v-if="!showReturnForm"
            class="btn-review btn-return"
            @click="showReturnForm = true"
          >
            Return
          </button>
          <div v-else class="return-form">
            <input
              type="text"
              v-model="returnNote"
              placeholder="Reason for return (required)..."
              class="return-input"
              @keyup.enter="handleReturn"
            />
            <button class="btn-review btn-return" @click="handleReturn" :disabled="!returnNote.trim() || loading">
              Confirm Return
            </button>
            <button class="btn-review btn-cancel" @click="showReturnForm = false">Cancel</button>
          </div>
        </template>
      </div>
    </div>

    <!-- Review Notes -->
    <div class="review-notes-section">
      <button class="notes-toggle" @click="showNotes = !showNotes">
        {{ showNotes ? '&#x25BE;' : '&#x25B8;' }} Review Notes ({{ review.notes.length }})
      </button>

      <div v-if="showNotes" class="notes-list">
        <div
          v-for="n in review.notes"
          :key="n.id"
          class="note-item"
          :class="actionClass(n.action)"
        >
          <div class="note-header">
            <span class="note-date">{{ formatDate(n.created_at) }}</span>
            <span class="note-user">{{ n.username }}</span>
            <span v-if="n.review_role" class="note-role">({{ formatRole(n.review_role) }})</span>
            <span class="note-action" :class="actionClass(n.action)">— {{ formatAction(n.action) }}</span>
          </div>
          <div v-if="n.note_text" class="note-body">{{ n.note_text }}</div>
        </div>
        <div v-if="!review.notes.length" class="notes-empty">No review activity yet.</div>
      </div>

      <!-- Add note -->
      <div class="add-note-row">
        <input
          type="text"
          v-model="noteText"
          placeholder="Add a note..."
          class="note-input"
          @keyup.enter="handleAddNote"
        />
        <button class="btn-review btn-note" @click="handleAddNote" :disabled="!noteText.trim() || loading">
          Post
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.review-panel {
  background: #f8f9fa;
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 12px 16px;
  margin-bottom: 16px;
  font-size: 13px;
}

.review-status-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 8px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
}

.status-text {
  font-size: 14px;
}

.review-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.btn-review {
  padding: 5px 14px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 500;
}
.btn-review:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-submit { background: #1565c0; color: white; }
.btn-submit:hover { background: #0d47a1; }
.btn-approve { background: #2e7d32; color: white; }
.btn-approve:hover { background: #1b5e20; }
.btn-return { background: #e65100; color: white; }
.btn-return:hover { background: #bf360c; }
.btn-cancel { background: #757575; color: white; }
.btn-cancel:hover { background: #616161; }
.btn-note { background: #1F4E79; color: white; }
.btn-note:hover { background: #163a5c; }

.approve-group {
  display: flex;
  gap: 4px;
  align-items: center;
}

.approve-input, .return-input {
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
  width: 200px;
}

.return-form {
  display: flex;
  gap: 4px;
  align-items: center;
}

/* Notes section */
.review-notes-section {
  border-top: 1px solid #dee2e6;
  padding-top: 8px;
}

.notes-toggle {
  background: none;
  border: none;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: #333;
  padding: 2px 0;
}
.notes-toggle:hover { color: #1565c0; }

.notes-list {
  max-height: 240px;
  overflow-y: auto;
  margin: 6px 0;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background: white;
}

.note-item {
  padding: 6px 10px;
  border-bottom: 1px solid #f0f0f0;
}
.note-item:last-child { border-bottom: none; }

.note-header {
  display: flex;
  gap: 6px;
  align-items: center;
  font-size: 12px;
  color: #666;
}

.note-date { color: #999; min-width: 100px; }
.note-user { font-weight: 600; color: #333; }
.note-role { font-size: 11px; color: #888; }

.note-action { font-weight: 500; }
.note-action.action-approve { color: #2e7d32; }
.note-action.action-return { color: #e65100; }
.note-action.action-submit { color: #1565c0; }

.note-body {
  margin-top: 2px;
  font-size: 12px;
  color: #444;
  padding-left: 106px;
}

.notes-empty {
  padding: 12px;
  color: #999;
  font-style: italic;
  text-align: center;
  font-size: 12px;
}

.add-note-row {
  display: flex;
  gap: 6px;
  margin-top: 6px;
}

.note-input {
  flex: 1;
  padding: 5px 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
}
</style>
