<script setup lang="ts">
import { useDataStore } from '../../stores/data'

const data = useDataStore()
</script>

<template>
  <div class="toast-container">
    <div
      v-for="toast in data.toasts"
      :key="toast.id"
      class="toast"
      :class="toast.type"
      @click="data.dismissToast(toast.id)"
    >
      <span class="toast-icon" v-if="toast.type === 'success'">&#10003;</span>
      <span class="toast-icon" v-else-if="toast.type === 'error'">&#10007;</span>
      <span class="toast-icon" v-else>&#9432;</span>
      <span class="toast-message">{{ toast.message }}</span>
    </div>
  </div>
</template>

<style scoped>
.toast-container {
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}

.toast {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 16px;
  border-radius: 6px;
  font-size: 13px;
  color: white;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  animation: slideIn 0.2s ease-out;
  max-width: 400px;
}

.toast.success { background: #2e7d32; }
.toast.error { background: #c62828; }
.toast.info { background: #1565c0; }

.toast-icon { font-size: 14px; flex-shrink: 0; }
.toast-message { flex: 1; }

@keyframes slideIn {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
</style>
