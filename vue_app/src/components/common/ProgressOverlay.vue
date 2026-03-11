<script setup lang="ts">
defineProps<{
  visible: boolean
  message?: string
  progress?: { current: number; total: number; deal: string } | null
}>()
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="overlay">
      <div class="overlay-content">
        <div class="spinner"></div>
        <p class="message">{{ message || 'Loading...' }}</p>
        <template v-if="progress && progress.total > 0">
          <div class="progress-bar">
            <div class="progress-fill"
                 :style="{ width: Math.round((progress.current / progress.total) * 100) + '%' }">
            </div>
          </div>
          <p class="progress-text">
            {{ progress.current }} / {{ progress.total }}
            <span v-if="progress.deal"> — {{ progress.deal }}</span>
          </p>
        </template>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.overlay-content {
  background: white;
  padding: 32px 48px;
  border-radius: 12px;
  text-align: center;
  min-width: 340px;
}

.spinner {
  width: 40px;
  height: 40px;
  border: 4px solid var(--color-border);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin: 0 auto 12px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.message {
  color: var(--color-text-secondary);
  font-size: 14px;
  margin-bottom: 8px;
}

.progress-bar {
  background: #e5e7eb;
  border-radius: 6px;
  height: 10px;
  overflow: hidden;
  margin-top: 12px;
}

.progress-fill {
  background: var(--color-accent, #1F4E79);
  height: 100%;
  border-radius: 6px;
  transition: width 0.3s ease;
}

.progress-text {
  color: var(--color-text-secondary);
  font-size: 12px;
  margin-top: 6px;
}
</style>
