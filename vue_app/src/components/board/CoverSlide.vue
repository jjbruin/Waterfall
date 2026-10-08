<script setup lang="ts">
/** The cover, as the January 2026 deck's: the logo, "Board Meeting" and the meeting date. */
import { computed } from 'vue'
import BoardSlide from './BoardSlide.vue'
import logo from '@/assets/brand/psc-logo.png'

const props = defineProps<{
  meetingDate: string
  version: 'full' | 'abbr'
  page: string | number
  notes?: { footnotes: string[]; disclosure?: string | null }
}>()
const when = computed(() => {
  const d = new Date(props.meetingDate + 'T00:00:00')
  return isNaN(d.getTime()) ? props.meetingDate
    : d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
})
</script>

<template>
  <BoardSlide :page="page" :footnotes="notes?.footnotes" :disclosure="notes?.disclosure">
    <div class="cover">
      <img :src="logo" alt="Peaceable Street Capital" class="logo" />
      <div class="t">Board Meeting</div>
      <div class="d">{{ when }}</div>
      <div v-if="version === 'abbr'" class="v">Results package</div>
    </div>
  </BoardSlide>
</template>

<style scoped>
.cover { height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.logo { width: 380px; height: auto; margin-bottom: 40px; }
.t { font-size: 42px; color: #000; }
.d { font-size: 30px; color: #404040; margin-top: 10px; font-weight: 300; }
.v { font-size: 18px; color: #6b7280; margin-top: 14px; letter-spacing: .5px; }
</style>
