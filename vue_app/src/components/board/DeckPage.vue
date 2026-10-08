<script setup lang="ts">
/**
 * One page of the package, whatever its kind. The viewer and the printed copy
 * both draw pages through this, so what is on screen is what prints.
 */
import AssetClassSlide from './AssetClassSlide.vue'
import CapitalizationSlide from './CapitalizationSlide.vue'
import ContentsSlide from './ContentsSlide.vue'
import CoverSlide from './CoverSlide.vue'
import DividerSlide from './DividerSlide.vue'
import InvestmentSummarySlide from './InvestmentSummarySlide.vue'
import NarrativePageSlide from './NarrativePageSlide.vue'
import PartnerSlide from './PartnerSlide.vue'
import PerformanceSlide from './PerformanceSlide.vue'
import PortfolioMetricsSlide from './PortfolioMetricsSlide.vue'
import PrefByYearSlide from './PrefByYearSlide.vue'

defineProps<{
  slide: any
  page: number
  view: any
  status?: string
  error?: string
  notes: { footnotes: string[]; disclosure?: string | null }
  images: Record<string, string>
  contents: any[]
  meetingDate: string
  version: 'full' | 'abbr'
}>()
const emit = defineEmits<{ (e: 'go', page: number): void; (e: 'retry'): void }>()
</script>

<template>
  <CoverSlide v-if="slide.kind === 'cover'" :meeting-date="meetingDate" :version="version" :page="page" :notes="notes" />
  <ContentsSlide v-else-if="slide.kind === 'contents'" :parts="contents" :page="page" :notes="notes"
                 @go="(p: number) => emit('go', p)" />
  <DividerSlide v-else-if="slide.kind === 'divider'" :roman="slide.roman" :title="slide.title" :page="page" :notes="notes" />
  <NarrativePageSlide v-else-if="slide.kind === 'narrative'" :title="slide.title" :blocks="slide.blocks"
                      :images="images" :page="page" :notes="notes" />
  <template v-else-if="status === 'ready' && view">
    <PrefByYearSlide v-if="slide.kind === 'pref_by_year'" :view="view" :page="page" :notes="notes" />
    <AssetClassSlide v-else-if="slide.kind === 'exposure_asset_class'" :view="view" :page="page" :notes="notes" />
    <PartnerSlide v-else-if="slide.kind === 'exposure_partner'" :view="view" :page="page" :notes="notes" />
    <CapitalizationSlide v-else-if="slide.kind === 'capitalization'" :view="view" :page="page" :notes="notes" />
    <PerformanceSlide v-else-if="slide.kind === 'performance'" :view="view" :page="page" :notes="notes" />
    <PortfolioMetricsSlide v-else-if="slide.kind === 'debt'" :view="view" :page="page" :notes="notes" />
    <InvestmentSummarySlide v-else-if="slide.kind === 'investment_summaries'" :view="view" :slide="slide.spec"
                            :page="page" :notes="notes" />
  </template>
  <div v-else class="placeholder">
    <div class="ph-title">{{ slide.title }}</div>
    <div v-if="status === 'error'" class="ph-err">Could not load: {{ error }}
      <button class="btn" @click="emit('retry')">Try again</button></div>
    <div v-else>Building the figures from the engines…</div>
  </div>
</template>

<style scoped>
.placeholder { width: 1100px; height: 825px; background: #fff; color: #374151; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 12px; font-size: 20px; font-family: Calibri, 'Segoe UI', sans-serif; }
.ph-title { font-size: 28px; }
.ph-err { color: #b42318; font-size: 16px; }
.btn { padding: 4px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; border: 1px solid #d1d5db; background: #fff; }
</style>
