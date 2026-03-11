import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/client'

export interface WaterfallStep {
  iOrder: number
  PropCode: string
  vState: string
  FXRate: number
  nPercent: number
  mAmount: number
  vtranstype: string
  vAmtType: string
  vNotes: string
}

export interface WaterfallEntity {
  vcode: string
  name: string
  label: string
  has_wf: boolean
}

interface Investor {
  investor_id: string
  ownership_pct: number
}

interface ValidationResult {
  errors: string[]
  warnings: string[]
}

interface PreviewAllocation {
  PropCode: string
  vState: string
  Allocated: number
}

export const useWaterfallStore = defineStore('waterfall', () => {
  const currentEntity = ref<string>('')
  const cfSteps = ref<WaterfallStep[]>([])
  const capSteps = ref<WaterfallStep[]>([])
  const entities = ref<WaterfallEntity[]>([])
  const investors = ref<Investor[]>([])
  const validation = ref<ValidationResult>({ errors: [], warnings: [] })
  const loading = ref(false)
  const saving = ref(false)
  const hasCf = ref(false)
  const hasCap = ref(false)
  const vstateOptions = ref<string[]>([
    'Pref', 'Initial', 'Add', 'Tag', 'Share', 'IRR',
    'Amt', 'Def&Int', 'Def_Int', 'Default', 'AMFee', 'Promote',
  ])

  // Preview state
  const previewResult = ref<PreviewAllocation[] | null>(null)
  const previewTotal = ref(0)
  const previewError = ref('')
  const previewing = ref(false)

  // Dirty tracking
  const cfDirty = ref(false)
  const capDirty = ref(false)
  const hasWaterfall = computed(() => hasCf.value || hasCap.value || cfSteps.value.length > 0 || capSteps.value.length > 0)

  async function loadEntities() {
    const res = await api.get('/api/waterfall-setup/entities')
    entities.value = res.data.entities
  }

  async function loadSteps(vcode: string) {
    loading.value = true
    try {
      const [stepsRes, invRes] = await Promise.all([
        api.get(`/api/waterfall-setup/${vcode}/steps`),
        api.get(`/api/waterfall-setup/${vcode}/investors`),
      ])
      cfSteps.value = stepsRes.data.cf_wf
      capSteps.value = stepsRes.data.cap_wf
      hasCf.value = stepsRes.data.has_cf
      hasCap.value = stepsRes.data.has_cap
      investors.value = invRes.data.investors
      currentEntity.value = vcode
      cfDirty.value = false
      capDirty.value = false
      validation.value = { errors: [], warnings: [] }
      previewResult.value = null
      previewError.value = ''
    } finally {
      loading.value = false
    }
  }

  async function saveSteps(vcode: string, wfType: string, steps: WaterfallStep[]) {
    saving.value = true
    try {
      // Include the other type's steps so they're preserved
      const otherSteps = wfType === 'CF_WF' ? capSteps.value : cfSteps.value
      const res = await api.put(`/api/waterfall-setup/${vcode}/steps`, {
        wf_type: wfType,
        steps,
        other_steps: otherSteps.length > 0 ? otherSteps : undefined,
      })
      if (res.data.success) {
        // Refresh steps from DB
        await loadSteps(vcode)
      }
      return res.data
    } finally {
      saving.value = false
    }
  }

  async function validateSteps(steps: WaterfallStep[], wfType: string) {
    const res = await api.post('/api/waterfall-setup/validate', { steps, wf_type: wfType })
    validation.value = res.data
    return res.data
  }

  async function previewWaterfall(vcode: string, wfType: string, steps: WaterfallStep[]) {
    previewing.value = true
    previewResult.value = null
    previewError.value = ''
    try {
      const res = await api.post(`/api/waterfall-setup/${vcode}/preview`, {
        steps,
        wf_type: wfType,
      })
      if (res.data.success) {
        previewResult.value = res.data.allocations
        previewTotal.value = res.data.total || 0
      } else {
        previewError.value = res.data.error || 'Preview failed'
      }
    } catch (e: any) {
      previewError.value = e.response?.data?.error || e.message
    } finally {
      previewing.value = false
    }
  }

  async function copyCfToCap(vcode: string, cfDraftSteps: WaterfallStep[]) {
    // Use current CF draft steps as the new Cap steps
    capSteps.value = cfDraftSteps.map((s) => ({ ...s }))
    capDirty.value = true
  }

  async function copyFromEntity(sourceVcode: string) {
    const res = await api.get(`/api/waterfall-setup/copy-from/${sourceVcode}`)
    cfSteps.value = res.data.cf_wf || []
    capSteps.value = res.data.cap_wf || []
    cfDirty.value = true
    capDirty.value = true
  }

  async function createFromTemplate(vcode: string, template: 'new' | 'pari-passu') {
    const res = await api.post(`/api/waterfall-setup/${vcode}/template/${template}`)
    cfSteps.value = res.data.cf_wf || []
    capSteps.value = res.data.cap_wf || []
    cfDirty.value = true
    capDirty.value = true
  }

  function resetSteps(vcode: string) {
    // Reload from DB (discards unsaved changes)
    return loadSteps(vcode)
  }

  function markDirty(wfType: string) {
    if (wfType === 'CF_WF') cfDirty.value = true
    else capDirty.value = true
  }

  return {
    currentEntity, cfSteps, capSteps, entities, investors,
    validation, loading, saving, hasCf, hasCap, vstateOptions,
    previewResult, previewTotal, previewError, previewing,
    cfDirty, capDirty, hasWaterfall,
    loadEntities, loadSteps, saveSteps, validateSteps,
    previewWaterfall, copyCfToCap, copyFromEntity,
    createFromTemplate, resetSteps, markDirty,
  }
})
