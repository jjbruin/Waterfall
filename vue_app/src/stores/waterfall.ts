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

interface OwnershipInvestor {
  investor_id: string
  name: string
  label: string
  ownership_pct: number
}

interface TreeEntity {
  id: string
  name: string
  pct?: number
  tags?: string[]
}

export interface OwnershipTree {
  owners: TreeEntity[]
  selected: { id: string; name: string } | null
  investments: TreeEntity[]
  investors: OwnershipInvestor[]
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

/**
 * Read a {cf_wf, cap_wf} body, or throw with a message worth showing.
 *
 * A response that fails to parse as JSON arrives here as a STRING, not an
 * object: axios's default `silentJSONParsing` swallows the parse error and
 * hands back the raw text instead of rejecting. `data.cf_wf` is then
 * `undefined`, and the old `|| []` turned that into an empty grid the caller
 * reported as a successful copy — which is exactly how a bare `NaN` in the
 * payload (fixed server-side in bf093c2) stayed invisible for 68 of 92 deals.
 *
 * So the shape is checked rather than defaulted. The server being fixed is not
 * a reason to keep trusting it: this is the layer that decides whether a
 * failure is visible.
 */
/**
 * Outcome of a call that loads steps into the drafts. Discriminated on
 * `success` so the caller cannot read a step count off a failure, and an
 * explicit annotation is required for that narrowing — an inferred return type
 * widens `true` to `boolean` and the union stops discriminating.
 */
export type StepsLoadResult =
  | { success: true; cf: number; cap: number }
  | { success: false; message: string }

/** The server's own error text where there is one, else the thrown message. */
function errMsg(e: any, fallback: string): string {
  return e?.response?.data?.error || e?.message || fallback
}

function readStepsPayload(data: any): { cf: WaterfallStep[]; cap: WaterfallStep[] } {
  if (data == null || typeof data !== 'object') {
    throw new Error(
      'The server returned a malformed response that could not be read as JSON.'
    )
  }
  if (!Array.isArray(data.cf_wf) && !Array.isArray(data.cap_wf)) {
    throw new Error('The server response contained no waterfall steps.')
  }
  return {
    cf: Array.isArray(data.cf_wf) ? data.cf_wf : [],
    cap: Array.isArray(data.cap_wf) ? data.cap_wf : [],
  }
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

  // Ownership tree
  const ownershipTree = ref<OwnershipTree>({ owners: [], selected: null, investments: [], investors: [] })

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

  const loadError = ref('')

  async function loadSteps(vcode: string) {
    loading.value = true
    loadError.value = ''
    try {
      const [stepsRes, invRes, treeRes] = await Promise.all([
        api.get(`/api/waterfall-setup/${vcode}/steps`),
        api.get(`/api/waterfall-setup/${vcode}/investors`).catch(() => ({ data: { investors: [] } })),
        api.get(`/api/waterfall-setup/${vcode}/ownership-tree`).catch(() => ({ data: { owners: [], selected: null, investments: [], investors: [] } })),
      ])
      // Same guard as the copy paths: a body that failed to parse must read as
      // a load error, not as "this entity has no waterfall".
      const { cf, cap } = readStepsPayload(stepsRes.data)
      cfSteps.value = cf
      capSteps.value = cap
      hasCf.value = stepsRes.data.has_cf || false
      hasCap.value = stepsRes.data.has_cap || false
      investors.value = invRes.data.investors || []
      ownershipTree.value = treeRes.data || { owners: [], selected: null, investments: [], investors: [] }
      currentEntity.value = vcode
      cfDirty.value = false
      capDirty.value = false
      validation.value = { errors: [], warnings: [] }
      previewResult.value = null
      previewError.value = ''
    } catch (e: any) {
      loadError.value = e.response?.data?.error || e.message || 'Failed to load waterfall steps'
      cfSteps.value = []
      capSteps.value = []
      hasCf.value = false
      hasCap.value = false
      currentEntity.value = vcode
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
        // Use fresh steps from response instead of 3 extra API calls
        if (res.data.cf_wf !== undefined) {
          cfSteps.value = res.data.cf_wf
          capSteps.value = res.data.cap_wf || []
          hasCf.value = res.data.has_cf || false
          hasCap.value = res.data.has_cap || false
        }
        cfDirty.value = false
        capDirty.value = false
        validation.value = { errors: [], warnings: [] }
        previewResult.value = null
        previewError.value = ''
      } else if (res.data.errors) {
        // Server-side validation failed
        validation.value = {
          errors: res.data.errors || [],
          warnings: res.data.warnings || [],
        }
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

  async function copyFromEntity(sourceVcode: string): Promise<StepsLoadResult> {
    try {
      const res = await api.get(`/api/waterfall-setup/copy-from/${sourceVcode}`)
      const { cf, cap } = readStepsPayload(res.data)
      cfSteps.value = cf
      capSteps.value = cap
      cfDirty.value = true
      capDirty.value = true
      return { success: true, cf: cf.length, cap: cap.length }
    } catch (e: any) {
      return { success: false, message: errMsg(e, 'Copy failed.') }
    }
  }

  async function createFromTemplate(vcode: string, template: 'new' | 'pari-passu'): Promise<StepsLoadResult> {
    try {
      const res = await api.post(`/api/waterfall-setup/${vcode}/template/${template}`)
      const { cf, cap } = readStepsPayload(res.data)
      cfSteps.value = cf
      capSteps.value = cap
      cfDirty.value = true
      capDirty.value = true
      return { success: true, cf: cf.length, cap: cap.length }
    } catch (e: any) {
      return { success: false, message: errMsg(e, 'Could not build the template.') }
    }
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
    currentEntity, cfSteps, capSteps, entities, investors, ownershipTree,
    validation, loading, loadError, saving, hasCf, hasCap, vstateOptions,
    previewResult, previewTotal, previewError, previewing,
    cfDirty, capDirty, hasWaterfall,
    loadEntities, loadSteps, saveSteps, validateSteps,
    previewWaterfall, copyCfToCap, copyFromEntity,
    createFromTemplate, resetSteps, markDirty,
  }
})
