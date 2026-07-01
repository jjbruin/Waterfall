import { ref, onUnmounted } from 'vue'

const updateAvailable = ref(false)
const knownVersion = ref<string | null>(null)
let intervalId: ReturnType<typeof setInterval> | null = null
let started = false

async function checkVersion() {
  try {
    const resp = await fetch('/api/data/version', { cache: 'no-store' })
    if (!resp.ok) return
    const data = await resp.json()
    const serverVersion = data.version as string
    if (!knownVersion.value) {
      knownVersion.value = serverVersion
    } else if (serverVersion !== knownVersion.value) {
      updateAvailable.value = true
    }
  } catch {
    // Network error — skip
  }
}

export function useVersionCheck() {
  if (!started) {
    started = true
    checkVersion()
    // Check every 5 minutes
    intervalId = setInterval(checkVersion, 5 * 60 * 1000)
  }

  function onNavigate() {
    checkVersion()
  }

  function reloadApp() {
    window.location.reload()
  }

  return { updateAvailable, onNavigate, reloadApp }
}
