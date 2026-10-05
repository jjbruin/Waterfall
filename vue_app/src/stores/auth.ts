import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/client'

interface User {
  id: number
  username: string
  role: string
  // Section keys this user may open (flask_app/auth/sections.py). Absent only
  // on a payload from before section access existed -- treated as "all", since
  // the server is the control and this only decides what is shown.
  sections?: string[]
}

export interface AppSection {
  key: string
  label: string
  routes: string[]
}

interface ManagedUser {
  id: number
  username: string
  role: string
  email: string | null
  must_change_password: boolean
  created_at: string
  denied_sections: string[]
  superuser: boolean
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(localStorage.getItem('token'))

  // Force password change state
  const mustChangePassword = ref(false)
  const pendingUsername = ref('')

  const isAuthenticated = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  // Mirrors ROLE_LEVELS in flask_app/auth/routes.py. isAnalyst means "at least
  // analyst", which is what every caller uses it for -- so the three accounting
  // roles belong in it. If this list and the backend's ever disagree the screen
  // hides a control the API would in fact have allowed, which is confusing but
  // not a security hole: the backend decides, this only decides what is shown.
  const ANALYST_OR_ABOVE = ['analyst', 'accountant', 'accounting_manager', 'cfo', 'admin']
  const isAnalyst = computed(() => ANALYST_OR_ABOVE.includes(user.value?.role || ''))
  const userRole = computed(() => user.value?.role || 'viewer')

  // Mirrors ACCOUNTING_ROLES in flask_app/auth/routes.py, and it is a
  // MEMBERSHIP list, not a level. Jim, Sep 17 2026: "Only the accountants,
  // accounting manager, and cfo should be able to edit anything in the
  // accounting section of the app generally. Me as admin, can edit only so I
  // can help them get something fixed while we are building and testing the
  // model." An analyst is deliberately NOT here -- note it IS in
  // ANALYST_OR_ABOVE above, so the two lists differ on purpose.
  const ACCOUNTING_ROLES = ['admin', 'cfo', 'accounting_manager', 'accountant']
  // ...AND the Accounting section (Jim, Oct 2 2026): an admin ROLE with
  // Accounting unticked builds the system but is not accounting. Mirrors
  // has_accounting_authority; the `admin` username has every section.
  const canEditAccounting = computed(
    () => ACCOUNTING_ROLES.includes(user.value?.role || '') && hasSection('accounting'))

  // Mirrors CLOSE_PLAN_ROLES. The CFO sets the PLAN of the close -- when it
  // opens, when each thing is due, and the order entities are worked in.
  // The team works inside it: syncing, naming preparers, setting a
  // property, signing off. Jim, Sep 17 2026, in three passes: cycles,
  // then "deadlines should be CFO only too", then "order number should be
  // CFO only too".
  const CLOSE_PLAN_ROLES = ['admin', 'cfo']
  const canSetClosePlan = computed(
    () => CLOSE_PLAN_ROLES.includes(user.value?.role || ''))

  // ── Section access by username ─────────────────────────────────────
  // The catalogue comes from the server (GET /auth/sections), which reads the
  // one registry in flask_app/auth/sections.py. Nothing here lists sections:
  // a section added there shows up in the sidebar gate, the router guard and
  // the User Management columns without touching this file.
  const sectionCatalog = ref<AppSection[]>([])
  // Only the `admin` USERNAME may change the boxes (Jim, Oct 1 2026); the
  // server says whether THIS user may, and refuses the write regardless.
  const canAssignSections = ref(false)
  // Sections granted together, e.g. Asset Management with New Business.
  const linkedSections = ref<string[][]>([])
  let catalogPromise: Promise<void> | null = null

  function loadSectionCatalog(force = false) {
    if (catalogPromise && !force) return catalogPromise
    catalogPromise = api.get('/auth/sections')
      .then(res => {
        sectionCatalog.value = res.data.sections || []
        canAssignSections.value = !!res.data.can_assign
        linkedSections.value = res.data.linked || []
      })
      .catch(() => { catalogPromise = null })
    return catalogPromise
  }

  function linkedTo(key: string) {
    const grp = linkedSections.value.find(g => g.includes(key))
    return grp ? grp.filter(k => k !== key) : []
  }

  // Sections ticked for NOBODY until granted (auth/sections.py `opt_in`). Before
  // the user's sections have loaded, everything else reads as allowed and these
  // do not -- so the Board link never flashes up for a user without it.
  const OPT_IN_SECTIONS = ['board']

  function hasSection(key: string) {
    const s = user.value?.sections
    if (!s) return !OPT_IN_SECTIONS.includes(key)
    return s.includes(key)
  }

  // The section owning a screen path, '' if none does. Prefix match, so
  // /portfolio-snapshot/print belongs with /portfolio-snapshot.
  function sectionForPath(path: string) {
    for (const sec of sectionCatalog.value) {
      if (sec.routes.some(r => path === r || path.startsWith(r + '/'))) return sec.key
    }
    return ''
  }

  // Where to land a user who may not open the screen they asked for.
  function firstAllowedPath() {
    const sec = sectionCatalog.value.find(s => hasSection(s.key))
    return sec?.routes[0] || '/settings'
  }

  function sectionLabel(key: string) {
    return sectionCatalog.value.find(s => s.key === key)?.label || key
  }

  // User management state (admin only)
  const users = ref<ManagedUser[]>([])
  const usersLoading = ref(false)

  async function login(username: string, password: string) {
    const res = await api.post('/auth/login', { username, password })
    catalogPromise = null  // can_assign belongs to whoever is signing in

    // Check if user must change password
    if (res.data.must_change_password) {
      mustChangePassword.value = true
      pendingUsername.value = res.data.username
      return { mustChangePassword: true }
    }

    token.value = res.data.token
    user.value = res.data.user
    mustChangePassword.value = false
    pendingUsername.value = ''
    localStorage.setItem('token', res.data.token)
    return { mustChangePassword: false }
  }

  async function forceChangePassword(currentPassword: string, newPassword: string) {
    const res = await api.post('/auth/force-change-password', {
      username: pendingUsername.value,
      current_password: currentPassword,
      new_password: newPassword,
    })
    token.value = res.data.token
    user.value = res.data.user
    mustChangePassword.value = false
    pendingUsername.value = ''
    localStorage.setItem('token', res.data.token)
  }

  async function fetchMe() {
    if (!token.value) return
    try {
      const res = await api.get('/auth/me')
      user.value = res.data.user
    } catch {
      logout()
    }
  }

  function logout() {
    catalogPromise = null
    canAssignSections.value = false
    token.value = null
    user.value = null
    mustChangePassword.value = false
    pendingUsername.value = ''
    localStorage.removeItem('token')
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    await api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
  }

  async function forgotPassword(email: string) {
    const res = await api.post('/auth/forgot-password', { email })
    return res.data.message
  }

  async function resetPassword(resetToken: string, newPassword: string) {
    const res = await api.post('/auth/reset-password', {
      token: resetToken,
      new_password: newPassword,
    })
    return res.data.message
  }

  async function validateResetToken(resetToken: string) {
    const res = await api.post('/auth/validate-reset-token', { token: resetToken })
    return res.data
  }

  // Admin: user management
  async function loadUsers() {
    usersLoading.value = true
    try {
      const res = await api.get('/auth/users')
      users.value = res.data.users
    } finally {
      usersLoading.value = false
    }
  }

  async function createUser(username: string, password: string, role: string,
                            email?: string, mustChange?: boolean,
                            sendWelcome?: boolean) {
    const res = await api.post('/auth/users', {
      username, password, role,
      email: email || undefined,
      must_change_password: mustChange ?? false,
      send_welcome_email: sendWelcome ?? false,
    })
    await loadUsers()
    return res.data
  }

  async function updateUserRole(userId: number, role: string) {
    await api.put(`/auth/users/${userId}/role`, { role })
    await loadUsers()
  }

  async function updateUserEmail(userId: number, email: string) {
    await api.put(`/auth/users/${userId}/email`, { email })
    await loadUsers()
  }

  async function updateUserSections(userId: number, sections: Record<string, boolean>) {
    await api.put(`/auth/users/${userId}/sections`, { sections })
    await loadUsers()
  }

  async function deleteUser(userId: number) {
    await api.delete(`/auth/users/${userId}`)
    await loadUsers()
  }

  return {
    user, token, isAuthenticated, isAdmin, isAnalyst, userRole,
    canEditAccounting, canSetClosePlan,
    sectionCatalog, loadSectionCatalog, hasSection, sectionForPath,
    canAssignSections, linkedSections, linkedTo,
    firstAllowedPath, sectionLabel, updateUserSections,
    mustChangePassword, pendingUsername,
    users, usersLoading,
    login, fetchMe, logout, changePassword,
    forceChangePassword, forgotPassword, resetPassword, validateResetToken,
    loadUsers, createUser, updateUserRole, updateUserEmail, deleteUser,
  }
})
