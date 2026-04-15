import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '../api/client'

interface User {
  id: number
  username: string
  role: string
}

interface ManagedUser {
  id: number
  username: string
  role: string
  email: string | null
  must_change_password: boolean
  created_at: string
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(localStorage.getItem('token'))

  // Force password change state
  const mustChangePassword = ref(false)
  const pendingUsername = ref('')

  const isAuthenticated = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.role === 'admin')
  const isAnalyst = computed(() => user.value?.role === 'analyst' || user.value?.role === 'admin')
  const userRole = computed(() => user.value?.role || 'viewer')

  // User management state (admin only)
  const users = ref<ManagedUser[]>([])
  const usersLoading = ref(false)

  async function login(username: string, password: string) {
    const res = await api.post('/auth/login', { username, password })

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

  async function deleteUser(userId: number) {
    await api.delete(`/auth/users/${userId}`)
    await loadUsers()
  }

  return {
    user, token, isAuthenticated, isAdmin, isAnalyst, userRole,
    mustChangePassword, pendingUsername,
    users, usersLoading,
    login, fetchMe, logout, changePassword,
    forceChangePassword, forgotPassword, resetPassword, validateResetToken,
    loadUsers, createUser, updateUserRole, updateUserEmail, deleteUser,
  }
})
