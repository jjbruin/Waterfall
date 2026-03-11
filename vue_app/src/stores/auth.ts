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
  created_at: string
}

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const token = ref<string | null>(localStorage.getItem('token'))

  const isAuthenticated = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.role === 'admin')
  const isAnalyst = computed(() => user.value?.role === 'analyst' || user.value?.role === 'admin')
  const userRole = computed(() => user.value?.role || 'viewer')

  // User management state (admin only)
  const users = ref<ManagedUser[]>([])
  const usersLoading = ref(false)

  async function login(username: string, password: string) {
    const res = await api.post('/auth/login', { username, password })
    token.value = res.data.token
    user.value = res.data.user
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
    localStorage.removeItem('token')
  }

  async function changePassword(currentPassword: string, newPassword: string) {
    await api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    })
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

  async function createUser(username: string, password: string, role: string) {
    const res = await api.post('/auth/users', { username, password, role })
    await loadUsers()
    return res.data
  }

  async function updateUserRole(userId: number, role: string) {
    await api.put(`/auth/users/${userId}/role`, { role })
    await loadUsers()
  }

  async function deleteUser(userId: number) {
    await api.delete(`/auth/users/${userId}`)
    await loadUsers()
  }

  return {
    user, token, isAuthenticated, isAdmin, isAnalyst, userRole,
    users, usersLoading,
    login, fetchMe, logout, changePassword,
    loadUsers, createUser, updateUserRole, deleteUser,
  }
})
