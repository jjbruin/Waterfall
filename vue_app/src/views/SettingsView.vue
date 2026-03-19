<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useAuthStore } from '../stores/auth'
import { useDataStore } from '../stores/data'
import DataTable from '../components/common/DataTable.vue'
import api from '../api/client'

const auth = useAuthStore()
const dataStore = useDataStore()

// Password change
const currentPw = ref('')
const newPw = ref('')
const confirmPw = ref('')

// New user form
const newUsername = ref('')
const newUserPw = ref('')
const newUserRole = ref('viewer')

onMounted(async () => {
  if (auth.isAdmin) {
    await auth.loadUsers()
    await loadReviewRoles()
  }
})

async function changePassword() {
  if (newPw.value !== confirmPw.value) {
    dataStore.addToast('Passwords do not match', 'error')
    return
  }
  try {
    await auth.changePassword(currentPw.value, newPw.value)
    dataStore.addToast('Password changed successfully', 'success')
    currentPw.value = ''
    newPw.value = ''
    confirmPw.value = ''
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to change password', 'error')
  }
}

async function createUser() {
  if (!newUsername.value.trim() || !newUserPw.value) {
    dataStore.addToast('Username and password required', 'error')
    return
  }
  try {
    await auth.createUser(newUsername.value.trim(), newUserPw.value, newUserRole.value)
    dataStore.addToast(`User '${newUsername.value}' created`, 'success')
    newUsername.value = ''
    newUserPw.value = ''
    newUserRole.value = 'viewer'
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to create user', 'error')
  }
}

async function updateRole(userId: number, role: string) {
  try {
    await auth.updateUserRole(userId, role)
    dataStore.addToast('Role updated', 'success')
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to update role', 'error')
  }
}

async function removeUser(userId: number, username: string) {
  if (!confirm(`Delete user '${username}'? This cannot be undone.`)) return
  try {
    await auth.deleteUser(userId)
    dataStore.addToast(`User '${username}' deleted`, 'success')
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to delete user', 'error')
  }
}

const roleOptions = ['viewer', 'analyst', 'admin']

// ── Review Role Management ──────────────────────────────────
interface ReviewRoleAssignment {
  id: number
  user_id: number
  username: string
  review_role: string
}

const reviewRoles = ref<ReviewRoleAssignment[]>([])
const availableReviewRoles = ref<string[]>([])
const reviewRolesLoading = ref(false)
const newReviewUserId = ref<number | null>(null)
const newReviewRole = ref('')

async function loadReviewRoles() {
  reviewRolesLoading.value = true
  try {
    const res = await api.get('/api/reviews/roles')
    reviewRoles.value = res.data.assignments || []
    availableReviewRoles.value = res.data.available_roles || []
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to load review roles', 'error')
  } finally {
    reviewRolesLoading.value = false
  }
}

async function addReviewRole() {
  if (!newReviewUserId.value || !newReviewRole.value) {
    dataStore.addToast('Select a user and role', 'error')
    return
  }
  try {
    await api.post('/api/reviews/roles', {
      user_id: newReviewUserId.value,
      review_role: newReviewRole.value,
    })
    dataStore.addToast('Review role assigned', 'success')
    newReviewUserId.value = null
    newReviewRole.value = ''
    await loadReviewRoles()
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to assign role', 'error')
  }
}

async function removeReviewRole(id: number) {
  try {
    await api.delete(`/api/reviews/roles/${id}`)
    dataStore.addToast('Review role removed', 'success')
    await loadReviewRoles()
  } catch (e: any) {
    dataStore.addToast(e.response?.data?.error || 'Failed to remove role', 'error')
  }
}

function formatReviewRole(role: string): string {
  return role.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}
</script>

<template>
  <div class="settings">
    <h2>Settings</h2>

    <!-- Current User Info -->
    <div class="section">
      <h3>Your Account</h3>
      <div class="info-row">
        <span class="info-label">Username:</span>
        <span class="info-value">{{ auth.user?.username }}</span>
      </div>
      <div class="info-row">
        <span class="info-label">Role:</span>
        <span class="role-badge" :class="auth.userRole">{{ auth.userRole }}</span>
      </div>
    </div>

    <!-- Change Password -->
    <div class="section">
      <h3>Change Password</h3>
      <div class="form-grid">
        <div class="form-row">
          <label>Current Password</label>
          <input type="password" v-model="currentPw" />
        </div>
        <div class="form-row">
          <label>New Password</label>
          <input type="password" v-model="newPw" />
        </div>
        <div class="form-row">
          <label>Confirm New Password</label>
          <input type="password" v-model="confirmPw" />
        </div>
        <button class="btn-action" @click="changePassword" :disabled="!currentPw || !newPw || !confirmPw">
          Change Password
        </button>
      </div>
    </div>

    <!-- User Management (admin only) -->
    <template v-if="auth.isAdmin">
      <div class="section">
        <h3>User Management</h3>

        <!-- Existing Users -->
        <div v-if="auth.usersLoading" class="placeholder">Loading users...</div>
        <table v-else-if="auth.users.length" class="users-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Username</th>
              <th>Role</th>
              <th>Created</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="u in auth.users" :key="u.id">
              <td>{{ u.id }}</td>
              <td>{{ u.username }}</td>
              <td>
                <select
                  :value="u.role"
                  @change="(e: any) => updateRole(u.id, e.target.value)"
                  :disabled="u.id === auth.user?.id"
                  class="role-select"
                >
                  <option v-for="r in roleOptions" :key="r" :value="r">{{ r }}</option>
                </select>
              </td>
              <td>{{ u.created_at?.slice(0, 10) || '—' }}</td>
              <td>
                <button
                  v-if="u.id !== auth.user?.id"
                  class="btn-delete"
                  @click="removeUser(u.id, u.username)"
                >
                  Delete
                </button>
                <span v-else class="self-tag">you</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Create New User -->
      <div class="section">
        <h3>Create New User</h3>
        <div class="form-grid">
          <div class="form-row">
            <label>Username</label>
            <input type="text" v-model="newUsername" placeholder="username" />
          </div>
          <div class="form-row">
            <label>Password</label>
            <input type="password" v-model="newUserPw" placeholder="password" />
          </div>
          <div class="form-row">
            <label>Role</label>
            <select v-model="newUserRole" class="role-select">
              <option v-for="r in roleOptions" :key="r" :value="r">{{ r }}</option>
            </select>
          </div>
          <button class="btn-action" @click="createUser" :disabled="!newUsername.trim() || !newUserPw">
            Create User
          </button>
        </div>
      </div>

      <!-- Role Descriptions -->
      <div class="section">
        <h3>Role Permissions</h3>
        <div class="role-desc">
          <div class="role-item">
            <span class="role-badge viewer">viewer</span>
            <span>Read-only access to all views and reports</span>
          </div>
          <div class="role-item">
            <span class="role-badge analyst">analyst</span>
            <span>Run computations, edit waterfalls, change report settings</span>
          </div>
          <div class="role-item">
            <span class="role-badge admin">admin</span>
            <span>Full access: manage users, import data, delete waterfalls, configure system</span>
          </div>
        </div>
      </div>

      <!-- Review Roles -->
      <div class="section">
        <h3>Review Roles (One Pager Approval)</h3>
        <p class="section-desc">Assign review workflow roles to users for the One Pager approval pipeline.</p>

        <!-- Current assignments -->
        <div v-if="reviewRolesLoading" class="placeholder">Loading review roles...</div>
        <table v-else-if="reviewRoles.length" class="users-table" style="margin-bottom: 16px;">
          <thead>
            <tr>
              <th>Username</th>
              <th>Review Role</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="rr in reviewRoles" :key="rr.id">
              <td>{{ rr.username }}</td>
              <td>
                <span class="review-role-badge">{{ formatReviewRole(rr.review_role) }}</span>
              </td>
              <td>
                <button class="btn-delete" @click="removeReviewRole(rr.id)">Remove</button>
              </td>
            </tr>
          </tbody>
        </table>
        <p v-else class="placeholder">No review roles assigned yet.</p>

        <!-- Add new assignment -->
        <div class="form-grid">
          <div class="form-row">
            <label>User</label>
            <select v-model="newReviewUserId" class="role-select">
              <option :value="null">-- Select user --</option>
              <option v-for="u in auth.users" :key="u.id" :value="u.id">{{ u.username }}</option>
            </select>
          </div>
          <div class="form-row">
            <label>Review Role</label>
            <select v-model="newReviewRole" class="role-select">
              <option value="">-- Select role --</option>
              <option v-for="r in availableReviewRoles" :key="r" :value="r">{{ formatReviewRole(r) }}</option>
            </select>
          </div>
          <button class="btn-action" @click="addReviewRole" :disabled="!newReviewUserId || !newReviewRole">
            Assign Review Role
          </button>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.settings { padding: 0 0 40px 0; }
h2 { font-size: 20px; margin-bottom: 16px; }
h3 { font-size: 15px; margin: 0 0 12px 0; }

.section {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 16px;
}

.info-row {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
  align-items: center;
}

.info-label { font-size: 13px; color: var(--color-text-secondary); min-width: 80px; }
.info-value { font-size: 14px; font-weight: 500; }

.form-grid { display: flex; flex-direction: column; gap: 10px; }

.form-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.form-row label {
  font-size: 13px;
  min-width: 140px;
  color: var(--color-text-secondary);
}

.form-row input, .form-row select {
  flex: 1;
  padding: 6px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 13px;
}

.btn-action {
  align-self: flex-start;
  padding: 8px 20px;
  background: var(--color-accent);
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
}
.btn-action:hover { opacity: 0.9; }
.btn-action:disabled { opacity: 0.5; cursor: not-allowed; }

/* Users table */
.users-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.users-table th {
  padding: 8px 12px;
  background: var(--color-accent);
  color: white;
  font-weight: 600;
  text-align: left;
}

.users-table td {
  padding: 6px 12px;
  border-bottom: 1px solid var(--color-border);
}

.role-select {
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  font-size: 12px;
}

.btn-delete {
  padding: 2px 10px;
  background: none;
  color: #c62828;
  border: 1px solid #c62828;
  border-radius: 4px;
  cursor: pointer;
  font-size: 11px;
}
.btn-delete:hover { background: #ffebee; }

.self-tag {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-style: italic;
}

.placeholder {
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 16px 0;
  text-align: center;
}

/* Role badges */
.role-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.role-badge.viewer { background: #e3f2fd; color: #1565c0; }
.role-badge.analyst { background: #e8f5e9; color: #2e7d32; }
.role-badge.admin { background: #fce4ec; color: #c62828; }

.role-desc { display: flex; flex-direction: column; gap: 8px; }

.role-item {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
}

.role-item .role-badge { min-width: 70px; text-align: center; }

.section-desc {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin: -8px 0 12px 0;
}

.review-role-badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  background: #e8eaf6;
  color: #283593;
}
</style>
