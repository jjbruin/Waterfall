<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const route = useRoute()
const auth = useAuthStore()

const token = ref('')
const username = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const loading = ref(false)
const validating = ref(true)
const error = ref('')
const success = ref(false)
const tokenInvalid = ref(false)

onMounted(async () => {
  token.value = (route.query.token as string) || ''
  if (!token.value) {
    tokenInvalid.value = true
    validating.value = false
    return
  }
  try {
    const res = await auth.validateResetToken(token.value)
    if (res.valid) {
      username.value = res.username
    } else {
      tokenInvalid.value = true
    }
  } catch {
    tokenInvalid.value = true
  } finally {
    validating.value = false
  }
})

async function handleReset() {
  error.value = ''
  if (newPassword.value !== confirmPassword.value) {
    error.value = 'Passwords do not match'
    return
  }
  if (newPassword.value.length < 8) {
    error.value = 'Password must be at least 8 characters'
    return
  }
  loading.value = true
  try {
    await auth.resetPassword(token.value, newPassword.value)
    success.value = true
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Failed to reset password'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-card">
    <h2>Waterfall XIRR</h2>

    <template v-if="validating">
      <p class="subtitle">Validating reset link...</p>
    </template>

    <template v-else-if="tokenInvalid">
      <p class="subtitle">Invalid Reset Link</p>
      <div class="error-box">
        <p>This password reset link is invalid or has expired.</p>
        <p>Please request a new reset link.</p>
      </div>
      <router-link to="/forgot-password" class="btn-action">Request New Link</router-link>
      <div class="forgot-link">
        <router-link to="/login">Back to Login</router-link>
      </div>
    </template>

    <template v-else-if="success">
      <p class="subtitle">Password Reset Successfully</p>
      <div class="success-box">
        <p>Your password has been changed. You can now log in with your new password.</p>
      </div>
      <router-link to="/login" class="btn-action">Log In</router-link>
    </template>

    <template v-else>
      <p class="subtitle">Set a new password</p>
      <p class="info-text">Hi <strong>{{ username }}</strong>, enter your new password below.</p>

      <form @submit.prevent="handleReset">
        <div class="field">
          <label>New Password</label>
          <input v-model="newPassword" type="password" autocomplete="new-password"
                 placeholder="At least 8 characters" required />
        </div>
        <div class="field">
          <label>Confirm New Password</label>
          <input v-model="confirmPassword" type="password" autocomplete="new-password" required />
        </div>
        <p v-if="error" class="error">{{ error }}</p>
        <button type="submit" :disabled="loading" class="btn-login">
          {{ loading ? 'Resetting...' : 'Reset Password' }}
        </button>
      </form>
      <div class="forgot-link">
        <router-link to="/login">Back to Login</router-link>
      </div>
    </template>
  </div>
</template>

<style scoped>
.login-card { background: var(--color-surface); padding: 40px; border-radius: 12px; box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1); width: 360px; }
.login-card h2 { margin-bottom: 4px; color: var(--color-primary); }
.subtitle { color: var(--color-text-secondary); margin-bottom: 24px; font-size: 14px; }
.info-text { font-size: 13px; margin-bottom: 20px; padding: 10px 12px; background: #f0f7ff; border-radius: 6px; border-left: 3px solid var(--color-primary); }
.field { margin-bottom: 16px; }
.field label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
.field input { width: 100%; padding: 8px 12px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 14px; }
.error { color: #dc3545; font-size: 13px; margin-bottom: 12px; }
.btn-login { width: 100%; padding: 10px; background: var(--color-primary); color: white; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; }
.btn-login:hover { background: #163d5f; }
.btn-login:disabled { opacity: 0.6; cursor: not-allowed; }
.btn-action { display: block; text-align: center; padding: 10px; background: var(--color-primary); color: white; border-radius: 6px; text-decoration: none; font-size: 14px; margin-bottom: 8px; }
.btn-action:hover { background: #163d5f; }
.forgot-link { text-align: center; margin-top: 16px; font-size: 13px; }
.forgot-link a { color: var(--color-primary); text-decoration: none; }
.forgot-link a:hover { text-decoration: underline; }
.success-box { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
.success-box p { color: #155724; font-size: 14px; margin: 0; }
.error-box { background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
.error-box p { color: #721c24; font-size: 14px; margin: 0 0 4px; }
.error-box p:last-child { margin-bottom: 0; }
</style>
