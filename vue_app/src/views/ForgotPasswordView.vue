<script setup lang="ts">
import { ref } from 'vue'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const email = ref('')
const loading = ref(false)
const error = ref('')
const success = ref('')

async function handleSubmit() {
  error.value = ''
  success.value = ''
  if (!email.value.trim()) {
    error.value = 'Please enter your email address'
    return
  }
  loading.value = true
  try {
    const msg = await auth.forgotPassword(email.value.trim())
    success.value = msg
  } catch (e: any) {
    error.value = e.response?.data?.error || 'Something went wrong. Please try again.'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-card">
    <h2>Waterfall XIRR</h2>
    <p class="subtitle">Reset your password</p>

    <template v-if="success">
      <div class="success-box">
        <p>{{ success }}</p>
        <p class="hint">Check your inbox and follow the link in the email.</p>
      </div>
      <router-link to="/login" class="btn-back-link">Back to Login</router-link>
    </template>

    <template v-else>
      <p class="info-text">Enter the email address associated with your account and we'll send you a link to reset your password.</p>

      <form @submit.prevent="handleSubmit">
        <div class="field">
          <label>Email Address</label>
          <input v-model="email" type="email" autocomplete="email" placeholder="you@peaceablestreet.com" required />
        </div>
        <p v-if="error" class="error">{{ error }}</p>
        <button type="submit" :disabled="loading" class="btn-login">
          {{ loading ? 'Sending...' : 'Send Reset Link' }}
        </button>
      </form>
      <div class="forgot-link">
        <router-link to="/login">Back to Login</router-link>
      </div>
    </template>
  </div>
</template>

<style scoped>
.login-card {
  background: var(--color-surface);
  padding: 40px;
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0, 0, 0, 0.1);
  width: 360px;
}
.login-card h2 { margin-bottom: 4px; color: var(--color-primary); }
.subtitle { color: var(--color-text-secondary); margin-bottom: 24px; font-size: 14px; }
.info-text { font-size: 13px; margin-bottom: 20px; color: var(--color-text-secondary); line-height: 1.5; }
.field { margin-bottom: 16px; }
.field label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 4px; }
.field input { width: 100%; padding: 8px 12px; border: 1px solid var(--color-border); border-radius: 6px; font-size: 14px; }
.error { color: #dc3545; font-size: 13px; margin-bottom: 12px; }
.btn-login { width: 100%; padding: 10px; background: var(--color-primary); color: white; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; }
.btn-login:hover { background: #163d5f; }
.btn-login:disabled { opacity: 0.6; cursor: not-allowed; }
.forgot-link { text-align: center; margin-top: 16px; font-size: 13px; }
.forgot-link a { color: var(--color-primary); text-decoration: none; }
.forgot-link a:hover { text-decoration: underline; }
.success-box { background: #d4edda; border: 1px solid #c3e6cb; border-radius: 6px; padding: 16px; margin-bottom: 16px; }
.success-box p { color: #155724; font-size: 14px; margin: 0; }
.success-box .hint { margin-top: 8px; font-size: 13px; color: #1b5e2b; }
.btn-back-link { display: block; text-align: center; padding: 10px; border: 1px solid var(--color-border); border-radius: 6px; color: var(--color-text-secondary); text-decoration: none; font-size: 13px; }
.btn-back-link:hover { background: #f5f5f5; }
</style>
