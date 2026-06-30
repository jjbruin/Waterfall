<script setup lang="ts">
import { ref, nextTick, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '../../api/client'
import { useDataStore } from '../../stores/data'
import { useDealsStore } from '../../stores/deals'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface ToolEvent {
  name: string
  input: Record<string, unknown>
}

const route = useRoute()
const dataStore = useDataStore()
const dealsStore = useDealsStore()

const isOpen = ref(false)
const isAvailable = ref(false)
const messages = ref<Message[]>([])
const inputText = ref('')
const isLoading = ref(false)
const toolActivity = ref<ToolEvent[]>([])
const chatBody = ref<HTMLElement | null>(null)
const suggestions = ref<string[]>([])

onMounted(async () => {
  try {
    const { data: status } = await api.get('/api/assistant/status')
    isAvailable.value = status.available
  } catch {
    isAvailable.value = false
  }
})

function getSuggestedQuestions(): string[] {
  const page = (route.name as string) || route.path
  const vcode = dealsStore.currentVcode || (route.query.vcode as string)
  const dealName = vcode ? dataStore.getDealName(vcode) : ''
  const label = dealName || 'this deal'

  if (page === 'dashboard' || route.path === '/') {
    return [
      'What is the total portfolio value?',
      'Which deals have the highest IRR?',
      'Show me all active deals',
    ]
  }
  if (page === 'deal-analysis' && vcode) {
    return [
      `What is the projected IRR for ${label}?`,
      `What are the expected sale proceeds?`,
      `Show me the capitalization stack`,
    ]
  }
  if (page === 'one-pager' && vcode) {
    return [
      `What is the current NOI for ${label}?`,
      `What is the PE exposure?`,
      `What is the DSCR?`,
    ]
  }
  if (page === 'property-financials' && vcode) {
    return [
      `Show me the income statement for ${label}`,
      `What is the occupancy trend?`,
      `Compare actual vs budget NOI`,
    ]
  }
  if (page === 'sold-portfolio') {
    return [
      'What are the sold portfolio returns?',
      'Which sold deal had the highest IRR?',
      'Show me the sold deal activity detail',
    ]
  }
  return [
    'List all active deals',
    'What is the portfolio summary?',
    'Compare two deals side-by-side',
  ]
}

function toggleChat() {
  isOpen.value = !isOpen.value
  if (isOpen.value && messages.value.length === 0) {
    suggestions.value = getSuggestedQuestions()
    messages.value.push({
      role: 'assistant',
      content: 'Hello! I\'m your AI assistant for the Waterfall app. I can help you look up deals, analyze returns, query financial data, and more. What would you like to know?',
    })
  }
}

function useSuggestion(text: string) {
  inputText.value = text
  suggestions.value = []
  sendMessage()
}

function scrollToBottom() {
  nextTick(() => {
    if (chatBody.value) {
      chatBody.value.scrollTop = chatBody.value.scrollHeight
    }
  })
}

function getPageContext(): Record<string, string> {
  const ctx: Record<string, string> = {}
  ctx.page = (route.name as string) || route.path
  ctx.path = route.path

  // Current deal from deals store (Deal Analysis, Property Financials, One Pager)
  const vcode = dealsStore.currentVcode
  if (vcode) {
    ctx.current_vcode = vcode
    ctx.current_deal_name = dataStore.getDealName(vcode)
  }

  // Query params that indicate deal/quarter selection
  if (route.query.vcode) ctx.current_vcode = route.query.vcode as string
  if (route.query.quarter) ctx.selected_quarter = route.query.quarter as string

  return ctx
}

async function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isLoading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  isLoading.value = true
  toolActivity.value = []
  suggestions.value = []
  scrollToBottom()

  // Build conversation for API (skip the initial greeting)
  const apiMessages = messages.value
    .filter((_, i) => i > 0)  // skip greeting
    .map(m => ({ role: m.role, content: m.content }))

  // Add empty assistant message to stream into
  messages.value.push({ role: 'assistant', content: '' })
  const assistantIdx = messages.value.length - 1

  try {
    const token = localStorage.getItem('token')
    const response = await fetch('/api/assistant/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({ messages: apiMessages, page_context: getPageContext() }),
    })

    if (!response.ok) {
      const err = await response.json()
      messages.value[assistantIdx].content = `Error: ${err.error || 'Request failed'}`
      isLoading.value = false
      return
    }

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const jsonStr = line.slice(6).trim()
        if (!jsonStr) continue

        try {
          const event = JSON.parse(jsonStr)

          if (event.type === 'text_delta') {
            messages.value[assistantIdx].content += event.text
            scrollToBottom()
          } else if (event.type === 'tool_use') {
            toolActivity.value.push({ name: event.name, input: event.input })
            scrollToBottom()
          } else if (event.type === 'error') {
            if (!messages.value[assistantIdx].content) {
              messages.value[assistantIdx].content = `Error: ${event.message}`
            }
          }
        } catch {
          // Skip malformed JSON
        }
      }
    }

    // Clean up empty assistant messages
    if (!messages.value[assistantIdx].content) {
      messages.value[assistantIdx].content = 'I completed the analysis but had no additional text to share.'
    }
  } catch (err: unknown) {
    const errMsg = err instanceof Error ? err.message : 'Unknown error'
    messages.value[assistantIdx].content = `Connection error: ${errMsg}`
  } finally {
    isLoading.value = false
    toolActivity.value = []
    scrollToBottom()
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    sendMessage()
  }
}

function clearChat() {
  messages.value = [{
    role: 'assistant',
    content: 'Chat cleared. How can I help you?',
  }]
  toolActivity.value = []
}

function formatToolName(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}
</script>

<template>
  <!-- Floating button -->
  <button
    v-if="isAvailable"
    class="ai-fab"
    :class="{ 'ai-fab--open': isOpen }"
    @click="toggleChat"
    title="AI Assistant"
  >
    <span v-if="!isOpen" class="ai-fab-icon">AI</span>
    <span v-else class="ai-fab-icon">&times;</span>
  </button>

  <!-- Chat panel -->
  <Transition name="slide">
    <div v-if="isOpen && isAvailable" class="ai-panel">
      <div class="ai-panel-header">
        <span class="ai-panel-title">AI Assistant</span>
        <button class="ai-clear-btn" @click="clearChat" title="Clear chat">Clear</button>
      </div>

      <div ref="chatBody" class="ai-panel-body">
        <div
          v-for="(msg, i) in messages"
          :key="i"
          class="ai-message"
          :class="msg.role === 'user' ? 'ai-message--user' : 'ai-message--assistant'"
        >
          <div class="ai-message-bubble" v-html="renderMarkdown(msg.content)" />
        </div>

        <!-- Suggested questions -->
        <div v-if="suggestions.length > 0 && !isLoading" class="ai-suggestions">
          <button
            v-for="(q, i) in suggestions"
            :key="i"
            class="ai-suggestion-chip"
            @click="useSuggestion(q)"
          >{{ q }}</button>
        </div>

        <!-- Tool activity indicator -->
        <div v-if="toolActivity.length > 0" class="ai-tool-activity">
          <div v-for="(tool, i) in toolActivity" :key="i" class="ai-tool-chip">
            <span class="ai-tool-spinner" /> {{ formatToolName(tool.name) }}
          </div>
        </div>

        <!-- Loading indicator -->
        <div v-if="isLoading && toolActivity.length === 0 && messages[messages.length - 1]?.content === ''" class="ai-typing">
          <span class="ai-typing-dot" /><span class="ai-typing-dot" /><span class="ai-typing-dot" />
        </div>
      </div>

      <div class="ai-panel-footer">
        <textarea
          v-model="inputText"
          class="ai-input"
          placeholder="Ask about deals, returns, data..."
          rows="2"
          :disabled="isLoading"
          @keydown="handleKeydown"
        />
        <button
          class="ai-send-btn"
          :disabled="isLoading || !inputText.trim()"
          @click="sendMessage"
        >
          Send
        </button>
      </div>
    </div>
  </Transition>
</template>

<script lang="ts">
// Simple markdown-to-HTML (bold, code, links, line breaks)
function renderMarkdown(text: string): string {
  if (!text) return ''
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>')
}
</script>

<style scoped>
.ai-fab {
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: var(--color-primary, #1F4E79);
  color: #fff;
  border: none;
  cursor: pointer;
  font-weight: 700;
  font-size: 14px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  z-index: 10000;
  transition: transform 0.15s, background 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.ai-fab:hover { transform: scale(1.08); }
.ai-fab--open { background: #666; }
.ai-fab-icon { line-height: 1; }

.ai-panel {
  position: fixed;
  bottom: 80px;
  right: 24px;
  width: 420px;
  max-height: 600px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.2);
  z-index: 9999;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.ai-panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--color-primary, #1F4E79);
  color: #fff;
}
.ai-panel-title { font-weight: 600; font-size: 14px; }
.ai-clear-btn {
  background: rgba(255,255,255,0.2);
  color: #fff;
  border: none;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}
.ai-clear-btn:hover { background: rgba(255,255,255,0.3); }

.ai-panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  min-height: 200px;
  max-height: 420px;
}

.ai-message { margin-bottom: 10px; display: flex; }
.ai-message--user { justify-content: flex-end; }
.ai-message--assistant { justify-content: flex-start; }

.ai-message-bubble {
  max-width: 85%;
  padding: 8px 12px;
  border-radius: 12px;
  font-size: 13px;
  line-height: 1.5;
  word-break: break-word;
}
.ai-message--user .ai-message-bubble {
  background: var(--color-primary, #1F4E79);
  color: #fff;
  border-bottom-right-radius: 4px;
}
.ai-message--assistant .ai-message-bubble {
  background: #f0f0f0;
  color: #333;
  border-bottom-left-radius: 4px;
}
.ai-message-bubble :deep(code) {
  background: rgba(0,0,0,0.08);
  padding: 1px 4px;
  border-radius: 3px;
  font-size: 12px;
}
.ai-message-bubble :deep(strong) { font-weight: 600; }

.ai-suggestions {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 8px 0;
}
.ai-suggestion-chip {
  background: #fff;
  color: var(--color-primary, #1F4E79);
  border: 1px solid var(--color-primary, #1F4E79);
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 12px;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s, color 0.15s;
}
.ai-suggestion-chip:hover {
  background: var(--color-primary, #1F4E79);
  color: #fff;
}

.ai-tool-activity {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin: 8px 0;
}
.ai-tool-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #e8f0fe;
  color: #1a73e8;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
}
.ai-tool-spinner {
  width: 10px;
  height: 10px;
  border: 2px solid #1a73e8;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

.ai-typing {
  display: flex;
  gap: 4px;
  padding: 8px 12px;
}
.ai-typing-dot {
  width: 6px;
  height: 6px;
  background: #999;
  border-radius: 50%;
  animation: bounce 1.2s infinite;
}
.ai-typing-dot:nth-child(2) { animation-delay: 0.2s; }
.ai-typing-dot:nth-child(3) { animation-delay: 0.4s; }

.ai-panel-footer {
  display: flex;
  gap: 8px;
  padding: 12px;
  border-top: 1px solid #eee;
  background: #fafafa;
}
.ai-input {
  flex: 1;
  resize: none;
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 8px 10px;
  font-size: 13px;
  font-family: inherit;
  outline: none;
}
.ai-input:focus { border-color: var(--color-primary, #1F4E79); }
.ai-send-btn {
  background: var(--color-primary, #1F4E79);
  color: #fff;
  border: none;
  padding: 8px 16px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  align-self: flex-end;
}
.ai-send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.ai-send-btn:hover:not(:disabled) { filter: brightness(1.1); }

/* Transitions */
.slide-enter-active, .slide-leave-active {
  transition: opacity 0.2s, transform 0.2s;
}
.slide-enter-from, .slide-leave-to {
  opacity: 0;
  transform: translateY(10px);
}

@keyframes spin { to { transform: rotate(360deg); } }
@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-4px); }
}

@media print { .ai-fab, .ai-panel { display: none !important; } }
@media (max-width: 480px) {
  .ai-panel { width: calc(100vw - 32px); right: 16px; bottom: 72px; }
}
</style>
