const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data.detail || data.message || '请求失败')
  }
  return data
}

export async function sendChatMessage(message, sessionId = 'default') {
  const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      session_id: sessionId,
    }),
  })

  return parseResponse(response)
}

export async function sendMultimodalMessage({ message, sessionId = 'default', audioFile }) {
  const formData = new FormData()
  formData.append('message', message || '')
  formData.append('session_id', sessionId)
  if (audioFile) {
    formData.append('audio', audioFile)
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/chat/multimodal`, {
    method: 'POST',
    body: formData,
  })

  return parseResponse(response)
}
