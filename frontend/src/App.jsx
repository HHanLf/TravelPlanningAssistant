import { useEffect, useMemo, useRef, useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { Sidebar } from './components/Sidebar'
import { checkApiHealth, sendChatMessage, sendMultimodalMessage } from './services/api'

const INITIAL_ASSISTANT_MESSAGE =
  '嗨，我是你的旅行规划助手。告诉我目的地、天数、预算、出发地和偏好，我会帮你整理成一份可直接执行的行程。'

function createSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID()
  }
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function friendlyRequestError(error) {
  const message = error?.message || '请求失败'
  if (message === 'Failed to fetch' || message.includes('fetch')) {
    return '无法连接后端服务。请确认 FastAPI 已启动，或检查 Vite 代理目标是否指向 http://127.0.0.1:8000。'
  }
  return `请求失败：${message}`
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => createSessionId())
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([{ role: 'assistant', content: INITIAL_ASSISTANT_MESSAGE }])
  const [loading, setLoading] = useState(false)
  const [state, setState] = useState({})
  const [error, setError] = useState('')
  const [apiStatus, setApiStatus] = useState({ state: 'checking', message: '正在连接后端' })
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioFile, setAudioFile] = useState(null)
  const [audioPreview, setAudioPreview] = useState('')
  const [speechSupported, setSpeechSupported] = useState(false)

  const chatListRef = useRef(null)
  const textareaRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const recordingTimerRef = useRef(null)
  const streamRef = useRef(null)

  const profile = state?.memory_context?.user_profile || state?.profile || {}
  const currentDestination = state?.problem?.destination || profile.destination || ''
  const hasUserMessages = messages.some((item) => item.role === 'user')

  const stats = useMemo(
    () => ({
      messageCount: Math.max(messages.filter((item) => item.role === 'user').length, 0),
      toolCount: state?.tool_results?.items?.length || state?.research_tasks?.length || 0,
    }),
    [messages, state],
  )

  const canSend = useMemo(() => (message.trim().length > 0 || audioFile) && !loading, [message, audioFile, loading])

  useEffect(() => {
    let cancelled = false

    async function verifyBackend() {
      try {
        const data = await checkApiHealth()
        if (!cancelled) {
          setApiStatus({
            state: 'online',
            message: data.service ? `已连接 ${data.service}` : '后端已连接',
          })
        }
      } catch (healthError) {
        if (!cancelled) {
          setApiStatus({
            state: 'offline',
            message: friendlyRequestError(healthError),
          })
        }
      }
    }

    verifyBackend()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    const el = chatListRef.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [message])

  useEffect(() => {
    const hasMediaRecorder = typeof window !== 'undefined' && typeof window.MediaRecorder !== 'undefined'
    const hasDevices = typeof navigator !== 'undefined' && Boolean(navigator.mediaDevices?.getUserMedia)
    setSpeechSupported(hasMediaRecorder && hasDevices)
  }, [])

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current)
      streamRef.current?.getTracks?.().forEach((track) => track.stop())
      if (audioPreview) URL.revokeObjectURL(audioPreview)
    }
  }, [audioPreview])

  function resetRecordingState() {
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current)
      recordingTimerRef.current = null
    }
    streamRef.current?.getTracks?.().forEach((track) => track.stop())
    streamRef.current = null
    mediaRecorderRef.current = null
    audioChunksRef.current = []
    setIsRecording(false)
    setRecordingTime(0)
  }

  function clearComposerMedia() {
    if (audioPreview) URL.revokeObjectURL(audioPreview)
    setAudioPreview('')
    setAudioFile(null)
    setRecordingTime(0)
  }

  function resetConversation() {
    setSessionId(createSessionId())
    setMessage('')
    setLoading(false)
    setState({})
    setError('')
    clearComposerMedia()
    resetRecordingState()
    setMessages([{ role: 'assistant', content: INITIAL_ASSISTANT_MESSAGE }])
    setMobileSidebarOpen(false)
  }

  async function handleStartRecording() {
    if (loading || isRecording || !speechSupported) return
    try {
      setError('')
      clearComposerMedia()
      setRecordingTime(0)

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : 'audio/webm'
      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const finalMimeType = mediaRecorder.mimeType || 'audio/webm'
        const blob = new Blob(audioChunksRef.current, { type: finalMimeType })
        if (blob.size > 0) {
          const extension = finalMimeType.includes('ogg') ? 'ogg' : finalMimeType.includes('mp4') ? 'm4a' : 'webm'
          const file = new File([blob], `voice-message.${extension}`, { type: finalMimeType })
          setAudioFile(file)
          setAudioPreview(URL.createObjectURL(blob))
        }
        resetRecordingState()
      }

      mediaRecorder.start()
      setIsRecording(true)
      recordingTimerRef.current = setInterval(() => setRecordingTime((prev) => prev + 1), 1000)
    } catch (recordingError) {
      resetRecordingState()
      setError(`无法开始录音：${recordingError.message}`)
    }
  }

  function handleStopRecording() {
    if (!isRecording || !mediaRecorderRef.current) return
    mediaRecorderRef.current.stop()
  }

  async function sendMessage() {
    if (!canSend) return

    const userMessage = message.trim()
    const hasAudio = Boolean(audioFile)
    const pendingAudioFile = audioFile
    const previewText = userMessage || '语音消息'

    setMessage('')
    setError('')
    setAudioFile(null)
    if (audioPreview) {
      URL.revokeObjectURL(audioPreview)
      setAudioPreview('')
    }
    setMessages((prev) => [...prev, { role: 'user', content: previewText }])
    setLoading(true)
    setMobileSidebarOpen(false)

    try {
      const data = hasAudio
        ? await sendMultimodalMessage({ message: userMessage, sessionId, audioFile: pendingAudioFile })
        : await sendChatMessage(userMessage, sessionId)

      setApiStatus({
        state: 'online',
        message: '后端已连接',
      })
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || data.final_answer || '暂时没有返回结果。',
        },
      ])
      setState(data)
    } catch (requestError) {
      const content = friendlyRequestError(requestError)
      setMessages((prev) => [...prev, { role: 'assistant', content }])
      setError(content)
      setApiStatus({
        state: 'offline',
        message: content,
      })
    } finally {
      setLoading(false)
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      sendMessage()
    }
  }

  function handleUsePrompt(prompt) {
    setMessage(prompt)
    requestAnimationFrame(() => textareaRef.current?.focus())
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'app-shell--sidebar-collapsed' : ''} ${mobileSidebarOpen ? 'app-shell--sidebar-open' : ''}`}>
      <Sidebar
        collapsed={sidebarCollapsed}
        mobileOpen={mobileSidebarOpen}
        onToggleCollapse={() => setSidebarCollapsed((prev) => !prev)}
        onCloseMobile={() => setMobileSidebarOpen(false)}
        onNewChat={resetConversation}
        sessionId={sessionId}
        profile={profile}
        stats={stats}
        currentDestination={currentDestination}
        apiStatus={apiStatus}
      />

      {mobileSidebarOpen ? <button type="button" className="sidebar-backdrop" aria-label="关闭侧边栏" onClick={() => setMobileSidebarOpen(false)} /> : null}

      <main className="app-main">
        <ChatPanel
          messages={messages}
          loading={loading}
          message={message}
          setMessage={setMessage}
          canSend={canSend}
          onSend={sendMessage}
          onUsePrompt={handleUsePrompt}
          textareaRef={textareaRef}
          chatListRef={chatListRef}
          onKeyDown={handleComposerKeyDown}
          error={error}
          speechSupported={speechSupported}
          isRecording={isRecording}
          recordingTime={recordingTime}
          onStartRecording={handleStartRecording}
          onStopRecording={handleStopRecording}
          audioFile={audioFile}
          audioPreview={audioPreview}
          onClearAudio={clearComposerMedia}
          audioTranscript={state?.audio_transcript || ''}
          onOpenSidebar={() => setMobileSidebarOpen(true)}
          destination={currentDestination}
          hasUserMessages={hasUserMessages}
          conversationState={state}
          apiStatus={apiStatus}
        />
      </main>
    </div>
  )
}
