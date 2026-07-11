import { useEffect, useMemo, useRef, useState } from 'react'
import { ChatPanel } from './components/ChatPanel'
import { Sidebar } from './components/Sidebar'
import { TravelDashboard } from './components/TravelDashboard'
import { WorkflowPanel } from './components/WorkflowPanel'
import { sendChatMessage, sendMultimodalMessage } from './services/api'

export default function App() {
  const [sessionId] = useState('default')
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        '你好，我是你的 AI 旅行规划 Agent。告诉我目的地、天数、预算、出发地和偏好，我会把路线、天气、住宿、餐厅和本地经验整理成一份可执行的旅行看板。',
    },
  ])
  const [loading, setLoading] = useState(false)
  const [state, setState] = useState({})
  const [error, setError] = useState('')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
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

  const canSend = useMemo(() => (message.trim().length > 0 || audioFile) && !loading, [message, audioFile, loading])
  const profile = state?.memory_context?.user_profile || state?.profile || {}
  const stats = {
    messageCount: Math.max(messages.length - 1, 0),
    toolCount: state?.tool_results?.items?.length || state?.research_tasks?.length || 0,
  }

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

  async function handleStartRecording() {
    if (loading || isRecording || !speechSupported) return
    try {
      setError('')
      if (audioPreview) URL.revokeObjectURL(audioPreview)
      setAudioPreview('')
      setAudioFile(null)
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

  function handleClearAudio() {
    if (audioPreview) URL.revokeObjectURL(audioPreview)
    setAudioPreview('')
    setAudioFile(null)
    setRecordingTime(0)
  }

  async function sendMessage() {
    if (!canSend) return
    const userMessage = message.trim()
    const hasAudio = Boolean(audioFile)
    const pendingAudioFile = audioFile
    const previewText = userMessage || '[语音消息]'

    setMessage('')
    setError('')
    setAudioFile(null)
    if (audioPreview) {
      URL.revokeObjectURL(audioPreview)
      setAudioPreview('')
    }
    setMessages((prev) => [...prev, { role: 'user', content: previewText }])
    setLoading(true)

    try {
      const data = hasAudio
        ? await sendMultimodalMessage({ message: userMessage, sessionId, audioFile: pendingAudioFile })
        : await sendChatMessage(userMessage, sessionId)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || data.final_answer || '暂时没有返回结果。',
        },
      ])
      setState(data)
    } catch (requestError) {
      const content = `请求失败：${requestError.message}`
      setMessages((prev) => [...prev, { role: 'assistant', content }])
      setError(content)
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
    <div className={`app-shell ${sidebarCollapsed ? 'app-shell--nav-collapsed' : ''}`}>
      <Sidebar
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed((prev) => !prev)}
        sessionId={sessionId}
        profile={profile}
        stats={stats}
      />

      <main className="main-layout">
        <div className="center-column">
          <WorkflowPanel loading={loading} state={state} />
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
            onClearAudio={handleClearAudio}
            audioTranscript={state?.audio_transcript || ''}
          />
        </div>

        <TravelDashboard state={state} loading={loading} />
      </main>
    </div>
  )
}

