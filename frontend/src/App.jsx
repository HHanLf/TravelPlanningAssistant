import { useEffect, useMemo, useRef, useState } from 'react'
import { sendChatMessage, sendMultimodalMessage } from './services/api'

const QUICK_PROMPTS = [
  '帮我规划杭州 3 天 2 晚，预算 5000，喜欢自然风景',
  '从济南去北京玩 3 天，偏好历史建筑，预算 3000',
  '想去成都吃美食和逛博物馆，2 人 4 天怎么安排',
  '比较一下去上海旅游坐高铁还是飞机更合适',
]

function buildSections(data) {
  const plan = data?.plan || {}
  const intent = data?.intent || {}
  const toolResults = data?.tool_results || {}
  const reflection = data?.reflection_result || {}
  const memoryContext = data?.memory_context || {}
  const retrievedDocs = data?.retrieved_docs || []
  const answer = data?.answer || data?.final_answer || '暂时没有返回结果。'

  const sections = []
  sections.push({
    title: '本轮回复摘要',
    content: answer,
    tone: 'primary',
  })

  if (plan?.cards?.length) {
    sections.push({
      title: '推荐行程卡片',
      content: plan.cards
        .map((card) => {
          const lines = [card.title]
          if (card.subtitle) lines.push(card.subtitle)
          if (card.details?.length) lines.push(...card.details.map((detail) => `• ${detail}`))
          return lines.join('\n')
        })
        .join('\n\n'),
    })
  }

  if (intent && Object.keys(intent).length > 0) {
    sections.push({
      title: '识别到的意图',
      content: JSON.stringify(intent, null, 2),
      code: true,
    })
  }

  if (plan && Object.keys(plan).length > 0) {
    sections.push({
      title: '规划状态',
      content: JSON.stringify(plan, null, 2),
      code: true,
    })
  }

  if (retrievedDocs.length > 0) {
    sections.push({
      title: '检索资料',
      content: retrievedDocs
        .slice(0, 5)
        .map((doc, index) => `${index + 1}. ${doc.title || doc.content || '无标题'}`)
        .join('\n'),
    })
  }

  if (toolResults && Object.keys(toolResults).length > 0) {
    sections.push({
      title: '工具执行结果',
      content: JSON.stringify(toolResults, null, 2),
      code: true,
    })
  }

  if (reflection && Object.keys(reflection).length > 0) {
    sections.push({
      title: '预算与反思',
      content: JSON.stringify(reflection, null, 2),
      code: true,
    })
  }

  if (memoryContext && Object.keys(memoryContext).length > 0) {
    sections.push({
      title: '记忆上下文',
      content: JSON.stringify(memoryContext, null, 2),
      code: true,
    })
  }

  return sections
}

function SectionCard({ title, content, code = false, tone = 'default' }) {
  return (
    <section className={`insight-card ${tone}`}>
      <div className="insight-card__header">
        <h3>{title}</h3>
      </div>
      <div className={code ? 'insight-card__content insight-card__content--code' : 'insight-card__content'}>{content}</div>
    </section>
  )
}

function EmptyState({ onUsePrompt }) {
  return (
    <div className="empty-state">
      <div className="empty-state__badge">Travel Planning Assistant</div>
      <h1>今天想去哪里玩？</h1>
      <p>
        像和 ChatGPT 对话一样直接描述你的需求：目的地、天数、预算、人数、兴趣偏好，
        我会尽量给你结构化、可执行的旅行建议。
      </p>
      <div className="prompt-grid">
        {QUICK_PROMPTS.map((prompt) => (
          <button key={prompt} type="button" className="prompt-card" onClick={() => onUsePrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [sessionId] = useState('default')
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: '你好，我是你的智能旅游助手。告诉我目的地、预算、出发地、天数或偏好，我会帮你一起把行程想清楚。',
    },
  ])
  const [loading, setLoading] = useState(false)
  const [state, setState] = useState({})
  const [showInspector, setShowInspector] = useState(true)
  const [error, setError] = useState('')
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
  const sections = useMemo(() => buildSections(state), [state])
  const chatMessages = useMemo(() => messages.slice(1), [messages])

  useEffect(() => {
    const el = chatListRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
  }, [messages, loading])

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`
  }, [message])

  useEffect(() => {
    const hasMediaRecorder = typeof window !== 'undefined' && typeof window.MediaRecorder !== 'undefined'
    const hasDevices = typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
    setSpeechSupported(hasMediaRecorder && hasDevices)
  }, [])

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current)
      }
      streamRef.current?.getTracks?.().forEach((track) => track.stop())
    }
  }, [])

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
      if (audioPreview) {
        URL.revokeObjectURL(audioPreview)
      }
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
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
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
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
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
    if (audioPreview) {
      URL.revokeObjectURL(audioPreview)
    }
    setAudioPreview('')
    setAudioFile(null)
    setRecordingTime(0)
  }

  function formatRecordingTime(totalSeconds) {
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0')
    const seconds = String(totalSeconds % 60).padStart(2, '0')
    return `${minutes}:${seconds}`
  }

  async function sendMessage() {
    if (!canSend) return
    const userMessage = message.trim()
    const hasAudio = !!audioFile
    const previewText = userMessage || '[语音消息]'
    const pendingAudioFile = audioFile

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
        ? await sendMultimodalMessage({
            message: userMessage,
            sessionId,
            audioFile: pendingAudioFile,
          })
        : await sendChatMessage(userMessage, sessionId)
      setMessages((prev) => [...prev, { role: 'assistant', content: data.answer || data.final_answer || '暂时没有返回结果。' }])
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

  const lastIntent = state?.intent?.type || state?.intent?.task || state?.intent || '待识别'
  const profile = state?.memory_context?.user_profile || state?.profile || {}
  const audioTranscript = state?.audio_transcript || ''

  return (
    <div className="app-shell">
      <aside className="sidebar-shell">
        <div className="brand-block">
          <div className="brand-mark">TP</div>
          <div>
            <p className="brand-kicker">Travel Planning Assistant</p>
            <h2>智能旅行顾问</h2>
          </div>
        </div>

        <div className="sidebar-card sidebar-card--soft">
          <p className="sidebar-card__label">当前会话</p>
          <h3>默认会话</h3>
          <p>围绕预算、交通、景点偏好和多轮上下文来生成建议。</p>
        </div>

        <div className="sidebar-card">
          <p className="sidebar-card__label">建议输入</p>
          <ul className="sidebar-list">
            <li>目的地 / 出发地</li>
            <li>天数 / 人数 / 预算</li>
            <li>偏好：自然、美食、人文、亲子等</li>
            <li>是否需要详细行程或交通比较</li>
          </ul>
        </div>

        <div className="sidebar-card">
          <p className="sidebar-card__label">快速状态</p>
          <div className="status-chip-row">
            <span className="status-chip">意图：{String(lastIntent)}</span>
            <span className="status-chip">消息数：{chatMessages.length}</span>
          </div>
          <div className="profile-preview">
            {Object.keys(profile).length > 0 ? JSON.stringify(profile, null, 2) : '等待用户画像生成…'}
          </div>
        </div>
      </aside>

      <div className="workspace-shell">
        <header className="topbar">
          <div>
            <div className="topbar__title">旅行规划对话</div>
            <div className="topbar__subtitle">更接近 ChatGPT 的聊天体验，同时保留结构化结果查看能力</div>
          </div>
          <button type="button" className="ghost-button" onClick={() => setShowInspector((prev) => !prev)}>
            {showInspector ? '隐藏结构化面板' : '显示结构化面板'}
          </button>
        </header>

        <main className={`content-shell ${showInspector ? '' : 'content-shell--collapsed'}`}>
          <section className="chat-shell">
            <div className="chat-scroll" ref={chatListRef}>
              {chatMessages.length === 0 ? (
                <EmptyState onUsePrompt={handleUsePrompt} />
              ) : (
                <div className="message-stream">
                  {chatMessages.map((item, index) => (
                    <article key={`${item.role}-${index}`} className={`message-row message-row--${item.role}`}>
                      <div className="message-avatar">{item.role === 'assistant' ? 'AI' : '你'}</div>
                      <div className="message-card">
                        <div className="message-role">{item.role === 'assistant' ? '旅行助手' : '你'}</div>
                        <div className="message-text">{item.content}</div>
                      </div>
                    </article>
                  ))}
                  {loading && (
                    <article className="message-row message-row--assistant">
                      <div className="message-avatar">AI</div>
                      <div className="message-card message-card--loading">
                        <div className="message-role">旅行助手</div>
                        <div className="typing-indicator">
                          <span />
                          <span />
                          <span />
                        </div>
                      </div>
                    </article>
                  )}
                </div>
              )}
            </div>

            <div className="composer-shell">
              {error ? <div className="error-banner">{error}</div> : null}
              <div className="composer-card">
                <textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(event) => setMessage(event.target.value)}
                  onKeyDown={handleComposerKeyDown}
                  placeholder="发送消息给旅行助手，或先录一段语音，例如：从济南去北京，2个人，3天2晚，预算3000，想看历史建筑，帮我排详细行程"
                  rows={1}
                />

                {speechSupported ? (
                  <div className="voice-toolbar">
                    <button
                      type="button"
                      className={`voice-button ${isRecording ? 'voice-button--recording' : ''}`}
                      onClick={isRecording ? handleStopRecording : handleStartRecording}
                      disabled={loading}
                    >
                      {isRecording ? `停止录音 ${formatRecordingTime(recordingTime)}` : '开始语音输入'}
                    </button>

                    {audioFile ? (
                      <div className="audio-pill">
                        <span>已录制语音</span>
                        <button type="button" className="audio-pill__clear" onClick={handleClearAudio}>
                          清除
                        </button>
                      </div>
                    ) : (
                      <span className="voice-hint">支持直接录音后自动转文字参与对话</span>
                    )}
                  </div>
                ) : (
                  <div className="voice-hint voice-hint--warning">当前浏览器不支持录音，将只使用文本对话。</div>
                )}

                {audioPreview ? (
                  <div className="audio-preview-shell">
                    <audio controls src={audioPreview} className="audio-preview" />
                  </div>
                ) : null}

                {audioTranscript ? <div className="transcript-banner">最近一次语音识别：{audioTranscript}</div> : null}

                <div className="composer-footer">
                  <div className="composer-tips">Enter 发送，Shift + Enter 换行；可仅发送语音</div>
                  <button type="button" className="send-button" onClick={sendMessage} disabled={!canSend}>
                    {loading ? '生成中...' : '发送'}
                  </button>
                </div>
              </div>
            </div>
          </section>

          {showInspector ? (
            <aside className="inspector-shell">
              <div className="inspector-panel">
                <div className="inspector-panel__header">
                  <h2>结构化结果</h2>
                  <span>{sections.length} 项</span>
                </div>
                <div className="inspector-panel__body">
                  {sections.length > 0 ? sections.map((item) => <SectionCard key={item.title} {...item} />) : <div className="muted-block">等待生成结果…</div>}
                </div>
              </div>

              <div className="inspector-panel">
                <div className="inspector-panel__header">
                  <h2>完整状态</h2>
                  <span>Debug</span>
                </div>
                <pre className="state-viewer">{JSON.stringify(state, null, 2) || '{}'}</pre>
              </div>
            </aside>
          ) : null}
        </main>
      </div>
    </div>
  )
}
