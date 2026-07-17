import { AnimatePresence, motion } from 'framer-motion'
import { AlertCircle, ArrowUp, Image, Loader2, Menu, Mic, Pause, Plus, Sparkles, X } from 'lucide-react'

const QUICK_PROMPTS = [
  '帮我规划 5 天东京亲子游，预算 8000 元',
  '预算 5000 元，适合情侣的云南路线',
  '第一次去欧洲，10 天怎么玩',
  '从上海出发，3 天周末海岛放松游',
]

function parseInline(text) {
  const imageMatch = text.match(/^!\[(.*?)\]\((.*?)\)$/)
  if (imageMatch) {
    return <img className="markdown-image" src={imageMatch[2]} alt={imageMatch[1] || '旅行图片'} />
  }
  return text
}

function isTableLine(line) {
  const trimmed = line.trim()
  return trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.includes('|')
}

function isSeparatorLine(line) {
  return /^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(line.trim())
}

function MarkdownTable({ rows }) {
  const parsedRows = rows
    .filter((row) => !isSeparatorLine(row))
    .map((row) =>
      row
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim()),
    )
    .filter((row) => row.some(Boolean))

  if (!parsedRows.length) {
    return null
  }

  const [head, ...body] = parsedRows
  return (
    <div className="markdown-table-wrap">
      <table className="markdown-table">
        <thead>
          <tr>
            {head.map((cell, index) => (
              <th key={`${cell}-${index}`}>{parseInline(cell)}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={`row-${rowIndex}`}>
              {head.map((_, cellIndex) => (
                <td key={`cell-${rowIndex}-${cellIndex}`}>{parseInline(row[cellIndex] || '')}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MarkdownLite({ content = '' }) {
  const lines = String(content).split('\n')
  const blocks = []
  let codeBuffer = []
  let inCode = false

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      if (inCode) {
        blocks.push(
          <pre className="code-block" key={`code-${index}`}>
            <code>{codeBuffer.join('\n')}</code>
          </pre>,
        )
        codeBuffer = []
      }
      inCode = !inCode
      continue
    }

    if (inCode) {
      codeBuffer.push(line)
      continue
    }

    if (!trimmed) {
      blocks.push(<div className="markdown-space" key={`space-${index}`} />)
      continue
    }

    if (/^#{1,3}\s/.test(trimmed)) {
      blocks.push(
        <h3 className="markdown-title" key={`title-${index}`}>
          {trimmed.replace(/^#{1,3}\s/, '')}
        </h3>,
      )
      continue
    }

    if (/^\s*[-*]\s/.test(line) || /^\s*\d+\.\s/.test(line)) {
      blocks.push(
        <div className="markdown-bullet" key={`bullet-${index}`}>
          <span />
          <p>{parseInline(line.replace(/^\s*([-*]|\d+\.)\s/, ''))}</p>
        </div>,
      )
      continue
    }

    if (isTableLine(line)) {
      const tableRows = []
      let nextIndex = index
      while (nextIndex < lines.length && isTableLine(lines[nextIndex])) {
        tableRows.push(lines[nextIndex])
        nextIndex += 1
      }
      blocks.push(<MarkdownTable rows={tableRows} key={`table-${index}`} />)
      index = nextIndex - 1
      continue
    }

    blocks.push(<p key={`p-${index}`}>{parseInline(line)}</p>)
  }

  if (inCode && codeBuffer.length) {
    blocks.push(
      <pre className="code-block" key="code-final">
        <code>{codeBuffer.join('\n')}</code>
      </pre>,
    )
  }

  return <div className="markdown-body">{blocks}</div>
}

function TypingIndicator() {
  return (
    <div className="typing-card">
      <div className="typing-dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p>正在整理路线、交通、住宿和预算建议</p>
    </div>
  )
}

function EmptyState({ onUsePrompt }) {
  return (
    <div className="empty-state">
      <h1>想去哪里旅行？</h1>
      <p>告诉我目的地、天数、预算和偏好，我来帮你规划。</p>
      <div className="prompt-grid" aria-label="旅行规划示例">
        {QUICK_PROMPTS.map((prompt) => (
          <button type="button" className="prompt-chip" key={prompt} onClick={() => onUsePrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  )
}

function TripFactChips({ state = {}, destination }) {
  const profile = state?.memory_context?.user_profile || state?.profile || {}
  const problem = state?.problem || {}
  const facts = [
    destination ? `目的地：${destination}` : '',
    problem.days || profile.days ? `${problem.days || profile.days} 天` : '天数待定',
    problem.budget || profile.budget ? `预算 ¥${problem.budget || profile.budget}` : '预算待定',
  ].filter(Boolean)

  return (
    <div className="trip-facts" aria-label="当前旅行要素">
      {facts.slice(0, 3).map((fact) => (
        <span key={fact}>{fact}</span>
      ))}
    </div>
  )
}

export function ChatPanel({
  messages,
  loading,
  message,
  setMessage,
  canSend,
  onSend,
  onUsePrompt,
  textareaRef,
  chatListRef,
  onKeyDown,
  error,
  speechSupported,
  isRecording,
  recordingTime,
  onStartRecording,
  onStopRecording,
  audioFile,
  audioPreview,
  onClearAudio,
  audioTranscript,
  onOpenSidebar,
  destination,
  hasUserMessages,
  conversationState,
  apiStatus,
}) {
  const visibleMessages = hasUserMessages ? messages : []

  return (
    <section className="chat-panel">
      <header className="chat-header">
        <button type="button" className="icon-button chat-header__menu" onClick={onOpenSidebar} aria-label="打开侧边栏">
          <Menu size={18} />
        </button>
        <div className="chat-header__copy">
          <button type="button" className="model-switch">
            Travel Planner
            <span>AI</span>
          </button>
          <TripFactChips state={conversationState} destination={destination} />
        </div>
        <div className={`status-pill status-pill--${apiStatus?.state || 'checking'} ${loading ? 'status-pill--loading' : ''}`}>
          {loading ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
          <span>{loading ? '规划中' : apiStatus?.state === 'online' ? '已连接' : apiStatus?.state === 'offline' ? '未连接' : '检查中'}</span>
        </div>
      </header>

      <div className="chat-scroll" ref={chatListRef}>
        {!hasUserMessages ? (
          <EmptyState onUsePrompt={onUsePrompt} />
        ) : (
          <div className="message-list">
            <AnimatePresence initial={false}>
              {visibleMessages.map((item, index) => (
                <motion.article
                  className={`message message--${item.role}`}
                  key={`${item.role}-${index}-${item.content.slice(0, 18)}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.18 }}
                >
                  <div className="message__avatar">{item.role === 'assistant' ? 'AI' : '你'}</div>
                  <div className="message__bubble">
                    <MarkdownLite content={item.content} />
                  </div>
                </motion.article>
              ))}
            </AnimatePresence>
            {loading ? (
              <motion.article className="message message--assistant" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="message__avatar">AI</div>
                <div className="message__bubble">
                  <TypingIndicator />
                </div>
              </motion.article>
            ) : null}
          </div>
        )}
      </div>

      <div className="composer-wrap">
        {error ? (
          <div className="error-banner" role="alert">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        ) : null}

        <div className={`composer ${message ? 'composer--active' : ''}`}>
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="告诉我你想去哪里、玩几天、预算多少、喜欢什么风格..."
            rows={1}
            aria-label="旅行规划需求"
          />

          <div className="composer__bottom">
            <div className="composer__tools">
              <button type="button" className="tool-button" title="添加内容" aria-label="添加内容">
                <Plus size={17} />
              </button>
              <button type="button" className="tool-button" title="图片功能预留" aria-label="图片功能预留">
                <Image size={17} />
              </button>
              {speechSupported ? (
                <button
                  type="button"
                  className={`tool-button ${isRecording ? 'tool-button--recording' : ''}`}
                  title={isRecording ? '停止录音' : '语音输入'}
                  aria-label={isRecording ? '停止录音' : '语音输入'}
                  onClick={isRecording ? onStopRecording : onStartRecording}
                >
                  {isRecording ? <Pause size={17} /> : <Mic size={17} />}
                  {isRecording ? <span>{formatTime(recordingTime)}</span> : null}
                </button>
              ) : null}
            </div>

            <button type="button" className="send-button" onClick={onSend} disabled={!canSend} aria-label="发送消息">
              {loading ? <Loader2 size={18} className="spin" /> : <ArrowUp size={18} />}
            </button>
          </div>

          {audioFile || audioPreview || audioTranscript ? (
            <div className="composer__attachments">
              {audioFile ? (
                <span className="attachment-pill">
                  语音已准备
                  <button type="button" onClick={onClearAudio} aria-label="清除语音">
                    <X size={13} />
                  </button>
                </span>
              ) : null}
              {audioPreview ? <audio controls src={audioPreview} /> : null}
              {audioTranscript ? <p>最近语音识别：{audioTranscript}</p> : null}
            </div>
          ) : null}
        </div>
        <p className="composer-hint">Enter 发送，Shift + Enter 换行。</p>
      </div>
    </section>
  )
}

function formatTime(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0')
  const seconds = String(totalSeconds % 60).padStart(2, '0')
  return `${minutes}:${seconds}`
}
