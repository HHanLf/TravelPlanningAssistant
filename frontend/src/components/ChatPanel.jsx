import { AnimatePresence, motion } from 'framer-motion'
import { ArrowUp, Image, Mic, Pause, Sparkles, X } from 'lucide-react'

function parseInline(text) {
  const imageMatch = text.match(/^!\[(.*?)\]\((.*?)\)$/)
  if (imageMatch) {
    return <img className="markdown-image" src={imageMatch[2]} alt={imageMatch[1] || '旅行图片'} />
  }
  return text
}

function MarkdownLite({ content = '' }) {
  const lines = String(content).split('\n')
  const blocks = []
  let codeBuffer = []
  let inCode = false

  lines.forEach((line, index) => {
    if (line.trim().startsWith('```')) {
      if (inCode) {
        blocks.push(
          <pre className="code-block" key={`code-${index}`}>
            <code>{codeBuffer.join('\n')}</code>
          </pre>,
        )
        codeBuffer = []
      }
      inCode = !inCode
      return
    }
    if (inCode) {
      codeBuffer.push(line)
      return
    }
    if (!line.trim()) {
      blocks.push(<div className="markdown-space" key={`space-${index}`} />)
    } else if (/^#{1,3}\s/.test(line)) {
      blocks.push(
        <h3 className="markdown-title" key={`title-${index}`}>
          {line.replace(/^#{1,3}\s/, '')}
        </h3>,
      )
    } else if (/^\s*[-*]\s/.test(line)) {
      blocks.push(
        <div className="markdown-bullet" key={`bullet-${index}`}>
          <span />
          <p>{parseInline(line.replace(/^\s*[-*]\s/, ''))}</p>
        </div>,
      )
    } else if (line.includes('|') && line.trim().startsWith('|')) {
      blocks.push(
        <pre className="table-block" key={`table-${index}`}>
          {line}
        </pre>,
      )
    } else {
      blocks.push(<p key={`p-${index}`}>{parseInline(line)}</p>)
    }
  })

  return <div className="markdown-body">{blocks}</div>
}

function TypingIndicator() {
  return (
    <div className="typing-card">
      <div className="typing-dots">
        <span />
        <span />
        <span />
      </div>
      <p>Agent 正在研究路线、天气、住宿和本地经验</p>
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
}) {
  const quickPrompts = [
    '帮我规划杭州 3 天 2 晚，预算 5000，喜欢自然风景',
    '从济南去北京玩 3 天，偏好历史建筑，预算 3000',
    '想去成都吃美食和逛博物馆，2 人 4 天怎么安排',
    '比较一下去上海旅游坐高铁还是飞机更合适',
  ]

  return (
    <section className="chat-panel">
      <div className="chat-panel__header">
        <div>
          <p>AI Chat</p>
          <h1>旅行规划对话</h1>
        </div>
        <div className="live-pill">
          <Sparkles size={15} />
          <span>{loading ? 'Researching' : 'Ready'}</span>
        </div>
      </div>

      <div className="chat-scroll" ref={chatListRef}>
        {messages.length <= 1 ? (
          <div className="welcome-card">
            <div className="welcome-card__icon">
              <Sparkles size={24} />
            </div>
            <h2>今天想去哪里？</h2>
            <p>告诉我目的地、天数、预算、出发地和偏好，我会把聊天、研究、工具结果和行程看板一起整理出来。</p>
            <div className="prompt-grid">
              {quickPrompts.map((prompt) => (
                <button type="button" className="prompt-card" key={prompt} onClick={() => onUsePrompt(prompt)}>
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            <AnimatePresence initial={false}>
              {messages.slice(1).map((item, index) => (
                <motion.article
                  className={`message message--${item.role}`}
                  key={`${item.role}-${index}-${item.content.slice(0, 12)}`}
                  initial={{ opacity: 0, y: 14, scale: 0.98 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -8 }}
                  transition={{ duration: 0.22 }}
                >
                  <div className="message__avatar">{item.role === 'assistant' ? 'AI' : '你'}</div>
                  <div className="message__bubble">
                    <div className="message__meta">{item.role === 'assistant' ? 'Travel Agent' : 'Traveler'}</div>
                    <MarkdownLite content={item.content} />
                  </div>
                </motion.article>
              ))}
            </AnimatePresence>
            {loading && (
              <motion.article className="message message--assistant" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
                <div className="message__avatar">AI</div>
                <div className="message__bubble">
                  <TypingIndicator />
                </div>
              </motion.article>
            )}
          </div>
        )}
      </div>

      <div className="composer-wrap">
        {error ? <div className="error-banner">{error}</div> : null}
        <div className="composer">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={onKeyDown}
            placeholder="描述你的旅行需求，例如：从济南出发去北京 3 天 2 晚，预算 3000，喜欢历史建筑，少走路。"
            rows={1}
          />

          <div className="composer__tools">
            <button type="button" className="tool-button" title="图片展示预留">
              <Image size={17} />
            </button>
            {speechSupported && (
              <button
                type="button"
                className={`tool-button ${isRecording ? 'tool-button--recording' : ''}`}
                title={isRecording ? '停止录音' : '语音输入'}
                onClick={isRecording ? onStopRecording : onStartRecording}
              >
                {isRecording ? <Pause size={17} /> : <Mic size={17} />}
                {isRecording && <span>{formatTime(recordingTime)}</span>}
              </button>
            )}
            <button type="button" className="send-button" onClick={onSend} disabled={!canSend}>
              <ArrowUp size={18} />
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
      </div>
    </section>
  )
}

function formatTime(totalSeconds) {
  const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0')
  const seconds = String(totalSeconds % 60).padStart(2, '0')
  return `${minutes}:${seconds}`
}

