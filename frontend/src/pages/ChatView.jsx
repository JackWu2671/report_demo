import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import MarkdownOutline from '../components/MarkdownOutline'
import ChatMessage from '../components/ChatMessage'
import QueryInput from '../components/QueryInput'

const AGENT_NAMES = { '1': '专家知识沉淀', '2': '大纲对话生成' }
const AGENT_DESCS = {
  '1': '将专家描述的业务逻辑整理成结构化报告大纲，支持多轮对话迭代完善。',
  '2': '根据分析需求实时生成报告大纲，支持聚焦方向、删减章节、设置参数等修改。',
}

function extractOutline(text) {
  const idx = text.indexOf('---OUTLINE---')
  return idx >= 0 ? text.slice(idx + 13).trim() : ''
}

export default function ChatView() {
  const { agentId } = useParams()
  const navigate = useNavigate()
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [outline, setOutline] = useState('')
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send() {
    const text = input.trim()
    if (!text || streaming) return

    const userMsg = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setStreaming(true)
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: parseInt(agentId), messages: newMessages }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let full = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue
          const data = line.slice(6)
          if (data === '[DONE]') break
          try {
            const { text } = JSON.parse(data)
            if (text) {
              full += text
              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = { role: 'assistant', content: full }
                return updated
              })
              const ol = extractOutline(full)
              if (ol) setOutline(ol)
            }
          } catch {}
        }
      }
    } catch (e) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = { role: 'assistant', content: `请求失败: ${e.message}` }
        return updated
      })
    }

    setStreaming(false)
  }

  return (
    <div className="chat-view">
      {/* 左：对话区 */}
      <div className="chat-panel">
        <div className="chat-panel__header">
          <button className="chat-panel__back" onClick={() => navigate('/chat')}>
            <svg width="17" height="17" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/>
            </svg>
          </button>
          <div>
            <div className="chat-panel__title">{AGENT_NAMES[agentId]}</div>
            <div className="chat-panel__subtitle">{AGENT_DESCS[agentId]}</div>
          </div>
        </div>

        <div className="chat-panel__messages">
          {messages.length === 0 && (
            <div className="chat-empty">
              <div className="chat-empty__icon">💬</div>
              <div className="chat-empty__text">发送消息开始对话</div>
            </div>
          )}
          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              message={msg}
              isStreaming={streaming && i === messages.length - 1 && msg.role === 'assistant'}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <QueryInput
          value={input}
          onChange={setInput}
          onSend={send}
          disabled={streaming}
        />
      </div>

      {/* 右：大纲预览 */}
      <div className="outline-panel">
        <div className="outline-panel__header">
          <span className="outline-panel__title">大纲预览</span>
          {outline && <span className="outline-panel__badge">已生成</span>}
        </div>
        <div className="outline-panel__body">
          <MarkdownOutline markdown={outline} />
        </div>
      </div>
    </div>
  )
}
