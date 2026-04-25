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

  // 更新最后一条 assistant 消息
  function updateLast(updater) {
    setMessages(prev => {
      const updated = [...prev]
      updated[updated.length - 1] = updater(updated[updated.length - 1])
      return updated
    })
  }

  async function send() {
    const text = input.trim()
    if (!text || streaming) return

    const userMsg = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setStreaming(true)
    // 添加空 assistant 消息占位
    setMessages(prev => [...prev, { role: 'assistant', content: '', steps: [] }])

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: parseInt(agentId), messages: newMessages }),
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        for (const line of decoder.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          if (raw === '[DONE]') break
          try {
            const evt = JSON.parse(raw)

            if (evt.type === 'step') {
              // 更新/追加步骤
              updateLast(msg => {
                const steps = [...(msg.steps || [])]
                const idx = steps.findIndex(s => s.step === evt.step)
                const entry = { step: evt.step, name: evt.name, status: evt.status, detail: evt.detail || '' }
                if (idx >= 0) steps[idx] = entry
                else steps.push(entry)
                return { ...msg, steps }
              })

            } else if (evt.type === 'text') {
              updateLast(msg => ({ ...msg, content: (msg.content || '') + evt.text }))

            } else if (evt.type === 'outline') {
              setOutline(evt.content)

            } else if (evt.type === 'error') {
              updateLast(msg => ({ ...msg, content: `请求失败: ${evt.error}` }))

            } else if (evt.text) {
              // 兼容旧格式 {"text": "..."}
              updateLast(msg => ({ ...msg, content: (msg.content || '') + evt.text }))
            }
          } catch {}
        }
      }
    } catch (e) {
      updateLast(msg => ({ ...msg, content: `请求失败: ${e.message}` }))
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
