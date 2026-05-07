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
  const [quickReplies, setQuickReplies] = useState([])
  const [outlineMd, setOutlineMd] = useState('')
  const [outlineLlm, setOutlineLlm] = useState('')
  const [outlineJson, setOutlineJson] = useState(null)
  const [outlineTab, setOutlineTab] = useState('md')  // 'md' | 'llm' | 'json'
  const sessionIdRef = useRef(null)
  const messagesEndRef = useRef(null)
  const assistantMsgIdxRef = useRef(-1)

  useEffect(() => {
    sessionIdRef.current = null
    setMessages([])
    setOutline('')
    setOutlineMd('')
    setOutlineLlm('')
    setOutlineJson(null)
    setOutlineTab('md')
    setQuickReplies([])

    fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ agent_id: parseInt(agentId) }),
    })
      .then(r => r.json())
      .then(d => { sessionIdRef.current = d.session_id })
      .catch(e => console.error('[ChatView] 创建 session 失败', e))
  }, [agentId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function updateAssistant(updater) {
    setMessages(prev => {
      const idx = assistantMsgIdxRef.current
      if (idx < 0 || idx >= prev.length) return prev
      const updated = [...prev]
      updated[idx] = updater(updated[idx])
      return updated
    })
  }

  function appendMsg(msg) {
    setMessages(prev => [...prev, msg])
  }

  async function sendText(text) {
    const t = text.trim()
    if (!t || streaming || !sessionIdRef.current) return

    setQuickReplies([])
    appendMsg({ role: 'user', content: t })
    setStreaming(true)
    setMessages(prev => {
      assistantMsgIdxRef.current = prev.length
      return [...prev, { role: 'assistant', content: '', steps: [] }]
    })

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionIdRef.current, message: t }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        updateAssistant(msg => ({ ...msg, content: `请求失败: ${err.detail}` }))
        setStreaming(false)
        return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const raw = line.slice(6)
          if (raw === '[DONE]') break
          try { handleEvent(JSON.parse(raw)) } catch {}
        }
      }
    } catch (e) {
      updateAssistant(msg => ({ ...msg, content: `请求失败: ${e.message}` }))
    }

    setStreaming(false)
  }

  async function send() {
    const text = input.trim()
    setInput('')
    await sendText(text)
  }

  function handleEvent(evt) {
    switch (evt.type) {
      case 'step':
        updateAssistant(msg => {
          const steps = [...(msg.steps || [])]
          const idx = steps.findIndex(s => s.name === evt.name)
          const existing = idx >= 0 ? steps[idx] : {}
          const entry = {
            ...existing,
            name: evt.name,
            status: evt.status,
            ...(evt.args !== undefined && { args: evt.args }),
            ...(evt.result !== undefined && { result: evt.result }),
          }
          if (idx >= 0) steps[idx] = entry
          else steps.push(entry)
          return { ...msg, steps }
        })
        break

      case 'text':
        updateAssistant(msg => ({
          ...msg,
          content: (msg.content || '') + (evt.chunk ?? evt.text ?? ''),
        }))
        break

      case 'outline': {
        const md = evt.markdown ?? evt.content ?? ''
        setOutline(md)
        setOutlineMd(md)
        if (evt.md_with_ids) setOutlineLlm(evt.md_with_ids)
        if (evt.outline_tree) setOutlineJson(evt.outline_tree)
        break
      }

      case 'confirm':
        setQuickReplies(evt.options || [])
        break

      case 'done':
        updateAssistant(msg => ({ ...msg, duration: evt.seconds }))
        break

      case 'extraction':
        appendMsg({
          role: 'extraction',
          scene_name: evt.scene_name,
          keywords: evt.keywords || [],
          summary: evt.summary || '',
        })
        break

case 'saved':
        appendMsg({ role: 'success', content: `模板已保存：${evt.scene_name}` })
        break

      case 'error':
        updateAssistant(msg => ({ ...msg, content: `[错误] ${evt.message}` }))
        break
    }
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

        {/* 快捷回复按钮（pending_confirm 时显示） */}
        {quickReplies.length > 0 && (
          <div className="quick-replies">
            {quickReplies.map(opt => (
              <button
                key={opt}
                className="quick-reply-btn"
                disabled={streaming}
                onClick={() => sendText(opt)}
              >
                {opt}
              </button>
            ))}
          </div>
        )}

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
          {outline && (
            <div className="outline-tabs">
              {[
                { key: 'md',   label: '用户' },
                { key: 'llm',  label: 'LLM' },
                { key: 'json', label: 'JSON' },
              ].map(tab => (
                <button
                  key={tab.key}
                  className={`outline-tab${outlineTab === tab.key ? ' outline-tab--active' : ''}`}
                  onClick={() => setOutlineTab(tab.key)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="outline-panel__body">
          {outlineTab === 'md' && <MarkdownOutline markdown={outlineMd} />}
          {outlineTab === 'llm' && (
            <pre className="outline-raw">{outlineLlm || '（暂无数据）'}</pre>
          )}
          {outlineTab === 'json' && (
            <pre className="outline-raw">
              {outlineJson ? JSON.stringify(outlineJson, null, 2) : '（暂无数据）'}
            </pre>
          )}
        </div>
      </div>
    </div>
  )
}
