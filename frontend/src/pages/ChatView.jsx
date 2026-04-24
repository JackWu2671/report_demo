import React, { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import MarkdownOutline from '../components/MarkdownOutline'

const AGENT_NAMES = { '1': '专家知识沉淀', '2': '大纲对话生成' }

function extractOutline(text) {
  const idx = text.indexOf('---OUTLINE---')
  return idx >= 0 ? text.slice(idx + 13).trim() : ''
}

function extractChat(text) {
  const idx = text.indexOf('---OUTLINE---')
  return idx >= 0 ? text.slice(0, idx).trim() : text
}

export default function ChatView() {
  const { agentId } = useParams()
  const navigate = useNavigate()
  const [messages, setMessages] = useState([])   // [{role, content}]
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

    // placeholder for assistant
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
        const chunk = decoder.decode(value)
        for (const line of chunk.split('\n')) {
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
        updated[updated.length - 1] = { role: 'assistant', content: `❌ 请求失败: ${e.message}` }
        return updated
      })
    }

    setStreaming(false)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>

      {/* 左：对话区 */}
      <div style={{ width: '42%', display: 'flex', flexDirection: 'column', borderRight: '1px solid #e8e8e8' }}>
        {/* 顶栏 */}
        <div style={{ padding: '14px 20px', borderBottom: '1px solid #e8e8e8', display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={() => navigate('/chat')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#636e72', fontSize: 20 }}>←</button>
          <span style={{ fontWeight: 700 }}>{AGENT_NAMES[agentId]}</span>
        </div>

        {/* 消息列表 */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
          {messages.length === 0 && (
            <div style={{ color: '#b2bec3', textAlign: 'center', marginTop: 60, fontSize: 14 }}>
              发送消息开始对话
            </div>
          )}
          {messages.map((msg, i) => (
            <div key={i} style={{ marginBottom: 16, display: 'flex', flexDirection: 'column', alignItems: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
              <div style={{
                maxWidth: '85%', padding: '10px 14px', borderRadius: 12, fontSize: 14, lineHeight: 1.6, whiteSpace: 'pre-wrap',
                background: msg.role === 'user' ? '#0984e3' : '#f5f6fa',
                color: msg.role === 'user' ? '#fff' : '#2d3436',
              }}>
                {msg.role === 'assistant' ? extractChat(msg.content) || '…' : msg.content}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* 输入框 */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid #e8e8e8', display: 'flex', gap: 8 }}>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            disabled={streaming}
            style={{
              flex: 1, padding: '10px 12px', borderRadius: 10, border: '1px solid #e8e8e8',
              resize: 'none', fontSize: 14, height: 72, outline: 'none', fontFamily: 'inherit',
            }}
          />
          <button
            onClick={send}
            disabled={streaming || !input.trim()}
            style={{
              padding: '0 20px', borderRadius: 10, border: 'none',
              background: streaming || !input.trim() ? '#dfe6e9' : '#0984e3',
              color: '#fff', cursor: streaming || !input.trim() ? 'not-allowed' : 'pointer',
              fontSize: 14, fontWeight: 600,
            }}
          >
            {streaming ? '…' : '发送'}
          </button>
        </div>
      </div>

      {/* 右：大纲预览 */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div style={{ padding: '14px 20px', borderBottom: '1px solid #e8e8e8', fontWeight: 700 }}>
          大纲预览
        </div>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          <MarkdownOutline markdown={outline} />
        </div>
      </div>
    </div>
  )
}
