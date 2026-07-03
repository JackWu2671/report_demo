import React, { useState, useRef, useEffect } from 'react'
import MarkdownOutline from '../components/MarkdownOutline'
import ReportView from '../components/ReportView'
import ChatMessage from '../components/ChatMessage'
import QueryInput from '../components/QueryInput'

export default function ChatView() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [outline, setOutline] = useState('')
  const [quickReplies, setQuickReplies] = useState([])
  const [outlineMd, setOutlineMd] = useState('')
  const [outlineLlm, setOutlineLlm] = useState('')
  const [outlineJson, setOutlineJson] = useState(null)
  const [outlineTab, setOutlineTab] = useState('md')  // 'md' | 'llm' | 'json'
  const [sceneMeta, setSceneMeta] = useState(null)   // {scene_name, summary, keywords, usage_conditions}
  const [rightTab, setRightTab] = useState('outline') // 'outline' | 'report'
  const [report, setReport] = useState('')           // 报告 Markdown 原文（fmt=md），只用于「Markdown」原文 tab
  const [reportReady, setReportReady] = useState(false) // 这个 session 是否已经生成过报告
  const [reportKey, setReportKey] = useState(0)       // 每次生成完 +1，强制报告 iframe 重新加载
  const [reportTab, setReportTab] = useState('view') // 'view' | 'md'
  const [generatingReport, setGeneratingReport] = useState(false)
  const [conversations, setConversations] = useState([]) // 历史会话列表
  const [activeSession, setActiveSession] = useState(null) // 当前会话 id（仅用于列表高亮）
  const sessionIdRef = useRef(null)
  const messagesEndRef = useRef(null)
  const assistantMsgIdxRef = useRef(-1)

  // 清空当前会话相关的本地状态（不含 session id）
  function resetLocalState() {
    setMessages([])
    setOutline('')
    setOutlineMd('')
    setOutlineLlm('')
    setOutlineJson(null)
    setOutlineTab('md')
    setSceneMeta(null)
    setQuickReplies([])
    setRightTab('outline')
    setReport('')
    setReportReady(false)
    setReportTab('view')
  }

  function refreshConversations() {
    fetch('/api/conversations')
      .then(r => r.json())
      .then(d => setConversations(d.conversations || []))
      .catch(e => console.error('[ChatView] 拉取历史会话失败', e))
  }

  // 新建对话：重置本地状态 + 创建新 session
  function startNewConversation() {
    if (streaming) return
    sessionIdRef.current = null
    setActiveSession(null)
    resetLocalState()
    fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then(r => r.json())
      .then(d => { sessionIdRef.current = d.session_id; setActiveSession(d.session_id) })
      .catch(e => console.error('[ChatView] 创建 session 失败', e))
  }

  // 打开历史会话：恢复消息、大纲，以及（如果生成过）报告
  async function openConversation(id) {
    if (streaming || id === activeSession) return
    try {
      const res = await fetch(`/api/conversations/${id}/open`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const d = await res.json()
      resetLocalState()
      sessionIdRef.current = d.session_id
      setActiveSession(d.session_id)
      setMessages(d.messages || [])
      if (d.outline_tree && Object.keys(d.outline_tree).length) {
        setOutlineJson(d.outline_tree)
        setOutlineMd(d.markdown || '')
        setOutlineLlm(d.outline_yaml || '')
      }
      if (d.extraction && d.extraction.scene_name) setSceneMeta(d.extraction)

      // 报告是否生成过，直接看 backend/data/report/{id}/ 里有没有 report.md——
      // 不在前端另外维护一份"是否已生成"的状态
      const reportRes = await fetch(`/api/session/${d.session_id}/report?fmt=md`)
      if (reportRes.ok) {
        setReport(await reportRes.text())
        setReportReady(true)
        setReportKey(k => k + 1)
      }
    } catch (e) {
      console.error('[ChatView] 打开历史会话失败', e)
    }
  }

  async function deleteConversation(id, e) {
    e.stopPropagation()
    if (!window.confirm('删除这条历史对话？')) return
    try {
      await fetch(`/api/conversations/${id}`, { method: 'DELETE' })
      if (id === activeSession) startNewConversation()
      refreshConversations()
    } catch (err) {
      console.error('[ChatView] 删除历史会话失败', err)
    }
  }

  useEffect(() => {
    startNewConversation()
    refreshConversations()
  }, [])

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
          try { handleEvent(JSON.parse(raw)) }
          catch (err) { console.warn('[SSE] 解析/处理聊天事件失败:', err, 'raw=', raw) }
        }
      }
    } catch (e) {
      updateAssistant(msg => ({ ...msg, content: `请求失败: ${e.message}` }))
    }

    setStreaming(false)
    refreshConversations()  // 一轮结束后刷新历史列表（标题/排序可能变化）
  }

  async function send() {
    const text = input.trim()
    setInput('')
    await sendText(text)
  }

  // 重新从 backend/data/report/{id}/ 拉取当前大纲——报告生成、条件跳过删节点等
  // 操作都会改这份文件，拉一次保证大纲面板跟刚生成的报告一致
  async function refreshOutline() {
    if (!sessionIdRef.current) return
    try {
      const res = await fetch(`/api/session/${sessionIdRef.current}/outline`)
      if (!res.ok) return
      const d = await res.json()
      setOutlineJson(d.outline_tree)
      setOutlineMd(d.markdown || '')
      setOutlineLlm(d.outline_yaml || '')
    } catch (e) {
      console.error('[ChatView] 刷新大纲失败', e)
    }
  }

  // 生成报告：只传 session_id，大纲从 backend/data/report/{id}/outline.json 读，
  // 生成什么、要不要重新生成完全由后端自己判断。生成完之前不做任何本地拼装/展示，
  // 结束后直接去 /api/session/{id}/report 取最终结果——backend/data/report/ 才是
  // 唯一权威数据源。
  async function generateReport() {
    if (!sessionIdRef.current || generatingReport) return
    setGeneratingReport(true)
    setReportTab('view')
    setRightTab('report')

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionIdRef.current }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        appendMsg({ role: 'error', content: `报告生成失败：${err.detail}` })
        return
      }
      const result = await res.json()
      if (result.skipped?.length) {
        const names = result.skipped.map(s => `「${s.node_name}」`).join('、')
        appendMsg({ role: 'info', content: `${names} 不符合展示条件，已从大纲中删除。` })
      }

      await refreshOutline()
      const mdRes = await fetch(`/api/session/${sessionIdRef.current}/report?fmt=md`)
      setReport(mdRes.ok ? await mdRes.text() : '')
      setReportReady(true)
      setReportKey(k => k + 1)
      appendMsg({ role: 'success', content: '报告已生成完成，请查看右侧报告面板。' })
    } catch (e) {
      appendMsg({ role: 'error', content: `报告生成失败：${e.message}` })
    } finally {
      setGeneratingReport(false)
    }
  }

  function handleEvent(evt) {
    switch (evt.type) {
      case 'step':
        updateAssistant(msg => {
          const steps = [...(msg.steps || [])]
          const idx = evt.call_id
            ? steps.findIndex(s => s.call_id === evt.call_id)
            : steps.findIndex(s => s.name === evt.name && s.status === 'running')
          const existing = idx >= 0 ? steps[idx] : {}
          const entry = {
            ...existing,
            name: evt.name,
            status: evt.status,
            ...(evt.call_id !== undefined && { call_id: evt.call_id }),
            ...(evt.args !== undefined && { args: evt.args }),
            ...(evt.result !== undefined && { result: evt.result }),
            ...(evt.detail !== undefined && { detail: evt.detail }),
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
        if (evt.outline_yaml) setOutlineLlm(evt.outline_yaml)
        if (evt.outline_tree) setOutlineJson(evt.outline_tree)
        break
      }

      case 'confirm':
        setQuickReplies(evt.options || [])
        break

      case 'start_report':
        generateReport()
        break

      case 'done':
        updateAssistant(msg => ({ ...msg, duration: evt.seconds }))
        break

      case 'metadata':
        setSceneMeta({
          scene_name: evt.scene_name || '',
          summary: evt.summary || '',
          keywords: evt.keywords || [],
          usage_conditions: evt.usage_conditions || '',
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
      {/* 最左：历史会话 */}
      <div className="conv-list">
        <button className="conv-new-btn" onClick={startNewConversation} disabled={streaming}>
          ＋ 新建对话
        </button>
        <div className="conv-items">
          {conversations.length === 0 && (
            <div className="conv-empty">暂无历史对话</div>
          )}
          {conversations.map(c => (
            <div
              key={c.session_id}
              className={`conv-item${c.session_id === activeSession ? ' conv-item--active' : ''}`}
              onClick={() => openConversation(c.session_id)}
              title={c.title}
            >
              <div className="conv-item__title">{c.title || '新对话'}</div>
              <div className="conv-item__meta">{(c.updated_at || '').replace('T', ' ')}</div>
              <button
                className="conv-item__del"
                onClick={(e) => deleteConversation(c.session_id, e)}
                title="删除"
              >×</button>
            </div>
          ))}
        </div>
      </div>

      {/* 中：对话区 */}
      <div className="chat-panel">
        <div className="chat-panel__header">
          <div>
            <div className="chat-panel__title">看网分析</div>
            <div className="chat-panel__subtitle">分析传送网络现状，覆盖评估、容量分析、部署规划等</div>
            {activeSession && (
              <div
                className="chat-panel__session-id"
                title={`点击复制完整 session_id: ${activeSession}`}
                onClick={() => navigator.clipboard?.writeText(activeSession)}
              >
                session: {activeSession.slice(0, 8)}
              </div>
            )}
          </div>
          {messages.length > 0 && sessionIdRef.current && (
            <button
              className="chat-panel__download"
              title="下载消息列表"
              onClick={async () => {
                const res = await fetch(`/api/session/${sessionIdRef.current}/messages`)
                const data = await res.json()
                const blob = new Blob([JSON.stringify(data.messages, null, 2)], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `messages_${sessionIdRef.current.slice(0, 8)}.json`
                a.click()
                URL.revokeObjectURL(url)
              }}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19 9h-4V3H9v6H5l7 7 7-7zm-8 2V5h2v6h1.17L12 13.17 9.83 11H11zm-6 8h14v2H5v-2z"/>
              </svg>
            </button>
          )}
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
          <div ref={messagesEndRef} />
        </div>

        {messages.length === 0 && (
          <div className="preset-questions">
            {['OLT现网评估分析', 'WiFi7升级怎么引导', '50GPON价值站点分析'].map(q => (
              <button
                key={q}
                className="preset-question-btn"
                onClick={() => setInput(q)}
              >
                {q}
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

      {/* 右：大纲 / 报告 */}
      <div className="outline-panel">

        {/* 顶层 Tab 栏 */}
        <div className="right-panel-tabs">
          {[
            { key: 'outline', label: '大纲' },
            { key: 'report',  label: '报告' },
          ].map(t => (
            <button
              key={t.key}
              className={`right-panel-tab${rightTab === t.key ? ' right-panel-tab--active' : ''}`}
              onClick={() => setRightTab(t.key)}
            >
              {t.label}
              {t.key === 'report' && reportReady && (
                <span className="right-panel-tab__dot" />
              )}
            </button>
          ))}
          {outlineJson && (
            <button
              className={`generate-report-btn${streaming ? ' generate-report-btn--disabled' : ''}`}
              disabled={streaming}
              onClick={generateReport}
            >
              {generatingReport ? '生成中…' : '生成报告'}
            </button>
          )}
        </div>

        {/* 大纲面板 */}
        {rightTab === 'outline' && (
          <>
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
            {sceneMeta && (
              <div className="scene-meta">
                <div className="scene-meta__row">
                  <span className="scene-meta__label">场景</span>
                  <span className="scene-meta__value">{sceneMeta.scene_name}</span>
                </div>
                <div className="scene-meta__row">
                  <span className="scene-meta__label">摘要</span>
                  <span className="scene-meta__value">{sceneMeta.summary}</span>
                </div>
                {sceneMeta.keywords.length > 0 && (
                  <div className="scene-meta__row">
                    <span className="scene-meta__label">关键词</span>
                    <span className="scene-meta__value">
                      {sceneMeta.keywords.map(k => (
                        <span key={k} className="scene-meta__tag">{k}</span>
                      ))}
                    </span>
                  </div>
                )}
                {sceneMeta.usage_conditions && (
                  <div className="scene-meta__row">
                    <span className="scene-meta__label">适用条件</span>
                    <span className="scene-meta__value">{sceneMeta.usage_conditions}</span>
                  </div>
                )}
              </div>
            )}
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
          </>
        )}

        {/* 报告面板 */}
        {rightTab === 'report' && (
          <>
            <div className="outline-panel__header">
              <span className="outline-panel__title">报告预览</span>
              {reportReady && (
                <div className="outline-tabs">
                  {[
                    { key: 'view', label: '报告' },
                    { key: 'md',   label: 'Markdown' },
                  ].map(tab => (
                    <button
                      key={tab.key}
                      className={`outline-tab${reportTab === tab.key ? ' outline-tab--active' : ''}`}
                      onClick={() => setReportTab(tab.key)}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="outline-panel__body">
              {reportTab === 'view' && (
                <ReportView
                  sessionId={sessionIdRef.current}
                  reportKey={reportKey}
                  ready={reportReady}
                  generating={generatingReport}
                />
              )}
              {reportTab === 'md' && (
                <pre className="outline-raw">{report || '（暂无数据）'}</pre>
              )}
            </div>
          </>
        )}

      </div>
    </div>
  )
}
