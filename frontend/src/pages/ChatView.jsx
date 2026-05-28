import React, { useState, useRef, useEffect } from 'react'
import MarkdownOutline from '../components/MarkdownOutline'
import ReportView from '../components/ReportView'
import ChatMessage from '../components/ChatMessage'
import QueryInput from '../components/QueryInput'

// 从大纲树生成带占位符的报告骨架
// 占位符格式：<!--PH:指标名-->_加载中…_
// ChatView 收到 report_metric 事件后，用实际数据替换对应占位符
function buildSkeleton(tree) {
  if (!tree) return ''

  // 找最浅的 level（L1-L4），让它对应 H1，其余相对偏移
  let minLevel = Infinity
  function scanMin(nodes) {
    for (const node of nodes || []) {
      const lv = node.level || 1
      if (lv >= 1 && lv <= 4) minLevel = Math.min(minLevel, lv)
      scanMin(node.children)
    }
  }
  scanMin(tree.children || [])
  if (minLevel === Infinity) minLevel = 1

  const lines = []
  function walk(nodes) {
    for (const node of nodes || []) {
      const lv = node.level || 1
      const h = lv - minLevel + 1  // 相对标题层级，最小为 1
      if (lv >= 1 && lv <= 3) {
        lines.push('#'.repeat(h) + ' ' + node.name + '\n\n')
        if (node.description) lines.push(node.description + '\n\n')
        walk(node.children)
        if (node.summarySuggestion) lines.push('> 总结\n\n<span data-ph-summary="' + node.id + '" class="ph-spin"></span>\n\n')
      } else if (lv === 4) {
        lines.push('#'.repeat(h) + ' ' + node.name + '\n\n')
        if (node.description) lines.push(node.description + '\n\n')
        for (const q of (node.children || []).filter(c => c.level === 5)) {
          lines.push('**' + q.name + '**\n\n')
          lines.push('<span data-ph="' + q.name + '" class="ph-spin"></span>\n\n')
          if (q.summarySuggestion) lines.push('> 总结\n\n<span data-ph-summary="' + q.id + '" class="ph-spin"></span>\n\n')
        }
        if (node.summarySuggestion) lines.push('> 总结\n\n<span data-ph-summary="' + node.id + '" class="ph-spin"></span>\n\n')
        lines.push('---\n\n')
      } else if (lv === 5) {
        // L5 直接挂在根节点或 L1-L3 下，没有 L4 父节点
        lines.push('**' + node.name + '**\n\n')
        lines.push('<span data-ph="' + node.name + '" class="ph-spin"></span>\n\n')
        if (node.summarySuggestion) lines.push('> 总结\n\n<span data-ph-summary="' + node.id + '" class="ph-spin"></span>\n\n')
      }
    }
  }
  walk(tree.children || [])
  return lines.join('')
}

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
  const [report, setReport] = useState('')
  const [skeleton, setSkeleton] = useState('')
  const [reportTab, setReportTab] = useState('view') // 'view' | 'skeleton' | 'md'
  const [generatingReport, setGeneratingReport] = useState(false)
  const [chartData, setChartData] = useState({}) // name → {render_type, col_x, col_y, rows}
  const metricCacheRef = useRef({}) // name → chunk/placeholder，跨次生成缓存
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
    setSceneMeta(null)
    setQuickReplies([])
    setRightTab('outline')
    setReport('')
    setSkeleton('')
    setReportTab('view')
    metricCacheRef.current = {}
    setChartData({})

    fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
      .then(r => r.json())
      .then(d => { sessionIdRef.current = d.session_id })
      .catch(e => console.error('[ChatView] 创建 session 失败', e))
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

  async function generateReport() {
    if (!outlineJson || generatingReport) return
    setGeneratingReport(true)
    const sk = buildSkeleton(outlineJson)
    setSkeleton(sk)
    setReportTab('view')
    setRightTab('report')

    // 预填缓存，收集本次需要后端执行的 name
    const cache = metricCacheRef.current
    const allNames = [...sk.matchAll(/data-ph="([^"]+)"/g)].map(m => m[1])
    const cachedNames = allNames.filter(n => cache[n] !== undefined)
    let prefilled = sk
    for (const n of cachedNames) {
      prefilled = prefilled.replace(`<span data-ph="${n}" class="ph-spin"></span>`, cache[n])
    }
    setReport(prefilled)

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ outline_tree: outlineJson, cached_names: cachedNames }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        setReport(`**错误：** ${err.detail}`)
        return
      }

      const reader  = res.body.getReader()
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
          try {
            const evt = JSON.parse(raw)
            if (evt.type === 'report_metric') {
              const ph = '<span data-ph="' + evt.name + '" class="ph-spin"></span>'
              const CHART = new Set(['BAR', 'LINE', 'PIE'])
              if (CHART.has(evt.render_type) && evt.rows?.length) {
                const info = { render_type: evt.render_type, col_x: evt.col_x, col_y: evt.col_y, rows: evt.rows }
                setChartData(prev => ({ ...prev, [evt.name]: info }))
                const placeholder = `<div data-echart="${evt.name}"></div>\n\n`
                metricCacheRef.current[evt.name] = placeholder
                setReport(prev => prev.includes(ph) ? prev.replace(ph, placeholder) : prev)
              } else {
                const chunk = evt.chunk ?? ''
                metricCacheRef.current[evt.name] = chunk
                setReport(prev => prev.includes(ph) ? prev.replace(ph, chunk) : prev)
              }
            } else if (evt.type === 'report_summary') {
              const ph = '<span data-ph-summary="' + evt.node_id + '" class="ph-spin"></span>'
              const chunk = evt.chunk ?? ''
              setReport(prev => prev.includes(ph) ? prev.replace(ph, chunk) : prev)
            }
          } catch {}
        }
      }
    } catch (e) {
      setReport(`**错误：** ${e.message}`)
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
        if (evt.md_with_ids) setOutlineLlm(evt.md_with_ids)
        if (evt.outline_tree) setOutlineJson(evt.outline_tree)
        break
      }

      case 'confirm':
        setQuickReplies(evt.options || [])
        break

      case 'report':
        setReport(prev => prev + (evt.chunk ?? evt.content ?? ''))
        setRightTab('report')
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
      {/* 左：对话区 */}
      <div className="chat-panel">
        <div className="chat-panel__header">
          <div>
            <div className="chat-panel__title">看网分析</div>
            <div className="chat-panel__subtitle">分析传送网络现状，覆盖评估、容量分析、部署规划等</div>
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
              {t.key === 'report' && report && (
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
              {(report || skeleton) && (
                <div className="outline-tabs">
                  {[
                    { key: 'view',     label: '报告' },
                    { key: 'md',       label: 'Markdown' },
                    { key: 'skeleton', label: '骨架' },
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
                <ReportView markdown={report} generating={generatingReport} chartData={chartData} />
              )}
              {reportTab === 'md' && (
                <pre className="outline-raw">{report || '（暂无数据）'}</pre>
              )}
              {reportTab === 'skeleton' && (
                <pre className="outline-raw">{skeleton || '（暂无数据）'}</pre>
              )}
            </div>
          </>
        )}

      </div>
    </div>
  )
}
