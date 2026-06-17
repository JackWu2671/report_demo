import React, { useState, useRef, useEffect } from 'react'
import MarkdownOutline from '../components/MarkdownOutline'
import ReportView from '../components/ReportView'
import ChatMessage from '../components/ChatMessage'
import QueryInput from '../components/QueryInput'

// 从大纲树生成带占位符的报告骨架
// 占位符格式：<!--PH:指标名-->_加载中…_
// ChatView 收到 report_metric 事件后，用实际数据替换对应占位符
// heading 层级由节点在树中的深度决定（depth=1 → h1），与节点的 level 字段无关，
// 保证同级兄弟节点无论 Lx 编号是否一致都渲染为相同 heading 层级。
function buildSkeleton(tree) {
  if (!tree) return ''

  const lines = []
  function walk(nodes, depth) {
    for (const node of nodes || []) {
      const lv = node.level || 1
      const h = Math.min(Math.max(1, depth), 6)
      if (lv === 5) {
        // L5 = 查询叶子节点，只渲染占位符
        lines.push('#'.repeat(h) + ' ' + node.name + '\n\n')
        lines.push('<span data-ph="' + node.name + '" class="ph-spin"></span>\n\n')
        if (node.summarySuggestion) lines.push('> 总结\n> \n> <span data-ph-summary="' + node.id + '" class="ph-spin"></span>\n\n')
      } else {
        // 结构节点（任意非 L5 层级）：递归处理所有子节点
        lines.push('#'.repeat(h) + ' ' + node.name + '\n\n')
        if (node.description) lines.push(node.description + '\n\n')
        walk(node.children, depth + 1)
        if (node.summarySuggestion) lines.push('> 总结\n> \n> <span data-ph-summary="' + node.id + '" class="ph-spin"></span>\n\n')
        // 只在叶子结构节点（子节点全为 L5 或无子节点）后加分隔线
        const hasStructuralChild = (node.children || []).some(c => (c.level || 1) !== 5)
        if (!hasStructuralChild) lines.push('---\n\n')
      }
    }
  }
  walk(tree.children || [], 1)
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
  const [reportTab, setReportTab] = useState('view') // 'view' | 'md'
  const [generatingReport, setGeneratingReport] = useState(false)
  const [chartData, setChartData] = useState({}) // name → {render_type, col_x, col_y, rows}
  const [tableData, setTableData] = useState({}) // name → rows[]
  const [conversations, setConversations] = useState([]) // 历史会话列表
  const [activeSession, setActiveSession] = useState(null) // 当前会话 id（仅用于列表高亮）
  const metricCacheRef = useRef({})   // name → { sig, value }，跨次生成缓存；sig 变化则失效
  const summaryCacheRef = useRef({})  // node_id → { key, chunk }，subtree 不变时复用
  const outlineJsonRef = useRef(null) // 同步镜像 outlineJson state，供事件回调同帧读取
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
    outlineJsonRef.current = null
    setOutlineTab('md')
    setSceneMeta(null)
    setQuickReplies([])
    setRightTab('outline')
    setReport('')
    setReportTab('view')
    metricCacheRef.current = {}
    summaryCacheRef.current = {}
    setChartData({})
    setTableData({})
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

  // 打开历史会话：恢复消息与大纲
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
        outlineJsonRef.current = d.outline_tree
        setOutlineJson(d.outline_tree)
        setOutlineMd(d.markdown || '')
        setOutlineLlm(d.outline_yaml || '')
      }
      if (d.extraction && d.extraction.scene_name) setSceneMeta(d.extraction)
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

  function collectSummaryNodes(nodes, acc = []) {
    for (const node of nodes || []) {
      if (node.summarySuggestion) acc.push(node)
      collectSummaryNodes(node.children, acc)
    }
    return acc
  }

  function findNodeById(nodes, id) {
    for (const node of nodes || []) {
      if (node.id === id) return node
      const found = findNodeById(node.children, id)
      if (found) return found
    }
    return null
  }

  function findNodeByName(nodes, name) {
    for (const node of nodes || []) {
      if (node.name === name) return node
      const found = findNodeByName(node.children, name)
      if (found) return found
    }
    return null
  }

  // metric 缓存签名：决定数据/渲染的字段，任一变化即缓存失效（如 exec_sql 改了）
  function metricSig(node) {
    if (!node) return ''
    return JSON.stringify({
      sql: node.exec_sql || '',
      rt:  node.renderType || '',
      x:   node.colX || '',
      y:   node.colY || '',
    })
  }

  function applySummaryChunk(str, nodeId, chunk) {
    const ph = '> <span data-ph-summary="' + nodeId + '" class="ph-spin"></span>'
    if (!str.includes(ph)) return str
    const replacement = chunk.trim().split('\n').map(l => '> ' + l).join('\n') + '\n'
    return str.replace(ph, replacement)
  }

  // 把 metric 缓存和 summary 缓存回填进 skeleton，返回已预填的报告字符串
  function replayCaches(sk, tree) {
    let out = sk
    const cache = metricCacheRef.current
    const treeChildren = (tree || outlineJson)?.children || []
    for (const [name, entry] of Object.entries(cache)) {
      // SQL 等签名变化时缓存失效，保留占位符让后端重新查询
      if (entry.sig !== metricSig(findNodeByName(treeChildren, name))) continue
      const ph = `<span data-ph="${name}" class="ph-spin"></span>`
      if (out.includes(ph)) out = out.replace(ph, entry.value)
    }
    for (const [nodeId, cached] of Object.entries(summaryCacheRef.current)) {
      const node = findNodeById((tree || outlineJson)?.children || [], nodeId)
      if (node && cached?.key === JSON.stringify(node)) {
        out = applySummaryChunk(out, nodeId, cached.chunk)
      }
    }
    return out
  }

  async function generateReport() {
    // 读 ref 而非 state：当 start_report 与 outline 事件在同一 SSE 批次内触发时，
    // React state 尚未刷新，但 ref 已同步更新
    const tree = outlineJsonRef.current
    if (!tree || generatingReport) return
    setGeneratingReport(true)
    const sk = buildSkeleton(tree)
    setReportTab('view')
    setRightTab('report')

    // 预填指标缓存：仅当签名（exec_sql 等）与当前节点一致才算命中，否则需重查
    const cache = metricCacheRef.current
    const allNames = [...sk.matchAll(/data-ph="([^"]+)"/g)].map(m => m[1])
    const cachedNames = allNames.filter(n => {
      const entry = cache[n]
      return entry !== undefined && entry.sig === metricSig(findNodeByName(tree.children || [], n))
    })

    // 预填 summary 缓存：子树 JSON 未变则直接填充，无需 LLM 重新生成
    const summaryNodes = collectSummaryNodes(tree.children || [])
    const cachedSummaryIds = []
    for (const node of summaryNodes) {
      const cached = summaryCacheRef.current[node.id]
      if (cached?.key === JSON.stringify(node)) {
        cachedSummaryIds.push(node.id)
      }
    }

    setReport(replayCaches(sk, tree))

    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionIdRef.current,
          outline_tree: tree,
          cached_names: cachedNames,
          cached_summary_ids: cachedSummaryIds,
        }),
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
              const sig = metricSig(findNodeByName(outlineJsonRef.current?.children || [], evt.name))
              const CHART = new Set(['BAR', 'LINE', 'PIE'])
              if (CHART.has(evt.render_type) && evt.rows?.length) {
                const info = { render_type: evt.render_type, col_x: evt.col_x, col_y: evt.col_y, rows: evt.rows }
                setChartData(prev => ({ ...prev, [evt.name]: info }))
                const placeholder = `<div data-echart="${evt.name}"></div>\n\n`
                metricCacheRef.current[evt.name] = { sig, value: placeholder }
                setReport(prev => prev.includes(ph) ? prev.replace(ph, placeholder) : prev)
              } else if (evt.render_type === 'TABLE' && evt.rows?.length) {
                setTableData(prev => ({ ...prev, [evt.name]: evt.rows }))
                const placeholder = `<div data-table="${evt.name}"></div>\n\n`
                metricCacheRef.current[evt.name] = { sig, value: placeholder }
                setReport(prev => prev.includes(ph) ? prev.replace(ph, placeholder) : prev)
              } else {
                const chunk = evt.chunk ?? ''
                metricCacheRef.current[evt.name] = { sig, value: chunk }
                setReport(prev => prev.includes(ph) ? prev.replace(ph, chunk) : prev)
              }
            } else if (evt.type === 'report_summary') {
              const span = '<span data-ph-summary="' + evt.node_id + '" class="ph-spin"></span>'
              const ph = '> ' + span  // span lives on a '> ' line inside the blockquote
              const chunk = (evt.chunk ?? '').trim()
              const replacement = chunk.split('\n').map(l => '> ' + l).join('\n') + '\n'
              setReport(prev => prev.includes(ph) ? prev.replace(ph, replacement) : prev)
              // 以当前大纲该节点的子树 JSON 为 key 写入缓存
              const node = findNodeById(outlineJson.children || [], evt.node_id)
              if (node) summaryCacheRef.current[evt.node_id] = { key: JSON.stringify(node), chunk: evt.chunk ?? '' }
            } else if (evt.type === 'report_skip') {
              appendMsg({ role: 'info', content: `「${evt.node_name}」不符合展示条件，已从报告中跳过。` })
            } else if (evt.type === 'outline') {
              // 条件跳过后，后端同步推送更新后的大纲，前端重建所有视图
              const newTree = evt.outline_tree
              outlineJsonRef.current = newTree
              setOutlineJson(newTree)
              setOutlineMd(evt.markdown || '')
              setOutlineLlm(evt.outline_yaml || '')
              const newSk = buildSkeleton(newTree)
              setReport(replayCaches(newSk, newTree))
            } else if (evt.type === 'report_done') {
              appendMsg({ role: 'success', content: '报告已生成完成，请查看右侧报告面板。' })
            }
          } catch (err) { console.warn('[SSE] 解析/处理报告事件失败:', err, 'raw=', raw) }
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
        if (evt.outline_yaml) setOutlineLlm(evt.outline_yaml)
        if (evt.outline_tree) {
          outlineJsonRef.current = evt.outline_tree
          setOutlineJson(evt.outline_tree)
        }
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
              {report && (
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
                <ReportView markdown={report} generating={generatingReport} chartData={chartData} tableData={tableData} />
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
