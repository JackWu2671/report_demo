import React, { useEffect, useState } from 'react'
import TreeNode from '../components/TreeNode'

function TemplateCard({ tpl }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="template-card">
      <div className={`card-header ${open ? 'open' : ''}`} onClick={() => setOpen(o => !o)}>
        <div className="card-title">{tpl.scene_name} {open ? '▾' : '▸'}</div>
        <div className="card-summary">{tpl.summary}</div>
        {tpl.keywords?.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {tpl.keywords.map(k => (
              <span key={k} className="keyword-tag">{k}</span>
            ))}
          </div>
        )}
        <div className="card-meta" style={{ marginTop: 8 }}>
          创建于 {tpl.created_at}
        </div>
      </div>

      {open && (
        <div className="card-body">
          {tpl.usage_conditions && (
            <>
              <div className="card-section-label">使用条件</div>
              <div className="usage-cond">{tpl.usage_conditions}</div>
            </>
          )}
          <div className="card-section-label">大纲结构</div>
          {tpl.outline ? (
            <TreeNode node={normalizeOutline(tpl.outline)} defaultOpen={true} />
          ) : (
            <div className="empty" style={{ padding: '20px 0' }}>无大纲数据</div>
          )}
        </div>
      )}
    </div>
  )
}

// template outline node may lack `level` field — derive from depth if needed
function normalizeOutline(node, depth = 2) {
  return {
    ...node,
    level: node.level ?? depth,
    children: (node.children || []).map(c => normalizeOutline(c, depth + 1))
  }
}

export default function TemplatePage() {
  const [templates, setTemplates] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/templates')
      .then(r => r.json())
      .then(setTemplates)
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="empty">加载失败：{error}</div>
  if (!templates) return <div className="loading">加载模板...</div>
  if (!templates.length) return <div className="empty">暂无已保存的大纲模板</div>

  return (
    <>
      <h1 className="page-title">大纲模板 · {templates.length} 个</h1>
      <div className="template-grid">
        {templates.map(t => <TemplateCard key={t.scene_name} tpl={t} />)}
      </div>
    </>
  )
}
