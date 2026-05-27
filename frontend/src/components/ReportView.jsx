import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function buildSkeleton(tree) {
  if (!tree) return ''
  const lines = []
  function walk(nodes) {
    for (const node of nodes || []) {
      const lv = node.level || 1
      if (lv >= 1 && lv <= 3) {
        lines.push('#'.repeat(lv) + ' ' + node.name + '\n\n')
        walk(node.children)
      } else if (lv === 4) {
        lines.push('#### ' + node.name + '\n\n')
        if (node.description) lines.push(node.description + '\n\n')
        lines.push('_数据加载中…_\n\n---\n\n')
      }
    }
  }
  walk(tree.children || [])
  return lines.join('')
}

export default function ReportView({ markdown, generating, outlineTree }) {
  if (!markdown && !generating) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">📄</div>
        <div className="report-empty__text">大纲确认后即可生成报告</div>
      </div>
    )
  }

  const content = markdown || buildSkeleton(outlineTree)

  return (
    <div className="report-body">
      {generating && !markdown && (
        <div style={{
          fontSize: 12, color: 'var(--color-text-placeholder)',
          display: 'flex', alignItems: 'center', gap: 6,
          marginBottom: 16, paddingBottom: 12,
          borderBottom: '1px solid var(--color-border)',
        }}>
          <span className="loading" style={{ width: 14, height: 14, margin: 0 }} />
          正在生成报告…
        </div>
      )}
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </div>
  )
}
