import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function ReportView({ markdown, generating }) {
  if (!markdown && !generating) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">📄</div>
        <div className="report-empty__text">大纲确认后即可生成报告</div>
      </div>
    )
  }

  return (
    <div className="report-body">
      {generating && (
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
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
    </div>
  )
}
