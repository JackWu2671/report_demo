import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function ReportView({ markdown }) {
  if (!markdown) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">📄</div>
        <div className="report-empty__text">大纲确认后即可生成报告</div>
      </div>
    )
  }

  return (
    <div className="report-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {markdown}
      </ReactMarkdown>
    </div>
  )
}
