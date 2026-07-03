import React from 'react'

// 报告的排版、图表、目录序号全部由后端 temp_store.py 渲染进 report.html，
// 前端不再自己解析 Markdown/画图——backend/data/report/{id}/ 才是报告内容的
// 唯一权威来源，这里只是把已经生成好的页面嵌进来。
export default function ReportView({ sessionId, reportKey, ready, generating }) {
  if (generating) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">⏳</div>
        <div className="report-empty__text">报告生成中，请稍候…</div>
      </div>
    )
  }

  if (!ready || !sessionId) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">📄</div>
        <div className="report-empty__text">大纲确认后即可生成报告</div>
      </div>
    )
  }

  return (
    <iframe
      key={reportKey}
      src={`/api/session/${sessionId}/report?fmt=html&t=${reportKey}`}
      title="报告"
      style={{ width: '100%', height: '100%', border: 'none', background: '#fff' }}
    />
  )
}
