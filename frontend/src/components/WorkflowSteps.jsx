import React, { useState } from 'react'

export default function WorkflowSteps({ steps }) {
  const [collapsed, setCollapsed] = useState(false)
  if (!steps || steps.length === 0) return null

  const doneCount = steps.filter(s => s.status === 'done').length
  const allDone = doneCount === steps.length
  const hasError = steps.some(s => s.status === 'error')

  return (
    <div className={`wf-steps${allDone ? ' wf-steps--done' : ''}${hasError ? ' wf-steps--error' : ''}`}>
      <div className="wf-steps__header" onClick={() => setCollapsed(c => !c)}>
        <span className="wf-steps__status-icon">
          {hasError ? '✗' : allDone ? '✓' : <span className="wf-steps__spin" />}
        </span>
        <span className="wf-steps__title">
          {hasError
            ? '工作流出错'
            : allDone
            ? `已完成 ${steps.length} 个步骤`
            : `执行中 ${doneCount}/${steps.length}`}
        </span>
        <span className="wf-steps__toggle">{collapsed ? '▸' : '▾'}</span>
      </div>

      {!collapsed && (
        <div className="wf-steps__list">
          {steps.map(s => (
            <div key={s.step} className={`wf-step wf-step--${s.status}`}>
              <span className="wf-step__icon">
                {s.status === 'running' && <span className="wf-step__spin" />}
                {s.status === 'done' && '✓'}
                {s.status === 'error' && '✗'}
                {s.status === 'pending' && '·'}
              </span>
              <span className="wf-step__body">
                <span className="wf-step__name">{s.name}</span>
                {s.detail && <span className="wf-step__detail">{s.detail}</span>}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
