import React, { useState } from 'react'

function formatArgs(name, args) {
  if (!args || Object.keys(args).length === 0) return null
  if (args.question) return `"${args.question}"`
  if (args.anchor_id) return `anchor_id: ${args.anchor_id}`
  if (args.scene_name) return `scene_name: ${args.scene_name}`
  if (args.ops) {
    const types = args.ops.map(o => o.op).join(', ')
    return `ops (${args.ops.length}): ${types}`
  }
  return Object.entries(args).map(([k, v]) =>
    `${k}: ${typeof v === 'string' ? `"${v}"` : JSON.stringify(v)}`
  ).join(', ')
}

function StepRow({ s }) {
  const [open, setOpen] = useState(false)
  const hasDetail = s.args || s.result
  const argStr = formatArgs(s.name, s.args)

  return (
    <div className={`wf-step wf-step--${s.status}`}>
      <span className="wf-step__icon">
        {s.status === 'running' && <span className="wf-step__spin" />}
        {s.status === 'done' && '✓'}
        {s.status === 'error' && '✗'}
        {s.status === 'pending' && '·'}
      </span>
      <span className="wf-step__body">
        <span className="wf-step__name-row">
          <span className="wf-step__name">{s.name}</span>
          {hasDetail && (
            <button className="wf-step__toggle" onClick={() => setOpen(o => !o)}>
              {open ? '▴' : '▾'}
            </button>
          )}
        </span>
        {open && (
          <span className="wf-step__detail">
            {argStr && <span className="wf-step__detail-row"><span className="wf-step__detail-label">入参</span>{argStr}</span>}
            {s.result && <span className="wf-step__detail-row"><span className="wf-step__detail-label">返回</span>{s.result}</span>}
          </span>
        )}
      </span>
    </div>
  )
}

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
          {steps.map(s => <StepRow key={s.name} s={s} />)}
        </div>
      )}
    </div>
  )
}
