import React, { useState } from 'react'

export default function TreeNode({ node, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)
  const hasChildren = node.children && node.children.length > 0
  const level = node.level || 1

  return (
    <div className={`tree-node level-${level}`}>
      <div className="tree-row" onClick={() => setOpen(o => !o)}>
        <span className="toggle">
          {hasChildren ? (open ? '▾' : '▸') : '·'}
        </span>
        <span className="level-badge">L{level}</span>
        <div className="node-body">
          <div className="node-name">{node.name}</div>
          {open && node.description && (
            <div className="node-desc">{node.description}</div>
          )}
          {open && node.keywords && node.keywords.length > 0 && (
            <div className="node-keywords">
              {node.keywords.map(k => (
                <span key={k} className="keyword-tag">{k}</span>
              ))}
            </div>
          )}
        </div>
      </div>
      {open && hasChildren && (
        <div className="tree-children">
          {node.children.map(child => (
            <TreeNode key={child.id} node={child} defaultOpen={false} />
          ))}
        </div>
      )}
    </div>
  )
}
