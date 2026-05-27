import React, { useEffect, useState } from 'react'

function buildNodeMap(nodes, relations) {
  const map = {}
  nodes.forEach(n => { map[n.id] = { ...n, children: [] } })
  const childIds = new Set(relations.map(r => r.child))
  relations.forEach(r => {
    if (map[r.parent] && map[r.child]) {
      map[r.parent].children.push(map[r.child])
    }
  })
  Object.values(map).forEach(n => {
    n.children.sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
  })
  return {
    map,
    roots: Object.values(map).filter(n => !childIds.has(n.id)),
  }
}

const COLS = [
  { label: 'L1', name: '场景',    color: '#6c5ce7' },
  { label: 'L2', name: '子场景',  color: '#2563eb' },
  { label: 'L3', name: '评估维度', color: '#00b894' },
  { label: 'L4', name: '评估项',  color: '#e17055' },
  { label: 'Q',  name: '评估指标', color: '#636e72' },
]

function cascadeSelect(map, colIdx, nodeId, prev) {
  const next = [...prev]
  next[colIdx] = nodeId
  let cur = map[nodeId]
  for (let i = colIdx + 1; i < 4; i++) {
    const first = cur?.children?.[0]
    next[i] = first?.id ?? null
    cur = first ? map[first.id] : null
  }
  return next
}

export default function KBPage() {
  const [kb, setKb] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState([null, null, null, null])

  useEffect(() => {
    fetch('/api/kb')
      .then(r => r.json())
      .then(data => {
        const kb = buildNodeMap(data.nodes, data.relations)
        setKb(kb)
        if (kb.roots.length > 0) {
          setSelected(cascadeSelect(kb.map, 0, kb.roots[0].id, [null, null, null, null]))
        }
      })
      .catch(e => setError(e.message))
  }, [])

  function handleSelect(colIdx, nodeId) {
    setSelected(cascadeSelect(kb.map, colIdx, nodeId, selected))
  }

  if (error) return <div className="empty">加载失败：{error}</div>
  if (!kb)   return <div className="loading">加载知识库...</div>

  const columns = [
    kb.roots,
    selected[0] ? (kb.map[selected[0]]?.children ?? []) : [],
    selected[1] ? (kb.map[selected[1]]?.children ?? []) : [],
    selected[2] ? (kb.map[selected[2]]?.children ?? []) : [],
    selected[3] ? (kb.map[selected[3]]?.children ?? []) : [],
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div style={{ padding: '16px 24px 12px', flexShrink: 0, borderBottom: '1px solid var(--color-border)' }}>
        <h1 className="page-title" style={{ marginBottom: 0 }}>
          知识库 · {Object.keys(kb.map).length} 个节点
        </h1>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {columns.map((nodes, colIdx) => {
          const col = COLS[colIdx]
          return (
            <div key={colIdx} style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              borderRight: colIdx < 4 ? '1px solid var(--color-border)' : 'none',
              overflow: 'hidden',
              minWidth: 0,
            }}>
              <div style={{
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                borderBottom: '1px solid var(--color-border)',
                background: 'var(--color-bg-secondary)',
                flexShrink: 0,
              }}>
                <span style={{
                  fontSize: 10, fontWeight: 700,
                  padding: '1px 6px', borderRadius: 4,
                  background: col.color, color: '#fff',
                }}>
                  {col.label}
                </span>
                <span style={{ fontSize: 12, color: 'var(--color-text-muted)', fontWeight: 500 }}>
                  {col.name}
                </span>
                <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--color-text-placeholder)' }}>
                  {nodes.length}
                </span>
              </div>

              <div style={{ flex: 1, overflowY: 'auto' }}>
                {nodes.length === 0
                  ? <div style={{ padding: 20, color: 'var(--color-text-placeholder)', fontSize: 12, textAlign: 'center' }}>—</div>
                  : nodes.map(node => {
                      const isActive   = colIdx < 4 && selected[colIdx] === node.id
                      const clickable  = colIdx < 4 && (node.children?.length ?? 0) > 0
                      return (
                        <KBNode
                          key={node.id}
                          node={node}
                          isActive={isActive}
                          clickable={clickable}
                          color={col.color}
                          onClick={() => clickable && handleSelect(colIdx, node.id)}
                        />
                      )
                    })
                }
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function KBNode({ node, isActive, clickable, color, onClick }) {
  const [hovered, setHovered] = useState(false)

  let bg = 'transparent'
  if (isActive)        bg = color + '12'
  else if (hovered && clickable) bg = 'var(--color-bg-tertiary)'

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding: '9px 12px',
        borderBottom: '1px solid var(--color-bg-tertiary)',
        borderLeft: isActive ? `3px solid ${color}` : '3px solid transparent',
        background: bg,
        cursor: clickable ? 'pointer' : 'default',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 6,
        transition: 'background 0.12s',
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{
          fontSize: 13,
          fontWeight: isActive ? 600 : 400,
          color: isActive ? color : 'var(--color-text)',
          lineHeight: 1.45,
          marginBottom: node.description ? 3 : 0,
        }}>
          {node.name}
        </div>
        {node.description && (
          <div style={{
            fontSize: 11,
            color: 'var(--color-text-placeholder)',
            lineHeight: 1.5,
            overflow: 'hidden',
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
          }}>
            {node.description}
          </div>
        )}
      </div>
      {isActive && clickable && (
        <span style={{ color, fontSize: 14, lineHeight: 1, marginTop: 2, opacity: 0.6, flexShrink: 0 }}>›</span>
      )}
    </div>
  )
}
