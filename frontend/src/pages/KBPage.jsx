import React, { useEffect, useState } from 'react'

function buildNodeMap(nodes, relations) {
  const map = {}
  nodes.forEach(n => { map[n.id] = { ...n, children: [] } })
  relations.forEach(r => {
    if (map[r.parent] && map[r.child]) {
      map[r.parent].children.push(map[r.child])
    }
  })
  Object.values(map).forEach(n => {
    n.children.sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
  })
  return map
}

const COLS = [
  { level: 1, label: 'L1', name: '场景',    color: '#6c5ce7' },
  { level: 2, label: 'L2', name: '子场景',  color: '#2563eb' },
  { level: 3, label: 'L3', name: '评估维度', color: '#00b894' },
  { level: 4, label: 'L4', name: '评估项',  color: '#e17055' },
  { level: 5, label: 'Q',  name: '评估指标', color: '#636e72' },
]

const LEVEL_COLOR = { 1: '#6c5ce7', 2: '#2563eb', 3: '#00b894', 4: '#e17055', 5: '#636e72' }

export default function KBPage() {
  const [map, setMap] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/kb')
      .then(r => r.json())
      .then(data => setMap(buildNodeMap(data.nodes, data.relations)))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="empty">加载失败：{error}</div>
  if (!map)  return <div className="loading">加载知识库...</div>

  const all = Object.values(map)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      <div style={{ padding: '16px 24px 12px', flexShrink: 0, borderBottom: '1px solid var(--color-border)' }}>
        <h1 className="page-title" style={{ marginBottom: 0 }}>
          知识库 · {all.length} 个节点
        </h1>
      </div>

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {COLS.map((col, idx) => {
          const nodes = all.filter(n => (n.level || 1) === col.level)
          return (
            <div key={col.level} style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              borderRight: idx < COLS.length - 1 ? '1px solid var(--color-border)' : 'none',
              overflow: 'hidden',
              minWidth: 0,
              background: 'var(--color-bg-secondary)',
            }}>
              <div style={{
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                borderBottom: '1px solid var(--color-border)',
                background: 'var(--color-bg)',
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

              <div style={{ flex: 1, overflowY: 'auto', padding: 8 }}>
                {nodes.length === 0
                  ? <div style={{ padding: 20, color: 'var(--color-text-placeholder)', fontSize: 12, textAlign: 'center' }}>—</div>
                  : nodes.map(node => <KBCard key={node.id} node={node} map={map} />)
                }
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function KBCard({ node, map }) {
  const [open, setOpen] = useState(false)
  const color = LEVEL_COLOR[node.level] || '#636e72'
  const hasChildren = node.children && node.children.length > 0
  const isQuery = node.level === 5
  const canExpand = hasChildren || (isQuery && node.exec_sql)

  return (
    <div style={{
      marginBottom: 8,
      border: '1px solid var(--color-border)',
      borderRadius: 'var(--radius-md)',
      background: 'var(--color-bg)',
      overflow: 'hidden',
    }}>
      <div
        onClick={() => canExpand && setOpen(o => !o)}
        style={{
          padding: '10px 12px',
          cursor: canExpand ? 'pointer' : 'default',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
          <span style={{
            fontSize: 9, fontWeight: 700, fontFamily: 'monospace',
            padding: '1px 5px', borderRadius: 3,
            background: color + '18', color,
          }}>
            {node.id}
          </span>
          {canExpand && (
            <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--color-text-placeholder)' }}>
              {open ? '▾' : '▸'} {hasChildren ? `${node.children.length} 子节点` : 'SQL'}
            </span>
          )}
        </div>

        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--color-text)', lineHeight: 1.45 }}>
          {node.name}
        </div>

        {node.description && (
          <div style={{ fontSize: 11, color: 'var(--color-text-muted)', lineHeight: 1.5, marginTop: 4 }}>
            {node.description}
          </div>
        )}

        {node.keywords && node.keywords.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {node.keywords.map(k => (
              <span key={k} className="keyword-tag">{k}</span>
            ))}
          </div>
        )}

        {node.summarySuggestion && (
          <div style={{
            marginTop: 6, fontSize: 11,
            padding: '4px 8px', borderRadius: 4,
            background: '#f0fdf4', color: '#15803d',
            border: '1px solid #bbf7d0',
          }}>
            <span style={{ opacity: 0.6, fontSize: 10, marginRight: 4 }}>总结</span>
            {node.summarySuggestion}
          </div>
        )}

        {node.condition && (
          <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 3 }}>
            <div style={{
              fontSize: 10, fontFamily: 'monospace',
              padding: '3px 7px', borderRadius: 3,
              background: '#fff7ed', color: '#c2410c',
              border: '1px solid #fed7aa',
              display: 'inline-flex', alignItems: 'center', gap: 4, alignSelf: 'flex-start',
            }}>
              <span style={{ opacity: 0.6 }}>@if</span> {node.condition}
            </div>
            {node.condition_queries && node.condition_queries.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
                {node.condition_queries.map(q => (
                  <span key={q} style={{
                    fontSize: 10, padding: '1px 6px', borderRadius: 3,
                    background: '#fff7ed', color: '#9a3412',
                    border: '1px dashed #fdba74',
                  }}>{q}</span>
                ))}
              </div>
            )}
          </div>
        )}

        {node.uuid && (
          <div style={{ fontSize: 9, color: 'var(--color-text-placeholder)', fontFamily: 'monospace', marginTop: 6 }}>
            {node.uuid}
          </div>
        )}
      </div>

      {open && hasChildren && (
        <div style={{ borderTop: '1px solid var(--color-border)', background: 'var(--color-bg-secondary)', padding: '6px 8px 6px 12px' }}>
          {node.children.map(child => (
            <SubTree key={child.id} node={map[child.id] || child} map={map} />
          ))}
        </div>
      )}

      {open && isQuery && node.exec_sql && (
        <div style={{ borderTop: '1px solid var(--color-border)', background: '#0d1117', padding: '10px 12px' }}>
          <div style={{ fontSize: 10, color: '#8b949e', marginBottom: 6, fontFamily: 'monospace', letterSpacing: '0.05em' }}>
            SQL
          </div>
          <pre style={{
            margin: 0,
            fontSize: 11,
            color: '#e6edf3',
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
            lineHeight: 1.6,
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}>
            {node.exec_sql}
          </pre>
        </div>
      )}
    </div>
  )
}

function SubTree({ node, map }) {
  const color = LEVEL_COLOR[node.level] || '#636e72'
  const hasChildren = node.children && node.children.length > 0

  return (
    <div style={{ borderLeft: `2px solid ${color}40`, paddingLeft: 8, marginBottom: 4 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 5, padding: '2px 0' }}>
        <span style={{
          fontSize: 8, fontWeight: 700, fontFamily: 'monospace',
          padding: '0 4px', borderRadius: 3,
          background: color + '18', color, flexShrink: 0,
        }}>
          {node.id}
        </span>
        <span style={{ fontSize: 12, color: 'var(--color-text)', lineHeight: 1.4 }}>
          {node.name}
        </span>
      </div>
      {node.description && (
        <div style={{ fontSize: 10, color: 'var(--color-text-placeholder)', lineHeight: 1.45, paddingLeft: 4, marginBottom: 2 }}>
          {node.description}
        </div>
      )}
      {hasChildren && node.children.map(child => (
        <SubTree key={child.id} node={map[child.id] || child} map={map} />
      ))}
    </div>
  )
}
