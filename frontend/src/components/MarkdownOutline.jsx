import React from 'react'

const LEVEL_COLORS = ['#6c5ce7', '#0984e3', '#00b894', '#e17055', '#636e72']

export default function MarkdownOutline({ markdown }) {
  if (!markdown) {
    return <div style={{ color: '#b2bec3', padding: 32, textAlign: 'center' }}>大纲将在对话中生成</div>
  }

  const lines = markdown.split('\n')
  const elements = []

  lines.forEach((line, i) => {
    const m = line.match(/^(#{1,5})\s+(.+)/)
    if (m) {
      const depth = m[1].length
      const text = m[2]
      const color = LEVEL_COLORS[depth - 1] || '#636e72'
      const [name, desc] = text.split(/：|:(.+)/).filter(Boolean)
      elements.push(
        <div key={i} style={{ marginLeft: (depth - 1) * 16, marginBottom: 6 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '1px 6px',
              borderRadius: 10, background: color, color: '#fff', flexShrink: 0,
            }}>H{depth}</span>
            <span style={{ fontSize: 14 - depth, fontWeight: depth <= 2 ? 700 : 500 }}>{name}</span>
          </div>
          {desc && <div style={{ marginLeft: 36, fontSize: 12, color: '#636e72', marginTop: 2 }}>{desc.replace(/^:/, '').trim()}</div>}
        </div>
      )
    } else if (line.trim()) {
      elements.push(
        <div key={i} style={{ fontSize: 13, color: '#636e72', marginBottom: 4, lineHeight: 1.6 }}>{line}</div>
      )
    }
  })

  return <div style={{ padding: '16px 20px' }}>{elements}</div>
}
