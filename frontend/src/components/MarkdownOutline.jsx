import React from 'react'

const COLORS = ['#6c5ce7', '#0984e3', '#00b894', '#e17055', '#636e72']
const LABELS = ['L1', 'L2', 'L3', 'L4', 'L5']

function parseMarkdown(markdown) {
  // Parse to_markdown() output: "# heading\n\ndescription" blocks
  const nodes = []
  let current = null

  for (const line of markdown.split('\n')) {
    const m = line.match(/^(#{1,6})\s+(.+)/)
    if (m) {
      if (current) nodes.push(current)
      current = { depth: m[1].length, name: m[2].trim(), desc: '' }
    } else if (current && line.trim()) {
      current.desc = current.desc
        ? current.desc + ' ' + line.trim()
        : line.trim()
    }
  }
  if (current) nodes.push(current)
  return nodes
}

export default function MarkdownOutline({ markdown }) {
  if (!markdown) {
    return (
      <div style={{ color: '#b2bec3', padding: 32, textAlign: 'center' }}>
        大纲将在对话中生成
      </div>
    )
  }

  const nodes = parseMarkdown(markdown)

  return (
    <div style={{ padding: '16px 20px' }}>
      {nodes.map((node, i) => {
        const color = COLORS[node.depth - 1] ?? '#636e72'
        const label = LABELS[node.depth - 1] ?? `H${node.depth}`
        const fontSize = Math.max(14 - node.depth, 12)
        const fontWeight = node.depth <= 2 ? 700 : node.depth <= 3 ? 600 : 500

        return (
          <div
            key={i}
            style={{ marginLeft: (node.depth - 1) * 18, marginBottom: 8 }}
          >
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 7 }}>
              <span style={{
                fontSize: 10, fontWeight: 700, padding: '1px 5px',
                borderRadius: 4, background: color, color: '#fff', flexShrink: 0,
              }}>
                {label}
              </span>
              <span style={{ fontSize, fontWeight, lineHeight: 1.4 }}>
                {node.name}
              </span>
            </div>
            {node.desc && (
              <div style={{
                marginLeft: 32, marginTop: 3,
                fontSize: 12, color: '#6b7280', lineHeight: 1.55,
              }}>
                {node.desc}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
