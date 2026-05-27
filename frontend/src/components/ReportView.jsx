import React, { useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'

function slugify(text) {
  return text.replace(/\s+/g, '-').replace(/[^\w一-龥-]/g, '')
}

function extractHeadings(markdown) {
  if (!markdown) return []
  const headings = []
  for (const line of markdown.split('\n')) {
    const m = line.match(/^(#{1,4})\s+(.+)/)
    if (m) headings.push({ level: m[1].length, text: m[2].trim() })
  }
  return headings
}

// 自定义标题渲染，注入 id 供锚点跳转
function headingComponent(level) {
  const Tag = `h${level}`
  return function Heading({ children, ...props }) {
    const text = typeof children === 'string' ? children
      : Array.isArray(children) ? children.map(c => typeof c === 'string' ? c : '').join('') : ''
    const id = slugify(text)
    return <Tag id={id} {...props}>{children}</Tag>
  }
}

const HEADING_COMPONENTS = {
  h1: headingComponent(1),
  h2: headingComponent(2),
  h3: headingComponent(3),
  h4: headingComponent(4),
}

const INDENT = { 1: 0, 2: 10, 3: 18, 4: 26 }
const TOC_COLOR = { 1: '#6c5ce7', 2: '#2563eb', 3: '#00b894', 4: '#e17055' }

export default function ReportView({ markdown, generating }) {
  const headings = useMemo(() => extractHeadings(markdown), [markdown])

  if (!markdown && !generating) {
    return (
      <div className="report-empty">
        <div className="report-empty__icon">📄</div>
        <div className="report-empty__text">大纲确认后即可生成报告</div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden' }}>
      {/* 左：目录导航 */}
      {headings.length > 0 && (
        <div style={{
          width: 180, flexShrink: 0,
          borderRight: '1px solid var(--color-border)',
          overflowY: 'auto', padding: '12px 0',
          background: 'var(--color-bg-secondary)',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--color-text-muted)', padding: '0 12px 8px', letterSpacing: '0.05em' }}>
            目录
          </div>
          {headings.map((h, i) => (
            <a
              key={i}
              href={'#' + slugify(h.text)}
              onClick={e => {
                e.preventDefault()
                document.getElementById(slugify(h.text))?.scrollIntoView({ behavior: 'smooth' })
              }}
              style={{
                display: 'block',
                paddingLeft: 12 + INDENT[h.level],
                paddingRight: 8,
                paddingTop: 3,
                paddingBottom: 3,
                fontSize: h.level <= 2 ? 12 : 11,
                fontWeight: h.level <= 2 ? 600 : 400,
                color: h.level <= 2 ? TOC_COLOR[h.level] : 'var(--color-text-muted)',
                textDecoration: 'none',
                lineHeight: 1.4,
                borderLeft: `2px solid transparent`,
                transition: 'all 0.12s',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.color = TOC_COLOR[h.level] || 'var(--color-primary)'
                e.currentTarget.style.borderLeftColor = TOC_COLOR[h.level] || 'var(--color-primary)'
                e.currentTarget.style.background = 'var(--color-bg-tertiary)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color = h.level <= 2 ? TOC_COLOR[h.level] : 'var(--color-text-muted)'
                e.currentTarget.style.borderLeftColor = 'transparent'
                e.currentTarget.style.background = 'transparent'
              }}
            >
              {h.text}
            </a>
          ))}
        </div>
      )}

      {/* 右：报告正文 */}
      <div style={{ flex: 1, overflowY: 'auto', minWidth: 0 }}>
        <div className="report-body">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeRaw]}
            components={HEADING_COMPONENTS}
          >
            {markdown}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
