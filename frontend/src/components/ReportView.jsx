import React, { useMemo, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import ReactECharts from 'echarts-for-react'

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
  const minLevel = headings.length ? Math.min(...headings.map(h => h.level)) : 1
  const counters = [0, 0, 0, 0]
  return headings.map(h => {
    const idx = h.level - minLevel
    counters[idx]++
    for (let i = idx + 1; i < 4; i++) counters[i] = 0
    return { ...h, number: counters.slice(0, idx + 1).join('.') }
  })
}

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

const PAGE_SIZE = 20

function PaginatedTable({ rows }) {
  const [page, setPage] = useState(0)
  if (!rows || !rows.length) return null

  const headers = Object.keys(rows[0])
  const totalPages = Math.ceil(rows.length / PAGE_SIZE)
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div style={{ margin: '8px 0' }}>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 13 }}>
          <thead>
            <tr>
              {headers.map(h => (
                <th key={h} style={{
                  border: '1px solid var(--color-border)',
                  padding: '6px 12px',
                  background: 'var(--color-bg-secondary)',
                  textAlign: 'left',
                  whiteSpace: 'nowrap',
                  fontWeight: 600,
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr key={i} style={{ background: i % 2 === 0 ? 'var(--color-bg)' : 'var(--color-bg-secondary)' }}>
                {headers.map(h => (
                  <td key={h} style={{
                    border: '1px solid var(--color-border)',
                    padding: '5px 12px',
                    maxWidth: 320,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}>{String(row[h] ?? '')}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 8, fontSize: 12, color: 'var(--color-text-muted)' }}>
          <button
            onClick={() => setPage(p => Math.max(0, p - 1))}
            disabled={page === 0}
            style={{ padding: '3px 12px', cursor: page === 0 ? 'default' : 'pointer', opacity: page === 0 ? 0.4 : 1 }}
          >‹ 上一页</button>
          <span>第 {page + 1} / {totalPages} 页（共 {rows.length} 行）</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))}
            disabled={page === totalPages - 1}
            style={{ padding: '3px 12px', cursor: page === totalPages - 1 ? 'default' : 'pointer', opacity: page === totalPages - 1 ? 0.4 : 1 }}
          >下一页 ›</button>
        </div>
      )}
    </div>
  )
}

function buildChartOption(info) {
  const { render_type, col_x, col_y, rows } = info
  if (!rows || !rows.length) return null

  const t = (render_type || '').toUpperCase()

  if (t === 'PIE') {
    const nameKey = col_x || Object.keys(rows[0])[0]
    const valueKey = col_y || Object.keys(rows[0])[1]
    return {
      tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
      series: [{
        type: 'pie',
        radius: ['30%', '60%'],
        data: rows.map(r => ({ name: String(r[nameKey] ?? ''), value: r[valueKey] })),
        label: { formatter: '{b}\n{d}%' },
      }],
    }
  }

  // BAR / LINE
  const xKey = col_x || Object.keys(rows[0])[0]
  const yKey = col_y || Object.keys(rows[0])[1] || Object.keys(rows[0])[0]
  const xData = rows.map(r => String(r[xKey] ?? ''))
  const yData = rows.map(r => r[yKey])

  return {
    tooltip: { trigger: 'axis' },
    grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
    xAxis: {
      type: 'category',
      data: xData,
      axisLabel: { rotate: xData.length > 6 ? 30 : 0, overflow: 'truncate', width: 80 },
    },
    yAxis: { type: 'value' },
    series: [{
      type: t === 'LINE' ? 'line' : 'bar',
      data: yData,
      smooth: t === 'LINE',
    }],
  }
}

export default function ReportView({ markdown, generating, chartData = {}, tableData = {} }) {
  const headings = useMemo(() => extractHeadings(markdown), [markdown])

  // Refs keep components memo stable ([] deps) so ReactMarkdown never remounts
  // existing chart/table components when new ones arrive.
  const chartDataRef = useRef(chartData)
  chartDataRef.current = chartData
  const tableDataRef = useRef(tableData)
  tableDataRef.current = tableData

  const components = useMemo(() => ({
    ...HEADING_COMPONENTS,
    div: ({ node, children, ...props }) => {
      const echart = props['data-echart']
      if (echart) {
        const info = chartDataRef.current[echart]
        if (info) {
          const option = buildChartOption(info)
          if (option) {
            return (
              <div style={{ margin: '12px 0' }}>
                <ReactECharts key={echart} option={option} notMerge={true} style={{ height: 280 }} />
              </div>
            )
          }
        }
        // data not yet arrived — show a small placeholder
        return <div style={{ height: 60, display: 'flex', alignItems: 'center', paddingLeft: 4, color: 'var(--color-text-muted)', fontSize: 12 }}>图表加载中…</div>
      }
      const table = props['data-table']
      if (table && tableDataRef.current[table]) {
        return <PaginatedTable rows={tableDataRef.current[table]} />
      }
      return <div {...props}>{children}</div>
    },
  }), [])  // stable — never changes reference, refs provide latest data

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
              <span style={{ color: TOC_COLOR[h.level], marginRight: 5, fontVariantNumeric: 'tabular-nums', flexShrink: 0 }}>{h.number}</span>
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
            components={components}
          >
            {markdown}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  )
}
