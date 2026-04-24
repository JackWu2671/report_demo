import React from 'react'
import { useNavigate } from 'react-router-dom'

const SCENARIOS = [
  {
    id: 1,
    icon: '🧠',
    title: '专家知识沉淀',
    desc: '将专家描述的业务逻辑整理成结构化报告大纲，支持多轮对话迭代完善。',
  },
  {
    id: 2,
    icon: '📊',
    title: '大纲对话生成',
    desc: '根据分析需求实时生成报告大纲，支持聚焦方向、删减章节、设置参数等修改。',
  },
]

export default function ChatPage() {
  const navigate = useNavigate()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: 24 }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 8 }}>选择对话场景</h1>
      <div style={{ display: 'flex', gap: 24 }}>
        {SCENARIOS.map(s => (
          <button
            key={s.id}
            onClick={() => navigate(`/chat/${s.id}`)}
            style={{
              width: 260, padding: '32px 24px', borderRadius: 16,
              border: '2px solid #e8e8e8', background: '#fff',
              cursor: 'pointer', textAlign: 'left',
              transition: 'all 0.15s', boxShadow: 'none',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#0984e3'; e.currentTarget.style.boxShadow = '0 4px 16px rgba(9,132,227,0.15)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#e8e8e8'; e.currentTarget.style.boxShadow = 'none' }}
          >
            <div style={{ fontSize: 36, marginBottom: 12 }}>{s.icon}</div>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>{s.title}</div>
            <div style={{ fontSize: 13, color: '#636e72', lineHeight: 1.6 }}>{s.desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
