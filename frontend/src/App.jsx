import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import KBPage from './pages/KBPage'
import TemplatePage from './pages/TemplatePage'
import ChatPage from './pages/ChatPage'
import ChatView from './pages/ChatView'

export default function App() {
  return (
    <>
      <nav className="sidebar">
        <h2>报告系统</h2>
        <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
          📚 知识库
        </NavLink>
        <NavLink to="/templates" className={({ isActive }) => isActive ? 'active' : ''}>
          📄 大纲模板
        </NavLink>
        <NavLink to="/chat" className={({ isActive }) => isActive ? 'active' : ''}>
          💬 对话生成
        </NavLink>
      </nav>
      <main className="main" style={{ padding: 0, overflow: 'hidden' }}>
        <Routes>
          <Route path="/" element={<div style={{ padding: 32 }}><KBPage /></div>} />
          <Route path="/templates" element={<div style={{ padding: 32 }}><TemplatePage /></div>} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/chat/:agentId" element={<ChatView />} />
        </Routes>
      </main>
    </>
  )
}
