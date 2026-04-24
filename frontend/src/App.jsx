import React from 'react'
import { Routes, Route, NavLink } from 'react-router-dom'
import KBPage from './pages/KBPage'
import TemplatePage from './pages/TemplatePage'
import ChatPage from './pages/ChatPage'

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
      <main className="main">
        <Routes>
          <Route path="/" element={<KBPage />} />
          <Route path="/templates" element={<TemplatePage />} />
          <Route path="/chat" element={<ChatPage />} />
        </Routes>
      </main>
    </>
  )
}
