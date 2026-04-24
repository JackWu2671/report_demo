import React, { useEffect, useState } from 'react'
import TreeNode from '../components/TreeNode'

function buildTree(nodes, relations) {
  const map = {}
  nodes.forEach(n => { map[n.id] = { ...n, children: [] } })

  const childIds = new Set(relations.map(r => r.child))
  relations.forEach(r => {
    if (map[r.parent] && map[r.child]) {
      map[r.parent].children.push(map[r.child])
    }
  })
  // sort children by order field if present
  Object.values(map).forEach(n => {
    n.children.sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
  })

  return Object.values(map).filter(n => !childIds.has(n.id))
}

export default function KBPage() {
  const [roots, setRoots] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch('/api/kb')
      .then(r => r.json())
      .then(data => setRoots(buildTree(data.nodes, data.relations)))
      .catch(e => setError(e.message))
  }, [])

  if (error) return <div className="empty">加载失败：{error}</div>
  if (!roots) return <div className="loading">加载知识库...</div>
  if (!roots.length) return <div className="empty">知识库为空</div>

  return (
    <>
      <h1 className="page-title">知识库 · {roots.reduce((s, r) => s + countNodes(r), 0)} 个节点</h1>
      {roots.map(root => (
        <TreeNode key={root.id} node={root} defaultOpen={true} />
      ))}
    </>
  )
}

function countNodes(node) {
  return 1 + (node.children || []).reduce((s, c) => s + countNodes(c), 0)
}
