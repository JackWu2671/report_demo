"""
search_graph_tree.py — search_graph_tree tool implementation.

Wraps retriever.search_graph_tree: embed query → FAISS search →
build ancestor paths → assemble tree dict list.

Returns the KB graph tree as structured data and a formatted text
representation for LLM consumption.

Used by: agent2
"""

import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.retriever import search_graph_tree as _search_graph_tree

logger = logging.getLogger(__name__)


def _tree_to_text(nodes: list[dict], depth: int = 0) -> str:
    """将树状 dict 列表渲染为缩进文本，★ 标记 FAISS 直接命中节点。"""
    lines = []
    for node in nodes:
        indent = "  " * depth
        hit_mark = " ★" if node.get("hit") else ""
        score_str = f" ({node['score']:.3f})" if node.get("score") is not None else ""
        id_str = f" {node['id']}" if node.get("id") else ""
        desc_str = f" — {node['description']}" if node.get("description") else ""
        lines.append(f"{indent}[L{node['level']}{id_str}] {node['name']}{hit_mark}{score_str}{desc_str}")
        if node.get("children"):
            lines.append(_tree_to_text(node["children"], depth + 1))
    return "\n".join(lines)


async def search_graph_tree(question: str) -> dict:
    """
    Search the knowledge graph and return a tree of relevant nodes.

    Returns:
        status="success"   — tree found; graph_tree + tree_text populated
        status="not_found" — no relevant nodes in the KB
    """
    logger.info("[Tool:search_graph_tree] question=%r", question)

    tree, _ = await _search_graph_tree(question)

    if not tree:
        return {
            "status": "not_found",
            "graph_tree": [],
            "tree_text": "",
            "message": f"知识库中未检索到与「{question}」相关的节点，系统暂不支持该分析场景。",
        }

    tree_text = _tree_to_text(tree)
    logger.info("[Tool:search_graph_tree] 完成\n%s", tree_text)
    return {
        "status": "success",
        "graph_tree": tree,
        "tree_text": tree_text,
        "message": "",
    }
