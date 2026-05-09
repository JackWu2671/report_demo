"""
definitions.py — agent2 的工具列表。

从 tools/shared_tools.py 导入所需工具定义，按新建大纲时的推荐调用顺序排列：
  1. search_outline_templates  — 向量检索，返回 top-N 候选模板列表（纯检索，不调 LLM）
  2. load_template_outline     — 按模板 id 直接加载完整大纲
  3. build_outline_from_anchor — 从 agent 选定的锚节点展开知识图谱子树
  4. search_graph_tree         — FAISS 检索知识库，构建带祖先路径的树状结构
  5. modify_outline            — 对当前大纲执行结构化 patch 操作
"""

import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_TOOLS_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.shared_tools import (
    SEARCH_OUTLINE_TEMPLATES_TOOL,
    LOAD_TEMPLATE_OUTLINE_TOOL,
    BUILD_OUTLINE_FROM_ANCHOR_TOOL,
    SEARCH_GRAPH_TREE_TOOL,
    MODIFY_OUTLINE_TOOL,
)

TOOLS: list[dict] = [
    SEARCH_OUTLINE_TEMPLATES_TOOL,
    LOAD_TEMPLATE_OUTLINE_TOOL,
    BUILD_OUTLINE_FROM_ANCHOR_TOOL,
    SEARCH_GRAPH_TREE_TOOL,
    MODIFY_OUTLINE_TOOL,
]
