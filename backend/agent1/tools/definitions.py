"""
definitions.py — agent1 的工具列表。

从 tools/shared_tools.py 导入所需工具定义，按专家知识沉淀流程排列：
  1. search_graph_tree         — FAISS 检索知识库，返回带祖先路径的树状结构
  2. set_outline_from_markdown — LLM 构造 md_with_ids 文本后调此工具渲染为大纲
  3. set_scene_metadata        — 填写场景名称、摘要、关键词、适用条件
  4. modify_outline            — 对当前大纲执行结构化 patch 操作
  5. save_outline_template     — 将当前大纲保存为可复用模板
"""

import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_TOOLS_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.shared_tools import (
    SEARCH_GRAPH_TREE_TOOL,
    SET_OUTLINE_FROM_MARKDOWN_TOOL,
    SET_SCENE_METADATA_TOOL,
    MODIFY_OUTLINE_TOOL,
    SAVE_OUTLINE_TEMPLATE_TOOL,
)

TOOLS: list[dict] = [
    SEARCH_GRAPH_TREE_TOOL,
    SET_OUTLINE_FROM_MARKDOWN_TOOL,
    SET_SCENE_METADATA_TOOL,
    MODIFY_OUTLINE_TOOL,
    SAVE_OUTLINE_TEMPLATE_TOOL,
]
