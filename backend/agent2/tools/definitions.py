"""
definitions.py — agent2 的 OpenAI 工具 schema 定义。

共五个工具，按新建大纲时的推荐调用顺序排列：
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

from tools.shared_tools import make_search_graph_tree_tool, make_modify_outline_tool

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_outline_templates",
            "description": (
                "向量检索模板库，返回与需求最相似的 top-N 候选模板列表（含 scene_name、summary、score）。"
                "用户提出新的分析需求时优先调用。根据返回的候选列表自行判断是否有匹配的模板："
                "有匹配 → 调用 load_template_outline 加载；无匹配 → 调用 search_graph_tree 从知识库生成。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回候选数量，默认 5",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_template_outline",
            "description": (
                "按模板 id 直接加载指定模板的完整大纲内容。"
                "在 search_outline_templates 返回候选后，判断有匹配时调用此工具加载大纲，再询问用户是否使用。"
                "template_id 必须取自 search_outline_templates 返回的候选列表中的 id 字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "template_id": {
                        "type": "string",
                        "description": "模板唯一 id，取自 search_outline_templates 返回的候选列表中的 id 字段",
                    },
                },
                "required": ["template_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_outline_from_anchor",
            "description": (
                "以指定节点为根，从知识图谱展开子树，生成初始报告大纲。"
                "必须在 search_graph_tree 成功后，从返回的树中选出最相关节点的 id，再调用此工具。"
                "anchor_id 取自 search_graph_tree 返回的树节点 id 字段，选择与用户需求最直接相关的节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "anchor_id": {
                        "type": "string",
                        "description": "锚节点 id，从 search_graph_tree 返回的树中选取，如 'L4_001'",
                    }
                },
                "required": ["anchor_id"],
            },
        },
    },
    make_search_graph_tree_tool(
        context_desc=(
            "search_outline_templates 无合适候选时调用此工具，再从结果树中选锚节点调用 build_outline_from_anchor。"
            "status=not_found 表示知识库无相关内容，应告知用户系统暂不支持该场景。"
        ),
        question_desc="用户的分析需求描述，原文传入",
    ),
    make_modify_outline_tool(
        extra_desc="ops 由你根据用户指令和当前大纲（system prompt 中）直接构造。"
    ),
]
