"""
definitions.py — agent1 的 OpenAI 工具 schema 定义。

共五个工具，按专家知识沉淀流程排列：
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

from tools.shared_tools import make_search_graph_tree_tool, make_modify_outline_tool

TOOLS: list[dict] = [
    make_search_graph_tree_tool(
        context_desc="专家提供场景描述后首先调用，获取可用的节点 id，再组合构造大纲。",
        question_desc="专家的业务场景描述，原文传入",
    ),
    {
        "type": "function",
        "function": {
            "name": "set_outline_from_markdown",
            "description": (
                "将 LLM 构造的 md_with_ids 格式大纲文本解析为结构化大纲并渲染到前端，供专家直接查看。"
                "调用后大纲将立即展示给专家，请确认内容完整、结构正确后再调用。"
                "L2/L3/L4 层级由 LLM 按专家意图自由设计；L5 必须引用 search_graph_tree 返回的知识库节点 id。"
                "调用此工具后，必须紧接着调用 set_scene_metadata 填写所有场景元数据。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "md_with_ids": {
                        "type": "string",
                        "description": (
                            "大纲文本，每行格式：{缩进}[L{层级} {id}] {名称}（新建节点名后加全角冒号和描述）\n"
                            "必须以唯一的 L1 节点作为根（报告总标题），所有 L2 节点均为其子节点。\n"
                            "示例：\n"
                            "[L1 new_001] 传送网络覆盖分析报告：面向OTN站点企业覆盖现状的专项分析\n"
                            "  [L2 new_002] 企业分布洞察：了解目标市场的行业与区域分布\n"
                            "    [L3 new_003] 行业与区域分布：统计价值企业分布，识别拓展方向\n"
                            "      [L4 new_004] 企业分布分析：从行业、行政区等维度统计企业分布\n"
                            "        [L5 L5_001] 企业行业分布\n"
                            "        [L5 L5_002] 企业行政区分布"
                        ),
                    },
                },
                "required": ["md_with_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_scene_metadata",
            "description": (
                "填写场景元数据（名称、摘要、关键词、适用条件），在 set_outline_from_markdown 之后立即调用。"
                "元数据与大纲渲染解耦，仅在保存模板时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_name": {
                        "type": "string",
                        "description": "场景名称，中文，不超过 10 字，如「传送网络覆盖分析」",
                    },
                    "summary": {
                        "type": "string",
                        "description": "一句话场景摘要，不超过 50 字，概括本次分析的核心目标",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3～8 个核心领域关键词，名词短语为主，代表分析维度、评估指标或技术名词",
                    },
                    "usage_conditions": {
                        "type": "string",
                        "description": "适用条件，说明在什么业务场景下适合使用这份大纲，以及有哪些前提要求，不超过 80 字",
                    },
                },
                "required": ["scene_name", "summary", "keywords", "usage_conditions"],
            },
        },
    },
    make_modify_outline_tool(one_op_per_call=True),
    {
        "type": "function",
        "function": {
            "name": "save_outline_template",
            "description": (
                "将当前大纲保存为可复用模板。"
                "仅在专家明确确认（如说'保存'、'好的就这样'）时调用，不得主动触发。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
