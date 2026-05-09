"""
shared_tools.py — 所有工具的 schema 定义全集。

各 agent 按需从此文件导入所需工具定义，组装自己的 TOOLS 列表。
共用的 handler 实现（handle_search_graph_tree、handle_modify_outline）也在此处统一维护。
"""

import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.search_graph_tree import search_graph_tree
from tools.modify_outline import modify_outline
from memory.store import AgentMemory

logger = logging.getLogger(__name__)


# ── Tool Schema Definitions ──────────────────────────────────────

SEARCH_GRAPH_TREE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "search_graph_tree",
        "description": "从知识图谱中检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、名称、描述）。",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "业务场景描述，原文传入",
                },
            },
            "required": ["question"],
        },
    },
}

MODIFY_OUTLINE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "modify_outline",
        "description": "对当前报告大纲执行修改，直接传入结构化操作列表。仅当已存在大纲时可用。",
        "parameters": {
            "type": "object",
            "properties": {
                "ops": {
                    "type": "array",
                    "description": (
                        "操作列表，每条操作包含 op 字段和对应参数。\n"
                        "支持的操作：\n"
                        "- add_node: {op, node_id, parent_id} — 新增知识库已有节点（node_id 必须来自 search_graph_tree 返回结果，不可新建）；顶层章节 parent_id 传 \"\"\n"
                        "- delete_node: {op, node_id} — 删除节点及其子树\n"
                        "- modify_node_name: {op, node_id, value} — 修改节点名称\n"
                        "- modify_node_description: {op, node_id, value} — 修改节点描述\n"
                        "- modify_node_condition: {op, node_id, value} — 设置或修改节点展示条件；value 格式必须为「当……时，本节才展示」；value 传空字符串表示删除条件\n"
                        "- keep_only_node: {op, node_id} — 保留该节点，删除同级其他节点（每个保留节点单独一条）"
                    ),
                    "items": {"type": "object"},
                }
            },
            "required": ["ops"],
        },
    },
}

SEARCH_OUTLINE_TEMPLATES_TOOL: dict = {
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
}

LOAD_TEMPLATE_OUTLINE_TOOL: dict = {
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
}

BUILD_OUTLINE_FROM_ANCHOR_TOOL: dict = {
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
}

SET_OUTLINE_FROM_MARKDOWN_TOOL: dict = {
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
}

SET_SCENE_METADATA_TOOL: dict = {
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
}

SAVE_OUTLINE_TEMPLATE_TOOL: dict = {
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
}


# ── Shared Handlers ──────────────────────────────────────────────

async def handle_search_graph_tree(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """从知识图谱检索相关节点，结果存入 memory 供后续构造大纲使用。"""
    result = await search_graph_tree(args.get("question", ""))
    if result["status"] == "success":
        memory.set_kb_tree(result["tree_text"])
        llm_str = f"[search_graph_tree] status=success\n\n{result['tree_text']}"
    else:
        llm_str = f"[search_graph_tree] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_modify_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """对当前大纲执行结构化修改操作，将修改后的大纲写入 memory。"""
    result = await modify_outline(args.get("ops", []), memory.outline_tree)
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        skipped = result.get("skipped", [])
        op_lines = "\n".join(_format_op(op) for op in result["ops"])
        llm_str = f"[modify_outline] status=success  ops={len(result['ops'])}\n{op_lines}"
        if skipped:
            skip_lines = "\n".join(
                f"  - {op.get('op')} node_id={op.get('node_id')} 原因: {op.get('_skip_reason', '未知')}"
                for op in skipped
            )
            llm_str += f"\n\n⚠️ 以下 {len(skipped)} 个操作未执行，请在下一步补救：\n{skip_lines}"
        llm_str += f"\n\n当前大纲：\n{result['md_with_ids']}"
    else:
        llm_str = f"[modify_outline] status=error  message={result['message']}"
    return result, llm_str


def _format_op(op: dict) -> str:
    """将单条操作格式化为可读单行字符串，用于前端步骤摘要和 LLM 历史。"""
    name = op.get("op", "?")
    if name == "add_node":
        return f"  + add_node    node_id={op.get('node_id')}  parent_id={op.get('parent_id') or '(root)'}"
    if name == "delete_node":
        return f"  - delete_node node_id={op.get('node_id')}"
    if name == "keep_only_node":
        return f"  ✓ keep_only_node node_id={op.get('node_id')}"
    if name == "modify_node_name":
        return f"  ~ modify_name node_id={op.get('node_id')}  value={op.get('value')!r}"
    if name == "modify_node_description":
        return f"  ~ modify_desc node_id={op.get('node_id')}  value={op.get('value', '')[:60]!r}"
    if name == "modify_node_condition":
        return f"  ~ modify_cond node_id={op.get('node_id')}  value={op.get('value', '')[:60]!r}"
    return f"  ? {name} {op}"
