"""
shared_tools.py — 所有工具的 schema 定义全集。

各 agent 按需从此文件导入所需工具定义，组装自己的 TOOLS 列表。
共用的 handler 实现（handle_search_graph_tree、handle_modify_outline）也在此处统一维护。

工具列表：
  业务工具（agent1 / agent2 按需选用）：
    SEARCH_GRAPH_TREE_TOOL        MODIFY_OUTLINE_TOOL
    SEARCH_OUTLINE_TEMPLATES_TOOL LOAD_TEMPLATE_OUTLINE_TOOL
    BUILD_OUTLINE_FROM_ANCHOR_TOOL
    SET_OUTLINE_FROM_MARKDOWN_TOOL SET_SCENE_METADATA_TOOL
    SAVE_OUTLINE_TEMPLATE_TOOL

  Skill 系统元工具（AgentWithSkills 专用）：
    READ_SKILL_TOOL
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
            "调用此工具后，必须紧接着调用 set_scene_metadata 填写所有场景元数据。"
            "构造大纲时必须遵守以下约束，违反任意一条视为无效输出：\n"
            "1. L1 必须存在且唯一，作为大纲根节点（报告总标题），所有 L2 均为其子节点；\n"
            "2. query节点必须是叶子节点，禁止在 L5 下方挂任何子节点；\n"
            "3. 禁止新建 query节点（即禁止 [Q new_xxx]），L5 只能引用 search_graph_tree 返回的知识库节点 id；\n"
            "4. 禁止 [L4 new_xxx] 作叶子节点，每个新建 L4 下方必须至少挂一个知识库已有的 query节点；\n"
            "5. 所有新建节点（new_xxx）名称后必须紧跟全角冒号和描述（50～100 字），说明分析目的和业务关联；\n"
            "6. L2/L3/L4 由 LLM 按专家意图自由设计，不得用知识库节点名称替代专家描述的分析板块名称。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "md_with_ids": {
                    "type": "string",
                    "description": (
                        "大纲文本，每行格式：{缩进}[L{层级} {id}] {名称}（新建节点名后加全角冒号和描述）\n"
                        "query节点必须是叶子节点，不可再有子节点。\n"
                        "条件节点在描述后追加「｜条件：当……时，本节才展示」。\n"
                        "示例：\n"
                        "[L1 new_001] 传送网络覆盖分析报告：面向OTN站点企业覆盖现状的专项分析\n"
                        "  [L2 new_002] 企业分布洞察：了解目标市场的行业与区域分布\n"
                        "    [L3 new_003] 行业与区域分布：统计价值企业分布，识别重点拓展方向\n"
                        "      [L4 new_004] 企业分布分析：从行业、行政区等维度统计企业分布\n"
                        "        [Q L5_001] 企业行业分布\n"
                        "        [Q L5_002] 企业行政区分布"
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

GRAPH_MANAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "graph_manage",
        "description": (
            "执行知识图谱融合写入：将 agent 分析确认好的 patch 写入 node.json + relation.json。"
            "agent 必须按 graph-fusion.md 的 SOP 完成分析并获得专家确认后才能调用本工具。"
            "L5（query）节点受保护，level 只允许传 2、3、4。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "来源模板 ID，取自 save_outline_template 的返回值，用于溯源",
                },
                "add_nodes": {
                    "type": "array",
                    "description": "要新增到图谱的节点列表（仅 L2-L4）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "level": {
                                "type": "integer",
                                "enum": [2, 3, 4],
                                "description": "节点层级，只允许 2、3、4",
                            },
                            "name": {
                                "type": "string",
                                "description": "节点名称",
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "3-6 个检索关键词",
                            },
                            "description": {
                                "type": "string",
                                "description": "节点业务描述，50-100 字",
                            },
                            "parent_id": {
                                "type": "string",
                                "description": "父节点 ID，必须是 node.json 中已有节点的 ID，如 L2_001",
                            },
                        },
                        "required": ["level", "name", "keywords", "description", "parent_id"],
                    },
                },
                "enrich_nodes": {
                    "type": "array",
                    "description": "要丰富描述的已有图谱节点列表（仅 L1-L4，L5 自动跳过）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "node_id": {
                                "type": "string",
                                "description": "要丰富的节点 ID，必须是 node.json 中已有节点，如 L3_001",
                            },
                            "append": {
                                "type": "string",
                                "description": "追加到 description 末尾的补充描述，30-80 字，不重复已有内容",
                            },
                        },
                        "required": ["node_id", "append"],
                    },
                },
            },
            "required": ["template_id", "add_nodes", "enrich_nodes"],
        },
    },
}

READ_SKILL_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "read_skill",
        "description": (
            "加载指定 skill 的完整 SOP（Level 1），或其内部支持文件（Level 2）。"
            "决定使用某个 skill 前必须先加载其 SOP，已加载的 skill 无需重复加载。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill 名称，如 analyze-network"},
                "path": {
                    "type": "string",
                    "description": "可选。skill 文件夹内的支持文件路径（Level 2）",
                },
            },
            "required": ["name"],
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
    ops = args.get("ops", [])
    if isinstance(ops, str):
        import json as _json
        try:
            ops = _json.loads(ops)
        except _json.JSONDecodeError:
            ops = []
    result = await modify_outline(ops, memory.outline_tree)
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
