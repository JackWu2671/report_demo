"""
shared_tools.py — agent1 / agent2 共用的工具定义和 handler。

目前共用工具：
  - search_graph_tree : FAISS 检索知识库
  - modify_outline    : 对大纲执行结构化 patch 操作

各 agent 通过工厂函数获取带有上下文描述的工具定义，handler 直接导入复用。
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


# ── Tool Definitions ─────────────────────────────────────────────

def make_search_graph_tree_tool(context_desc: str, question_desc: str = "业务场景描述，原文传入") -> dict:
    """构造 search_graph_tree 工具定义，context_desc 说明该 agent 的调用时机和后续动作。"""
    return {
        "type": "function",
        "function": {
            "name": "search_graph_tree",
            "description": (
                "从知识图谱中检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、名称、描述）。"
                + context_desc
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": question_desc,
                    },
                },
                "required": ["question"],
            },
        },
    }


_MODIFY_OUTLINE_OPS_BASE = (
    "支持的操作：\n"
    "- add_node: {op, node_id, parent_id} — 新增知识库已有节点（node_id 必须来自 search_graph_tree 返回结果，不可新建）；顶层章节 parent_id 传 \"\"\n"
    "- delete_node: {op, node_id} — 删除节点及其子树\n"
    "- modify_node_name: {op, node_id, value} — 修改节点名称\n"
    "- modify_node_description: {op, node_id, value} — 修改节点描述\n"
    "- modify_node_condition: {op, node_id, value} — 设置或修改节点展示条件；value 格式必须为「当……时，本节才展示」；value 传空字符串表示删除条件\n"
    "- keep_only_node: {op, node_id} — 保留该节点，删除同级其他节点（每个保留节点单独一条）"
)


def make_modify_outline_tool(extra_desc: str = "", one_op_per_call: bool = False) -> dict:
    """构造 modify_outline 工具定义。

    Args:
        extra_desc      : 追加到 description 末尾的上下文说明
        one_op_per_call : True 时在描述和 ops 中加入"每次只传一个 op"限制（agent1）
    """
    desc = "对当前报告大纲执行修改，直接传入结构化操作列表。仅当已存在大纲时可用。"
    if one_op_per_call:
        desc += " 每次只传一个 op。"
    if extra_desc:
        desc += " " + extra_desc

    header = "操作列表，每条操作包含 op 字段和对应参数。"
    if one_op_per_call:
        header += " 每次只传一个 op。"
    ops_desc = header + "\n" + _MODIFY_OUTLINE_OPS_BASE

    return {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "ops": {
                        "type": "array",
                        "description": ops_desc,
                        "items": {"type": "object"},
                    }
                },
                "required": ["ops"],
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


# ── Shared Utilities ─────────────────────────────────────────────

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
