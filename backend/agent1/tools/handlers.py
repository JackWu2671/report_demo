"""
handlers.py — agent1 工具分发层（薄适配层）。

从 backend/tools/ 引入各工具的具体实现，并与 Agent1Memory 对接。
每个 handler 返回 (result_dict, llm_str)。
"""

import logging

from agent1.memory import Agent1Memory
from tools.search_graph_tree import search_graph_tree
from tools.set_outline_from_markdown import set_outline_from_markdown
from tools.set_scene_metadata import set_scene_metadata
from tools.modify_outline import modify_outline
from tools.save_template import save_outline_template

logger = logging.getLogger(__name__)


async def handle_search_graph_tree(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    """从知识图谱检索相关节点，返回带祖先路径的树状结构，供构造大纲时引用节点 id。"""
    result = await search_graph_tree(args.get("question", ""))
    if result["status"] == "success":
        memory.set_kb_tree(result["tree_text"])
        llm_str = f"[search_graph_tree] status=success\n\n{result['tree_text']}"
    else:
        llm_str = f"[search_graph_tree] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_set_outline_from_markdown(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    """解析 LLM 构造的 md_with_ids 文本，将大纲写入 memory。"""
    result = await set_outline_from_markdown(md_with_ids=args.get("md_with_ids", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[set_outline_from_markdown] status=success\n"
            f"大纲已渲染。接下来请调用 set_scene_metadata 填写场景元数据。\n\n"
            f"当前大纲：\n{result['md_with_ids']}"
        )
    else:
        llm_str = f"[set_outline_from_markdown] status=error  message={result['message']}"
    return result, llm_str


async def handle_set_scene_metadata(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    """将场景元数据合并写入 memory，供后续 save_outline_template 使用。"""
    result = await set_scene_metadata(
        scene_name=args.get("scene_name", ""),
        summary=args.get("summary", ""),
        keywords=args.get("keywords", []),
        usage_conditions=args.get("usage_conditions", ""),
    )
    if result["status"] == "success":
        memory.set_extraction({
            "scene_name": result["scene_name"],
            "summary": result["summary"],
            "keywords": result["keywords"],
            "usage_conditions": result["usage_conditions"],
        })
        llm_str = (
            f"[set_scene_metadata] status=success\n"
            f"场景：{result['scene_name']}\n"
            f"关键词：{', '.join(result['keywords'])}\n"
            f"适用条件：{result['usage_conditions']}\n"
            f"元数据已记录，请询问专家是否需要修改或保存。"
        )
    else:
        llm_str = f"[set_scene_metadata] status=error  message={result['message']}"
    return result, llm_str


async def handle_modify_outline(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
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


async def handle_save_outline_template(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    """将当前大纲和场景元数据保存为可复用模板文件。"""
    result = await save_outline_template(memory.extraction, memory.outline_tree)
    if result["status"] == "success":
        llm_str = (
            f"[save_outline_template] status=success\n"
            f"场景: {result['scene_name']}\n"
            f"路径: {result['path']}"
        )
    else:
        llm_str = f"[save_outline_template] status=error  message={result['message']}"
    return result, llm_str


def _format_op(op: dict) -> str:
    """将单条操作格式化为可读单行字符串。"""
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


HANDLERS: dict = {
    "search_graph_tree":         handle_search_graph_tree,
    "set_outline_from_markdown": handle_set_outline_from_markdown,
    "set_scene_metadata":        handle_set_scene_metadata,
    "modify_outline":            handle_modify_outline,
    "save_outline_template":     handle_save_outline_template,
}
