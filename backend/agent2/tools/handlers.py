"""
handlers.py — agent2 tool dispatch (thin adapter layer).

Imports implementations from backend/tools/ and wires them to AgentMemory.
Each handler returns (result_dict, llm_str):
  result_dict — full result (agent uses for outline event)
  llm_str     — compact string for LLM history (md_with_ids only, no full markdown)
"""

import logging

from memory.store import AgentMemory
from tools.search_template import match_outline_template, search_outline_templates, load_template_outline
from tools.build_outline_from_anchor import build_outline_from_anchor
from tools.search_graph_tree import search_graph_tree
from tools.modify_outline import modify_outline

logger = logging.getLogger(__name__)


async def handle_match_outline_template(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await match_outline_template(args.get("question", ""))
    if result["status"] == "pending_confirm":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[match_outline_template] status=pending_confirm  scene={result['scene_name']}\n"
            f"大纲已预览，请询问用户：使用此模板还是重新从知识库生成？\n\n"
            f"当前大纲：\n{result['md_with_ids']}"
        )
    else:
        llm_str = f"[match_outline_template] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_search_outline_templates(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await search_outline_templates(args.get("question", ""), args.get("top_k", 5))
    if result["status"] == "found":
        lines = [f"  {i+1}. {c['scene_name']} (score={c['score']}) — {c.get('summary', '')}"
                 for i, c in enumerate(result["candidates"])]
        llm_str = f"[search_outline_templates] 找到 {len(result['candidates'])} 个候选:\n" + "\n".join(lines)
    else:
        llm_str = f"[search_outline_templates] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_load_template_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = load_template_outline(args.get("scene_name", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[load_template_outline] status=success  scene={result['scene_name']}\n"
            f"大纲已加载，请询问用户是否满意或需要调整。\n\n"
            f"当前大纲：\n{result['md_with_ids']}"
        )
    else:
        llm_str = f"[load_template_outline] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_search_graph_tree(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await search_graph_tree(args.get("question", ""))
    if result["status"] == "success":
        memory.set_kb_tree(result["tree_text"])
        llm_str = f"[search_graph_tree] status=success\n\n{result['tree_text']}"
    else:
        llm_str = f"[search_graph_tree] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_build_outline_from_anchor(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await build_outline_from_anchor(args.get("anchor_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = f"[build_outline_from_anchor] status=success\n\n当前大纲：\n{result['md_with_ids']}"
    else:
        llm_str = f"[build_outline_from_anchor] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_modify_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
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
            llm_str += (f"\n\n⚠️ 以下 {len(skipped)} 个操作未执行，请在下一步补救：\n{skip_lines}")
        llm_str += f"\n\n当前大纲：\n{result['md_with_ids']}"
    else:
        llm_str = f"[modify_outline] status=error  message={result['message']}"
    return result, llm_str


def _format_op(op: dict) -> str:
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
    return f"  ? {name} {op}"


HANDLERS: dict = {
    "match_outline_template": handle_match_outline_template,
    "search_outline_templates": handle_search_outline_templates,
    "load_template_outline": handle_load_template_outline,
    "build_outline_from_anchor": handle_build_outline_from_anchor,
    "search_graph_tree": handle_search_graph_tree,
    "modify_outline": handle_modify_outline,
}
