"""
handlers.py — agent1 tool dispatch (thin adapter layer).

Imports implementations from backend/tools/ and wires them to Agent1Memory.
Each handler returns (result_dict, llm_str).
"""

import logging

from agent1.memory import Agent1Memory
from tools.analyze_expert import analyze_expert_knowledge
from tools.modify_outline import modify_outline
from tools.save_template import save_outline_template

logger = logging.getLogger(__name__)


async def handle_analyze_expert_knowledge(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    expert_text = args.get("expert_text", "")
    result = await analyze_expert_knowledge(expert_text)

    if result["status"] == "success":
        result["expert_text"] = expert_text
        memory.set_analysis_result(result)

        new_summary = (
            f"{len(result['new_nodes'])} 个 [new] 节点: "
            + ", ".join(n["name"] for n in result["new_nodes"][:5])
            if result["new_nodes"] else "无 [new] 节点"
        )
        llm_str = (
            f"[analyze_expert_knowledge] status=success\n"
            f"场景: {result['extraction'].get('scene_name')}\n"
            f"{new_summary}\n\n"
            f"{result['md_with_ids']}"
        )
    else:
        llm_str = f"[analyze_expert_knowledge] status=error  message={result['message']}"

    return result, llm_str


async def handle_modify_outline(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
    result = await modify_outline(args.get("instruction", ""), memory.outline_tree)
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        ops_summary = ", ".join(op.get("op", "?") for op in result["ops"])
        llm_str = (
            f"[modify_outline] status=success  ops={len(result['ops'])} ({ops_summary})\n\n"
            f"{result['md_with_ids']}"
        )
    else:
        llm_str = f"[modify_outline] status=error  message={result['message']}"
    return result, llm_str


async def handle_save_outline_template(args: dict, memory: Agent1Memory) -> tuple[dict, str]:
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


HANDLERS: dict = {
    "analyze_expert_knowledge": handle_analyze_expert_knowledge,
    "modify_outline": handle_modify_outline,
    "save_outline_template": handle_save_outline_template,
}
