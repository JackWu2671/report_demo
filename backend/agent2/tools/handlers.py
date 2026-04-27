"""
handlers.py — agent2 tool dispatch (thin adapter layer).

Imports implementations from backend/tools/ and wires them to AgentMemory.
Each handler returns (result_dict, llm_str):
  result_dict — full result (agent uses for outline event)
  llm_str     — compact string for LLM history (md_with_ids only, no full markdown)
"""

import logging

from memory.store import AgentMemory
from tools.search_template import search_outline_template
from tools.generate_outline import generate_outline
from tools.modify_outline import modify_outline

logger = logging.getLogger(__name__)


async def handle_search_outline_template(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await search_outline_template(args.get("question", ""))
    if result["status"] == "found":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (f"[search_outline_template] status=found  scene={result['scene_name']}\n\n"
                   f"{result['md_with_ids']}")
    else:
        llm_str = f"[search_outline_template] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_generate_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await generate_outline(args.get("question", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = f"[generate_outline] status=success\n\n{result['md_with_ids']}"
    else:
        llm_str = f"[generate_outline] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_modify_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await modify_outline(args.get("instruction", ""), memory.outline_tree)
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        ops_summary = ", ".join(op.get("op", "?") for op in result["ops"])
        llm_str = (f"[modify_outline] status=success  ops={len(result['ops'])} ({ops_summary})\n\n"
                   f"{result['md_with_ids']}")
    else:
        llm_str = f"[modify_outline] status=error  message={result['message']}"
    return result, llm_str


HANDLERS: dict = {
    "search_outline_template": handle_search_outline_template,
    "generate_outline": handle_generate_outline,
    "modify_outline": handle_modify_outline,
}
