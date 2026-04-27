"""
handlers.py — Tool dispatch for agent2.

Each handler:
  1. Executes the tool function
  2. Updates AgentMemory if outline was produced
  3. Returns (result_dict, llm_str):
       result_dict — full result (agent uses this to emit outline SSE event)
       llm_str     — compact string injected as tool message in LLM history
                     (md_with_ids only, NO full markdown — markdown goes via outline event)
"""

import logging

from agent2.memory.store import AgentMemory
from .search_template import search_outline_template
from .generate import generate_outline
from .modify import modify_outline

logger = logging.getLogger(__name__)


# ── Handlers ──────────────────────────────────────────────────

async def handle_search_outline_template(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await search_outline_template(args.get("question", ""))
    if result["status"] == "found":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[search_outline_template] status=found  scene={result['scene_name']}\n\n"
            f"{result['md_with_ids']}"
        )
    else:
        llm_str = f"[search_outline_template] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_generate_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    result = await generate_outline(args.get("question", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[generate_outline] status=success\n\n"
            f"{result['md_with_ids']}"
        )
    else:
        llm_str = f"[generate_outline] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_modify_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
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


# ── Dispatch table ─────────────────────────────────────────────

HANDLERS: dict = {
    "search_outline_template": handle_search_outline_template,
    "generate_outline": handle_generate_outline,
    "modify_outline": handle_modify_outline,
}
