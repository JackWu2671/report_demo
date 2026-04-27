"""
handlers.py — Tool dispatch table for agent2.

Each handler receives the parsed tool arguments plus the shared agent state dict,
executes the tool, and returns the tool result string (injected as a
role=tool message back into the conversation).
"""

import json
import logging

from .generate import generate_outline
from .modify import modify_outline

logger = logging.getLogger(__name__)


async def handle_generate_outline(args: dict, state: dict) -> str:
    """
    Execute generate_outline and update shared state.

    state keys written: outline_tree, markdown, md_with_ids
    """
    question = args.get("question", "")
    result = await generate_outline(question)

    state["outline_tree"] = result["outline_tree"]
    state["markdown"] = result["markdown"]
    state["md_with_ids"] = result["md_with_ids"]

    if not result["outline_tree"]:
        return result["markdown"]  # error message

    # Tool result injected into LLM context: LLM-readable tree + clean markdown
    return (
        f"[大纲已生成]\n\n"
        f"## 带ID视图（供引用）\n{result['md_with_ids']}\n\n"
        f"## Markdown视图（供展示）\n{result['markdown']}"
    )


async def handle_modify_outline(args: dict, state: dict) -> str:
    """
    Execute modify_outline and update shared state.

    state keys read: outline_tree
    state keys written: outline_tree, markdown, md_with_ids
    """
    instruction = args.get("instruction", "")
    result = await modify_outline(instruction, state.get("outline_tree", {}))

    state["outline_tree"] = result["outline_tree"]
    state["markdown"] = result["markdown"]
    state["md_with_ids"] = result["md_with_ids"]

    if not result["outline_tree"]:
        return result["markdown"]  # error message

    ops_summary = json.dumps(result["ops"], ensure_ascii=False)
    return (
        f"[大纲已修改，共 {len(result['ops'])} 个操作: {ops_summary}]\n\n"
        f"## 带ID视图（供引用）\n{result['md_with_ids']}\n\n"
        f"## Markdown视图（供展示）\n{result['markdown']}"
    )


# Dispatch table: tool_name → handler coroutine
HANDLERS: dict = {
    "generate_outline": handle_generate_outline,
    "modify_outline": handle_modify_outline,
}
