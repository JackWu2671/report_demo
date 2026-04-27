"""
modify_outline.py — modify_outline tool implementation.

Wraps case_workflow_2 patcher: parse_patch (LLM) + apply_patch (pure Python).
Used by: agent1, agent2 (shared)
"""

import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)
_WF2_DIR = os.path.join(_BACKEND_DIR, "case_workflow_2")

for _p in [_BACKEND_DIR, _WF2_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from patcher import parse_patch, apply_patch
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def modify_outline(instruction: str, outline_tree: dict) -> dict:
    """
    Parse and apply a modification instruction to the current outline.

    Returns:
        {status: "success"|"error", outline_tree, markdown, md_with_ids, ops, message}
    """
    if not outline_tree:
        return {"status": "error", "outline_tree": {}, "markdown": "",
                "md_with_ids": "", "ops": [],
                "message": "当前没有可修改的大纲，请先生成大纲。"}

    logger.info("[Tool:modify_outline] instruction=%r", instruction)
    ops = await parse_patch(instruction, outline_tree)
    new_tree = apply_patch(outline_tree, ops)
    clean_tree = to_clean_json(new_tree)
    logger.info("[Tool:modify_outline] 完成，%d 个操作: %s",
                len(ops), [op.get("op") for op in ops])
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "ops": ops,
        "message": "",
    }
