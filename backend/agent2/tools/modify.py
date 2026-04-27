"""
modify.py — Tool implementation for modify_outline.

Uses case_workflow_2 patcher (Steps 8-9) to apply natural-language instructions
to the current outline tree, then re-renders all three representations.
"""

import logging
import os
import sys

_AGENT2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BACKEND_DIR = os.path.dirname(_AGENT2_DIR)
_WF2_DIR = os.path.join(_BACKEND_DIR, "case_workflow_2")

for _p in [_BACKEND_DIR, _WF2_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from case_workflow_2.patcher import parse_patch, apply_patch
from outline_utils import to_markdown, to_markdown_with_ids, to_clean_json

logger = logging.getLogger(__name__)


async def modify_outline(instruction: str, outline_tree: dict) -> dict:
    """
    Apply a natural-language modification to outline_tree via the patcher.

    Args:
        instruction  : user's modification request
        outline_tree : current clean JSON tree (from agent state)

    Returns:
        {
            "outline_tree": dict,      # updated clean JSON tree
            "markdown": str,           # updated human-readable Markdown
            "md_with_ids": str,        # updated LLM-context Markdown
            "ops": list[dict],         # patch operations that were applied
        }
    """
    logger.info("[Tool:modify_outline] instruction=%r", instruction)

    if not outline_tree:
        return {
            "outline_tree": {},
            "markdown": "当前没有可修改的大纲，请先生成大纲。",
            "md_with_ids": "",
            "ops": [],
        }

    ops = await parse_patch(instruction, outline_tree)
    new_tree = apply_patch(outline_tree, ops)
    clean_tree = to_clean_json(new_tree)
    markdown = to_markdown(clean_tree)
    md_with_ids = to_markdown_with_ids(clean_tree)

    logger.info(
        "[Tool:modify_outline] 完成，应用 %d 个操作: %s",
        len(ops),
        [op.get("op") for op in ops],
    )
    return {
        "outline_tree": clean_tree,
        "markdown": markdown,
        "md_with_ids": md_with_ids,
        "ops": ops,
    }
