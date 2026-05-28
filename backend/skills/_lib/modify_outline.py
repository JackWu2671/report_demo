"""
modify_outline.py — modify_outline tool implementation.

Pure Python: applies a structured ops list to the outline tree.
The main agent LLM is responsible for constructing the ops.
Used by: agent2
"""

import logging
import os
import sys

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from patcher import apply_patch
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def modify_outline(ops: list[dict], outline_tree: dict) -> dict:
    """
    Apply a structured ops list to the current outline tree.

    Args:
        ops          : list of patch operations constructed by the agent LLM
        outline_tree : current outline tree dict

    Returns:
        {status: "success"|"error", outline_tree, markdown, md_with_ids, ops, message}
    """
    if not outline_tree:
        return {"status": "error", "outline_tree": {}, "markdown": "",
                "md_with_ids": "", "ops": [],
                "message": "当前没有可修改的大纲，请先生成大纲。"}

    logger.info("[Tool:modify_outline] %d 个操作: %s", len(ops), [op.get("op") for op in ops])
    new_tree, skipped = await apply_patch(outline_tree, ops)
    clean_tree = to_clean_json(new_tree)
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "ops": ops,
        "skipped": skipped,
        "message": "",
    }
