"""
build_outline_from_anchor.py — build_outline_from_anchor tool implementation.

Pure Python: builds an outline subtree from a given anchor node ID.
The agent LLM selects the anchor from the search_graph_tree result.
Used by: agent2
"""

import logging
import os
import sys

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.subtree import build_subtree
from utils.loader import load_resources
from utils.outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def build_outline_from_anchor(anchor_id: str) -> dict:
    """
    Build an outline subtree rooted at the given anchor node.

    Args:
        anchor_id : node ID selected by the agent from search_graph_tree result

    Returns:
        {status: "success"|"not_found", outline_tree, markdown, md_with_ids, message}
    """
    logger.info("[Tool:build_outline_from_anchor] anchor_id=%r", anchor_id)

    _, nodes_dict, children_map = load_resources()

    try:
        tree = build_subtree(anchor_id, nodes_dict, children_map)
    except ValueError as e:
        return {"status": "not_found", "outline_tree": {}, "markdown": "",
                "md_with_ids": "", "message": str(e)}

    clean_tree = to_clean_json(tree)
    # 用虚拟根节点包裹，使 add_node parent_id="" 能正确地与一级章节平行
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": [clean_tree]}
    logger.info("[Tool:build_outline_from_anchor] 完成，根节点: %s", clean_tree.get("name"))
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "md_with_ids": to_markdown_with_ids(wrapped),
        "message": "",
    }
