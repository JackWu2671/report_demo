"""
build_outline_from_anchor.py — build_outline_from_anchor tool implementation.

Pure Python: builds an outline subtree from a given anchor node ID.
The agent LLM selects the anchor from the search_graph_tree result.
Used by: agent2
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

from subtree import build_subtree
from loader import load_resources
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

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
    logger.info("[Tool:build_outline_from_anchor] 完成，根节点: %s", clean_tree.get("name"))
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "message": "",
    }
