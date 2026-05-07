"""
init_outline_from_graph.py — init_outline_from_graph tool implementation.

Anchor selection + subtree expansion + initial patch.
Requires candidates from search_graph_tree (no internal FAISS search).
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

from anchor import select_anchor
from subtree import build_subtree
from patcher import parse_patch, apply_patch
from loader import load_resources
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def init_outline_from_graph(question: str, kb_tree_text: str) -> dict:
    """
    Anchor selection + subtree expansion + initial patch.

    Args:
        question     : user's analysis question
        kb_tree_text : full tree text from search_graph_tree (★ marks FAISS-hit nodes)

    Returns:
        {status: "success"|"not_found", outline_tree, markdown, md_with_ids, message}
    """
    logger.info("[Tool:init_outline_from_graph] question=%r", question)

    if not kb_tree_text:
        return _not_found("没有可用的知识图谱树，请先调用 search_graph_tree。")

    _, nodes_dict, children_map = load_resources()

    anchor = await select_anchor(question, kb_tree_text)

    try:
        tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)
    except ValueError as e:
        return _not_found(str(e))

    ops = await parse_patch(question, tree)
    if ops:
        tree = apply_patch(tree, ops)

    clean_tree = to_clean_json(tree)
    logger.info("[Tool:init_outline_from_graph] 完成，%d 字", len(to_markdown(clean_tree)))
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "message": "",
    }


def _not_found(message: str) -> dict:
    return {"status": "not_found", "outline_tree": {}, "markdown": "",
            "md_with_ids": "", "message": message}
