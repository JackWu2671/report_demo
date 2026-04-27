"""
generate_outline.py — generate_outline tool implementation (KB-only path).

Runs the full case_workflow_2 KB retrieval pipeline.
Called when search_outline_template returns not_found.
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

from retriever import embed_query, search_nodes, build_candidate_paths
from anchor import select_anchor
from subtree import build_subtree
from patcher import parse_patch, apply_patch
from loader import load_resources
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def generate_outline(question: str) -> dict:
    """
    KB retrieval + subtree build + initial patch from question.

    Returns:
        {status: "success"|"not_found", outline_tree, markdown, md_with_ids, message}
    """
    logger.info("[Tool:generate_outline] question=%r", question)

    query_embedding = await embed_query(question)
    faiss_svc, nodes_dict, children_map = load_resources()

    hits = search_nodes(query_embedding, faiss_svc)
    if not hits:
        return _not_found(f"知识库中未检索到与「{question}」相关的节点，系统暂不支持该分析场景。")

    candidates = build_candidate_paths(hits, nodes_dict, children_map)
    anchor = await select_anchor(question, candidates)

    try:
        tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)
    except ValueError as e:
        return _not_found(str(e))

    ops = await parse_patch(question, tree)
    if ops:
        tree = apply_patch(tree, ops)

    clean_tree = to_clean_json(tree)
    logger.info("[Tool:generate_outline] 完成，%d 字", len(to_markdown(clean_tree)))
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
