"""
generate.py — Tool: generate_outline

KB-only outline generation (Steps 1-7 of case_workflow_2, no template search).
Called when search_outline_template returns not_found.

Returns {status, outline_tree, markdown, md_with_ids}.
status="not_found" when the KB has no relevant nodes → agent LLM should reject.
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

from retriever import embed_query, search_nodes, build_candidate_paths
from anchor import select_anchor
from subtree import build_subtree
from renderer import render_outline
from patcher import parse_patch, apply_patch
from loader import load_resources
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def generate_outline(question: str) -> dict:
    """
    Run the KB retrieval + build pipeline and return all three representations.

    Returns:
        {
            "status"      : "success" | "not_found",
            "outline_tree": dict,
            "markdown"    : str,
            "md_with_ids" : str,
            "message"     : str,  # human-readable reason when not_found
        }
    """
    logger.info("[Tool:generate_outline] question=%r", question)

    query_embedding = await embed_query(question)
    faiss_svc, nodes_dict, children_map = load_resources()

    hits = search_nodes(query_embedding, faiss_svc)
    if not hits:
        logger.info("[Tool:generate_outline] FAISS 无命中")
        return _not_found(f"知识库中未检索到与「{question}」相关的节点，系统暂不支持该分析场景。")

    candidates = build_candidate_paths(hits, nodes_dict, children_map)
    anchor = await select_anchor(question, candidates)

    try:
        tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)
    except ValueError as e:
        logger.warning("[Tool:generate_outline] build_subtree 失败: %s", e)
        return _not_found(str(e))

    # Apply initial param patch from the question itself
    ops = await parse_patch(question, tree)
    if ops:
        tree = apply_patch(tree, ops)

    clean_tree = to_clean_json(tree)
    markdown = to_markdown(clean_tree)
    md_with_ids = to_markdown_with_ids(clean_tree)

    logger.info(
        "[Tool:generate_outline] 完成，Markdown %d 字",
        len(markdown),
    )
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": markdown,
        "md_with_ids": md_with_ids,
        "message": "",
    }


def _not_found(message: str) -> dict:
    return {
        "status": "not_found",
        "outline_tree": {},
        "markdown": "",
        "md_with_ids": "",
        "message": message,
    }
