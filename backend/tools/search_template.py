"""
search_template.py — template search and matching tools.

Two functions at different granularities:

  search_outline_templates  — vector search only, returns top-N raw candidates
  match_outline_template    — search + LLM judge, returns best match or not_found

Used by: agent2, workflow.py
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

from retriever import embed_query
from template_selector import search_templates, select_template
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def search_outline_templates(question: str, top_k: int = 5) -> dict:
    """
    Vector search only — return top-N template candidates without LLM judgment.

    Returns:
        status="found"     — candidates list populated
        status="not_found" — template library is empty or no hits
    """
    logger.info("[Tool:search_outline_templates] question=%r top_k=%d", question, top_k)

    query_embedding = await embed_query(question)
    candidates = await search_templates(query_embedding, top_k=top_k)

    if not candidates:
        return {"status": "not_found", "candidates": [], "reason": "模板库为空"}

    logger.info(
        "[Tool:search_outline_templates] 命中 %d 个候选: %s",
        len(candidates),
        [(t["scene_name"], round(t["_score"], 3)) for t in candidates],
    )
    return {
        "status": "found",
        "candidates": [
            {
                "scene_name": t.get("scene_name", ""),
                "summary": t.get("summary", ""),
                "usage_conditions": t.get("usage_conditions", ""),
                "score": round(t.get("_score", 0), 3),
            }
            for t in candidates
        ],
    }


async def match_outline_template(question: str) -> dict:
    """
    Vector search + LLM judge — return the best-matching template or not_found.

    Returns:
        status="pending_confirm"  — matched a template; outline_tree populated
        status="not_found"        — no suitable template found
    """
    logger.info("[Tool:match_outline_template] question=%r", question)

    query_embedding = await embed_query(question)
    candidates = await search_templates(query_embedding, top_k=5)

    if not candidates:
        return _not_found("模板库为空，请走知识库生成")

    selected = await select_template(question, candidates)
    if not selected:
        return _not_found("未找到与需求匹配的预制大纲")

    raw_tree = selected.get("outline", {})
    if not raw_tree:
        return _not_found("模板存在但缺少 outline 字段")

    clean_tree = to_clean_json(raw_tree)
    logger.info("[Tool:match_outline_template] 选中: %s (score=%.3f)",
                selected.get("scene_name"), selected.get("_score", 0))
    return {
        "status": "pending_confirm",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "scene_name": selected.get("scene_name", ""),
        "reason": "",
    }


def _not_found(reason: str) -> dict:
    return {"status": "not_found", "outline_tree": {}, "markdown": "",
            "md_with_ids": "", "scene_name": "", "reason": reason}
