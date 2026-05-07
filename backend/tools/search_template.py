"""
search_template.py — template search and matching tools.

Three functions at different granularities:

  search_outline_templates  — vector search only, returns top-N raw candidates
  match_outline_template    — search + LLM judge, returns best match or not_found
  load_template_outline     — load full outline for a specific template by scene_name

Used by: agent2, workflow.py
"""

import json
import logging
import os
import sys
from pathlib import Path

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

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")


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
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": [clean_tree]}
    logger.info("[Tool:match_outline_template] 选中: %s (score=%.3f)",
                selected.get("scene_name"), selected.get("_score", 0))
    return {
        "status": "pending_confirm",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "md_with_ids": to_markdown_with_ids(wrapped),
        "scene_name": selected.get("scene_name", ""),
        "reason": "",
    }


def load_template_outline(scene_name: str) -> dict:
    """
    Load the full outline for a specific template by its scene_name.

    Skips vector search and LLM judgment — direct lookup by name.
    Use when the user has already seen a candidate list and wants to
    preview one specific template's structure.

    Returns:
        status="success"   — outline_tree / markdown / md_with_ids populated
        status="not_found" — no template with that scene_name exists
    """
    logger.info("[Tool:load_template_outline] scene_name=%r", scene_name)

    if not os.path.isdir(_TEMPLATE_DIR):
        return _not_found("模板目录不存在")

    for path in sorted(Path(_TEMPLATE_DIR).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                t = json.load(f)
        except Exception:
            continue
        if t.get("scene_name") == scene_name:
            raw_tree = t.get("outline", {})
            if not raw_tree:
                return _not_found(f"模板「{scene_name}」缺少 outline 字段")
            clean_tree = to_clean_json(raw_tree)
            wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": [clean_tree]}
            logger.info("[Tool:load_template_outline] 已加载: %s", scene_name)
            return {
                "status": "success",
                "outline_tree": wrapped,
                "markdown": to_markdown(wrapped),
                "md_with_ids": to_markdown_with_ids(wrapped),
                "scene_name": scene_name,
            }

    return _not_found(f"未找到名为「{scene_name}」的模板")


def _not_found(reason: str) -> dict:
    return {"status": "not_found", "outline_tree": {}, "markdown": "",
            "md_with_ids": "", "scene_name": "", "reason": reason}
