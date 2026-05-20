"""
search_template.py — 模板检索工具。

  search_outline_templates  — 向量检索，返回 top-N 原始候选列表
  load_template_outline     — 按模板 id 直接加载完整大纲

Used by: agent2
"""

import json
import logging
import os
import sys
from pathlib import Path

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.retriever import embed_query
from utils.template_selector import search_templates
from utils.outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

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
                "id": t.get("id", ""),
                "scene_name": t.get("scene_name", ""),
                "summary": t.get("summary", ""),
                "usage_conditions": t.get("usage_conditions", ""),
                "score": round(t.get("_score", 0), 3),
            }
            for t in candidates
        ],
    }


def load_template_outline(template_id: str) -> dict:
    """
    按模板 id 直接加载完整大纲（O(1) 文件查找）。

    Returns:
        status="success"   — outline_tree / markdown / md_with_ids populated
        status="not_found" — 无对应 id 的模板
    """
    logger.info("[Tool:load_template_outline] template_id=%r", template_id)

    if not os.path.isdir(_TEMPLATE_DIR):
        return _not_found("模板目录不存在")

    path = (Path(_TEMPLATE_DIR) / f"{template_id}.json").resolve()
    if not path.is_relative_to(Path(_TEMPLATE_DIR).resolve()):
        return _not_found(f"无效的 template_id: {template_id}")
    if not path.is_file():
        return _not_found(f"未找到 id={template_id} 的模板")

    try:
        with open(path, encoding="utf-8") as f:
            t = json.load(f)
    except Exception as e:
        return _not_found(f"模板文件读取失败: {e}")

    raw_tree = t.get("outline", {})
    if not raw_tree:
        return _not_found(f"模板 id={template_id} 缺少 outline 字段")

    clean_tree = to_clean_json(raw_tree)
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": [clean_tree]}
    scene_name = t.get("scene_name", "")
    logger.info("[Tool:load_template_outline] 已加载: %s (id=%s)", scene_name, template_id)
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "md_with_ids": to_markdown_with_ids(wrapped),
        "scene_name": scene_name,
        "template_id": template_id,
    }


def _not_found(reason: str) -> dict:
    return {"status": "not_found", "outline_tree": {}, "markdown": "",
            "md_with_ids": "", "scene_name": "", "reason": reason}
