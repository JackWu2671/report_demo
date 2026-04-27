"""
search_template.py — search_outline_template tool implementation.

Encapsulates: embed_query → vector search → LLM judge (select_template).
Returns clean {status, outline_tree, markdown, md_with_ids}.
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

from retriever import embed_query
from template_selector import search_templates, select_template
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)


async def search_outline_template(question: str) -> dict:
    """
    Search pre-built templates and let an internal LLM judge if any match.

    Returns:
        {status: "found"|"not_found", outline_tree, markdown, md_with_ids,
         scene_name, reason}
    """
    logger.info("[Tool:search_outline_template] question=%r", question)

    query_embedding = await embed_query(question)
    candidates = await search_templates(query_embedding, top_k=10)

    if not candidates:
        return _not_found("模板库为空，请走知识库生成")

    selected = await select_template(question, candidates)
    if not selected:
        return _not_found("未找到与需求匹配的预制大纲")

    raw_tree = selected.get("outline", {})
    if not raw_tree:
        return _not_found("模板存在但缺少 outline 字段")

    clean_tree = to_clean_json(raw_tree)
    logger.info("[Tool:search_outline_template] 命中: %s (score=%.3f)",
                selected.get("scene_name"), selected.get("_score", 0))
    return {
        "status": "found",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "scene_name": selected.get("scene_name", ""),
        "reason": "",
    }


def _not_found(reason: str) -> dict:
    return {"status": "not_found", "outline_tree": {}, "markdown": "",
            "md_with_ids": "", "scene_name": "", "reason": reason}
