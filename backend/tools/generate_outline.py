"""
generate_outline.py — generate_outline tool implementation.

LLM generates a report outline in md_with_ids format based on the
KB graph tree returned by search_graph_tree. The output is parsed back
into outline_tree via from_md_with_ids().

Used by: agent2, workflow.py (Step 3)
"""

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

from services.llm_service import LLMService
from outline_utils import from_md_with_ids, to_markdown, to_markdown_with_ids, to_clean_json

logger = logging.getLogger(__name__)

_PROMPT = (Path(_WF2_DIR) / "prompts" / "generate_outline.txt").read_text(encoding="utf-8")


async def generate_outline(question: str, tree_text: str) -> dict:
    """
    Generate a report outline from KB graph tree using LLM.

    LLM outputs md_with_ids format → parsed into outline_tree.

    Args:
        question  : user's analysis question
        tree_text : formatted KB graph tree text from search_graph_tree

    Returns:
        {status: "success"|"error", outline_tree, markdown, md_with_ids, message}
    """
    logger.info("[Tool:generate_outline] question=%r", question)

    llm = LLMService.from_env()
    messages = [
        {"role": "system", "content": _PROMPT},
        {"role": "user", "content": f"## 用户需求\n{question}\n\n## 知识图谱节点\n{tree_text}"},
    ]

    logger.info(
        "[Tool:generate_outline] Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    raw = await llm.complete(messages)
    logger.info("[Tool:generate_outline] LLM 输出:\n%s", raw)

    tree = from_md_with_ids(raw)
    if tree is None:
        return {
            "status": "error",
            "outline_tree": {},
            "markdown": "",
            "md_with_ids": "",
            "message": f"大纲解析失败，LLM 原始输出：\n{raw}",
        }

    clean_tree = to_clean_json(tree)
    logger.info("[Tool:generate_outline] 解析成功，根节点: %s", clean_tree.get("name"))
    return {
        "status": "success",
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "message": "",
    }
