"""
generate.py — Tool implementation for generate_outline.

Calls the full case_workflow_2 pipeline (Steps 0-10) and returns:
  - outline_tree : dict  (source of truth, stored in agent state)
  - markdown     : str   (human-readable)
  - md_with_ids  : str   (LLM-readable, used as tool result injected into messages)
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

from case_workflow_2.workflow import main as _wf2_main
from outline_utils import to_markdown, to_markdown_with_ids, to_clean_json

logger = logging.getLogger(__name__)


async def generate_outline(question: str) -> dict:
    """
    Run the full case_workflow_2 pipeline and return all three representations.

    Returns:
        {
            "outline_tree": dict,      # clean JSON tree (source of truth)
            "markdown": str,           # human-readable Markdown
            "md_with_ids": str,        # LLM-context Markdown with [Lx id] prefixes
        }
    """
    logger.info("[Tool:generate_outline] question=%r", question)

    raw_tree, _ = await _wf2_main(question)

    if not raw_tree:
        return {
            "outline_tree": {},
            "markdown": f"未能为「{question}」生成大纲，请检查知识库索引。",
            "md_with_ids": "",
        }

    clean_tree = to_clean_json(raw_tree)
    markdown = to_markdown(clean_tree)
    md_with_ids = to_markdown_with_ids(clean_tree)

    logger.info(
        "[Tool:generate_outline] 完成，节点数=%d，Markdown %d 字",
        _count_nodes(clean_tree),
        len(markdown),
    )
    return {
        "outline_tree": clean_tree,
        "markdown": markdown,
        "md_with_ids": md_with_ids,
    }


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))
