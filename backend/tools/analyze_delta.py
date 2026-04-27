"""
analyze_delta.py — analyze_kb_delta tool implementation (Step 5).

Runs case_workflow_1 Step 5: LLM compares the expert's outline against the
existing KB tree and summarises what the expert contributes beyond the KB.

Kept separate from analyze_expert_knowledge because:
  - It's optional (expert may just want the outline, not the delta report)
  - It's another LLM call — expensive to run every time

Used by: agent1
"""

import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)
_WF1_DIR = os.path.join(_BACKEND_DIR, "case_workflow_1")

for _p in [_BACKEND_DIR, _WF1_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from outline_gen import generate_delta

logger = logging.getLogger(__name__)


async def analyze_kb_delta(expert_text: str, tree_text: str, outline_md_annotated: str) -> dict:
    """
    Compare expert outline against KB tree and summarise the delta.

    Args:
        expert_text           : original expert input
        tree_text             : KB tree from Step 2 (stored in Agent1Memory)
        outline_md_annotated  : [id]/[new]-marked outline from Step 3

    Returns:
        {status: "success"|"error", delta_text, message}
    """
    if not tree_text or not outline_md_annotated:
        return {"status": "error", "delta_text": "",
                "message": "请先调用 analyze_expert_knowledge 生成大纲。"}

    logger.info("[Tool:analyze_kb_delta]")
    delta_text = await generate_delta(expert_text, tree_text, outline_md_annotated)
    return {"status": "success", "delta_text": delta_text, "message": ""}
