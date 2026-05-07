"""
anchor.py — Step 5: 使用 LLM 从候选节点中选出最符合用户意图的锚节点。

输入: question (str), candidates (list[dict] 来自 retriever.build_candidate_paths)
输出: anchor dict
  {
    "selected_id"   : str,   # 锚节点 id
    "selected_name" : str,   # 锚节点名称
    "selected_path" : str,   # 祖先路径
    "level"         : int,   # 层级
    "reason"        : str,   # 选择理由
  }

Prompt 从 prompts/anchor.txt 加载，候选以树状结构传给 LLM（见 retriever.candidates_to_tree_text）。
"""

import logging
import os
import sys
from pathlib import Path

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService

logger = logging.getLogger(__name__)

# 在模块加载时读取 prompt 文件，避免重复 IO
_PROMPT_DIR = Path(__file__).parent / "prompts"
ANCHOR_PROMPT: str = (_PROMPT_DIR / "anchor.txt").read_text(encoding="utf-8")


async def select_anchor(question: str, tree_text: str) -> dict:
    """
    调用 LLM，从知识图谱树中选出最符合用户核心意图的锚节点。

    Args:
        question  : 用户的自然语言问题
        tree_text : search_graph_tree 返回的完整树文本（★ 标记 FAISS 命中节点）

    Returns:
        anchor dict，含 selected_id / selected_name / selected_path / level / reason
    """
    llm = LLMService.from_env()
    display_tree = tree_text

    messages = [
        {"role": "system", "content": ANCHOR_PROMPT},
        {"role": "user", "content": f"## 候选（树状结构）\n{display_tree}\n\n## 问题\n{question}"},
    ]

    logger.info(
        "[Step 5] Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    answer = await llm.complete(messages)
    logger.info("[Step 5] LLM 完整输出:\n%s", answer)
    anchor = LLMService._parse_json(answer)

    logger.info(
        "[Step 5] 选锚: '%s' (L%s), reason=%s",
        anchor.get("selected_name"),
        anchor.get("level"),
        anchor.get("reason", ""),
    )
    return anchor
