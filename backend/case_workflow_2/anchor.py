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
from retriever import candidates_to_tree_text

logger = logging.getLogger(__name__)

# 在模块加载时读取 prompt 文件，避免重复 IO
_PROMPT_DIR = Path(__file__).parent / "prompts"
ANCHOR_PROMPT: str = (_PROMPT_DIR / "anchor.txt").read_text(encoding="utf-8")


async def select_anchor(question: str, candidates: list[dict]) -> dict:
    """
    调用 LLM，从候选节点树中选出最符合用户核心意图的锚节点。

    候选以树状结构（★ 标记命中节点）呈现给 LLM，帮助 LLM 感知父子覆盖关系。
    若 LLM 调用失败，回退到 FAISS 分数最高的候选节点。

    Args:
        question   : 用户的自然语言问题
        candidates : retriever.build_candidate_paths() 返回的候选列表

    Returns:
        anchor dict，含 selected_id / selected_name / selected_path / level / reason
    """
    llm = LLMService.from_env()
    tree_text = candidates_to_tree_text(candidates)

    messages = [
        {"role": "system", "content": ANCHOR_PROMPT},
        {"role": "user", "content": f"## 候选（树状结构）\n{tree_text}\n\n## 问题\n{question}"},
    ]

    logger.info(
        "[Step 5] Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 5] LLM 流式输出 ↓", flush=True)
    print("-" * 50, flush=True)

    try:
        answer = await llm.stream_and_collect(messages)
        print("\n" + "-" * 50, flush=True)
        logger.info("[Step 5] LLM 完整输出:\n%s", answer)
        anchor = LLMService._parse_json(answer)
    except Exception as e:
        logger.warning("[Step 5] LLM 选锚失败，回退到 score 最高的候选: %s", e)
        f = candidates[0]
        anchor = {
            "selected_id": f["id"],
            "selected_name": f["name"],
            "selected_path": f["path"],
            "level": f["level"],
            "reason": "fallback",
        }

    logger.info(
        "[Step 5] 选锚: '%s' (L%s), reason=%s",
        anchor.get("selected_name"),
        anchor.get("level"),
        anchor.get("reason", ""),
    )
    return anchor
