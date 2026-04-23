"""
extractor.py — Step 1: LLM 从专家输入文本中抽取关键词、摘要和场景名称。

输出: {"keywords": [...], "summary": str, "scene_name": str}
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

_PROMPT_DIR = Path(__file__).parent / "prompts"
EXTRACT_PROMPT: str = (_PROMPT_DIR / "extract.txt").read_text(encoding="utf-8")


async def extract_from_expert(expert_text: str) -> dict:
    """
    LLM 从专家输入文本中抽取结构化信息。

    Args:
        expert_text: 专家输入的自然语言场景描述

    Returns:
        {"keywords": [...], "summary": "...", "scene_name": "..."}
    """
    llm = LLMService.from_env()

    messages = [
        {"role": "system", "content": EXTRACT_PROMPT},
        {"role": "user", "content": expert_text},
    ]

    logger.info(
        "[Step 1] Extract Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 1] LLM 关键词抽取 ↓", flush=True)
    print("-" * 50, flush=True)

    answer = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 1] LLM 完整输出:\n%s", answer)

    result = LLMService._parse_json(answer)
    logger.info(
        "[Step 1] 抽取完成 — 场景: %s | 关键词: %s",
        result.get("scene_name"),
        result.get("keywords"),
    )
    return result
