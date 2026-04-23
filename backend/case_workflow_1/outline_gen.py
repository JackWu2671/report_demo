"""
outline_gen.py — Step 3: LLM 基于专家文本和知识库候选节点生成 Markdown 大纲。

输出: Markdown 格式的报告大纲字符串
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
OUTLINE_PROMPT: str = (_PROMPT_DIR / "outline.txt").read_text(encoding="utf-8")


def _candidates_to_text(candidates: list[dict]) -> str:
    """将候选节点渲染为带路径的列表文本，供 LLM 参考。"""
    if not candidates:
        return "（未检索到相关节点）"
    lines = []
    for c in candidates:
        lines.append(f"  ★ [L{c['level']} {c['id']}] {c['name']}  路径: {c['path']}")
    return "\n".join(lines)


async def generate_outline(expert_text: str, candidates: list[dict]) -> str:
    """
    LLM 基于专家描述和知识库候选节点生成 Markdown 大纲。

    Args:
        expert_text : 专家输入的原始文本
        candidates  : dual_search() 返回的候选节点列表

    Returns:
        Markdown 格式大纲字符串
    """
    llm = LLMService.from_env()
    candidates_text = _candidates_to_text(candidates)

    user_content = (
        f"## 专家描述\n{expert_text}\n\n"
        f"## 知识库相关节点\n{candidates_text}"
    )

    messages = [
        {"role": "system", "content": OUTLINE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "[Step 3] Outline Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 3] LLM 大纲生成 ↓", flush=True)
    print("-" * 50, flush=True)

    outline_md = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 3] 大纲生成完成 (%d 字符)", len(outline_md))
    return outline_md
