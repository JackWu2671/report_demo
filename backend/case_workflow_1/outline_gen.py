"""
outline_gen.py — Step 3: LLM 基于专家文本和知识库树生成带 id 标注的 Markdown 大纲。

输出格式: 每个节点携带 [id] 或 [new] 标注
  - [L2_001] 等真实 id  → 引用现有知识库节点
  - [new]              → 专家描述中出现但知识库没有的新概念（仅 L2~L4，L5 不允许）

示例:
  ## [L2_001] fgOTN部署
  ### [new] IP承载网分析
  #### [L4_001] 企业分布分析
  ##### [L5_001] 企业行业分布
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


async def generate_outline(expert_text: str, tree_text: str) -> str:
    """
    LLM 基于专家描述和完整知识库树生成带 id 标注的 Markdown 大纲。

    Args:
        expert_text : 专家输入的原始文本
        tree_text   : build_kb_tree_text() 生成的完整 KB 树状文本（含 ★ 标记）

    Returns:
        带 [id]/[new] 标注的 Markdown 大纲字符串
    """
    llm = LLMService.from_env()

    user_content = (
        f"## 专家描述\n{expert_text}\n\n"
        f"## 知识库节点树\n{tree_text}"
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
