"""
template_selector.py — Step 0: 检索已沉淀模板，LLM 决策是否复用。

Step 0a search_templates : 实时 embedding，余弦相似度检索 top-K 模板
Step 0b select_template  : LLM 从候选模板中决策是否复用，返回选中模板或 None
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")
_PROMPT_DIR = Path(__file__).parent / "prompts"
SELECT_PROMPT: str = (_PROMPT_DIR / "select.txt").read_text(encoding="utf-8")


def _load_templates() -> list[dict]:
    """加载 templates/ 目录下所有 JSON 模板文件。"""
    if not os.path.isdir(_TEMPLATE_DIR):
        return []
    templates = []
    for path in sorted(Path(_TEMPLATE_DIR).glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                t = json.load(f)
            templates.append(t)
        except Exception as e:
            logger.warning("[Step 0] 加载模板失败 %s: %s", path.name, e)
    return templates


async def search_templates(
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """
    用查询向量在所有模板中检索最相似的 top_k 个。

    检索文本 = scene_name + summary + usage_conditions（拼接）。
    向量化在本函数内实时完成，无需预建索引。

    Args:
        query_embedding : embed_query() 返回的 shape (1, dim) 归一化向量
        top_k           : 最多返回几个候选模板

    Returns:
        按相似度降序排列的模板列表（每个 dict 含原始字段 + _score）
    """
    templates = _load_templates()
    if not templates:
        logger.info("[Step 0] templates/ 目录为空，跳过模板检索")
        return []

    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )

    texts = [
        f"{t.get('scene_name', '')} {t.get('summary', '')} {t.get('usage_conditions', '')}"
        for t in templates
    ]
    template_vecs = await emb_svc.get_embeddings_batch(texts)  # (N, dim)，已归一化

    # 向量已归一化，点积 = 余弦相似度
    scores = (template_vecs @ query_embedding.T).flatten()  # (N,)

    top_indices = np.argsort(scores)[::-1][: min(top_k, len(templates))]
    results = []
    for i in top_indices:
        t = dict(templates[i])
        t["_score"] = float(scores[i])
        results.append(t)

    logger.info(
        "[Step 0] 模板检索 top%d: %s",
        len(results),
        [(t["scene_name"], round(t["_score"], 3)) for t in results],
    )
    return results


async def select_template(
    question: str,
    candidates: list[dict],
) -> dict | None:
    """
    LLM 从候选模板中决策：是否复用已有模板。

    Args:
        question   : 用户原始问题
        candidates : search_templates() 返回的候选列表

    Returns:
        选中的模板 dict（含 outline 字段），或 None（走 KB 实时生成）
    """
    if not candidates:
        return None

    llm = LLMService.from_env()

    candidates_text = "\n".join(
        f"{i + 1}. 【{t['scene_name']}】\n"
        f"   摘要：{t.get('summary', '')}\n"
        f"   使用条件：{t.get('usage_conditions', '')}"
        for i, t in enumerate(candidates)
    )

    user_content = f"## 用户问题\n{question}\n\n## 候选模板\n{candidates_text}"

    messages = [
        {"role": "system", "content": SELECT_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "[Step 0] Select Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 0] LLM 模板选择 ↓", flush=True)
    print("-" * 50, flush=True)

    answer = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 0] LLM 完整输出:\n%s", answer)

    result = LLMService._parse_json(answer)

    if result.get("use_template"):
        scene_name = result.get("selected", "")
        matched = next((t for t in candidates if t.get("scene_name") == scene_name), None)
        if matched:
            logger.info(
                "[Step 0] 使用模板: %s (score=%.3f) | 原因: %s",
                scene_name, matched.get("_score", 0), result.get("reason", ""),
            )
            return matched
        logger.warning("[Step 0] LLM 选择的模板 '%s' 不在候选列表，改走 KB 生成", scene_name)

    logger.info("[Step 0] 不使用已有模板，走 KB 实时生成 | 原因: %s", result.get("reason", ""))
    return None
