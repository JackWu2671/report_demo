"""
template_selector.py — 模板向量检索。

search_templates : 实时 embedding，余弦相似度检索 top-K 模板候选
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

from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")


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
            logger.warning("[template_selector] 加载模板失败 %s: %s", path.name, e)
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
        logger.info("[template_selector] templates/ 目录为空，跳过模板检索")
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
        "[template_selector] top%d: %s",
        len(results),
        [(t["scene_name"], round(t["_score"], 3)) for t in results],
    )
    return results
