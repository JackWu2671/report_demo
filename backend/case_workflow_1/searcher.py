"""
searcher.py — Step 2: 双路 FAISS 检索，合并结果并构建候选节点祖先路径。

双路检索：
  路径 A: 关键词拼接字符串 → Embedding → FAISS
  路径 B: 摘要文本 → Embedding → FAISS
两路结果按 score 去重合并（取高分），再构建每个候选节点的祖先路径。
"""

import logging
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)


async def dual_search(
    extraction: dict,
    faiss_svc: FAISSService,
    nodes_dict: dict,
    children_map: dict,
) -> list[dict]:
    """
    双路 FAISS 检索，合并去重后返回候选节点列表（含祖先路径）。

    Args:
        extraction   : extract_from_expert() 的输出 {keywords, summary, scene_name}
        faiss_svc    : 已载入索引的 FAISSService 实例
        nodes_dict   : {node_id -> node_dict}
        children_map : {parent_id -> [child_id, ...]}

    Returns:
        [{id, name, level, score, path}, ...] 按分数降序
    """
    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    threshold = float(os.getenv("FAISS_SCORE_THRESHOLD", 0.3))

    kw_text = " ".join(extraction.get("keywords", []))
    summary_text = extraction.get("summary", "")

    # 两路 embedding（串行，避免连接复用问题）
    vec_kw = await emb_svc.get_embedding(kw_text)
    vec_sum = await emb_svc.get_embedding(summary_text)

    hits_kw = faiss_svc.search(vec_kw, top_k=8, threshold=threshold)
    hits_sum = faiss_svc.search(vec_sum, top_k=8, threshold=threshold)

    # 合并去重，相同节点保留更高分数
    merged: dict[str, dict] = {}
    for h in hits_kw + hits_sum:
        if h["id"] not in merged or h["score"] > merged[h["id"]]["score"]:
            merged[h["id"]] = h
    hits = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    logger.info(
        "[Step 2] 双路检索命中 %d 个节点 (KW路 %d, 摘要路 %d): %s",
        len(hits), len(hits_kw), len(hits_sum),
        ", ".join(f"{h['name']}({h['score']:.3f})" for h in hits),
    )

    return _build_candidate_paths(hits, nodes_dict, children_map)


def _build_candidate_paths(
    hits: list[dict],
    nodes_dict: dict,
    children_map: dict,
) -> list[dict]:
    """为每个命中节点补全祖先路径，格式与 case_workflow_2/retriever.py 一致。"""
    parent_map: dict[str, str] = {}
    for pid, cids in children_map.items():
        for cid in cids:
            parent_map[cid] = pid

    candidates = []
    for hit in hits:
        chain: list[str] = []
        cur: str | None = hit["id"]
        while cur and cur in nodes_dict:
            chain.append(nodes_dict[cur]["name"])
            cur = parent_map.get(cur)
        chain.reverse()
        candidates.append({
            "id": hit["id"],
            "name": hit["name"],
            "level": hit["level"],
            "score": hit["score"],
            "path": " > ".join(chain),
        })

    logger.info(
        "[Step 2] %d 个候选节点:\n%s",
        len(candidates),
        "\n".join(
            f"  L{c['level']} {c['name']} | {c['path']} | score={c['score']:.3f}"
            for c in candidates
        ),
    )
    return candidates
