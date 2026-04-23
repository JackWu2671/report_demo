"""
searcher.py — Step 2: 双路 FAISS 检索，合并结果并构建候选节点祖先路径。

双路检索：
  路径 A: 关键词拼接字符串 → Embedding → FAISS
  路径 B: 摘要文本 → Embedding → FAISS
两路结果按 score 去重合并（取高分），再构建每个候选节点的祖先路径。

build_kb_tree_text(): 将整个知识库渲染为树状文本（★ 标记命中节点），供 LLM 生成带 id 大纲用。
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
    双路 FAISS 检索，合并去重后返回命中节点列表（含祖先路径）。

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

    vec_kw = await emb_svc.get_embedding(kw_text)
    vec_sum = await emb_svc.get_embedding(summary_text)

    hits_kw = faiss_svc.search(vec_kw, top_k=8, threshold=threshold)
    hits_sum = faiss_svc.search(vec_sum, top_k=8, threshold=threshold)

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

    return hits


def build_kb_tree_text(
    hit_ids: set,
    nodes_dict: dict,
    children_map: dict,
) -> str:
    """
    将整个知识库渲染为带层级缩进的树状文本，★ 标记 FAISS 命中节点。

    供 LLM 生成带 id 标注大纲时参考完整的知识结构和节点 id。

    Args:
        hit_ids      : FAISS 命中节点的 id 集合（用于标 ★）
        nodes_dict   : {node_id -> node_dict}
        children_map : {parent_id -> [child_id, ...]}

    Returns:
        多行缩进字符串，每行格式: [node_id L层级] 节点名 ★（命中时）
    """
    all_child_ids = {cid for cids in children_map.values() for cid in cids}
    root_ids = sorted(nid for nid in nodes_dict if nid not in all_child_ids)

    lines: list[str] = []

    def _render(node_id: str, depth: int) -> None:
        node = nodes_dict.get(node_id)
        if not node:
            return
        indent = "  " * depth
        marker = " ★" if node_id in hit_ids else ""
        lines.append(f"{indent}[{node_id} L{node['level']}] {node['name']}{marker}")
        for child_id in children_map.get(node_id, []):
            _render(child_id, depth + 1)

    for root_id in root_ids:
        _render(root_id, 0)

    return "\n".join(lines)

