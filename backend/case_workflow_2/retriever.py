"""
retriever.py — Step 2 / 3 / 4: 用户问题向量化、FAISS 检索、候选节点路径构建。

Step 2 embed_query       : 调用 Embedding 服务将问题向量化
Step 3 search_nodes      : 在 FAISS 索引中检索相关候选节点
Step 4 build_candidate_paths: 为候选节点补全祖先路径信息
       candidates_to_tree_text: 将候选节点渲染为树状文本，供 LLM 选锚使用
"""

import logging
import os
import sys

import numpy as np

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)


# ── Step 2 ────────────────────────────────────────────────────

async def embed_query(question: str) -> np.ndarray:
    """
    调用 Embedding 服务将用户问题向量化。

    Args:
        question: 用户输入的自然语言问题

    Returns:
        shape (1, dim) 的 float32 ndarray
    """
    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    vec = await emb_svc.get_embedding(question)
    logger.info("[Step 2] 问题向量化完成: '%s'", question[:60])
    return vec


# ── Step 3 ────────────────────────────────────────────────────

def search_nodes(
    query_embedding: np.ndarray,
    faiss_svc: FAISSService,
    top_k: int = 10,
    threshold: float | None = None,
) -> list[dict]:
    """
    在 FAISS 索引中检索与问题最相关的候选节点。

    Args:
        query_embedding : shape (1, dim) 的 float32 ndarray
        faiss_svc       : 已载入索引的 FAISSService 实例
        top_k           : 最大返回数量，默认 10
        threshold       : 最低余弦相似度阈值，None 时读 FAISS_SCORE_THRESHOLD 环境变量

    Returns:
        [{id, name, level, score, ...}, ...]，按相似度降序
    """
    th = threshold if threshold is not None else float(os.getenv("FAISS_SCORE_THRESHOLD", 0.3))
    hits = faiss_svc.search(query_embedding, top_k=top_k, threshold=th)
    logger.info(
        "[Step 3] FAISS 命中 %d 个节点: %s",
        len(hits),
        ", ".join(f"{h['name']}({h['score']:.3f})" for h in hits),
    )
    return hits


# ── Step 4 ────────────────────────────────────────────────────

def build_candidate_paths(
    hits: list[dict],
    nodes_dict: dict,
    children_map: dict,
) -> list[dict]:
    """
    为每个 FAISS 命中节点构建完整祖先路径，供 LLM 理解节点在图中的位置。

    Args:
        hits         : search_nodes() 返回的命中节点列表
        nodes_dict   : {node_id -> node_dict}
        children_map : {parent_id -> [child_id, ...]}

    Returns:
        [{id, name, level, score, path}, ...]
        path 格式: "根节点名 > 中间节点名 > ... > 当前节点名"
    """
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
        candidates.append(
            {
                "id": hit["id"],
                "name": hit["name"],
                "level": hit["level"],
                "score": hit["score"],
                "path": " > ".join(chain),
            }
        )

    logger.info(
        "[Step 4] %d 个候选节点:\n%s",
        len(candidates),
        "\n".join(
            f"  L{c['level']} {c['name']} | {c['path']} | score={c['score']:.3f}"
            for c in candidates
        ),
    )
    return candidates


def candidates_to_tree_text(candidates: list[dict]) -> str:
    """
    将候选节点渲染为带层级缩进的树状文本，供 LLM 选锚节点时使用。

    ★ 标记 FAISS 命中节点，无★ 的中间节点仅提供祖先路径上下文。
    每个节点格式: [L{level} {id}] {name} ★（命中时）

    Args:
        candidates: build_candidate_paths() 返回的候选节点列表

    Returns:
        多行缩进字符串，反映候选节点在知识图谱中的树状层级
    """
    hit_ids = {c["id"] for c in candidates}
    id_by_name: dict[str, str] = {c["name"]: c["id"] for c in candidates}
    level_by_name: dict[str, int] = {c["name"]: c["level"] for c in candidates}

    # 从 path 字符串还原树结构: name -> {id, level, children}
    tree: dict[str, dict] = {}
    roots: list[str] = []

    for c in candidates:
        parts = [p.strip() for p in c["path"].split(">")]
        for i, name in enumerate(parts):
            if name not in tree:
                tree[name] = {
                    "id": id_by_name.get(name),
                    "level": level_by_name.get(name, i + 1),
                    "children": [],
                }
            if i > 0:
                parent_name = parts[i - 1]
                if name not in tree[parent_name]["children"]:
                    tree[parent_name]["children"].append(name)
            elif name not in roots:
                roots.append(name)

    lines: list[str] = []

    def _render(name: str, depth: int) -> None:
        node = tree[name]
        indent = "  " * depth
        id_str = f" {node['id']}" if node["id"] else ""
        marker = " ★" if node["id"] in hit_ids else ""
        lines.append(f"{indent}[L{node['level']}{id_str}] {name}{marker}")
        for child in node["children"]:
            _render(child, depth + 1)

    for root in roots:
        _render(root, 0)

    return "\n".join(lines)
