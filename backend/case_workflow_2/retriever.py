"""
retriever.py — Step 2 / 3 / 4: 用户问题向量化、FAISS 检索、候选节点路径构建。

Step 2 embed_query          : 调用 Embedding 服务将问题向量化
Step 3 search_nodes         : 在 FAISS 索引中检索相关候选节点
Step 4 build_candidate_paths: 为候选节点补全祖先路径信息
       candidates_to_tree_text: 将候选节点渲染为树状文本，供 LLM 选锚使用

search_graph_tree           : 组合函数，embed → search → build paths → 返回树状 dict 列表
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
from loader import load_resources

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


# ── 组合接口 ──────────────────────────────────────────────────

async def search_graph_tree(question: str) -> tuple[list[dict], list[dict]]:
    """
    搜索知识图谱并返回树状结构，供 LLM 生成大纲时作为上下文。

    流程: embed_query → search_nodes → build_candidate_paths → 组装树 dict

    每个节点格式:
        {
            "id"      : str | None,   # 知识图谱节点 id（含祖先节点，名称不在图谱中时为 None）
            "name"    : str,
            "level"   : int,
            "hit"     : bool,         # True = FAISS 直接命中
            "score"   : float | None, # FAISS 相似度，祖先节点为 None
            "children": [...]
        }

    Args:
        question: 用户的自然语言问题

    Returns:
        根节点列表（通常 1～3 个），每个根节点下挂完整子树
    """
    faiss_svc, nodes_dict, children_map = load_resources()
    query_embedding = await embed_query(question)
    hits = search_nodes(query_embedding, faiss_svc)
    if not hits:
        logger.info("[search_graph_tree] 无命中节点，返回空树")
        return []

    candidates = build_candidate_paths(hits, nodes_dict, children_map)
    hit_ids = {c["id"] for c in candidates}
    score_by_id = {c["id"]: c["score"] for c in candidates}

    # 反查表：name → (id, description)（从完整 nodes_dict 中建，覆盖祖先节点）
    name_to_id = {n["name"]: n["id"] for n in nodes_dict.values()}
    name_to_desc = {n["name"]: n.get("description", "") for n in nodes_dict.values()}

    # 从 path 字符串还原 name → {id, level, children_names} 映射
    name_meta: dict[str, dict] = {}
    roots: list[str] = []

    for c in candidates:
        parts = [p.strip() for p in c["path"].split(">")]
        for i, name in enumerate(parts):
            if name not in name_meta:
                name_meta[name] = {
                    "id": name_to_id.get(name),
                    "description": name_to_desc.get(name, ""),
                    "level": c["level"] if name == c["name"] else i + 1,
                    "children_names": [],
                }
            if i > 0:
                parent = parts[i - 1]
                if name not in name_meta[parent]["children_names"]:
                    name_meta[parent]["children_names"].append(name)
            elif name not in roots:
                roots.append(name)

    # 补全：对每个 FAISS 命中节点，递归展开所有后代节点
    # 避免 LLM 因阈值过滤漏掉未命中但实际存在的子节点
    def _expand_all(node_id: str, parent_name: str) -> None:
        for child_id in children_map.get(node_id, []):
            child_node = nodes_dict.get(child_id)
            if not child_node:
                continue
            child_name = child_node["name"]
            if child_name not in name_meta:
                name_meta[child_name] = {
                    "id": child_id,
                    "description": child_node.get("description", ""),
                    "level": child_node.get("level", 0),
                    "children_names": [],
                }
            if child_name not in name_meta[parent_name]["children_names"]:
                name_meta[parent_name]["children_names"].append(child_name)
            _expand_all(child_id, child_name)

    for hit_id in hit_ids:
        hit_node = nodes_dict.get(hit_id, {})
        hit_name = hit_node.get("name", "")
        if hit_name not in name_meta:
            continue
        _expand_all(hit_id, hit_name)

    def _to_dict(name: str) -> dict:
        meta = name_meta[name]
        node_id = meta["id"]
        return {
            "id": node_id,
            "name": name,
            "level": meta["level"],
            "description": meta.get("description", ""),
            "hit": node_id in hit_ids,
            "score": score_by_id.get(node_id),
            "children": [_to_dict(child) for child in meta["children_names"]],
        }

    tree = [_to_dict(r) for r in roots]
    total = sum(_count_tree(r) for r in tree)
    hit_count = sum(1 for c in candidates)
    logger.info("[search_graph_tree] 返回 %d 棵根树，共 %d 个节点（%d 命中 + %d 补全）",
                len(tree), total, hit_count, total - hit_count)
    return tree, candidates


def _count_tree(node: dict) -> int:
    return 1 + sum(_count_tree(c) for c in node.get("children", []))
