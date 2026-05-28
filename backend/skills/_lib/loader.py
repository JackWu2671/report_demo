"""
loader.py — Step 1: 加载 FAISS 索引和 JSON 知识图谱。

输入: 无（从环境变量读取路径）
输出: (FAISSService, nodes_dict, children_map)
  - FAISSService  : 已载入索引的检索服务
  - nodes_dict    : {node_id -> node_dict}，全量节点查找表
  - children_map  : {parent_id -> [child_id, ...]}，父子关系映射
"""

import json
import logging
import os
import sys

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(_BACKEND_DIR, "data")
_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")


def _build_index_if_missing() -> None:
    """索引文件不存在时自动构建，省去手动跑 build_index.py。"""
    index_path  = os.path.join(_DATA_DIR, "faiss.index")
    id_map_path = os.path.join(_DATA_DIR, "faiss_id_map.json")
    if os.path.exists(index_path) and os.path.exists(id_map_path):
        return

    logger.info("[Step 1] FAISS 索引不存在，开始自动构建…")
    import asyncio
    import sys
    sys.path.insert(0, _BACKEND_DIR)
    from services.embedding_service import EmbeddingService
    import numpy as np

    node_path = os.path.join(_EXPERT_DIR, "node.json")
    with open(node_path, encoding="utf-8") as f:
        nodes = json.load(f)

    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    texts = [n["name"] + " " + " ".join(n.get("keywords", [])) for n in nodes]

    async def _embed():
        return await emb_svc.get_embeddings_batch(texts, batch_size=32)

    embeddings = asyncio.run(_embed())

    os.makedirs(_DATA_DIR, exist_ok=True)
    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.build(nodes, embeddings)
    faiss_svc.save(index_path, id_map_path)
    logger.info("[Step 1] FAISS 索引自动构建完成，共 %d 条向量", faiss_svc.total)


def load_resources() -> tuple[FAISSService, dict, dict]:
    """
    加载 FAISS 索引（data/faiss.index）和 JSON 知识图谱（expert_knowledge/）。
    索引不存在时自动构建。

    Returns:
        faiss_svc    : FAISSService 实例，已载入向量索引
        nodes_dict   : {node_id -> node_dict}，用于按 id 快速查节点
        children_map : {parent_id -> [child_id, ...]}，用于构建子树
    """
    _build_index_if_missing()

    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.load(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )

    with open(os.path.join(_EXPERT_DIR, "node.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(_EXPERT_DIR, "relation.json"), encoding="utf-8") as f:
        relations = json.load(f)

    nodes_dict = {n["id"]: n for n in nodes}
    children_map: dict[str, list[str]] = {}
    for rel in relations:
        children_map.setdefault(rel["parent"], []).append(rel["child"])

    logger.info(
        "[Step 1] 加载完成 — 节点: %d, 关系: %d, FAISS 向量数: %d",
        len(nodes_dict), len(relations), faiss_svc.total,
    )
    return faiss_svc, nodes_dict, children_map
