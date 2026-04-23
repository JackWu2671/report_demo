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

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(_BACKEND_DIR, "data")
_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")


def load_resources() -> tuple[FAISSService, dict, dict]:
    """
    加载 FAISS 索引（data/faiss.index）和 JSON 知识图谱（expert_knowledge/）。

    Returns:
        faiss_svc    : FAISSService 实例，已载入向量索引
        nodes_dict   : {node_id -> node_dict}，用于按 id 快速查节点
        children_map : {parent_id -> [child_id, ...]}，用于构建子树
    """
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
