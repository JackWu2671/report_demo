import json
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSService:
    """FAISS 向量索引服务（无 Neo4j 依赖）。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.index: faiss.IndexFlatIP | None = None
        self.id_map: list[dict] = []

    def build(self, nodes: list[dict], embeddings: np.ndarray) -> None:
        """
        从节点列表和对应向量矩阵构建索引。

        Args:
            nodes     : 节点列表，每个元素至少包含 {id, name, level}
            embeddings: shape (N, dim) 的 float32 ndarray，与 nodes 一一对应
        """
        vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)
        self.id_map = list(nodes)
        logger.info(f"FAISS 索引构建完成: {self.index.ntotal} 条向量")

    def save(self, index_path: str, id_map_path: str) -> None:
        """将索引和 ID 映射持久化到文件。"""
        faiss.write_index(self.index, index_path)
        with open(id_map_path, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f, ensure_ascii=False, indent=2)
        logger.info(f"FAISS 索引已保存: {index_path}")

    def load(self, index_path: str, id_map_path: str) -> None:
        """从文件加载索引和 ID 映射。"""
        self.index = faiss.read_index(index_path)
        with open(id_map_path, "r", encoding="utf-8") as f:
            self.id_map = json.load(f)
        logger.info(f"FAISS 索引已加载: {self.index.ntotal} 条向量")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10, threshold: float = 0.3
    ) -> list[dict]:
        """
        向量检索。

        Args:
            query_embedding: shape (1, dim) 的 float32 ndarray
            top_k          : 返回的最大候选数
            threshold      : 最低相似度分数（余弦相似度，范围 0~1）

        Returns:
            [{id, name, level, score, ...}, ...]  按分数降序
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        qe = query_embedding.copy().astype(np.float32)
        faiss.normalize_L2(qe)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(qe, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or float(score) < threshold:
                continue
            node = dict(self.id_map[idx])
            node["score"] = float(score)
            results.append(node)
        return results

    @property
    def total(self) -> int:
        return self.index.ntotal if self.index else 0
