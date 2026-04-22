"""
build_index.py — 构建 FAISS 向量索引。

从 expert_knowledge/node.json 读取知识节点，
调用 Embedding 服务获取向量，构建 FAISS 索引并保存到 data/ 目录。

使用方法:
    cd backend
    python scripts/build_index.py

输出:
    data/faiss.index        FAISS 向量索引文件
    data/faiss_id_map.json  节点 ID 映射文件
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPERT_DIR = os.path.join(_BASE_DIR, "expert_knowledge")
_DATA_DIR = os.path.join(_BASE_DIR, "data")


async def build_index() -> None:
    node_path = os.path.join(_EXPERT_DIR, "node.json")
    if not os.path.exists(node_path):
        logger.error(f"找不到节点文件: {node_path}")
        sys.exit(1)

    with open(node_path, encoding="utf-8") as f:
        nodes = json.load(f)
    logger.info(f"读取到 {len(nodes)} 个知识节点")

    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    texts = [
        n["name"] + " " + " ".join(n.get("keywords", []))
        for n in nodes
    ]
    logger.info(f"开始获取 {len(texts)} 个节点的 Embedding...")
    embeddings = await emb_svc.get_embeddings_batch(texts, batch_size=32)
    logger.info(f"Embedding 完成，shape: {embeddings.shape}")

    os.makedirs(_DATA_DIR, exist_ok=True)
    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.build(nodes, embeddings)
    faiss_svc.save(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )
    logger.info(f"✅ 索引构建完成，共 {faiss_svc.total} 条向量")
    logger.info(f"   → {os.path.join(_DATA_DIR, 'faiss.index')}")
    logger.info(f"   → {os.path.join(_DATA_DIR, 'faiss_id_map.json')}")


if __name__ == "__main__":
    asyncio.run(build_index())
