"""
embedding_service_test.py — 测试 EmbeddingService 单条和批量向量接口。

使用方法:
    cd backend
    python services/services_test/embedding_service_test.py
    python services/services_test/embedding_service_test.py "自定义文本"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

from services.embedding_service import EmbeddingService


def _build_service() -> EmbeddingService:
    base_url = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
    dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    return EmbeddingService(base_url=base_url, dim=dim)


async def test_single(text: str) -> None:
    svc = _build_service()
    print(f"地址  : {svc.base_url}")
    print(f"模型  : {svc.model}")
    print(f"维度  : {svc.dim}")
    print(f"文本  : {text}")
    print("-" * 50)

    vec = await svc.get_embedding(text)
    print(f"shape : {vec.shape}")
    print(f"norm  : {float((vec ** 2).sum() ** 0.5):.6f}  (归一化后应≈1.0)")
    print(f"前5维 : {vec[0, :5].tolist()}")


async def test_batch() -> None:
    svc = _build_service()
    texts = [
        "用户留存分析",
        "商品转化漏斗",
        "GMV 同比增长",
        "客单价分布",
    ]
    print("-" * 50)
    print(f"批量文本数: {len(texts)}")

    mat = await svc.get_embeddings_batch(texts, batch_size=2)
    print(f"矩阵 shape: {mat.shape}  (期望 {len(texts)} x {svc.dim})")

    import numpy as np
    norms = np.linalg.norm(mat, axis=1)
    print(f"各行 norm : {norms.tolist()}  (均应≈1.0)")

    sim = float(mat[0] @ mat[1])
    print(f"文本0 · 文本1 余弦相似度: {sim:.4f}")


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "用户活跃度分析报告"

    async def main():
        await test_single(text)
        print()
        await test_batch()

    asyncio.run(main())
