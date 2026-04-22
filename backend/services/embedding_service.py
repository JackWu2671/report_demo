import logging

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, base_url: str, model: str = "bge-m3", dim: int = 1024):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim

    async def get_embedding(self, text: str) -> np.ndarray:
        """获取单条文本的向量，返回 shape (1, dim) 的 float32 ndarray。"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": text},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Embedding 失败: status={resp.status}")
                data = await resp.json()
                vec = np.array(data["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
                return vec.reshape(1, -1)

    async def get_embeddings_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> np.ndarray:
        """批量获取向量，返回 shape (N, dim) 的 float32 ndarray。"""
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": batch},
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        raise RuntimeError(f"Embedding 批量失败: status={resp.status}")
                    data = await resp.json()
                    for item in data["data"]:
                        vec = np.array(item["embedding"], dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        if norm > 0:
                            vec /= norm
                        all_embeddings.append(vec)
            logger.info(f"Embedding: {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(all_embeddings, dtype=np.float32)
