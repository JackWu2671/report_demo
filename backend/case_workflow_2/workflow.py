"""
case_workflow_2 — 基于 JSON 知识图谱 + FAISS 的报告大纲生成工作流。

对应 report_system 的 case_2（outline-generate）：
  - 用 expert_knowledge/node.json + relation.json 替代 Neo4j
  - 用 services/faiss_service.py 做向量检索
  - 用 LLM 生成最终大纲文本

使用方法:
    cd backend
    python case_workflow_2/workflow.py "分析企业网络时延问题"
"""

import asyncio
import json
import logging
import os
import sys

import aiohttp
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_EXPERT_DIR = os.path.join(_BASE_DIR, "expert_knowledge")


# ─────────────────────────────────────────────────────────────
# Step 1: 加载资源（FAISS 索引 + JSON 知识图谱）
# ─────────────────────────────────────────────────────────────

def step1_load_resources() -> tuple[FAISSService, dict, dict]:
    """
    加载 FAISS 索引和 JSON 知识图谱。

    Returns:
        faiss_svc   : FAISSService 实例（已载入索引）
        nodes_dict  : {node_id -> node_dict}
        children_map: {parent_id -> [child_id, ...]}
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
        children_map.setdefault(rel["from_id"], []).append(rel["to_id"])

    logger.info(
        f"[Step 1] 加载完成 — 节点: {len(nodes_dict)}, 关系: {len(relations)}, "
        f"FAISS 向量数: {faiss_svc.total}"
    )
    return faiss_svc, nodes_dict, children_map


# ─────────────────────────────────────────────────────────────
# Step 2: 用户问题向量化
# ─────────────────────────────────────────────────────────────

async def step2_embed_query(question: str) -> np.ndarray:
    """
    调用 Embedding 服务将用户问题向量化。

    Returns:
        shape (1, dim) 的 float32 ndarray
    """
    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    vec = await emb_svc.get_embedding(question)
    logger.info(f"[Step 2] 问题向量化完成: '{question[:60]}'")
    return vec


# ─────────────────────────────────────────────────────────────
# Step 3: FAISS 检索最相关节点
# ─────────────────────────────────────────────────────────────

def step3_search_nodes(
    query_embedding: np.ndarray,
    faiss_svc: FAISSService,
    top_k: int = 5,
    threshold: float | None = None,
) -> list[dict]:
    """
    在 FAISS 索引中检索与问题最相关的知识节点。

    Returns:
        [{id, name, level, intro_text, score}, ...]  按分数降序
    """
    th = threshold if threshold is not None else float(os.getenv("FAISS_SCORE_THRESHOLD", 0.3))
    hits = faiss_svc.search(query_embedding, top_k=top_k, threshold=th)
    logger.info(
        f"[Step 3] FAISS 命中 {len(hits)} 个节点: "
        + ", ".join(f"{h['name']}({h['score']:.3f})" for h in hits)
    )
    return hits


# ─────────────────────────────────────────────────────────────
# Step 4: 从命中节点向上找根，再向下构建子树
# ─────────────────────────────────────────────────────────────

def _find_root(node_id: str, children_map: dict) -> str:
    """沿父节点链向上走到最顶层（level-1）节点。"""
    parent_map: dict[str, str] = {}
    for pid, cids in children_map.items():
        for cid in cids:
            parent_map[cid] = pid
    current = node_id
    while current in parent_map:
        current = parent_map[current]
    return current


def _build_subtree(node_id: str, nodes_dict: dict, children_map: dict) -> dict:
    """递归构建以 node_id 为根的子树 dict（含 children 列表）。"""
    node = dict(nodes_dict[node_id])
    node["children"] = [
        _build_subtree(cid, nodes_dict, children_map)
        for cid in children_map.get(node_id, [])
    ]
    return node


def step4_build_subtree(
    hits: list[dict],
    nodes_dict: dict,
    children_map: dict,
) -> dict:
    """
    取得分最高的命中节点，向上找到 Level-1 根节点，
    再递归构建完整子树。

    Returns:
        subtree dict，结构: {id, name, level, intro_text, children: [...]}
    """
    if not hits:
        raise ValueError("FAISS 未命中任何节点，请先运行 build_index.py 构建索引")

    best_id = hits[0]["id"]
    root_id = _find_root(best_id, children_map)
    subtree = _build_subtree(root_id, nodes_dict, children_map)
    logger.info(
        f"[Step 4] 子树根: '{nodes_dict[root_id]['name']}' (id={root_id}), "
        f"命中节点: '{hits[0]['name']}' (score={hits[0]['score']:.3f})"
    )
    return subtree


# ─────────────────────────────────────────────────────────────
# Step 5: 将子树转换为 LLM 上下文文本
# ─────────────────────────────────────────────────────────────

_LEVEL_LABELS = {1: "场景", 2: "子场景", 3: "维度", 4: "评估项", 5: "指标"}


def _subtree_to_text(node: dict, depth: int = 0) -> str:
    indent = "  " * depth
    label = _LEVEL_LABELS.get(node.get("level", 0), "")
    intro = f" — {node['intro_text']}" if node.get("intro_text") else ""
    line = f"{indent}[L{node.get('level', 0)} {label}] {node['name']}{intro}"
    child_lines = [_subtree_to_text(c, depth + 1) for c in node.get("children", [])]
    return "\n".join([line] + child_lines)


def step5_build_context(subtree: dict) -> str:
    """
    将子树 dict 序列化为缩进文本，供 LLM 理解知识图谱结构。

    Returns:
        多行字符串，每行标注层级类型和节点名称
    """
    context = _subtree_to_text(subtree)
    logger.info(f"[Step 5] 知识上下文 ({len(context)} 字符):\n{context}")
    return context


# ─────────────────────────────────────────────────────────────
# Step 6: LLM 生成报告大纲
# ─────────────────────────────────────────────────────────────

async def step6_generate_outline(question: str, context: str) -> str:
    """
    调用 LLM，基于检索到的知识图谱结构为用户问题生成报告大纲。

    Returns:
        Markdown 格式的报告大纲字符串
    """
    llm_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    llm_model = os.getenv("LLM_MODEL_NAME", "qwen3-27b")
    llm_api_key = os.getenv("LLM_API_KEY", "")

    system_prompt = (
        "你是一个专业的分析报告大纲生成助手。\n"
        "根据用户问题和提供的知识图谱结构，生成一份层次清晰的分析报告大纲。\n"
        "要求：\n"
        "- 使用 Markdown 格式（# 报告标题，## 一级章节，### 二级章节）\n"
        "- 严格基于知识图谱中的 L3 维度和 L4 评估项组织章节\n"
        "- L3 维度对应 ## 章节，L4 评估项对应 ### 章节\n"
        "- 可在每个 ## 章节后加一句简短说明\n"
        "- 语言简洁专业，直接输出大纲，不加额外解释"
    )

    user_prompt = (
        f"用户问题：{question}\n\n"
        f"知识图谱结构：\n{context}\n\n"
        "请基于以上知识图谱，生成分析报告大纲。"
    )

    headers = {"Content-Type": "application/json"}
    if llm_api_key:
        headers["Authorization"] = f"Bearer {llm_api_key}"

    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(os.getenv("LLM_TEMPERATURE", 0.1)),
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{llm_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"LLM 调用失败: status={resp.status}, body={body[:300]}")
            data = await resp.json()
            outline = data["choices"][0]["message"]["content"]

    logger.info(f"[Step 6] 大纲生成完成: {len(outline)} 字符")
    return outline


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

async def main(question: str) -> str:
    """
    完整工作流：输入用户问题，输出 Markdown 格式报告大纲。

    Args:
        question: 用户的分析问题，例如 "分析企业网络时延问题"
    Returns:
        Markdown 格式的报告大纲字符串
    """
    # Step 1: 加载资源
    faiss_svc, nodes_dict, children_map = step1_load_resources()

    # Step 2: 问题向量化
    query_embedding = await step2_embed_query(question)

    # Step 3: FAISS 检索
    hits = step3_search_nodes(query_embedding, faiss_svc)
    if not hits:
        return f"未找到与"{question}"相关的知识节点，请检查索引或降低 FAISS_SCORE_THRESHOLD。"

    # Step 4: 构建子树
    subtree = step4_build_subtree(hits, nodes_dict, children_map)

    # Step 5: 生成 LLM 上下文
    context = step5_build_context(subtree)

    # Step 6: LLM 生成大纲
    outline = await step6_generate_outline(question, context)

    return outline


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "分析企业网络时延问题"
    result = asyncio.run(main(q))
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
