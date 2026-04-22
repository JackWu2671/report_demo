"""
case_workflow_2 — 基于 JSON 知识图谱 + FAISS 的报告大纲生成工作流。

对应 report_system 的 case_2（outline-generate）：
  - 用 expert_knowledge/node.json + relation.json 替代 Neo4j
  - 用 services/faiss_service.py 做向量检索
  - 用 ANCHOR_PROMPT + LLM 选锚节点（同 report_system/graph_rag_executor.py）
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

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService
from services.llm_service import LLMService

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_EXPERT_DIR = os.path.join(_BASE_DIR, "expert_knowledge")

# 与 report_system/graph_rag_executor.py 保持一致
ANCHOR_PROMPT = """你是知识库节点选择专家。从候选中选出最符合用户意图的唯一节点。
判断原则: 宽泛→高层级，具体→低层级，父子关系时按粒度判断。
用 ```json ``` 代码块包裹输出，不要加解释文字。格式:
```json
{"selected_id":"","selected_name":"","selected_path":"","level":0,"reason":""}
```"""


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
# Step 3: FAISS 检索候选节点
# ─────────────────────────────────────────────────────────────

def step3_search_nodes(
    query_embedding: np.ndarray,
    faiss_svc: FAISSService,
    top_k: int = 10,
    threshold: float | None = None,
) -> list[dict]:
    """
    在 FAISS 索引中检索与问题最相关的候选节点。

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
# Step 4: 为候选节点构建祖先路径
# ─────────────────────────────────────────────────────────────

def step4_build_candidate_paths(
    hits: list[dict],
    nodes_dict: dict,
    children_map: dict,
) -> list[dict]:
    """
    为每个 FAISS 命中节点构建完整祖先路径，供 LLM 理解节点在图中的位置。

    格式同 report_system/graph_rag_executor.py：
      id=... name=... level=... path=（根 > ... > 节点）

    Returns:
        [{id, name, level, score, path}, ...]
    """
    # 反向映射：child_id -> parent_id
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
        chain.reverse()  # 根 → 叶
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
        f"[Step 4] {len(candidates)} 个候选节点:\n"
        + "\n".join(
            f"  L{c['level']} {c['name']} | {c['path']} | score={c['score']:.3f}"
            for c in candidates
        )
    )
    return candidates


# ─────────────────────────────────────────────────────────────
# Step 5: LLM 选锚节点
# ─────────────────────────────────────────────────────────────

async def step5_select_anchor(question: str, candidates: list[dict]) -> dict:
    """
    使用 ANCHOR_PROMPT 让 LLM 从候选节点中选出最符合用户意图的锚节点。

    候选格式（同 report_system）：
        - id=... name=... level=... path=...

    Returns:
        {"selected_id": ..., "selected_name": ..., "selected_path": ...,
         "level": ..., "reason": ...}
    """
    llm = LLMService.from_env()

    cs = "\n".join(
        f"- id={c['id']} name={c['name']} level={c['level']} path={c['path']}"
        for c in candidates
    )

    messages = [
        {"role": "system", "content": ANCHOR_PROMPT},
        {"role": "user", "content": f"## 候选\n{cs}\n\n## 问题\n{question}"},
    ]

    try:
        anchor = await llm.complete_json(messages)
    except Exception as e:
        logger.warning(f"[Step 5] LLM 选锚失败，回退到 score 最高的候选: {e}")
        f = candidates[0]
        anchor = {
            "selected_id": f["id"],
            "selected_name": f["name"],
            "selected_path": f["path"],
            "level": f["level"],
            "reason": "fallback",
        }

    logger.info(
        f"[Step 5] 选锚: '{anchor.get('selected_name')}' "
        f"(L{anchor.get('level')}), reason={anchor.get('reason', '')}"
    )
    return anchor


# ─────────────────────────────────────────────────────────────
# Step 6: 从锚节点构建子树
# ─────────────────────────────────────────────────────────────

def step6_build_subtree(
    anchor_id: str, nodes_dict: dict, children_map: dict
) -> dict:
    """
    从锚节点递归构建完整子树 dict（含 children 列表）。

    Returns:
        subtree dict，结构: {id, name, level, intro_text, children: [...]}
    """
    if anchor_id not in nodes_dict:
        raise ValueError(f"锚节点 '{anchor_id}' 不在知识图谱中")

    subtree = _build_subtree(anchor_id, nodes_dict, children_map)
    logger.info(
        f"[Step 6] 子树构建完成: 根='{nodes_dict[anchor_id]['name']}', "
        f"共 {_count_nodes(subtree)} 个节点"
    )
    return subtree


# ─────────────────────────────────────────────────────────────
# Step 7: 子树直接渲染为 Markdown 大纲
# ─────────────────────────────────────────────────────────────

def step7_render_outline(subtree: dict) -> str:
    """
    将锚节点子树直接渲染为 Markdown 大纲，无需 LLM。
    锚节点对应 # 标题，每深一层标题级别加一。

    Returns:
        Markdown 格式的报告大纲字符串
    """
    blocks = _render_node(subtree, heading_level=1)
    outline = "\n\n".join(blocks)
    logger.info(f"[Step 7] 大纲渲染完成: {len(outline)} 字符")
    return outline


# ─────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────

def _build_subtree(node_id: str, nodes_dict: dict, children_map: dict) -> dict:
    node = dict(nodes_dict[node_id])
    node["children"] = [
        _build_subtree(cid, nodes_dict, children_map)
        for cid in children_map.get(node_id, [])
    ]
    return node


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


def _render_node(node: dict, heading_level: int) -> list[str]:
    prefix = "#" * min(heading_level, 6)
    block = f"{prefix} {node['name']}"
    if node.get("intro_text"):
        block += f"\n\n{node['intro_text']}"
    blocks = [block]
    for child in node.get("children", []):
        blocks.extend(_render_node(child, heading_level + 1))
    return blocks


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

    # Step 3: FAISS 检索候选节点
    hits = step3_search_nodes(query_embedding, faiss_svc)
    if not hits:
        return f"未找到与'{question}'相关的知识节点，请检查索引或降低 FAISS_SCORE_THRESHOLD。"

    # Step 4: 构建候选节点祖先路径
    candidates = step4_build_candidate_paths(hits, nodes_dict, children_map)

    # Step 5: LLM 选锚节点
    anchor = await step5_select_anchor(question, candidates)

    # Step 6: 从锚节点构建子树
    subtree = step6_build_subtree(anchor["selected_id"], nodes_dict, children_map)

    # Step 7: 渲染大纲
    outline = step7_render_outline(subtree)

    return outline


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "分析企业网络时延问题"
    result = asyncio.run(main(q))
    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)
