"""
kb_updater.py — Step 4: LLM 分析新知识点，更新 node.json / relation.json，增量重建 FAISS。

Step 4a propose_updates : LLM 判断哪些概念需要补充到知识库，输出 add_nodes 列表
Step 4b apply_updates   : 为新节点自动分配 ID，生成对应 relation 记录
Step 4c save_json_files : 写回 expert_knowledge/node.json 和 relation.json
Step 4d rebuild_index   : 仅对新增节点生成 Embedding，追加到现有 FAISS 索引
"""

import json
import logging
import os
import sys
from pathlib import Path

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
UPDATE_PROMPT: str = (_PROMPT_DIR / "update.txt").read_text(encoding="utf-8")

_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")


# ── Step 4a ────────────────────────────────────────────────────

async def propose_updates(
    expert_text: str,
    outline_md: str,
    nodes_dict: dict,
) -> dict:
    """
    LLM 分析专家描述中未覆盖的知识点，提出新增节点建议。

    Args:
        expert_text : 专家输入的原始文本
        outline_md  : generate_outline() 生成的 Markdown 大纲
        nodes_dict  : 当前知识库 {node_id -> node_dict}

    Returns:
        {"add_nodes": [{level, name, keywords, description, parent_id, order}, ...]}
    """
    llm = LLMService.from_env()

    kb_lines = [
        f"  [L{n['level']} {n['id']}] {n['name']}"
        for n in sorted(nodes_dict.values(), key=lambda x: (x["level"], x["id"]))
    ]
    kb_text = "\n".join(kb_lines)

    user_content = (
        f"## 现有知识库节点\n{kb_text}\n\n"
        f"## 专家描述\n{expert_text}\n\n"
        f"## 本次生成的大纲\n{outline_md}"
    )

    messages = [
        {"role": "system", "content": UPDATE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "[Step 4] Update Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 4] LLM 知识库更新分析 ↓", flush=True)
    print("-" * 50, flush=True)

    answer = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 4] LLM 完整输出:\n%s", answer)

    result = LLMService._parse_json(answer)
    add_nodes = result.get("add_nodes", [])
    logger.info("[Step 4a] LLM 建议新增 %d 个节点", len(add_nodes))
    return {"add_nodes": add_nodes}


# ── Step 4b ────────────────────────────────────────────────────

def apply_updates(
    patch: dict,
    nodes_list: list,
    relations_list: list,
) -> tuple[list, list, list]:
    """
    将 LLM 建议的新节点写入内存（不涉及文件 IO）。

    自动为每个新节点分配 ID（格式: L{level}_{seq:03d}，从当前最大序号续接），
    并基于 parent_id + order 自动生成对应的 relation 记录。

    Args:
        patch          : propose_updates() 的输出 {"add_nodes": [...]}
        nodes_list     : 当前 node.json 内容（list of dicts）
        relations_list : 当前 relation.json 内容（list of dicts）

    Returns:
        (updated_nodes, updated_relations, newly_added_nodes)
        newly_added_nodes 用于后续 rebuild_index 仅对新节点做 Embedding
    """
    add_nodes = patch.get("add_nodes", [])
    if not add_nodes:
        return nodes_list, relations_list, []

    # 计算各层级当前最大序号，例如 L3_012 → level=3, seq=12
    level_max: dict[int, int] = {}
    for n in nodes_list:
        level = n.get("level", 0)
        try:
            seq = int(n["id"].split("_")[-1])
        except (ValueError, IndexError):
            seq = 0
        level_max[level] = max(level_max.get(level, 0), seq)

    new_nodes: list[dict] = []
    new_relations: list[dict] = []

    for spec in add_nodes:
        level = spec.get("level")
        if level not in (2, 3, 4, 5):
            logger.warning("[Step 4b] 跳过无效 level=%s (节点: %s)", level, spec.get("name"))
            continue
        parent_id = spec.get("parent_id", "")
        if not parent_id:
            logger.warning("[Step 4b] 跳过缺少 parent_id 的节点: %s", spec.get("name"))
            continue

        level_max[level] = level_max.get(level, 0) + 1
        new_id = f"L{level}_{level_max[level]:03d}"

        new_node = {
            "id": new_id,
            "level": level,
            "name": spec.get("name", ""),
            "keywords": spec.get("keywords", []),
            "description": spec.get("description", ""),
        }
        new_nodes.append(new_node)
        new_relations.append({
            "parent": parent_id,
            "child": new_id,
            "order": spec.get("order", 99),
        })
        logger.info(
            "[Step 4b] 新增节点 %s (%s) → 挂载到 %s",
            new_id, new_node["name"], parent_id,
        )

    updated_nodes = nodes_list + new_nodes
    updated_relations = relations_list + new_relations
    return updated_nodes, updated_relations, new_nodes


# ── Step 4c ────────────────────────────────────────────────────

def save_json_files(nodes_list: list, relations_list: list) -> None:
    """将更新后的节点和关系列表写回 expert_knowledge/ 目录。"""
    node_path = os.path.join(_EXPERT_DIR, "node.json")
    rel_path = os.path.join(_EXPERT_DIR, "relation.json")

    with open(node_path, "w", encoding="utf-8") as f:
        json.dump(nodes_list, f, ensure_ascii=False, indent=2)
    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump(relations_list, f, ensure_ascii=False, indent=2)

    logger.info(
        "[Step 4c] 知识库已写入 — 节点: %d, 关系: %d",
        len(nodes_list), len(relations_list),
    )


# ── Step 4d ────────────────────────────────────────────────────

async def rebuild_index(new_nodes: list) -> None:
    """
    为新增节点生成 Embedding 并增量追加到现有 FAISS 索引。

    不重建整个索引，仅对 new_nodes 做追加，节省时间。
    追加完成后保存索引文件（覆盖原文件）。

    Args:
        new_nodes: apply_updates() 返回的 newly_added_nodes 列表
    """
    if not new_nodes:
        logger.info("[Step 4d] 无新节点，跳过 FAISS 更新")
        return

    import numpy as np
    import faiss as faiss_lib

    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.load(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )

    texts = [n["name"] + " " + " ".join(n.get("keywords", [])) for n in new_nodes]
    logger.info("[Step 4d] 为 %d 个新节点生成 Embedding...", len(new_nodes))
    embeddings = await emb_svc.get_embeddings_batch(texts, batch_size=32)

    vecs = embeddings.copy().astype(np.float32)
    faiss_lib.normalize_L2(vecs)
    faiss_svc.index.add(vecs)
    faiss_svc.id_map.extend(new_nodes)

    faiss_svc.save(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )
    logger.info(
        "[Step 4d] FAISS 索引已更新，新增 %d 条，共 %d 条向量",
        len(new_nodes), faiss_svc.total,
    )
