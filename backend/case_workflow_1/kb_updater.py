"""
kb_updater.py — Step 4: 从带 id 标注的大纲中解析新节点，补充元数据，写入 KB，更新 FAISS。

Step 4a parse_new_nodes  : 解析大纲 Markdown，提取 [new] 节点及其层级/父节点关系
Step 4b enrich_new_nodes : LLM 为 [new] 节点补充 keywords 和 description
Step 4c apply_updates    : 分配 ID（支持父节点也是 [new] 的链式情况），生成 relation 记录
Step 4d save_json_files  : 写回 expert_knowledge/node.json 和 relation.json
Step 4e rebuild_index    : 仅对新节点生成 Embedding，增量追加到 FAISS 索引
"""

import json
import logging
import os
import re
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
ENRICH_PROMPT: str = (_PROMPT_DIR / "enrich.txt").read_text(encoding="utf-8")

_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")


# ── Step 4a ────────────────────────────────────────────────────

def parse_new_nodes(outline_md: str) -> list[dict]:
    """
    解析带 id 标注的 Markdown 大纲，提取所有 [new] 节点及其层级/父节点信息。

    解析规则:
    - 每行格式: `## [L2_001] 节点名` 或 `### [new] 新节点名`
    - 标题深度（# 数量）= 知识库层级（## = L2, ### = L3, ...）
    - 用栈跟踪各层级当前节点，确定 [new] 节点的父节点

    Args:
        outline_md: generate_outline() 输出的带 id 标注 Markdown

    Returns:
        [{name, level, parent_id, parent_name, order}, ...]
        parent_id   : 父节点的真实 KB id（若父节点也是 [new]，则为 None）
        parent_name : 父节点名称（供 apply_updates 做 name→id 解析，处理链式 [new]）
        order       : 在同级节点中的顺序（按出现顺序计数）
    """
    level_stack: dict[int, dict] = {}   # depth → {ref, name}
    order_counter: dict[str, int] = {}  # parent_ref → 已出现的子节点数

    new_nodes: list[dict] = []

    for line in outline_md.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^(#+)\s+\[([^\]]+)\]\s+(.+)$", line)
        if not m:
            continue

        hashes = m.group(1)
        node_ref = m.group(2).strip()
        name = m.group(3).strip()
        depth = len(hashes)  # ## = 2 = L2

        # 清除比当前深度更深的层级（向上或同层移动时清空子树）
        for d in [d for d in level_stack if d >= depth]:
            del level_stack[d]
        level_stack[depth] = {"ref": node_ref, "name": name}

        parent = level_stack.get(depth - 1)
        parent_ref = parent["ref"] if parent else None
        parent_key = parent_ref or "root"
        order_counter[parent_key] = order_counter.get(parent_key, 0) + 1

        if node_ref.lower() == "new":
            new_nodes.append({
                "name": name,
                "level": depth,
                "parent_id": parent_ref if (parent_ref and parent_ref.lower() != "new") else None,
                "parent_name": parent["name"] if parent else None,
                "order": order_counter[parent_key],
            })

    logger.info(
        "[Step 4a] 解析到 %d 个 [new] 节点: %s",
        len(new_nodes),
        [f"L{n['level']} {n['name']}" for n in new_nodes],
    )
    return new_nodes


# ── Step 4b ────────────────────────────────────────────────────

async def enrich_new_nodes(new_nodes: list[dict], expert_text: str) -> dict:
    """
    LLM 为 [new] 节点补充 keywords 和 description。

    Args:
        new_nodes   : parse_new_nodes() 返回的列表
        expert_text : 专家输入的原始文本（提供业务背景）

    Returns:
        {"add_nodes": [{level, name, keywords, description, parent_id, parent_name, order}, ...]}
    """
    if not new_nodes:
        return {"add_nodes": []}

    llm = LLMService.from_env()

    nodes_text = "\n".join(
        f"- L{n['level']} 节点「{n['name']}」"
        f"（父节点: {n.get('parent_name') or n.get('parent_id') or '未知'}）"
        for n in new_nodes
    )

    user_content = f"## 专家描述\n{expert_text}\n\n## 需要补充信息的新节点\n{nodes_text}"

    messages = [
        {"role": "system", "content": ENRICH_PROMPT},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "[Step 4b] Enrich Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 4b] LLM 补充新节点元数据 ↓", flush=True)
    print("-" * 50, flush=True)

    answer = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 4b] LLM 完整输出:\n%s", answer)

    result = LLMService._parse_json(answer)
    enriched = result.get("nodes", [])

    # 将 LLM 补充的 keywords/description 与结构信息（parent_id, order 等）合并
    name_to_struct = {n["name"]: n for n in new_nodes}
    add_nodes = []
    for e in enriched:
        name = e.get("name", "")
        struct = name_to_struct.get(name, {})
        add_nodes.append({
            "name": name,
            "level": struct.get("level", e.get("level")),
            "keywords": e.get("keywords", []),
            "description": e.get("description", ""),
            "parent_id": struct.get("parent_id"),
            "parent_name": struct.get("parent_name"),
            "order": struct.get("order", 99),
        })

    logger.info("[Step 4b] 补充完成，%d 个节点", len(add_nodes))
    return {"add_nodes": add_nodes}


# ── Step 4c ────────────────────────────────────────────────────

def apply_updates(
    patch: dict,
    nodes_list: list,
    relations_list: list,
) -> tuple[list, list, list]:
    """
    将新节点写入内存（不涉及文件 IO）。

    自动分配 ID（L{level}_{seq:03d}，从当前最大序号续接）。
    支持父节点也是 [new] 的链式情况：通过 name→id 映射解析 parent_name。

    Args:
        patch          : enrich_new_nodes() 的输出 {"add_nodes": [...]}
        nodes_list     : 当前 node.json 内容（list of dicts）
        relations_list : 当前 relation.json 内容（list of dicts）

    Returns:
        (updated_nodes, updated_relations, newly_added_nodes)
    """
    add_nodes = patch.get("add_nodes", [])
    if not add_nodes:
        return nodes_list, relations_list, []

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
    name_to_new_id: dict[str, str] = {}  # 本批新建节点的 name→id（供链式 [new] 父节点解析）

    for spec in add_nodes:
        level = spec.get("level")
        if level not in (2, 3, 4, 5):
            logger.warning("[Step 4c] 跳过无效 level=%s 节点: %s", level, spec.get("name"))
            continue

        # 优先用明确的 parent_id，再尝试从 parent_name 查本批新增节点的 id
        parent_id = spec.get("parent_id")
        if not parent_id and spec.get("parent_name"):
            parent_id = name_to_new_id.get(spec["parent_name"])
        if not parent_id:
            logger.warning(
                "[Step 4c] 跳过找不到父节点的节点: %s (parent_name=%s)",
                spec.get("name"), spec.get("parent_name"),
            )
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
        name_to_new_id[spec["name"]] = new_id

        logger.info(
            "[Step 4c] 新增节点 %s (%s) → 挂载到 %s",
            new_id, new_node["name"], parent_id,
        )

    return nodes_list + new_nodes, relations_list + new_relations, new_nodes


# ── Step 4d ────────────────────────────────────────────────────

def save_json_files(nodes_list: list, relations_list: list) -> None:
    """将更新后的节点和关系列表写回 expert_knowledge/ 目录。"""
    node_path = os.path.join(_EXPERT_DIR, "node.json")
    rel_path = os.path.join(_EXPERT_DIR, "relation.json")

    with open(node_path, "w", encoding="utf-8") as f:
        json.dump(nodes_list, f, ensure_ascii=False, indent=2)
    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump(relations_list, f, ensure_ascii=False, indent=2)

    logger.info(
        "[Step 4d] 知识库已写入 — 节点: %d, 关系: %d",
        len(nodes_list), len(relations_list),
    )


# ── Step 4e ────────────────────────────────────────────────────

async def rebuild_index(new_nodes: list) -> None:
    """
    为新增节点生成 Embedding 并增量追加到现有 FAISS 索引。

    Args:
        new_nodes: apply_updates() 返回的 newly_added_nodes 列表
    """
    if not new_nodes:
        logger.info("[Step 4e] 无新节点，跳过 FAISS 更新")
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
    logger.info("[Step 4e] 为 %d 个新节点生成 Embedding...", len(new_nodes))
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
        "[Step 4e] FAISS 索引已更新，新增 %d 条，共 %d 条向量",
        len(new_nodes), faiss_svc.total,
    )
