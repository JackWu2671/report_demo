"""
graph_manage.py — 知识图谱融合执行工具。

接收 agent 已分析好的 patch，直接写入 node.json + relation.json。
分析和决策由 agent 按 graph-fusion.md 的 SOP 完成，本工具只负责写入，不再内部调用 LLM。

规则：
  - L5（query）节点绑定 SQL/API 查询，受保护，写入时自动跳过
  - add_nodes 中 level 只允许 2、3、4
  - 变更直接写入磁盘；FAISS 索引需重建后生效
"""

import json
import logging
import os
import sys

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logger = logging.getLogger(__name__)

_NODE_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "node.json")
_RELATION_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "relation.json")


def _next_id(level: int, nodes: list[dict]) -> str:
    """生成该层级的下一个可用 ID，如 L3_006。"""
    prefix = f"L{level}_"
    max_num = 0
    for n in nodes:
        nid = n.get("id", "")
        if nid.startswith(prefix):
            try:
                max_num = max(max_num, int(nid[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{max_num + 1:03d}"


async def graph_manage(
    template_id: str,
    add_nodes: list[dict],
    enrich_nodes: list[dict],
) -> dict:
    """
    将 agent 分析好的图谱融合方案写入 node.json + relation.json。

    Args:
        template_id  : 来源模板 ID，用于日志溯源
        add_nodes    : 要新增的节点列表，每项格式：
                       {level: 2|3|4|5, name, keywords, description, parent_id}
                       level=5 为 query 节点，description 即查询参数
        enrich_nodes : 要丰富描述的已有节点列表，每项格式：
                       {node_id, append}

    Returns:
        {status, added_nodes, enriched_nodes, message}
    """
    with open(_NODE_PATH, encoding="utf-8") as f:
        nodes: list[dict] = json.load(f)
    with open(_RELATION_PATH, encoding="utf-8") as f:
        relations: list[dict] = json.load(f)

    existing_ids = {n["id"] for n in nodes}

    added_names: list[str] = []
    enriched_ids: list[str] = []

    # ── 新增节点 ──────────────────────────────────────────────────
    max_order: dict[str, int] = {}
    for rel in relations:
        pid = rel["parent"]
        max_order[pid] = max(max_order.get(pid, 0), rel.get("order", 0))

    for spec in add_nodes:
        level = spec.get("level")
        if level not in (2, 3, 4, 5):
            logger.warning("[graph_manage] 跳过非法层级: level=%s name=%s", level, spec.get("name"))
            continue
        parent_id = spec.get("parent_id", "")
        if parent_id not in existing_ids:
            logger.warning("[graph_manage] 跳过无效父节点: parent_id=%s", parent_id)
            continue

        new_id = _next_id(level, nodes)
        new_node = {
            "id": new_id,
            "level": level,
            "name": spec["name"],
            "keywords": spec.get("keywords", []),
            "description": spec.get("description", ""),
        }
        nodes.append(new_node)
        existing_ids.add(new_id)

        order = max_order.get(parent_id, 0) + 1
        relations.append({"parent": parent_id, "child": new_id, "order": order})
        max_order[parent_id] = order

        added_names.append(spec["name"])
        logger.info("[graph_manage] 新增: %s  %s → parent=%s", new_id, spec["name"], parent_id)

    # ── 丰富已有节点描述 ──────────────────────────────────────────
    nodes_dict = {n["id"]: n for n in nodes}
    for spec in enrich_nodes:
        nid = spec.get("node_id", "")
        append_text = spec.get("append", "").strip()
        if not append_text or nid not in nodes_dict:
            logger.warning("[graph_manage] 跳过无效丰富: node_id=%s", nid)
            continue
        node = nodes_dict[nid]
        if node.get("level", 5) >= 5:
            logger.warning("[graph_manage] 跳过 L5 节点: %s", nid)
            continue
        existing = node.get("description", "")
        node["description"] = (existing + "；" + append_text) if existing else append_text
        enriched_ids.append(nid)
        logger.info("[graph_manage] 丰富描述: %s", nid)

    # ── 写回磁盘 ──────────────────────────────────────────────────
    with open(_NODE_PATH, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(_RELATION_PATH, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    logger.info("[graph_manage] template=%s  新增=%d  丰富=%d",
                template_id, len(added_names), len(enriched_ids))

    msg_parts = []
    if added_names:
        msg_parts.append(f"新增节点 {len(added_names)} 个：{', '.join(added_names)}")
    if enriched_ids:
        msg_parts.append(f"丰富描述 {len(enriched_ids)} 个：{', '.join(enriched_ids)}")
    if not msg_parts:
        msg_parts.append("无变更")
    msg_parts.append("FAISS 索引需重建后生效（运行 scripts/build_index.py）")

    return {
        "status": "success",
        "added_nodes": added_names,
        "enriched_nodes": enriched_ids,
        "message": "；".join(msg_parts),
    }
