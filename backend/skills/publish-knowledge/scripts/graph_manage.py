#!/usr/bin/env python3
"""
执行知识图谱融合写入。

用法:
  python3 graph_manage.py --template-id <id> \
      --add-nodes "<json_array>" \
      --enrich-nodes "<json_array>"

add_nodes 每项：{"level": 2|3|4|5, "name": "...", "keywords": [...], "description": "...", "parent_id": "L4_001", "condition": "当...时才展示（可选）"}
  level=5 为 query 节点，description 即查询参数，parent_id 须为 L4 节点
enrich_nodes 每项：{"node_id": "L3_001", "append": "补充描述"}

成功时输出 JSON 摘要：{"added": [...], "enriched": [...], "message": "..."}
"""
import argparse
import asyncio
import json
import logging
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.environ.get("REPORT_BACKEND_DIR", "") or os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPTS)))
_NODE_PATH     = os.path.join(_BACKEND_DIR, "reference", "node.json")
_RELATION_PATH = os.path.join(_BACKEND_DIR, "reference", "relation.json")


def _next_id(level: int, nodes: list[dict]) -> str:
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


async def graph_manage(template_id: str, add_nodes: list[dict], enrich_nodes: list[dict]) -> dict:
    with open(_NODE_PATH, encoding="utf-8") as f:
        nodes: list[dict] = json.load(f)
    with open(_RELATION_PATH, encoding="utf-8") as f:
        relations: list[dict] = json.load(f)

    existing_ids = {n["id"] for n in nodes}
    added_names:  list[str] = []
    enriched_ids: list[str] = []

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
            "id": new_id, "level": level, "name": spec["name"],
            "keywords": spec.get("keywords", []),
            "description": spec.get("description", ""),
            "condition": spec.get("condition", ""),
        }
        nodes.append(new_node)
        existing_ids.add(new_id)
        order = max_order.get(parent_id, 0) + 1
        relations.append({"parent": parent_id, "child": new_id, "order": order})
        max_order[parent_id] = order
        added_names.append(spec["name"])
        logger.info("[graph_manage] 新增: %s  %s → parent=%s", new_id, spec["name"], parent_id)

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

    with open(_NODE_PATH, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(_RELATION_PATH, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    logger.info("[graph_manage] template=%s  新增=%d  丰富=%d", template_id, len(added_names), len(enriched_ids))
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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--add-nodes", default="[]")
    parser.add_argument("--enrich-nodes", default="[]")
    args = parser.parse_args()

    def _parse_json_arg(s: str) -> list:
        s = " ".join(s.split())
        if s.startswith("'") and s.endswith("'"):
            s = s[1:-1]
        return json.loads(s)

    try:
        add_nodes    = _parse_json_arg(args.add_nodes)
        enrich_nodes = _parse_json_arg(args.enrich_nodes)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"JSON 解析失败：{e}"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    result = await graph_manage(
        template_id=args.template_id,
        add_nodes=add_nodes,
        enrich_nodes=enrich_nodes,
    )
    print(json.dumps({
        "added": result["added_nodes"],
        "enriched": result["enriched_nodes"],
        "message": result["message"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
