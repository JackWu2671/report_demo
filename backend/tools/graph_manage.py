"""
graph_manage.py — 知识图谱融合工具。

在 save_outline_template 之后调用，分析模板中的新概念，
将专家经验自动回流到 node.json + relation.json。

规则：
  - L5（query）节点绑定 SQL/API 查询，受保护，禁止修改或新增
  - L2-L4 新节点：专家创建的新概念，可提升为图谱节点
  - 已有节点：可根据专家使用场景丰富 description
  - 变更直接写入磁盘；FAISS 索引需重建后生效
"""

import json
import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")
_NODE_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "node.json")
_RELATION_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "relation.json")

_SYSTEM_PROMPT = """\
你是传送网分析知识图谱的管理员，负责将专家的业务经验融合进图谱。

## 图谱层级说明
- L1：分析场景顶层（如「政企OTN升级」）
- L2：分析方向（如「fgOTN部署」）
- L3：分析视角（如「传送网络覆盖分析」）
- L4：具体分析模块（如「企业分布分析」）
- L5：数据查询节点，绑定 SQL/API，禁止新增或修改

## 你的任务
分析专家模板后，输出图谱更新方案，包含三类操作：
1. `new_nodes`：建议新增到图谱的节点（仅 L2-L4）
2. `new_relations`：新节点挂载的父节点关系
3. `enrichments`：建议丰富描述的已有节点

## 判断标准
**新增节点**（加入 new_nodes）：
  - 专家创建的新概念，图谱中没有语义相近的节点
  - 该概念有通用性，未来其他专家分析时可能复用
  - 排除：过于具体的一次性节点（如「南宁市2024Q3覆盖分析」）

**丰富描述**（加入 enrichments）：
  - 专家引用了已有节点，但其使用场景提供了新的业务视角
  - 现有 description 没有覆盖这个视角
  - append 字段只写补充内容（30-80字），不重复已有描述

**忽略**（既不新增也不丰富）：
  - 与已有节点完全重复的概念
  - 太具体的临时性概念

## 输出格式（严格 JSON，不加任何解释文字）
```json
{
  "new_nodes": [
    {
      "level": 3,
      "name": "节点名称",
      "keywords": ["关键词1", "关键词2", "关键词3"],
      "description": "该分析维度的业务含义，50-100字"
    }
  ],
  "new_relations": [
    {
      "parent_id": "L2_001",
      "child_name": "节点名称"
    }
  ],
  "enrichments": [
    {
      "node_id": "L3_001",
      "append": "补充的业务视角，30-80字"
    }
  ]
}
```

注意：new_relations 中 parent_id 必须是已有图谱节点的 ID，不能是新增节点的名称。"""


def _collect_nodes(tree: dict, existing_ids: set[str]) -> tuple[list[dict], list[dict]]:
    """
    递归遍历大纲树，分别收集：
    - novel: 专家新建的 L2-L4 节点（id 以 new_ 开头）
    - referenced: 模板引用的已有 L2-L4 图谱节点
    """
    novel, referenced = [], []
    _walk(tree, existing_ids, novel, referenced)
    return novel, referenced


def _walk(node: dict, existing_ids: set[str], novel: list, referenced: list) -> None:
    node_id = node.get("id", "") or ""
    level = node.get("level", 0)

    if 2 <= level <= 4:
        entry = {
            "id": node_id,
            "name": node.get("name", ""),
            "level": level,
            "description": node.get("description", ""),
        }
        if node_id.startswith("new_"):
            novel.append(entry)
        elif node_id in existing_ids:
            referenced.append(entry)

    for child in node.get("children", []):
        _walk(child, existing_ids, novel, referenced)


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


async def graph_manage(template_id: str) -> dict:
    """
    分析已保存模板，将专家经验融合回知识图谱。

    Args:
        template_id: save_outline_template 返回的模板 ID

    Returns:
        {status, added_nodes, enriched_nodes, message}
    """
    # 1. 加载模板
    template_path = os.path.join(_TEMPLATE_DIR, f"{template_id}.json")
    if not os.path.exists(template_path):
        return {"status": "error", "added_nodes": [], "enriched_nodes": [],
                "message": f"模板不存在: {template_id}"}

    with open(template_path, encoding="utf-8") as f:
        template = json.load(f)

    # 2. 加载现有图谱
    with open(_NODE_PATH, encoding="utf-8") as f:
        nodes: list[dict] = json.load(f)
    with open(_RELATION_PATH, encoding="utf-8") as f:
        relations: list[dict] = json.load(f)

    existing_ids = {n["id"] for n in nodes}

    # 3. 从大纲树收集候选节点
    outline = template.get("outline", {})
    novel_nodes, referenced_nodes = _collect_nodes(outline, existing_ids)

    logger.info("[graph_manage] 模板=%s  新建节点=%d  引用已有=%d",
                template_id, len(novel_nodes), len(referenced_nodes))

    if not novel_nodes and not referenced_nodes:
        return {
            "status": "no_change",
            "added_nodes": [],
            "enriched_nodes": [],
            "message": "模板完全复用已有图谱节点且无新概念，图谱无需更新。",
        }

    # 4. 调用 LLM 分析
    user_content = f"""## 已有图谱节点（L1-L4）
{json.dumps([n for n in nodes if n.get("level", 5) < 5], ensure_ascii=False, indent=2)}

## 已有图谱关系
{json.dumps(relations, ensure_ascii=False, indent=2)}

## 专家模板信息
场景名称：{template.get("scene_name", "")}
摘要：{template.get("summary", "")}
关键词：{", ".join(template.get("keywords", []))}
适用条件：{template.get("usage_conditions", "")}

## 模板中的新建节点（判断是否加入图谱）
{json.dumps(novel_nodes, ensure_ascii=False, indent=2)}

## 模板引用的已有节点（判断是否丰富描述）
{json.dumps(referenced_nodes, ensure_ascii=False, indent=2)}

请输出图谱更新方案。"""

    llm = LLMService.from_env()
    patch = await llm.complete_json([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ])

    # 5. 应用变更
    added_names: list[str] = []
    enriched_ids: list[str] = []

    # 新增节点：分配 ID
    name_to_new_id: dict[str, str] = {}
    for spec in patch.get("new_nodes", []):
        level = spec.get("level", 3)
        if not (2 <= level <= 4):
            logger.warning("[graph_manage] 跳过非法层级节点: level=%d name=%s", level, spec.get("name"))
            continue
        new_id = _next_id(level, nodes)
        nodes.append({
            "id": new_id,
            "level": level,
            "name": spec["name"],
            "keywords": spec.get("keywords", []),
            "description": spec.get("description", ""),
        })
        name_to_new_id[spec["name"]] = new_id
        added_names.append(spec["name"])
        logger.info("[graph_manage] 新增: %s  %s", new_id, spec["name"])

    # 新关系
    max_order: dict[str, int] = {}
    for rel in relations:
        pid = rel["parent"]
        max_order[pid] = max(max_order.get(pid, 0), rel.get("order", 0))

    current_ids = {n["id"] for n in nodes}
    for rel_spec in patch.get("new_relations", []):
        parent_id = rel_spec.get("parent_id", "")
        child_name = rel_spec.get("child_name", "")
        child_id = name_to_new_id.get(child_name)
        if not child_id:
            logger.warning("[graph_manage] 跳过关系（子节点未新增）: child=%s", child_name)
            continue
        if parent_id not in current_ids:
            logger.warning("[graph_manage] 跳过关系（父节点不存在）: parent=%s", parent_id)
            continue
        order = max_order.get(parent_id, 0) + 1
        relations.append({"parent": parent_id, "child": child_id, "order": order})
        max_order[parent_id] = order

    # 丰富描述
    nodes_dict = {n["id"]: n for n in nodes}
    for enrich in patch.get("enrichments", []):
        nid = enrich.get("node_id", "")
        append_text = enrich.get("append", "").strip()
        if not append_text or nid not in nodes_dict:
            continue
        node = nodes_dict[nid]
        if node.get("level", 5) >= 5:
            logger.warning("[graph_manage] 跳过 L5 节点丰富描述: %s", nid)
            continue
        existing = node.get("description", "")
        node["description"] = (existing + "；" + append_text) if existing else append_text
        enriched_ids.append(nid)
        logger.info("[graph_manage] 丰富描述: %s", nid)

    # 6. 写回磁盘
    with open(_NODE_PATH, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(_RELATION_PATH, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    logger.info("[graph_manage] 完成: 新增 %d 节点, 丰富 %d 节点描述",
                len(added_names), len(enriched_ids))

    msg_parts = []
    if added_names:
        msg_parts.append(f"新增节点 {len(added_names)} 个：{', '.join(added_names)}")
    if enriched_ids:
        msg_parts.append(f"丰富描述 {len(enriched_ids)} 个：{', '.join(enriched_ids)}")
    msg_parts.append("FAISS 索引需重建后生效（运行 scripts/build_index.py）")

    return {
        "status": "success",
        "added_nodes": added_names,
        "enriched_nodes": enriched_ids,
        "message": "；".join(msg_parts),
    }
