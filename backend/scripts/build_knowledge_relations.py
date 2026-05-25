#!/usr/bin/env python3
"""
build_knowledge_relations.py — 从各层级 JSON 自动生成 knowledge_relations.json

父子关系推导规则：
  场景      .dimensions[i].id (UUID) → 子场景.id       → 子场景.nodeId
  子场景    .dimensions[i].id (UUID) → 评估维度.id     → 评估维度.nodeId
  评估维度  .dimensions[i].id (UUID) → 评估项.id       → 评估项.nodeId
  评估项    .dimensions[i] (name)    → 评估指标.name   → 评估指标.nodeId

输出格式: [{parent: nodeId, child: nodeId, order: int}]

用法:
  python3 build_knowledge_relations.py

输出:
  expert_knowledge/knowledge_relations.json
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

# ── 配置区 ────────────────────────────────────────────────────────────
# 按层级顺序列出，前三个用 UUID 匹配，最后一个用 name 匹配
UUID_LEVELS = ["场景.json", "子场景.json", "评估维度.json"]
NAME_LEVEL  = "评估项.json"
LEAF_LEVEL  = "评估指标.json"
OUTPUT_FILE = os.path.join(_KB_DIR, "knowledge_relations.json")
# ─────────────────────────────────────────────────────────────────────


def load_json(filename: str) -> list[dict]:
    path = os.path.join(_KB_DIR, filename)
    if not os.path.exists(path):
        print(f"[跳过] 文件不存在: {filename}")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ── 建立查找表 ─────────────────────────────────────────────────────
    uuid_to_id: dict[str, str] = {}   # uuid → id（所有层级）
    name_to_id: dict[str, str] = {}   # name → id（仅评估指标）

    all_files = UUID_LEVELS + [NAME_LEVEL, LEAF_LEVEL]
    all_data: dict[str, list[dict]] = {}

    for filename in all_files:
        data = load_json(filename)
        all_data[filename] = data
        for record in data:
            uid = record.get("uuid", "")
            nid = record.get("id", "")
            if uid and nid:
                uuid_to_id[uid] = nid
            if record.get("level") == 5 and record.get("name") and nid:
                name_to_id[record["name"]] = nid

    # ── 生成关系 ───────────────────────────────────────────────────────
    relations = []
    missing   = 0

    # 场景 / 子场景 / 评估维度 → 下一层（UUID 匹配）
    for filename in UUID_LEVELS:
        for record in all_data.get(filename, []):
            parent_id = record.get("id", "")
            dims = record.get("dimensions") or []
            for dim in dims:
                child_uuid = dim.get("uuid", "")
                child_id   = uuid_to_id.get(child_uuid)
                if not child_id:
                    missing += 1
                    continue
                relations.append({
                    "parent": parent_id,
                    "child":  child_id,
                    "order":  dim.get("rank") or len(relations) + 1,
                })

    # 评估项 → 评估指标（name 匹配）
    for record in all_data.get(NAME_LEVEL, []):
        parent_id = record.get("id", "")
        dims = record.get("dimensions") or []   # list of str
        for i, metric_name in enumerate(dims):
            child_id = name_to_id.get(metric_name)
            if not child_id:
                missing += 1
                continue
            relations.append({
                "parent": parent_id,
                "child":  child_id,
                "order":  i + 1,
            })

    # ── 输出 ───────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(relations, f, ensure_ascii=False, indent=2)

    print(f"\n生成完成，共 {len(relations)} 条关系，{missing} 条未匹配（子节点数据不存在）→ {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
