#!/usr/bin/env python3
"""
build_knowledge_nodes.py — 将各层级 JSON 合并成统一的节点总表

读取（本地生成的）各层级 JSON，提取 id / level / nodeId / name / description，
合并输出为单一 knowledge_nodes.json，供知识图谱检索使用。

用法:
  python3 build_knowledge_nodes.py

输出:
  expert_knowledge/knowledge_nodes.json

特殊处理:
  评估指标 没有 description 字段，用 question 字段代替。
"""

import json
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

# ── 配置区 ────────────────────────────────────────────────────────────
# (文件名, level值, description 来源字段)
INPUT_CONFIGS = [
    ("场景.json",    1, "description"),
    ("子场景.json",  2, "description"),
    ("评估维度.json", 3, "description"),
    ("评估项.json",  4, "description"),
    ("评估指标.json", 5, None),     # 无 description，留空字符串
]
OUTPUT_FILE = os.path.join(_KB_DIR, "knowledge_nodes.json")
# ─────────────────────────────────────────────────────────────────────


def extract_node(record: dict, desc_field: str | None) -> dict:
    node = {
        "uuid":        record.get("uuid", ""),
        "id":          record.get("id", ""),
        "level":       record.get("level", ""),
        "name":        record.get("name", ""),
        "description": record.get(desc_field, "") if desc_field else "",
    }
    # L4 节点额外携带 condition / condition_queries，供大纲执行时判断是否展示
    if record.get("level") == 4:
        node["condition"]         = record.get("condition", "")
        node["condition_queries"] = record.get("condition_queries", [])
    return node


def main():
    all_nodes = []

    for filename, level, desc_field in INPUT_CONFIGS:
        path = os.path.join(_KB_DIR, filename)
        if not os.path.exists(path):
            print(f"[跳过] 文件不存在: {filename}")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[错误] {filename} 不是数组，跳过", file=sys.stderr)
            continue

        nodes = [extract_node(r, desc_field) for r in data]
        all_nodes.extend(nodes)
        print(f"[{level}] {len(nodes)} 条")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成，共 {len(all_nodes)} 个节点 → {OUTPUT_FILE}")

    sync_target = os.path.join(_KB_DIR, "node.json")
    shutil.copy(OUTPUT_FILE, sync_target)
    print(f"→ 已同步到 node.json")


if __name__ == "__main__":
    main()
