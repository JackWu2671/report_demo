#!/usr/bin/env python3
"""
build_knowledge_nodes.py — 将各层级 JSON 合并成统一的节点总表

读取（本地生成的）各层级 JSON，提取 id / level / nodeId / name / description，
合并输出为单一 knowledge_nodes.json，供知识图谱检索使用。

用法:
  python3 build_knowledge_nodes.py

输出:
  reference/knowledge_nodes.json

特殊处理:
  评估指标 没有 description 字段，用 question 字段代替。
"""

import json
import logging
import os
import shutil


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "reference")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── 配置区 ────────────────────────────────────────────────────────────
# (文件名, level值, description 来源字段)
INPUT_CONFIGS = [
    ("场景.json", 1, "description"),
    ("子场景.json", 2, "description"),
    ("评估维度.json", 3, "description"),
    ("评估项.json", 4, "description"),
    ("评估指标.json", 5, None),     # 无 description，留空字符串
]
OUTPUT_FILE = os.path.join(_KB_DIR, "knowledge_nodes.json")
# ─────────────────────────────────────────────────────────────────────


def extract_node(record: dict, desc_field: str | None) -> dict:
    node = {
        "uuid": record.get("uuid", ""),
        "id": record.get("id", ""),
        "level": record.get("level", ""),
        "name": record.get("name", ""),
        "description": record.get(desc_field, "") if desc_field else "",
        "descriptionSuggestion": record.get("descriptionSuggestion", ""),
        "condition": record.get("condition", ""),
        "condition_queries": record.get("condition_queries", []),
        "summarySuggestion": record.get("summarySuggestion", ""),
        "summary": record.get("summary", ""),
    }
    # L5 指标节点附带 SQL 数据源配置
    if record.get("sql_config"):
        node["sql_config"] = record["sql_config"]
    return node


def main():
    all_nodes = []

    for filename, level, desc_field in INPUT_CONFIGS:
        path = os.path.join(_KB_DIR, filename)
        if not os.path.exists(path):
            logging.info("文件不存在，跳过: %s", filename)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logging.error("%s 不是数组，跳过", filename)
            continue

        nodes = [extract_node(r, desc_field) for r in data]
        all_nodes.extend(nodes)
        logging.info("[%s] %d 条", level, len(nodes))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_nodes, f, ensure_ascii=False, indent=2)

    logging.info("合并完成，共 %d 个节点 → %s", len(all_nodes), OUTPUT_FILE)

    sync_target = os.path.join(_KB_DIR, "node.json")
    shutil.copy(OUTPUT_FILE, sync_target)
    logging.info("已同步到 node.json")


if __name__ == "__main__":
    main()
