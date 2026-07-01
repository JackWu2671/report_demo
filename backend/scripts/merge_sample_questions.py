#!/usr/bin/env python3
"""
merge_sample_questions.py — 合并 appSampleQuestion.json 和 sampleQuestion.json

用法:
  python3 merge_sample_questions.py

输出:
  reference/评估指标.json

字段顺序: nodeId, level, id, name, domain, sql_config
  nodeId  — 短编号 L5_001 / L5_002 ...（合并后按顺序生成）
  level   — 固定 "评估指标"
  id      — 原始 UUID
  name    — 指标名称（源字段 question）
  sql_config — SQL 数据源配置（见 docs/node-schema.md），由源字段 answer /
              renderType / colX / colY 转换而来：
                sql_config.exec_sql    ← answer.exec_sql
                sql_config.tables      ← answer.extracted_table
                sql_config.renderType  ← renderType
                sql_config.colX        ← colX
                sql_config.colY        ← colY
              answer 缺失或无 exec_sql 时 sql_config 为 null。
缺失字段补 null。
"""

import json
import logging
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "reference")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

INPUT_FILES = [
    os.path.join(_KB_DIR, "appSampleQuestion.json"),
    os.path.join(_KB_DIR, "sampleQuestion.json"),
]
OUTPUT_FILE = os.path.join(_KB_DIR, "评估指标.json")
NODE_PREFIX = "L5"
NODE_START = 1

FIELDS = ["id", "question", "answer", "domain", "renderType", "colX", "colY"]


def make_node_id(index: int) -> str:
    return f"{NODE_PREFIX}_{index:03d}"


def build_sql_config(raw_answer, render_type, col_x, col_y) -> dict | None:
    """
    把源字段 answer(JSON 字符串) + renderType/colX/colY 合并为 sql_config。
    answer 缺失、非字符串或解析失败、无 exec_sql 时返回 None。
    """
    if not isinstance(raw_answer, str):
        return None
    try:
        answer = json.loads(raw_answer)
    except json.JSONDecodeError:
        return None

    exec_sql = answer.get("exec_sql", "")
    if not exec_sql:
        return None

    extracted = answer.get("extracted_table", "[]")
    tables = json.loads(extracted) if isinstance(extracted, str) else (extracted or [])

    sql_config = {"exec_sql": exec_sql, "tables": tables}
    if render_type:
        sql_config["renderType"] = render_type
    if col_x:
        sql_config["colX"] = col_x
    if col_y:
        sql_config["colY"] = col_y
    return sql_config


def extract(record: dict) -> dict:
    """从原始记录里只取 7 个字段，缺失的补 None，并合并出 sql_config。"""
    item = {field: record.get(field, None) for field in FIELDS}
    item["sql_config"] = build_sql_config(
        item.pop("answer"), item.pop("renderType"), item.pop("colX"), item.pop("colY"),
    )
    # question → name 对齐其他层级；id(UUID) → uuid 对齐命名规范
    item["name"] = item.pop("question")
    item["uuid"] = item.pop("id")
    return item


def main():
    merged = []
    seen_ids = set()

    for path in INPUT_FILES:
        filename = os.path.basename(path)
        if not os.path.exists(path):
            logging.warning("文件不存在，跳过: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            logging.error("%s 不是 JSON 数组，跳过", filename)
            continue

        before = len(merged)
        dup = 0
        for record in data:
            item = extract(record)
            rid = item["uuid"]
            if rid in seen_ids:
                dup += 1
                continue
            seen_ids.add(rid)
            item["id"] = make_node_id(NODE_START + len(merged))
            item["level"] = 5
            # 调整字段顺序：id / level 放最前
            item = {"id": item.pop("id"), "level": item.pop("level"), **item}
            merged.append(item)

        added = len(merged) - before
        logging.info("[%s] 读入 %d 条，新增 %d 条，跳过重复 %d 条", filename, len(data), added, dup)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logging.info("合并完成，共 %d 条，nodeId 范围: %s ~ %s → %s",
                 len(merged),
                 make_node_id(NODE_START),
                 make_node_id(NODE_START + len(merged) - 1),
                 OUTPUT_FILE)


if __name__ == "__main__":
    main()
