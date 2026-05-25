#!/usr/bin/env python3
"""
merge_sample_questions.py — 合并 appSampleQuestion.json 和 sampleQuestion.json

用法:
  python3 merge_sample_questions.py

输出:
  expert_knowledge/评估指标.json

字段顺序: nodeId, level, id, name, answer, domain, renderType, colX, colY
  nodeId  — 短编号 L5_001 / L5_002 ...（合并后按顺序生成）
  level   — 固定 "评估指标"
  id      — 原始 UUID
  name    — 指标名称（源字段 question）
缺失字段补 null。
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

INPUT_FILES = [
    os.path.join(_KB_DIR, "appSampleQuestion.json"),
    os.path.join(_KB_DIR, "sampleQuestion.json"),
]
OUTPUT_FILE  = os.path.join(_KB_DIR, "评估指标.json")
NODE_PREFIX  = "L5"
NODE_START   = 1

FIELDS = ["id", "question", "answer", "domain", "renderType", "colX", "colY"]


def make_node_id(index: int) -> str:
    return f"{NODE_PREFIX}_{index:03d}"


def normalize_answer(raw_answer) -> str | None:
    """
    确保 answer JSON 字符串里包含 apiName: "NL2SQL"。
    - 若 answer 为 null / 非字符串，原样返回
    - 若解析失败，原样返回（不破坏原始数据）
    - 若已有 apiName，不覆盖
    """
    if not isinstance(raw_answer, str):
        return raw_answer
    try:
        obj = json.loads(raw_answer)
    except json.JSONDecodeError:
        return raw_answer  # 解析失败，保持原样

    if "apiName" not in obj:
        # 把 apiName 插到最前面，保持可读性
        obj = {"apiName": "NL2SQL", **obj}

    return json.dumps(obj, ensure_ascii=False)


def extract(record: dict) -> dict:
    """从原始记录里只取 7 个字段，缺失的补 None，并统一 answer 格式。"""
    item = {field: record.get(field, None) for field in FIELDS}
    item["answer"] = normalize_answer(item["answer"])
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
            print(f"[警告] 文件不存在，跳过: {path}", file=sys.stderr)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"[错误] {filename} 不是 JSON 数组，跳过", file=sys.stderr)
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
            item["id"]    = make_node_id(NODE_START + len(merged))
            item["level"] = 5
            # 调整字段顺序：id / level 放最前
            item = {"id": item.pop("id"), "level": item.pop("level"), **item}
            merged.append(item)

        added = len(merged) - before
        print(f"[{filename}] 读入 {len(data)} 条，新增 {added} 条，跳过重复 {dup} 条")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成，共 {len(merged)} 条，nodeId 范围: {make_node_id(NODE_START)} ~ {make_node_id(NODE_START + len(merged) - 1)} → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
