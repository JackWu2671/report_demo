#!/usr/bin/env python3
"""
merge_sample_questions.py — 合并 appSampleQuestion.json 和 sampleQuestion.json

用法:
  python3 merge_sample_questions.py

输出:
  expert_knowledge/mergedSampleQuestions.json

字段顺序: id, question, answer, domain, renderType, colX, colY
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
OUTPUT_FILE = os.path.join(_KB_DIR, "mergedSampleQuestions.json")

FIELDS = ["id", "question", "answer", "domain", "renderType", "colX", "colY"]


def extract(record: dict) -> dict:
    """从原始记录里只取 7 个字段，缺失的补 None。"""
    return {field: record.get(field, None) for field in FIELDS}


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
            rid = item["id"]
            if rid in seen_ids:
                dup += 1
                continue
            seen_ids.add(rid)
            merged.append(item)

        added = len(merged) - before
        print(f"[{filename}] 读入 {len(data)} 条，新增 {added} 条，跳过重复 {dup} 条")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n合并完成，共 {len(merged)} 条 → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
