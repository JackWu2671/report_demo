#!/usr/bin/env python3
"""
migrate_mock_sql_config.py — 一次性迁移脚本

用途: 评估指标.json 已经改为新格式（用 sql_config 取代
answer/renderType/colX/colY），但 评估指标_mock.json 是用旧格式跑出来的，
已经带有耗时获取的 mock_data，不想重新拉一次。

本脚本按 id 对齐两个文件：
  - 结构（含 sql_config）以 评估指标.json 为准
  - mock_data 从旧的 评估指标_mock.json 里原样保留
输出覆盖回 评估指标_mock.json（覆盖前自动备份）。

用法:
  cd backend
  python3 scripts/migrate_mock_sql_config.py
"""

import json
import logging
import os
import shutil

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "reference")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

NODE_FILE = os.path.join(_KB_DIR, "评估指标.json")
MOCK_FILE = os.path.join(_KB_DIR, "评估指标_mock.json")


def main():
    for path, label in [(NODE_FILE, "评估指标.json"), (MOCK_FILE, "评估指标_mock.json")]:
        if not os.path.exists(path):
            logging.error("文件不存在: %s", path)
            return

    with open(NODE_FILE, encoding="utf-8") as f:
        nodes = json.load(f)
    with open(MOCK_FILE, encoding="utf-8") as f:
        old_mocks = json.load(f)

    old_mock_data = {r["id"]: r.get("mock_data") for r in old_mocks if r.get("id")}

    missing = 0
    merged = []
    for node in nodes:
        nid = node.get("id")
        mock_data = old_mock_data.get(nid)
        if mock_data is None and nid not in old_mock_data:
            missing += 1
        merged.append({**node, "mock_data": mock_data})

    backup = MOCK_FILE + ".bak"
    shutil.copy(MOCK_FILE, backup)
    logging.info("原文件已备份至 %s", backup)

    with open(MOCK_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    logging.info("迁移完成，共 %d 条，%d 条在旧 mock 文件里找不到（mock_data=null）→ %s",
                 len(merged), missing, MOCK_FILE)


if __name__ == "__main__":
    main()
