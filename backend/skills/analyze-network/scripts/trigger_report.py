#!/usr/bin/env python3
"""
触发报告生成：在 session 文件写入 generate_report 标记，
agent 检测到后向前端推送 start_report 事件，前端自动开始生成报告。

用法:
  python3 trigger_report.py
"""
import os
import sys

_BACKEND_DIR = os.environ.get("REPORT_BACKEND_DIR", "")
if _BACKEND_DIR and _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from skills._lib import session

data = session.read()
if not data.get("outline_tree"):
    print('{"status": "error", "message": "没有大纲，无法生成报告"}')
    sys.exit(1)

data["generate_report"] = True
session.write(data)
print("报告生成已触发")
