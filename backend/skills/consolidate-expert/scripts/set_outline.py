#!/usr/bin/env python3
"""
将 YAML 格式大纲文本解析并写入会话状态。

用法（从 stdin 读取 YAML）:
  echo "<yaml>" | python3 set_outline.py
  python3 set_outline.py < outline.yaml

成功时输出解析后的 YAML 大纲（用于 LLM 确认）。
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_outline
from set_outline_from_markdown import set_outline_from_yaml


async def main():
    outline_yaml = sys.stdin.read()
    if not outline_yaml.strip():
        print(json.dumps({"status": "error", "message": "stdin 为空，请通过管道传入 YAML 大纲"}), file=sys.stderr)
        sys.exit(1)

    result = await set_outline_from_yaml(outline_yaml)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["outline_yaml"], result["markdown"])
        print(result["outline_yaml"])
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
