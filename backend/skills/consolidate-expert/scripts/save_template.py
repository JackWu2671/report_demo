#!/usr/bin/env python3
"""
将当前会话的大纲和元数据保存为可复用模板。

用法:
  python3 save_template.py

成功时输出 JSON：{"template_id": "...", "scene_name": "...", "path": "..."}
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import get_outline_tree, get_extraction
from tools.save_template import save_outline_template


async def main():
    outline_tree = get_outline_tree()
    extraction = get_extraction()

    if not outline_tree:
        print(json.dumps({"status": "error", "message": "当前没有大纲"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
    if not extraction.get("scene_name"):
        print(json.dumps({"status": "error", "message": "缺少场景元数据，请先调用 set_metadata"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    result = await save_outline_template(extraction, outline_tree)

    if result["status"] == "success":
        print(json.dumps({
            "template_id": result["template_id"],
            "scene_name": result["scene_name"],
            "path": result["path"],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
