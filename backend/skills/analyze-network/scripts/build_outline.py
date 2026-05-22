#!/usr/bin/env python3
"""
从锚节点展开知识图谱子树，构建大纲并写入会话状态。

用法:
  python3 build_outline.py <anchor_id>

成功时输出带 id 的 Markdown 大纲。
失败时输出 JSON：{"status": "error", "message": "..."}
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_outline
from tools.build_outline_from_anchor import build_outline_from_anchor


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "缺少 anchor_id 参数"}), file=sys.stderr)
        sys.exit(1)

    anchor_id = sys.argv[1]
    result = await build_outline_from_anchor(anchor_id)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["md_with_ids"], result["markdown"])
        print(result["md_with_ids"])
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
