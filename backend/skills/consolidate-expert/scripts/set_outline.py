#!/usr/bin/env python3
"""
将 md_with_ids 格式大纲文本解析并写入会话状态。

用法:
  python3 set_outline.py "<md_with_ids>"

节点之间用 \\n 分隔（字面量反斜线n），缩进用空格表示层级：
  "[L1 new_001] 标题：描述\\n  [L2 new_002] 章节：描述\\n    [Q L5_001] query名称"

成功时输出解析后的带 id Markdown 大纲（用于 LLM 确认）。
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_outline
from set_outline_from_markdown import set_outline_from_markdown


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "缺少 md_with_ids 参数"}), file=sys.stderr)
        sys.exit(1)

    # On Windows cmd.exe single quotes are literal, splitting args on spaces.
    # Rejoin all argv[1:], strip surrounding single quotes, then decode \n.
    raw = " ".join(sys.argv[1:]).strip()
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
    md_with_ids = raw.replace("\\n", "\n")

    result = await set_outline_from_markdown(md_with_ids)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["md_with_ids"], result["markdown"])
        print(result["md_with_ids"])
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["md_with_ids"], result["markdown"])
        print(result["md_with_ids"])
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
