#!/usr/bin/env python3
"""
对当前大纲应用 patch 操作，写入会话状态。

用法:
  python3 modify_outline.py '<ops_json>'

ops_json 是 JSON 数组，支持的操作：
  add_node            node_id, parent_id?
  delete_node         node_id
  modify_node_name    node_id, value
  modify_node_description  node_id, value  （query 节点描述决定查询范围）
  modify_node_condition    node_id, value  （格式：「当……时，本节才展示」）
  keep_only_node      node_id

成功时输出修改后的带 id Markdown 大纲。
跳过的操作在末尾以 # SKIPPED: 开头输出。
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import get_outline_tree, set_outline
from modify_outline import modify_outline


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "缺少 ops_json 参数"}), file=sys.stderr)
        sys.exit(1)

    # On Windows cmd.exe, single quotes are NOT string delimiters, so a
    # single-quoted JSON arg gets split on spaces into multiple argv entries.
    # Rejoin them and strip surrounding single quotes if present.
    raw = " ".join(sys.argv[1:]).strip()
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]

    try:
        ops = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"ops_json 解析失败：{e}"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    outline_tree = get_outline_tree()
    if not outline_tree:
        print(json.dumps({"status": "error", "message": "当前没有大纲，请先调用 build_outline 或 load_template"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    result = await modify_outline(ops, outline_tree)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["md_with_ids"], result["markdown"])
        print(result["md_with_ids"])
        for s in result.get("skipped", []):
            reason = s.get("_skip_reason", "未知原因") if isinstance(s, dict) else str(s)
            op = s.get("op", "?") if isinstance(s, dict) else "?"
            nid = s.get("node_id", "") if isinstance(s, dict) else ""
            print(f"# SKIPPED: {op} node_id={nid} → {reason}")
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
