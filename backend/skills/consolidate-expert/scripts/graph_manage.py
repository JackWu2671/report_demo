#!/usr/bin/env python3
"""
执行知识图谱融合写入。

用法:
  python3 graph_manage.py --template-id <id> \
      --add-nodes '<json_array>' \
      --enrich-nodes '<json_array>'

add_nodes 每项格式：{"level": 2|3|4|5, "name": "...", "keywords": [...], "description": "...", "parent_id": "L2_001", "condition": "当...时才展示（可选）"}
  level=5 为 query 节点，description 即查询参数，parent_id 须为 L4 节点
enrich_nodes 每项格式：{"node_id": "L3_001", "append": "补充描述"}

成功时输出 JSON 摘要：{"added": [...], "enriched": [...], "message": "..."}
"""
import sys
import os
import asyncio
import json
import argparse

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from graph_manage import graph_manage


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--add-nodes", default="[]")
    parser.add_argument("--enrich-nodes", default="[]")
    args = parser.parse_args()

    try:
        add_nodes = json.loads(args.add_nodes)
        enrich_nodes = json.loads(args.enrich_nodes)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"JSON 解析失败：{e}"}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)

    result = await graph_manage(
        template_id=args.template_id,
        add_nodes=add_nodes,
        enrich_nodes=enrich_nodes,
    )

    print(json.dumps({
        "added": result["added_nodes"],
        "enriched": result["enriched_nodes"],
        "message": result["message"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
