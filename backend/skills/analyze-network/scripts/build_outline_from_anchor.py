#!/usr/bin/env python3
"""
从锚节点展开知识图谱子树，构建大纲并写入会话状态。

用法:
  python3 build_outline_from_anchor.py <anchor_id>

成功时输出 YAML 格式大纲。
失败时输出 JSON：{"status": "error", "message": "..."}
"""
import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from loader import load_resources
from outline_utils import to_clean_json, to_markdown, to_yaml
from session import set_outline
from subtree import build_subtree

logger = logging.getLogger(__name__)


async def build_outline_from_anchor(anchor_id: str) -> dict:
    _, nodes_dict, children_map = await load_resources()
    try:
        tree = build_subtree(anchor_id, nodes_dict, children_map)
    except ValueError as e:
        return {"status": "not_found", "outline_tree": {}, "markdown": "",
                "outline_yaml": "", "message": str(e)}
    clean_tree = to_clean_json(tree)
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": [clean_tree]}
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "outline_yaml": to_yaml(wrapped),
        "message": "",
    }


async def main():
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "缺少 anchor_id 参数"}), file=sys.stderr)
        sys.exit(1)

    anchor_id = sys.argv[1]
    result = await build_outline_from_anchor(anchor_id)

    if result["status"] == "success":
        set_outline(result["outline_tree"], result["outline_yaml"], result["markdown"])
        print(result["outline_yaml"])
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
