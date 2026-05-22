#!/usr/bin/env python3
"""
填写场景元数据，写入会话状态。

用法:
  python3 set_metadata.py --scene-name "名称" --summary "摘要" \
      --keywords "kw1,kw2,kw3" --usage-conditions "适用条件"

成功时输出已记录的元数据 JSON。
"""
import sys
import os
import asyncio
import json
import argparse

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_extraction
from tools.set_scene_metadata import set_scene_metadata


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--keywords", required=True, help="逗号分隔")
    parser.add_argument("--usage-conditions", required=True)
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    result = await set_scene_metadata(
        scene_name=args.scene_name,
        summary=args.summary,
        keywords=keywords,
        usage_conditions=args.usage_conditions,
    )

    if result["status"] == "success":
        set_extraction({
            "scene_name": result["scene_name"],
            "summary": result["summary"],
            "keywords": result["keywords"],
            "usage_conditions": result["usage_conditions"],
        })
        print(json.dumps({
            "scene_name": result["scene_name"],
            "keywords": result["keywords"],
            "summary": result["summary"],
            "usage_conditions": result["usage_conditions"],
        }, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"status": "error", "message": result["message"]}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
