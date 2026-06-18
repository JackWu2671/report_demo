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
import logging

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import set_extraction

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-name", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--keywords", required=True, help="逗号分隔")
    parser.add_argument("--usage-conditions", required=True)
    args = parser.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    logger.info("[set_metadata] scene=%r keywords=%s", args.scene_name, keywords)

    set_extraction({
        "scene_name": args.scene_name,
        "summary": args.summary,
        "keywords": keywords,
        "usage_conditions": args.usage_conditions,
    })

    print(json.dumps({
        "scene_name": args.scene_name,
        "keywords": keywords,
        "summary": args.summary,
        "usage_conditions": args.usage_conditions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
