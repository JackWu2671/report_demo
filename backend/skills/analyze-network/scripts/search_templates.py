#!/usr/bin/env python3
"""
语义检索大纲模板库。

用法:
  python3 search_templates.py "查询词" [--topk 3]

输出 JSON 数组，每项含 id、scene_name、summary、score。
无匹配时输出 []。
"""
import sys
import os
import argparse
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from tools.search_template import search_outline_templates


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索词")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    result = await search_outline_templates(args.query, args.topk)
    if result["status"] == "found":
        out = [
            {
                "id": c["id"],
                "scene_name": c["scene_name"],
                "summary": c.get("summary", ""),
                "usage_conditions": c.get("usage_conditions", ""),
                "score": round(c["score"], 4),
            }
            for c in result["candidates"]
        ]
    else:
        out = []

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
