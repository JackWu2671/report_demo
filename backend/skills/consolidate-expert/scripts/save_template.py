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
import logging
import uuid
from datetime import datetime

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import get_outline_tree, get_extraction

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.environ.get("REPORT_BACKEND_DIR", "") or os.path.dirname(
    os.path.dirname(os.path.dirname(_SCRIPTS))
)
_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")


async def save_outline_template(extraction: dict, outline_tree: dict) -> dict:
    if not extraction or not extraction.get("scene_name"):
        return {"status": "error", "path": "", "scene_name": "",
                "message": "缺少场景元数据，请先调用 analyze_expert_knowledge。"}

    if not outline_tree:
        return {"status": "error", "path": "", "scene_name": "",
                "message": "当前没有大纲，请先调用 analyze_expert_knowledge。"}

    template_id = str(uuid.uuid4())
    scene_name = extraction["scene_name"]
    template = {
        "id": template_id,
        "scene_name": scene_name,
        "keywords": extraction.get("keywords", []),
        "summary": extraction.get("summary", ""),
        "usage_conditions": extraction.get("usage_conditions", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outline": outline_tree,
    }

    os.makedirs(_TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(_TEMPLATE_DIR, f"{template_id}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    logger.info("[save_template] 已保存: %s (id=%s)", path, template_id)
    return {"status": "success", "path": path, "scene_name": scene_name,
            "template_id": template_id, "message": ""}


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
