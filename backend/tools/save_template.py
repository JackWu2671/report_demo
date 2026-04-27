"""
save_template.py — save_outline_template tool implementation.

Saves the current outline_tree + extraction metadata to templates/{scene_name}.json.

Uses outline_tree (JSON) directly — does NOT re-parse the annotated Markdown,
because the tree may have been modified by modify_outline after generation.

Kept as a separate tool (not bundled with analyze) so the LLM must explicitly
call it — ensuring the user has consciously confirmed before saving.

Used by: agent1
"""

import json
import logging
import os
import sys
from datetime import datetime

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logger = logging.getLogger(__name__)
_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")


async def save_outline_template(extraction: dict, outline_tree: dict) -> dict:
    """
    Save extraction metadata + outline_tree to templates/{scene_name}.json.

    Args:
        extraction   : {scene_name, keywords, summary, usage_conditions}
        outline_tree : current clean JSON tree (may differ from annotated version)

    Returns:
        {status: "success"|"error", path, scene_name, message}
    """
    if not extraction or not extraction.get("scene_name"):
        return {"status": "error", "path": "", "scene_name": "",
                "message": "缺少场景元数据，请先调用 analyze_expert_knowledge。"}

    if not outline_tree:
        return {"status": "error", "path": "", "scene_name": "",
                "message": "当前没有大纲，请先调用 analyze_expert_knowledge。"}

    scene_name = extraction["scene_name"]
    template = {
        "scene_name": scene_name,
        "keywords": extraction.get("keywords", []),
        "summary": extraction.get("summary", ""),
        "usage_conditions": extraction.get("usage_conditions", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outline": outline_tree,
    }

    os.makedirs(_TEMPLATE_DIR, exist_ok=True)
    path = os.path.join(_TEMPLATE_DIR, f"{scene_name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    logger.info("[Tool:save_outline_template] 已保存: %s", path)
    return {"status": "success", "path": path, "scene_name": scene_name, "message": ""}
