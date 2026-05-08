"""
set_scene_metadata.py — set_scene_metadata 工具实现。

在 set_outline_from_markdown 之后调用，补充 keywords 和 usage_conditions，
供 save_outline_template 使用。

Used by: agent1
"""

import logging

logger = logging.getLogger(__name__)


async def set_scene_metadata(
    scene_name: str,
    summary: str,
    keywords: list[str],
    usage_conditions: str,
) -> dict:
    """记录场景元数据（名称、摘要、关键词、适用条件），合并写入 memory。"""
    logger.info("[Tool:set_scene_metadata] scene=%r keywords=%s", scene_name, keywords)
    return {
        "status": "success",
        "scene_name": scene_name,
        "summary": summary,
        "keywords": keywords if isinstance(keywords, list) else [],
        "usage_conditions": usage_conditions,
        "message": "",
    }
