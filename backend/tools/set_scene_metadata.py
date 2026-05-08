"""
set_scene_metadata.py — set_scene_metadata 工具实现。

在 set_outline_from_markdown 之后调用，补充 keywords 和 usage_conditions，
供 save_outline_template 使用。

Used by: agent1
"""

import logging

logger = logging.getLogger(__name__)


async def set_scene_metadata(keywords: list[str], usage_conditions: str) -> dict:
    """记录场景关键词和适用条件，合并到已有的 extraction 中。"""
    logger.info("[Tool:set_scene_metadata] keywords=%s", keywords)
    return {
        "status": "success",
        "keywords": keywords if isinstance(keywords, list) else [],
        "usage_conditions": usage_conditions,
        "message": "",
    }
