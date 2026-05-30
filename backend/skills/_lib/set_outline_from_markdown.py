"""
set_outline_from_markdown.py — set_outline_from_yaml 工具实现。

LLM 根据知识图谱返回的节点，自行组合构造 YAML 格式的大纲文本，
调用此工具将文本解析为 outline_tree 并渲染到前端。

场景元数据仅含 scene_name 和 summary；keywords / usage_conditions 由
set_scene_metadata 工具单独设置。

Used by: agent1
"""

import logging
import os
import sys

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from outline_utils import from_yaml, to_markdown, to_yaml, VIRTUAL_ROOT_ID

logger = logging.getLogger(__name__)


async def set_outline_from_yaml(outline_yaml: str) -> dict:
    """将 LLM 构造的 YAML 大纲文本解析为 outline_tree，渲染到前端。"""
    logger.info("[Tool:set_outline_from_yaml] text_len=%d", len(outline_yaml))

    if not outline_yaml.strip():
        return _error("outline_yaml 不能为空")

    tree = from_yaml(outline_yaml)
    if tree is None:
        return _error("YAML 解析失败，请检查格式是否正确")

    # 确保有虚拟根节点，支持 add_node parent_id=""
    if tree.get("id") != VIRTUAL_ROOT_ID:
        wrapped = {
            "id": VIRTUAL_ROOT_ID, "name": "", "level": 0,
            "description": "", "children": [tree],
        }
    else:
        wrapped = tree

    top_count = len(wrapped.get("children", []))
    logger.info("[Tool:set_outline_from_yaml] 解析完成，顶层章节数=%d", top_count)
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "outline_yaml": to_yaml(wrapped),
        "message": "",
    }


def _error(message: str) -> dict:
    return {"status": "error", "outline_tree": {}, "markdown": "", "outline_yaml": "", "message": message}
