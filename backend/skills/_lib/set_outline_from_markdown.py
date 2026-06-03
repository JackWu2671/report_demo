"""
set_outline_from_markdown.py — set_outline 工具实现。

LLM 根据知识图谱返回的节点，自行组合构造大纲并调用 set_outline 工具，
将其解析为 outline_tree 并渲染到前端。

入参为 JSON 节点列表（工具参数原生数组，经 json.loads 直接得到 Python
对象），不经过 YAML 文本，因此不受换行/缩进等空白格式影响——这正是从
YAML 字符串改为 JSON 结构的原因（YAML 被 LLM 压成一行即解析失败）。
保留 set_outline_from_yaml 仅供向后兼容/其他入口使用。

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

from outline_utils import from_data, from_yaml, to_markdown, to_yaml, VIRTUAL_ROOT_ID

logger = logging.getLogger(__name__)


async def set_outline_from_tree(outline) -> dict:
    """将 LLM 通过 JSON 工具参数传入的大纲节点列表还原为 outline_tree，渲染到前端。"""
    node_count = len(outline) if isinstance(outline, list) else (1 if outline else 0)
    logger.info("[Tool:set_outline_from_tree] 顶层节点数=%d", node_count)

    if not outline:
        return _error("outline 不能为空")

    tree = from_data(outline)
    if tree is None:
        return _error("大纲结构解析失败，请检查是否为合法的节点列表（每项含 id/name，子节点放在 children 数组）")

    return _finalize(tree)


async def set_outline_from_yaml(outline_yaml: str) -> dict:
    """将 LLM 构造的 YAML 大纲文本解析为 outline_tree，渲染到前端（兼容旧入口）。"""
    logger.info("[Tool:set_outline_from_yaml] text_len=%d", len(outline_yaml))

    if not outline_yaml.strip():
        return _error("outline_yaml 不能为空")

    tree = from_yaml(outline_yaml)
    if tree is None:
        return _error("YAML 解析失败，请检查格式是否正确")

    return _finalize(tree)


def _finalize(tree: dict) -> dict:
    """包裹虚拟根节点并渲染三视图，组装成功返回值。"""
    # 确保有虚拟根节点，支持 add_node parent_id=""
    if tree.get("id") != VIRTUAL_ROOT_ID:
        wrapped = {
            "id": VIRTUAL_ROOT_ID, "name": "", "level": 0,
            "description": "", "children": [tree],
        }
    else:
        wrapped = tree

    top_count = len(wrapped.get("children", []))
    logger.info("[Tool:set_outline] 解析完成，顶层章节数=%d", top_count)
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "outline_yaml": to_yaml(wrapped),
        "message": "",
    }


def _error(message: str) -> dict:
    return {"status": "error", "outline_tree": {}, "markdown": "", "outline_yaml": "", "message": message}
