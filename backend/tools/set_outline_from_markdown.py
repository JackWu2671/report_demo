"""
set_outline_from_markdown.py — set_outline_from_markdown 工具实现。

LLM 根据 search_graph_tree 返回的节点，自行组合构造 md_with_ids 格式的大纲文本，
调用此工具将文本解析为 outline_tree 并渲染到前端。

场景元数据仅含 scene_name 和 summary；keywords / usage_conditions 由
set_scene_metadata 工具单独设置。

Used by: agent1
"""

import logging
import os
import re
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from outline_utils import to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)

_LINE_RE = re.compile(r'^(\s*)\[L(\d+)\s+(\S+)\]\s+(.+)$')


async def set_outline_from_markdown(md_with_ids: str) -> dict:
    """将 LLM 构造的 md_with_ids 文本解析为 outline_tree，渲染到前端。"""
    logger.info("[Tool:set_outline_from_markdown] text_len=%d", len(md_with_ids))

    if not md_with_ids.strip():
        return _error("md_with_ids 不能为空")

    roots: list[dict] = []
    stack: list[tuple[int, dict]] = []  # (depth, node)

    for line in md_with_ids.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        indent, level, node_id, rest = m.groups()
        depth = len(indent) // 2
        name, _, description = rest.partition('：')
        node = {
            'id': node_id.strip(),
            'name': name.strip(),
            'level': int(level),
            'description': description.strip(),
            'children': [],
        }
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if stack:
            stack[-1][1]['children'].append(node)
        else:
            roots.append(node)
        stack.append((depth, node))

    if not roots:
        return _error("解析失败，请检查 md_with_ids 格式是否正确（需含 [Lx id] 前缀）")

    # 用虚拟根节点包裹，支持 add_node parent_id=""
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": roots}

    logger.info("[Tool:set_outline_from_markdown] 解析完成，顶层章节数=%d", len(roots))
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "md_with_ids": to_markdown_with_ids(wrapped),
        "message": "",
    }


def _error(message: str) -> dict:
    return {"status": "error", "outline_tree": {}, "markdown": "", "md_with_ids": "", "message": message}
