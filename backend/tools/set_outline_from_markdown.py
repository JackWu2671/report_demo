"""
set_outline_from_markdown.py — set_outline_from_markdown 工具实现。

LLM 根据 search_graph_tree 返回的节点，自行组合构造 md_with_ids 格式的大纲文本，
调用此工具将文本解析为 outline_tree 并渲染到前端。

同时接收场景元数据（scene_name、keywords、summary、usage_conditions），
存入 memory 供后续 save_outline_template 使用。

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


async def set_outline_from_markdown(
    md_with_ids: str,
    scene_name: str,
    keywords: list[str],
    summary: str,
    usage_conditions: str,
) -> dict:
    """
    将 LLM 构造的 md_with_ids 文本解析为 outline_tree，并封装场景元数据。

    Args:
        md_with_ids       : LLM 用知识图谱节点 id 组合而成的大纲文本
        scene_name        : 场景名称，用于模板保存
        keywords          : 关键词列表
        summary           : 场景简要描述
        usage_conditions  : 适用条件说明

    Returns:
        {status, outline_tree, markdown, md_with_ids, extraction, message}
    """
    logger.info("[Tool:set_outline_from_markdown] scene=%r text_len=%d", scene_name, len(md_with_ids))

    if not md_with_ids.strip():
        return _error("md_with_ids 不能为空")

    # 解析 md_with_ids，支持多个顶层章节
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

    # 用虚拟根节点包裹，与 agent2 保持一致，支持 add_node parent_id=""
    wrapped = {"id": "__root__", "name": "", "level": 0, "description": "", "children": roots}

    extraction = {
        "scene_name": scene_name,
        "keywords": keywords if isinstance(keywords, list) else [],
        "summary": summary,
        "usage_conditions": usage_conditions,
    }

    logger.info("[Tool:set_outline_from_markdown] 解析完成，顶层章节数=%d", len(roots))
    return {
        "status": "success",
        "outline_tree": wrapped,
        "markdown": to_markdown(wrapped),
        "md_with_ids": to_markdown_with_ids(wrapped),
        "extraction": extraction,
        "message": "",
    }


def _error(message: str) -> dict:
    return {"status": "error", "outline_tree": {}, "markdown": "", "md_with_ids": "",
            "extraction": {}, "message": message}
