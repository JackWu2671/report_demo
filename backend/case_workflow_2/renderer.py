"""
renderer.py — Step 7: 将大纲子树渲染为 Markdown 文本。

输入: subtree dict（来自 subtree.build_subtree 或 patcher.apply_patch）
输出: Markdown 格式字符串

渲染规则：
  - 锚节点（根）对应 # 标题，每深一层标题级别加一（最深 ######）
  - 节点的 description 字段渲染为标题下方段落
  - 节点的 params 字段渲染为 > 参数设置 标注行
"""

import logging

logger = logging.getLogger(__name__)


def render_outline(subtree: dict) -> str:
    """
    将大纲子树直接渲染为 Markdown，无需调用 LLM。

    Args:
        subtree: build_subtree() 或 apply_patch() 返回的树 dict

    Returns:
        Markdown 格式的报告大纲字符串
    """
    blocks = _render_node(subtree, heading_level=1)
    outline = "\n\n".join(blocks)
    logger.info("[Step 7] 大纲渲染完成: %d 字符", len(outline))
    return outline


# ── 内部工具 ──────────────────────────────────────────────────

def _render_node(node: dict, heading_level: int) -> list[str]:
    """
    递归渲染单个节点及其子节点为 Markdown block 列表。

    Args:
        node          : 节点 dict（含 name, description, params, children）
        heading_level : 当前标题级别（1 = #, 2 = ##, ...）

    Returns:
        每个元素为一个 Markdown block（标题 + 可选描述/参数）
    """
    prefix = "#" * min(heading_level, 6)
    block = f"{prefix} {node['name']}"

    if node.get("description"):
        block += f"\n\n{node['description']}"

    if node.get("params"):
        param_str = "、".join(
            f"{k}: {v['value']}{v['unit']}" for k, v in node["params"].items()
        )
        block += f"\n\n> 参数设置 — {param_str}"

    blocks = [block]
    for child in node.get("children", []):
        blocks.extend(_render_node(child, heading_level + 1))
    return blocks
