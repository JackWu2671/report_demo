"""
outline_utils.py — 大纲三种表示之间的转化工具。

唯一数据源是 outline_tree（dict），从它派生出三种视图：

  to_markdown(tree)           → 纯 Markdown，供用户阅读
  to_markdown_with_ids(tree)  → 带 id 缩进树，供 LLM 上下文使用
  to_clean_json(tree)         → 干净 JSON dict，供程序存储/执行

逆向解析：

  from_md_with_ids(text)      → outline_tree，将 LLM 输出的 md_with_ids 还原为树

三者可从同一个 tree 独立生成，互不依赖，也不需要 node.json / relation.json。
"""

import re

VIRTUAL_ROOT_ID = "__root__"


def _is_virtual_root(tree: dict) -> bool:
    return tree.get("id") == VIRTUAL_ROOT_ID


# ── 纯 Markdown（用户视图）─────────────────────────────────────

def to_markdown(tree: dict) -> str:
    """
    将大纲树渲染为纯 Markdown，用户可读，不含 id。

    渲染规则：
      - 根节点对应 # 标题，每深一层加一级（最深 ######）
      - description 渲染为标题下方段落
      - 虚拟根节点（__root__）被跳过，其子节点作为顶层章节渲染
    """
    if _is_virtual_root(tree):
        blocks = []
        for child in tree.get("children", []):
            blocks.extend(_md_node(child, heading_level=1))
        return "\n\n".join(blocks)
    blocks = _md_node(tree, heading_level=1)
    return "\n\n".join(blocks)


def _md_node(node: dict, heading_level: int) -> list[str]:
    prefix = "#" * min(heading_level, 6)
    block = f"{prefix} {node['name']}"

    if node.get("description"):
        block += f"\n\n{node['description']}"

    if node.get("condition"):
        block += f"\n\n@if {node['condition']}"

    blocks = [block]
    for child in node.get("children", []):
        blocks.extend(_md_node(child, heading_level + 1))
    return blocks


# ── 带 id Markdown（LLM 上下文视图）──────────────────────────────

def to_markdown_with_ids(tree: dict) -> str:
    """
    将大纲树渲染为带 id 的缩进树，供 LLM 上下文使用。

    格式：
      [L1 L1_001] 节点名称：description（无 description 则省略冒号后内容）
        [L2 L2_003] 子节点：描述
          [L3 L3_011] 孙节点：描述

    LLM 可通过 id 精确引用节点，输出 patch 操作时不会指错目标。
    虚拟根节点（__root__）被跳过，其子节点作为顶层章节渲染。
    """
    lines: list[str] = []
    if _is_virtual_root(tree):
        for child in tree.get("children", []):
            _id_md_node(child, depth=0, lines=lines)
    else:
        _id_md_node(tree, depth=0, lines=lines)
    return "\n".join(lines)


def _id_md_node(node: dict, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    nid = node.get("id", "?")
    level = node.get("level", "?")
    name = node.get("name", "")

    level_str = "Q" if level == 5 else f"L{level}"
    suffix = f"：{node['description']}" if node.get("description") else ""
    if node.get("condition"):
        suffix += f"｜条件：{node['condition']}"
    lines.append(f"{indent}[{level_str} {nid}] {name}{suffix}")

    for child in node.get("children", []):
        _id_md_node(child, depth + 1, lines)


# ── md_with_ids → outline_tree（逆向解析）────────────────────────

_LINE_RE = re.compile(r'^(\s*)\[(L(\d+)|Q)\s+(\S+)\]\s+(.+)$')


def _parse_level(level_token: str, level_digit: str) -> int:
    """将格式标记转为内部 level 整数：Q → 5，L1..L4 → 1..4。"""
    return 5 if level_token == "Q" else int(level_digit)


def from_md_with_ids(text: str) -> dict | None:
    """
    将 LLM 输出的 md_with_ids 文本解析为 outline_tree dict。

    格式约定（与 to_markdown_with_ids 一致）：
      {indent}[L{level} {id}] {name}：{description}   （章节节点）
      {indent}[Q {id}] {name}：{description}           （query 节点）
      缩进每层 2 个空格，描述可省略。

    Returns:
        根节点 dict，解析失败时返回 None。
    """
    nodes: list[tuple[int, dict]] = []  # (depth, node)

    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        indent, level_token, level_digit, node_id, rest = m.groups()
        depth = len(indent) // 2

        condition = ''
        if '｜条件：' in rest:
            rest, condition = rest.split('｜条件：', 1)

        if '：' in rest:
            name, description = rest.split('：', 1)
        else:
            name, description = rest, ''

        node = {
            'id': node_id.strip(),
            'name': name.strip(),
            'level': _parse_level(level_token, level_digit),
            'description': description.strip(),
            'condition': condition.strip(),
            'children': [],
        }

        # 找父节点：弹出所有深度 >= 当前的栈帧
        while nodes and nodes[-1][0] >= depth:
            nodes.pop()

        if nodes:
            nodes[-1][1]['children'].append(node)

        nodes.append((depth, node))

    # 根节点是第一个 depth=0 的节点
    for depth, node in nodes:
        if depth == 0:
            return node
    return None


# ── 干净 JSON（程序视图）─────────────────────────────────────────

def to_clean_json(tree: dict) -> dict:
    """
    将大纲树导出为干净 JSON，去除检索/内部字段（keywords、score 等）。

    保留字段：id, name, level, description, children
    """
    _KEEP = {"id", "name", "level", "description", "condition"}
    node = {k: v for k, v in tree.items() if k in _KEEP}
    node["children"] = [to_clean_json(c) for c in tree.get("children", [])]
    return node
