"""
outline_utils.py — 大纲三种表示之间的转化工具。

唯一数据源是 outline_tree（dict），从它派生出三种视图：

  to_markdown(tree)  → 纯 Markdown，供用户阅读
  to_yaml(tree)      → 简洁 YAML，供 LLM 上下文使用
  to_clean_json(tree)→ 干净 JSON dict，供程序存储/执行

逆向解析：

  from_data(obj)     → outline_tree，将已解析的 JSON 节点结构（list/dict）还原为树

三者可从同一个 tree 独立生成，互不依赖，也不需要 node.json / relation.json。
LLM 上下文用 to_yaml 只读输出；写回走 from_data（JSON 结构，不依赖空白格式）。
"""

import re

import yaml

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


# ── YAML（LLM 上下文视图）────────────────────────────────────────

_YAML_OMIT = {
    "level", "exec_sql", "apiName", "extracted_table",
    "renderType", "colX", "colY", "summarySuggestion", "keywords", "score",
}


def to_yaml(tree: dict) -> str:
    """
    将大纲树渲染为简洁 YAML，供 LLM 上下文使用。

    只保留 id/name/description/condition/condition_queries/children，
    省略 level、SQL 字段和空的可选字段。
    虚拟根节点（__root__）被跳过，其子节点作为顶层列表渲染；
    但在 YAML 头部保留一行注释，使 LLM 知道新增顶层章节时 parent_id 填 __root__。
    """
    if _is_virtual_root(tree):
        nodes = [_yaml_node(c) for c in tree.get("children", [])]
        body = yaml.dump(nodes, allow_unicode=True, default_flow_style=False, sort_keys=False)
        return f"# 顶层节点的父节点均为虚拟根 __root__（add_node 新增顶层章节时 parent_id 填 \"__root__\"）\n{body}"
    nodes = [_yaml_node(tree)]
    return yaml.dump(nodes, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _yaml_node(node: dict) -> dict:
    out: dict = {"id": node.get("id", ""), "name": node.get("name", "")}
    if node.get("description"):
        out["description"] = node["description"]
    if node.get("condition"):
        out["condition"] = node["condition"]
    if node.get("condition_queries"):
        out["condition_queries"] = node["condition_queries"]
    children = [_yaml_node(c) for c in node.get("children", [])]
    if children:
        out["children"] = children
    return out


# ── YAML → outline_tree（逆向解析）───────────────────────────────

_LEVEL_RE = re.compile(r'^(?:new_)?L(\d+)_')


def _infer_level(node_id: str) -> int:
    """从节点 ID 前缀推断 level：L1_xxx→1, L5_xxx→5, new_L2_xxx→2；无法识别→5。"""
    m = _LEVEL_RE.match(node_id)
    return int(m.group(1)) if m else 5


def _yaml_to_node(item: dict, depth: int = 1) -> dict:
    node_id = str(item.get("id", ""))
    m = _LEVEL_RE.match(node_id)
    # level 优先从 id 前缀显式读取：
    #   KB 节点 L1_/L5_… ，新建结构节点约定命名 new_L2_xxx / new_L3_xxx（显式编码层级）。
    # 兜底：未按约定命名的 new_ 节点按树深度推断并封顶 4，确保结构节点绝不等于 5
    # （level==5 是"查询指标叶子"的判定标志，会被报告当 SQL 指标处理）。
    level = int(m.group(1)) if m else min(depth, 4)
    return {
        "id": node_id,
        "name": str(item.get("name", "")),
        "level": level,
        "description": str(item.get("description", "")),
        "condition": str(item.get("condition", "")),
        "condition_queries": list(item.get("condition_queries") or []),
        "children": [_yaml_to_node(c, depth + 1) for c in (item.get("children") or [])],
    }


def from_data(data) -> dict | None:
    """
    将已解析的结构（list / dict，通常来自 JSON 工具参数）还原为 outline_tree。

    本函数接收已解析好的 Python 对象（不经过 YAML 文本），因此不受换行/缩进
    等空白格式影响——set_outline 工具用 JSON 数组而非 YAML 字符串正是为此。
    顶层为列表时包裹虚拟根节点；顶层为单个 dict 时直接返回；空或类型不符返回 None。
    """
    if not data:
        return None
    if isinstance(data, list):
        roots = [_yaml_to_node(item) for item in data if isinstance(item, dict)]
        if not roots:
            return None
        if len(roots) == 1:
            return roots[0]
        return {
            "id": VIRTUAL_ROOT_ID, "name": "", "level": 0,
            "description": "", "condition": "", "condition_queries": [],
            "children": roots,
        }
    if isinstance(data, dict):
        return _yaml_to_node(data)
    return None


# ── 干净 JSON（程序视图）─────────────────────────────────────────

def to_clean_json(tree: dict) -> dict:
    """
    将大纲树导出为干净 JSON，去除检索/内部字段（keywords、score 等）。

    保留字段：id, name, level, description, condition, condition_queries, summarySuggestion,
             renderType, colX, colY, apiName, exec_sql, extracted_table（L5）, children
    缺失的通用字段补默认值，确保所有层级的节点结构一致。
    """
    _KEEP = {"id", "name", "level", "description", "condition", "condition_queries", "summarySuggestion",
             "renderType", "colX", "colY", "apiName", "exec_sql", "extracted_table"}
    node = {k: v for k, v in tree.items() if k in _KEEP}
    # 补默认值：保证任意层级的节点都有这三个字段
    node.setdefault("condition", "")
    node.setdefault("condition_queries", [])
    node.setdefault("summarySuggestion", "")
    node["children"] = [to_clean_json(c) for c in tree.get("children", [])]
    return node
