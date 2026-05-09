"""
subtree.py — Step 6: 从锚节点递归构建完整子树。

输入: anchor_id, nodes_dict, children_map
输出: subtree dict，结构为 {id, name, level, description, children: [...]}
      children 列表按 relation.json 中的 order 顺序排列
"""

import logging

logger = logging.getLogger(__name__)


def build_subtree(anchor_id: str, nodes_dict: dict, children_map: dict) -> dict:
    """
    从锚节点开始，递归构建包含所有后代节点的子树 dict。

    Args:
        anchor_id    : 锚节点 id，必须存在于 nodes_dict 中
        nodes_dict   : {node_id -> node_dict}
        children_map : {parent_id -> [child_id, ...]}

    Returns:
        subtree dict，每个节点包含原始字段 + children 列表

    Raises:
        ValueError: 锚节点 id 不在知识图谱中
    """
    if anchor_id not in nodes_dict:
        raise ValueError(f"锚节点 '{anchor_id}' 不在知识图谱中")

    tree = _build_recursive(anchor_id, nodes_dict, children_map)
    logger.info(
        "[Step 6] 子树构建完成: 根='%s', 共 %d 个节点",
        nodes_dict[anchor_id]["name"], _count_nodes(tree),
    )
    return tree


# ── 内部工具 ──────────────────────────────────────────────────

def _build_recursive(node_id: str, nodes_dict: dict, children_map: dict) -> dict:
    """递归构建节点子树，保留节点的所有原始字段。"""
    node = dict(nodes_dict[node_id])
    node["children"] = [
        _build_recursive(cid, nodes_dict, children_map)
        for cid in children_map.get(node_id, [])
    ]
    return node


def _count_nodes(node: dict) -> int:
    """统计子树节点总数（含根节点）。"""
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))
