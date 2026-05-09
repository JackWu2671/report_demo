"""
patcher.py — 将结构化操作列表应用到大纲树。

支持的 patch 操作:
  add_node              — 从知识图谱新增节点，挂到指定父节点下
  delete_node           — 删除指定节点及其所有子节点
  modify_node_name      — 修改节点的 name
  modify_node_description — 修改节点的 description
  modify_node_condition — 设置或修改节点展示条件
  keep_only_node        — 保留指定节点，删除同级兄弟节点
"""

import copy
import logging
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.loader import load_resources

logger = logging.getLogger(__name__)


# ── apply_patch ───────────────────────────────────────────────

def apply_patch(outline_tree: dict, ops: list[dict]) -> tuple[dict, list[dict]]:
    """
    将 patch 操作列表应用到大纲树，返回 (新树, 跳过/失败的操作列表)。

    Returns:
        (tree, skipped)
          tree    : deepcopy 后修改过的新树 dict
          skipped : 未成功执行的操作列表，每项含 op / node_id / reason 字段
    """
    tree = copy.deepcopy(outline_tree)
    skipped: list[dict] = []

    # 懒加载 KB 资源，仅当存在 add_node op 时才加载
    _kb_cache: dict | None = None

    def _get_kb():
        nonlocal _kb_cache
        if _kb_cache is None:
            _, nd, cm = load_resources()
            _kb_cache = {"nodes_dict": nd, "children_map": cm}
        return _kb_cache

    # 先收集所有 keep_only_node id，统一批量处理
    keep_ids = [op["node_id"] for op in ops if op.get("op") == "keep_only_node" and op.get("node_id")]
    if keep_ids:
        reasons = [op.get("reason", "") for op in ops if op.get("op") == "keep_only_node"]
        _keep_only_nodes(tree, keep_ids)
        logger.info("[Step 9] keep_only_node: 保留节点 %s | 原因: %s", keep_ids, " / ".join(r for r in reasons if r))

    for op in ops:
        node_id = op.get("node_id", "")
        reason = op.get("reason", "")
        op_name = op.get("op", "")

        if op_name == "keep_only_node":
            continue  # 已批量处理

        elif op_name == "add_node":
            subtree = op.get("subtree")
            if not subtree:
                kb = _get_kb()
                subtree = _build_kb_subtree(node_id, kb["nodes_dict"], kb["children_map"])
            if not subtree:
                msg = f"节点 {node_id} 在知识图谱中不存在"
                logger.warning("[Step 9] add_node: %s，跳过", msg)
                skipped.append({**op, "_skip_reason": msg})
                continue
            ids_before = _collect_ids(tree)
            parent_id = op.get("parent_id") or ""
            if not parent_id:
                tree.setdefault("children", []).append(subtree)
                logger.info("[Step 9] add_node: 新增顶层章节 %s（与现有一级章节平行）", node_id)
            else:
                added = _add_node(tree, parent_id, subtree)
                if not added:
                    tree.setdefault("children", []).append(subtree)
                    logger.warning("[Step 9] add_node: 未找到父节点 %s，已作为顶层章节新增", parent_id)
                else:
                    logger.info("[Step 9] add_node: 新增节点 %s → 父节点 %s | 原因: %s", node_id, parent_id, reason)
            new_ids = _collect_ids(subtree)
            duplicates = [i for i in new_ids if i in ids_before]
            if duplicates:
                logger.warning("[Step 9] add_node: 新增后发现重复节点 %s", duplicates)
                skipped.append({**op, "_skip_reason": f"新增成功但产生了重复节点: {duplicates}，请用 delete_node 删除重复项"})

        elif op_name == "delete_node":
            removed = _delete_node(tree, node_id)
            if removed:
                logger.info("[Step 9] delete_node: 已删除节点 %s | 原因: %s", node_id, reason)
            else:
                msg = f"节点 {node_id} 不存在"
                logger.warning("[Step 9] delete_node: 未找到节点 %s", node_id)
                skipped.append({**op, "_skip_reason": msg})

        elif op_name == "modify_node_name":
            found = _modify_field(tree, node_id, "name", op.get("value", ""))
            if found:
                logger.info("[Step 9] modify_node_name: 节点 %s | 原因: %s", node_id, reason)
            else:
                msg = f"节点 {node_id} 不存在"
                logger.warning("[Step 9] modify_node_name: 未找到节点 %s", node_id)
                skipped.append({**op, "_skip_reason": msg})

        elif op_name == "modify_node_description":
            found = _modify_field(tree, node_id, "description", op.get("value", ""))
            if found:
                logger.info("[Step 9] modify_node_description: 节点 %s | 原因: %s", node_id, reason)
            else:
                msg = f"节点 {node_id} 不存在"
                logger.warning("[Step 9] modify_node_description: 未找到节点 %s", node_id)
                skipped.append({**op, "_skip_reason": msg})

        elif op_name == "modify_node_condition":
            found = _modify_field(tree, node_id, "condition", op.get("value", ""))
            if found:
                logger.info("[Step 9] modify_node_condition: 节点 %s | 原因: %s", node_id, reason)
            else:
                msg = f"节点 {node_id} 不存在"
                logger.warning("[Step 9] modify_node_condition: 未找到节点 %s", node_id)
                skipped.append({**op, "_skip_reason": msg})

        else:
            logger.warning("[Step 9] 未知操作: %s", op_name)
            skipped.append({**op, "_skip_reason": f"未知操作类型 {op_name}"})

    return tree, skipped


# ── 内部工具 ──────────────────────────────────────────────────

def tree_to_id_text(node: dict, depth: int = 0) -> str:
    """将大纲树渲染为带 id 的缩进文本，供 LLM 在 patch 时引用节点 id。"""
    if node.get("id") == "__root__":
        return "\n".join(tree_to_id_text(c, 0) for c in node.get("children", []))
    indent = "  " * depth
    desc = f" — {node['description']}" if node.get("description") else ""
    cond = f" ｜条件：{node['condition']}" if node.get("condition") else ""
    line = f"{indent}[id={node['id']} L{node['level']}] {node['name']}{desc}{cond}"
    child_lines = [tree_to_id_text(c, depth + 1) for c in node.get("children", [])]
    return "\n".join([line] + child_lines)


def _keep_only_nodes(tree: dict, node_ids: list[str]) -> None:
    """
    保留 node_ids 中的节点，删除同层中不包含任何 keep 节点的兄弟分支。

    过滤规则：当某节点的直接子节点中存在 keep 节点时，将 children 过滤为
    "自身在 keep_set 中"或"子树中包含 keep 节点"的节点。其他层级不动。
    """
    keep_set = set(node_ids)

    def _contains_keep(node: dict) -> bool:
        if node["id"] in keep_set:
            return True
        return any(_contains_keep(c) for c in node.get("children", []))

    def _prune(node: dict) -> None:
        children = node.get("children", [])
        if any(c["id"] in keep_set for c in children):
            node["children"] = [c for c in children if _contains_keep(c)]
        for child in node.get("children", []):
            _prune(child)

    _prune(tree)



def _build_kb_subtree(node_id: str, nodes_dict: dict, children_map: dict) -> dict | None:
    """从知识图谱递归构建以 node_id 为根的完整子树。"""
    node = nodes_dict.get(node_id)
    if not node:
        return None
    return {
        "id": node_id,
        "name": node["name"],
        "level": node.get("level", 0),
        "description": node.get("description", ""),
        "children": [
            child
            for child_id in children_map.get(node_id, [])
            if (child := _build_kb_subtree(child_id, nodes_dict, children_map)) is not None
        ],
    }


def _collect_ids(node: dict, result: set | None = None) -> set:
    """递归收集树中所有节点 id。"""
    if result is None:
        result = set()
    result.add(node["id"])
    for c in node.get("children", []):
        _collect_ids(c, result)
    return result


def _add_node(tree: dict, parent_id: str, new_node: dict) -> bool:
    """将 new_node 追加到 parent_id 节点的 children 末尾，返回是否找到父节点。"""
    if tree["id"] == parent_id:
        tree.setdefault("children", []).append(new_node)
        return True
    for child in tree.get("children", []):
        if _add_node(child, parent_id, new_node):
            return True
    return False


def _delete_node(tree: dict, node_id: str) -> bool:
    """从树中删除 node_id 对应的节点（含子树），返回是否找到目标节点。"""
    children = tree.get("children", [])
    for i, child in enumerate(children):
        if child["id"] == node_id:
            children.pop(i)
            return True
        if _delete_node(child, node_id):
            return True
    return False


def _modify_field(tree: dict, node_id: str, field: str, value: str) -> bool:
    """修改 node_id 节点的指定字段，返回是否找到目标节点。"""
    if tree["id"] == node_id:
        tree[field] = value
        return True
    for child in tree.get("children", []):
        if _modify_field(child, node_id, field, value):
            return True
    return False
