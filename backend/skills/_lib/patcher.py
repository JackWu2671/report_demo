"""
patcher.py — 将结构化操作列表应用到大纲树。

支持的操作:
  delete_node    — 删除指定节点及其所有子节点
  add_node       — 新增节点：node_id 在 KB 中存在时拉取完整子树；否则需传 name（和可选 description）创建自定义节点
  keep_only_node — 保留指定节点，删除同级兄弟节点
  update_node    — 修改节点属性：field=name/description/condition/exec_sql
"""

import copy
import logging
import os
import re
import sys

_LEVEL_RE = re.compile(r'^(?:new_)?L(\d+)_')


def _infer_level_from_id(node_id: str) -> int:
    """从节点 ID 前缀推断 level，无法识别时默认 4（结构节点）。"""
    m = _LEVEL_RE.match(node_id)
    return int(m.group(1)) if m else 4

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_LIB_DIR))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from loader import load_resources

logger = logging.getLogger(__name__)


# ── apply_patch ───────────────────────────────────────────────

async def apply_patch(outline_tree: dict, ops: list[dict]) -> tuple[dict, list[dict]]:
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

    async def _get_kb():
        nonlocal _kb_cache
        if _kb_cache is None:
            _, nd, cm = await load_resources()
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

        # update_node → 路由到对应的专属 op，复用已有处理逻辑
        if op_name == "update_node":
            field = op.get("field", "")
            if not field:
                skipped.append({**op, "_skip_reason": "缺少 field 参数"})
                continue
            _FIELD_TO_OP = {
                "name":        "modify_node_name",
                "description": "modify_node_description",
                "condition":   "modify_node_condition",
                "exec_sql":    "modify_node_exec_sql",
            }
            op = {**op, "op": _FIELD_TO_OP.get(field, "set_node_field")}
            op_name = op["op"]

        if op_name == "add_node":
            subtree = op.get("subtree")
            if not subtree:
                kb = await _get_kb()
                subtree = _build_kb_subtree(node_id, kb["nodes_dict"], kb["children_map"])
            if not subtree:
                # KB 中不存在时，尝试用 name/description 创建自定义结构节点
                custom_name = str(op.get("name", "")).strip()
                if custom_name:
                    subtree = {
                        "id":                node_id,
                        "name":              custom_name,
                        "level":             _infer_level_from_id(node_id),
                        "description":       str(op.get("description", "")),
                        "condition":         "",
                        "condition_queries": [],
                        "children":          [],
                    }
                    logger.info("[Step 9] add_node: 自定义节点 %s（%r）", node_id, custom_name)
                else:
                    msg = f"节点 {node_id} 在知识图谱中不存在，且未提供 name 参数"
                    logger.warning("[Step 9] add_node: %s，跳过", msg)
                    skipped.append({**op, "_skip_reason": msg})
                    continue
            ids_before = _collect_ids(tree)
            parent_id = op.get("parent_id") or ""
            after_id  = op.get("after_id")  or ""
            if not parent_id:
                tree.setdefault("children", []).append(subtree)
                logger.info("[Step 9] add_node: 新增顶层章节 %s（与现有一级章节平行）", node_id)
            else:
                added = _add_node(tree, parent_id, subtree, after_id=after_id)
                if not added:
                    tree.setdefault("children", []).append(subtree)
                    logger.warning("[Step 9] add_node: 未找到父节点 %s，已作为顶层章节新增", parent_id)
                else:
                    pos_info = f"after_id={after_id}" if after_id else "末尾"
                    logger.info("[Step 9] add_node: 新增节点 %s → 父节点 %s [%s] | 原因: %s", node_id, parent_id, pos_info, reason)
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
            new_name = op.get("value", "")
            found = _modify_field(tree, node_id, "name", new_name)
            if found:
                # L5 节点改名后，从 KB 同步所有关联字段（exec_sql / renderType 等）
                if _find_node_level(tree, node_id) == 5:
                    kb = await _get_kb()
                    kb_node = next(
                        (n for n in kb["nodes_dict"].values() if n.get("name") == new_name),
                        None,
                    )
                    if kb_node:
                        _update_node_fields(tree, node_id, {
                            "id":              kb_node.get("id", node_id),
                            "exec_sql":        kb_node.get("exec_sql", ""),
                            "renderType":      kb_node.get("renderType", ""),
                            "colX":            kb_node.get("colX", ""),
                            "colY":            kb_node.get("colY", ""),
                            "apiName":         kb_node.get("apiName", ""),
                            "extracted_table": kb_node.get("extracted_table") or [],
                        })
                        logger.info("[Step 9] modify_node_name: L5 节点 %s → %r，已从 KB 同步字段 (new_id=%s)",
                                    node_id, new_name, kb_node.get("id"))
                    else:
                        logger.warning("[Step 9] modify_node_name: 新名称 %r 在 KB 中不存在，exec_sql 等字段未同步",
                                       new_name)
                logger.info("[Step 9] modify_node_name: 节点 %s → %r | 原因: %s", node_id, new_name, reason)
            else:
                msg = f"节点 {node_id} 不存在"
                logger.warning("[Step 9] modify_node_name: 未找到节点 %s", node_id)
                skipped.append({**op, "_skip_reason": msg})

        elif op_name == "modify_node_description":
            target_level = _find_node_level(tree, node_id)
            if target_level == 5:
                msg = f"L5 query 节点 {node_id} 的 description 禁止修改"
                logger.warning("[Step 9] modify_node_description: %s", msg)
                skipped.append({**op, "_skip_reason": msg})
            else:
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

        elif op_name == "modify_node_exec_sql":
            target_level = _find_node_level(tree, node_id)
            if target_level != 5:
                msg = f"节点 {node_id} 不是 L5 query 节点，exec_sql 只能在 L5 节点上修改"
                logger.warning("[Step 9] modify_node_exec_sql: %s", msg)
                skipped.append({**op, "_skip_reason": msg})
            else:
                found = _modify_field(tree, node_id, "exec_sql", op.get("value", ""))
                if found:
                    logger.info("[Step 9] modify_node_exec_sql: 节点 %s | 原因: %s", node_id, reason)
                else:
                    msg = f"节点 {node_id} 不存在"
                    logger.warning("[Step 9] modify_node_exec_sql: 未找到节点 %s", node_id)
                    skipped.append({**op, "_skip_reason": msg})

        elif op_name == "set_node_field":
            field = op.get("field", "")
            val = op.get("value")
            if not field:
                skipped.append({**op, "_skip_reason": "缺少 field 参数"})
            else:
                found = _modify_field(tree, node_id, field, val)
                if found:
                    logger.info("[Step 9] set_node_field: 节点 %s.%s | 原因: %s", node_id, field, reason)
                else:
                    msg = f"节点 {node_id} 不存在"
                    skipped.append({**op, "_skip_reason": msg})

        else:
            logger.warning("[Step 9] 未知操作: %s", op_name)
            skipped.append({**op, "_skip_reason": f"未知操作类型 {op_name}"})

    return tree, skipped


# ── 内部工具 ──────────────────────────────────────────────────

def tree_to_id_text(node: dict, depth: int = 0) -> str:
    """将大纲树渲染为带 id 的缩进文本，供 LLM 在 patch 时引用节点 id。"""
    if node.get("id") == "__root__":
        header = "[id=__root__ L0] （虚拟根节点，add_node 新增顶层章节时 parent_id 填 __root__）"
        children_text = "\n".join(tree_to_id_text(c, 1) for c in node.get("children", []))
        return f"{header}\n{children_text}" if children_text else header
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
    entry = {
        "id":                node_id,
        "name":              node["name"],
        "level":             node.get("level", 0),
        "description":       node.get("description", ""),
        "condition":         node.get("condition", ""),
        "condition_queries": node.get("condition_queries") or [],
        "summarySuggestion": node.get("summarySuggestion", ""),
        "children": [
            child
            for child_id in children_map.get(node_id, [])
            if (child := _build_kb_subtree(child_id, nodes_dict, children_map)) is not None
        ],
    }
    if node.get("level") == 5:
        entry["renderType"]       = node.get("renderType", "")
        entry["colX"]             = node.get("colX", "")
        entry["colY"]             = node.get("colY", "")
        entry["apiName"]          = node.get("apiName", "")
        entry["exec_sql"]         = node.get("exec_sql", "")
        entry["extracted_table"]  = node.get("extracted_table") or []
    return entry


def _collect_ids(node: dict, result: set | None = None) -> set:
    """递归收集树中所有节点 id。"""
    if result is None:
        result = set()
    result.add(node["id"])
    for c in node.get("children", []):
        _collect_ids(c, result)
    return result


def _add_node(tree: dict, parent_id: str, new_node: dict, after_id: str = "") -> bool:
    """
    将 new_node 插入到 parent_id 节点的 children 中。
    after_id 非空时，插入到该兄弟节点之后；否则追加到末尾。
    返回是否找到父节点。
    """
    if tree["id"] == parent_id:
        children = tree.setdefault("children", [])
        if after_id:
            idx = next((i for i, c in enumerate(children) if c["id"] == after_id), None)
            if idx is not None:
                children.insert(idx + 1, new_node)
                return True
        children.append(new_node)
        return True
    for child in tree.get("children", []):
        if _add_node(child, parent_id, new_node, after_id=after_id):
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


def _update_node_fields(tree: dict, node_id: str, updates: dict) -> bool:
    """找到 node_id 节点并批量更新多个字段，返回是否找到目标节点。"""
    if tree["id"] == node_id:
        tree.update(updates)
        return True
    for child in tree.get("children", []):
        if _update_node_fields(child, node_id, updates):
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


def _find_node_level(tree: dict, node_id: str) -> int | None:
    """返回 node_id 节点的 level，未找到返回 None。"""
    if tree["id"] == node_id:
        return tree.get("level")
    for child in tree.get("children", []):
        result = _find_node_level(child, node_id)
        if result is not None:
            return result
    return None
