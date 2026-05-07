"""
patcher.py — Step 8 / 9: 解析用户修改指令并应用到大纲树。

Step 8 parse_patch  : LLM 将自然语言指令翻译为结构化 patch 操作列表
Step 9 apply_patch  : 纯 Python，将 patch 操作列表应用到大纲树（deepcopy，原树不变）

支持的 patch 操作:
  add_node              — 从知识图谱新增节点，挂到指定父节点下
  delete_node           — 删除指定节点及其所有子节点
  modify_node_name      — 修改节点的 name
  modify_node_description — 修改节点的 description（也用于调整阈值等说明）
  keep_only_node        — 保留指定节点，删除同级兄弟节点

Prompt 从 prompts/patch.txt 加载。
"""

import copy
import logging
import os
import sys
from pathlib import Path

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from loader import load_resources

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
PATCH_PROMPT: str = (_PROMPT_DIR / "patch.txt").read_text(encoding="utf-8")


# ── Step 8 ────────────────────────────────────────────────────

async def parse_patch(user_request: str, outline_tree: dict) -> list[dict]:
    """
    调用 LLM，将用户的自然语言修改指令翻译为结构化 patch 操作列表。

    LLM 看到当前大纲（含节点 id）后判断是否需要修改：
    - 无修改意图 → 返回 []
    - 有修改意图 → 返回 [{op, ...}, ...]

    add_node op 在 LLM 输出后自动从 nodes_dict 补全 name/level/description。

    Args:
        user_request : 用户的自然语言问题或修改指令
        outline_tree : 当前大纲树 dict

    Returns:
        patch 操作列表，可能为空列表
    """
    llm = LLMService.from_env()
    tree_text = tree_to_id_text(outline_tree)

    messages = [
        {"role": "system", "content": PATCH_PROMPT},
        {"role": "user", "content": f"## 当前大纲\n{tree_text}\n\n## 修改指令\n{user_request}"},
    ]

    logger.info(
        "[Step 8] Patch Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    answer = await llm.complete(messages)
    logger.info("[Step 8] LLM 完整输出:\n%s", answer)

    raw = LLMService._parse_json(answer)
    ops: list[dict] = raw if isinstance(raw, list) else [raw]

    # 补全 add_node ops：从 nodes_dict 查出 name/level/description
    add_ops = [op for op in ops if op.get("op") == "add_node"]
    if add_ops:
        _, nodes_dict, _ = load_resources()
        for op in add_ops:
            node = nodes_dict.get(op.get("node_id", ""))
            if node:
                op.setdefault("name", node["name"])
                op.setdefault("level", node.get("level", 0))
                op.setdefault("description", node.get("description", ""))

    logger.info("[Step 8] 解析 patch 操作 (%d 条): %s", len(ops), ops)
    return ops


# ── Step 9 ────────────────────────────────────────────────────

def apply_patch(outline_tree: dict, ops: list[dict]) -> dict:
    """
    将 patch 操作列表应用到大纲树，返回修改后的新树（原树不变）。

    keep_only_node 操作汇总后统一处理：收集所有 node_id 组成 keep_set，
    按层级从高到低依次删除各节点的非 keep_set 兄弟节点。

    Args:
        outline_tree : 当前大纲树 dict
        ops          : parse_patch() 返回的操作列表

    Returns:
        deepcopy 后修改过的新树 dict
    """
    tree = copy.deepcopy(outline_tree)

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
            new_node = {
                "id": node_id,
                "name": op.get("name", node_id),
                "level": op.get("level", 0),
                "description": op.get("description", ""),
                "children": [],
            }
            added = _add_node(tree, op.get("parent_id", ""), new_node)
            if added:
                logger.info("[Step 9] add_node: 新增节点 %s → 父节点 %s | 原因: %s", node_id, op.get("parent_id"), reason)
            else:
                logger.warning("[Step 9] add_node: 未找到父节点 %s", op.get("parent_id"))

        elif op_name == "delete_node":
            removed = _delete_node(tree, node_id)
            if removed:
                logger.info("[Step 9] delete_node: 已删除节点 %s | 原因: %s", node_id, reason)
            else:
                logger.warning("[Step 9] delete_node: 未找到节点 %s", node_id)

        elif op_name == "modify_node_name":
            found = _modify_field(tree, node_id, "name", op.get("value", ""))
            if found:
                logger.info("[Step 9] modify_node_name: 节点 %s | 原因: %s", node_id, reason)
            else:
                logger.warning("[Step 9] modify_node_name: 未找到节点 %s", node_id)

        elif op_name == "modify_node_description":
            found = _modify_field(tree, node_id, "description", op.get("value", ""))
            if found:
                logger.info("[Step 9] modify_node_description: 节点 %s | 原因: %s", node_id, reason)
            else:
                logger.warning("[Step 9] modify_node_description: 未找到节点 %s", node_id)

        else:
            logger.warning("[Step 9] 未知操作: %s", op_name)

    return tree


# ── 内部工具 ──────────────────────────────────────────────────

def tree_to_id_text(node: dict, depth: int = 0) -> str:
    """将大纲树渲染为带 id 的缩进文本，供 LLM 在 patch 时引用节点 id。"""
    indent = "  " * depth
    desc = f" — {node['description']}" if node.get("description") else ""
    line = f"{indent}[id={node['id']} L{node['level']}] {node['name']}{desc}"
    child_lines = [tree_to_id_text(c, depth + 1) for c in node.get("children", [])]
    return "\n".join([line] + child_lines)


def _keep_only_nodes(tree: dict, node_ids: list[str]) -> None:
    """
    对每个 node_id，删除其兄弟节点中不在 keep_set 内的节点。
    按层级从高（L2）到低（L5）处理，确保高层删除后低层无需重复处理。
    """
    keep_set = set(node_ids)
    id_to_level: dict[str, int] = {}

    def _collect(node: dict) -> None:
        id_to_level[node["id"]] = node.get("level", 0)
        for c in node.get("children", []):
            _collect(c)

    _collect(tree)

    sorted_ids = sorted(
        (nid for nid in node_ids if nid in id_to_level),
        key=lambda nid: id_to_level[nid],
    )
    for node_id in sorted_ids:
        _prune_siblings(tree, node_id, keep_set)


def _prune_siblings(tree: dict, target_id: str, keep_set: set) -> bool:
    """找到 target_id 的父节点，将父节点的 children 过滤为只保留 keep_set 内的节点。"""
    children = tree.get("children", [])
    for child in children:
        if child["id"] == target_id:
            tree["children"] = [c for c in children if c["id"] in keep_set]
            return True
        if _prune_siblings(child, target_id, keep_set):
            return True
    return False


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
