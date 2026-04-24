"""
patcher.py — Step 8 / 9: 解析用户修改指令并应用到大纲树。

Step 8 parse_patch  : LLM 将自然语言指令翻译为结构化 patch 操作列表
Step 9 apply_patch  : 纯 Python，将 patch 操作列表应用到大纲树（deepcopy，原树不变）

支持的 patch 操作:
  delete    — 删除指定节点及其所有子节点
  set_param — 为节点添加参数标注（如阈值、时间窗口等）

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

logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).parent / "prompts"
PATCH_PROMPT: str = (_PROMPT_DIR / "patch.txt").read_text(encoding="utf-8")


# ── Step 8 ────────────────────────────────────────────────────

async def parse_patch(user_request: str, outline_tree: dict) -> list[dict]:
    """
    调用 LLM，将用户的自然语言修改指令翻译为结构化 patch 操作列表。

    LLM 看到当前大纲（含节点 id）后判断是否需要修改：
    - 无修改意图 → 返回 []
    - 有修改意图 → 返回 [{op, node_id, ...}, ...]

    Args:
        user_request : 用户的自然语言问题或修改指令
        outline_tree : 当前大纲树 dict（来自 subtree.build_subtree 或上一轮 apply_patch）

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

    print("\n[Step 8] LLM Patch 流式输出 ↓", flush=True)
    print("-" * 50, flush=True)

    answer = await llm.stream_and_collect(messages)
    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 8] LLM 完整输出:\n%s", answer)

    raw = LLMService._parse_json(answer)
    ops: list[dict] = raw if isinstance(raw, list) else [raw]
    logger.info("[Step 8] 解析 patch 操作 (%d 条): %s", len(ops), ops)
    return ops


# ── Step 9 ────────────────────────────────────────────────────

def apply_patch(outline_tree: dict, ops: list[dict]) -> dict:
    """
    将 patch 操作列表应用到大纲树，返回修改后的新树（原树不变）。

    Args:
        outline_tree : 当前大纲树 dict
        ops          : parse_patch() 返回的操作列表

    Returns:
        deepcopy 后修改过的新树 dict
    """
    tree = copy.deepcopy(outline_tree)
    for op in ops:
        node_id = op.get("node_id", "")
        reason = op.get("reason", "")
        if op["op"] == "delete":
            removed = _delete_node(tree, node_id)
            if removed:
                logger.info("[Step 9] delete: 已删除节点 %s | 原因: %s", node_id, reason)
            else:
                logger.warning("[Step 9] delete: 未找到节点 %s", node_id)
        elif op["op"] == "set_param":
            found = _set_param_node(tree, node_id, op["key"], op["value"], op.get("unit", ""))
            if found:
                logger.info(
                    "[Step 9] set_param: 节点 %s → %s=%s%s | 原因: %s",
                    node_id, op["key"], op["value"], op.get("unit", ""), reason,
                )
            else:
                logger.warning("[Step 9] set_param: 未找到节点 %s", node_id)
        else:
            logger.warning("[Step 9] 未知操作: %s", op["op"])
    return tree


# ── 内部工具 ──────────────────────────────────────────────────

def tree_to_id_text(node: dict, depth: int = 0) -> str:
    """
    将大纲树渲染为带 id 的缩进文本，供 LLM 在 patch 时引用节点 id。

    格式: [id={id} L{level}] {name}
    """
    indent = "  " * depth
    desc = f" — {node['description']}" if node.get("description") else ""
    line = f"{indent}[id={node['id']} L{node['level']}] {node['name']}{desc}"
    child_lines = [tree_to_id_text(c, depth + 1) for c in node.get("children", [])]
    return "\n".join([line] + child_lines)


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


def _set_param_node(tree: dict, node_id: str, key: str, value, unit: str) -> bool:
    """在 node_id 节点上添加/更新参数标注，返回是否找到目标节点。"""
    if tree["id"] == node_id:
        tree.setdefault("params", {})[key] = {"value": value, "unit": unit}
        return True
    for child in tree.get("children", []):
        if _set_param_node(child, node_id, key, value, unit):
            return True
    return False
