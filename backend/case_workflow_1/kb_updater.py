"""
kb_updater.py — Step 4: 从带 id 标注的大纲中解析 [new] 节点。

解析格式: [Lx id] name  或  [Lx new] name：description
"""

import logging
import re

logger = logging.getLogger(__name__)

# Matches: optional indent, [Lx node_ref] name（：description optional）
_NODE_RE = re.compile(r"^\s*\[L(\d+)\s+([^\]]+)\]\s+(.+)$")


def parse_new_nodes(outline_md: str) -> list[dict]:
    """
    解析带 [Lx id] 标注的大纲，提取所有 [new] 节点及其层级/父节点信息。

    Args:
        outline_md: generate_outline() 输出的带 [Lx id] 标注文本

    Returns:
        [{name, description, level, parent_id, parent_name, order}, ...]
    """
    level_stack: dict[int, dict] = {}
    order_counter: dict[str, int] = {}
    new_nodes: list[dict] = []

    for line in outline_md.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue

        level = int(m.group(1))
        node_ref = m.group(2).strip()
        raw_text = m.group(3).strip()

        if node_ref.lower() == "new" and "：" in raw_text:
            name, inline_desc = raw_text.split("：", 1)
            name, inline_desc = name.strip(), inline_desc.strip()
        else:
            name, inline_desc = raw_text, ""

        # Prune stack at same or deeper level
        for d in [d for d in level_stack if d >= level]:
            del level_stack[d]
        level_stack[level] = {"ref": node_ref, "name": name}

        parent = level_stack.get(level - 1)
        parent_ref = parent["ref"] if parent else None
        parent_key = parent_ref or "root"
        order_counter[parent_key] = order_counter.get(parent_key, 0) + 1

        if node_ref.lower() == "new":
            new_nodes.append({
                "name": name,
                "description": inline_desc,
                "level": level,
                "parent_id": parent_ref if (parent_ref and parent_ref.lower() != "new") else None,
                "parent_name": parent["name"] if parent else None,
                "order": order_counter[parent_key],
            })

    logger.info(
        "[Step 4] 解析到 %d 个 [new] 节点: %s",
        len(new_nodes),
        [f"L{n['level']} {n['name']}" for n in new_nodes],
    )
    return new_nodes
