"""
kb_updater.py — Step 4: 从带 id 标注的大纲中解析 [new] 节点。
"""

import logging
import re

logger = logging.getLogger(__name__)


def parse_new_nodes(outline_md: str) -> list[dict]:
    """
    解析带 id 标注的 Markdown 大纲，提取所有 [new] 节点及其层级/父节点信息。

    Args:
        outline_md: generate_outline() 输出的带 id 标注 Markdown

    Returns:
        [{name, description, level, parent_id, parent_name, order}, ...]
    """
    level_stack: dict[int, dict] = {}
    order_counter: dict[str, int] = {}
    new_nodes: list[dict] = []

    for line in outline_md.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^(#+)\s+\[([^\]]+)\]\s+(.+)$", line)
        if not m:
            continue

        hashes = m.group(1)
        node_ref = m.group(2).strip()
        raw_text = m.group(3).strip()
        depth = len(hashes)

        if node_ref.lower() == "new" and ": " in raw_text:
            name, inline_desc = raw_text.split(": ", 1)
            name, inline_desc = name.strip(), inline_desc.strip()
        else:
            name, inline_desc = raw_text, ""

        for d in [d for d in level_stack if d >= depth]:
            del level_stack[d]
        level_stack[depth] = {"ref": node_ref, "name": name}

        parent = level_stack.get(depth - 1)
        parent_ref = parent["ref"] if parent else None
        parent_key = parent_ref or "root"
        order_counter[parent_key] = order_counter.get(parent_key, 0) + 1

        if node_ref.lower() == "new":
            new_nodes.append({
                "name": name,
                "description": inline_desc,
                "level": depth,
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
