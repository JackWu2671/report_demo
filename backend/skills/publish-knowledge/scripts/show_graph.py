#!/usr/bin/env python3
"""
输出当前知识图谱的完整结构。

用法:
  python3 show_graph.py

以 md_with_ids 格式输出完整节点树，供 LLM 与模板大纲对比分析。
"""
import sys
import os
import json

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPTS)))
sys.path.insert(0, _BACKEND_DIR)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from subtree import build_subtree
from outline_utils import to_markdown_with_ids

_NODE_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "node.json")
_RELATION_PATH = os.path.join(_BACKEND_DIR, "expert_knowledge", "relation.json")


def main():
    with open(_NODE_PATH, encoding="utf-8") as f:
        nodes = json.load(f)
    with open(_RELATION_PATH, encoding="utf-8") as f:
        relations = json.load(f)

    nodes_dict = {n["id"]: n for n in nodes}
    children_map: dict[str, list[str]] = {}
    for rel in sorted(relations, key=lambda r: r.get("order", 0)):
        children_map.setdefault(rel["parent"], []).append(rel["child"])

    # 找到所有根节点（L1，即没有被任何节点作为 child 的节点）
    all_children = {c for children in children_map.values() for c in children}
    roots = [n for n in nodes if n["id"] not in all_children]

    lines = []
    def _render(node_id: str, depth: int) -> None:
        node = nodes_dict.get(node_id)
        if not node:
            return
        indent = "  " * depth
        level = node.get("level", 0)
        level_str = "Q" if level == 5 else f"L{level}"
        desc = f"：{node['description']}" if node.get("description") else ""
        lines.append(f"{indent}[{level_str} {node_id}] {node['name']}{desc}")
        for child_id in children_map.get(node_id, []):
            _render(child_id, depth + 1)

    for root in roots:
        _render(root["id"], 0)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
