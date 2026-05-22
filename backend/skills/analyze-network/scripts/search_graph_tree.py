#!/usr/bin/env python3
"""
知识图谱语义检索。

用法:
  python3 search_graph_tree.py "查询词" [--topk 5] [--threshold 0.3]

输出两部分：
  1. === 匹配节点 ===  平铺列表（带路径和得分）
  2. === 相关知识树（★ 为命中节点）=== 完整子树，供选 anchor_id
"""
import sys
import os
import argparse
import asyncio

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", ".."))
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from search_graph_tree import search_graph_tree as _search


def _render_tree(node: dict, depth: int, hit_scores: dict) -> None:
    indent = "  " * depth
    level = node.get("level", 0)
    level_str = "Q" if level == 5 else f"L{level}"
    nid = node.get("id", "")
    desc = f"：{node['description']}" if node.get("description") else ""
    hit = f"  ★{hit_scores[nid]:.4f}" if nid in hit_scores else ""
    print(f"{indent}[{level_str} {nid}] {node['name']}{desc}{hit}")
    for child in node.get("children", []):
        _render_tree(child, depth + 1, hit_scores)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="检索词")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    tree, candidates = await _search(args.query)

    if not candidates:
        print("（无匹配节点）")
        return

    hit_scores = {c["id"]: c["score"] for c in candidates}

    print("=== 匹配节点 ===\n")
    for c in sorted(candidates, key=lambda x: -x["score"])[:args.topk]:
        level_str = "Q" if c["level"] == 5 else f"L{c['level']}"
        print(f"[{level_str} {c['id']}] {c['name']}  ({c['score']:.4f})")
        print(f"  路径: {c['path']}")
        print()

    print("=== 相关知识树（★ 为命中节点）===\n")
    for root in tree:
        _render_tree(root, 0, hit_scores)
    print()


if __name__ == "__main__":
    asyncio.run(main())
