#!/usr/bin/env python3
"""
查询已生成报告中某节点的数据内容（每指标前 10 行）。

用法:
  python3 get_report_data.py <node_id>

输出该节点子树所有 L5 指标的查询结果（每指标最多 10 行）以及各结构节点的总结文本。
报告尚未生成时输出提示并退出。
"""
import json
import os
import sys
from pathlib import Path

SESSION_ID  = os.environ.get("REPORT_SESSION_ID", "")
SESSION_DIR = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))

_MAX_ROWS = 10


def _read_session() -> dict:
    p = SESSION_DIR / f"{SESSION_ID}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _find_node(tree: dict, node_id: str) -> dict | None:
    if tree.get("id") == node_id:
        return tree
    for child in tree.get("children", []):
        found = _find_node(child, node_id)
        if found:
            return found
    return None


def _rows_to_md(rows: list) -> list[str]:
    if not rows:
        return ["  （暂无数据）"]
    if not isinstance(rows[0], dict):
        lines = [f"  {r}" for r in rows]
        if len(rows) == _MAX_ROWS:
            lines.append(f"  （仅显示前 {_MAX_ROWS} 行）")
        return lines
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    if len(rows) == _MAX_ROWS:
        lines.append(f"（仅显示前 {_MAX_ROWS} 行）")
    return lines


def _render(node: dict, report_data: dict, report_descriptions: dict, report_summaries: dict, depth: int = 0) -> list[str]:
    lines = []
    level  = node.get("level", 0)
    name   = node.get("name", "")
    nid    = node.get("id", "")
    indent = "  " * max(depth - 1, 0)

    if level == 5:
        stored = report_data.get(name)
        lines.append(f"{indent}■ {name} ({nid})")
        lines.extend(_rows_to_md(stored) if stored is not None else [f"{indent}  （尚未生成或无数据）"])
    else:
        if depth > 0:
            heading = "#" * min(depth, 4)
            lines.append(f"\n{indent}{heading} {name} ({nid})")
        description = report_descriptions.get(nid)
        if description:
            lines.append(f"{indent}**节描述**：{description.strip()}")
        for child in node.get("children", []):
            lines.extend(_render(child, report_data, report_descriptions, report_summaries, depth + 1))
        summary = report_summaries.get(nid)
        if summary:
            lines.append(f"\n{indent}**节总结**")
            lines.append(f"{indent}{summary.strip()}")

    return lines


def main():
    node_id = sys.argv[1] if len(sys.argv) > 1 else ""
    if not node_id:
        print("用法: python3 get_report_data.py <node_id>", file=sys.stderr)
        sys.exit(1)

    session             = _read_session()
    outline_tree        = session.get("outline_tree", {})
    report_data         = session.get("report_data", {})
    report_descriptions = session.get("report_descriptions", {})
    report_summaries    = session.get("report_summaries", {})

    if not report_data and not report_descriptions and not report_summaries:
        print("（报告尚未生成，暂无数据）")
        sys.exit(0)

    target = _find_node(outline_tree, node_id)
    if not target:
        print(f"（节点 {node_id} 不在当前大纲中）")
        sys.exit(1)

    node_name = target.get("name", node_id)
    lines = [f"=== {node_name} ({node_id}) ==="]
    top_description = report_descriptions.get(node_id)
    if top_description:
        lines.append(f"**节描述**：{top_description.strip()}")
    for child in target.get("children", []):
        lines.extend(_render(child, report_data, report_descriptions, report_summaries, depth=1))

    top_summary = report_summaries.get(node_id)
    if top_summary:
        lines.append("\n**节总结**")
        lines.append(top_summary.strip())

    print("\n".join(lines))


if __name__ == "__main__":
    main()
