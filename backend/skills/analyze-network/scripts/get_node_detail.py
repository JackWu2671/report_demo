#!/usr/bin/env python3
"""
查询知识库节点的完整信息。

用法:
  python3 get_node_detail.py <node_id> [node_id2 ...]

输出节点的全量字段，包括 summarySuggestion、exec_sql、renderType 等
在大纲 YAML 视图中被省略的字段。可同时查询多个节点。

失败时（节点不存在）输出 {"error": "..."}。
"""
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.environ.get("REPORT_BACKEND_DIR", "") or os.path.join(_SCRIPTS, "..", "..", "..", "..")
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")
_NODE_FILE = (
    os.path.join(_KB_DIR, "node.json")
    if os.path.exists(os.path.join(_KB_DIR, "node.json"))
    else os.path.join(_KB_DIR, "knowledge_nodes.json")
)

_DISPLAY_FIELDS = [
    "id", "name", "level", "description",
    "condition", "condition_queries", "summarySuggestion",
    "renderType", "colX", "colY",
    "apiName", "exec_sql", "extracted_table",
]


def main():
    ids = sys.argv[1:]
    if not ids:
        print(json.dumps({"error": "缺少 node_id 参数"}, ensure_ascii=False))
        sys.exit(1)

    if not os.path.exists(_NODE_FILE):
        print(json.dumps({"error": f"找不到 node.json: {_NODE_FILE}"}, ensure_ascii=False))
        sys.exit(1)

    with open(_NODE_FILE, encoding="utf-8") as f:
        all_nodes = json.load(f)

    index = {n["id"]: n for n in all_nodes if n.get("id")}

    results = []
    for nid in ids:
        node = index.get(nid)
        if not node:
            results.append({"id": nid, "error": "节点不存在"})
            continue
        detail = {k: node[k] for k in _DISPLAY_FIELDS if k in node}
        results.append(detail)

    output = results[0] if len(results) == 1 else results
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
