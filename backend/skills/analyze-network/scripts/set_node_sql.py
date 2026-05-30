#!/usr/bin/env python3
"""
set_node_sql.py — 直接修改 L5 节点的 exec_sql，SQL 通过 stdin 或文件传入。

为什么单独做这个脚本：
  SQL 经常包含反引号 `…`、单/双引号、换行。若把 SQL 拼进命令行参数，
  后端 bash 会对双引号里的反引号做命令替换（``…`` → 执行命令），
  导致 SQL 被破坏、修改静默失败（表现为脚本 (no output)、改了却查不到）。
  让 SQL 走 stdin（配引号 heredoc）或独立文件，可原样传入任意 SQL，
  完全绕开 shell 的引号/反引号解析。

用法一（推荐，Linux/Mac 后端）—— 引号 heredoc，SQL 原样传入：
  python3 set_node_sql.py L5_068 <<'SQL'
  SELECT COUNT(DISTINCT CONCAT(neIPAddress,'-',neType)) AS `OLT总数`
  FROM ads_aggr_unb_eval_an_all_netelement_info
  SQL

用法二（Windows 后端或需要程序化写入）—— 先把 SQL 写进文件再传路径：
  python3 set_node_sql.py L5_068 /path/to/sql.txt

成功时输出修改后的 YAML 大纲；跳过的操作以 # SKIPPED: 开头。
失败时输出 JSON 错误并以非 0 退出。
"""
import sys
import os
import asyncio
import json

sys.path.insert(0, os.environ.get("REPORT_BACKEND_DIR", ""))
_SCRIPTS = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_SCRIPTS, "..", "..", "_lib"))

from session import get_outline_tree, set_outline
from modify_outline import modify_outline


def _err(message: str) -> None:
    print(json.dumps({"status": "error", "message": message}, ensure_ascii=False), file=sys.stderr)
    sys.exit(1)


async def main():
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        _err("用法: set_node_sql.py <node_id>  （SQL 经 stdin 传入，或附加文件路径作为第 2 个参数）")

    node_id = sys.argv[1].strip()

    # SQL 来源：第 2 个参数是已存在文件 → 读文件；否则读 stdin
    if len(sys.argv) >= 3 and os.path.isfile(sys.argv[2]):
        with open(sys.argv[2], encoding="utf-8") as f:
            sql = f.read().strip()
    else:
        sql = sys.stdin.read().strip()

    if not sql:
        _err("未读到 SQL：请用引号 heredoc 经 stdin 传入，或提供 SQL 文件路径")

    outline_tree = get_outline_tree()
    if not outline_tree:
        _err("当前没有大纲，请先调用 build_outline 或 load_template")

    ops = [{"op": "modify_node_exec_sql", "node_id": node_id, "value": sql}]
    result = await modify_outline(ops, outline_tree)

    if result["status"] != "success":
        _err(result["message"])

    set_outline(result["outline_tree"], result["outline_yaml"], result["markdown"])
    print(result["outline_yaml"])
    for s in result.get("skipped", []):
        reason = s.get("_skip_reason", "未知原因") if isinstance(s, dict) else str(s)
        op = s.get("op", "?") if isinstance(s, dict) else "?"
        nid = s.get("node_id", "") if isinstance(s, dict) else ""
        print(f"# SKIPPED: {op} node_id={nid} → {reason}")


if __name__ == "__main__":
    asyncio.run(main())
