"""
report_executor.py — 遍历 outline_tree，执行 SQL，通过结构化事件推送结果。

前端用 buildSkeleton() 生成包含占位符的骨架作为报告初始状态，
本模块只负责执行查询并推送每条指标的数据，不再推送标题文本。

事件格式：
  {"type": "report_metric",  "name": str, "chunk": str}   — 单条指标数据
  {"type": "report_skip_l4", "name": str}                  — L4 节条件不满足，跳过
"""

import logging
from typing import Callable, Dict, List

from services.de_sql_execution_client import DeApiClient
from services.sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


def run_report(
    outline_tree: Dict,
    on_event: Callable[[dict], None],
) -> None:
    executor = SqlExecutor()
    with DeApiClient() as client:
        _walk(outline_tree.get("children", []), client, executor, on_event)


def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
) -> None:
    for node in nodes:
        level = node.get("level", 0)
        if level == 4:
            _process_l4(node, client, executor, on_event)
        elif 1 <= level <= 3:
            _walk(node.get("children", []), client, executor, on_event)


def _process_l4(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
) -> None:
    name              = node.get("name", "")
    condition         = node.get("condition", "")
    condition_queries = node.get("condition_queries", [])
    l5_nodes          = [c for c in node.get("children", []) if c.get("level") == 5]
    query_nodes       = [n for n in l5_nodes if n.get("name") not in condition_queries]

    # ── condition 检查 ───────────────────────────────────────────
    if condition:
        if not executor.eval_condition(condition, client):
            logger.info("[report] 跳过 L4 %r（condition 不满足）", name)
            for l5 in query_nodes:
                on_event({"type": "report_metric", "name": l5.get("name", ""), "chunk": "_（条件不满足，已跳过）_\n\n"})
            return

    # ── 逐条执行查询，即时推送 ────────────────────────────────────
    for l5 in query_nodes:
        metric_name = l5.get("name", "")
        logger.info("[report] 查询: %r", metric_name)
        result = executor.execute_metric(metric_name, client)

        if not result or not result.get("rows"):
            chunk = "_（暂无数据）_\n\n"
        else:
            rows = result["rows"]
            if len(rows) == 1 and len(rows[0]) == 1:
                val = next(iter(rows[0].values()))
                chunk = f"{val}\n\n"
            else:
                chunk = SqlExecutor.rows_to_markdown(rows) + "\n\n"

        on_event({"type": "report_metric", "name": metric_name, "chunk": chunk})
