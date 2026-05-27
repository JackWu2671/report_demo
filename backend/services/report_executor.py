"""
report_executor.py — 遍历 outline_tree，执行 SQL，生成 Markdown 报告

report_executor 是同步函数，通过 on_chunk(text) 回调逐段推送结果。
调用方（api_server）在线程池中运行它，用 asyncio.Queue 桥接到 SSE 流。

流程（每个 L4 节点）:
  1. 有 condition → eval_condition → 不满足则跳过
  2. 执行 L5 子节点的 SQL（跳过纯条件指标）
  3. 拼装 Markdown（标题 + description + 数据表 + summarySuggestion）
  4. on_chunk(text)
"""

import logging
from typing import Callable, Dict, List

from services.de_sql_execution_client import DeApiClient
from services.sql_executor import SqlExecutor

logger = logging.getLogger(__name__)


def run_report(
    outline_tree: Dict,
    on_chunk: Callable[[str], None],
) -> None:
    """
    同步执行报告生成，每完成一个 L4 节调用一次 on_chunk。

    Args:
        outline_tree : 完整大纲 JSON（__root__ 节点）
        on_chunk     : 每段 Markdown 文本的回调
    """
    executor = SqlExecutor()
    with DeApiClient() as client:
        _walk(outline_tree.get("children", []), client, executor, on_chunk)


# ── 遍历 ──────────────────────────────────────────────────────────────────

def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_chunk: Callable[[str], None],
) -> None:
    for node in nodes:
        level = node.get("level", 0)
        if level == 4:
            _process_l4(node, client, executor, on_chunk)
        elif 1 <= level <= 3:
            heading = "#" * level + " " + node.get("name", "")
            on_chunk(heading + "\n\n")
            _walk(node.get("children", []), client, executor, on_chunk)


def _process_l4(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_chunk: Callable[[str], None],
) -> None:
    name              = node.get("name", "")
    description       = node.get("description", "")
    condition         = node.get("condition", "")
    condition_queries = node.get("condition_queries", [])
    summary_hint      = node.get("summarySuggestion", "")
    l5_nodes          = [c for c in node.get("children", []) if c.get("level") == 5]

    # ── 1. condition 检查 ────────────────────────────────────────
    if condition:
        if not executor.eval_condition(condition, client):
            logger.info("[report] 跳过 %r（condition 不满足）", name)
            return

    # ── 2. 立即推送标题和描述 ────────────────────────────────────
    on_chunk(f"#### {name}\n\n")
    if description:
        on_chunk(f"{description}\n\n")

    # ── 3. 逐条执行查询并即时推送 ────────────────────────────────
    query_nodes = [n for n in l5_nodes if n.get("name") not in condition_queries]
    for l5 in query_nodes:
        metric_name = l5.get("name", "")
        logger.info("[report] 查询: %r", metric_name)
        result = executor.execute_metric(metric_name, client)

        on_chunk(f"**{metric_name}**\n\n")
        if not result or not result.get("rows"):
            on_chunk("_（暂无数据）_\n\n")
            continue

        rows = result["rows"]
        if len(rows) == 1 and len(rows[0]) == 1:
            val = next(iter(rows[0].values()))
            on_chunk(f"{val}\n\n")
        else:
            on_chunk(SqlExecutor.rows_to_markdown(rows) + "\n\n")

    # ── 4. 分析建议 ──────────────────────────────────────────────
    if summary_hint and summary_hint.strip().upper() != "NA":
        on_chunk(f"> **分析建议：** {summary_hint}\n\n")

    on_chunk("---\n\n")
