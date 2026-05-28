"""
report_executor.py — 遍历 outline_tree，执行 SQL，通过结构化事件推送结果。

前端用 buildSkeleton() 生成包含占位符的骨架作为报告初始状态，
本模块只负责执行查询并推送每条指标的数据，不再推送标题文本。

事件格式：
  {"type": "report_metric",  "name": str, "chunk": str}         — 单条指标数据
  {"type": "report_summary", "node_id": str, "chunk": str}      — 节点总结（LLM 生成）

并行策略：
  condition 检查仍串行（共享单一 client）；
  metric 查询用线程池并行，每个 worker 独立创建 DeApiClient；
  所有 metric 完成后，若节点有 summarySuggestion 则调 LLM 生成总结。
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

from services.de_sql_execution_client import DeApiClient
from services.sql_executor import SqlExecutor

MAX_PARALLEL = 5  # 同时执行的 metric 查询数

logger = logging.getLogger(__name__)


def run_report(
    outline_tree: Dict,
    on_event: Callable[[dict], None],
    cached_names: set = None,
) -> None:
    cached_names = cached_names or set()
    executor = SqlExecutor()
    with DeApiClient() as client:
        _walk(outline_tree.get("children", []), client, executor, on_event, cached_names)


def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
) -> None:
    standalone_l5 = []
    for node in nodes:
        level = node.get("level", 0)
        if level == 4:
            _process_l4(node, client, executor, on_event, cached_names)
        elif 1 <= level <= 3:
            _walk(node.get("children", []), client, executor, on_event, cached_names)
        elif level == 5:
            standalone_l5.append(node)

    # 没有 L4 父节点的孤立 L5 节点，直接并行执行（无 condition 检查）
    if standalone_l5:
        uncached = [n for n in standalone_l5 if n.get("name") not in cached_names]
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(_run_metric, l5, executor, on_event): l5 for l5 in uncached}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    l5 = futures[future]
                    logger.error("[report] 查询异常 %r: %s", l5.get("name", ""), e)
                    on_event({"type": "report_metric", "name": l5.get("name", ""), "chunk": "_（查询异常）_\n\n"})


def _process_l4(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
) -> None:
    name              = node.get("name", "")
    condition         = node.get("condition", "")
    condition_queries = node.get("condition_queries", [])
    l5_nodes          = [c for c in node.get("children", []) if c.get("level") == 5]
    query_nodes       = [n for n in l5_nodes if n.get("name") not in condition_queries]

    # 过滤掉前端已缓存的指标，无需重新执行
    uncached = [n for n in query_nodes if n.get("name") not in cached_names]
    if not uncached:
        logger.info("[report] L4 %r 所有指标均已缓存，跳过", name)
        return

    # ── condition 检查 ───────────────────────────────────────────
    if condition:
        if not executor.eval_condition(condition, client):
            logger.info("[report] 跳过 L4 %r（condition 不满足）", name)
            for l5 in uncached:
                on_event({"type": "report_metric", "name": l5.get("name", ""), "chunk": "_（条件不满足，已跳过）_\n\n"})
            return

    # ── 并行执行查询，即时推送，同时收集 rows ────────────────────
    collected: Dict[str, List] = {}  # metric_name -> rows
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {
            pool.submit(_run_metric, l5, executor, on_event): l5
            for l5 in uncached
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.get("rows"):
                    collected[result["name"]] = result["rows"]
            except Exception as e:
                l5 = futures[future]
                logger.error("[report] 查询异常 %r: %s", l5.get("name", ""), e)
                on_event({"type": "report_metric", "name": l5.get("name", ""), "chunk": "_（查询异常）_\n\n"})

    # ── 所有指标完成后，有 summarySuggestion 则生成总结 ──────────
    if node.get("summarySuggestion") and collected:
        _generate_summary(node, collected, on_event)


_CHART_TYPES = {"BAR", "LINE", "PIE"}


def _run_metric(
    l5: Dict,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
) -> Optional[Dict]:
    """
    在独立线程中执行单条指标查询并推送结果。每次创建自己的 DeApiClient。
    返回 {"name": ..., "rows": [...]} 供调用方收集用于总结，失败返回 None。
    """
    metric_name = l5.get("name", "")
    logger.info("[report] 查询: %r", metric_name)

    with DeApiClient() as client:
        result = executor.execute_metric(metric_name, client)

    if not result or not result.get("rows"):
        on_event({"type": "report_metric", "name": metric_name, "chunk": "_（暂无数据）_\n\n"})
        return None

    rows = result["rows"]
    dict_rows = [r for r in rows if isinstance(r, dict)]
    render_type = (result.get("render_type") or "").upper()

    if render_type in _CHART_TYPES and dict_rows:
        on_event({
            "type":        "report_metric",
            "name":        metric_name,
            "chunk":       SqlExecutor.rows_to_markdown(dict_rows) + "\n\n",
            "render_type": render_type,
            "col_x":       result.get("col_x") or "",
            "col_y":       result.get("col_y") or "",
            "rows":        dict_rows,
        })
    elif len(dict_rows) == 1 and len(dict_rows[0]) == 1:
        val = next(iter(dict_rows[0].values()))
        on_event({"type": "report_metric", "name": metric_name, "chunk": f"{val}\n\n"})
    else:
        on_event({"type": "report_metric", "name": metric_name, "chunk": SqlExecutor.rows_to_markdown(rows) + "\n\n"})

    return {"name": metric_name, "rows": dict_rows or rows}


def _generate_summary(
    node: Dict,
    collected: Dict[str, List],
    on_event: Callable[[dict], None],
) -> None:
    """所有指标查询完成后，调 LLM 生成节点总结并推送 report_summary 事件。"""
    from services.llm_service import LLMService

    node_id   = node.get("id", "")
    node_name = node.get("name", "")

    # ── 拼接详细信息 ─────────────────────────────────────────────
    lines = [f"章节名称：{node_name}"]
    if node.get("description"):
        lines.append(f"章节说明：{node['description']}")
    lines.append("\n指标查询结果：")
    for child in node.get("children", []):
        if child.get("level") != 5:
            continue
        child_name = child.get("name", "")
        lines.append(f"\n■ {child_name}")
        rows = collected.get(child_name)
        if rows:
            lines.append(SqlExecutor.rows_to_markdown(rows))
        else:
            lines.append("（暂无数据）")

    detail = "\n".join(lines)
    summary_suggestion = node["summarySuggestion"]

    prompt = (
        f"【详细信息】\n{detail}\n\n"
        f"【总结建议规则】\n{summary_suggestion}\n\n"
        "请严格按照总结建议规则的格式，用上方真实数据中的具体数字替换其中的 XX，"
        "直接输出总结内容，不要解释。"
    )

    logger.info("[report] 生成总结: %r", node_name)
    try:
        llm     = LLMService.from_env()
        summary = asyncio.run(llm.complete([{"role": "user", "content": prompt}]))
        on_event({"type": "report_summary", "node_id": node_id, "chunk": summary + "\n\n"})
        logger.info("[report] 总结完成: %r", node_name)
    except Exception as e:
        logger.error("[report] 总结生成失败 %r: %s", node_name, e)
        on_event({"type": "report_summary", "node_id": node_id, "chunk": "_（总结生成失败）_\n\n"})
