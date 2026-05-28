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
  某节点的所有后代 metric 都完成后，若该节点有 summarySuggestion 则调 LLM 生成总结。

总结范围：
  任意层级（L1~L5）节点均支持。_collect_node_data() 递归收集该节点
  子树下所有 L5 指标的查询结果，作为总结的数据输入。
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
    # collected 贯穿全局，所有 metric 的 rows 都写入这里
    collected: Dict[str, List] = {}
    with DeApiClient() as client:
        _walk(outline_tree.get("children", []), client, executor, on_event, cached_names, collected)


def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
    collected: Dict[str, List],
) -> None:
    standalone_l5 = []
    for node in nodes:
        level = node.get("level", 0)
        if level == 4:
            _process_l4(node, client, executor, on_event, cached_names, collected)
        elif 1 <= level <= 3:
            # 先递归处理所有后代
            _walk(node.get("children", []), client, executor, on_event, cached_names, collected)
            # 后代全部完成后，若本节点有 summarySuggestion 则生成总结
            if node.get("summarySuggestion"):
                _generate_summary(node, _collect_node_data(node, collected), on_event)
        elif level == 5:
            standalone_l5.append(node)

    # 没有 L4 父节点的孤立 L5 节点，直接并行执行（无 condition 检查）
    if standalone_l5:
        uncached = [n for n in standalone_l5 if n.get("name") not in cached_names]
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
            futures = {pool.submit(_run_metric, l5, executor, on_event): l5 for l5 in uncached}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result and result.get("rows"):
                        collected[result["name"]] = result["rows"]
                except Exception as e:
                    l5 = futures[future]
                    logger.error("[report] 查询异常 %r: %s", l5.get("name", ""), e)
                    on_event({"type": "report_metric", "name": l5.get("name", ""), "chunk": "_（查询异常）_\n\n"})

        # 孤立 L5 节点自身的总结
        for l5 in standalone_l5:
            if l5.get("summarySuggestion") and l5.get("name") in collected:
                _generate_summary(l5, {l5["name"]: collected[l5["name"]]}, on_event)


def _process_l4(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
    collected: Dict[str, List],
) -> None:
    name              = node.get("name", "")
    condition         = node.get("condition", "")
    condition_queries = node.get("condition_queries", [])
    l5_nodes          = [c for c in node.get("children", []) if c.get("level") == 5]
    query_nodes       = [n for n in l5_nodes if n.get("name") not in condition_queries]

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

    # ── 并行执行查询，即时推送，写入全局 collected ───────────────
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

    # ── L4 自身总结：用子树数据 ──────────────────────────────────
    if node.get("summarySuggestion"):
        _generate_summary(node, _collect_node_data(node, collected), on_event)


def _collect_node_data(node: Dict, collected: Dict[str, List]) -> Dict[str, List]:
    """
    递归收集节点子树下所有 L5 指标的查询结果。
    适用于任意层级（L1~L4）节点的总结数据准备。
    """
    data = {}
    for child in node.get("children", []):
        if child.get("level") == 5:
            name = child.get("name", "")
            if name in collected:
                data[name] = collected[name]
        else:
            data.update(_collect_node_data(child, collected))
    return data


_CHART_TYPES = {"BAR", "LINE", "PIE"}


def _run_metric(
    l5: Dict,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
) -> Optional[Dict]:
    """
    在独立线程中执行单条指标查询并推送结果。每次创建自己的 DeApiClient。
    返回 {"name": ..., "rows": [...]} 供 collected 收集，失败返回 None。
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
    elif render_type == "TABLE" and dict_rows:
        on_event({
            "type":        "report_metric",
            "name":        metric_name,
            "render_type": "TABLE",
            "rows":        dict_rows,
        })
    else:
        on_event({"type": "report_metric", "name": metric_name, "chunk": SqlExecutor.rows_to_markdown(dict_rows or rows) + "\n\n"})

    return {"name": metric_name, "rows": dict_rows or rows}


def _generate_summary(
    node: Dict,
    node_data: Dict[str, List],
    on_event: Callable[[dict], None],
) -> None:
    """
    调 LLM 生成节点总结并推送 report_summary 事件。
    node_data 是 _collect_node_data() 返回的该节点子树所有 L5 查询结果。
    """
    from services.llm_service import LLMService

    node_id   = node.get("id", "")
    node_name = node.get("name", "")

    # ── 拼接详细信息（按大纲 JSON 中的子节点顺序展示）────────────
    def _render_node(n: Dict, depth: int = 0) -> List[str]:
        lines = []
        indent = "  " * depth
        level  = n.get("level", 0)
        name   = n.get("name", "")
        if level == 5:
            lines.append(f"{indent}■ {name}")
            rows = node_data.get(name)
            if rows:
                for row_line in SqlExecutor.rows_to_markdown(rows).splitlines():
                    lines.append(f"{indent}{row_line}")
            else:
                lines.append(f"{indent}  （暂无数据）")
        else:
            if depth > 0:  # 根节点自身已在外层写了章节名/说明
                lines.append(f"\n{indent}【{name}】")
                if n.get("description"):
                    lines.append(f"{indent}{n['description']}")
            for child in n.get("children", []):
                lines.extend(_render_node(child, depth + 1))
        return lines

    detail_lines = [f"章节名称：{node_name}"]
    if node.get("description"):
        detail_lines.append(f"章节说明：{node['description']}")
    detail_lines.append("\n指标查询结果：")
    for child in node.get("children", []):
        detail_lines.extend(_render_node(child, depth=0))

    detail            = "\n".join(detail_lines)
    summary_suggestion = node["summarySuggestion"]

    prompt = (
        f"【详细信息】\n{detail}\n\n"
        f"【总结建议规则】\n{summary_suggestion}\n\n"
        "请严格按照总结建议规则的格式，用上方真实数据中的具体数字替换其中的 XX，"
        "直接输出总结内容，不要解释。"
    )

    logger.info("[report] 生成总结: %r（数据指标数: %d）", node_name, len(node_data))
    try:
        llm     = LLMService.from_env()
        summary = asyncio.run(llm.complete([{"role": "user", "content": prompt}]))
        on_event({"type": "report_summary", "node_id": node_id, "chunk": summary + "\n\n"})
        logger.info("[report] 总结完成: %r", node_name)
    except Exception as e:
        logger.error("[report] 总结生成失败 %r: %s", node_name, e)
        on_event({"type": "report_summary", "node_id": node_id, "chunk": "_（总结生成失败）_\n\n"})
