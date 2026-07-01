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
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional

from services.de_sql_execution_client import DeApiClient
from services.sql_executor import SqlExecutor

_LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "skills", "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

MAX_PARALLEL = 5  # 同时执行的 metric 查询数

logger = logging.getLogger(__name__)


def run_report(
    outline_tree: Dict,
    on_event: Callable[[dict], None],
    cached_names: set = None,
    cached_summary_ids: set = None,
    session_id: str = "",
) -> None:
    cached_names = cached_names or set()
    cached_summary_ids = cached_summary_ids or set()
    executor = SqlExecutor()
    # collected 贯穿全局，所有 metric 的 rows 都写入这里
    collected: Dict[str, List] = {}
    summaries: Dict[str, str] = {}

    def _capturing_on_event(event: dict) -> None:
        if event.get("type") == "report_summary":
            summaries[event.get("node_id", "")] = event.get("chunk", "").rstrip("\n")
        on_event(event)

    with DeApiClient() as client:
        _walk(outline_tree.get("children", []), client, executor, _capturing_on_event, cached_names, cached_summary_ids, collected)

    if session_id:
        _persist_report_data(session_id, collected, summaries)
        from services.temp_store import write_outline as _write_temp_outline
        from services.temp_store import write_report as _write_temp_report
        if summaries:
            # 生成的总结已回填进 outline_tree 各节点的 summary 字段（见 _generate_summary），
            # 这里把更新后的树重新落盘到 outline.json，避免只留在 report_sessions 的临时总结里
            try:
                from outline_utils import to_markdown, to_yaml
                _write_temp_outline(session_id, outline_tree, to_markdown(outline_tree), to_yaml(outline_tree))
                logger.info("[report] summary 已回填 outline.json（session=%s, 节点数=%d）", session_id, len(summaries))
            except Exception as e:
                logger.error("[report] summary 回填 outline.json 失败: %s", e, exc_info=True)
        _write_temp_report(session_id, outline_tree, summaries, collected)


def _persist_report_data(session_id: str, collected: Dict[str, List], summaries: Dict[str, str]) -> None:
    """将指标查询结果（最多 10 行）和节点总结持久化到 session 文件，供 agent 按需查询。"""
    _session_dir = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))
    p = _session_dir / f"{session_id}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data["report_data"] = {name: rows[:10] for name, rows in collected.items()}
        data["report_summaries"] = summaries
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[report] 报告数据已写入会话 (metrics=%d, summaries=%d)", len(collected), len(summaries))
    except Exception as e:
        logger.warning("[report] 写入会话文件失败: %s", e)


def _run_batch(
    nodes: List[Dict],
    cached_names: set,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    collected: Dict[str, List],
    *,
    silent_names: set | None = None,
) -> None:
    """并行执行一批 L5 指标查询，结果写入 collected。

    silent_names: 这些指标即使在 cached_names 里也会被查询，但不推 SSE 事件。
    用于总结需要重新生成时，静默补全 collected 里的缓存指标数据。
    """
    silent_names = silent_names or set()
    to_query = [n for n in nodes if n.get("name") not in cached_names or n.get("name") in silent_names]
    if not to_query:
        return

    def _emit(event: dict) -> None:
        if event.get("name") not in silent_names:
            on_event(event)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_run_metric, n, executor, _emit): n for n in to_query}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.get("rows"):
                    collected[result["name"]] = result["rows"]
            except Exception as e:
                n = futures[future]
                logger.error("[report] 查询异常 %r: %s", n.get("name", ""), e)
                _emit({"type": "report_metric", "name": n.get("name", ""), "chunk": "_（查询异常）_\n\n"})


def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
    cached_summary_ids: set,
    collected: Dict[str, List],
) -> None:
    """递归遍历节点列表。L5 是查询叶子，其他任意层级均视为结构节点。"""
    metric_nodes = []
    for node in nodes:
        if node.get("level") == 5:
            metric_nodes.append(node)
        else:
            _process_structural(node, client, executor, on_event, cached_names, cached_summary_ids, collected)

    # 处理当前层级的孤立 L5 节点（无结构父节点直接挂在这一层）
    if metric_nodes:
        # 若某 L5 节点总结需重新生成，该节点即使被缓存也需静默查询
        summary_needed = {
            n.get("name") for n in metric_nodes
            if n.get("summarySuggestion") and n.get("id") not in cached_summary_ids
        }
        silent = summary_needed & cached_names
        _run_batch(metric_nodes, cached_names, executor, on_event, collected, silent_names=silent)
        for node in metric_nodes:
            if node.get("summarySuggestion") and node.get("id") not in cached_summary_ids:
                _generate_summary(node, {node["name"]: collected.get(node["name"], [])}, on_event)


def _process_structural(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    cached_names: set,
    cached_summary_ids: set,
    collected: Dict[str, List],
) -> None:
    """
    处理任意非 L5 结构节点（L1~L4 或用户自定义层级）。

    流程：
      1. 查 condition_queries 指标（直属 L5 优先，找不到则从子树深处找）
      2. LLM 判断 condition → 不满足则跳过整个子树
      3. 查普通直属 L5 指标
      4. 递归处理结构子节点
      5. 生成本节点总结
    """
    condition         = node.get("condition", "")
    condition_queries = set(node.get("condition_queries") or [])

    l5_children          = [c for c in node.get("children", []) if c.get("level") == 5]
    structural_children  = [c for c in node.get("children", []) if c.get("level") != 5]

    # condition 指标：直属 L5 先找，剩余从结构子树里找
    cond_direct = [n for n in l5_children if n.get("name") in condition_queries]
    cond_deep   = _find_l5_by_names({"children": structural_children},
                                     condition_queries - {n.get("name") for n in cond_direct})

    # Step 1: 查 condition 指标（不走缓存——collected 每次报告都是全新的，
    # 若被 cached_names 跳过则 collected 里没有数据，条件判断会误判为 False）
    _run_batch(cond_direct + cond_deep, set(), executor, on_event, collected)

    # Step 2: condition 判断
    if condition:
        cond_data = {n: collected.get(n, []) for n in condition_queries}
        if not _eval_condition_llm(node, cond_data):
            logger.info("[report] 跳过节点 %r（LLM 判断 condition 不满足）", node.get("name", ""))
            on_event({"type": "report_skip", "node_id": node.get("id", ""), "node_name": node.get("name", "")})
            return

    needs_fresh_summary = bool(node.get("summarySuggestion")) and node.get("id") not in cached_summary_ids

    # Step 3: 查普通直属 L5 指标
    # 若本节点总结需重新生成，已缓存的指标也要静默查询以填充 collected
    regular_l5 = [n for n in l5_children if n.get("name") not in condition_queries]
    if needs_fresh_summary:
        silent = {n.get("name") for n in regular_l5 if n.get("name") in cached_names}
        _run_batch(regular_l5, cached_names, executor, on_event, collected, silent_names=silent)
    else:
        _run_batch(regular_l5, cached_names, executor, on_event, collected)

    # Step 4: 递归处理结构子节点
    _walk(structural_children, client, executor, on_event, cached_names, cached_summary_ids, collected)

    # Step 5: 总结
    if needs_fresh_summary:
        # structural_children 的深层 L5 若也被缓存跳过，先静默补查
        _backfill_cached_for_summary(node, cached_names, executor, collected)
        _generate_summary(node, _collect_node_data(node, collected), on_event)


def _eval_condition_llm(node: Dict, cond_data: Dict[str, List]) -> bool:
    """
    调 LLM 判断节点是否有必要展示。适用于任意层级（L1~L5）。

    cond_data: condition_queries 的实际查询结果（可能为空列表）。
    返回 True 表示展示，False 表示跳过；LLM 调用失败时默认返回 True。
    """
    from services.llm_service import LLMService

    node_name = node.get("name", "")
    condition = node.get("condition", "")

    # 构建数据描述
    data_lines = []
    for metric_name, rows in cond_data.items():
        data_lines.append(f"■ {metric_name}")
        if rows:
            data_lines.append(SqlExecutor.rows_to_markdown(rows))
        else:
            data_lines.append("  （未查询到数据）")
    data_str = "\n".join(data_lines) if data_lines else "（无条件查询数据）"

    prompt = (
        f"你是报告生成助手，请判断以下报告章节是否有必要展示给用户。\n\n"
        f"【章节名称】{node_name}\n"
        f"【展示条件】{condition}\n"
        f"【条件指标查询结果】\n{data_str}\n\n"
        f"请综合考虑展示条件和实际查询结果（包括数据为空的情况），"
        f"判断该章节是否应该展示。\n"
        f"只回答 true 或 false，不要解释。"
    )

    logger.info("[report] LLM 判断节点 %r condition（条件指标数: %d）", node_name, len(cond_data))
    try:
        llm      = LLMService.from_env()
        response = asyncio.run(llm.complete([{"role": "user", "content": prompt}]))
        result   = response.strip().lower().startswith("true")
        logger.info("[report] LLM condition 判断 %r → %s", node_name, result)
        return result
    except Exception as e:
        logger.error("[report] LLM condition 判断失败 %r: %s，默认展示", node_name, e)
        return True



def _find_l5_by_names(node: Dict, names: set) -> List[Dict]:
    """在节点子树中找到 name 在 names 集合内的所有 L5 节点。"""
    result = []
    for child in node.get("children", []):
        if child.get("level") == 5 and child.get("name") in names:
            result.append(child)
        else:
            result.extend(_find_l5_by_names(child, names))
    return result


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


def _backfill_cached_for_summary(
    node: Dict,
    cached_names: set,
    executor: SqlExecutor,
    collected: Dict[str, List],
) -> None:
    """
    对 node 子树中所有"已被前端缓存但 collected 里仍缺失"的 L5 指标，静默补查。
    用于 structural_children 经 _walk 处理后仍有缺口的情况。
    """
    missing: List[Dict] = []

    def _find(n: Dict) -> None:
        for child in n.get("children", []):
            if child.get("level") == 5:
                name = child.get("name", "")
                if name in cached_names and name not in collected:
                    missing.append(child)
            else:
                _find(child)

    _find(node)
    if not missing:
        return

    logger.info("[report] 总结补查 %d 条缓存指标（静默）", len(missing))
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_run_metric, n, executor, lambda _: None): n for n in missing}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result.get("rows"):
                    collected[result["name"]] = result["rows"]
            except Exception as e:
                logger.warning("[report] 补查异常: %s", e)


_CHART_TYPES = {"BAR", "LINE", "PIE"}


def _run_metric(
    l5: Dict,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
) -> Optional[Dict]:
    """
    在独立线程中执行单条指标查询并推送结果。每次创建自己的 DeApiClient。
    直接使用节点自身的 sql_config.exec_sql 执行查询，mock_data 按节点 id 回落。
    返回 {"name": ..., "rows": [...]} 供 collected 收集，失败返回 None。
    """
    sql_config  = l5.get("sql_config") or {}
    metric_name = l5.get("name", "")
    node_id     = l5.get("id", "")
    exec_sql    = sql_config.get("exec_sql") or ""
    tables      = sql_config.get("tables") or []
    render_type = (sql_config.get("renderType") or "").upper()
    col_x       = sql_config.get("colX") or ""
    col_y       = sql_config.get("colY") or ""

    logger.info("[report] 查询: %r", metric_name)

    force_mock = os.environ.get("FORCE_MOCK", "").lower() in ("1", "true", "yes")

    def _query_real():
        """实时执行 SQL；无 SQL 或查空返回 None。"""
        if not exec_sql:
            logger.warning("[report] 节点 %r 无 exec_sql", metric_name)
            return None
        with DeApiClient() as client:
            table = tables[0] if tables else ""
            r = client.execute_sql_query(exec_sql, table)
        if r:
            logger.info("[report] 真实查询成功: %r，%d 行", metric_name, len(r))
        else:
            logger.warning("[report] 真实查询返回空: %r", metric_name)
        return r

    def _query_mock():
        """取离线 mock；SQL 与生成时不一致则视为无 mock，返回 None。"""
        r = executor.get_mock(node_id, exec_sql)
        if r:
            logger.info("[report] 使用 mock_data: %r", metric_name)
        return r

    # FORCE_MOCK=true：离线优先，没有再实时；否则在线优先，空了再回落离线。
    # 两种模式下 mock 都要求 SQL 与当前节点一致才复用。
    if force_mock:
        logger.info("[report] FORCE_MOCK=true，离线数据优先: %r", metric_name)
        rows = _query_mock() or _query_real()
    else:
        rows = _query_real() or _query_mock()

    if not rows:
        on_event({"type": "report_metric", "name": metric_name, "chunk": "_（暂无数据）_\n\n"})
        return None

    dict_rows = [r for r in rows if isinstance(r, dict)]

    if render_type in _CHART_TYPES and dict_rows:
        on_event({
            "type":        "report_metric",
            "name":        metric_name,
            "chunk":       SqlExecutor.rows_to_markdown(dict_rows) + "\n\n",
            "render_type": render_type,
            "col_x":       col_x,
            "col_y":       col_y,
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
                MAX_ROWS = 20
                display = rows[:MAX_ROWS]
                for row_line in SqlExecutor.rows_to_markdown(display).splitlines():
                    lines.append(f"{indent}{row_line}")
                if len(rows) > MAX_ROWS:
                    lines.append(f"{indent}  （数据共 {len(rows)} 行，已截断展示前 {MAX_ROWS} 行）")
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
        detail_lines.extend(_render_node(child, depth=1))

    detail            = "\n".join(detail_lines)
    summary_suggestion = node["summarySuggestion"]

    prompt = (
        f"【详细信息】\n{detail}\n\n"
        f"【总结建议规则】\n{summary_suggestion}\n\n"
        "请分两步输出：\n"
        "1. 严格按照总结建议规则的格式，用上方真实数据中的具体数字替换其中的 XX；\n"
        "2. 在规则内容之后，结合数据自由补充 1-2 句分析观点，"
        "指出值得关注的趋势、异常或改进方向，语言简练专业。\n"
        "直接输出内容，不要解释步骤。"
    )

    logger.info("[report] 生成总结: %r（数据指标数: %d）", node_name, len(node_data))
    try:
        llm     = LLMService.from_env()
        summary = asyncio.run(llm.complete([{"role": "user", "content": prompt}]))
        node["summary"] = summary.strip()  # 回填到大纲节点，供 outline.json 持久化
        on_event({"type": "report_summary", "node_id": node_id, "chunk": summary + "\n\n"})
        logger.info("[report] 总结完成: %r", node_name)
    except Exception as e:
        logger.error("[report] 总结生成失败 %r: %s", node_name, e)
        on_event({"type": "report_summary", "node_id": node_id, "chunk": "_（总结生成失败）_\n\n"})
