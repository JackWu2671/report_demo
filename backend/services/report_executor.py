"""
report_executor.py — 遍历 outline_tree，执行 SQL，生成描述/总结，落盘到
backend/data/report/{session_id}/（唯一权威数据源，见 temp_store.py）。

on_event 回调仅供内部串联描述/总结生成结果（见 run_report 里的 _capturing_on_event），
不再是前端展示的数据来源——前端只通过 /api/session/{id}/report 读取生成完成后的
report.md/report.html。

描述与总结的区别：
  descriptionSuggestion → description：只用单值型指标（_extract_scalar_data 挑出的单行
    数字），写成一段自然语言文本，只陈述数字，不做分析判断，也不重复罗列分布类明细
    （那些数据下方会单独渲染）；具体行文格式由 descriptionSuggestion 文本本身给出。
  summarySuggestion → summary：用子树下全部指标数据（含分布明细），在给定格式基础上
    结合数据给出分析观点。

生成范围：
  任意层级（L1~L5）节点均支持 summarySuggestion；descriptionSuggestion 仅结构节点
  （L1~L4）有意义，L5 指标节点的 description 按 SOP 约定永远为空。
  _collect_node_data() 递归收集该节点子树下所有 L5 指标的查询结果，作为生成的数据输入。

看网逻辑分析（固定注入节点，id 恒为 VIEW_LOGIC_NODE_ID）：
  run_report() 自己把这个节点插到 outline_tree.children[0]（不需要业务模板显式配置，
  也不需要前端参与注入）。它总结的是"整份报告的分析思路"（先从哪个角度、再从哪个
  角度……），依赖的是其余章节的结构、descriptionSuggestion（大纲设计时写的内容要点，
  往往本身就是成体系的看网逻辑）和已生成的 description/summary，而不是自己的子树
  数据（它没有子树）——所以必须等其余节点都处理完之后才能生成。落盘 outline.json 时
  会剔除这个节点（它是报告生成专属的装饰节点，不属于业务大纲本身），但落盘
  report.md/report.html 时会保留（那是最终报告，理应包含这一节）。

生成缓存（避免大纲没变时重复重查 SQL / 重新调 LLM）：
  完全由后端自己判断，不再依赖前端传入 cached_names/cached_summary_ids。做法是给
  每个节点算一个"稳定签名"（_node_sig：节点自身 + 全部后代，但排除 description/
  summary 等生成结果字段），跟上次生成时记录的签名比较，不一致才重新生成；
  L5 指标另外按 sql_config 算 _metric_sig，签名不一致才重新查询，命中缓存的行数据
  从上次持久化的 report_data.json 里直接复用。签名记录在 _gen_cache.json 里，
  是内部实现细节，不写进 outline.json（temp_store.read_gen_cache/write_gen_cache）。
"""

import asyncio
import hashlib
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

_PROMPTS_DIR = Path(__file__).parent
_DESCRIPTION_PROMPT_FILE = _PROMPTS_DIR / "report_description_prompt.txt"
_SUMMARY_PROMPT_FILE = _PROMPTS_DIR / "report_summary_prompt.txt"
_VIEW_LOGIC_PROMPT_FILE = _PROMPTS_DIR / "report_view_logic_prompt.txt"

VIEW_LOGIC_NODE_ID = "__view_logic__"

logger = logging.getLogger(__name__)


def _load_prompt_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── 生成缓存签名 ─────────────────────────────────────────────

_VOLATILE_KEYS = {"description", "summary"}  # 生成结果字段，不能算进签名（否则会有自引用问题）


def _stable_repr(node: Dict) -> Dict:
    """去掉生成结果字段，只保留决定"生成什么"的输入字段，递归处理 children。"""
    out = {k: v for k, v in node.items() if k not in _VOLATILE_KEYS and k != "children"}
    out["children"] = [_stable_repr(c) for c in node.get("children") or []]
    return out


def _node_sig(node: Dict) -> str:
    """节点自身 + 全部后代（不含生成结果字段）的稳定签名，用于判断内容是否需要重新生成。"""
    payload = json.dumps(_stable_repr(node), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_sig(node: Dict) -> str:
    """L5 指标节点的查询签名，只看决定查询结果的字段。"""
    cfg = node.get("sql_config") or {}
    payload = json.dumps({
        "sql": cfg.get("exec_sql") or "",
        "rt":  (cfg.get("renderType") or "").upper(),
        "x":   cfg.get("colX") or "",
        "y":   cfg.get("colY") or "",
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_view_logic_node(outline_tree: Dict) -> Dict:
    """大纲 children[0] 不是看网逻辑分析节点时自动补一个；已存在则原样返回。"""
    children = outline_tree.get("children", [])
    if children and children[0].get("id") == VIEW_LOGIC_NODE_ID:
        return outline_tree
    view_logic_node = {
        "id": VIEW_LOGIC_NODE_ID,
        "name": "看网逻辑分析",
        "level": 1,
        "description": "",
        "descriptionSuggestion": "系统自动生成：概括整份报告的分析思路（先……再……）",
    }
    return {**outline_tree, "children": [view_logic_node, *children]}


def run_report(
    outline_tree: Dict,
    on_event: Callable[[dict], None],
    session_id: str = "",
) -> None:
    """
    outline_tree 应该是从 backend/data/report/{session_id}/outline.json 读出来的当前
    业务大纲（唯一权威来源，见 api_server.py 的 /api/report）。是否需要重新查询/重新
    生成完全由本函数自己按签名判断，调用方不需要也不应该传入任何缓存提示。
    """
    from services.temp_store import read_collected, read_gen_cache, write_gen_cache

    outline_tree = _ensure_view_logic_node(outline_tree)
    executor = SqlExecutor()
    # collected 用上次持久化的指标数据预填：签名没变的指标直接复用，不用重查
    collected: Dict[str, List] = dict(read_collected(session_id)) if session_id else {}
    descriptions: Dict[str, str] = {}
    summaries: Dict[str, str] = {}

    gen_cache = read_gen_cache(session_id) if session_id else {}
    metric_sigs: Dict[str, str] = dict(gen_cache.get("metric_sig") or {})
    content_sigs: Dict[str, str] = dict(gen_cache.get("content_sig") or {})

    def _capturing_on_event(event: dict) -> None:
        etype = event.get("type")
        if etype == "report_description":
            descriptions[event.get("node_id", "")] = event.get("chunk", "").rstrip("\n")
        elif etype == "report_summary":
            summaries[event.get("node_id", "")] = event.get("chunk", "").rstrip("\n")
        on_event(event)

    children = outline_tree.get("children", [])
    view_logic_node = children[0]
    rest_children = children[1:]

    with DeApiClient() as client:
        _walk(rest_children, client, executor, _capturing_on_event, collected, metric_sigs, content_sigs)

    # 看网逻辑分析依赖其余所有章节已生成的内容，必须放在 _walk() 之后单独处理；
    # 它自己没有子树，签名要用"其余章节整体"来算，不能用它自己的节点签名（那样永远不变）
    view_logic_id = view_logic_node.get("id", "")
    rest_sig = _node_sig({"children": rest_children})
    if content_sigs.get(view_logic_id) != rest_sig:
        _generate_view_logic(view_logic_node, rest_children, _capturing_on_event)
        content_sigs[view_logic_id] = rest_sig

    if session_id:
        write_gen_cache(session_id, {"metric_sig": metric_sigs, "content_sig": content_sigs})
        _persist_report_data(session_id, collected, descriptions, summaries)
        from services.temp_store import write_outline as _write_temp_outline
        from services.temp_store import write_report as _write_temp_report
        if descriptions or summaries:
            # 生成的描述/总结已回填进 outline_tree 各节点的 description/summary 字段
            # （见 _generate_description/_generate_summary），这里把更新后的树重新落盘到
            # outline.json，避免只留在 report_sessions 的临时数据里
            try:
                from outline_utils import to_markdown, to_yaml
                # 看网逻辑分析是报告生成专属的装饰节点，不属于业务大纲本身，落盘
                # outline.json 时要剔除，避免污染 modify_outline/set_outline 等
                # 后续会读写这份"真实业务大纲"的流程
                persisted_tree = {**outline_tree, "children": rest_children}
                _write_temp_outline(session_id, persisted_tree, to_markdown(persisted_tree), to_yaml(persisted_tree))
                logger.info("[report] description/summary 已回填 outline.json（session=%s, 描述数=%d, 总结数=%d）",
                            session_id, len(descriptions), len(summaries))
            except Exception as e:
                logger.error("[report] description/summary 回填 outline.json 失败: %s", e, exc_info=True)
        _write_temp_report(session_id, outline_tree, summaries, collected)


def _persist_report_data(
    session_id: str,
    collected: Dict[str, List],
    descriptions: Dict[str, str],
    summaries: Dict[str, str],
) -> None:
    """将指标查询结果（最多 10 行）、节点描述与总结持久化到 session 文件，供 agent 按需查询。"""
    _session_dir = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))
    p = _session_dir / f"{session_id}.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data["report_data"] = {name: rows[:10] for name, rows in collected.items()}
        data["report_descriptions"] = descriptions
        data["report_summaries"] = summaries
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[report] 报告数据已写入会话 (metrics=%d, descriptions=%d, summaries=%d)",
                    len(collected), len(descriptions), len(summaries))
    except Exception as e:
        logger.warning("[report] 写入会话文件失败: %s", e)


def _run_batch(
    nodes: List[Dict],
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    collected: Dict[str, List],
    metric_sigs: Dict[str, str],
    *,
    force: bool = False,
) -> None:
    """并行执行一批 L5 指标查询，结果写入 collected，查询签名写入 metric_sigs。

    force=True 用于 condition 指标：条件判断必须看当前真实结果，永远重查，不做签名缓存。
    其余情况下，签名跟上次一致的指标直接跳过——collected 已经在 run_report() 里用
    上次持久化的 report_data.json 预填过，跳过的指标数据本来就在里面。
    """
    to_query = nodes if force else [n for n in nodes if metric_sigs.get(n.get("id", "")) != _metric_sig(n)]
    if not to_query:
        return

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(_run_metric, n, executor, on_event): n for n in to_query}
        for future in as_completed(futures):
            n = futures[future]
            try:
                result = future.result()
                if result and result.get("rows"):
                    collected[result["name"]] = result["rows"]
            except Exception as e:
                logger.error("[report] 查询异常 %r: %s", n.get("name", ""), e)
                on_event({"type": "report_metric", "name": n.get("name", ""), "chunk": "_（查询异常）_\n\n"})
            # 无论查到、查空还是异常，都记录本次签名——跟前端旧缓存逻辑一致：只要
            # 查询配置没变就不重查，配置变了（哪怕之前失败过）才会因签名不同再查一次
            metric_sigs[n.get("id", "")] = _metric_sig(n)


def _walk(
    nodes: List[Dict],
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    collected: Dict[str, List],
    metric_sigs: Dict[str, str],
    content_sigs: Dict[str, str],
) -> None:
    """递归遍历节点列表。L5 是查询叶子，其他任意层级均视为结构节点。"""
    metric_nodes = []
    for node in nodes:
        if node.get("level") == 5:
            metric_nodes.append(node)
        else:
            _process_structural(node, client, executor, on_event, collected, metric_sigs, content_sigs)

    # 处理当前层级的孤立 L5 节点（无结构父节点直接挂在这一层）
    if metric_nodes:
        _run_batch(metric_nodes, executor, on_event, collected, metric_sigs)
        for node in metric_nodes:
            if not node.get("summarySuggestion"):
                continue
            node_id, sig = node.get("id", ""), _node_sig(node)
            if content_sigs.get(node_id) != sig:
                _generate_summary(node, {node["name"]: collected.get(node["name"], [])}, on_event)
                content_sigs[node_id] = sig


def _process_structural(
    node: Dict,
    client: DeApiClient,
    executor: SqlExecutor,
    on_event: Callable[[dict], None],
    collected: Dict[str, List],
    metric_sigs: Dict[str, str],
    content_sigs: Dict[str, str],
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

    # Step 1: 查 condition 指标（永远重查——条件判断必须看当前真实结果，不能拿旧数据判断）
    _run_batch(cond_direct + cond_deep, executor, on_event, collected, metric_sigs, force=True)

    # Step 2: condition 判断
    if condition:
        cond_data = {n: collected.get(n, []) for n in condition_queries}
        if not _eval_condition_llm(node, cond_data):
            logger.info("[report] 跳过节点 %r（LLM 判断 condition 不满足）", node.get("name", ""))
            on_event({"type": "report_skip", "node_id": node.get("id", ""), "node_name": node.get("name", "")})
            return

    # descriptionSuggestion / summarySuggestion 共用同一个"是否需要重新生成"的判定：
    # 节点自身 + 全部后代的签名（_node_sig）跟上次记录的不一致，才需要重新生成。
    node_id, node_sig = node.get("id", ""), _node_sig(node)
    needs_fresh_content = (
        bool(node.get("descriptionSuggestion")) or bool(node.get("summarySuggestion"))
    ) and content_sigs.get(node_id) != node_sig

    # Step 3: 查普通直属 L5 指标（签名没变的直接跳过，collected 里已有上次持久化的数据）
    regular_l5 = [n for n in l5_children if n.get("name") not in condition_queries]
    _run_batch(regular_l5, executor, on_event, collected, metric_sigs)

    # Step 4: 递归处理结构子节点
    _walk(structural_children, client, executor, on_event, collected, metric_sigs, content_sigs)

    # Step 5: 描述与总结（描述先生成，对应渲染在数据之前；总结渲染在数据之后）
    if needs_fresh_content:
        node_data = _collect_node_data(node, collected)
        if node.get("descriptionSuggestion"):
            _generate_description(node, node_data, on_event)
        if node.get("summarySuggestion"):
            _generate_summary(node, node_data, on_event)
        content_sigs[node_id] = node_sig


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


def _render_node_detail(node: Dict, node_data: Dict[str, List]) -> str:
    """
    拼接节点子树的详细数据描述文本，供 _generate_description / _generate_summary 共用。
    node_data 是 _collect_node_data() 返回的该节点子树所有 L5 查询结果。
    """
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

    detail_lines = [f"章节名称：{node.get('name', '')}"]
    if node.get("description"):
        detail_lines.append(f"章节说明：{node['description']}")
    detail_lines.append("\n指标查询结果：")
    for child in node.get("children", []):
        detail_lines.extend(_render_node(child, depth=1))
    return "\n".join(detail_lines)


def _extract_scalar_data(node_data: Dict[str, List]) -> Dict[str, str]:
    """
    只保留可当作单一数字看待的指标（单行结果），供 description 使用。
    多行的分布类指标（如"XX分布""XX占比明细"）跳过——那些数据下方会以表格/图表
    形式渲染，description 不重复展示，避免和数据区内容重复。
    """
    scalars: Dict[str, str] = {}
    for name, rows in node_data.items():
        if not rows or not isinstance(rows[0], dict) or len(rows) != 1:
            continue
        row = rows[0]
        if len(row) == 1:
            scalars[name] = str(next(iter(row.values())))
        else:
            scalars[name] = "，".join(f"{k}={v}" for k, v in row.items())
    return scalars


def _generate_description(
    node: Dict,
    node_data: Dict[str, List],
    on_event: Callable[[dict], None],
) -> None:
    """
    调 LLM 生成节点描述并推送 report_description 事件。
    跟 _generate_summary 的区别：
      - 数据输入只给单值型指标（_extract_scalar_data），不把分布类明细数据也塞进去——
        那些已经在数据区渲染过一遍，description 重复罗列没有意义。
      - 输出必须是一段连贯的自然语言文本，只客观陈述数字本身，不做任何分析/判断/建议。
    descriptionSuggestion 当作"内容要点/大纲"参考（历史遗留字段，文本里可能带
    ${指标名} 这类占位符语法），不是要逐字照抄的最终文案模板——LLM 需要结合原有
    的 description（章节说明）重新组织成一段人话，而不是把占位符原样输出。

    descriptionSuggestion 是大纲设计时写的静态文本，大纲结构后续可能被增删指标/
    章节调整过，它提到的点不一定还对应得上当前数据，当前数据里也可能有它没预料到
    的新内容——所以 prompt 里明确要求"以当前数据为准，descriptionSuggestion 只是
    表达角度参考"，避免 LLM 死板地按建议模板的覆盖范围写，而忽略大纲已经变化的事实。
    """
    from services.llm_service import LLMService

    node_id   = node.get("id", "")
    node_name = node.get("name", "")
    scalars   = _extract_scalar_data(node_data)
    data_str  = "\n".join(f"{name}：{val}" for name, val in scalars.items()) or "（无可引用的单值数据）"
    description_suggestion = node["descriptionSuggestion"]
    original_description   = node.get("description") or "（无）"

    prompt = (
        _load_prompt_template(_DESCRIPTION_PROMPT_FILE)
        .replace("{original_description}", original_description)
        .replace("{description_suggestion}", description_suggestion)
        .replace("{data_str}", data_str)
    )

    logger.info("[report] 生成描述: %r（可用数字指标数: %d/%d）", node_name, len(scalars), len(node_data))
    try:
        llm         = LLMService.from_env()
        description = asyncio.run(llm.complete([{"role": "user", "content": prompt}]))
        node["description"] = description.strip()  # 回填到大纲节点，供 outline.json 持久化
        on_event({"type": "report_description", "node_id": node_id, "chunk": description + "\n\n"})
        logger.info("[report] 描述完成: %r", node_name)
    except Exception as e:
        logger.error("[report] 描述生成失败 %r: %s", node_name, e)
        on_event({"type": "report_description", "node_id": node_id, "chunk": "_（描述生成失败）_\n\n"})


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
    detail    = _render_node_detail(node, node_data)
    summary_suggestion = node["summarySuggestion"]

    prompt = (
        _load_prompt_template(_SUMMARY_PROMPT_FILE)
        .replace("{detail}", detail)
        .replace("{summary_suggestion}", summary_suggestion)
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


def _collect_outline_summary_text(nodes: List[Dict], depth: int = 1) -> List[str]:
    """
    递归收集章节结构 + descriptionSuggestion + 已生成的 description/summary，
    供 _generate_view_logic 使用。只关心结构节点（章节），跳过 L5 指标叶子——
    看网逻辑分析讲的是"怎么组织分析"，不是具体指标明细。

    descriptionSuggestion 是大纲设计时写的内容要点，很多时候本身就包含成体系的
    看网逻辑（先看什么、再看什么、怎么判断），比生成后的 description/summary
    更接近"分析思路"本身，所以即使还没生成 description，也要把它带上。
    """
    lines: List[str] = []
    for node in nodes:
        if node.get("level") == 5:
            continue
        indent = "  " * (depth - 1)
        lines.append(f"{indent}{'#' * depth} {node.get('name', '')}")
        if node.get("descriptionSuggestion"):
            lines.append(f"{indent}内容要点：{node['descriptionSuggestion']}")
        if node.get("description"):
            lines.append(f"{indent}说明：{node['description']}")
        if node.get("summary"):
            lines.append(f"{indent}总结：{node['summary']}")
        lines.extend(_collect_outline_summary_text(node.get("children") or [], depth + 1))
    return lines


def _generate_view_logic(
    node: Dict,
    sibling_children: List[Dict],
    on_event: Callable[[dict], None],
) -> None:
    """
    调 LLM 生成"看网逻辑分析"节点内容，复用 report_description 事件上报（平铺文字，
    非 blockquote 总结）。跟其余节点的 description/summary 不同：这里的内容依赖整份
    报告的章节结构和其余节点（sibling_children）已生成的内容，而不是自己的子树数据
    （这个节点没有子树），所以调用方必须保证其余节点都已处理完（见 run_report）。
    """
    from services.llm_service import LLMService

    node_id = node.get("id", "")
    outline_text = "\n".join(_collect_outline_summary_text(sibling_children)) or "（暂无章节内容）"

    prompt = _load_prompt_template(_VIEW_LOGIC_PROMPT_FILE).replace("{outline_text}", outline_text)

    logger.info("[report] 生成看网逻辑分析")
    try:
        llm  = LLMService.from_env()
        text = asyncio.run(llm.complete([{"role": "user", "content": prompt}])).strip()
        node["description"] = text  # 回填到大纲节点，供 outline.json 持久化
        on_event({"type": "report_description", "node_id": node_id, "chunk": text + "\n\n"})
        logger.info("[report] 看网逻辑分析完成")
    except Exception as e:
        logger.error("[report] 看网逻辑分析生成失败: %s", e)
        on_event({"type": "report_description", "node_id": node_id, "chunk": "_（看网逻辑分析生成失败）_\n\n"})
