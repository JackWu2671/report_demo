"""
temp_store.py — 将 session 状态持久化到 backend/data/report/{session_id}/

每个 session 目录包含：
  outline.json      大纲树（JSON，纯业务数据，不含报告生成专属的装饰节点/缓存签名）
  outline.md        大纲（Markdown，供人阅读）
  outline.yaml      大纲（YAML，LLM 上下文视图）
  report_data.json  指标原始数据（供重新渲染/签名缓存复用，全量）
  report.md         最终报告（Markdown，含数据表格 + LLM 总结 + 标题序号）
  report.html       最终报告（HTML，含 ECharts 交互图表，需联网加载 CDN）
  _gen_cache.json   report_executor 用的生成缓存签名（metric_sig/content_sig，按节点 id 索引），
                    内部实现细节，不属于业务大纲，前端 /api/session/{id}/outline 不会返回这个文件

这个目录是报告和大纲的唯一权威数据源：report_executor.py 生成报告时只信任这里持久化的
outline.json/report_data.json，前端也只通过 /api/session/{id}/outline 和
/api/session/{id}/report 读取这里的内容，不再自己在内存里拼装/缓存展示状态。
"""

import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_DATA_ROOT   = Path(os.environ.get("REPORT_DATA_DIR") or str(_BACKEND_DIR / "data"))
_REPORT_ROOT = _DATA_ROOT / "report"

logger.info("[temp_store] 会话产物根目录: %s", _REPORT_ROOT)



def _session_dir(session_id: str) -> Path:
    d = _REPORT_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 公开写入接口 ──────────────────────────────────────────────

def _cached_report_data(session_id: str) -> tuple[dict, dict] | None:
    """读取已缓存的报告数据，返回 (collected, summaries) 或 None。

    优先从 data/report/{session_id}/report_data.json 读取 collected（最多 500 行/指标），
    再从 session JSON 读取 summaries（文本，体积小）。
    这样大纲变更后重渲染时能拿到比 session JSON 里 10 行更完整的数据。
    """
    collected: dict = {}
    summaries: dict = {}

    # 1. 从 data/report 目录读 collected（更完整）
    report_data_path = _REPORT_ROOT / session_id / "report_data.json"
    if report_data_path.exists():
        try:
            collected = json.loads(report_data_path.read_text(encoding="utf-8"))
        except Exception:
            collected = {}

    # 2. 从 session JSON 读 summaries；若 data/report 没有 collected 则也从这里回退
    session_dir = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))
    p = session_dir / f"{session_id}.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            summaries = data.get("report_summaries", {})
            if not collected:
                collected = data.get("report_data", {})
        except Exception:
            pass

    if collected or summaries:
        return collected, summaries
    return None


def write_outline(session_id: str, outline_tree: dict, markdown: str, outline_yaml: str) -> None:
    """大纲变更时调用，同步写三视图文件。
    若报告已生成过（report.md 存在），用新大纲 + 缓存数据同步更新报告文件。
    """
    if not session_id or not outline_tree:
        return
    try:
        d = _session_dir(session_id)
        (d / "outline.json").write_text(
            json.dumps(outline_tree, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (d / "outline.md").write_text(markdown or "", encoding="utf-8")
        (d / "outline.yaml").write_text(outline_yaml or "", encoding="utf-8")

        # 如果之前已生成过报告，用新大纲重新渲染（保持报告与大纲同步）
        if (d / "report.md").exists():
            cached = _cached_report_data(session_id)
            if cached:
                write_report(session_id, outline_tree, cached[1], cached[0])
    except Exception as e:
        logger.warning("[temp_store] 写大纲失败 session=%s: %s", session_id, e)


def write_report(
    session_id: str,
    outline_tree: dict,
    summaries: Dict[str, str],
    collected: Dict[str, List],
) -> None:
    """报告生成完成后调用，渲染并写 report.md、report.html 和 report_data.json。"""
    if not session_id or not outline_tree:
        return
    summaries = summaries or {}
    collected = collected or {}
    try:
        d = _session_dir(session_id)

        # 持久化原始数据（全量），供大纲变更后重渲染
        (d / "report_data.json").write_text(
            json.dumps(collected, ensure_ascii=False), encoding="utf-8"
        )

        md = _render_report_md(outline_tree, summaries, collected)
        (d / "report.md").write_text(md, encoding="utf-8")

        ht = _render_report_html(outline_tree, summaries, collected)
        (d / "report.html").write_text(ht, encoding="utf-8")

        logger.info("[temp_store] report.md / report.html 已写入 session=%s", session_id)
    except Exception as e:
        logger.warning("[temp_store] 写报告失败 session=%s: %s", session_id, e)


def read_outline_views(session_id: str) -> dict | None:
    """读取当前持久化的大纲三视图（JSON/Markdown/YAML）；大纲还没生成过则返回 None。

    这是唯一权威的大纲读取入口——/api/session/{id}/outline 和 /api/report 都应该
    调用这个函数，而不是各自拼路径读文件。
    """
    if not session_id:
        return None
    d = _REPORT_ROOT / session_id
    p = d / "outline.json"
    if not p.exists():
        return None
    try:
        outline_tree = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    def _read(name: str) -> str:
        fp = d / name
        return fp.read_text(encoding="utf-8") if fp.exists() else ""

    return {
        "outline_tree": outline_tree,
        "markdown":     _read("outline.md"),
        "outline_yaml": _read("outline.yaml"),
    }


def read_collected(session_id: str) -> Dict[str, List]:
    """读取上次持久化的指标原始数据（report_data.json），供 report_executor 按签名判断
    是否需要重新查询——签名没变的指标直接复用这里的行数据，不用重查。"""
    if not session_id:
        return {}
    p = _REPORT_ROOT / session_id / "report_data.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_gen_cache(session_id: str) -> dict:
    """读取上次的生成缓存签名（metric_sig/content_sig，按节点 id 索引）。
    这是 report_executor 的内部实现细节，不属于业务大纲，故不放进 outline.json。"""
    if not session_id:
        return {}
    p = _REPORT_ROOT / session_id / "_gen_cache.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_gen_cache(session_id: str, cache: dict) -> None:
    if not session_id:
        return
    try:
        d = _session_dir(session_id)
        (d / "_gen_cache.json").write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning("[temp_store] 写生成缓存失败 session=%s: %s", session_id, e)


# ── 公共工具 ─────────────────────────────────────────────────

def _rows_to_md_table(rows: List) -> str:
    if not rows or not isinstance(rows[0], dict):
        return "_（暂无数据）_"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


# ── Markdown 渲染 ─────────────────────────────────────────────

def _render_report_md(
    outline_tree: dict,
    summaries: Dict[str, str],
    collected: Dict[str, List],
) -> str:
    lines: List[str] = []
    _cnt: Dict[int, int] = {}   # depth → 当前计数，用于自动序号

    def _section_num(depth: int) -> str:
        """更新计数器并返回如 "1.2.3" 的序号字符串。"""
        _cnt[depth] = _cnt.get(depth, 0) + 1
        for d in list(_cnt.keys()):
            if d > depth:
                del _cnt[d]
        return ".".join(str(_cnt[d]) for d in sorted(_cnt.keys()))

    def _walk(node: dict, depth: int = 1) -> None:
        node_id  = node.get("id", "")
        name     = node.get("name", "")
        level    = node.get("level", 0)
        desc     = node.get("description", "")
        children = node.get("children", [])

        if node_id == "__root__":
            for child in children:
                _walk(child, depth)   # __root__ 不占一个层级
            return

        h = min(max(1, depth), 6)
        hashes = "#" * h

        if level == 5:
            # L5 指标节点：小标题 + 数据表格
            rows = collected.get(name, [])
            lines.append(f"\n{hashes} {name}\n")
            lines.append(_rows_to_md_table(rows))
            return

        # 结构节点：标题（带序号）+ 描述
        sec = _section_num(depth)
        lines.append(f"\n{hashes} {sec} {name}\n")
        if desc:
            lines.append(f"{desc}\n")

        # 先渲染所有子节点
        for child in children:
            _walk(child, depth + 1)

        # 再渲染当前节点的 summary（与前端 buildSkeleton 顺序一致）
        if node_id in summaries:
            quoted = "\n".join(f"> {ln}" for ln in summaries[node_id].splitlines())
            lines.append(f"\n{quoted}\n")

        # 叶子结构节点（子节点全为 L5 或无子节点）后加分隔线
        has_structural_child = any(c.get("level", 5) != 5 for c in children)
        if not has_structural_child and children:
            lines.append("\n---\n")

    _walk(outline_tree)
    return "\n".join(lines).strip() + "\n"


# ── HTML 渲染 ─────────────────────────────────────────────────

def _build_chart_option(render_type: str, col_x: str, col_y: str, rows: List[dict]) -> dict:
    t = render_type.upper()
    if t == "PIE":
        data = [{"name": str(r.get(col_x, "")), "value": r.get(col_y, 0)} for r in rows]
        return {
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "series":  [{"type": "pie", "radius": ["35%", "65%"], "data": data,
                         "label": {"formatter": "{b}\n{d}%"}}],
        }
    categories = [str(r.get(col_x, "")) for r in rows]
    values     = [r.get(col_y, 0) for r in rows]
    return {
        "tooltip": {"trigger": "axis"},
        "xAxis":   {"type": "category", "data": categories,
                    "axisLabel": {"rotate": 30 if len(categories) > 6 else 0}},
        "yAxis":   {"type": "value"},
        "series":  [{"type": "line" if t == "LINE" else "bar",
                     "data": values, "smooth": t == "LINE"}],
    }


def _rows_to_html_table(rows: List) -> str:
    if not rows or not isinstance(rows[0], dict):
        return '<p class="no-data">（暂无数据）</p>'
    headers = list(rows[0].keys())
    th  = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for row in rows:
        td = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        trs.append(f"<tr>{td}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _summary_to_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = escaped.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f"<p>{escaped}</p>"


def _get_report_title(outline_tree: dict) -> str:
    if outline_tree.get("id") == "__root__":
        children = outline_tree.get("children", [])
        if children:
            return children[0].get("name", "报告")
    return outline_tree.get("name", "报告")


_HTML_TMPL = """\
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    *{{box-sizing:border-box}}
    body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif;
          max-width:960px;margin:0 auto;padding:24px 32px;color:#1a1a1a;line-height:1.75}}
    h1{{border-bottom:2px solid #e0e0e0;padding-bottom:8px;margin-bottom:16px}}
    h2{{color:#222;margin-top:40px;border-bottom:1px solid #eee;padding-bottom:4px}}
    h3{{color:#333;margin-top:28px}}
    h4,h5,h6{{color:#555;margin-top:20px}}
    p{{margin:8px 0}}
    blockquote{{border-left:4px solid #4a9eff;margin:16px 0;padding:10px 16px;
                background:#f0f7ff;color:#333;border-radius:0 4px 4px 0}}
    blockquote p{{margin:4px 0}}
    hr{{border:none;border-top:1px solid #e8e8e8;margin:24px 0}}
    .metric-chart{{width:100%;height:320px;margin:12px 0;
                   border:1px solid #f0f0f0;border-radius:4px}}
    table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:.9em}}
    th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left}}
    th{{background:#f5f5f5;font-weight:600}}
    tr:nth-child(even){{background:#fafafa}}
    .metric-label{{font-weight:600;margin:20px 0 4px;color:#333;
                   padding-left:8px;border-left:3px solid #aaa}}
    .no-data{{color:#999;font-style:italic}}
  </style>
</head>
<body>
{body}
<script>
{scripts}
</script>
</body>
</html>"""


def _render_report_html(
    outline_tree: dict,
    summaries: Dict[str, str],
    collected: Dict[str, List],
) -> str:
    body_parts:    List[str] = []
    chart_scripts: List[str] = []
    _cnt: Dict[int, int] = {}
    chart_counter = [0]

    def _section_num(depth: int) -> str:
        _cnt[depth] = _cnt.get(depth, 0) + 1
        for d in list(_cnt.keys()):
            if d > depth:
                del _cnt[d]
        return ".".join(str(_cnt[d]) for d in sorted(_cnt.keys()))

    def _walk(node: dict, depth: int = 1) -> None:
        node_id  = node.get("id", "")
        name     = node.get("name", "")
        level    = node.get("level", 0)
        desc     = node.get("description", "")
        sql_config = node.get("sql_config") or {}
        rt       = (sql_config.get("renderType") or "").upper()
        col_x    = sql_config.get("colX", "")
        col_y    = sql_config.get("colY", "")
        children = node.get("children", [])

        if node_id == "__root__":
            for child in children:
                _walk(child, depth)   # __root__ 不占一个层级
            return

        h = min(max(1, depth), 6)
        tag = f"h{h}"

        if level == 5:
            rows = collected.get(name, [])
            body_parts.append(
                f'<p class="metric-label">■ {html.escape(name)}</p>'
            )
            if rows and rt in ("BAR", "LINE", "PIE") and col_x and col_y:
                cid = f"ec-{node_id}-{chart_counter[0]}"
                chart_counter[0] += 1
                body_parts.append(f'<div id="{cid}" class="metric-chart"></div>')
                option = _build_chart_option(rt, col_x, col_y, rows)
                chart_scripts.append(
                    f'echarts.init(document.getElementById("{cid}"))'
                    f'.setOption({json.dumps(option, ensure_ascii=False)});'
                )
            elif rows:
                body_parts.append(_rows_to_html_table(rows))
            else:
                body_parts.append('<p class="no-data">（暂无数据）</p>')
            return

        sec = _section_num(depth)
        body_parts.append(
            f"<{tag}>{sec}&nbsp;{html.escape(name)}</{tag}>"
        )
        if desc:
            body_parts.append(f"<p>{html.escape(desc)}</p>")

        for child in children:
            _walk(child, depth + 1)

        if node_id in summaries:
            body_parts.append(
                f"<blockquote>{_summary_to_html(summaries[node_id])}</blockquote>"
            )

        has_structural_child = any(c.get("level", 5) != 5 for c in children)
        if not has_structural_child and children:
            body_parts.append("<hr>")

    _walk(outline_tree)

    return _HTML_TMPL.format(
        title=html.escape(_get_report_title(outline_tree)),
        body="\n".join(body_parts),
        scripts="\n".join(chart_scripts),
    )
