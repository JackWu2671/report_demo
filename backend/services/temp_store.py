"""
temp_store.py — 将 session 状态持久化到 backend/temp/{session_id}/

每个 session 目录包含：
  outline.json   大纲树（JSON）
  outline.md     大纲（Markdown，供人阅读）
  outline.yaml   大纲（YAML，LLM 上下文视图）
  report.md      最终报告（Markdown，含数据表格 + LLM 总结）
  report.html    最终报告（HTML，含 ECharts 交互图表，需联网加载 CDN）
"""

import html
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent
_TEMP_ROOT   = Path(os.environ.get("REPORT_TEMP_DIR", str(_BACKEND_DIR / "temp")))

_LEVEL_HEADING = {1: "#", 2: "##", 3: "###", 4: "####"}


def _session_dir(session_id: str) -> Path:
    d = _TEMP_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 公开写入接口 ──────────────────────────────────────────────

def write_outline(session_id: str, outline_tree: dict, markdown: str, outline_yaml: str) -> None:
    """大纲变更时调用，同步写三视图文件。"""
    if not session_id or not outline_tree:
        return
    try:
        d = _session_dir(session_id)
        (d / "outline.json").write_text(
            json.dumps(outline_tree, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (d / "outline.md").write_text(markdown or "", encoding="utf-8")
        (d / "outline.yaml").write_text(outline_yaml or "", encoding="utf-8")
    except Exception as e:
        logger.warning("[temp_store] 写大纲失败 session=%s: %s", session_id, e)


def write_report(
    session_id: str,
    outline_tree: dict,
    summaries: Dict[str, str],
    collected: Dict[str, List],
) -> None:
    """报告生成完成后调用，渲染并写 report.md 和 report.html。"""
    if not session_id or not outline_tree:
        return
    summaries  = summaries or {}
    collected  = collected or {}
    try:
        d = _session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)

        md = _render_report_md(outline_tree, summaries, collected)
        (d / "report.md").write_text(md, encoding="utf-8")

        ht = _render_report_html(outline_tree, summaries, collected)
        (d / "report.html").write_text(ht, encoding="utf-8")

        logger.info("[temp_store] report.md / report.html 已写入 session=%s", session_id)
    except Exception as e:
        logger.warning("[temp_store] 写报告失败 session=%s: %s", session_id, e)


# ── Markdown 渲染 ─────────────────────────────────────────────

def _rows_to_md_table(rows: List[dict]) -> str:
    if not rows or not isinstance(rows[0], dict):
        return "_（暂无数据）_"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _render_report_md(
    outline_tree: dict,
    summaries: Dict[str, str],
    collected: Dict[str, List],
) -> str:
    lines: List[str] = []

    def _walk(node: dict) -> None:
        node_id  = node.get("id", "")
        name     = node.get("name", "")
        level    = node.get("level", 0)
        desc     = node.get("description", "")
        children = node.get("children", [])

        if node_id == "__root__":
            for child in children:
                _walk(child)
            return

        if level == 5:
            lines.append(f"\n**{name}**\n")
            rows = collected.get(name, [])
            lines.append(_rows_to_md_table(rows))
        else:
            heading = _LEVEL_HEADING.get(level, "####")
            lines.append(f"\n{heading} {name}\n")
            if desc:
                lines.append(f"{desc}\n")
            if node_id in summaries:
                # indent each summary line with blockquote marker
                quoted = "\n".join(f"> {l}" for l in summaries[node_id].splitlines())
                lines.append(f"{quoted}\n")

        for child in children:
            _walk(child)

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
    else:
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


def _rows_to_html_table(rows: List[dict]) -> str:
    if not rows or not isinstance(rows[0], dict):
        return '<p class="no-data">（暂无数据）</p>'
    headers = list(rows[0].keys())
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = []
    for row in rows:
        td = "".join(f"<td>{html.escape(str(row.get(h, '')))}</td>" for h in headers)
        trs.append(f"<tr>{td}</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(trs)}</tbody></table>"


def _summary_to_html(text: str) -> str:
    """把 LLM 生成的 markdown-ish 总结转为安全 HTML。"""
    escaped = html.escape(text)
    # **bold**
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # 换行
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
    h4{{color:#555;margin-top:20px}}
    p{{margin:8px 0}}
    blockquote{{border-left:4px solid #4a9eff;margin:12px 0;padding:10px 16px;
                background:#f0f7ff;color:#333;border-radius:0 4px 4px 0}}
    blockquote p{{margin:4px 0}}
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
    chart_counter = [0]

    def _walk(node: dict) -> None:
        node_id  = node.get("id", "")
        name     = node.get("name", "")
        level    = node.get("level", 0)
        desc     = node.get("description", "")
        rt       = (node.get("renderType") or "").upper()
        col_x    = node.get("colX", "")
        col_y    = node.get("colY", "")
        children = node.get("children", [])

        if node_id == "__root__":
            for child in children:
                _walk(child)
            return

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
        else:
            tag = f"h{min(level, 4)}"
            body_parts.append(f"<{tag}>{html.escape(name)}</{tag}>")
            if desc:
                body_parts.append(f"<p>{html.escape(desc)}</p>")
            if node_id in summaries:
                body_parts.append(
                    f"<blockquote>{_summary_to_html(summaries[node_id])}</blockquote>"
                )

        for child in children:
            _walk(child)

    _walk(outline_tree)

    return _HTML_TMPL.format(
        title=html.escape(_get_report_title(outline_tree)),
        body="\n".join(body_parts),
        scripts="\n".join(chart_scripts),
    )
