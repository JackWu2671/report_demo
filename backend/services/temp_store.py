"""
temp_store.py — 将 session 状态持久化到 backend/temp/{session_id}/

每个 session 目录包含：
  outline.json   大纲树（JSON）
  outline.md     大纲（Markdown，供人阅读）
  outline.yaml   大纲（YAML，LLM 上下文视图）
  report.md      最终报告（写入时机：report_executor 完成后）
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_BACKEND_DIR = Path(__file__).parent.parent
_TEMP_ROOT   = Path(os.environ.get("REPORT_TEMP_DIR", str(_BACKEND_DIR / "temp")))


def _session_dir(session_id: str) -> Path:
    d = _TEMP_ROOT / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


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


def write_report(session_id: str, outline_tree: dict, summaries: Dict[str, str]) -> None:
    """报告生成完成后调用，渲染并写 report.md。"""
    if not session_id or not outline_tree:
        return
    try:
        md = _render_report_md(outline_tree, summaries or {})
        ((_session_dir(session_id)) / "report.md").write_text(md, encoding="utf-8")
        logger.info("[temp_store] report.md 已写入 session=%s", session_id)
    except Exception as e:
        logger.warning("[temp_store] 写报告失败 session=%s: %s", session_id, e)


# ── 报告 Markdown 渲染 ────────────────────────────────────────

_LEVEL_HEADING = {1: "#", 2: "##", 3: "###", 4: "####"}


def _render_report_md(outline_tree: dict, summaries: Dict[str, str]) -> str:
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
            lines.append(f"- {name}")
        else:
            heading = _LEVEL_HEADING.get(level, "####")
            lines.append(f"\n{heading} {name}\n")
            if desc:
                lines.append(f"{desc}\n")
            if node_id in summaries:
                lines.append(f"{summaries[node_id]}\n")

        for child in children:
            _walk(child)

    _walk(outline_tree)
    return "\n".join(lines).strip() + "\n"
