"""
会话状态管理，供所有 skill 脚本读写大纲和元数据。

Agent 在执行 bash 命令前将当前内存状态写入 session 文件，
脚本修改后 agent 读回并推送前端事件。

环境变量（由 bash 工具自动注入）：
  REPORT_SESSION_ID  — 会话唯一 ID
  REPORT_SESSION_DIR — session 文件目录（默认 /tmp/report_sessions）
  REPORT_BACKEND_DIR — backend 根目录，供脚本加入 sys.path
"""
import json
import os
from pathlib import Path

SESSION_ID = os.environ.get("REPORT_SESSION_ID", "")
SESSION_DIR = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))
BACKEND_DIR = os.environ.get("REPORT_BACKEND_DIR", "")


def _path() -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{SESSION_ID}.json"


def read() -> dict:
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write(data: dict) -> None:
    _path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def set_outline(outline_tree: dict, md_with_ids: str, markdown: str) -> None:
    data = read()
    data["outline_tree"] = outline_tree
    data["md_with_ids"] = md_with_ids
    data["markdown"] = markdown
    write(data)


def set_extraction(extraction: dict) -> None:
    data = read()
    data["extraction"] = {**data.get("extraction", {}), **extraction}
    write(data)


def get_outline_tree() -> dict:
    return read().get("outline_tree", {})


def get_md_with_ids() -> str:
    return read().get("md_with_ids", "")


def get_extraction() -> dict:
    return read().get("extraction", {})
