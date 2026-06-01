"""
conversation_store.py — 会话历史持久化到 backend/logs/（不提交 git）。

每个会话存为 logs/{session_id}.json，包含对话历史和大纲快照。
用于：用户刷新/重启后仍能看到并继续历史对话。
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_BACKEND_DIR, "logs")


def _path(session_id: str) -> str:
    return os.path.join(_LOGS_DIR, f"{session_id}.json")


def _derive_title(history: list[dict]) -> str:
    """取第一条用户消息作为标题，截断到 30 字。"""
    for msg in history:
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            text = msg["content"].strip().replace("\n", " ")
            if text:
                return text[:30]
    return "新对话"


def save_conversation(agent) -> None:
    """把 agent 的对话历史与大纲快照写入 logs/{session_id}.json。

    无任何用户消息时不保存（避免空会话刷屏）。
    created_at 在首次保存时确定，后续保留。
    """
    mem = agent.memory
    history = getattr(mem, "_history", []) or []
    if not any(m.get("role") == "user" for m in history):
        return

    os.makedirs(_LOGS_DIR, exist_ok=True)
    p = _path(agent.session_id)

    created_at = datetime.now().isoformat(timespec="seconds")
    if os.path.exists(p):
        try:
            created_at = json.loads(open(p, encoding="utf-8").read()).get("created_at", created_at)
        except Exception:
            pass

    data = {
        "session_id":   agent.session_id,
        "title":        _derive_title(history),
        "created_at":   created_at,
        "updated_at":   datetime.now().isoformat(timespec="seconds"),
        "history":      history,
        "outline_tree": getattr(mem, "outline_tree", {}) or {},
        "outline_yaml": getattr(mem, "outline_yaml", "") or "",
        "markdown":     getattr(mem, "markdown", "") or "",
        "extraction":   getattr(mem, "extraction", {}) or {},
    }
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[Conversation] 已保存 session=%s (%d 条消息)", agent.session_id, len(history))
    except Exception as e:
        logger.warning("[Conversation] 保存失败 session=%s: %s", agent.session_id, e)


def list_conversations() -> list[dict]:
    """列出所有历史会话，按更新时间倒序，返回轻量元信息。"""
    if not os.path.isdir(_LOGS_DIR):
        return []
    items = []
    for name in os.listdir(_LOGS_DIR):
        if not name.endswith(".json"):
            continue
        try:
            d = json.loads(open(os.path.join(_LOGS_DIR, name), encoding="utf-8").read())
        except Exception:
            continue
        items.append({
            "session_id":  d.get("session_id", name[:-5]),
            "title":       d.get("title", "新对话"),
            "created_at":  d.get("created_at", ""),
            "updated_at":  d.get("updated_at", ""),
            "msg_count":   sum(1 for m in d.get("history", []) if m.get("role") in ("user", "assistant")),
        })
    items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return items


def load_conversation(session_id: str) -> Optional[dict]:
    """读取单个会话完整数据，不存在返回 None。"""
    p = _path(session_id)
    if not os.path.exists(p):
        return None
    try:
        return json.loads(open(p, encoding="utf-8").read())
    except Exception as e:
        logger.warning("[Conversation] 读取失败 session=%s: %s", session_id, e)
        return None


def delete_conversation(session_id: str) -> bool:
    """删除会话日志文件。"""
    p = _path(session_id)
    if os.path.exists(p):
        try:
            os.remove(p)
            return True
        except Exception as e:
            logger.warning("[Conversation] 删除失败 session=%s: %s", session_id, e)
    return False


def chat_messages(history: list[dict]) -> list[dict]:
    """把内部历史精简成前端可渲染的消息列表（只保留有文字的 user/assistant）。"""
    out = []
    for m in history:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            out.append({"role": role, "content": content})
    return out
