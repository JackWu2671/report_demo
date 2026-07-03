"""
api_server.py — FastAPI server for AgentWithSkills SSE stream.

Session lifecycle:
  POST /api/session  { agent_id? }  → { session_id }
  POST /api/chat     { session_id, message }  → text/event-stream

Each session keeps one AgentWithSkills instance alive across turns.
Sessions are stored in-process; they are lost on server restart.
"""

import glob
import json
import logging
import os
import subprocess
import sys
import uuid

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

from agent_with_skills.agent import AgentWithSkills
from skills._lib.loader import load_resources
from services import conversation_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("[Startup] 检查 FAISS 索引…")
        await load_resources()
        logger.info("[Startup] FAISS 索引就绪")
    except Exception as e:
        logger.warning("[Startup] FAISS 索引初始化失败（不影响启动）: %s", e)
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_KB_DIR = os.path.join(_DIR, "reference")
_TEMPLATE_DIR = os.path.join(_DIR, "templates")

import time

# session_id → AgentWithSkills
_sessions: dict[str, AgentWithSkills] = {}
# session_id → 最近活跃时间戳（用于过期清理）
_session_last_active: dict[str, float] = {}

# 内存中 session 的存活上限与数量上限，可由环境变量覆盖。
# 过期被清理的 session 不会丢数据——每轮对话已由 conversation_store 落盘到
# logs/，前端再次打开会经 /open 端点从磁盘重建到内存。
_SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", "7200"))   # 默认 2 小时
_MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "500"))


def _touch_session(session_id: str) -> None:
    _session_last_active[session_id] = time.time()


def _drop_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    _session_last_active.pop(session_id, None)


def _evict_sessions() -> None:
    """清理过期的内存 session；若仍超出数量上限，按最久未活跃淘汰。"""
    now = time.time()
    expired = [sid for sid, ts in _session_last_active.items() if now - ts > _SESSION_TTL]
    for sid in expired:
        _drop_session(sid)

    overflow = len(_sessions) - _MAX_SESSIONS
    if overflow > 0:
        oldest = sorted(_session_last_active.items(), key=lambda kv: kv[1])[:overflow]
        for sid, _ in oldest:
            _drop_session(sid)

    removed = len(expired) + max(overflow, 0)
    if removed:
        logger.info("[Session] 清理内存 session %d 个（过期%d + 超限%d），当前存活 %d",
                    removed, len(expired), max(overflow, 0), len(_sessions))


# —— 知识库 & 模板接口 ————————————————————————————————————————————

@app.get("/api/kb")
def get_kb():
    def _load(name):
        p = os.path.join(_KB_DIR, name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return json.load(f) or []

    # node.json 是本地构建产物（gitignore），L5 节点已内含 sql_config 等字段
    nodes     = _load("node.json") or _load("knowledge_nodes.json")
    relations = _load("relation.json") or _load("knowledge_relations.json")

    return {"nodes": nodes, "relations": relations}


@app.get("/api/templates")
def get_templates():
    if not os.path.isdir(_TEMPLATE_DIR):
        return []
    templates = []
    for path in sorted(glob.glob(os.path.join(_TEMPLATE_DIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                templates.append(json.load(f))
        except Exception:
            pass
    return templates


# —— Session 产物读取（供轮询） ————————————————————————————————————

_DATA_ROOT   = os.environ.get("REPORT_DATA_DIR") or os.path.join(_DIR, "data")
_REPORT_ROOT = os.path.join(_DATA_ROOT, "report")


@app.get("/api/session/{session_id}/outline")
def get_session_outline(session_id: str):
    """返回最新大纲三视图（JSON / Markdown / YAML）。"""
    from services import temp_store
    view = temp_store.read_outline_views(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="大纲尚未生成")
    return view


@app.get("/api/session/{session_id}/report")
def get_session_report(session_id: str, fmt: str = "html"):
    """返回最新报告内容。fmt=html（默认）或 md。"""
    d = os.path.join(_REPORT_ROOT, session_id)
    filename = "report.html" if fmt == "html" else "report.md"
    p = os.path.join(d, filename)
    if not os.path.exists(p):
        raise HTTPException(status_code=404, detail="报告尚未生成")
    from fastapi.responses import PlainTextResponse
    content_type = "text/html; charset=utf-8" if fmt == "html" else "text/markdown; charset=utf-8"
    return PlainTextResponse(open(p, encoding="utf-8").read(), media_type=content_type)


# —— Session 管理 ————————————————————————————————————————————————

class SessionRequest(BaseModel):
    agent_id: int = 3


@app.post("/api/session")
def create_session(req: SessionRequest):
    _evict_sessions()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = AgentWithSkills(session_id=session_id)
    _touch_session(session_id)
    logger.info("[Session] 创建 session=%s（当前存活 %d）", session_id, len(_sessions))
    return {"session_id": session_id}


# —— 历史会话 ————————————————————————————————————————————————————

@app.get("/api/conversations")
def list_conversations():
    return {"conversations": conversation_store.list_conversations()}


@app.post("/api/conversations/{session_id}/open")
def open_conversation(session_id: str):
    """把历史会话恢复到内存并返回供前端渲染的消息与大纲。"""
    from services import temp_store

    data = conversation_store.load_conversation(session_id)
    if data is None:
        raise HTTPException(status_code=404, detail="历史会话不存在")

    # backend/data/report/{id}/ 是大纲的权威数据源——报告生成和对话里的
    # set_outline/modify_outline 等操作都会实时写到这里，比 conversation_store
    # 里的大纲快照更新（快照只在每轮对话结束时保存一次）。存在就优先用它，
    # 避免出现"重开历史会话看到的大纲是生成报告之前的旧版本"。
    outline_view = temp_store.read_outline_views(session_id)
    if outline_view is not None:
        outline_tree, markdown, outline_yaml = (
            outline_view["outline_tree"], outline_view["markdown"], outline_view["outline_yaml"],
        )
    else:
        outline_tree = data.get("outline_tree", {}) or {}
        markdown     = data.get("markdown", "") or ""
        outline_yaml = data.get("outline_yaml", "") or ""

    # 重建 agent 内存状态（已在内存则直接复用）
    agent = _sessions.get(session_id)
    if agent is None:
        agent = AgentWithSkills(session_id=session_id)
        agent.memory._history = data.get("history", [])
        agent.memory.set_outline(outline_tree, markdown, outline_yaml)
        if hasattr(agent.memory, "set_extraction") and data.get("extraction"):
            agent.memory.set_extraction(data["extraction"])
        _sessions[session_id] = agent
        # 同步大纲到 /tmp session 文件，使后续脚本与报告生成可用
        from agent_with_skills.agent import _write_session
        _write_session(session_id, {
            "outline_tree": agent.memory.outline_tree or {},
            "outline_yaml": agent.memory.outline_yaml or "",
            "markdown":     agent.memory.markdown or "",
            "extraction":   getattr(agent.memory, "extraction", {}) or {},
        })
    _touch_session(session_id)

    return {
        "session_id":   session_id,
        "messages":     conversation_store.chat_messages(data.get("history", [])),
        "outline_tree": outline_tree,
        "outline_yaml": outline_yaml,
        "markdown":     markdown,
        "extraction":   data.get("extraction", {}) or {},
    }


@app.delete("/api/conversations/{session_id}")
def delete_conversation(session_id: str):
    conversation_store.delete_conversation(session_id)
    _drop_session(session_id)
    return {"ok": True}


# —— Chat SSE 流式接口 ————————————————————————————————————————————

class ChatRequest(BaseModel):
    session_id: str
    message: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_agent(session_id: str, message: str):
    _evict_sessions()
    agent = _sessions.get(session_id)
    if agent is None:
        yield _sse({"type": "error", "message": "Session 不存在，请刷新页面重试"})
        yield "data: [DONE]\n\n"
        return
    _touch_session(session_id)

    try:
        async for event in agent.chat_stream(message):
            yield _sse(event)
    except Exception as e:
        logger.error("[Chat] 异常 session=%s: %s", session_id, e, exc_info=True)
        yield _sse({"type": "error", "message": str(e)})

    # 一轮结束后把对话历史与大纲快照持久化到 logs/，供历史会话列表使用
    conversation_store.save_conversation(agent)

    yield "data: [DONE]\n\n"


@app.get("/api/session/{session_id}/messages")
def get_session_messages(session_id: str):
    agent = _sessions.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Session 不存在")
    system_prompt = getattr(agent, "_system_prompt", None) or getattr(agent, "system_prompt", None)
    if system_prompt:
        messages = agent.memory.build_messages(system_prompt)
    else:
        messages = agent.memory._history
    return {"messages": messages}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session 不存在，请刷新页面重试")
    return StreamingResponse(
        _stream_agent(req.session_id, req.message),
        media_type="text/event-stream",
    )


# —— 报告生成 ————————————————————————————————————————————————————
#
# backend/data/report/{session_id}/ 是唯一权威数据源：大纲从这里读，生成结果也写回
# 这里；前端不再传大纲 JSON 或缓存提示，只传 session_id，生成完直接去
# /api/session/{id}/report 取最终 report.md/report.html。

import asyncio

class ReportRequest(BaseModel):
    session_id: str


@app.post("/api/report")
async def generate_report(req: ReportRequest):
    from services import temp_store
    from services.report_executor import run_report

    if not req.session_id:
        raise HTTPException(status_code=400, detail="缺少 session_id")
    _touch_session(req.session_id)

    view = temp_store.read_outline_views(req.session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="大纲尚未生成，无法生成报告")

    skipped: list[dict] = []   # {node_id, node_name}，因 condition 不满足被跳过的节点

    def on_event(event: dict) -> None:
        if event.get("type") == "report_skip":
            skipped.append({"node_id": event.get("node_id", ""), "node_name": event.get("node_name", "")})

    try:
        await asyncio.to_thread(run_report, view["outline_tree"], on_event, session_id=req.session_id)
    except Exception as e:
        logger.error("[Report] 生成异常 session=%s: %s", req.session_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"报告生成失败: {e}")

    final_view = temp_store.read_outline_views(req.session_id) or view

    # 若有节点因 condition 不满足被跳过，从大纲里删除并重新落盘（连带重渲染报告）
    if skipped:
        try:
            _SKILLS_LIB = os.path.join(_DIR, "skills", "_lib")
            if _SKILLS_LIB not in sys.path:
                sys.path.insert(0, _SKILLS_LIB)
            from patcher import apply_patch
            from outline_utils import to_clean_json, to_markdown, to_yaml

            ops = [{"op": "delete_node", "node_id": s["node_id"]} for s in skipped]
            new_tree, _ = await apply_patch(final_view["outline_tree"], ops)
            updated_tree = to_clean_json(new_tree)
            md, yaml_str = to_markdown(updated_tree), to_yaml(updated_tree)
            temp_store.write_outline(req.session_id, updated_tree, md, yaml_str)
            final_view = {"outline_tree": updated_tree, "markdown": md, "outline_yaml": yaml_str}
        except Exception as e:
            logger.error("[Report] 处理条件跳过节点失败 session=%s: %s", req.session_id, e)

    # 把生成结果（description/summary 回填、条件跳过的删除）同步回 agent 内存 + 历史会话
    # 快照，这是之前"前端和后端沉淀内容不一致"问题的根源——之前这里没有回写，
    # conversation_store 存的一直是生成之前的旧大纲
    agent = _sessions.get(req.session_id)
    if agent is not None:
        agent.memory.set_outline(final_view["outline_tree"], final_view["markdown"], final_view["outline_yaml"])
        if skipped:
            names = "、".join(f"「{s['node_name']}」" for s in skipped)
            agent.memory.add_message({
                "role": "assistant",
                "content": f"[系统通知] 报告生成过程中，以下章节因数据条件不满足，已自动从大纲删除：{names}。大纲已同步更新。",
            })
        conversation_store.save_conversation(agent)

    return {"ok": True, "skipped": skipped}
