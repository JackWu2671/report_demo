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

_KB_DIR = os.path.join(_DIR, "expert_knowledge")
_TEMPLATE_DIR = os.path.join(_DIR, "templates")

# session_id → AgentWithSkills
_sessions: dict[str, AgentWithSkills] = {}


# —— 知识库 & 模板接口 ————————————————————————————————————————————

@app.get("/api/kb")
def get_kb():
    def _load(name):
        p = os.path.join(_KB_DIR, name)
        if not os.path.exists(p):
            return []
        with open(p, encoding="utf-8") as f:
            return json.load(f) or []

    # node.json 是本地构建产物（gitignore），优先使用；否则退回空列表
    nodes     = _load("node.json") or _load("knowledge_nodes.json")
    relations = _load("relation.json") or _load("knowledge_relations.json")

    # 从评估指标.json（或 sample_query_sql.json）提取 id → exec_sql 映射
    sql_source = _load("评估指标.json") or _load("sample_query_sql.json")
    sql_map = {}
    for item in sql_source:
        if item.get("answer"):
            try:
                sql_map[item["id"]] = json.loads(item["answer"]).get("exec_sql", "")
            except Exception:
                pass

    # 将 exec_sql 注入对应 L5 节点；若 nodes 里无 L5，则从 sql_source 补充
    existing_ids = {n["id"] for n in nodes}
    for node in nodes:
        if node.get("level") == 5 and node["id"] in sql_map:
            node["exec_sql"] = sql_map[node["id"]]

    for item in sql_source:
        if item["id"] not in existing_ids:
            node = {k: v for k, v in item.items() if k != "answer"}
            node["exec_sql"] = sql_map.get(item["id"], "")
            nodes.append(node)

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


# —— Session 管理 ————————————————————————————————————————————————

class SessionRequest(BaseModel):
    agent_id: int = 3


@app.post("/api/session")
def create_session(req: SessionRequest):
    session_id = str(uuid.uuid4())
    _sessions[session_id] = AgentWithSkills(session_id=session_id)
    logger.info("[Session] 创建 session=%s", session_id)
    return {"session_id": session_id}


# —— Chat SSE 流式接口 ————————————————————————————————————————————

class ChatRequest(BaseModel):
    session_id: str
    message: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_agent(session_id: str, message: str):
    agent = _sessions.get(session_id)
    if agent is None:
        yield _sse({"type": "error", "message": "Session 不存在，请刷新页面重试"})
        yield "data: [DONE]\n\n"
        return

    try:
        async for event in agent.chat_stream(message):
            yield _sse(event)
    except Exception as e:
        logger.error("[Chat] 异常 session=%s: %s", session_id, e, exc_info=True)
        yield _sse({"type": "error", "message": str(e)})

    yield "data: [DONE]\n\n"


@app.get("/api/session/{session_id}/messages")
def get_session_messages(session_id: str):
    agent = _sessions.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Session 不存在")
    system_prompt = getattr(agent, "system_prompt", None)
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


# —— 报告生成 SSE 流式接口 ————————————————————————————————————————————

import asyncio
import re as _re
import threading

class ReportRequest(BaseModel):
    session_id: str = ""
    outline_tree: dict
    cached_names: list[str] = []
    cached_summary_ids: list[str] = []


def _remove_nodes(tree: dict, ids: set) -> dict:
    """递归删除指定 id 的节点，返回新树（不修改原树）。"""
    new_children = [
        _remove_nodes(c, ids)
        for c in tree.get("children", [])
        if c.get("id") not in ids
    ]
    return {**tree, "children": new_children}


def _render_outline(node: dict, depth: int, with_ids: bool) -> list[str]:
    name    = node.get("name", "")
    node_id = node.get("id", "")
    h       = "#" * min(depth, 6)
    suffix  = f" [{node_id}]" if with_ids and node_id else ""
    lines   = [f"{h} {name}{suffix}", ""]
    if node.get("description"):
        lines += [node["description"], ""]
    for child in node.get("children", []):
        lines.extend(_render_outline(child, depth + 1, with_ids))
    return lines


def _build_outline_md(tree: dict, with_ids: bool) -> str:
    lines = []
    for child in tree.get("children", []):
        lines.extend(_render_outline(child, 1, with_ids))
    return "\n".join(lines).strip()


async def _stream_report(session_id: str, outline_tree: dict, cached_names: set, cached_summary_ids: set):
    from services.report_executor import run_report
    from agent_with_skills.agent import _read_session, _write_session

    loop  = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    skipped: list[dict] = []   # {node_id, node_name}

    def on_event(event: dict):
        if event.get("type") == "report_skip":
            skipped.append({"node_id": event["node_id"], "node_name": event["node_name"]})
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def worker():
        try:
            run_report(outline_tree, on_event, cached_names, cached_summary_ids)
        except Exception as e:
            logger.error("[Report] 生成异常: %s", e, exc_info=True)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {"type": "report_metric", "name": "__error__", "chunk": f"\n\n**[错误]** {e}\n\n"}
            )
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        event = await queue.get()
        if event is None:
            break
        yield _sse(event)

    # 若有节点被条件跳过，从大纲中删除并同步所有状态
    if skipped and session_id:
        try:
            skipped_ids = {s["node_id"] for s in skipped}
            session     = _read_session(session_id)
            updated_tree = _remove_nodes(session.get("outline_tree") or outline_tree, skipped_ids)
            md_with_ids  = _build_outline_md(updated_tree, with_ids=True)
            markdown     = _build_outline_md(updated_tree, with_ids=False)

            # 更新 session 文件
            _write_session(session_id, {
                **session,
                "outline_tree": updated_tree,
                "md_with_ids":  md_with_ids,
                "markdown":     markdown,
            })

            # 同步 agent 内存（防止下次 bash 调用前被旧 memory 覆盖）
            agent = _sessions.get(session_id)
            if agent:
                # 与 _detect_events 保持相同的调用顺序
                agent.memory.set_outline(updated_tree, md_with_ids, markdown)

            yield _sse({
                "type":         "outline",
                "markdown":     markdown,
                "md_with_ids":  md_with_ids,
                "outline_tree": updated_tree,
            })
        except Exception as e:
            logger.error("[Report] 更新大纲失败: %s", e)

    yield _sse({"type": "report_done"})
    yield "data: [DONE]\n\n"


@app.post("/api/report")
async def generate_report(req: ReportRequest):
    return StreamingResponse(
        _stream_report(req.session_id, req.outline_tree, set(req.cached_names), set(req.cached_summary_ids)),
        media_type="text/event-stream",
    )
