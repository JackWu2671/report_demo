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


# —— 报告生成 SSE 流式接口 ————————————————————————————————————————————

import asyncio
import threading

class ReportRequest(BaseModel):
    session_id: str = ""
    outline_tree: dict
    cached_names: list[str] = []
    cached_summary_ids: list[str] = []


async def _stream_report(session_id: str, outline_tree: dict, cached_names: set, cached_summary_ids: set):
    from services.report_executor import run_report
    from agent_with_skills.agent import _read_session

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

    # 若有节点被条件跳过，调 modify_outline.py 删除，保持脚本格式一致
    if skipped and session_id:
        try:
            ops = json.dumps([{"op": "delete_node", "node_id": s["node_id"]} for s in skipped])
            script = os.path.join(_DIR, "skills", "analyze-network", "scripts", "modify_outline.py")
            env = {
                **os.environ,
                "REPORT_SESSION_ID":  session_id,
                "REPORT_SESSION_DIR": os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"),
                "REPORT_BACKEND_DIR": _DIR,
            }

            proc = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, script, ops],
                env=env, capture_output=True, text=True, timeout=30,
            )

            if proc.returncode == 0:
                session      = _read_session(session_id)
                updated_tree = session.get("outline_tree", {})

                # 同步 agent 内存（与 _detect_events 调用顺序一致）
                agent = _sessions.get(session_id)
                if agent:
                    agent.memory.set_outline(
                        updated_tree,
                        session.get("md_with_ids", ""),
                        session.get("markdown", ""),
                    )
                    names = "、".join(f"「{s['node_name']}」" for s in skipped)
                    agent.memory.add_message({
                        "role": "assistant",
                        "content": f"[系统通知] 报告生成过程中，以下章节因数据条件不满足，已自动从大纲删除：{names}。大纲已同步更新。",
                    })

                yield _sse({
                    "type":         "outline",
                    "markdown":     session.get("markdown", ""),
                    "md_with_ids":  session.get("md_with_ids", ""),
                    "outline_tree": updated_tree,
                })
            else:
                logger.warning("[Report] modify_outline.py 删除跳过节点失败: %s", proc.stderr[:300])
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
