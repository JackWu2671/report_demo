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

app = FastAPI()
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
    try:
        with open(os.path.join(_KB_DIR, "node.json"), encoding="utf-8") as f:
            nodes = json.load(f)
        with open(os.path.join(_KB_DIR, "relation.json"), encoding="utf-8") as f:
            relations = json.load(f)
        return {"nodes": nodes, "relations": relations}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
    _sessions[session_id] = AgentWithSkills()
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
