"""
api_server.py — 前端数据接口，供 report_demo frontend 使用。

启动:
    cd backend
    uvicorn api_server:app --port 8888 --reload
"""

import glob
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DIR = os.path.dirname(os.path.abspath(__file__))
_KB_DIR = os.path.join(_DIR, "expert_knowledge")
_TEMPLATE_DIR = os.path.join(_DIR, "templates")
_AGENT_DIR = os.path.join(_DIR, "agent")

# 加载 agent system prompts
def _load_prompt(name: str) -> str:
    with open(os.path.join(_AGENT_DIR, name), encoding="utf-8") as f:
        return f.read()

AGENT_PROMPTS = {
    1: _load_prompt("agent1_prompt.txt"),
    2: _load_prompt("agent2_prompt.txt"),
}

# LLM 客户端
_llm = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY") or "EMPTY",
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
)
_MODEL = os.getenv("LLM_MODEL_NAME", "qwen3-27b")


# ── 知识库 & 模板接口 ──────────────────────────────────────────

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


# ── 对话接口（SSE 流式） ───────────────────────────────────────

class ChatRequest(BaseModel):
    agent_id: int
    messages: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]


@app.post("/api/chat")
async def chat(req: ChatRequest):
    system_prompt = AGENT_PROMPTS.get(req.agent_id)
    if not system_prompt:
        raise HTTPException(status_code=400, detail=f"未知 agent_id: {req.agent_id}")

    full_messages = [{"role": "system", "content": system_prompt}] + req.messages

    async def generate():
        try:
            stream = await _llm.chat.completions.create(
                model=_MODEL,
                messages=full_messages,
                stream=True,
                temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield f"data: {json.dumps({'text': content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
