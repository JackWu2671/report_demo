"""
api_server.py — 前端数据接口，供 report_demo frontend 使用。

启动:
    cd backend
    uvicorn api_server:app --port 8888 --reload
"""

import glob
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
