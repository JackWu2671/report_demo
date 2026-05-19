"""
tool_server.py — 工具执行 REST 服务。

供 Java Agent 调用，每个端点对应一个工具，全部无状态（输入→输出）。
outline_tree 等状态由 Java Agent 维护，调用时随请求体传入。

启动：uvicorn tool_server:app --port 8889
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from tools.search_graph_tree import search_graph_tree
from tools.search_template import search_outline_templates, load_template_outline
from tools.build_outline_from_anchor import build_outline_from_anchor
from tools.modify_outline import modify_outline
from tools.set_outline_from_markdown import set_outline_from_markdown
from tools.save_template import save_outline_template

app = FastAPI(title="Tool Server")


@app.post("/tools/search_graph_tree")
async def api_search_graph_tree(req: dict):
    return await search_graph_tree(req["question"], req.get("top_k", 5))


@app.post("/tools/search_outline_templates")
async def api_search_outline_templates(req: dict):
    return await search_outline_templates(req["question"], req.get("top_k", 5))


@app.post("/tools/load_template_outline")
async def api_load_template_outline(req: dict):
    return load_template_outline(req["template_id"])


@app.post("/tools/build_outline_from_anchor")
async def api_build_outline_from_anchor(req: dict):
    return await build_outline_from_anchor(req["anchor_id"])


@app.post("/tools/modify_outline")
async def api_modify_outline(req: dict):
    return await modify_outline(req["ops"], req["outline_tree"])


@app.post("/tools/set_outline_from_markdown")
async def api_set_outline_from_markdown(req: dict):
    return await set_outline_from_markdown(req["markdown"])


@app.post("/tools/save_outline_template")
async def api_save_outline_template(req: dict):
    return await save_outline_template(req["extraction"], req["outline_tree"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8889)
