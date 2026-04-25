"""
api_server.py — 前端数据接口，接入实际工作流，以 SSE 流式推送步骤进度。
"""

import glob
import json
import logging
import os
import sys

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

# —— 工作流模块路径注入 ——————————————————————————————————————————
_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_DIR, os.path.join(_DIR, "case_workflow_1"), os.path.join(_DIR, "case_workflow_2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# workflow 1 步骤
from extractor import extract_from_expert
from searcher import dual_search, build_kb_tree_text
from outline_gen import generate_outline, generate_delta
from kb_updater import parse_new_nodes

# workflow 2 步骤
from template_selector import search_templates, select_template
from loader import load_resources
from retriever import embed_query, search_nodes, build_candidate_paths
from anchor import select_anchor
from subtree import build_subtree
from renderer import render_outline
from patcher import parse_patch, apply_patch
from exporter import export_json

# —— FastAPI 应用 ————————————————————————————————————————————————
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_KB_DIR = os.path.join(_DIR, "expert_knowledge")
_TEMPLATE_DIR = os.path.join(_DIR, "templates")


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


# —— SSE 工具函数 ————————————————————————————————————————————————

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _step(n: int, name: str, status: str, detail: str = "") -> str:
    p: dict = {"type": "step", "step": n, "name": name, "status": status}
    if detail:
        p["detail"] = detail
    return _sse(p)


def _text(t: str) -> str:
    return _sse({"type": "text", "text": t})


def _outline(content: str) -> str:
    return _sse({"type": "outline", "content": content})


# —— 对话接口 ————————————————————————————————————————————————————

class ChatRequest(BaseModel):
    agent_id: int
    messages: list[dict]


# —— Workflow 1: 专家知识沉淀（5 步）———————————————————————————

# 知识库资源懒加载，进程生命周期内只加载一次
_KB1_RESOURCES: tuple | None = None


def _get_kb1_resources() -> tuple:
    global _KB1_RESOURCES
    if _KB1_RESOURCES is None:
        from services.faiss_service import FAISSService

        with open(os.path.join(_KB_DIR, "node.json"), encoding="utf-8") as f:
            nodes_list = json.load(f)
        with open(os.path.join(_KB_DIR, "relation.json"), encoding="utf-8") as f:
            relations_list = json.load(f)

        nodes_dict = {n["id"]: n for n in nodes_list}
        children_map: dict[str, list[str]] = {}
        for rel in relations_list:
            children_map.setdefault(rel["parent"], []).append(rel["child"])

        data_dir = os.path.join(_DIR, "data")
        faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
        faiss_svc.load(
            os.path.join(data_dir, "faiss.index"),
            os.path.join(data_dir, "faiss_id_map.json"),
        )
        _KB1_RESOURCES = (faiss_svc, nodes_dict, children_map)
        logger.info(
            "[KB1] 加载完成: %d 节点 / %d 关系 / %d FAISS向量",
            len(nodes_dict), len(relations_list), faiss_svc.total,
        )
    return _KB1_RESOURCES


async def _run_workflow1(messages: list[dict]):
    """Agent 1 — 专家知识沉淀 5步工作流，流式推送步骤进度。"""
    expert_text = messages[-1]["content"]
    logger.info("[WF1] 启动 | 输入 %d 字", len(expert_text))

    try:
        faiss_svc, nodes_dict, children_map = _get_kb1_resources()

        # Step 1: 关键词抽取
        yield _step(1, "提取关键词与场景信息", "running")
        extraction = await extract_from_expert(expert_text)
        logger.info("[WF1] Step1 完成: %s", extraction)
        yield _step(1, "提取关键词与场景信息", "done",
                    f"场景: {extraction.get('scene_name')} | 关键词: {', '.join(extraction.get('keywords', []))}")

        # Step 2: 双路检索
        yield _step(2, "知识库双路检索", "running")
        hits = await dual_search(extraction, faiss_svc, nodes_dict, children_map)
        hit_ids = {h["id"] for h in hits}
        tree_text = build_kb_tree_text(hit_ids, nodes_dict, children_map)
        logger.info("[WF1] Step2 完成: 命中 %d 节点", len(hit_ids))
        yield _step(2, "知识库双路检索", "done", f"命中 {len(hit_ids)} 个相关节点")

        # Step 3: 生成大纲
        yield _step(3, "生成带标注大纲", "running")
        outline_md = await generate_outline(expert_text, tree_text)
        logger.info("[WF1] Step3 完成: 大纲 %d 字", len(outline_md))
        yield _step(3, "生成带标注大纲", "done")

        # Step 4: 解析 [new] 节点
        yield _step(4, "解析新增知识节点", "running")
        new_nodes = parse_new_nodes(outline_md)
        logger.info("[WF1] Step4 完成: %d 个 [new] 节点", len(new_nodes))
        yield _step(4, "解析新增知识节点", "done",
                    f"发现 {len(new_nodes)} 个新节点" if new_nodes else "大纲全部引用现有节点")

        # Step 5: 差异分析
        yield _step(5, "分析与知识库的逻辑差异", "running")
        delta_text = await generate_delta(expert_text, tree_text, outline_md)
        logger.info("[WF1] Step5 完成")
        yield _step(5, "分析与知识库的逻辑差异", "done")

        yield _text(delta_text)
        yield _outline(outline_md)

    except Exception as e:
        logger.error("[WF1] 工作流异常: %s", e, exc_info=True)
        yield _sse({"type": "error", "error": str(e)})

    finally:
        yield "data: [DONE]\n\n"


# —— Workflow 2: 大纲对话生成（最多 10 步）————————————————————————

async def _run_workflow2(messages: list[dict]):
    """Agent 2 — 大纲对话生成最多10步工作流，流式推送步骤进度。"""
    question = messages[-1]["content"]
    logger.info("[WF2] 启动 | 问题: %s", question)

    try:
        # Step 1: 向量化问题
        yield _step(1, "问题向量化", "running")
        query_embedding = await embed_query(question)
        yield _step(1, "问题向量化", "done")

        # Step 2: 检索相似模板
        yield _step(2, "检索已沉淀模板", "running")
        template_candidates = await search_templates(query_embedding, top_k=10)
        yield _step(2, "检索已沉淀模板", "done",
                    f"找到 {len(template_candidates)} 个候选模板")

        # Step 3: LLM 决策是否复用
        yield _step(3, "决策：复用模板或重新生成", "running")
        selected = await select_template(question, template_candidates)
        if selected:
            logger.info("[WF2] 复用模板: %s", selected.get("scene_name"))
            yield _step(3, "决策：复用模板或重新生成", "done",
                        f"复用已有模板: {selected.get('scene_name')}")
            tree = selected["outline"]
            outline_md = render_outline(tree)
            yield _text(f"已复用已有大纲模板：{selected.get('scene_name')}")
            yield _outline(outline_md)
            yield "data: [DONE]\n\n"
            return

        yield _step(3, "决策：复用模板或重新生成", "done", "从知识库实时生成")

        # Step 4: 加载资源
        yield _step(4, "加载知识库与FAISS索引", "running")
        faiss_svc, nodes_dict, children_map = load_resources()
        yield _step(4, "加载知识库与FAISS索引", "done")

        # Step 5: FAISS 检索
        yield _step(5, "FAISS向量检索候选节点", "running")
        hits = search_nodes(query_embedding, faiss_svc)
        if not hits:
            yield _step(5, "FAISS向量检索候选节点", "done", "未找到相关节点")
            yield _text(f"未找到与"{question}"相关的知识节点，请检查索引或降低阈值。")
            yield "data: [DONE]\n\n"
            return
        logger.info("[WF2] Step5 完成: %d 个候选节点", len(hits))
        yield _step(5, "FAISS向量检索候选节点", "done", f"检索到 {len(hits)} 个候选节点")

        # Step 6: 构建祖先路径
        yield _step(6, "构建候选节点祖先路径", "running")
        candidates = build_candidate_paths(hits, nodes_dict, children_map)
        yield _step(6, "构建候选节点祖先路径", "done", f"构建 {len(candidates)} 条路径")

        # Step 7: 选锚节点
        yield _step(7, "LLM选择锚节点", "running")
        anchor = await select_anchor(question, candidates)
        logger.info("[WF2] Step7 锚节点: %s (id=%s)", anchor.get("selected_name"), anchor.get("selected_id"))
        yield _step(7, "LLM选择锚节点", "done",
                    f"{anchor.get('selected_name')} — {anchor.get('selected_path', '')}")

        # Step 8: 构建子树
        yield _step(8, "递归构建子树", "running")
        tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)
        yield _step(8, "递归构建子树", "done")

        # Step 9: 渲染大纲 + 解析修改
        yield _step(9, "渲染大纲 & 解析修改意图", "running")
        outline_md = render_outline(tree)
        ops = await parse_patch(question, tree)
        logger.info("[WF2] Step9 解析到 %d 个修改操作", len(ops))
        if ops:
            yield _step(9, "渲染大纲 & 解析修改意图", "done", f"{len(ops)} 个修改操作")

            # Step 10: 应用修改
            yield _step(10, "应用大纲修改", "running")
            tree = apply_patch(tree, ops)
            outline_md = render_outline(tree)
            yield _step(10, "应用大纲修改", "done")
        else:
            yield _step(9, "渲染大纲 & 解析修改意图", "done", "无需修改")

        yield _text(f"已根据您的需求生成大纲。")
        yield _outline(outline_md)

    except Exception as e:
        logger.error("[WF2] 工作流异常: %s", e, exc_info=True)
        yield _sse({"type": "error", "error": str(e)})

    finally:
        yield "data: [DONE]\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.agent_id == 1:
        return StreamingResponse(_run_workflow1(req.messages), media_type="text/event-stream")
    elif req.agent_id == 2:
        return StreamingResponse(_run_workflow2(req.messages), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=400, detail=f"未知 agent_id: {req.agent_id}")
