"""
workflow.py — 大纲生成工作流编排入口。

流程:
  Step 1  search_outline_template()  检索已沉淀模板，LLM 决策是否复用
            ├── 命中 → 直接返回模板大纲
            └── 未命中 → Step 2
  Step 2  search_graph_tree()        FAISS 检索知识图谱 → 补全祖先路径 → 返回候选树
  Step 3  generate_outline()         LLM 基于候选树生成完整大纲（待实现）

使用方法:
    cd backend
    python case_workflow_2/workflow.py "帮我分析传送网络覆盖情况"
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WF2_DIR = os.path.dirname(os.path.abspath(__file__))

for _p in [_BACKEND_DIR, _WF2_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from tools.search_template import search_outline_template
from tools.search_graph_tree import search_graph_tree
from patcher import parse_patch, apply_patch
from renderer import render_outline
from exporter import export_json


async def main(question: str) -> tuple[dict, str]:
    """
    完整工作流：输入用户问题，返回 (outline_tree, outline_markdown)。

    Step 1: 优先复用已沉淀模板；
    Step 2: 模板未命中时检索知识图谱，获取候选节点树；
    Step 3: TODO — generate_outline(question, graph_tree) → 正式大纲。
    """
    # Step 1: 检索模板
    template_result = await search_outline_template(question)
    if template_result["status"] == "pending_confirm":
        logger.info("[workflow] 命中模板: %s", template_result["scene_name"])
        tree = template_result["outline_tree"]
        return tree, template_result["markdown"]

    # Step 2: 知识图谱检索
    graph_result = await search_graph_tree(question)
    if graph_result["status"] == "not_found":
        logger.info("[workflow] 知识图谱无命中")
        return {}, graph_result["message"]

    # Step 3: TODO generate_outline(question, graph_result["graph_tree"])
    # 暂时返回候选树文本，待 generate_outline 实现后替换
    logger.info("[workflow] graph_tree 已就绪，等待 generate_outline 实现")
    return graph_result["graph_tree"], graph_result["tree_text"]


async def modify(user_request: str, outline_tree: dict) -> tuple[dict, str]:
    """
    对已有大纲执行修改：LLM 解析指令 → 应用 patch → 重新渲染。
    """
    ops = await parse_patch(user_request, outline_tree)
    new_tree = apply_patch(outline_tree, ops)
    return new_tree, render_outline(new_tree)


# ── 本地运行入口 ──────────────────────────────────────────────

def _print_outline(outline: str, json_tree) -> None:
    print("\n" + "=" * 60)
    print("大纲 / 候选树")
    print("=" * 60)
    print(outline)
    print("=" * 60)
    print("\n" + "=" * 60)
    print("JSON")
    print("=" * 60)
    print(json.dumps(json_tree, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "分析政企OTN升级"
    tree, outline = asyncio.run(main(q))
    _print_outline(outline, export_json(tree) if isinstance(tree, dict) and tree else tree)

    while True:
        try:
            cmd = input("\n修改指令（直接回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            break
        tree, outline = asyncio.run(modify(cmd, tree))
        _print_outline(outline, export_json(tree))
