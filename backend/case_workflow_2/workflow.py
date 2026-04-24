"""
workflow.py — 大纲生成工作流的编排入口。

完整流程:
  Step 0  template_selector.search_templates()  Embedding 检索 top-10 相似模板
          template_selector.select_template()   LLM 决策：复用模板 or 走 KB 生成
            ├── 复用模板 → 直接加载 outline → Step 8 patch
            └── 走 KB   → Steps 1~7
  Step 1  loader.load_resources()               加载 FAISS 索引 + JSON 知识图谱
  Step 2  retriever.embed_query()               用户问题向量化（Step 0 已完成，复用）
  Step 3  retriever.search_nodes()              FAISS 检索候选节点
  Step 4  retriever.build_candidate_paths()     构建候选节点祖先路径
  Step 5  anchor.select_anchor()                LLM 选锚节点
  Step 6  subtree.build_subtree()               从锚节点递归构建子树
  Step 7  renderer.render_outline()             子树渲染为 Markdown 大纲
  Step 8  patcher.parse_patch()                 LLM 解析修改意图
  Step 9  patcher.apply_patch()                 应用 patch 到树
  Step 10 exporter.export_json()                导出干净 JSON 树

使用方法:
    cd backend
    python case_workflow_2/workflow.py "帮我看传送网络覆盖情况，覆盖率阈值设为 85%"
"""

import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from loader import load_resources
from retriever import embed_query, search_nodes, build_candidate_paths
from anchor import select_anchor
from subtree import build_subtree
from renderer import render_outline
from patcher import parse_patch, apply_patch
from exporter import export_json
from template_selector import search_templates, select_template


async def main(question: str) -> tuple[dict, str]:
    """
    完整工作流：输入用户问题，返回大纲树和 Markdown 文本。

    Step 0 先检索已沉淀模板，LLM 决策是否复用；
    若复用则跳过 KB 检索流程直接进入 patch；
    若不复用则走完整 KB 生成流程。

    Args:
        question: 用户的分析问题

    Returns:
        (outline_tree, outline_markdown)
    """
    # Step 0 / Step 2: 问题向量化（Step 0 和 Step 3 共用）
    query_embedding = await embed_query(question)

    # Step 0a: 检索相似模板
    template_candidates = await search_templates(query_embedding, top_k=10)

    # Step 0b: LLM 决策是否复用模板
    selected = await select_template(question, template_candidates)

    if selected:
        print(f"\n[Step 0] 复用已有模板：{selected['scene_name']}", flush=True)
        tree = selected["outline"]
        return tree, render_outline(tree)

    # Steps 1~7: KB 实时生成
    faiss_svc, nodes_dict, children_map = load_resources()

    hits = search_nodes(query_embedding, faiss_svc)
    if not hits:
        return {}, f"未找到与'{question}'相关的知识节点，请检查索引或降低 FAISS_SCORE_THRESHOLD。"

    candidates = build_candidate_paths(hits, nodes_dict, children_map)
    anchor = await select_anchor(question, candidates)
    tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)

    # Step 7: 渲染初版大纲
    outline = render_outline(tree)

    # Step 8+9: 提取修改意图并应用
    ops = await parse_patch(question, tree)
    if ops:
        tree = apply_patch(tree, ops)
        outline = render_outline(tree)

    return tree, outline


async def modify(user_request: str, outline_tree: dict) -> tuple[dict, str]:
    """
    对已有大纲执行修剪：LLM 解析指令 → 应用 patch → 重新渲染。

    Args:
        user_request : 自然语言修改指令
        outline_tree : main() 或上一轮 modify() 返回的 outline_tree

    Returns:
        (new_tree, outline_markdown)
    """
    ops = await parse_patch(user_request, outline_tree)
    new_tree = apply_patch(outline_tree, ops)
    outline = render_outline(new_tree)
    return new_tree, outline


# ── 本地运行入口 ──────────────────────────────────────────────

def _print_outline(outline: str, json_tree: dict) -> None:
    print("\n" + "=" * 60)
    print("大纲 Markdown")
    print("=" * 60)
    print(outline)
    print("=" * 60)
    print("\n" + "=" * 60)
    print("大纲 JSON")
    print("=" * 60)
    print(json.dumps(json_tree, ensure_ascii=False, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "分析政企OTN升级"
    tree, outline = asyncio.run(main(q))
    _print_outline(outline, export_json(tree))

    while True:
        try:
            cmd = input("\n修改指令（直接回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            break
        tree, outline = asyncio.run(modify(cmd, tree))
        _print_outline(outline, export_json(tree))
