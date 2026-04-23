"""
workflow.py — 大纲生成工作流的编排入口。

完整流程（10 步）:
  Step 1  loader.load_resources()           加载 FAISS 索引 + JSON 知识图谱
  Step 2  retriever.embed_query()           用户问题向量化
  Step 3  retriever.search_nodes()          FAISS 检索候选节点
  Step 4  retriever.build_candidate_paths() 构建候选节点祖先路径
  Step 5  anchor.select_anchor()            LLM 选锚节点（树状结构呈现候选）
  Step 6  subtree.build_subtree()           从锚节点递归构建子树
  Step 7  renderer.render_outline()         子树渲染为 Markdown 大纲
  Step 8  patcher.parse_patch()             LLM 解析用户问题中的修改意图
  Step 9  patcher.apply_patch()             将 patch 应用到子树（可能为空操作）
  Step 10 exporter.export_json()            导出干净的 JSON 树结构

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

# 将 backend/ 加入路径，使 services/ 和 case_workflow_2/ 下的模块均可导入
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 各步骤模块（位于 case_workflow_2/ 同级目录，运行时已在 sys.path 中）
from loader import load_resources
from retriever import embed_query, search_nodes, build_candidate_paths
from anchor import select_anchor
from subtree import build_subtree
from renderer import render_outline
from patcher import parse_patch, apply_patch
from exporter import export_json


async def main(question: str) -> tuple[dict, str]:
    """
    完整工作流：输入用户问题，返回大纲树和 Markdown 文本。

    step8+9 在 step7 之后执行，从原始问题中提取修改意图（如阈值参数）并应用。
    若问题不含修改意图，step8 返回 []，大纲保持不变。

    Args:
        question: 用户的分析问题，例如 "帮我看传送网络覆盖情况，覆盖率阈值设为 85%"

    Returns:
        (outline_tree, outline_markdown)
        outline_tree     : 大纲的内存树结构，可传入 modify() 做后续修剪
        outline_markdown : Markdown 格式的报告大纲字符串
    """
    # Step 1: 加载资源
    faiss_svc, nodes_dict, children_map = load_resources()

    # Step 2: 问题向量化
    query_embedding = await embed_query(question)

    # Step 3: FAISS 检索候选节点
    hits = search_nodes(query_embedding, faiss_svc)
    if not hits:
        return {}, f"未找到与'{question}'相关的知识节点，请检查索引或降低 FAISS_SCORE_THRESHOLD。"

    # Step 4: 构建候选节点祖先路径
    candidates = build_candidate_paths(hits, nodes_dict, children_map)

    # Step 5: LLM 选锚节点
    anchor = await select_anchor(question, candidates)

    # Step 6: 从锚节点构建子树
    tree = build_subtree(anchor["selected_id"], nodes_dict, children_map)

    # Step 7: 渲染初版大纲
    outline = render_outline(tree)

    # Step 8+9: 从原始问题提取修改意图并应用（无意图时 ops=[]，树不变）
    ops = await parse_patch(question, tree)
    if ops:
        tree = apply_patch(tree, ops)
        outline = render_outline(tree)

    return tree, outline


async def modify(user_request: str, outline_tree: dict) -> tuple[dict, str]:
    """
    对已有大纲执行修剪：LLM 解析指令 → 应用 patch → 重新渲染。

    每次调用在上一轮 outline_tree 的基础上叠加，支持多轮修改。

    Args:
        user_request : 自然语言修改指令，例如 "删除企业分布分析"
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

    # 多轮修改交互循环
    while True:
        try:
            cmd = input("\n修改指令（直接回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            break
        tree, outline = asyncio.run(modify(cmd, tree))
        _print_outline(outline, export_json(tree))
