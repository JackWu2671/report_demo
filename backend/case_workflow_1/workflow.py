"""
workflow.py — case_workflow_1 编排入口：专家知识沉淀到 JSON 知识库。

完整流程（4 步）:
  Step 1  extractor.extract_from_expert()    LLM 抽取关键词 + 摘要 + 场景名称
  Step 2  searcher.dual_search()             双路 FAISS 检索（关键词 + 摘要），得到命中 id 集合
          searcher.build_kb_tree_text()      渲染完整 KB 树（★ 标命中节点）
  Step 3  outline_gen.generate_outline()     LLM 生成带 [id]/[new] 标注的 Markdown 大纲
  Step 4  kb_updater.parse_new_nodes()       解析大纲中的 [new] 节点（含层级/父节点信息）
          kb_updater.enrich_new_nodes()      LLM 为 [new] 节点补充 keywords/description
          kb_updater.apply_updates()          分配 ID，生成 relation 记录
          kb_updater.save_json_files()        写回 node.json + relation.json
          kb_updater.rebuild_index()          增量追加 FAISS 向量（仅新节点）

Step 4 写入操作需用户在命令行中确认（y/n）。

使用方法:
    cd backend
    python case_workflow_1/workflow.py "专家输入文本..."
    python case_workflow_1/workflow.py  # 交互式多行输入，Ctrl+D 结束
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

from extractor import extract_from_expert
from searcher import dual_search, build_kb_tree_text
from outline_gen import generate_outline
from kb_updater import parse_new_nodes, enrich_new_nodes, apply_updates, save_json_files, rebuild_index

_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")


def _load_kb() -> tuple:
    """加载知识库 JSON 文件和 FAISS 索引。"""
    from services.faiss_service import FAISSService

    with open(os.path.join(_EXPERT_DIR, "node.json"), encoding="utf-8") as f:
        nodes_list = json.load(f)
    with open(os.path.join(_EXPERT_DIR, "relation.json"), encoding="utf-8") as f:
        relations_list = json.load(f)

    nodes_dict = {n["id"]: n for n in nodes_list}
    children_map: dict[str, list[str]] = {}
    for rel in relations_list:
        children_map.setdefault(rel["parent"], []).append(rel["child"])

    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.load(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )

    logger.info(
        "[资源加载] 节点: %d, 关系: %d, FAISS 向量数: %d",
        len(nodes_dict), len(relations_list), faiss_svc.total,
    )
    return faiss_svc, nodes_dict, children_map, nodes_list, relations_list


async def main(expert_text: str) -> tuple[str, list, list, list]:
    """
    完整工作流：专家输入 → 关键词抽取 → 双路检索 → 带 id 大纲生成 → 解析 [new] 节点。

    Args:
        expert_text: 专家输入的自然语言场景描述

    Returns:
        (outline_md, new_nodes_raw, nodes_list, relations_list)
        outline_md      : 带 [id]/[new] 标注的 Markdown 大纲
        new_nodes_raw   : parse_new_nodes() 结果，用于展示给用户确认
        nodes_list      : 当前 KB 节点列表（供 confirm_and_save 使用）
        relations_list  : 当前 KB 关系列表（供 confirm_and_save 使用）
    """
    faiss_svc, nodes_dict, children_map, nodes_list, relations_list = _load_kb()

    # Step 1: 关键词抽取
    extraction = await extract_from_expert(expert_text)
    print(
        f"\n[Step 1 结果] 场景: {extraction.get('scene_name')} "
        f"| 关键词: {extraction.get('keywords')}",
        flush=True,
    )

    # Step 2: 双路检索 + 渲染完整 KB 树
    hits = await dual_search(extraction, faiss_svc, nodes_dict, children_map)
    hit_ids = {h["id"] for h in hits}
    print(f"\n[Step 2 结果] 命中 {len(hit_ids)} 个相关节点", flush=True)

    tree_text = build_kb_tree_text(hit_ids, nodes_dict, children_map)
    logger.info("[Step 2] KB 树状结构:\n%s", tree_text)

    # Step 3: 带 id 标注的大纲生成
    outline_md = await generate_outline(expert_text, tree_text)

    # Step 4a: 解析 [new] 节点（不调用 LLM，纯解析）
    new_nodes_raw = parse_new_nodes(outline_md)

    return outline_md, new_nodes_raw, nodes_list, relations_list


async def confirm_and_save(
    new_nodes_raw: list,
    expert_text: str,
    nodes_list: list,
    relations_list: list,
) -> None:
    """
    应用 KB 更新：LLM 补充元数据 → 写入 JSON → 增量重建 FAISS。

    Args:
        new_nodes_raw  : parse_new_nodes() 返回的 [new] 节点列表
        expert_text    : 专家输入原文（供 enrich LLM 使用）
        nodes_list     : main() 返回的当前节点列表
        relations_list : main() 返回的当前关系列表
    """
    patch = await enrich_new_nodes(new_nodes_raw, expert_text)
    updated_nodes, updated_relations, new_nodes = apply_updates(patch, nodes_list, relations_list)
    save_json_files(updated_nodes, updated_relations)
    await rebuild_index(new_nodes)
    print(f"\n✅ 知识库已更新，新增 {len(new_nodes)} 个节点，FAISS 索引已增量重建。", flush=True)


# ── 本地运行入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("请输入专家场景描述（多行输入，Ctrl+D 结束）:")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        text = "\n".join(lines).strip()

    if not text:
        print("错误: 输入不能为空")
        sys.exit(1)

    outline, new_nodes_raw, nodes_list, relations_list = asyncio.run(main(text))

    print("\n" + "=" * 60)
    print("生成大纲（带 id 标注）")
    print("=" * 60)
    print(outline)
    print("=" * 60)

    if not new_nodes_raw:
        print("\n[Step 4] 大纲完全引用现有知识库节点，无需新增。")
    else:
        print(f"\n[Step 4] 大纲中包含 {len(new_nodes_raw)} 个 [new] 节点:")
        for n in new_nodes_raw:
            parent_label = n.get("parent_id") or n.get("parent_name") or "未知"
            print(f"  L{n['level']} 「{n['name']}」→ 父节点: {parent_label}")

        try:
            ans = input("\n是否将新知识点沉淀到知识库? (y/n) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"

        if ans == "y":
            asyncio.run(confirm_and_save(new_nodes_raw, text, nodes_list, relations_list))
        else:
            print("已取消，知识库未修改。")
