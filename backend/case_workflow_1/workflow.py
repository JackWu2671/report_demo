"""
workflow.py — case_workflow_1 编排入口：专家知识沉淀到模板 JSON。

完整流程（5 步）:
  Step 1  extractor.extract_from_expert()    LLM 抽取关键词 + 摘要 + 场景名称
  Step 2  searcher.dual_search()             双路 FAISS 检索，得到命中 id 集合
          searcher.build_kb_tree_text()      渲染完整 KB 树（★ 标命中节点）
  Step 3  outline_gen.generate_outline()     LLM 生成带 [id]/[new] 标注的 Markdown 大纲
  Step 4  kb_updater.parse_new_nodes()       解析大纲中的 [new] 节点（仅展示）
  Step 5  outline_gen.generate_delta()       LLM 总结专家逻辑与现有 KB 框架的差异

CLI 交互:
  - 展示大纲、[new] 节点列表、Step 5 差异总结
  - 询问是否保存大纲模板 JSON（y/n）

使用方法:
    cd backend
    python case_workflow_1/workflow.py "专家输入文本..."
    python case_workflow_1/workflow.py  # 交互式多行输入，--- 单独一行结束
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
from outline_gen import generate_outline, generate_delta
from kb_updater import parse_new_nodes
from template_saver import save_template

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
    return faiss_svc, nodes_dict, children_map, nodes_list


async def main(expert_text: str) -> tuple[dict, str, str, list, dict]:
    """
    完整工作流：专家输入 → 关键词抽取 → 双路检索 → 大纲生成 → [new] 解析 → 差异分析。

    Returns:
        (extraction, outline_md, delta_text, new_nodes_raw, nodes_dict)
    """
    faiss_svc, nodes_dict, children_map, nodes_list = _load_kb()

    # Step 1: 关键词抽取
    extraction = await extract_from_expert(expert_text)
    print(
        f"\n[Step 1 结果] 场景: {extraction.get('scene_name')} "
        f"| 关键词: {extraction.get('keywords')}\n"
        f"           使用条件: {extraction.get('usage_conditions')}",
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

    # Step 4: 解析 [new] 节点（仅展示，不写入 KB）
    new_nodes_raw = parse_new_nodes(outline_md)

    # Step 5: 分析专家逻辑与现有 KB 框架的差异
    delta_text = await generate_delta(expert_text, tree_text, outline_md)

    return extraction, outline_md, delta_text, new_nodes_raw, nodes_dict


async def cli_main(expert_text: str) -> None:
    """CLI 交互流程：运行工作流，询问是否保存大纲模板 JSON。"""
    extraction, outline_md, delta_text, new_nodes_raw, nodes_dict = await main(expert_text)

    # 展示大纲
    print("\n" + "=" * 60)
    print("生成大纲（带 id 标注）")
    print("=" * 60)
    print(outline_md)
    print("=" * 60)

    if not new_nodes_raw:
        print("\n[Step 4] 大纲完全引用现有知识库节点。")
    else:
        print(f"\n[Step 4] 大纲中包含 {len(new_nodes_raw)} 个 [new] 节点:")
        for n in new_nodes_raw:
            parent_label = n.get("parent_id") or n.get("parent_name") or "未知"
            print(f"  L{n['level']} 「{n['name']}」→ 父节点: {parent_label}")

    # 展示差异分析
    print("\n" + "=" * 60)
    print("【Step 5】与现有知识库的逻辑差异")
    print("=" * 60)
    print(delta_text)
    print("=" * 60)

    # 确认保存模板
    try:
        ans = input("\n是否保存大纲模板到 templates/? (y/n) > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans = "n"

    if ans == "y":
        path = save_template(extraction, outline_md, nodes_dict)
        print(f"✅ 模板已保存: {path}", flush=True)
    else:
        print("已跳过模板保存。")


# ── 本地运行入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("请输入专家场景描述（多行输入，输入 --- 单独一行结束）:")
        lines = []
        for line in sys.stdin:
            if line.rstrip("\n") == "---":
                break
            lines.append(line.rstrip("\n"))
        text = "\n".join(lines).strip()

    if not text:
        print("错误: 输入不能为空")
        sys.exit(1)

    asyncio.run(cli_main(text))
