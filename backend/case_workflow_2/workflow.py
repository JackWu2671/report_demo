"""
case_workflow_2 — 基于 JSON 知识图谱 + FAISS 的报告大纲生成工作流。

对应 report_system 的 case_2（outline-generate）：
  - 用 expert_knowledge/node.json + relation.json 替代 Neo4j
  - 用 services/faiss_service.py 做向量检索
  - 用 ANCHOR_PROMPT + LLM 选锚节点（同 report_system/graph_rag_executor.py）
  - 子树直接渲染为大纲，支持 delete / set_param 修剪

使用方法:
    cd backend
    python case_workflow_2/workflow.py "分析政企OTN升级"
"""

import asyncio
import copy
import json
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.embedding_service import EmbeddingService
from services.faiss_service import FAISSService
from services.llm_service import LLMService

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_EXPERT_DIR = os.path.join(_BASE_DIR, "expert_knowledge")

# 与 report_system/graph_rag_executor.py 保持一致
ANCHOR_PROMPT = """你是知识库节点选择专家。从候选中选出最符合用户【核心分析主题】的唯一锚节点。

候选节点以树状结构呈现（★ 为候选节点，无★ 为路径上下文）。

选择原则：
1. 以用户的核心分析主题（即"要分析什么范围"）为锚点依据，而非具体的阈值、参数或条件
   - "低阶交叉利用率阈值设为70%"、"时间范围选近3个月"等属于参数条件，不是锚点选择依据
2. 宽泛意图 → 高层级节点；具体意图 → 低层级节点
3. 候选中存在父子关系时，按主题粒度判断：
   - 问题指向某一方向的全面分析 → 选更高层级的父节点，使大纲覆盖该方向的完整内容
   - 问题只关注某一个具体指标 → 才选该指标所在的低层级节点
4. 当某个节点的多个下级子节点都出现在候选（★）中时，优先选择该父节点，而非其中某一子节点

用 ```json``` 代码块包裹输出，不要加解释文字。格式:
```json
{"selected_id":"","selected_name":"","selected_path":"","level":0,"reason":""}
```"""

PATCH_PROMPT = """你是大纲修改助手。根据用户问题和当前大纲结构，判断是否需要对大纲进行修改。

【重要】大多数情况下无需修改，应直接输出 []。
仅当用户问题中明确包含以下意图时才输出操作：
- 明确要求删除某个章节 → delete
- 明确指定某个指标的具体阈值或参数 → set_param

当前大纲以 [id=... L层级] 节点名 格式展示，修改时必须使用节点的 id 字段。

支持的操作：
1. delete    — 删除指定节点及其所有子节点
2. set_param — 为节点设置参数标注（如阈值、时间窗口等）

用 ```json``` 代码块输出操作列表，格式：
```json
[
  {"op": "delete",    "node_id": "L4_001"},
  {"op": "set_param", "node_id": "L5_009", "key": "threshold", "value": 70, "unit": "%"}
]
```
无需修改时输出 []，不要输出任何解释文字。"""


# ─────────────────────────────────────────────────────────────
# Step 1: 加载资源（FAISS 索引 + JSON 知识图谱）
# ─────────────────────────────────────────────────────────────

def step1_load_resources() -> tuple[FAISSService, dict, dict]:
    """
    加载 FAISS 索引和 JSON 知识图谱。

    Returns:
        faiss_svc   : FAISSService 实例（已载入索引）
        nodes_dict  : {node_id -> node_dict}
        children_map: {parent_id -> [child_id, ...]}
    """
    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.load(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )

    with open(os.path.join(_EXPERT_DIR, "node.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    with open(os.path.join(_EXPERT_DIR, "relation.json"), encoding="utf-8") as f:
        relations = json.load(f)

    nodes_dict = {n["id"]: n for n in nodes}
    children_map: dict[str, list[str]] = {}
    for rel in relations:
        children_map.setdefault(rel["parent"], []).append(rel["child"])

    logger.info(
        f"[Step 1] 加载完成 — 节点: {len(nodes_dict)}, 关系: {len(relations)}, "
        f"FAISS 向量数: {faiss_svc.total}"
    )
    return faiss_svc, nodes_dict, children_map


# ─────────────────────────────────────────────────────────────
# Step 2: 用户问题向量化
# ─────────────────────────────────────────────────────────────

async def step2_embed_query(question: str) -> np.ndarray:
    """
    调用 Embedding 服务将用户问题向量化。

    Returns:
        shape (1, dim) 的 float32 ndarray
    """
    emb_svc = EmbeddingService(
        base_url=os.getenv("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
        dim=int(os.getenv("EMBEDDING_DIM", 1024)),
    )
    vec = await emb_svc.get_embedding(question)
    logger.info(f"[Step 2] 问题向量化完成: '{question[:60]}'")
    return vec


# ─────────────────────────────────────────────────────────────
# Step 3: FAISS 检索候选节点
# ─────────────────────────────────────────────────────────────

def step3_search_nodes(
    query_embedding: np.ndarray,
    faiss_svc: FAISSService,
    top_k: int = 10,
    threshold: float | None = None,
) -> list[dict]:
    """
    在 FAISS 索引中检索与问题最相关的候选节点。

    Returns:
        [{id, name, level, score, ...}, ...]  按分数降序
    """
    th = threshold if threshold is not None else float(os.getenv("FAISS_SCORE_THRESHOLD", 0.3))
    hits = faiss_svc.search(query_embedding, top_k=top_k, threshold=th)
    logger.info(
        f"[Step 3] FAISS 命中 {len(hits)} 个节点: "
        + ", ".join(f"{h['name']}({h['score']:.3f})" for h in hits)
    )
    return hits


# ─────────────────────────────────────────────────────────────
# Step 4: 为候选节点构建祖先路径
# ─────────────────────────────────────────────────────────────

def step4_build_candidate_paths(
    hits: list[dict],
    nodes_dict: dict,
    children_map: dict,
) -> list[dict]:
    """
    为每个 FAISS 命中节点构建完整祖先路径，供 LLM 理解节点在图中的位置。

    格式同 report_system/graph_rag_executor.py：
      id=... name=... level=... path=（根 > ... > 节点）

    Returns:
        [{id, name, level, score, path}, ...]
    """
    parent_map: dict[str, str] = {}
    for pid, cids in children_map.items():
        for cid in cids:
            parent_map[cid] = pid

    candidates = []
    for hit in hits:
        chain: list[str] = []
        cur: str | None = hit["id"]
        while cur and cur in nodes_dict:
            chain.append(nodes_dict[cur]["name"])
            cur = parent_map.get(cur)
        chain.reverse()  # 根 → 叶
        candidates.append(
            {
                "id": hit["id"],
                "name": hit["name"],
                "level": hit["level"],
                "score": hit["score"],
                "path": " > ".join(chain),
            }
        )

    logger.info(
        f"[Step 4] {len(candidates)} 个候选节点:\n"
        + "\n".join(
            f"  L{c['level']} {c['name']} | {c['path']} | score={c['score']:.3f}"
            for c in candidates
        )
    )
    return candidates


# ─────────────────────────────────────────────────────────────
# Step 5: LLM 选锚节点
# ─────────────────────────────────────────────────────────────

async def step5_select_anchor(question: str, candidates: list[dict]) -> dict:
    """
    使用 ANCHOR_PROMPT 让 LLM 从候选节点中选出最符合用户意图的锚节点。

    候选格式（同 report_system）：
        - id=... name=... level=... path=...

    Returns:
        {"selected_id": ..., "selected_name": ..., "selected_path": ...,
         "level": ..., "reason": ...}
    """
    llm = LLMService.from_env()

    # 将候选节点渲染为树状结构，让 LLM 能看清父子覆盖关系
    tree_text = _candidates_to_tree_text(candidates)

    messages = [
        {"role": "system", "content": ANCHOR_PROMPT},
        {"role": "user", "content": f"## 候选（树状结构）\n{tree_text}\n\n## 问题\n{question}"},
    ]

    logger.info(
        "[Step 5] Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 5] LLM 流式输出 ↓", flush=True)
    print("-" * 50, flush=True)
    full_text = ""
    try:
        async for chunk in llm.complete_stream(messages):
            print(chunk, end="", flush=True)
            full_text += chunk
        print("\n" + "-" * 50, flush=True)
        logger.info("[Step 5] LLM 完整输出:\n%s", full_text)
        anchor = LLMService._parse_json(full_text)
    except Exception as e:
        logger.warning(f"[Step 5] LLM 选锚失败，回退到 score 最高的候选: {e}")
        f = candidates[0]
        anchor = {
            "selected_id": f["id"],
            "selected_name": f["name"],
            "selected_path": f["path"],
            "level": f["level"],
            "reason": "fallback",
        }

    logger.info(
        "[Step 5] 选锚: '%s' (L%s), reason=%s",
        anchor.get("selected_name"),
        anchor.get("level"),
        anchor.get("reason", ""),
    )
    return anchor


# ─────────────────────────────────────────────────────────────
# Step 6: 从锚节点构建子树
# ─────────────────────────────────────────────────────────────

def step6_build_subtree(
    anchor_id: str, nodes_dict: dict, children_map: dict
) -> dict:
    """
    从锚节点递归构建完整子树 dict（含 children 列表）。

    Returns:
        subtree dict，结构: {id, name, level, description, children: [...]}
    """
    if anchor_id not in nodes_dict:
        raise ValueError(f"锚节点 '{anchor_id}' 不在知识图谱中")

    subtree = _build_subtree(anchor_id, nodes_dict, children_map)
    logger.info(
        f"[Step 6] 子树构建完成: 根='{nodes_dict[anchor_id]['name']}', "
        f"共 {_count_nodes(subtree)} 个节点"
    )
    return subtree


# ─────────────────────────────────────────────────────────────
# Step 7: 子树渲染为 Markdown 大纲
# ─────────────────────────────────────────────────────────────

def step7_render_outline(subtree: dict) -> str:
    """
    将锚节点子树直接渲染为 Markdown 大纲，无需 LLM。
    锚节点对应 # 标题，每深一层标题级别加一。
    若节点有 params，渲染为标注行。

    Returns:
        Markdown 格式的报告大纲字符串
    """
    blocks = _render_node(subtree, heading_level=1)
    outline = "\n\n".join(blocks)
    logger.info(f"[Step 7] 大纲渲染完成: {len(outline)} 字符")
    return outline


# ─────────────────────────────────────────────────────────────
# Step 8: LLM 解析用户修改指令 → patch 操作列表
# ─────────────────────────────────────────────────────────────

async def step8_parse_patch(user_request: str, outline_tree: dict) -> list[dict]:
    """
    将用户的自然语言修改指令翻译为结构化 patch 操作列表。

    支持操作:
        delete    — 删除节点及其子节点
        set_param — 为节点打参数标注

    Returns:
        [{"op": ..., "node_id": ..., ...}, ...]
    """
    llm = LLMService.from_env()
    tree_text = _tree_to_id_text(outline_tree)

    messages = [
        {"role": "system", "content": PATCH_PROMPT},
        {"role": "user", "content": f"## 当前大纲\n{tree_text}\n\n## 修改指令\n{user_request}"},
    ]

    logger.info(
        "[Step 8] Patch Prompt:\n[SYSTEM]\n%s\n\n[USER]\n%s",
        messages[0]["content"],
        messages[1]["content"],
    )

    print("\n[Step 8] LLM Patch 流式输出 ↓", flush=True)
    print("-" * 50, flush=True)
    full_text = ""
    async for chunk in llm.complete_stream(messages):
        print(chunk, end="", flush=True)
        full_text += chunk
    print("\n" + "-" * 50, flush=True)
    logger.info("[Step 8] LLM 完整输出:\n%s", full_text)

    raw = LLMService._parse_json(full_text)
    ops: list[dict] = raw if isinstance(raw, list) else [raw]
    logger.info("[Step 8] 解析 patch 操作 (%d 条): %s", len(ops), ops)
    return ops


# ─────────────────────────────────────────────────────────────
# Step 9: 将 patch 操作应用到子树
# ─────────────────────────────────────────────────────────────

def step9_apply_patch(outline_tree: dict, ops: list[dict]) -> dict:
    """
    将 patch 操作列表应用到大纲子树，返回修改后的新树（deepcopy，原树不变）。

    支持操作:
        delete    — 从树中删除目标节点及其子节点
        set_param — 在目标节点上添加/更新参数标注
    """
    tree = copy.deepcopy(outline_tree)
    for op in ops:
        node_id = op.get("node_id", "")
        if op["op"] == "delete":
            removed = _delete_node(tree, node_id)
            if removed:
                logger.info("[Step 9] delete: 已删除节点 %s", node_id)
            else:
                logger.warning("[Step 9] delete: 未找到节点 %s", node_id)
        elif op["op"] == "set_param":
            found = _set_param_node(
                tree, node_id, op["key"], op["value"], op.get("unit", "")
            )
            if found:
                logger.info(
                    "[Step 9] set_param: 节点 %s → %s=%s%s",
                    node_id, op["key"], op["value"], op.get("unit", ""),
                )
            else:
                logger.warning("[Step 9] set_param: 未找到节点 %s", node_id)
        else:
            logger.warning("[Step 9] 未知操作: %s", op["op"])
    return tree


# ─────────────────────────────────────────────────────────────
# 内部工具函数
# ─────────────────────────────────────────────────────────────

def _candidates_to_tree_text(candidates: list[dict]) -> str:
    """
    将 FAISS 候选节点渲染为带层级缩进的树状文本。
    ★ 标记命中节点，无★ 的中间节点仅提供路径上下文。
    """
    hit_ids = {c["id"] for c in candidates}

    # 从候选节点的 path 字符串中还原 name→id 映射（仅候选节点本身有 id）
    # 中间祖先节点只需要显示 name，不强求 id
    id_by_name: dict[str, str] = {c["name"]: c["id"] for c in candidates}
    level_by_name: dict[str, int] = {c["name"]: c["level"] for c in candidates}

    # 用有序 dict 构建树：node_name -> {children: [name,...], level: int, id: str|None}
    tree: dict[str, dict] = {}
    roots: list[str] = []

    for c in candidates:
        parts = [p.strip() for p in c["path"].split(">")]
        for i, name in enumerate(parts):
            if name not in tree:
                tree[name] = {
                    "id": id_by_name.get(name),
                    "level": level_by_name.get(name, i + 1),
                    "children": [],
                }
            if i > 0:
                parent_name = parts[i - 1]
                if name not in tree[parent_name]["children"]:
                    tree[parent_name]["children"].append(name)
            elif name not in roots:
                roots.append(name)

    lines: list[str] = []

    def render(name: str, depth: int) -> None:
        node = tree[name]
        indent = "  " * depth
        id_str = f" {node['id']}" if node["id"] else ""
        marker = " ★" if node["id"] in hit_ids else ""
        lines.append(f"{indent}[L{node['level']}{id_str}] {name}{marker}")
        for child in node["children"]:
            render(child, depth + 1)

    for root in roots:
        render(root, 0)

    return "\n".join(lines)


def _build_subtree(node_id: str, nodes_dict: dict, children_map: dict) -> dict:
    node = dict(nodes_dict[node_id])
    node["children"] = [
        _build_subtree(cid, nodes_dict, children_map)
        for cid in children_map.get(node_id, [])
    ]
    return node


def _count_nodes(node: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


def _tree_to_id_text(node: dict, depth: int = 0) -> str:
    """将子树渲染为带 id 的缩进文本，供 LLM patch 时引用。"""
    indent = "  " * depth
    line = f"{indent}[id={node['id']} L{node['level']}] {node['name']}"
    child_lines = [_tree_to_id_text(c, depth + 1) for c in node.get("children", [])]
    return "\n".join([line] + child_lines)


def _render_node(node: dict, heading_level: int) -> list[str]:
    prefix = "#" * min(heading_level, 6)
    block = f"{prefix} {node['name']}"
    if node.get("description"):
        block += f"\n\n{node['description']}"
    if node.get("params"):
        param_str = "、".join(
            f"{k}: {v['value']}{v['unit']}" for k, v in node["params"].items()
        )
        block += f"\n\n> 参数设置 — {param_str}"
    blocks = [block]
    for child in node.get("children", []):
        blocks.extend(_render_node(child, heading_level + 1))
    return blocks


def _delete_node(tree: dict, node_id: str) -> bool:
    """从树中删除 node_id 对应的节点（含子树），返回是否找到。"""
    children = tree.get("children", [])
    for i, child in enumerate(children):
        if child["id"] == node_id:
            children.pop(i)
            return True
        if _delete_node(child, node_id):
            return True
    return False


def _set_param_node(tree: dict, node_id: str, key: str, value, unit: str) -> bool:
    """在 node_id 节点上设置参数标注，返回是否找到。"""
    if tree["id"] == node_id:
        tree.setdefault("params", {})[key] = {"value": value, "unit": unit}
        return True
    for child in tree.get("children", []):
        if _set_param_node(child, node_id, key, value, unit):
            return True
    return False


# ─────────────────────────────────────────────────────────────
# Step 10: 大纲树导出为 JSON
# ─────────────────────────────────────────────────────────────

def step10_export_json(outline_tree: dict) -> dict:
    """
    将大纲树导出为干净的 JSON 结构（去除 score、keywords 等内部字段）。
    保留: id, name, level, description, params, children
    params 仅在节点有参数标注时出现。
    """
    _KEEP = {"id", "name", "level", "description", "params"}
    node = {k: v for k, v in outline_tree.items() if k in _KEEP}
    node["children"] = [step10_export_json(c) for c in outline_tree.get("children", [])]
    return node




async def main(question: str) -> tuple[dict, str]:
    """
    完整工作流：输入用户问题，返回 (outline_tree, markdown_outline)。

    Args:
        question: 用户的分析问题，例如 "分析政企OTN升级"
    Returns:
        outline_tree : 大纲的内存树结构（可传入 modify() 做后续修剪）
        outline      : Markdown 格式的报告大纲字符串
    """
    faiss_svc, nodes_dict, children_map = step1_load_resources()
    query_embedding = await step2_embed_query(question)
    hits = step3_search_nodes(query_embedding, faiss_svc)
    if not hits:
        err = f"未找到与'{question}'相关的知识节点，请检查索引或降低 FAISS_SCORE_THRESHOLD。"
        return {}, err
    candidates = step4_build_candidate_paths(hits, nodes_dict, children_map)
    anchor = await step5_select_anchor(question, candidates)
    subtree = step6_build_subtree(anchor["selected_id"], nodes_dict, children_map)
    outline = step7_render_outline(subtree)

    # Step 8+9: 从原始问题中提取修改意图并应用（无意图时 ops=[]，树不变）
    ops = await step8_parse_patch(question, subtree)
    if ops:
        subtree = step9_apply_patch(subtree, ops)
        outline = step7_render_outline(subtree)

    return subtree, outline


async def modify(user_request: str, outline_tree: dict) -> tuple[dict, str]:
    """
    对已有大纲执行修剪：LLM 解析指令 → 应用 patch → 重新渲染。

    Args:
        user_request : 自然语言修改指令，例如 "删除企业分布分析，低阶交叉利用率阈值设为70%"
        outline_tree : main() 返回的 outline_tree（或上一轮 modify() 返回的树）
    Returns:
        new_tree  : 修改后的树（可继续传入下一轮 modify()）
        outline   : 新的 Markdown 大纲
    """
    ops = await step8_parse_patch(user_request, outline_tree)
    new_tree = step9_apply_patch(outline_tree, ops)
    outline = step7_render_outline(new_tree)
    return new_tree, outline


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
    _print_outline(outline, step10_export_json(tree))

    # 多轮修改交互循环
    while True:
        try:
            cmd = input("\n修改指令（直接回车退出）> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            break
        tree, outline = asyncio.run(modify(cmd, tree))
        _print_outline(outline, step10_export_json(tree))
