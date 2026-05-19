"""
handlers.py — agent2 工具分发层（薄适配层）。

从 backend/tools/ 引入各工具的具体实现，并与 AgentMemory 对接。
每个 handler 返回 (result_dict, llm_str)：
  result_dict — 完整结果（agent 用于触发 outline 事件）
  llm_str     — 写入 LLM 历史的紧凑字符串（只含 md_with_ids，不含完整 markdown）
"""

import logging

from memory.store import AgentMemory
from tools.search_template import search_outline_templates, load_template_outline
from tools.build_outline_from_anchor import build_outline_from_anchor
from tools.shared_tools import handle_search_graph_tree, handle_modify_outline

logger = logging.getLogger(__name__)


async def handle_search_outline_templates(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """仅做向量检索，返回 top-N 候选模板列表，由 LLM 自行判断是否匹配。"""
    result = await search_outline_templates(args.get("question", ""), args.get("top_k", 5))
    if result["status"] == "found":
        candidates = result["candidates"]
        lines = [
            f"  {i+1}. id={c['id']}  scene_name={c['scene_name']}  score={c['score']}\n"
            f"      summary: {c.get('summary', '')}\n"
            f"      usage_conditions: {c.get('usage_conditions', '')}"
            for i, c in enumerate(candidates)
        ]
        llm_str = (
            f"[search_outline_templates] 找到 {len(candidates)} 个候选:\n"
            + "\n".join(lines)
            + "\n\n请根据用户需求与上述候选的 scene_name / summary / usage_conditions 自行判断相关性："
            f"若有高度匹配的模板，调用 load_template_outline 加载；若无匹配，改用 search_graph_tree 从知识库生成。"
        )
    else:
        llm_str = f"[search_outline_templates] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_load_template_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """按模板 id 加载完整大纲，写入 memory，并向用户展示预览和确认选项。"""
    result = load_template_outline(args.get("template_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        # 大纲已在 result 中，agent 循环会推送 outline 事件（预览）
        # 同时触发 confirm 事件，让用户决定是否采用
        result = {**result, "status": "pending_confirm"}
        llm_str = (
            f"[load_template_outline] status=success  scene={result['scene_name']}\n"
            f"模板已加载并展示给用户，同时提供「使用此模板」和「重新从知识库生成」两个选项。"
            f"请向用户简要说明模板内容并等待其选择。\n\n"
            f"当前大纲：\n{result['md_with_ids']}"
        )
    else:
        llm_str = f"[load_template_outline] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_build_outline_from_anchor(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """以锚节点为根展开知识图谱子树，生成初始大纲并写入 memory。"""
    result = await build_outline_from_anchor(args.get("anchor_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = f"[build_outline_from_anchor] status=success\n\n当前大纲：\n{result['md_with_ids']}"
    else:
        llm_str = f"[build_outline_from_anchor] status=not_found  message={result['message']}"
    return result, llm_str


HANDLERS: dict = {
    "search_outline_templates":  handle_search_outline_templates,
    "load_template_outline":     handle_load_template_outline,
    "build_outline_from_anchor": handle_build_outline_from_anchor,
    "search_graph_tree":         handle_search_graph_tree,
    "modify_outline":            handle_modify_outline,
}
