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
    """仅做向量检索，返回 top-N 候选模板列表，不经 LLM 判断。"""
    result = await search_outline_templates(args.get("question", ""), args.get("top_k", 5))
    if result["status"] == "found":
        candidates = result["candidates"]
        lines = [
            f"  {i+1}. id={c['id']}  scene_name={c['scene_name']}  score={c['score']}\n"
            f"      summary: {c.get('summary', '')}\n"
            f"      usage_conditions: {c.get('usage_conditions', '')}"
            for i, c in enumerate(candidates)
        ]
        best_id = candidates[0]["id"]

        # 预加载最佳候选的完整大纲，推送到前端预览（不写入 memory）
        preview = load_template_outline(best_id)
        preview_fields = {}
        if preview["status"] == "success":
            preview_fields = {
                "outline_tree": preview["outline_tree"],
                "markdown":     preview["markdown"],
                "md_with_ids":  preview["md_with_ids"],
            }

        llm_str = (
            f"[search_outline_templates] 找到 {len(candidates)} 个候选:\n"
            + "\n".join(lines)
            + f"\n\n已向用户展示模板预览和两个快捷选项：「使用此模板」和「重新从知识库生成」。"
            f"请向用户简要介绍找到的模板并等待其选择。"
            f"若用户选择「使用此模板」，请调用 load_template_outline 加载 id={best_id}（最佳匹配）；"
            f"若用户选择「重新从知识库生成」，请改用 search_graph_tree 检索知识图谱。"
        )
        result = {**result, "status": "pending_confirm", **preview_fields}
    else:
        llm_str = f"[search_outline_templates] status=not_found  reason={result['reason']}"
    return result, llm_str


async def handle_load_template_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    """按模板 id 直接加载指定模板的完整大纲，将大纲写入 memory。"""
    result = load_template_outline(args.get("template_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[load_template_outline] status=success  scene={result['scene_name']}\n"
            f"大纲已加载，请询问用户是否满意或需要调整。\n\n"
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
