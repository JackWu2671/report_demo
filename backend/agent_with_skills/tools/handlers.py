"""
handlers.py — AgentWithSkills 工具分发层。

合并了原 agent1 和 agent2 的 handler，统一使用 AgentWithSkillsMemory。
每个 handler 返回 (result_dict, llm_str)。
"""

import logging

from agent_with_skills.memory import AgentWithSkillsMemory
from tools.search_template import search_outline_templates, load_template_outline
from tools.build_outline_from_anchor import build_outline_from_anchor
from tools.set_outline_from_markdown import set_outline_from_markdown
from tools.set_scene_metadata import set_scene_metadata
from tools.save_template import save_outline_template
from tools.graph_manage import graph_manage
from tools.shared_tools import handle_search_graph_tree, handle_modify_outline

logger = logging.getLogger(__name__)


async def handle_search_outline_templates(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
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


async def handle_load_template_outline(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = load_template_outline(args.get("template_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
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


async def handle_build_outline_from_anchor(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = await build_outline_from_anchor(args.get("anchor_id", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = f"[build_outline_from_anchor] status=success\n\n当前大纲：\n{result['md_with_ids']}"
    else:
        llm_str = f"[build_outline_from_anchor] status=not_found  message={result['message']}"
    return result, llm_str


async def handle_set_outline_from_markdown(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = await set_outline_from_markdown(md_with_ids=args.get("md_with_ids", ""))
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
        llm_str = (
            f"[set_outline_from_markdown] status=success\n"
            f"大纲已渲染。接下来请调用 set_scene_metadata 填写场景元数据。\n\n"
            f"当前大纲：\n{result['md_with_ids']}"
        )
    else:
        llm_str = f"[set_outline_from_markdown] status=error  message={result['message']}"
    return result, llm_str


async def handle_set_scene_metadata(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = await set_scene_metadata(
        scene_name=args.get("scene_name", ""),
        summary=args.get("summary", ""),
        keywords=args.get("keywords", []),
        usage_conditions=args.get("usage_conditions", ""),
    )
    if result["status"] == "success":
        memory.set_extraction({
            "scene_name": result["scene_name"],
            "summary": result["summary"],
            "keywords": result["keywords"],
            "usage_conditions": result["usage_conditions"],
        })
        llm_str = (
            f"[set_scene_metadata] status=success\n"
            f"场景：{result['scene_name']}\n"
            f"关键词：{', '.join(result['keywords'])}\n"
            f"适用条件：{result['usage_conditions']}\n"
            f"元数据已记录，请询问专家是否需要修改或保存。"
        )
    else:
        llm_str = f"[set_scene_metadata] status=error  message={result['message']}"
    return result, llm_str


async def handle_save_outline_template(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = await save_outline_template(memory.extraction, memory.outline_tree)
    if result["status"] == "success":
        llm_str = (
            f"[save_outline_template] status=success\n"
            f"场景: {result['scene_name']}\n"
            f"路径: {result['path']}\n"
            f"template_id: {result['template_id']}\n\n"
            f"请立即调用 read_skill(\"graph-fusion\") 加载知识图谱融合工作流。"
        )
    else:
        llm_str = f"[save_outline_template] status=error  message={result['message']}"
    return result, llm_str


async def handle_graph_manage(args: dict, memory: AgentWithSkillsMemory) -> tuple[dict, str]:
    result = await graph_manage(
        template_id=args.get("template_id", ""),
        add_nodes=args.get("add_nodes", []),
        enrich_nodes=args.get("enrich_nodes", []),
    )
    status = result["status"]
    if status == "success":
        llm_str = (
            f"[graph_manage] status=success\n"
            f"新增节点: {result['added_nodes']}\n"
            f"丰富描述: {result['enriched_nodes']}\n"
            f"说明: {result['message']}"
        )
    else:
        llm_str = f"[graph_manage] status=error  {result['message']}"
    return result, llm_str


HANDLERS: dict = {
    "search_outline_templates":  handle_search_outline_templates,
    "load_template_outline":     handle_load_template_outline,
    "build_outline_from_anchor": handle_build_outline_from_anchor,
    "search_graph_tree":         handle_search_graph_tree,
    "modify_outline":            handle_modify_outline,
    "set_outline_from_markdown": handle_set_outline_from_markdown,
    "set_scene_metadata":        handle_set_scene_metadata,
    "save_outline_template":     handle_save_outline_template,
    "graph_manage":              handle_graph_manage,
}
