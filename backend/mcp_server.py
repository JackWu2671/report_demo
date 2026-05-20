"""
mcp_server.py — 标准 MCP server，将 report_demo 工具暴露给任意 MCP 客户端。

Transport: stdio（MCP 标准传输，所有兼容客户端均支持）
Run:       python -m backend.mcp_server

状态设计：
  outline_tree 由调用方在上下文中跟踪，每次调用时以参数传入。
  工具本身无状态，与 tool_server.py 的设计一致。

Tools exposed:
  search_graph_tree, search_outline_templates, load_template_outline,
  build_outline_from_anchor, modify_outline,
  set_outline_from_markdown, set_scene_metadata, save_outline_template

Resources exposed:
  skill://{skill_name}            → SKILL.md SOP 正文（Level 1）
  skill://{skill_name}/{path}     → skill 文件夹内的支持文件（Level 2）
"""

import json
import logging
import os
import sys
from pathlib import Path

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

from tools.search_graph_tree import search_graph_tree as _search_graph_tree
from tools.modify_outline import modify_outline as _modify_outline
from tools.build_outline_from_anchor import build_outline_from_anchor as _build_outline
from tools.search_template import search_outline_templates as _search_templates
from tools.search_template import load_template_outline as _load_template
from tools.set_outline_from_markdown import set_outline_from_markdown as _set_outline_from_md
from tools.set_scene_metadata import set_scene_metadata as _set_metadata
from tools.save_template import save_outline_template as _save_template
from agent_with_skills.skill_loader import read_skill as _read_skill

_SKILLS_DIR = Path(_BACKEND_DIR) / "skills"

logging.basicConfig(level=logging.WARNING)

mcp = FastMCP("report-demo-看网分析")


# ── Tools ──────────────────────────────────────────────────────────

@mcp.tool()
async def search_graph_tree(question: str) -> str:
    """
    从知识图谱检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、名称、描述）。
    用于了解知识库有哪些可用的 query 节点，以及它们的层级关系。
    """
    result = await _search_graph_tree(question)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def search_outline_templates(question: str, top_k: int = 5) -> str:
    """
    向量检索模板库，返回与需求最相似的候选模板列表（含 id、scene_name、summary、score）。
    有匹配时用 id 调用 load_template_outline；无匹配时调用 search_graph_tree 从知识库构建。
    """
    result = await _search_templates(question, top_k)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def load_template_outline(template_id: str) -> str:
    """
    按模板 id 加载完整大纲。template_id 取自 search_outline_templates 返回的 id 字段。
    返回 outline_tree（JSON）、markdown、md_with_ids，请保存 outline_tree 供后续修改使用。
    """
    result = _load_template(template_id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def build_outline_from_anchor(anchor_id: str) -> str:
    """
    以指定节点为根从知识图谱展开子树，生成初始报告大纲。
    anchor_id 取自 search_graph_tree 返回的树节点 id（如 'L4_001'）。
    返回 outline_tree（JSON）、markdown、md_with_ids，请保存 outline_tree 供后续修改使用。
    """
    result = await _build_outline(anchor_id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def modify_outline(ops: list[dict], outline_tree: dict) -> str:
    """
    对当前报告大纲执行结构化修改操作。
    outline_tree 取自上次 build_outline_from_anchor / load_template_outline / modify_outline 的返回值。
    返回修改后的完整大纲（含新的 outline_tree），请用新的 outline_tree 替换旧值。

    ops 支持的操作：
    - add_node: {op, node_id, parent_id}
    - delete_node: {op, node_id}
    - modify_node_name: {op, node_id, value}
    - modify_node_description: {op, node_id, value}
    - modify_node_condition: {op, node_id, value}
    - keep_only_node: {op, node_id}
    """
    result = await _modify_outline(ops, outline_tree)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def set_outline_from_markdown(md_with_ids: str) -> str:
    """
    将 LLM 自行构造的 md_with_ids 格式大纲文本解析为结构化大纲。
    返回 outline_tree（JSON），请保存供后续修改使用。
    """
    result = await _set_outline_from_md(md_with_ids)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def set_scene_metadata(
    scene_name: str,
    summary: str,
    keywords: list[str],
    usage_conditions: str,
) -> str:
    """
    记录场景元数据（名称、摘要、关键词、适用条件）。
    在 set_outline_from_markdown 之后调用，返回值需传给 save_outline_template。
    """
    result = await _set_metadata(scene_name, summary, keywords, usage_conditions)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def save_outline_template(extraction: dict, outline_tree: dict) -> str:
    """
    将大纲保存为可复用模板。仅在用户明确确认时调用。
    extraction: {scene_name, summary, keywords, usage_conditions}，取自 set_scene_metadata 的返回值。
    outline_tree: 当前大纲树，取自最近一次 build/modify/set 的返回值。
    """
    result = await _save_template(extraction, outline_tree)
    return json.dumps(result, ensure_ascii=False, indent=2)


# ── Resources (Skills) ─────────────────────────────────────────────

@mcp.resource("skill://{skill_name}")
def skill_sop(skill_name: str) -> str:
    """读取指定 skill 的完整 SOP（SKILL.md 正文，不含 frontmatter）。"""
    skill_path = _SKILLS_DIR / skill_name
    if not skill_path.exists():
        return f"Skill '{skill_name}' not found. Available: {[p.name for p in _SKILLS_DIR.iterdir() if p.is_dir()]}"
    return _read_skill(skill_path)


@mcp.resource("skill://{skill_name}/{file_path}")
def skill_file(skill_name: str, file_path: str) -> str:
    """读取 skill 文件夹内的支持文件（Level 2，如 node-text-format.md）。"""
    skill_path = _SKILLS_DIR / skill_name
    if not skill_path.exists():
        return f"Skill '{skill_name}' not found."
    return _read_skill(skill_path, file_path)


if __name__ == "__main__":
    mcp.run()
