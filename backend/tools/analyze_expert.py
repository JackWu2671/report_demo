"""
analyze_expert.py — analyze_expert_knowledge tool implementation (Steps 1-4).

Bundles Steps 1-4 of case_workflow_1:
  Step 1  extract_from_expert   — LLM extracts metadata from expert text
  Step 2  dual_search           — dual FAISS search on keywords + summary
          build_kb_tree_text    — render full KB tree with hit markers
  Step 3  generate_outline      — LLM produces [id]/[new]-annotated Markdown outline
  Step 4  parse_new_nodes       — extract [new] nodes for display

Why bundle 1-4 together instead of exposing each step?
  Steps 2-3 are tightly coupled: search builds tree_text, outline needs tree_text.
  Step 1 result is not useful to show alone. Bundling gives the agent LLM one
  clean result to present: "here is the outline, these are new nodes."

Used by: agent1
"""

import json
import logging
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_TOOLS_DIR)
_WF1_DIR = os.path.join(_BACKEND_DIR, "case_workflow_1")

for _p in [_BACKEND_DIR, _WF1_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services.faiss_service import FAISSService
from extractor import extract_from_expert
from searcher import dual_search, build_kb_tree_text
from outline_gen import generate_outline as wf1_generate_outline
from kb_updater import parse_new_nodes
from template_saver import outline_md_to_json
from outline_utils import to_clean_json, to_markdown, to_markdown_with_ids

logger = logging.getLogger(__name__)

_EXPERT_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")
_DATA_DIR = os.path.join(_BACKEND_DIR, "data")


def _load_kb():
    with open(os.path.join(_EXPERT_DIR, "node.json"), encoding="utf-8") as f:
        nodes_list = json.load(f)
    with open(os.path.join(_EXPERT_DIR, "relation.json"), encoding="utf-8") as f:
        relations_list = json.load(f)

    nodes_dict = {n["id"]: n for n in nodes_list}
    children_map: dict = {}
    for rel in relations_list:
        children_map.setdefault(rel["parent"], []).append(rel["child"])

    faiss_svc = FAISSService(dim=int(os.getenv("EMBEDDING_DIM", 1024)))
    faiss_svc.load(
        os.path.join(_DATA_DIR, "faiss.index"),
        os.path.join(_DATA_DIR, "faiss_id_map.json"),
    )
    return faiss_svc, nodes_dict, children_map


async def analyze_expert_knowledge(expert_text: str) -> dict:
    """
    Run Steps 1-4 of case_workflow_1 and return all outputs.

    Returns:
        {
            "status"               : "success" | "error",
            "extraction"           : dict,    # scene_name, keywords, summary, usage_conditions
            "outline_md_annotated" : str,     # [id]/[new]-marked Markdown (for template_saver)
            "outline_tree"         : dict,    # clean JSON tree (source of truth for modify)
            "markdown"             : str,     # human-readable Markdown
            "md_with_ids"          : str,     # LLM-readable with [Lx id] prefixes
            "new_nodes"            : list,    # [{name, level, parent_id, ...}]
            "tree_text"            : str,     # KB tree used in Step 3 (needed by delta tool)
            "nodes_dict"           : dict,    # KB nodes (needed by save tool)
            "message"              : str,
        }
    """
    logger.info("[Tool:analyze_expert_knowledge] text length=%d", len(expert_text))

    try:
        faiss_svc, nodes_dict, children_map = _load_kb()
    except Exception as e:
        logger.error("[Tool:analyze_expert_knowledge] KB加载失败: %s", e)
        return _error(f"知识库加载失败: {e}")

    # Step 1: metadata extraction
    extraction = await extract_from_expert(expert_text)

    # Step 2: dual search + KB tree
    hits = await dual_search(extraction, faiss_svc, nodes_dict, children_map)
    hit_ids = {h["id"] for h in hits}
    tree_text = build_kb_tree_text(hit_ids, nodes_dict, children_map)
    logger.info("[Tool:analyze_expert_knowledge] 命中 %d 个节点", len(hit_ids))

    # Step 3: [id]/[new]-annotated outline
    outline_md_annotated = await wf1_generate_outline(expert_text, tree_text)

    # Step 4: parse [new] nodes
    new_nodes = parse_new_nodes(outline_md_annotated)

    # Convert to clean tree for display and modification
    roots = outline_md_to_json(outline_md_annotated, nodes_dict)
    raw_tree = roots[0] if len(roots) == 1 else {"id": "root", "name": extraction.get("scene_name", "大纲"), "level": 1, "description": "", "children": roots}
    clean_tree = to_clean_json(raw_tree)

    logger.info(
        "[Tool:analyze_expert_knowledge] 完成 — 场景: %s, [new]节点: %d",
        extraction.get("scene_name"), len(new_nodes),
    )
    return {
        "status": "success",
        "extraction": extraction,
        "outline_md_annotated": outline_md_annotated,
        "outline_tree": clean_tree,
        "markdown": to_markdown(clean_tree),
        "md_with_ids": to_markdown_with_ids(clean_tree),
        "new_nodes": new_nodes,
        "tree_text": tree_text,
        "nodes_dict": nodes_dict,
        "message": "",
    }


def _error(message: str) -> dict:
    return {
        "status": "error", "extraction": {}, "outline_md_annotated": "",
        "outline_tree": {}, "markdown": "", "md_with_ids": "",
        "new_nodes": [], "tree_text": "", "nodes_dict": {}, "message": message,
    }
