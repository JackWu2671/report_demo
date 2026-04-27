"""
template_saver.py — 将大纲模板保存为 JSON 文件。

保存内容：
  - 场景元数据（来自 Step 1 extraction）：scene_name, keywords, summary, usage_conditions
  - 大纲 JSON 树（从带 [Lx id] 标注的文本解析）
  - 创建时间戳

输出目录: backend/templates/{scene_name}.json
"""

import json
import logging
import os
import re
import sys
from datetime import datetime

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = os.path.join(_BACKEND_DIR, "templates")

# Matches: optional indent, [Lx node_ref] name（：description optional）
_NODE_RE = re.compile(r"^\s*\[L(\d+)\s+([^\]]+)\]\s+(.+)$")


def outline_md_to_json(outline_md: str, nodes_dict: dict) -> list[dict]:
    """
    将带 [Lx id] 标注的大纲文本转换为 JSON 树列表。

    - 已有 id 节点：从 nodes_dict 取 name / description
    - [new] 节点：使用大纲中的内联名称和描述（格式: name：description）
    - children 按 [Lx] 层级嵌套

    Args:
        outline_md : generate_outline() 输出的带 [Lx id] 标注文本
        nodes_dict : {node_id -> node_dict}（含 KB 节点的 name/description）

    Returns:
        顶层节点列表（通常只有一个根节点，但允许多个并列顶层）
    """
    stack: list[tuple[int, dict]] = []  # (level, node_dict)
    roots: list[dict] = []

    for line in outline_md.splitlines():
        m = _NODE_RE.match(line)
        if not m:
            continue

        level = int(m.group(1))
        node_ref = m.group(2).strip()
        raw_text = m.group(3).strip()

        if node_ref.lower() == "new" and "：" in raw_text:
            name, inline_desc = raw_text.split("：", 1)
            name, inline_desc = name.strip(), inline_desc.strip()
        else:
            name, inline_desc = raw_text, ""

        if node_ref.lower() == "new":
            node: dict = {
                "id": "new",
                "level": level,
                "name": name,
                "description": inline_desc,
                "children": [],
            }
        else:
            kb = nodes_dict.get(node_ref, {})
            node = {
                "id": node_ref,
                "level": level,
                "name": kb.get("name", name),
                "description": kb.get("description", ""),
                "children": [],
            }

        while stack and stack[-1][0] >= level:
            stack.pop()

        if stack:
            stack[-1][1]["children"].append(node)
        else:
            roots.append(node)

        stack.append((level, node))

    return roots


def save_template(
    extraction: dict,
    outline_md: str,
    nodes_dict: dict,
) -> str:
    """
    将场景元数据 + 大纲 JSON 树保存到 templates/{scene_name}.json。

    Args:
        extraction : extract_from_expert() 的输出
        outline_md : 带 [Lx id] 标注的大纲文本
        nodes_dict : 知识库节点字典

    Returns:
        保存的文件路径
    """
    scene_name = extraction.get("scene_name", "unnamed")
    outline_nodes = outline_md_to_json(outline_md, nodes_dict)

    template = {
        "scene_name": scene_name,
        "keywords": extraction.get("keywords", []),
        "summary": extraction.get("summary", ""),
        "usage_conditions": extraction.get("usage_conditions", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "outline": outline_nodes[0] if len(outline_nodes) == 1 else outline_nodes,
    }

    os.makedirs(_TEMPLATE_DIR, exist_ok=True)
    file_path = os.path.join(_TEMPLATE_DIR, f"{scene_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    logger.info("[Template] 模板已保存: %s", file_path)
    return file_path
