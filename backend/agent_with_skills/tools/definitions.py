"""
definitions.py — AgentWithSkills 的完整工具列表。

合并了原 agent1 和 agent2 的工具，agent2 工具优先（modify_outline 等共用工具去重）。
"""

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(os.path.dirname(_DIR))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from tools.shared_tools import (
    SEARCH_OUTLINE_TEMPLATES_TOOL,
    LOAD_TEMPLATE_OUTLINE_TOOL,
    BUILD_OUTLINE_FROM_ANCHOR_TOOL,
    SEARCH_GRAPH_TREE_TOOL,
    MODIFY_OUTLINE_TOOL,
    SET_OUTLINE_FROM_MARKDOWN_TOOL,
    SET_SCENE_METADATA_TOOL,
    SAVE_OUTLINE_TEMPLATE_TOOL,
    GRAPH_MANAGE_TOOL,
)

TOOLS: list[dict] = [
    SEARCH_OUTLINE_TEMPLATES_TOOL,
    LOAD_TEMPLATE_OUTLINE_TOOL,
    BUILD_OUTLINE_FROM_ANCHOR_TOOL,
    SEARCH_GRAPH_TREE_TOOL,
    MODIFY_OUTLINE_TOOL,
    SET_OUTLINE_FROM_MARKDOWN_TOOL,
    SET_SCENE_METADATA_TOOL,
    SAVE_OUTLINE_TEMPLATE_TOOL,
    GRAPH_MANAGE_TOOL,
]
