"""
memory.py — Agent1Memory: extends AgentMemory with expert-knowledge state.

Extra fields beyond the base class:
  expert_text           : original expert input text (for delta analysis)
  extraction            : {scene_name, keywords, summary, usage_conditions}
  outline_md_annotated  : [id]/[new]-marked Markdown from Step 3
  tree_text             : KB tree text from Step 2 (for delta analysis)
  new_nodes             : [{name, level, parent_id, ...}] from Step 4
  nodes_dict            : KB node dict (needed if save path calls template_saver)

build_messages() injects extraction metadata + new_nodes summary
alongside the current outline, giving the LLM full context.
"""

import os
import sys

_AGENT1_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT1_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from memory.store import AgentMemory


class Agent1Memory(AgentMemory):
    def __init__(self) -> None:
        super().__init__()
        self.expert_text: str = ""
        self.extraction: dict = {}
        self.outline_md_annotated: str = ""
        self.tree_text: str = ""
        self.new_nodes: list = []
        self.nodes_dict: dict = {}

    @property
    def has_analysis(self) -> bool:
        return bool(self.extraction)

    def set_analysis_result(self, result: dict) -> None:
        """Store the full output of analyze_expert_knowledge."""
        self.expert_text = result.get("expert_text", self.expert_text)
        self.extraction = result["extraction"]
        self.outline_md_annotated = result["outline_md_annotated"]
        self.tree_text = result["tree_text"]
        self.new_nodes = result["new_nodes"]
        self.nodes_dict = result.get("nodes_dict", {})
        self.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])

    def reset(self) -> None:
        super().reset()
        self.expert_text = ""
        self.extraction = {}
        self.outline_md_annotated = ""
        self.tree_text = ""
        self.new_nodes = []
        self.nodes_dict = {}

    def build_messages(self, system_prompt: str) -> list[dict]:
        content = system_prompt

        if self.has_analysis:
            meta = self.extraction
            content += (
                f"\n\n## 当前场景元数据\n"
                f"场景名：{meta.get('scene_name', '')}\n"
                f"使用条件：{meta.get('usage_conditions', '')}"
            )

        if self.has_outline:
            content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"

        if self.new_nodes:
            new_list = "\n".join(
                f"- L{n['level']} {n['name']}" for n in self.new_nodes
            )
            content += f"\n\n## [new] 节点（知识库暂未收录，共{len(self.new_nodes)}个）\n{new_list}"

        return [{"role": "system", "content": content}, *self._history]
