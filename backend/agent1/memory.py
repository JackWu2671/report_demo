"""
memory.py — Agent1Memory: 在 AgentMemory 基础上扩展专家知识沉淀所需状态。

额外字段：
  extraction : {scene_name, keywords, summary, usage_conditions}
               由 set_outline_from_markdown 写入，save_outline_template 读取

build_messages() 将场景元数据注入 system prompt，让 LLM 在修改大纲时始终知道当前场景。
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
        self.extraction: dict = {}

    @property
    def has_extraction(self) -> bool:
        return bool(self.extraction)

    def set_extraction(self, extraction: dict) -> None:
        """合并写入 extraction，保留已有字段（支持 set_outline / set_metadata 分步调用）。"""
        self.extraction = {**self.extraction, **extraction}

    def reset(self) -> None:
        super().reset()
        self.extraction = {}

    def build_messages(self, system_prompt: str) -> list[dict]:
        content = system_prompt

        if self.has_extraction:
            meta = self.extraction
            content += (
                f"\n\n## 当前场景元数据\n"
                f"场景名：{meta.get('scene_name', '')}\n"
                f"关键词：{', '.join(meta.get('keywords', []))}\n"
                f"使用条件：{meta.get('usage_conditions', '')}"
            )

        if self.has_outline:
            content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"

        return [{"role": "system", "content": content}, *self._history]
