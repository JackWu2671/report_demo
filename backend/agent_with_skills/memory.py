import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from memory.store import AgentMemory


class AgentWithSkillsMemory(AgentMemory):
    """
    AgentMemory 扩展：增加 extraction 字段，用于 consolidate-expert skill。

    extraction: {scene_name, keywords, summary, usage_conditions}
    由 set_scene_metadata 写入，save_outline_template 读取。
    """

    def __init__(self) -> None:
        super().__init__()
        self.extraction: dict = {}

    @property
    def has_extraction(self) -> bool:
        return bool(self.extraction)

    def set_extraction(self, extraction: dict) -> None:
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
            content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.outline_yaml}"

        return [{"role": "system", "content": content}, *self._history]
