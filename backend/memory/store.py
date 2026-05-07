"""
store.py — Shared AgentMemory base class.

Manages outline state (outline_tree / markdown / md_with_ids) and
conversation history, keeping them separate so outline JSON never bloats
the message history.

Subclass this to add agent-specific state (see agent1/memory.py).
"""


class AgentMemory:
    def __init__(self) -> None:
        self.outline_tree: dict = {}
        self.markdown: str = ""
        self.md_with_ids: str = ""
        self.kb_tree_text: str = ""
        self._history: list[dict] = []

    @property
    def has_outline(self) -> bool:
        return bool(self.outline_tree)

    def set_outline(self, outline_tree: dict, markdown: str, md_with_ids: str) -> None:
        self.outline_tree = outline_tree
        self.markdown = markdown
        self.md_with_ids = md_with_ids

    def set_kb_tree(self, tree_text: str) -> None:
        self.kb_tree_text = tree_text

    def clear_outline(self) -> None:
        self.outline_tree = {}
        self.markdown = ""
        self.md_with_ids = ""

    def add_message(self, msg: dict) -> None:
        self._history.append(msg)

    def reset(self) -> None:
        self._history.clear()
        self.clear_outline()

    def build_messages(self, system_prompt: str) -> list[dict]:
        """
        Build LLM message list: system_prompt (+ outline if exists) + history.

        Outline is appended to the system prompt content — not as a separate
        system message — because most models require system messages only at
        the start of the conversation.
        """
        content = system_prompt
        if self.kb_tree_text:
            content += f"\n\n## 知识图谱树（add_node 可用的节点）\n\n{self.kb_tree_text}"
        if self.has_outline:
            content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"
        return [{"role": "system", "content": content}, *self._history]
