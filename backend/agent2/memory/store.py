"""
store.py — AgentMemory: outline state + conversation history management.

Separates two concerns:
  1. outline_tree (JSON, source of truth) — never put in message history
  2. conversation history — kept compact (tool results use md_with_ids, not full markdown)

Before each LLM call, build_messages() injects the current md_with_ids as a
trailing system message so the LLM always sees up-to-date node IDs for referencing.
"""


class AgentMemory:
    def __init__(self) -> None:
        # Outline state — written by tool handlers, never serialized into messages
        self.outline_tree: dict = {}
        self.markdown: str = ""       # human-readable (for outline SSE event)
        self.md_with_ids: str = ""    # LLM-readable (injected into context)

        # Conversation turns (user / assistant / tool messages only, no system)
        self._history: list[dict] = []

    # ── Outline state ─────────────────────────────────────────

    @property
    def has_outline(self) -> bool:
        return bool(self.outline_tree)

    def set_outline(self, outline_tree: dict, markdown: str, md_with_ids: str) -> None:
        self.outline_tree = outline_tree
        self.markdown = markdown
        self.md_with_ids = md_with_ids

    def clear_outline(self) -> None:
        self.outline_tree = {}
        self.markdown = ""
        self.md_with_ids = ""

    # ── History management ────────────────────────────────────

    def add_message(self, msg: dict) -> None:
        self._history.append(msg)

    def reset(self) -> None:
        self._history.clear()
        self.clear_outline()

    # ── Context building ──────────────────────────────────────

    def build_messages(self, system_prompt: str) -> list[dict]:
        """
        Assemble the full messages list for an LLM call:
          [system_prompt] + history + [current outline context if exists]

        The outline is injected as a trailing system message so the LLM always
        sees the latest node IDs without the outline growing inside history.
        """
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history)
        if self.has_outline:
            messages.append({
                "role": "system",
                "content": f"## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}",
            })
        return messages
