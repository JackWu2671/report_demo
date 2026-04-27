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
          [system_prompt (+ outline if exists)] + history

        The current outline is appended to the system prompt (not as a separate
        system message) because many models require all system messages to be
        at the beginning of the conversation.
        """
        if self.has_outline:
            system_content = (
                f"{system_prompt}\n\n"
                f"## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"
            )
        else:
            system_content = system_prompt

        return [{"role": "system", "content": system_content}, *self._history]
