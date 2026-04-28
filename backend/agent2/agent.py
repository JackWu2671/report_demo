"""
agent.py — Multi-turn tool-calling agent (async generator interface).

Agent2.chat_stream(user_message) is an async generator that yields typed events:

  {"type": "step",    "name": str, "status": "running"|"done"}
  {"type": "outline", "markdown": str}          ← emitted by tool, no LLM streaming
  {"type": "text",    "chunk": str}             ← LLM's brief acknowledgment
  {"type": "done",    "seconds": float}
  {"type": "error",   "message": str}

The outline is always delivered via the "outline" event — the LLM never outputs
the markdown text itself, keeping its reply to 1-2 short sentences.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

_AGENT2_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT2_DIR)

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from memory.store import AgentMemory
from agent2.tools import TOOLS, HANDLERS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT2_DIR) / "prompt.txt").read_text(encoding="utf-8")
_MAX_TOOL_ROUNDS = 6


class Agent2:
    """
    Stateful multi-turn agent.

    State lives in self.memory (AgentMemory):
      - outline_tree / markdown / md_with_ids: current outline
      - _history: conversation turns (compact tool results, no raw markdown)

    To start fresh, call agent.memory.reset() or create a new Agent2().
    """

    def __init__(self) -> None:
        self.memory = AgentMemory()

    # ── Public interface ───────────────────────────────────────

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        Process one user turn, yielding events as they happen.

        Outline events are emitted immediately when a tool produces an outline —
        no LLM streaming delay. The LLM's text reply is a short acknowledgment.
        """
        self.memory.add_message({"role": "user", "content": user_message})
        logger.info("[Agent2] user: %r", user_message)
        t0 = time.time()

        for _round in range(_MAX_TOOL_ROUNDS):
            response = await self._call_llm()
            choice = response.choices[0]
            msg = choice.message

            self.memory.add_message(msg.model_dump(exclude_none=True))

            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    yield {"type": "step", "name": name, "status": "running"}

                    result_dict, llm_str = await self._execute_tool(tc)

                    # Emit outline event immediately — no LLM round-trip needed
                    if result_dict.get("outline_tree"):
                        yield {"type": "outline", "markdown": result_dict["markdown"]}

                    # Template found but needs user confirmation before committing
                    if result_dict.get("status") == "pending_confirm":
                        yield {
                            "type": "confirm",
                            "options": ["使用此模板", "重新从知识库生成"],
                        }

                    yield {"type": "step", "name": name, "status": "done"}

                    self.memory.add_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": llm_str,
                    })
                continue  # let LLM respond to tool results

            # Final text — should be short (LLM instructed to be brief)
            text = (msg.content or "").strip()
            if text:
                yield {"type": "text", "chunk": text}

            yield {"type": "done", "seconds": round(time.time() - t0, 1)}
            return

        yield {"type": "error", "message": "工具调用次数超限，请重试"}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}

    # ── Internal ───────────────────────────────────────────────

    async def _call_llm(self):
        """Build context-injected messages and call the LLM."""
        llm = LLMService.from_env()
        messages = self.memory.build_messages(_SYSTEM_PROMPT)
        logger.info("[Agent2] LLM call: %d messages", len(messages))

        return await llm._client.chat.completions.create(
            model=llm.default_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=llm._temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _execute_tool(self, tool_call) -> tuple[dict, str]:
        """Execute one tool call, return (result_dict, llm_str)."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error("[Agent2] args parse error: %s", e)
            return {}, f"工具参数解析失败: {e}"

        handler = HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"

        try:
            result_dict, llm_str = await handler(args, self.memory)
        except Exception as e:
            logger.exception("[Agent2] tool %r failed: %s", name, e)
            return {}, f"工具执行失败: {e}"

        return result_dict, llm_str
