"""
agent.py — Multi-turn tool-calling agent for outline generation and modification.

Agent2 wraps:
  - Conversation history (messages list)
  - Shared state (outline_tree, markdown, md_with_ids)
  - Tool-calling loop: LLM decides which tool to invoke, agent executes it

Usage:
    agent = Agent2()
    response = await agent.chat("帮我分析政企OTN升级")
    response = await agent.chat("删掉设备利用率那一节")
"""

import json
import logging
import os
import sys
from pathlib import Path

_AGENT2_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT2_DIR)

for _p in [_BACKEND_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services.llm_service import LLMService
from .tools import TOOLS, HANDLERS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT2_DIR) / "prompt.txt").read_text(encoding="utf-8")

# Max tool-call iterations per user turn (prevents infinite loops)
_MAX_TOOL_ROUNDS = 5


class Agent2:
    """
    Stateful multi-turn agent.

    Attributes:
        messages      : full conversation history (system + turns)
        state         : shared mutable state across tool calls
            outline_tree : current outline JSON tree (or {} if none)
            markdown     : current human-readable Markdown (or "")
            md_with_ids  : current LLM-readable Markdown with [Lx id] (or "")
    """

    def __init__(self) -> None:
        self.messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        self.state: dict = {
            "outline_tree": {},
            "markdown": "",
            "md_with_ids": "",
        }

    async def chat(self, user_message: str) -> str:
        """
        Send a user message and return the assistant's final text response.

        Internally runs the tool-calling loop until the LLM stops calling tools.

        Args:
            user_message: the user's input text

        Returns:
            assistant's final text reply
        """
        self.messages.append({"role": "user", "content": user_message})
        logger.info("[Agent2] user: %r", user_message)

        for _round in range(_MAX_TOOL_ROUNDS):
            response = await self._call_llm()

            finish_reason = response.choices[0].finish_reason
            msg = response.choices[0].message

            # Always append assistant message to history (exclude None to keep payload clean)
            self.messages.append(msg.model_dump(exclude_none=True))

            if finish_reason == "tool_calls" and msg.tool_calls:
                # Execute each tool call and append results
                for tc in msg.tool_calls:
                    tool_result = await self._execute_tool(tc)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    })
                # Loop again so LLM can respond to tool results
                continue

            # No more tool calls — return text
            final_text = msg.content or ""
            logger.info("[Agent2] assistant: %r (round %d)", final_text[:120], _round + 1)
            return final_text

        # Safety fallback
        logger.warning("[Agent2] 达到最大工具调用轮数 %d", _MAX_TOOL_ROUNDS)
        return "（内部错误：工具调用次数超限，请重试）"

    async def _call_llm(self):
        """Call the LLM with current messages and tools."""
        llm = LLMService.from_env()

        logger.info(
            "[Agent2] LLM 调用 messages=%d条",
            len(self.messages),
        )

        response = await llm._client.chat.completions.create(
            model=llm.default_model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=llm._temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response

    async def _execute_tool(self, tool_call) -> str:
        """Execute a single tool call and return the result string."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error("[Agent2] tool args JSON解析失败: %s | raw=%r", e, tool_call.function.arguments)
            return f"工具参数解析失败: {e}"

        logger.info("[Agent2] 执行工具 %r args=%s", name, args)

        handler = HANDLERS.get(name)
        if handler is None:
            logger.error("[Agent2] 未知工具: %r", name)
            return f"未知工具: {name}"

        try:
            result = await handler(args, self.state)
        except Exception as e:
            logger.exception("[Agent2] 工具 %r 执行失败: %s", name, e)
            result = f"工具执行失败: {e}"

        return result

    @property
    def has_outline(self) -> bool:
        return bool(self.state.get("outline_tree"))
