"""
agent.py — Agent1: expert knowledge → outline template pipeline.

Same ReAct loop structure as Agent2, different tools and memory.
chat_stream() yields typed events:
  {"type": "step",    "name": str, "status": "running"|"done"}
  {"type": "outline", "markdown": str}     ← emitted immediately by tool
  {"type": "delta",   "text": str}         ← delta analysis result
  {"type": "saved",   "scene_name": str, "path": str}
  {"type": "text",    "chunk": str}        ← LLM brief acknowledgment
  {"type": "done",    "seconds": float}
  {"type": "error",   "message": str}
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

_AGENT1_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT1_DIR)
_WF1_DIR = os.path.join(_BACKEND_DIR, "case_workflow_1")

for _p in [_BACKEND_DIR, _WF1_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from services.llm_service import LLMService
from agent1.memory import Agent1Memory
from agent1.tools import TOOLS, HANDLERS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT1_DIR) / "prompt.txt").read_text(encoding="utf-8")
_MAX_TOOL_ROUNDS = 6


class Agent1:
    def __init__(self) -> None:
        self.memory = Agent1Memory()

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        self.memory.add_message({"role": "user", "content": user_message})
        logger.info("[Agent1] user: %r", user_message)
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

                    # Emit typed events based on which tool ran
                    if name == "analyze_expert_knowledge" and result_dict.get("outline_tree"):
                        yield {"type": "outline", "markdown": result_dict["markdown"]}
                        if result_dict.get("new_nodes"):
                            yield {"type": "new_nodes", "nodes": result_dict["new_nodes"]}

                    elif name == "analyze_kb_delta" and result_dict.get("delta_text"):
                        yield {"type": "delta", "text": result_dict["delta_text"]}

                    elif name == "modify_outline" and result_dict.get("outline_tree"):
                        yield {"type": "outline", "markdown": result_dict["markdown"]}

                    elif name == "save_outline_template" and result_dict.get("status") == "success":
                        yield {"type": "saved",
                               "scene_name": result_dict["scene_name"],
                               "path": result_dict["path"]}

                    yield {"type": "step", "name": name, "status": "done"}
                    self.memory.add_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": llm_str,
                    })
                continue

            text = (msg.content or "").strip()
            if text:
                yield {"type": "text", "chunk": text}
            yield {"type": "done", "seconds": round(time.time() - t0, 1)}
            return

        yield {"type": "error", "message": "工具调用次数超限，请重试"}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}

    async def _call_llm(self):
        llm = LLMService.from_env()
        messages = self.memory.build_messages(_SYSTEM_PROMPT)
        logger.info("[Agent1] LLM call: %d messages", len(messages))
        return await llm._client.chat.completions.create(
            model=llm.default_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=llm._temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _execute_tool(self, tool_call) -> tuple[dict, str]:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return {}, f"工具参数解析失败: {e}"

        handler = HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"

        try:
            return await handler(args, self.memory)
        except Exception as e:
            logger.exception("[Agent1] tool %r failed: %s", name, e)
            return {}, f"工具执行失败: {e}"
