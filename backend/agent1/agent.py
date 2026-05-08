"""
agent.py — Agent1：专家知识 → 报告大纲模板沉淀流程。

chat_stream() 产出的事件类型：
  {"type": "step",    "name": str, "call_id": str, "status": "running"|"done", "args": dict, "result": str, "detail": str}
  {"type": "outline", "markdown": str, "md_with_ids": str, "outline_tree": dict}
  {"type": "saved",   "scene_name": str, "path": str}
  {"type": "text",    "chunk": str}
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

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from agent1.memory import Agent1Memory
from agent1.tools import TOOLS, HANDLERS

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT1_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_MAX_TOOL_ROUNDS = 10


class Agent1:
    def __init__(self) -> None:
        self.memory = Agent1Memory()
        self.system_prompt = _SYSTEM_PROMPT

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
                    try:
                        args_for_display = json.loads(tc.function.arguments)
                    except Exception:
                        args_for_display = {}
                    call_id = tc.id
                    yield {"type": "step", "name": name, "status": "running",
                           "call_id": call_id, "args": args_for_display}

                    result_dict, llm_str = await self._execute_tool(tc)

                    if result_dict.get("outline_tree"):
                        yield {"type": "outline",
                               "markdown": result_dict["markdown"],
                               "md_with_ids": result_dict["md_with_ids"],
                               "outline_tree": result_dict["outline_tree"]}

                    if name == "save_outline_template" and result_dict.get("status") == "success":
                        yield {"type": "saved",
                               "scene_name": result_dict["scene_name"],
                               "path": result_dict["path"]}

                    yield {"type": "step", "name": name, "status": "done",
                           "call_id": call_id,
                           "result": _result_display(name, result_dict),
                           "detail": llm_str}

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
        logger.info("[Agent1] tool_call: %s args=%s", name, json.dumps(args, ensure_ascii=False)[:200])
        handler = HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"
        try:
            return await handler(args, self.memory)
        except Exception as e:
            logger.exception("[Agent1] tool %r failed: %s", name, e)
            return {}, f"工具执行失败: {e}"


def _result_display(name: str, result: dict) -> str:
    """生成前端步骤摘要的单行字符串。"""
    status = result.get("status", "?")
    if name == "search_graph_tree":
        if status == "success":
            lines = [l for l in result.get("tree_text", "").splitlines() if l.strip()]
            return f"返回 {len(lines)} 个节点"
        return f"未找到：{result.get('message', '')}"
    if name == "set_outline_from_markdown":
        if status == "success":
            ext = result.get("extraction", {})
            return f"场景：{ext.get('scene_name', '')}，大纲已渲染"
        return f"失败：{result.get('message', '')}"
    if name == "modify_outline":
        if status == "success":
            ops = result.get("ops", [])
            return f"{len(ops)} 个操作：{', '.join(op.get('op', '?') for op in ops)}"
        return f"失败：{result.get('message', '')}"
    if name == "save_outline_template":
        if status == "success":
            return f"已保存：{result.get('scene_name', '')}"
        return f"失败：{result.get('message', '')}"
    return status
