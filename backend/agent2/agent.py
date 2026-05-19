"""
agent.py — Agent2：报告大纲生成流程。

chat_stream() 产出的事件类型：
  {"type": "step",    "name": str, "call_id": str, "status": "running"|"done", "args": dict, "result": str, "detail": str}
  {"type": "outline", "markdown": str, "md_with_ids": str, "outline_tree": dict}
  {"type": "text",    "chunk": str}
  {"type": "done",    "seconds": float}
  {"type": "error",   "message": str}

大纲始终通过 outline 事件推送，LLM 的文字回复保持在 1-2 句话。
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

_SYSTEM_PROMPT = (Path(_AGENT2_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_MAX_TOOL_ROUNDS = 10


class Agent2:
    """
    报告大纲生成 agent（有状态，多轮对话）。

    状态保存在 self.memory（AgentMemory）：
      - outline_tree / markdown / md_with_ids：当前大纲
      - _history：对话历史（工具结果以紧凑字符串存入，不存原始 markdown）

    重置会话：调用 agent.memory.reset() 或新建 Agent2()。
    """

    def __init__(self) -> None:
        self.memory = AgentMemory()
        self.system_prompt = _SYSTEM_PROMPT

    # ── Public ────────────────────────────────────────────────────

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        处理一轮用户输入，以事件流形式 yield 结果。

        大纲事件在工具返回后立即推送，无需等待 LLM 文字回复。
        LLM 的文字回复应保持在 1-2 句话（由 system prompt 约束）。
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
                    try:
                        args_for_display = json.loads(tc.function.arguments)
                    except Exception:
                        args_for_display = {}
                    call_id = tc.id
                    yield {"type": "step", "name": name, "status": "running",
                           "call_id": call_id, "args": args_for_display}

                    result_dict, llm_str = await self._execute_tool(tc)

                    # 大纲事件立即推送，无需等待 LLM 回复
                    if result_dict.get("outline_tree"):
                        yield {"type": "outline",
                               "markdown": result_dict["markdown"],
                               "md_with_ids": result_dict["md_with_ids"],
                               "outline_tree": result_dict["outline_tree"]}

                    # 模板命中，等待用户确认是否采用
                    if result_dict.get("status") == "pending_confirm":
                        yield {
                            "type": "confirm",
                            "options": ["使用此模板", "重新从知识库生成"],
                        }

                    yield {"type": "step", "name": name, "status": "done",
                           "call_id": call_id,
                           "result": _result_display(name, result_dict),
                           "detail": llm_str}

                    self.memory.add_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": llm_str,
                    })
                continue  # 让 LLM 处理工具结果后继续

            # LLM 给出文字回复，结束本轮
            text = (msg.content or "").strip()
            if text:
                yield {"type": "text", "chunk": text}

            yield {"type": "done", "seconds": round(time.time() - t0, 1)}
            return

        yield {"type": "error", "message": "工具调用次数超限，请重试"}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}

    # ── Internal ──────────────────────────────────────────────────

    async def _call_llm(self):
        """将当前大纲注入 system prompt 后调用 LLM。"""
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
        """执行单条工具调用，返回 (result_dict, llm_str)。"""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error("[Agent2] args parse error: %s", e)
            return {}, f"工具参数解析失败: {e}"

        logger.info("[Agent2] tool_call: %s args=%s", name, json.dumps(args, ensure_ascii=False))

        handler = HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"

        try:
            result_dict, llm_str = await handler(args, self.memory)
        except Exception as e:
            logger.exception("[Agent2] tool %r failed: %s", name, e)
            return {}, f"工具执行失败: {e}"

        return result_dict, llm_str


# ── 展示辅助 ──────────────────────────────────────────────────────

def _count_nodes(tree: dict) -> int:
    return 1 + sum(_count_nodes(c) for c in tree.get("children", []))


def _result_display(name: str, result: dict) -> str:
    """将工具结果转为前端步骤面板显示的单行摘要。"""
    status = result.get("status", "?")
    if name == "search_outline_templates":
        n = len(result.get("candidates", []))
        return f"找到 {n} 个候选模板" if status == "found" else f"未找到：{result.get('reason', '')}"
    if name == "load_template_outline":
        if status in ("success", "pending_confirm"):
            return f"已加载：{result.get('scene_name', '')}，等待用户确认"
        return f"未找到：{result.get('reason', '')}"
    if name == "search_graph_tree":
        if status == "success":
            lines = [l for l in result.get("tree_text", "").splitlines() if l.strip()]
            return f"返回 {len(lines)} 个节点"
        return f"未找到：{result.get('message', '')}"
    if name == "build_outline_from_anchor":
        if status == "success":
            tree = result.get("outline_tree", {})
            return f"根节点：{tree.get('name', '')}，共 {_count_nodes(tree)} 个节点"
        return f"失败：{result.get('message', '')}"
    if name == "modify_outline":
        if status == "success":
            ops = result.get("ops", [])
            return f"{len(ops)} 个操作：{', '.join(op.get('op', '?') for op in ops)}"
        return f"失败：{result.get('message', '')}"
    return status
