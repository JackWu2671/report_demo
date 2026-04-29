"""
agent.py — 单一 agent，内置 skill 渐进式加载。

流程：
  1. 启动：discover_skills() 只读 frontmatter，构建 skill 列表注入 system prompt
  2. 用户发消息：LLM 判断意图，决定调用哪个 skill
  3. LLM 调用 load_skill(skill_name)：读取完整 SKILL.md SOP，注入对话历史
  4. LLM 按 SOP 调用工具（search_outline_template / build_outline_from_anchor / modify_outline）
  5. 已加载的 skill 无需重复加载，后续轮次直接调工具
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from memory.store import AgentMemory
from services.llm_service import LLMService
from agent2.tools.definitions import TOOLS as _OUTLINE_TOOLS
from agent2.tools.handlers import HANDLERS as _OUTLINE_HANDLERS
from agent_with_skills.skill_loader import discover_skills, load_skill_content

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(_AGENT_DIR) / "skills"
_MAX_ROUNDS = 8

# load_skill 是元工具，让 LLM 按需拉取 skill SOP
_LOAD_SKILL_TOOL = {
    "type": "function",
    "function": {
        "name": "load_skill",
        "description": (
            "加载指定 skill 的完整 SOP 操作流程。"
            "使用某个 skill 前必须先调用此工具获取流程说明，已加载的 skill 无需重复加载。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "skill 名称，如 generate-outline",
                }
            },
            "required": ["skill_name"],
        },
    },
}

TOOLS = [_LOAD_SKILL_TOOL] + _OUTLINE_TOOLS


class AgentWithSkills:
    def __init__(self) -> None:
        self._skill_meta = discover_skills(_SKILLS_DIR)
        self._loaded: set[str] = set()   # 已加载 SOP 的 skill，避免重复注入
        self.memory = AgentMemory()
        logger.info(
            "[AgentWithSkills] skills: %s", [m["name"] for m in self._skill_meta]
        )

    # ── Public ────────────────────────────────────────────────────

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        self.memory.add_message({"role": "user", "content": user_message})
        t0 = time.time()

        for _ in range(_MAX_ROUNDS):
            response = await self._call_llm()
            choice = response.choices[0]
            msg = choice.message
            self.memory.add_message(msg.model_dump(exclude_none=True))

            if choice.finish_reason == "tool_calls" and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    yield {"type": "step", "name": name, "status": "running"}

                    result_dict, llm_str = await self._execute_tool(tc)

                    if result_dict.get("outline_tree"):
                        yield {
                            "type": "outline",
                            "markdown": result_dict["markdown"],
                            "md_with_ids": result_dict["md_with_ids"],
                            "outline_tree": result_dict["outline_tree"],
                        }
                    if result_dict.get("status") == "pending_confirm":
                        yield {"type": "confirm", "options": ["使用此模板", "重新从知识库生成"]}

                    yield {"type": "step", "name": name, "status": "done"}
                    self.memory.add_message(
                        {"role": "tool", "tool_call_id": tc.id, "content": llm_str}
                    )
                continue

            if msg.content:
                yield {"type": "text", "chunk": msg.content}
            yield {"type": "done", "seconds": round(time.time() - t0, 1)}
            return

        yield {"type": "error", "message": "工具调用次数超限，请重试"}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}

    def reset(self) -> None:
        self.memory.reset()
        self._loaded.clear()

    # ── Internal ──────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        skill_list = "\n".join(
            f"- {m['name']}: {m['description']}" for m in self._skill_meta
        )
        return (
            "你是一个报告生成助手。\n\n"
            "## 可用 Skill\n\n"
            "遇到用户请求时，先判断需要哪个 skill，"
            "调用 load_skill 获取详细 SOP，再按 SOP 步骤调用工具。\n"
            "已加载过的 skill 无需重复加载，直接按流程操作。\n\n"
            f"{skill_list}"
        )

    async def _call_llm(self):
        llm = LLMService.from_env()
        messages = self.memory.build_messages(self._build_system_prompt())
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
            return {}, f"参数解析失败: {e}"

        # 元工具：渐进式加载 skill SOP
        if name == "load_skill":
            return self._handle_load_skill(args)

        # 业务工具：路由到 outline handlers
        handler = _OUTLINE_HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"
        try:
            return await handler(args, self.memory)
        except Exception as e:
            logger.exception("[AgentWithSkills] tool %r failed", name)
            return {}, f"工具执行失败: {e}"

    def _handle_load_skill(self, args: dict) -> tuple[dict, str]:
        skill_name = args.get("skill_name", "")
        meta = next((m for m in self._skill_meta if m["name"] == skill_name), None)
        if meta is None:
            return {}, f"[load_skill] skill 不存在: {skill_name}"
        if skill_name in self._loaded:
            return {}, f"[load_skill] {skill_name} 已加载，请直接按 SOP 操作"
        content = load_skill_content(meta["_path"])
        self._loaded.add(skill_name)
        logger.info("[AgentWithSkills] loaded skill SOP: %s", skill_name)
        return {}, f"[load_skill] {skill_name} SOP 已加载：\n\n{content}"
