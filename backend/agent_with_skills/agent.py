"""
agent.py — 单一 agent，hermes-agent 风格三级渐进式 skill 加载。

启动时注入 Level 0 skill 列表（只有 name/description/category，~极少 token）。
LLM 按需调用 skill_view 加载完整 SOP（Level 1），或加载支持文件（Level 2）。
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
from agent_with_skills.skill_loader import discover_skills, skill_view as _skill_view

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(_AGENT_DIR) / "skills"
_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "prompt.txt").read_text(encoding="utf-8")
_MAX_ROUNDS = 8

# ── Skill 元工具定义 ──────────────────────────────────────────────

_SKILLS_LIST_TOOL = {
    "type": "function",
    "function": {
        "name": "skills_list",
        "description": "列出所有可用 skill 的名称、描述和分类（Level 0）。不确定有哪些能力时调用。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SKILL_VIEW_TOOL = {
    "type": "function",
    "function": {
        "name": "skill_view",
        "description": (
            "加载指定 skill 的完整 SOP（Level 1），或其内部支持文件（Level 2）。"
            "决定使用某个 skill 前必须先加载其 SOP，已加载的 skill 无需重复加载。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "skill 名称，如 generate-outline"},
                "path": {
                    "type": "string",
                    "description": "可选。skill 文件夹内的支持文件路径，如 references/faq.md（Level 2）",
                },
            },
            "required": ["name"],
        },
    },
}

TOOLS = [_SKILLS_LIST_TOOL, _SKILL_VIEW_TOOL] + _OUTLINE_TOOLS

_SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
调用工具时，遇到复杂任务先用 skill_view(<skill_name>) 阅读工作流指导。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


class AgentWithSkills:
    def __init__(self) -> None:
        self._skill_meta = discover_skills(_SKILLS_DIR)
        self._loaded: set[str] = set()  # 已加载 SOP 的 skill，避免重复注入
        self.memory = AgentMemory()
        logger.info(
            "[AgentWithSkills] discovered: %s", [m["name"] for m in self._skill_meta]
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
        lines = []
        for m in self._skill_meta:
            cat = f"[{m['category']}] " if m.get("category") else ""
            lines.append(f"- {cat}{m['name']}: {m.get('description', '')}")
        skill_entries = "\n".join(lines)
        skill_block = _SKILL_SYSTEM_TEMPLATE.format(skill_entries=skill_entries)
        return f"{_SYSTEM_PROMPT}\n\n{skill_block}"

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

        if name == "skills_list":
            return self._handle_skills_list()

        if name == "skill_view":
            return self._handle_skill_view(args)

        handler = _OUTLINE_HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"
        try:
            return await handler(args, self.memory)
        except Exception as e:
            logger.exception("[AgentWithSkills] tool %r failed", name)
            return {}, f"工具执行失败: {e}"

    def _handle_skills_list(self) -> tuple[dict, str]:
        items = [
            {"name": m["name"], "description": m.get("description", ""), "category": m.get("category", "")}
            for m in self._skill_meta
        ]
        return {}, f"[skills_list]\n{json.dumps(items, ensure_ascii=False, indent=2)}"

    def _handle_skill_view(self, args: dict) -> tuple[dict, str]:
        skill_name = args.get("name", "")
        ref_path = args.get("path")
        meta = next((m for m in self._skill_meta if m["name"] == skill_name), None)
        if meta is None:
            return {}, f"[skill_view] skill 不存在: {skill_name}"
        if not ref_path and skill_name in self._loaded:
            return {}, f"[skill_view] {skill_name} SOP 已加载，请直接按流程操作"
        content = _skill_view(meta["_path"], ref_path)
        if not ref_path:
            self._loaded.add(skill_name)
            logger.info("[AgentWithSkills] loaded skill SOP: %s", skill_name)
        level = "2" if ref_path else "1"
        label = f"{skill_name}/{ref_path}" if ref_path else skill_name
        return {}, f"[skill_view Level {level}] {label}:\n\n{content}"
