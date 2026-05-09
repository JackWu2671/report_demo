"""
agent.py — 单一 agent，hermes-agent 风格三级渐进式 skill 加载。

启动时注入 Level 0 skill 列表（只有 name/description/category，~极少 token）。
LLM 按需调用 read_skill 加载完整 SOP（Level 1），或加载支持文件（Level 2）。

支持两个 skill：
  generate-report    — 面向普通用户，生成分析报告
  consolidate-expert — 面向专家，沉淀知识为可复用模板
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

from agent1.memory import Agent1Memory
from services.llm_service import LLMService
from agent1.tools.definitions import TOOLS as _AGENT1_TOOLS
from agent1.tools.handlers import HANDLERS as _AGENT1_HANDLERS
from agent2.tools.definitions import TOOLS as _AGENT2_TOOLS
from agent2.tools.handlers import HANDLERS as _AGENT2_HANDLERS
from agent_with_skills.skill_registry import SkillRegistry
from tools.shared_tools import SKILLS_LIST_TOOL, READ_SKILL_TOOL

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(_BACKEND_DIR) / "skills"
_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "prompt.txt").read_text(encoding="utf-8")
_MAX_ROUNDS = 8

# ── 合并业务工具（agent2 优先，agent1 补充独有工具，modify_outline 去重）──────
_business_tools: dict[str, dict] = {t["function"]["name"]: t for t in _AGENT2_TOOLS}
_business_tools.update({t["function"]["name"]: t for t in _AGENT1_TOOLS})
_BUSINESS_TOOLS = list(_business_tools.values())

# agent1 handlers 覆盖 agent2 同名 handler（modify_outline 两者逻辑一致）
_BUSINESS_HANDLERS = {**_AGENT2_HANDLERS, **_AGENT1_HANDLERS}

TOOLS = [SKILLS_LIST_TOOL, READ_SKILL_TOOL] + _BUSINESS_TOOLS

_SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
调用工具时，遇到复杂任务先用 read_skill(<skill_name>) 阅读工作流指导。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


class AgentWithSkills:
    """
    渐进式 skill 加载的单一 agent（有状态，多轮对话）。

    启动时仅将 skill 的 name/description 注入 system prompt（Level 0，极少 token）。
    LLM 按需调用 read_skill 加载完整 SOP（Level 1），或 skill 内支持文件（Level 2）。
    业务工具来自 agent1 + agent2 的合集，通过 _BUSINESS_HANDLERS 分发。
    状态使用 Agent1Memory（agent1/agent2 所需字段的超集）。
    """

    def __init__(self) -> None:
        self.registry = SkillRegistry(_SKILLS_DIR)
        self._loaded: set[str] = set()  # 已注入 context 的 skill SOP，避免重复加载
        self.memory = Agent1Memory()    # 超集，兼容两个 skill 所需的所有状态字段

    # ── Public ────────────────────────────────────────────────────

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        处理一轮用户输入，以事件流形式 yield 结果。

        LLM 遇到复杂任务时会先调 read_skill 加载 SOP，再执行业务工具。
        大纲、元数据等事件在工具返回后立即推送，无需等待 LLM 文字回复。
        """
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
                    logger.info("[AgentWithSkills] LLM decided: %s(%s)", name, tc.function.arguments)
                    yield {"type": "step", "name": name, "status": "running"}

                    result_dict, llm_str = await self._execute_tool(tc)

                    # generate-report 事件
                    if result_dict.get("outline_tree"):
                        yield {
                            "type": "outline",
                            "markdown": result_dict["markdown"],
                            "md_with_ids": result_dict["md_with_ids"],
                            "outline_tree": result_dict["outline_tree"],
                        }
                    # consolidate-expert: 场景元数据确认后推送给前端
                    if name == "set_scene_metadata" and result_dict.get("status") == "success":
                        yield {
                            "type": "extraction",
                            "scene_name": result_dict.get("scene_name", ""),
                            "keywords": result_dict.get("keywords", []),
                            "summary": result_dict.get("summary", ""),
                        }
                    if name == "save_outline_template" and result_dict.get("status") == "success":
                        yield {"type": "saved",
                               "scene_name": result_dict["scene_name"],
                               "path": result_dict["path"]}

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
        """重置会话状态，清空对话历史、大纲和已加载的 skill SOP。"""
        self.memory.reset()
        self._loaded.clear()

    # ── Internal ──────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        """将 Level 0 skill 列表拼入 system prompt，每次 LLM 调用前动态构建。"""
        lines = []
        for m in self.registry.list_all():
            cat = f"[{m['category']}] " if m.get("category") else ""
            lines.append(f"- {cat}{m['name']}: {m.get('description', '')}")
        skill_entries = "\n".join(lines)
        skill_block = _SKILL_SYSTEM_TEMPLATE.format(skill_entries=skill_entries)
        return f"{_SYSTEM_PROMPT}\n\n{skill_block}"

    async def _call_llm(self):
        """将当前大纲和 skill 列表注入 system prompt 后调用 LLM。"""
        llm = LLMService.from_env()
        messages = self.memory.build_messages(self._build_system_prompt())
        logger.info(
            "[AgentWithSkills._call_llm] messages=%d\n%s",
            len(messages),
            "\n---\n".join(
                f"[{m['role']}]\n{m['content'] if isinstance(m.get('content'), str) else m.get('content')}"
                for m in messages
            ),
        )
        return await llm._client.chat.completions.create(
            model=llm.default_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=llm._temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _execute_tool(self, tool_call) -> tuple[dict, str]:
        """
        分发工具调用：skill 元工具由本地处理，业务工具转发给 _BUSINESS_HANDLERS。
        返回 (result_dict, llm_str)。
        """
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return {}, f"参数解析失败: {e}"

        if name == "skills_list":
            return self._handle_skills_list()
        if name == "read_skill":
            return self._handle_read_skill(args)

        handler = _BUSINESS_HANDLERS.get(name)
        if handler is None:
            return {}, f"未知工具: {name}"
        try:
            return await handler(args, self.memory)
        except Exception as e:
            logger.exception("[AgentWithSkills] tool %r failed", name)
            return {}, f"工具执行失败: {e}"

    def _handle_skills_list(self) -> tuple[dict, str]:
        """返回所有可用 skill 的 Level 0 元数据列表（name、description、category）。"""
        items = [
            {"name": m["name"], "description": m.get("description", ""), "category": m.get("category", "")}
            for m in self.registry.list_all()
        ]
        return {}, f"[skills_list]\n{json.dumps(items, ensure_ascii=False, indent=2)}"

    def _handle_read_skill(self, args: dict) -> tuple[dict, str]:
        """
        加载 skill SOP 正文（Level 1）或内部支持文件（Level 2）。
        已加载过的 SOP 直接返回提示，不重复注入 context。
        """
        skill_name = args.get("name", "")
        ref_path = args.get("path")
        if self.registry.get(skill_name) is None:
            return {}, f"[read_skill] skill 不存在: {skill_name}"
        if not ref_path and skill_name in self._loaded:
            return {}, f"[read_skill] {skill_name} SOP 已加载，请直接按流程操作"
        content = self.registry.read_sop(skill_name, ref_path)
        if not ref_path:
            self._loaded.add(skill_name)
            logger.info("[AgentWithSkills] loaded skill SOP: %s", skill_name)
        level = "2" if ref_path else "1"
        label = f"{skill_name}/{ref_path}" if ref_path else skill_name
        return {}, f"[read_skill Level {level}] {label}:\n\n{content}"
