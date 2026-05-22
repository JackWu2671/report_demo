"""
agent.py — 脚本驱动的单一 agent，无业务 tool schema。

LLM 只有两个工具：
  read_skill — 加载 SKILL.md SOP（Level 1）或支持文件（Level 2）
  bash       — 执行 bash 命令（通常是 skills/<name>/scripts/*.py）

所有业务逻辑以 Python 脚本形式存放在 skills/<name>/scripts/，
LLM 通过 SKILL.md 了解脚本 CLI 接口，无需感知任何 JSON tool schema。

状态通过 session 文件（/tmp/report_sessions/{session_id}.json）在
agent 内存与脚本之间同步：
  - bash 调用前：将内存状态写入 session 文件
  - bash 调用后：读回 session 文件，检测变化并推送前端事件
"""

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_AGENT_DIR)
_SKILLS_DIR = Path(_BACKEND_DIR) / "skills"
_SESSION_DIR = Path(os.environ.get("REPORT_SESSION_DIR", "/tmp/report_sessions"))

if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.llm_service import LLMService
from agent_with_skills.memory import AgentWithSkillsMemory
from agent_with_skills.skill_registry import SkillRegistry
from tools.shared_tools import READ_SKILL_TOOL, BASH_TOOL

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_MAX_ROUNDS = 12

TOOLS = [READ_SKILL_TOOL, BASH_TOOL]

_SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
遇到复杂任务先用 read_skill(<skill_name>) 阅读工作流指导，再用 bash 执行对应脚本。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


class AgentWithSkills:
    """
    脚本驱动的单一 agent。LLM 只感知 read_skill + bash 两个工具。
    业务逻辑以 Python 脚本实现，通过 SKILL.md 文档化 CLI 接口。
    """

    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.registry = SkillRegistry(_SKILLS_DIR)
        self._loaded: set[str] = set()
        self.memory = AgentWithSkillsMemory()
        self._system_prompt = self._build_system_prompt()

    # ── Public ────────────────────────────────────────────────────

    async def chat_stream(self, user_message: str) -> AsyncGenerator[dict, None]:
        """处理一轮用户输入，以事件流形式 yield 结果。"""
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
                    call_id = tc.id
                    try:
                        args = json.loads(tc.function.arguments)
                    except Exception:
                        args = {}

                    logger.info("[Agent] tool=%s args=%s", name, tc.function.arguments[:200])
                    yield {"type": "step", "name": name, "status": "running",
                           "call_id": call_id, "args": args}

                    result_dict, llm_str = await self._execute_tool(name, args)

                    # bash 执行后推送检测到的状态变化事件
                    for event in result_dict.get("_events", []):
                        yield event

                    yield {"type": "step", "name": name, "status": "done",
                           "call_id": call_id,
                           "result": _result_display(name, result_dict, llm_str),
                           "detail": llm_str}
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
        _session_path(self.session_id).unlink(missing_ok=True)

    # ── Internal ──────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        lines = []
        for m in self.registry.list_all():
            cat = f"[{m['category']}] " if m.get("category") else ""
            lines.append(f"- {cat}{m['name']}: {m.get('description', '')}")
        skill_block = _SKILL_SYSTEM_TEMPLATE.format(skill_entries="\n".join(lines))
        return f"{_SYSTEM_PROMPT}\n\n{skill_block}"

    async def _call_llm(self):
        llm = LLMService.from_env()
        messages = self.memory.build_messages(self._system_prompt)
        logger.info("[Agent._call_llm] messages=%d", len(messages))
        return await llm._client.chat.completions.create(
            model=llm.default_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=llm._temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    async def _execute_tool(self, name: str, args: dict) -> tuple[dict, str]:
        if name == "read_skill":
            return self._handle_read_skill(args)
        if name == "bash":
            return await self._handle_bash(args.get("command", ""))
        return {}, f"未知工具: {name}"

    def _handle_read_skill(self, args: dict) -> tuple[dict, str]:
        skill_name = args.get("name", "")
        ref_path = args.get("path")
        if self.registry.get(skill_name) is None:
            return {}, f"[read_skill] skill 不存在: {skill_name}"
        if not ref_path and skill_name in self._loaded:
            return {}, f"[read_skill] {skill_name} SOP 已加载，请直接按流程操作"
        content = self.registry.read_sop(skill_name, ref_path)
        if not ref_path:
            self._loaded.add(skill_name)
        level = "2" if ref_path else "1"
        label = f"{skill_name}/{ref_path}" if ref_path else skill_name
        return {}, f"[read_skill Level {level}] {label}:\n\n{content}"

    async def _handle_bash(self, command: str) -> tuple[dict, str]:
        """执行 bash 命令，同步 session 状态，返回事件列表。"""
        # 执行前将内存状态写入 session 文件，供脚本读取
        before = _read_session(self.session_id)
        _write_session(self.session_id, {
            "outline_tree": self.memory.outline_tree or {},
            "md_with_ids":  self.memory.md_with_ids or "",
            "markdown":     self.memory.markdown or "",
            "extraction":   self.memory.extraction or {},
        })

        env = {
            **os.environ,
            "REPORT_SESSION_ID":  self.session_id,
            "REPORT_SESSION_DIR": str(_SESSION_DIR),
            "REPORT_BACKEND_DIR": _BACKEND_DIR,
        }

        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                env=env, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {"_events": []}, "[bash] 执行超时（60s）"

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # 执行后读回 session 文件，检测变化
        after = _read_session(self.session_id)
        events = self._detect_events(before, after)

        llm_output = stdout if stdout else "(no output)"
        if stderr:
            llm_output += f"\n[stderr]\n{stderr}"
        if proc.returncode != 0:
            llm_output += f"\n[exit code: {proc.returncode}]"

        logger.info("[bash] returncode=%d stdout_len=%d events=%d",
                    proc.returncode, len(stdout), len(events))
        return {"_events": events}, llm_output

    def _detect_events(self, before: dict, after: dict) -> list[dict]:
        """对比 session 前后状态，生成需要推送给前端的事件列表。"""
        events = []

        after_outline = after.get("outline_tree") or {}
        before_outline = before.get("outline_tree") or {}
        if after_outline and after_outline != before_outline:
            self.memory.set_outline(after_outline, after.get("md_with_ids", ""), after.get("markdown", ""))
            events.append({
                "type":        "outline",
                "markdown":    after.get("markdown", ""),
                "md_with_ids": after.get("md_with_ids", ""),
                "outline_tree": after_outline,
            })

        after_ext = after.get("extraction") or {}
        before_ext = before.get("extraction") or {}
        if after_ext and after_ext != before_ext:
            self.memory.set_extraction(after_ext)
            if after_ext.get("scene_name"):
                events.append({
                    "type":       "extraction",
                    "scene_name": after_ext.get("scene_name", ""),
                    "keywords":   after_ext.get("keywords", []),
                    "summary":    after_ext.get("summary", ""),
                })

        return events


# ── Session 文件操作 ──────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR / f"{session_id}.json"


def _read_session(session_id: str) -> dict:
    p = _session_path(session_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _write_session(session_id: str, data: dict) -> None:
    _session_path(session_id).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 前端步骤面板摘要 ─────────────────────────────────────────────

def _result_display(name: str, result: dict, llm_str: str) -> str:
    if name == "read_skill":
        lines = [l for l in llm_str.splitlines() if l.strip()]
        return lines[0] if lines else "已读取"
    if name == "bash":
        first = llm_str.splitlines()[0] if llm_str.splitlines() else ""
        events = result.get("_events", [])
        tag = " | ".join(e["type"] for e in events) if events else ""
        summary = first[:80] if first else "(no output)"
        return f"{summary}  [{tag}]" if tag else summary
    return "完成"
