"""
agent.py — 脚本驱动的单一 agent，无业务 tool schema。

LLM 有三个工具：
  read_skill — 加载 SKILL.md SOP（Level 1）或支持文件（Level 2）
  bash       — 执行 bash 命令（通常是 skills/<name>/scripts/*.py）
  edit_node  — 直接修改大纲节点属性（参数走 JSON、不过 shell，含特殊字符的 SQL 安全）

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
import platform
import re
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
from tools.shared_tools import READ_SKILL_TOOL, BASH_TOOL, EDIT_NODE_TOOL, SET_OUTLINE_TOOL

_LIB_DIR = str(_SKILLS_DIR / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from modify_outline import modify_outline  # noqa: E402  (imported after sys.path setup)
from set_outline_from_markdown import set_outline_from_tree  # noqa: E402

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_MAX_ROUNDS = 12

TOOLS = [READ_SKILL_TOOL, BASH_TOOL, EDIT_NODE_TOOL, SET_OUTLINE_TOOL]

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
                report_triggered = False
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
                        if event.get("type") == "start_report":
                            report_triggered = True
                        yield event

                    yield {"type": "step", "name": name, "status": "done",
                           "call_id": call_id,
                           "result": _result_display(name, result_dict, llm_str),
                           "detail": llm_str}
                    self.memory.add_message(
                        {"role": "tool", "tool_call_id": tc.id, "content": llm_str}
                    )

                # 触发报告生成后立即结束本回合：报告在独立的 /api/report 流中渲染，
                # 无需再跑一轮 LLM。否则那轮 LLM 会与报告自身的 LLM 调用抢占后端，
                # 导致聊天流迟迟不关闭、前端输入框一直转圈无法输入。
                if report_triggered:
                    reply = "好的，开始生成报告。"
                    self.memory.add_message({"role": "assistant", "content": reply})
                    yield {"type": "text", "chunk": reply}
                    yield {"type": "done", "seconds": round(time.time() - t0, 1)}
                    return

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
        if name == "edit_node":
            return await self._handle_edit_node(args)
        if name == "set_outline":
            return await self._handle_set_outline(args)
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

    async def _handle_set_outline(self, args: dict) -> tuple[dict, str]:
        """一次性写入完整大纲，参数为 JSON 节点数组（经工具参数传入，不过 shell、不过 YAML）。"""
        outline = args.get("outline")
        if not outline or not isinstance(outline, list):
            return {"_events": []}, "[set_outline] 缺少 outline 参数（应为完整大纲的 JSON 节点数组，顶层含一个 L1 根节点）"

        result = await set_outline_from_tree(outline)
        if result["status"] != "success":
            # 失败必须明确告知，禁止当成功（治"静默失败+谎报"）
            return {"_events": []}, f"[set_outline] 写入失败: {result['message']}（大纲未生成，请修正后重试，不要告知用户已生成）"

        self.memory.set_outline(
            result["outline_tree"],
            result["markdown"],
            result["outline_yaml"],
        )
        events = [{
            "type":         "outline",
            "markdown":     result["markdown"],
            "outline_yaml": result["outline_yaml"],
            "outline_tree": result["outline_tree"],
        }]
        return {"_events": events}, "[set_outline] 大纲已写入并推送\n" + result["outline_yaml"]

    async def _handle_edit_node(self, args: dict) -> tuple[dict, str]:
        """直接修改大纲节点属性，参数走 JSON、不过 shell。"""
        node_id = args.get("node_id", "").strip()
        field = str(args.get("field", "")).strip()
        value = args.get("value")

        outline_tree = self.memory.outline_tree
        if not outline_tree:
            return {"_events": []}, "[edit_node] 当前没有大纲，请先生成大纲"

        # 有专属 patcher op 的字段，走 modify_outline 管线（name 会触发 L5 KB 同步等逻辑）
        _field_op = {
            "name":        "modify_node_name",
            "description": "modify_node_description",
            "condition":   "modify_node_condition",
            "exec_sql":    "modify_node_exec_sql",
        }
        if field in _field_op:
            ops = [{"op": _field_op[field], "node_id": node_id, "value": value}]
        else:
            # summarySuggestion / renderType / colX / colY / condition_queries 等
            ops = [{"op": "set_node_field", "node_id": node_id, "field": field, "value": value}]

        result = await modify_outline(ops, outline_tree)

        if result["status"] != "success":
            return {"_events": []}, f"[edit_node] {result['message']}"

        self.memory.set_outline(
            result["outline_tree"],
            result["markdown"],
            result["outline_yaml"],
        )
        events = [
            {
                "type":         "outline",
                "markdown":     result["markdown"],
                "outline_yaml": result["outline_yaml"],
                "outline_tree": result["outline_tree"],
            },
            {"type": "confirm", "options": ["生成报告"]},
        ]

        lines = [f"[edit_node] 已更新 {node_id}.{field}"]
        for s in result.get("skipped", []):
            reason = s.get("_skip_reason", "未知") if isinstance(s, dict) else str(s)
            lines.append(f"SKIPPED: {s.get('op','?')} node_id={s.get('node_id','')} → {reason}")
        return {"_events": events}, "\n".join(lines)

    async def _handle_bash(self, command: str) -> tuple[dict, str]:
        """执行 bash 命令，同步 session 状态，返回事件列表。"""
        # 执行前将内存状态写入 session 文件，供脚本读取
        before = _read_session(self.session_id)
        _write_session(self.session_id, {
            "outline_tree": self.memory.outline_tree or {},
            "outline_yaml": self.memory.outline_yaml or "",
            "markdown":     self.memory.markdown or "",
            "extraction":   self.memory.extraction or {},
        })

        env = {
            **os.environ,
            "REPORT_SESSION_ID":  self.session_id,
            "REPORT_SESSION_DIR": str(_SESSION_DIR),
            "REPORT_BACKEND_DIR": _BACKEND_DIR,
            "SKILLS_DIR":         str(_SKILLS_DIR),
        }

        # Expand $VAR references so the command runs correctly on all platforms.
        # cmd.exe (Windows) doesn't expand $VAR, so we do it ourselves before
        # handing the command to the shell.
        for key, val in env.items():
            command = command.replace(f"${key}", val)

        # On Windows python3 is not on PATH; replace with the running interpreter.
        if platform.system() == "Windows":
            command = re.sub(r"\bpython3\b", sys.executable.replace("\\", "/"), command)

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
            self.memory.set_outline(after_outline, after.get("markdown", ""), after.get("outline_yaml", ""))
            events.append({
                "type":         "outline",
                "markdown":     after.get("markdown", ""),
                "outline_yaml": after.get("outline_yaml", ""),
                "outline_tree": after_outline,
            })
            events.append({"type": "confirm", "options": ["生成报告"]})

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

        if after.get("generate_report") and not before.get("generate_report"):
            after["generate_report"] = False
            _write_session(self.session_id, after)
            events.append({"type": "start_report"})

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
