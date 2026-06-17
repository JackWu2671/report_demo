"""
agent.py — 脚本驱动的单一 agent，无业务 tool schema。

LLM 有四个工具：
  read_skill      — 加载 SKILL.md SOP（Level 1）或支持文件（Level 2）
  bash            — 执行 bash 命令（通常是 skills/<name>/scripts/*.py）
  set_outline     — 一次性写入完整大纲
  modify_outline  — 对已有大纲执行修改（增删节点、保留分支、修改节点属性）

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
from tools.shared_tools import READ_SKILL_TOOL, BASH_TOOL, SET_OUTLINE_TOOL, MODIFY_OUTLINE_TOOL

_LIB_DIR = str(_SKILLS_DIR / "_lib")
if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from patcher import apply_patch  # noqa: E402
from outline_utils import to_clean_json, to_markdown, to_yaml  # noqa: E402
from set_outline_from_markdown import set_outline_from_tree  # noqa: E402
from services.temp_store import write_outline as _write_temp_outline  # noqa: E402

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (Path(_AGENT_DIR) / "system_prompt.txt").read_text(encoding="utf-8")
_MAX_ROUNDS = 12

TOOLS = [READ_SKILL_TOOL, BASH_TOOL, SET_OUTLINE_TOOL, MODIFY_OUTLINE_TOOL]

_SKILL_SYSTEM_TEMPLATE = """\
<skill_system>
遇到复杂任务先用 read_skill(<skill_name>) 阅读工作流指导，再用 bash 执行对应脚本。
只在需要时读取，不要预先读取所有技能。

<available_skills>
{skill_entries}
</available_skills>
</skill_system>"""


def _repair_truncated_json(s: str) -> list | None:
    """补全因 token 截断导致末尾括号不完整的 JSON 字符串，成功返回 list，否则返回 None。

    只在确认有未闭合括号时才修复；若括号本身已平衡则说明不是截断问题，返回 None。
    """
    stack: list[str] = []
    in_string = False
    escape_next = False

    for ch in s:
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append('}' if ch == '{' else ']')
        elif ch in '}]':
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack:
        return None  # 括号已平衡，不是截断问题

    repaired = s + ''.join(reversed(stack))
    try:
        result = json.loads(repaired)
    except json.JSONDecodeError:
        return None
    if isinstance(result, list):
        logger.info("[set_outline] 截断 JSON 修复成功，补全了 %d 个括号: %s",
                    len(stack), ''.join(reversed(stack)))
        return result
    return None


def _coerce_outline_str(raw: str):
    """LLM 误将 outline 数组序列化为字符串时尝试还原，返回 list 或抛出 ValueError。

    处理顺序（顺序不能乱）：
      1. json.loads → list：直接返回
      2. json.loads → str（双重序列化）：对结果再试一次 json.loads
      3. json.loads 失败 → 去掉外层多余引号后再试 json.loads
      4. 仍失败 → 补全未闭合括号修复截断 JSON（LLM 生成过长被 token 截断）
      5. 仍失败 → ast.literal_eval（兼容单引号 Python repr）
      均失败则抛 ValueError，调用方返回明确报错给 LLM。

    注意：步骤 1/2 必须先于步骤 3，否则双重序列化的 backslash 会被提前破坏。
    """
    import ast

    s = raw.strip()

    # ① ② 先走 json.loads，最多两轮（处理双重序列化）
    for _ in range(2):
        try:
            result = json.loads(s)
        except json.JSONDecodeError:
            break  # json.loads 失败，走后续降级路径
        if isinstance(result, list):
            return result
        if isinstance(result, str):
            s = result  # 双重序列化，展开一层再试
            continue
        raise ValueError(f"outline 解析后类型为 {type(result).__name__}，期望 list")

    # ③ json.loads 失败，若有外层多余引号则去掉后再试一次 json.loads
    if s.startswith('"') and s.endswith('"'):
        inner = s[1:-1]
        try:
            result = json.loads(inner)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            s = inner  # 让后续步骤也用去掉引号后的内容

    # ④ 补全截断 JSON（LLM 生成的字符串因 token 限制在末尾被截断，缺少闭合括号）
    repaired = _repair_truncated_json(s)
    if repaired is not None:
        return repaired

    # ⑤ 最后尝试 ast.literal_eval（兼容单引号 Python repr）
    try:
        result = ast.literal_eval(s)
    except Exception as exc:
        logger.warning("[set_outline] 所有解析方式均失败 | head=%r | err=%s", s[:120], exc)
        raise ValueError("outline 字符串无法解析为 JSON 数组，请直接传入 array 而非字符串") from exc

    if isinstance(result, list):
        return result
    raise ValueError(f"ast.literal_eval 结果类型为 {type(result).__name__}，期望 list")


# ── 文本 tool call 兜底解析 ────────────────────────────────────────
# vLLM 的 tool-call-parser 与模型模板失配时（如 Qwen3 未配 --tool-call-parser
# hermes），模型会把 tool call 当普通文本输出，原生 tool_calls 字段为空。
# 这里把泄漏到 content 里的 tool call 文本解析回结构化调用，避免直接吐给用户。

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNC_NAME_RE = re.compile(r"<function=([A-Za-z_]\w*)")
_FUNC_CALL_RE = re.compile(r"<function=[A-Za-z_]\w*\((.*)\)\s*", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_one_text_call(block: str) -> dict | None:
    """解析单个 tool call 文本块，返回 {"name", "arguments"}，失败返回 None。

    兼容三种 Qwen/vLLM 常见变体：
      A. Hermes JSON  : {"name": "x", "arguments": {...}}
      B. Qwen XML     : <function=x><parameter=k>v</parameter></function>
      C. Python 调用式: <function=read_skill(name="analyze-network")>
    """
    import ast

    block = block.strip()

    # A. JSON 体（Hermes 风格）
    if block.startswith("{"):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and obj.get("name"):
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}

    name_m = _FUNC_NAME_RE.search(block)
    if not name_m:
        return None
    name = name_m.group(1)
    args: dict = {}

    # C. Python 调用式 <function=name(k="v", ...)>，用 ast 解析 kwargs
    call_m = _FUNC_CALL_RE.search(block)
    if call_m and call_m.group(1).strip():
        try:
            node = ast.parse(f"f({call_m.group(1)})", mode="eval").body
            for kw in node.keywords:  # type: ignore[attr-defined]
                if kw.arg:
                    args[kw.arg] = ast.literal_eval(kw.value)
        except Exception:
            args = {}

    # B. Qwen XML <parameter=key>value</parameter>
    if not args:
        for pk, pv in _PARAM_RE.findall(block):
            pv = pv.strip()
            try:
                args[pk.strip()] = json.loads(pv)
            except json.JSONDecodeError:
                args[pk.strip()] = pv

    return {"name": name, "arguments": args}


def _parse_text_tool_calls(content: str) -> list[dict]:
    """从模型文本输出中解析被当成普通文本泄漏的 tool call，解析不出返回 []。"""
    if not content or ("<tool_call>" not in content and "<function=" not in content):
        return []
    blocks = _TOOL_CALL_BLOCK_RE.findall(content)
    if not blocks:
        blocks = [content]  # 没有成对 <tool_call> 标签，退而整体扫描 <function=>
    calls = []
    for block in blocks:
        parsed = _parse_one_text_call(block)
        if parsed:
            calls.append(parsed)
    return calls


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

            # 归一化 tool call：优先用服务端解析好的原生 tool_calls；
            # 若服务端 parser 失配把 tool call 当文本吐出来，则从 content 兜底解析。
            calls = self._normalize_tool_calls(choice, msg)

            if calls:
                report_triggered = False
                for call_id, name, args_str in calls:
                    try:
                        args = json.loads(args_str)
                    except Exception:
                        args = {}

                    logger.info("[Agent] tool=%s args=%s", name, args_str[:200])
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
                        {"role": "tool", "tool_call_id": call_id, "content": llm_str}
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

    def _normalize_tool_calls(self, choice, msg) -> list[tuple[str, str, str]]:
        """归一化本轮的 tool call，并把 assistant 消息写入 memory。

        返回 [(call_id, name, arguments_str), ...]；无 tool call 时返回 []。
        优先用服务端原生 tool_calls；服务端 parser 失配时从 content 文本兜底解析。
        """
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            self.memory.add_message(msg.model_dump(exclude_none=True))
            return [(tc.id, tc.function.name, tc.function.arguments) for tc in msg.tool_calls]

        text_calls = _parse_text_tool_calls(msg.content or "")
        if not text_calls:
            self.memory.add_message(msg.model_dump(exclude_none=True))
            return []

        # 文本兜底：重写 assistant 消息为规范 tool_calls 形态并清掉泄漏的原始文本，
        # 否则后续 role:tool 引用的 tool_call_id 在历史中找不到，部分模板会报错。
        synth_tcs, calls = [], []
        for i, c in enumerate(text_calls):
            cid = f"call_text_{i}_{uuid.uuid4().hex[:8]}"
            args_str = json.dumps(c["arguments"], ensure_ascii=False)
            synth_tcs.append({
                "id": cid, "type": "function",
                "function": {"name": c["name"], "arguments": args_str},
            })
            calls.append((cid, c["name"], args_str))
        self.memory.add_message({"role": "assistant", "content": None, "tool_calls": synth_tcs})
        logger.warning("[Agent] 服务端未返回原生 tool_calls，已从文本兜底解析 %d 个: %s",
                       len(calls), [c[1] for c in calls])
        return calls

    async def _execute_tool(self, name: str, args: dict) -> tuple[dict, str]:
        if name == "read_skill":
            return self._handle_read_skill(args)
        if name == "bash":
            return await self._handle_bash(args.get("command", ""))
        if name == "set_outline":
            return await self._handle_set_outline(args)
        if name == "modify_outline":
            return await self._handle_modify_outline(args)
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
        # LLM 有时会把数组序列化成字符串（甚至双重序列化）再传入，逐层尝试解析
        if isinstance(outline, str):
            try:
                outline = _coerce_outline_str(outline)
            except ValueError as exc:
                return {"_events": []}, f"[set_outline] outline 参数解析失败：{exc}（大纲未写入，请重新调用并直接传入 JSON 数组）"
        if not outline or not isinstance(outline, list):
            return {"_events": []}, "[set_outline] outline 参数缺失或类型错误（应为 JSON 节点数组，顶层含一个 L1 根节点）"

        result = await set_outline_from_tree(outline)
        if result["status"] != "success":
            # 失败必须明确告知，禁止当成功（治"静默失败+谎报"）
            return {"_events": []}, f"[set_outline] 写入失败: {result['message']}（大纲未生成，请修正后重试，不要告知用户已生成）"

        self.memory.set_outline(
            result["outline_tree"],
            result["markdown"],
            result["outline_yaml"],
        )
        _write_temp_outline(self.session_id, result["outline_tree"], result["markdown"], result["outline_yaml"])
        events = [{
            "type":         "outline",
            "markdown":     result["markdown"],
            "outline_yaml": result["outline_yaml"],
            "outline_tree": result["outline_tree"],
        }]
        return {"_events": events}, "[set_outline] 大纲已写入并推送\n" + result["outline_yaml"]

    async def _handle_modify_outline(self, args: dict) -> tuple[dict, str]:
        """对已有大纲执行一个或多个结构化修改操作，参数走 JSON、不过 shell。"""
        ops = args.get("ops")
        if not ops or not isinstance(ops, list):
            return {"_events": []}, "[modify_outline] ops 参数缺失或类型错误（应为操作对象数组）"

        outline_tree = self.memory.outline_tree
        if not outline_tree:
            return {"_events": []}, "[modify_outline] 当前没有大纲，请先生成大纲"

        new_tree, skipped = await apply_patch(outline_tree, ops)

        lines = [f"[modify_outline] 已执行 {len(ops) - len(skipped)}/{len(ops)} 个操作"]
        for s in skipped:
            reason = s.get("_skip_reason", "未知") if isinstance(s, dict) else str(s)
            lines.append(f"SKIPPED: {s.get('op', '?')} node_id={s.get('node_id', '')} → {reason}")

        if skipped and len(skipped) >= len(ops):
            return {"_events": []}, "\n".join(lines)

        clean_tree = to_clean_json(new_tree)
        md = to_markdown(clean_tree)
        yaml_str = to_yaml(clean_tree)
        self.memory.set_outline(clean_tree, md, yaml_str)
        _write_temp_outline(self.session_id, clean_tree, md, yaml_str)
        events = [
            {"type": "outline", "markdown": md, "outline_yaml": yaml_str, "outline_tree": clean_tree},
            {"type": "confirm", "options": ["生成报告"]},
        ]
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
            # 子进程统一用 UTF-8 编码 stdout/stderr，与父进程解码一致；
            # 否则 Windows 下子进程默认按 GBK 输出中文，父进程 UTF-8 解码会乱码。
            "PYTHONIOENCODING":   "utf-8",
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
                encoding="utf-8", errors="replace",
                env=env, timeout=60,
            )
        except subprocess.TimeoutExpired:
            return {"_events": []}, "[bash] 执行超时（60s）"

        # 脚本统一输出 UTF-8，显式指定解码避免 Windows 默认 GBK 解码中文失败；
        # errors="replace" 再兜一层，坏字节也不会让 stdout 变 None。
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()

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
            _write_temp_outline(self.session_id, after_outline, after.get("markdown", ""), after.get("outline_yaml", ""))
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
