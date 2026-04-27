"""
agent_test.py — Interactive CLI for testing Agent2.

Usage:
    cd backend
    python agent2/agent_test.py

Commands inside the REPL:
    /state    — print current outline_tree JSON
    /md       — print current human-readable Markdown
    /ids      — print current LLM-context Markdown (with [Lx id] prefixes)
    /reset    — start a fresh agent session
    /help     — show this help
    Ctrl-C / empty input → exit

Workflow:
    1. Type your analysis request to generate an initial outline
    2. Type modification instructions to refine it
    3. Use /state or /md to inspect the current outline at any time
"""

import asyncio
import json
import logging
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_WF2 = os.path.join(_BACKEND, "case_workflow_2")

for _p in [_BACKEND, _WF2]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
# Quiet chatty loggers during interactive use
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from agent2.agent import Agent2  # noqa: E402 (after sys.path setup)


# ── REPL ──────────────────────────────────────────────────────────────────────

_DIVIDER = "─" * 60


def _print_assistant(text: str) -> None:
    print(f"\n🤖 助手\n{_DIVIDER}\n{text}\n{_DIVIDER}\n")


def _print_outline_state(agent: Agent2) -> None:
    print(f"\n{_DIVIDER}")
    print("当前大纲 (JSON):")
    print(json.dumps(agent.state["outline_tree"], ensure_ascii=False, indent=2))
    print(_DIVIDER)


def _print_outline_md(agent: Agent2) -> None:
    md = agent.state.get("markdown", "（暂无大纲）")
    print(f"\n{_DIVIDER}\n{md}\n{_DIVIDER}")


def _print_outline_ids(agent: Agent2) -> None:
    ids = agent.state.get("md_with_ids", "（暂无大纲）")
    print(f"\n{_DIVIDER}\n{ids}\n{_DIVIDER}")


def _print_help() -> None:
    print("""
命令:
  /state   打印当前大纲 JSON
  /md      打印当前 Markdown 大纲（用户视图）
  /ids     打印带 ID 的大纲（LLM 上下文视图）
  /reset   重置会话（清空历史和大纲）
  /help    显示此帮助
  Ctrl-C 或空行  退出
""")


async def repl() -> None:
    agent = Agent2()
    print(f"\n{'=' * 60}")
    print("  报告大纲生成 Agent — 交互测试")
    print(f"{'=' * 60}")
    print("输入分析需求开始，输入 /help 查看命令。\n")

    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            print("退出。")
            break

        # Built-in commands
        if user_input == "/help":
            _print_help()
            continue
        if user_input == "/state":
            _print_outline_state(agent)
            continue
        if user_input == "/md":
            _print_outline_md(agent)
            continue
        if user_input == "/ids":
            _print_outline_ids(agent)
            continue
        if user_input == "/reset":
            agent = Agent2()
            print("✓ 会话已重置。\n")
            continue

        # Send to agent
        try:
            response = await agent.chat(user_input)
        except Exception as e:
            print(f"\n[错误] {e}\n")
            logging.exception("agent.chat 异常")
            continue

        _print_assistant(response)


if __name__ == "__main__":
    asyncio.run(repl())
