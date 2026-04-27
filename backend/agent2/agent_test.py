"""
agent_test.py — Interactive CLI for testing Agent2.

Usage:
    cd backend
    python agent2/agent_test.py

Each user turn streams events:
  [running] search_outline_template    ← tool executing
  [done]    search_outline_template
  ── 大纲 ──────────────────────────    ← outline event (instant, no LLM streaming)
  # 传送网专项分析
  ...
  ──────────────────────────────────
  🤖 已找到匹配大纲，共5个章节。      ← LLM brief text

Built-in commands:
  /state    print current outline_tree JSON
  /md       print current Markdown (human view)
  /ids      print current md_with_ids (LLM view)
  /reset    start a fresh session
  /help     show this help
  empty line or Ctrl-C → exit
"""

import asyncio
import json
import logging
import os
import sys
import time

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
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from agent2.agent import Agent2  # noqa: E402

_DIV = "─" * 56


def _print_outline(markdown: str) -> None:
    print(f"\n{_DIV}\n{markdown}\n{_DIV}")


def _print_help() -> None:
    print("""
命令:
  /state   打印当前大纲 JSON
  /md      打印 Markdown 大纲（用户视图）
  /ids     打印带 ID 大纲（LLM 视图）
  /reset   重置会话
  /help    显示此帮助
  空行 / Ctrl-C  退出
""")


async def repl() -> None:
    agent = Agent2()
    print(f"\n{'=' * 56}")
    print("  报告大纲生成 Agent — 交互测试")
    print(f"{'=' * 56}")
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

        if user_input == "/help":
            _print_help()
            continue
        if user_input == "/state":
            print(json.dumps(agent.memory.outline_tree, ensure_ascii=False, indent=2))
            continue
        if user_input == "/md":
            print(agent.memory.markdown or "（暂无大纲）")
            continue
        if user_input == "/ids":
            print(agent.memory.md_with_ids or "（暂无大纲）")
            continue
        if user_input == "/reset":
            agent.memory.reset()
            print("✓ 会话已重置。\n")
            continue

        # Stream events
        t0 = time.time()
        try:
            async for event in agent.chat_stream(user_input):
                etype = event.get("type")

                if etype == "step":
                    icon = "▶" if event["status"] == "running" else "✓"
                    print(f"  {icon} {event['name']}", flush=True)

                elif etype == "outline":
                    _print_outline(event["markdown"])

                elif etype == "text":
                    print(f"\n🤖 {event['chunk']}\n")

                elif etype == "done":
                    elapsed = event.get("seconds", round(time.time() - t0, 1))
                    print(f"   耗时 {elapsed}s\n")

                elif etype == "error":
                    print(f"\n[错误] {event['message']}\n")

        except Exception as e:
            print(f"\n[异常] {e}\n")
            logging.exception("chat_stream 异常")


if __name__ == "__main__":
    asyncio.run(repl())
