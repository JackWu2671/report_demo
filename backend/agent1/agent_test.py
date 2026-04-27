"""
agent_test.py — Interactive CLI for testing Agent1.

Usage:
    cd backend
    python agent1/agent_test.py

Or pass expert text directly:
    python agent1/agent_test.py "专家场景描述..."

Commands:
  /state    print current outline_tree JSON
  /md       print Markdown outline (human view)
  /ids      print md_with_ids (LLM view)
  /meta     print extraction metadata
  /new      print [new] nodes list
  /reset    reset session
  /help     show this help
  empty / Ctrl-C → exit
"""

import asyncio
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_WF1 = os.path.join(_BACKEND, "case_workflow_1")

for _p in [_BACKEND, _WF1]:
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

from agent1.agent import Agent1

_DIV = "─" * 56


def _print_help() -> None:
    print("""
命令:
  /state   打印大纲 JSON
  /md      打印 Markdown 大纲
  /ids     打印带 ID 大纲（LLM 视图）
  /meta    打印场景元数据
  /new     打印 [new] 节点列表
  /reset   重置会话
  /help    显示此帮助
  空行 / Ctrl-C  退出
""")


def _read_input() -> str | None:
    """Read a slash-command (single line) or multi-line expert text (end with '---')."""
    try:
        first = input("你 > ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    # Commands and empty lines: return immediately
    if not first or first.startswith("/"):
        return first if first else None

    # Multi-line mode for expert descriptions
    print("  (继续输入，单独一行输入 --- 结束多行输入)")
    lines = [first]
    while True:
        try:
            line = input("... ").rstrip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "---":
            break
        lines.append(line)
    return "\n".join(lines)


async def repl(initial_text: str = "") -> None:
    agent = Agent1()
    print(f"\n{'=' * 56}")
    print("  专家知识沉淀 Agent — 交互测试")
    print(f"{'=' * 56}")
    print("输入专家场景描述，或 /help 查看命令。")
    print("多行输入请换行继续，最后一行单独输入 --- 结束。\n")

    first_input = initial_text

    while True:
        if first_input:
            user_input = first_input
            first_input = ""
            print(f"你 > {user_input}")
        else:
            user_input = _read_input()
            if user_input is None:
                print("\n退出。")
                break

        if not user_input:
            continue

        if user_input == "/help":
            _print_help(); continue
        if user_input == "/state":
            print(json.dumps(agent.memory.outline_tree, ensure_ascii=False, indent=2)); continue
        if user_input == "/md":
            print(agent.memory.markdown or "（暂无大纲）"); continue
        if user_input == "/ids":
            print(agent.memory.md_with_ids or "（暂无大纲）"); continue
        if user_input == "/meta":
            print(json.dumps(agent.memory.extraction, ensure_ascii=False, indent=2)); continue
        if user_input == "/new":
            for n in agent.memory.new_nodes:
                print(f"  L{n['level']} {n['name']} → 父: {n.get('parent_name', '?')}"); continue
        if user_input == "/reset":
            agent.memory.reset(); print("✓ 已重置\n"); continue

        t0 = time.time()
        try:
            async for event in agent.chat_stream(user_input):
                etype = event.get("type")
                if etype == "step":
                    icon = "▶" if event["status"] == "running" else "✓"
                    print(f"  {icon} {event['name']}", flush=True)
                elif etype == "outline":
                    print(f"\n{_DIV}\n{event['markdown']}\n{_DIV}")
                elif etype == "new_nodes":
                    nodes = event["nodes"]
                    print(f"\n  ⚠ [new] 节点 ({len(nodes)}个): "
                          + ", ".join(n["name"] for n in nodes))
                elif etype == "saved":
                    print(f"\n  ✅ 模板已保存: {event['scene_name']}  →  {event['path']}")
                elif etype == "text":
                    print(f"\n🤖 {event['chunk']}\n")
                elif etype == "done":
                    print(f"   耗时 {event.get('seconds', round(time.time()-t0,1))}s\n")
                elif etype == "error":
                    print(f"\n[错误] {event['message']}\n")
        except Exception as e:
            print(f"\n[异常] {e}\n")
            logging.exception("chat_stream 异常")


if __name__ == "__main__":
    initial = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    asyncio.run(repl(initial))
