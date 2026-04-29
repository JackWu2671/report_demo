"""
agent_test.py — AgentWithSkills 交互测试。

可以直观看到 skill 渐进式加载的完整过程：

  你 > 帮我分析 fgOTN 部署情况

  ▶ skills_list                     ← Level 0：查询有哪些 skill（如需要）
  ✓ skills_list
  ▶ skill_view                      ← Level 1：加载 generate-outline SOP
  ✓ skill_view
  ▶ search_outline_template         ← 按 SOP 调工具
  ✓ search_outline_template
  ── 大纲 ──────────────────────────
  # fgOTN 部署分析
  ...
  ──────────────────────────────────
  🤖 已找到匹配模板，共 3 个分析维度。

用法:
  cd backend
  python agent_with_skills/agent_test.py

内置命令:
  /skills   显示已发现的 skill 列表及加载状态
  /state    打印当前大纲 JSON
  /md       打印 Markdown 大纲（用户视图）
  /reset    重置会话（清空对话历史和 skill 加载状态）
  /help     显示此帮助
  空行 / Ctrl-C  退出
"""

import asyncio
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from agent_with_skills.agent import AgentWithSkills  # noqa: E402

_DIV = "─" * 56

# skill 元工具，显示时加特殊标记，方便区分"skill 加载"和"业务工具"
_SKILL_TOOLS = {"skills_list", "skill_view"}


def _print_outline(markdown: str) -> None:
    print(f"\n── 大纲 {'─' * 48}\n{markdown}\n{_DIV}")


def _print_skills(agent: AgentWithSkills) -> None:
    print("\n已发现的 Skill：")
    for m in agent._skill_meta:
        loaded = "✓ 已加载" if m["name"] in agent._loaded else "  未加载"
        cat = f"[{m['category']}] " if m.get("category") else ""
        print(f"  {loaded}  {cat}{m['name']} — {m.get('description', '')[:40]}")
    print()


def _print_help() -> None:
    print("""
命令:
  /skills  显示 skill 列表及加载状态
  /state   打印当前大纲 JSON
  /md      打印 Markdown 大纲（用户视图）
  /reset   重置会话
  /help    显示此帮助
  空行 / Ctrl-C  退出
""")


async def repl() -> None:
    print(f"\n{'=' * 56}")
    print("  AgentWithSkills — 交互测试")
    print(f"{'=' * 56}")

    agent = AgentWithSkills()  # 在 header 之后创建，日志不会和提示符交错

    print(f"已发现 {len(agent._skill_meta)} 个 skill，输入需求开始，/help 查看命令。\n")

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
        if user_input == "/skills":
            _print_skills(agent)
            continue
        if user_input == "/state":
            print(json.dumps(agent.memory.outline_tree, ensure_ascii=False, indent=2))
            continue
        if user_input == "/md":
            print(agent.memory.markdown or "（暂无大纲）")
            continue
        if user_input == "/reset":
            agent.reset()
            print("✓ 会话已重置（对话历史 + skill 加载状态已清空）。\n")
            continue

        t0 = time.time()
        try:
            async for event in agent.chat_stream(user_input):
                etype = event.get("type")

                if etype == "step":
                    name = event["name"]
                    icon = "▶" if event["status"] == "running" else "✓"
                    # skill 元工具用不同颜色前缀区分
                    prefix = "  [skill] " if name in _SKILL_TOOLS else "  "
                    print(f"{prefix}{icon} {name}", flush=True)

                elif etype == "extraction":
                    print(f"\n  [场景] {event.get('scene_name', '')}  关键词: {', '.join(event.get('keywords', []))}\n")

                elif etype == "new_nodes":
                    names = [n.get("name", "") for n in event.get("nodes", [])]
                    print(f"  [新节点] {', '.join(names)}\n")

                elif etype == "saved":
                    print(f"\n  [已保存] {event.get('scene_name', '')}  → {event.get('path', '')}\n")

                elif etype == "outline":
                    _print_outline(event["markdown"])

                elif etype == "confirm":
                    opts = " / ".join(event.get("options", []))
                    print(f"\n  [确认] {opts}\n")

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
