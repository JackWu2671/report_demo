"""
tool_test.py — 单轮 LLM 调用测试，直接打印大模型原始输出的 JSON 结构。

用途：看清楚大模型 function calling 的原始输出长什么样。

用法：
    cd backend
    python tests/tool_test.py
    python tests/tool_test.py "帮我分析一下fgOTN的部署情况"
"""

import asyncio
import json
import os
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from dotenv import load_dotenv
load_dotenv(os.path.join(_BACKEND, ".env"))

from services.llm_service import LLMService
from agent2.tools import TOOLS
from pathlib import Path

_SYSTEM_PROMPT = (Path(_BACKEND) / "agent2" / "prompt.txt").read_text(encoding="utf-8")

_DIV  = "─" * 60
_DIV2 = "═" * 60


async def single_round(user_input: str) -> None:
    llm = LLMService.from_env()

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_input},
    ]

    print(f"\n{_DIV2}")
    print("  单轮 LLM tool-call 测试 (Agent2)")
    print(_DIV2)
    print(f"\n[用户输入]\n{user_input}\n")
    print(f"[传给大模型的 tools 列表] — 共 {len(TOOLS)} 个工具")
    for t in TOOLS:
        fn = t["function"]
        params = list(fn["parameters"]["properties"].keys())
        print(f"  • {fn['name']}({', '.join(params)})")

    print(f"\n{_DIV} 发送请求... {_DIV}\n")

    response = await llm._client.chat.completions.create(
        model=llm.default_model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=llm._temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    choice = response.choices[0]
    msg = choice.message

    # ── 打印原始响应结构 ──────────────────────────────────────────
    print("[大模型原始输出 — response.choices[0]]\n")
    print(f"  finish_reason : {choice.finish_reason!r}")
    print(f"  message.role  : {msg.role!r}")
    print(f"  message.content: {msg.content!r}")
    print()

    if msg.tool_calls:
        print(f"  message.tool_calls: [{len(msg.tool_calls)} 个]\n")
        for i, tc in enumerate(msg.tool_calls):
            print(f"  [{i}] id       : {tc.id!r}")
            print(f"  [{i}] type     : {tc.type!r}")
            print(f"  [{i}] function.name      : {tc.function.name!r}")
            print(f"  [{i}] function.arguments : (原始字符串)")
            print(f"       {tc.function.arguments}")
            try:
                parsed = json.loads(tc.function.arguments)
                print(f"\n       (解析后 JSON):")
                print(json.dumps(parsed, ensure_ascii=False, indent=6))
            except json.JSONDecodeError as e:
                print(f"       ⚠ arguments 解析失败: {e}")
            print()
    else:
        print("  message.tool_calls: None  (大模型直接回复文字，未调工具)\n")

    # ── 打印完整原始 JSON ────────────────────────────────────────
    print(f"{_DIV}")
    print("[完整原始 JSON — response.model_dump()]\n")
    raw = response.model_dump()
    print(json.dumps(raw, ensure_ascii=False, indent=2))

    # ── 结论 ─────────────────────────────────────────────────────
    print(f"\n{_DIV}")
    if choice.finish_reason == "tool_calls":
        names = [tc.function.name for tc in msg.tool_calls]
        print(f"[结论] 大模型决定调工具: {names}")
        print("       agent.py 会执行对应 handler，把结果追加到 messages 后再调一轮 LLM")
    else:
        print("[结论] 大模型直接回复文字，未触发工具调用")
    print()


if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]).strip() or "帮我生成一份fgOTN部署分析的报告大纲"
    asyncio.run(single_round(user_input))
