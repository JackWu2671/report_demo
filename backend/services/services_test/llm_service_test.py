"""
llm_service_test.py — 测试 LLMService.stream_and_collect() 流式打印效果。

使用方法:
    cd backend
    python services/services_test/llm_service_test.py
    python services/services_test/llm_service_test.py "你的问题"
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

load_dotenv(os.path.join(_BACKEND_DIR, ".env"))

from services.llm_service import LLMService


async def test_stream(question: str) -> None:
    llm = LLMService.from_env()

    print(f"模型  : {llm.default_model}")
    print(f"地址  : {llm.base_url}")
    print(f"Think : {llm.think_tag_mode}")
    print(f"问题  : {question}")
    print("-" * 50)

    messages = [{"role": "user", "content": question}]

    answer = await llm.stream_and_collect(messages)

    print("\n" + "-" * 50)
    print(f"[收集到的正式回答，共 {len(answer)} 字符]")
    print(answer)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "用一句话介绍你自己"
    asyncio.run(test_stream(q))
