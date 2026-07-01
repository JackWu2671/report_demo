#!/usr/bin/env python3
"""
test_llm_ping.py — 最简单的模型连通性测试，不依赖 LLMService。

用法:
  cd backend
  python3 scripts/test_llm_ping.py "你好，介绍一下你自己"
  python3 scripts/test_llm_ping.py            # 不传参数用默认问题

读取 .env 里的 LLM_BASE_URL / LLM_MODEL_NAME / LLM_API_KEY 直接调用，
非流式，打印原始返回内容，用于快速确认模型是否可用。
"""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))


def ask(question: str) -> str:
    """输入一句话，返回大模型的回答（非流式，直接拿 message.content）。"""
    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("LLM_API_KEY") or "EMPTY",
    )
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", ""),
        messages=[{"role": "user", "content": question}],
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
    )
    message = resp.choices[0].message
    if not message.content:
        # content 为空时把整条 message 打出来，方便排查是不是内容全跑进了
        # reasoning_content（思考模型没关思考时的常见现象，见 llm_service.py 的处理）
        print(f"[警告] content 为空，原始 message: {message!r}")
    return message.content or ""


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "你好，用一句话介绍一下你自己"
    print(f"问题: {q}\n")
    answer = ask(q)
    print(f"回答: {answer!r}")
