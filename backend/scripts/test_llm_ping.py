#!/usr/bin/env python3
"""
test_llm_ping.py — 最简单的模型连通性测试，不依赖 LLMService。

用法:
  python3 scripts/test_llm_ping.py "你好，介绍一下你自己"
  python3 scripts/test_llm_ping.py            # 不传参数用默认问题

直接改下面的常量填模型地址，非流式，打印原始返回内容，
用于快速确认模型是否可用。
"""

import sys

from openai import OpenAI

LLM_BASE_URL = "http://10.118.238.104:8003/v1"
LLM_MODEL_NAME = "qwen3.6-27b"
LLM_API_KEY = "EMPTY"
LLM_TEMPERATURE = 0.1


def ask(question: str) -> str:
    """输入一句话，返回大模型的回答（非流式，直接拿 message.content）。"""
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    resp = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[{"role": "user", "content": question}],
        temperature=LLM_TEMPERATURE,
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
