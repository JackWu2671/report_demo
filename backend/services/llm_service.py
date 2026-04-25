"""
LLM 服务层，封装 OpenAI-compatible chat completions API（openai SDK）。

  stream_and_collect()  流式调用，实时打印，返回正式回答内容字符串
  complete()            流式调用，不打印，返回正式回答内容字符串
  complete_json()       complete() + JSON 解析

enable_thinking=True 时，服务端返回的 delta.reasoning_content 字段为思考过程，
delta.content 字段为正式回答；enable_thinking=False 时，仅有 delta.content。
两种情况下，返回值均为正式回答内容（reasoning_content 不计入返回值）。
"""

import json
import logging
import os
import re

from openai import AsyncOpenAI

from llm.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        base_url: str,
        model: str = "",
        api_key: str = "",
        temperature: float = 0.1,
        top_p: float = 1.0,
        timeout: int = 120,
        enable_thinking: bool = False,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            timeout=timeout,
        )
        self.base_url = base_url
        self.default_model = model
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self.enable_thinking = enable_thinking

    @classmethod
    def from_env(cls) -> "LLMService":
        """从环境变量构造实例（需在调用前 load_dotenv）。"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            model=os.getenv("LLM_MODEL_NAME", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            top_p=float(os.getenv("LLM_TOP_P", 1.0)),
            timeout=int(os.getenv("LLM_TIMEOUT", 120)),
            enable_thinking=os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true",
        )

    # ─── 公开接口 ───────────────────────────────────────────────

    async def stream_and_collect(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """流式调用，实时打印到终端，返回正式回答内容字符串（不含思考内容）。"""
        return await self._stream(messages, config, print_stream=True)

    async def complete(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """流式调用，不打印，返回正式回答内容字符串（不含思考内容）。"""
        return await self._stream(messages, config, print_stream=False)

    async def complete_json(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> dict:
        """complete() + 自动解析 JSON。"""
        raw = await self.complete(messages, config)
        return self._parse_json(raw)

    # ─── 内部方法 ───────────────────────────────────────────────

    async def _stream(
        self,
        messages: list[dict],
        config: LLMConfig | None = None,
        print_stream: bool = False,
    ) -> str:
        cfg = config or LLMConfig()
        model = cfg.model or self.default_model
        temperature = cfg.temperature if cfg.temperature is not None else self._temperature

        logger.info(
            "[LLM] 调用 model=%s temperature=%.2f messages=%d条",
            model, temperature, len(messages),
        )
        for m in messages:
            logger.info("[LLM Prompt][%s]\n%s", m["role"], m["content"])

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=cfg.top_p if cfg.top_p is not None else self._top_p,
            max_tokens=cfg.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": self.enable_thinking},
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_content += reasoning
                if print_stream and not is_answering:
                    print(reasoning, end="", flush=True)

            content = getattr(delta, "content", None)
            if content:
                if not is_answering:
                    is_answering = True
                answer_content += content
                if print_stream:
                    print(content, end="", flush=True)

        logger.info(
            "[LLM Output] (%d字):\n%s",
            len(answer_content), answer_content,
        )
        return answer_content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        从 LLM 输出中提取 JSON，兼容三种格式：
        1. ```json … ``` 代码块
        2. 裸 JSON 对象
        3. 文本中嵌套的 {...} 块
        """
        s = raw.strip()
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(s[first : last + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 LLM 输出中提取 JSON: {s[:200]}")
