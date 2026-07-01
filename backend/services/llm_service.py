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

# 单条 prompt 消息打日志时的截断长度（避免每次 LLM 调用都把整段 system prompt
# 写进日志造成膨胀）。设为 0 可关闭截断打印全文，调试时用。
_PROMPT_LOG_LIMIT = int(os.environ.get("PROMPT_LOG_LIMIT", "800"))

# 绕过代理直连 LLM 服务，避免内网地址被代理拦截
os.environ.setdefault("NO_PROXY", "oneapi.rnd.huawei.com")
os.environ.setdefault("no_proxy", "oneapi.rnd.huawei.com")


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
        max_tokens: int = 4096,
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
        self._max_tokens = max_tokens

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
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", 4096)),
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
        max_tokens = cfg.max_tokens if cfg.max_tokens is not None else self._max_tokens

        logger.info(
            "[LLM] 调用 model=%s temperature=%.2f max_tokens=%d messages=%d条",
            model, temperature, max_tokens, len(messages),
        )
        for m in messages:
            content = m.get("content")
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            if _PROMPT_LOG_LIMIT and text and len(text) > _PROMPT_LOG_LIMIT:
                text = f"{text[:_PROMPT_LOG_LIMIT]}…（截断，共{len(text)}字，PROMPT_LOG_LIMIT=0 看全文）"
            logger.info("[LLM Prompt][%s]\n%s", m["role"], text)

        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=cfg.top_p if cfg.top_p is not None else self._top_p,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
        )

        reasoning_content = ""
        answer_content = ""
        is_answering = False
        finish_reason = None
        usage = None

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
            delta = choice.delta

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
            "[LLM Output] (%d字，reasoning=%d字，finish_reason=%s，usage=%s):\n%s",
            len(answer_content), len(reasoning_content), finish_reason, usage, answer_content,
        )
        if reasoning_content:
            logger.info("[LLM reasoning_content 全文]\n%s", reasoning_content)

        if not answer_content and reasoning_content:
            if finish_reason == "length":
                logger.warning(
                    "[LLM] content 为空、reasoning_content 有 %d 字，finish_reason=length——"
                    "输出被截断，可调大 LLM_MAX_TOKENS",
                    len(reasoning_content),
                )
            else:
                # 部分部署未正确遵守 enable_thinking=false，把完整回答整段写进了
                # reasoning_content、content 从未被填充。finish_reason 正常结束时，
                # 兜底把 reasoning_content 当作回答返回，而不是丢弃整个结果。
                logger.warning(
                    "[LLM] content 为空但 reasoning_content 有 %d 字（finish_reason=%s，非截断），"
                    "服务端可能未遵守 LLM_ENABLE_THINKING=false——回退使用 reasoning_content 作为回答",
                    len(reasoning_content), finish_reason,
                )
                answer_content = reasoning_content

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
