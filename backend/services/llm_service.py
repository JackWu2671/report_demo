"""
LLM 服务层，封装 OpenAI-compatible chat completions API。

参考 report_system/backend/llm/service.py，简化为 demo 所需功能：
  - complete()        非流式调用，返回 content 字符串
  - complete_json()   非流式调用，解析并返回 JSON dict
  - complete_stream() 流式调用，async generator，yield content 片段
  - from_env()        从 .env 环境变量构造实例

think_tag_mode:
  "qwen3" : 响应以思考内容开头，<think>…</think> 后才是正文，剥除思考部分
  "r1"    : 同 qwen3，适用于 DeepSeek-R1 等模型
  "none"  : 不解析标签，直接返回全部 content（适用于非推理模型）
"""

import json
import logging
import os
import re

import aiohttp

from llm.config import LLMConfig

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self,
        base_url: str,
        default_model: str = "",
        api_key: str = "",
        temperature: float = 0.1,
        top_p: float = 1.0,
        timeout: int = 120,
        think_tag_mode: str = "none",
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout
        self.think_tag_mode = think_tag_mode

    @classmethod
    def from_env(cls) -> "LLMService":
        """从环境变量构造实例（需在调用前 load_dotenv）。"""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
            default_model=os.getenv("LLM_MODEL_NAME", ""),
            api_key=os.getenv("LLM_API_KEY", ""),
            temperature=float(os.getenv("LLM_TEMPERATURE", 0.1)),
            top_p=float(os.getenv("LLM_TOP_P", 1.0)),
            timeout=int(os.getenv("LLM_TIMEOUT", 120)),
            think_tag_mode=os.getenv("LLM_THINK_TAG_MODE", "none"),
        )

    # ─── 公开接口 ───────────────────────────────────────────────

    async def complete(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """非流式调用，返回最终 content 字符串（已剥除 think 内容）。"""
        cfg = config or LLMConfig()
        payload = self._build_payload(messages, cfg, stream=False)
        data = await self._post(payload, cfg)
        content = data["choices"][0]["message"]["content"]
        return self._strip_think(content)

    async def complete_json(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> dict:
        """
        非流式调用，自动解析 LLM 响应中的 JSON。
        支持 ```json``` 代码块和裸 JSON 两种格式。
        """
        raw = await self.complete(messages, config)
        return self._parse_json(raw)

    async def complete_stream(
        self,
        messages: list[dict],
        config: LLMConfig | None = None,
        strip_think: bool = False,
    ):
        """
        流式调用，async generator，逐 chunk yield content 字符串片段。

        Args:
            strip_think: True 时在流式阶段过滤 <think> 内容（用户看不到思考过程）。
                         False（默认）时全量输出，调用方收集完整文本后可自行调用
                         _strip_think() 剥除思考内容。建议保持 False 以获得实时流式体验。
        """
        cfg = config or LLMConfig()
        payload = self._build_payload(messages, cfg, stream=True)
        headers = self._headers()
        timeout = aiohttp.ClientTimeout(total=cfg.timeout or self._timeout)

        is_inside_think = False
        parse_think = self.think_tag_mode != "none" and strip_think

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"LLM 流式调用失败: status={resp.status}, body={body[:200]}"
                    )

                buf = bytearray()
                async for raw_bytes in resp.content:
                    buf.extend(raw_bytes)
                    while b"\n" in buf:
                        end = buf.find(b"\n")
                        line = bytes(buf[:end])
                        del buf[: end + 1]
                        if not line.startswith(b"data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == b"[DONE]":
                            return
                        try:
                            chunk = json.loads(data_str)
                            text = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if not text:
                                continue
                            if not parse_think:
                                yield text
                            elif is_inside_think:
                                if "</think>" in text:
                                    is_inside_think = False
                                    after = text[text.index("</think>") + 8 :]
                                    if after:
                                        yield after
                            else:
                                if "<think>" in text:
                                    is_inside_think = True
                                else:
                                    yield text
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    # ─── 内部方法 ───────────────────────────────────────────────

    async def stream_and_collect(
        self, messages: list[dict], config: LLMConfig | None = None
    ) -> str:
        """
        流式调用并实时打印到终端，返回最终正式回答内容。

        - <think>...</think> 思考内容：灰色字体实时打印，不计入返回值
        - 正式回答内容：正常字体实时打印，收集后返回
        - think_tag_mode=none 时：全部内容正常打印并返回

        Returns:
            仅正式回答内容（已剥除思考标签）的完整字符串
        """
        cfg = config or LLMConfig()
        payload = self._build_payload(messages, cfg, stream=True)
        headers = self._headers()
        timeout = aiohttp.ClientTimeout(total=cfg.timeout or self._timeout)

        use_think = self.think_tag_mode in ("qwen3", "r1")
        _OPEN, _CLOSE = "<think>", "</think>"
        _PAD = max(len(_OPEN), len(_CLOSE)) - 1  # 滚动缓冲区保留长度，防止标签被截断

        result: list[str] = []
        is_thinking = False
        pending = ""  # 未确认输出的滚动缓冲区

        def _drain(text: str) -> None:
            """将新文本追加到缓冲区，识别并处理 think 标签，刷出安全内容。"""
            nonlocal pending, is_thinking
            pending += text

            if not use_think:
                print(pending, end="", flush=True)
                result.append(pending)
                pending = ""
                return

            while True:
                tag = _CLOSE if is_thinking else _OPEN
                idx = pending.find(tag)
                if idx != -1:
                    segment = pending[:idx]
                    if segment:
                        if is_thinking:
                            print(f"\033[90m{segment}\033[0m", end="", flush=True)
                        else:
                            print(segment, end="", flush=True)
                            result.append(segment)
                    is_thinking = not is_thinking
                    pending = pending[idx + len(tag):]
                else:
                    # 保留最后 _PAD 个字符以防标签跨 chunk 截断
                    safe_len = max(0, len(pending) - _PAD)
                    if safe_len:
                        safe = pending[:safe_len]
                        if is_thinking:
                            print(f"\033[90m{safe}\033[0m", end="", flush=True)
                        else:
                            print(safe, end="", flush=True)
                            result.append(safe)
                        pending = pending[safe_len:]
                    break

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"LLM 流式调用失败: status={resp.status}, body={body[:200]}"
                    )

                raw_buf = bytearray()
                done = False
                async for raw_bytes in resp.content:
                    if done:
                        break
                    raw_buf.extend(raw_bytes)
                    while b"\n" in raw_buf:
                        end = raw_buf.find(b"\n")
                        line = bytes(raw_buf[:end])
                        del raw_buf[: end + 1]
                        if not line.startswith(b"data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == b"[DONE]":
                            done = True
                            break
                        try:
                            chunk = json.loads(data_str)
                            text = (
                                chunk.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if text:
                                _drain(text)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        # 刷出缓冲区剩余内容
        if pending:
            if is_thinking:
                print(f"\033[90m{pending}\033[0m", end="", flush=True)
            else:
                print(pending, end="", flush=True)
                result.append(pending)

        return "".join(result)

    def _build_payload(
        self, messages: list[dict], cfg: LLMConfig, stream: bool
    ) -> dict:
        payload: dict = {
            "model": cfg.model or self.default_model,
            "messages": messages,
            "temperature": cfg.temperature if cfg.temperature is not None else self._temperature,
            "top_p": cfg.top_p if cfg.top_p is not None else self._top_p,
            "max_tokens": cfg.max_tokens,
            "stream": stream,
        }
        if cfg.extra_payload:
            payload.update(cfg.extra_payload)
        return payload

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _post(self, payload: dict, cfg: LLMConfig) -> dict:
        timeout = aiohttp.ClientTimeout(total=cfg.timeout or self._timeout)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"LLM 调用失败: status={resp.status}, body={body[:300]}"
                    )
                return await resp.json()

    def _strip_think(self, content: str) -> str:
        """剥除 think 标签内的推理内容，返回正文。"""
        if self.think_tag_mode == "none":
            return content
        if "</think>" in content:
            return content[content.index("</think>") + 8 :].strip()
        return content

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """
        从 LLM 输出中提取 JSON，兼容三种格式：
        1. ```json … ``` 代码块
        2. 裸 JSON 对象
        3. 文本中嵌套的 {...} 块
        """
        s = raw.strip()
        # 1. ```json``` 代码块
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        # 2. 直接解析
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # 3. 最外层 { ... }
        first, last = s.find("{"), s.rfind("}")
        if first != -1 and last > first:
            try:
                return json.loads(s[first : last + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"无法从 LLM 输出中提取 JSON: {s[:200]}")
