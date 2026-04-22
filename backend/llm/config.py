from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """
    单次 LLM 调用的运行时参数，可覆盖 LLMService 的全局默认值。

    None 表示"不覆盖，使用 LLMService 的默认值"。
    """

    model: str = ""
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int = 4096
    timeout: int = 0          # 0 = 使用 LLMService 默认值
    max_retry: int = 2
    extra_payload: dict = field(default_factory=dict)
