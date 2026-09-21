"""llama-cpp-python 本地推理适配器：在消费级设备跑 GGUF 量化模型，离线可用。"""

from __future__ import annotations

import os

from stellarai.core.config import StellarConfig
from stellarai.core.errors import ConfigError, ProviderError
from stellarai.core.registry import provider
from stellarai.core.types import LLMResponse, Message
from stellarai.models.llm import LLMProvider


@provider("llm", "llamacpp")
def _make_llamacpp(config: StellarConfig) -> LLMProvider:
    return LlamaCppLLM(config)


class LlamaCppLLM(LLMProvider):
    name = "llamacpp"

    def __init__(self, config: StellarConfig) -> None:
        self.config = config
        self.model = config.llm_model
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - 依赖可选
            raise ProviderError(
                "未安装 llama-cpp-python，请执行: pip install stellarai[llm-llamacpp]"
            ) from exc
        model_path = config.llm_base_url
        if not model_path or not os.path.exists(model_path):
            raise ConfigError("llamacpp 需在 STELLAR_LLM_BASE_URL 指定本地 .gguf 模型路径")
        self._llm = Llama(model_path=model_path, n_ctx=max(512, config.llm_max_tokens * 4))

    def _to_prompt(self, messages: list[Message]) -> str:
        parts = []
        for m in messages:
            role = "user" if m.role == "user" else ("assistant" if m.role == "assistant" else "system")
            parts.append(f"{role}: {m.content}")
        return "\n".join(parts)

    def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        out = self._llm(prompt, max_tokens=max_tokens or self.config.llm_max_tokens, temperature=temperature)
        return out["choices"][0]["text"]

    def chat(self, messages: list[Message], *, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
        prompt = self._to_prompt(messages)
        text = self.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        return LLMResponse(text=text, model=self.model, provider=self.name)
