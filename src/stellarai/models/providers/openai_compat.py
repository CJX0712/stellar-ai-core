"""OpenAI 兼容协议适配器：复用 vLLM / Ollama / OpenAI 等任意兼容端点。

LLM 与 Embedding 共享同一套 HTTP 协议；通过 STELLAR_LLM_BASE_URL /
STELLAR_EMBEDDER_BASE_URL 指向不同后端，实现一行配置切换 SOTA 推理。
"""

from __future__ import annotations

import httpx
import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.errors import ConfigError, ProviderError
from stellarai.core.registry import registry
from stellarai.core.types import LLMResponse, Message
from stellarai.models.embedder import Embedder
from stellarai.models.llm import LLMProvider

_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"


def _make_compat_llm(label: str):
    def factory(config: StellarConfig) -> LLMProvider:
        return OpenAICompatLLM(config, name=label)

    return factory


registry.register("llm", "openai", _make_compat_llm("openai"))
registry.register("llm", "vllm", _make_compat_llm("vllm"))
registry.register("llm", "ollama", _make_compat_llm("ollama"))


class OpenAICompatLLM(LLMProvider):
    name = "openai-compat"

    def __init__(self, config: StellarConfig, name: str = "openai") -> None:
        self.config = config
        self.label = name
        if config.llm_base_url:
            self.base = config.llm_base_url.rstrip("/")
        elif name == "openai":
            self.base = _OPENAI_DEFAULT_BASE
        else:
            raise ConfigError(f"{name} 需设置 STELLAR_LLM_BASE_URL 指向兼容端点")
        self.api_key = config.llm_api_key
        self.model = config.llm_model

    def chat(self, messages: list[Message], *, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
        url = f"{self.base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.llm_max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"调用 {self.label} 失败: {exc}") from exc
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=self.model, provider=self.label, usage=data.get("usage", {}), raw=data)

    def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        return self.chat([Message(role="user", content=prompt)], max_tokens=max_tokens, temperature=temperature).text


def _make_compat_embedder(label: str):
    def factory(config: StellarConfig) -> Embedder:
        return OpenAICompatEmbedder(config, name=label)

    return factory


registry.register("embedder", "openai", _make_compat_embedder("openai"))
registry.register("embedder", "vllm", _make_compat_embedder("vllm"))


class OpenAICompatEmbedder(Embedder):
    name = "openai-compat"

    def __init__(self, config: StellarConfig, name: str = "openai") -> None:
        self.config = config
        self.label = name
        if config.embedder_base_url:
            self.base = config.embedder_base_url.rstrip("/")
        elif name == "openai":
            self.base = _OPENAI_DEFAULT_BASE
        else:
            raise ConfigError(f"{name} 嵌入需设置 STELLAR_EMBEDDER_BASE_URL")
        self.api_key = config.embedder_api_key
        self.model = config.embedder_model
        self.dim = config.embedder_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        url = f"{self.base}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(url, json={"model": self.model, "input": texts}, headers=headers, timeout=60)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"调用 {self.label} 嵌入失败: {exc}") from exc
        data = resp.json()
        return np.array([d["embedding"] for d in data["data"]], dtype=np.float64)
