"""确定性 Mock 适配器：零依赖、零网络，保障全链路可验证与可复现。"""

from __future__ import annotations

import hashlib

import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.registry import provider
from stellarai.core.types import LLMResponse, Message
from stellarai.models.embedder import Embedder
from stellarai.models.llm import LLMProvider


@provider("llm", "mock")
def _make_mock_llm(config: StellarConfig) -> LLMProvider:
    return MockLLM(config)


class MockLLM(LLMProvider):
    """回显式 Mock：chat 返回用户最后一条消息内容（RAG 测试可借此验证上下文注入）。"""

    name = "mock"

    def __init__(self, config: StellarConfig) -> None:
        self.config = config
        self.model = config.llm_model

    def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        return f"[mock-complete] {prompt}"

    def chat(self, messages: list[Message], *, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
        last = messages[-1].content if messages else ""
        return LLMResponse(text=last, model=self.model, provider=self.name)


@provider("llm", "mock-scripted")
def _make_scripted_llm(config: StellarConfig) -> LLMProvider:
    return ScriptedLLM(config)


class ScriptedLLM(LLMProvider):
    """脚本化 Mock：按既定脚本逐次返回，用于确定性验证 Agent 推理循环。"""

    name = "mock-scripted"

    def __init__(self, config: StellarConfig | None = None, script: list[str] | None = None) -> None:
        self.config = config
        self.model = "mock-scripted"
        self._script = list(script or ["Final Answer: 完成"])
        self._idx = 0

    def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        return self._next()

    def chat(self, messages: list[Message], *, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
        return LLMResponse(text=self._next(), model=self.model, provider=self.name)

    def _next(self) -> str:
        val = self._script[min(self._idx, len(self._script) - 1)]
        self._idx += 1
        return val


@provider("embedder", "mock")
def _make_mock_embedder(config: StellarConfig) -> Embedder:
    return MockEmbedder(config)


class MockEmbedder(Embedder):
    """基于文本哈希的确定性向量：相同文本恒定相同向量，保证可复现。"""

    name = "mock"

    def __init__(self, config: StellarConfig) -> None:
        self.dim = config.embedder_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim)
            norm = float(np.linalg.norm(vec)) or 1.0
            arr[i] = vec / norm
        return arr
