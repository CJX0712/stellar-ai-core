"""LLM 提供方接口与构建入口。

接口（抽象基类）定义能力边界；具体实现经注册表按配置注入。
模块可独立验证：使用 MockLLM 即可脱离 GPU/网络测试全链路。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from stellarai.core.config import StellarConfig
from stellarai.core.registry import registry
from stellarai.core.types import LLMResponse, Message


class LLMProvider(ABC):
    """LLM 提供方统一接口。"""

    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int | None = None, temperature: float = 0.0) -> str:
        """纯文本补全。"""

    @abstractmethod
    def chat(self, messages: list[Message], *, max_tokens: int | None = None, temperature: float = 0.0) -> LLMResponse:
        """多轮对话。"""


def build_llm(config: StellarConfig) -> LLMProvider:
    """按配置实例化 LLM 提供方。"""
    return registry.build("llm", config.llm_provider, config)
