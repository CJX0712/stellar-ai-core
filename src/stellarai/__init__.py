"""StellarAI - 模块化、可编排、接口驱动的端到端 AI 系统。

设计原则：
- 单一职责：每个模块只做一件事，通过 Protocol/ABC 接口通信。
- 能力注册表 + 依赖注入：组件按名字注册，按配置实例化，支持热插拔。
- 零依赖可验证：内置 Mock 适配器，无需 GPU/网络即可跑通全链路。
- 复用领先 OSS：vLLM / llama-cpp-python / sentence-transformers / Chroma / LangGraph
  经适配器接入，一行配置切换。
"""

from stellarai.core.config import StellarConfig, get_config
from stellarai.core.errors import (
    ConfigError,
    ProviderError,
    RegistryError,
    StellarError,
)
from stellarai.core.registry import Registry, provider, registry
from stellarai.core.types import (
    Chunk,
    Document,
    LLMResponse,
    Message,
    RetrievalResult,
)
from stellarai.memory import stores as _stores  # noqa: F401

# 注册内置 provider 工厂（重依赖在工厂函数内惰性导入，导入本包不触发安装）
from stellarai.models import providers as _providers  # noqa: F401
from stellarai.tools import builtins as _tools  # noqa: F401

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "StellarConfig",
    "get_config",
    "registry",
    "provider",
    "Registry",
    "StellarError",
    "ConfigError",
    "ProviderError",
    "RegistryError",
    "Message",
    "LLMResponse",
    "Document",
    "Chunk",
    "RetrievalResult",
    "__version__",
    "__author__",
]
