"""模型网关：统一 LLM 与 Embedding 抽象，按配置注入具体实现。"""

from stellarai.models import providers  # noqa: F401  (注册内置 provider 工厂)
from stellarai.models.embedder import Embedder, build_embedder
from stellarai.models.llm import LLMProvider, build_llm

__all__ = ["LLMProvider", "build_llm", "Embedder", "build_embedder"]
