"""向量存储接口与构建入口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from stellarai.core.config import StellarConfig
from stellarai.core.registry import registry
from stellarai.core.types import Chunk, RetrievalResult


class VectorStore(ABC):
    """向量存储统一接口。"""

    @abstractmethod
    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        """写入一条切片与其向量。"""

    @abstractmethod
    def query(self, vector: list[float], top_k: int = 4) -> list[RetrievalResult]:
        """按余弦相似度返回 top_k 结果。"""

    @abstractmethod
    def count(self) -> int:
        ...

    def clear(self) -> None:  # noqa: B027
        """可选：清空存储。"""


def build_vector_store(config: StellarConfig) -> VectorStore:
    return registry.build("vector_store", config.vector_store, config)
