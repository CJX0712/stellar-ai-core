"""Embedding 提供方接口与构建入口。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.registry import registry


class Embedder(ABC):
    """文本向量化提供方统一接口。"""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """返回形状为 (len(texts), dim) 的向量矩阵。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()


def build_embedder(config: StellarConfig) -> Embedder:
    """按配置实例化 Embedding 提供方。"""
    return registry.build("embedder", config.embedder_provider, config)
