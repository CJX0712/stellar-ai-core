"""测试公共夹具：全程使用 Mock 适配器，零 GPU/网络依赖。"""

from __future__ import annotations

import pytest

from stellarai.core.config import StellarConfig
from stellarai.memory.stores.numpy_store import NumPyCosineStore
from stellarai.models.providers.mock import MockEmbedder, MockLLM
from stellarai.rag.pipeline import RAGPipeline


@pytest.fixture
def cfg() -> StellarConfig:
    return StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=32)


@pytest.fixture
def mock_store(cfg) -> NumPyCosineStore:
    return NumPyCosineStore(cfg)


@pytest.fixture
def mock_rag(cfg, mock_store) -> RAGPipeline:
    return RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=mock_store)
