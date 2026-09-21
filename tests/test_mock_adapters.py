from __future__ import annotations

import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.types import Message
from stellarai.models.providers.mock import MockEmbedder, MockLLM, ScriptedLLM


def test_mock_llm_complete_and_chat():
    cfg = StellarConfig(llm_provider="mock", embedder_dim=16)
    llm = MockLLM(cfg)
    assert "hello" in llm.complete("hello")
    resp = llm.chat([Message(role="user", content="world")])
    assert resp.text == "world"
    assert resp.provider == "mock"


def test_mock_embedder_deterministic_and_unit():
    cfg = StellarConfig(embedder_dim=16)
    emb = MockEmbedder(cfg)
    v1 = emb.embed_one("同一段文本")
    v2 = emb.embed_one("同一段文本")
    assert np.allclose(v1, v2, atol=1e-12)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-9


def test_scripted_llm_returns_sequence():
    llm = ScriptedLLM(script=["第一步", "Final Answer: 完成"])
    assert llm.complete("x") == "第一步"
    assert llm.complete("x") == "Final Answer: 完成"
    # 超出脚本长度时回退到最后一条
    assert llm.complete("x") == "Final Answer: 完成"
