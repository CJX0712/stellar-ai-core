from __future__ import annotations

from stellarai.core.types import Document


def test_rag_ingest_and_retrieve(mock_rag):
    n = mock_rag.ingest(
        [Document(content="StellarAI 是由晨星打造的模块化 AI 系统。", source="kb")],
        source="kb",
    )
    assert n >= 1
    results = mock_rag.retrieve("晨星 是谁")
    assert len(results) >= 1
    assert "晨星" in results[0].chunk.text


def test_rag_answer_includes_context(mock_rag):
    mock_rag.ingest(
        [Document(content="StellarAI 是由晨星打造的模块化 AI 系统，支持 RAG 与 Agent。", source="kb")],
        source="kb",
    )
    res = mock_rag.answer("StellarAI 是谁打造的？")
    # MockLLM 回显用户消息（含注入的上下文），故答案中应含事实
    assert "晨星" in res["answer"]
    assert len(res["sources"]) >= 1


def test_rag_empty_store(mock_rag):
    res = mock_rag.answer("任意问题")
    assert isinstance(res["answer"], str)
    assert res["sources"] == []
