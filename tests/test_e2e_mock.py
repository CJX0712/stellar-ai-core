from __future__ import annotations

from stellarai.agents.agent import Agent
from stellarai.core.config import StellarConfig
from stellarai.core.types import Document
from stellarai.memory.stores.numpy_store import NumPyCosineStore
from stellarai.models.providers.mock import MockEmbedder, MockLLM, ScriptedLLM
from stellarai.rag.pipeline import RAGPipeline


def test_end_to_end_mock_chain():
    cfg = StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=32)
    store = NumPyCosineStore(cfg)
    rag = RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=store)

    rag.ingest(
        [Document(content="StellarAI 是由晨星打造的模块化 AI 系统，支持 RAG 与 Agent 编排。", source="kb")],
        source="kb",
    )
    rag_res = rag.answer("StellarAI 是谁打造的？")
    assert "晨星" in rag_res["answer"]

    # Agent 复用同一存储做检索 + 计算
    script = [
        "Thought: 先检索\nAction: retriever\nAction Input: StellarAI 是什么",
        "Thought: 再计算\nAction: calculator\nAction Input: 6*7",
        "Final Answer: 已检索并计算得到 42",
    ]
    from stellarai.tools.builtins import CalculatorTool, RetrieverTool

    calls: list[str] = []

    class SpyRetriever(RetrieverTool):
        def run(self, action_input: str) -> str:
            result = super().run(action_input)
            calls.append(result)
            return result

    tools = {
        "retriever": SpyRetriever(cfg, store),
        "calculator": CalculatorTool(),
    }
    agent = Agent(cfg, llm=ScriptedLLM(script=script), tools=tools, store=store)
    out = agent.run("检索 StellarAI 信息并计算 6*7")
    assert "42" in out
    # 检索工具确实返回了知识库中的事实
    assert calls and "晨星" in calls[0]
