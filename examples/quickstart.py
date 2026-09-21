"""StellarAI 快速上手示例。

默认使用 Mock 适配器（零 GPU/网络）跑通全链路；
将下方 cfg 改为真实提供方即可切换到生产模型（见 USAGE.md）。
"""

from __future__ import annotations

from stellarai.core.config import StellarConfig
from stellarai.core.types import Document
from stellarai.rag.pipeline import RAGPipeline
from stellarai.agents.agent import Agent
from stellarai.memory.stores.numpy_store import NumPyCosineStore
from stellarai.models.providers.mock import MockEmbedder, MockLLM, ScriptedLLM


def main() -> None:
    # 切换真实模型示例：
    # cfg = StellarConfig(
    #     llm_provider="vllm", llm_base_url="http://localhost:8000/v1",
    #     llm_model="Qwen2.5-0.5B-Instruct",
    #     embedder_provider="openai", embedder_base_url="http://localhost:8000/v1",
    #     embedder_model="bge-small-zh", vector_store="chroma",
    # )
    cfg = StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=32)
    store = NumPyCosineStore(cfg)
    rag = RAGPipeline(cfg, llm=MockLLM(cfg), embedder=MockEmbedder(cfg), store=store)

    # RAG：摄取本地知识
    rag.ingest(
        [
            Document(
                content="StellarAI 是由晨星打造的模块化 AI 系统，支持 RAG 与 Agent 编排。",
                source="kb",
            )
        ],
        source="kb",
    )
    print("RAG 回答:", rag.answer("StellarAI 是谁打造的？")["answer"])

    # Agent：ReAct + 计算器工具
    script = [
        "Action: calculator\nAction Input: 12*8",
        "Final Answer: 96",
    ]
    from stellarai.tools.builtins import CalculatorTool

    agent = Agent(cfg, llm=ScriptedLLM(script=script), tools={"calculator": CalculatorTool()}, store=store)
    print("Agent 结果:", agent.run("请帮我计算 12 乘以 8"))


if __name__ == "__main__":
    main()
