"""RAG 流水线：摄取 -> 切片 -> 向量化 -> 检索 -> 生成。"""

from __future__ import annotations

from stellarai.core.config import StellarConfig
from stellarai.core.registry import registry
from stellarai.core.types import Document, Message, RetrievalResult
from stellarai.ingestion.chunker import RecursiveCharChunker
from stellarai.memory.vector_store import VectorStore
from stellarai.models.embedder import Embedder
from stellarai.models.llm import LLMProvider


def build_rag(config: StellarConfig, **kwargs) -> RAGPipeline:
    return RAGPipeline(config, **kwargs)


class RAGPipeline:
    """检索增强生成流水线，全部组件经接口注入，可独立替换/验证。"""

    def __init__(
        self,
        config: StellarConfig,
        llm: LLMProvider | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.config = config
        self.llm: LLMProvider = llm or registry.build("llm", config.llm_provider, config)
        self.embedder: Embedder = embedder or registry.build("embedder", config.embedder_provider, config)
        self.store: VectorStore = store or registry.build("vector_store", config.vector_store, config)
        self.chunker = RecursiveCharChunker(config.chunk_size, config.chunk_overlap)

    def ingest(self, docs: list[Document | str], source: str = "memory") -> int:
        n = 0
        for d in docs:
            doc = d if isinstance(d, Document) else Document(content=d, source=source)
            for c in self.chunker.chunk(doc):
                vec = self.embedder.embed_one(c.text)
                self.store.upsert(c, vec)
                n += 1
        if hasattr(self.store, "persist_now"):
            self.store.persist_now()
        return n

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        top_k = top_k or self.config.top_k
        vec = self.embedder.embed_one(query)
        return self.store.query(vec, top_k)

    def answer(self, question: str, top_k: int | None = None) -> dict:
        results = self.retrieve(question, top_k)
        context = "\n\n".join(f"[资料 {i + 1}] {r.chunk.text}" for i, r in enumerate(results))
        prompt = (
            "你是一个严谨的问答助手。仅依据下方资料回答，资料不足时如实说明。\n\n"
            f"资料:\n{context}\n\n问题: {question}\n\n回答:"
        )
        resp = self.llm.chat([Message(role="user", content=prompt)])
        return {
            "answer": resp.text,
            "sources": [r.chunk.text for r in results],
            "scores": [r.score for r in results],
        }
