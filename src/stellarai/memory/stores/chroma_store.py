"""Chroma 向量存储适配器：生产级持久化向量库，经适配器接入。"""

from __future__ import annotations

from stellarai.core.config import StellarConfig
from stellarai.core.errors import ProviderError
from stellarai.core.registry import provider
from stellarai.core.types import Chunk, RetrievalResult
from stellarai.memory.vector_store import VectorStore


@provider("vector_store", "chroma")
def _make_chroma_store(config: StellarConfig) -> VectorStore:
    return ChromaStore(config)


class ChromaStore(VectorStore):
    def __init__(self, config: StellarConfig) -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - 依赖可选
            raise ProviderError("未安装 chromadb，请执行 pip install stellarai[vector]") from exc
        self.config = config
        self.dim = config.embedder_dim
        path = config.vector_persist_path or "./.chroma"
        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection("stellarai", metadata={"hnsw:space": "cosine"})
        self._seq = 0

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        cid = f"{chunk.doc_id}__{chunk.chunk_index}__{self._seq}"
        self._seq += 1
        self._col.upsert(
            ids=[cid],
            embeddings=[vector],
            documents=[chunk.text],
            metadatas=[{"doc_id": chunk.doc_id, "chunk_index": chunk.chunk_index}],
        )

    def query(self, vector: list[float], top_k: int = 4) -> list[RetrievalResult]:
        n = max(1, self._col.count())
        res = self._col.query(query_embeddings=[vector], n_results=min(top_k, n))
        out: list[RetrievalResult] = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for doc, dist in zip(res["documents"][0], res["distances"][0], strict=True):
            out.append(RetrievalResult(chunk=Chunk(text=doc, doc_id="chroma", chunk_index=0), score=float(1.0 - dist)))
        return out

    def count(self) -> int:
        return self._col.count()
