"""零依赖向量存储：NumPy 余弦相似度，无需任何外部服务即可运行与复现。"""

from __future__ import annotations

import json
import os

import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.registry import provider
from stellarai.core.types import Chunk, RetrievalResult
from stellarai.memory.vector_store import VectorStore


@provider("vector_store", "numpy")
def _make_numpy_store(config: StellarConfig) -> VectorStore:
    return NumPyCosineStore(config)


class NumPyCosineStore(VectorStore):
    def __init__(self, config: StellarConfig) -> None:
        self.dim = config.embedder_dim
        self.persist = config.vector_persist_path
        self._items: list[tuple[Chunk, np.ndarray]] = []
        self._matrix: np.ndarray | None = None
        if self.persist:
            self._load()

    def upsert(self, chunk: Chunk, vector: list[float]) -> None:
        self._items.append((chunk, np.asarray(vector, dtype=np.float64)))
        self._matrix = None

    def query(self, vector: list[float], top_k: int = 4) -> list[RetrievalResult]:
        if not self._items:
            return []
        if self._matrix is None:
            self._matrix = np.stack([v for _, v in self._items])
        q = np.asarray(vector, dtype=np.float64)
        nq = float(np.linalg.norm(q)) or 1.0
        q = q / nq
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0] = 1.0
        sims = (self._matrix / norms[:, None]) @ q
        k = min(top_k, len(sims))
        idx = np.argsort(-sims)[:k]
        return [RetrievalResult(chunk=self._items[i][0], score=float(sims[i])) for i in idx]

    def count(self) -> int:
        return len(self._items)

    def persist_now(self) -> None:
        if not self.persist:
            return
        arr = np.stack([v for _, v in self._items])
        np.save(self.persist + ".npy", arr)
        with open(self.persist + ".json", "w", encoding="utf-8") as f:
            json.dump({"chunks": [c.__dict__ for c, _ in self._items]}, f, ensure_ascii=False)

    def _load(self) -> None:
        vec_file = self.persist + ".npy"
        meta_file = self.persist + ".json"
        if os.path.isfile(vec_file) and os.path.isfile(meta_file):
            arr = np.load(vec_file)
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            self._items = [(Chunk(**m), v) for m, v in zip(meta["chunks"], arr, strict=True)]
            self._matrix = None
