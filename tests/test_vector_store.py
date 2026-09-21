from __future__ import annotations

import numpy as np

from stellarai.core.config import StellarConfig
from stellarai.core.types import Chunk
from stellarai.memory.stores.numpy_store import NumPyCosineStore


def _chunk(text: str, idx: int) -> Chunk:
    return Chunk(text=text, doc_id="d", chunk_index=idx)


def test_numpy_store_query_returns_topk_sorted():
    cfg = StellarConfig(embedder_dim=4)
    store = NumPyCosineStore(cfg)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    store.upsert(_chunk("x-axis", 0), base.tolist())
    store.upsert(_chunk("y-axis", 1), [0.0, 1.0, 0.0, 0.0])
    store.upsert(_chunk("z-axis", 2), [0.0, 0.0, 1.0, 0.0])

    res = store.query([0.9, 0.1, 0.0, 0.0], top_k=2)
    assert len(res) == 2
    assert res[0].chunk.text == "x-axis"
    assert res[0].score >= res[1].score
    # 余弦相似度在 [-1,1]，归一化后最高约 0.9+
    assert res[0].score > 0.8


def test_numpy_store_count_and_persist(tmp_path):
    cfg = StellarConfig(embedder_dim=3, vector_persist_path=str(tmp_path / "vec"))
    store = NumPyCosineStore(cfg)
    store.upsert(_chunk("a", 0), [1.0, 0.0, 0.0])
    store.persist_now()
    assert (tmp_path / "vec.npy").exists()

    reloaded = NumPyCosineStore(cfg)
    assert reloaded.count() == 1
    assert reloaded.query([1.0, 0.0, 0.0], top_k=1)[0].chunk.text == "a"
