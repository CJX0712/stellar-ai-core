from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stellarai.api.server import create_app
from stellarai.core.config import StellarConfig


@pytest.fixture
def client():
    cfg = StellarConfig(llm_provider="mock", embedder_provider="mock", embedder_dim=16)
    app = create_app(cfg)
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat(client):
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["text"] == "hi"


def test_embed(client):
    r = client.post("/embed", json={"texts": ["a", "b"]})
    assert r.status_code == 200
    assert r.json()["dim"] == 16
    assert len(r.json()["embeddings"]) == 2


def test_ingest_and_query(client):
    r = client.post(
        "/ingest",
        json={"documents": [{"content": "StellarAI 由晨星打造。", "source": "api"}]},
    )
    assert r.status_code == 200
    assert r.json()["ingested_chunks"] >= 1

    q = client.post("/query", json={"question": "谁打造了 StellarAI？"})
    assert q.status_code == 200
    assert "晨星" in q.json()["answer"]
