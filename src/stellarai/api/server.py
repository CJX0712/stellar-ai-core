"""FastAPI 服务：把各模块暴露为 REST 接口（OpenAPI 自动生成）。"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from stellarai.core.config import StellarConfig, get_config
from stellarai.core.logging import configure_logging
from stellarai.core.types import Document, Message
from stellarai.memory.vector_store import build_vector_store
from stellarai.models.embedder import build_embedder
from stellarai.models.llm import build_llm
from stellarai.rag.pipeline import RAGPipeline


class _AppContext:
    def __init__(self, config: StellarConfig) -> None:
        self.config = config
        self.llm = build_llm(config)
        self.embedder = build_embedder(config)
        self.store = build_vector_store(config)
        self.rag = RAGPipeline(config, llm=self.llm, embedder=self.embedder, store=self.store)


class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., description="角色/内容列表")


class EmbedRequest(BaseModel):
    texts: list[str]


class IngestRequest(BaseModel):
    documents: list[dict[str, Any]]


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


def create_app(config: StellarConfig | None = None) -> FastAPI:
    configure_logging()
    cfg = config or get_config()
    ctx = _AppContext(cfg)

    app = FastAPI(title="StellarAI", version="0.1.0")
    app.state.ctx = ctx

    @app.get("/health")
    def health():
        return {"status": "ok", "config": cfg.model_summary()}

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            messages = [Message(role=m["role"], content=m["content"]) for m in req.messages]
            resp = ctx.llm.chat(messages)
            return {"text": resp.text, "model": resp.model, "provider": resp.provider}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/embed")
    def embed(req: EmbedRequest):
        matrix = ctx.embedder.embed(req.texts)
        return {"embeddings": matrix.tolist(), "dim": ctx.embedder.dim}

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        docs = [
            Document(content=d.get("content", ""), source=d.get("source", "api"))
            for d in req.documents
        ]
        n = ctx.rag.ingest(docs, source="api")
        return {"ingested_chunks": n}

    @app.post("/query")
    def query(req: QueryRequest):
        try:
            return ctx.rag.answer(req.question, top_k=req.top_k)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app
