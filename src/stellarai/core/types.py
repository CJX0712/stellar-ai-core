"""跨模块共享的数据模型（无业务依赖）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """对话消息。role: system/user/assistant/tool。"""

    role: str
    content: str
    name: str | None = None


@dataclass
class LLMResponse:
    """LLM 一次调用的结果。"""

    text: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class Document:
    """原始文档。"""

    content: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """文档切片。"""

    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.doc_id}#{self.chunk_index}"


@dataclass
class RetrievalResult:
    """单条检索结果。"""

    chunk: Chunk
    score: float
