"""文档切片：递归字符切分，尽量在语义边界（段落/标点）处断开。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from stellarai.core.types import Chunk, Document

DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "，", ".", " ", ""]


class ChunkingStrategy(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        ...


class RecursiveCharChunker(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, doc: Document) -> list[Chunk]:
        segs = self._split(doc.content)
        merged = self._merge(segs)
        out: list[Chunk] = []
        prev = ""
        for i, text in enumerate(merged):
            if prev and self.chunk_overlap > 0:
                text = prev[-self.chunk_overlap :] + " " + text
            out.append(
                Chunk(text=text, doc_id=doc.source or "doc", chunk_index=i, metadata={**doc.metadata})
            )
            prev = text
        return out

    def _split(self, text: str, seps: list[str] | None = None) -> list[str]:
        seps = seps if seps is not None else self.separators
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if not seps:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = seps[0]
        parts = text.split(sep)
        result: list[str] = []
        for p in parts:
            if not p:
                continue
            if len(p) <= self.chunk_size:
                result.append(p)
            else:
                result.extend(self._split(p, seps[1:]))
        return result

    def _merge(self, segs: list[str]) -> list[str]:
        chunks: list[str] = []
        cur = ""
        for s in segs:
            if not s.strip():
                continue
            if not cur:
                cur = s
            elif len(cur) + len(s) + 1 <= self.chunk_size:
                cur = cur + " " + s
            else:
                chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
        return chunks
